import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from datasets import load_dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

# Imagenette class names
IMAGENETTE_LABELS = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="sd-imagenette-erased")
    parser.add_argument("--erase_concept", type=str, required=True)
    parser.add_argument("--negative_guidance", type=float, default=2.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)

    # Freeze VAE + text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=os.path.dirname(args.train_data_dir))
    train_dataset = dataset["train"]

    train_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        pixel_values = [train_transforms(img) for img in images]

        captions = [
            f"a photo of {IMAGENETTE_LABELS[list(IMAGENETTE_LABELS.keys())[label]]}"
            for label in examples["label"]
        ]

        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids}

    train_dataset = train_dataset.with_transform(preprocess)

    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        input_ids = torch.stack([e["input_ids"] for e in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    num_training_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Prepare erase + null embeddings once
    with torch.no_grad():
        erase_prompt = f"a photo of {args.erase_concept}"
        null_prompt = ""

        erase_input = tokenizer(
            erase_prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        null_input = tokenizer(
            null_prompt,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        erase_embeds = text_encoder(erase_input.input_ids)[0]
        null_embeds = text_encoder(null_input.input_ids)[0]

    # ===========================
    # TRAINING LOOP (ERASURE)
    # ===========================

    for epoch in range(args.num_train_epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):

            pixel_values = batch["pixel_values"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.fp16):

                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)

                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                )

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Standard prediction
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Erase direction
                noise_pred_erase = unet(noisy_latents, timesteps, erase_embeds.repeat(latents.size(0),1,1)).sample
                noise_pred_null = unet(noisy_latents, timesteps, null_embeds.repeat(latents.size(0),1,1)).sample

                target = noise_pred_erase - args.negative_guidance * (noise_pred_erase - noise_pred_null)

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_dataloader):.4f}")

    # Save erased model
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )

    pipeline.save_pretrained(args.output_dir)
    print("Concept erasure complete!")


if __name__ == "__main__":
    main()