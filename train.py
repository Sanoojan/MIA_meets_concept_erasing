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
import json

# Load BLIP captions
with open("/egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json", "r") as f:
    caption_data = json.load(f)

# Build mapping per split
caption_dict = {
    "train": {},
    "val": {}
}

for item in caption_data:
    split = item["split"]
    path = item["image_path"]  # e.g. train/n033.../file.JPEG
    caption_dict[split][path] = item["caption"]



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
    parser.add_argument("--output_dir", type=str, default="sd-imagenette")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=5)
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

    # Freeze VAE and text encoder
    vae.eval()
    vae.requires_grad_(False)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Load dataset with train and validation
    dataset = load_dataset("imagefolder", data_dir=os.path.dirname(args.train_data_dir))
    dataset = dataset.cast_column("image", dataset["train"].features["image"])

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # Transform
    train_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        
        # images = [img.convert("RGB") for img in examples["image"]]
        # pixel_values = [train_transforms(img) for img in images]

        # # Map integer label to ImageNet class string
        # captions = [f"a photo of {IMAGENETTE_LABELS[list(IMAGENETTE_LABELS.keys())[label]]}"
        #             for label in examples["label"]]
        
        images = []
        captions = []

        for img in examples["image"]:
            images.append(img.convert("RGB"))

            # Full path like:
            # /root/.../train/n03394916/n03394916_30968.JPEG
            full_path = img.filename

            # Extract relative path starting from train/ or validation/
            rel_path = "/".join(full_path.split(os.sep)[-3:])

            # Determine split
            split = "train" if rel_path.startswith("train") else "val"

            caption = caption_dict[split].get(rel_path)

            if caption is None:
                raise ValueError(f"Caption not found for {rel_path}")

            captions.append(caption)

        pixel_values = [train_transforms(img) for img in images]

        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {"pixel_values": pixel_values, "input_ids": inputs.input_ids}

    train_dataset = train_dataset.with_transform(preprocess)
    val_dataset = val_dataset.with_transform(preprocess)

    # Collate function
    def collate_fn(examples):
        pixel_values = torch.stack([e["pixel_values"] for e in examples])
        input_ids = [e["input_ids"] for e in examples]
        input_ids = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")["input_ids"]
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn)

    num_training_steps = args.num_train_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Training loop
    for epoch in range(args.num_train_epochs):
        total_loss = 0
        unet.train()
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.fp16):
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{args.num_train_epochs} - Training Loss: {avg_loss:.4f}")

        # Optional: Evaluate on validation set
        val_loss = 0
        unet.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(input_ids)[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                val_loss += F.mse_loss(model_pred.float(), noise.float(), reduction="mean").item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")

    # Save pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )
    pipeline.save_pretrained(args.output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()