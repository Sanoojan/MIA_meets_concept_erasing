import os
import json
import random
import argparse
from itertools import cycle
import torch
from torch.nn import MSELoss
from torchvision import transforms
from tqdm.auto import tqdm
from datasets import load_dataset
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# --------------------------
# CONFIG / PARAMETERS
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--erase_concept", type=str, default="parachute")
    parser.add_argument("--erase_from", type=str, default=None)
    parser.add_argument("--pipeline_path", type=str, default="/egr/research-sprintai/baliahsa/projects/SecMI-LDM/Checkpoints/sd-imagenette-full_20_ep_blip")
    parser.add_argument("--dataset_root", type=str, default="/egr/research-sprintai/baliahsa/projects/SecMI-LDM/dataset/Datasets-Vision/imagenette2-320")
    parser.add_argument("--caption_json", type=str, default="/egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_method", type=str, default="esd-u", choices=["esd-x", "esd-u", "esd-all", "esd-x-strict"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--negative_guidance", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--mia_lambda", type=float, default=1.0)
    parser.add_argument("--mia_timesteps_per_sample", type=int, default=1)
    parser.add_argument("--mia_partials_per_sample", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


args = parse_args()
ERASE_CONCEPT = args.erase_concept
ERASE_FROM = args.erase_from
BATCH_SIZE = args.batch_size
HEIGHT = WIDTH = args.resolution
NUM_INFERENCE_STEPS = args.num_inference_steps
NEGATIVE_GUIDANCE = args.negative_guidance
TRAIN_METHOD = args.train_method
ITERATIONS = args.iterations
LR = args.learning_rate
MIA_LAMBDA = args.mia_lambda
MIA_TIMESTEPS_PER_SAMPLE = args.mia_timesteps_per_sample
MIA_PARTIALS_PER_SAMPLE = args.mia_partials_per_sample
DEVICE = args.device
OUTPUT_DIR = args.output_dir or (
    f"Checkpoints/{os.path.splitext(os.path.basename(args.caption_json))[0]}_"
    f"{TRAIN_METHOD.replace('-', '_')}_{ERASE_CONCEPT.replace(' ', '_')}_mia"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# LOAD PIPELINE & MODELS
# --------------------------
print("Loading models...")
PIPELINE_PATH = args.pipeline_path
DATASET_ROOT = args.dataset_root
CAPTION_JSON = args.caption_json

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

tokenizer = CLIPTokenizer.from_pretrained(PIPELINE_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PIPELINE_PATH, subfolder="text_encoder").to(DEVICE)
base_unet = UNet2DConditionModel.from_pretrained(PIPELINE_PATH, subfolder="unet").to(DEVICE)
vae = AutoencoderKL.from_pretrained(PIPELINE_PATH, subfolder="vae").to(DEVICE)

# Make a copy of UNet for training
esd_unet = UNet2DConditionModel.from_pretrained(PIPELINE_PATH, subfolder="unet").to(DEVICE)

for model in (base_unet, esd_unet):
    try:
        model.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(f"Could not enable xformers memory efficient attention: {exc}")

try:
    esd_unet.enable_gradient_checkpointing()
except Exception as exc:
    print(f"Could not enable UNet gradient checkpointing: {exc}")

# Freeze base UNet
base_unet.requires_grad_(False)
text_encoder.requires_grad_(False)
vae.requires_grad_(False)
base_unet.eval()
text_encoder.eval()
vae.eval()
esd_unet.train()


def get_imagenette_wnid(concept):
    concept = concept.lower()
    for wnid, label in IMAGENETTE_LABELS.items():
        if label.lower() == concept:
            return wnid
    raise ValueError(
        f"Unknown Imagenette concept '{concept}'. "
        f"Expected one of: {', '.join(IMAGENETTE_LABELS.values())}"
    )


def load_caption_dict(caption_json):
    with open(caption_json, "r") as f:
        caption_data = json.load(f)

    caption_dict = {"train": {}, "val": {}}
    for item in caption_data:
        caption_dict[item["split"]][item["image_path"]] = item["caption"]
    return caption_dict


def build_class_train_dataloader(dataset_root, caption_json, erase_concept, batch_size, resolution):
    caption_dict = load_caption_dict(caption_json)
    class_wnid = get_imagenette_wnid(erase_concept)

    dataset = load_dataset("imagefolder", data_dir=dataset_root)
    train_dataset = dataset["train"]
    label_feature = train_dataset.features["label"]
    class_idx = label_feature.str2int(class_wnid)
    train_dataset = train_dataset.filter(lambda example: example["label"] == class_idx)

    train_transforms = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def preprocess(examples):
        images = []
        captions = []

        for img in examples["image"]:
            images.append(img.convert("RGB"))
            rel_path = "/".join(img.filename.split(os.sep)[-3:])
            caption = caption_dict["train"].get(rel_path)
            if caption is None:
                raise ValueError(f"Caption not found for {rel_path}")
            captions.append(caption)

        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pixel_values": [train_transforms(img) for img in images],
            "input_ids": inputs.input_ids,
        }

    train_dataset = train_dataset.with_transform(preprocess)

    def collate_fn(examples):
        return {
            "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
            "input_ids": torch.stack([example["input_ids"] for example in examples]),
        }

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    if len(dataloader) == 0:
        raise ValueError(f"No training images found for class '{erase_concept}' ({class_wnid})")

    return dataloader

def compute_mia_loss(
    unet,
    scheduler,
    latents,
    text_encoder,
    tokenizer,
    input_ids,
    device,
    timesteps_per_sample=1,
    partials_per_sample=1,
):
    """
    Differentiable CLID-inspired MIA regularizer
    """

    B = latents.shape[0]
    T = scheduler.timesteps

    # Sample a small stochastic CLID probe from the middle denoising region.
    # Keeping this tiny avoids retaining many full UNet graphs before backward.
    candidate_timesteps = T[len(T)//3 : 2*len(T)//3]
    if len(candidate_timesteps) == 0:
        candidate_timesteps = T

    total_loss = latents.new_tensor(0.0)

    for b in range(B):

        full_ids = input_ids[b:b+1]

        # decode caption
        text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

        L = len(text)
        partial_prompts = [
            text[:L//3],
            text[L//3:2*L//3],
            text[2*L//3:],
            ""
        ]
        sampled_prompts = random.sample(
            partial_prompts,
            k=min(partials_per_sample, len(partial_prompts)),
        )

        partial_ids = tokenizer(
            sampled_prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to(device)

        emb_full = text_encoder(full_ids)[0]
        emb_partial = text_encoder(partial_ids)[0]

        latent = latents[b:b+1]
        sampled_timestep_indices = torch.randperm(
            len(candidate_timesteps),
            device=candidate_timesteps.device,
        )[:timesteps_per_sample]
        sampled_timesteps = candidate_timesteps[sampled_timestep_indices]

        sample_loss = latents.new_tensor(0.0)
        probe_count = 0

        for t in sampled_timesteps:
            t = t.to(device=device, dtype=scheduler.timesteps.dtype)
            t_batch = t.expand(1)

            noise = torch.randn_like(latent)
            noisy_latent = scheduler.add_noise(latent, noise, t_batch)

            # full prompt prediction
            pred_full = unet(noisy_latent, t_batch, emb_full).sample
            loss_full = ((pred_full - noise) ** 2).mean()

            partial_loss = latents.new_tensor(0.0)
            for i in range(emb_partial.shape[0]):
                pred_partial = unet(noisy_latent, t_batch, emb_partial[i:i+1]).sample
                partial_loss = partial_loss + ((pred_partial - noise) ** 2).mean()
            partial_loss = partial_loss / emb_partial.shape[0]

            # CLID score is partial_loss - full_loss; minimize that gap so
            # member images no longer look much better under the full caption.
            sample_loss = sample_loss + torch.relu(partial_loss - loss_full)
            probe_count += 1

        total_loss = total_loss + sample_loss / probe_count

    return total_loss / B

# Optimizer: only train ESD parameters
def get_esd_trainable_parameters(unet, method="esd-x"):
    params = []
    names = []
    for name, module in unet.named_modules():
        if module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
            if method == "esd-x" and "attn2" in name:
                for n, p in module.named_parameters():
                    params.append(p)
                    names.append(name + "." + n)
            elif method == "esd-u" and "attn2" not in name:
                for n, p in module.named_parameters():
                    params.append(p)
                    names.append(name + "." + n)
            elif method == "esd-all":
                for n, p in module.named_parameters():
                    params.append(p)
                    names.append(name + "." + n)
            elif method == "esd-x-strict" and ("attn2.to_k" in name or "attn2.to_v" in name):
                for n, p in module.named_parameters():
                    params.append(p)
                    names.append(name + "." + n)
    return names, params

esd_param_names, esd_params = get_esd_trainable_parameters(esd_unet, TRAIN_METHOD)
optimizer = torch.optim.Adam(esd_params, lr=LR)
criterion = MSELoss()

print(f"Loading train images and captions from {CAPTION_JSON} for '{ERASE_CONCEPT}'...")
train_dataloader = build_class_train_dataloader(
    DATASET_ROOT,
    CAPTION_JSON,
    ERASE_CONCEPT,
    BATCH_SIZE,
    HEIGHT,
)
train_batches = cycle(train_dataloader)
print(f"Loaded {len(train_dataloader.dataset)} train samples for MIA regularization.")

# Scheduler: timesteps for diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    PIPELINE_PATH, 
    unet=base_unet,
    text_encoder=text_encoder,
    vae=vae,
    safety_checker=None
)
pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=DEVICE)

# --------------------------
# ENCODE PROMPTS
# --------------------------
with torch.no_grad():
    input_ids = tokenizer(
        ERASE_CONCEPT,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    erase_embeds = text_encoder(input_ids)[0]

    # Null embedding for classifier-free guidance
    null_input_ids = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(DEVICE)
    null_embeds = text_encoder(null_input_ids)[0]

    if ERASE_FROM is not None:
        erase_from_input_ids = tokenizer(
            ERASE_FROM,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(DEVICE)
        erase_from_embeds = text_encoder(erase_from_input_ids)[0]
    else:
        erase_from_embeds = erase_embeds

# --------------------------
# TRAINING LOOP
# --------------------------
pbar = tqdm(range(ITERATIONS), desc="Training ESD")
for i in pbar:
    optimizer.zero_grad()
    batch = next(train_batches)
    pixel_values = batch["pixel_values"].to(DEVICE)
    batch_input_ids = batch["input_ids"].to(DEVICE)
    batch_size = pixel_values.shape[0]

    with torch.no_grad():
        image_latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

    # Random timestep
    t_idx = random.randint(0, NUM_INFERENCE_STEPS - 1)
    timestep = pipe.scheduler.timesteps[t_idx].to(DEVICE)
    timestep_batch = timestep.expand(batch_size)

    # Sample random latent
    latents = torch.randn((batch_size, base_unet.in_channels, HEIGHT // 8, WIDTH // 8), device=DEVICE)

    # Add noise
    noise = torch.randn_like(latents)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep_batch)

    erase_batch_embeds = erase_embeds.expand(batch_size, -1, -1)
    null_batch_embeds = null_embeds.expand(batch_size, -1, -1)
    erase_from_batch_embeds = erase_from_embeds.expand(batch_size, -1, -1)

    # Base UNet predictions
    with torch.no_grad():
        noise_pred_erase = base_unet(noisy_latents, timestep_batch, encoder_hidden_states=erase_batch_embeds).sample
        noise_pred_null = base_unet(noisy_latents, timestep_batch, encoder_hidden_states=null_batch_embeds).sample
        noise_pred_erase_from = base_unet(
            noisy_latents,
            timestep_batch,
            encoder_hidden_states=erase_from_batch_embeds,
        ).sample

    # ESD UNet prediction
    noise_pred_esd = esd_unet(noisy_latents, timestep_batch, encoder_hidden_states=erase_from_batch_embeds).sample

    # Loss and backward
    # loss = criterion(noise_pred_esd, noise_pred_erase_from - (NEGATIVE_GUIDANCE * (noise_pred_erase - noise_pred_null)))
    
    # ESD loss (existing)
    loss_esd = criterion(
        noise_pred_esd,
        noise_pred_erase_from - (NEGATIVE_GUIDANCE * (noise_pred_erase - noise_pred_null))
    )
    loss_esd.backward()
    loss_esd_value = loss_esd.detach()

    del (
        latents,
        noise,
        noisy_latents,
        noise_pred_erase,
        noise_pred_null,
        noise_pred_erase_from,
        noise_pred_esd,
        erase_batch_embeds,
        null_batch_embeds,
        erase_from_batch_embeds,
    )

    # --------------------------
    # MIA LOSS (NEW)
    # --------------------------
    mia_loss = compute_mia_loss(
        esd_unet,
        pipe.scheduler,
        image_latents.detach(),
        text_encoder,
        tokenizer,
        batch_input_ids,
        DEVICE,
        timesteps_per_sample=MIA_TIMESTEPS_PER_SAMPLE,
        partials_per_sample=MIA_PARTIALS_PER_SAMPLE,
    )

    (MIA_LAMBDA * mia_loss).backward()
    optimizer.step()

    total_loss_value = loss_esd_value + MIA_LAMBDA * mia_loss.detach()
    pbar.set_postfix({
        "loss": total_loss_value.item(),
        "esd": loss_esd_value.item(),
        "mia": mia_loss.item(),
    })

# --------------------------
# SAVE FULL PIPELINE
# --------------------------
pipe.unet = esd_unet
pipe.save_pretrained(OUTPUT_DIR)
print(f"ESD concept-erased pipeline saved at: {OUTPUT_DIR}")
