import os
import random
import torch
from torch.nn import MSELoss
from safetensors.torch import save_file
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

# --------------------------
# CONFIG / PARAMETERS
# --------------------------
ERASE_CONCEPT = "garbage truck"  # Concept to erase
ERASE_FROM = None  # if None, erase concept from itself
BATCH_SIZE = 1
HEIGHT = WIDTH = 512
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 3.0
NEGATIVE_GUIDANCE = 2.0
TRAIN_METHOD = "esd-u"  # esd-x, esd-u, esd-all, esd-x-strict
ITERATIONS = 100
LR = 5e-5
DEVICE = "cuda:0"
OUTPUT_DIR = "Checkpoints/esd_u_sd1_5_garbage_truck"  # Change this for different concepts / methods
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# LOAD PIPELINE & MODELS
# --------------------------
print("Loading models...")
# PIPELINE_PATH = "/egr/research-sprintai/baliahsa/projects/SecMI-LDM/sd-imagenette-full_20_ep_blip"
PIPELINE_PATH = "runwayml/stable-diffusion-v1-5"
tokenizer = CLIPTokenizer.from_pretrained(PIPELINE_PATH, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(PIPELINE_PATH, subfolder="text_encoder").to(DEVICE)
base_unet = UNet2DConditionModel.from_pretrained(PIPELINE_PATH, subfolder="unet").to(DEVICE)
vae = None  # Not needed for ESD-style noise prediction

# Make a copy of UNet for training
esd_unet = UNet2DConditionModel.from_pretrained(PIPELINE_PATH, subfolder="unet").to(DEVICE)

# Freeze base UNet
base_unet.requires_grad_(False)
text_encoder.requires_grad_(False)

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

# Scheduler: timesteps for diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    PIPELINE_PATH, 
    unet=base_unet,
    text_encoder=text_encoder,
    safety_checker=None
)
pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS)

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

    # Random timestep
    t_idx = random.randint(0, NUM_INFERENCE_STEPS - 1)
    timestep = pipe.scheduler.timesteps[t_idx]

    # Sample random latent
    latents = torch.randn((BATCH_SIZE, base_unet.in_channels, HEIGHT // 8, WIDTH // 8), device=DEVICE)

    # Add noise
    noise = torch.randn_like(latents)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep)

    # Base UNet predictions
    pipe.unet = base_unet
    noise_pred_erase = pipe.unet(noisy_latents, timestep, encoder_hidden_states=erase_embeds)[0]
    noise_pred_null = pipe.unet(noisy_latents, timestep, encoder_hidden_states=null_embeds)[0]
    noise_pred_erase_from = pipe.unet(noisy_latents, timestep, encoder_hidden_states=erase_from_embeds)[0]

    # ESD UNet prediction
    pipe.unet = esd_unet
    noise_pred_esd = pipe.unet(noisy_latents, timestep, encoder_hidden_states=erase_from_embeds)[0]

    # Loss and backward
    loss = criterion(noise_pred_esd, noise_pred_erase_from - (NEGATIVE_GUIDANCE * (noise_pred_erase - noise_pred_null)))
    loss.backward()
    optimizer.step()

    pbar.set_postfix({"loss": loss.item()})

# --------------------------
# SAVE FULL PIPELINE
# --------------------------
pipe.unet = esd_unet
pipe.save_pretrained(OUTPUT_DIR)
print(f"ESD concept-erased pipeline saved at: {OUTPUT_DIR}")