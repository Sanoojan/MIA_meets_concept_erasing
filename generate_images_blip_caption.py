import os
import json
import random
import torch
from diffusers import StableDiffusionPipeline

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda"
NUM_SAMPLES_PER_SPLIT = 100   # 10 train + 10 val
BATCH_SIZE = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512
BASE_SEED = 1234

CAPTION_JSON = "/egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json"
# CAPTION_JSON = "/egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_llava_large_captions.json"

# MODELS = {
#     "esd_parachute": "blip_20_ep_trained_esd_parachute",
#     "esd_golf_ball": "blip_20_ep_trained_esd_golf_ball",
#     "esd_garbage_truck": "blip_20_ep_trained_esd_garbage_truck",
#     "sd_imagenette_full": "sd-imagenette-full_20_ep_blip",
#     "sd_v15": "runwayml/stable-diffusion-v1-5",
#     "esd_all_parachute": "blip_20_ep_trained_esd_all_parachute",
#     "esd_u_parachute": "blip_20_ep_trained_esd_u_parachute",
#     "esd_x_150_parachute": "blip_20_ep_trained_esd_x_parachute_150_iter",
#     "esd_x_normal_trained_parachute": "esd_pipeline_golf_ball",
# }

MODELS = {
    "IMN_blip_trd_esd_u_golf_ball": "Checkpoints/blip_20_ep_trained_esd_u_golf_ball",
    "IMN_blip_trd_esd_u_garbage_truck": "Checkpoints/blip_20_ep_trained_esd_u_garbage_truck",
    "IMN_blip_trd_esd_u_golf_ball_mia_blip": "Checkpoints/final_blip_esd_u_golf_ball_mia",
    "IMN_blip_trd_esd_u_garbage_truck_mia_blip": "Checkpoints/final_blip_esd_u_garbage_truck_mia",
    "IMN_blip_trd_esd_u_parachute_mia_blip": "Checkpoints/final_blip_esd_u_parachute_mia",
    "sd_v15": "runwayml/stable-diffusion-v1-5",
    "sd_imagenette_full": "Checkpoints/sd-imagenette-full_20_ep_blip",
    "SD_1_5_esd_u_parachute":"Checkpoints/esd_u_sd1_5_parachute",
    "SD_1_5_esd_u_garbage_truck": "Checkpoints/esd_u_sd1_5_garbage_truck",
    "IMN_blip_trd_esd_u_parachute": "Checkpoints/blip_20_ep_trained_esd_u_parachute",
 
}

OUTPUT_ROOT = "All_generated_images/generated_imagenette_from_blip_text_100"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# -----------------------------
# LOAD CAPTIONS
# -----------------------------
with open(CAPTION_JSON, "r") as f:
    captions_data = json.load(f)

# Organize by class and split
# structure: class_dict[class_name]["train"] / ["val"]
class_dict = {}

for item in captions_data:
    split = item["split"]
    image_path = item["image_path"]  # e.g. train/n03394916/n03394916_30968.JPEG
    caption = item["caption"]

    class_name = image_path.split("/")[1]  # n03394916
    image_filename = os.path.basename(image_path)

    if class_name not in class_dict:
        class_dict[class_name] = {"train": [], "val": []}

    class_dict[class_name][split].append(
        {
            "caption": caption,
            "image_filename": image_filename,
        }
    )

# -----------------------------
# GENERATION LOOP
# -----------------------------
for model_name, model_path in MODELS.items():
    print(f"\nLoading model: {model_name}")
    model_rng = random.Random(f"{BASE_SEED}:{model_name}")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
    ).to(DEVICE)

    pipe.enable_attention_slicing()

    model_output_dir = os.path.join(OUTPUT_ROOT, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    for class_name, splits in class_dict.items():
        print(f"  Generating for class: {class_name}")

        class_dir = os.path.join(model_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        class_rng = random.Random(f"{BASE_SEED}:{model_name}:{class_name}")

        # Randomly sample 10 train and 10 val
        train_samples = class_rng.sample(
            splits["train"], 
            min(NUM_SAMPLES_PER_SPLIT, len(splits["train"]))
        )
        val_samples = class_rng.sample(
            splits["val"], 
            min(NUM_SAMPLES_PER_SPLIT, len(splits["val"]))
        )

        selected_samples = train_samples + val_samples
        sample_seeds = [model_rng.randint(0, 2**31 - 1) for _ in selected_samples]

        for batch_start in range(0, len(selected_samples), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(selected_samples))
            batch_samples = selected_samples[batch_start:batch_end]

            batch_prompts = [sample["caption"] for sample in batch_samples]
            batch_generators = [
                torch.Generator(device=DEVICE).manual_seed(seed)
                for seed in sample_seeds[batch_start:batch_end]
            ]

            batch_images = pipe(
                batch_prompts,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=batch_generators,
            ).images

            for sample, image in zip(batch_samples, batch_images):
                original_filename = sample["image_filename"]
                base_name = os.path.splitext(original_filename)[0]
                save_name = f"{base_name}_replicate.JPEG"
                save_path = os.path.join(class_dir, save_name)

                if os.path.exists(save_path):
                    print(f"    Skipping {save_name} (already exists)")
                    continue

                image.save(save_path)

    del pipe
    torch.cuda.empty_cache()

print("\nAll BLIP-caption images generated successfully!")