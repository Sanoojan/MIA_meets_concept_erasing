import os
import random
import torch
from diffusers import StableDiffusionPipeline

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda"
NUM_IMAGES_PER_CLASS = 100
BATCH_SIZE = 10
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512
BASE_SEED = 1234

# Imagenette labels
IMAGENETTE_CLASSES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]

# Models to evaluate
# MODELS = {
#     "SD_1_5_esd_u_parachute":"Checkpoints/esd_u_sd1_5_parachute",
#     "esd_parachute": "Checkpoints/blip_20_ep_trained_esd_parachute",
#     "esd_golf_ball": "Checkpoints/blip_20_ep_trained_esd_golf_ball",
#     "esd_garbage_truck": "Checkpoints/blip_20_ep_trained_esd_garbage_truck",
#     "sd_imagenette_full": "Checkpoints/sd-imagenette-full_20_ep_blip",
#     "sd_v15": "runwayml/stable-diffusion-v1-5",
#     "esd_all_parachute": "Checkpoints/blip_20_ep_trained_esd_all_parachute",
#     "esd_u_parachute": "Checkpoints/blip_20_ep_trained_esd_u_parachute",
#     "esd_x_150_parachute": "Checkpoints/blip_20_ep_trained_esd_x_parachute_150_iter",
# }
MODELS = {
    # "IMN_blip_trd_esd_u_golf_ball": "Checkpoints/blip_20_ep_trained_esd_u_golf_ball",
    # "IMN_blip_trd_esd_u_garbage_truck": "Checkpoints/blip_20_ep_trained_esd_u_garbage_truck",
    # "IMN_blip_trd_esd_u_parachute_mia_blip": "Checkpoints/final_blip_esd_u_parachute_mia",
    # "IMN_blip_trd_esd_u_garbage_truck_mia_blip": "Checkpoints/final_blip_esd_u_garbage_truck_mia",
    "IMN2_blip_trd_esd_u_golf_ball_mia_blip": "Checkpoints/final_blip_esd_u_golf_ball_mia",
    # "IMN2_blip_trd_esd_u_garbage_truck_mia_blip": "Checkpoints/final_blip_esd_u_garbage_truck_mia",
    # "IMN2_blip_trd_esd_u_parachute_mia_blip": "Checkpoints/final_blip_esd_u_parachute_mia",
    # "sd_v15": "runwayml/stable-diffusion-v1-5",
    # "sd_imagenette_full": "Checkpoints/sd-imagenette-full_20_ep_blip",
    # "SD_1_5_esd_u_parachute":"Checkpoints/esd_u_sd1_5_parachute",
    # "SD_1_5_esd_u_garbage_truck": "Checkpoints/esd_u_sd1_5_garbage_truck",
    # "IMN_blip_trd_esd_u_parachute": "Checkpoints/blip_20_ep_trained_esd_u_parachute",
 
}


OUTPUT_ROOT = "All_generated_images/generated_imagenette_100"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

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

    for class_name in IMAGENETTE_CLASSES:
        print(f"  Generating for class: {class_name}")

        class_dir = os.path.join(model_output_dir, class_name.replace(" ", "_"))
        os.makedirs(class_dir, exist_ok=True)
        if class_dir is not None and os.path.exists(class_dir) and len(os.listdir(class_dir)) >= NUM_IMAGES_PER_CLASS:
            print(f"    Skipping {class_name} (already generated)")
            continue

        prompt = f"a photo of a {class_name}"
        seeds = [model_rng.randint(0, 2**31 - 1) for _ in range(NUM_IMAGES_PER_CLASS)]

        for batch_start in range(0, NUM_IMAGES_PER_CLASS, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, NUM_IMAGES_PER_CLASS)
            batch_prompts = [prompt] * (batch_end - batch_start)
            batch_generators = [
                torch.Generator(device=DEVICE).manual_seed(seed)
                for seed in seeds[batch_start:batch_end]
            ]

            batch_images = pipe(
                batch_prompts,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=batch_generators,
            ).images

            for image_offset, image in enumerate(batch_images, start=batch_start):
                image.save(os.path.join(class_dir, f"{class_name.replace(' ','_')}_{image_offset}.png"))

    del pipe
    torch.cuda.empty_cache()

print("\nAll images generated successfully!")