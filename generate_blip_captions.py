import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# =====================================================
# CONFIG
# =====================================================
DATASET_ROOT = "/egr/research-sprintai/baliahsa/projects/PPML/CLiD/dataset/Datasets-Vision/imagenette2-320"
OUTPUT_PATH = "imagenette_blip_large_captions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD BLIP-LARGE
# =====================================================
print("Loading BLIP-Large...")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-large"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(DEVICE)

model.eval()
print("Model loaded on", DEVICE)

# =====================================================
# CAPTION FUNCTION
# =====================================================
@torch.no_grad()
def generate_caption(image):
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    out = model.generate(
        **inputs,
        max_length=50,
        num_beams=5
    )
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# =====================================================
# PROCESS DATASET
# =====================================================
all_results = []

for split in ["train", "val"]:
    split_path = os.path.join(DATASET_ROOT, split)

    for root, _, files in os.walk(split_path):
        for file in tqdm(files, desc=f"Processing {split}"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)

            try:
                image = Image.open(image_path).convert("RGB")
                caption = generate_caption(image)

                all_results.append({
                    "split": split,
                    "image_path": os.path.relpath(image_path, DATASET_ROOT),
                    "caption": caption
                })

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

# =====================================================
# SAVE JSON
# =====================================================
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nSaved captions to {OUTPUT_PATH}")