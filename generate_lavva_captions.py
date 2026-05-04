import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaProcessor, LlavaForConditionalGeneration

# =====================================================
# CONFIG
# =====================================================
DATASET_ROOT = "/egr/research-sprintai/baliahsa/projects/PPML/CLiD/dataset/Datasets-Vision/imagenette2-320"
OUTPUT_PATH = "imagenette_llava_large_captions.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# LOAD LLaVA
# =====================================================
print("Loading LLaVA-1.5...")
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16
).to(DEVICE)
model.eval()
print("Model loaded on", DEVICE)

# =====================================================
# CAPTION FUNCTION
# =====================================================
@torch.no_grad()
def generate_caption(image: Image.Image):
    try:
        # LLaVA-1.5 requires the <image> token in the prompt
        prompt = "USER: <image>\nDescribe this image in one sentence.\nASSISTANT:"
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE, torch.float16)
        
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )
        
        # Decode only the newly generated tokens (strip the prompt)
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        caption = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        raise RuntimeError(f"Model generation failed: {e}")

# =====================================================
# PROCESS DATASET
# =====================================================
all_results = []
error_log = []

for split in ["train", "val"]:
    split_path = os.path.join(DATASET_ROOT, split)
    for root, _, files in os.walk(split_path):
        for file in tqdm(files, desc=f"Processing {split}"):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)
            rel_path = os.path.relpath(image_path, DATASET_ROOT)

            try:
                image = Image.open(image_path).convert("RGB")
                caption = generate_caption(image)
                all_results.append({
                    "split": split,
                    "image_path": rel_path,
                    "caption": caption
                })
            except Exception as e:
                error_msg = f"{rel_path}: {e}"
                print("Error processing", error_msg)
                error_log.append(error_msg)

# =====================================================
# SAVE RESULTS
# =====================================================
with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=4)

if error_log:
    with open("imagenette_llava_errors.log", "w") as f:
        for line in error_log:
            f.write(line + "\n")
    print(f"\nCompleted with {len(error_log)} errors. See imagenette_llava_errors.log")

print(f"\nSaved captions to {OUTPUT_PATH}")