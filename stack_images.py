import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def get_images_from_class(class_dir, num_images=2):
    """Get first num_images from a class directory."""
    if not os.path.exists(class_dir):
        return []
    
    image_files = sorted([
        f for f in os.listdir(class_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG'))
    ])
    return image_files[:num_images]


def get_random_other_images(model_dir, exclude_classes, num_images=4):
    """Get random images from classes not in exclude_classes."""
    all_classes = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d not in exclude_classes
    ]
    
    random_images = []
    for class_name in random.sample(all_classes, min(num_images, len(all_classes))):
        class_dir = os.path.join(model_dir, class_name)
        image_files = sorted([
            f for f in os.listdir(class_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG'))
        ])
        if image_files:
            random_images.append((class_name, image_files[0]))
    
    return random_images


def create_comparison_grid(model_dirs, model_names, classes_to_stack, output_path, img_size=256):
    """
    Create a comparison grid of images from multiple models.
    
    Args:
        model_dirs: List of model directory paths
        model_names: List of model display names
        classes_to_stack: Dict with class names and number of images to select
        output_path: Path to save the output image
        img_size: Size of each individual image in the grid
    """
    num_models = len(model_dirs)
    num_classes = sum(classes_to_stack.values())
    
    # Calculate grid dimensions
    grid_width = num_models
    grid_height = num_classes + 1  # +1 for header row with model names
    
    # Create grid
    grid_img = Image.new(
        'RGB',
        (grid_width * img_size, grid_height * img_size),
        color='white'
    )
    draw = ImageDraw.Draw(grid_img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add model name headers
    for col, model_name in enumerate(model_names):
        x = col * img_size + img_size // 2
        y = img_size // 2
        text_bbox = draw.textbbox((0, 0), model_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.text(
            (x - text_width // 2, y - text_height // 2),
            model_name,
            fill='black',
            font=font
        )
    
    # Collect all images for each model
    model_images = []
    exclude_classes = set(classes_to_stack.keys())
    
    for model_dir in model_dirs:
        images = []
        
        # Get specified classes
        for class_name, num_imgs in classes_to_stack.items():
            class_dir = os.path.join(model_dir, class_name)
            img_files = get_images_from_class(class_dir, num_imgs)
            for img_file in img_files:
                images.append(os.path.join(class_dir, img_file))
        
        # Get random other images
        random_others = get_random_other_images(model_dir, exclude_classes, 4)
        for class_name, img_file in random_others:
            images.append(os.path.join(model_dir, class_name, img_file))
        
        model_images.append(images)
    
    # Paste images into grid
    row_offset = 1  # Skip header row
    for row_idx in range(num_classes):
        for col_idx, images in enumerate(model_images):
            if row_idx < len(images):
                try:
                    img = Image.open(images[row_idx])
                    img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                    x = col_idx * img_size
                    y = (row_offset + row_idx) * img_size
                    grid_img.paste(img, (x, y))
                except Exception as e:
                    print(f"Failed to load {images[row_idx]}: {e}")
    
    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid_img.save(output_path)
    print(f"Saved comparison grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stack images from multiple models for visual comparison"
    )
    parser.add_argument(
        "--paths",
        nargs='+',
        required=True,
        help="List of model directory paths"
    )
    parser.add_argument(
        "--names",
        nargs='+',
        required=True,
        help="List of model display names"
    )
    parser.add_argument(
        "--output",
        default="All_generated_images/comparison_grid.png",
        help="Output file path for comparison grid"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Size of each image in the grid"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    if len(args.paths) != len(args.names):
        print("Error: Number of paths must match number of names")
        return
    
    random.seed(args.seed)
    
    # Define classes to stack: 2 garbage_truck, 2 golf_ball, 2 parachute, 4 others random
    classes_to_stack = {
        "garbage_truck": 2,
        "golf_ball": 2,
        "parachute": 2,
    }
    
    create_comparison_grid(
        model_dirs=args.paths,
        model_names=args.names,
        classes_to_stack=classes_to_stack,
        output_path=args.output,
        img_size=args.img_size
    )


if __name__ == "__main__":
    main()
