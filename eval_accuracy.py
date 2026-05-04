import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


IMAGENETTE_CLASSES = {
    "n01440764": {"name": "tench", "imagenet_idx": 0},
    "n02102040": {"name": "english springer", "imagenet_idx": 217},
    "n02979186": {"name": "cassette player", "imagenet_idx": 482},
    "n03000684": {"name": "chain saw", "imagenet_idx": 491},
    "n03028079": {"name": "church", "imagenet_idx": 497},
    "n03394916": {"name": "french horn", "imagenet_idx": 566},
    "n03417042": {"name": "garbage truck", "imagenet_idx": 569},
    "n03425413": {"name": "gas pump", "imagenet_idx": 571},
    "n03445777": {"name": "golf ball", "imagenet_idx": 574},
    "n03888257": {"name": "parachute", "imagenet_idx": 701},
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(label):
    return label.lower().replace("_", " ").replace("-", " ").strip()


LABEL_ALIASES = {}
for wnid, meta in IMAGENETTE_CLASSES.items():
    LABEL_ALIASES[normalize_label(wnid)] = wnid
    LABEL_ALIASES[normalize_label(meta["name"])] = wnid


def resolve_class_dir_name(name):
    return LABEL_ALIASES.get(normalize_label(name))


def is_image(path):
    return path.suffix.lower() in IMAGE_EXTENSIONS


def has_class_dirs(path):
    if not path.is_dir():
        return False
    return any(resolve_class_dir_name(child.name) is not None for child in path.iterdir() if child.is_dir())


def discover_eval_roots(input_root):
    input_root = Path(input_root)

    if (input_root / "val").is_dir() or (input_root / "train").is_dir():
        roots = []
        if (input_root / "val").is_dir():
            roots.append(("val", input_root / "val"))
        if (input_root / "train").is_dir():
            roots.append(("train", input_root / "train"))
        return roots

    if has_class_dirs(input_root):
        return [(input_root.name, input_root)]

    roots = []
    for child in sorted(input_root.iterdir()):
        if child.is_dir() and has_class_dirs(child):
            roots.append((child.name, child))

    if not roots:
        raise ValueError(f"No Imagenette class folders found under {input_root}")

    return roots


def iter_images(eval_root, max_per_class=None):
    eval_root = Path(eval_root)

    for class_dir in sorted(child for child in eval_root.iterdir() if child.is_dir()):
        wnid = resolve_class_dir_name(class_dir.name)
        if wnid is None:
            continue

        paths = sorted(path for path in class_dir.rglob("*") if path.is_file() and is_image(path))
        if max_per_class is not None:
            paths = paths[:max_per_class]

        for path in paths:
            yield path, wnid


def load_resnet50(device, weights_name):
    if weights_name == "IMAGENET1K_V1":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    elif weights_name == "IMAGENET1K_V2":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
    else:
        raise ValueError("--weights must be IMAGENET1K_V1 or IMAGENET1K_V2")

    model = models.resnet50(weights=weights)
    model.eval()
    model.to(device)
    return model, weights.transforms()


@torch.no_grad()
def evaluate_root(model, preprocess, root_name, eval_root, device, batch_size, max_per_class):
    stats = defaultdict(lambda: {
        "total": 0,
        "top1": 0,
        "top5": 0,
        "confidence_sum": 0.0,
    })

    batch_images = []
    batch_labels = []
    batch_paths = []

    def flush():
        if not batch_images:
            return

        images = torch.stack(batch_images).to(device, non_blocking=True)
        target_indices = torch.tensor(
            [IMAGENETTE_CLASSES[wnid]["imagenet_idx"] for wnid in batch_labels],
            device=device,
            dtype=torch.long,
        )

        logits = model(images)
        probs = logits.softmax(dim=1)
        top5 = logits.topk(5, dim=1).indices
        top1 = top5[:, 0]

        for i, wnid in enumerate(batch_labels):
            target = target_indices[i]
            class_stats = stats[wnid]
            class_stats["total"] += 1
            class_stats["top1"] += int(top1[i].item() == target.item())
            class_stats["top5"] += int((top5[i] == target).any().item())
            class_stats["confidence_sum"] += float(probs[i, target].item())

        batch_images.clear()
        batch_labels.clear()
        batch_paths.clear()

    image_items = list(iter_images(eval_root, max_per_class=max_per_class))
    for path, wnid in tqdm(image_items, desc=f"Evaluating {root_name}"):
        try:
            image = Image.open(path).convert("RGB")
            batch_images.append(preprocess(image))
            batch_labels.append(wnid)
            batch_paths.append(path)
        except Exception as exc:
            print(f"Skipping unreadable image {path}: {exc}")
            continue

        if len(batch_images) >= batch_size:
            flush()

    flush()
    return stats


def summarize(root_name, stats, erase_wnid):
    rows = []
    total = top1 = top5 = 0
    conf_sum = 0.0
    erased_total = erased_top1 = 0
    other_total = other_top1 = 0

    for wnid, meta in IMAGENETTE_CLASSES.items():
        class_stats = stats[wnid]
        n = class_stats["total"]
        c1 = class_stats["top1"]
        c5 = class_stats["top5"]
        avg_conf = class_stats["confidence_sum"] / n if n else 0.0

        rows.append({
            "eval_root": root_name,
            "wnid": wnid,
            "class_name": meta["name"],
            "is_erased_concept": wnid == erase_wnid,
            "num_images": n,
            "top1_correct": c1,
            "top5_correct": c5,
            "top1_accuracy": c1 / n if n else 0.0,
            "top5_accuracy": c5 / n if n else 0.0,
            "target_confidence": avg_conf,
        })

        total += n
        top1 += c1
        top5 += c5
        conf_sum += class_stats["confidence_sum"]
        if wnid == erase_wnid:
            erased_total += n
            erased_top1 += c1
        else:
            other_total += n
            other_top1 += c1

    summary = {
        "eval_root": root_name,
        "num_images": total,
        "top1_accuracy": top1 / total if total else 0.0,
        "top5_accuracy": top5 / total if total else 0.0,
        "target_confidence": conf_sum / total if total else 0.0,
        "erase_concept": IMAGENETTE_CLASSES[erase_wnid]["name"] if erase_wnid else "",
        "erase_top1_accuracy": erased_top1 / erased_total if erased_total else 0.0,
        "other_top1_accuracy": other_top1 / other_total if other_total else 0.0,
        "erase_num_images": erased_total,
        "other_num_images": other_total,
    }
    return rows, summary


def write_outputs(rows, summaries, output_csv, output_json):
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(output_json, "w") as f:
        json.dump({"summary": summaries, "per_class": rows}, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Imagenette class accuracy with an ImageNet-trained ResNet50."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="All_generated_images/generated_imagenette_from_blip",
        help="Root with model/class image folders, one model/class folder, or Imagenette train/val folders.",
    )
    parser.add_argument("--erase-concept", type=str, default="garbage truck")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V2")
    parser.add_argument("--output-csv", type=str, default="Logs/accuracy/resnet50_imagenette_accuracy.csv")
    parser.add_argument("--output-json", type=str, default="Logs/accuracy/resnet50_imagenette_accuracy.json")
    return parser.parse_args()


def main():
    args = parse_args()
    erase_wnid = resolve_class_dir_name(args.erase_concept)
    if erase_wnid is None:
        valid = ", ".join(meta["name"] for meta in IMAGENETTE_CLASSES.values())
        raise ValueError(f"Unknown --erase-concept {args.erase_concept!r}. Valid Imagenette names: {valid}")

    model, preprocess = load_resnet50(args.device, args.weights)
    eval_roots = discover_eval_roots(args.input_root)

    all_rows = []
    summaries = []

    for root_name, eval_root in eval_roots:
        stats = evaluate_root(
            model,
            preprocess,
            root_name,
            eval_root,
            args.device,
            args.batch_size,
            args.max_per_class,
        )
        rows, summary = summarize(root_name, stats, erase_wnid)
        all_rows.extend(rows)
        summaries.append(summary)

        print(
            f"{root_name}: top1={summary['top1_accuracy']:.4f}, "
            f"erase={summary['erase_top1_accuracy']:.4f}, "
            f"other={summary['other_top1_accuracy']:.4f}, "
            f"n={summary['num_images']}"
        )

    if not all_rows:
        raise ValueError(f"No images found under {args.input_root}")

    write_outputs(all_rows, summaries, args.output_csv, args.output_json)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
