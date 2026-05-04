
import tqdm
from sklearn import metrics
from datasets import load_from_disk
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
sys.path.append('/egr/research-sprintai/baliahsa/projects/SecMI-LDM')

from src.diffusers import DDIMScheduler
from src.diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from torchvision.datasets import CocoDetection
import os
from typing import Iterable, Callable, Optional, Any, Tuple, List
from omegaconf import OmegaConf
import argparse
from safetensors.torch import load_file
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import json

caption_dict = {"train": {}, "val": {}}


def load_caption_dict(caption_file):
    with open(caption_file, "r") as f:
        caption_data = json.load(f)

    captions = {"train": {}, "val": {}}
    for item in caption_data:
        split = item["split"]
        path = item["image_path"]  # e.g. train/n033.../file.JPEG
        captions[split][path] = item["caption"]
    return captions

IMAGENETTE_LABELS = {
    "n03888257": "parachute",
    "n01440764": "tench",
    "n02102040": "english springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "french horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    
}


def load_imagenette_datasets(dataset_root, class_name, num_samples=100, batch_size=4, num_workers=4):
    resolution = 512

    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_raw = ImageFolder(
        root=os.path.join(dataset_root, "train"),
        transform=transform
    )

    val_raw = ImageFolder(
        root=os.path.join(dataset_root, "val"),
        transform=transform
    )

    # 🔥 Get class index
    class_idx = train_raw.class_to_idx[class_name]

    # Filter by ImageFolder metadata. Iterating over ImageFolder itself opens every image.
    train_indices = [i for i, (_, label) in enumerate(train_raw.samples) if label == class_idx]
    val_indices = [i for i, (_, label) in enumerate(val_raw.samples) if label == class_idx]

    # Random sample 50 each
    train_indices = random.sample(train_indices, min(num_samples, len(train_indices)))
    val_indices = random.sample(val_indices, min(num_samples, len(val_indices)))

    train_subset = Subset(train_raw, train_indices)
    val_subset = Subset(val_raw, val_indices)

    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, split):
            self.dataset = dataset
            self.split = split  # "train" or "val"

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, _ = self.dataset[idx]

            # Get original ImageFolder
            base_dataset = self.dataset.dataset if isinstance(
                self.dataset, torch.utils.data.Subset
            ) else self.dataset

            # Get original index inside ImageFolder
            real_idx = self.dataset.indices[idx] if isinstance(
                self.dataset, torch.utils.data.Subset
            ) else idx

            # Full path
            path, _ = base_dataset.samples[real_idx]

            # Extract relative path: train/... or val/...
            rel_path = "/".join(path.split(os.sep)[-3:])

            caption = caption_dict[self.split].get(rel_path)

            if caption is None:
                raise ValueError(f"Caption not found for {rel_path}")

            # Tokenize caption
            input_ids = tokenizer(
                caption,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids[0]

            return {
                "pixel_values": img,
                "input_ids": input_ids
            }

    train_dataset = WrappedDataset(train_subset, "train")
    val_dataset = WrappedDataset(val_subset, "val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader

def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
            # for unknown caption
            # captions.append('None')
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
            # for unknown caption
            # captions.append('None')
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def preprocess_train(examples):
    resolution = 512
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [transform(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def load_pokemon_datasets(dataset_root):
    dataset = load_from_disk(os.path.join(dataset_root, 'pokemon'))
    train_dataset = dataset["train"].with_transform(preprocess_train)
    test_dataset = dataset["test"].with_transform(preprocess_train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=True, batch_size=1, collate_fn=collate_fn
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader

class CocoCaptionsDict(CocoDetection):

    def __init__(
            self,
            split,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
            tokenizer=None,
    ) -> None:
        assert split in ['train', 'val']
        annFile = os.path.join(root, 'annotations/captions_val2017.json')
        self.split = split
        conf = OmegaConf.load(os.path.join(root, 'coco_split.yaml'))
        root = os.path.join(root, 'val2017')
        self._train_ids = conf['train']
        self._val_ids = conf['test']
        self.tokenizer = tokenizer

        super().__init__(root, annFile, transform, target_transform, transforms)

        if split == 'train':
            self.ids = [self.ids[i] for i in self._train_ids]
        elif split == 'val':
            self.ids = [self.ids[i] for i in self._val_ids]
        self._init_tokenize_captions()

    def _init_tokenize_captions(self):
        captions = []
        for id in self.ids:
            caption = [ann["caption"] for ann in super()._load_target(id)]
            # caption = ['None' for ann in super()._load_target(id)]
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                # captions.append(random.choice(caption) if is_train else caption[0])
                captions.append(caption[0])
            else:
                raise ValueError()
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs.input_ids

    def _load_target(self, id: int):
        # return super()._load_target(id)
        return self.input_ids[id]

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(index)
        caption = super()._load_target(id)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # return image, target
        return {"pixel_values": image, "input_ids": target, 'caption': caption}

def load_coco_datasets(dataset_root):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = CocoCaptionsDict(split='train', transform=transform, tokenizer=tokenizer,
                                     root=os.path.join(dataset_root, 'coco2017val'))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    test_dataset = CocoCaptionsDict(split='val', transform=transform, tokenizer=tokenizer,
                                    root=os.path.join(dataset_root, 'coco2017val'))
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=1
    )
    return train_dataset, test_dataset, train_dataloader, test_dataloader

class StandardTransform:
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, input: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input = self.transform(input)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return input, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return [f"{head}{lines[0]}"] + ["{}{}".format(" " * len(head), line) for line in lines[1:]]

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, "Target transform: ")

        return "\n".join(body)


class LaionSet(torch.utils.data.Dataset):

    def __init__(
            self,
            img_root,
            listfile_path: str,
            transforms: Optional[Callable] = None,
            tokenizer=None,
    ) -> None:
        self.img_root = img_root
        self.tokenizer = tokenizer
        self.transforms = transforms
        # load list file
        self.img_list = np.load(listfile_path)

        self._init_tokenize_captions()


    def __len__(self):
        return len(self.img_list)


    def _init_tokenize_captions(self):
        captions = []
        for img_info in self.img_list:
            caption = img_info[1]
            captions.append(caption)

        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        )
        self.input_ids = inputs.input_ids

    def _load_target(self, id: int):
        return self.input_ids[id]

    def __getitem__(self, index: int):
        path = os.path.join(self.img_root, self.img_list[index][0] + '.jpg')
        image = Image.open(path).convert("RGB")

        target = self._load_target(index)
        caption = self.img_list[index][1]

        if self.transforms is not None:
            image, target = StandardTransform(self.transforms, None)(image, target)

        # return image, target
        return {"pixel_values": image, "input_ids": target, 'caption': caption}



def load_laion_dataset(dataset_root):
    resolution = 512
    transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_dataset = LaionSet(img_root=os.path.join(dataset_root, 'laion-2.5k/images'),
                             listfile_path=os.path.join(dataset_root, 'laion-2.5k/captions.npy'),
                             transforms=transform, tokenizer=tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=1
    )
    return train_dataset, train_dataloader


def resolve_model_path(model_path):
    if os.path.exists(model_path):
        return model_path

    if os.path.isabs(model_path):
        checkpoint_path = os.path.join(
            os.path.dirname(model_path),
            "Checkpoints",
            os.path.basename(model_path)
        )
        if os.path.exists(checkpoint_path):
            print(f"Resolved missing model path {model_path} to {checkpoint_path}")
            return checkpoint_path

    return model_path


def load_pipeline(ckpt_path, device='cuda:0'):
    ckpt_path = resolve_model_path(ckpt_path)
    pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    return pipe
def _resolve_unet_checkpoint(unet_path):
    if not unet_path:
        return None

    if os.path.isfile(unet_path):
        return unet_path

    candidates = [
        os.path.join(unet_path, "diffusion_pytorch_model.safetensors"),
        os.path.join(unet_path, "pytorch_model.bin"),
        os.path.join(unet_path, "unet.pt"),
        os.path.join(unet_path, "unet.pth"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def _extract_state_dict(checkpoint):
    if hasattr(checkpoint, "state_dict"):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("state_dict", "model_state_dict", "unet_state_dict", "unet", "model"):
        value = checkpoint.get(key)
        if hasattr(value, "state_dict"):
            return value.state_dict()
        if isinstance(value, dict):
            return value

    return checkpoint


def _match_unet_state_dict(state_dict, unet):
    if not isinstance(state_dict, dict):
        raise ValueError("UNet checkpoint did not contain a state dict.")

    target_keys = set(unet.state_dict().keys())
    prefixes = ("module.", "unet.", "model.", "model_ema.", "model.diffusion_model.")
    matched = {}
    unexpected = []

    for key, value in state_dict.items():
        candidates = [key]
        for prefix in prefixes:
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])

        matched_key = next((candidate for candidate in candidates if candidate in target_keys), None)
        if matched_key is None:
            unexpected.append(key)
        else:
            matched[matched_key] = value

    return matched, unexpected


def _load_unet_checkpoint(unet, checkpoint_path):
    if checkpoint_path.endswith(".safetensors"):
        checkpoint = load_file(checkpoint_path)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = _extract_state_dict(checkpoint)
    state_dict, unexpected = _match_unet_state_dict(state_dict, unet)

    if not state_dict:
        raise ValueError(
            f"No UNet weights in {checkpoint_path} matched the Diffusers UNet key names."
        )

    missing, load_unexpected = unet.load_state_dict(state_dict, strict=False)
    print(f"Loaded UNet checkpoint: {checkpoint_path}")
    print(f"Matched UNet tensors: {len(state_dict)}")
    if missing:
        print(f"Missing UNet tensors after load: {len(missing)}")
    if unexpected or load_unexpected:
        print(f"Ignored checkpoint tensors: {len(unexpected) + len(load_unexpected)}")


def load_pipeline_unet(ckpt_path, unet_path, base_model=None, device="cuda"):
    ckpt_path = resolve_model_path(ckpt_path)
    checkpoint_path = _resolve_unet_checkpoint(unet_path)
    model_source = base_model or ckpt_path

    pipe = StableDiffusionPipeline.from_pretrained(
        model_source,
        torch_dtype=torch.float32
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if checkpoint_path is not None:
        _load_unet_checkpoint(pipe.unet, checkpoint_path)
    elif unet_path:
        print(f"No supported UNet checkpoint found at {unet_path}. Using UNet from {model_source}.")

    pipe = pipe.to(device)
    return pipe


def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def get_reverse_denoise_results(pipe, dataloader, prefix='member'):

    weight_dtype = torch.float32
    mean_l2 = 0
    scores = []
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        # Convert images to latent space
        pixel_values = batch["pixel_values"].to(weight_dtype)
        pixel_values = pixel_values.cuda()
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215
        # Get the text embedding for conditioning
        input_ids = batch["input_ids"].cuda()
        encoder_hidden_states = text_encoder(input_ids)[0]

        image, reverse_results, denoising_results = \
            pipe(prompt=None, latents=latents, text_embeddings=encoder_hidden_states, guidance_scale=1.0)

        score = ((denoising_results[-15] - reverse_results[14]) ** 2).sum()
        scores.append(score.reshape(-1, 1))
        mean_l2 += score
        print(f'[{batch_idx}/{len(dataloader)}] mean l2-sum: {mean_l2 / (batch_idx + 1):.8f}')

    return torch.concat(scores).reshape(-1)

@torch.no_grad()
def get_clid_scores(unet, vae, text_encoder, tokenizer, dataloader, device,
                    T=1000, even_gap=20, n_steps=5, scheduler=None):

    if scheduler is None:
        scheduler = DDIMScheduler.from_pretrained(
            args.ckpt_path,
            subfolder="scheduler"
        )

    all_scores = []

    for batch in tqdm.tqdm(dataloader):

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)

        # Encode to latent
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215

        B = latents.shape[0]

        # Choose timesteps near middle
        mid = T // 2
        timesteps = torch.linspace(
            mid - even_gap * (n_steps // 2),
            mid + even_gap * (n_steps // 2),
            n_steps
        ).long().to(device)

        partial_prompts = []
        for ids in input_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            L = len(text)
            partial_prompts.extend([
                text[:L//3],
                text[L//3:2*L//3],
                text[2*L//3:],
                "",
            ])

        partial_ids = tokenizer(
            partial_prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to(device, non_blocking=True)

        emb_full = text_encoder(input_ids)[0]
        emb_partial = text_encoder(partial_ids)[0]
        encoder_hidden_states = torch.cat([emb_full, emb_partial], dim=0)

        sum_full = torch.zeros(B, device=device)
        sum_partial = torch.zeros(B, device=device)

        for t in timesteps:
            noise = torch.randn_like(latents)
            timestep = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
            noisy_latents = scheduler.add_noise(latents, noise, timestep)

            model_latents = torch.cat([
                noisy_latents,
                noisy_latents.repeat_interleave(4, dim=0),
            ], dim=0)
            model_timesteps = torch.cat([
                timestep,
                timestep.repeat_interleave(4),
            ], dim=0)
            target_noise = torch.cat([
                noise,
                noise.repeat_interleave(4, dim=0),
            ], dim=0)

            pred = unet(
                model_latents,
                model_timesteps,
                encoder_hidden_states
            ).sample

            losses = ((pred - target_noise) ** 2).mean(dim=(1, 2, 3))
            sum_full += losses[:B]
            sum_partial += losses[B:].view(B, 4).mean(dim=1)

        scores = (sum_partial - sum_full) / len(timesteps)
        all_scores.extend(scores.detach().cpu().tolist())

    return torch.tensor(all_scores)

def main(args):
    if args.dataset == 'pokemon':
        _, _, train_loader, test_loader = load_pokemon_datasets(args.dataset_root)
    elif args.dataset == 'laion':
        _, _, _, test_loader = load_coco_datasets(args.dataset_root)
        _, train_loader = load_laion_dataset(args.dataset_root)
    elif args.dataset == 'imagenette':

        pipe = load_pipeline_unet(args.ckpt_path, args.unet_path, device=args.device)
        pipe.unet.eval()
        scheduler = DDIMScheduler.from_pretrained(
            args.ckpt_path,
            subfolder="scheduler"
        )

        results = {}

        for wnid, class_name in IMAGENETTE_LABELS.items():

            print(f"\n==============================")
            print(f"Evaluating class: {class_name}")
            print(f"==============================")

            train_loader, test_loader = load_imagenette_datasets(
                args.dataset_root,
                class_name=wnid,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            
            

            # member_scores = get_reverse_denoise_results(pipe, train_loader)
            # nonmember_scores = get_reverse_denoise_results(pipe, test_loader)
            
            member_scores = get_clid_scores(
                pipe.unet, vae, text_encoder, tokenizer,
                train_loader,
                args.device,
                even_gap=args.even_gap,
                n_steps=args.n_steps,
                scheduler=scheduler,
            )

            nonmember_scores = get_clid_scores(
                pipe.unet, vae, text_encoder, tokenizer,
                test_loader,
                args.device,
                even_gap=args.even_gap,
                n_steps=args.n_steps,
                scheduler=scheduler,
            )


            labels = torch.cat([
                torch.ones_like(member_scores),
                torch.zeros_like(nonmember_scores)
            ])

            scores = torch.cat([member_scores, nonmember_scores])

            fpr, tpr, thresholds = metrics.roc_curve(
                labels.cpu().numpy(),
                scores.cpu().numpy()  # positive if higher = more member-like else negative if lower = more member-like, depending on how you define the score
            )

            auc = metrics.auc(fpr, tpr)

            print(f"Class: {class_name}  |  AUROC: {auc:.4f}")

            results[class_name] = auc

        print("\n==============================")
        print("FINAL PER-CLASS RESULTS")
        print("==============================")

        for cls, auc in results.items():
            print(f"{cls:20s}: {auc:.4f}")

        print(f"\nMean AUROC: {np.mean(list(results.values())):.4f}")

        return
    
    else:
        
        raise NotImplementedError

    pipe = load_pipeline_unet(args.ckpt_path, args.unet_path, device=args.device)

    member_scores = get_reverse_denoise_results(pipe, train_loader)
    nonmember_scores = get_reverse_denoise_results(pipe, test_loader)

    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []

    total = member_scores.size(0) + nonmember_scores.size(0)

    for threshold in torch.range(min_score, max_score, (max_score - min_score) / 10000):
        acc = ((member_scores <= threshold).sum() + (nonmember_scores > threshold).sum()) / total

        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        TPR_list.append(TPR.item())
        FPR_list.append(FPR.item())

        print(f'Score threshold = {threshold:.16f} \t ASR: {acc:.8f} \t TPR: {TPR:.8f} \t FPR: {FPR:.8f}')
    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(f'AUROC: {auc}')



def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='pokemon',
                    choices=['pokemon', 'laion', 'imagenette'])
    parser.add_argument('--dataset-root', default='dataset/Datasets-Vision/SecMI-LDM-Data/datasets', type=str)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--ckpt-path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--unet-path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--caption_file', type=str, default='/egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--n-steps', type=int, default=5)
    parser.add_argument('--even-gap', type=int, default=20)
    args = parser.parse_args()
    args.ckpt_path = resolve_model_path(args.ckpt_path)
    caption_dict = load_caption_dict(args.caption_file)
    print(f"Using caption file: {args.caption_file}")

    dataset_name_mapping = {
        "lambdalabs/pokemon-blip-captions": ("image", "text"),
    }

    image_column = 'image'
    caption_column = 'text'
    
    
    
    # with open(args.caption_file, "r") as f:
    #     caption_data = json.load(f)

   
    # # Build mapping per split
    # caption_dict = {
    #     "train": {},
    #     "val": {}
    # }

    # for item in caption_data:
    #     split = item["split"]
    #     path = item["image_path"]  # e.g. train/n033.../file.JPEG
    #     caption_dict[split][path] = item["caption"]
    

    # image.save("astronaut_rides_horse.png")
    # ckpt_path = "/home/jd3734@drexel.edu/workspace/SecMI-LDM/checkpoints/sd-pokemon-checkpoint"
    # ckpt_path = 'runwayml/stable-diffusion-v1-5'
    # args.ckpt_path = ckpt_path

    tokenizer = CLIPTokenizer.from_pretrained(
        args.ckpt_path, subfolder="tokenizer", revision=None
    )
    # tokenizer = tokenizer.cuda()

    text_encoder = CLIPTextModel.from_pretrained(
        args.ckpt_path, subfolder="text_encoder", revision=None
    )
    text_encoder = text_encoder.to(args.device)
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained(args.ckpt_path, subfolder="vae", revision=None)
    vae = vae.to(args.device)
    vae.eval()

    unet = None
    if not args.unet_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.ckpt_path, subfolder="unet", revision=None
        )
        unet = unet.to(args.device)
        unet.eval()

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    fix_seed(args.seed)

    main(args)
