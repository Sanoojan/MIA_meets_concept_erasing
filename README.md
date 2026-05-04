# Does Concept Erasure in Diffusion Models Remove Training Data Membership Signals?
### (CSE 895 Course Project)

This project studies whether **concept erasure in diffusion models** removes **training data membership signals**, evaluated using Membership Inference Attacks (MIA).

---

## Overview

We extend SecMI with:
- ESD (Erased Stable Diffusion)
- ESD + MIA-aware erasure (ours)
- ClID-based MIA evaluation
- Attack Success Rate (ASR)

---

## Environment Setup

Follow the same setup as SecMI-LDM.

---

## Dataset

Use Imagenette dataset.

Place it at:
```
/path/to/imagenette
```

---

## Caption Generation

BLIP captions:
```
python generate_images_blip_caption.py
```

LLaVA captions:
```
python generate_images_llava_caption.py
```

---

## Training

```
sh train.sh
```

---

## Concept Erasure

ESD:
```
python esd_erase.py
```

ESD + MIA:
```
python esd_erase_with_mia.py
```

OR

```
sh erase.sh
```

---

## Evaluation

ClID MIA:
```
sh clid.sh
```

Run full pipeline:
```
sh Final_run_all.sh
```

Compute ASR:
```
sh eval_accuracy.sh
```

---

## Workflow

1. Download dataset  
2. Generate captions  
3. Train model  
4. Apply erasure  
5. Evaluate MIA  
6. Compute ASR  
