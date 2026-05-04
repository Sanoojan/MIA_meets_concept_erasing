# Does Concept Erasure in Diffusion Models Remove Training Data Membership Signals?
### (CSE 895 Course Project)

This project investigates whether **concept erasure methods in diffusion models** effectively remove **training data membership signals**, focusing on Membership Inference Attacks (MIA).

---

## Overview

We extend the SecMI framework to analyze:

- Concept erasure using ESD (Erased Stable Diffusion)
- Our method: ESD + MIA-aware erasure
- Membership inference using ClID
- Attack Success Rate (ASR)

---

## Environment Setup

Same as SecMI-LDM.

---

## Dataset

Use Imagenette dataset.

---

## Caption Generation

BLIP:
    python generate_images_blip_caption.py

LLaVA:
    python generate_images_llava_caption.py

---

## Training

    sh train.sh

---

## Concept Erasure

ESD:
    python esd_erase.py

ESD + MIA:
    python esd_erase_with_mia.py
or
    sh erase.sh

---

## Evaluation

ClID MIA:
    sh clid.sh

Run all:
    sh Final_run_all.sh

ASR:
    sh eval_accuracy.sh

---

## Workflow

1. Prepare dataset
2. Generate captions
3. Train model
4. Apply erasure
5. Evaluate MIA
6. Compute ASR
