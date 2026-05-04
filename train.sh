CUDA_VISIBLE_DEVICES=3 python train.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --train_data_dir /egr/research-sprintai/baliahsa/projects/SecMI-LDM/dataset/Datasets-Vision/imagenette2-320/train \
  --output_dir sd-imagenette-full_20_ep_blip \
  --train_batch_size 2 \
  --num_train_epochs 20 \
  --learning_rate 1e-5 \
  --fp16 > Logs/train_sd_imagenette_full_with_blip.log 2>&1 &