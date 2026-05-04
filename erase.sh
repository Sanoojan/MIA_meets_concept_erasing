# python erase.py \
#   --pretrained_model_name_or_path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/sd-imagenette-full \
#   --train_data_dir /egr/research-sprintai/baliahsa/projects/SecMI-LDM/dataset/Datasets-Vision/imagenette2-320/train \
#   --erase_concept parachute \
#   --train_batch_size 16 \
#   --num_train_epochs 3 \
#   --learning_rate 1e-5 \
#   --fp16 > Logs/erase_sd_imagenette_parachute.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python esd_erase.py > Logs/esd/esd_u_erase_sd1.5_garbage_truck.log 2>&1 &
python esd_erase_with_mia.py > Logs/esd/esd_u_erase_sd1.5_parachute_mia_check2.log 2>&1 &