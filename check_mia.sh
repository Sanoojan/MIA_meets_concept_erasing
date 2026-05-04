
# CUDA_VISIBLE_DEVICES=6 python -m src.mia.secmi \
# --dataset coco \
# --dataset-root /egr/research-sprintai/baliahsa/projects/SecMI-LDM/dataset/Datasets-Vision/SecMI-LDM-Data/datasets \
# --ckpt-path runwayml/stable-diffusion-v1-5 > Logs/secmi_mia_coco.log 2>&1

# CUDA_VISIBLE_DEVICES=7 python -m src.mia.secmi \
# --dataset imagenette \
# --dataset-root /egr/research-sprintai/baliahsa/projects/SecMI-LDM/dataset/Datasets-Vision/SecMI-LDM-Data/datasets \
# --ckpt-path runwayml/stable-diffusion-v1-5 > Logs/secmi_mia_imagenette.log 2>&1 &


# CUDA_VISIBLE_DEVICES=6 python -m src.mia.secmi \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path runwayml/stable-diffusion-v1-5 > Logs/secmi_sd_imagenette_classwise.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 python -m src.mia.secmi_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path esd_pipeline_golf_ball > Logs/secmi_subset_sd_imagenette_trained_esd_pipeline_golf_ball_classwise_batch.log 2>&1 &

Log_Folder=Logs/Imagenette_Finetuned_20ep
mkdir -p $Log_Folder

# CUDA_VISIBLE_DEVICES=5 python -m src.mia.secmi_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path sd-imagenette-full_20_ep > $Log_Folder/secmi_classwise_batch.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 python -m src.mia.secmi \
--dataset imagenette \
--dataset-root dataset/Datasets-Vision/imagenette2-320 \
--ckpt-path sd-imagenette-full_20_ep > $Log_Folder/secmi_all_levels_batch_a_photo_of.log 2>&1 &