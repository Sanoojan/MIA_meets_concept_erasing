
# CUDA_VISIBLE_DEVICES=1 python -m src.mia.secmi \
# --dataset laion \
# --dataset-root ./datasets \
# --ckpt-path runwayml/stable-diffusion-v1-5

CUDA_VISIBLE_DEVICES=2 python -m src.mia.secmi \
--dataset laion \
--dataset-root ./datasets \
--ckpt-path runwayml/stable-diffusion-v1-5 \
--unet-path /egr/research-sprintai/baliahsa/projects/PPML/erasing/sd_full_unet_imagenette > Logs/secmi_sd_laion_imagenette_trained.log 2>&1
