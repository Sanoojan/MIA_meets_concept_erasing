#conda activate secmi

# CUDA_VISIBLE_DEVICES=6 python -m clid_my_implementation_stp \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/sd-imagenette-full_20_ep

# CUDA_VISIBLE_DEVICES=7 python -m clid_my_implementation \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/sd-imagenette-full_20_ep_blip \
# > Logs/clid_imagenette_trained_blip_text.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_golf_ball \
# > Logs/clid_classwise_blip_text_esd_golf_ball.log 2>&1 &

# CUDA_VISIBLE_DEVICES=3  python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_garbage_truck \
# > Logs/clid_classwise_blip_text_esd_garbage_truck.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path runwayml/stable-diffusion-v1-5 \
# > Logs/clid_classwise_SD_using_llava.log 2>&1 &

# CUDA_VISIBLE_DEVICES=6 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/sd-imagenette-full_20_ep_blip \
# > Logs/clid_classwise_blip_text_using_lavva.log 2>&1 &


# CUDA_VISIBLE_DEVICES=1 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_garbage_truck \
# --caption_file /egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json \
# > Logs/BLIP_Trained/CLID_BLIP/blip_trained_esd_garbage_truck2.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python -m clid_classwise \
--dataset imagenette \
--dataset-root dataset/Datasets-Vision/imagenette2-320 \
--ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/Checkpoints/sd-imagenette-full_20_ep_blip \
--unet-path /egr/research-sprintai/baliahsa/projects/PPML/robust-concept-erasing/stereo_weights_from_IMN_trd2/garbage_truck/ste_stage_model.pt \
--caption_file /egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_llava_large_captions.json \
--batch-size 4 \
--num-workers 4 \
> Logs/BLIP_Trained/CLID_LLAVA/stereo_garbage_truck_ste_stage_model.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_u_golf_ball \
# --caption_file /egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json \
# > Logs/BLIP_Trained/CLID_BLIP/blip_trained_esd_u_golf_ball.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_u_garbage_truck \
# --caption_file /egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json \
# > Logs/BLIP_Trained/CLID_BLIP/blip_trained_esd_u_garbage_truck.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_u_golf_ball \
# --caption_file /egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_blip_large_captions.json \
# > Logs/BLIP_Trained/CLID_BLIP/blip_trained_esd_u_golf_ball.log 2>&1 &

# CUDA_VISIBLE_DEVICES=4 python -m clid_classwise \
# --dataset imagenette \
# --dataset-root dataset/Datasets-Vision/imagenette2-320 \
# --ckpt-path /egr/research-sprintai/baliahsa/projects/SecMI-LDM/blip_20_ep_trained_esd_golf_ball \
# --caption_file /egr/research-sprintai/baliahsa/projects/SecMI-LDM/imagenette_llava_large_captions.json \
# > Logs/BLIP_Trained/CLID_LLava/blip_trained_esd_golf_ball.log 2>&1 &
