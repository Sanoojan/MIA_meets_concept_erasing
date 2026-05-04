python stack_images.py \
  --paths All_generated_images/generated_imagenette_100/sd_v15 \
           All_generated_images/generated_imagenette_100/sd_imagenette_full \
           All_generated_images/generated_imagenette_100/IMN_blip_trd_esd_u_parachute \
  --names "SD-1.5" "Imagenette finetuned" "ESD-U parachute" \
  --output stacked/comparison_grid.png \
  --img_size 256