#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PIPELINE_PATH="$PROJECT_ROOT/Checkpoints/sd-imagenette-full_20_ep_blip"
DATASET_ROOT="$PROJECT_ROOT/dataset/Datasets-Vision/imagenette2-320"

TRAIN_LOG_DIR="$PROJECT_ROOT/Logs/esd/final_run"
CLID_LOG_DIR="$PROJECT_ROOT/Logs/Final_CLID"
CHECKPOINT_DIR="$PROJECT_ROOT/Checkpoints"
mkdir -p "$TRAIN_LOG_DIR" "$CLID_LOG_DIR" "$CHECKPOINT_DIR"

GPUS=(4 5)
CONCEPTS=("golf ball" "garbage truck")
CAPTION_TAGS=("blip" "llava")
CAPTION_FILES=(
  "$PROJECT_ROOT/imagenette_blip_large_captions.json"
  "$PROJECT_ROOT/imagenette_llava_large_captions.json"
)

TRAIN_METHOD="esd-u"
ITERATIONS=100
LR="5e-5"
MIA_LAMBDA="1.0"
MIA_TIMESTEPS_PER_SAMPLE=1
MIA_PARTIALS_PER_SAMPLE=2

pids=()

slugify() {
  local value="$1"
  value="${value,,}"
  value="${value// /_}"
  value="${value//-/_}"
  echo "$value"
}

wait_for_batch() {
  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  pids=()
  if [[ "$status" -ne 0 ]]; then
    echo "At least one job failed. Check the corresponding log above." >&2
    exit "$status"
  fi
}

# echo "=============================="
# echo "Training 6 ESD+MIA checkpoints"
# echo "=============================="

# job_idx=0
# for concept in "${CONCEPTS[@]}"; do
#   concept_slug="$(slugify "$concept")"

#   for caption_idx in "${!CAPTION_TAGS[@]}"; do
#     caption_tag="${CAPTION_TAGS[$caption_idx]}"
#     caption_file="${CAPTION_FILES[$caption_idx]}"
#     gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"

#     output_dir="$CHECKPOINT_DIR/final_${caption_tag}_${TRAIN_METHOD//-/_}_${concept_slug}_mia"
#     train_log="$TRAIN_LOG_DIR/final_${caption_tag}_${TRAIN_METHOD//-/_}_${concept_slug}_mia.log"

#     echo "[$(date)] Train: concept='$concept' captions='$caption_tag' gpu=$gpu"
#     CUDA_VISIBLE_DEVICES="$gpu" python esd_erase_with_mia.py \
#       --erase_concept "$concept" \
#       --pipeline_path "$PIPELINE_PATH" \
#       --dataset_root "$DATASET_ROOT" \
#       --caption_json "$caption_file" \
#       --output_dir "$output_dir" \
#       --train_method "$TRAIN_METHOD" \
#       --iterations "$ITERATIONS" \
#       --learning_rate "$LR" \
#       --mia_lambda "$MIA_LAMBDA" \
#       --mia_timesteps_per_sample "$MIA_TIMESTEPS_PER_SAMPLE" \
#       --mia_partials_per_sample "$MIA_PARTIALS_PER_SAMPLE" \
#       --device cuda:0 \
#       > "$train_log" 2>&1 &

#     pids+=("$!")
#     job_idx=$((job_idx + 1))

#     if [[ "${#pids[@]}" -eq "${#GPUS[@]}" ]]; then
#       wait_for_batch
#     fi
#   done
# done
# wait_for_batch

echo "=============================="
echo "Running 12 CLID evaluations"
echo "=============================="

job_idx=0
for train_caption_idx in "${!CAPTION_TAGS[@]}"; do

  
  for eval_caption_idx in "${!CAPTION_TAGS[@]}"; do
  
    train_caption_tag="${CAPTION_TAGS[$train_caption_idx]}"
    

    for concept in "${CONCEPTS[@]}"; do
      concept_slug="$(slugify "$concept")"
      ckpt_path="$CHECKPOINT_DIR/final_${train_caption_tag}_${TRAIN_METHOD//-/_}_${concept_slug}_mia"
      eval_caption_tag="${CAPTION_TAGS[$eval_caption_idx]}"
      eval_caption_file="${CAPTION_FILES[$eval_caption_idx]}"
      gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"

      clid_log="$CLID_LOG_DIR/final2_${train_caption_tag}_${TRAIN_METHOD//-/_}_${concept_slug}_mia_eval_${eval_caption_tag}.log"

      echo "[$(date)] CLID: checkpoint='$(basename "$ckpt_path")' eval_captions='$eval_caption_tag' gpu=$gpu"
      CUDA_VISIBLE_DEVICES="$gpu" python -m clid_classwise \
        --dataset imagenette \
        --dataset-root "$DATASET_ROOT" \
        --ckpt-path "$ckpt_path" \
        --caption_file "$eval_caption_file" \
        --device cuda:0 \
        > "$clid_log" 2>&1 &

      pids+=("$!")
      job_idx=$((job_idx + 1))

      if [[ "${#pids[@]}" -eq "${#GPUS[@]}" ]]; then
        wait_for_batch
      fi
    done
  done
done
wait_for_batch

echo "All runs finished."
echo "Training logs: $TRAIN_LOG_DIR"
echo "CLID logs: $CLID_LOG_DIR"
