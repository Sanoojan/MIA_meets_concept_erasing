# #!/usr/bin/env bash
# # set -euo pipefail

# # conda activate secmi

# GPU="${GPU:-0}"
# INPUT_ROOT="${INPUT_ROOT:-/egr/research-sprintai/baliahsa/projects/SecMI-LDM/All_generated_images/generated_imagenette_from_blip/esd_garbage_truck}"
# ERASE_CONCEPT="${ERASE_CONCEPT:-garbage truck}"
# BATCH_SIZE="${BATCH_SIZE:-64}"
# MAX_PER_CLASS="${MAX_PER_CLASS:-}"
# OUT_DIR="${OUT_DIR:-Logs/accuracy}"

# mkdir -p "${OUT_DIR}"

# CMD=(
#   python -m eval_accuracy
#   --input-root "${INPUT_ROOT}"
#   --erase-concept "${ERASE_CONCEPT}"
#   --batch-size "${BATCH_SIZE}"
#   --output-csv "${OUT_DIR}/resnet50_imagenette_accuracy.csv"
#   --output-json "${OUT_DIR}/resnet50_imagenette_accuracy.json"
# )

# if [[ -n "${MAX_PER_CLASS}" ]]; then
#   CMD+=(--max-per-class "${MAX_PER_CLASS}")
# fi

# CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" \
#   > "${OUT_DIR}/resnet50_imagenette_accuracy.log" 2>&1


#!/usr/bin/env bash
set -euo pipefail

# conda activate secmi

GPU="${GPU:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_PER_CLASS="${MAX_PER_CLASS:-}"
BASE_OUT_DIR="${OUT_DIR:-Logs/accuracy}"

mkdir -p "${BASE_OUT_DIR}"

# ----------------------------
# DEFINE FOLDERS + CONCEPTS
# ----------------------------
INPUT_ROOTS=(
  # "All_generated_images/generated_imagenette_100/sd_v15"
  # "All_generated_images/generated_imagenette_100/sd_imagenette_full"
  # "All_generated_images/generated_imagenette_100/IMN_blip_trd_esd_u_parachute_mia_blip"
  # "All_generated_images/generated_imagenette_100/IMN_blip_trd_esd_u_garbage_truck_mia_blip"
  # "All_generated_images/generated_imagenette_100/IMN_blip_trd_esd_u_golf_ball"
  All_generated_images/generated_imagenette_100/IMN_blip_trd_esd_u_golf_ball_mia_blip
)

CONCEPTS=(
  # "golf ball"
  # "golf ball"
  # "parachute"
  # "garbage truck"
  "golf ball"
)

# sanity check
if [[ ${#INPUT_ROOTS[@]} -ne ${#CONCEPTS[@]} ]]; then
  echo "Error: INPUT_ROOTS and CONCEPTS must have same length"
  exit 1
fi

# ----------------------------
# RUN SEQUENTIALLY
# ----------------------------
for i in "${!INPUT_ROOTS[@]}"; do

  INPUT_ROOT="${INPUT_ROOTS[$i]}"
  ERASE_CONCEPT="${CONCEPTS[$i]}"

  # take last folder name
  EXP_NAME=$(basename "${INPUT_ROOT}")

  OUT_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
  mkdir -p "${OUT_DIR}"

  echo "======================================"
  echo "Running evaluation"
  echo "Input root   : ${INPUT_ROOT}"
  echo "Concept      : ${ERASE_CONCEPT}"
  echo "Output dir   : ${OUT_DIR}"
  echo "======================================"

  CMD=(
    python -m eval_accuracy
    --input-root "${INPUT_ROOT}"
    --erase-concept "${ERASE_CONCEPT}"
    --batch-size "${BATCH_SIZE}"
    --output-csv "${OUT_DIR}/resnet50_imagenette_accuracy.csv"
    --output-json "${OUT_DIR}/resnet50_imagenette_accuracy.json"
  )

  if [[ -n "${MAX_PER_CLASS}" ]]; then
    CMD+=(--max-per-class "${MAX_PER_CLASS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" \
    > "${OUT_DIR}/run.log" 2>&1

done