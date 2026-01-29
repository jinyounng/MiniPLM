#!/bin/bash
# Save AE latent (z) for 5% of data — same forwarding as 1stage feature matching.
#
# Usage:
#   bash save_ae_latent_5pct.sh
#   bash save_ae_latent_5pct.sh /path/to/teacher /path/to/ae.pt /path/to/save_dir

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_PATH=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}
AE_CHECKPOINT=${3-"/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/logit_only/layernorm/best_ae_ld40.pt"}
DATA_PATH=${4-"/home/jiwonyoon/data1/data/pile_dataset/data_0"}
SAVE_DIR=${5-"${BASE_PATH}/results/AE/latent_5pct"}
DATA_FRACTION=${6-0.05}
MAX_LENGTH=${7-1024}
BATCH_SIZE=${8-8}
LATENT_DIM=${9-40}

export PYTHONPATH="${BASE_PATH}"

echo "Save AE latent (5% data, same forward as 1stage)"
echo "  Teacher: ${TEACHER_PATH}"
echo "  AE: ${AE_CHECKPOINT}"
echo "  Data: ${DATA_PATH} (data_fraction=${DATA_FRACTION})"
echo "  Save: ${SAVE_DIR}"
echo ""

python "${BASE_PATH}/scripts/AE/kd/save_ae_latent_5pct.py" \
  --data_path "${DATA_PATH}" \
  --teacher_path "${TEACHER_PATH}" \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_dim "${LATENT_DIM}" \
  --data_fraction "${DATA_FRACTION}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --save_dir "${SAVE_DIR}"

echo "Done. Load with: np.load('.../latent_chunk_000000.npz') → z, y_token"
