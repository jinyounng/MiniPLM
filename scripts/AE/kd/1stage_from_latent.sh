#!/bin/bash
# 1-stage KD from pre-saved latent: Teacher forward 없이 저장된 z, y_token만 사용.
# save_ae_latent_5pct.sh로 저장한 latent_dir과 동일한 data_path, data_fraction, batch_size 사용.
#
# Usage:
#   bash 1stage_from_latent.sh
#   bash 1stage_from_latent.sh <base_path> <latent_dir> ... <loss_type>
#   포트만 지정: MAIN_PORT=29501 bash 1stage_from_latent.sh

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
LATENT_DIR=${2-"${BASE_PATH}/results/AE/latent_5pct"}
TEACHER_PATH=${3-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}
STUDENT_PATH=${4-"/home/jiwonyoon/data1/checkpoints/qwen/200M"}
FROM_SCRATCH=${5-1}
AE_CHECKPOINT=${6-"/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/logit_only/layernorm/best_ae_ld40.pt"}
DATA_PATH=${7-"/home/jiwonyoon/data1/data/pile_dataset/data_0"}
OUTPUT_DIR=${8-"${BASE_PATH}/results/AE/kd/1stage_from_latent_ld40"}

# save_ae_latent 시와 동일해야 함
DATA_FRACTION=${9-1}
MAX_LENGTH=${10-1024}
BATCH_SIZE=${11-16}

EPOCHS=${12-1}
LR=${13-5e-4}
LATENT_DIM=${14-40}
LOSS_TYPE=${15-mse}
MAIN_PORT=${MAIN_PORT:-29500}

export PYTHONPATH="${BASE_PATH}"

FROM_SCRATCH_FLAG=""
if [ "${FROM_SCRATCH}" = "1" ]; then
  FROM_SCRATCH_FLAG="--from_scratch"
fi

echo "1-stage from pre-saved latent (decoder only; no teacher forward)"
echo "  Latent dir: ${LATENT_DIR}"
echo "  Teacher (embed only): ${TEACHER_PATH}"
echo "  Student: ${STUDENT_PATH} (from_scratch=${FROM_SCRATCH})"
echo "  AE checkpoint: ${AE_CHECKPOINT}"
echo "  Data: ${DATA_PATH} (data_fraction=${DATA_FRACTION}, batch_size=${BATCH_SIZE})"
echo "  Output: ${OUTPUT_DIR}"
echo "  Loss: ${LOSS_TYPE}"
echo "  Launch: accelerate (multi-GPU), port=${MAIN_PORT}"
echo ""

accelerate launch --main_process_port "${MAIN_PORT}" "${BASE_PATH}/scripts/AE/kd/1stage_from_latent.py" \
  --latent_dir "${LATENT_DIR}" \
  --data_path "${DATA_PATH}" \
  --teacher_path "${TEACHER_PATH}" \
  --student_path "${STUDENT_PATH}" \
  ${FROM_SCRATCH_FLAG} \
  --ae_checkpoint "${AE_CHECKPOINT}" \
  --latent_dim "${LATENT_DIM}" \
  --data_fraction "${DATA_FRACTION}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --output_dir "${OUTPUT_DIR}" \
  --loss_type "${LOSS_TYPE}"
