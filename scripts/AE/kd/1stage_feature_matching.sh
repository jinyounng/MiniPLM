#!/bin/bash
# 1-stage KD: Student hidden ~ AE(teacher hidden). Uses 5% of training data.
#
# Usage:
#   bash 1stage_feature_matching.sh
#   bash 1stage_feature_matching.sh /path/to/teacher /path/to/student /path/to/ae.pt

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_PATH=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}
# Student: config path for from-scratch (e.g. checkpoints/qwen/200M), or checkpoint with weights to resume
STUDENT_PATH=${3-"/home/jiwonyoon/data1/checkpoints/qwen/200M"}
FROM_SCRATCH=${4-1}   # 1 = pretrain from scratch (config only), 0 = load weights
AE_CHECKPOINT=${5-"/home/jiwonyoon/data1/projects/MiniPLM/checkpoints/AE/logit_only/layernorm/best_ae_ld40.pt"}
DATA_PATH=${6-"/home/jiwonyoon/data1/data/pile_dataset/data_0"}
OUTPUT_DIR=${7-"${BASE_PATH}/results/AE/kd/1stage_fm_8e-4"}

DATA_FRACTION=${8-0.05}   # 5% of data
MAX_LENGTH=${9-1024}
BATCH_SIZE=${10-8}
EPOCHS=${11-1}
LR=${12-8e-4}
LATENT_DIM=${13-40}
LOSS_TYPE=${14-mse}        # mse or cosine

export PYTHONPATH="${BASE_PATH}"

# Build --from_scratch flag for Python
FROM_SCRATCH_FLAG=""
if [ "${FROM_SCRATCH}" = "1" ]; then
  FROM_SCRATCH_FLAG="--from_scratch"
fi

echo "1-stage Feature Matching (student hidden ~ AE(teacher hidden)); pretrain from scratch=${FROM_SCRATCH}"
echo "  Teacher: ${TEACHER_PATH}"
echo "  Student: ${STUDENT_PATH} (from_scratch=${FROM_SCRATCH})"
echo "  AE checkpoint: ${AE_CHECKPOINT}"
echo "  Data: ${DATA_PATH} (data_fraction=${DATA_FRACTION})"
echo "  Output: ${OUTPUT_DIR}"
echo "  Loss: ${LOSS_TYPE}"
echo ""

python "${BASE_PATH}/scripts/AE/kd/1stage_feature_matching.py" \
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
