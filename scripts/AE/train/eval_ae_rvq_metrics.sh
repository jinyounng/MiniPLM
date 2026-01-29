#!/bin/bash
# Eval AE / RVQ: hidden → logits 복원 품질 (Logit MSE/KL, Hidden MSE/Cosine)
#
# Usage:
#   bash eval_ae_rvq_metrics.sh ae   /path/to/best_ae_ld25.pt
#   bash eval_ae_rvq_metrics.sh rvq /path/to/best_rvq_s4_c1024_d1024.pt

set -e
MODEL_TYPE=${1:-ae}
CKPT=${2:?"Usage: $0 <ae|rvq> <checkpoint_path>"}

BASE_PATH=${BASE_PATH:-/home/jiwonyoon/data1/projects/MiniPLM}
TEACHER_PATH=${TEACHER_PATH:-/home/jiwonyoon/data1/checkpoints/qwen/7B}
DATA_PATH=${DATA_PATH:-/home/jiwonyoon/data1/data/pile_dataset/data_0}

export PYTHONPATH=${BASE_PATH}
MAX_SAMPLES=${MAX_SAMPLES:-2000}
MAX_LENGTH=${MAX_LENGTH:-512}
BATCH_SIZE=${BATCH_SIZE:-32}

if [[ "$MODEL_TYPE" == "ae" ]]; then
  LATENT_DIM=${LATENT_DIM:-25}
  python "${BASE_PATH}/scripts/AE/train/eval_ae_rvq_metrics.py" \
    --model_type ae \
    --checkpoint_path "$CKPT" \
    --teacher_path "$TEACHER_PATH" \
    --data_path "$DATA_PATH" \
    --latent_dim "$LATENT_DIM" \
    --max_samples "$MAX_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE"
elif [[ "$MODEL_TYPE" == "rvq" ]]; then
  NUM_STAGES=${NUM_STAGES:-4}
  NUM_CODES=${NUM_CODES:-1024}
  COMPRESSED_DIM=${COMPRESSED_DIM:-1024}
  python "${BASE_PATH}/scripts/AE/train/eval_ae_rvq_metrics.py" \
    --model_type rvq \
    --checkpoint_path "$CKPT" \
    --teacher_path "$TEACHER_PATH" \
    --data_path "$DATA_PATH" \
    --num_stages "$NUM_STAGES" \
    --num_codes "$NUM_CODES" \
    --compressed_dim "$COMPRESSED_DIM" \
    --max_samples "$MAX_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE"
else
  echo "Unknown model_type: $MODEL_TYPE (use ae or rvq)"
  exit 1
fi
