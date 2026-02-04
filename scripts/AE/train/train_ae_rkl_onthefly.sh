#!/bin/bash
# Train Conditional AutoEncoder with Reverse KL (on-the-fly, DDP)
#
# train_ae_rkl_onthefly.py: On-the-fly + KL(recon || teacher)
#
# 사용 예시:
#   bash train_ae_rkl_onthefly.sh
#   bash train_ae_rkl_onthefly.sh /path/to/teacher/model 25
#
# 주의: accelerate config를 먼저 실행하여 설정해야 합니다.

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/logit_only/layernorm_rkl"

LATENT_DIM=${3-40}
VAL_SAMPLES=${4-1000}
BATCH_SIZE=${5-8}
EPOCHS=${6-1}
LR=${7-5e-4}
PATIENCE=${8-10}
MAX_LENGTH=${9-1024}

ALPHA_MSE=${10-0.0}
ALPHA_COSINE=${11-0.0}
ALPHA_LOGIT=${12-1.0}           # Reverse KL weight
ALPHA_LOGIT_MSE=${13-0.0}

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting AutoEncoder Training (Reverse KL, On-the-fly)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH}"
echo "   Output: ${OUTPUT_DIR}"
echo "   Latent Dim: ${LATENT_DIM}"
echo "   Val Samples: ${VAL_SAMPLES}"
echo "   Batch Size: ${BATCH_SIZE} per GPU"
echo "   Epochs: ${EPOCHS}"
echo ""

mkdir -p ${OUTPUT_DIR}

accelerate launch ${BASE_PATH}/scripts/AE/train/train_ae_rkl_onthefly.py \
    --data_path ${DATA_PATH} \
    --teacher_path ${TEACHER_MODEL} \
    --latent_dim ${LATENT_DIM} \
    --val_samples ${VAL_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --patience ${PATIENCE} \
    --max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --alpha_mse ${ALPHA_MSE} \
    --alpha_cosine ${ALPHA_COSINE} \
    --alpha_logit ${ALPHA_LOGIT} \
    --alpha_logit_mse ${ALPHA_LOGIT_MSE}

echo ""
echo "✅ AutoEncoder (Reverse KL, on-the-fly) training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_rkl_ld${LATENT_DIM}.pt"
