#!/bin/bash
# Train Conditional AutoEncoder with Reverse KL (KL(recon || teacher))
#
# train_ae_rkl.py 사용: Logit loss로 reverse KL divergence 사용
#
# 사용 예시:
#   bash train_ae_rkl.sh
#   bash train_ae_rkl.sh /path/to/teacher/model
#   bash train_ae_rkl.sh /path/to/teacher/model 25

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# GPU 설정
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda:0"

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/logit_only/layernorm_rkl"
PRE_EXTRACTED_PATH="${BASE_PATH}/data/hidden_states/data_0_hidden_states.pt"

# Training parameters
LATENT_DIM=${3-40}
VAL_SAMPLES=${4-500}
BATCH_SIZE=${5-16}
EPOCHS=${6-1}
LR=${7-5e-4}
PATIENCE=${8-10}
MAX_LENGTH=${9-1024}

# Loss weights
ALPHA_MSE=${10-0.0}
ALPHA_COSINE=${11-0.0}
ALPHA_LOGIT=${12-1.0}           # Weight for logit reverse KL loss (default: 1.0)
ALPHA_LOGIT_MSE=${13-0.0}

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting AutoEncoder Training (Reverse KL)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH}"
echo "   Output: ${OUTPUT_DIR}"
echo "   Latent Dim: ${LATENT_DIM}"
echo "   Val Samples: ${VAL_SAMPLES}"
echo "   Batch Size: ${BATCH_SIZE}"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Device: ${DEVICE}"
echo ""

mkdir -p ${OUTPUT_DIR}

if [ -f "${PRE_EXTRACTED_PATH}" ]; then
    echo "✅ Using pre-extracted hidden states: ${PRE_EXTRACTED_PATH}"
    PRE_EXTRACTED_ARG="--pre_extracted_path ${PRE_EXTRACTED_PATH}"
else
    echo "⚠️  Pre-extracted hidden states not found: ${PRE_EXTRACTED_PATH}"
    PRE_EXTRACTED_ARG=""
fi

python ${BASE_PATH}/scripts/AE/train/train_ae_rkl.py \
    --data_path ${DATA_PATH} \
    --teacher_path ${TEACHER_MODEL} \
    --latent_dim ${LATENT_DIM} \
    --val_samples ${VAL_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --patience ${PATIENCE} \
    --device ${DEVICE} \
    --max_length ${MAX_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --alpha_mse ${ALPHA_MSE} \
    --alpha_cosine ${ALPHA_COSINE} \
    --alpha_logit ${ALPHA_LOGIT} \
    --alpha_logit_mse ${ALPHA_LOGIT_MSE} \
    ${PRE_EXTRACTED_ARG}

echo ""
echo "✅ AutoEncoder (Reverse KL) training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_y_ld${LATENT_DIM}.pt"
