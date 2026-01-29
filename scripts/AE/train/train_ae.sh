#!/bin/bash
# Train Conditional AutoEncoder for Knowledge Distillation
#
# 이 스크립트는 Teacher 모델의 Hidden States를 압축하는 Conditional AutoEncoder를 학습합니다.
#
# 주요 기능:
#   - Teacher 모델의 hidden states를 작은 latent space로 압축
#   - 다음 토큰 예측(y)을 condition으로 사용하여 더 나은 압축 성능 달성
#   - data_0.bin 파일의 모든 데이터 사용
#
# 사용 예시:
#   bash train_ae.sh
#   bash train_ae.sh /path/to/teacher/model
#   bash train_ae.sh /path/to/teacher/model 25

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# GPU 설정
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda:0"

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"  # data_0.bin만 사용
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/logit_only/layernorm"
PRE_EXTRACTED_PATH="${BASE_PATH}/data/hidden_states/data_0_hidden_states.pt"  # 미리 추출된 hidden states (없으면 자동 추출)

# Training parameters
LATENT_DIM=${3-40}              # Latent dimension (default: 25)
# TRAIN_SAMPLES는 전달하지 않아서 data_0의 모든 데이터 사용
VAL_SAMPLES=${4-500}           # Number of validation samples (default: 1000)
BATCH_SIZE=${5-512}             # Batch size (default: 256)
EPOCHS=${6-1}                   # Number of epochs (default: 1)
LR=${7-5e-4}                    # Learning rate (default: 5e-4)
PATIENCE=${8-10}                 # Early stopping patience (default: 10)
MAX_LENGTH=${9-1024}             # Max sequence length (default: 512)

# Loss weights
ALPHA_MSE=${10-0.0}             # Weight for MSE loss (default: 2.0)
ALPHA_COSINE=${11-0.0}          # Weight for cosine similarity loss (default: 0.0)
ALPHA_LOGIT=${12-1.0}           # Weight for logit KL divergence loss (default: 1.0)
ALPHA_LOGIT_MSE=${13-0.0}       # Weight for logit MSE loss (default: 0.0)

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting AutoEncoder Training..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH} (ALL samples)"
echo "   Output: ${OUTPUT_DIR}"
echo "   Latent Dim: ${LATENT_DIM}"
echo "   Train Samples: ALL (no limit)"
echo "   Val Samples: ${VAL_SAMPLES}"
echo "   Batch Size: ${BATCH_SIZE}"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Device: ${DEVICE}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if pre-extracted hidden states exist
if [ -f "${PRE_EXTRACTED_PATH}" ]; then
    echo "✅ Using pre-extracted hidden states: ${PRE_EXTRACTED_PATH}"
    PRE_EXTRACTED_ARG="--pre_extracted_path ${PRE_EXTRACTED_PATH}"
else
    echo "⚠️  Pre-extracted hidden states not found: ${PRE_EXTRACTED_PATH}"
    echo "   Will extract during training (slower)"
    PRE_EXTRACTED_ARG=""
fi

# Build command (--train_samples는 전달하지 않아서 모든 데이터 사용)
python ${BASE_PATH}/scripts/AE/train/train_ae.py \
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
echo "✅ AutoEncoder training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_y_ld${LATENT_DIM}.pt"
