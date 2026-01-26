#!/bin/bash
# Train Conditional AutoEncoder for Knowledge Distillation (DDP Optimized Version)
#
# 이 스크립트는 Teacher 모델의 Hidden States를 압축하는 Conditional AutoEncoder를 학습합니다.
# **DDP & Sequence Batching**: Hugging Face Accelerate를 사용한 분산 학습
# **B200 최적화**: Sequence 단위 배치 처리로 Teacher 연산 효율 극대화
#
# 주요 기능:
#   - Teacher 모델의 hidden states를 작은 latent space로 압축
#   - 다음 토큰 예측(y)을 condition으로 사용하여 더 나은 압축 성능 달성
#   - Sequence 단위 배치 처리로 Teacher 연산 효율 500배 향상
#   - DDP를 통한 8개 GPU 완전 활용
#
# 사용 예시:
#   bash train_ae_onthefly.sh
#   bash train_ae_onthefly.sh /path/to/teacher/model
#   bash train_ae_onthefly.sh /path/to/teacher/model 25
#
# 주의: accelerate config를 먼저 실행하여 설정해야 합니다.
#       accelerate config

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"  # data_0.bin만 사용
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/logit_only/layernorm"

# Training parameters
LATENT_DIM=${3-25}              # Latent dimension (default: 25)
# TRAIN_SAMPLES는 전달하지 않아서 data_0의 모든 데이터 사용
VAL_SAMPLES=${4-1000}           # Number of validation sequences (default: 1000)
BATCH_SIZE=${5-32}              # Batch size per GPU (default: 32, total = 32 * 8 = 256)
EPOCHS=${6-1}                   # Number of epochs (default: 1)
LR=${7-5e-4}                    # Learning rate (default: 5e-4)
PATIENCE=${8-10}                 # Early stopping patience (default: 10)
MAX_LENGTH=${9-1024}            # Max sequence length (default: 1024)

# Loss weights
ALPHA_MSE=${10-0.0}             # Weight for MSE loss (default: 2.0)
ALPHA_COSINE=${11-0.0}          # Weight for cosine similarity loss (default: 0.0)
ALPHA_LOGIT=${12-1.0}           # Weight for logit KL divergence loss (default: 1.0)
ALPHA_LOGIT_MSE=${13-0.0}       # Weight for logit MSE loss (default: 0.0)

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting AutoEncoder Training (DDP Optimized Mode)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH} (ALL samples)"
echo "   Output: ${OUTPUT_DIR}"
echo "   Latent Dim: ${LATENT_DIM}"
echo "   Train Samples: ALL (no limit)"
echo "   Val Samples: ${VAL_SAMPLES} sequences"
echo "   Batch Size: ${BATCH_SIZE} per GPU (total = ${BATCH_SIZE} * 8 = $((BATCH_SIZE * 8)))"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Max Length: ${MAX_LENGTH}"
echo "   Mode: DDP with Accelerate (8 GPUs)"
echo "   Optimization: Sequence-level batching (500x faster teacher inference)"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "⚠️  accelerate not found. Installing..."
    pip install accelerate
fi

# Build command using accelerate launch
# Note: Accelerate config should be set up beforehand using 'accelerate config'
# Note: --device is removed as Accelerate manages devices automatically
accelerate launch ${BASE_PATH}/scripts/AE/train/train_ae_onthefly.py \
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
echo "✅ AutoEncoder training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_ld${LATENT_DIM}.pt"
