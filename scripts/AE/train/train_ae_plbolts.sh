#!/bin/bash
# Train Conditional AutoEncoder using pl_bolts wrapper
#
# 핵심:
#   - pl_bolts의 AE를 그대로 사용 (wrapper)
#   - 입력을 [x, y_emb]로 concatenate
#   - encoder/decoder에 y 넣는 건 스위치로 토글 가능
#
# 사용 예시:
#   bash train_ae_plbolts.sh
#   bash train_ae_plbolts.sh /path/to/teacher/model
#   bash train_ae_plbolts.sh /path/to/teacher/model 25

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/plbolts_wrapper"

# Model parameters
LATENT_DIM=${3-25}              # Latent dimension (default: 25)
USE_Y_IN_ENCODER=${4-true}      # Use y_emb in encoder input (default: true)
USE_Y_IN_DECODER=${5-true}      # Use y_emb in decoder input (default: true)

# Training parameters
VAL_SAMPLES=${6-1000}           # Number of validation sequences (default: 1000)
BATCH_SIZE=${7-16}              # Batch size per GPU (default: 16)
EPOCHS=${8-1}                   # Number of epochs (default: 1)
LR=${9-5e-4}                    # Learning rate (default: 5e-4)
PATIENCE=${10-10}               # Early stopping patience (default: 10)
MAX_LENGTH=${11-1024}           # Max sequence length (default: 1024)

# Loss weights
ALPHA_MSE=${12-0.0}             # Weight for MSE loss (default: 0.0)
ALPHA_COSINE=${13-0.0}          # Weight for cosine similarity loss (default: 0.0)
ALPHA_LOGIT=${14-1.0}           # Weight for logit KL divergence loss (default: 1.0)
ALPHA_LOGIT_MSE=${15-0.0}       # Weight for logit MSE loss (default: 0.0)

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting pl_bolts AE Wrapper Training (DDP Optimized Mode)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH} (ALL samples)"
echo "   Output: ${OUTPUT_DIR}"
echo "   Latent Dim: ${LATENT_DIM}"
echo "   Use Y in Encoder: ${USE_Y_IN_ENCODER}"
echo "   Use Y in Decoder: ${USE_Y_IN_DECODER}"
echo "   Train Samples: ALL (no limit)"
echo "   Val Samples: ${VAL_SAMPLES} sequences"
echo "   Batch Size: ${BATCH_SIZE} per GPU (total = ${BATCH_SIZE} * 8 = $((BATCH_SIZE * 8)))"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Max Length: ${MAX_LENGTH}"
echo "   Mode: DDP with Accelerate (8 GPUs)"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "⚠️  accelerate not found. Installing..."
    pip install accelerate
fi

# Build flags for y usage
ENCODER_FLAG=""
DECODER_FLAG=""
if [[ "${USE_Y_IN_ENCODER}" == "true" ]]; then
    ENCODER_FLAG="--use_y_in_encoder"
fi
if [[ "${USE_Y_IN_DECODER}" == "true" ]]; then
    DECODER_FLAG="--use_y_in_decoder"
fi

# Build command using accelerate launch
accelerate launch ${BASE_PATH}/scripts/AE/train/train_ae_plbolts.py \
    --data_path ${DATA_PATH} \
    --teacher_path ${TEACHER_MODEL} \
    --latent_dim ${LATENT_DIM} \
    ${ENCODER_FLAG} \
    ${DECODER_FLAG} \
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
echo "✅ pl_bolts AE Wrapper training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_plbolts_ld${LATENT_DIM}_enc${USE_Y_IN_ENCODER}_dec${USE_Y_IN_DECODER}.pt"
