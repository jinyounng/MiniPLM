#!/bin/bash

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/zonly"

# Model parameters
LATENT_DIM=${3-40}              # Latent dimension (default: 40)

# Training parameters
VAL_SAMPLES=${4-1000}           # Number of validation sequences (default: 1000)
BATCH_SIZE=${5-32}              # Batch size per GPU
EPOCHS=${6-1}                  # Number of epochs (default: 30)
LR=${7-5e-4}                    # Learning rate (default: 5e-4)
PATIENCE=${8-10}                # Early stopping patience (default: 10)
MAX_LENGTH=${9-1024}             # Max sequence length (default: 512)

# Loss weights
ALPHA_MSE=${10-0.0}             # MSE loss weight (default: 2.0)
ALPHA_COSINE=${11-0.0}          # Cosine loss weight (default: 0.0)
ALPHA_LOGIT=${12-1.0}           # Logit KL loss weight (default: 1.0)
ALPHA_LOGIT_MSE=${13-0.0}       # Logit MSE loss weight (default: 0.0)

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting AutoEncoder Training (Z-only Decoder)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH}"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model Config:"
echo "     - Latent Dim: ${LATENT_DIM}"
echo "     - Encoder: [hidden + Y_emb] → Z"
echo "     - Decoder: Z ONLY → hidden (Y_emb ignored)"
echo "   Train Samples: ALL (no limit)"
echo "   Val Samples: ${VAL_SAMPLES} sequences"
echo "   Batch Size: ${BATCH_SIZE} per GPU"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Max Length: ${MAX_LENGTH}"
echo "   Loss Weights: MSE=${ALPHA_MSE}, Cosine=${ALPHA_COSINE}, Logit=${ALPHA_LOGIT}, LogitMSE=${ALPHA_LOGIT_MSE}"
echo "   Mode: DDP with Accelerate"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if accelerate is installed
if ! command -v accelerate &> /dev/null; then
    echo "⚠️  accelerate not found. Installing..."
    pip install accelerate
fi

# Build command using accelerate launch
accelerate launch ${BASE_PATH}/scripts/AE/train/train_ae_zonly.py \
    --data_path ${DATA_PATH} \
    --teacher_path ${TEACHER_MODEL} \
    --latent_dim ${LATENT_DIM} \
    --val_samples ${VAL_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --patience ${PATIENCE} \
    --max_length ${MAX_LENGTH} \
    --alpha_mse ${ALPHA_MSE} \
    --alpha_cosine ${ALPHA_COSINE} \
    --alpha_logit ${ALPHA_LOGIT} \
    --alpha_logit_mse ${ALPHA_LOGIT_MSE} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "✅ AutoEncoder training (Z-only) completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_zonly_ld${LATENT_DIM}.pt"
