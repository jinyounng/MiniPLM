#!/bin/bash
# Train Conditional AutoEncoder with Top-K Logit Loss
#
# Logit loss를 teacher 상위 K개 로짓에만 적용 (나머지 버림).
# --topk_logit 5000 이면 top-5000 로짓만 사용.
#
# 사용 예시:
#   bash train_ae_top-k.sh                    # 기본 (full vocab 로짓 로스)
#   bash train_ae_top-k.sh "" "" "" "" "" "" "" "" "" "" "" "" "" 5000   # top-5000 로짓만
#   TOPK_LOGIT=5000 bash train_ae_top-k.sh     # 환경변수로 top-5000 지정
#
# 주의: accelerate config를 먼저 실행하여 설정해야 합니다.
#       accelerate config

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"

# Training parameters
LATENT_DIM=${3-40}
VAL_SAMPLES=${4-1000}
BATCH_SIZE=${5-16}
EPOCHS=${6-1}
LR=${7-5e-4}
PATIENCE=${8-10}
MAX_LENGTH=${9-1024}

# Loss weights
ALPHA_MSE=${10-0.0}
ALPHA_COSINE=${11-0.0}
ALPHA_LOGIT=${12-1.0}
ALPHA_LOGIT_MSE=${13-0.0}

# Top-K logit: 로짓 로스에 teacher 상위 K개만 사용. 비우면 full vocab.
# 예: 5000, 2000, 10000
TOPK_LOGIT=${14-5000}
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/topk_logit/layernorm/topk_${TOPK_LOGIT}"

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting AutoEncoder Training (Top-K Logit Loss)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH}"
echo "   Output: ${OUTPUT_DIR}"
echo "   Latent Dim: ${LATENT_DIM}"
echo "   Val Samples: ${VAL_SAMPLES}"
echo "   Batch Size: ${BATCH_SIZE} per GPU"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Max Length: ${MAX_LENGTH}"
echo "   Top-K Logit: ${TOPK_LOGIT:-full vocab}"
echo ""

mkdir -p ${OUTPUT_DIR}

if ! command -v accelerate &> /dev/null; then
    echo "⚠️  accelerate not found. Installing..."
    pip install accelerate
fi

CMD="accelerate launch ${BASE_PATH}/scripts/AE/train/train_ae_top-k.py \
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
    --alpha_logit_mse ${ALPHA_LOGIT_MSE} \
    --topk_logit ${TOPK_LOGIT}" 

if [ -n "${TOPK_LOGIT}" ]; then
    CMD="${CMD} --topk_logit ${TOPK_LOGIT}"
fi

eval ${CMD}

echo ""
echo "✅ AutoEncoder training completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_ae_ld${LATENT_DIM}.pt"
