#!/bin/bash
# Extract Hidden States from Teacher Model
#
# Teacher 모델로부터 hidden states를 미리 추출하여 저장합니다.
# 저장된 파일은 train_ae.py에서 --pre_extracted_path로 사용할 수 있습니다.

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# GPU 설정 (8개 사용)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"
OUTPUT_PATH="${BASE_PATH}/data/hidden_states/data_0_hidden_states.pt"

# Extraction parameters
MAX_LENGTH=${3-1024}            # Max sequence length (default: 1024)
BATCH_SIZE=${4-160}               # Batch size (default: 32)
MAX_SAMPLES=${5-}                # Maximum samples (default: all, leave empty)
START_IDX=${6-0}                 # Starting index (default: 0)
CHECKPOINT_INTERVAL=${7-100}     # Checkpoint every N batches (default: 100, 0 to disable)

export PYTHONPATH=${BASE_PATH}

echo "🚀 Extracting Hidden States (Multi-GPU)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH}"
echo "   Output: ${OUTPUT_PATH}"
echo "   GPUs: ${NUM_GPUS}"
echo "   Max Length: ${MAX_LENGTH}"
echo "   Batch Size: ${BATCH_SIZE}"
if [ -n "${MAX_SAMPLES}" ]; then
    echo "   Max Samples: ${MAX_SAMPLES}"
fi
echo "   Checkpoint Interval: ${CHECKPOINT_INTERVAL} batches"
echo ""

# Create output directory
mkdir -p $(dirname ${OUTPUT_PATH})

python ${BASE_PATH}/scripts/AE/train/extract_hidden_states.py \
    --data_path ${DATA_PATH} \
    --teacher_path ${TEACHER_MODEL} \
    --output_path ${OUTPUT_PATH} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --num_gpus ${NUM_GPUS} \
    ${MAX_SAMPLES:+--max_samples ${MAX_SAMPLES}} \
    --start_idx ${START_IDX} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL}

echo ""
echo "✅ Extraction completed!"
echo "   Output: ${OUTPUT_PATH}"
if [ "${CHECKPOINT_INTERVAL}" -gt 0 ]; then
    CHECKPOINT_FILE="${OUTPUT_PATH%.pt}_checkpoint.pt"
    if [ -f "${CHECKPOINT_FILE}" ]; then
        echo "   Checkpoint: ${CHECKPOINT_FILE}"
    fi
fi
echo ""
echo "Usage in train_ae.py:"
echo "  --pre_extracted_path ${OUTPUT_PATH}"
