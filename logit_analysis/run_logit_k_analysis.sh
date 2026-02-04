#!/bin/bash
# =============================================================================
# Teacher Logit K Analysis (Distributed)
# =============================================================================
# Teacher의 softmax 확률 분포에서 누적확률 0.99, 0.999를 달성하는
# 최소 K를 분석합니다.
#
# 사용법:
#   ./run_logit_k_analysis.sh          # Single GPU (default)
#   ./run_logit_k_analysis.sh multi    # Multi-GPU (8 GPUs)
#
# 출력:
#   - analysis_report.txt: 요약 리포트
#   - aggregate_stats.json: 집계 통계 (JSON)
#   - coverage_rate_0.99.json: 고정 K(1000,2000,5000,10000) coverage rate 요약
#   - k_distribution.png: 분포 시각화
#   - raw_stats.npz: 원시 데이터 (추가 분석용)
# =============================================================================

# Paths
BASE_PATH="/home/jiwonyoon/data1/projects/MiniPLM"
TEACHER_MODEL="/home/jiwonyoon/data1/checkpoints/qwen/7B"
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
OUTPUT_DIR="${BASE_PATH}/logit_analysis/results/qwen_7B"

# Common parameters
MAX_LENGTH=1024
TEMPERATURE=1.0
DTYPE="bf16"
NUM_WORKERS=4
PREFETCH_FACTOR=2

# Environment
export PYTHONPATH=${BASE_PATH}:${PYTHONPATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Check argument
MODE=${1:-"single"}

if [ "$MODE" == "multi" ]; then
    # ===========================================
    # Multi-GPU Mode (8 GPUs)
    # ===========================================
    NUM_GPUS=8
    MAX_SAMPLES=100000
    BATCH_SIZE=64
    
    echo "=============================================="
    echo "Teacher Logit K Analysis (Multi-GPU: ${NUM_GPUS})"
    echo "=============================================="
    echo "Teacher Model: ${TEACHER_MODEL}"
    echo "Data Dir: ${DATA_DIR}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "Max Samples: ${MAX_SAMPLES}"
    echo "Batch Size: ${BATCH_SIZE} (per GPU)"
    echo "Num Workers: ${NUM_WORKERS}"
    echo "=============================================="
    
    mkdir -p ${OUTPUT_DIR}
    
    torchrun --nproc_per_node=${NUM_GPUS} \
        ${BASE_PATH}/logit_analysis/logit_k_analysis.py \
        --model-path ${TEACHER_MODEL} \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --max-samples ${MAX_SAMPLES} \
        --batch-size ${BATCH_SIZE} \
        --max-length ${MAX_LENGTH} \
        --temperature ${TEMPERATURE} \
        --dtype ${DTYPE} \
        --num-workers ${NUM_WORKERS} \
        --prefetch-factor ${PREFETCH_FACTOR} \
        --distributed

else
    # ===========================================
    # Single GPU Mode
    # ===========================================
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    MAX_SAMPLES=10000
    BATCH_SIZE=32
    
    echo "=============================================="
    echo "Teacher Logit K Analysis (Single GPU)"
    echo "=============================================="
    echo "Teacher Model: ${TEACHER_MODEL}"
    echo "Data Dir: ${DATA_DIR}"
    echo "Output Dir: ${OUTPUT_DIR}"
    echo "Max Samples: ${MAX_SAMPLES}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Num Workers: ${NUM_WORKERS}"
    echo "=============================================="
    
    mkdir -p ${OUTPUT_DIR}
    
    python ${BASE_PATH}/logit_analysis/logit_k_analysis.py \
        --model-path ${TEACHER_MODEL} \
        --data-dir ${DATA_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --max-samples ${MAX_SAMPLES} \
        --batch-size ${BATCH_SIZE} \
        --max-length ${MAX_LENGTH} \
        --temperature ${TEMPERATURE} \
        --dtype ${DTYPE} \
        --num-workers ${NUM_WORKERS} \
        --prefetch-factor ${PREFETCH_FACTOR}
fi

echo ""
echo "=============================================="
echo "Analysis completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
