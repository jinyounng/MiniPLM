#!/bin/bash
BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
TEACHER_MODEL=${2-"/home/jiwonyoon/data1/checkpoints/qwen/7B"}

# Data paths
DATA_PATH="/home/jiwonyoon/data1/data/pile_dataset/data_0"  # data_0.bin만 사용
OUTPUT_DIR="${BASE_PATH}/checkpoints/AE/condition_RVQ"

# RVQ Model parameters
NUM_STAGES=${3-25}             # Number of quantization stages (default: 25)
NUM_CODES=${4-4096}            # Codebook size per stage (default: 1024)
COMPRESSED_DIM=${5-1024}       # Compressed dimension (default: 1024)
GAMMA=${6-0.90}                # EMA decay rate (default: 0.99)
G_TAG="g$(echo ${GAMMA} | tr '.' 'p')"   # for save filename (0.99 -> g0p99)

# Training parameters
# TRAIN_SAMPLES는 전달하지 않아서 data_0의 모든 데이터 사용
VAL_SAMPLES=${7-1000}          # Number of validation sequences (default: 1000)
BATCH_SIZE=${8-64}             # Batch size per GPU (default: 16, total = 16 * 8 = 128)
EPOCHS=${9-1}                 # Number of epochs (default: 10)
LR=${10-5e-4}                  # Learning rate (default: 1e-4)
PATIENCE=${11-10}               # Early stopping patience (default: 10)
MAX_LENGTH=${12-1024}          # Max sequence length (default: 1024)
COMMITMENT_WEIGHT=${13-0.25}   # Commitment loss weight (default: 0.25)
THRESHOLD_EMA_DEAD_CODE=${14-2}  # Dead code threshold (default: 10, recommended: batch_size * 0.5 to batch_size)
PROGRESSIVE_TRAINING=${15-""}  # Progressive training flag (set to "--progressive_training" to enable)
PERPLEXITY_PERIOD=${16-0}      # Perplexity compute period (0 = never during training, only eval)
ENCODER_DEPTH=${17-4}          # Number of encoder blocks (default: 3)
DECODER_DEPTH=${18-4}          # Number of decoder blocks (default: 3)

export PYTHONPATH=${BASE_PATH}

echo "🚀 Starting RVQ Compressor Training with Embedding Conditioning (DDP Optimized Mode)..."
echo "   Teacher: ${TEACHER_MODEL}"
echo "   Data: ${DATA_PATH} (ALL samples)"
echo "   Output: ${OUTPUT_DIR}"
echo "   RVQ Config:"
echo "     - Stages: ${NUM_STAGES}"
echo "     - Codes/Stage: ${NUM_CODES}"
echo "     - Gamma (EMA decay): ${GAMMA}"
echo "     - Compressed Dim: ${COMPRESSED_DIM}"
echo "     - Encoder Depth: ${ENCODER_DEPTH}"
echo "     - Decoder Depth: ${DECODER_DEPTH}"
echo "     - Compression: ~$((NUM_STAGES * 10)) bits per sample"
echo "     - Save name: best_rvq_emb_s${NUM_STAGES}_c${NUM_CODES}_${G_TAG}_d${COMPRESSED_DIM}_enc${ENCODER_DEPTH}_dec${DECODER_DEPTH}.pt"
echo "   Conditioning:"
echo "     - Encoder input: [Y_emb + hidden] → compressed"
echo "     - Decoder input: [Z + Y_emb] → hidden"
echo "   Train Samples: ALL (no limit)"
echo "   Val Samples: ${VAL_SAMPLES} sequences"
echo "   Batch Size: ${BATCH_SIZE} per GPU (total = ${BATCH_SIZE} * 8 = $((BATCH_SIZE * 8)))"
echo "   Epochs: ${EPOCHS}"
echo "   LR: ${LR}"
echo "   Max Length: ${MAX_LENGTH}"
echo "   Commitment Weight: ${COMMITMENT_WEIGHT}"
echo "   Mode: DDP with Accelerate (8 GPUs)"
echo "   Optimization: Sequence-level batching"
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
accelerate launch ${BASE_PATH}/scripts/AE/train/train_RVQ_emb.py \
    --data_path ${DATA_PATH} \
    --teacher_path ${TEACHER_MODEL} \
    --num_stages ${NUM_STAGES} \
    --num_codes ${NUM_CODES} \
    --compressed_dim ${COMPRESSED_DIM} \
    --gamma ${GAMMA} \
    --val_samples ${VAL_SAMPLES} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --patience ${PATIENCE} \
    --max_length ${MAX_LENGTH} \
    --commitment_weight ${COMMITMENT_WEIGHT} \
    --threshold_ema_dead_code ${THRESHOLD_EMA_DEAD_CODE} \
    --encoder_depth ${ENCODER_DEPTH} \
    --decoder_depth ${DECODER_DEPTH} \
    ${PROGRESSIVE_TRAINING} \
    --perplexity_compute_period ${PERPLEXITY_PERIOD} \
    --output_dir ${OUTPUT_DIR}

echo ""
echo "✅ RVQ Compressor training with embedding conditioning completed!"
echo "   Output: ${OUTPUT_DIR}"
echo "   Model: best_rvq_emb_s${NUM_STAGES}_c${NUM_CODES}_${G_TAG}_d${COMPRESSED_DIM}_enc${ENCODER_DEPTH}_dec${DECODER_DEPTH}.pt"
