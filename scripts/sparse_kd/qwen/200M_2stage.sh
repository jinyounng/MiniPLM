#! /bin/bash
# Sparse KD 2stage: offline_kd (sparse) initialized from 1stage AE-KD checkpoint
#
# Cached teacher logits + Sparse loss, student = 1stage epoch_1

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo
export MASTER_ADDR=192.168.129.92
BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
MASTER_PORT=${2-29702}
GPUS_PER_NODE=${3-8}
export MASTER_PORT
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT"

# type
TYPE="offline_kd"
# model: 2stage = Sparse KD initialized from 1stage AE-KD checkpoint
CKPT_NAME="qwen/200M-2stage-sparse-from-1stage"
CKPT="${BASE_PATH}/results/AE/kd/1stage_ld40_5e-4/epoch_1"
# cached logits (Sparse 사용)
CACHED_LOGITS_DIR="${CACHED_LOGITS_DIR:-/home/jiwonyoon/data1/data/logits_hdf5}"
KD_METHOD="sparse"
# data
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
DATA_NAME="miniplm_refined_corpus"
WANDB_NAME="200M-sparse-kd-2stage-from-1stage"
# hp (vanilla_kd 2stage와 동일)
BATCH_SIZE=32
LR=0.0006
LR_MIN=0.00006
GRAD_ACC=2
ALPHA=0.5
KD_TEMPERATURE=1.0
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}/sparse_kd"
SEED=10

OPTS=""
OPTS+=" --type ${TYPE}"
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
# Load from 1stage (do NOT use --from-scratch)
# OPTS+=" --from-scratch"
# offline KD (Sparse)
OPTS+=" --cached-logits-dir ${CACHED_LOGITS_DIR}"
OPTS+=" --kd-method ${KD_METHOD}"
OPTS+=" --alpha ${ALPHA}"
OPTS+=" --kd-temperature ${KD_TEMPERATURE}"
# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --bin-data"
OPTS+=" --no-shuffle"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-min ${LR_MIN}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 2000"
OPTS+=" --scheduler-name cosine"
OPTS+=" --weight-decay 0.1"
OPTS+=" --clip-grad 1.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.98"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 100000"
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --do-train"
OPTS+=" --save-interval 20000"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-eval-when-start"
OPTS+=" --seed ${SEED}"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --wandb-group offline_kd_sparse"
OPTS+=" --wandb-name ${WANDB_NAME}"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
