#! /bin/bash

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export MASTER_ADDR=192.168.129.92
export MASTER_PORT=29600

# =========================
# paths
# =========================
BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
GPUS_PER_NODE=8
NNODES=1

DISTRIBUTED_ARGS="--num_gpus ${GPUS_PER_NODE} \
                  --num_nodes ${NNODES} \
                  --master_port ${MASTER_PORT}"


TYPE="pretrain"
CKPT_NAME="qwen/500M"
CKPT="/home/jiwonyoon/data1/checkpoints/${CKPT_NAME}"
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
DATA_NAME="miniplm_refined_corpus"
WANDB_NAME="qwen500m-pretrain-from-scratch"
SAVE_PATH="${BASE_PATH}/results/${TYPE}/qwen_500m"

BATCH_SIZE=32
GRAD_ACC=2

LR=6e-4
LR_MIN=6e-5

MAX_LENGTH=1024
SEED=10

# =========================
# options
# =========================
OPTS=""
OPTS+=" --type ${TYPE}"
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --from-scratch"

# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --bin-data"
OPTS+=" --num-workers 8"

# optimization
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

# runtime
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --do-train"
OPTS+=" --save-interval 10000"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-eval-when-start"
OPTS+=" --seed ${SEED}"

# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

# wandb
OPTS+=" --wandb-group pretrain_scratch"
OPTS+=" --wandb-name ${WANDB_NAME}"

# =========================
# env
# =========================
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=3

CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD}
