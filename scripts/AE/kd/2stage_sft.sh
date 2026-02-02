#! /bin/bash
# 2-stage SFT: Continue LM training from 1stage feature-matching checkpoint (next-token prediction).
# Structure follows scripts/pretrain/qwen/200M.sh
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo
export MASTER_ADDR=192.168.129.92
export MASTER_PORT=29500

BASE_PATH=${1-"/home/jiwonyoon/data1/projects/MiniPLM"}
MASTER_PORT=${2-29500}
GPUS_PER_NODE=${3-8}
NNODES=1

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT "

# type
TYPE="pretrain"
# model: load from 1stage FM checkpoint (no --from-scratch)
CKPT_NAME="qwen/200M-2stage-sft"
CKPT="${BASE_PATH}/results/AE/kd/1stage_ld40_5e-4/epoch_1"
# Tokenizer: 1stage ckpt has no tokenizer files; use original student config path (must have tokenizer.json etc.)
TOKENIZER_PATH="/home/jiwonyoon/data1/checkpoints/qwen/200M"
# or use a fixed path: CKPT="/home/jiwonyoon/data1/checkpoints/qwen/200M/"
# data
DATA_DIR="/home/jiwonyoon/data1/data/pile_dataset"
DATA_NAME="miniplm_refined_corpus"
WANDB_NAME="200M-2stage-sft"
# hp
BATCH_SIZE=64
LR=0.0006
LR_MIN=0.00006
GRAD_ACC=1
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10

OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
# load from 1stage ckpt (do NOT use --from-scratch)
# OPTS+=" --from-scratch"
# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --bin-data"
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
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --save-interval 10000"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --no-eval-when-start"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# wandb
OPTS+=" --wandb-group 2stage_sft"
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
