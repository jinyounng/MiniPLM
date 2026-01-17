#! /bin/bash

BASE_PATH=${1-"/home/MiniPLM"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-16}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# type
TYPE="pt_lm_infer"
# model
CKPT_NAME="qwen_104M"
CKPT="${BASE_PATH}/checkpoints/qwen/104M_ref"
# data
DATA_DIR="${BASE_PATH}/processed_data/pretrain/pile_ref/qwen-1025"
# hp
EVAL_BATCH_SIZE=16
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --model-type qwen"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# data
OPTS+=" --data-name pile"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --bin-data"
OPTS+=" --data-split data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-infer"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-interval 10"
OPTS+=" --save-interval 2500"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type ${TYPE}"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/inference.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
