BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/convert_tokenization.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name pile_converted \
    --old-model-type qwen \
    --old-model-path $BASE_PATH/checkpoints/qwen/500M \
    --model-type mamba \
    --model-path $BASE_PATH/checkpoints/mamba/130M \
    --data-dir $BASE_PATH/processed_data/pretrain/pile/qwen-1025 \
    --save $BASE_PATH/processed_data/pretrain/ \
    --max-length 1025 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --chunk-num-per-shard 1000000 \
