#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

CONFIG=$1
GPUS=$2
PORT=${PORT:-6666}

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main.py --config $CONFIG ${@:3}
