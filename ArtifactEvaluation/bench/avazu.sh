#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FILE="$DIR/../dlrm_s_pytorch.py"

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python $FILE \
--use-gpu \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-set=avazu \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1024 \
--test-freq=30000 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--data_path="$DIR/../datasets/avazu" \
$dlrm_extra_option 2>&1

echo "done"

