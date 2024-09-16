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
FILE="$DIR/../main.py"

CUDA_VISIBLE_DEVICES=0 \
CUDA_LAUNCH_BLOCKING=1 \
python $FILE \
--use_gpu \
--embedding_dim=16 \
--dataset=avazu \
--learning_rate=0.1 \
--mini_batch_size=128 \
--print_freq=1024 \
--test_freq=30000 \
--data_path="$DIR/../datasets/avazu" \
$dlrm_extra_option 2>&1

echo "done"

