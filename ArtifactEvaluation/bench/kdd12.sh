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
python $FILE \
--use_gpu \
--embedding_dim=64 \
--arch_mlp_bot="13-512-256-64-64" \
--arch_mlp_top="512-256-1" \
--dataset=kdd12 \
--learning_rate=0.1 \
--mini_batch_size=128 \
--print_freq=1024 \
--test_freq=30000 \
--print_time \
--test_mini_batch_size=16384 \
--data_path="$DIR/../datasets/kdd12" \
$dlrm_extra_option 2>&1

echo "done"

