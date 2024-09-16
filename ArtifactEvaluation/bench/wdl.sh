#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_optionf

CUDA_VISIBLE_DEVICES=1 \
python ../main.py \
--model wdl \
--use_gpu \
--embedding_dim=128 \
--max_ind_range=40000000 \
--dataset=criteotb \
--learning_rate=0.1 \
--mini_batch_size=2048 \
--print_freq=2048 \
--test_freq=102400 \
--data_path="/home/zhl/criteo_24days" \
$dlrm_extra_option 2>&1 | tee wdl.log

echo "done"
