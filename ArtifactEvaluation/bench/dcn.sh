#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_optionf

CUDA_VISIBLE_DEVICES=0 \
python ../main.py \
--model dcn \
--use-gpu \
--embedding_dim=128 \
--max-ind-range=40000000 \
--data-set=criteotb \
--learning-rate=0.1 \
--mini-batch-size=2048 \
--print-freq=2048 \
--print-time \
--test-freq=102400 \
--test-mini-batch-size=16384 \
--data_path="/home/zhl/criteo_24days" \
--test-num-workers=16 \
$dlrm_extra_option 2>&1 | tee dcn.log

echo "done"
