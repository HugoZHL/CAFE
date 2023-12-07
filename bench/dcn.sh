#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_optionf

python dcn.py \
--use-gpu \
--arch-sparse-feature-size=128 \
--max-ind-range=40000000 \
--data-generation=dataset \
--data-set=terabyte \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=2048 \
--print-freq=2048 \
--print-time \
--test-freq=102400 \
--test-mini-batch-size=16384 \
--cat-path="../criteo_24days/sparse" \
--dense-path="../criteo_24days/dense" \
--label-path="../criteo_24days/label" \
--count-path="../criteo_24days/processed_count.bin" \
--test-num-workers=16 \
$dlrm_extra_option 2>&1 | tee dcn.log

echo "done"
