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
python $FILE \
--use-gpu \
--arch-sparse-feature-size=64 \
--arch-mlp-bot="13-512-256-64-64" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kdd12 \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1024 \
--test-freq=30000 \
--print-time \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--cat-path="$DIR/../datasets/kdd12/processed_sparse_sep.bin" \
--dense-path="$DIR/../datasets/kdd12/processed_dense.bin" \
--label-path="$DIR/../datasets/kdd12/processed_label.bin" \
--count-path="$DIR/../datasets/kdd12/processed_count.bin" \
$dlrm_extra_option 2>&1

echo "done"


# --sketch-flag \
# --notinsert-test \
# --compress-rate=0.005 \
# --hash-rate=0.8 \
# --sketch-threshold=500 \