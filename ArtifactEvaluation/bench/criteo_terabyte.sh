#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_optionf
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FILE="$DIR/../main.py"

CUDA_VISIBLE_DEVICES=0 \
python $FILE \
--use-gpu \
--embedding_dim=128 \
--arch-mlp-bot="13-512-256-128" \
--arch-mlp-top="1024-1024-512-256-1" \
--max-ind-range=40000000 \
--data-set=criteotb \
--learning-rate=1.0 \
--mini-batch-size=2048 \
--print-freq=2048 \
--print-time \
--test-freq=102400 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--compress-rate=0.001 \
--hash-flag \
--data_path="/home/zhl/criteo_24days" \
$dlrm_extra_option 2>&1

echo "done"
