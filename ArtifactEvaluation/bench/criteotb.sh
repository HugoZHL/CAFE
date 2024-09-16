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
--use_gpu \
--embedding_dim=128 \
--max_ind_range=40000000 \
--dataset=criteotb \
--learning_rate=1.0 \
--mini_batch_size=2048 \
--print_freq=2048 \
--test_freq=102400 \
--compress_rate=0.001 \
--data_path="/home/zhl/criteo_24days" \
$dlrm_extra_option 2>&1

echo "done"
