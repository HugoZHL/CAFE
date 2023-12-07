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

CUDA_VISIBLE_DEVICES=0 \
python dlrm_s_pytorch.py \
--use-gpu \
--arch-sparse-feature-size=128 \
--arch-mlp-bot="13-512-256-128" \
--arch-mlp-top="1024-1024-512-256-1" \
--max-ind-range=40000000 \
--data-generation=dataset \
--data-set=terabyte \
--loss-function=bce \
--round-targets=True \
--learning-rate=1.0 \
--mini-batch-size=2048 \
--print-freq=2048 \
--print-time \
--test-freq=102400 \
--test-mini-batch-size=16384 \
--cat-path="/path/to/data" \
--dense-path="/path/to/data" \
--label-path="/path/to/data" \
--test-num-workers=16 \
$dlrm_extra_option 2>&1 | tee terabyte.log

echo "done"
