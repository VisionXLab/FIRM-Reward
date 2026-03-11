#!/usr/bin/env bash
set -euo pipefail

export REWARD_SERVER="${REWARD_SERVER:-127.0.0.1:12341}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-qwen_mllm_reward_16g}"

torchrun --nproc_per_node="${N_GPUS:-8}" \
  scripts/train_nft_qwen_image_edit.py \
  --config "config/qwen_image_edit_nft.py:${CONFIG_NAME}"
