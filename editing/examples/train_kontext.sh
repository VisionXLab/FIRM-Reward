#!/usr/bin/env bash
set -euo pipefail

export REWARD_SERVER="${REWARD_SERVER:-127.0.0.1:12341}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
CONFIG_NAME="${CONFIG_NAME:-kontext_mllm_reward}"
CONFIG_FILE="${CONFIG_FILE:-config/kontext_nft_qwen3vl_8b.py}"

torchrun --nproc_per_node="${N_GPUS:-8}" \
  scripts/train_nft_kontext.py \
  --config "${CONFIG_FILE}:${CONFIG_NAME}"
