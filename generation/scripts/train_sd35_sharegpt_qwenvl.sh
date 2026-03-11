#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen-VL reward server separately before running this script:
#   python -m flow_grpo.qwen_vl_reward_server

export N_GPUS="${N_GPUS:-8}"

torchrun --nproc_per_node="${N_GPUS}" \
  scripts/train_nft_sd3.py \
  --config config/nft.py:sd3_sharegpt_qwenvl
