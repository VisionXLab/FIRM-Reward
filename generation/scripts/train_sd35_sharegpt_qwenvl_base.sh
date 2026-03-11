#!/usr/bin/env bash
set -euo pipefail

# Start reward servers (on each machine) before running this script:
#   python flow_grpo/reward_model_server_base.py
#
# Multi-endpoint routing:
# - QWEN_VL_REWARD_URLS supports comma-separated endpoints.
# - QWEN_VL_REWARD_ROUTING supports: random / round_robin / p2c.
# - random is the default for 3-machine load sharing.
export QWEN_VL_REWARD_BATCH="${QWEN_VL_REWARD_BATCH:-64}"
export N_GPUS="${N_GPUS:-8}"

export QWEN_VL_REWARD_URLS="${QWEN_VL_REWARD_URLS:-http://127.0.0.1:12341}"
# Backward compatibility: downstream still reads QWEN_VL_REWARD_URL.
export QWEN_VL_REWARD_URL="${QWEN_VL_REWARD_URL:-${QWEN_VL_REWARD_URLS}}"
export QWEN_VL_REWARD_ROUTING="${QWEN_VL_REWARD_ROUTING:-random}"

# Throughput tuning for 3 reward endpoints.
export REWARD_CLIENT_WORKERS="${REWARD_CLIENT_WORKERS:-24}"
export REWARD_MAX_PENDING="${REWARD_MAX_PENDING:-3}"
export QWEN_VL_REWARD_POOL_CONNECTIONS="${QWEN_VL_REWARD_POOL_CONNECTIONS:-192}"
export QWEN_VL_REWARD_POOL_MAXSIZE="${QWEN_VL_REWARD_POOL_MAXSIZE:-192}"
export QWEN_VL_REWARD_TIMEOUT="${QWEN_VL_REWARD_TIMEOUT:-1800}"

# Reward blend mode in trainer-side postprocess:
#   legacy: keep server default reward (current behavior)
#   ins: use instruction score only
#   mix: reward=ins*(0.4+0.6*quality), where quality=normalize(min(q1,q2,q3), [1,5])
export QWEN_VL_REWARD_BLEND_MODE="${QWEN_VL_REWARD_BLEND_MODE:-mix}"
export TRAIN_DATA_JSON="${TRAIN_DATA_JSON:-data/sharegpt/text_to_image.json}"
export NFT_LOG_ROOT="${NFT_LOG_ROOT:-logs/nft/mix_reward_base}"
export EXP_NAME="${EXP_NAME:-sd35_sharegpt_qwenvl_flowgrpo_base}"

torchrun --nproc_per_node="${N_GPUS}" \
  scripts/train_nft_sd3.py \
  --config config/nft.py:sd35_sharegpt_qwenvl_flowgrpo_base
