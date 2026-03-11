#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen-VL reward server separately before running this script:
#   python -m flow_grpo.qwen_vl_reward_server

export N_GPUS="${N_GPUS:-8}"
export NUM_GROUPS="${NUM_GROUPS:-32}"
export NUM_IMAGE_PER_PROMPT="${NUM_IMAGE_PER_PROMPT:-16}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export EXP_NAME="${EXP_NAME:-qwen_image_diffusionnft_qwenvl_8gpu}"

export QWEN_IMAGE_MODEL_PATH="${QWEN_IMAGE_MODEL_PATH:-Qwen/Qwen-Image}"
export QWEN_IMAGE_DATA_JSON="${QWEN_IMAGE_DATA_JSON:-data/sharegpt/text_to_image.json}"

# WandB offline + requested storage layout
export WANDB_MODE="offline"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
export LOG_DIR="${LOG_DIR:-logs/wandb/${EXP_NAME}}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Reward parallelism
export REWARD_CLIENT_WORKERS="${REWARD_CLIENT_WORKERS:-16}"
export QWEN_VL_REWARD_URL="${QWEN_VL_REWARD_URL:-http://127.0.0.1:12341}"
export QWEN_VL_REWARD_TIMEOUT="${QWEN_VL_REWARD_TIMEOUT:-1800}"
export QWEN_VL_REWARD_RETRIES="${QWEN_VL_REWARD_RETRIES:-10}"
export QWEN_VL_REWARD_BACKOFF="${QWEN_VL_REWARD_BACKOFF:-2}"
export QWEN_VL_REWARD_BATCH="${QWEN_VL_REWARD_BATCH:-24}"
export QWEN_VL_REWARD_POOL_CONNECTIONS="${QWEN_VL_REWARD_POOL_CONNECTIONS:-128}"
export QWEN_VL_REWARD_POOL_MAXSIZE="${QWEN_VL_REWARD_POOL_MAXSIZE:-128}"


torchrun \
  --nproc_per_node="${N_GPUS}" \
  scripts/train_nft_qwen_image.py \
  --config=config/nft_qwen_image.py:qwen_image_sharegpt_qwenvl \
  --config.pretrained.model="${QWEN_IMAGE_MODEL_PATH}" \
  --config.dataset_json_path="${QWEN_IMAGE_DATA_JSON}" \
  --config.output_dir="${OUTPUT_DIR}" \
  --config.log_dir="${LOG_DIR}" \
  --config.save_dir="${OUTPUT_DIR}" \
  --config.wandb_mode="offline"
