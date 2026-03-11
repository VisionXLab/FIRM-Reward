#!/usr/bin/env bash
set -euo pipefail

# Start the Qwen-VL reward server separately before running this script:
#   python -m flow_grpo.qwen_vl_reward_server

export N_GPUS="${N_GPUS:-8}"
export NUM_GROUPS="${NUM_GROUPS:-48}"
export NUM_IMAGE_PER_PROMPT="${NUM_IMAGE_PER_PROMPT:-12}"
export EXP_NAME="${EXP_NAME:-zimage_turbo_diffusionnft_sharegpt_8gpu_reward}"

# Model and dataset settings
export ZIMAGE_MODEL_ID_WITH_ORIGIN_PATHS="${ZIMAGE_MODEL_ID_WITH_ORIGIN_PATHS:-models/Z-Image-Turbo:transformer/*.safetensors,models/Z-Image-Turbo:text_encoder/*.safetensors,models/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors}"
export ZIMAGE_MODEL_PATHS="${ZIMAGE_MODEL_PATHS:-}"
export ZIMAGE_TOKENIZER_PATH="${ZIMAGE_TOKENIZER_PATH:-models/Z-Image-Turbo/tokenizer}"
export ZIMAGE_DATA_JSON="${ZIMAGE_DATA_JSON:-data/sharegpt/text_to_image.json}"

# WandB offline + requested storage layout
export WANDB_MODE="offline"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
export LOG_DIR="${LOG_DIR:-logs/wandb/${EXP_NAME}}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Qwen-VL reward server settings
export QWEN_VL_REWARD_URL="${QWEN_VL_REWARD_URL:-http://127.0.0.1:12341}"
export REWARD_MAX_PENDING="${REWARD_MAX_PENDING:-1}"
export QWEN_VL_REWARD_TIMEOUT="${QWEN_VL_REWARD_TIMEOUT:-1800}"

torchrun \
  --nproc_per_node="${N_GPUS}" \
  scripts/train_nft_zimage_turbo.py \
  --config=config/nft_zimage_turbo.py:zimage_turbo_sharegpt_qwenvl \
  --config.output_dir="${OUTPUT_DIR}" \
  --config.log_dir="${LOG_DIR}" \
  --config.dataset_json_path="${ZIMAGE_DATA_JSON}" \
  --config.pretrained.model_id_with_origin_paths="${ZIMAGE_MODEL_ID_WITH_ORIGIN_PATHS}" \
  --config.pretrained.model_paths="${ZIMAGE_MODEL_PATHS}" \
  --config.pretrained.tokenizer_path="${ZIMAGE_TOKENIZER_PATH}" \
  --config.wandb_mode="offline"
