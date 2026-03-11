#!/usr/bin/env bash
set -euo pipefail

export N_GPUS="${N_GPUS:-8}"
export NUM_GROUPS="${NUM_GROUPS:-48}"
export NUM_IMAGE_PER_PROMPT="${NUM_IMAGE_PER_PROMPT:-12}"
export EXP_NAME="${EXP_NAME:-flux2_klein_diffusionnft_sharegpt_8gpu_base_model}"

# Fixed by request (direct prompt JSON + local model)
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-models/Flux.2}"
export FLUX2_DATA_JSON="${FLUX2_DATA_JSON:-data/sharegpt/text_to_image.json}"

# WandB offline + requested storage layout
export WANDB_MODE="offline"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/${EXP_NAME}}"
export LOG_DIR="${LOG_DIR:-logs/wandb/${EXP_NAME}}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Reward settings aligned with scripts/train_sd35_sharegpt_qwenvl_flowgrpo.sh
export QWEN_VL_REWARD_URL="${QWEN_VL_REWARD_URL:-http://127.0.0.1:12341}"
export REWARD_MAX_PENDING="${REWARD_MAX_PENDING:-1}"
export QWEN_VL_REWARD_TIMEOUT="${QWEN_VL_REWARD_TIMEOUT:-1800}"

# Optional overrides consumed by config
# export FLUX2_DATA_JSON
# export FLUX2_MODEL_PATH
# export EXP_NAME


torchrun \
  --nproc_per_node="${N_GPUS}" \
  scripts/train_nft_flux2_klein.py \
  --config=config/nft_flux2_klein.py:flux2_klein_sharegpt_qwenvl \
  --config.output_dir="${OUTPUT_DIR}" \
  --config.log_dir="${LOG_DIR}" \
  --config.pretrained.model="${FLUX2_MODEL_PATH}" \
  --config.dataset_json_path="${FLUX2_DATA_JSON}" \
  --config.wandb_mode="offline"
