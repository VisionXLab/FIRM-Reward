# GenerationRL

This subproject contains the generation-side reinforcement learning code used in TrustYourCritic.

## Main Components

- `scripts/train_nft_sd3.py`: SD3.5-based RL training
- `scripts/train_nft_qwen_image.py`: Qwen-Image RL training
- `scripts/train_nft_zimage_turbo.py`: Z-Image-Turbo RL training
- `scripts/train_nft_flux2_klein.py`: FLUX/Kontext-style RL training
- `flow_grpo/`: reward functions, training utilities, and rollout logic
- `reward_server/`: optional reward model services

## Install

```bash
pip install -e .
```

## Common Environment Variables

- `TRAIN_DATA_JSON`: prompt JSON path
- `SD35_MODEL_PATH`: SD3.5 model path or model id
- `QWEN_IMAGE_MODEL_PATH`: Qwen-Image model path or model id
- `FLUX2_MODEL_PATH`: FLUX model path
- `QWEN_VL_REWARD_URL`: reward server endpoint

## Example

```bash
export TRAIN_DATA_JSON=data/sharegpt/text_to_image.json
export SD35_MODEL_PATH=stabilityai/stable-diffusion-3.5-medium
export QWEN_VL_REWARD_URL=http://127.0.0.1:12341

torchrun --nproc_per_node=8 \
  scripts/train_nft_sd3.py \
  --config config/nft.py:sd35_sharegpt_qwenvl_flowgrpo_base
```
