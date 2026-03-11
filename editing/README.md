# EditRL

This subproject contains the image editing reinforcement learning code used in TrustYourCritic.

## Main Components

- `scripts/train_nft_qwen_image_edit.py`: Qwen-Image-Edit RL training
- `scripts/train_nft_kontext.py`: Kontext/FLUX editing RL training
- `reward_server/`: MLLM-based reward services
- `reproduction/`: sampling and reproduction scripts

## Install

```bash
pip install -e .
```

## Dataset Format

```text
dataset-root/
├── images/
├── train_metadata.jsonl
└── test_metadata.jsonl
```

Each line of `train_metadata.jsonl` / `test_metadata.jsonl`:

```json
{"prompt": "edit instruction", "image": "images/example.jpg", "requirement": "optional constraint"}
```

## Example

```bash
export EDIT_DATASET_ROOT=data/edit_dataset
export QWEN_IMAGE_EDIT_MODEL_PATH=Qwen/Qwen-Image-Edit
export REWARD_SERVER=127.0.0.1:12341

torchrun --nproc_per_node=8 \
  scripts/train_nft_qwen_image_edit.py \
  --config config/qwen_image_edit_nft.py:qwen_mllm_reward_16g
```
