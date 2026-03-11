# TrustYourCritic

## Repository Layout

```text
TrustYourCritic/
├── generation/   # GenerationRL training and reward serving
└── editing/      # EditRL training, reward serving, reproduction scripts
```

## Important Notes

- To avoid Python package conflicts, install and run GenRL/EditRL in separate environments.

## Quick Start

### 1) GenerationRL

```bash
cd generation
conda create -n FIRM-Gen python=3.10 -y
conda activate FIRM-Gen
pip install -e .
```



#### i ) Launch Reward Server First

```
python generation/flow_grpo/reward_model_server.py
```

#### ii ) Change the Training Configuration

- `generation/config/nft_flux2_klein.py`
- `generation/config/nft_qwen_image.py`
- `generation/config/nft_zimage_turbo.py`
- `generation/config/nft.py`


#### iii ) Start Training

```bash
bash generation/scripts/train_sd35_sharegpt_qwenvl.sh
```

### 2) EditRL

```bash
cd editing
conda create -n FIRM-Edit python=3.10 -y
conda activate FIRM-Edit
pip install -e .
```

#### i ) Launch Reward Server First

```python
## Change the default ip and port to your perference
python editing/reward_server/reward_server_qwen3_vl_8b_sft.py
```

#### ii ) Change the Training Configuration

- `editing/config/kontext_nft_qwen3vl_8b_sft.py`
- `editing/config/kontext_nft_qwen3vl_8b.py`
- `editing/config/kontext_nft_qwen25vl_32b_non_logits.py`

#### iii ) Start Training

```bash
bash editing/examples/train_qwen_image_edit.sh
```

## Data Perparation

### GenerationRL

Expected JSON file like:

```json
[
  {"input_prompt": "A cinematic portrait of a fox in snow."}
]
```

### EditRL

Expected dataset layout:

```text
dataset-root/
├── images/
├── train_metadata.jsonl
└── test_metadata.jsonl
```

Each JSONL line:

```json
{"prompt": "make the sky sunset orange", "image": "images/example.jpg", "requirement": "preserve identity"}
```

