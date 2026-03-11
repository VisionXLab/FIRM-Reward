# TrustYourCritic

## Repository Layout

```text
FIRM-Reward/
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

## Evaluation

The code and data for **FIRM-Bench** are hosted on [Hugging Face](https://huggingface.co/datasets/zpy777/FIRM-Bench).

We provide inference and evaluation scripts for **FIRM-Bench**. We recommend deploying the model with vLLM for inference.


### FIRM-Bench-Edit

#### Inference
```python
python FIRM-Bench-Edit/vllm_infer.py \
  --input FIRM-Bench-Edit/bench_v1.jsonl \
  --output FIRM-Bench-Edit/result/xxx.jsonl \
  --image-root FIRM-Bench-Edit/ \
  --api-url xxxxx
```

#### MAE Calculation
```python
python FIRM-Bench-Edit/edit_mae.py \
  --gt FIRM-Bench-Edit/result/human_bench_v1.jsonl \
  --pred FIRM-Bench-Edit/result/xxx.jsonl
```

### FIRM-Bench-Gen

#### Inference
```python
python FIRM-Bench-Gen/vllm_infer.py \
  --input FIRM-Bench-Gen/bench_v1.jsonl \
  --output FIRM-Bench-Gen/result/xxx.jsonl \
  --image-root FIRM-Bench-Gen/ \
  --api-url xxxxx
```

#### MAE Calculation
```python
python FIRM-Bench-Gen/gen_mae.py \
  --gt FIRM-Bench-Gen/result/human_bench_v1.jsonl \
  --pred FIRM-Bench-Gen/result/xxx.jsonl
```
