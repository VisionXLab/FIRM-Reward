<p align="center">
  <img src="assets/FIRM_reward.png" alt="Trust Your Critic hero card" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Focus-Robust%20Reward%20Modeling-A6543D?style=flat-square" alt="Focus: Robust Reward Modeling">
  <img src="https://img.shields.io/badge/Tasks-Image%20Editing%20%2B%20T2I%20Generation-166A5C?style=flat-square" alt="Tasks: Image Editing and T2I Generation">
  <img src="https://img.shields.io/badge/Benchmark-FIRM--Bench%20(807%20samples)-6D5A96?style=flat-square" alt="Benchmark: FIRM-Bench">
  <img src="https://img.shields.io/badge/Status-Research%20Codebase-333333?style=flat-square" alt="Status: Research Codebase">
</p>

<p align="center">
  <strong>Trust Your Critic: Robust Reward Modeling and Reinforcement Learning for Faithful Image Editing and Generation</strong>
</p>


## Why This Repo

- **Critics are the bottleneck.** FIRM is built around the idea that RL for visual generation only works when the reward model is faithful, stable, and hard to hack.
- **Two task-specific data pipelines.** `FIRM-Edit` uses a difference-first scoring pipeline, while `FIRM-Gen` uses a plan-then-score pipeline to reduce MLLM hallucinations.
- **One benchmark for both sides.** `FIRM-Bench` provides a human-annotated test bed for editing and generation critics.
- **RL reward shaping that actually holds up.** `CME` and `QMA` are designed to prevent the shortcut behavior that appears when rewards are naively combined.

## FIRM At A Glance

| Component | What it adds |
| --- | --- |
| `FIRM-Edit-370K` | Large-scale editing reward data with separate execution and consistency supervision. |
| `FIRM-Gen-293K` | Structured generation reward data built from prompt-specific checklists. |
| `FIRM-Edit-8B` / `FIRM-Gen-8B` | Reward models specialized for faithful editing and instruction-following generation. |
| `FIRM-Bench` | 807 human-annotated samples spanning editing execution, editing consistency, and generation alignment. |
| `CME` / `QMA` | Base-and-bonus reward formulations for editing RL and generation RL. |

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

## Acknowledgements

This repository was shaped by several open-source projects that pushed RL for image generation and image editing forward:

- [flow-grpo](https://github.com/yifan123/flow_grpo) 
- [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) 
- [Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1) 
