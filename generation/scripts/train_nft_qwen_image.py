# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import os
import sys
from pathlib import Path
import datetime
from concurrent import futures
import time
import json
import gc
from absl import app, flags
import logging
from diffusers import DiffusionPipeline
import numpy as np

# Allow running via torchrun without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob import pipeline_with_logprob
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/nft_qwen_image.py:qwen_image_sharegpt_qwenvl", "Training configuration."
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ShareGPTPromptDataset(Dataset):
    def __init__(self, dataset_json_path, split="train", test_ratio=0.02, seed=42):
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        if not isinstance(items, list):
            raise ValueError(f"Dataset json should be a list, got: {type(items)}")

        indices = list(range(len(items)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        split_idx = int(len(indices) * (1 - test_ratio))
        if split == "train":
            indices = indices[:split_idx]
        else:
            indices = indices[split_idx:]

        if not indices and split == "test" and len(items) > 0:
            indices = [0]

        self.items = [items[idx] for idx in indices]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt = ""
        for key in ("input_prompt", "caption", "prompt", "text"):
            if key in item and item[key] is not None:
                prompt = str(item[key])
                break

        metadata = {}
        if isinstance(item, dict):
            for key in ("requirement", "source", "category"):
                if key in item:
                    metadata[key] = item[key]

        return {"prompt": prompt, "metadata": metadata}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class TextPromptDataset(Dataset):
    def __init__(self, dataset_dir, split="train"):
        self.file_path = os.path.join(dataset_dir, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        if self.total_samples % self.k != 0:
            self.m = (self.total_samples + self.k - 1) // self.k
            logger.warning(
                "Sampler auto-adjust: k=%d, world_batch=%d not divisible. Using m=%d prompt ids then trimming to %d samples.",
                self.k,
                self.total_samples,
                self.m,
                self.total_samples,
            )
        else:
            self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices][: self.total_samples]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    if world_size == 1:
        return tensor.detach().cpu()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def gather_list_to_all(local_list, world_size):
    if world_size == 1:
        return list(local_list)
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, list(local_list))
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


def load_pipeline_staggered(config, rank, world_size, model_dtype):
    load_kwargs = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }
    model_path = config.pretrained.model

    if world_size <= 1:
        logger.info("Loading Qwen-Image pipeline from %s", model_path)
        pipeline = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
        logger.info("Loaded pipeline class: %s", pipeline.__class__.__name__)
        return pipeline

    pipeline = None
    for load_rank in range(world_size):
        if rank == load_rank:
            logger.info(
                "Rank %d/%d loading Qwen-Image pipeline from %s",
                rank,
                world_size,
                model_path,
            )
            pipeline = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
            logger.info("Rank %d loaded pipeline class: %s", rank, pipeline.__class__.__name__)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        dist.barrier()
    return pipeline


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def build_qwen_img_shapes(batch_size, resolution, vae_scale_factor):
    base_hw = resolution // vae_scale_factor // 2
    return [[(1, base_hw, base_hw)] for _ in range(batch_size)]


def qwenimage_cfg_predict_batch(
    transformer,
    latents,
    timesteps,
    prompt_embeds,
    prompt_embeds_mask,
    guidance_scale,
    resolution,
    vae_scale_factor,
    negative_prompt_embeds=None,
    negative_prompt_embeds_mask=None,
):
    batch_size = latents.shape[0]
    img_shapes = build_qwen_img_shapes(batch_size, resolution, vae_scale_factor)

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
    max_txt_len = max(txt_seq_lens)
    prompt_embeds = prompt_embeds[:, :max_txt_len]
    prompt_embeds_mask = prompt_embeds_mask[:, :max_txt_len]

    timestep_input = timesteps.float() / 1000.0
    cond_pred = transformer(
        hidden_states=latents,
        timestep=timestep_input,
        guidance=None,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_embeds_mask,
        img_shapes=img_shapes,
        txt_seq_lens=txt_seq_lens,
    )[0]
    cond_pred = cond_pred[:, : latents.shape[1], ...]

    if guidance_scale <= 1.0 or negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
        return cond_pred

    neg_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
    max_neg_len = max(neg_txt_seq_lens)
    negative_prompt_embeds = negative_prompt_embeds[:, :max_neg_len]
    negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :max_neg_len]
    uncond_pred = transformer(
        hidden_states=latents,
        timestep=timestep_input,
        guidance=None,
        encoder_hidden_states=negative_prompt_embeds,
        encoder_hidden_states_mask=negative_prompt_embeds_mask,
        img_shapes=img_shapes,
        txt_seq_lens=neg_txt_seq_lens,
    )[0]
    uncond_pred = uncond_pred[:, : latents.shape[1], ...]

    cfg_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    cond_norm = torch.norm(cond_pred, dim=-1, keepdim=True).clamp_min(1e-6)
    cfg_norm = torch.norm(cfg_pred, dim=-1, keepdim=True).clamp_min(1e-6)
    return cfg_pred * (cond_norm / cfg_norm)


def eval_fn(
    pipeline,
    test_dataloader,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()
    all_rewards = defaultdict(list)
    eval_guidance_scale = float(getattr(config, "eval_guidance_scale", config.sample.guidance_scale))

    test_sampler = (
        DistributedSampler(
            test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        with torch_autocast(
            enabled=(config.mixed_precision in ["fp16", "bf16"]),
            dtype=mixed_precision_dtype,
        ):
            with torch.no_grad():
                sampled = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    negative_prompt=[""] * len(prompts),
                    num_inference_steps=config.sample.eval_num_steps,
                    true_cfg_scale=eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0.0,
                    process_index=rank,
                    sde_window_size=0,
                )
                images = sampled["images"]

        rewards_future = executor.submit(
            reward_fn, images, prompts, prompt_metadata, only_strict=False
        )
        time.sleep(0)
        rewards, _ = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {
            key: np.concatenate(value_list) for key, value_list in all_rewards.items()
        }

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray(
                    (image.float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [
                {k: final_rewards[k][i] for k in final_rewards}
                for i in range(num_samples_to_log)
            ]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts_log, sampled_rewards_log)
                        )
                    ],
                    **{
                        f"eval_reward_{key}": float(np.mean(value[np.isfinite(value)]))
                        for key, value in final_rewards.items()
                    },
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir,
    transformer_ddp,
    global_step,
    rank,
    ema,
    transformer_trainable_parameters,
    config,
    optimizer,
    scaler,
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    log_dir = getattr(config, "log_dir", "") or os.path.join(config.logdir, config.run_name)
    config.log_dir = log_dir
    if not getattr(config, "save_dir", ""):
        config.save_dir = getattr(config, "output_dir", log_dir)
    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(config.save_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DISABLE_GIT", "true")
        wandb.init(
            project="flow-grpo",
            name=config.run_name,
            config=config.to_dict(),
            dir=log_dir,
            mode=getattr(config, "wandb_mode", "offline"),
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=(mixed_precision_dtype == torch.float16))
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=(mixed_precision_dtype == torch.float16))
    else:
        scaler = GradScaler(enabled=(mixed_precision_dtype == torch.float16))

    # --- Load pipeline and models ---
    model_dtype = torch.float32
    if config.mixed_precision == "fp16":
        model_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        model_dtype = torch.bfloat16

    pipeline = load_pipeline_staggered(config, rank, world_size, model_dtype)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    if hasattr(pipeline, "safety_checker"):
        pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)
    if bool(getattr(config.train, "gradient_checkpointing", False)) and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    if config.use_lora:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
            "txt_mlp.net.0.proj",
            "txt_mlp.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=int(getattr(config.lora, "rank", 64)),
            lora_alpha=int(getattr(config.lora, "alpha", 128)),
            init_lora_weights="gaussian",
            target_modules=list(getattr(config.lora, "target_modules", target_modules)),
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    else:
        raise ValueError("Qwen-Image DiffusionNFT currently supports LoRA fine-tuning only (config.use_lora=True).")

    transformer_ddp = DDP(
        transformer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    use_text_prompt_dataset = (
        str(getattr(config, "prompt_fn", "")) == "general_ocr"
        and bool(getattr(config, "dataset", ""))
    )
    if use_text_prompt_dataset:
        train_dataset = TextPromptDataset(config.dataset, split="train")
        test_dataset = TextPromptDataset(config.dataset, split="test")
        logger.info("Using text prompt dataset from %s", config.dataset)
    else:
        dataset_json_path = str(config.dataset_json_path)
        train_dataset = ShareGPTPromptDataset(
            dataset_json_path,
            split="train",
            test_ratio=float(getattr(config, "dataset_test_ratio", 0.02)),
            seed=int(getattr(config, "dataset_split_seed", config.seed)),
        )
        test_dataset = ShareGPTPromptDataset(
            dataset_json_path,
            split="test",
            test_ratio=float(getattr(config, "dataset_test_ratio", 0.02)),
            seed=int(getattr(config, "dataset_split_seed", config.seed)),
        )
        logger.info("Using json prompt dataset from %s", dataset_json_path)

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=int(getattr(config, "num_workers", 0)),
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=True,
    )

    # --- Prompt Trackering ---
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        stat_tracker = None

    reward_workers = int(os.getenv("REWARD_CLIENT_WORKERS", str(getattr(config, "reward_client_workers", 16))))
    executor = futures.ThreadPoolExecutor(max_workers=max(1, reward_workers))
    logger.info("Reward client workers = %s", max(1, reward_workers))

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size * world_size * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device, config.reward_fn
    )  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device, config.reward_fn
    )  # Pass device

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(
                lora_path, adapter_name="default", is_trainable=True
            )
            transformer_ddp.module.load_adapter(
                lora_path, adapter_name="old", is_trainable=False
            )
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(
                config.resume_from, "transformer_model.pt"
            )  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(
                    torch.load(model_ckpt_path, map_location=device)
                )

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(
                f"Resumed global_step to {global_step}. Epoch estimation might be needed."
            )
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=0.9,
            update_step_interval=1,
            device=device,
        )

    num_train_timesteps = int((config.sample.num_steps - 1) * config.train.timestep_fraction)
    if int(getattr(config.sample, "sde_window_size", 0)) > 0:
        num_train_timesteps = int(getattr(config.sample, "sde_window_size"))
    if num_train_timesteps < 1:
        num_train_timesteps = 1
    train_guidance_scale = float(getattr(config, "train_guidance_scale", config.sample.guidance_scale))

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters,
        old_transformer_trainable_parameters,
        strict=True,
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(
                train_sampler, DistributedKRepeatSampler
            ):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            try:
                prompts, prompt_metadata = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                prompts, prompt_metadata = next(train_iter)

            prompt_ids = pipeline.tokenizer(
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            if i == 0 and epoch > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if (
                i == 0
                and epoch % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
            ):
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    sampled = pipeline_with_logprob(
                        pipeline,
                        prompt=prompts,
                        negative_prompt=[""] * len(prompts),
                        num_inference_steps=config.sample.num_steps,
                        true_cfg_scale=train_guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=float(getattr(config.sample, "noise_level", 0.0)),
                        process_index=rank,
                        sde_window_size=int(getattr(config.sample, "sde_window_size", 0)),
                        sde_window_range=tuple(getattr(config.sample, "sde_window_range", (0, config.sample.num_steps // 2))),
                    )
                    images = sampled["images"]
                    latents_clean = sampled["final_latents"]
                    prompt_embeds = sampled["prompt_embeds"]
                    prompt_embeds_mask = sampled["prompt_embeds_mask"]
                    negative_prompt_embeds = sampled["negative_prompt_embeds"]
                    negative_prompt_embeds_mask = sampled["negative_prompt_embeds_mask"]
                    if sampled["all_timesteps"]:
                        timestep_values = torch.stack(sampled["all_timesteps"]).to(device)
                    else:
                        timestep_values = sampled["scheduler_timesteps"][:-1].to(device)
            transformer_ddp.module.set_adapter("default")

            if timestep_values.numel() == 0:
                raise ValueError("Qwen-Image sampler returned empty timesteps.")
            timestep_values = timestep_values[:num_train_timesteps]
            timesteps = timestep_values.repeat(len(prompts), 1)

            rewards_future = executor.submit(
                reward_fn,
                images,
                prompts,
                prompt_metadata,
                only_strict=True,
            )
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "prompt_embeds_mask": prompt_embeds_mask,
                    "negative_prompt_embeds": negative_prompt_embeds,
                    "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate(
                        [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1
                    ),
                    "latents_clean": latents_clean,
                    "rewards_future": rewards_future,  # Store future
                    "prompts": list(prompts),
                }
            )


        for sample_item in tqdm(
            samples_data_list,
            desc="Waiting for rewards",
            disable=not is_main_process(rank),
            position=0,
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {
                k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()
            }
            del sample_item["rewards_future"]

        max_prompt_len = max(sample["prompt_embeds"].shape[1] for sample in samples_data_list)
        for sample in samples_data_list:
            pad_len = max_prompt_len - sample["prompt_embeds"].shape[1]
            if pad_len <= 0:
                continue
            sample["prompt_embeds"] = torch.nn.functional.pad(
                sample["prompt_embeds"], (0, 0, 0, pad_len), value=0
            )
            sample["prompt_embeds_mask"] = torch.nn.functional.pad(
                sample["prompt_embeds_mask"], (0, pad_len), value=0
            )
            sample["negative_prompt_embeds"] = torch.nn.functional.pad(
                sample["negative_prompt_embeds"], (0, 0, 0, pad_len), value=0
            )
            sample["negative_prompt_embeds_mask"] = torch.nn.functional.pad(
                sample["negative_prompt_embeds_mask"], (0, pad_len), value=0
            )

        collated_samples = {
            "prompt_ids": torch.cat([s["prompt_ids"] for s in samples_data_list], dim=0),
            "prompt_embeds": torch.cat([s["prompt_embeds"] for s in samples_data_list], dim=0),
            "prompt_embeds_mask": torch.cat([s["prompt_embeds_mask"] for s in samples_data_list], dim=0),
            "negative_prompt_embeds": torch.cat([s["negative_prompt_embeds"] for s in samples_data_list], dim=0),
            "negative_prompt_embeds_mask": torch.cat([s["negative_prompt_embeds_mask"] for s in samples_data_list], dim=0),
            "timesteps": torch.cat([s["timesteps"] for s in samples_data_list], dim=0),
            "next_timesteps": torch.cat([s["next_timesteps"] for s in samples_data_list], dim=0),
            "latents_clean": torch.cat([s["latents_clean"] for s in samples_data_list], dim=0),
            "prompts": [p for s in samples_data_list for p in s["prompts"]],
            "rewards": {
                key: torch.cat([s["rewards"][key] for s in samples_data_list], dim=0)
                for key in samples_data_list[0]["rewards"].keys()
            },
        }

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][
                -len(images_to_log) :
            ].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray(
                        (img_data.float().numpy().transpose(1, 2, 0) * 255).astype(
                            np.uint8
                        )
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        train_timesteps_this_epoch = min(num_train_timesteps, collated_samples["timesteps"].shape[1])
        if train_timesteps_this_epoch < 1:
            raise ValueError("No valid timesteps collected for Qwen-Image training.")
        collated_samples["timesteps"] = collated_samples["timesteps"][:, :train_timesteps_this_epoch]
        collated_samples["next_timesteps"] = collated_samples["next_timesteps"][:, :train_timesteps_this_epoch]
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"]
            .unsqueeze(1)
            .repeat(1, train_timesteps_this_epoch)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(
                value_tensor, world_size
            ).numpy()

        if is_main_process(rank):  # logging
            reward_mean_parts = []
            reward_log_values = {}
            for k, v in gathered_rewards_dict.items():
                flat = np.asarray(v).reshape(-1)
                finite = np.isfinite(flat)
                if np.any(flat == -10):
                    finite = finite & (flat != -10)
                mean_val = float(np.mean(flat[finite])) if np.any(finite) else float("nan")
                reward_mean_parts.append(f"{k}={mean_val:.4f}")
                if np.any(finite) and "_strict_accuracy" not in k and "_accuracy" not in k:
                    reward_log_values[f"reward_{k}"] = mean_val
            logger.info("Epoch %s reward mean | %s", epoch, " | ".join(reward_mean_parts))

            wandb.log(
                {
                    "epoch": epoch,
                    **reward_log_values,
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompts_all = gather_list_to_all(collated_samples["prompts"], world_size)
            advantages = stat_tracker.update(prompts_all, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                    prompts_all, gathered_rewards_dict
                )
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (
                avg_rewards_all.std() + 1e-4
            )
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            raise ValueError(
                f"advantages shape mismatch: {advantages.shape[0]} vs {world_size} * {samples_per_gpu}"
            )
        
        if is_main_process(rank):
            logger.info(
                f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}"
            )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        del collated_samples["prompts"]

        num_batches = (
            config.sample.num_batches_per_epoch
            * config.sample.train_batch_size
            // config.train.batch_size
        )

        filtered_samples = collated_samples
        filtered_samples["valid_mask"] = torch.ones_like(collated_samples["advantages"])

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples[
            "timesteps"
        ].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = (
            config.train.gradient_accumulation_steps * train_timesteps_this_epoch
        )

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):

            if total_batch_size_filtered == 0: # If all samples are banned, break
                print("All samples are banned, break")
                break
            
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {
                k: v[perm] for k, v in filtered_samples.items()
            }

            perms_time = torch.stack(
                [
                    torch.randperm(num_timesteps_filtered, device=device)
                    for _ in range(total_batch_size_filtered)
                ]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None],
                    perms_time,
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(
                list
            )  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                prompt_embeds = train_sample_batch["prompt_embeds"]
                prompt_embeds_mask = train_sample_batch["prompt_embeds_mask"]
                negative_prompt_embeds = train_sample_batch["negative_prompt_embeds"]
                negative_prompt_embeds_mask = train_sample_batch["negative_prompt_embeds_mask"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(train_timesteps_this_epoch)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(
                        enabled=enable_amp, dtype=mixed_precision_dtype
                    ):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            old_prediction = qwenimage_cfg_predict_batch(
                                transformer=transformer_ddp.module,
                                latents=xt,
                                timesteps=train_sample_batch["timesteps"][:, j_idx],
                                prompt_embeds=prompt_embeds,
                                prompt_embeds_mask=prompt_embeds_mask,
                                guidance_scale=train_guidance_scale,
                                resolution=config.resolution,
                                vae_scale_factor=pipeline.vae_scale_factor,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                            ).detach()
                        transformer_ddp.module.set_adapter("default")

                        forward_prediction = qwenimage_cfg_predict_batch(
                            transformer=transformer_ddp.module,
                            latents=xt,
                            timesteps=train_sample_batch["timesteps"][:, j_idx],
                            prompt_embeds=prompt_embeds,
                            prompt_embeds_mask=prompt_embeds_mask,
                            guidance_scale=train_guidance_scale,
                            resolution=config.resolution,
                            vae_scale_factor=pipeline.vae_scale_factor,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                        )

                        with torch.no_grad():  # Reference model part
                            with transformer_ddp.module.disable_adapter():
                                ref_forward_prediction = qwenimage_cfg_predict_batch(
                                    transformer=transformer_ddp.module,
                                    latents=xt,
                                    timesteps=train_sample_batch["timesteps"][:, j_idx],
                                    prompt_embeds=prompt_embeds,
                                    prompt_embeds_mask=prompt_embeds_mask,
                                    guidance_scale=train_guidance_scale,
                                    resolution=config.resolution,
                                    vae_scale_factor=pipeline.vae_scale_factor,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                                )
                            transformer_ddp.module.set_adapter("default")

                    loss_terms = {}

                    valid_mask = train_sample_batch["valid_mask"][:, j_idx].float()
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(
                                advantages_clip, 0, config.train.adv_clip_max
                            )
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(
                                advantages_clip, -config.train.adv_clip_max, 0
                            )
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0,
                                torch.ones_like(advantages_clip),
                                torch.zeros_like(advantages_clip),
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (
                        advantages_clip / config.train.adv_clip_max
                    ) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()
                    loss_terms["old_deviate_max"] = torch.max(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()

                    positive_prediction = (
                        config.beta * forward_prediction
                        + (1 - config.beta) * old_prediction.detach()
                    )
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                    negative_x0_prediction = (
                        xt - t_expanded * implicit_negative_prediction
                    )
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = (
                        (negative_x0_prediction - x0) ** 2 / negative_weight_factor
                    ).mean(dim=tuple(range(1, x0.ndim)))
                    ori_policy_loss = (
                        r * positive_loss * valid_mask / config.beta
                        + (1.0 - r) * negative_loss * valid_mask / config.beta
                    )

                    def mean_by_mask(x, mask):
                        if mask.sum() == 0:
                            return x.sum() * 0
                        return x.sum() / mask.sum()

                    policy_loss = mean_by_mask(ori_policy_loss * config.train.adv_clip_max, valid_mask)
                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = (
                        mean_by_mask(ori_policy_loss, valid_mask).detach()
                    )

                    kl_div_loss = (
                        (forward_prediction - ref_forward_prediction) ** 2
                    ).mean(dim=tuple(range(1, x0.ndim))) * valid_mask

                    loss += config.train.beta * mean_by_mask(kl_div_loss, valid_mask)
                    kl_div_loss = mean_by_mask(kl_div_loss, valid_mask)
                    loss_terms["kl_div_loss"] = kl_div_loss.detach()
                    loss_terms["kl_div"] = mean_by_mask(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        ) * valid_mask, valid_mask
                    ).detach()
                    loss_terms["old_kl_div"] = mean_by_mask(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        ) * valid_mask, valid_mask
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    if not torch.isfinite(loss):
                        logger.warning(
                            "Non-finite loss at epoch=%s step=%s timestep_idx=%s, skip this micro step.",
                            epoch,
                            global_step,
                            j_idx,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            transformer_ddp.module.parameters(),
                            config.train.max_grad_norm,
                        )
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {
                            k: torch.mean(torch.stack(v_list)).item()
                            for k, v_list in info_accumulated.items()
                        }
                        info_tensor = torch.tensor(
                            [log_info[k] for k in sorted(log_info.keys())],
                            device=device,
                        )
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.SUM)
                        info_tensor = info_tensor / world_size
                        reduced_log_info = {
                            k: info_tensor[ki].item()
                            for ki, k in enumerate(sorted(log_info.keys()))
                        }
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(
                            list
                        )  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters,
                old_transformer_trainable_parameters,
                strict=True,
            ):
                tgt_param.data.copy_(
                    tgt_param.detach().data * decay
                    + src_param.detach().clone().data * (1.0 - decay)
                )

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
