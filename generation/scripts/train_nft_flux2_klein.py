# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from concurrent import futures
import contextlib
import datetime
import fnmatch
import gc
import json
import logging
import os
from pathlib import Path
import random
import sys
import tempfile

from absl import app, flags
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
import tqdm
import wandb

from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model

# Allow running via torchrun without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import flow_grpo.rewards
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker

try:
    from diffusers import Flux2KleinPipeline
except ImportError as exc:
    raise ImportError(
        "Flux2KleinPipeline is required for FLUX.2 training. "
        "Please install a diffusers version that includes Flux2KleinPipeline."
    ) from exc


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "config/nft_flux2_klein.py:flux2_klein_sharegpt_qwenvl",
    "Training configuration.",
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
tqdm = tqdm.tqdm


DEFAULT_FLUX_LORA_TARGET_MODULES = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.linear_in",
    "ff.linear_out",
    "ff_context.linear_in",
    "ff_context.linear_out",
    "attn.to_qkv_mlp_proj",
]


def resolve_lora_target_modules(transformer, target_patterns):
    module_map = dict(transformer.named_modules())
    module_names = list(module_map.keys())

    resolved = []
    missing = []
    for pattern in target_patterns:
        if not pattern:
            continue
        if any(ch in pattern for ch in "*?[]"):
            matched = [name for name in module_names if fnmatch.fnmatch(name, pattern)]
        else:
            matched = [name for name in module_names if name == pattern or name.endswith(f".{pattern}")]
        matched = [name for name in matched if name and hasattr(module_map[name], "weight")]
        if matched:
            resolved.extend(matched)
        else:
            missing.append(pattern)

    resolved = sorted(set(resolved))
    if missing:
        logger.warning("Some LoRA target patterns did not match any module: %s", missing)
    if not resolved:
        raise ValueError(f"No LoRA target modules matched. patterns={target_patterns}")
    return resolved


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1 and not dist.is_initialized():
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


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


def all_reduce_mean(value, device, world_size):
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / world_size
    return tensor.item()


def autocast_cuda(enabled, dtype):
    if enabled and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def tensor_to_pil(image_tensor, resolution=None):
    image = image_tensor.detach().float().cpu().clamp(0, 1)
    image_np = (image.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(image_np)
    if resolution is not None:
        pil = pil.resize((resolution, resolution))
    return pil


def _resolve_model_load_dtype(config):
    if getattr(config, "model_load_dtype", ""):
        dtype_name = str(config.model_load_dtype).lower()
        if dtype_name == "fp32":
            return torch.float32
        if dtype_name in ("fp16", "float16"):
            return torch.float16
        if dtype_name in ("bf16", "bfloat16"):
            return torch.bfloat16
    if str(getattr(config, "mixed_precision", "")).lower() == "fp16":
        return torch.float16
    if str(getattr(config, "mixed_precision", "")).lower() == "bf16":
        return torch.bfloat16
    return torch.float32


def _materialize_to_device_early(pipeline, device, load_dtype):
    # Immediately move heavy modules to GPU to release host memory.
    if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
        pipeline.transformer.to(device=device, dtype=load_dtype)
    if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
        pipeline.text_encoder.to(device=device, dtype=load_dtype)
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        vae_dtype = load_dtype if load_dtype != torch.float32 else torch.float32
        pipeline.vae.to(device=device, dtype=vae_dtype)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_pipeline_staggered(config, rank, world_size, device):
    # Avoid host-memory spikes when 8 ranks materialize a large model simultaneously.
    load_dtype = _resolve_model_load_dtype(config)
    load_kwargs = {
        "torch_dtype": load_dtype,
        "low_cpu_mem_usage": True,
    }

    if world_size <= 1:
        logger.info("Loading pipeline with dtype=%s", str(load_dtype))
        pipeline = Flux2KleinPipeline.from_pretrained(config.pretrained.model, **load_kwargs)
        logger.info("Loaded pipeline class: %s", pipeline.__class__.__name__)
        _materialize_to_device_early(pipeline, device, load_dtype)
        return pipeline

    pipeline = None
    for load_rank in range(world_size):
        if rank == load_rank:
            logger.info(
                "Rank %d/%d loading pipeline from %s (dtype=%s)",
                rank,
                world_size,
                config.pretrained.model,
                str(load_dtype),
            )
            pipeline = Flux2KleinPipeline.from_pretrained(config.pretrained.model, **load_kwargs)
            logger.info("Loaded pipeline class: %s", pipeline.__class__.__name__)
            _materialize_to_device_early(pipeline, device, load_dtype)
        dist.barrier()
    return pipeline


def _extract_prompt(item):
    for key in ("input_prompt", "caption", "prompt", "text"):
        if key in item and item[key] is not None:
            return str(item[key])
    return ""


def _build_metadata(item):
    metadata = {}
    if isinstance(item, dict):
        if "requirement" in item:
            metadata["requirement"] = item["requirement"]
        if "source" in item:
            metadata["source"] = item["source"]
        if "category" in item:
            metadata["category"] = item["category"]
    return metadata


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
        prompt = _extract_prompt(item)
        metadata = _build_metadata(item)

        return {
            "prompt": prompt,
            "metadata": metadata,
        }

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
            # Allow non-divisible settings by sampling slightly more prompt ids
            # and trimming to exactly total_samples after shuffle.
            self.m = (self.total_samples + self.k - 1) // self.k
            logger.warning(
                "Sampler auto-adjust: k=%d, world_batch=%d not divisible. "
                "Using m=%d prompt ids then trimming to %d samples.",
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
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            shuffled_samples = shuffled_samples[: self.total_samples]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _repeat_negative_embeddings(negative_prompt_embeds, negative_text_ids, batch_size, device):
    neg_prompt = negative_prompt_embeds
    neg_text = negative_text_ids

    if neg_prompt.shape[0] == 1 and batch_size > 1:
        neg_prompt = neg_prompt.repeat(batch_size, 1, 1)
    elif neg_prompt.shape[0] != batch_size:
        neg_prompt = neg_prompt[:batch_size]

    if neg_text.shape[0] == 1 and batch_size > 1:
        neg_text = neg_text.repeat(batch_size, 1, 1)
    elif neg_text.shape[0] != batch_size:
        neg_text = neg_text[:batch_size]

    return neg_prompt.to(device), neg_text.to(device)


def compute_empirical_mu(image_seq_len, num_steps):
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def set_flux_timesteps(scheduler, num_inference_steps, device, image_seq_len):
    sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)
    timesteps = scheduler.timesteps
    scheduler_sigmas = getattr(scheduler, "sigmas", None)
    if scheduler_sigmas is not None and scheduler_sigmas.ndim == 1 and scheduler_sigmas.shape[0] >= timesteps.shape[0]:
        # Scheduler commonly keeps one extra terminal sigma; trim to match sampled timesteps.
        train_sigmas = scheduler_sigmas[: timesteps.shape[0]].to(device=device, dtype=torch.float32)
    else:
        # Fallback for schedulers without exposed sigma buffers.
        train_sigmas = timesteps.float().to(device=device) / 1000.0
    return timesteps, train_sigmas


def decode_flux_latents(pipeline, latents, latent_ids, height, width, output_type="pt"):
    if hasattr(pipeline, "_unpack_latents_with_ids"):
        unpacked = pipeline._unpack_latents_with_ids(latents, latent_ids)
    elif hasattr(pipeline, "_unpack_latents"):
        unpacked = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    else:
        raise AttributeError("Pipeline is missing latent unpacking method for FLUX decoding.")

    if hasattr(pipeline.vae, "bn") and hasattr(pipeline, "_unpatchify_latents"):
        latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        latents_bn_std = torch.sqrt(
            pipeline.vae.bn.running_var.view(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps
        ).to(unpacked.device, unpacked.dtype)
        unpacked = unpacked * latents_bn_std + latents_bn_mean
        unpacked = pipeline._unpatchify_latents(unpacked)
    else:
        scaling_factor = getattr(pipeline.vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(pipeline.vae.config, "shift_factor", 0.0)
        unpacked = (unpacked / scaling_factor) + shift_factor

    # Keep decode input dtype/device aligned with VAE parameters.
    vae_param = next(pipeline.vae.parameters())
    unpacked = unpacked.to(device=vae_param.device, dtype=vae_param.dtype)
    image = pipeline.vae.decode(unpacked, return_dict=False)[0]
    return pipeline.image_processor.postprocess(image, output_type=output_type)


def flux_predict(
    transformer,
    latents,
    timesteps,
    prompt_embeds,
    text_ids,
    latent_ids,
    guidance_scale,
    embedded_guidance=1.0,
    negative_prompt_embeds=None,
    negative_text_ids=None,
):
    timestep_float = timesteps.float() / 1000.0
    guidance = None
    if embedded_guidance is not None:
        guidance = torch.full(
            (latents.shape[0],),
            float(embedded_guidance),
            device=latents.device,
            dtype=latents.dtype,
        )

    cond_pred = transformer(
        hidden_states=latents,
        timestep=timestep_float,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    cond_pred = cond_pred[:, : latents.size(1), ...]

    if guidance_scale <= 1.0 or negative_prompt_embeds is None or negative_text_ids is None:
        return cond_pred

    neg_prompt_embeds, neg_text_ids = _repeat_negative_embeddings(
        negative_prompt_embeds,
        negative_text_ids,
        batch_size=latents.shape[0],
        device=latents.device,
    )
    uncond_pred = transformer(
        hidden_states=latents,
        timestep=timestep_float,
        guidance=guidance,
        encoder_hidden_states=neg_prompt_embeds,
        txt_ids=neg_text_ids,
        img_ids=latent_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    uncond_pred = uncond_pred[:, : latents.size(1), ...]
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)


def sample_flux_rollout(
    pipeline,
    transformer,
    prompt_embeds,
    text_ids,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    device,
    generator,
    embedded_guidance=1.0,
    negative_prompt_embeds=None,
    negative_text_ids=None,
    mixed_precision_dtype=torch.bfloat16,
):
    batch_size = prompt_embeds.shape[0]
    num_latent_channels = transformer.config.in_channels // 4

    latents, latent_ids = pipeline.prepare_latents(
        batch_size=batch_size,
        num_latents_channels=num_latent_channels,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=None,
    )
    timesteps, sigmas = set_flux_timesteps(pipeline.scheduler, num_inference_steps, device, latents.shape[1])

    all_latents = [latents.detach()]

    for t in timesteps:
        timestep = t.expand(batch_size).to(device)
        with autocast_cuda(enabled=(mixed_precision_dtype is not None), dtype=mixed_precision_dtype):
            noise_pred = flux_predict(
                transformer=transformer,
                latents=latents,
                timesteps=timestep,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                latent_ids=latent_ids,
                guidance_scale=guidance_scale,
                embedded_guidance=embedded_guidance,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
            )
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        all_latents.append(latents.detach())

    images = decode_flux_latents(
        pipeline=pipeline,
        latents=latents,
        latent_ids=latent_ids,
        height=height,
        width=width,
        output_type="pt",
    )
    all_latents = torch.stack(all_latents, dim=1)
    return images, all_latents, timesteps, sigmas, latent_ids


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
        raise ValueError(f"Unsupported decay_type: {decay_type}")

    if step < flat:
        return 0.0
    decay = (step - flat) * uprate
    return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    if len(unique_prompts) == 0:
        return 0.0, 0.0
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def save_ckpt(
    save_dir,
    transformer,
    global_step,
    rank,
    ema,
    transformer_trainable_parameters,
    config,
    optimizer,
    scaler,
):
    if not is_main_process(rank):
        return

    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)

    model_to_save = _unwrap_model(transformer)

    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    model_to_save.save_pretrained(save_root_lora)
    torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    logger.info("Saved checkpoint to %s", save_root)


def eval_fn(
    pipeline,
    transformer,
    test_dataloader,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    eval_embedded_guidance,
    text_encoder_device,
    negative_prompt_embeds,
    negative_text_ids,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    model = _unwrap_model(transformer)
    model.eval()

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
        pin_memory=True,
    )

    all_rewards = defaultdict(list)
    last_images = None
    last_prompts = None

    max_eval_prompts = int(getattr(config, "eval_num_prompts", 0) or 0)
    seen_prompts = 0

    for test_batch in tqdm(
        eval_loader,
        desc="Eval",
        disable=not is_main_process(rank),
        dynamic_ncols=True,
    ):
        prompts, prompt_metadata = test_batch
        with torch.no_grad():
            prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt=prompts,
                device=text_encoder_device,
                num_images_per_prompt=1,
                max_sequence_length=int(getattr(config, "max_sequence_length", 512)),
                text_encoder_out_layers=tuple(getattr(config, "text_encoder_out_layers", [9, 18, 27])),
            )
        prompt_embeds = prompt_embeds.to(device, non_blocking=True)
        text_ids = text_ids.to(device, non_blocking=True)

        with torch.no_grad():
            images, _, _, _, _ = sample_flux_rollout(
                pipeline=pipeline,
                transformer=model,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=config.sample.eval_num_steps,
                guidance_scale=getattr(config, "eval_guidance_scale", config.sample.guidance_scale),
                device=device,
                generator=None,
                embedded_guidance=eval_embedded_guidance,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_text_ids=negative_text_ids,
                mixed_precision_dtype=mixed_precision_dtype,
            )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        rewards, _ = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

        last_images = images
        last_prompts = prompts
        seen_prompts += len(prompts)
        if max_eval_prompts > 0 and seen_prompts >= max_eval_prompts:
            break

    if is_main_process(rank) and all_rewards:
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        if last_images is not None and last_prompts is not None:
            images_to_log = last_images.cpu()
            prompts_to_log = last_prompts
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples_to_log = min(12, len(images_to_log))
                for idx in range(num_samples_to_log):
                    pil = tensor_to_pil(images_to_log[idx], resolution=config.resolution)
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]
                wandb.log(
                    {
                        "eval_images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt[:120]} | "
                                + " | ".join(
                                    f"{k}: {float(v):.2f}" for k, v in reward.items() if np.isfinite(v)
                                ),
                            )
                            for idx, (prompt, reward) in enumerate(zip(prompts_to_log[:num_samples_to_log], sampled_rewards_log))
                        ],
                        **{f"eval_reward_{key}": float(np.mean(value)) for key, value in final_rewards.items()},
                    },
                    step=global_step,
                )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def main(_):
    config = FLAGS.config

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    text_encoder_device = device if str(getattr(config, "text_encoder_device", "cuda")).lower() == "cuda" else torch.device("cpu")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name = f"{config.run_name}_{unique_id}"

    log_dir = getattr(config, "log_dir", "") or os.path.join(config.logdir, config.run_name)
    output_dir = (
        getattr(config, "output_dir", "")
        or getattr(config, "save_dir", "")
        or os.path.join("outputs", config.run_name)
    )
    config.log_dir = log_dir
    config.output_dir = output_dir
    config.save_dir = output_dir

    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        wandb.init(
            project="diffusion-nft-flux2-klein",
            name=config.run_name,
            config=config.to_dict(),
            dir=log_dir,
            mode=getattr(config, "wandb_mode", "offline"),
        )
    logger.info("\n%s", config)

    set_seed(config.seed, rank)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))

    pipeline = load_pipeline_staggered(config, rank, world_size, device)

    if hasattr(pipeline, "text_encoder"):
        pipeline.text_encoder.requires_grad_(False)
        text_encoder_dtype = mixed_precision_dtype if mixed_precision_dtype is not None else torch.float32
        pipeline.text_encoder.to(device=text_encoder_device, dtype=text_encoder_dtype)
        pipeline.text_encoder.eval()

    pipeline.vae.requires_grad_(False)
    vae_dtype = mixed_precision_dtype if mixed_precision_dtype is not None else torch.float32
    pipeline.vae.to(device=device, dtype=vae_dtype)
    pipeline.vae.eval()

    transformer = pipeline.transformer
    transformer.requires_grad_(not config.use_lora)
    if bool(getattr(config.train, "gradient_checkpointing", False)):
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        else:
            logger.warning("gradient_checkpointing requested but transformer does not expose enable_gradient_checkpointing().")

    if config.use_lora:
        lora_target_modules = list(getattr(config.lora, "target_modules", DEFAULT_FLUX_LORA_TARGET_MODULES))
        lora_target_modules = resolve_lora_target_modules(transformer, lora_target_modules)
        lora_rank = int(getattr(config.lora, "rank", 64))
        lora_alpha = int(getattr(config.lora, "alpha", 128))
        logger.info("Resolved %d LoRA target modules.", len(lora_target_modules))

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, lora_cfg)

        transformer.add_adapter("old", lora_cfg)
        transformer.set_adapter("default")
    else:
        raise ValueError("This training script is designed for LoRA fine-tuning. Set config.use_lora=True.")

    transformer = transformer.to(device)
    transformer_ddp = (
        DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if world_size > 1
        else transformer
    )

    model = _unwrap_model(transformer_ddp)
    model.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model.set_adapter("default")

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

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

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=True,
    )

    negative_prompt_embeds = None
    negative_text_ids = None
    train_guidance_scale = float(getattr(config, "train_guidance_scale", config.sample.guidance_scale))
    train_embedded_guidance = float(getattr(config, "train_embedded_guidance", 1.0))
    eval_embedded_guidance = float(getattr(config, "eval_embedded_guidance", train_embedded_guidance))
    if train_guidance_scale > 1.0:
        with torch.no_grad():
            negative_prompt_embeds, negative_text_ids = pipeline.encode_prompt(
                prompt="",
                device=text_encoder_device,
                num_images_per_prompt=1,
                max_sequence_length=int(getattr(config, "max_sequence_length", 512)),
                text_encoder_out_layers=tuple(getattr(config, "text_encoder_out_layers", [9, 18, 27])),
            )
        negative_prompt_embeds = negative_prompt_embeds.to(device)
        negative_text_ids = negative_text_ids.to(device)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    stat_tracker = PerPromptStatTracker(config.sample.global_std) if config.per_prompt_stat_tracking else None

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)

    reward_workers_cfg = getattr(config, "reward_client_workers", None)
    reward_workers_env = os.getenv("REWARD_CLIENT_WORKERS", "")
    try:
        if reward_workers_env:
            reward_workers = max(1, int(reward_workers_env))
        elif reward_workers_cfg is not None:
            reward_workers = max(1, int(reward_workers_cfg))
        else:
            reward_workers = 8
    except ValueError:
        reward_workers = 8
    executor = futures.ThreadPoolExecutor(max_workers=reward_workers)

    max_pending_rewards_cfg = getattr(config, "reward_max_pending", None)
    max_pending_rewards_env = os.getenv("REWARD_MAX_PENDING", "")
    try:
        if max_pending_rewards_env:
            max_pending_rewards = max(0, int(max_pending_rewards_env))
        elif max_pending_rewards_cfg is not None:
            max_pending_rewards = max(0, int(max_pending_rewards_cfg))
        else:
            max_pending_rewards = 0
    except ValueError:
        max_pending_rewards = 0

    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info("Resuming from %s", config.resume_from)
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):
            model.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            model.load_adapter(lora_path, adapter_name="old", is_trainable=False)

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and config.mixed_precision == "fp16":
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
        except ValueError:
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.995, update_step_interval=1, device=device)

    for src_param, tgt_param in zip(transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True):
        tgt_param.data.copy_(src_param.detach().data)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    if num_train_timesteps < 1:
        num_train_timesteps = 1

    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running FLUX.2-Klein DiffusionNFT training *****")
    logger.info("  Num Epochs = %s", config.num_epochs)
    logger.info("  Sample batch size per device = %s", config.sample.train_batch_size)
    logger.info("  Train batch size per device = %s", config.train.batch_size)
    logger.info("  Gradient Accumulation steps = %s", config.train.gradient_accumulation_steps)
    logger.info("  Total samples per epoch = %s", samples_per_epoch)
    logger.info("  Total train batch size = %s", total_train_batch_size)

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        model.eval()
        samples_data_list = []
        pending_start_idx = 0
        pending_futures = 0

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            dynamic_ncols=True,
        ):
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            try:
                prompts, prompt_metadata = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                prompts, prompt_metadata = next(train_iter)

            with torch.no_grad():
                prompt_embeds, text_ids = pipeline.encode_prompt(
                    prompt=prompts,
                    device=text_encoder_device,
                    num_images_per_prompt=1,
                    max_sequence_length=int(getattr(config, "max_sequence_length", 512)),
                    text_encoder_out_layers=tuple(getattr(config, "text_encoder_out_layers", [9, 18, 27])),
                )
            prompt_embeds = prompt_embeds.to(device, non_blocking=True)
            text_ids = text_ids.to(device, non_blocking=True)

            do_eval = False
            if i == 0 and not config.debug:
                eval_steps = getattr(config, "eval_steps", 0)
                if eval_steps and global_step > 0 and global_step % eval_steps == 0:
                    do_eval = True
                elif eval_steps == 0 and epoch % config.eval_freq == 0:
                    do_eval = True
            if do_eval:
                eval_fn(
                    pipeline=pipeline,
                    transformer=transformer_ddp,
                    test_dataloader=test_dataloader,
                    config=config,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    global_step=global_step,
                    reward_fn=eval_reward_fn,
                    executor=executor,
                    mixed_precision_dtype=mixed_precision_dtype,
                    eval_embedded_guidance=eval_embedded_guidance,
                    text_encoder_device=text_encoder_device,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    ema=ema,
                    transformer_trainable_parameters=transformer_trainable_parameters,
                )

            if i == 0 and epoch % config.save_freq == 0 and not config.debug:
                save_ckpt(
                    save_dir=config.save_dir,
                    transformer=transformer_ddp,
                    global_step=global_step,
                    rank=rank,
                    ema=ema,
                    transformer_trainable_parameters=transformer_trainable_parameters,
                    config=config,
                    optimizer=optimizer,
                    scaler=scaler,
                )

            model.set_adapter("old")
            with torch.no_grad():
                generator = None
                if bool(getattr(config.sample, "deterministic", False)):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(config.seed + epoch * config.sample.num_batches_per_epoch + i + rank)

                images, rollout_latents, timesteps, sigmas, latent_ids = sample_flux_rollout(
                    pipeline=pipeline,
                    transformer=model,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=train_guidance_scale,
                    device=device,
                    generator=generator,
                    embedded_guidance=train_embedded_guidance,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_text_ids=negative_text_ids,
                    mixed_precision_dtype=mixed_precision_dtype,
                )
            model.set_adapter("default")

            if latent_ids.dim() == 2:
                latent_ids = latent_ids.unsqueeze(0).repeat(len(prompts), 1, 1)
            elif latent_ids.dim() == 3 and latent_ids.shape[0] == 1:
                latent_ids = latent_ids.repeat(len(prompts), 1, 1)

            timesteps = timesteps.to(device).repeat(len(prompts), 1)
            sigmas = sigmas.to(device).repeat(len(prompts), 1)
            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)

            samples_data_list.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "text_ids": text_ids,
                    "latent_ids": latent_ids.to(device),
                    "timesteps": timesteps,
                    "next_timesteps": torch.cat([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "sigmas": sigmas,
                    "next_sigmas": torch.cat([sigmas[:, 1:], torch.zeros_like(sigmas[:, :1])], dim=1),
                    "latents_clean": rollout_latents[:, -1],
                    "rewards_future": rewards_future,
                    "prompts": list(prompts),
                }
            )

            pending_futures += 1
            if max_pending_rewards and pending_futures > max_pending_rewards:
                sample_item = samples_data_list[pending_start_idx]
                rewards, _ = sample_item["rewards_future"].result()
                sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
                del sample_item["rewards_future"]
                pending_start_idx += 1
                pending_futures -= 1

        for sample_item in tqdm(
            samples_data_list,
            desc="Waiting rewards",
            disable=not is_main_process(rank),
            dynamic_ncols=True,
        ):
            if "rewards_future" not in sample_item:
                continue
            rewards, _ = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        collated_samples = {
            "prompt_embeds": torch.cat([s["prompt_embeds"] for s in samples_data_list], dim=0),
            "text_ids": torch.cat([s["text_ids"] for s in samples_data_list], dim=0),
            "latent_ids": torch.cat([s["latent_ids"] for s in samples_data_list], dim=0),
            "timesteps": torch.cat([s["timesteps"] for s in samples_data_list], dim=0),
            "next_timesteps": torch.cat([s["next_timesteps"] for s in samples_data_list], dim=0),
            "sigmas": torch.cat([s["sigmas"] for s in samples_data_list], dim=0),
            "next_sigmas": torch.cat([s["next_sigmas"] for s in samples_data_list], dim=0),
            "latents_clean": torch.cat([s["latents_clean"] for s in samples_data_list], dim=0),
            "prompts": [p for s in samples_data_list for p in s["prompts"]],
            "rewards": {
                key: torch.cat([s["rewards"][key] for s in samples_data_list], dim=0)
                for key in samples_data_list[0]["rewards"].keys()
            },
        }

        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()
            prompts_to_log = prompts
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu().numpy()
            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(10, len(images_to_log))
                for idx in range(num_to_log):
                    pil = tensor_to_pil(images_to_log[idx], resolution=config.resolution)
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx][:120]} | avg: {float(rewards_to_log[idx]):.2f}",
                            )
                            for idx in range(num_to_log)
                        ]
                    },
                    step=global_step,
                )

        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):
            reward_log_values = {}
            for k, v in gathered_rewards_dict.items():
                flat = np.asarray(v).reshape(-1)
                finite = np.isfinite(flat)
                if np.any(flat == -10):
                    finite = finite & (flat != -10)
                if np.any(finite):
                    reward_log_values[f"reward_{k}"] = float(np.mean(flat[finite]))
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
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all, gathered_rewards_dict)
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
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)

        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] != world_size * samples_per_gpu:
            raise ValueError(
                f"advantages shape mismatch: {advantages.shape[0]} vs {world_size} * {samples_per_gpu}"
            )

        collated_samples["advantages"] = torch.from_numpy(advantages.reshape(world_size, samples_per_gpu, -1)[rank]).to(device)

        if is_main_process(rank):
            logger.info("Advantages abs mean: %.6f", collated_samples["advantages"].abs().mean().item())

        del collated_samples["rewards"]
        del collated_samples["prompts"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        model.train()
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps", "sigmas", "next_sigmas"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches
            samples_batched_list = []
            for k_batch in range(num_batches):
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                samples_batched_list.append({k: v[start:end] for k, v in shuffled_filtered_samples.items()})

            info_accumulated = defaultdict(list)
            for i_train, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: train",
                disable=not is_main_process(rank),
                dynamic_ncols=True,
            ):
                del i_train
                for j_idx in range(num_train_timesteps):
                    x0 = train_sample_batch["latents_clean"]
                    timestep_batch = train_sample_batch["timesteps"][:, j_idx]
                    sigma_batch = train_sample_batch["sigmas"][:, j_idx].float()
                    sigma_expanded = sigma_batch.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())
                    xt = (1 - sigma_expanded) * x0 + sigma_expanded * noise

                    with autocast_cuda(enabled=enable_amp, dtype=mixed_precision_dtype):
                        model.set_adapter("old")
                        with torch.no_grad():
                            old_prediction = flux_predict(
                                transformer=model,
                                latents=xt,
                                timesteps=timestep_batch,
                                prompt_embeds=train_sample_batch["prompt_embeds"],
                                text_ids=train_sample_batch["text_ids"],
                                latent_ids=train_sample_batch["latent_ids"],
                                guidance_scale=train_guidance_scale,
                                embedded_guidance=train_embedded_guidance,
                                negative_prompt_embeds=negative_prompt_embeds,
                                negative_text_ids=negative_text_ids,
                            ).detach()

                        model.set_adapter("default")
                        forward_prediction = flux_predict(
                            transformer=model,
                            latents=xt,
                            timesteps=timestep_batch,
                            prompt_embeds=train_sample_batch["prompt_embeds"],
                            text_ids=train_sample_batch["text_ids"],
                            latent_ids=train_sample_batch["latent_ids"],
                            guidance_scale=train_guidance_scale,
                            embedded_guidance=train_embedded_guidance,
                            negative_prompt_embeds=negative_prompt_embeds,
                            negative_text_ids=negative_text_ids,
                        )

                        with torch.no_grad():
                            with model.disable_adapter():
                                ref_forward_prediction = flux_predict(
                                    transformer=model,
                                    latents=xt,
                                    timesteps=timestep_batch,
                                    prompt_embeds=train_sample_batch["prompt_embeds"],
                                    text_ids=train_sample_batch["text_ids"],
                                    latent_ids=train_sample_batch["latent_ids"],
                                    guidance_scale=train_guidance_scale,
                                    embedded_guidance=train_embedded_guidance,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    negative_text_ids=negative_text_ids,
                                )
                            model.set_adapter("default")

                    loss_terms = {}

                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)

                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (1.0 + config.beta) * old_prediction.detach() - config.beta * forward_prediction

                    x0_prediction = xt - sigma_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=1e-5)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))

                    negative_x0_prediction = xt - sigma_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=1e-5)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss_scale = float(getattr(config.train, "policy_loss_scale", config.train.adv_clip_max))
                    policy_loss = (ori_policy_loss * policy_loss_scale).mean()

                    kl_div = ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    loss = policy_loss + config.train.beta * torch.mean(kl_div)

                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()
                    loss_terms["kl_div_loss"] = torch.mean(kl_div).detach()
                    loss_terms["total_loss"] = loss.detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()

                    if not torch.isfinite(loss):
                        logger.warning("Non-finite loss at epoch=%s step=%s timestep_idx=%s, skip this micro step.", epoch, global_step, j_idx)
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    scaled_loss = loss / effective_grad_accum_steps
                    if config.mixed_precision == "fp16":
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    current_accumulated_steps += 1
                    for key, value in loss_terms.items():
                        info_accumulated[key].append(value)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if config.mixed_precision == "fp16":
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_trainable_parameters, config.train.max_grad_norm)

                        if config.mixed_precision == "fp16":
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                        gradient_update_times += 1
                        reduced_log_info = {
                            k: all_reduce_mean(torch.mean(torch.stack(v_list)).item(), device, world_size)
                            for k, v_list in info_accumulated.items()
                        }

                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                },
                                step=global_step,
                            )

                        global_step += 1
                        info_accumulated = defaultdict(list)

                        if config.train.ema and ema is not None:
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
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().data * (1.0 - decay))

    if is_main_process(rank):
        wandb.finish()

    executor.shutdown(wait=True)
    cleanup_distributed(world_size)


if __name__ == "__main__":
    app.run(main)
