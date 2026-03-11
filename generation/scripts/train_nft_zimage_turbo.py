# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from concurrent import futures
import contextlib
import datetime
import fnmatch
import gc
import glob
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

# Allow importing local DiffSynth-Studio package.
_DIFFSYNTH_ROOT = _ROOT / "DiffSynth-Studio"
if str(_DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_DIFFSYNTH_ROOT))

import flow_grpo.rewards
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.stat_tracking import PerPromptStatTracker

try:
    from diffsynth.core import ModelConfig
    from diffsynth.diffusion.training_module import DiffusionTrainingModule
    from diffsynth.pipelines.z_image import ZImagePipeline, ZImageUnit_PromptEmbedder, model_fn_z_image_turbo
except ImportError as exc:
    raise ImportError(
        "Failed to import DiffSynth-Studio Z-Image modules. "
        "Please ensure DiffSynth-Studio is available at DiffusionNFT/DiffSynth-Studio."
    ) from exc


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "config/nft_zimage_turbo.py:zimage_turbo_sharegpt_qwenvl",
    "Training configuration.",
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
tqdm = tqdm.tqdm

DEFAULT_ZIMAGE_LORA_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "w1",
    "w2",
    "w3",
]


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


def log_epoch_reward_details(epoch, rewards_dict):
    detail_parts = []
    for key in sorted(rewards_dict.keys()):
        values = np.asarray(rewards_dict[key]).reshape(-1)
        if values.size == 0:
            detail_parts.append(f"{key}: empty")
            continue

        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        if finite_values.size == 0:
            detail_parts.append(f"{key}: n={values.size}, finite=0")
            continue

        # Qwen-VL rewards can contain -10 for invalid samples; prioritize stats on valid subset.
        valid_values = finite_values[finite_values != -10]
        stats_source = valid_values if valid_values.size > 0 else finite_values

        detail_parts.append(
            (
                f"{key}: n={values.size}, finite={finite_values.size}, valid={valid_values.size}, "
                f"mean={float(np.mean(stats_source)):.4f}, std={float(np.std(stats_source)):.4f}, "
                f"min={float(np.min(stats_source)):.4f}, p25={float(np.percentile(stats_source, 25)):.4f}, "
                f"p50={float(np.percentile(stats_source, 50)):.4f}, p75={float(np.percentile(stats_source, 75)):.4f}, "
                f"max={float(np.max(stats_source)):.4f}"
            )
        )

    logger.info("Epoch %s reward details | %s", epoch, " | ".join(detail_parts))


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


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


def _build_model_configs(config, device):
    helper = DiffusionTrainingModule()
    model_paths_raw = getattr(config.pretrained, "model_paths", None) or None
    model_id_with_origin_paths_raw = getattr(config.pretrained, "model_id_with_origin_paths", None) or None

    model_path_entries = []
    remote_model_specs = []

    def _append_model_path_entry(entry):
        entry = str(entry).strip()
        if not entry:
            return
        if any(ch in entry for ch in "*?[]"):
            matched = sorted(glob.glob(entry, recursive=True))
            if matched:
                files = [path for path in matched if os.path.isfile(path)]
                if files:
                    model_path_entries.append(files if len(files) > 1 else files[0])
            return
        if os.path.isdir(entry):
            grouped_matches = []
            for pattern in (
                "transformer/*.safetensors",
                "text_encoder/*.safetensors",
                "vae/diffusion_pytorch_model.safetensors",
                "vae/*.safetensors",
            ):
                files = sorted(glob.glob(os.path.join(entry, pattern)))
                files = [path for path in files if os.path.isfile(path)]
                if files:
                    grouped_matches.append(files if len(files) > 1 else files[0])
            if grouped_matches:
                model_path_entries.extend(grouped_matches)
                return
            # Fallback: keep all safetensors in one grouped entry.
            discovered = sorted(
                path for path in glob.glob(os.path.join(entry, "**/*.safetensors"), recursive=True) if os.path.isfile(path)
            )
            if discovered:
                model_path_entries.append(discovered if len(discovered) > 1 else discovered[0])
            return
        model_path_entries.append(entry)

    if isinstance(model_paths_raw, (list, tuple)):
        for path in model_paths_raw:
            _append_model_path_entry(path)
    elif isinstance(model_paths_raw, str):
        model_paths_raw = model_paths_raw.strip()
        if model_paths_raw:
            if model_paths_raw.startswith("["):
                parsed_paths = json.loads(model_paths_raw)
                for path in parsed_paths:
                    _append_model_path_entry(path)
            else:
                for part in model_paths_raw.split(","):
                    _append_model_path_entry(part)

    if isinstance(model_id_with_origin_paths_raw, str) and model_id_with_origin_paths_raw.strip():
        for spec in [part.strip() for part in model_id_with_origin_paths_raw.split(",") if part.strip()]:
            if ":" in spec:
                split_id = spec.rfind(":")
                model_id_or_path = spec[:split_id]
                origin_pattern = spec[split_id + 1 :]
                if os.path.exists(model_id_or_path):
                    matched_files = sorted(glob.glob(os.path.join(model_id_or_path, origin_pattern)))
                    if not matched_files:
                        raise FileNotFoundError(
                            f"No local files matched spec `{spec}`. "
                            f"Checked `{os.path.join(model_id_or_path, origin_pattern)}`"
                        )
                    model_path_entries.append(matched_files if len(matched_files) > 1 else matched_files[0])
                else:
                    remote_model_specs.append(spec)
            elif os.path.exists(spec):
                model_path_entries.append(spec)
            else:
                remote_model_specs.append(spec)

    dedup_paths = []
    seen_paths = set()
    for path_entry in model_path_entries:
        if isinstance(path_entry, list):
            key = ("list", tuple(path_entry))
        else:
            key = ("str", path_entry)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        dedup_paths.append(path_entry)

    model_paths = json.dumps(dedup_paths) if dedup_paths else None
    model_id_with_origin_paths = ",".join(remote_model_specs) if remote_model_specs else None

    if not model_paths and not model_id_with_origin_paths:
        raise ValueError("Need one of config.pretrained.model_paths / config.pretrained.model_id_with_origin_paths")

    model_configs = helper.parse_model_configs(
        model_paths=model_paths,
        model_id_with_origin_paths=model_id_with_origin_paths,
        fp8_models=getattr(config.pretrained, "fp8_models", None),
        offload_models=getattr(config.pretrained, "offload_models", None),
        device=device,
    )

    tokenizer_path = getattr(config.pretrained, "tokenizer_path", None) or None
    tokenizer_config = helper.parse_path_or_model_id(
        tokenizer_path,
        default_value=ModelConfig(
            model_id=os.getenv("ZIMAGE_DEFAULT_MODEL_ID", "models/Z-Image-Turbo"),
            origin_file_pattern="tokenizer/",
        ),
    )
    return model_configs, tokenizer_config


def _materialize_to_device_early(pipeline, device, load_dtype):
    if hasattr(pipeline, "dit") and pipeline.dit is not None:
        pipeline.dit.to(device=device, dtype=load_dtype)
    if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
        pipeline.text_encoder.to(device=device, dtype=load_dtype)
    if hasattr(pipeline, "vae_encoder") and pipeline.vae_encoder is not None:
        pipeline.vae_encoder.to(device=device, dtype=load_dtype)
    if hasattr(pipeline, "vae_decoder") and pipeline.vae_decoder is not None:
        pipeline.vae_decoder.to(device=device, dtype=load_dtype)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_pipeline_staggered(config, rank, world_size, device):
    load_dtype = _resolve_model_load_dtype(config)
    model_configs, tokenizer_config = _build_model_configs(config, device)

    def _load():
        pipe = ZImagePipeline.from_pretrained(
            torch_dtype=load_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            enable_npu_patch=bool(getattr(config, "enable_npu_patch", False)),
        )
        _materialize_to_device_early(pipe, device, load_dtype)
        return pipe

    if world_size <= 1:
        logger.info("Loading Z-Image-Turbo pipeline (dtype=%s)", str(load_dtype))
        return _load()

    pipeline = None
    for load_rank in range(world_size):
        if rank == load_rank:
            logger.info(
                "Rank %d/%d loading Z-Image-Turbo pipeline (dtype=%s)",
                rank,
                world_size,
                str(load_dtype),
            )
            pipeline = _load()
        dist.barrier()
    return pipeline


def resolve_lora_target_modules(model, target_patterns):
    module_map = dict(model.named_modules())
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


def encode_zimage_prompts(pipe, prompts, prompt_embedder, device):
    pipe.load_models_to_device(("text_encoder",))
    with torch.no_grad():
        embeds = prompt_embedder.encode_prompt(pipe, prompt=list(prompts), device=device)
    embeds = [emb.to(device=device, dtype=pipe.torch_dtype) for emb in embeds]
    return embeds


def zimage_predict_batch(
    dit,
    latents,
    timesteps,
    prompt_embeds,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
):
    preds = []
    for idx in range(latents.shape[0]):
        latent_i = latents[idx : idx + 1]
        timestep_i = timesteps[idx : idx + 1].to(device=latents.device, dtype=latents.dtype)
        pred_i = model_fn_z_image_turbo(
            dit=dit,
            latents=latent_i,
            timestep=timestep_i,
            prompt_embeds=prompt_embeds[idx],
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        preds.append(pred_i)
    return torch.cat(preds, dim=0)


def zimage_cfg_predict_batch(
    dit,
    latents,
    timesteps,
    prompt_embeds,
    guidance_scale,
    negative_prompt_embeds,
    use_gradient_checkpointing,
    use_gradient_checkpointing_offload,
):
    cond_pred = zimage_predict_batch(
        dit=dit,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    if guidance_scale <= 1.0 or negative_prompt_embeds is None:
        return cond_pred

    uncond_pred = zimage_predict_batch(
        dit=dit,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=negative_prompt_embeds,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    return uncond_pred + guidance_scale * (cond_pred - uncond_pred)


def sample_zimage_rollout(
    pipeline,
    dit,
    prompt_embeds,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    device,
    seed_base,
    rank,
    deterministic,
    sigma_shift,
    denoising_strength,
    negative_prompt_embeds,
    mixed_precision_dtype,
):
    batch_size = len(prompt_embeds)

    pipeline.scheduler.set_timesteps(
        num_inference_steps,
        denoising_strength=denoising_strength,
        shift=sigma_shift,
    )
    timesteps = pipeline.scheduler.timesteps.to(device=device, dtype=torch.float32)
    sigmas = pipeline.scheduler.sigmas.to(device=device, dtype=torch.float32)

    latents = []
    for idx in range(batch_size):
        seed = None
        if deterministic:
            seed = seed_base + idx + rank * 100000
        latent = pipeline.generate_noise(
            shape=(1, 16, height // 8, width // 8),
            seed=seed,
            rand_device="cpu",
            rand_torch_dtype=pipeline.torch_dtype,
            device=device,
            torch_dtype=pipeline.torch_dtype,
        )
        latents.append(latent)
    latents = torch.cat(latents, dim=0)
    all_latents = [latents.detach()]

    for t in timesteps:
        timestep_batch = t.repeat(batch_size).to(device=device, dtype=pipeline.torch_dtype)
        with autocast_cuda(enabled=(mixed_precision_dtype is not None), dtype=mixed_precision_dtype):
            noise_pred = zimage_cfg_predict_batch(
                dit=dit,
                latents=latents,
                timesteps=timestep_batch,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
                negative_prompt_embeds=negative_prompt_embeds,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )
        latents = pipeline.scheduler.step(noise_pred, t, latents)
        all_latents.append(latents.detach())

    pipeline.load_models_to_device(("vae_decoder",))
    with torch.no_grad():
        image_latents = pipeline.vae_decoder(latents)
        images = ((image_latents.float() + 1.0) / 2.0).clamp(0, 1)

    all_latents = torch.stack(all_latents, dim=1)
    return images, all_latents, timesteps, sigmas


def save_ckpt(
    save_dir,
    dit_model,
    global_step,
    rank,
    ema,
    trainable_parameters,
    config,
    optimizer,
    scaler,
):
    if not is_main_process(rank):
        return

    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)

    model_to_save = _unwrap_model(dit_model)

    if config.train.ema and ema is not None:
        ema.copy_ema_to(trainable_parameters, store_temp=True)

    model_to_save.save_pretrained(save_root_lora)
    torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

    if config.train.ema and ema is not None:
        ema.copy_temp_to(trainable_parameters)

    logger.info("Saved checkpoint to %s", save_root)


def eval_fn(
    pipeline,
    dit_model,
    test_dataloader,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    prompt_embedder,
    eval_guidance_scale,
    eval_sigma_shift,
    eval_denoising_strength,
    ema,
    trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(trainable_parameters, store_temp=True)

    model = _unwrap_model(dit_model)
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
        prompt_embeds = encode_zimage_prompts(pipeline, prompts, prompt_embedder, device)
        negative_prompt_embeds = None
        if eval_guidance_scale > 1.0:
            negative_prompt_embeds = encode_zimage_prompts(pipeline, [""] * len(prompts), prompt_embedder, device)

        with torch.no_grad():
            images, _, _, _ = sample_zimage_rollout(
                pipeline=pipeline,
                dit=model,
                prompt_embeds=prompt_embeds,
                height=config.resolution,
                width=config.resolution,
                num_inference_steps=config.sample.eval_num_steps,
                guidance_scale=eval_guidance_scale,
                device=device,
                seed_base=global_step,
                rank=rank,
                deterministic=True,
                sigma_shift=eval_sigma_shift,
                denoising_strength=eval_denoising_strength,
                negative_prompt_embeds=negative_prompt_embeds,
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
                            for idx, (prompt, reward) in enumerate(
                                zip(prompts_to_log[:num_samples_to_log], sampled_rewards_log)
                            )
                        ],
                        **{f"eval_reward_{key}": float(np.mean(value)) for key, value in final_rewards.items()},
                    },
                    step=global_step,
                )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(trainable_parameters)

    if world_size > 1:
        dist.barrier()


def _apply_permutation_to_list(items, perm):
    perm_cpu = perm.detach().cpu().tolist()
    return [items[idx] for idx in perm_cpu]


def main(_):
    config = FLAGS.config

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

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
            project="diffusion-nft-zimage-turbo",
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
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=(config.mixed_precision == "fp16"))
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=(config.mixed_precision == "fp16"))
    else:
        scaler = GradScaler(enabled=(config.mixed_precision == "fp16"))

    pipeline = load_pipeline_staggered(config, rank, world_size, device)
    prompt_embedder = ZImageUnit_PromptEmbedder()

    for required_name in ("dit", "text_encoder", "vae_decoder"):
        if getattr(pipeline, required_name, None) is None:
            raise ValueError(f"Z-Image pipeline missing required model `{required_name}`. Check model paths/config.")

    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder.to(device=device, dtype=mixed_precision_dtype or torch.float32)
    pipeline.text_encoder.eval()

    if pipeline.vae_encoder is not None:
        pipeline.vae_encoder.requires_grad_(False)
    pipeline.vae_decoder.requires_grad_(False)
    vae_dtype = mixed_precision_dtype if mixed_precision_dtype is not None else torch.float32
    if pipeline.vae_encoder is not None:
        pipeline.vae_encoder.to(device=device, dtype=vae_dtype)
    pipeline.vae_decoder.to(device=device, dtype=vae_dtype)
    if pipeline.vae_encoder is not None:
        pipeline.vae_encoder.eval()
    pipeline.vae_decoder.eval()

    dit = pipeline.dit
    dit.requires_grad_(not config.use_lora)

    if bool(getattr(config.train, "gradient_checkpointing", False)) and hasattr(dit, "enable_gradient_checkpointing"):
        dit.enable_gradient_checkpointing()

    if config.use_lora:
        lora_target_modules = list(getattr(config.lora, "target_modules", DEFAULT_ZIMAGE_LORA_TARGET_MODULES))
        lora_target_modules = resolve_lora_target_modules(dit, lora_target_modules)
        lora_rank = int(getattr(config.lora, "rank", 32))
        lora_alpha = int(getattr(config.lora, "alpha", 32))

        logger.info("Resolved %d LoRA target modules.", len(lora_target_modules))
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=lora_target_modules,
        )

        if config.train.lora_path:
            dit = PeftModel.from_pretrained(dit, config.train.lora_path)
            dit.set_adapter("default")
        else:
            dit = get_peft_model(dit, lora_cfg)

        dit.add_adapter("old", lora_cfg)
        dit.set_adapter("default")
    else:
        raise ValueError("This training script is designed for LoRA fine-tuning. Set config.use_lora=True.")

    pipeline.dit = dit.to(device)
    dit_ddp = (
        DDP(pipeline.dit, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if world_size > 1
        else pipeline.dit
    )

    model = _unwrap_model(dit_ddp)
    model.set_adapter("default")
    trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model.set_adapter("old")
    old_trainable_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model.set_adapter("default")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
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

    train_guidance_scale = float(getattr(config, "train_guidance_scale", config.sample.guidance_scale))
    eval_guidance_scale = float(getattr(config, "eval_guidance_scale", train_guidance_scale))
    train_sigma_shift = float(getattr(config.sample, "sigma_shift", 3.0))
    eval_sigma_shift = float(getattr(config.sample, "eval_sigma_shift", train_sigma_shift))
    train_denoising_strength = float(getattr(config.sample, "denoising_strength", 1.0))
    eval_denoising_strength = float(getattr(config.sample, "eval_denoising_strength", train_denoising_strength))

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
        ema = EMAModuleWrapper(trainable_parameters, decay=0.995, update_step_interval=1, device=device)

    for src_param, tgt_param in zip(trainable_parameters, old_trainable_parameters, strict=True):
        tgt_param.data.copy_(src_param.detach().data)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    if num_train_timesteps < 1:
        num_train_timesteps = 1

    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running Z-Image-Turbo DiffusionNFT training *****")
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

            prompt_embeds = encode_zimage_prompts(pipeline, prompts, prompt_embedder, device)
            negative_prompt_embeds = None
            if train_guidance_scale > 1.0:
                negative_prompt_embeds = encode_zimage_prompts(
                    pipeline,
                    [""] * len(prompts),
                    prompt_embedder,
                    device,
                )

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
                    dit_model=dit_ddp,
                    test_dataloader=test_dataloader,
                    config=config,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    global_step=global_step,
                    reward_fn=eval_reward_fn,
                    executor=executor,
                    mixed_precision_dtype=mixed_precision_dtype,
                    prompt_embedder=prompt_embedder,
                    eval_guidance_scale=eval_guidance_scale,
                    eval_sigma_shift=eval_sigma_shift,
                    eval_denoising_strength=eval_denoising_strength,
                    ema=ema,
                    trainable_parameters=trainable_parameters,
                )

            if i == 0 and epoch % config.save_freq == 0 and not config.debug:
                save_ckpt(
                    save_dir=config.save_dir,
                    dit_model=dit_ddp,
                    global_step=global_step,
                    rank=rank,
                    ema=ema,
                    trainable_parameters=trainable_parameters,
                    config=config,
                    optimizer=optimizer,
                    scaler=scaler,
                )

            model.set_adapter("old")
            with torch.no_grad():
                images, rollout_latents, timesteps, sigmas = sample_zimage_rollout(
                    pipeline=pipeline,
                    dit=model,
                    prompt_embeds=prompt_embeds,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=train_guidance_scale,
                    device=device,
                    seed_base=config.seed + epoch * config.sample.num_batches_per_epoch + i,
                    rank=rank,
                    deterministic=bool(getattr(config.sample, "deterministic", False)),
                    sigma_shift=train_sigma_shift,
                    denoising_strength=train_denoising_strength,
                    negative_prompt_embeds=negative_prompt_embeds,
                    mixed_precision_dtype=mixed_precision_dtype,
                )
            model.set_adapter("default")

            timesteps = timesteps.repeat(len(prompts), 1)
            sigmas = sigmas.repeat(len(prompts), 1)
            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)

            samples_data_list.append(
                {
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
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
            "prompt_embeds": [emb for s in samples_data_list for emb in s["prompt_embeds"]],
            "negative_prompt_embeds": None,
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
        if any(s["negative_prompt_embeds"] is not None for s in samples_data_list):
            collated_samples["negative_prompt_embeds"] = [
                emb
                for s in samples_data_list
                for emb in (s["negative_prompt_embeds"] or [])
            ]

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

        gathered_rewards_raw = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_raw[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):
            log_epoch_reward_details(epoch, gathered_rewards_raw)
            reward_log_values = {}
            for k, v in gathered_rewards_raw.items():
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

        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        gathered_rewards_dict = {
            "avg": gather_tensor_to_all(collated_samples["rewards"]["avg"], world_size).numpy(),
        }

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

        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, samples_per_gpu, -1)[rank]
        ).to(device)

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

        tensor_keys = [
            "timesteps",
            "next_timesteps",
            "sigmas",
            "next_sigmas",
            "latents_clean",
            "advantages",
        ]

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: filtered_samples[k][perm] for k in tensor_keys}
            shuffled_filtered_samples["prompt_embeds"] = _apply_permutation_to_list(filtered_samples["prompt_embeds"], perm)
            if filtered_samples["negative_prompt_embeds"] is not None:
                shuffled_filtered_samples["negative_prompt_embeds"] = _apply_permutation_to_list(
                    filtered_samples["negative_prompt_embeds"], perm
                )
            else:
                shuffled_filtered_samples["negative_prompt_embeds"] = None

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
                batch_item = {k: v[start:end] for k, v in shuffled_filtered_samples.items() if torch.is_tensor(v)}
                batch_item["prompt_embeds"] = shuffled_filtered_samples["prompt_embeds"][start:end]
                if shuffled_filtered_samples["negative_prompt_embeds"] is not None:
                    batch_item["negative_prompt_embeds"] = shuffled_filtered_samples["negative_prompt_embeds"][start:end]
                else:
                    batch_item["negative_prompt_embeds"] = None
                samples_batched_list.append(batch_item)

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

                    use_gc = bool(getattr(config.train, "gradient_checkpointing", False))
                    use_gc_offload = bool(getattr(config.train, "gradient_checkpointing_offload", False))

                    with autocast_cuda(enabled=enable_amp, dtype=mixed_precision_dtype):
                        model.set_adapter("old")
                        with torch.no_grad():
                            old_prediction = zimage_cfg_predict_batch(
                                dit=model,
                                latents=xt,
                                timesteps=timestep_batch,
                                prompt_embeds=train_sample_batch["prompt_embeds"],
                                guidance_scale=train_guidance_scale,
                                negative_prompt_embeds=train_sample_batch["negative_prompt_embeds"],
                                use_gradient_checkpointing=False,
                                use_gradient_checkpointing_offload=False,
                            ).detach()

                        model.set_adapter("default")
                        forward_prediction = zimage_cfg_predict_batch(
                            dit=model,
                            latents=xt,
                            timesteps=timestep_batch,
                            prompt_embeds=train_sample_batch["prompt_embeds"],
                            guidance_scale=train_guidance_scale,
                            negative_prompt_embeds=train_sample_batch["negative_prompt_embeds"],
                            use_gradient_checkpointing=use_gc,
                            use_gradient_checkpointing_offload=use_gc_offload,
                        )

                        with torch.no_grad():
                            with model.disable_adapter():
                                ref_forward_prediction = zimage_cfg_predict_batch(
                                    dit=model,
                                    latents=xt,
                                    timesteps=timestep_batch,
                                    prompt_embeds=train_sample_batch["prompt_embeds"],
                                    guidance_scale=train_guidance_scale,
                                    negative_prompt_embeds=train_sample_batch["negative_prompt_embeds"],
                                    use_gradient_checkpointing=False,
                                    use_gradient_checkpointing_offload=False,
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
                        logger.warning(
                            "Non-finite loss at epoch=%s step=%s timestep_idx=%s, skip this micro step.",
                            epoch,
                            global_step,
                            j_idx,
                        )
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
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, config.train.max_grad_norm)

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
                            ema.step(trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                trainable_parameters,
                old_trainable_parameters,
                strict=True,
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().data * (1.0 - decay))

    if is_main_process(rank):
        wandb.finish()

    executor.shutdown(wait=True)
    cleanup_distributed(world_size)


if __name__ == "__main__":
    app.run(main)
