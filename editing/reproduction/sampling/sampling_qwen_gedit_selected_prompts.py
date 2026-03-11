import argparse
import os
import re
import types
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk
from tqdm import tqdm


DEFAULT_TARGET_PROMPTS = [
    "Remove the Christmas tree on the left side of the image.",
    "Replace the dog with a robot."
]


def ensure_pil_exiftags_compat() -> None:
    # datasets Image feature may expect Pillow symbols missing in old Pillow builds.
    import PIL.ExifTags
    import PIL.Image

    if not hasattr(PIL.Image, "ExifTags"):
        PIL.Image.ExifTags = PIL.ExifTags
    if not hasattr(PIL.Image.ExifTags, "Base"):
        PIL.Image.ExifTags.Base = types.SimpleNamespace(Orientation=274)


def normalize_prompt(text: str) -> str:
    text = " ".join(text.strip().lower().split())
    text = re.sub(r"[.。!?！？;；]+$", "", text)
    text = text.replace("hiah definition", "high definition")
    return text


def get_prompt_from_item(item: Dict) -> str:
    for key in ("input_prompt", "instruction", "prompt"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Cannot find prompt field in sample")


def load_pipeline(pretrained_name_or_path: str, lora_path: Optional[str] = None):
    try:
        from diffusers import QwenImageEditPlusPipeline as QwenPipeline
    except Exception:
        try:
            from diffusers import QwenImageEditPipeline as QwenPipeline
        except Exception:
            from diffusers import DiffusionPipeline as QwenPipeline

    pipeline = QwenPipeline.from_pretrained(
        pretrained_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipeline.to("cuda")

    if lora_path:
        pipeline.load_lora_weights(
            lora_path,
            weight_name="adapter_model_converted.safetensors",
            adapter_name="lora",
        )
        pipeline.set_adapters(["lora"], adapter_weights=[1])
        print(f"Loaded LoRA: {lora_path}")
    else:
        print("No lora path provided, using base model")
    return pipeline


def _sample_worker(
    jobs: List[Tuple[int, int]],
    gedit_bench_path: str,
    pretrained_name_or_path: str,
    lora_path: Optional[str],
    output_dir: str,
    base_seed: int,
    num_repeats: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    guidance_scale: float,
):
    ensure_pil_exiftags_compat()
    dataset = load_from_disk(gedit_bench_path)
    pipeline = load_pipeline(pretrained_name_or_path, lora_path)

    for idx, repeat_idx in tqdm(jobs, desc="Generating"):
        item = dataset[idx]
        if item.get("instruction_language") not in (None, "en"):
            continue

        key = item["key"]
        task_type = item["task_type"]
        prompt = get_prompt_from_item(item)
        input_image = item["input_image_raw"].convert("RGB")

        image_output_dir = os.path.join(output_dir, task_type)
        os.makedirs(image_output_dir, exist_ok=True)
        output_path = os.path.join(image_output_dir, f"{key}_r{repeat_idx + 1}.png")
        if os.path.exists(output_path):
            continue

        # Make every generation use a unique, deterministic seed.
        run_seed = base_seed + idx * num_repeats + repeat_idx
        generator = torch.Generator(device="cuda").manual_seed(run_seed)
        output_image = pipeline(
            prompt=prompt,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            image=input_image,
            negative_prompt=" ",
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        output_image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_name_or_path",
        type=str,
        default="Qwen/Qwen-Image-Edit",
    )
    parser.add_argument(
        "--gedit_bench_path",
        type=str,
        default="data/GEdit-Bench",
    )
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_repeats", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/qwen_image_edit_selected_prompts",
        help="path to save the output images",
    )
    parser.add_argument(
        "--target_prompts",
        nargs="*",
        default=None,
        help="Prompt list to filter. If omitted, uses the 4 prompts in your request.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print matched cases without generation.",
    )
    args = parser.parse_args()

    ensure_pil_exiftags_compat()

    target_prompts = args.target_prompts if args.target_prompts else DEFAULT_TARGET_PROMPTS
    target_norm_to_raw = {normalize_prompt(p): p for p in target_prompts}

    dataset = load_from_disk(args.gedit_bench_path)
    fields = set(dataset.column_names)
    prompt_column = "input_prompt" if "input_prompt" in fields else "instruction"

    prompts = dataset[prompt_column]
    languages = dataset["instruction_language"] if "instruction_language" in fields else None
    task_types = dataset["task_type"] if "task_type" in fields else ["unknown"] * len(dataset)
    keys = dataset["key"] if "key" in fields else [str(i) for i in range(len(dataset))]

    matched_indices: List[int] = []
    grouped_matches = defaultdict(list)
    for idx, prompt in enumerate(prompts):
        if not isinstance(prompt, str):
            continue
        if languages is not None and languages[idx] != "en":
            continue
        norm_prompt = normalize_prompt(prompt)
        if norm_prompt in target_norm_to_raw:
            matched_indices.append(idx)
            grouped_matches[target_norm_to_raw[norm_prompt]].append(
                (idx, keys[idx], task_types[idx], prompt)
            )

    print(f"Prompt column: {prompt_column}")
    print(f"Matched cases: {len(matched_indices)}")
    for target in target_prompts:
        cases = grouped_matches.get(target, [])
        print(f"- {target} -> {len(cases)} case(s)")
        for idx, key, task_type, actual_prompt in cases:
            print(
                f"    idx={idx}, key={key}, task_type={task_type}, actual_prompt={actual_prompt}"
            )

    if not matched_indices:
        raise ValueError("No matched samples found for target prompts.")

    if args.dry_run:
        return

    os.makedirs(args.output_dir, exist_ok=True)
    jobs = [(idx, repeat_idx) for idx in matched_indices for repeat_idx in range(args.num_repeats)]

    try:
        import ray
    except Exception as exc:
        raise ImportError(
            "ray is required for multi-GPU parallel generation. Please install ray in your env."
        ) from exc

    ray.init(ignore_reinit_error=True)
    available_gpus = int(ray.available_resources().get("GPU", 0))
    if available_gpus <= 0:
        raise RuntimeError("No GPU resources found by ray.")

    worker_count = min(args.num_gpus, available_gpus)
    print(f"Using {worker_count} GPU workers (available={available_gpus}, requested={args.num_gpus})")

    worker_fn = ray.remote(num_gpus=1)(_sample_worker)
    slices = [jobs[i::worker_count] for i in range(worker_count)]

    ray.get(
        [
            worker_fn.remote(
                sliced_jobs,
                args.gedit_bench_path,
                args.pretrained_name_or_path,
                args.lora_path,
                args.output_dir,
                args.seed,
                args.num_repeats,
                args.num_inference_steps,
                args.true_cfg_scale,
                args.guidance_scale,
            )
            for sliced_jobs in slices
            if sliced_jobs
        ]
    )


if __name__ == "__main__":
    main()
