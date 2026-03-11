import argparse
import os
import torch

from diffusers import StableDiffusion3Pipeline
from peft import PeftModel


DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def resolve_lora_dir(lora_path, checkpoint_path):
    if lora_path:
        return lora_path
    if checkpoint_path:
        return os.path.join(checkpoint_path, "lora")
    raise ValueError("Please provide --lora_path or --checkpoint_path.")


def main():
    parser = argparse.ArgumentParser(description="Merge SD3.5 LoRA into base model and save merged weights.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base SD3.5 model ID or local path.",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="",
        help="Path to LoRA directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Path to checkpoint dir that contains a 'lora' subfolder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged weights.",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        default="transformer",
        choices=["transformer", "pipeline"],
        help="Save only merged transformer or the whole pipeline.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Weights dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for merging.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TF32 for matmul and cuDNN.",
    )
    args = parser.parse_args()

    lora_dir = resolve_lora_dir(args.lora_path, args.checkpoint_path)
    if not os.path.exists(lora_dir):
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = DTYPE_MAP[args.dtype]
    print(f"Loading base model: {args.base_model}")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.base_model, torch_dtype=dtype)
    pipe.to(args.device)

    print(f"Loading LoRA from: {lora_dir}")
    transformer = PeftModel.from_pretrained(pipe.transformer, lora_dir)
    print("Merging LoRA into base transformer...")
    transformer = transformer.merge_and_unload()
    pipe.transformer = transformer

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_mode == "pipeline":
        print(f"Saving merged pipeline to: {args.output_dir}")
        pipe.save_pretrained(args.output_dir)
    else:
        print(f"Saving merged transformer to: {args.output_dir}")
        pipe.transformer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
