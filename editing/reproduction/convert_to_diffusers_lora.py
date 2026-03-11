import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA keys to diffusers-compatible keys.")
    parser.add_argument("--lora-dir", required=True, help="Directory containing adapter_model.safetensors")
    parser.add_argument("--input-name", default="adapter_model.safetensors")
    parser.add_argument("--output-name", default="adapter_model_converted.safetensors")
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    input_path = lora_dir / args.input_name
    output_path = lora_dir / args.output_name

    state_dict = load_file(str(input_path))
    converted = {key.replace("base_model.model", "transformer"): value for key, value in state_dict.items()}
    save_file(converted, str(output_path))
    print(f"Saved converted LoRA to {output_path}")


if __name__ == "__main__":
    main()
