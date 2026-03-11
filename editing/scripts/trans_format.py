import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert image-edit JSON data into training JSONL metadata.")
    parser.add_argument("--input-json", required=True, help="Input JSON file")
    parser.add_argument("--output-jsonl", required=True, help="Output JSONL file")
    parser.add_argument("--requirement", default="", help="Default requirement field if missing")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    output_path = Path(args.output_jsonl)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            image_value = item.get("image")
            if not image_value:
                image_value = item.get("input_image", [None])[0]
            jsonl_item = {
                "prompt": item.get("prompt") or item.get("input_prompt", ""),
                "image": image_value or "",
                "requirement": item.get("requirement", args.requirement),
            }
            f.write(json.dumps(jsonl_item, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
