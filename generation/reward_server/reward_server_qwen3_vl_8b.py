import os

# Respect external GPU selection; default to GPU 0.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("REWARD_VISIBLE_DEVICES", "0"))

import json
import re
from typing import List, Optional
from vllm import LLM, SamplingParams
import vllm
from PIL import Image
from io import BytesIO
import base64
import pickle
import traceback
import time
import uuid
from flask import Flask, request
import ray
import asyncio
import prompt_template
# if vllm.__version__ != "0.9.2":
#     raise ValueError("vLLM version must be 0.9.2")

# os.environ["VLLM_USE_V1"] = "0"  # IMPORTANT

app = Flask(__name__)

# Global variables
workers = []  # Ray actors for each GPU
MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "Qwen/Qwen3-VL-32B-Instruct")
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))
NUM_TP = int(os.getenv("NUM_TP", "1"))
PROMPT_TEMPLATES = {
    "score_consistency": prompt_template.SCORE_CONSISTENCY,
    "score_execution": prompt_template.SCORE_EXECUTION,
}
LOG_DIR = os.getenv("REWARD_LOG_DIR", "reward_logs")


def _ensure_log_dir(path):
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def _save_reward_records(
    mode,
    prompts,
    metadatas,
    scores,
    reasonings,
):
    log_dir = _ensure_log_dir(LOG_DIR)
    if not log_dir:
        return
    request_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}"
    request_dir = os.path.join(log_dir, mode, request_id)
    os.makedirs(request_dir, exist_ok=True)

    max_len = len(scores)
    for idx in range(max_len):
        record_dir = os.path.join(request_dir, f"{idx:06d}")
        os.makedirs(record_dir, exist_ok=True)
        prompt = prompts[idx] if idx < len(prompts) else ""
        metadata = metadatas[idx] if idx < len(metadatas) else {}
        requirement = metadata.get("requirement", "")
        reasoning = reasonings[idx] if idx < len(reasonings) else ""
        record = {
            "mode": mode,
            "prompt": prompt,
            "requirement": requirement,
            "score": scores[idx],
            "reasoning": reasoning,
        }
        with open(os.path.join(record_dir, "record.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=True, indent=2)

def get_base64(image):
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode("utf-8")
    return encoded_image


@ray.remote(num_gpus=NUM_TP)
class ModelWorker:
    def __init__(self):
        self.llm = None
        self.load_model()

    def load_model(self):
        """Load the Qwen2-VL model using vLLM on specific GPU"""
        self.llm = LLM(
            MODEL_PATH, limit_mm_per_prompt={"image": 3}, tensor_parallel_size=NUM_TP
        )

    def evaluate_image(
        self,
        image_bytes,
        prompt,
        ref_image_bytes=None,
        requirement: str = "",
        score_mode: str = "score_consistency",
    ):
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
        ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])
        prompt_template_text = PROMPT_TEMPLATES.get(score_mode)
        if prompt_template_text is None:
            raise ValueError(f"Unsupported score mode: {score_mode}")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": ref_image},
                    {"type": "image_pil", "image_pil": image},
                    {
                        "type": "text",
                        "text": prompt_template_text.format(
                            prompt=prompt, requirement=requirement
                        ),
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation)

    def evaluate_images(
        self,
        image_bytes_list,
        prompts,
        ref_image_bytes_list=None,
        requirements: Optional[List[str]] = None,
        score_mode: str = "score_consistency",
    ):
        if not requirements:
            requirements = [""] * len(prompts)
        if ref_image_bytes_list is None:
            ref_image_bytes_list = [None] * len(prompts)

        prompt_template_text = PROMPT_TEMPLATES.get(score_mode)
        if prompt_template_text is None:
            raise ValueError(f"Unsupported score mode: {score_mode}")

        conversations = []
        for image_bytes, prompt, ref_image_bytes, requirement in zip(
            image_bytes_list, prompts, ref_image_bytes_list, requirements
        ):
            image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
            ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_pil", "image_pil": ref_image},
                            {"type": "image_pil", "image_pil": image},
                            {
                                "type": "text",
                                "text": prompt_template_text.format(
                                    prompt=prompt, requirement=requirement
                                ),
                            },
                        ],
                    }
                ]
            )

        return self._vllm_evaluate_batch(conversations)

    def _extract_score_and_reasoning(self, text: str):
        if not text:
            return None, ""

        stripped = text.strip()
        json_candidate = None
        if "{" in stripped and "}" in stripped:
            json_candidate = stripped[stripped.find("{") : stripped.rfind("}") + 1]
        if json_candidate:
            try:
                payload = json.loads(json_candidate)
                reasoning = payload.get("reasoning", "")
                score = payload.get("score")
                if isinstance(score, list) and score:
                    score = score[0]
                if isinstance(score, str):
                    score = score.strip()
                    if score.isdigit():
                        score = int(score)
                if isinstance(score, (int, float)) and 1 <= score <= 5:
                    return int(score), reasoning
            except json.JSONDecodeError:
                pass

        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', stripped)
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        score_match = re.search(r'"score"\s*:\s*\[?\s*([1-5])\s*\]?', stripped)
        if score_match:
            return int(score_match.group(1)), reasoning
        return None, reasoning

    def _vllm_evaluate(self, conversation, max_tokens=256):
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.chat(conversation, sampling_params=sampling_params)
        try:
            if outputs and outputs[0].outputs:
                text = outputs[0].outputs[0].text
                score, reasoning = self._extract_score_and_reasoning(text)
                if score is None:
                    print(f"Failed to extract score from: {text}")
                    return 0, reasoning
                print(f"Score: {score}")
                return score, reasoning
            else:
                print("No outputs received")
                return 0, ""
        except Exception as e:
            print(f"Error in _vllm_evaluate: {e}")
            score = 0
            text = ""

        return score, text

    def _vllm_evaluate_batch(self, conversations, max_tokens=256):
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.chat(conversations, sampling_params=sampling_params)
        results = []
        if not outputs:
            return [(0, "") for _ in range(len(conversations))]

        for output in outputs:
            try:
                if output and output.outputs:
                    text = output.outputs[0].text
                    score, reasoning = self._extract_score_and_reasoning(text)
                    if score is None:
                        print(f"Failed to extract score from: {text}")
                        results.append((0, reasoning))
                    else:
                        results.append((score, reasoning))
                else:
                    print("No outputs received")
                    results.append((0, ""))
            except Exception as e:
                print(f"Error in _vllm_evaluate_batch: {e}")
                results.append((0, ""))

        return results


def initialize_ray_workers(num_gpus=8, num_tp=4):
    global workers
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()

    # Create workers for each GPU
    workers = []
    for _ in range(num_gpus // num_tp):
        worker = ModelWorker.remote()
        workers.append(worker)

    print(f"Initialized {num_gpus//num_tp} Ray workers")
    return workers


async def evaluate_images_async(
    image_bytes_list,
    prompts,
    ref_image_bytes_list=None,
    requirements: Optional[List[str]] = None,
    score_mode: str = "score_consistency",
):
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    if not requirements:
        requirements = [""] * len(prompts)
    if ref_image_bytes_list is None:
        ref_image_bytes_list = [None] * len(prompts)

    batch_size = min(len(image_bytes_list), len(prompts))
    per_worker_batches = [[] for _ in range(len(workers))]
    for i, (image_bytes, prompt, ref_image_bytes, requirement) in enumerate(
        zip(
            image_bytes_list[:batch_size],
            prompts[:batch_size],
            ref_image_bytes_list[:batch_size],
            requirements[:batch_size],
        )
    ):
        worker_idx = i % len(workers)
        per_worker_batches[worker_idx].append(
            (i, image_bytes, prompt, ref_image_bytes, requirement)
        )

    tasks = []
    for worker_idx, batch in enumerate(per_worker_batches):
        if not batch:
            continue
        indices, images, batch_prompts, ref_images, batch_requirements = zip(*batch)
        task = workers[worker_idx].evaluate_images.remote(
            list(images),
            list(batch_prompts),
            list(ref_images),
            list(batch_requirements),
            score_mode,
        )
        tasks.append((indices, task))

    results = [None] * batch_size
    for indices, task in tasks:
        batch_results = ray.get(task)
        for idx, item in zip(indices, batch_results):
            results[idx] = item

    return results


def evaluate_images(
    image_bytes_list,
    prompts,
    ref_image_bytes_list=None,
    requirements=None,
    score_mode: str = "score_consistency",
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_images_async(
                image_bytes_list,
                prompts,
                ref_image_bytes_list,
                requirements,
                score_mode,
            )
        )
        return scores
    finally:
        loop.close()


@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()

    assert mode in ["score_consistency", "score_execution"], "Invalid mode"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        ref_image_bytes_list = data.get("ref_images", None)
        prompts = data["prompts"]
        metadatas = data.get("metadatas", [])
        requirements = []
        for metadata in metadatas:
            requirements.append(metadata.get("requirement", ""))

        results = evaluate_images(
            image_bytes_list,
            prompts,
            ref_image_bytes_list,
            requirements,
            mode,
        )
        scores = []
        reasonings = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                score, reasoning = item[0], item[1]
            else:
                score, reasoning = item, ""
            scores.append(score)
            reasonings.append(reasoning)
        _save_reward_records(
            mode,
            prompts,
            metadatas,
            scores,
            reasonings,
        )

        response = {"scores": scores, "reasonings": reasonings}
        response = pickle.dumps(response)
        returncode = 200
    except KeyError as e:
        response = f"KeyError: {str(e)}"
        response = response.encode("utf-8")
        returncode = 500
    except Exception as e:
        response = traceback.format_exc()
        response = response.encode("utf-8")
        returncode = 500

    return response, returncode


if __name__ == "__main__":
    initialize_ray_workers(NUM_GPUS, NUM_TP)
    print(f"Starting Flask server with {NUM_GPUS//NUM_TP} Ray workers...")
    app.run(host="0.0.0.0", port=12342, debug=False)
