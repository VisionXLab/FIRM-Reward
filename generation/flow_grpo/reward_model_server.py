import os
import sys
from pathlib import Path

# Respect external GPU selection. If you need to force a subset, set
# REWARD_FORCE_CUDA_VISIBLE_DEVICES (e.g., "0,1,2,3").
_force_visible = os.getenv("REWARD_FORCE_CUDA_VISIBLE_DEVICES")
if _force_visible:
    os.environ["CUDA_VISIBLE_DEVICES"] = _force_visible

# Prefer external configuration; only set conservative defaults here.
os.environ.setdefault("REWARD_MAX_TOKENS", "8192")
os.environ.setdefault("REWARD_RAY_GET_TIMEOUT", "360")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("REWARD_VISIBLE_DEVICES", "0")

import json
import math
import re
from typing import List, Optional
from vllm import LLM, SamplingParams
from PIL import Image
from io import BytesIO
import base64
import pickle
import traceback
import time
import uuid
import logging
import threading
from flask import Flask, request
import ray
import asyncio

# Allow running this file directly without installing as a package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flow_grpo import prompt_template

# if vllm.__version__ != "0.9.2":
#     raise ValueError("vLLM version must be 0.9.2")

# os.environ["VLLM_USE_V1"] = "0"  # IMPORTANT

app = Flask(__name__)
LOG_LEVEL = os.getenv("REWARD_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen_vl_reward_server")
_worker_rr = 0
_worker_rr_lock = threading.Lock()
_inflight_sem = None
_max_inflight_env = os.getenv("REWARD_MAX_INFLIGHT")
if _max_inflight_env:
    try:
        _inflight_sem = threading.Semaphore(max(1, int(_max_inflight_env)))
    except ValueError:
        _inflight_sem = None
_max_tokens_env = os.getenv("REWARD_MAX_TOKENS", "8192")
try:
    _max_tokens_default = max(64, int(_max_tokens_env))
except ValueError:
    _max_tokens_default = 8192

# Global variables
workers = []  # Ray actors for each GPU
MODEL_PATH = os.getenv("REWARD_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))
NUM_TP = int(os.getenv("NUM_TP", "1"))
PROMPT_TEMPLATES = {
    "score": prompt_template.SCORE,
    "quality": prompt_template.Quality_PROMPT,
}
LOG_DIR = os.getenv("REWARD_LOG_DIR", "reward_logs")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_score(value, low: float, high: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    if high <= low:
        return 0.0
    return _clamp01((value - low) / (high - low))


def _normalize_blend_mode(value) -> Optional[str]:
    if value is None:
        return None
    mode = str(value).strip().lower()
    if mode in ("ins", "instruction", "instruction_only"):
        return "ins"
    if mode in ("mix", "weighted_mix", "linear_mix"):
        return "mix"
    if mode in ("legacy", "server", "default"):
        return "legacy"
    return None


def _resolve_reward_blend_config(payload) -> tuple[str, float]:
    mode = _normalize_blend_mode(payload.get("reward_blend_mode")) if isinstance(payload, dict) else None
    if mode is None:
        mode = _normalize_blend_mode(
            os.getenv("REWARD_BLEND_MODE", os.getenv("QWEN_VL_REWARD_BLEND_MODE", "legacy"))
        )
    if mode is None:
        mode = "legacy"

    ratio_value = payload.get("reward_quality_ratio") if isinstance(payload, dict) else None
    if ratio_value is None:
        ratio_value = os.getenv("REWARD_QUALITY_RATIO", os.getenv("QWEN_VL_REWARD_QUALITY_RATIO", "0.5"))
    try:
        quality_ratio = float(ratio_value)
    except (TypeError, ValueError):
        quality_ratio = 0.5
    quality_ratio = _clamp01(quality_ratio)
    return mode, quality_ratio


def _quality_from_total_or_subscores(quality_score, quality_subscores):
    if isinstance(quality_subscores, (list, tuple)) and len(quality_subscores) >= 3:
        parsed_subscores = []
        for item in quality_subscores[:3]:
            try:
                value = float(item)
            except (TypeError, ValueError):
                parsed_subscores = []
                break
            if not math.isfinite(value):
                parsed_subscores = []
                break
            parsed_subscores.append(value)
        if len(parsed_subscores) == 3:
            return _normalize_score(min(parsed_subscores), 1.0, 5.0), True

    try:
        total = float(quality_score)
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(total):
        return None, False
    return _normalize_score(total, 3.0, 15.0), True


def _compose_reward(ins_score, quality_score, quality_subscores, reward_mode: str, quality_ratio: float):
    ins = _normalize_score(ins_score, 1.0, 5.0)
    quality, quality_available = _quality_from_total_or_subscores(quality_score, quality_subscores)
    if reward_mode == "ins":
        reward = ins
    elif reward_mode == "mix":
        if not quality_available:
            reward = ins
        else:
            reward = ins * (0.4 + 0.6 * quality)
    else:
        quality = quality if quality_available else 0.0
        reward = ins * (0.5 + 0.5 * quality)
    return _clamp01(reward), ins, quality


def _reward_to_server_score(reward: float) -> float:
    # Keep response compatible with downstream normalization: (score - 1) / 4.
    return _clamp01(reward) * 4.0 + 1.0


def _coerce_quality_subscores(values):
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return None
    subscores = []
    for item in values[:3]:
        try:
            value = int(float(item))
        except (TypeError, ValueError):
            return None
        if value < 1 or value > 5:
            return None
        subscores.append(value)
    return subscores


def _parse_quality_reasoning_payload(reasoning):
    payload = None
    if isinstance(reasoning, dict):
        payload = reasoning
    elif isinstance(reasoning, str):
        stripped = reasoning.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return str(reasoning) if isinstance(reasoning, str) else "", None

    reasoning_text = payload.get("reasoning", "")
    if not isinstance(reasoning_text, str):
        reasoning_text = str(reasoning_text)

    quality_subscores = _coerce_quality_subscores(
        payload.get("subscores", payload.get("score"))
    )
    return reasoning_text, quality_subscores


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
        requirement: str = "",
        score_mode: str = "score",
    ):
        image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
        prompt_template_text = PROMPT_TEMPLATES.get(score_mode)
        if prompt_template_text is None:
            raise ValueError(f"Unsupported score mode: {score_mode}")
        rendered_prompt = self._render_prompt_text(
            prompt_template_text, prompt, requirement
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {
                        "type": "text",
                        "text": rendered_prompt,
                    },
                ],
            },
        ]
        return self._vllm_evaluate(conversation, score_mode=score_mode)

    def evaluate_images(
        self,
        image_bytes_list,
        prompts,
        requirements: Optional[List[str]] = None,
        score_mode: str = "score",
    ):
        if not requirements:
            requirements = [""] * len(prompts)

        prompt_template_text = PROMPT_TEMPLATES.get(score_mode)
        if prompt_template_text is None:
            raise ValueError(f"Unsupported score mode: {score_mode}")

        conversations = []
        for image_bytes, prompt, requirement in zip(
            image_bytes_list, prompts, requirements
        ):
            image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
            rendered_prompt = self._render_prompt_text(
                prompt_template_text, prompt, requirement
            )
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_pil", "image_pil": image},
                            {
                                "type": "text",
                                "text": rendered_prompt,
                            },
                        ],
                    }
                ]
            )

        return self._vllm_evaluate_batch(conversations, score_mode=score_mode)

    def _render_prompt_text(
        self,
        prompt_template_text: str,
        prompt: str,
        requirement: str,
    ) -> str:
        # Only the instruction-following prompt requires formatting fields.
        if "{prompt}" in prompt_template_text or "{requirement}" in prompt_template_text:
            return prompt_template_text.format(prompt=prompt, requirement=requirement)
        return prompt_template_text

    def _extract_quality_score_and_reasoning(self, text: str):
        if not text:
            return None, ""

        stripped = text.strip()
        reasoning = ""

        json_candidate = None
        if "{" in stripped and "}" in stripped:
            json_candidate = stripped[stripped.find("{") : stripped.rfind("}") + 1]
        if json_candidate:
            try:
                payload = json.loads(json_candidate)
                payload_reasoning = payload.get("reasoning", "")
                if isinstance(payload_reasoning, str):
                    reasoning = payload_reasoning
                score = payload.get("score")
                values = _coerce_quality_subscores(score)
                if values is not None:
                    return sum(values), {"reasoning": reasoning, "subscores": values}
            except json.JSONDecodeError:
                pass

        score_match = re.search(
            r'"score"\s*:\s*\[\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*\]',
            stripped,
        )
        if score_match:
            values = [int(score_match.group(i)) for i in range(1, 4)]
            quality_total = sum(values)
            return quality_total, {"reasoning": reasoning, "subscores": values}

        list_match = re.search(
            r"\[\s*([1-5])\s*,\s*([1-5])\s*,\s*([1-5])\s*\]",
            stripped,
        )
        if list_match:
            values = [int(list_match.group(i)) for i in range(1, 4)]
            quality_total = sum(values)
            return quality_total, {"reasoning": reasoning, "subscores": values}

        return None, reasoning

    def _extract_score_and_reasoning(self, text: str, score_mode: str = "score"):
        if score_mode == "quality":
            return self._extract_quality_score_and_reasoning(text)
        if not text:
            return None, ""

        stripped = text.strip()
        boxed_match = re.search(
            r"\\boxed\{+\s*(?:\[\s*)?([1-5])(?:\s*\])?\s*\}+",
            stripped,
        )
        if boxed_match:
            score = int(boxed_match.group(1))
            analysis_match = re.search(
                r"\*\*Final Analysis\*\*:\s*(.*?)(?:\n\*\*Final Alignment Rating|\Z)",
                stripped,
                re.S,
            )
            reasoning = analysis_match.group(1).strip() if analysis_match else ""
            return score, reasoning

        rating_match = re.search(
            r"Final Alignment Rating\s*:\s*(?:\[\s*)?([1-5])(?:\s*\])?",
            stripped,
        )
        if rating_match:
            score = int(rating_match.group(1))
            analysis_match = re.search(
                r"\*\*Final Analysis\*\*:\s*(.*?)(?:\n\*\*Final Alignment Rating|\Z)",
                stripped,
                re.S,
            )
            reasoning = analysis_match.group(1).strip() if analysis_match else ""
            return score, reasoning

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

        final_score_match = re.search(
            r"Final Score\s*:\s*(?:\[\s*)?([1-5])(?:\s*\])?",
            stripped,
        )
        if final_score_match:
            return int(final_score_match.group(1)), ""

        reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', stripped)
        reasoning = reasoning_match.group(1) if reasoning_match else ""
        score_match = re.search(r'"score"\s*:\s*\[?\s*([1-5])\s*\]?', stripped)
        if score_match:
            return int(score_match.group(1)), reasoning
        return None, reasoning

    def _vllm_evaluate(self, conversation, max_tokens=None, score_mode: str = "score"):
        if max_tokens is None:
            max_tokens = _max_tokens_default
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.chat(conversation, sampling_params=sampling_params)
        try:
            if outputs and outputs[0].outputs:
                text = outputs[0].outputs[0].text
                score, reasoning = self._extract_score_and_reasoning(
                    text, score_mode=score_mode
                )
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

    def _vllm_evaluate_batch(
        self, conversations, max_tokens=None, score_mode: str = "score"
    ):
        if max_tokens is None:
            max_tokens = _max_tokens_default
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0)
        outputs = self.llm.chat(conversations, sampling_params=sampling_params)
        results = []
        if not outputs:
            return [(0, "") for _ in range(len(conversations))]

        for output in outputs:
            try:
                if output and output.outputs:
                    text = output.outputs[0].text
                    score, reasoning = self._extract_score_and_reasoning(
                        text, score_mode=score_mode
                    )
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
    if not ray.is_initialized():
        ray.init()

    try:
        import torch

        visible_count = torch.cuda.device_count()
        logger.info(
            "cuda_visible_devices=%s detected_gpus=%d requested_gpus=%d tp=%d",
            os.getenv("CUDA_VISIBLE_DEVICES", "<not set>"),
            visible_count,
            num_gpus,
            num_tp,
        )
        if visible_count > 0 and num_gpus > visible_count:
            logger.warning(
                "Requested NUM_GPUS=%d but only %d GPUs visible. "
                "Reduce NUM_GPUS or set CUDA_VISIBLE_DEVICES to include all GPUs.",
                num_gpus,
                visible_count,
            )
            num_gpus = visible_count
    except Exception:
        logger.info(
            "cuda_visible_devices=%s requested_gpus=%d tp=%d",
            os.getenv("CUDA_VISIBLE_DEVICES", "<not set>"),
            num_gpus,
            num_tp,
        )

    workers = []
    for _ in range(num_gpus // num_tp):
        worker = ModelWorker.remote()
        workers.append(worker)

    print(f"Initialized {num_gpus//num_tp} Ray workers")
    return workers


async def evaluate_images_async(
    image_bytes_list,
    prompts,
    requirements: Optional[List[str]] = None,
    score_mode: str = "score",
):
    global workers

    if not workers:
        raise RuntimeError("Ray workers not initialized")

    if not requirements:
        requirements = [""] * len(prompts)

    batch_size = min(len(image_bytes_list), len(prompts))
    max_workers_env = os.getenv("REWARD_WORKERS_PER_REQUEST")
    if max_workers_env:
        try:
            max_workers = max(1, int(max_workers_env))
        except ValueError:
            max_workers = len(workers)
    else:
        max_workers = len(workers)
    num_active = min(len(workers), max_workers)
    if num_active <= 0:
        raise RuntimeError("No active workers available")
    global _worker_rr
    with _worker_rr_lock:
        start_idx = _worker_rr % len(workers)
        _worker_rr = (start_idx + num_active) % len(workers)
    active_workers = [workers[(start_idx + i) % len(workers)] for i in range(num_active)]
    logger.debug(
        "dispatch_batches mode=%s batch_size=%d workers=%d",
        score_mode,
        batch_size,
        len(active_workers),
    )
    per_worker_batches = [[] for _ in range(len(active_workers))]
    for i, (image_bytes, prompt, requirement) in enumerate(
        zip(
            image_bytes_list[:batch_size],
            prompts[:batch_size],
            requirements[:batch_size],
        )
    ):
        worker_idx = i % len(active_workers)
        per_worker_batches[worker_idx].append(
            (i, image_bytes, prompt, requirement)
        )

    tasks = []
    for worker_idx, batch in enumerate(per_worker_batches):
        if not batch:
            continue
        indices, images, batch_prompts, batch_requirements = zip(*batch)
        task = active_workers[worker_idx].evaluate_images.remote(
            list(images),
            list(batch_prompts),
            list(batch_requirements),
            score_mode,
        )
        tasks.append((indices, task, worker_idx, time.monotonic()))

    results = [None] * batch_size
    ray_timeout_env = os.getenv("REWARD_RAY_GET_TIMEOUT")
    ray_timeout = float(ray_timeout_env) if ray_timeout_env else None
    for indices, task, worker_idx, start_time in tasks:
        if ray_timeout is None:
            batch_results = ray.get(task)
        else:
            ready, _ = ray.wait([task], timeout=ray_timeout)
            if not ready:
                raise TimeoutError(
                    f"Ray task timeout after {ray_timeout}s (worker_idx={worker_idx})"
                )
            batch_results = ray.get(ready[0])
        logger.info(
            "worker_done worker_idx=%d items=%d elapsed_sec=%.2f",
            worker_idx,
            len(indices),
            time.monotonic() - start_time,
        )
        for idx, item in zip(indices, batch_results):
            results[idx] = item

    return results


def evaluate_images(
    image_bytes_list,
    prompts,
    requirements=None,
    score_mode: str = "score",
):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        scores = loop.run_until_complete(
            evaluate_images_async(
                image_bytes_list,
                prompts,
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
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    t0 = time.monotonic()
    payload_size = len(data) if data is not None else 0
    sem_wait_start = None
    if _inflight_sem is not None:
        sem_wait_start = time.monotonic()
        _inflight_sem.acquire()

    assert mode in ["score"], "Invalid mode"

    try:
        data = pickle.loads(data)
        image_bytes_list = data["images"]
        prompts = data["prompts"]
        metadatas = data.get("metadatas", [])
        reward_mode, quality_ratio = _resolve_reward_blend_config(data)
        compute_quality = reward_mode != "ins"
        requirements = []
        for metadata in metadatas:
            requirements.append(metadata.get("requirement", ""))

        num_images = len(image_bytes_list)
        prompt_chars = sum(len(p) for p in prompts)
        image_bytes_total = sum(len(b) for b in image_bytes_list)
        logger.info(
            "request_start id=%s mode=%s reward_mode=%s quality_ratio=%.3f images=%d payload_bytes=%d image_bytes=%d prompt_chars=%d",
            request_id,
            mode,
            reward_mode,
            quality_ratio,
            num_images,
            payload_size,
            image_bytes_total,
            prompt_chars,
        )

        if os.getenv("REWARD_DRY_RUN", "0").lower() in ("1", "true", "yes"):
            scores = [3 for _ in range(len(prompts))]
            reasonings = ["" for _ in range(len(prompts))]
            response = {"scores": scores, "reasonings": reasonings}
            response = pickle.dumps(response)
            returncode = 200
            elapsed = time.monotonic() - t0
            logger.info(
                "request_done id=%s mode=%s images=%d status=%d elapsed_sec=%.2f dry_run=1",
                request_id,
                mode,
                len(scores),
                returncode,
                elapsed,
            )
            return response, returncode

        ins_results = evaluate_images(
            image_bytes_list,
            prompts,
            requirements,
            "score",
        )
        if compute_quality:
            quality_results = evaluate_images(
                image_bytes_list,
                prompts,
                requirements,
                "quality",
            )
            if len(ins_results) != len(quality_results):
                raise RuntimeError(
                    "Mismatched score lengths from score/quality evaluation."
                )
        else:
            quality_results = [None] * len(ins_results)

        scores = []
        reasonings = []
        for ins_item, quality_item in zip(ins_results, quality_results):
            if isinstance(ins_item, (list, tuple)) and len(ins_item) >= 2:
                ins_score, ins_reasoning = ins_item[0], ins_item[1]
            else:
                ins_score, ins_reasoning = ins_item, ""

            if isinstance(quality_item, (list, tuple)) and len(quality_item) >= 2:
                quality_score, quality_reasoning = quality_item[0], quality_item[1]
            else:
                quality_score, quality_reasoning = quality_item, ""

            if quality_item is None:
                quality_score = float("nan")
                quality_reasoning_text = ""
                quality_subscores = [float("nan"), float("nan"), float("nan")]
            else:
                quality_reasoning_text, quality_subscores = _parse_quality_reasoning_payload(
                    quality_reasoning
                )
                if quality_score is None:
                    quality_score = float("nan")
                if quality_subscores is None:
                    quality_subscores = [float("nan"), float("nan"), float("nan")]

            reward, ins_norm, quality_norm = _compose_reward(
                ins_score,
                quality_score,
                quality_subscores,
                reward_mode,
                quality_ratio,
            )
            score1 = quality_subscores[0] if quality_subscores else None
            score2 = quality_subscores[1] if quality_subscores else None
            score3 = quality_subscores[2] if quality_subscores else None
            scores.append(_reward_to_server_score(reward))
            reasonings.append(
                json.dumps(
                    {
                        "ins_score": ins_score,
                        "ins_norm": ins_norm,
                        "quality_score": quality_score,
                        "quality_norm": quality_norm,
                        "quality_subscores": quality_subscores,
                        "quality_score1": score1,
                        "quality_score2": score2,
                        "quality_score3": score3,
                        "reward": reward,
                        "reward_mode": reward_mode,
                        "quality_ratio": quality_ratio,
                        "quality_evaluated": compute_quality,
                        "ins_reasoning": ins_reasoning,
                        "quality_reasoning": quality_reasoning_text,
                    },
                    ensure_ascii=True,
                )
            )
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
        elapsed = time.monotonic() - t0
        if sem_wait_start is not None:
            logger.info(
                "request_queue_wait id=%s wait_sec=%.2f",
                request_id,
                time.monotonic() - sem_wait_start,
            )
        logger.info(
            "request_done id=%s mode=%s images=%d status=%d elapsed_sec=%.2f",
            request_id,
            mode,
            len(scores),
            returncode,
            elapsed,
        )
    except KeyError as e:
        response = f"KeyError: {str(e)}"
        response = response.encode("utf-8")
        returncode = 500
        elapsed = time.monotonic() - t0
        if sem_wait_start is not None:
            logger.info(
                "request_queue_wait id=%s wait_sec=%.2f",
                request_id,
                time.monotonic() - sem_wait_start,
            )
        logger.exception(
            "request_error id=%s mode=%s status=%d elapsed_sec=%.2f error=%s",
            request_id,
            mode,
            returncode,
            elapsed,
            str(e),
        )
    except Exception as e:
        response = traceback.format_exc()
        response = response.encode("utf-8")
        returncode = 500
        elapsed = time.monotonic() - t0
        if sem_wait_start is not None:
            logger.info(
                "request_queue_wait id=%s wait_sec=%.2f",
                request_id,
                time.monotonic() - sem_wait_start,
            )
        logger.exception(
            "request_error id=%s mode=%s status=%d elapsed_sec=%.2f error=%s",
            request_id,
            mode,
            returncode,
            elapsed,
            str(e),
        )
    finally:
        if _inflight_sem is not None:
            _inflight_sem.release()

    return response, returncode


if __name__ == "__main__":
    initialize_ray_workers(NUM_GPUS, NUM_TP)
    print(f"Starting Flask server with {NUM_GPUS//NUM_TP} Ray workers...")
    http_host = os.getenv("REWARD_HTTP_HOST", "0.0.0.0")
    http_port = int(os.getenv("REWARD_HTTP_PORT", "12341"))
    http_threaded = os.getenv("REWARD_HTTP_THREADED", "1").lower() in ("1", "true", "yes")
    app.run(host=http_host, port=http_port, debug=False, threaded=http_threaded)
