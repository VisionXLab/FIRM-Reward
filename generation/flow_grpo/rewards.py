from PIL import Image
import io
import json
import re
import random
import threading
import numpy as np
import torch
from collections import defaultdict


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device):
    from flow_grpo.gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn


def qwen_vl_reward_score(device, score_mode="score"):
    import os
    import pickle
    from io import BytesIO
    import requests
    from requests.adapters import HTTPAdapter, Retry

    def _parse_reward_urls():
        urls = []
        raw_urls = os.getenv("QWEN_VL_REWARD_URLS", "").strip()
        if raw_urls:
            urls.extend(re.split(r"[,\s]+", raw_urls))
        fallback_url = os.getenv("QWEN_VL_REWARD_URL", "http://127.0.0.1:12341").strip()
        if fallback_url:
            urls.extend(re.split(r"[,\s]+", fallback_url))

        deduped = []
        seen = set()
        for url in urls:
            normalized = str(url).strip()
            if not normalized:
                continue
            if "://" not in normalized:
                normalized = f"http://{normalized}"
            normalized = normalized.rstrip("/")
            if normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)
        if not deduped:
            deduped = ["http://127.0.0.1:12341"]
        return deduped

    base_urls = _parse_reward_urls()
    routing_mode = os.getenv(
        "QWEN_VL_REWARD_ROUTING",
        "random" if len(base_urls) > 1 else "single",
    ).strip().lower()
    timeout = float(os.getenv("QWEN_VL_REWARD_TIMEOUT", "1800"))
    batch_size = int(os.getenv("QWEN_VL_REWARD_BATCH", "16"))
    reward_blend_mode = os.getenv("QWEN_VL_REWARD_BLEND_MODE", "legacy").strip().lower()
    quality_ratio_env = os.getenv("QWEN_VL_REWARD_QUALITY_RATIO", "0.5")
    try:
        quality_ratio = float(quality_ratio_env)
    except ValueError:
        quality_ratio = 0.5
    quality_ratio = max(0.0, min(1.0, quality_ratio))
    total_retries = int(os.getenv("QWEN_VL_REWARD_RETRIES", "10"))
    backoff = float(os.getenv("QWEN_VL_REWARD_BACKOFF", "2"))
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=False,
    )
    pool_connections_env = os.getenv("QWEN_VL_REWARD_POOL_CONNECTIONS", os.getenv("QWEN_VL_REWARD_POOL", "64"))
    pool_maxsize_env = os.getenv("QWEN_VL_REWARD_POOL_MAXSIZE", os.getenv("QWEN_VL_REWARD_POOL", "64"))
    try:
        pool_connections = max(1, int(pool_connections_env))
    except ValueError:
        pool_connections = 64
    try:
        pool_maxsize = max(1, int(pool_maxsize_env))
    except ValueError:
        pool_maxsize = 64
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool_connections, pool_maxsize=pool_maxsize)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    endpoint_lock = threading.Lock()
    endpoint_inflight = [0] * len(base_urls)
    endpoint_rr_cursor = 0

    def _pick_endpoint_index():
        nonlocal endpoint_rr_cursor
        if len(base_urls) == 1:
            return 0
        if routing_mode in ("round_robin", "rr"):
            with endpoint_lock:
                idx = endpoint_rr_cursor % len(base_urls)
                endpoint_rr_cursor = (endpoint_rr_cursor + 1) % len(base_urls)
            return idx
        if routing_mode in ("least_loaded", "p2c", "power_of_two"):
            if len(base_urls) == 2:
                candidates = [0, 1]
            else:
                candidates = random.sample(range(len(base_urls)), k=2)
            with endpoint_lock:
                return (
                    candidates[0]
                    if endpoint_inflight[candidates[0]] <= endpoint_inflight[candidates[1]]
                    else candidates[1]
                )
        return random.randrange(len(base_urls))

    def _post_reward(mode, request_payload):
        encoded_payload = pickle.dumps(request_payload)
        first_idx = _pick_endpoint_index()
        endpoint_order = [first_idx] + [idx for idx in range(len(base_urls)) if idx != first_idx]
        if len(endpoint_order) > 1:
            remaining = endpoint_order[1:]
            random.shuffle(remaining)
            endpoint_order = [endpoint_order[0]] + remaining

        errors = []
        for idx in endpoint_order:
            endpoint = base_urls[idx]
            with endpoint_lock:
                endpoint_inflight[idx] += 1
            try:
                response = session.post(
                    f"{endpoint}/mode/{mode}",
                    data=encoded_payload,
                    timeout=timeout,
                )
                if response.status_code == 200:
                    return response
                errors.append(
                    f"{endpoint} status={response.status_code} body={response.content[:120]!r}"
                )
            except Exception as exc:
                errors.append(f"{endpoint} exception={exc}")
            finally:
                with endpoint_lock:
                    endpoint_inflight[idx] -= 1
        raise RuntimeError(
            f"Qwen-VL reward server request failed on all endpoints ({mode}): " + "; ".join(errors)
        )

    def _image_to_jpeg_bytes(image):
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def _normalize_metadatas(metadata, count):
        if metadata is None:
            return [{} for _ in range(count)]
        if isinstance(metadata, dict):
            return [metadata for _ in range(count)]
        return list(metadata)

    def _parse_qwen_reasoning_payload(reasoning_item):
        payload = None
        if isinstance(reasoning_item, dict):
            payload = reasoning_item
        elif isinstance(reasoning_item, str):
            stripped = reasoning_item.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    payload = None
        return payload if isinstance(payload, dict) else {}

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _clamp01(value):
        return max(0.0, min(1.0, value))

    def _blend_reward(server_norm, ins_norm, quality_norm):
        if reward_blend_mode in ("ins", "instruction", "instruction_only"):
            if ins_norm is None or not np.isfinite(ins_norm):
                return server_norm
            return _clamp01(ins_norm)

        if reward_blend_mode in ("mix", "weighted_mix", "linear_mix"):
            if ins_norm is None or not np.isfinite(ins_norm):
                return server_norm
            if quality_norm is None or not np.isfinite(quality_norm):
                # If quality is missing, fall back to instruction-only behavior.
                return _clamp01(ins_norm)
            return _clamp01(ins_norm * (0.4 + 0.6 * quality_norm))

        # legacy/server: keep reward server output unchanged.
        return server_norm

    def _parse_triplet(values):
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            return None
        parsed = []
        for item in values[:3]:
            value = _to_float(item)
            if value is None:
                return None
            parsed.append(value)
        return parsed

    def _parse_quality_subscores(payload):
        if not isinstance(payload, dict):
            return None
        candidates = [
            payload.get("quality_subscores"),
            payload.get("subscores"),
            payload.get("score"),
        ]
        if any(k in payload for k in ("quality_score1", "quality_score2", "quality_score3")):
            candidates.append(
                [
                    payload.get("quality_score1"),
                    payload.get("quality_score2"),
                    payload.get("quality_score3"),
                ]
            )
        for values in candidates:
            parsed = _parse_triplet(values)
            if parsed is not None:
                return parsed
        return None

    def _parse_quality_subscores_from_text(reasoning_item):
        if not isinstance(reasoning_item, str):
            return None
        match = re.search(
            r"\[\s*([1-5](?:\.\d+)?)\s*,\s*([1-5](?:\.\d+)?)\s*,\s*([1-5](?:\.\d+)?)\s*\]",
            reasoning_item,
        )
        if not match:
            return None
        return [float(match.group(i)) for i in range(1, 4)]

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        elif isinstance(images, np.ndarray):
            if images.ndim == 4:
                images = [Image.fromarray(image.astype(np.uint8)) for image in images]
            else:
                images = [Image.fromarray(images.astype(np.uint8))]
        else:
            images = list(images)

        metadatas = _normalize_metadatas(metadata, len(prompts))
        if len(metadatas) != len(prompts):
            raise ValueError("Metadata length must match prompts length for Qwen-VL reward.")

        all_scores = []
        all_reasonings = []
        all_instruction_scores = []
        all_instruction_norms = []
        all_quality_scores = []
        all_quality_score1 = []
        all_quality_score2 = []
        all_quality_score3 = []
        quality_mode_supported = None
        for start in range(0, len(images), batch_size):
            end = start + batch_size
            image_batch = images[start:end]
            prompt_batch = list(prompts)[start:end]
            metadata_batch = metadatas[start:end]

            image_bytes_list = [_image_to_jpeg_bytes(image) for image in image_batch]
            request_payload = {
                "images": image_bytes_list,
                "prompts": prompt_batch,
                "metadatas": metadata_batch,
                "reward_blend_mode": reward_blend_mode,
                "reward_quality_ratio": quality_ratio,
            }
            response = _post_reward(score_mode, request_payload)
            result = pickle.loads(response.content)
            scores = result.get("scores", [])
            reasonings = result.get("reasonings", [""] * len(scores))
            server_norm_scores = [(float(score) - 1.0) / 4.0 for score in scores]
            if len(reasonings) < len(scores):
                reasonings = list(reasonings) + [""] * (len(scores) - len(reasonings))
            all_reasonings += reasonings[: len(scores)]
            batch_quality_scores = [None] * len(scores)
            batch_quality_subscores = [None] * len(scores)
            batch_instruction_norms = [None] * len(scores)
            for idx, reasoning_item in enumerate(reasonings[: len(scores)]):
                reasoning_payload = _parse_qwen_reasoning_payload(reasoning_item)
                ins_score = _to_float(reasoning_payload.get("ins_score"))
                if ins_score is None:
                    ins_score = _to_float(scores[idx])
                if ins_score is None:
                    ins_score = float("nan")
                all_instruction_scores.append(ins_score)

                ins_norm = _to_float(reasoning_payload.get("ins_norm"))
                if ins_norm is None:
                    ins_norm = (ins_score - 1.0) / 4.0 if np.isfinite(ins_score) else float("nan")
                all_instruction_norms.append(ins_norm)
                batch_instruction_norms[idx] = ins_norm

                quality_subscores = _parse_quality_subscores(reasoning_payload)
                if quality_subscores is None:
                    quality_subscores = _parse_quality_subscores_from_text(reasoning_item)
                quality_score = _to_float(reasoning_payload.get("quality_score"))
                if quality_score is None and quality_subscores is not None:
                    quality_score = float(sum(quality_subscores))
                batch_quality_scores[idx] = quality_score
                batch_quality_subscores[idx] = quality_subscores

            need_quality_fallback = (
                score_mode == "score" and any(item is None for item in batch_quality_subscores)
            )
            if need_quality_fallback and quality_mode_supported is not False:
                try:
                    quality_response = _post_reward("quality", request_payload)
                    quality_result = pickle.loads(quality_response.content)
                    q_scores = quality_result.get("scores", [])
                    q_reasonings = quality_result.get("reasonings", [""] * len(q_scores))
                    if len(q_reasonings) < len(q_scores):
                        q_reasonings = list(q_reasonings) + [""] * (len(q_scores) - len(q_reasonings))
                    if len(q_scores) >= len(scores):
                        quality_mode_supported = True
                        for idx in range(len(scores)):
                            if batch_quality_subscores[idx] is not None:
                                continue
                            q_payload = _parse_qwen_reasoning_payload(q_reasonings[idx])
                            q_subscores = _parse_quality_subscores(q_payload)
                            if q_subscores is None:
                                q_subscores = _parse_quality_subscores_from_text(q_reasonings[idx])
                            q_score = _to_float(q_payload.get("quality_score"))
                            if q_score is None:
                                q_score = _to_float(q_scores[idx])
                            if q_score is None and q_subscores is not None:
                                q_score = float(sum(q_subscores))
                            batch_quality_scores[idx] = q_score
                            batch_quality_subscores[idx] = q_subscores
                    else:
                        quality_mode_supported = False
                except Exception:
                    quality_mode_supported = False

            for idx in range(len(scores)):
                quality_score = batch_quality_scores[idx]
                quality_subscores = batch_quality_subscores[idx]
                quality_norm = None
                if quality_subscores is not None:
                    valid_subscores = [sub for sub in quality_subscores[:3] if np.isfinite(sub)]
                    if len(valid_subscores) == 3:
                        quality_norm = (min(valid_subscores) - 1.0) / 4.0
                if quality_score is None:
                    quality_score = float("nan")
                if quality_subscores is None:
                    quality_subscores = [float("nan"), float("nan"), float("nan")]
                ins_norm = batch_instruction_norms[idx]
                if ins_norm is not None and not np.isfinite(ins_norm):
                    ins_norm = None
                all_scores.append(
                    _blend_reward(
                        server_norm_scores[idx],
                        ins_norm,
                        quality_norm,
                    )
                )
                all_quality_scores.append(quality_score)
                all_quality_score1.append(quality_subscores[0])
                all_quality_score2.append(quality_subscores[1])
                all_quality_score3.append(quality_subscores[2])

        return all_scores, {
            "reasonings": all_reasonings,
            "instruction_score": all_instruction_scores,
            "instruction_norm": all_instruction_norms,
            "quality_score": all_quality_scores,
            "quality_score1": all_quality_score1,
            "quality_score2": all_quality_score2,
            "quality_score3": all_quality_score3,
            "quality_aesthetics": all_quality_score1,
            "quality_quality": all_quality_score2,
            "quality_structure": all_quality_score3,
        }

    return _fn


def qwen_vl_score(device):
    return qwen_vl_reward_score(device, "score")


def reward_model_server_score(device):
    # Alias used by FLUX.2-Klein DiffusionNFT config/scripts.
    return qwen_vl_reward_score(device, "score")


def multi_score(device, score_dict):
    score_functions = {
        "ocr": ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
        "qwen_vl": qwen_vl_score,
        "reward_model_server": reward_model_server_score,
        "qwen_vl_reward_server": reward_model_server_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = (
            score_functions[score_name](device)
            if "device" in score_functions[score_name].__code__.co_varnames
            else score_functions[score_name]()
        )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            if isinstance(rewards, dict):
                for metric_name, metric_values in rewards.items():
                    if metric_name == "reasonings":
                        continue
                    detail_key = f"{score_name}_{metric_name}"
                    score_details[detail_key] = metric_values
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "test_cases/nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"unifiedreward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
