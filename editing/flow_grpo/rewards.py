from PIL import Image
import io
import os
import numpy as np
import torch
from collections import defaultdict
import random


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


def _normalize_mllm_scores(raw_scores, normalize_scores, normalize_mode, normalize_range):
    if not normalize_scores:
        return [float(score) for score in raw_scores]
    mode = normalize_mode or "div5"
    if mode == "div5":
        return [float(score) / 5.0 for score in raw_scores]
    if mode in (
        "range_1_5",
        "range_1_5_range",
        "range_1_5_0.2_1",
        # Backward-compatible aliases from the previous attempt.
        "minmax",
        "minmax_range",
        "minmax_0.2_1",
    ):
        if mode in ("range_1_5_0.2_1", "minmax_0.2_1"):
            low, high = 0.2, 1.0
        else:
            low, high = normalize_range
        # Fixed global normalization: r' = (r - 1) / (5 - 1)
        return [
            float(low) + (float(score) - 1.0) * ((high - low) / 4.0)
            for score in raw_scores
        ]
    # Fallback to div5 to preserve prior behavior if an unknown mode is provided.
    return [float(score) / 5.0 for score in raw_scores]


def _mllm_score(
    mode,
    normalize_scores=True,
    normalize_mode="div5",
    normalize_range=(0.0, 1.0),
):
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = f"http://{os.getenv('REWARD_SERVER', 'localhost:12341')}/mode/{mode}"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(ref_images, images, prompts, metadatas):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        elif not isinstance(images, np.ndarray):
            images = np.array([np.array(img) for img in images])

        if isinstance(ref_images, torch.Tensor):
            ref_images = (
                (ref_images * 255)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            ref_images = ref_images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        elif not isinstance(ref_images, np.ndarray):
            ref_images = np.array([np.array(img) for img in ref_images])

        if not isinstance(prompts, list):
            prompts = list(prompts)
        if metadatas is None:
            metadatas = [{} for _ in range(len(prompts))]
        elif not isinstance(metadatas, list):
            metadatas = list(metadatas)

        all_scores_raw = []
        all_reasonings = []
        for start in range(0, len(images), batch_size):
            end = start + batch_size
            image_batch = images[start:end]
            ref_image_batch = ref_images[start:end]
            prompt_batch = prompts[start:end]
            metadata_batch = metadatas[start:end]

            jpeg_images = []
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            ref_jpeg_images = []
            for ref_image in ref_image_batch:
                img = Image.fromarray(ref_image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                ref_jpeg_images.append(buffer.getvalue())

            data = {
                "ref_images": ref_jpeg_images,
                "images": jpeg_images,
                "prompts": prompt_batch,
                "metadatas": metadata_batch,
            }
            data_bytes = pickle.dumps(data)

            response = sess.post(url, data=data_bytes, timeout=360)
            response_data = pickle.loads(response.content)
            scores = response_data["scores"]
            reasonings = response_data.get("reasonings", [""] * len(scores))
            all_scores_raw += scores
            all_reasonings += reasonings

        normalized_scores = _normalize_mllm_scores(
            all_scores_raw,
            normalize_scores=normalize_scores,
            normalize_mode=normalize_mode,
            normalize_range=normalize_range,
        )
        return normalized_scores, {
            "reasonings": all_reasonings,
            "raw_scores": all_scores_raw,
        }

    return _fn


def mllm_score_continue(
    device,
    normalize_mllm_score=True,
    normalize_mllm_score_mode="div5",
    normalize_mllm_score_range=(0.0, 1.0),
):
    return _mllm_score(
        "score_consistency",
        normalize_scores=normalize_mllm_score,
        normalize_mode=normalize_mllm_score_mode,
        normalize_range=normalize_mllm_score_range,
    )


def mllm_score_consistency(
    device,
    normalize_mllm_score=True,
    normalize_mllm_score_mode="div5",
    normalize_mllm_score_range=(0.0, 1.0),
):
    return _mllm_score(
        "score_consistency",
        normalize_scores=normalize_mllm_score,
        normalize_mode=normalize_mllm_score_mode,
        normalize_range=normalize_mllm_score_range,
    )


def mllm_score_execution(
    device,
    normalize_mllm_score=True,
    normalize_mllm_score_mode="div5",
    normalize_mllm_score_range=(0.0, 1.0),
):
    return _mllm_score(
        "score_execution",
        normalize_scores=normalize_mllm_score,
        normalize_mode=normalize_mllm_score_mode,
        normalize_range=normalize_mllm_score_range,
    )

def mllm_score_non_logits(
    device,
    normalize_mllm_score=True,
    normalize_mllm_score_mode="div5",
    normalize_mllm_score_range=(0.0, 1.0),
):
    return _mllm_score(
        "logits_non_cot",
        normalize_scores=normalize_mllm_score,
        normalize_mode=normalize_mllm_score_mode,
        normalize_range=normalize_mllm_score_range,
    )

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
        return score, {"raw_outputs": text_outputs}

    return _fn

def dummy():
    def _fn(images, prompts, metadata):
        return [random.random() for _ in range(len(images))], {}
    return _fn
    

def multi_score(
    device,
    score_dict,
    normalize_mllm_score=True,
    normalize_mllm_score_mode="div5",
    normalize_mllm_score_range=(0.0, 1.0),
    exec_consistency_guard=False,
    exec_consistency_exec_lt=3.0,
    exec_consistency_cons_ge=4.0,
    use_total_score=False,
    total_score_w1=1.0,
    total_score_w2=1.0,
    total_score_mode=None,
):
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
        "mllm_score_continue": mllm_score_continue,
        "mllm_score_consistency": mllm_score_consistency,
        "mllm_score_execution": mllm_score_execution,
        "mllm_score_non_logits": mllm_score_non_logits,
        "dummy": dummy
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        if score_name.startswith("mllm_"):
            score_fns[score_name] = score_functions[score_name](
                device,
                normalize_mllm_score=normalize_mllm_score,
                normalize_mllm_score_mode=normalize_mllm_score_mode,
                normalize_mllm_score_range=normalize_mllm_score_range,
            )
        else:
            score_fns[score_name] = (
                score_functions[score_name](device)
                if "device" in score_functions[score_name].__code__.co_varnames
                else score_functions[score_name]()
            )

    extra_score_fns = {}
    if use_total_score:
        for score_name in ("mllm_score_execution", "mllm_score_consistency"):
            if score_name not in score_fns:
                extra_score_fns[score_name] = score_functions[score_name](
                    device,
                    normalize_mllm_score=normalize_mllm_score,
                    normalize_mllm_score_mode=normalize_mllm_score_mode,
                    normalize_mllm_score_range=normalize_mllm_score_range,
                )

    # only_strict is only for geneval. During training, only the strict reward is needed, and non-strict rewards don't need to be computed, reducing reward calculation time.
    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        total_scores = []
        score_details = {}
        score_metadata = {}

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
            elif score_name.startswith("mllm_"):
                scores, rewards = score_fns[score_name](ref_images, images, prompts, metadata)
                if rewards:
                    score_metadata[score_name] = rewards
            else:
                scores, rewards = score_fns[score_name](images, prompts, metadata)
                if rewards:
                    score_metadata[score_name] = rewards
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        if (
            exec_consistency_guard
            and not use_total_score
            and "mllm_score_execution" in score_details
            and "mllm_score_consistency" in score_details
        ):
            exec_weight = score_dict.get("mllm_score_execution", 1.0)
            exec_scores = score_details["mllm_score_execution"]
            cons_scores = score_details["mllm_score_consistency"]
            if normalize_mllm_score:
                if normalize_mllm_score_mode == "div5":
                    exec_raw = [score * 5.0 for score in exec_scores]
                    cons_raw = [score * 5.0 for score in cons_scores]
                else:
                    exec_meta = score_metadata.get("mllm_score_execution", {})
                    cons_meta = score_metadata.get("mllm_score_consistency", {})
                    exec_raw = exec_meta.get("raw_scores")
                    cons_raw = cons_meta.get("raw_scores")
                    if exec_raw is None:
                        exec_raw = [score * 4.0 + 1.0 for score in exec_scores]
                    if cons_raw is None:
                        cons_raw = [score * 4.0 + 1.0 for score in cons_scores]
                exec_norm = exec_scores
            else:
                exec_raw = exec_scores
                cons_raw = cons_scores
                exec_norm = [score / 5.0 for score in exec_scores]
            for idx, (exec_score, cons_score) in enumerate(zip(exec_raw, cons_raw)):
                if exec_score <= exec_consistency_exec_lt:
                    total_scores[idx] = exec_weight * exec_norm[idx]

        if use_total_score:
            if "mllm_score_execution" not in score_details:
                scores, rewards = extra_score_fns["mllm_score_execution"](
                    ref_images, images, prompts, metadata
                )
                score_details["mllm_score_execution"] = scores
                if rewards:
                    score_metadata["mllm_score_execution"] = rewards
            if "mllm_score_consistency" not in score_details:
                scores, rewards = extra_score_fns["mllm_score_consistency"](
                    ref_images, images, prompts, metadata
                )
                score_details["mllm_score_consistency"] = scores
                if rewards:
                    score_metadata["mllm_score_consistency"] = rewards

            exec_scores = score_details["mllm_score_execution"]
            cons_scores = score_details["mllm_score_consistency"]
            if normalize_mllm_score:
                exec_norm = [float(score) for score in exec_scores]
                cons_norm = [float(score) for score in cons_scores]
            else:
                exec_norm = [float(score) / 5.0 for score in exec_scores]
                cons_norm = [float(score) / 5.0 for score in cons_scores]

            w1 = float(total_score_w1)
            w2 = float(total_score_w2)
            denom = w1 + w2
            mode = (total_score_mode or "exec_consistency").lower()

            total_scores = []
            for exec_score, cons_score in zip(exec_norm, cons_norm):
                if mode in ("execution_only", "exec_only", "exec_only_sum"):
                    value = exec_score * (w1 + w2)
                else:
                    if denom <= 0:
                        value = 0.0
                    else:
                        value = exec_score * (w1 + w2 * cons_score) / denom
                if value < 0.0:
                    value = 0.0
                elif value > 1.0:
                    value = 1.0
                total_scores.append(value)
            score_details["total_score"] = total_scores

        score_details["avg"] = total_scores
        return score_details, score_metadata

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
