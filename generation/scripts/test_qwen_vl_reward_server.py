#!/usr/bin/env python3
import argparse
import pickle
import socket
import sys
import time
from typing import List

import requests

try:
    from PIL import Image
except Exception as exc:
    Image = None


def _make_image_bytes(width: int, height: int) -> bytes:
    if Image is None:
        raise RuntimeError("PIL is required to build test images.")
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = bytearray()
    with Image.new("RGB", (width, height), color=(128, 128, 128)) as img:
        from io import BytesIO

        bio = BytesIO()
        img.save(bio, format="JPEG")
        buf = bio.getvalue()
    return bytes(buf)


def tcp_ping(host: str, port: int, timeout: float) -> float:
    t0 = time.monotonic()
    with socket.create_connection((host, port), timeout=timeout):
        pass
    return time.monotonic() - t0


def build_payload(
    batch_size: int,
    width: int,
    height: int,
    prompt_len: int,
) -> bytes:
    img_bytes = _make_image_bytes(width, height)
    images: List[bytes] = [img_bytes for _ in range(batch_size)]
    prompts = ["a test prompt " + ("x" * max(0, prompt_len - 13)) for _ in range(batch_size)]
    metadatas = [{"requirement": ""} for _ in range(batch_size)]
    payload = {"images": images, "prompts": prompts, "metadatas": metadatas}
    return pickle.dumps(payload)


def post_score(
    url: str,
    payload: bytes,
    connect_timeout: float,
    read_timeout: float,
):
    t0 = time.monotonic()
    resp = requests.post(
        url,
        data=payload,
        timeout=(connect_timeout, read_timeout),
    )
    elapsed = time.monotonic() - t0
    return resp, elapsed


def main():
    parser = argparse.ArgumentParser(description="Connectivity test for Qwen-VL reward server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12341)
    parser.add_argument("--mode", default="score")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--connect-timeout", type=float, default=3.0)
    parser.add_argument("--read-timeout", type=float, default=30.0)
    parser.add_argument("--tcp-only", action="store_true")
    parser.add_argument("--get-only", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/mode/{args.mode}"

    try:
        tcp_time = tcp_ping(args.host, args.port, args.connect_timeout)
        print(f"[tcp] ok in {tcp_time:.3f}s")
    except Exception as exc:
        print(f"[tcp] failed: {exc}")
        if args.tcp_only:
            return 2

    if args.tcp_only:
        return 0

    if args.get_only:
        t0 = time.monotonic()
        try:
            resp = requests.get(f"http://{args.host}:{args.port}/", timeout=args.connect_timeout)
            elapsed = time.monotonic() - t0
            print(f"[get /] status={resp.status_code} elapsed={elapsed:.3f}s")
        except Exception as exc:
            print(f"[get /] failed: {exc}")
            return 2
        return 0

    payload = build_payload(args.batch_size, args.width, args.height, args.prompt_len)
    print(f"[post] url={url} payload_bytes={len(payload)} batch={args.batch_size}")
    try:
        resp, elapsed = post_score(
            url,
            payload,
            args.connect_timeout,
            args.read_timeout,
        )
        print(f"[post] status={resp.status_code} elapsed={elapsed:.3f}s body_bytes={len(resp.content)}")
    except Exception as exc:
        print(f"[post] failed: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
