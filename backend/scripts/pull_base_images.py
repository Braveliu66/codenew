from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Iterable


DEFAULT_BASE_IMAGES = (
    ("API_BASE_IMAGE", "python:3.12-slim"),
    ("PREVIEW_CUDA_BASE_IMAGE", "nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04"),
    ("LINGBOT_CUDA_BASE_IMAGE", "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"),
)


def unique_nonempty(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        image = value.strip()
        if not image or image in seen:
            continue
        seen.add(image)
        result.append(image)
    return result


def configured_images() -> list[str]:
    return unique_nonempty(os.environ.get(env_name, default) for env_name, default in DEFAULT_BASE_IMAGES)


def pull_image(image: str, *, retries: int, delay_seconds: float) -> bool:
    for attempt in range(1, retries + 1):
        print(f"pulling base image {attempt}/{retries}: {image}", flush=True)
        result = subprocess.run(["docker", "pull", image], check=False)
        if result.returncode == 0:
            print(f"base image ready: {image}", flush=True)
            return True
        if attempt < retries:
            print(f"pull failed for {image}; retrying in {delay_seconds:g}s", flush=True)
            time.sleep(delay_seconds)
    print(f"failed to pull base image after {retries} attempts: {image}", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-pull Docker base images with retries before compose build.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Explicit image references. Defaults to API/PREVIEW/LINGBOT base image environment settings.",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    args = parser.parse_args()

    images = unique_nonempty(args.images) if args.images is not None else configured_images()
    if not images:
        print("no base images configured", flush=True)
        return 0

    retries = max(args.retries, 1)
    delay_seconds = max(args.delay_seconds, 0)
    failures = [image for image in images if not pull_image(image, retries=retries, delay_seconds=delay_seconds)]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
