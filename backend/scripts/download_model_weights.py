from __future__ import annotations

import argparse
import contextlib
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_RETRIES = 5


@dataclass(frozen=True)
class ModelWeight:
    key: str
    repo: str
    filename: str
    cache_subdir: str
    env_path: str | None = None
    env_url: str | None = None

    def target_path(self, cache_root: Path) -> Path:
        if self.env_path and os.environ.get(self.env_path):
            return Path(os.environ[self.env_path])
        return cache_root / self.cache_subdir / self.filename

    def url(self, endpoint: str) -> str:
        if self.env_url and os.environ.get(self.env_url):
            return os.environ[self.env_url]
        return f"{endpoint.rstrip('/')}/{self.repo}/resolve/main/{self.filename}"


MODEL_WEIGHTS = {
    "litevggt": ModelWeight(
        key="litevggt",
        repo="ZhijianShu/LiteVGGT",
        filename="te_dict.pt",
        cache_subdir="litevggt",
        env_url="LITEVGGT_WEIGHT_URL",
    ),
    "lingbot-map": ModelWeight(
        key="lingbot-map",
        repo="robbyant/lingbot-map",
        filename="lingbot-map-long.pt",
        cache_subdir="lingbot-map",
        env_path="LINGBOT_MODEL_PATH",
        env_url="LINGBOT_WEIGHT_URL",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights into a resumable shared cache.")
    parser.add_argument("--cache-root", default=os.environ.get("MODEL_CACHE_ROOT", "model-cache"))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_WEIGHTS),
        default=sorted(MODEL_WEIGHTS),
        help="Model keys to ensure in the cache.",
    )
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    for model_key in args.models:
        ensure_model_weight(MODEL_WEIGHTS[model_key], cache_root, str(args.hf_endpoint), retries=max(args.retries, 1))
    return 0


def ensure_model_weight(model: ModelWeight, cache_root: Path, endpoint: str, *, retries: int = DEFAULT_RETRIES) -> Path:
    target = model.target_path(cache_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    with file_lock(lock_path):
        expected_size = remote_content_length(model.url(endpoint))
        if is_complete(target, expected_size):
            print(f"{model.key} weight already cached: {target} ({target.stat().st_size} bytes)", flush=True)
            return target

        part = target.with_suffix(target.suffix + ".part")
        if target.exists() and target.stat().st_size > 0 and not is_complete(target, expected_size):
            target.replace(part)

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                download_with_resume(model.url(endpoint), part, expected_size)
                if not is_complete(part, expected_size):
                    found = part.stat().st_size if part.exists() else 0
                    raise RuntimeError(f"incomplete download for {model.key}: expected {expected_size}, got {found}")
                part.replace(target)
                print(f"Cached {model.key} weight: {target} ({target.stat().st_size} bytes)", flush=True)
                return target
            except Exception as exc:  # pragma: no cover - exercised through integration/network runs
                last_error = exc
                if attempt >= retries:
                    break
                sleep_seconds = min(2 ** attempt, 30)
                print(f"{model.key} download attempt {attempt} failed: {exc}; retrying in {sleep_seconds}s", flush=True)
                time.sleep(sleep_seconds)
        raise RuntimeError(f"failed to download {model.key} after {retries} attempts: {last_error}") from last_error


def remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None


def is_complete(path: Path, expected_size: int | None) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    return expected_size is None or path.stat().st_size == expected_size


def download_with_resume(url: str, part: Path, expected_size: int | None) -> None:
    existing_size = part.stat().st_size if part.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {url} -> {part} (resume_at={existing_size})", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", response.getcode())
        if existing_size > 0 and status != 206:
            existing_size = 0
        mode = "ab" if existing_size > 0 else "wb"
        with part.open(mode) as file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
    if expected_size is not None and part.stat().st_size > expected_size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded file is larger than expected: {part}")


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.25)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
