from __future__ import annotations

import argparse
import os
from pathlib import Path

from backend.scripts.download_model_weights import MODEL_WEIGHTS, ensure_model_weight


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker startup preflight for required model weights.")
    parser.add_argument("--cache-root", default=os.environ.get("MODEL_CACHE_ROOT", "/model-cache"))
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_WEIGHTS), required=True)
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    for model_key in args.models:
        path = ensure_model_weight(MODEL_WEIGHTS[model_key], cache_root, str(args.hf_endpoint))
        print(f"model weight ready: {model_key} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
