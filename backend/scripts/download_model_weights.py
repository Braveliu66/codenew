from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path


LITEVGGT_WEIGHT_REPO = "ZhijianShu/LiteVGGT"
LITEVGGT_WEIGHT_FILE = "te_dict.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights into the local project cache.")
    parser.add_argument("--cache-root", default="model-cache")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = parser.parse_args()

    endpoint = str(args.hf_endpoint).rstrip("/")
    target = Path(args.cache_root) / "litevggt" / LITEVGGT_WEIGHT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"LiteVGGT weight already cached: {target} ({target.stat().st_size} bytes)")
        return 0

    url = f"{endpoint}/{LITEVGGT_WEIGHT_REPO}/resolve/main/{LITEVGGT_WEIGHT_FILE}"
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as file:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded an empty weight file: {tmp}")
    tmp.replace(target)
    print(f"Cached LiteVGGT weight: {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

