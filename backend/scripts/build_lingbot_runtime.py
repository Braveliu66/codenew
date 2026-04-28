from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


LINGBOT_REPO = "https://github.com/Robbyant/lingbot-map.git"
SPARK_REPO = "https://github.com/sparkjsdev/spark.git"
LINGBOT_LICENSE = "Apache-2.0"
LINGBOT_WEIGHT_FILE = "lingbot-map-long.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build LingBot-Map video preview runtime.")
    parser.add_argument("--runtime-root", default="/opt/three-dgs-lingbot")
    parser.add_argument("--registry-output", default="/opt/three-dgs-lingbot/runtime/algorithm_registry.generated.json")
    parser.add_argument("--weight-cache-root", default="/workspace/model-cache")
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    repos_root = runtime_root / "repos"
    registry_output = Path(args.registry_output)
    weight_cache_root = Path(args.weight_cache_root)
    workspace = Path(args.workspace)
    repos_root.mkdir(parents=True, exist_ok=True)
    registry_output.parent.mkdir(parents=True, exist_ok=True)

    configure_git()
    lingbot = clone_checkout(LINGBOT_REPO, repos_root / "lingbot-map", os.environ.get("LINGBOT_COMMIT", "main"))
    spark = clone_checkout(SPARK_REPO, repos_root / "spark", os.environ.get("SPARK_COMMIT", "main"))
    weight_path = resolve_lingbot_weight(weight_cache_root)
    if not args.skip_install:
        install_lingbot_runtime(lingbot)
        install_spark_runtime(spark)
    write_registry(
        registry_output=registry_output,
        workspace=workspace,
        lingbot=lingbot,
        spark=spark,
        lingbot_weight=weight_path,
    )
    return 0


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=str(cwd) if cwd else None, check=check, text=True)


def configure_git() -> None:
    run(["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"], check=False)


def clone_checkout(url: str, target: Path, ref: str) -> Path:
    if not target.exists():
        run(["git", "clone", url, str(target)])
    run(["git", "-C", str(target), "fetch", "origin", ref, "--depth", "1"], check=False)
    run(["git", "-C", str(target), "checkout", ref])
    return target.resolve()


def resolve_lingbot_weight(weight_cache_root: Path) -> Path:
    configured = os.environ.get("LINGBOT_MODEL_PATH")
    candidates = [Path(configured)] if configured else []
    candidates.append(weight_cache_root / "lingbot-map" / LINGBOT_WEIGHT_FILE)
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise RuntimeError(
        "LingBot-Map weight is missing. Place it at "
        f"{weight_cache_root / 'lingbot-map' / LINGBOT_WEIGHT_FILE} before building."
    )


def install_lingbot_runtime(lingbot: Path) -> None:
    pip_install(["--upgrade", "pip", "setuptools", "wheel"])
    pip_install(["opencv-python-headless", "numpy==1.26.4", "einops", "safetensors", "huggingface_hub"])
    pip_install(["--no-build-isolation", "-e", str(lingbot)])
    pip_install(["numpy==1.26.4"])


def install_spark_runtime(spark: Path) -> None:
    registry = os.environ.get("NPM_CONFIG_REGISTRY", "https://registry.npmmirror.com")
    run(["npm", "config", "set", "registry", registry])
    run(["npm", "ci"], cwd=spark)
    run(["npm", "run", "build"], cwd=spark)


def pip_install(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def commit_hash(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_registry(
    *,
    registry_output: Path,
    workspace: Path,
    lingbot: Path,
    spark: Path,
    lingbot_weight: Path,
) -> None:
    script_dir = workspace / "backend" / "scripts"
    algorithms = [
        {
            "name": "LingBot-Map",
            "repo_url": "https://github.com/Robbyant/lingbot-map",
            "license": LINGBOT_LICENSE,
            "commit_hash": commit_hash(lingbot),
            "weight_source": "local model-cache/lingbot-map/lingbot-map-long.pt",
            "local_path": str(lingbot),
            "enabled": True,
            "notes": "Docker-built video preview stage. Uses local model-cache weight only.",
            "weight_paths": [str(lingbot_weight)],
            "commands": {"run_preview": ["python3", str(script_dir / "run_lingbot_map_preview.py")]},
        },
        {
            "name": "Spark-SPZ",
            "repo_url": "https://github.com/sparkjsdev/spark",
            "license": "MIT",
            "commit_hash": commit_hash(spark),
            "weight_source": None,
            "local_path": str(spark),
            "enabled": True,
            "notes": "Docker-built local Spark/SPZ conversion stage.",
            "weight_paths": [],
            "commands": {"compress": ["python3", str(script_dir / "run_spz_convert.py")]},
        },
    ]
    registry_output.write_text(json.dumps({"algorithms": algorithms}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
