from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


LITEVGGT_REPO = "https://github.com/GarlicBa/LiteVGGT-repo.git"
EDGS_REPO = "https://github.com/CompVis/EDGS.git"
SPARK_REPO = "https://github.com/sparkjsdev/spark.git"
LITEVGGT_WEIGHT_REPO = "ZhijianShu/LiteVGGT"
LITEVGGT_WEIGHT_FILE = "te_dict.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap real preview algorithm runtime.")
    parser.add_argument("--runtime-root", default=".runtime")
    parser.add_argument("--registry", default="backend/config/algorithm_registry.example.json")
    parser.add_argument("--install-deps", action="store_true")
    args = parser.parse_args()

    workspace = Path.cwd()
    runtime_root = (workspace / args.runtime_root).resolve()
    repos_root = runtime_root / "repos"
    models_root = runtime_root / "models"
    repos_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    litevggt = clone_repo(LITEVGGT_REPO, repos_root / "LiteVGGT-repo")
    edgs = clone_repo(EDGS_REPO, repos_root / "EDGS")
    spark = clone_repo(SPARK_REPO, repos_root / "spark")
    weight_path = models_root / "litevggt" / LITEVGGT_WEIGHT_FILE
    download_file(litevggt_weight_url(), weight_path)

    if args.install_deps:
        pip_install(litevggt / "requirements.txt")
        pip_install(edgs / "requirements.txt")
        if (spark / "package.json").exists():
            subprocess.run(["npm", "install"], cwd=str(spark), check=True)

    registry_path = (workspace / args.registry).resolve()
    update_registry(
        registry_path=registry_path,
        workspace=workspace,
        paths={
            "LiteVGGT": litevggt,
            "EDGS": edgs,
            "Spark-SPZ": spark,
        },
        litevggt_weight=weight_path,
    )
    return 0


def litevggt_weight_url() -> str:
    configured = os.environ.get("LITEVGGT_WEIGHT_URL")
    if configured:
        return configured
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").strip().rstrip("/")
    return f"{endpoint}/{LITEVGGT_WEIGHT_REPO}/resolve/main/{LITEVGGT_WEIGHT_FILE}"


def clone_repo(url: str, target: Path) -> Path:
    if target.exists():
        subprocess.run(["git", "-C", str(target), "fetch", "--all", "--tags"], check=False)
    else:
        subprocess.run(["git", "clone", url, str(target)], check=True)
    return target.resolve()


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response:
        target.write_bytes(response.read())
    if target.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded an empty file: {target}")


def pip_install(requirements: Path) -> None:
    if requirements.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements)], check=True)


def commit_hash(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def update_registry(
    *,
    registry_path: Path,
    workspace: Path,
    paths: dict[str, Path],
    litevggt_weight: Path,
) -> None:
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    script_dir = workspace / "backend" / "scripts"
    command_scripts = {
        "LiteVGGT": ("run_demo", script_dir / "run_litevggt_preview.py"),
        "EDGS": ("train", script_dir / "run_edgs_preview.py"),
        "Spark-SPZ": ("compress", script_dir / "run_spz_convert.py"),
        "FFmpeg": ("extract_frames", script_dir / "run_ffmpeg_extract.py"),
    }
    for entry in data["algorithms"]:
        name = entry["name"]
        if name in paths:
            entry["enabled"] = True
            entry["local_path"] = str(paths[name])
            entry["commit_hash"] = commit_hash(paths[name])
        if name == "LiteVGGT":
            entry["weight_paths"] = [str(litevggt_weight)]
        if name == "FFmpeg":
            entry["enabled"] = bool(shutil.which("ffmpeg"))
            entry["commit_hash"] = "system-package"
            entry["source_type"] = "command"
        if name in command_scripts:
            key, script = command_scripts[name]
            entry["commands"] = {key: [sys.executable, str(script.resolve())]}
    registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

