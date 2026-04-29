from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path


LITEVGGT_REPO = "https://github.com/GarlicBa/LiteVGGT-repo.git"
EDGS_REPO = "https://github.com/CompVis/EDGS.git"
SPARK_REPO = "https://github.com/sparkjsdev/spark.git"
LITEVGGT_WEIGHT_REPO = "ZhijianShu/LiteVGGT"
LITEVGGT_WEIGHT_FILE = "te_dict.pt"
EDGS_LICENSE = "Non-commercial research and personal use (see EDGS LICENSE.txt)"
EDGS_WHEEL_BASE_URL = "https://huggingface.co/spaces/CompVis/EDGS/resolve/main/wheels"
DEFAULT_TRANSFORMER_ENGINE_VERSION = "2.14.0"
EDGS_WHEELS = {
    "diff_gaussian_rasterization": "diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
    "simple_knn": "simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build preview algorithm runtime for Docker images.")
    parser.add_argument("--runtime-root", default="/opt/three-dgs")
    parser.add_argument("--registry-output", default="/opt/three-dgs/runtime/algorithm_registry.generated.json")
    parser.add_argument("--weight-cache-root", default="/workspace/model-cache")
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    repos_root = runtime_root / "repos"
    models_root = runtime_root / "models"
    weight_cache_root = Path(args.weight_cache_root)
    registry_output = Path(args.registry_output)
    workspace = Path(args.workspace)
    repos_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    registry_output.parent.mkdir(parents=True, exist_ok=True)

    configure_git()
    litevggt = clone_checkout(LITEVGGT_REPO, repos_root / "LiteVGGT-repo", env_commit("LITEVGGT_COMMIT"))
    edgs = clone_checkout(EDGS_REPO, repos_root / "EDGS", env_commit("EDGS_COMMIT"), recursive=True)
    spark = clone_checkout(SPARK_REPO, repos_root / "spark", env_commit("SPARK_COMMIT"))
    weight_path = resolve_litevggt_weight(models_root, weight_cache_root)

    if not args.skip_install:
        install_python_runtime(litevggt, edgs)
        install_spark_runtime(spark)

    write_registry(
        registry_output=registry_output,
        workspace=workspace,
        litevggt=litevggt,
        edgs=edgs,
        spark=spark,
        litevggt_weight=weight_path,
    )
    return 0


def env_commit(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=str(cwd) if cwd else None, check=check, text=True)


def configure_git() -> None:
    run(["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"], check=False)


def clone_checkout(url: str, target: Path, commit: str, *, recursive: bool = False) -> Path:
    if not target.exists():
        command = ["git", "clone"]
        if recursive:
            command.append("--recursive")
        command.extend([url, str(target)])
        run(command)
    run(["git", "-C", str(target), "fetch", "origin", commit, "--depth", "1"], check=False)
    run(["git", "-C", str(target), "checkout", commit])
    if recursive:
        run(["git", "-C", str(target), "submodule", "sync", "--recursive"], check=False)
        run(["git", "-C", str(target), "submodule", "update", "--init", "--recursive"])
    return target.resolve()


def resolve_litevggt_weight(models_root: Path, weight_cache_root: Path) -> Path:
    runtime_weight = Path("/model-cache/litevggt") / LITEVGGT_WEIGHT_FILE
    cached = weight_cache_root / "litevggt" / LITEVGGT_WEIGHT_FILE
    target = models_root / "litevggt" / LITEVGGT_WEIGHT_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists() and cached.stat().st_size > 0:
        print(f"Using cached LiteVGGT weight from {cached}", flush=True)
        shutil.copy2(cached, target)
        return target.resolve()
    if target.exists() and target.stat().st_size > 0:
        return target.resolve()
    print(
        "LiteVGGT weight is not embedded during Docker build; "
        f"worker startup will ensure {runtime_weight}",
        flush=True,
    )
    return runtime_weight


def install_python_runtime(litevggt: Path, edgs: Path) -> None:
    pip_install(["--upgrade", "pip", "setuptools", "wheel"])
    pip_install(["-r", str(litevggt / "requirements.txt")])
    install_transformer_engine()
    diff_raster = edgs / "submodules" / "gaussian-splatting" / "submodules" / "diff-gaussian-rasterization"
    simple_knn = edgs / "submodules" / "gaussian-splatting" / "submodules" / "simple-knn"
    roma = edgs / "submodules" / "RoMa"
    if diff_raster.exists():
        install_edgs_extension("diff_gaussian_rasterization", diff_raster)
    if simple_knn.exists():
        install_edgs_extension("simple_knn", simple_knn)
    pip_install(
        [
            "pycolmap",
            "wandb",
            "hydra-core",
            "tqdm",
            "torchmetrics",
            "lpips",
            "matplotlib",
            "rich",
            "plyfile",
            "imageio",
            "imageio-ffmpeg",
            "numpy==1.26.4",
        ]
    )
    if roma.exists():
        pip_install(["--no-build-isolation", "-e", str(roma)])
    pip_install(["numpy==1.26.4"])


def install_transformer_engine() -> None:
    version = os.environ.get("TRANSFORMER_ENGINE_VERSION", DEFAULT_TRANSFORMER_ENGINE_VERSION).strip()
    if not version:
        raise RuntimeError("TRANSFORMER_ENGINE_VERSION cannot be empty")
    pip_uninstall(["transformer-engine-cu13"])
    pip_install([f"transformer-engine=={version}", f"transformer-engine-cu12=={version}"])
    pip_install(["--no-build-isolation", f"transformer-engine-torch=={version}"])
    validate_transformer_engine_packages(version)


def validate_transformer_engine_packages(version: str) -> None:
    expected = {
        "transformer-engine": version,
        "transformer-engine-cu12": version,
        "transformer-engine-torch": version,
    }
    installed = {name: installed_package_version(name) for name in expected}
    mismatched = {name: found for name, found in installed.items() if found != expected[name]}
    cu13_version = installed_package_version("transformer-engine-cu13")
    if mismatched or cu13_version is not None:
        raise RuntimeError(
            "Transformer Engine package mismatch: "
            f"expected={expected}, installed={installed}, transformer-engine-cu13={cu13_version}"
        )
    print(f"Transformer Engine packages pinned to CUDA 12 version {version}", flush=True)


def installed_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def install_edgs_extension(package: str, source_path: Path) -> None:
    mode = os.environ.get("EDGS_EXTENSION_INSTALL_MODE", "source").strip().lower()
    if mode not in {"auto", "wheel", "source"}:
        raise RuntimeError(f"Unsupported EDGS_EXTENSION_INSTALL_MODE={mode!r}; expected auto, wheel, or source")

    wheel = EDGS_WHEELS.get(package)
    can_use_official_wheel = (
        wheel is not None
        and sys.version_info[:2] == (3, 10)
        and platform.machine() == "x86_64"
        and sys.platform.startswith("linux")
    )
    if mode in {"auto", "wheel"} and can_use_official_wheel:
        url = f"{EDGS_WHEEL_BASE_URL}/{wheel}"
        try:
            pip_install([url])
            return
        except subprocess.CalledProcessError:
            if mode == "wheel":
                raise
            print(f"Official EDGS wheel failed for {package}; falling back to local CUDA build.", flush=True)
    elif mode == "wheel":
        raise RuntimeError(f"No official EDGS wheel is configured for {package} on this Python/platform")

    compile_edgs_extension(package, source_path)


def compile_edgs_extension(package: str, source_path: Path) -> None:
    if not source_path.exists():
        raise RuntimeError(f"EDGS extension source is missing for {package}: {source_path}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Cannot compile {package}: torch must be installed first") from exc
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.6+PTX")
    os.environ.setdefault("MAX_JOBS", str(max((os.cpu_count() or 2) // 2, 1)))
    pip_uninstall([package])
    pip_install(["--no-build-isolation", str(source_path)])


def pip_install(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def pip_uninstall(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "uninstall", "-y", *args], check=False)


def install_spark_runtime(spark: Path) -> None:
    registry = os.environ.get("NPM_CONFIG_REGISTRY", "https://registry.npmmirror.com")
    run(["npm", "config", "set", "registry", registry])
    run(["npm", "ci"], cwd=spark)
    run(["npm", "run", "build"], cwd=spark)


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
    litevggt: Path,
    edgs: Path,
    spark: Path,
    litevggt_weight: Path,
) -> None:
    script_dir = workspace / "backend" / "scripts"
    algorithms = [
        {
            "name": "LiteVGGT",
            "repo_url": "https://github.com/GarlicBa/LiteVGGT-repo",
            "license": "MIT",
            "commit_hash": commit_hash(litevggt),
            "weight_source": f"{os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')}/{LITEVGGT_WEIGHT_REPO}/resolve/main/{LITEVGGT_WEIGHT_FILE}",
            "local_path": str(litevggt),
            "enabled": True,
            "notes": "Docker-built preview geometry stage.",
            "weight_paths": [str(litevggt_weight)],
            "commands": {"run_demo": ["python3", str(script_dir / "run_litevggt_preview.py")]},
        },
        {
            "name": "EDGS",
            "repo_url": "https://github.com/CompVis/EDGS",
            "license": EDGS_LICENSE,
            "commit_hash": commit_hash(edgs),
            "weight_source": None,
            "local_path": str(edgs),
            "enabled": True,
            "notes": "Docker-built preview Gaussian training stage.",
            "weight_paths": [],
            "commands": {"train": ["python3", str(script_dir / "run_edgs_preview.py")]},
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
        {
            "name": "FFmpeg",
            "repo_url": "https://ffmpeg.org/",
            "license": "LGPL/GPL depending on build configuration",
            "commit_hash": "system-package",
            "weight_source": None,
            "local_path": None,
            "enabled": shutil.which("ffmpeg") is not None,
            "notes": "Video preview frame extraction.",
            "weight_paths": [],
            "source_type": "command",
            "commands": {"extract_frames": ["python3", str(script_dir / "run_ffmpeg_extract.py")]},
        },
        {
            "name": "LingBot-Map",
            "repo_url": "https://github.com/Robbyant/lingbot-map",
            "license": "Apache-2.0",
            "commit_hash": None,
            "weight_source": "local model-cache/lingbot-map/lingbot-map-long.pt",
            "local_path": None,
            "enabled": False,
            "notes": "Video preview algorithm; enabled by the separate LingBot-Map worker image.",
            "weight_paths": ["/model-cache/lingbot-map/lingbot-map-long.pt"],
            "commands": {"run_preview": ["python3", str(script_dir / "run_lingbot_map_preview.py")]},
        },
    ]
    registry_output.write_text(json.dumps({"algorithms": algorithms}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
