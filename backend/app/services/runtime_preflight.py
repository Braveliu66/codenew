from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from backend.app.algorithms.registry import AlgorithmRegistry


def build_runtime_preflight(registry: AlgorithmRegistry) -> dict[str, Any]:
    algorithms = [algorithm_status(entry.to_dict()) for entry in registry.list_entries()]
    transformer_engine = transformer_engine_status()
    edgs_extensions = edgs_cuda_extension_status()
    lingbot_runtime = lingbot_runtime_status()
    errors: list[str] = []
    warnings: list[str] = []
    for item in algorithms:
        if item["enabled"] and not item["ready"]:
            errors.extend(f"{item['name']}: {issue}" for issue in item["issues"])
        if not item["enabled"]:
            warnings.append(f"{item['name']} is disabled")
    if any(item["name"] == "LiteVGGT" and item["enabled"] for item in algorithms) and not transformer_engine["available"]:
        errors.append(f"Transformer Engine: {transformer_engine['message']}")
    if any(item["name"] == "EDGS" and item["enabled"] for item in algorithms) and not edgs_extensions["available"]:
        errors.extend(f"EDGS CUDA extension: {issue}" for issue in edgs_extensions["issues"])
    if any(item["name"] == "LingBot-Map" and item["enabled"] for item in algorithms) and not lingbot_runtime["available"]:
        errors.extend(f"LingBot-Map runtime: {issue}" for issue in lingbot_runtime["issues"])
    return {
        "python": python_status(),
        "gpu": gpu_status(),
        "torch": torch_status(),
        "transformer_engine": transformer_engine,
        "edgs_cuda_extensions": edgs_extensions,
        "lingbot_runtime": lingbot_runtime,
        "algorithms": algorithms,
        "errors": errors,
        "warnings": warnings,
    }


def python_status() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
    }


def gpu_status() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "message": "nvidia-smi is not available"}
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "message": str(exc)}
    gpus = []
    for line in completed.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_percent": float(parts[4]),
                }
            )
    return {"available": bool(gpus), "gpus": gpus}


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {"available": False, "message": "torch is not installed"}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def transformer_engine_status() -> dict[str, Any]:
    try:
        import transformer_engine.pytorch  # noqa: F401
    except ModuleNotFoundError as exc:
        return {"available": False, "message": f"{exc.name or 'transformer_engine'} is not installed"}
    except Exception as exc:
        return {"available": False, "message": str(exc)}

    def package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "available": True,
        "version": package_version("transformer-engine"),
        "cuda12_version": package_version("transformer-engine-cu12"),
        "cuda13_version": package_version("transformer-engine-cu13"),
        "torch_extension_version": package_version("transformer-engine-torch"),
    }


def edgs_cuda_extension_status() -> dict[str, Any]:
    checks = [
        ("simple_knn._C", "simple_knn"),
        ("diff_gaussian_rasterization", "diff_gaussian_rasterization"),
    ]
    issues: list[str] = []
    for module_name, label in checks:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            issues.append(f"{label} cannot be imported: {exc}")
    return {"available": not issues, "issues": issues}


def lingbot_runtime_status() -> dict[str, Any]:
    checks = [
        ("torch", "torch"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
    ]
    issues: list[str] = []
    for module_name, label in checks:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            issues.append(f"{label} cannot be imported: {exc}")
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("torch.cuda.is_available() is false")
    except Exception:
        pass
    flashinfer_available = True
    try:
        importlib.import_module("flashinfer")
    except Exception:
        flashinfer_available = False
    return {"available": not issues, "issues": issues, "flashinfer_available": flashinfer_available}


def algorithm_status(entry: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    local_path = Path(entry["local_path"]) if entry.get("local_path") else None
    if entry.get("enabled"):
        if entry.get("source_type") != "command":
            if local_path is None:
                issues.append("local_path is not configured")
            elif not local_path.exists():
                issues.append(f"local_path does not exist: {local_path}")
            elif entry.get("commit_hash"):
                actual = git_head(local_path)
                if actual and actual.lower() != str(entry["commit_hash"]).lower():
                    issues.append(f"commit mismatch: expected {entry['commit_hash']}, actual {actual}")
                elif not actual:
                    issues.append("git commit cannot be read")
        for weight in entry.get("weight_paths") or []:
            path = Path(str(weight))
            if not path.is_absolute() and local_path is not None:
                path = local_path / path
            if not path.exists() or (path.is_file() and path.stat().st_size <= 0):
                issues.append(f"weight is missing or empty: {path}")
        for key, command in (entry.get("commands") or {}).items():
            if not command:
                issues.append(f"command is empty: {key}")
            elif not command_exists(str(command[0])):
                issues.append(f"command executable is not available: {command[0]}")
    return {
        "name": entry.get("name"),
        "enabled": bool(entry.get("enabled")),
        "ready": bool(entry.get("enabled")) and not issues,
        "repo_url": entry.get("repo_url"),
        "license": entry.get("license"),
        "commit_hash": entry.get("commit_hash"),
        "local_path": entry.get("local_path"),
        "weight_paths": entry.get("weight_paths") or [],
        "commands": entry.get("commands") or {},
        "issues": issues,
    }


def git_head(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def command_exists(command: str) -> bool:
    if Path(command).is_absolute():
        return Path(command).exists()
    return shutil.which(command) is not None
