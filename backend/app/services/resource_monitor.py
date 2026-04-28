from __future__ import annotations

import shutil
import subprocess
from typing import Any


CPU_SAMPLE_SECONDS = 0.25


def current_cpu_resources() -> dict[str, Any]:
    try:
        import psutil
    except ModuleNotFoundError:
        return {
            "available": False,
            "usage_percent": None,
            "source": None,
            "sample_seconds": None,
            "message": "psutil is not installed",
        }
    usage = psutil.cpu_percent(interval=CPU_SAMPLE_SECONDS)
    return {
        "available": True,
        "usage_percent": round(float(usage), 1),
        "source": "psutil",
        "sample_seconds": CPU_SAMPLE_SECONDS,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "message": None,
    }


def current_memory_resources() -> dict[str, Any]:
    try:
        import psutil
    except ModuleNotFoundError:
        return {
            "available": False,
            "usage_percent": None,
            "total": None,
            "used": None,
            "available_bytes": None,
            "source": None,
            "message": "psutil is not installed",
        }
    memory = psutil.virtual_memory()
    return {
        "available": True,
        "usage_percent": round(float(memory.percent), 1),
        "total": int(memory.total),
        "used": int(memory.used),
        "available_bytes": int(memory.available),
        "source": "psutil",
        "message": None,
    }


def current_gpu_resources() -> dict[str, Any]:
    nvml = current_nvml_gpu_resources()
    if nvml.get("available"):
        return nvml
    nvidia_smi = current_nvidia_smi_resources()
    if nvidia_smi.get("available"):
        return nvidia_smi
    return nvml if nvml.get("message") else nvidia_smi


def current_nvml_gpu_resources() -> dict[str, Any]:
    try:
        import pynvml
    except ModuleNotFoundError:
        return unavailable_gpu("pynvml is not installed", source="pynvml")
    try:
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        gpus = []
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_mb = float(memory.total) / 1024 / 1024
            used_mb = float(memory.used) / 1024 / 1024
            gpus.append(
                {
                    "index": index,
                    "name": str(name),
                    "usage_percent": float(utilization.gpu),
                    "memory_total": round(total_mb, 1),
                    "memory_used": round(used_mb, 1),
                    "memory_usage_percent": round((used_mb / total_mb) * 100, 1) if total_mb > 0 else None,
                    "memory_utilization_percent": float(utilization.memory),
                }
            )
    except Exception as exc:
        return unavailable_gpu(str(exc), source="pynvml")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return aggregate_gpus(gpus, source="pynvml")


def current_nvidia_smi_resources() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return unavailable_gpu("nvidia-smi is not available in this runtime", source="nvidia-smi")
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,name,utilization.gpu,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return unavailable_gpu(str(exc), source="nvidia-smi")
    gpus = parse_nvidia_smi_gpus(completed.stdout)
    if not gpus:
        return unavailable_gpu("nvidia-smi returned no GPU utilization data", source="nvidia-smi")
    return aggregate_gpus(gpus, source="nvidia-smi")


def parse_nvidia_smi_gpus(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            usage = float(parts[2])
            total = float(parts[3])
            used = float(parts[4])
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "usage_percent": usage,
                "memory_total": total,
                "memory_used": used,
                "memory_usage_percent": round((used / total) * 100, 1) if total > 0 else None,
            }
        )
    return gpus


def aggregate_gpus(gpus: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    if not gpus:
        return unavailable_gpu("no GPU utilization data is available", source=source)
    total = sum(float(gpu["memory_total"]) for gpu in gpus)
    used = sum(float(gpu["memory_used"]) for gpu in gpus)
    return {
        "available": True,
        "usage_percent": max(float(gpu["usage_percent"]) for gpu in gpus),
        "memory_total": round(total, 1),
        "memory_used": round(used, 1),
        "memory_usage_percent": round((used / total) * 100, 1) if total > 0 else None,
        "gpus": gpus,
        "source": source,
        "message": None,
    }


def unavailable_gpu(message: str, *, source: str) -> dict[str, Any]:
    return {
        "available": False,
        "usage_percent": None,
        "memory_total": None,
        "memory_used": None,
        "memory_usage_percent": None,
        "gpus": [],
        "source": source,
        "message": message,
    }


def gpu_resources_from_workers(workers: list[Any]) -> dict[str, Any]:
    latest_by_gpu: dict[tuple[str | None, int | None], Any] = {}
    for worker in workers:
        if worker.gpu_index is None or worker.gpu_memory_total is None or worker.gpu_memory_used is None:
            continue
        key = (worker.hostname, worker.gpu_index)
        existing = latest_by_gpu.get(key)
        if existing is None or worker.last_seen_at > existing.last_seen_at:
            latest_by_gpu[key] = worker
    gpus = [
        {
            "index": worker.gpu_index,
            "name": worker.gpu_name,
            "usage_percent": float(worker.gpu_utilization or 0),
            "memory_total": float(worker.gpu_memory_total or 0),
            "memory_used": float(worker.gpu_memory_used or 0),
            "memory_usage_percent": (
                round((float(worker.gpu_memory_used or 0) / float(worker.gpu_memory_total or 0)) * 100, 1)
                if float(worker.gpu_memory_total or 0) > 0
                else None
            ),
        }
        for worker in latest_by_gpu.values()
    ]
    if not gpus:
        return unavailable_gpu("no worker GPU heartbeat is available", source="worker-heartbeat")
    return aggregate_gpus(gpus, source="worker-heartbeat")
