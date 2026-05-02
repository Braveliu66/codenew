from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CPU_SAMPLE_SECONDS = 1.0
WORKER_HEARTBEAT_MAX_AGE_SECONDS = 20
STREAM_SAMPLE_MAX_AGE_SECONDS = 5
NVIDIA_SMI_WINDOWS_PATH = Path("C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe")
POWERSHELL_COUNTER_SCRIPT = r"""
$ErrorActionPreference = "Stop"
Get-Counter "\Processor(_Total)\% Processor Time" | Out-Null
Start-Sleep -Milliseconds 500
$os = Get-CimInstance Win32_OperatingSystem
[float]$totalMemoryMB = $os.TotalVisibleMemorySize / 1KB
while ($true) {
    $cpu = (Get-Counter "\Processor(_Total)\% Processor Time").CounterSamples.CookedValue
    $availableMB = (Get-Counter "\Memory\Available MBytes").CounterSamples.CookedValue
    $usedMB = $totalMemoryMB - $availableMB
    $memPercent = [math]::Round(($usedMB / $totalMemoryMB) * 100, 2)
    Write-Output "$cpu $memPercent $usedMB $totalMemoryMB"
    Start-Sleep -Seconds 1
}
""".strip()


class LineStreamMonitor:
    def __init__(self, *, name: str, command_factory: Any, parser: Any, merge: Any | None = None) -> None:
        self.name = name
        self.command_factory = command_factory
        self.parser = parser
        self.merge = merge
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.latest: Any = None
        self.latest_at: float | None = None
        self.last_error: str | None = None

    def start(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                return
            command = self.command_factory()
            if not command:
                self.last_error = f"{self.name} command is not available"
                return
            try:
                self.process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                self.process = None
                self.last_error = str(exc)
                return
            self.thread = threading.Thread(target=self._read_loop, name=f"{self.name}-reader", daemon=True)
            self.thread.start()

    def _read_loop(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            parsed = self.parser(line)
            if parsed is None:
                continue
            with self.lock:
                self.latest = self.merge(self.latest, parsed) if self.merge else parsed
                self.latest_at = time.monotonic()
                self.last_error = None

    def snapshot(self, *, max_age_seconds: float = STREAM_SAMPLE_MAX_AGE_SECONDS) -> Any | None:
        self.start()
        with self.lock:
            if self.latest_at is None or self.latest is None:
                return None
            if time.monotonic() - self.latest_at > max_age_seconds:
                return None
            return self.latest

    def stop(self) -> None:
        with self.lock:
            process = self.process
            self.process = None
        if process and process.poll() is None:
            process.terminate()


def gpu_loop_command() -> list[str] | None:
    nvidia_smi = find_nvidia_smi()
    if not nvidia_smi:
        return None
    return [
        nvidia_smi,
        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
        "--loop=1",
    ]


def cpu_memory_loop_command() -> list[str] | None:
    powershell = shutil.which("powershell") or shutil.which("powershell.exe") or shutil.which("pwsh") or shutil.which("pwsh.exe")
    if not powershell:
        return None
    return [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", POWERSHELL_COUNTER_SCRIPT]


def parse_gpu_loop_line(line: str) -> dict[str, Any] | None:
    gpus = parse_nvidia_smi_gpus(line)
    return gpus[0] if gpus else None


def merge_gpu_loop_sample(latest: Any, gpu: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = {
        (item.get("uuid") or item.get("index")): dict(item)
        for item in latest
    } if isinstance(latest, list) else {}
    gpus[gpu.get("uuid") or gpu.get("index")] = gpu
    return sorted(gpus.values(), key=lambda item: int(item.get("index") or 0))


def parse_cpu_memory_loop_line(line: str) -> dict[str, float] | None:
    parts = line.strip().split()
    if len(parts) < 4:
        return None
    try:
        return {
            "cpu_percent": float(parts[0]),
            "memory_percent": float(parts[1]),
            "memory_used_mb": float(parts[2]),
            "memory_total_mb": float(parts[3]),
        }
    except ValueError:
        return None


GPU_LOOP_MONITOR = LineStreamMonitor(
    name="nvidia-smi-loop",
    command_factory=gpu_loop_command,
    parser=parse_gpu_loop_line,
    merge=merge_gpu_loop_sample,
)
CPU_MEMORY_LOOP_MONITOR = LineStreamMonitor(
    name="powershell-counter-loop",
    command_factory=cpu_memory_loop_command,
    parser=parse_cpu_memory_loop_line,
)

atexit.register(GPU_LOOP_MONITOR.stop)
atexit.register(CPU_MEMORY_LOOP_MONITOR.stop)


def current_cpu_resources() -> dict[str, Any]:
    stream = CPU_MEMORY_LOOP_MONITOR.snapshot()
    if stream:
        return {
            "available": True,
            "usage_percent": round(float(stream["cpu_percent"]), 1),
            "source": "powershell-counter-loop",
            "sample_seconds": 1.0,
            "logical_cpu_count": None,
            "physical_cpu_count": None,
            "message": None,
        }
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
        "source": "psutil-system",
        "sample_seconds": CPU_SAMPLE_SECONDS,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "message": None,
    }


def current_memory_resources() -> dict[str, Any]:
    stream = CPU_MEMORY_LOOP_MONITOR.snapshot()
    if stream:
        total = int(float(stream["memory_total_mb"]) * 1024 * 1024)
        used = int(float(stream["memory_used_mb"]) * 1024 * 1024)
        return {
            "available": True,
            "usage_percent": round(float(stream["memory_percent"]), 1),
            "total": total,
            "used": used,
            "available_bytes": max(total - used, 0),
            "source": "powershell-counter-loop",
            "message": None,
        }
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
        "source": "psutil-system",
        "message": None,
    }


def current_gpu_resources() -> dict[str, Any]:
    gpus = GPU_LOOP_MONITOR.snapshot()
    if gpus:
        return aggregate_gpus(gpus, source="nvidia-smi-loop")
    nvidia_smi = current_nvidia_smi_resources()
    if nvidia_smi.get("available"):
        return nvidia_smi
    nvml = current_nvml_gpu_resources()
    if nvml.get("available"):
        return nvml
    return nvidia_smi if nvidia_smi.get("message") else nvml


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
    nvidia_smi = find_nvidia_smi()
    if not nvidia_smi:
        return unavailable_gpu("nvidia-smi is not available in this runtime", source="nvidia-smi")
    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total",
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


def find_nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe") or (
        str(NVIDIA_SMI_WINDOWS_PATH) if NVIDIA_SMI_WINDOWS_PATH.exists() else None
    )


def parse_nvidia_smi_gpus(output: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            if len(parts) >= 6:
                uuid = parts[1]
                name = parts[2]
                usage = float(parts[3])
                used = float(parts[4])
                total = float(parts[5])
            else:
                uuid = None
                name = parts[1]
                usage = float(parts[2])
                total = float(parts[3])
                used = float(parts[4])
        except ValueError:
            continue
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
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


def gpu_resources_from_workers(workers: list[Any], *, max_age_seconds: int | None = WORKER_HEARTBEAT_MAX_AGE_SECONDS) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    latest_by_gpu: dict[tuple[int | None, str | None, int], Any] = {}
    stale_worker_count = 0
    for worker in workers:
        if worker.gpu_index is None or worker.gpu_memory_total is None or worker.gpu_memory_used is None:
            continue
        age_seconds = heartbeat_age_seconds(worker.last_seen_at, now)
        if max_age_seconds is not None and age_seconds is not None and age_seconds > max_age_seconds:
            stale_worker_count += 1
            continue
        key = (worker.gpu_index, worker.gpu_name, int(worker.gpu_memory_total or 0))
        existing = latest_by_gpu.get(key)
        if existing is None or comparable_datetime(worker.last_seen_at) > comparable_datetime(existing.last_seen_at):
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
            "heartbeat_age_seconds": heartbeat_age_seconds(worker.last_seen_at, now),
        }
        for worker in latest_by_gpu.values()
    ]
    if not gpus:
        result = unavailable_gpu("no fresh worker GPU heartbeat is available", source="worker-device-sample")
        result["stale_worker_count"] = stale_worker_count
        return result
    result = aggregate_gpus(gpus, source="worker-device-sample")
    result["stale_worker_count"] = stale_worker_count
    return result


def fresh_worker_heartbeats(workers: list[Any], *, max_age_seconds: int | None = WORKER_HEARTBEAT_MAX_AGE_SECONDS) -> list[Any]:
    if max_age_seconds is None:
        return workers
    now = datetime.now(timezone.utc)
    fresh = []
    for worker in workers:
        age_seconds = heartbeat_age_seconds(getattr(worker, "last_seen_at", None), now)
        if age_seconds is not None and age_seconds <= max_age_seconds:
            fresh.append(worker)
    return fresh


def heartbeat_age_seconds(last_seen_at: Any, now: datetime) -> float | None:
    if not isinstance(last_seen_at, datetime):
        return None
    if last_seen_at.tzinfo is None:
        last_seen_at = last_seen_at.replace(tzinfo=timezone.utc)
    return round(max((now - last_seen_at).total_seconds(), 0.0), 1)


def comparable_datetime(value: Any) -> datetime:
    if not isinstance(value, datetime):
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
