from __future__ import annotations

import os
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.algorithms.errors import AlgorithmErrorCode
from backend.app.algorithms.models import PreviewTaskRequest, TaskExecutionResult
from backend.app.algorithms.preview_engine import PreviewEngine
from backend.app.core.config import get_settings
from backend.app.db import models
from backend.app.db.session import SessionLocal, init_database
from backend.app.services.object_storage import ObjectStorage
from backend.app.services.registry_store import load_registry_from_db
from backend.app.services.seed import seed_database
from backend.app.services.task_queue import PreviewTaskQueue, TaskQueueError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    init_database()
    with SessionLocal() as db:
        seed_database(db)
    storage = ObjectStorage()
    storage.ensure_bucket()
    queue = PreviewTaskQueue()
    worker_id = os.environ.get("WORKER_ID", f"preview-{socket.gethostname()}-{uuid.uuid4().hex[:8]}")
    while True:
        with SessionLocal() as db:
            write_heartbeat(db, worker_id=worker_id, current_task_id=None)
        try:
            task_id = queue.pop_preview(timeout_seconds=5)
        except TaskQueueError as exc:
            with SessionLocal() as db:
                write_heartbeat(db, worker_id=worker_id, current_task_id=None)
            print(f"preview worker queue unavailable: {exc}", flush=True)
            time.sleep(5)
            continue
        if not task_id:
            continue
        with SessionLocal() as db:
            write_heartbeat(db, worker_id=worker_id, current_task_id=task_id)
            process_preview_task(db, task_id, worker_id=worker_id, storage=storage)


def process_preview_task(
    db: Session,
    task_id: str,
    *,
    worker_id: str,
    storage: ObjectStorage | None = None,
) -> models.Task | None:
    storage = storage or ObjectStorage()
    task = db.scalar(
        select(models.Task)
        .where(models.Task.id == task_id)
        .options(selectinload(models.Task.project).selectinload(models.Project.media_assets))
    )
    if task is None:
        return None
    project = task.project
    if task.status == "canceled":
        return task

    task.status = "running"
    task.progress = 5
    task.worker_id = worker_id
    task.current_stage = "materializing_inputs"
    task.started_at = task.started_at or models.utc_now()
    task.error_code = None
    task.error_message = None
    project.status = "PREVIEW_RUNNING"
    project.error_message = None
    db.commit()

    try:
        request = build_preview_request(task, project, storage)
        task.current_stage = "preview_engine"
        task.progress = 15
        db.commit()
        result = PreviewEngine(load_registry_from_db(db)).execute(request)
        if result.status != "succeeded":
            mark_failed(db, task, project, result)
            return task
        persist_success(db, task, project, result, storage)
        return task
    except Exception as exc:
        code = AlgorithmErrorCode.PREVIEW_ARTIFACT_INVALID.value
        task.status = "failed"
        task.progress = 100
        task.current_stage = "failed"
        task.error_code = code
        task.error_message = f"{code}: {exc}"
        task.finished_at = models.utc_now()
        project.status = "FAILED"
        project.error_message = task.error_message
        db.commit()
        return task
    finally:
        write_heartbeat(db, worker_id=worker_id, current_task_id=None)


def build_preview_request(task: models.Task, project: models.Project, storage: ObjectStorage) -> PreviewTaskRequest:
    settings = get_settings()
    work_dir = settings.storage_root / "work" / task.id
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    media = list(project.media_assets)
    if project.input_type == "images":
        images = [item for item in media if item.kind == "image"]
        if not images:
            raise ValueError("preview requires at least one uploaded image")
        min_frames = settings.preview_min_input_frames
        max_frames = int((task.options or {}).get("max_preview_frames") or settings.preview_max_input_frames)
        max_frames = min(max(max_frames, min_frames), settings.preview_max_input_frames)
        if len(images) < min_frames:
            raise ValueError(f"preview requires at least {min_frames} uploaded images")
        images = select_evenly(images, max_frames)
        image_dir = raw_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(images):
            suffix = Path(item.file_name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                suffix = ".jpg"
            target = image_dir / f"{index:04d}{suffix}"
            storage.download_to_path(item.object_uri, target)
        raw_uri = str(image_dir)
    elif project.input_type == "video":
        videos = [item for item in media if item.kind == "video"]
        if not videos:
            raise ValueError("preview requires an uploaded video")
        source = videos[0]
        suffix = Path(source.file_name).suffix or ".mp4"
        target = raw_dir / f"source{suffix}"
        storage.download_to_path(source.object_uri, target)
        raw_uri = str(target)
    else:
        raise ValueError("camera preview is not implemented")

    return PreviewTaskRequest(
        task_id=task.id,
        project_id=project.id,
        user_id=project.owner_id,
        input_type=project.input_type,
        raw_uri=raw_uri,
        work_dir=work_dir,
        output_prefix=f"minio://{storage.bucket}/users/{project.owner_id}/projects/{project.id}/preview",
        timeout_seconds=int((task.options or {}).get("timeout_seconds") or 300),
        options=task.options or {},
    )


def select_evenly(items: list[models.MediaAsset], max_items: int) -> list[models.MediaAsset]:
    if len(items) <= max_items:
        return items
    if max_items <= 1:
        return items[:1]
    last = len(items) - 1
    indexes = [round(i * last / (max_items - 1)) for i in range(max_items)]
    selected: list[models.MediaAsset] = []
    seen: set[int] = set()
    for index in indexes:
        if index not in seen:
            selected.append(items[index])
            seen.add(index)
    return selected


def mark_failed(db: Session, task: models.Task, project: models.Project, result: TaskExecutionResult) -> None:
    errors = result.errors or []
    first_error = errors[0] if errors else {}
    message = "; ".join(str(error.get("message") or "") for error in errors).strip() or "Preview failed"
    task.status = "failed"
    task.progress = 100
    task.current_stage = "failed"
    task.error_code = str(first_error.get("code") or AlgorithmErrorCode.PREVIEW_ARTIFACT_INVALID.value)
    task.error_message = message
    task.metrics = result.to_dict()
    task.logs = result.logs or []
    task.finished_at = models.utc_now()
    project.status = "FAILED"
    project.error_message = message
    db.commit()


def persist_success(db: Session, task: models.Task, project: models.Project, result: TaskExecutionResult, storage: ObjectStorage) -> None:
    preview = next((item for item in result.artifacts if item.get("kind") == "preview_spz"), None)
    if not preview:
        raise RuntimeError("successful preview result did not include preview_spz")
    source = Path(str(preview.get("path") or ""))
    if not source.exists() or source.stat().st_size <= 0:
        raise RuntimeError("preview_spz artifact is missing or empty")
    object_name = f"users/{project.owner_id}/projects/{project.id}/preview/preview.spz"
    object_uri = storage.put_file(object_name, source, content_type="application/octet-stream")
    artifact = models.Artifact(
        project_id=project.id,
        task_id=task.id,
        kind="preview_spz",
        object_uri=object_uri,
        file_name="preview.spz",
        file_size=source.stat().st_size,
        artifact_metadata={"pipeline": "litevggt_edgs_spz"},
    )
    db.add(artifact)
    task.status = "succeeded"
    task.progress = 100
    task.current_stage = "preview_ready"
    task.metrics = result.to_dict()
    task.logs = result.logs or []
    task.finished_at = models.utc_now()
    project.status = "PREVIEW_READY"
    project.error_message = None
    db.commit()


def write_heartbeat(db: Session, *, worker_id: str, current_task_id: str | None) -> None:
    gpu = detect_gpu()
    heartbeat = db.get(models.WorkerHeartbeat, worker_id)
    if heartbeat is None:
        heartbeat = models.WorkerHeartbeat(worker_id=worker_id, hostname=socket.gethostname())
        db.add(heartbeat)
    heartbeat.hostname = socket.gethostname()
    heartbeat.current_task_id = current_task_id
    heartbeat.last_seen_at = models.utc_now()
    heartbeat.gpu_index = gpu.get("gpu_index")
    heartbeat.gpu_name = gpu.get("gpu_name")
    heartbeat.gpu_memory_total = gpu.get("gpu_memory_total")
    heartbeat.gpu_memory_used = gpu.get("gpu_memory_used")
    heartbeat.gpu_utilization = gpu.get("gpu_utilization")
    db.commit()


def detect_gpu() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 5:
        return {}
    return {
        "gpu_index": int(parts[0]),
        "gpu_name": parts[1],
        "gpu_memory_total": int(parts[2]),
        "gpu_memory_used": int(parts[3]),
        "gpu_utilization": float(parts[4]),
    }


if __name__ == "__main__":
    main()
