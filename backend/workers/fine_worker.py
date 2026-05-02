from __future__ import annotations

import os
import socket
import time
import uuid
from pathlib import Path
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from backend.app.algorithms.errors import AlgorithmErrorCode
from backend.app.algorithms.fine_engine import FineSynthesisEngine
from backend.app.algorithms.models import FineTaskRequest, TaskExecutionResult
from backend.app.core.config import get_settings
from backend.app.db import models
from backend.app.db.session import SessionLocal, init_database
from backend.app.services.object_storage import ObjectStorage
from backend.app.services.registry_store import load_registry_from_db
from backend.app.services.seed import seed_database
from backend.app.services.task_queue import FineTaskQueue, TaskQueueError
from backend.workers.preview_worker import collect_task_logs, start_heartbeat_thread, write_heartbeat


def main() -> None:
    init_database()
    with SessionLocal() as db:
        seed_database(db)
    storage = ObjectStorage()
    storage.ensure_bucket()
    queue = FineTaskQueue()
    worker_id = os.environ.get("WORKER_ID", f"fine-{socket.gethostname()}-{uuid.uuid4().hex[:8]}")
    while True:
        with SessionLocal() as db:
            write_heartbeat(db, worker_id=worker_id, current_task_id=None)
        try:
            task_id = queue.pop_fine(timeout_seconds=5)
        except TaskQueueError as exc:
            with SessionLocal() as db:
                write_heartbeat(db, worker_id=worker_id, current_task_id=None)
            print(f"fine worker queue unavailable: {exc}", flush=True)
            time.sleep(5)
            continue
        if not task_id:
            continue
        with SessionLocal() as db:
            write_heartbeat(db, worker_id=worker_id, current_task_id=task_id)
            process_fine_task(db, task_id, worker_id=worker_id, storage=storage)


def process_fine_task(
    db: Session,
    task_id: str,
    *,
    worker_id: str,
    storage: ObjectStorage | None = None,
    engine_factory: Callable[[Session], FineSynthesisEngine] | None = None,
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
    if task.type != "fine":
        return task

    task.status = "running"
    task.progress = 5
    task.worker_id = worker_id
    task.current_stage = "materializing_inputs"
    task.started_at = task.started_at or models.utc_now()
    task.error_code = None
    task.error_message = None
    project.status = "FINE_RUNNING"
    project.error_message = None
    db.commit()
    stop_heartbeat = start_heartbeat_thread(worker_id=worker_id, current_task_id=task_id)

    try:
        request = build_fine_request(task, project, storage)
        task.current_stage = "fine_engine"
        task.progress = 15
        db.commit()
        engine = engine_factory(db) if engine_factory else FineSynthesisEngine(load_registry_from_db(db))
        result = engine.execute(request)
        if result.status != "succeeded":
            mark_failed(db, task, project, result)
            return task
        persist_success(db, task, project, result, storage)
        return task
    except Exception as exc:
        code = AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID.value
        task.status = "failed"
        task.progress = 100
        task.current_stage = "failed"
        task.error_code = code
        task.error_message = f"{code}: {exc}"
        task.logs = [str(exc)]
        task.finished_at = models.utc_now()
        project.status = "FAILED"
        project.error_message = task.error_message
        db.commit()
        return task
    finally:
        stop_heartbeat()
        write_heartbeat(db, worker_id=worker_id, current_task_id=None)


def build_fine_request(task: models.Task, project: models.Project, storage: ObjectStorage) -> FineTaskRequest:
    settings = get_settings()
    work_dir = settings.storage_root / "work" / task.id
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    media = list(project.media_assets)
    options = task.options or {}
    input_policy = options.get("input_policy") if isinstance(options.get("input_policy"), dict) else {}
    if project.input_type == "images":
        images = [item for item in media if item.kind == "image"]
        if not images:
            raise ValueError("fine reconstruction requires at least one uploaded image")
        image_dir = raw_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(images):
            suffix = Path(item.file_name).suffix.lower() or ".jpg"
            target = image_dir / f"{index:04d}{suffix}"
            storage.download_to_path(item.object_uri, target)
        raw_uri = str(image_dir)
        frame_count = int(input_policy.get("frame_count") or len(images))
    elif project.input_type == "video":
        videos = [item for item in media if item.kind == "video"]
        if not videos:
            raise ValueError("fine reconstruction requires an uploaded video")
        source = sorted(videos, key=lambda item: item.created_at)[-1]
        suffix = Path(source.file_name).suffix or ".mp4"
        target = raw_dir / f"source{suffix}"
        storage.download_to_path(source.object_uri, target)
        raw_uri = str(target)
        frame_count = int(input_policy.get("frame_count") or 0)
    else:
        raise ValueError("camera fine reconstruction is not implemented")

    return FineTaskRequest(
        task_id=task.id,
        project_id=project.id,
        input_type=project.input_type,
        raw_uri=raw_uri,
        work_dir=work_dir,
        output_prefix=f"minio://{storage.bucket}/users/{project.owner_id}/projects/{project.id}/fine",
        frame_count=frame_count,
        blur_detected=bool(options.get("blur_detected", False)),
        enable_long_video_global_optimization=bool(options.get("enable_long_video_global_optimization", False)),
        timeout_seconds=int(options.get("timeout_seconds") or 7200),
        options=options,
    )


def mark_failed(db: Session, task: models.Task, project: models.Project, result: TaskExecutionResult) -> None:
    errors = result.errors or []
    first_error = errors[0] if errors else {}
    message = "; ".join(str(error.get("message") or "") for error in errors).strip() or "Fine reconstruction failed"
    task.status = "failed"
    task.progress = 100
    task.current_stage = "failed"
    task.error_code = str(first_error.get("code") or "FINE_RECONSTRUCTION_FAILED")
    task.error_message = message
    task.metrics = result.to_dict()
    task.logs = collect_task_logs(result)
    task.finished_at = models.utc_now()
    project.status = "FAILED"
    project.error_message = message
    db.commit()


def persist_success(db: Session, task: models.Task, project: models.Project, result: TaskExecutionResult, storage: ObjectStorage) -> None:
    if not result.artifacts:
        raise RuntimeError("successful fine result did not include artifacts")
    for item in result.artifacts:
        source = Path(str(item.get("path") or ""))
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"fine artifact is missing or empty: {source}")
        file_name = str(item.get("file_name") or source.name)
        kind = str(item.get("kind") or "fine_artifact")
        object_name = f"users/{project.owner_id}/projects/{project.id}/fine/{file_name}"
        object_uri = storage.put_file(object_name, source, content_type="application/octet-stream")
        db.add(
            models.Artifact(
                project_id=project.id,
                task_id=task.id,
                kind=kind,
                object_uri=object_uri,
                file_name=file_name,
                file_size=source.stat().st_size,
                checksum=item.get("checksum"),
                artifact_metadata={
                    "pipeline": "fused3dgs_fine",
                    **dict(item.get("metadata") or {}),
                },
            )
        )
    task.status = "succeeded"
    task.progress = 100
    task.current_stage = "completed"
    task.metrics = result.to_dict()
    task.logs = collect_task_logs(result)
    task.finished_at = models.utc_now()
    project.status = "COMPLETED"
    project.error_message = None
    db.commit()


if __name__ == "__main__":
    main()
