from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.app.core.config import get_settings
from backend.app.db import models


def iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def user_to_dict(user: models.User) -> dict[str, Any]:
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "created_at": iso_datetime(user.created_at),
    }


def project_to_dict(project: models.Project) -> dict[str, Any]:
    return {
        "id": project.id,
        "owner_id": project.owner_id,
        "name": project.name,
        "input_type": project.input_type,
        "status": project.status,
        "tags": project.tags or [],
        "total_size_bytes": project.total_size_bytes or 0,
        "preview_image_uri": project.preview_image_uri,
        "error_message": project.error_message,
        "created_at": iso_datetime(project.created_at),
        "updated_at": iso_datetime(project.updated_at),
    }


def media_to_dict(asset: models.MediaAsset) -> dict[str, Any]:
    return {
        "id": asset.id,
        "project_id": asset.project_id,
        "kind": asset.kind,
        "object_uri": asset.object_uri,
        "thumbnail_uri": asset.thumbnail_uri,
        "file_name": asset.file_name,
        "file_size": asset.file_size,
        "width": asset.width,
        "height": asset.height,
        "duration_seconds": asset.duration_seconds,
        "quality_flags": asset.quality_flags or {},
        "created_at": iso_datetime(asset.created_at),
    }


def task_to_dict(task: models.Task) -> dict[str, Any]:
    progress, eta_seconds = task_progress_snapshot(task)
    return {
        "id": task.id,
        "project_id": task.project_id,
        "type": task.type,
        "status": task.status,
        "priority": task.priority,
        "progress": progress,
        "worker_id": task.worker_id,
        "options": task.options or {},
        "metrics": task.metrics or {},
        "current_stage": task.current_stage or "",
        "eta_seconds": eta_seconds,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "logs": task.logs or [],
        "created_at": iso_datetime(task.created_at),
        "started_at": iso_datetime(task.started_at),
        "finished_at": iso_datetime(task.finished_at),
    }


def task_progress_snapshot(task: models.Task) -> tuple[int, int | None]:
    base_progress = clamp_int(task.progress or 0, 0, 100)
    if task.status in {"succeeded", "failed"}:
        return 100, 0
    if task.status == "canceled":
        return base_progress, 0
    if task.status != "running" or task.started_at is None:
        return base_progress, task.eta_seconds

    started_at = task.started_at
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    elapsed = max(0, int((datetime.now(timezone.utc) - started_at.astimezone(timezone.utc)).total_seconds()))
    total = estimated_task_duration_seconds(task)
    if total <= 0:
        return base_progress, task.eta_seconds

    start, end = task_progress_range(task)
    computed_progress = start + int((min(elapsed, total) / total) * max(end - start, 1))
    progress = clamp_int(max(base_progress, computed_progress), 0, end)
    eta_seconds = None if elapsed >= total else max(0, total - elapsed)
    return progress, eta_seconds


def task_progress_range(task: models.Task) -> tuple[int, int]:
    if task.type != "preview":
        return (15, 95)
    stage = task.current_stage or ""
    if stage == "materializing_inputs":
        return (5, 15)
    if stage == "video_frame_extraction":
        return (20, 30)
    if stage in {"video_lingbot_map", "camera_lingbot_map"}:
        return (35, 90)
    if stage == "geometry_litevggt":
        return (35, 55)
    if stage == "training_edgs":
        return (60, 94)
    if stage == "spz_conversion":
        return (92, 98)
    return (15, 94)


def estimated_task_duration_seconds(task: models.Task) -> int:
    options = task.options or {}
    explicit = positive_int(options.get("estimated_duration_seconds"))
    if explicit is not None:
        return explicit
    timeout = positive_int(options.get("timeout_seconds"))
    if task.type == "preview":
        settings = get_settings()
        timeout = timeout or 300
        frame_policy = options.get("input_frame_policy") if isinstance(options.get("input_frame_policy"), dict) else {}
        selected_frames = positive_int(frame_policy.get("selected_input_frames")) or positive_int(options.get("max_preview_frames"))
        selected_frames = selected_frames or settings.preview_min_input_frames
        edgs_epochs = positive_int(options.get("edgs_epochs")) or settings.preview_default_edgs_epochs
        frame_factor = min(max(selected_frames, settings.preview_min_input_frames), settings.preview_max_input_frames) / max(settings.preview_max_input_frames, 1)
        epoch_factor = edgs_epochs / max(settings.preview_default_edgs_epochs, 1)
        estimated = int(45 + (90 * frame_factor) + (60 * epoch_factor))
        return min(max(estimated, 60), timeout)
    if task.type == "fine":
        return timeout or 7200
    return timeout or 300


def positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def artifact_to_dict(artifact: models.Artifact) -> dict[str, Any]:
    return {
        "id": artifact.id,
        "project_id": artifact.project_id,
        "task_id": artifact.task_id,
        "kind": artifact.kind,
        "object_uri": artifact.object_uri,
        "file_name": artifact.file_name,
        "file_size": artifact.file_size,
        "checksum": artifact.checksum,
        "metadata": artifact.artifact_metadata or {},
        "created_at": iso_datetime(artifact.created_at),
    }


def feedback_to_dict(feedback: models.Feedback) -> dict[str, Any]:
    return {
        "id": feedback.id,
        "user_id": feedback.user_id,
        "project_id": feedback.project_id,
        "title": feedback.title,
        "content": feedback.content,
        "status": feedback.status,
        "created_at": iso_datetime(feedback.created_at),
    }


def algorithm_to_dict(record: models.AlgorithmRegistryRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "repo_url": record.repo_url,
        "license": record.license,
        "commit_hash": record.commit_hash,
        "weight_source": record.weight_source,
        "local_path": record.local_path,
        "enabled": record.enabled,
        "notes": record.notes,
        "commands": record.commands or {},
        "weight_paths": record.weight_paths or [],
        "source_type": record.source_type,
    }


def worker_to_dict(worker: models.WorkerHeartbeat) -> dict[str, Any]:
    return {
        "worker_id": worker.worker_id,
        "hostname": worker.hostname,
        "gpu_index": worker.gpu_index,
        "gpu_name": worker.gpu_name,
        "gpu_memory_total": worker.gpu_memory_total,
        "gpu_memory_used": worker.gpu_memory_used,
        "gpu_utilization": worker.gpu_utilization,
        "current_task_id": worker.current_task_id,
        "last_seen_at": iso_datetime(worker.last_seen_at),
    }
