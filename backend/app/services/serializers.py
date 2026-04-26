from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

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
    return {
        "id": task.id,
        "project_id": task.project_id,
        "type": task.type,
        "status": task.status,
        "priority": task.priority,
        "progress": task.progress,
        "worker_id": task.worker_id,
        "options": task.options or {},
        "metrics": task.metrics or {},
        "current_stage": task.current_stage or "",
        "eta_seconds": task.eta_seconds,
        "error_code": task.error_code,
        "error_message": task.error_message,
        "logs": task.logs or [],
        "created_at": iso_datetime(task.created_at),
        "started_at": iso_datetime(task.started_at),
        "finished_at": iso_datetime(task.finished_at),
    }


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
