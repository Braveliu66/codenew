from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings
from sqlalchemy import Select, select
from sqlalchemy.orm import Session, selectinload

from backend.app.db import models
from backend.app.services.object_storage import ObjectStorage
from backend.app.services.serializers import artifact_to_dict, media_to_dict, project_to_dict, task_to_dict


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def safe_filename(name: str) -> str:
    return Path(name).name.replace("\\", "_").replace("/", "_") or "upload.bin"


def detect_kind(filename: str, content_type: str | None = None) -> str:
    suffix = Path(filename).suffix.lower()
    content_type = content_type or ""
    if content_type.startswith("image/") or suffix in IMAGE_EXTENSIONS:
        return "image"
    if content_type.startswith("video/") or suffix in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError("unsupported media file type")


def project_query_for_user(user: models.User, project_id: str) -> Select[tuple[models.Project]]:
    query = select(models.Project).where(models.Project.id == project_id)
    if user.role != "admin":
        query = query.where(models.Project.owner_id == user.id)
    return query


def get_project_for_user(db: Session, user: models.User, project_id: str) -> models.Project | None:
    return db.scalar(project_query_for_user(user, project_id))


def create_project(db: Session, user: models.User, payload: dict[str, Any]) -> models.Project:
    input_type = str(payload.get("input_type") or "images")
    if input_type not in {"images", "video", "camera"}:
        raise ValueError("input_type must be images, video, or camera")
    project = models.Project(
        owner_id=user.id,
        name=str(payload.get("name") or "Untitled reconstruction"),
        input_type=input_type,
        tags=[str(item) for item in (payload.get("tags") or [])],
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def list_projects(db: Session, user: models.User) -> list[models.Project]:
    query = select(models.Project).order_by(models.Project.updated_at.desc())
    if user.role != "admin":
        query = query.where(models.Project.owner_id == user.id)
    return list(db.scalars(query))


def project_summary(db: Session, user: models.User) -> dict[str, Any]:
    projects = list_projects(db, user)
    return {
        "project_count": len(projects),
        "training_count": sum(1 for item in projects if item.status in {"PREVIEW_RUNNING", "FINE_RUNNING", "FINE_QUEUED"}),
        "completed_count": sum(1 for item in projects if item.status in {"PREVIEW_READY", "COMPLETED"}),
        "failed_count": sum(1 for item in projects if item.status == "FAILED"),
        "total_size_bytes": sum(int(item.total_size_bytes or 0) for item in projects),
    }


def project_detail(db: Session, project: models.Project) -> dict[str, Any]:
    loaded = db.scalar(
        select(models.Project)
        .where(models.Project.id == project.id)
        .options(
            selectinload(models.Project.media_assets),
            selectinload(models.Project.tasks),
            selectinload(models.Project.artifacts),
        )
    )
    if loaded is None:
        raise ValueError("project not found")
    tasks = sorted(loaded.tasks, key=lambda item: item.created_at, reverse=True)
    artifacts = sorted(loaded.artifacts, key=lambda item: item.created_at, reverse=True)
    return {
        **project_to_dict(loaded),
        "media": [media_to_dict(item) for item in sorted(loaded.media_assets, key=lambda item: item.created_at)],
        "tasks": [task_to_dict(item) for item in tasks],
        "artifacts": [artifact_to_dict(item) for item in artifacts],
    }


def save_upload(
    db: Session,
    storage: ObjectStorage,
    user: models.User,
    project: models.Project,
    filename: str,
    content: bytes,
    content_type: str | None,
) -> models.MediaAsset:
    if not content:
        raise ValueError("uploaded file is empty")
    kind = detect_kind(filename, content_type)
    if project.input_type == "images" and kind != "image":
        raise ValueError("this project accepts image uploads")
    if project.input_type == "video" and kind != "video":
        raise ValueError("this project accepts a video upload")
    media_id = models.uuid_str()
    safe_name = safe_filename(filename)
    object_name = f"users/{user.id}/projects/{project.id}/raw/{media_id}_{safe_name}"
    object_uri = storage.put_bytes(object_name, content, content_type=content_type)
    asset = models.MediaAsset(
        id=media_id,
        project_id=project.id,
        kind=kind,
        object_uri=object_uri,
        file_name=safe_name,
        file_size=len(content),
    )
    project.total_size_bytes = int(project.total_size_bytes or 0) + len(content)
    project.status = "UPLOADING"
    if kind == "image" and not project.preview_image_uri:
        project.preview_image_uri = object_uri
    db.add(asset)
    db.commit()
    db.refresh(asset)
    return asset


def media_stats(db: Session, project: models.Project) -> dict[str, Any]:
    media = list(db.scalars(select(models.MediaAsset).where(models.MediaAsset.project_id == project.id)))
    return {
        "image_count": sum(1 for item in media if item.kind == "image"),
        "video_count": sum(1 for item in media if item.kind == "video"),
        "file_count": len(media),
        "total_size_bytes": sum(int(item.file_size or 0) for item in media),
    }


def validate_preview_inputs(project: models.Project, options: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = get_settings()
    min_frames = settings.preview_min_input_frames
    max_frames = settings.preview_max_input_frames
    requested_max = int((options or {}).get("max_preview_frames") or max_frames)
    if requested_max < min_frames:
        raise ValueError(f"max_preview_frames must be at least {min_frames}")
    if requested_max > max_frames:
        requested_max = max_frames

    media = list(project.media_assets)
    if project.input_type == "images":
        image_count = sum(1 for item in media if item.kind == "image")
        if image_count < min_frames:
            raise ValueError(f"preview requires at least {min_frames} uploaded images")
        return {
            "input_type": "images",
            "available_input_frames": image_count,
            "selected_input_frames": min(image_count, requested_max),
            "min_input_frames": min_frames,
            "max_input_frames": max_frames,
        }
    if project.input_type == "video":
        video_count = sum(1 for item in media if item.kind == "video")
        if video_count < 1:
            raise ValueError("preview requires an uploaded video")
        return {
            "input_type": "video",
            "available_input_frames": None,
            "selected_input_frames": requested_max,
            "min_input_frames": min_frames,
            "max_input_frames": max_frames,
        }
    raise ValueError("camera preview is not implemented")


def create_preview_task(db: Session, project: models.Project, options: dict[str, Any] | None = None) -> models.Task:
    options = dict(options or {})
    preview_inputs = validate_preview_inputs(project, options)
    options["max_preview_frames"] = preview_inputs["selected_input_frames"]
    options.setdefault("min_preview_frames", preview_inputs["min_input_frames"])
    options.setdefault("input_frame_policy", preview_inputs)
    task = models.Task(
        project_id=project.id,
        type="preview",
        status="queued",
        priority=100,
        progress=0,
        current_stage="queued",
        options=options,
    )
    project.status = "PREVIEW_RUNNING"
    project.error_message = None
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def validate_fine_inputs(project: models.Project) -> dict[str, Any]:
    media = list(project.media_assets)
    if project.input_type == "images":
        image_count = sum(1 for item in media if item.kind == "image")
        if image_count < 1:
            raise ValueError("fine reconstruction requires at least one uploaded image")
        return {"input_type": "images", "frame_count": image_count}
    if project.input_type == "video":
        video_count = sum(1 for item in media if item.kind == "video")
        if video_count < 1:
            raise ValueError("fine reconstruction requires an uploaded video")
        return {"input_type": "video", "frame_count": 0}
    raise ValueError("camera fine reconstruction is not implemented")


def create_fine_task(db: Session, project: models.Project, options: dict[str, Any] | None = None) -> models.Task:
    options = dict(options or {})
    options.setdefault("input_policy", validate_fine_inputs(project))
    task = models.Task(
        project_id=project.id,
        type="fine",
        status="queued",
        priority=80,
        progress=0,
        current_stage="queued",
        options=options,
    )
    project.status = "FINE_QUEUED"
    project.error_message = None
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def user_can_access_task(db: Session, user: models.User, task: models.Task) -> bool:
    if user.role == "admin":
        return True
    owner_id = db.scalar(select(models.Project.owner_id).where(models.Project.id == task.project_id))
    return owner_id == user.id


def list_artifacts(db: Session, project: models.Project) -> list[models.Artifact]:
    return list(
        db.scalars(
            select(models.Artifact)
            .where(models.Artifact.project_id == project.id)
            .order_by(models.Artifact.created_at.desc())
        )
    )


def latest_preview_artifact(db: Session, project: models.Project) -> models.Artifact | None:
    return db.scalar(
        select(models.Artifact)
        .where(models.Artifact.project_id == project.id, models.Artifact.kind == "preview_spz")
        .order_by(models.Artifact.created_at.desc())
        .limit(1)
    )


def create_feedback(db: Session, user: models.User, payload: dict[str, Any]) -> models.Feedback:
    project_id = payload.get("project_id")
    if project_id:
        project = get_project_for_user(db, user, str(project_id))
        if not project:
            raise ValueError("project not found")
    feedback = models.Feedback(
        user_id=user.id,
        project_id=str(project_id) if project_id else None,
        title=str(payload.get("title") or "Untitled feedback"),
        content=str(payload.get("content") or ""),
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


def all_tasks(db: Session) -> list[models.Task]:
    return list(db.scalars(select(models.Task).order_by(models.Task.created_at.desc())))


def worker_heartbeats(db: Session) -> list[models.WorkerHeartbeat]:
    return list(db.scalars(select(models.WorkerHeartbeat).order_by(models.WorkerHeartbeat.last_seen_at.desc())))


def delete_project(db: Session, project: models.Project) -> None:
    db.delete(project)
    db.commit()
