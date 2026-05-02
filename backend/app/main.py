from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.algorithms.fine_engine import FineSynthesisEngine
from backend.app.algorithms.models import FineTaskRequest, PreviewTaskRequest
from backend.app.algorithms.preview_engine import PreviewEngine
from backend.app.core.config import get_settings
from backend.app.core.security import (
    create_access_token,
    create_artifact_token,
    decode_access_token,
    decode_artifact_token,
    hash_password,
    verify_password,
)
from backend.app.db import models
from backend.app.db.session import SessionLocal, get_db, init_database
from backend.app.services.object_storage import ObjectStorage, ObjectStorageError
from backend.app.services.project_store import (
    all_tasks,
    create_feedback,
    create_fine_task,
    create_preview_task,
    create_project,
    delete_project,
    get_project_for_user,
    latest_preview_artifact,
    list_artifacts,
    list_projects,
    media_stats,
    project_detail,
    project_summary,
    save_upload,
    user_can_access_task,
    worker_heartbeats,
)
from backend.app.services.registry_store import load_registry_from_db, registry_to_response
from backend.app.services.resource_monitor import (
    current_cpu_resources,
    current_gpu_resources,
    current_memory_resources,
    fresh_worker_heartbeats,
    gpu_resources_from_workers,
    parse_nvidia_smi_gpus,
)
from backend.app.services.runtime_preflight import build_runtime_preflight
from backend.app.services.seed import seed_database
from backend.app.services.serializers import (
    artifact_to_dict,
    feedback_to_dict,
    media_to_dict,
    project_to_dict,
    task_to_dict,
    user_to_dict,
    worker_to_dict,
)
from backend.app.services.task_queue import FineTaskQueue, PreviewTaskQueue, TaskQueueError


bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> models.User:
    if not credentials:
        raise HTTPException(status_code=401, detail="authentication required")
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="invalid token")
    user = db.get(models.User, str(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="user not found")
    return user


def require_admin(user: models.User = Depends(get_current_user)) -> models.User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user


def storage_dependency() -> ObjectStorage:
    return ObjectStorage()


def queue_dependency() -> PreviewTaskQueue:
    return PreviewTaskQueue()


def fine_queue_dependency() -> FineTaskQueue:
    return FineTaskQueue()


def current_cpu_percent() -> float | None:
    return current_cpu_resources()["usage_percent"]


def collect_task_logs(result: Any) -> list[str]:
    logs = list(getattr(result, "logs", []) or [])
    for error in getattr(result, "errors", []) or []:
        code = str(error.get("code") or "ERROR")
        message = str(error.get("message") or "")
        logs.append(f"[{code}] {message}")
        details = error.get("details") or {}
        stdout = str(details.get("stdout") or "").strip()
        stderr = str(details.get("stderr") or "").strip()
        stdout_path = str(details.get("stdout_path") or "").strip()
        stderr_path = str(details.get("stderr_path") or "").strip()
        if stdout_path or stderr_path:
            logs.append(f"log files:\nstdout: {stdout_path or '-'}\nstderr: {stderr_path or '-'}")
        if stdout:
            logs.append("stdout:\n" + stdout)
        if stderr:
            logs.append("stderr:\n" + stderr)
    return logs


def build_fine_request(task: models.Task, project: models.Project, storage: ObjectStorage) -> FineTaskRequest:
    settings = get_settings()
    work_dir = settings.storage_root / "work" / task.id
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    media = list(project.media_assets)
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
        frame_count = len(images)
    elif project.input_type == "video":
        videos = [item for item in media if item.kind == "video"]
        if not videos:
            raise ValueError("fine reconstruction requires an uploaded video")
        source = videos[0]
        suffix = Path(source.file_name).suffix or ".mp4"
        target = raw_dir / f"source{suffix}"
        storage.download_to_path(source.object_uri, target)
        raw_uri = str(target)
        frame_count = 0
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
        timeout_seconds=int((task.options or {}).get("timeout_seconds") or 7200),
        options=task.options or {},
    )


def process_fine_task_background(task_id: str) -> None:
    storage = ObjectStorage()
    with SessionLocal() as db:
        task = db.get(models.Task, task_id)
        if not task:
            return
        project = db.get(models.Project, task.project_id)
        if not project or task.status == "canceled":
            return
        task.status = "running"
        task.progress = 5
        task.current_stage = "materializing_inputs"
        task.started_at = task.started_at or models.utc_now()
        task.error_code = None
        task.error_message = None
        project.status = "FINE_RUNNING"
        project.error_message = None
        db.commit()
        try:
            request = build_fine_request(task, project, storage)
            task.current_stage = "fine_engine"
            task.progress = 15
            db.commit()
            result = FineSynthesisEngine(load_registry_from_db(db)).execute(request)
            if result.status != "succeeded":
                message = "; ".join(str(error.get("message") or "") for error in result.errors).strip() or "Fine reconstruction failed"
                first_error = result.errors[0] if result.errors else {}
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
                return
            for item in result.artifacts:
                source = Path(str(item.get("path") or ""))
                if not source.is_file():
                    continue
                file_name = str(item.get("file_name") or source.name)
                object_name = f"users/{project.owner_id}/projects/{project.id}/fine/{file_name}"
                object_uri = storage.put_file(object_name, source, content_type="application/octet-stream")
                db.add(models.Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind=str(item.get("kind") or "fine_artifact"),
                    object_uri=object_uri,
                    file_name=file_name,
                    file_size=source.stat().st_size,
                    checksum=item.get("checksum"),
                    artifact_metadata={"pipeline": "fine_engine"},
                ))
            task.status = "succeeded"
            task.progress = 100
            task.current_stage = "completed"
            task.metrics = result.to_dict()
            task.logs = collect_task_logs(result)
            task.finished_at = models.utc_now()
            project.status = "COMPLETED"
            project.error_message = None
            db.commit()
        except Exception as exc:
            task.status = "failed"
            task.progress = 100
            task.current_stage = "failed"
            task.error_code = "FINE_RECONSTRUCTION_FAILED"
            task.error_message = str(exc)
            task.logs = [str(exc)]
            task.finished_at = models.utc_now()
            project.status = "FAILED"
            project.error_message = str(exc)
            db.commit()


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        init_database()
        with SessionLocal() as db:
            seed_database(db)
        ObjectStorage().ensure_bucket()
        yield

    app = FastAPI(title="3DGS Reconstruction Backend", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin for origin in settings.cors_origins if origin],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/auth/register")
    def register(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        email = str(payload.get("email") or "").strip() or None
        if len(username) < 3:
            raise HTTPException(status_code=400, detail="username must be at least 3 characters")
        if len(password) < 6:
            raise HTTPException(status_code=400, detail="password must be at least 6 characters")
        if db.scalar(select(models.User).where(models.User.username == username)):
            raise HTTPException(status_code=409, detail="username already exists")
        user = models.User(username=username, email=email, password_hash=hash_password(password), role="user")
        db.add(user)
        db.commit()
        db.refresh(user)
        return {
            "access_token": create_access_token(user.id, user.role),
            "token_type": "bearer",
            "user": user_to_dict(user),
        }

    @app.post("/api/auth/login")
    def login(payload: dict[str, Any], db: Session = Depends(get_db)) -> dict[str, Any]:
        username = str(payload.get("username") or "").strip()
        password = str(payload.get("password") or "")
        user = db.scalar(select(models.User).where(models.User.username == username))
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=401, detail="invalid username or password")
        return {
            "access_token": create_access_token(user.id, user.role),
            "token_type": "bearer",
            "user": user_to_dict(user),
        }

    @app.post("/api/auth/logout")
    def logout() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/api/me")
    def me(user: models.User = Depends(get_current_user)) -> dict[str, Any]:
        return user_to_dict(user)

    @app.get("/api/admin/system/resources")
    def resources(_: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        workers = worker_heartbeats(db)
        fresh_workers = fresh_worker_heartbeats(workers)
        gpu = current_gpu_resources()
        if not gpu.get("available"):
            worker_gpu = gpu_resources_from_workers(workers)
            if worker_gpu.get("available"):
                gpu = worker_gpu
        return {
            "cpu": current_cpu_resources(),
            "memory": current_memory_resources(),
            "gpu": gpu,
            "workers": {"count": len(fresh_workers), "active_task_count": sum(1 for item in fresh_workers if item.current_task_id)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/api/algorithms")
    def public_algorithms(db: Session = Depends(get_db)) -> dict[str, Any]:
        return registry_to_response(db)

    @app.get("/api/admin/algorithms")
    def admin_algorithms(_: models.User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
        return registry_to_response(db)

    @app.get("/api/admin/workers")
    def admin_workers(_: models.User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
        workers = [worker_to_dict(item) for item in worker_heartbeats(db)]
        return {"workers": workers}

    @app.get("/api/admin/runtime/preflight")
    def admin_runtime_preflight(_: models.User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
        return build_runtime_preflight(load_registry_from_db(db))

    @app.get("/api/admin/tasks")
    def admin_tasks(_: models.User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
        return {"tasks": [task_to_dict(task) for task in all_tasks(db)]}

    @app.post("/api/projects")
    def create_project_endpoint(
        payload: dict[str, Any],
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            return project_to_dict(create_project(db, user, payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/projects/summary")
    def summary_endpoint(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        return project_summary(db, user)

    @app.get("/api/projects")
    def list_projects_endpoint(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict[str, Any]:
        return {"projects": [project_to_dict(project) for project in list_projects(db, user)]}

    @app.get("/api/projects/{project_id}")
    def get_project_endpoint(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return project_detail(db, project)

    @app.delete("/api/projects/{project_id}")
    def delete_project_endpoint(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, bool]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        delete_project(db, project)
        return {"deleted": True}

    @app.post("/api/projects/{project_id}/media")
    async def upload_media(
        project_id: str,
        file: UploadFile = File(...),
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
        storage: ObjectStorage = Depends(storage_dependency),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        content = await file.read()
        try:
            asset = save_upload(db, storage, user, project, file.filename or "upload.bin", content, file.content_type)
        except (ValueError, ObjectStorageError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return media_to_dict(asset)

    @app.get("/api/projects/{project_id}/media")
    def list_media_endpoint(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return {"media": [media_to_dict(item) for item in project.media_assets]}

    @app.get("/api/projects/{project_id}/media/stats")
    def media_stats_endpoint(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return media_stats(db, project)

    @app.post("/api/projects/{project_id}/tasks/preview")
    def create_preview_task_endpoint(
        project_id: str,
        payload: dict[str, Any] | None = None,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
        queue: PreviewTaskQueue = Depends(queue_dependency),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            task = create_preview_task(db, project, (payload or {}).get("options") or {})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            queue.enqueue_preview(task.id, input_type=project.input_type)
        except TaskQueueError as exc:
            task.status = "failed"
            task.progress = 100
            task.current_stage = "queue_unavailable"
            task.error_code = "QUEUE_UNAVAILABLE"
            task.error_message = str(exc)
            project.status = "FAILED"
            project.error_message = str(exc)
            db.commit()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return task_to_dict(task)

    @app.post("/api/camera/sessions")
    def create_camera_session(
        payload: dict[str, Any] | None = None,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        project = create_project(
            db,
            user,
            {
                "name": str((payload or {}).get("name") or "Realtime camera reconstruction"),
                "input_type": "camera",
                "tags": (payload or {}).get("tags") or ["camera", "lingbot-map"],
            },
        )
        return project_to_dict(project)

    @app.post("/api/projects/{project_id}/camera/chunks")
    async def upload_camera_chunk(
        project_id: str,
        file: UploadFile = File(...),
        segment_index: int = Query(default=0),
        segment_start_seconds: float = Query(default=0),
        segment_end_seconds: float | None = Query(default=None),
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
        storage: ObjectStorage = Depends(storage_dependency),
        queue: PreviewTaskQueue = Depends(queue_dependency),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        if project.input_type != "camera":
            raise HTTPException(status_code=400, detail="project is not a camera session")
        content = await file.read()
        try:
            asset = save_upload(
                db,
                storage,
                user,
                project,
                file.filename or f"camera-segment-{segment_index:04d}.webm",
                content,
                file.content_type or "video/webm",
            )
            task = create_preview_task(
                db,
                project,
                {
                    "media_asset_id": asset.id,
                    "preview_pipeline": "lingbot_map_spark",
                    "video_preview_mode": "streaming",
                    "progressive": True,
                    "segment_index": segment_index,
                    "segment_start_seconds": segment_start_seconds,
                    "segment_end_seconds": segment_end_seconds,
                    "max_preview_frames": get_settings().preview_max_input_frames,
                    "min_preview_frames": 1,
                },
            )
            queue.enqueue_preview(task.id, input_type="camera")
        except (ValueError, ObjectStorageError, TaskQueueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"media": media_to_dict(asset), "task": task_to_dict(task)}

    @app.post("/api/projects/{project_id}/camera/finish")
    def finish_camera_session(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        if project.input_type != "camera":
            raise HTTPException(status_code=400, detail="project is not a camera session")
        project.status = "PREVIEW_READY"
        project.updated_at = models.utc_now()
        db.commit()
        db.refresh(project)
        return project_to_dict(project)

    @app.post("/api/projects/{project_id}/tasks/fine")
    def create_fine_task_endpoint(
        project_id: str,
        payload: dict[str, Any] | None = None,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
        queue: FineTaskQueue = Depends(fine_queue_dependency),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        try:
            task = create_fine_task(db, project, (payload or {}).get("options") or {})
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            queue.enqueue_fine(task.id)
        except TaskQueueError as exc:
            task.status = "failed"
            task.progress = 100
            task.current_stage = "queue_unavailable"
            task.error_code = "QUEUE_UNAVAILABLE"
            task.error_message = str(exc)
            project.status = "FAILED"
            project.error_message = str(exc)
            db.commit()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return task_to_dict(task)

    @app.post("/api/tasks/preview/plan")
    def plan_preview_task(
        payload: dict[str, Any],
        _: models.User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            request = PreviewTaskRequest.from_mapping(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        engine = PreviewEngine(load_registry_from_db(db))
        plan = engine.build_plan(request)
        issues = engine.validate_plan(plan, request)
        return {"plan": plan.to_dict(), "environment_issues": [issue.to_dict() for issue in issues]}

    @app.post("/api/tasks/preview/run")
    def run_preview_direct(
        payload: dict[str, Any],
        _: models.User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            request = PreviewTaskRequest.from_mapping(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PreviewEngine(load_registry_from_db(db)).execute(request).to_dict()

    @app.get("/api/tasks/{task_id}")
    def get_task(
        task_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        task = db.get(models.Task, task_id)
        if not task or not user_can_access_task(db, user, task):
            raise HTTPException(status_code=404, detail="task not found")
        return task_to_dict(task)

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(
        task_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        task = db.get(models.Task, task_id)
        if not task or not user_can_access_task(db, user, task):
            raise HTTPException(status_code=404, detail="task not found")
        task.status = "canceled"
        task.current_stage = "canceled"
        task.finished_at = models.utc_now()
        db.commit()
        db.refresh(task)
        return task_to_dict(task)

    @app.get("/api/projects/{project_id}/artifacts")
    def list_artifacts_endpoint(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        return {"artifacts": [artifact_to_dict(item) for item in list_artifacts(db, project)]}

    @app.get("/api/projects/{project_id}/events")
    def project_events(
        project_id: str,
        token: str | None = Query(default=None),
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ):
        payload = decode_access_token(credentials.credentials) if credentials else (decode_access_token(token) if token else None)
        if not payload:
            raise HTTPException(status_code=401, detail="authentication required")
        user_id = str(payload["sub"])

        async def stream():
            seen_artifacts: set[str] = set()
            last_task_state: dict[str, tuple[str, int, str]] = {}
            for _ in range(1800):
                with SessionLocal() as event_db:
                    user = event_db.get(models.User, user_id)
                    if not user:
                        yield sse_event("error", {"message": "user not found"})
                        return
                    project = get_project_for_user(event_db, user, project_id)
                    if not project:
                        yield sse_event("error", {"message": "project not found"})
                        return
                    detail = project_detail(event_db, project)
                    yield sse_event("project_snapshot", detail)
                    for task in detail.get("tasks", []):
                        task_id = str(task["id"])
                        state = (str(task["status"]), int(task["progress"]), str(task.get("current_stage") or ""))
                        if last_task_state.get(task_id) != state:
                            last_task_state[task_id] = state
                            event_name = {
                                "succeeded": "task_succeeded",
                                "failed": "task_failed",
                                "canceled": "task_canceled",
                            }.get(state[0], "task_progress")
                            yield sse_event(event_name, task)
                    for artifact in detail.get("artifacts", []):
                        artifact_id = str(artifact["id"])
                        if artifact_id in seen_artifacts:
                            continue
                        seen_artifacts.add(artifact_id)
                        event_name = "preview_segment_ready" if artifact.get("kind") == "preview_spz_segment" else "artifact_created"
                        yield sse_event(event_name, artifact)
                await asyncio.sleep(1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/artifacts/{artifact_id}/download-url")
    def artifact_download_url(
        artifact_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
        storage: ObjectStorage = Depends(storage_dependency),
    ) -> dict[str, Any]:
        artifact = db.get(models.Artifact, artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="artifact not found")
        project = get_project_for_user(db, user, artifact.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="artifact not found")
        presigned = storage.presigned_url(artifact.object_uri)
        if presigned:
            return {"url": presigned, "expires_in_seconds": 3600}
        token = create_artifact_token(artifact.id)
        return {"url": f"/api/artifacts/{artifact.id}/file?token={token}", "expires_in_seconds": 3600}

    @app.get("/api/artifacts/{artifact_id}/file")
    def artifact_file(
        artifact_id: str,
        token: str | None = Query(default=None),
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
        db: Session = Depends(get_db),
        storage: ObjectStorage = Depends(storage_dependency),
    ):
        artifact = db.get(models.Artifact, artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail="artifact not found")
        authenticated_user: models.User | None = None
        if credentials:
            payload = decode_access_token(credentials.credentials)
            if payload:
                authenticated_user = db.get(models.User, str(payload["sub"]))
        token_artifact_id = decode_artifact_token(token) if token else None
        allowed = token_artifact_id == artifact.id
        if authenticated_user and get_project_for_user(db, authenticated_user, artifact.project_id):
            allowed = True
        if not allowed:
            raise HTTPException(status_code=401, detail="artifact token or authorization required")
        try:
            return storage.response_for_object(artifact.object_uri)
        except ObjectStorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/viewer-config")
    def viewer_config(
        project_id: str,
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
        storage: ObjectStorage = Depends(storage_dependency),
    ) -> dict[str, Any]:
        project = get_project_for_user(db, user, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="project not found")
        artifacts = list_artifacts(db, project)
        final = next((artifact for artifact in artifacts if artifact.kind == "final_web_spz"), None)
        if final:
            presigned = storage.presigned_url(final.object_uri)
            token = create_artifact_token(final.id)
            lod_payloads = []
            for lod_artifact in sorted(
                [artifact for artifact in artifacts if artifact.kind == "lod_rad"],
                key=lambda item: int((item.artifact_metadata or {}).get("lod") or 0),
            ):
                metadata = lod_artifact.artifact_metadata or {}
                lod_presigned = storage.presigned_url(lod_artifact.object_uri)
                lod_token = create_artifact_token(lod_artifact.id)
                lod_payloads.append(
                    {
                        "artifact_id": lod_artifact.id,
                        "model_url": lod_presigned or f"/api/artifacts/{lod_artifact.id}/file?token={lod_token}",
                        "format": "rad",
                        "lod": int(metadata.get("lod") or 0),
                        "target_gaussians": metadata.get("target_gaussians"),
                        "actual_gaussians": metadata.get("actual_gaussians"),
                        "file_size": lod_artifact.file_size,
                    }
                )
            return {
                "status": "ready",
                "mode": "single",
                "source": "final",
                "artifact_id": final.id,
                "model_url": presigned or f"/api/artifacts/{final.id}/file?token={token}",
                "format": "spz",
                "lods": lod_payloads,
            }
        segments = [
            artifact
            for artifact in artifacts
            if artifact.kind == "preview_spz_segment"
        ]
        if segments:
            segment_payloads = []
            for segment in sorted(
                segments,
                key=lambda item: int((item.artifact_metadata or {}).get("segment_index") or 0),
            ):
                presigned = storage.presigned_url(segment.object_uri)
                token = create_artifact_token(segment.id)
                metadata = segment.artifact_metadata or {}
                segment_payloads.append(
                    {
                        "artifact_id": segment.id,
                        "model_url": presigned or f"/api/artifacts/{segment.id}/file?token={token}",
                        "format": "spz",
                        "segment_index": int(metadata.get("segment_index") or 0),
                        "segment_start_seconds": metadata.get("segment_start_seconds"),
                        "segment_end_seconds": metadata.get("segment_end_seconds"),
                        "lod": metadata.get("lod", 0),
                        "estimated_splats": metadata.get("estimated_splats"),
                        "file_size": segment.file_size,
                    }
                )
            return {
                "status": "ready",
                "mode": "progressive",
                "source": "preview",
                "format": "spz",
                "segments": segment_payloads,
                "progressive": True,
            }
        preview = latest_preview_artifact(db, project)
        if not preview:
            return {"status": "unavailable", "message": "No real preview_spz artifact is available", "model_url": None}
        presigned = storage.presigned_url(preview.object_uri)
        token = create_artifact_token(preview.id)
        return {
            "status": "ready",
            "mode": "single",
            "source": "preview",
            "artifact_id": preview.id,
            "model_url": presigned or f"/api/artifacts/{preview.id}/file?token={token}",
            "format": "spz",
        }

    @app.post("/api/feedback")
    def feedback_endpoint(
        payload: dict[str, Any],
        user: models.User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            return feedback_to_dict(create_feedback(db, user, payload))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/tasks/fine/plan")
    def plan_fine_task(
        payload: dict[str, Any],
        _: models.User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            request = FineTaskRequest.from_mapping(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        engine = FineSynthesisEngine(load_registry_from_db(db))
        plan = engine.build_plan(request)
        issues = engine.validate_plan(plan)
        return {"plan": plan.to_dict(), "environment_issues": [issue.to_dict() for issue in issues]}

    @app.post("/api/tasks/fine/run")
    def run_fine_task(
        payload: dict[str, Any],
        _: models.User = Depends(require_admin),
        db: Session = Depends(get_db),
    ) -> dict[str, Any]:
        try:
            request = FineTaskRequest.from_mapping(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FineSynthesisEngine(load_registry_from_db(db)).execute(request).to_dict()

    return app


app = create_app()


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
