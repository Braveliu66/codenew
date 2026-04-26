from __future__ import annotations

import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
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
from backend.app.services.task_queue import PreviewTaskQueue, TaskQueueError


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
    def resources(_: models.User = Depends(require_admin), db: Session = Depends(get_db)) -> dict[str, Any]:
        workers = worker_heartbeats(db)
        return {
            "cpu": {"available": True, "usage_percent": None},
            "gpu": {
                "available": bool(shutil.which("nvidia-smi")),
                "usage_percent": None,
                "memory_total": None,
                "memory_used": None,
                "message": None if shutil.which("nvidia-smi") else "nvidia-smi is not available in this runtime",
            },
            "workers": {"count": len(workers), "active_task_count": sum(1 for item in workers if item.current_task_id)},
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
            queue.enqueue_preview(task.id)
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
        preview = latest_preview_artifact(db, project)
        if not preview:
            return {"status": "unavailable", "message": "No real preview_spz artifact is available", "model_url": None}
        presigned = storage.presigned_url(preview.object_uri)
        token = create_artifact_token(preview.id)
        return {
            "status": "ready",
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
