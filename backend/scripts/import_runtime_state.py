from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from backend.app.core.security import hash_password
from backend.app.db import models
from backend.app.db.session import SessionLocal, init_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy storage/state.json into the SQL database.")
    parser.add_argument("--state", default="backend/storage/state.json", help="Read-only legacy JSON state path.")
    parser.add_argument("--user", default="legacy-user", help="Owner username for imported legacy projects.")
    args = parser.parse_args()
    state_path = Path(args.state)
    if not state_path.exists():
        raise SystemExit(f"state file not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    init_database()
    with SessionLocal() as db:
        user = db.scalar(select(models.User).where(models.User.username == args.user))
        if user is None:
            user = models.User(username=args.user, password_hash=hash_password("legacy-import-disabled"), role="user")
            db.add(user)
            db.flush()
        project_id_map: dict[str, str] = {}
        for item in state.get("projects", {}).values():
            if db.get(models.Project, item["id"]):
                project_id_map[item["id"]] = item["id"]
                continue
            project = models.Project(
                id=item["id"],
                owner_id=user.id,
                name=item.get("name") or "Legacy project",
                input_type=item.get("input_type") or "images",
                status=item.get("status") or "CREATED",
                tags=item.get("tags") or [],
                total_size_bytes=int(item.get("total_size_bytes") or 0),
                preview_image_uri=item.get("preview_image_uri"),
                error_message=item.get("error_message"),
            )
            db.add(project)
            project_id_map[item["id"]] = project.id
        for item in state.get("media", {}).values():
            if db.get(models.MediaAsset, item["id"]) or item.get("project_id") not in project_id_map:
                continue
            db.add(
                models.MediaAsset(
                    id=item["id"],
                    project_id=project_id_map[item["project_id"]],
                    kind=item.get("kind") or "image",
                    object_uri=item.get("object_uri") or item.get("path") or "",
                    file_name=item.get("file_name") or "legacy-upload",
                    file_size=int(item.get("file_size") or 0),
                    quality_flags=item.get("quality_flags") or {},
                )
            )
        for item in state.get("tasks", {}).values():
            if db.get(models.Task, item["id"]) or item.get("project_id") not in project_id_map:
                continue
            db.add(
                models.Task(
                    id=item["id"],
                    project_id=project_id_map[item["project_id"]],
                    type=item.get("type") or "preview",
                    status=item.get("status") or "queued",
                    progress=int(item.get("progress") or 0),
                    current_stage=item.get("current_stage"),
                    error_message=item.get("error_message"),
                    options=item.get("options") or {},
                    metrics=item.get("metrics") or {},
                    logs=item.get("logs") or [],
                )
            )
        for item in state.get("artifacts", {}).values():
            if db.get(models.Artifact, item["id"]) or item.get("project_id") not in project_id_map:
                continue
            task_id = item.get("task_id")
            if not task_id or db.get(models.Task, task_id) is None:
                continue
            db.add(
                models.Artifact(
                    id=item["id"],
                    project_id=project_id_map[item["project_id"]],
                    task_id=task_id,
                    kind=item.get("kind") or "artifact",
                    object_uri=item.get("object_uri") or item.get("path") or "",
                    file_name=item.get("file_name") or "legacy-artifact",
                    file_size=int(item.get("file_size") or 0),
                    artifact_metadata=item.get("metadata") or {},
                )
            )
        db.commit()


if __name__ == "__main__":
    main()
