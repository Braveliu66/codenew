from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.db import models
from backend.app.db.session import SessionLocal, configure_database, init_database
from backend.app.main import create_app, queue_dependency
from backend.app.services.object_storage import ObjectStorage
from backend.app.services.project_store import create_preview_task, create_project, save_upload
from backend.app.services.seed import seed_database
from backend.workers.preview_worker import process_preview_task


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


class FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue_preview(self, task_id: str) -> None:
        self.enqueued.append(task_id)


def configure_test_database(name: str) -> Path:
    TEST_TMP_ROOT.mkdir(exist_ok=True)
    db_path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}.sqlite"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
    os.environ["THREE_DGS_STORAGE_ROOT"] = str(TEST_TMP_ROOT / f"storage-{uuid.uuid4().hex}")
    os.environ.pop("MINIO_ENDPOINT", None)
    configure_database(os.environ["DATABASE_URL"])
    init_database()
    with SessionLocal() as db:
        seed_database(db)
    return db_path


class EngineeringBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_test_database(self.id().split(".")[-1])

    def make_client(self, fake_queue: FakeQueue | None = None) -> TestClient:
        app = create_app()
        if fake_queue is not None:
            app.dependency_overrides[queue_dependency] = lambda: fake_queue
        return TestClient(app)

    def register(self, client: TestClient, username: str) -> str:
        response = client.post("/api/auth/register", json={"username": username, "password": "secret123"})
        self.assertEqual(response.status_code, 200, response.text)
        return str(response.json()["access_token"])

    def test_register_login_and_admin_permission(self) -> None:
        with self.make_client() as client:
            token = self.register(client, "alice")
            me = client.get("/api/me", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(me.status_code, 200)
            self.assertEqual(me.json()["username"], "alice")

            denied = client.get("/api/admin/tasks", headers={"Authorization": f"Bearer {token}"})
            self.assertEqual(denied.status_code, 403)

            login = client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
            self.assertEqual(login.status_code, 200, login.text)
            admin_token = login.json()["access_token"]
            allowed = client.get("/api/admin/tasks", headers={"Authorization": f"Bearer {admin_token}"})
            self.assertEqual(allowed.status_code, 200)

    def test_project_owner_isolation(self) -> None:
        with self.make_client() as client:
            alice = self.register(client, "alice")
            bob = self.register(client, "bob")
            created = client.post(
                "/api/projects",
                json={"name": "isolated", "input_type": "images", "tags": []},
                headers={"Authorization": f"Bearer {alice}"},
            )
            self.assertEqual(created.status_code, 200, created.text)
            project_id = created.json()["id"]

            forbidden = client.get(f"/api/projects/{project_id}", headers={"Authorization": f"Bearer {bob}"})
            self.assertEqual(forbidden.status_code, 404)

    def test_preview_task_is_queued_without_creating_artifact(self) -> None:
        fake_queue = FakeQueue()
        with self.make_client(fake_queue) as client:
            token = self.register(client, "previewer")
            project = client.post(
                "/api/projects",
                json={"name": "queued", "input_type": "images", "tags": ["test"]},
                headers={"Authorization": f"Bearer {token}"},
            ).json()
            upload = client.post(
                f"/api/projects/{project['id']}/media",
                files={"file": ("image.jpg", b"real upload bytes", "image/jpeg")},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(upload.status_code, 200, upload.text)

            task = client.post(
                f"/api/projects/{project['id']}/tasks/preview",
                json={"options": {}},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(task.status_code, 200, task.text)
            self.assertEqual(task.json()["status"], "queued")
            self.assertEqual(fake_queue.enqueued, [task.json()["id"]])

            artifacts = client.get(
                f"/api/projects/{project['id']}/artifacts",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(artifacts.status_code, 200)
            self.assertEqual(artifacts.json()["artifacts"], [])

    def test_worker_fails_unconfigured_preview_without_artifact(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="worker-user", password_hash="unused", role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "worker", "input_type": "images", "tags": []})
            storage = ObjectStorage()
            save_upload(db, storage, user, project, "image.jpg", b"real upload bytes", "image/jpeg")
            task = create_preview_task(db, project, {"skip_backend_cuda_check": True})

            processed = process_preview_task(db, task.id, worker_id="test-worker", storage=storage)
            self.assertIsNotNone(processed)
            self.assertEqual(processed.status, "failed")
            self.assertEqual(processed.error_code, "ALGORITHM_NOT_CONFIGURED")
            artifacts = list(db.scalars(select(models.Artifact).where(models.Artifact.task_id == task.id)))
            self.assertEqual(artifacts, [])


if __name__ == "__main__":
    unittest.main()
