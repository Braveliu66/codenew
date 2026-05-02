from __future__ import annotations

import os
import json
import unittest
import uuid
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.core.security import hash_password
from backend.app.db import models
from backend.app.db.session import SessionLocal, configure_database, init_database
from backend.app.algorithms.models import TaskExecutionResult
from backend.app.main import create_app, fine_queue_dependency, queue_dependency
from backend.app.services.object_storage import ObjectStorage
from backend.app.services.project_store import create_fine_task, create_preview_task, create_project, save_upload
from backend.app.services.registry_store import seed_algorithm_registry
from backend.app.services.seed import seed_database
from backend.workers.preview_worker import process_preview_task
from backend.workers.fine_worker import process_fine_task


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


class FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, str]] = []

    def enqueue_preview(self, task_id: str, input_type: str = "images") -> None:
        self.enqueued.append((task_id, input_type))


class FakeFineQueue:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue_fine(self, task_id: str) -> None:
        self.enqueued.append(task_id)


class FakeFineEngine:
    def __init__(self, artifacts: list[dict[str, object]] | None = None, status: str = "succeeded") -> None:
        self.artifacts = artifacts or []
        self.status = status

    def execute(self, request):
        artifacts = self.artifacts
        if self.status == "succeeded" and not artifacts:
            final_dir = request.work_dir / "final"
            lod_dir = final_dir / "lod"
            final_dir.mkdir(parents=True, exist_ok=True)
            lod_dir.mkdir(parents=True, exist_ok=True)
            final_ply = final_dir / "final.ply"
            final_spz = final_dir / "final_web.spz"
            metrics = final_dir / "metrics.json"
            final_ply.write_bytes(b"ply")
            final_spz.write_bytes(b"spz")
            metrics.write_text("{}", encoding="utf-8")
            artifacts = [
                {"kind": "final_ply", "path": str(final_ply), "file_name": "final.ply"},
                {"kind": "final_web_spz", "path": str(final_spz), "file_name": "final_web.spz"},
                {"kind": "metrics_json", "path": str(metrics), "file_name": "metrics.json"},
            ]
            for lod in range(4):
                path = lod_dir / f"final_lod{lod}.rad"
                path.write_bytes(f"rad-{lod}".encode("ascii"))
                artifacts.append(
                    {
                        "kind": "lod_rad",
                        "path": str(path),
                        "file_name": path.name,
                        "metadata": {"lod": lod, "target_gaussians": [1_000_000, 500_000, 200_000, 50_000][lod], "actual_gaussians": 10},
                    }
                )
        return TaskExecutionResult(
            task_id=request.task_id,
            status=self.status,
            artifacts=artifacts,
            metrics={"fake": True},
            errors=[] if self.status == "succeeded" else [{"code": "ALGORITHM_OUTPUT_INVALID", "message": "fake failure"}],
        )


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

    def make_client_with_fine_queue(self, fake_queue: FakeFineQueue) -> TestClient:
        app = create_app()
        app.dependency_overrides[fine_queue_dependency] = lambda: fake_queue
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
            for index in range(8):
                upload = client.post(
                    f"/api/projects/{project['id']}/media",
                    files={"file": (f"image-{index}.jpg", b"real upload bytes", "image/jpeg")},
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
            self.assertEqual(fake_queue.enqueued, [(task.json()["id"], "images")])

            artifacts = client.get(
                f"/api/projects/{project['id']}/artifacts",
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(artifacts.status_code, 200)
            self.assertEqual(artifacts.json()["artifacts"], [])

    def test_preview_task_allows_fewer_than_eight_images(self) -> None:
        fake_queue = FakeQueue()
        with self.make_client(fake_queue) as client:
            token = self.register(client, "few-images")
            project = client.post(
                "/api/projects",
                json={"name": "too few", "input_type": "images", "tags": []},
                headers={"Authorization": f"Bearer {token}"},
            ).json()
            for index in range(7):
                upload = client.post(
                    f"/api/projects/{project['id']}/media",
                    files={"file": (f"image-{index}.jpg", b"real upload bytes", "image/jpeg")},
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
            self.assertEqual(fake_queue.enqueued, [(task.json()["id"], "images")])

    def test_fine_task_is_queued_without_creating_artifact(self) -> None:
        fake_queue = FakeFineQueue()
        with self.make_client_with_fine_queue(fake_queue) as client:
            token = self.register(client, "fine-queue")
            project = client.post(
                "/api/projects",
                json={"name": "fine", "input_type": "images", "tags": []},
                headers={"Authorization": f"Bearer {token}"},
            ).json()
            upload = client.post(
                f"/api/projects/{project['id']}/media",
                files={"file": ("image.jpg", b"real upload bytes", "image/jpeg")},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(upload.status_code, 200, upload.text)

            task = client.post(
                f"/api/projects/{project['id']}/tasks/fine",
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
            self.assertEqual(artifacts.json()["artifacts"], [])

    def test_video_preview_task_uses_video_queue_and_lingbot_pipeline(self) -> None:
        fake_queue = FakeQueue()
        with self.make_client(fake_queue) as client:
            token = self.register(client, "video-preview")
            project = client.post(
                "/api/projects",
                json={"name": "video", "input_type": "video", "tags": []},
                headers={"Authorization": f"Bearer {token}"},
            ).json()
            upload = client.post(
                f"/api/projects/{project['id']}/media",
                files={"file": ("clip.mp4", b"real video bytes", "video/mp4")},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(upload.status_code, 200, upload.text)

            task = client.post(
                f"/api/projects/{project['id']}/tasks/preview",
                json={"options": {}},
                headers={"Authorization": f"Bearer {token}"},
            )

            self.assertEqual(task.status_code, 200, task.text)
            self.assertEqual(task.json()["options"]["preview_pipeline"], "lingbot_map_spark")
            self.assertEqual(fake_queue.enqueued, [(task.json()["id"], "video")])

    def test_camera_chunk_creates_segment_preview_task(self) -> None:
        fake_queue = FakeQueue()
        with self.make_client(fake_queue) as client:
            token = self.register(client, "camera-preview")
            session = client.post(
                "/api/camera/sessions",
                json={"name": "camera", "tags": ["test"]},
                headers={"Authorization": f"Bearer {token}"},
            )
            self.assertEqual(session.status_code, 200, session.text)
            self.assertEqual(session.json()["input_type"], "camera")

            chunk = client.post(
                f"/api/projects/{session.json()['id']}/camera/chunks"
                "?segment_index=2&segment_start_seconds=10&segment_end_seconds=15",
                files={"file": ("chunk.webm", b"real camera bytes", "video/webm")},
                headers={"Authorization": f"Bearer {token}"},
            )

            self.assertEqual(chunk.status_code, 200, chunk.text)
            task = chunk.json()["task"]
            self.assertEqual(task["options"]["preview_pipeline"], "lingbot_map_spark")
            self.assertTrue(task["options"]["progressive"])
            self.assertEqual(task["options"]["segment_index"], 2)
            self.assertEqual(fake_queue.enqueued, [(task["id"], "camera")])

    def test_viewer_config_returns_progressive_segments(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="segment-user", password_hash=hash_password("secret123"), role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "segments", "input_type": "camera", "tags": []})
            project_id = project.id
            task = models.Task(project_id=project.id, type="preview", status="succeeded", progress=100)
            db.add(task)
            db.commit()
            db.refresh(task)
            source = TEST_TMP_ROOT / f"segment-{uuid.uuid4().hex}.spz"
            source.write_bytes(b"spz")
            storage = ObjectStorage()
            object_uri = storage.put_file(f"users/{user.id}/projects/{project.id}/preview/preview_segment_0000.spz", source)
            db.add(
                models.Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind="preview_spz_segment",
                    object_uri=object_uri,
                    file_name="preview_segment_0000.spz",
                    file_size=source.stat().st_size,
                    artifact_metadata={
                        "segment_index": 0,
                        "segment_start_seconds": 0,
                        "segment_end_seconds": 5,
                        "progressive": True,
                    },
                )
            )
            db.commit()

        with self.make_client() as client:
            login = client.post("/api/auth/login", json={"username": "segment-user", "password": "secret123"})
            self.assertEqual(login.status_code, 200, login.text)
            response = client.get(
                f"/api/projects/{project_id}/viewer-config",
                headers={"Authorization": f"Bearer {login.json()['access_token']}"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["mode"], "progressive")
            self.assertEqual(payload["segments"][0]["segment_index"], 0)

    def test_viewer_config_prefers_final_over_preview(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="final-viewer", password_hash=hash_password("secret123"), role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "final", "input_type": "images", "tags": []})
            project_id = project.id
            task = models.Task(project_id=project.id, type="fine", status="succeeded", progress=100)
            db.add(task)
            db.commit()
            db.refresh(task)
            storage = ObjectStorage()
            preview_source = TEST_TMP_ROOT / f"preview-{uuid.uuid4().hex}.spz"
            preview_source.write_bytes(b"preview")
            final_source = TEST_TMP_ROOT / f"final-{uuid.uuid4().hex}.spz"
            final_source.write_bytes(b"final")
            lod_source = TEST_TMP_ROOT / f"final-lod-{uuid.uuid4().hex}.rad"
            lod_source.write_bytes(b"rad")
            db.add(
                models.Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind="preview_spz",
                    object_uri=storage.put_file(f"users/{user.id}/projects/{project.id}/preview/preview.spz", preview_source),
                    file_name="preview.spz",
                    file_size=preview_source.stat().st_size,
                )
            )
            db.add(
                models.Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind="final_web_spz",
                    object_uri=storage.put_file(f"users/{user.id}/projects/{project.id}/fine/final_web.spz", final_source),
                    file_name="final_web.spz",
                    file_size=final_source.stat().st_size,
                )
            )
            db.add(
                models.Artifact(
                    project_id=project.id,
                    task_id=task.id,
                    kind="lod_rad",
                    object_uri=storage.put_file(f"users/{user.id}/projects/{project.id}/fine/final_lod0.rad", lod_source),
                    file_name="final_lod0.rad",
                    file_size=lod_source.stat().st_size,
                    artifact_metadata={"lod": 0, "target_gaussians": 1000000, "actual_gaussians": 900000},
                )
            )
            db.commit()

        with self.make_client() as client:
            login = client.post("/api/auth/login", json={"username": "final-viewer", "password": "secret123"})
            self.assertEqual(login.status_code, 200, login.text)
            response = client.get(
                f"/api/projects/{project_id}/viewer-config",
                headers={"Authorization": f"Bearer {login.json()['access_token']}"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["source"], "final")
            self.assertEqual(payload["artifact_id"], next(item["id"] for item in client.get(
                f"/api/projects/{project_id}/artifacts",
                headers={"Authorization": f"Bearer {login.json()['access_token']}"},
            ).json()["artifacts"] if item["kind"] == "final_web_spz"))
            self.assertEqual(payload["lods"][0]["target_gaussians"], 1000000)

    def test_preview_task_caps_selected_image_frames_at_800(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="cap-user", password_hash="unused", role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "cap", "input_type": "images", "tags": []})
            for index in range(805):
                db.add(
                    models.MediaAsset(
                        project_id=project.id,
                        kind="image",
                        object_uri=f"file:///tmp/image-{index}.jpg",
                        file_name=f"image-{index}.jpg",
                        file_size=100,
                    )
                )
            db.commit()
            db.refresh(project)

            task = create_preview_task(db, project, {"max_preview_frames": 1200})

            self.assertEqual(task.options["max_preview_frames"], 800)
            self.assertEqual(task.options["input_frame_policy"]["available_input_frames"], 805)
            self.assertEqual(task.options["input_frame_policy"]["selected_input_frames"], 800)

    def test_algorithm_registry_seed_upserts_existing_records(self) -> None:
        registry_path = TEST_TMP_ROOT / f"registry-{uuid.uuid4().hex}.json"
        registry_path.write_text(
            json.dumps(
                {
                    "algorithms": [
                        {
                            "name": "LiteVGGT",
                            "repo_url": "old",
                            "license": "old",
                            "commit_hash": "old",
                            "enabled": False,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with SessionLocal() as db:
            seed_algorithm_registry(db, registry_path)
            registry_path.write_text(
                json.dumps(
                    {
                        "algorithms": [
                            {
                                "name": "LiteVGGT",
                                "repo_url": "new",
                                "license": "MIT",
                                "commit_hash": "new-commit",
                                "local_path": "/opt/three-dgs/repos/LiteVGGT-repo",
                                "enabled": True,
                                "commands": {"run_demo": ["python3", "run.py"]},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            seed_algorithm_registry(db, registry_path)

            record = db.scalar(select(models.AlgorithmRegistryRecord).where(models.AlgorithmRegistryRecord.name == "LiteVGGT"))
            self.assertIsNotNone(record)
            self.assertTrue(record.enabled)
            self.assertEqual(record.repo_url, "new")
            self.assertEqual(record.commit_hash, "new-commit")

    def test_worker_fails_unconfigured_preview_without_artifact(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="worker-user", password_hash="unused", role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "worker", "input_type": "images", "tags": []})
            storage = ObjectStorage()
            for index in range(8):
                save_upload(db, storage, user, project, f"image-{index}.jpg", b"real upload bytes", "image/jpeg")
            task = create_preview_task(db, project, {"skip_backend_cuda_check": True})

            processed = process_preview_task(db, task.id, worker_id="test-worker", storage=storage)
            self.assertIsNotNone(processed)
            self.assertEqual(processed.status, "failed")
            self.assertEqual(processed.error_code, "ALGORITHM_NOT_CONFIGURED")
            artifacts = list(db.scalars(select(models.Artifact).where(models.Artifact.task_id == task.id)))
            self.assertEqual(artifacts, [])

    def test_fine_worker_fails_unconfigured_algorithms_without_artifact(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="fine-worker-fail", password_hash="unused", role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "fine worker fail", "input_type": "images", "tags": []})
            storage = ObjectStorage()
            save_upload(db, storage, user, project, "image.jpg", b"real upload bytes", "image/jpeg")
            task = create_fine_task(db, project, {})

            processed = process_fine_task(db, task.id, worker_id="fine-test-worker", storage=storage)

            self.assertIsNotNone(processed)
            self.assertEqual(processed.status, "failed")
            self.assertEqual(processed.error_code, "ALGORITHM_NOT_CONFIGURED")
            artifacts = list(db.scalars(select(models.Artifact).where(models.Artifact.task_id == task.id)))
            self.assertEqual(artifacts, [])

    def test_fine_worker_persists_complete_final_artifacts(self) -> None:
        with SessionLocal() as db:
            user = models.User(username="fine-worker-ok", password_hash="unused", role="user")
            db.add(user)
            db.commit()
            db.refresh(user)
            project = create_project(db, user, {"name": "fine worker ok", "input_type": "images", "tags": []})
            storage = ObjectStorage()
            save_upload(db, storage, user, project, "image.jpg", b"real upload bytes", "image/jpeg")
            task = create_fine_task(db, project, {})

            processed = process_fine_task(
                db,
                task.id,
                worker_id="fine-test-worker",
                storage=storage,
                engine_factory=lambda _db: FakeFineEngine(),
            )

            self.assertIsNotNone(processed)
            self.assertEqual(processed.status, "succeeded")
            self.assertEqual(processed.project.status, "COMPLETED")
            artifacts = list(db.scalars(select(models.Artifact).where(models.Artifact.task_id == task.id)))
            self.assertEqual(
                sorted(item.kind for item in artifacts),
                ["final_ply", "final_web_spz", "lod_rad", "lod_rad", "lod_rad", "lod_rad", "metrics_json"],
            )


if __name__ == "__main__":
    unittest.main()
