from __future__ import annotations

import shutil
import time
from datetime import timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from fastapi.responses import FileResponse, RedirectResponse

from backend.app.core.config import Settings, get_settings


class ObjectStorageError(RuntimeError):
    pass


class ObjectStorage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.bucket = self.settings.minio_bucket
        self.local_root = self.settings.storage_root / "objects"
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def using_minio(self) -> bool:
        return bool(self.settings.minio_endpoint)

    def ensure_bucket(self) -> None:
        if not self.using_minio:
            self.local_root.mkdir(parents=True, exist_ok=True)
            return
        last_error: Exception | None = None
        for _ in range(5):
            try:
                client = self._minio_client()
                if not client.bucket_exists(self.bucket):
                    client.make_bucket(self.bucket)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(2)
        raise ObjectStorageError(f"MinIO bucket is unavailable: {last_error}") from last_error

    def put_bytes(self, object_name: str, content: bytes, content_type: str | None = None) -> str:
        if not content:
            raise ObjectStorageError("empty objects are not accepted")
        object_name = self._normalize_object_name(object_name)
        if self.using_minio:
            from io import BytesIO

            self.ensure_bucket()
            self._minio_client().put_object(
                self.bucket,
                object_name,
                BytesIO(content),
                length=len(content),
                content_type=content_type or "application/octet-stream",
            )
            return f"minio://{self.bucket}/{object_name}"
        target = self.local_root / object_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return target.as_uri()

    def put_file(self, object_name: str, source: Path, content_type: str | None = None) -> str:
        if not source.exists() or source.stat().st_size <= 0:
            raise ObjectStorageError(f"source artifact is missing or empty: {source}")
        object_name = self._normalize_object_name(object_name)
        if self.using_minio:
            self.ensure_bucket()
            self._minio_client().fput_object(
                self.bucket,
                object_name,
                str(source),
                content_type=content_type or "application/octet-stream",
            )
            return f"minio://{self.bucket}/{object_name}"
        target = self.local_root / object_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return target.as_uri()

    def download_to_path(self, object_uri: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        parsed = urlparse(object_uri)
        if parsed.scheme == "minio":
            bucket = parsed.netloc
            object_name = unquote(parsed.path.lstrip("/"))
            self._minio_client().fget_object(bucket, object_name, str(target))
            return
        if parsed.scheme == "file":
            source = self._path_from_file_uri(object_uri)
        else:
            source = Path(object_uri)
        if not source.exists() or source.stat().st_size <= 0:
            raise ObjectStorageError(f"stored object is missing or empty: {object_uri}")
        shutil.copy2(source, target)

    def presigned_url(self, object_uri: str, expires_seconds: int = 3600) -> str | None:
        parsed = urlparse(object_uri)
        if parsed.scheme != "minio":
            return None
        bucket = parsed.netloc
        object_name = unquote(parsed.path.lstrip("/"))
        return self._minio_client().presigned_get_object(
            bucket,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )

    def response_for_object(self, object_uri: str):
        url = self.presigned_url(object_uri)
        if url:
            return RedirectResponse(url=url)
        parsed = urlparse(object_uri)
        path = self._path_from_file_uri(object_uri) if parsed.scheme == "file" else Path(object_uri)
        if not path.exists() or path.stat().st_size <= 0:
            raise ObjectStorageError("artifact file not found")
        return FileResponse(path)

    def _minio_client(self):
        if self._client is not None:
            return self._client
        try:
            from minio import Minio
        except ModuleNotFoundError as exc:
            raise ObjectStorageError("MinIO client is not installed; install backend/requirements.txt") from exc
        self._client = Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
        )
        return self._client

    def _normalize_object_name(self, object_name: str) -> str:
        return object_name.replace("\\", "/").lstrip("/")

    def _path_from_file_uri(self, object_uri: str) -> Path:
        parsed = urlparse(object_uri)
        return Path(url2pathname(unquote(parsed.path)))
