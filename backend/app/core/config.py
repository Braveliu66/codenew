from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    redis_url: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    minio_secure: bool
    jwt_secret: str
    jwt_algorithm: str
    access_token_minutes: int
    storage_root: Path
    algorithm_registry_path: Path
    cors_origins: list[str]
    preview_min_input_frames: int
    preview_max_input_frames: int
    preview_default_edgs_epochs: int
    preview_default_pipeline: str
    lingbot_model_path: Path
    video_preview_mode: str
    video_preview_target_frames: int | None


def get_settings() -> Settings:
    backend_root = Path(__file__).resolve().parents[2]
    default_storage = backend_root / "storage"
    default_registry = backend_root / "config" / "algorithm_registry.example.json"
    return Settings(
        database_url=os.environ.get("DATABASE_URL", f"sqlite:///{default_storage / 'app.db'}"),
        redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        minio_endpoint=os.environ.get("MINIO_ENDPOINT", ""),
        minio_access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
        minio_secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
        minio_bucket=os.environ.get("MINIO_BUCKET", "three-dgs"),
        minio_secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        jwt_secret=os.environ.get("JWT_SECRET", "dev-only-change-me"),
        jwt_algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
        access_token_minutes=int(os.environ.get("ACCESS_TOKEN_MINUTES", "1440")),
        storage_root=Path(os.environ.get("THREE_DGS_STORAGE_ROOT", str(default_storage))),
        algorithm_registry_path=Path(os.environ.get("ALGORITHM_REGISTRY_PATH", str(default_registry))),
        cors_origins=os.environ.get(
            "THREE_DGS_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
        ).split(","),
        preview_min_input_frames=int(os.environ.get("PREVIEW_MIN_INPUT_FRAMES", "8")),
        preview_max_input_frames=int(os.environ.get("PREVIEW_MAX_INPUT_FRAMES", "800")),
        preview_default_edgs_epochs=int(os.environ.get("PREVIEW_DEFAULT_EDGS_EPOCHS", "3000")),
        preview_default_pipeline=os.environ.get("PREVIEW_DEFAULT_PIPELINE", "edgs"),
        lingbot_model_path=Path(os.environ.get("LINGBOT_MODEL_PATH", "/model-cache/lingbot-map/lingbot-map-long.pt")),
        video_preview_mode=os.environ.get("VIDEO_PREVIEW_MODE", "windowed"),
        video_preview_target_frames=(
            int(os.environ["VIDEO_PREVIEW_TARGET_FRAMES"])
            if os.environ.get("VIDEO_PREVIEW_TARGET_FRAMES")
            else None
        ),
    )

