from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmRequirement:
    name: str
    stage: str
    role: str
    requires_weights: bool = False
    requires_command: bool = False
    command_key: str | None = None


@dataclass(frozen=True)
class PipelineStage:
    name: str
    algorithm: str
    role: str
    reason: str
    requires_weights: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "role": self.role,
            "reason": self.reason,
            "requires_weights": self.requires_weights,
        }


@dataclass(frozen=True)
class SkippedStage:
    name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "reason": self.reason}


@dataclass(frozen=True)
class FineTaskRequest:
    task_id: str
    project_id: str
    input_type: str
    raw_uri: str
    work_dir: Path
    output_prefix: str
    frame_count: int = 0
    effective_view_count: int | None = None
    colmap_succeeded: bool = True
    blur_detected: bool = False
    enable_long_video_global_optimization: bool = False
    request_mesh_export: bool = False
    timeout_seconds: int = 7200
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "FineTaskRequest":
        missing = [
            key
            for key in ("task_id", "project_id", "input_type", "raw_uri", "work_dir", "output_prefix")
            if not data.get(key)
        ]
        if missing:
            raise ValueError(f"Missing required fine task fields: {', '.join(missing)}")

        return cls(
            task_id=str(data["task_id"]),
            project_id=str(data["project_id"]),
            input_type=str(data["input_type"]),
            raw_uri=str(data["raw_uri"]),
            work_dir=Path(str(data["work_dir"])),
            output_prefix=str(data["output_prefix"]),
            frame_count=int(data.get("frame_count") or 0),
            effective_view_count=(
                int(data["effective_view_count"])
                if data.get("effective_view_count") is not None
                else None
            ),
            colmap_succeeded=bool(data.get("colmap_succeeded", True)),
            blur_detected=bool(data.get("blur_detected", False)),
            enable_long_video_global_optimization=bool(
                data.get("enable_long_video_global_optimization", False)
            ),
            request_mesh_export=bool(data.get("request_mesh_export", False)),
            timeout_seconds=int(data.get("timeout_seconds") or 7200),
            options=dict(data.get("options") or {}),
        )


@dataclass(frozen=True)
class FineEnginePlan:
    task_id: str
    project_id: str
    entrypoint_algorithm: str
    command_key: str
    stages: list[PipelineStage]
    skipped_stages: list[SkippedStage]
    requirements: list[AlgorithmRequirement]
    engine_options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "entrypoint_algorithm": self.entrypoint_algorithm,
            "command_key": self.command_key,
            "stages": [stage.to_dict() for stage in self.stages],
            "skipped_stages": [stage.to_dict() for stage in self.skipped_stages],
            "requirements": [
                {
                    "name": requirement.name,
                    "stage": requirement.stage,
                    "role": requirement.role,
                    "requires_weights": requirement.requires_weights,
                    "requires_command": requirement.requires_command,
                    "command_key": requirement.command_key,
                }
                for requirement in self.requirements
            ],
            "engine_options": self.engine_options,
        }


@dataclass(frozen=True)
class TaskExecutionResult:
    task_id: str
    status: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "logs": self.logs,
            "errors": self.errors,
        }
        if self.plan is not None:
            payload["plan"] = self.plan
        return payload


@dataclass(frozen=True)
class PreviewTaskRequest:
    task_id: str
    project_id: str
    user_id: str
    input_type: str
    raw_uri: str
    work_dir: Path
    output_prefix: str
    timeout_seconds: int = 300
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PreviewTaskRequest":
        missing = [
            key
            for key in ("task_id", "project_id", "user_id", "input_type", "raw_uri", "work_dir", "output_prefix")
            if not data.get(key)
        ]
        if missing:
            raise ValueError(f"Missing required preview task fields: {', '.join(missing)}")
        input_type = str(data["input_type"])
        if input_type not in {"images", "video", "camera"}:
            raise ValueError("Preview input_type must be 'images', 'video', or 'camera'")
        return cls(
            task_id=str(data["task_id"]),
            project_id=str(data["project_id"]),
            user_id=str(data["user_id"]),
            input_type=input_type,
            raw_uri=str(data["raw_uri"]),
            work_dir=Path(str(data["work_dir"])),
            output_prefix=str(data["output_prefix"]),
            timeout_seconds=int(data.get("timeout_seconds") or 300),
            options=dict(data.get("options") or {}),
        )


@dataclass(frozen=True)
class PreviewPipelinePlan:
    task_id: str
    project_id: str
    stages: list[PipelineStage]
    skipped_stages: list[SkippedStage]
    requirements: list[AlgorithmRequirement]
    pipeline_options: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "project_id": self.project_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "skipped_stages": [stage.to_dict() for stage in self.skipped_stages],
            "requirements": [
                {
                    "name": requirement.name,
                    "stage": requirement.stage,
                    "role": requirement.role,
                    "requires_weights": requirement.requires_weights,
                    "requires_command": requirement.requires_command,
                    "command_key": requirement.command_key,
                }
                for requirement in self.requirements
            ],
            "pipeline_options": self.pipeline_options,
        }
