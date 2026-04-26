from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from backend.app.core.config import get_settings

from .environment import AlgorithmEnvironmentChecker
from .errors import AlgorithmErrorCode, AlgorithmIssue
from .models import (
    AlgorithmRequirement,
    PipelineStage,
    PreviewPipelinePlan,
    PreviewTaskRequest,
    SkippedStage,
    TaskExecutionResult,
)
from .registry import AlgorithmRegistry, AlgorithmRegistryEntry
from .runner import RealAlgorithmCommandRunner


class PreviewEngine:
    """Runs the real preview pipeline through configured external commands."""

    def __init__(
        self,
        registry: AlgorithmRegistry,
        runner: RealAlgorithmCommandRunner | None = None,
    ) -> None:
        self.registry = registry
        self.checker = AlgorithmEnvironmentChecker(registry)
        self.runner = runner or RealAlgorithmCommandRunner()

    def build_plan(self, request: PreviewTaskRequest) -> PreviewPipelinePlan:
        settings = get_settings()
        min_frames = settings.preview_min_input_frames
        max_frames = settings.preview_max_input_frames
        requested_max = int(request.options.get("max_preview_frames") or max_frames)
        requested_max = min(max(requested_max, min_frames), max_frames)
        stages: list[PipelineStage] = []
        skipped: list[SkippedStage] = []
        requirements: list[AlgorithmRequirement] = []

        if request.input_type == "video":
            stage = PipelineStage(
                name="video_frame_extraction",
                algorithm="FFmpeg",
                role="video_to_preview_frames",
                reason="video preview uses sampled frames before LiteVGGT and EDGS",
            )
            stages.append(stage)
            requirements.append(
                AlgorithmRequirement(
                    name=stage.algorithm,
                    stage=stage.name,
                    role=stage.role,
                    requires_command=True,
                    command_key="extract_frames",
                )
            )
        else:
            skipped.append(
                SkippedStage(
                    name="video_frame_extraction",
                    reason="image preview uses uploaded image files directly",
                )
            )

        for stage in (
            PipelineStage(
                name="geometry_litevggt",
                algorithm="LiteVGGT",
                role="camera_and_geometry_estimation",
                reason="LiteVGGT provides fast camera and point cloud initialization",
                requires_weights=True,
            ),
            PipelineStage(
                name="training_edgs",
                algorithm="EDGS",
                role="dense_gaussian_preview_training",
                reason="EDGS trains the preview Gaussian model from the LiteVGGT dataset",
            ),
            PipelineStage(
                name="spz_conversion",
                algorithm="Spark-SPZ",
                role="real_spz_conversion",
                reason="viewer must load a real SPZ artifact, not a zip placeholder",
            ),
        ):
            stages.append(stage)
            requirements.append(
                AlgorithmRequirement(
                    name=stage.algorithm,
                    stage=stage.name,
                    role=stage.role,
                    requires_weights=stage.requires_weights,
                    requires_command=True,
                    command_key={
                        "LiteVGGT": "run_demo",
                        "EDGS": "train",
                        "Spark-SPZ": "compress",
                    }[stage.algorithm],
                )
            )

        return PreviewPipelinePlan(
            task_id=request.task_id,
            project_id=request.project_id,
            stages=stages,
            skipped_stages=skipped,
            requirements=requirements,
            pipeline_options={
                "input_type": request.input_type,
                "frame_sample_fps": int(request.options.get("frame_sample_fps", 2)),
                "min_preview_frames": min_frames,
                "max_preview_frames": requested_max,
                "preview_frame_cap": max_frames,
                "edgs_epochs": int(request.options.get("edgs_epochs") or settings.preview_default_edgs_epochs),
                "require_real_spz": True,
            },
        )

    def validate_plan(self, plan: PreviewPipelinePlan, request: PreviewTaskRequest) -> list[AlgorithmIssue]:
        issues = [self._normalize_preview_issue(issue) for issue in self.checker.check_many(plan.requirements)]
        if not issues and not bool(request.options.get("skip_backend_cuda_check", False)):
            if not self._has_cuda_runtime_signal():
                issues.append(
                    AlgorithmIssue(
                        code=AlgorithmErrorCode.GPU_RESOURCE_UNAVAILABLE,
                        message="CUDA/NVIDIA runtime is not available to the backend; preview algorithms require GPU execution",
                        stage="environment",
                        details={"hint": "Run inside the Docker/WSL CUDA environment or set skip_backend_cuda_check only when external runners manage CUDA."},
                    )
                )
        return issues

    def _normalize_preview_issue(self, issue: AlgorithmIssue) -> AlgorithmIssue:
        if issue.algorithm == "Spark-SPZ" and issue.code == AlgorithmErrorCode.ALGORITHM_RUNNER_NOT_CONFIGURED:
            return AlgorithmIssue(
                code=AlgorithmErrorCode.SPZ_CONVERTER_NOT_CONFIGURED,
                message=issue.message,
                algorithm=issue.algorithm,
                stage=issue.stage,
                details=issue.details,
            )
        if issue.algorithm == "FFmpeg" and issue.code == AlgorithmErrorCode.ALGORITHM_RUNNER_NOT_CONFIGURED:
            return AlgorithmIssue(
                code=AlgorithmErrorCode.VIDEO_FRAME_EXTRACTION_FAILED,
                message=issue.message,
                algorithm=issue.algorithm,
                stage=issue.stage,
                details=issue.details,
            )
        return issue

    def execute(self, request: PreviewTaskRequest) -> TaskExecutionResult:
        try:
            plan = self.build_plan(request)
        except ValueError as exc:
            issue = AlgorithmIssue(
                code=AlgorithmErrorCode.INVALID_TASK_OPTIONS,
                message=str(exc),
            )
            return TaskExecutionResult(task_id=request.task_id, status="failed", errors=[issue.to_dict()])

        issues = self.validate_plan(plan, request)
        if issues:
            return TaskExecutionResult(
                task_id=request.task_id,
                status="failed",
                errors=[issue.to_dict() for issue in issues],
                plan=plan.to_dict(),
            )

        request.work_dir.mkdir(parents=True, exist_ok=True)
        logs: list[str] = []
        stage_results: dict[str, dict[str, Any]] = {}

        source_path = self._resolve_local_path(request.raw_uri)
        if not source_path.exists():
            return self._failed(
                request,
                plan,
                AlgorithmErrorCode.INVALID_TASK_OPTIONS,
                f"Preview input path does not exist: {source_path}",
            )

        image_dir = source_path
        if request.input_type == "video":
            frame_result = self._run_stage(
                request=request,
                plan=plan,
                stage_name="video_frame_extraction",
                algorithm="FFmpeg",
                command_key="extract_frames",
                spec={
                    "video_path": str(source_path),
                    "output_dir": str(request.work_dir / "frames"),
                    "frame_sample_fps": plan.pipeline_options["frame_sample_fps"],
                    "max_preview_frames": plan.pipeline_options["max_preview_frames"],
                },
            )
            if frame_result[1]:
                return self._failed_from_issue(request, plan, frame_result[1], logs)
            stage_results["video_frame_extraction"] = frame_result[0] or {}
            logs.extend(frame_result[2])
            image_dir_value = self._artifact_path(stage_results["video_frame_extraction"], "frame_dir")
            if image_dir_value is None:
                return self._failed(
                    request,
                    plan,
                    AlgorithmErrorCode.VIDEO_FRAME_EXTRACTION_FAILED,
                    "FFmpeg stage did not report a frame_dir artifact",
                    logs=logs,
                )
            image_dir = Path(image_dir_value)
            frame_count = int((stage_results["video_frame_extraction"].get("metrics") or {}).get("frame_count") or 0)
            if frame_count < int(plan.pipeline_options["min_preview_frames"]):
                return self._failed(
                    request,
                    plan,
                    AlgorithmErrorCode.INVALID_TASK_OPTIONS,
                    (
                        "Video preview requires at least "
                        f"{plan.pipeline_options['min_preview_frames']} extracted frames; got {frame_count}"
                    ),
                    logs=logs,
                )
        else:
            image_count = len([path for path in image_dir.iterdir() if path.is_file()])
            if image_count < int(plan.pipeline_options["min_preview_frames"]):
                return self._failed(
                    request,
                    plan,
                    AlgorithmErrorCode.INVALID_TASK_OPTIONS,
                    (
                        "Image preview requires at least "
                        f"{plan.pipeline_options['min_preview_frames']} images; got {image_count}"
                    ),
                    logs=logs,
                )

        litevggt_result = self._run_stage(
            request=request,
            plan=plan,
            stage_name="geometry_litevggt",
            algorithm="LiteVGGT",
            command_key="run_demo",
            spec={
                "image_dir": str(image_dir),
                "output_dir": str(request.work_dir / "litevggt"),
                "mode": "preview",
                "min_input_frames": plan.pipeline_options["min_preview_frames"],
                "max_input_frames": plan.pipeline_options["max_preview_frames"],
            },
        )
        if litevggt_result[1]:
            return self._failed_from_issue(request, plan, litevggt_result[1], logs)
        stage_results["geometry_litevggt"] = litevggt_result[0] or {}
        logs.extend(litevggt_result[2])

        dataset_dir = (
            self._artifact_path(stage_results["geometry_litevggt"], "dataset_dir")
            or self._artifact_path(stage_results["geometry_litevggt"], "colmap_dir")
            or self._artifact_path(stage_results["geometry_litevggt"], "output_dir")
        )
        if dataset_dir is None:
            return self._failed(
                request,
                plan,
                AlgorithmErrorCode.PREVIEW_ARTIFACT_INVALID,
                "LiteVGGT stage did not report a dataset_dir, colmap_dir, or output_dir artifact",
                logs=logs,
            )

        edgs_result = self._run_stage(
            request=request,
            plan=plan,
            stage_name="training_edgs",
            algorithm="EDGS",
            command_key="train",
            spec={
                "source_path": dataset_dir,
                "output_dir": str(request.work_dir / "edgs"),
                "edgs_epochs": plan.pipeline_options["edgs_epochs"],
            },
        )
        if edgs_result[1]:
            return self._failed_from_issue(request, plan, edgs_result[1], logs)
        stage_results["training_edgs"] = edgs_result[0] or {}
        logs.extend(edgs_result[2])

        preview_ply = (
            self._artifact_path(stage_results["training_edgs"], "preview_ply")
            or self._artifact_path(stage_results["training_edgs"], "trained_ply")
            or self._artifact_path(stage_results["training_edgs"], "point_cloud")
        )
        if preview_ply is None:
            return self._failed(
                request,
                plan,
                AlgorithmErrorCode.PREVIEW_ARTIFACT_INVALID,
                "EDGS stage did not report a preview_ply, trained_ply, or point_cloud artifact",
                logs=logs,
            )

        spz_path = request.work_dir / "preview" / "preview.spz"
        spz_result = self._run_stage(
            request=request,
            plan=plan,
            stage_name="spz_conversion",
            algorithm="Spark-SPZ",
            command_key="compress",
            spec={
                "input_ply": preview_ply,
                "output_spz": str(spz_path),
            },
        )
        if spz_result[1]:
            return self._failed_from_issue(request, plan, spz_result[1], logs)
        stage_results["spz_conversion"] = spz_result[0] or {}
        logs.extend(spz_result[2])

        preview_spz = self._artifact_path(stage_results["spz_conversion"], "preview_spz")
        if preview_spz is None:
            return self._failed(
                request,
                plan,
                AlgorithmErrorCode.PREVIEW_ARTIFACT_INVALID,
                "SPZ conversion stage did not report a preview_spz artifact",
                logs=logs,
            )

        return TaskExecutionResult(
            task_id=request.task_id,
            status="succeeded",
            artifacts=[
                {
                    "kind": "preview_spz",
                    "path": preview_spz,
                    "object_uri": f"{request.output_prefix.rstrip('/')}/preview.spz",
                    "file_size": Path(preview_spz).stat().st_size,
                }
            ],
            metrics={
                "pipeline": "litevggt_edgs_spz",
                "input_type": request.input_type,
                "stages": list(stage_results.keys()),
            },
            logs=logs,
            plan=plan.to_dict(),
        )

    def _run_stage(
        self,
        *,
        request: PreviewTaskRequest,
        plan: PreviewPipelinePlan,
        stage_name: str,
        algorithm: str,
        command_key: str,
        spec: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, AlgorithmIssue | None, list[str]]:
        entry = self.registry.get(algorithm)
        if entry is None:
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                message=f"{algorithm} is not configured",
                algorithm=algorithm,
                stage=stage_name,
            ), []

        stage_dir = request.work_dir / "stage_specs"
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec_path = stage_dir / f"{stage_name}.json"
        result_path = stage_dir / f"{stage_name}.result.json"
        payload = {
            "task_id": request.task_id,
            "project_id": request.project_id,
            "stage": stage_name,
            "algorithm": algorithm,
            "work_dir": str(request.work_dir),
            "plan": plan.to_dict(),
            "algorithms": self._algorithm_context(),
            **spec,
        }
        spec_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        result, issue = self.runner.run(
            entry=entry,
            command_key=command_key,
            spec_path=spec_path,
            result_path=result_path,
            timeout_seconds=request.timeout_seconds,
        )
        if issue:
            issue = AlgorithmIssue(
                code=issue.code,
                message=issue.message,
                algorithm=issue.algorithm,
                stage=stage_name,
                details=issue.details,
            )
        log_entries = [str(spec_path), str(result_path)]
        if result and isinstance(result.get("_runner"), dict):
            runner_info = result["_runner"]
            if runner_info.get("stdout_tail"):
                log_entries.append(f"{algorithm} stdout: {runner_info['stdout_tail']}")
            if runner_info.get("stderr_tail"):
                log_entries.append(f"{algorithm} stderr: {runner_info['stderr_tail']}")
        if issue and issue.details:
            if issue.details.get("stdout"):
                log_entries.append(f"{algorithm} stdout: {issue.details['stdout']}")
            if issue.details.get("stderr"):
                log_entries.append(f"{algorithm} stderr: {issue.details['stderr']}")
        return result, issue, log_entries

    def _algorithm_context(self) -> dict[str, dict[str, Any]]:
        return {
            entry.name: {
                "repo_url": entry.repo_url,
                "license": entry.license,
                "commit_hash": entry.commit_hash,
                "local_path": str(entry.local_path) if entry.local_path else None,
                "weight_paths": [str(path) for path in entry.weight_paths],
                "weight_source": entry.weight_source,
                "commands": entry.commands,
                "source_type": entry.source_type,
            }
            for entry in self.registry.list_entries()
        }

    def _artifact_path(self, result: dict[str, Any], kind: str) -> str | None:
        for artifact in result.get("artifacts", []):
            if artifact.get("kind") == kind and artifact.get("path"):
                return str(artifact["path"])
        artifacts = result.get("artifacts")
        if isinstance(artifacts, dict) and artifacts.get(kind):
            return str(artifacts[kind])
        return None

    def _failed(
        self,
        request: PreviewTaskRequest,
        plan: PreviewPipelinePlan,
        code: AlgorithmErrorCode,
        message: str,
        *,
        logs: list[str] | None = None,
    ) -> TaskExecutionResult:
        issue = AlgorithmIssue(code=code, message=message)
        return TaskExecutionResult(
            task_id=request.task_id,
            status="failed",
            errors=[issue.to_dict()],
            logs=logs or [],
            plan=plan.to_dict(),
        )

    def _failed_from_issue(
        self,
        request: PreviewTaskRequest,
        plan: PreviewPipelinePlan,
        issue: AlgorithmIssue,
        logs: list[str],
    ) -> TaskExecutionResult:
        return TaskExecutionResult(
            task_id=request.task_id,
            status="failed",
            errors=[issue.to_dict()],
            logs=logs,
            plan=plan.to_dict(),
        )

    def _resolve_local_path(self, uri: str) -> Path:
        if uri.startswith("file://"):
            return Path(uri[7:])
        return Path(uri)

    def _has_cuda_runtime_signal(self) -> bool:
        if shutil.which("nvidia-smi"):
            try:
                subprocess.run(
                    ["nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                return True
            except (subprocess.SubprocessError, OSError):
                pass
        try:
            import torch
        except ModuleNotFoundError:
            return False
        return bool(torch.cuda.is_available())
