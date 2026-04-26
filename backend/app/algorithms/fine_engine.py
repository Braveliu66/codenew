from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .environment import AlgorithmEnvironmentChecker
from .errors import AlgorithmErrorCode, AlgorithmIssue
from .models import (
    AlgorithmRequirement,
    FineEnginePlan,
    FineTaskRequest,
    PipelineStage,
    SkippedStage,
    TaskExecutionResult,
)
from .registry import AlgorithmRegistry
from .runner import RealAlgorithmCommandRunner


class FineSynthesisEngine:
    """Plans and invokes the one-process fine synthesis engine.

    The adapter only succeeds when a real Faster-GS entrypoint command is
    configured and returns verified artifacts. It never writes synthetic model
    outputs to make a task look successful.
    """

    ENTRYPOINT_ALGORITHM = "Faster-GS"
    ENTRYPOINT_COMMAND = "fine_engine"

    def __init__(
        self,
        registry: AlgorithmRegistry,
        runner: RealAlgorithmCommandRunner | None = None,
    ) -> None:
        self.registry = registry
        self.checker = AlgorithmEnvironmentChecker(registry)
        self.runner = runner or RealAlgorithmCommandRunner()

    def build_plan(self, request: FineTaskRequest) -> FineEnginePlan:
        stages: list[PipelineStage] = []
        skipped: list[SkippedStage] = []
        requirements: list[AlgorithmRequirement] = []

        long_video = request.input_type == "video" and request.frame_count > 500
        if long_video and request.enable_long_video_global_optimization:
            for algorithm in ("LingBot-Map", "MASt3R", "Pi3"):
                stage = PipelineStage(
                    name="long_video_global_optimization",
                    algorithm=algorithm,
                    role="pose_correction",
                    reason="video has more than 500 frames and global optimization is enabled",
                    requires_weights=algorithm in {"MASt3R", "Pi3"},
                )
                stages.append(stage)
                requirements.append(
                    AlgorithmRequirement(
                        name=algorithm,
                        stage=stage.name,
                        role=stage.role,
                        requires_weights=stage.requires_weights,
                    )
                )
        elif long_video:
            skipped.append(
                SkippedStage(
                    name="long_video_global_optimization",
                    reason="video has more than 500 frames, but optional global optimization is disabled",
                )
            )

        sparse_or_pose_failed = (
            request.effective_view_count is not None
            and request.effective_view_count < 15
        ) or not request.colmap_succeeded
        if sparse_or_pose_failed:
            stage = PipelineStage(
                name="sparse_or_pose_free_initialization",
                algorithm="FreeSplatter",
                role="initial_gaussian_generation",
                reason="effective views are below 15 or COLMAP failed",
                requires_weights=True,
            )
            stages.append(stage)
            requirements.append(
                AlgorithmRequirement(
                    name=stage.algorithm,
                    stage=stage.name,
                    role=stage.role,
                    requires_weights=True,
                )
            )

        main_stage = PipelineStage(
            name="fine_training_main_process",
            algorithm=self.ENTRYPOINT_ALGORITHM,
            role="one_process_engine",
            reason="Faster-GS is the required host framework for fine reconstruction",
        )
        stages.append(main_stage)
        requirements.append(
            AlgorithmRequirement(
                name=self.ENTRYPOINT_ALGORITHM,
                stage=main_stage.name,
                role=main_stage.role,
                requires_command=True,
                command_key=self.ENTRYPOINT_COMMAND,
            )
        )

        fastgs_stage = PipelineStage(
            name="densify_and_prune_hook",
            algorithm="FastGS",
            role="fast_dropgaussian_callback",
            reason="FastGS injects densify_and_prune acceleration into Faster-GS",
        )
        stages.append(fastgs_stage)
        requirements.append(
            AlgorithmRequirement(
                name=fastgs_stage.algorithm,
                stage=fastgs_stage.name,
                role=fastgs_stage.role,
            )
        )

        if request.blur_detected:
            deblur_stage = PipelineStage(
                name="deblur_forward_hook",
                algorithm="Deblurring-3DGS",
                role="learnable_blur_kernel",
                reason="preprocessing detected blurry input",
            )
            stages.append(deblur_stage)
            requirements.append(
                AlgorithmRequirement(
                    name=deblur_stage.algorithm,
                    stage=deblur_stage.name,
                    role=deblur_stage.role,
                )
            )
        else:
            skipped.append(
                SkippedStage(
                    name="deblur_forward_hook",
                    reason="preprocessing did not detect blurry input; inference path remains clean",
                )
            )

        lm_stage = PipelineStage(
            name="optimizer_switch_70_percent",
            algorithm="3DGS-LM",
            role="levenberg_marquardt_optimizer",
            reason="switch Adam to LM after 70 percent of iterations",
        )
        stages.append(lm_stage)
        requirements.append(
            AlgorithmRequirement(
                name=lm_stage.algorithm,
                stage=lm_stage.name,
                role=lm_stage.role,
            )
        )

        if request.request_mesh_export:
            mesh_stage = PipelineStage(
                name="same_cuda_context_mesh_export",
                algorithm="MeshSplatting",
                role="mesh_optimization_and_export",
                reason="mesh export requested after fine reconstruction",
            )
            stages.append(mesh_stage)
            requirements.append(
                AlgorithmRequirement(
                    name=mesh_stage.algorithm,
                    stage=mesh_stage.name,
                    role=mesh_stage.role,
                )
            )

        engine_options = {
            "input_type": request.input_type,
            "frame_count": request.frame_count,
            "effective_view_count": request.effective_view_count,
            "colmap_succeeded": request.colmap_succeeded,
            "blur_detected": request.blur_detected,
            "enable_long_video_global_optimization": (
                request.enable_long_video_global_optimization
            ),
            "request_mesh_export": request.request_mesh_export,
            "optimizer_switch_iteration_ratio": 0.7,
            "hooks": {
                "densify_and_prune": "FastGS",
                "deblur_forward": "Deblurring-3DGS" if request.blur_detected else None,
                "optimizer": "3DGS-LM",
                "sparse_initializer": "FreeSplatter" if sparse_or_pose_failed else None,
                "mesh_exporter": "MeshSplatting" if request.request_mesh_export else None,
            },
        }

        return FineEnginePlan(
            task_id=request.task_id,
            project_id=request.project_id,
            entrypoint_algorithm=self.ENTRYPOINT_ALGORITHM,
            command_key=self.ENTRYPOINT_COMMAND,
            stages=stages,
            skipped_stages=skipped,
            requirements=requirements,
            engine_options=engine_options,
        )

    def validate_plan(self, plan: FineEnginePlan) -> list[AlgorithmIssue]:
        return self.checker.check_many(plan.requirements)

    def execute(self, request: FineTaskRequest) -> TaskExecutionResult:
        try:
            plan = self.build_plan(request)
        except ValueError as exc:
            issue = AlgorithmIssue(
                code=AlgorithmErrorCode.INVALID_TASK_OPTIONS,
                message=str(exc),
            )
            return TaskExecutionResult(
                task_id=request.task_id,
                status="failed",
                errors=[issue.to_dict()],
            )

        issues = self.validate_plan(plan)
        if issues:
            return TaskExecutionResult(
                task_id=request.task_id,
                status="failed",
                errors=[issue.to_dict() for issue in issues],
                plan=plan.to_dict(),
            )

        entry = self.registry.get(self.ENTRYPOINT_ALGORITHM)
        if entry is None:
            issue = AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                message="Faster-GS entrypoint disappeared after validation",
                algorithm=self.ENTRYPOINT_ALGORITHM,
            )
            return TaskExecutionResult(
                task_id=request.task_id,
                status="failed",
                errors=[issue.to_dict()],
                plan=plan.to_dict(),
            )

        request.work_dir.mkdir(parents=True, exist_ok=True)
        spec_path = request.work_dir / f"{request.task_id}.fine_engine_spec.json"
        result_path = request.work_dir / f"{request.task_id}.fine_engine_result.json"
        spec_path.write_text(
            json.dumps(self._execution_spec(request, plan), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result, issue = self.runner.run(
            entry=entry,
            command_key=self.ENTRYPOINT_COMMAND,
            spec_path=spec_path,
            result_path=result_path,
            timeout_seconds=request.timeout_seconds,
        )
        if issue:
            return TaskExecutionResult(
                task_id=request.task_id,
                status="failed",
                errors=[issue.to_dict()],
                logs=[str(spec_path)],
                plan=plan.to_dict(),
            )

        return TaskExecutionResult(
            task_id=request.task_id,
            status="succeeded",
            artifacts=list(result.get("artifacts", [])) if result else [],
            metrics=dict(result.get("metrics", {})) if result else {},
            logs=[str(spec_path), str(result_path)],
            plan=plan.to_dict(),
        )

    def _execution_spec(
        self,
        request: FineTaskRequest,
        plan: FineEnginePlan,
    ) -> dict[str, Any]:
        algorithms: dict[str, dict[str, Any]] = {}
        for requirement in plan.requirements:
            entry = self.registry.get(requirement.name)
            if entry is None:
                continue
            algorithms[requirement.name] = {
                "repo_url": entry.repo_url,
                "license": entry.license,
                "commit_hash": entry.commit_hash,
                "local_path": str(entry.local_path) if entry.local_path else None,
                "weight_paths": [str(path) for path in entry.weight_paths],
                "role": requirement.role,
            }

        return {
            "task_id": request.task_id,
            "project_id": request.project_id,
            "raw_uri": request.raw_uri,
            "output_prefix": request.output_prefix,
            "work_dir": str(request.work_dir),
            "plan": plan.to_dict(),
            "algorithms": algorithms,
            "options": request.options,
            "expected_artifacts": [
                "final_ply",
                "final_web_spz",
                "metrics_json",
            ],
        }

