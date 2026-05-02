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
    """Runs the real fine reconstruction chain through configured commands.

    The platform owns orchestration and validation only. It never fabricates
    model files; every required model artifact must come from a configured
    external algorithm command and must be non-empty.
    """

    ENTRYPOINT_ALGORITHM = "Faster-GS"
    ENTRYPOINT_COMMAND = "fine_engine"
    SPZ_ALGORITHM = "Spark-SPZ"
    SPZ_COMMAND = "compress_final"
    LOD_ALGORITHM = "RAD-LOD"
    LOD_COMMAND = "export_rad"

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
        options = self._normalized_options(request)

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

        if self._module_enabled(options, "fused3dgs", "use_vcd", True) and self._module_enabled(options, "vcd", "enabled", True):
            fastgs_stage = PipelineStage(
                name="vcd_densify_and_prune_hook",
                algorithm="FastGS",
                role="multi_view_consistency_densification",
                reason="FastGS VCD controls densification and pruning inside the Faster-GS host loop",
            )
            stages.append(fastgs_stage)
            requirements.append(
                AlgorithmRequirement(
                    name=fastgs_stage.algorithm,
                    stage=fastgs_stage.name,
                    role=fastgs_stage.role,
                )
            )
        else:
            skipped.append(SkippedStage(name="vcd_densify_and_prune_hook", reason="FastGS VCD is disabled in fine options"))

        use_deblur = self._should_use_deblur(request, options)
        if use_deblur:
            deblur_stage = PipelineStage(
                name="deblur_covariance_modulation_hook",
                algorithm="Deblurring-3DGS",
                role="training_only_covariance_mlp",
                reason="blur was detected or explicitly enabled; Deblurring MLP is active only during training",
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
                    name="deblur_covariance_modulation_hook",
                    reason="Deblurring MLP is disabled or no blurry input was detected",
                )
            )

        if self._module_enabled(options, "fused3dgs", "use_lm_optimizer", True) and self._module_enabled(options, "lm_optimizer", "enabled", True):
            lm_options = self._section(options, "lm_optimizer")
            lm_stage = PipelineStage(
                name="lm_interval_optimizer",
                algorithm="3DGS-LM",
                role="levenberg_marquardt_interval_step",
                reason=(
                    "LM steps run at fixed intervals after "
                    f"iteration {int(lm_options.get('start_iter') or 3000)}"
                ),
            )
            stages.append(lm_stage)
            requirements.append(
                AlgorithmRequirement(
                    name=lm_stage.algorithm,
                    stage=lm_stage.name,
                    role=lm_stage.role,
                )
            )
        else:
            skipped.append(SkippedStage(name="lm_interval_optimizer", reason="3DGS-LM optimizer is disabled in fine options"))

        requested = self._requested_outputs(options)
        if "final_web_spz" in requested:
            spz_stage = PipelineStage(
                name="final_spz_conversion",
                algorithm=self.SPZ_ALGORITHM,
                role="final_web_spz_conversion",
                reason="final viewer loads a real SPZ converted from final.ply",
            )
            stages.append(spz_stage)
            requirements.append(
                AlgorithmRequirement(
                    name=spz_stage.algorithm,
                    stage=spz_stage.name,
                    role=spz_stage.role,
                    requires_command=True,
                    command_key=self.SPZ_COMMAND,
                )
            )
        if any(item.startswith("lod_rad:") for item in requested):
            lod_stage = PipelineStage(
                name="rad_lod_export",
                algorithm=self.LOD_ALGORITHM,
                role="rad_lod_export",
                reason="LOD exporter generates RAD files with quantized Gaussian count targets",
            )
            stages.append(lod_stage)
            requirements.append(
                AlgorithmRequirement(
                    name=lod_stage.algorithm,
                    stage=lod_stage.name,
                    role=lod_stage.role,
                    requires_command=True,
                    command_key=self.LOD_COMMAND,
                )
            )

        engine_options = {
            **options,
            "input_type": request.input_type,
            "frame_count": request.frame_count,
            "effective_view_count": request.effective_view_count,
            "colmap_succeeded": request.colmap_succeeded,
            "blur_detected": request.blur_detected,
            "enable_long_video_global_optimization": request.enable_long_video_global_optimization,
            "request_mesh_export": request.request_mesh_export,
            "requested_outputs": requested,
            "hooks": {
                "densify_and_prune": "FastGS" if self._stage_present(stages, "FastGS") else None,
                "deblur_forward": "Deblurring-3DGS" if use_deblur else None,
                "optimizer": "3DGS-LM" if self._stage_present(stages, "3DGS-LM") else None,
                "sparse_initializer": "FreeSplatter" if sparse_or_pose_failed else None,
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
            return TaskExecutionResult(task_id=request.task_id, status="failed", errors=[issue.to_dict()])

        issues = self.validate_plan(plan)
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
        artifacts: list[dict[str, Any]] = []

        train_result, train_issue, train_logs = self._run_stage(
            request=request,
            plan=plan,
            stage_name="fine_training_main_process",
            algorithm=self.ENTRYPOINT_ALGORITHM,
            command_key=self.ENTRYPOINT_COMMAND,
            spec={
                "raw_uri": request.raw_uri,
                "output_dir": str(request.work_dir / "final"),
                "fine_options": plan.engine_options,
            },
        )
        logs.extend(train_logs)
        if train_issue:
            return self._failed_from_issue(request, plan, train_issue, logs)
        stage_results["fine_training_main_process"] = train_result or {}
        final_ply = self._artifact_path(stage_results["fine_training_main_process"], "final_ply")
        if final_ply is None:
            return self._failed(
                request,
                plan,
                AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                "Faster-GS fine stage did not report a final_ply artifact",
                logs=logs,
            )
        artifacts.append(self._artifact_payload("final_ply", final_ply, file_name="final.ply"))

        requested = set(plan.engine_options.get("requested_outputs") or [])
        if "final_web_spz" in requested:
            spz_path = request.work_dir / "final" / "final_web.spz"
            spz_result, spz_issue, spz_logs = self._run_stage(
                request=request,
                plan=plan,
                stage_name="final_spz_conversion",
                algorithm=self.SPZ_ALGORITHM,
                command_key=self.SPZ_COMMAND,
                spec={
                    "input_ply": final_ply,
                    "output_spz": str(spz_path),
                },
            )
            logs.extend(spz_logs)
            if spz_issue:
                return self._failed_from_issue(request, plan, spz_issue, logs)
            stage_results["final_spz_conversion"] = spz_result or {}
            final_spz = self._artifact_path(stage_results["final_spz_conversion"], "final_web_spz")
            if final_spz is None:
                return self._failed(
                    request,
                    plan,
                    AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                    "Spark-SPZ final conversion did not report a final_web_spz artifact",
                    logs=logs,
                )
            artifacts.append(self._artifact_payload("final_web_spz", final_spz, file_name="final_web.spz"))

        if any(item.startswith("lod_rad:") for item in requested):
            lod_dir = request.work_dir / "final" / "lod"
            lod_result, lod_issue, lod_logs = self._run_stage(
                request=request,
                plan=plan,
                stage_name="rad_lod_export",
                algorithm=self.LOD_ALGORITHM,
                command_key=self.LOD_COMMAND,
                spec={
                    "input_ply": final_ply,
                    "output_dir": str(lod_dir),
                    "lod_targets": plan.engine_options.get("lod_targets") or {},
                },
            )
            logs.extend(lod_logs)
            if lod_issue:
                return self._failed_from_issue(request, plan, lod_issue, logs)
            stage_results["rad_lod_export"] = lod_result or {}
            lod_artifacts = self._lod_artifacts(stage_results["rad_lod_export"])
            missing_lods = [
                int(item.split(":", 1)[1])
                for item in requested
                if item.startswith("lod_rad:")
                and int(item.split(":", 1)[1]) not in {lod for lod, _ in lod_artifacts}
            ]
            if missing_lods:
                return self._failed(
                    request,
                    plan,
                    AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                    f"RAD LOD export is missing LOD levels: {missing_lods}",
                    logs=logs,
                )
            for lod, artifact in sorted(lod_artifacts):
                payload = self._artifact_payload(
                    "lod_rad",
                    artifact["path"],
                    file_name=f"final_lod{lod}.rad",
                    metadata={
                        "lod": lod,
                        "target_gaussians": artifact.get("target_gaussians"),
                        "actual_gaussians": artifact.get("actual_gaussians"),
                    },
                )
                artifacts.append(payload)

        metrics = {
            "input_type": request.input_type,
            "stages": list(stage_results.keys()),
            "requested_outputs": sorted(requested),
            "stage_metrics": {
                stage: result.get("metrics") or {}
                for stage, result in stage_results.items()
            },
            "lod_targets": plan.engine_options.get("lod_targets") or {},
        }
        if "metrics_json" in requested:
            metrics_path = request.work_dir / "final" / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts.append(self._artifact_payload("metrics_json", str(metrics_path), file_name="metrics.json"))

        output_issue = self._validate_requested_outputs(requested, artifacts)
        if output_issue:
            return TaskExecutionResult(
                task_id=request.task_id,
                status="failed",
                errors=[output_issue.to_dict()],
                logs=logs,
                plan=plan.to_dict(),
            )

        return TaskExecutionResult(
            task_id=request.task_id,
            status="succeeded",
            artifacts=artifacts,
            artifact_paths=self._artifact_paths(artifacts),
            metrics=metrics,
            logs=logs,
            plan=plan.to_dict(),
        )

    def _run_stage(
        self,
        *,
        request: FineTaskRequest,
        plan: FineEnginePlan,
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

        stage_dir = request.work_dir / "fine_stage_specs"
        stage_dir.mkdir(parents=True, exist_ok=True)
        spec_path = stage_dir / f"{stage_name}.json"
        result_path = stage_dir / f"{stage_name}.result.json"
        payload = {
            "task_id": request.task_id,
            "project_id": request.project_id,
            "stage": stage_name,
            "algorithm": algorithm,
            "work_dir": str(request.work_dir),
            "output_prefix": request.output_prefix,
            "plan": plan.to_dict(),
            "algorithms": self._algorithm_context(),
            "options": request.options,
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
            log_entries.extend(self._runner_log_entries(algorithm, result["_runner"]))
        if issue and issue.details:
            log_entries.extend(self._runner_log_entries(algorithm, issue.details))
        return result, issue, log_entries

    def _runner_log_entries(self, algorithm: str, details: dict[str, Any]) -> list[str]:
        logs: list[str] = []
        if details.get("stdout_path") or details.get("stderr_path"):
            logs.append(
                f"{algorithm} log files:\n"
                f"stdout: {details.get('stdout_path') or '-'}\n"
                f"stderr: {details.get('stderr_path') or '-'}"
            )
        if details.get("stdout"):
            logs.append(f"{algorithm} stdout:\n{details['stdout']}")
        if details.get("stderr"):
            logs.append(f"{algorithm} stderr:\n{details['stderr']}")
        return logs

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

    def _lod_artifacts(self, result: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
        lods: list[tuple[int, dict[str, Any]]] = []
        artifacts = result.get("artifacts") or []
        if isinstance(artifacts, dict):
            artifacts = [
                {"kind": "lod_rad", "path": path, "lod": int(str(key).replace("lod", ""))}
                for key, path in artifacts.items()
                if str(key).startswith("lod")
            ]
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("kind") != "lod_rad":
                continue
            lod_value = artifact.get("lod")
            if lod_value is None:
                metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
                lod_value = metadata.get("lod")
            try:
                lod = int(lod_value)
            except (TypeError, ValueError):
                continue
            if artifact.get("path"):
                lods.append((lod, artifact))
        return lods

    def _artifact_payload(
        self,
        kind: str,
        path: str,
        *,
        file_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = Path(path)
        return {
            "kind": kind,
            "path": str(source),
            "file_name": file_name or source.name,
            "file_size": source.stat().st_size if source.exists() and source.is_file() else None,
            "metadata": metadata or {},
        }

    def _validate_requested_outputs(self, requested: set[str], artifacts: list[dict[str, Any]]) -> AlgorithmIssue | None:
        available = {str(artifact.get("kind")) for artifact in artifacts}
        lods = {
            int((artifact.get("metadata") or {}).get("lod"))
            for artifact in artifacts
            if artifact.get("kind") == "lod_rad" and (artifact.get("metadata") or {}).get("lod") is not None
        }
        missing: list[str] = []
        for output in requested:
            if output.startswith("lod_rad:"):
                lod = int(output.split(":", 1)[1])
                if lod not in lods:
                    missing.append(output)
            elif output not in available:
                missing.append(output)
        if missing:
            return AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                message=f"Fine reconstruction did not produce requested outputs: {', '.join(sorted(missing))}",
                stage="requested_output_validation",
                details={"missing": sorted(missing)},
            )
        return None

    def _artifact_paths(self, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
        paths: dict[str, Any] = {}
        lods: list[dict[str, Any]] = []
        for artifact in artifacts:
            kind = str(artifact.get("kind") or "")
            path = artifact.get("path")
            if not kind or not path:
                continue
            if kind == "lod_rad":
                metadata = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else {}
                if metadata.get("lod") is None:
                    continue
                lods.append(
                    {
                        "lod": metadata.get("lod"),
                        "path": str(path),
                        "target_gaussians": metadata.get("target_gaussians"),
                        "actual_gaussians": metadata.get("actual_gaussians"),
                    }
                )
            else:
                paths[kind] = str(path)
        if lods:
            paths["lod_rad"] = sorted(lods, key=lambda item: int(item["lod"]))
        return paths

    def _failed(
        self,
        request: FineTaskRequest,
        plan: FineEnginePlan,
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
        request: FineTaskRequest,
        plan: FineEnginePlan,
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

    def _normalized_options(self, request: FineTaskRequest) -> dict[str, Any]:
        options = dict(request.options or {})
        options.setdefault("fused3dgs", {})
        options.setdefault("deblurring", {})
        options.setdefault("lm_optimizer", {})
        options.setdefault("vcd", {})
        options.setdefault("outputs", {})
        options.setdefault("lod_targets", {"0": 1_000_000, "1": 500_000, "2": 200_000, "3": 50_000})
        self._section(options, "fused3dgs").setdefault("use_deblur", "auto")
        self._section(options, "fused3dgs").setdefault("use_vcd", True)
        self._section(options, "fused3dgs").setdefault("use_lm_optimizer", True)
        self._section(options, "deblurring").setdefault("start_iter", 0)
        self._section(options, "deblurring").setdefault("mlp_hidden", 64)
        self._section(options, "lm_optimizer").setdefault("enabled", True)
        self._section(options, "lm_optimizer").setdefault("start_iter", 3000)
        self._section(options, "lm_optimizer").setdefault("interval", 200)
        self._section(options, "lm_optimizer").setdefault("pcg_rtol", 0.05)
        self._section(options, "lm_optimizer").setdefault("pcg_max_iter", 8)
        self._section(options, "vcd").setdefault("enabled", True)
        self._section(options, "vcd").setdefault("loss_thresh", 0.1)
        self._section(options, "vcd").setdefault("grad_thresh", 0.0002)
        self._section(options, "outputs").setdefault("spz", True)
        self._section(options, "outputs").setdefault("lod", True)
        self._section(options, "outputs").setdefault("metrics", True)
        return options

    def _section(self, options: dict[str, Any], key: str) -> dict[str, Any]:
        section = options.get(key)
        if not isinstance(section, dict):
            section = {}
            options[key] = section
        return section

    def _module_enabled(self, options: dict[str, Any], section: str, key: str, default: bool) -> bool:
        value = self._section(options, section).get(key, default)
        return bool(value)

    def _should_use_deblur(self, request: FineTaskRequest, options: dict[str, Any]) -> bool:
        value = self._section(options, "fused3dgs").get("use_deblur", "auto")
        if isinstance(value, bool):
            return value
        if str(value).lower() == "auto":
            return request.blur_detected or bool(options.get("blur_detected"))
        return str(value).lower() in {"1", "true", "yes", "on"}

    def _requested_outputs(self, options: dict[str, Any]) -> list[str]:
        outputs = self._section(options, "outputs")
        requested = ["final_ply"]
        if bool(outputs.get("spz", True)):
            requested.append("final_web_spz")
        if bool(outputs.get("lod", True)):
            requested.extend([f"lod_rad:{lod}" for lod in range(4)])
        if bool(outputs.get("metrics", True)):
            requested.append("metrics_json")
        return requested

    def _stage_present(self, stages: list[PipelineStage], algorithm: str) -> bool:
        return any(stage.algorithm == algorithm for stage in stages)
