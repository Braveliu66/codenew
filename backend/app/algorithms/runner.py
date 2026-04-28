from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .errors import AlgorithmErrorCode, AlgorithmIssue
from .registry import AlgorithmRegistryEntry


class RealAlgorithmCommandRunner:
    """Runs a configured algorithm command and validates its reported outputs."""

    def run(
        self,
        entry: AlgorithmRegistryEntry,
        command_key: str,
        spec_path: Path,
        result_path: Path,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any] | None, AlgorithmIssue | None]:
        command = entry.command(command_key)
        if not command:
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_RUNNER_NOT_CONFIGURED,
                message=f"{entry.name} command '{command_key}' is not configured",
                algorithm=entry.name,
            )

        env = os.environ.copy()
        env["GS_TASK_SPEC"] = str(spec_path)
        env["GS_STAGE_RESULT"] = str(result_path)

        try:
            completed = subprocess.run(
                command,
                cwd=str(entry.local_path) if entry.local_path else None,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = self._text_output(exc.stdout)
            stderr = self._text_output(exc.stderr)
            stdout_path, stderr_path = self._write_command_logs(result_path, stdout, stderr)
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_COMMAND_FAILED,
                message=f"{entry.name} command '{command_key}' timed out",
                algorithm=entry.name,
                details={
                    "timeout_seconds": timeout_seconds,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "stdout": stdout,
                    "stderr": stderr,
                },
            )
        except FileNotFoundError as exc:
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_COMMAND_FAILED,
                message=f"{entry.name} command executable was not found",
                algorithm=entry.name,
                details={"error": str(exc)},
            )

        stdout_path, stderr_path = self._write_command_logs(result_path, completed.stdout, completed.stderr)
        log_details = {
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if completed.returncode != 0:
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_COMMAND_FAILED,
                message=f"{entry.name} command '{command_key}' failed",
                algorithm=entry.name,
                details={
                    "returncode": completed.returncode,
                    **log_details,
                },
            )

        if not result_path.exists():
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                message=f"{entry.name} command did not write a result JSON",
                algorithm=entry.name,
                details={"result_path": str(result_path)},
            )

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return None, AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                message=f"{entry.name} result JSON is invalid",
                algorithm=entry.name,
                details={"error": str(exc), "result_path": str(result_path)},
            )

        result["_runner"] = {
            **log_details,
        }
        artifact_issue = self._validate_artifacts(entry.name, result)
        if artifact_issue:
            return None, AlgorithmIssue(
                code=artifact_issue.code,
                message=artifact_issue.message,
                algorithm=artifact_issue.algorithm,
                stage=artifact_issue.stage,
                details={**artifact_issue.details, **log_details},
            )
        return result, None

    def _write_command_logs(self, result_path: Path, stdout: str, stderr: str) -> tuple[Path, Path]:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = result_path.with_name(result_path.stem.replace(".result", "") + ".stdout.log")
        stderr_path = result_path.with_name(result_path.stem.replace(".result", "") + ".stderr.log")
        stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
        return stdout_path, stderr_path

    def _text_output(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    def _validate_artifacts(
        self,
        algorithm: str,
        result: dict[str, Any],
    ) -> AlgorithmIssue | None:
        if result.get("status") != "succeeded":
            return AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                message="Algorithm result did not report succeeded status",
                algorithm=algorithm,
                details={"status": result.get("status")},
            )
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return AlgorithmIssue(
                code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                message="Algorithm result did not report verifiable artifacts",
                algorithm=algorithm,
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                return AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                    message="Algorithm artifact entry is not an object",
                    algorithm=algorithm,
                )
            local_path = artifact.get("path")
            object_uri = artifact.get("object_uri")
            if local_path:
                path = Path(str(local_path))
                if not path.exists():
                    return AlgorithmIssue(
                        code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                        message="Algorithm reported a missing artifact",
                        algorithm=algorithm,
                        details={"path": str(path)},
                    )
                if path.is_file() and path.stat().st_size <= 0:
                    return AlgorithmIssue(
                        code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                        message="Algorithm reported an empty artifact",
                        algorithm=algorithm,
                        details={"path": str(path)},
                    )
                if path.is_dir() and not any(path.iterdir()):
                    return AlgorithmIssue(
                        code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                        message="Algorithm reported an empty artifact directory",
                        algorithm=algorithm,
                        details={"path": str(path)},
                    )
            elif object_uri and artifact.get("file_size", 0) > 0 and artifact.get("checksum"):
                continue
            else:
                return AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_OUTPUT_INVALID,
                    message=(
                        "Algorithm artifact must include a non-empty local path, or "
                        "an object URI with file_size and checksum from storage verification"
                    ),
                    algorithm=algorithm,
                    details={"artifact": artifact},
                )
        return None
