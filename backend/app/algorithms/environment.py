from __future__ import annotations

import subprocess
from pathlib import Path

from .errors import AlgorithmErrorCode, AlgorithmIssue
from .models import AlgorithmRequirement
from .registry import AlgorithmRegistry, AlgorithmRegistryEntry


class AlgorithmEnvironmentChecker:
    def __init__(self, registry: AlgorithmRegistry) -> None:
        self.registry = registry

    def check(self, requirement: AlgorithmRequirement) -> list[AlgorithmIssue]:
        entry = self.registry.get(requirement.name)
        if entry is None or not entry.enabled:
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                    message=f"{requirement.name} is not enabled in the algorithm registry",
                    algorithm=requirement.name,
                    stage=requirement.stage,
                )
            ]

        issues: list[AlgorithmIssue] = []
        issues.extend(self._check_compliance(entry, requirement))
        issues.extend(self._check_source(entry, requirement))
        if requirement.requires_weights:
            issues.extend(self._check_weights(entry, requirement))
        if requirement.requires_command and requirement.command_key:
            issues.extend(self._check_command(entry, requirement))
        return issues

    def check_many(self, requirements: list[AlgorithmRequirement]) -> list[AlgorithmIssue]:
        issues: list[AlgorithmIssue] = []
        seen: set[tuple[str, str, str | None]] = set()
        for requirement in requirements:
            key = (requirement.name, requirement.stage, requirement.command_key)
            if key in seen:
                continue
            seen.add(key)
            issues.extend(self.check(requirement))
        return issues

    def _check_compliance(
        self,
        entry: AlgorithmRegistryEntry,
        requirement: AlgorithmRequirement,
    ) -> list[AlgorithmIssue]:
        issues: list[AlgorithmIssue] = []
        if not entry.license:
            issues.append(
                AlgorithmIssue(
                    code=AlgorithmErrorCode.LICENSE_NOT_REGISTERED,
                    message=f"{entry.name} license is not registered",
                    algorithm=entry.name,
                    stage=requirement.stage,
                )
            )
        if not entry.repo_url:
            issues.append(
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                    message=f"{entry.name} repository URL is not registered",
                    algorithm=entry.name,
                    stage=requirement.stage,
                )
            )
        if not entry.commit_hash:
            issues.append(
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                    message=f"{entry.name} commit hash is not registered",
                    algorithm=entry.name,
                    stage=requirement.stage,
                )
            )
        return issues

    def _check_source(
        self,
        entry: AlgorithmRegistryEntry,
        requirement: AlgorithmRequirement,
    ) -> list[AlgorithmIssue]:
        if entry.source_type == "command":
            return []
        if entry.local_path is None:
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                    message=f"{entry.name} local source path is not configured",
                    algorithm=entry.name,
                    stage=requirement.stage,
                )
            ]
        if not entry.local_path.exists():
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED,
                    message=f"{entry.name} local source path does not exist",
                    algorithm=entry.name,
                    stage=requirement.stage,
                    details={"local_path": str(entry.local_path)},
                )
            ]
        git_dir = entry.local_path / ".git"
        if not git_dir.exists():
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_SOURCE_MISMATCH,
                    message=f"{entry.name} source path is not a git checkout; commit cannot be verified",
                    algorithm=entry.name,
                    stage=requirement.stage,
                    details={"local_path": str(entry.local_path)},
                )
            ]
        actual_commit = self._git_head(entry.local_path)
        if actual_commit is None:
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_SOURCE_MISMATCH,
                    message=f"{entry.name} git commit cannot be read",
                    algorithm=entry.name,
                    stage=requirement.stage,
                    details={"local_path": str(entry.local_path)},
                )
            ]
        if entry.commit_hash and actual_commit.lower() != entry.commit_hash.lower():
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_SOURCE_MISMATCH,
                    message=f"{entry.name} git commit does not match registry",
                    algorithm=entry.name,
                    stage=requirement.stage,
                    details={
                        "expected": entry.commit_hash,
                        "actual": actual_commit,
                    },
                )
            ]
        return []

    def _check_weights(
        self,
        entry: AlgorithmRegistryEntry,
        requirement: AlgorithmRequirement,
    ) -> list[AlgorithmIssue]:
        if not entry.weight_source:
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.WEIGHTS_NOT_FOUND,
                    message=f"{entry.name} weight source is not registered",
                    algorithm=entry.name,
                    stage=requirement.stage,
                )
            ]
        missing_paths = [
            str(path)
            for path in entry.weight_paths
            if not self._resolve_path(path, entry.local_path).exists()
        ]
        if not entry.weight_paths or missing_paths:
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.WEIGHTS_NOT_FOUND,
                    message=f"{entry.name} required weight files are not available",
                    algorithm=entry.name,
                    stage=requirement.stage,
                    details={"missing_paths": missing_paths},
                )
            ]
        return []

    def _check_command(
        self,
        entry: AlgorithmRegistryEntry,
        requirement: AlgorithmRequirement,
    ) -> list[AlgorithmIssue]:
        command = entry.command(requirement.command_key or "")
        if not command:
            return [
                AlgorithmIssue(
                    code=AlgorithmErrorCode.ALGORITHM_RUNNER_NOT_CONFIGURED,
                    message=(
                        f"{entry.name} command '{requirement.command_key}' is not configured; "
                        "the worker will not simulate algorithm execution"
                    ),
                    algorithm=entry.name,
                    stage=requirement.stage,
                )
            ]
        return []

    def _git_head(self, path: Path) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return None
        return completed.stdout.strip() or None

    def _resolve_path(self, path: Path, local_path: Path | None) -> Path:
        if path.is_absolute() or local_path is None:
            return path
        return local_path / path
