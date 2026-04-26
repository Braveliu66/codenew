from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlgorithmErrorCode(str, Enum):
    ALGORITHM_NOT_CONFIGURED = "ALGORITHM_NOT_CONFIGURED"
    WEIGHTS_NOT_FOUND = "WEIGHTS_NOT_FOUND"
    GPU_RESOURCE_UNAVAILABLE = "GPU_RESOURCE_UNAVAILABLE"
    LICENSE_NOT_REGISTERED = "LICENSE_NOT_REGISTERED"
    ALGORITHM_SOURCE_MISMATCH = "ALGORITHM_SOURCE_MISMATCH"
    ALGORITHM_RUNNER_NOT_CONFIGURED = "ALGORITHM_RUNNER_NOT_CONFIGURED"
    ALGORITHM_COMMAND_FAILED = "ALGORITHM_COMMAND_FAILED"
    ALGORITHM_OUTPUT_INVALID = "ALGORITHM_OUTPUT_INVALID"
    INVALID_TASK_OPTIONS = "INVALID_TASK_OPTIONS"
    SPZ_CONVERTER_NOT_CONFIGURED = "SPZ_CONVERTER_NOT_CONFIGURED"
    VIDEO_FRAME_EXTRACTION_FAILED = "VIDEO_FRAME_EXTRACTION_FAILED"
    PREVIEW_ARTIFACT_INVALID = "PREVIEW_ARTIFACT_INVALID"


@dataclass(frozen=True)
class AlgorithmIssue:
    code: AlgorithmErrorCode
    message: str
    algorithm: str | None = None
    stage: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
        }
        if self.algorithm:
            payload["algorithm"] = self.algorithm
        if self.stage:
            payload["stage"] = self.stage
        if self.details:
            payload["details"] = self.details
        return payload
