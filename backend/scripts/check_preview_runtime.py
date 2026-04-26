from __future__ import annotations

import json
import sys

from backend.app.algorithms.registry import AlgorithmRegistry
from backend.app.core.config import get_settings
from backend.app.services.runtime_preflight import build_runtime_preflight


def main() -> int:
    settings = get_settings()
    registry = AlgorithmRegistry.from_json_file(settings.algorithm_registry_path)
    result = build_runtime_preflight(registry)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
