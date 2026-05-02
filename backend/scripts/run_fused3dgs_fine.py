from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    spec_path = Path(os.environ["GS_TASK_SPEC"])
    result_path = Path(os.environ["GS_STAGE_RESULT"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    work_dir = Path(spec["work_dir"])
    train_spec = work_dir / "fused3dgs_train_spec.json"
    train_result = work_dir / "fused3dgs_train_result.json"
    train_spec.parent.mkdir(parents=True, exist_ok=True)
    train_spec.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "fused3dgs.train",
        "--spec",
        str(train_spec),
        "--result",
        str(train_result),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "fused3dgs training command failed")
    if not train_result.exists():
        raise RuntimeError("fused3dgs training did not write a result JSON")

    result = json.loads(train_result.read_text(encoding="utf-8"))
    final_ply = find_artifact(result, "final_ply")
    if final_ply is None:
        raise RuntimeError("fused3dgs training did not report final_ply")
    final_path = Path(final_ply)
    if not final_path.exists() or final_path.stat().st_size <= 0:
        raise RuntimeError(f"fused3dgs final_ply is missing or empty: {final_path}")

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [
                {
                    "kind": "final_ply",
                    "path": str(final_path.resolve()),
                    "file_size": final_path.stat().st_size,
                }
            ],
            "metrics": {
                **dict(result.get("metrics") or {}),
                "fused3dgs_stdout": completed.stdout,
            },
        },
    )
    return 0


def find_artifact(result: dict, kind: str) -> str | None:
    artifacts = result.get("artifacts") or []
    if isinstance(artifacts, dict):
        value = artifacts.get(kind)
        return str(value) if value else None
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("kind") == kind and artifact.get("path"):
            return str(artifact["path"])
    return None


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1)
