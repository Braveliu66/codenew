from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from fused3dgs.config import Fused3DGSConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fused3DGS fine reconstruction command wrapper.")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)

    spec_path = Path(args.spec)
    result_path = Path(args.result)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_path = str(spec.get("raw_uri") or spec.get("source_path") or "")
    model_path = str(spec.get("output_dir") or Path(spec.get("work_dir", ".")) / "final")
    config = Fused3DGSConfig.from_options(
        source_path=source_path,
        model_path=model_path,
        options=dict(spec.get("fine_options") or spec.get("options") or {}),
    )
    Path(model_path).mkdir(parents=True, exist_ok=True)

    command_template = os.environ.get("FUSED3DGS_TRAIN_COMMAND")
    if not command_template:
        raise RuntimeError("FUSED3DGS_TRAIN_COMMAND is not configured; refusing to simulate fine reconstruction")

    external_result = Path(model_path) / "fused3dgs_external_result.json"
    command = [
        part.format(
            spec=str(spec_path.resolve()),
            result=str(external_result.resolve()),
            source=str(Path(source_path).resolve() if source_path else source_path),
            model_path=str(Path(model_path).resolve()),
        )
        for part in shlex.split(command_template)
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "external Fused3DGS train command failed")

    payload = read_external_result(external_result)
    final_ply = find_artifact(payload, "final_ply") or str(Path(model_path) / "final.ply")
    final_path = Path(final_ply)
    if not final_path.exists() or final_path.stat().st_size <= 0:
        raise RuntimeError(f"external Fused3DGS train command did not produce a non-empty final.ply: {final_path}")

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [{"kind": "final_ply", "path": str(final_path.resolve())}],
            "metrics": {
                **dict(payload.get("metrics") or {}),
                "config": config.to_dict(),
                "stdout": completed.stdout,
            },
        },
    )
    return 0


def read_external_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_artifact(result: dict[str, Any], kind: str) -> str | None:
    artifacts = result.get("artifacts") or []
    if isinstance(artifacts, dict):
        value = artifacts.get(kind)
        return str(value) if value else None
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("kind") == kind and artifact.get("path"):
            return str(artifact["path"])
    return None


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1)
