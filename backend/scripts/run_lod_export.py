from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_LOD_TARGETS = {"0": 1_000_000, "1": 500_000, "2": 200_000, "3": 50_000}


def main() -> int:
    spec_path = Path(os.environ["GS_TASK_SPEC"])
    result_path = Path(os.environ["GS_STAGE_RESULT"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    input_ply = Path(spec["input_ply"])
    output_dir = Path(spec["output_dir"])
    lod_targets = normalize_lod_targets(spec.get("lod_targets") or DEFAULT_LOD_TARGETS)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_ply.exists() or input_ply.stat().st_size <= 0:
        raise RuntimeError(f"final PLY is missing or empty: {input_ply}")

    command_template = os.environ.get("RAD_LOD_EXPORT_COMMAND")
    if not command_template:
        raise RuntimeError("RAD_LOD_EXPORT_COMMAND is not configured; refusing to create placeholder LOD files")

    targets_path = output_dir / "lod_targets.json"
    targets_path.write_text(json.dumps(lod_targets, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        part.format(
            input=str(input_ply.resolve()),
            output_dir=str(output_dir.resolve()),
            targets=str(targets_path.resolve()),
        )
        for part in shlex.split(command_template)
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "RAD LOD export command failed")

    artifacts = []
    for lod, target_count in sorted(lod_targets.items()):
        path = output_dir / f"final_lod{lod}.rad"
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"RAD LOD exporter did not produce a non-empty file for LOD{lod}: {path}")
        artifacts.append(
            {
                "kind": "lod_rad",
                "lod": lod,
                "path": str(path.resolve()),
                "file_size": path.stat().st_size,
                "target_gaussians": target_count,
                "actual_gaussians": read_actual_count(output_dir, lod),
            }
        )

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": artifacts,
            "metrics": {"lod_targets": lod_targets},
        },
    )
    return 0


def normalize_lod_targets(raw: object) -> dict[int, int]:
    if not isinstance(raw, dict):
        raw = DEFAULT_LOD_TARGETS
    targets: dict[int, int] = {}
    for key, value in raw.items():
        lod = int(key)
        if lod < 0 or lod > 3:
            continue
        count = int(value)
        if count <= 0:
            raise RuntimeError(f"LOD{lod} target_gaussians must be positive")
        targets[lod] = count
    missing = {0, 1, 2, 3} - set(targets)
    if missing:
        raise RuntimeError(f"LOD targets must include levels 0..3; missing {sorted(missing)}")
    return targets


def read_actual_count(output_dir: Path, lod: int) -> int | None:
    metrics_path = output_dir / f"final_lod{lod}.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("actual_gaussians")
    try:
        return int(value)
    except (TypeError, ValueError):
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
