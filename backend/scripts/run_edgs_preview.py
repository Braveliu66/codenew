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
    algorithm = spec["algorithms"]["EDGS"]
    repo_path = Path(algorithm["local_path"])
    train_py = repo_path / "train.py"
    source_path = Path(spec["source_path"])
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not repo_path.exists() or not train_py.exists():
        raise RuntimeError("EDGS official repository or train.py is missing")
    if not source_path.exists():
        raise RuntimeError(f"EDGS source dataset path is missing: {source_path}")

    python_bin = os.environ.get("EDGS_PYTHON") or sys.executable
    epochs = int(spec.get("edgs_epochs") or 3000)
    command = [
        python_bin,
        str(train_py),
        f"train.gs_epochs={epochs}",
        "train.no_densify=True",
        f"gs.dataset.source_path={source_path.resolve()}",
        f"gs.dataset.model_path={output_dir.resolve()}",
        "init_wC.use=True",
        "init_wC.matches_per_ref=15000",
        "init_wC.nns_per_ref=3",
        "init_wC.num_refs=180",
        "wandb.mode=disabled",
    ]
    completed = subprocess.run(command, cwd=str(repo_path), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "EDGS train.py failed")

    preview_ply = find_point_cloud(output_dir)
    if preview_ply is None:
        raise RuntimeError("EDGS did not produce a point cloud PLY")

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [
                {"kind": "model_dir", "path": str(output_dir.resolve())},
                {"kind": "preview_ply", "path": str(preview_ply.resolve())},
            ],
            "metrics": {"edgs_epochs": epochs},
        },
    )
    return 0


def find_point_cloud(output_dir: Path) -> Path | None:
    point_root = output_dir / "point_cloud"
    if point_root.exists():
        iterations = sorted(
            [path for path in point_root.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
            key=lambda path: int(path.name.split("_")[-1]) if path.name.split("_")[-1].isdigit() else -1,
        )
        for iteration in reversed(iterations):
            candidate = iteration / "point_cloud.ply"
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
    matches = sorted(output_dir.rglob("*.ply"))
    return matches[0] if matches else None


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1)

