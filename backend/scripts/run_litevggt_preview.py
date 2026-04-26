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
    algorithm = spec["algorithms"]["LiteVGGT"]
    repo_path = Path(algorithm["local_path"])
    weights = [Path(path) for path in algorithm.get("weight_paths") or []]
    checkpoint = weights[0] if weights else None
    image_dir = Path(spec["image_dir"])
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_demo = repo_path / "run_demo.py"
    if not repo_path.exists() or not run_demo.exists():
        raise RuntimeError("LiteVGGT official repository or run_demo.py is missing")
    if checkpoint is None or not checkpoint.exists():
        raise RuntimeError("LiteVGGT checkpoint te_dict.pt is missing")
    if not image_dir.exists() or not any(path.is_file() for path in image_dir.iterdir()):
        raise RuntimeError("LiteVGGT image directory has no input images")

    python_bin = os.environ.get("LITEVGGT_PYTHON") or sys.executable
    command = [
        python_bin,
        str(run_demo),
        "--ckpt_path",
        str(checkpoint),
        "--img_dir",
        str(image_dir.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
    ]
    completed = subprocess.run(command, cwd=str(repo_path), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "LiteVGGT run_demo.py failed")

    files = [path for path in output_dir.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError("LiteVGGT did not produce any output files")

    dataset_dir = find_dataset_dir(output_dir)
    point_cloud = first_existing(output_dir, ["*.ply", "*.pcd"])
    artifacts = [{"kind": "output_dir", "path": str(output_dir.resolve())}]
    if dataset_dir:
        artifacts.append({"kind": "dataset_dir", "path": str(dataset_dir.resolve())})
    if point_cloud:
        artifacts.append({"kind": "point_cloud", "path": str(point_cloud.resolve())})

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": artifacts,
            "metrics": {"output_file_count": len(files)},
        },
    )
    return 0


def find_dataset_dir(output_dir: Path) -> Path | None:
    candidates = [output_dir / "colmap", output_dir / "dataset", output_dir]
    for candidate in candidates:
        if (candidate / "sparse").exists() or (candidate / "images").exists():
            return candidate
    for candidate in output_dir.rglob("*"):
        if candidate.is_dir() and ((candidate / "sparse").exists() or (candidate / "images").exists()):
            return candidate
    return None


def first_existing(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
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

