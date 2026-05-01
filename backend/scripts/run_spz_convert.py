from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    spec_path = Path(os.environ["GS_TASK_SPEC"])
    result_path = Path(os.environ["GS_STAGE_RESULT"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    algorithm = spec["algorithms"]["Spark-SPZ"]
    repo_path_value = algorithm.get("local_path")
    repo_path = Path(repo_path_value) if repo_path_value else None
    input_ply = Path(spec["input_ply"])
    output_spz = Path(spec["output_spz"])
    output_spz.parent.mkdir(parents=True, exist_ok=True)

    if not input_ply.exists() or input_ply.stat().st_size <= 0:
        raise RuntimeError(f"input PLY is missing or empty: {input_ply}")

    custom_command = os.environ.get("SPZ_CONVERTER_COMMAND")
    if custom_command:
        command = [
            part.format(input=str(input_ply.resolve()), output=str(output_spz.resolve()))
            for part in shlex.split(custom_command)
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    else:
        if repo_path is None or not repo_path.exists():
            raise RuntimeError("Spark/SPZ converter repository is not configured")
        converter_script = repo_path / "scripts" / "compress-to-spz.js"
        if converter_script.exists():
            if not shutil.which("node"):
                raise RuntimeError("node executable is not available for Spark/SPZ conversion")
            command = ["node", str(converter_script), str(input_ply.resolve())]
        else:
            command = ["npm", "run", "assets:compress", "--", str(input_ply.resolve())]
        completed = subprocess.run(command, cwd=str(repo_path), capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "SPZ conversion command failed")

    if not output_spz.exists():
        generated = find_newest_spz(repo_path or input_ply.parent, input_ply)
        if generated is None:
            raise RuntimeError("SPZ converter did not produce an SPZ file")
        shutil.copy2(generated, output_spz)
    if output_spz.stat().st_size <= 0:
        raise RuntimeError("SPZ converter produced an empty SPZ file")

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [
                {
                    "kind": "preview_spz",
                    "path": str(output_spz.resolve()),
                    "file_size": output_spz.stat().st_size,
                }
            ],
            "metrics": {"spz_size_bytes": output_spz.stat().st_size},
        },
    )
    return 0


def find_newest_spz(root: Path, input_ply: Path) -> Path | None:
    candidates = sorted(
        [path for path in root.rglob("*.spz") if path.is_file() and path.stat().st_size > 0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    sibling = input_ply.with_suffix(".spz")
    if sibling.exists() and sibling.stat().st_size > 0:
        return sibling
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

