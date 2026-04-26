from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    spec_path = Path(os.environ["GS_TASK_SPEC"])
    result_path = Path(os.environ["GS_STAGE_RESULT"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    video_path = Path(spec["video_path"])
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg executable is not available")
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"video input is missing or empty: {video_path}")

    fps = max(int(spec.get("frame_sample_fps") or 2), 1)
    min_frames = max(int(spec.get("min_preview_frames") or 8), 1)
    max_frames = max(int(spec.get("max_preview_frames") or 800), min_frames)
    pattern = output_dir / "frame_%05d.jpg"
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        str(pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "ffmpeg frame extraction failed")

    frames = sorted(output_dir.glob("frame_*.jpg"))
    if len(frames) < min_frames:
        raise RuntimeError(f"ffmpeg produced {len(frames)} frames; at least {min_frames} are required")

    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [
                {"kind": "frame_dir", "path": str(output_dir.resolve())},
            ],
            "metrics": {"frame_count": len(frames), "frame_sample_fps": fps, "max_preview_frames": max_frames},
        },
    )
    return 0


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1)

