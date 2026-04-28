from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def main() -> int:
    spec_path = Path(os.environ["GS_TASK_SPEC"])
    result_path = Path(os.environ["GS_STAGE_RESULT"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    algorithm = spec["algorithms"]["LingBot-Map"]
    repo_path = Path(algorithm["local_path"])
    weights = [Path(path) for path in algorithm.get("weight_paths") or []]
    model_path = Path(os.environ.get("LINGBOT_MODEL_PATH") or (weights[0] if weights else ""))
    video_path = Path(spec["video_path"])
    output_dir = Path(spec["output_dir"])
    frame_dir = output_dir / "frames"
    ply_path = output_dir / "preview.ply"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not repo_path.exists():
        raise RuntimeError("LingBot-Map repository is missing")
    if not model_path.exists() or model_path.stat().st_size <= 0:
        raise RuntimeError(f"LingBot-Map model weight is missing or empty: {model_path}")
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise RuntimeError(f"video input is missing or empty: {video_path}")

    extraction = extract_video_frames(
        video_path=video_path,
        output_dir=frame_dir,
        target_frame_count=positive_int(spec.get("target_frame_count")),
        frame_sample_fps=positive_float(spec.get("frame_sample_fps")),
        min_frames=max(int(spec.get("min_preview_frames") or 1), 1),
        max_frames=max(int(spec.get("max_preview_frames") or 800), 1),
    )
    result = run_lingbot(
        repo_path=repo_path,
        model_path=model_path,
        image_dir=frame_dir,
        output_ply=ply_path,
        mode=str(spec.get("video_preview_mode") or os.environ.get("VIDEO_PREVIEW_MODE") or "windowed"),
        mask_sky=bool(spec.get("mask_sky", False)),
        max_points=positive_int(spec.get("max_preview_points")) or 300000,
    )
    metrics = {**extraction, **result}
    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [
                {"kind": "frame_dir", "path": str(frame_dir.resolve())},
                {"kind": "preview_ply", "path": str(ply_path.resolve())},
                {"kind": "point_cloud", "path": str(ply_path.resolve())},
            ],
            "metrics": metrics,
        },
    )
    return 0


def extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    target_frame_count: int | None,
    frame_sample_fps: float | None,
    min_frames: int,
    max_frames: int,
) -> dict[str, Any]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required for LingBot-Map video frame extraction") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video input: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        raise RuntimeError("video frame count cannot be read")
    duration = (total_frames / source_fps) if source_fps > 0 else None

    if target_frame_count is not None:
        selected_count = target_frame_count
    elif frame_sample_fps is not None and duration is not None:
        selected_count = int(duration * frame_sample_fps) + 1
    else:
        selected_count = total_frames
    selected_count = min(max(selected_count, min_frames), max_frames, total_frames)
    if selected_count < min_frames:
        raise RuntimeError(f"LingBot-Map video preview requires at least {min_frames} sampled frames; got {selected_count}")

    indexes = sorted({int(round(value)) for value in np.linspace(0, total_frames - 1, selected_count)})
    written = 0
    for index in indexes:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        target = output_dir / f"frame_{written:05d}.jpg"
        if not cv2.imwrite(str(target), frame):
            raise RuntimeError(f"failed to write sampled frame: {target}")
        written += 1
    capture.release()
    if written < min_frames:
        raise RuntimeError(f"LingBot-Map video preview sampled {written} usable frames; at least {min_frames} are required")
    return {
        "source_frame_count": total_frames,
        "source_fps": source_fps,
        "source_duration_seconds": duration,
        "selected_frame_count": written,
        "sampled_first_frame": indexes[0],
        "sampled_last_frame": indexes[-1],
    }


def run_lingbot(
    *,
    repo_path: Path,
    model_path: Path,
    image_dir: Path,
    output_ply: Path,
    mode: str,
    mask_sky: bool,
    max_points: int,
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for LingBot-Map") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for LingBot-Map")

    sys.path.insert(0, str(repo_path))
    try:
        from demo import load_images, load_model, postprocess
    except Exception as exc:
        raise RuntimeError(f"cannot import LingBot-Map demo helpers: {exc}") from exc

    device = torch.device(os.environ.get("LINGBOT_DEVICE", "cuda:0"))
    args = SimpleNamespace(
        image_folder=str(image_dir),
        model_path=str(model_path),
        image_size=int(os.environ.get("LINGBOT_IMAGE_SIZE", "518")),
        patch_size=int(os.environ.get("LINGBOT_PATCH_SIZE", "14")),
        mode=mode,
        enable_3d_rope=True,
        use_sdpa=parse_bool(os.environ.get("LINGBOT_USE_SDPA"), default=True),
        mask_sky=mask_sky,
        camera_num_iterations=int(os.environ.get("LINGBOT_CAMERA_NUM_ITERATIONS", "4")),
        max_frame_num=positive_int(os.environ.get("LINGBOT_MAX_FRAME_NUM")) or 4096,
        kv_cache_sliding_window=positive_int(os.environ.get("LINGBOT_KV_CACHE_SLIDING_WINDOW")) or 64,
        window_size=positive_int(os.environ.get("LINGBOT_WINDOW_SIZE")) or 32,
        overlap_size=positive_int(os.environ.get("LINGBOT_OVERLAP_SIZE")) or 8,
        num_scale_frames=positive_int(os.environ.get("LINGBOT_NUM_SCALE_FRAMES")) or 8,
        keyframe_interval=positive_int(os.environ.get("LINGBOT_KEYFRAME_INTERVAL")),
        offload_to_cpu=parse_bool(os.environ.get("LINGBOT_OFFLOAD_TO_CPU"), default=True),
        conf_threshold=float(os.environ.get("LINGBOT_CONF_THRESHOLD", "1.5")),
    )

    model = load_model(args, device)
    images, _paths, _resolved_image_folder = load_images(
        image_folder=args.image_folder,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)
    images = images.to(device)
    print(
        json.dumps(
            {
                "event": "lingbot_cuda",
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device.index or 0),
                "input_frames": int(images.shape[0]),
                "mode": args.mode,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            keyframe_interval = args.keyframe_interval or (1 if images.shape[0] <= 320 else max(1, images.shape[0] // 320))
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )
        else:
            keyframe_interval = args.keyframe_interval or 1
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=keyframe_interval,
                output_device=torch.device("cpu"),
            )
    images_for_post = predictions.get("images", images)
    predictions, images_cpu = postprocess(predictions, images_for_post)
    point_count = write_lingbot_point_cloud(
        output_ply,
        predictions=predictions,
        images=images_cpu,
        max_points=max_points,
        confidence_threshold=args.conf_threshold,
    )
    return {
        "point_count": point_count,
        "keyframe_interval": keyframe_interval,
        "window_size": None if args.mode == "streaming" else args.window_size,
        "video_preview_mode": args.mode,
        "max_preview_points": max_points,
    }


def write_lingbot_point_cloud(
    path: Path,
    *,
    predictions: dict[str, Any],
    images: Any | None,
    max_points: int,
    confidence_threshold: float,
) -> int:
    world_points = tensor_to_numpy(predictions["world_points"])
    confidence = tensor_to_numpy(predictions.get("world_points_conf"))
    images = tensor_to_numpy(images)
    if world_points.ndim == 5:
        world_points = world_points[0]
    if confidence is not None and confidence.ndim == 4:
        confidence = confidence[0]
    if images is not None and images.ndim == 5:
        images = images[0]
    if images is not None and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    points = world_points.reshape(-1, 3)
    colors = np.ones((len(points), 3), dtype=np.uint8) * 255
    if images is not None:
        colors = np.clip(images.reshape(-1, 3) * 255.0, 0, 255).astype(np.uint8)
    valid = np.isfinite(points).all(axis=1)
    if confidence is not None:
        conf_flat = confidence.reshape(-1)
        valid &= np.isfinite(conf_flat) & (conf_flat >= confidence_threshold)
    points = points[valid]
    colors = colors[valid]
    if len(points) <= 0:
        raise RuntimeError("LingBot-Map did not produce valid 3D points")
    if len(points) > max_points:
        if confidence is not None:
            confidence_valid = confidence.reshape(-1)[valid]
            keep = np.argsort(confidence_valid)[::-1][:max_points]
        else:
            keep = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[keep]
        colors = colors[keep]
    write_binary_point_ply(path, points, colors)
    return int(len(points))


def write_binary_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        file.write(header.encode("ascii"))
        for point, color in zip(points, colors):
            file.write(
                struct.pack(
                    "<fffBBB",
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )


def tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def positive_int(value: object) -> int | None:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def positive_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1)
