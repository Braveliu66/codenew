from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    spec_path = Path(os.environ["GS_TASK_SPEC"])
    result_path = Path(os.environ["GS_STAGE_RESULT"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    algorithm = spec["algorithms"]["LiteVGGT"]
    repo_path = Path(algorithm["local_path"])
    sys.path.insert(0, str(repo_path))

    weights = [Path(path) for path in algorithm.get("weight_paths") or []]
    checkpoint = weights[0] if weights else None
    image_dir = Path(spec["image_dir"])
    output_dir = Path(spec["output_dir"])
    dataset_dir = output_dir / "dataset"
    processed_image_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if not repo_path.exists():
        raise RuntimeError("LiteVGGT repository is missing")
    if checkpoint is None or not checkpoint.exists():
        raise RuntimeError("LiteVGGT checkpoint te_dict.pt is missing")
    if not image_dir.exists():
        raise RuntimeError(f"LiteVGGT image directory is missing: {image_dir}")

    min_frames = max(int(spec.get("min_input_frames") or 8), 1)
    max_frames = max(int(spec.get("max_input_frames") or 800), min_frames)
    image_paths = select_evenly(sorted_images(image_dir), max_frames)
    if len(image_paths) < min_frames:
        raise RuntimeError(f"LiteVGGT requires at least {min_frames} image; got {len(image_paths)}")

    result = run_litevggt(
        repo_path=repo_path,
        checkpoint=checkpoint,
        image_paths=image_paths,
        processed_image_dir=processed_image_dir,
        sparse_dir=sparse_dir,
        output_dir=output_dir,
    )
    write_result(
        result_path,
        {
            "status": "succeeded",
            "artifacts": [
                {"kind": "output_dir", "path": str(output_dir.resolve())},
                {"kind": "dataset_dir", "path": str(dataset_dir.resolve())},
                {"kind": "colmap_dir", "path": str(sparse_dir.resolve())},
                {"kind": "preview_ply", "path": str(result["ply_path"].resolve())},
                {"kind": "point_cloud", "path": str(result["ply_path"].resolve())},
            ],
            "metrics": {
                "input_frame_count": len(image_paths),
                "processed_width": result["width"],
                "processed_height": result["height"],
                "colmap_point_count": result["point_count"],
            },
        },
    )
    return 0


def run_litevggt(
    *,
    repo_path: Path,
    checkpoint: Path,
    image_paths: list[Path],
    processed_image_dir: Path,
    sparse_dir: Path,
    output_dir: Path,
    ) -> dict[str, object]:
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    from vggt.models.vggt import VGGT
    from vggt.utils.eval_utils import load_image_file_crop
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = os.environ.get("LITEVGGT_DEVICE", "cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for LiteVGGT")
    device_index = torch.device(device).index or 0
    print(
        json.dumps(
            {
                "event": "litevggt_cuda",
                "device": device,
                "device_name": torch.cuda.get_device_name(device_index),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "input_frames": len(image_paths),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT().to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.to(torch.bfloat16)
    model.eval()

    images_np: list[np.ndarray] = []
    saved_names: list[str] = []
    for index, image_path in enumerate(image_paths):
        image = load_image_file_crop(str(image_path))
        images_np.append(image)
        name = f"{index:05d}.png"
        saved_names.append(name)
        Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(processed_image_dir / name)

    images = torch.stack(
        [torch.from_numpy(np.transpose(image, (2, 0, 1))) for image in images_np],
        dim=0,
    ).to(device)
    height = int(images.shape[-2])
    width = int(images.shape[-1])
    if hasattr(model, "update_patch_dimensions"):
        model.update_patch_dimensions(width // 14, height // 14)

    images_batched = images[None]
    with torch.no_grad():
        fp8_recipe = DelayedScaling(fp8_format=Format.E4M3, amax_history_len=80, amax_compute_algo="max")
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            aggregated_tokens_list, patch_start_idx = model.aggregator(images_batched)
        with torch.amp.autocast("cuda", enabled=True, dtype=dtype):
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            w2c_pre, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batched.shape[-2:])
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batched, patch_start_idx)

    points_3d = unproject_depth_map_to_point_map(
        depth_map.squeeze(0),
        w2c_pre.squeeze(0),
        intrinsic.squeeze(0),
    )
    points = points_3d.reshape(-1, 3)
    colors = np.clip(images_batched[0].permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    confidence = depth_conf.reshape(-1).detach().cpu().numpy()
    valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    points = points[valid]
    colors = colors[valid]
    confidence = confidence[valid]
    keep_count = min(max(int(len(points) * 0.25), 1000), 200000, len(points))
    if keep_count <= 0:
        raise RuntimeError("LiteVGGT did not produce valid 3D points")
    keep_indices = np.argsort(confidence)[::-1][:keep_count]
    points = points[keep_indices]
    colors = colors[keep_indices]

    w2c = w2c_pre.squeeze(0).detach().cpu().numpy()
    intrinsics = intrinsic.squeeze(0).detach().cpu().numpy()
    write_colmap_text(
        sparse_dir=sparse_dir,
        image_names=saved_names,
        width=width,
        height=height,
        w2c=w2c,
        intrinsics=intrinsics,
        points=points,
        colors=colors,
    )
    try_convert_colmap_to_binary(sparse_dir)
    ply_path = output_dir / "recon.ply"
    write_point_cloud_ply(ply_path, points, colors)
    if torch.cuda.is_available():
        print(
            json.dumps(
                {
                    "event": "litevggt_cuda_memory",
                    "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device_index) / 1024 / 1024, 1),
                    "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device_index) / 1024 / 1024, 1),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return {"ply_path": ply_path, "point_count": len(points), "width": width, "height": height}


def sorted_images(image_dir: Path) -> list[Path]:
    return [path for path in sorted(image_dir.iterdir()) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def select_evenly(items: list[Path], max_items: int) -> list[Path]:
    if len(items) <= max_items:
        return items
    last = len(items) - 1
    indexes = [round(i * last / (max_items - 1)) for i in range(max_items)]
    selected: list[Path] = []
    seen: set[int] = set()
    for index in indexes:
        if index not in seen:
            selected.append(items[index])
            seen.add(index)
    return selected


def write_colmap_text(
    *,
    sparse_dir: Path,
    image_names: list[str],
    width: int,
    height: int,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with (sparse_dir / "cameras.txt").open("w", encoding="utf-8") as file:
        file.write("# Camera list with one line of data per camera:\n")
        file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for index, intrinsic in enumerate(intrinsics, start=1):
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            file.write(f"{index} PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")
    with (sparse_dir / "images.txt").open("w", encoding="utf-8") as file:
        file.write("# Image list with two lines of data per image:\n")
        file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for index, name in enumerate(image_names, start=1):
            matrix = w2c[index - 1]
            rotation = matrix[:3, :3]
            translation = matrix[:3, 3]
            qvec = rotation_matrix_to_qvec(rotation)
            file.write(
                f"{index} {qvec[0]:.12g} {qvec[1]:.12g} {qvec[2]:.12g} {qvec[3]:.12g} "
                f"{translation[0]:.12g} {translation[1]:.12g} {translation[2]:.12g} {index} {name}\n\n"
            )
    with (sparse_dir / "points3D.txt").open("w", encoding="utf-8") as file:
        file.write("# 3D point list with one line of data per point:\n")
        file.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for index, (point, color) in enumerate(zip(points, colors), start=1):
            file.write(
                f"{index} {point[0]:.12g} {point[1]:.12g} {point[2]:.12g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0\n"
            )


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = np.linalg.norm(qvec)
    return qvec / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def try_convert_colmap_to_binary(sparse_dir: Path) -> None:
    try:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
        reconstruction.write(str(sparse_dir))
    except Exception as exc:
        (sparse_dir / "binary_conversion_warning.txt").write_text(str(exc), encoding="utf-8")


def write_point_cloud_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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


def write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1)
