from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from backend.app.algorithms.errors import AlgorithmErrorCode
from backend.app.algorithms.fine_engine import FineSynthesisEngine
from backend.app.algorithms.models import FineTaskRequest
from backend.app.algorithms.registry import AlgorithmRegistry, AlgorithmRegistryEntry


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def fine_request(**overrides: object) -> FineTaskRequest:
    base = {
        "task_id": "task-1",
        "project_id": "project-1",
        "input_type": "images",
        "raw_uri": "s3://bucket/users/u/projects/p/raw/images/",
        "work_dir": TEST_TMP_ROOT / "codenew-test-work",
        "output_prefix": "s3://bucket/users/u/projects/p/final/",
    }
    base.update(overrides)
    return FineTaskRequest(**base)


def command_entry(name: str, commands: dict[str, list[str]] | None = None) -> AlgorithmRegistryEntry:
    return AlgorithmRegistryEntry(
        name=name,
        repo_url=f"https://example.invalid/{name}",
        license="test-license",
        commit_hash="system-command",
        enabled=True,
        source_type="command",
        commands=commands or {},
    )


def complete_registry() -> AlgorithmRegistry:
    return AlgorithmRegistry(
        [
            command_entry("Faster-GS", {"fine_engine": ["python", "-c", "pass"]}),
            command_entry("FastGS"),
            command_entry("Deblurring-3DGS"),
            command_entry("3DGS-LM"),
            command_entry("Spark-SPZ", {"compress_final": ["python", "-c", "pass"]}),
            command_entry("RAD-LOD", {"export_rad": ["python", "-c", "pass"]}),
        ]
    )


class FakeFineRunner:
    def __init__(self, *, omit_spz: bool = False) -> None:
        self.omit_spz = omit_spz

    def run(self, entry: AlgorithmRegistryEntry, command_key: str, spec_path: Path, result_path: Path, timeout_seconds: int):
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        work_dir = Path(spec["work_dir"])
        if command_key == "fine_engine":
            final_ply = work_dir / "final" / "final.ply"
            final_ply.parent.mkdir(parents=True, exist_ok=True)
            final_ply.write_bytes(b"ply")
            return {"status": "succeeded", "artifacts": [{"kind": "final_ply", "path": str(final_ply)}], "metrics": {}}, None
        if command_key == "compress_final":
            if self.omit_spz:
                return {"status": "succeeded", "artifacts": [{"kind": "other", "path": str(spec_path)}], "metrics": {}}, None
            output_spz = Path(spec["output_spz"])
            output_spz.parent.mkdir(parents=True, exist_ok=True)
            output_spz.write_bytes(b"spz")
            return {"status": "succeeded", "artifacts": [{"kind": "final_web_spz", "path": str(output_spz)}], "metrics": {}}, None
        if command_key == "export_rad":
            output_dir = Path(spec["output_dir"])
            artifacts = []
            for lod in range(4):
                path = output_dir / f"final_lod{lod}.rad"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"rad-{lod}".encode("ascii"))
                artifacts.append(
                    {
                        "kind": "lod_rad",
                        "lod": lod,
                        "path": str(path),
                        "target_gaussians": [1_000_000, 500_000, 200_000, 50_000][lod],
                        "actual_gaussians": [900_000, 450_000, 180_000, 45_000][lod],
                    }
                )
            return {"status": "succeeded", "artifacts": artifacts, "metrics": {}}, None
        raise AssertionError(command_key)


class FineSynthesisEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)

    def make_work_dir(self) -> Path:
        path = TEST_TMP_ROOT / f"case-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_plan_selects_sparse_deblur_and_long_video_stages(self) -> None:
        engine = FineSynthesisEngine(AlgorithmRegistry())
        request = fine_request(
            input_type="video",
            frame_count=700,
            effective_view_count=8,
            blur_detected=True,
            enable_long_video_global_optimization=True,
        )

        plan = engine.build_plan(request)
        stage_algorithms = [stage.algorithm for stage in plan.stages]

        self.assertEqual(
            stage_algorithms,
            [
                "LingBot-Map",
                "MASt3R",
                "Pi3",
                "FreeSplatter",
                "Faster-GS",
                "FastGS",
                "Deblurring-3DGS",
                "3DGS-LM",
                "Spark-SPZ",
                "RAD-LOD",
            ],
        )
        self.assertEqual(plan.engine_options["lm_optimizer"]["start_iter"], 3000)
        self.assertEqual(plan.engine_options["lm_optimizer"]["interval"], 200)
        self.assertIn("lod_rad:3", plan.engine_options["requested_outputs"])

    def test_long_video_global_optimization_is_not_default(self) -> None:
        engine = FineSynthesisEngine(AlgorithmRegistry())
        request = fine_request(input_type="video", frame_count=700)

        plan = engine.build_plan(request)

        self.assertNotIn("LingBot-Map", [stage.algorithm for stage in plan.stages])
        self.assertEqual(plan.skipped_stages[0].name, "long_video_global_optimization")

    def test_execute_fails_without_configured_algorithms_and_creates_no_artifacts(self) -> None:
        engine = FineSynthesisEngine(AlgorithmRegistry())
        tmpdir = self.make_work_dir()
        request = fine_request(work_dir=tmpdir, blur_detected=True)

        result = engine.execute(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.artifacts, [])
        self.assertIn(AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED.value, {error["code"] for error in result.errors})
        self.assertFalse((tmpdir / "final.ply").exists())
        self.assertFalse((tmpdir / "preview.spz").exists())

    def test_enabled_entry_without_fine_command_is_rejected(self) -> None:
        tmpdir = self.make_work_dir()
        registry = AlgorithmRegistry(
            [
                command_entry("Faster-GS", {}),
            ]
        )
        request = fine_request(
            work_dir=tmpdir,
            options={
                "fused3dgs": {"use_vcd": False, "use_lm_optimizer": False, "use_deblur": False},
                "outputs": {"spz": False, "lod": False, "metrics": False},
            },
        )

        result = FineSynthesisEngine(registry).execute(request)

        self.assertIn(AlgorithmErrorCode.ALGORITHM_RUNNER_NOT_CONFIGURED.value, {error["code"] for error in result.errors})

    def test_complete_chain_succeeds_with_verified_artifacts(self) -> None:
        tmpdir = self.make_work_dir()
        result = FineSynthesisEngine(complete_registry(), runner=FakeFineRunner()).execute(fine_request(work_dir=tmpdir))

        self.assertEqual(result.status, "succeeded")
        self.assertEqual(
            [artifact["kind"] for artifact in result.artifacts],
            ["final_ply", "final_web_spz", "lod_rad", "lod_rad", "lod_rad", "lod_rad", "metrics_json"],
        )
        self.assertIn("final_ply", result.artifact_paths)
        self.assertIn("final_web_spz", result.artifact_paths)
        self.assertEqual([item["lod"] for item in result.artifact_paths["lod_rad"]], [0, 1, 2, 3])
        self.assertTrue((tmpdir / "final" / "metrics.json").is_file())

    def test_requested_output_missing_fails(self) -> None:
        tmpdir = self.make_work_dir()
        result = FineSynthesisEngine(complete_registry(), runner=FakeFineRunner(omit_spz=True)).execute(fine_request(work_dir=tmpdir))

        self.assertEqual(result.status, "failed")
        self.assertIn("final_web_spz", result.errors[0]["message"])


if __name__ == "__main__":
    unittest.main()
