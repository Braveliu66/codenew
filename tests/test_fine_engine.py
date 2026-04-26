from __future__ import annotations

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


class FineSynthesisEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)

    def make_work_dir(self) -> Path:
        path = TEST_TMP_ROOT / f"case-{uuid.uuid4().hex}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def test_plan_selects_sparse_deblur_mesh_and_long_video_stages(self) -> None:
        engine = FineSynthesisEngine(AlgorithmRegistry())
        request = fine_request(
            input_type="video",
            frame_count=700,
            effective_view_count=8,
            blur_detected=True,
            enable_long_video_global_optimization=True,
            request_mesh_export=True,
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
                "MeshSplatting",
            ],
        )
        self.assertEqual(plan.engine_options["optimizer_switch_iteration_ratio"], 0.7)
        self.assertEqual(plan.engine_options["hooks"]["densify_and_prune"], "FastGS")
        self.assertEqual(plan.engine_options["hooks"]["sparse_initializer"], "FreeSplatter")

    def test_long_video_global_optimization_is_not_default(self) -> None:
        engine = FineSynthesisEngine(AlgorithmRegistry())
        request = fine_request(input_type="video", frame_count=700)

        plan = engine.build_plan(request)

        self.assertNotIn("LingBot-Map", [stage.algorithm for stage in plan.stages])
        self.assertEqual(
            plan.skipped_stages[0].name,
            "long_video_global_optimization",
        )

    def test_execute_fails_without_configured_algorithms_and_creates_no_artifacts(self) -> None:
        engine = FineSynthesisEngine(AlgorithmRegistry())
        tmpdir = self.make_work_dir()
        request = fine_request(work_dir=tmpdir, blur_detected=True)

        result = engine.execute(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.artifacts, [])
        error_codes = {error["code"] for error in result.errors}
        self.assertIn(AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED.value, error_codes)
        self.assertFalse((tmpdir / "final.ply").exists())
        self.assertFalse((tmpdir / "preview.spz").exists())

    def test_enabled_entry_without_command_is_rejected(self) -> None:
        tmpdir = self.make_work_dir()
        repo = tmpdir / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        registry = AlgorithmRegistry(
            [
                AlgorithmRegistryEntry(
                    name="Faster-GS",
                    repo_url="https://example.invalid/faster-gs",
                    license="Apache-2.0",
                    commit_hash="abc123",
                    local_path=repo,
                    enabled=True,
                    commands={},
                )
            ]
        )
        engine = FineSynthesisEngine(registry)

        result = engine.execute(fine_request(work_dir=tmpdir))
        error_codes = {error["code"] for error in result.errors}

        self.assertIn(
            AlgorithmErrorCode.ALGORITHM_RUNNER_NOT_CONFIGURED.value,
            error_codes,
        )


if __name__ == "__main__":
    unittest.main()
