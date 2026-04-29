from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from backend.app.algorithms.errors import AlgorithmErrorCode
from backend.app.algorithms.preview_engine import PreviewEngine
from backend.app.algorithms.models import PreviewTaskRequest
from backend.app.algorithms.registry import AlgorithmRegistry, AlgorithmRegistryEntry


TEST_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_tests"


def preview_request(**overrides: object) -> PreviewTaskRequest:
    work_dir = TEST_TMP_ROOT / f"preview-{uuid.uuid4().hex}"
    raw_dir = work_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "image.jpg").write_bytes(b"not-empty")
    base = {
        "task_id": "preview-task-1",
        "project_id": "project-1",
        "user_id": "demo-user",
        "input_type": "images",
        "raw_uri": str(raw_dir),
        "work_dir": work_dir,
        "output_prefix": str(work_dir / "preview"),
        "options": {"skip_backend_cuda_check": True},
    }
    base.update(overrides)
    return PreviewTaskRequest(**base)


class PreviewEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        TEST_TMP_ROOT.mkdir(exist_ok=True)

    def test_video_plan_uses_lingbot_map_before_spz(self) -> None:
        engine = PreviewEngine(AlgorithmRegistry())
        request = preview_request(input_type="video", raw_uri=str(TEST_TMP_ROOT / "video.mp4"))

        plan = engine.build_plan(request)

        self.assertEqual(
            [stage.algorithm for stage in plan.stages],
            ["LingBot-Map", "Spark-SPZ"],
        )
        self.assertEqual(plan.pipeline_options["preview_pipeline"], "lingbot_map_spark")

    def test_camera_plan_uses_lingbot_map_progressive_streaming(self) -> None:
        engine = PreviewEngine(AlgorithmRegistry())
        request = preview_request(
            input_type="camera",
            raw_uri=str(TEST_TMP_ROOT / "camera.webm"),
            options={"skip_backend_cuda_check": True, "progressive": True, "segment_index": 3},
        )

        plan = engine.build_plan(request)

        self.assertEqual([stage.algorithm for stage in plan.stages], ["LingBot-Map", "Spark-SPZ"])
        self.assertEqual(plan.stages[0].name, "camera_lingbot_map")
        self.assertEqual(plan.pipeline_options["video_preview_mode"], "streaming")
        self.assertTrue(plan.pipeline_options["progressive"])
        self.assertEqual(plan.pipeline_options["segment_index"], 3)

    def test_image_plan_defaults_to_litevggt_edgs_spz(self) -> None:
        engine = PreviewEngine(AlgorithmRegistry())
        request = preview_request()

        plan = engine.build_plan(request)

        self.assertEqual(
            [stage.algorithm for stage in plan.stages],
            ["LiteVGGT", "EDGS", "Spark-SPZ"],
        )
        self.assertEqual(plan.pipeline_options["preview_pipeline"], "edgs")

    def test_image_direct_pipeline_skips_edgs(self) -> None:
        engine = PreviewEngine(AlgorithmRegistry())
        request = preview_request(options={"skip_backend_cuda_check": True, "preview_pipeline": "litevggt_spark"})

        plan = engine.build_plan(request)

        self.assertEqual(
            [stage.algorithm for stage in plan.stages],
            ["LiteVGGT", "Spark-SPZ"],
        )
        self.assertIn("training_edgs", [stage.name for stage in plan.skipped_stages])

    def test_execute_fails_without_configured_preview_algorithms(self) -> None:
        engine = PreviewEngine(AlgorithmRegistry())
        request = preview_request()

        result = engine.execute(request)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.artifacts, [])
        self.assertIn(
            AlgorithmErrorCode.ALGORITHM_NOT_CONFIGURED.value,
            {error["code"] for error in result.errors},
        )

    def test_missing_spz_converter_command_is_explicit(self) -> None:
        weight = TEST_TMP_ROOT / "te_dict.pt"
        weight.write_bytes(b"real-weight-path-marker")
        registry = AlgorithmRegistry(
            [
                AlgorithmRegistryEntry(
                    name="LiteVGGT",
                    repo_url="https://github.com/GarlicBa/LiteVGGT-repo",
                    license="MIT",
                    commit_hash="test-command",
                    weight_source="https://huggingface.co/ZhijianShu/LiteVGGT/resolve/main/te_dict.pt",
                    weight_paths=(weight,),
                    enabled=True,
                    source_type="command",
                    commands={"run_demo": ["python", "-c", "pass"]},
                ),
                AlgorithmRegistryEntry(
                    name="EDGS",
                    repo_url="https://github.com/CompVis/EDGS",
                    license="Apache-2.0",
                    commit_hash="test-command",
                    enabled=True,
                    source_type="command",
                    commands={"train": ["python", "-c", "pass"]},
                ),
                AlgorithmRegistryEntry(
                    name="Spark-SPZ",
                    repo_url="https://github.com/sparkjsdev/spark",
                    license="MIT",
                    commit_hash="test-command",
                    enabled=True,
                    source_type="command",
                    commands={},
                ),
            ]
        )
        request = preview_request()

        issues = PreviewEngine(registry).validate_plan(PreviewEngine(registry).build_plan(request), request)

        self.assertIn(
            AlgorithmErrorCode.SPZ_CONVERTER_NOT_CONFIGURED.value,
            {issue.code.value for issue in issues},
        )


if __name__ == "__main__":
    unittest.main()

