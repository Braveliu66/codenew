from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

from backend.app.db import models
from backend.app.algorithms.registry import AlgorithmRegistryEntry
from backend.app.algorithms.runner import RealAlgorithmCommandRunner
from backend.app.main import current_gpu_resources, parse_nvidia_smi_gpus
from backend.app.services.serializers import task_to_dict
from backend.scripts import build_preview_runtime, download_model_weights


class RuntimeConfigurationTests(unittest.TestCase):
    def test_transformer_engine_install_pins_cuda12_packages_to_one_version(self) -> None:
        pip_installs: list[list[str]] = []
        pip_uninstalls: list[list[str]] = []
        validated: list[str] = []

        with (
            patch.dict("os.environ", {"TRANSFORMER_ENGINE_VERSION": "2.14.0"}),
            patch.object(build_preview_runtime, "pip_install", side_effect=lambda args: pip_installs.append(args)),
            patch.object(build_preview_runtime, "pip_uninstall", side_effect=lambda args: pip_uninstalls.append(args)),
            patch.object(build_preview_runtime, "validate_transformer_engine_packages", side_effect=lambda version: validated.append(version)),
        ):
            build_preview_runtime.install_transformer_engine()

        self.assertEqual(pip_uninstalls, [["transformer-engine-cu13"]])
        self.assertIn(["transformer-engine==2.14.0", "transformer-engine-cu12==2.14.0"], pip_installs)
        self.assertIn(["--no-build-isolation", "transformer-engine-torch==2.14.0"], pip_installs)
        self.assertEqual(validated, ["2.14.0"])

    def test_transformer_engine_validation_rejects_cuda13_package(self) -> None:
        versions = {
            "transformer-engine": "2.14.0",
            "transformer-engine-cu12": "2.14.0",
            "transformer-engine-torch": "2.14.0",
            "transformer-engine-cu13": "2.14.0",
        }

        with patch.object(build_preview_runtime, "installed_package_version", side_effect=lambda name: versions.get(name)):
            with self.assertRaisesRegex(RuntimeError, "transformer-engine-cu13=2.14.0"):
                build_preview_runtime.validate_transformer_engine_packages("2.14.0")

    def test_edgs_extensions_default_to_source_build(self) -> None:
        compiled: list[str] = []

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(build_preview_runtime, "compile_edgs_extension", side_effect=lambda package, _: compiled.append(package)),
        ):
            build_preview_runtime.install_edgs_extension("simple_knn", build_preview_runtime.Path("unused"))

        self.assertEqual(compiled, ["simple_knn"])

    def test_edgs_source_build_uninstalls_incompatible_existing_package(self) -> None:
        pip_installs: list[list[str]] = []
        pip_uninstalls: list[list[str]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.dict("sys.modules", {"torch": object()}),
                patch.object(build_preview_runtime, "pip_install", side_effect=lambda args: pip_installs.append(args)),
                patch.object(build_preview_runtime, "pip_uninstall", side_effect=lambda args: pip_uninstalls.append(args)),
            ):
                build_preview_runtime.compile_edgs_extension("simple_knn", build_preview_runtime.Path(tmpdir))

        self.assertEqual(pip_uninstalls, [["simple_knn"]])
        self.assertEqual(pip_installs, [["--no-build-isolation", tmpdir]])

    def test_parse_nvidia_smi_gpus_keeps_all_devices(self) -> None:
        parsed = parse_nvidia_smi_gpus(
            "0, RTX A, 17, 12282, 1024\n"
            "1, RTX B, 83, 24564, 12000\n"
        )

        self.assertEqual([gpu["index"] for gpu in parsed], [0, 1])
        self.assertEqual(parsed[1]["usage_percent"], 83)
        self.assertEqual(parsed[0]["memory_usage_percent"], 8.3)

    def test_current_gpu_resources_aggregates_multi_gpu_status(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, RTX A, 17, 12282, 1024\n1, RTX B, 83, 24564, 12000\n",
            stderr="",
        )
        with (
            patch(
                "backend.app.services.resource_monitor.current_nvml_gpu_resources",
                return_value={"available": False, "message": "no nvml", "source": "pynvml"},
            ),
            patch("backend.app.services.resource_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("backend.app.services.resource_monitor.subprocess.run", return_value=completed),
        ):
            resources = current_gpu_resources()

        self.assertTrue(resources["available"])
        self.assertEqual(resources["usage_percent"], 83)
        self.assertEqual(resources["memory_total"], 36846)
        self.assertEqual(resources["memory_used"], 13024)
        self.assertEqual(len(resources["gpus"]), 2)

    def test_algorithm_runner_keeps_complete_stdout_and_stderr_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = build_preview_runtime.Path(tmpdir)
            spec_path = root / "stage.json"
            result_path = root / "stage.result.json"
            spec_path.write_text("{}", encoding="utf-8")
            entry = AlgorithmRegistryEntry(
                name="TestAlgorithm",
                repo_url=None,
                license=None,
                commit_hash="test-command",
                enabled=True,
                source_type="command",
                commands={
                    "run": [
                        sys.executable,
                        "-c",
                        (
                            "import json, os, pathlib, sys; "
                            "print('O' * 5000); "
                            "print('E' * 5000, file=sys.stderr); "
                            "pathlib.Path(os.environ['GS_STAGE_RESULT']).write_text("
                            "json.dumps({'status':'succeeded','artifacts':[{'kind':'out','path':os.environ['GS_TASK_SPEC']}]}),"
                            "encoding='utf-8')"
                        ),
                    ]
                },
            )

            result, issue = RealAlgorithmCommandRunner().run(entry, "run", spec_path, result_path, 10)

            self.assertIsNone(issue)
            self.assertIsNotNone(result)
            runner = result["_runner"]  # type: ignore[index]
            self.assertIn("O" * 5000, runner["stdout"])
            self.assertIn("E" * 5000, runner["stderr"])
            self.assertIn("O" * 5000, build_preview_runtime.Path(runner["stdout_path"]).read_text(encoding="utf-8"))
            self.assertIn("E" * 5000, build_preview_runtime.Path(runner["stderr_path"]).read_text(encoding="utf-8"))

    def test_running_task_serializer_estimates_progress_and_eta(self) -> None:
        now = models.utc_now()
        task = models.Task(
            id="task-1",
            project_id="project-1",
            type="preview",
            status="running",
            progress=15,
            current_stage="training_edgs",
            options={"estimated_duration_seconds": 120},
            created_at=now - timedelta(minutes=2),
            started_at=now - timedelta(seconds=60),
        )

        payload = task_to_dict(task)

        self.assertGreater(payload["progress"], 15)
        self.assertLess(payload["progress"], 100)
        self.assertIsNotNone(payload["eta_seconds"])
        self.assertLessEqual(payload["eta_seconds"], 60)

    def test_weight_downloader_reuses_complete_cached_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = download_model_weights.MODEL_WEIGHTS["litevggt"].target_path(build_preview_runtime.Path(tmpdir))
            target.parent.mkdir(parents=True)
            target.write_bytes(b"cached")

            with (
                patch.dict("os.environ", {}, clear=True),
                patch.object(download_model_weights, "remote_content_length", return_value=len(b"cached")),
                patch.object(download_model_weights, "download_with_resume") as downloader,
            ):
                result = download_model_weights.ensure_model_weight(
                    download_model_weights.MODEL_WEIGHTS["litevggt"],
                    build_preview_runtime.Path(tmpdir),
                    "https://example.invalid",
                )

            self.assertEqual(result, target)
            downloader.assert_not_called()

    def test_weight_downloader_resumes_part_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {}, clear=True):
                root = build_preview_runtime.Path(tmpdir)
                model = download_model_weights.MODEL_WEIGHTS["lingbot-map"]
                target = model.target_path(root)
                target.parent.mkdir(parents=True)
                part = target.with_suffix(target.suffix + ".part")
                part.write_bytes(b"abc")

                def fake_download(_: str, part_path: build_preview_runtime.Path, __: int | None) -> None:
                    self.assertEqual(part_path.read_bytes(), b"abc")
                    with part_path.open("ab") as file:
                        file.write(b"def")

                with (
                    patch.object(download_model_weights, "remote_content_length", return_value=6),
                    patch.object(download_model_weights, "download_with_resume", side_effect=fake_download),
                ):
                    result = download_model_weights.ensure_model_weight(model, root, "https://example.invalid")

                self.assertEqual(result.read_bytes(), b"abcdef")
                self.assertFalse(part.exists())


if __name__ == "__main__":
    unittest.main()
