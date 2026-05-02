from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.db import models
from backend.app.algorithms.registry import AlgorithmRegistryEntry
from backend.app.algorithms.runner import RealAlgorithmCommandRunner
from backend.app.main import current_gpu_resources, gpu_resources_from_workers, parse_nvidia_smi_gpus
from backend.app.services.serializers import task_to_dict
from backend.scripts import (
    build_gpu_runtime,
    build_preview_runtime,
    download_algorithm_repos,
    download_model_weights,
    pull_base_images,
    run_spz_convert,
)


class RuntimeConfigurationTests(unittest.TestCase):
    def test_gpu_runtime_filters_litevggt_torch_and_cuda_stack_pins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "requirements.txt"
            target = root / "filtered.txt"
            source.write_text(
                "torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "
                "numpy==1.26.1 opencv-python Pillow einops safetensors\n",
                encoding="utf-8",
            )

            result = build_gpu_runtime.filter_litevggt_requirements(source, target)

            requirements = result.read_text(encoding="utf-8").splitlines()
            self.assertEqual(requirements, ["Pillow", "einops", "safetensors"])

    def test_gpu_runtime_installs_one_torch_cuda128_stack(self) -> None:
        calls: list[tuple[list[str], str, Path]] = []

        with (
            patch.dict("os.environ", {}, clear=True),
            patch.object(
                build_gpu_runtime,
                "install_pip_requirements_via_wheelhouse",
                side_effect=lambda *, requirements, index_url, wheelhouse: calls.append((requirements, index_url, wheelhouse)),
            ),
        ):
            build_gpu_runtime.install_torch_runtime()

        self.assertEqual(
            calls,
            [
                (
                    ["torch==2.8.0+cu128", "torchvision==0.23.0+cu128", "torchaudio==2.8.0+cu128"],
                    "https://mirrors.aliyun.com/pytorch-wheels/cu128",
                    Path("/root/.cache/three-dgs-wheelhouse/torch-cu128"),
                )
            ],
        )

    def test_gpu_runtime_torch_index_falls_back_to_official_source(self) -> None:
        calls: list[str] = []

        def fake_install(*, requirements: list[str], index_url: str, wheelhouse: Path) -> None:
            calls.append(index_url)
            if "mirror.invalid" in index_url:
                raise RuntimeError("mirror unavailable")

        with (
            patch.dict(
                "os.environ",
                {"TORCH_INDEX_URLS": "https://mirror.invalid/cu128,https://download.pytorch.org/whl/cu128"},
                clear=True,
            ),
            patch.object(build_gpu_runtime, "install_pip_requirements_via_wheelhouse", side_effect=fake_install),
        ):
            build_gpu_runtime.install_torch_runtime()

        self.assertEqual(calls, ["https://mirror.invalid/cu128", "https://download.pytorch.org/whl/cu128"])

    def test_gpu_runtime_torch_resolution_tries_find_links_for_flat_mirror(self) -> None:
        with patch.dict("os.environ", {"PIP_INDEX_URL": "https://pypi.tuna.tsinghua.edu.cn/simple"}):
            commands = build_gpu_runtime.pip_resolution_commands(
                requirements=["torch==2.8.0+cu128"],
                index_url="https://mirrors.aliyun.com/pytorch-wheels/cu128",
                report=Path("/tmp/report.json"),
            )

        self.assertEqual(commands[0][0], "index-url")
        self.assertIn("--extra-index-url", commands[0][1])
        self.assertEqual(commands[1][0], "find-links")
        self.assertIn("--find-links", commands[1][1])
        self.assertIn("https://mirrors.aliyun.com/pytorch-wheels/cu128", commands[1][1])

    def test_wheel_report_extracts_urls_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = Path(tmpdir) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "install": [
                            {
                                "download_info": {
                                    "url": "https://download.example/torch-2.8.0.whl",
                                    "archive_info": {"hashes": {"sha256": "abc"}},
                                }
                            },
                            {
                                "download_info": {
                                    "url": "https://download.example/torchvision-0.23.0.whl",
                                    "archive_info": {"hash": "sha256=def"},
                                }
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            downloads = build_gpu_runtime.wheel_downloads_from_report(report)

        self.assertEqual(
            downloads,
            [
                {"url": "https://download.example/torch-2.8.0.whl", "sha256": "abc"},
                {"url": "https://download.example/torchvision-0.23.0.whl", "sha256": "def"},
            ],
        )

    def test_algorithm_repo_download_defaults_to_github_without_mirror(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            commands = download_algorithm_repos.clone_attempt_commands(
                "https://github.com/GarlicBa/LiteVGGT-repo.git",
                Path("repo-cache/LiteVGGT-repo"),
            )

        self.assertEqual(len(commands), 1)
        self.assertIn("https://github.com/GarlicBa/LiteVGGT-repo.git", commands[0])

    def test_algorithm_repo_download_allows_explicit_mirror_prefixes(self) -> None:
        with patch.dict("os.environ", {"ALGORITHM_REPO_MIRROR_PREFIXES": "https://mirror.example/github/"}, clear=True):
            commands = download_algorithm_repos.clone_attempt_commands(
                "https://github.com/GarlicBa/LiteVGGT-repo.git",
                Path("repo-cache/LiteVGGT-repo"),
            )

        self.assertIn("https://mirror.example/github/GarlicBa/LiteVGGT-repo.git", commands[0])
        self.assertIn("https://github.com/GarlicBa/LiteVGGT-repo.git", commands[-1])

    def test_gpu_runtime_uses_local_repo_cache_without_clone(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            commands.append(command)
            return subprocess.CompletedProcess(args=command, returncode=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_repo = root / "repo-cache" / "LiteVGGT-repo"
            target = root / "runtime" / "repos" / "LiteVGGT-repo"
            (cache_repo / ".git").mkdir(parents=True)
            (cache_repo / "requirements.txt").write_text("Pillow\n", encoding="utf-8")

            with patch("backend.scripts.build_gpu_runtime.subprocess.run", side_effect=fake_run):
                build_gpu_runtime.clone_checkout(
                    build_gpu_runtime.LITEVGGT_REPO,
                    target,
                    "commit",
                    repo_cache_root=root / "repo-cache",
                )
            target_exists = target.exists()

        self.assertTrue(target_exists)
        self.assertFalse(any("clone" in command for command in commands))
        self.assertTrue(any(command[-2:] == ["checkout", "commit"] for command in commands))

    def test_wheel_download_resumes_part_file(self) -> None:
        class FakeResponse:
            status = 206
            headers = {"Content-Length": "3"}

            def __init__(self) -> None:
                self._chunks = [b"def", b""]

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def getcode(self) -> int:
                return self.status

            def read(self, _: int) -> bytes:
                return self._chunks.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            part = Path(tmpdir) / "package.whl.part"
            part.write_bytes(b"abc")
            seen_ranges: list[str | None] = []

            def fake_urlopen(request: object, **_: object) -> FakeResponse:
                seen_ranges.append(request.get_header("Range"))  # type: ignore[attr-defined]
                return FakeResponse()

            with patch("backend.scripts.build_gpu_runtime.urllib.request.urlopen", side_effect=fake_urlopen):
                build_gpu_runtime.download_url_with_resume("https://example.invalid/package.whl", part, expected_size=6)

            self.assertEqual(part.read_bytes(), b"abcdef")
            self.assertEqual(seen_ranges, ["bytes=3-"])

    def test_gpu_runtime_installs_lingbot_without_re_resolving_torch(self) -> None:
        pip_installs: list[list[str]] = []

        with (
            patch.dict("os.environ", {"INSTALL_FLASHINFER": "false"}),
            patch.object(build_gpu_runtime, "pip_install", side_effect=lambda args: pip_installs.append(args)),
        ):
            build_gpu_runtime.install_lingbot_runtime(Path("/opt/three-dgs/repos/lingbot-map"))

        self.assertIn(["--no-build-isolation", "--no-deps", "-e", str(Path("/opt/three-dgs/repos/lingbot-map"))], pip_installs)

    def test_gpu_runtime_normalizes_spark_shell_scripts_before_npm_ci(self) -> None:
        commands: list[tuple[list[str], Path | None]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            spark = Path(tmpdir) / "spark"
            script = spark / "rust" / "build_rust_wasm.sh"
            script.parent.mkdir(parents=True)
            script.write_bytes(b"#!/usr/bin/env bash\r\necho build\r\n")

            with patch.object(build_gpu_runtime, "run", side_effect=lambda command, cwd=None, **_: commands.append((command, cwd))):
                build_gpu_runtime.install_spark_runtime(spark)

            self.assertEqual(script.read_bytes(), b"#!/usr/bin/env bash\necho build\n")

        self.assertEqual(commands[1], (["npm", "ci"], spark))

    def test_gpu_runtime_edgs_wheel_uses_hf_endpoint(self) -> None:
        pip_installs: list[list[str]] = []

        with (
            patch.dict("os.environ", {"EDGS_EXTENSION_INSTALL_MODE": "wheel", "HF_ENDPOINT": "https://hf-mirror.com"}),
            patch.object(build_gpu_runtime.sys, "version_info", (3, 10, 0)),
            patch.object(build_gpu_runtime.sys, "platform", "linux"),
            patch.object(build_gpu_runtime.platform, "machine", return_value="x86_64"),
            patch.object(build_gpu_runtime, "pip_install", side_effect=lambda args: pip_installs.append(args)),
        ):
            build_gpu_runtime.install_edgs_extension(
                "simple_knn",
                Path("/opt/three-dgs/repos/EDGS/submodules/gaussian-splatting/submodules/simple-knn"),
            )

        self.assertEqual(
            pip_installs,
            [
                [
                    "https://hf-mirror.com/spaces/CompVis/EDGS/resolve/main/wheels/"
                    "simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl"
                ]
            ],
        )

    def test_gpu_runtime_registries_enable_image_and_video_algorithms_separately(self) -> None:
        with (
            patch.object(build_gpu_runtime, "commit_hash", return_value="commit"),
            patch.object(build_gpu_runtime.shutil, "which", return_value="/usr/bin/ffmpeg"),
        ):
            image_algorithms = build_gpu_runtime.image_registry_algorithms(
                Path("/workspace"),
                Path("/opt/three-dgs/repos/LiteVGGT-repo"),
                Path("/opt/three-dgs/repos/EDGS"),
                Path("/opt/three-dgs/repos/lingbot-map"),
                Path("/opt/three-dgs/repos/spark"),
                Path("/model-cache/litevggt/te_dict.pt"),
                Path("/model-cache/lingbot-map/lingbot-map-long.pt"),
            )
            video_algorithms = build_gpu_runtime.video_registry_algorithms(
                Path("/workspace"),
                Path("/opt/three-dgs/repos/LiteVGGT-repo"),
                Path("/opt/three-dgs/repos/EDGS"),
                Path("/opt/three-dgs/repos/lingbot-map"),
                Path("/opt/three-dgs/repos/spark"),
                Path("/model-cache/litevggt/te_dict.pt"),
                Path("/model-cache/lingbot-map/lingbot-map-long.pt"),
            )

        image_enabled = {item["name"] for item in image_algorithms if item["enabled"]}
        video_enabled = {item["name"] for item in video_algorithms if item["enabled"]}
        self.assertEqual(image_enabled, {"LiteVGGT", "EDGS", "Spark-SPZ", "FFmpeg"})
        self.assertEqual(video_enabled, {"LingBot-Map", "Spark-SPZ", "FFmpeg"})

    def test_base_image_pull_defaults_to_unified_gpu_build_and_runtime_images(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            images = pull_base_images.configured_images()

        self.assertEqual(
            images,
            [
                "python:3.12-slim",
                "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
                "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04",
            ],
        )

    def test_spark_converter_uses_direct_node_script_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spark = root / "spark"
            scripts = spark / "scripts"
            scripts.mkdir(parents=True)
            converter = scripts / "compress-to-spz.js"
            converter.write_text("console.log('convert')\n", encoding="utf-8")
            input_ply = root / "input.ply"
            input_ply.write_bytes(b"ply")
            output_spz = root / "output.spz"
            spec_path = root / "spec.json"
            result_path = root / "result.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "algorithms": {"Spark-SPZ": {"local_path": str(spark)}},
                        "input_ply": str(input_ply),
                        "output_spz": str(output_spz),
                    }
                ),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                input_ply.with_suffix(".spz").write_bytes(b"spz")
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with (
                patch.dict(os.environ, {"GS_TASK_SPEC": str(spec_path), "GS_STAGE_RESULT": str(result_path)}),
                patch("backend.scripts.run_spz_convert.shutil.which", return_value="/usr/bin/node"),
                patch("backend.scripts.run_spz_convert.subprocess.run", side_effect=fake_run),
            ):
                self.assertEqual(run_spz_convert.main(), 0)

            self.assertEqual(commands[0], ["node", str(converter), str(input_ply.resolve())])
            self.assertTrue(output_spz.exists())

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

    def test_worker_gpu_resources_uses_fresh_latest_heartbeat_per_gpu(self) -> None:
        now = datetime.now(timezone.utc)
        workers = [
            SimpleNamespace(
                hostname="container-a",
                gpu_index=0,
                gpu_name="RTX 6000",
                gpu_memory_total=36864,
                gpu_memory_used=4096,
                gpu_utilization=12,
                last_seen_at=now - timedelta(seconds=6),
            ),
            SimpleNamespace(
                hostname="container-b",
                gpu_index=0,
                gpu_name="RTX 6000",
                gpu_memory_total=36864,
                gpu_memory_used=8192,
                gpu_utilization=73,
                last_seen_at=now - timedelta(seconds=1),
            ),
            SimpleNamespace(
                hostname="container-c",
                gpu_index=1,
                gpu_name="RTX 6000",
                gpu_memory_total=36864,
                gpu_memory_used=1024,
                gpu_utilization=8,
                last_seen_at=now - timedelta(seconds=90),
            ),
        ]

        resources = gpu_resources_from_workers(workers, max_age_seconds=20)

        self.assertTrue(resources["available"])
        self.assertEqual(resources["usage_percent"], 73)
        self.assertEqual(resources["memory_total"], 36864)
        self.assertEqual(resources["memory_used"], 8192)
        self.assertEqual(resources["stale_worker_count"], 1)
        self.assertEqual(len(resources["gpus"]), 1)

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
