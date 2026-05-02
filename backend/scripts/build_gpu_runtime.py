from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from hashlib import sha256
from importlib import metadata
from pathlib import Path


LITEVGGT_REPO = "https://github.com/GarlicBa/LiteVGGT-repo.git"
EDGS_REPO = "https://github.com/CompVis/EDGS.git"
LINGBOT_REPO = "https://github.com/Robbyant/lingbot-map.git"
SPARK_REPO = "https://github.com/sparkjsdev/spark.git"

LITEVGGT_WEIGHT_REPO = "ZhijianShu/LiteVGGT"
LITEVGGT_WEIGHT_FILE = "te_dict.pt"
LINGBOT_WEIGHT_FILE = "lingbot-map-long.pt"

EDGS_LICENSE = "Non-commercial research and personal use (see EDGS LICENSE.txt)"
LINGBOT_LICENSE = "Apache-2.0"
EDGS_WHEEL_REPO_PATH = "spaces/CompVis/EDGS/resolve/main/wheels"
EDGS_WHEELS = {
    "diff_gaussian_rasterization": "diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl",
    "simple_knn": "simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl",
}

GITHUB_HTTPS_PREFIX = "https://github.com/"
GITHUB_SSH_PREFIX = "git@github.com:"
DEFAULT_ALGORITHM_REPO_MIRROR_PREFIXES = ""
DEFAULT_TORCH_INDEX_URLS = "https://mirrors.aliyun.com/pytorch-wheels/cu128,https://download.pytorch.org/whl/cu128"
DEFAULT_TORCH_CUDA_SUFFIX = "cu128"
DEFAULT_TORCH_VERSION = "2.8.0"
DEFAULT_TORCHVISION_VERSION = "0.23.0"
DEFAULT_TORCHAUDIO_VERSION = "2.8.0"
DEFAULT_TRANSFORMER_ENGINE_VERSION = "2.14.0"
DEFAULT_TORCH_WHEELHOUSE = "/root/.cache/three-dgs-wheelhouse/torch-cu128"
DEFAULT_PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
DOWNLOAD_RETRIES = 8
LITEVGGT_SKIPPED_REQUIREMENTS = {"torch", "torchvision", "torchaudio", "numpy", "opencv-python"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unified GPU preview runtime.")
    parser.add_argument("--runtime-root", default="/opt/three-dgs")
    parser.add_argument("--image-registry-output", default="/opt/three-dgs/runtime/image_algorithm_registry.generated.json")
    parser.add_argument("--video-registry-output", default="/opt/three-dgs/runtime/video_algorithm_registry.generated.json")
    parser.add_argument("--weight-cache-root", default="/model-cache")
    parser.add_argument("--repo-cache-root", default="/repo-cache")
    parser.add_argument("--workspace", default="/workspace")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root)
    repos_root = runtime_root / "repos"
    temp_root = runtime_root / "build"
    image_registry_output = Path(args.image_registry_output)
    video_registry_output = Path(args.video_registry_output)
    weight_cache_root = Path(args.weight_cache_root)
    repo_cache_root = Path(args.repo_cache_root)
    workspace = Path(args.workspace)

    repos_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    image_registry_output.parent.mkdir(parents=True, exist_ok=True)
    video_registry_output.parent.mkdir(parents=True, exist_ok=True)

    configure_git()
    litevggt = clone_checkout(
        LITEVGGT_REPO,
        repos_root / "LiteVGGT-repo",
        env_ref("LITEVGGT_COMMIT"),
        repo_cache_root=repo_cache_root,
    )
    edgs = clone_checkout(
        EDGS_REPO,
        repos_root / "EDGS",
        env_ref("EDGS_COMMIT"),
        repo_cache_root=repo_cache_root,
        recursive=True,
    )
    lingbot = clone_checkout(
        LINGBOT_REPO,
        repos_root / "lingbot-map",
        os.environ.get("LINGBOT_COMMIT", "main"),
        repo_cache_root=repo_cache_root,
    )
    spark = clone_checkout(
        SPARK_REPO,
        repos_root / "spark",
        env_ref("SPARK_COMMIT"),
        repo_cache_root=repo_cache_root,
    )

    if not args.skip_install:
        install_unified_python_runtime(
            litevggt=litevggt,
            edgs=edgs,
            lingbot=lingbot,
            temp_root=temp_root,
        )
        install_spark_runtime(spark)
        prune_spark_runtime(spark)

    litevggt_weight = resolve_litevggt_weight(weight_cache_root)
    lingbot_weight = resolve_lingbot_weight(weight_cache_root)
    write_image_registry(
        registry_output=image_registry_output,
        workspace=workspace,
        litevggt=litevggt,
        edgs=edgs,
        lingbot=lingbot,
        spark=spark,
        litevggt_weight=litevggt_weight,
        lingbot_weight=lingbot_weight,
    )
    write_video_registry(
        registry_output=video_registry_output,
        workspace=workspace,
        litevggt=litevggt,
        edgs=edgs,
        lingbot=lingbot,
        spark=spark,
        litevggt_weight=litevggt_weight,
        lingbot_weight=lingbot_weight,
    )
    return 0


def env_ref(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=str(cwd) if cwd else None, check=check, text=True)


def configure_git() -> None:
    run(["git", "config", "--global", "url.https://github.com/.insteadOf", "git@github.com:"], check=False)


def clone_checkout(url: str, target: Path, ref: str, *, repo_cache_root: Path | None = None, recursive: bool = False) -> Path:
    if target.exists() and not (target / ".git").exists():
        raise RuntimeError(f"{target} exists but is not a git repository")
    if not target.exists():
        cached = cached_repo_path(url, repo_cache_root)
        if cached:
            print(f"Using cached repository for {url}: {cached}", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(cached, target, symlinks=True)
        else:
            print(f"No local repo cache found for {url}; cloning with mirror fallback.", flush=True)
            clone_repo_with_fallback(url, target, recursive=recursive)
            run(["git", "-C", str(target), "remote", "set-url", "origin", url], check=False)
    if not git_has_ref(target, ref):
        fetch_ref_with_fallback(target, ref)
    run(["git", "-C", str(target), "checkout", ref])
    if recursive:
        run(["git", "-C", str(target), "submodule", "sync", "--recursive"], check=False)
        run_git_with_fallback(["-C", str(target), "submodule", "update", "--init", "--recursive"])
    return target.resolve()


def cached_repo_path(url: str, repo_cache_root: Path | None) -> Path | None:
    if repo_cache_root is None:
        return None
    candidate = repo_cache_root / repo_cache_name(url)
    if (candidate / ".git").exists():
        return candidate
    return None


def repo_cache_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name if parsed.path else url.rstrip("/").rsplit("/", 1)[-1]
    return name[:-4] if name.endswith(".git") else name


def git_has_ref(repo: Path, ref: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{ref}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.returncode == 0


def fetch_ref_with_fallback(repo: Path, ref: str) -> None:
    run_git_with_fallback(["-C", str(repo), "fetch", "origin", ref, "--depth", "1"])


def clone_repo_with_fallback(url: str, target: Path, *, recursive: bool = False) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for command in clone_attempt_commands(url, target, recursive=recursive):
        try:
            run(command)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if target.exists():
                shutil.rmtree(target)
            print(f"clone attempt failed for {url}: {exc}; trying next source", flush=True)
    raise RuntimeError(f"failed to clone {url}: {last_error}") from last_error


def run_git_with_fallback(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    last_result: subprocess.CompletedProcess[str] | None = None
    last_error: subprocess.CalledProcessError | None = None
    for command in git_attempt_commands(args):
        try:
            completed = run(command, check=check)
            if check or completed.returncode == 0:
                return completed
            last_result = completed
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"git attempt failed: {exc}; trying next source", flush=True)
    if check:
        raise RuntimeError(f"git command failed after all sources: {args}: {last_error}") from last_error
    return last_result or subprocess.CompletedProcess(args=args, returncode=1)


def clone_attempt_commands(url: str, target: Path, *, recursive: bool = False) -> list[list[str]]:
    commands: list[list[str]] = []
    repo_path = github_repo_path(url)
    if repo_path:
        for prefix in algorithm_repo_mirror_prefixes():
            command = ["clone"]
            if recursive:
                command.append("--recursive")
            command.extend([mirror_github_url(repo_path, prefix), str(target)])
            commands.append(git_command_with_mirror(prefix, command))
    command = ["clone"]
    if recursive:
        command.append("--recursive")
    command.extend([url, str(target)])
    commands.append(git_command_official(command))
    return commands


def git_attempt_commands(args: list[str]) -> list[list[str]]:
    commands = [git_command_with_mirror(prefix, args) for prefix in algorithm_repo_mirror_prefixes()]
    commands.append(git_command_official(args))
    return commands


def git_command_with_mirror(prefix: str, args: list[str]) -> list[str]:
    return [
        "git",
        "-c",
        f"url.{prefix}.insteadOf={GITHUB_HTTPS_PREFIX}",
        "-c",
        f"url.{prefix}.insteadOf={GITHUB_SSH_PREFIX}",
        *args,
    ]


def git_command_official(args: list[str]) -> list[str]:
    return ["git", "-c", f"url.{GITHUB_HTTPS_PREFIX}.insteadOf={GITHUB_SSH_PREFIX}", *args]


def algorithm_repo_mirror_prefixes() -> list[str]:
    raw = os.environ.get("ALGORITHM_REPO_MIRROR_PREFIXES", DEFAULT_ALGORITHM_REPO_MIRROR_PREFIXES)
    return [normalize_prefix(item) for item in split_csv(raw)]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_prefix(value: str) -> str:
    return value if value.endswith("/") else value + "/"


def github_repo_path(url: str) -> str | None:
    if url.startswith(GITHUB_HTTPS_PREFIX):
        return url[len(GITHUB_HTTPS_PREFIX) :]
    if url.startswith(GITHUB_SSH_PREFIX):
        return url[len(GITHUB_SSH_PREFIX) :]
    return None


def mirror_github_url(repo_path: str, prefix: str) -> str:
    return normalize_prefix(prefix) + repo_path


def install_unified_python_runtime(*, litevggt: Path, edgs: Path, lingbot: Path, temp_root: Path) -> None:
    pip_install(["--upgrade", "pip", "setuptools", "wheel"])
    backend_requirements = Path("/workspace/backend/requirements.txt")
    if backend_requirements.exists():
        pip_install(["-r", str(backend_requirements)])
    install_torch_runtime()
    install_litevggt_runtime(litevggt, temp_root)
    install_transformer_engine()
    install_edgs_runtime(edgs)
    install_lingbot_runtime(lingbot)
    pip_install(["numpy==1.26.4"])


def install_torch_runtime() -> None:
    torch_version = os.environ.get("TORCH_VERSION", DEFAULT_TORCH_VERSION).strip()
    torchvision_version = os.environ.get("TORCHVISION_VERSION", DEFAULT_TORCHVISION_VERSION).strip()
    torchaudio_version = os.environ.get("TORCHAUDIO_VERSION", DEFAULT_TORCHAUDIO_VERSION).strip()
    requirements = [
        torch_requirement("torch", torch_version),
        torch_requirement("torchvision", torchvision_version),
        torch_requirement("torchaudio", torchaudio_version),
    ]
    wheelhouse = Path(os.environ.get("TORCH_WHEELHOUSE", DEFAULT_TORCH_WHEELHOUSE))
    last_error: Exception | None = None
    for index_url in torch_index_urls():
        try:
            install_pip_requirements_via_wheelhouse(requirements=requirements, index_url=index_url, wheelhouse=wheelhouse)
            return
        except Exception as exc:
            last_error = exc
            print(f"Torch wheel install failed via {index_url}: {exc}; trying next index.", flush=True)
    raise RuntimeError(f"failed to install Torch runtime from configured indexes: {torch_index_urls()}") from last_error


def torch_requirement(package: str, version: str) -> str:
    suffix = os.environ.get("TORCH_CUDA_SUFFIX", DEFAULT_TORCH_CUDA_SUFFIX).strip()
    normalized = version.strip()
    if suffix and "+" not in normalized:
        normalized = f"{normalized}+{suffix}"
    return f"{package}=={normalized}"


def torch_index_urls() -> list[str]:
    raw = os.environ.get("TORCH_INDEX_URLS")
    if raw is None:
        raw = os.environ.get("TORCH_INDEX_URL", DEFAULT_TORCH_INDEX_URLS)
    urls = split_csv(raw)
    if not urls:
        raise RuntimeError("TORCH_INDEX_URLS cannot be empty")
    return urls


def pypi_index_url() -> str:
    return os.environ.get("PIP_INDEX_URL", DEFAULT_PIP_INDEX_URL).strip() or DEFAULT_PIP_INDEX_URL


def install_pip_requirements_via_wheelhouse(*, requirements: list[str], index_url: str, wheelhouse: Path) -> None:
    wheelhouse.mkdir(parents=True, exist_ok=True)
    download_report = wheelhouse / "download-report.json"
    last_error: Exception | None = None
    for label, command in pip_resolution_commands(requirements=requirements, index_url=index_url, report=download_report):
        try:
            download_report.unlink(missing_ok=True)
            print(f"Resolving wheels via {label}: {index_url}", flush=True)
            run(command)
            downloads = wheel_downloads_from_report(download_report)
            if not downloads:
                raise RuntimeError(f"pip dry-run did not produce wheel downloads for: {requirements}")
            for download in downloads:
                download_wheel_with_resume(
                    url=download["url"],
                    wheelhouse=wheelhouse,
                    expected_sha256=download.get("sha256"),
                    retries=DOWNLOAD_RETRIES,
                )
            pip_install(["--no-index", "--find-links", str(wheelhouse), *requirements])
            return
        except Exception as exc:
            last_error = exc
            print(f"Wheel resolution failed via {label} {index_url}: {exc}", flush=True)
    raise RuntimeError(f"failed to resolve wheels from {index_url} as index or find-links") from last_error


def pip_resolution_commands(*, requirements: list[str], index_url: str, report: Path) -> list[tuple[str, list[str]]]:
    common = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--only-binary=:all:",
        "--prefer-binary",
        "--retries",
        str(DOWNLOAD_RETRIES),
        "--timeout",
        "120",
        "--report",
        str(report),
    ]
    pypi_url = pypi_index_url()
    index_command = [*common, "--index-url", index_url]
    if pypi_url and pypi_url != index_url:
        index_command.extend(["--extra-index-url", pypi_url])
    find_links_command = [*common, "--index-url", pypi_url, "--find-links", index_url]
    return [
        ("index-url", [*index_command, *requirements]),
        ("find-links", [*find_links_command, *requirements]),
    ]


def wheel_downloads_from_report(path: Path) -> list[dict[str, str]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    downloads: list[dict[str, str]] = []
    for item in report.get("install", []):
        info = item.get("download_info") or {}
        url = str(info.get("url") or "")
        if not url:
            continue
        archive_info = info.get("archive_info") or {}
        hashes = archive_info.get("hashes") or {}
        digest = hashes.get("sha256")
        if not digest and isinstance(archive_info.get("hash"), str):
            algorithm, _, value = str(archive_info["hash"]).partition("=")
            if algorithm == "sha256":
                digest = value
        downloads.append({"url": url, **({"sha256": str(digest)} if digest else {})})
    return downloads


def download_wheel_with_resume(*, url: str, wheelhouse: Path, expected_sha256: str | None, retries: int) -> Path:
    filename = wheel_filename_from_url(url)
    target = wheelhouse / filename
    if wheel_is_complete(target, expected_sha256):
        print(f"wheel already cached: {target}", flush=True)
        return target

    part = target.with_suffix(target.suffix + ".part")
    if target.exists():
        target.unlink()

    last_error: Exception | None = None
    for attempt in range(1, max(retries, 1) + 1):
        try:
            expected_size = remote_content_length(url)
            download_url_with_resume(url, part, expected_size=expected_size)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise RuntimeError(f"incomplete wheel download: expected {expected_size}, got {part.stat().st_size}")
            if expected_sha256 and file_sha256(part) != expected_sha256:
                raise RuntimeError(f"sha256 mismatch for {filename}")
            part.replace(target)
            print(f"cached wheel: {target}", flush=True)
            return target
        except Exception as exc:
            last_error = exc
            if attempt >= max(retries, 1):
                break
            sleep_seconds = min(2 ** attempt, 30)
            print(f"wheel download attempt {attempt} failed for {filename}: {exc}; retrying in {sleep_seconds}s", flush=True)
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to download wheel after {retries} attempts: {filename}: {last_error}") from last_error


def wheel_filename_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    filename = Path(urllib.parse.unquote(parsed.path)).name
    if not filename.endswith(".whl"):
        raise RuntimeError(f"download URL does not look like a wheel: {url}")
    return filename


def wheel_is_complete(path: Path, expected_sha256: str | None) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    return not expected_sha256 or file_sha256(path) == expected_sha256


def remote_content_length(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None


def download_url_with_resume(url: str, part: Path, *, expected_size: int | None) -> None:
    part.parent.mkdir(parents=True, exist_ok=True)
    existing_size = part.stat().st_size if part.exists() else 0
    if expected_size is not None and existing_size > expected_size:
        part.unlink()
        existing_size = 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading wheel {url} -> {part} (resume_at={existing_size})", flush=True)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        append = existing_size > 0 and status == 206
        mode = "ab" if append else "wb"
        with part.open(mode) as handle:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def install_litevggt_runtime(litevggt: Path, temp_root: Path) -> None:
    source = litevggt / "requirements.txt"
    filtered = filter_litevggt_requirements(source, temp_root / "litevggt-requirements.filtered.txt")
    if filtered.exists() and filtered.stat().st_size > 0:
        pip_install(["-r", str(filtered)])


def filter_litevggt_requirements(source: Path, target: Path) -> Path:
    requirements = parse_requirement_tokens(source.read_text(encoding="utf-8"))
    kept = [requirement for requirement in requirements if not should_skip_litevggt_requirement(requirement)]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    skipped = sorted(set(requirements) - set(kept))
    print(f"LiteVGGT requirements filtered: kept={len(kept)} skipped={skipped}", flush=True)
    return target


def parse_requirement_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        tokens.extend(part.strip() for part in clean.split() if part.strip())
    return tokens


def should_skip_litevggt_requirement(requirement: str) -> bool:
    name = requirement_name(requirement)
    return name in LITEVGGT_SKIPPED_REQUIREMENTS


def requirement_name(requirement: str) -> str:
    normalized = requirement.strip().split(";", 1)[0].strip()
    if "@" in normalized and not normalized.startswith(("http://", "https://", "git+")):
        normalized = normalized.split("@", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9_.-]+)", normalized)
    return match.group(1).lower().replace("_", "-") if match else normalized.lower()


def install_transformer_engine() -> None:
    version = os.environ.get("TRANSFORMER_ENGINE_VERSION", DEFAULT_TRANSFORMER_ENGINE_VERSION).strip()
    if not version:
        raise RuntimeError("TRANSFORMER_ENGINE_VERSION cannot be empty")
    pip_uninstall(["transformer-engine-cu13"])
    pip_install([f"transformer-engine=={version}", f"transformer-engine-cu12=={version}"])
    pip_install(["--no-build-isolation", f"transformer-engine-torch=={version}"])
    validate_transformer_engine_packages(version)


def validate_transformer_engine_packages(version: str) -> None:
    expected = {
        "transformer-engine": version,
        "transformer-engine-cu12": version,
        "transformer-engine-torch": version,
    }
    installed = {name: installed_package_version(name) for name in expected}
    mismatched = {name: found for name, found in installed.items() if found != expected[name]}
    cu13_version = installed_package_version("transformer-engine-cu13")
    if mismatched or cu13_version is not None:
        raise RuntimeError(
            "Transformer Engine package mismatch: "
            f"expected={expected}, installed={installed}, transformer-engine-cu13={cu13_version}"
        )
    print(f"Transformer Engine packages pinned to CUDA 12 version {version}", flush=True)


def installed_package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def install_edgs_runtime(edgs: Path) -> None:
    diff_raster = edgs / "submodules" / "gaussian-splatting" / "submodules" / "diff-gaussian-rasterization"
    simple_knn = edgs / "submodules" / "gaussian-splatting" / "submodules" / "simple-knn"
    roma = edgs / "submodules" / "RoMa"
    if diff_raster.exists():
        install_edgs_extension("diff_gaussian_rasterization", diff_raster)
    if simple_knn.exists():
        install_edgs_extension("simple_knn", simple_knn)
    pip_install(
        [
            "pycolmap",
            "wandb",
            "hydra-core",
            "tqdm",
            "torchmetrics",
            "lpips",
            "matplotlib",
            "rich",
            "plyfile",
            "imageio",
            "imageio-ffmpeg",
            "numpy==1.26.4",
        ]
    )
    if roma.exists():
        pip_install(["--no-build-isolation", "-e", str(roma)])


def install_edgs_extension(package: str, source_path: Path) -> None:
    mode = os.environ.get("EDGS_EXTENSION_INSTALL_MODE", "source").strip().lower()
    if mode not in {"auto", "wheel", "source"}:
        raise RuntimeError(f"Unsupported EDGS_EXTENSION_INSTALL_MODE={mode!r}; expected auto, wheel, or source")

    wheel = EDGS_WHEELS.get(package)
    can_use_official_wheel = (
        wheel is not None
        and sys.version_info[:2] == (3, 10)
        and platform.machine() == "x86_64"
        and sys.platform.startswith("linux")
    )
    if mode in {"auto", "wheel"} and can_use_official_wheel:
        try:
            pip_install([edgs_wheel_url(wheel)])
            return
        except subprocess.CalledProcessError:
            if mode == "wheel":
                raise
            print(f"Official EDGS wheel failed for {package}; falling back to local CUDA build.", flush=True)
    elif mode == "wheel":
        raise RuntimeError(f"No official EDGS wheel is configured for {package} on this Python/platform")

    compile_edgs_extension(package, source_path)


def edgs_wheel_url(wheel: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").strip().rstrip("/")
    return f"{endpoint}/{EDGS_WHEEL_REPO_PATH}/{wheel}"


def compile_edgs_extension(package: str, source_path: Path) -> None:
    if not source_path.exists():
        raise RuntimeError(f"EDGS extension source is missing for {package}: {source_path}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Cannot compile {package}: torch must be installed first") from exc
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.6+PTX")
    os.environ.setdefault("MAX_JOBS", str(max((os.cpu_count() or 2) // 2, 1)))
    pip_uninstall([package])
    pip_install(["--no-build-isolation", str(source_path)])


def install_lingbot_runtime(lingbot: Path) -> None:
    pip_install(["opencv-python-headless", "numpy==1.26.4", "Pillow", "huggingface_hub", "einops", "safetensors", "tqdm", "scipy"])
    pip_install(["--no-build-isolation", "--no-deps", "-e", str(lingbot)])
    if os.environ.get("INSTALL_FLASHINFER", "false").strip().lower() in {"1", "true", "yes", "on"}:
        pip_install(["-i", os.environ.get("FLASHINFER_INDEX_URL", "https://pypi.org/simple"), "flashinfer-python"])


def install_spark_runtime(spark: Path) -> None:
    registry = os.environ.get("NPM_CONFIG_REGISTRY", "https://registry.npmmirror.com")
    run(["npm", "config", "set", "registry", registry])
    normalize_spark_shell_scripts(spark)
    run(["npm", "ci"], cwd=spark)
    run(["npm", "run", "build"], cwd=spark)


def normalize_spark_shell_scripts(spark: Path) -> None:
    for script in spark.rglob("*.sh"):
        content = script.read_bytes()
        normalized = content.replace(b"\r\n", b"\n")
        if normalized != content:
            script.write_bytes(normalized)


def prune_spark_runtime(spark: Path) -> None:
    for relative in [
        "node_modules",
        "rust/target",
        "rust/spark-rs/target",
        "rust/spark-worker-rs/target",
        "rust/spark-rs/pkg",
        "rust/spark-worker-rs/pkg",
        "examples/assets",
    ]:
        shutil.rmtree(spark / relative, ignore_errors=True)


def pip_install(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", *args])


def pip_uninstall(args: list[str]) -> None:
    run([sys.executable, "-m", "pip", "uninstall", "-y", *args], check=False)


def resolve_litevggt_weight(weight_cache_root: Path) -> Path:
    cached = weight_cache_root / "litevggt" / LITEVGGT_WEIGHT_FILE
    if cached.exists() and cached.stat().st_size > 0:
        return cached.resolve()
    return Path("/model-cache/litevggt") / LITEVGGT_WEIGHT_FILE


def resolve_lingbot_weight(weight_cache_root: Path) -> Path:
    configured = os.environ.get("LINGBOT_MODEL_PATH")
    if configured and Path(configured).exists() and Path(configured).stat().st_size > 0:
        return Path(configured).resolve()
    cached = weight_cache_root / "lingbot-map" / LINGBOT_WEIGHT_FILE
    if cached.exists() and cached.stat().st_size > 0:
        return cached.resolve()
    return Path(configured or f"/model-cache/lingbot-map/{LINGBOT_WEIGHT_FILE}")


def commit_hash(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def write_image_registry(
    *,
    registry_output: Path,
    workspace: Path,
    litevggt: Path,
    edgs: Path,
    lingbot: Path,
    spark: Path,
    litevggt_weight: Path,
    lingbot_weight: Path,
) -> None:
    registry_output.write_text(
        json.dumps(
            {"algorithms": image_registry_algorithms(workspace, litevggt, edgs, lingbot, spark, litevggt_weight, lingbot_weight)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_video_registry(
    *,
    registry_output: Path,
    workspace: Path,
    litevggt: Path,
    edgs: Path,
    lingbot: Path,
    spark: Path,
    litevggt_weight: Path,
    lingbot_weight: Path,
) -> None:
    registry_output.write_text(
        json.dumps(
            {"algorithms": video_registry_algorithms(workspace, litevggt, edgs, lingbot, spark, litevggt_weight, lingbot_weight)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def image_registry_algorithms(
    workspace: Path,
    litevggt: Path,
    edgs: Path,
    lingbot: Path,
    spark: Path,
    litevggt_weight: Path,
    lingbot_weight: Path,
) -> list[dict[str, object]]:
    script_dir = workspace / "backend" / "scripts"
    return [
        litevggt_entry(script_dir, litevggt, litevggt_weight, enabled=True),
        edgs_entry(script_dir, edgs, enabled=True),
        spark_entry(script_dir, spark, enabled=True),
        ffmpeg_entry(script_dir),
        lingbot_entry(script_dir, lingbot, lingbot_weight, enabled=False),
    ]


def video_registry_algorithms(
    workspace: Path,
    litevggt: Path,
    edgs: Path,
    lingbot: Path,
    spark: Path,
    litevggt_weight: Path,
    lingbot_weight: Path,
) -> list[dict[str, object]]:
    script_dir = workspace / "backend" / "scripts"
    return [
        litevggt_entry(script_dir, litevggt, litevggt_weight, enabled=False),
        edgs_entry(script_dir, edgs, enabled=False),
        spark_entry(script_dir, spark, enabled=True),
        ffmpeg_entry(script_dir),
        lingbot_entry(script_dir, lingbot, lingbot_weight, enabled=True),
    ]


def litevggt_entry(script_dir: Path, litevggt: Path, weight: Path, *, enabled: bool) -> dict[str, object]:
    return {
        "name": "LiteVGGT",
        "repo_url": "https://github.com/GarlicBa/LiteVGGT-repo",
        "license": "MIT",
        "commit_hash": commit_hash(litevggt),
        "weight_source": f"{os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com').rstrip('/')}/{LITEVGGT_WEIGHT_REPO}/resolve/main/{LITEVGGT_WEIGHT_FILE}",
        "local_path": str(litevggt),
        "enabled": enabled,
        "notes": "Unified GPU runtime image geometry stage.",
        "weight_paths": [str(weight)],
        "commands": {"run_demo": ["python3", str(script_dir / "run_litevggt_preview.py")]},
    }


def edgs_entry(script_dir: Path, edgs: Path, *, enabled: bool) -> dict[str, object]:
    return {
        "name": "EDGS",
        "repo_url": "https://github.com/CompVis/EDGS",
        "license": EDGS_LICENSE,
        "commit_hash": commit_hash(edgs),
        "weight_source": None,
        "local_path": str(edgs),
        "enabled": enabled,
        "notes": "Unified GPU runtime image Gaussian training stage.",
        "weight_paths": [],
        "commands": {"train": ["python3", str(script_dir / "run_edgs_preview.py")]},
    }


def spark_entry(script_dir: Path, spark: Path, *, enabled: bool) -> dict[str, object]:
    return {
        "name": "Spark-SPZ",
        "repo_url": "https://github.com/sparkjsdev/spark",
        "license": "MIT",
        "commit_hash": commit_hash(spark),
        "weight_source": None,
        "local_path": str(spark),
        "enabled": enabled,
        "notes": "Unified GPU runtime local Spark/SPZ conversion stage.",
        "weight_paths": [],
        "commands": {"compress": ["python3", str(script_dir / "run_spz_convert.py")]},
    }


def ffmpeg_entry(script_dir: Path) -> dict[str, object]:
    return {
        "name": "FFmpeg",
        "repo_url": "https://ffmpeg.org/",
        "license": "LGPL/GPL depending on build configuration",
        "commit_hash": "system-package",
        "weight_source": None,
        "local_path": None,
        "enabled": shutil.which("ffmpeg") is not None,
        "notes": "Legacy video utility. Default LingBot preview uses native video_path input.",
        "weight_paths": [],
        "source_type": "command",
        "commands": {"extract_frames": ["python3", str(script_dir / "run_ffmpeg_extract.py")]},
    }


def lingbot_entry(script_dir: Path, lingbot: Path, weight: Path, *, enabled: bool) -> dict[str, object]:
    return {
        "name": "LingBot-Map",
        "repo_url": "https://github.com/Robbyant/lingbot-map",
        "license": LINGBOT_LICENSE,
        "commit_hash": commit_hash(lingbot),
        "weight_source": "local model-cache/lingbot-map/lingbot-map-long.pt",
        "local_path": str(lingbot),
        "enabled": enabled,
        "notes": "Unified GPU runtime video and camera preview stage.",
        "weight_paths": [str(weight)],
        "commands": {"run_preview": ["python3", str(script_dir / "run_lingbot_map_preview.py")]},
    }


if __name__ == "__main__":
    raise SystemExit(main())
