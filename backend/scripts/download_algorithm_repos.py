from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


LITEVGGT_REPO = "https://github.com/GarlicBa/LiteVGGT-repo.git"
EDGS_REPO = "https://github.com/CompVis/EDGS.git"
LINGBOT_REPO = "https://github.com/Robbyant/lingbot-map.git"
SPARK_REPO = "https://github.com/sparkjsdev/spark.git"

DEFAULT_LITEVGGT_REF = "4767c17f8b6f176bb751566e92f60eb885040033"
DEFAULT_EDGS_REF = "9a897645eb47c1b24d4f9e4428cd745927bf1ee1"
DEFAULT_LINGBOT_REF = "main"
DEFAULT_SPARK_REF = "915c474795e0c78f7cd1b7f4eb97695028b495c0"

GITHUB_HTTPS_PREFIX = "https://github.com/"
GITHUB_SSH_PREFIX = "git@github.com:"
DEFAULT_ALGORITHM_REPO_MIRROR_PREFIXES = ""


@dataclass(frozen=True)
class AlgorithmRepo:
    key: str
    url: str
    cache_subdir: str
    env_ref: str
    default_ref: str
    recursive: bool = False

    def ref(self) -> str:
        return os.environ.get(self.env_ref, self.default_ref).strip()

    def target_path(self, cache_root: Path) -> Path:
        return cache_root / self.cache_subdir


ALGORITHM_REPOS = {
    "litevggt": AlgorithmRepo("litevggt", LITEVGGT_REPO, "LiteVGGT-repo", "LITEVGGT_COMMIT", DEFAULT_LITEVGGT_REF),
    "edgs": AlgorithmRepo("edgs", EDGS_REPO, "EDGS", "EDGS_COMMIT", DEFAULT_EDGS_REF, recursive=True),
    "lingbot-map": AlgorithmRepo("lingbot-map", LINGBOT_REPO, "lingbot-map", "LINGBOT_COMMIT", DEFAULT_LINGBOT_REF),
    "spark": AlgorithmRepo("spark", SPARK_REPO, "spark", "SPARK_COMMIT", DEFAULT_SPARK_REF),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download algorithm repositories into a reusable local cache.")
    parser.add_argument("--cache-root", default=os.environ.get("ALGORITHM_REPO_CACHE_ROOT", "repo-cache"))
    parser.add_argument(
        "--repos",
        nargs="+",
        choices=sorted(ALGORITHM_REPOS),
        default=sorted(ALGORITHM_REPOS),
        help="Algorithm repositories to ensure in the cache.",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root)
    for repo_key in args.repos:
        ensure_algorithm_repo(ALGORITHM_REPOS[repo_key], cache_root)
    return 0


def ensure_algorithm_repo(repo: AlgorithmRepo, cache_root: Path) -> Path:
    target = repo.target_path(cache_root)
    ref = repo.ref()
    if not ref:
        raise RuntimeError(f"{repo.env_ref} cannot be empty")

    if (target / ".git").exists():
        print(f"{repo.key} repository already cached: {target}", flush=True)
    elif target.exists() and any(target.iterdir()):
        raise RuntimeError(f"{target} exists but is not a git repository; move it aside or delete it before retrying")
    else:
        temp = target.with_name(f".{target.name}.tmp")
        if temp.exists():
            shutil.rmtree(temp)
        target.parent.mkdir(parents=True, exist_ok=True)
        clone_repo_with_fallback(repo.url, temp)
        run(["git", "-C", str(temp), "remote", "set-url", "origin", repo.url])
        if target.exists():
            target.rmdir()
        temp.rename(target)

    try:
        run_git_with_fallback(["-C", str(target), "fetch", "origin", ref], action=f"fetch {repo.key}")
    except RuntimeError:
        if not git_has_ref(target, ref):
            raise
        print(f"{repo.key} fetch failed; using existing cached ref {ref}", flush=True)

    run(["git", "-C", str(target), "checkout", ref])
    if repo.recursive:
        update_submodules_with_fallback(target)
    print(f"Cached {repo.key}: {target.resolve()} @ {ref}", flush=True)
    return target.resolve()


def clone_repo_with_fallback(url: str, target: Path) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for command in clone_attempt_commands(url, target):
        try:
            run(command)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if target.exists():
                shutil.rmtree(target)
            print(f"clone attempt failed: {exc}; trying next source", flush=True)
    raise RuntimeError(f"failed to clone {url}: {last_error}") from last_error


def update_submodules_with_fallback(repo: Path) -> None:
    run(["git", "-C", str(repo), "submodule", "sync", "--recursive"], check=False)
    run_git_with_fallback(
        ["-C", str(repo), "submodule", "update", "--init", "--recursive"],
        action=f"update submodules for {repo}",
    )


def run_git_with_fallback(args: list[str], *, action: str) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for command in git_attempt_commands(args):
        try:
            run(command)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print(f"{action} failed: {exc}; trying next source", flush=True)
    raise RuntimeError(f"{action} failed after all git sources: {last_error}") from last_error


def clone_attempt_commands(url: str, target: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    repo_path = github_repo_path(url)
    if repo_path:
        for prefix in algorithm_repo_mirror_prefixes():
            commands.append(git_command_with_mirror(prefix, ["clone", mirror_github_url(repo_path, prefix), str(target)]))
    commands.append(git_command_official(["clone", url, str(target)]))
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


def git_has_ref(repo: Path, ref: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{ref}^{{commit}}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.returncode == 0


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, check=check, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
