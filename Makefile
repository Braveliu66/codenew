COMPOSE_FILE ?= deploy/docker-compose.preview.yml
MODEL_CACHE_ROOT ?= model-cache
MODEL_WEIGHTS ?= litevggt lingbot-map
ALGORITHM_REPO_CACHE_ROOT ?= repo-cache
ALGORITHM_REPOS ?= litevggt edgs lingbot-map spark
API_BASE_IMAGE ?= python:3.12-slim
DEBIAN_APT_MIRROR ?= https://mirrors.tuna.tsinghua.edu.cn/debian
DEBIAN_APT_SECURITY_MIRROR ?= https://mirrors.tuna.tsinghua.edu.cn/debian-security
UBUNTU_APT_MIRROR ?= https://mirrors.tuna.tsinghua.edu.cn/ubuntu
GPU_CUDA_BUILD_IMAGE ?= nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
GPU_CUDA_RUNTIME_IMAGE ?= nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

export API_BASE_IMAGE
export DEBIAN_APT_MIRROR
export DEBIAN_APT_SECURITY_MIRROR
export UBUNTU_APT_MIRROR
export GPU_CUDA_BUILD_IMAGE
export GPU_CUDA_RUNTIME_IMAGE

.PHONY: weights algorithm-repos base-images build-gpu-runtime rebuild-gpu-runtime rebuild-gpu-runtime-no-cache deploy up down preflight-image preflight-video preflight-camera

weights:
	python backend/scripts/download_model_weights.py --cache-root $(MODEL_CACHE_ROOT) --models $(MODEL_WEIGHTS)

algorithm-repos:
	python backend/scripts/download_algorithm_repos.py --cache-root $(ALGORITHM_REPO_CACHE_ROOT) --repos $(ALGORITHM_REPOS)

base-images:
	python backend/scripts/pull_base_images.py --images $(API_BASE_IMAGE) $(GPU_CUDA_BUILD_IMAGE) $(GPU_CUDA_RUNTIME_IMAGE)

build-gpu-runtime: algorithm-repos
	docker compose -f $(COMPOSE_FILE) build image-worker

rebuild-gpu-runtime: algorithm-repos
	docker compose -f $(COMPOSE_FILE) build image-worker

rebuild-gpu-runtime-no-cache: algorithm-repos
	docker compose -f $(COMPOSE_FILE) build --no-cache image-worker

deploy: weights algorithm-repos base-images
	docker compose -f $(COMPOSE_FILE) up -d --build

up:
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down

preflight-image:
	docker compose -f $(COMPOSE_FILE) run --rm image-worker python -m backend.scripts.check_preview_runtime

preflight-video:
	docker compose -f $(COMPOSE_FILE) run --rm video-worker python -m backend.scripts.check_preview_runtime

preflight-camera:
	docker compose -f $(COMPOSE_FILE) run --rm camera-worker python -m backend.scripts.check_preview_runtime
