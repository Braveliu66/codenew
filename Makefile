COMPOSE_FILE ?= deploy/docker-compose.preview.yml
MODEL_CACHE_ROOT ?= model-cache
MODEL_WEIGHTS ?= litevggt lingbot-map
API_BASE_IMAGE ?= python:3.12-slim
PREVIEW_CUDA_BASE_IMAGE ?= nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
LINGBOT_CUDA_BASE_IMAGE ?= nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

export API_BASE_IMAGE
export PREVIEW_CUDA_BASE_IMAGE
export LINGBOT_CUDA_BASE_IMAGE

.PHONY: weights base-images deploy up down preflight-image preflight-video preflight-camera

weights:
	python backend/scripts/download_model_weights.py --cache-root $(MODEL_CACHE_ROOT) --models $(MODEL_WEIGHTS)

base-images:
	python backend/scripts/pull_base_images.py --images $(API_BASE_IMAGE) $(PREVIEW_CUDA_BASE_IMAGE) $(LINGBOT_CUDA_BASE_IMAGE)

deploy: weights base-images
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
