# 部署与运行说明

本文档记录当前 Docker/WSL GPU 预览部署方式。目标是单机 GPU 先打通真实预览链路：图片走 LiteVGGT/EDGS，视频和实时摄像头走 LingBot-Map，最终转换为 SPZ 并在前端查看。

## 服务组成

`deploy/docker-compose.preview.yml` 包含：

- `backend`：轻量 FastAPI API 镜像，只运行 API、迁移和对象存储访问，不携带大算法依赖。
- `image-worker`：独立图片 GPU preview worker，运行 LiteVGGT/EDGS/Spark 命令。
- `video-worker`：独立视频 GPU preview worker，运行 LingBot-Map/Spark 命令。
- `camera-worker`：独立实时摄像头 GPU preview worker，消费 `preview_camera_tasks`，运行 LingBot-Map streaming/Spark 命令。
- `postgres`、`redis`、`minio`、`frontend`：分别提供元数据、队列、对象存储和 Next.js 前端。

## 为什么 Docker 中断后会从头下载

Docker build 的每个 `RUN` 是一个镜像层。大权重如果在某个 `RUN` 中下载，下载中断后该层不会提交，容器临时文件也会丢失；下一次 build 只能重新开始。解决方式是把权重下载移到宿主机/共享卷 `model-cache`，并用 `.part` 文件与 HTTP Range 续传。

当前规则：

- Docker build 只安装系统依赖、Python/CUDA 依赖、算法仓库和 registry，不远端下载大模型权重。
- Worker 启动前执行 `backend.scripts.ensure_model_weights`，检查共享 `/model-cache`。
- 缺失权重时，下载脚本写入 `*.part` 并支持 Range 续传；完成后原子替换为正式权重文件。
- 多个 Worker 同时启动时使用 lock 文件避免重复下载。

## 基础镜像拉取 EOF 排障

如果错误出现在类似 `[image-worker stage-0 2/13] WORKDIR /workspace`，并且日志里有 `httpReadSeeker`、`failed to copy`、`image-mirror.r2.daocloud.vip`、`EOF`，根因不是 Dockerfile 的 `WORKDIR`，也不是模型权重下载，而是 Docker 在拉取基础镜像 layer 时镜像源中断。

处理方式：

```bash
python backend/scripts/pull_base_images.py --retries 5
docker compose -f deploy/docker-compose.preview.yml build image-worker
```

`make deploy` 会先执行模型权重预检，再预拉取基础镜像，最后执行 compose build。如果 DaoCloud 镜像持续 EOF，需要在 Docker Desktop/daemon 中删除或替换故障 registry mirror；或者显式指定可访问的基础镜像：

```bash
$env:API_BASE_IMAGE="python:3.12-slim"
$env:PREVIEW_CUDA_BASE_IMAGE="nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04"
$env:LINGBOT_CUDA_BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"
```

这些变量会传给 `backend/Dockerfile.api`、`backend/Dockerfile.preview` 和 `backend/Dockerfile.lingbot_preview` 的 `FROM`。注意：如果 Docker daemon 配置了损坏的 registry mirror，同名 Docker Hub 镜像仍可能被 daemon 代理到该 mirror，需要修复 daemon 的 mirror 配置。

## 模型缓存

缓存路径：

```text
model-cache/litevggt/te_dict.pt
model-cache/lingbot-map/lingbot-map-long.pt
```

准备缓存：

```bash
python backend/scripts/download_model_weights.py --cache-root model-cache --models litevggt lingbot-map
```

如果默认 Hugging Face 路径不适合当前网络，可设置：

```bash
$env:HF_ENDPOINT="https://hf-mirror.com"
$env:LITEVGGT_WEIGHT_URL="<direct-url>"
$env:LINGBOT_WEIGHT_URL="<direct-url>"
```

## 一键部署

推荐：

```bash
make deploy
```

等价于：

```bash
python backend/scripts/download_model_weights.py --cache-root model-cache --models litevggt lingbot-map
python backend/scripts/pull_base_images.py --images python:3.12-slim nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04 nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
docker compose -f deploy/docker-compose.preview.yml up -d --build
```

如果本机没有 `make`：

```bash
python backend/scripts/download_model_weights.py --cache-root model-cache --models litevggt lingbot-map
python backend/scripts/pull_base_images.py --retries 5
docker compose -f deploy/docker-compose.preview.yml up -d --build
```

Worker 启动时仍会自动补齐缺失权重。

## GPU 与依赖隔离

- `backend/Dockerfile.api`：API 镜像，Python 3.12 slim，不占用 GPU。
- `backend/Dockerfile.preview`：图片 Worker 镜像，Python 3.12 + CUDA 12.6，包含 LiteVGGT、EDGS、Spark。
- `backend/Dockerfile.lingbot_preview`：视频/摄像头 Worker 镜像，Python 3.10 + CUDA 12.8 + PyTorch 2.8 cu128，包含 LingBot-Map、Spark。
- Compose 通过 NVIDIA Container Toolkit 的 `deploy.resources.reservations.devices` 透传 GPU 给 Worker。
- 新增算法时按固定模式扩展：新增 Dockerfile 或复用算法镜像、新增 compose 服务块、新增 registry 能力和 Worker 队列。

## 实时摄像头与渐进式预览

实时摄像头路径：

```text
Browser MediaRecorder -> POST /api/projects/{id}/camera/chunks
-> preview_camera_tasks -> camera-worker
-> LingBot-Map streaming -> Spark-SPZ
-> preview_spz_segment artifact -> SSE preview_segment_ready
-> Viewer 增量加载
```

`GET /api/projects/{project_id}/viewer-config` 返回：

- `mode=single`：传统单文件 `preview.spz`。
- `mode=progressive`：多个 `preview_segment_*.spz`，前端按时间线加载。

## 常用命令

```bash
docker compose -f deploy/docker-compose.preview.yml up -d
docker compose -f deploy/docker-compose.preview.yml restart image-worker video-worker camera-worker
docker compose -f deploy/docker-compose.preview.yml run --rm image-worker python -m backend.scripts.check_preview_runtime
docker compose -f deploy/docker-compose.preview.yml run --rm video-worker python -m backend.scripts.check_preview_runtime
docker compose -f deploy/docker-compose.preview.yml run --rm camera-worker python -m backend.scripts.check_preview_runtime
```

服务地址：

- 前端：http://localhost:3001
- 后端健康检查：http://localhost:8000/health
- MinIO Console：http://localhost:9001

## 验收标准

- 图片任务成功后 MinIO 中存在非空 `preview.spz`，viewer-config 返回 `mode=single`。
- 视频任务使用 LingBot-Map，不回退旧 FFmpeg -> LiteVGGT -> EDGS 管线。
- 实时摄像头页面能启动摄像头、分片上传、创建 camera preview task，并在 segment 完成后通过 SSE 刷新 Viewer。
- Viewer 以 90 FPS 为目标自适应质量；progressive 模式支持时间线和增量加载。
