# 部署用户使用说明

本文用于指导部署人员在本机或服务器上启动 3DGS 预览平台。系统通过 Docker Compose 运行前端、后端、数据库、对象存储、队列和 GPU worker。

## 1. 部署前准备

请先确认环境已具备：

- Docker Desktop 或 Docker Engine，并已启动 Docker。
- NVIDIA 显卡驱动和 NVIDIA Container Toolkit，容器内需要能访问 GPU。
- Python 3.10 及以上，用于执行权重、源码和镜像预拉脚本。
- 项目代码已完整下载到本地。
- 首次部署需要网络访问，用于下载模型权重、算法源码、Docker 基础镜像和依赖包。

Windows 推荐在 PowerShell 或 WSL 中执行命令；Linux 服务器可直接在项目根目录执行。

## 2. 一键部署

在项目根目录执行：

```bash
make deploy
```

该命令会自动完成：

- 下载模型权重到 `model-cache/`。
- 下载算法源码到 `repo-cache/`。
- 预拉 Docker 基础镜像。
- 构建并启动全部服务。

如果当前环境没有 `make`，可按顺序执行：

```bash
python backend/scripts/download_model_weights.py --cache-root model-cache --models litevggt lingbot-map
python backend/scripts/download_algorithm_repos.py --cache-root repo-cache --repos litevggt edgs lingbot-map spark
python backend/scripts/pull_base_images.py --images python:3.12-slim nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
docker compose -f deploy/docker-compose.preview.yml up -d --build
```

首次构建 GPU runtime 时间较长，属于正常情况。

## 3. 访问地址

服务启动后访问：

- 前端页面：`http://localhost:3001`
- 后端接口：`http://localhost:8000`
- 后端健康检查：`http://localhost:8000/health`
- MinIO 控制台：`http://localhost:9001`

默认登录账号：

```text
用户名：admin
密码：admin123
```

MinIO 默认账号：

```text
用户名：minioadmin
密码：minioadmin
```

生产环境部署前请修改 `deploy/docker-compose.preview.yml` 中的 `JWT_SECRET`、`DEFAULT_ADMIN_PASSWORD`、`MINIO_ROOT_USER` 和 `MINIO_ROOT_PASSWORD`。

## 4. 常用运维命令

启动服务：

```bash
docker compose -f deploy/docker-compose.preview.yml up -d
```

停止服务：

```bash
docker compose -f deploy/docker-compose.preview.yml down
```

重新构建并启动：

```bash
docker compose -f deploy/docker-compose.preview.yml up -d --build
```

重启 GPU worker：

```bash
docker compose -f deploy/docker-compose.preview.yml restart image-worker video-worker camera-worker
```

查看服务状态：

```bash
docker compose -f deploy/docker-compose.preview.yml ps
```

查看日志：

```bash
docker compose -f deploy/docker-compose.preview.yml logs -f backend
docker compose -f deploy/docker-compose.preview.yml logs -f image-worker
docker compose -f deploy/docker-compose.preview.yml logs -f video-worker
docker compose -f deploy/docker-compose.preview.yml logs -f camera-worker
```

## 5. 部署后检查

建议首次部署后执行预检：

```bash
docker compose -f deploy/docker-compose.preview.yml run --rm image-worker python -m backend.scripts.check_preview_runtime
docker compose -f deploy/docker-compose.preview.yml run --rm video-worker python -m backend.scripts.check_preview_runtime
docker compose -f deploy/docker-compose.preview.yml run --rm camera-worker python -m backend.scripts.check_preview_runtime
```

如果预检通过，再在前端页面上传图片、视频或摄像头数据创建预览任务。

## 6. 数据与缓存目录

部署过程中会使用以下本地目录：

- `model-cache/`：模型权重缓存。
- `repo-cache/`：算法源码缓存。
- `storage/`：运行时生成的本地文件。
- `.runtime/`：运行时临时状态。
- Docker volume：PostgreSQL、MinIO 和前端依赖缓存。

如需迁移部署，优先保留 `model-cache/`、`repo-cache/`、`storage/` 和数据库、MinIO 数据卷。

## 7. Windows 镜像保存与恢复

如需在重装系统前保存本地 Docker 镜像，可执行：

```powershell
powershell -ExecutionPolicy Bypass -File deploy/docker-preview-boot.ps1 -Mode Save
```

重装 Docker Desktop 后恢复并启动：

```powershell
powershell -ExecutionPolicy Bypass -File deploy/docker-preview-boot.ps1
```

默认镜像归档路径为 `Q:\docker-image-cache\three-dgs-preview-images.tar`。

## 8. 常见问题

基础镜像拉取失败或出现 EOF：

```bash
python backend/scripts/pull_base_images.py --retries 5
```

模型下载中断：

- 重新执行部署命令即可续传。
- 确认 `model-cache/` 目录可写。

GPU worker 启动失败：

- 确认宿主机 NVIDIA 驱动可用。
- 确认 Docker 已启用 GPU 支持。
- 执行第 5 节预检命令查看具体错误。

前端无法访问后端：

- 确认 `backend` 服务已启动。
- 确认 `http://localhost:8000` 可访问。
- 确认前端环境变量 `NEXT_PUBLIC_API_BASE_URL` 指向 `http://localhost:8000`。

## 9. 精细重建部署补充

当前 compose 已加入 `fine-worker`，但精细重建真实运行还需要额外配置真实算法命令。未配置时任务会失败，不会生成假产物。

### 9.1 必需命令

部署精细重建前至少需要确认：

- `Faster-GS.fine_engine`：调用 `backend/scripts/run_fused3dgs_fine.py`。
- `Spark-SPZ.compress_final`：调用 `backend/scripts/run_final_spz_convert.py`。
- `RAD-LOD.export_rad`：调用 `backend/scripts/run_lod_export.py`。

同时需要在运行环境中配置：

```bash
FUSED3DGS_TRAIN_COMMAND="真实训练命令，必须输出 final.ply"
SPZ_CONVERTER_COMMAND="可选；真实 final.ply -> final_web.spz 转换命令"
RAD_LOD_EXPORT_COMMAND="真实 final.ply -> final_lod0..3.rad 导出命令"
```

如果 `RAD_LOD_EXPORT_COMMAND` 未配置，LOD 阶段会失败，系统不会创建占位 `.rad` 文件。

### 9.2 GPU 环境要求

精细重建部署必须在带 NVIDIA GPU 的环境验证：

- 宿主机 NVIDIA driver 可用。
- Docker 能访问 GPU。
- Faster-GS / FastGS / Deblurring-3DGS / 3DGS-LM 所需 CUDA 扩展可编译和 import。
- `fused3dgs` 的实际训练命令能读取 stage spec，并输出真实非空 `final.ply`。

当前无独显开发环境只完成代码和单元测试，不代表 CUDA runtime 已经可用。

### 9.3 部署验收顺序

建议按以下顺序验收：

1. `npm install` 后执行 `npm run typecheck`。
2. 构建 Docker runtime：`docker compose -f deploy/docker-compose.preview.yml build`。
3. 启动服务：`docker compose -f deploy/docker-compose.preview.yml up -d`。
4. 执行 preview runtime preflight。
5. 增加 fine runtime preflight，检查 Fused3DGS、Spark final converter、RAD exporter。
6. 用小样本跑通 preview。
7. 用同一项目启动 fine task，确认 `fine_tasks` 入队、`fine-worker` 执行。
8. 确认 artifact 表中存在 `final_ply`、`final_web_spz`、4 个 `lod_rad` 和 `metrics_json`。
9. 确认 viewer 返回 `source="final"`，并能加载 final 模型和 LOD 列表。
