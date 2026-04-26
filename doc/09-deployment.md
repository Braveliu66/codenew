# 部署与运行说明

本文档记录当前 Docker/WSL GPU 预览部署方式。目标是单机 GPU 先打通真实预览链路：上传图片或视频、生成 COLMAP 数据集、运行 EDGS、转换 SPZ，并在前端查看。

## 服务组成

`deploy/docker-compose.preview.yml` 包含：

- `backend`：FastAPI API，启动前执行 Alembic migration。
- `worker`：独立 GPU preview worker，运行真实算法命令。
- `postgres`：PostgreSQL 元数据库。
- `redis`：预览任务队列。
- `minio`：上传文件和算法产物对象存储。
- `frontend`：Next.js 前端，Spark 通过 npm 构建打包。

## Docker 构建期自动完成

`backend/Dockerfile.preview` 使用 Python 3.12 和 CUDA devel 镜像，在镜像构建期完成：

- 安装系统工具：git、ffmpeg、build-essential、cmake、ninja、Rust、Node/npm。
- clone 并固定算法仓库：
  - LiteVGGT `4767c17f8b6f176bb751566e92f60eb885040033`
  - EDGS `9a897645eb47c1b24d4f9e4428cd745927bf1ee1`
  - Spark `915c474795e0c78f7cd1b7f4eb97695028b495c0`
- 下载 LiteVGGT 权重 `te_dict.pt`。
- 安装 LiteVGGT、EDGS、Spark 依赖。
- 生成 `/opt/three-dgs/runtime/algorithm_registry.generated.json`。

默认下载源：

| 用途 | 默认值 |
| --- | --- |
| Hugging Face | `HF_ENDPOINT=https://hf-mirror.com` |
| PyPI | `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` |
| npm | `NPM_CONFIG_REGISTRY=https://registry.npmmirror.com` |

这些值可通过 Docker build args 覆盖。

## 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy 数据库连接，Compose 默认使用 PostgreSQL |
| `REDIS_URL` | Redis 队列连接 |
| `MINIO_ENDPOINT` | MinIO 服务地址 |
| `MINIO_BUCKET` | 对象存储 bucket |
| `JWT_SECRET` | JWT 签名密钥，正式部署必须替换 |
| `DEFAULT_ADMIN_USERNAME` | 默认管理员用户名 |
| `DEFAULT_ADMIN_PASSWORD` | 默认管理员密码，正式部署必须替换 |
| `ALGORITHM_REGISTRY_PATH` | Docker 生成的算法 registry 路径 |
| `THREE_DGS_STORAGE_ROOT` | worker 临时工作目录和本地测试存储根目录 |
| `PREVIEW_MIN_INPUT_FRAMES` | 预览输入最少帧数，默认 `8` |
| `PREVIEW_MAX_INPUT_FRAMES` | 预览输入生产上限，默认 `800` |
| `PREVIEW_DEFAULT_EDGS_EPOCHS` | EDGS 预览训练迭代数，默认 `3000` |
| `VIEWER_TARGET_FPS` | 前端 3DGS 查看器目标 FPS，默认 `90` |
| `VIEWER_QUALITY_UP_FPS` | 高于该 FPS 并稳定后提升画质，默认 `105` |
| `VIEWER_QUALITY_DOWN_FPS` | 低于该 FPS 时降低画质，默认 `90` |

## 启动

在项目根目录执行：

```bash
docker compose -f deploy/docker-compose.preview.yml up --build
```

启动后：

- 前端：http://localhost:3000
- 后端健康检查：http://localhost:8000/health
- MinIO Console：http://localhost:9001

默认开发管理员为 `admin / admin123`，只用于本地验证。

## 运行时预检

管理员接口：

```bash
GET /api/admin/runtime/preflight
```

容器内命令：

```bash
python -m backend.scripts.check_preview_runtime
```

预检会检查 Python、CUDA、torch、nvidia-smi、算法仓库路径、commit、权重文件和命令入口。

## 预览验收

图片项目：

- 上传至少 8 张图片。
- 超过 800 张时，worker 会均匀采样 800 张参与预览。
- LiteVGGT 输出 EDGS 可用的 `images/` 和 `sparse/0` COLMAP 结构。

视频项目：

- worker 使用 ffmpeg 抽帧。
- 抽帧结果少于 8 帧时任务失败，不创建 artifact。
- 最多抽取 800 帧。

成功标准：

- task 状态为 `succeeded`。
- MinIO 中存在非空 `preview.spz`。
- `GET /api/projects/{project_id}/viewer-config` 返回 `ready`。
- 前端项目详情页 Spark Viewer 加载模型，并以 90 FPS 为目标自动调节清晰度。

## 常见问题

- 构建阶段下载失败：检查 GitHub、`hf-mirror.com`、PyPI 镜像和 npm 镜像访问。
- `runtime/preflight` 显示 commit mismatch：清理镜像缓存后重新 build。
- `torch.cuda.is_available=false`：检查 Docker Desktop WSL2 GPU、NVIDIA 驱动和 Compose GPU reservation。
- EDGS CUDA 扩展编译失败：优先检查 CUDA_HOME、PyTorch CUDA 版本和编译工具链。
