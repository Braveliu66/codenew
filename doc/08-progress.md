# 当前实现进度

更新时间：2026-04-30

## 已完成

1. 后端数据层使用 SQLAlchemy 模型和 Alembic migration，覆盖用户、项目、素材、任务、产物、反馈、worker 心跳和算法 registry。
2. 已实现用户名/密码登录、注册、JWT Bearer 鉴权、user/admin 角色和普通用户项目隔离。
3. 上传接口写入对象存储适配层；Docker 环境使用 MinIO，本地测试可使用本地文件后端。
4. 预览任务由 API 创建 DB task 并推送 Redis 队列，真实算法只在 worker 中运行。
5. 图片默认管线为 LiteVGGT -> EDGS -> Spark-SPZ；视频和摄像头默认管线为 LingBot-Map -> Spark-SPZ。
6. 预览队列已拆分为 `preview_image_tasks`、`preview_video_tasks`、`preview_camera_tasks`，API 按项目输入类型路由到对应 worker。
7. Docker GPU runtime 已统一为 `three-dgs-gpu-runtime:local`，三个 worker 复用同一个 CUDA 12.8 / PyTorch 2.8 镜像。
8. GPU runtime 改为多阶段构建：builder 编译 CUDA/Node/Rust 依赖，runtime 只保留运行所需文件。
9. LiteVGGT requirements 中的 Torch/Torchvision/Numpy/OpenCV pin 已在统一构建脚本中过滤，避免覆盖统一 PyTorch cu128 栈。
10. EDGS CUDA 扩展继续源码编译；LingBot-Map 使用 `--no-deps -e` 安装，避免 pip 重新解析 Torch。
11. Spark-SPZ 运行期改为直接调用 `node scripts/compress-to-spz.js`，不再依赖运行期 npm/Rust 构建。
12. 模型权重缓存规则固定为共享 `model-cache/<model-name>/...`；worker 启动预检命令优先使用本地缓存，缺失时用 `.part` 和 HTTP Range 断点续传。
13. 算法源码缓存规则固定为共享 `repo-cache/<repo-name>`；构建前可用 `make algorithm-repos` 预下载，Docker build 优先复用本地缓存。
14. `.dockerignore` 已排除 `model-cache`，并允许 `repo-cache` 进入 BuildKit 只读挂载，避免权重和续传文件进入 Docker build context。
15. 新增 `GET /api/admin/runtime/preflight` 和 `python -m backend.scripts.check_preview_runtime`，用于检查 GPU、torch、算法仓库、权重、命令和 CUDA 扩展。
16. 前端 Spark Viewer 已增加自适应质量控制；实时摄像头已支持 progressive segment 和 SSE 通知。

## 当前预览规则

- 图片项目至少 1 张图片；默认 `preview_pipeline=edgs`，超过 800 张时均匀采样 800 张参与预览。
- 图片可选 `preview_pipeline=litevggt_spark`，直接将 LiteVGGT 点云转为 Spark-SPZ，不运行 EDGS。
- 视频/摄像头固定使用 `preview_pipeline=lingbot_map_spark`；LingBot 未配置时明确失败，不回退到 LiteVGGT/EDGS。
- 视频默认按完整时长均匀采样；可用 `VIDEO_PREVIEW_TARGET_FRAMES` 或 `frame_sample_fps` 控制采样数量。
- 摄像头分片生成 `preview_spz_segment`，前端通过 SSE 收到 `preview_segment_ready` 后增量加载。
- 任务失败时不创建假的 `preview.spz` artifact。

## 已验证

- `docker compose -f deploy/docker-compose.preview.yml config` 可解析。
- `docker compose -f deploy/docker-compose.preview.yml config --images` 可看到三个 worker 均引用 `three-dgs-gpu-runtime:local`。
- 单元测试覆盖 registry upsert、runtime preflight、权重断点续传、算法源码缓存、统一 GPU runtime 依赖过滤和 base image 列表。
- 前端类型定义和 API 调用已覆盖 Runtime Preflight。

## 待真实环境验证

1. 在 Docker/WSL GPU 环境执行完整构建：`docker compose -f deploy/docker-compose.preview.yml build image-worker`。
2. 执行 image/video/camera worker 的 runtime preflight，确认 Torch CUDA、Transformer Engine、EDGS 扩展、LingBot 权重和 Spark 转换器可用。
3. 用真实图片验证 LiteVGGT -> EDGS -> Spark-SPZ 和 `litevggt_spark` 两条路径。
4. 用真实视频验证 LingBot-Map 全时长抽帧覆盖开头、中段、结尾，并输出非空 `preview.spz`。
5. 用摄像头分片验证 progressive segment、SSE 和 Viewer 增量加载。
6. 对比改造前后的 `docker images`、`docker system df -v` 和镜像保存包大小。

## 当前限制

1. 精细重建、Mesh 和完整 LOD 产物导出仍未纳入本阶段可用范围。
2. EDGS 使用原仓库许可证，当前记录为非商业研究和个人用途。
3. Docker 构建默认优先使用国内镜像和本地 `repo-cache`；Docker Hub / NVIDIA 基础镜像仍依赖宿主机 Docker daemon 的镜像源配置。
4. 统一 CUDA 12.8 / Torch 2.8 偏离 EDGS 官方 CUDA 12.1 建议，需要以真实 GPU 构建和预检结果作为最终准入。
