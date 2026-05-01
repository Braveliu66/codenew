# 系统优化点与架构融合

## Docker 体积优化

- **统一 GPU runtime**：`image-worker`、`video-worker`、`camera-worker` 复用 `three-dgs-gpu-runtime:local`，不再维护图片和视频两套 CUDA 镜像。
- **统一 CUDA/PyTorch 基线**：GPU runtime 固定 Python 3.10、CUDA 12.8.1、PyTorch 2.8.0/cu128，减少基础层和 wheel 重复下载。
- **多阶段构建**：builder 阶段保留 nvcc、Node、Rust、CMake、ninja 等编译工具；runtime 阶段只保留运行所需的 venv、算法仓库、Spark 转换脚本、Node 运行时和 ffmpeg。
- **模型权重外置**：LiteVGGT 和 LingBot-Map 权重只进入共享 `model-cache`，通过 `.part`、HTTP Range 和文件锁支持断点续传，不写入镜像。
- **算法源码缓存**：LiteVGGT、EDGS、LingBot-Map 和 Spark 先下载到宿主机 `repo-cache`，Docker build 优先复制本地缓存，缺失时才按国内镜像优先、官方源兜底在线 clone。
- **Torch wheelhouse 缓存**：PyTorch/cu128 相关 wheel 先下载到 BuildKit cache mount `/root/.cache/three-dgs-wheelhouse`，完整 wheel 复用，`.part` 文件断点续传，避免大包失败后全部重下。
- **构建上下文控制**：`.dockerignore` 排除 `model-cache`，同时保留 `repo-cache` 的源码工作树供 BuildKit 只读挂载，避免权重随 Docker build context 上传。

## 依赖融合策略

- LiteVGGT 官方 requirements 中的 `torch==2.7.1`、`torchvision==0.22.1`、`numpy`、`opencv-python` 在统一构建脚本中过滤，避免覆盖 cu128 统一栈。
- LingBot-Map 按官方推荐使用 PyTorch 2.8.0 / CUDA 12.8；安装项目本体时使用 `--no-deps -e`，避免 pip 重新解析 Torch。
- EDGS 官方偏 Python 3.10 / CUDA 12.1，本项目在统一 runtime 中继续源码编译 `diff_gaussian_rasterization` 和 `simple_knn`，用预检结果确认可用。
- Spark-SPZ 在构建期完成 npm/Rust build，运行期直接执行 `node scripts/compress-to-spz.js`，不保留完整 dev 构建链路。

## 架构融合点

- **一个镜像，多队列 worker**：同一 GPU runtime 通过 `PREVIEW_WORKER_INPUT_TYPE=images|video|camera` 区分消费队列和默认管线。
- **两套 registry 视图**：图片 worker 使用 image registry，只启用 LiteVGGT、EDGS、Spark、FFmpeg；视频/摄像头 worker 使用 video registry，只启用 LingBot-Map、Spark、FFmpeg。
- **统一 Spark 输出**：图片、视频、摄像头最终都转为 SPZ，后端 artifact 和前端 Viewer 不需要区分算法来源。
- **SSE 驱动增量加载**：摄像头分片完成后写入 `preview_spz_segment` artifact，并通过 `preview_segment_ready` 通知 Viewer 增量加载。

## 运行风险与观测

- 统一版本后重点观察 LiteVGGT Transformer Engine、EDGS CUDA 扩展和 LingBot-Map 推理是否同时可用。
- `python -m backend.scripts.check_preview_runtime` 是镜像准入检查；image/video/camera 三种 worker 都必须通过。
- `docker system df -v` 和镜像保存包大小用于验证瘦身收益。
- 如果 Torch/CUDA 版本冲突，优先保留 CUDA 12.8 / Torch 2.8 基线，再针对 LiteVGGT 或 EDGS 做局部依赖修正。
