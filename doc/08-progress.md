# 当前实现进度

更新时间：2026-04-26

## 已完成

1. 后端数据层使用 SQLAlchemy 模型和 Alembic migration，覆盖用户、项目、素材、任务、产物、反馈、worker 心跳和算法 registry。
2. 已实现用户名/密码登录、注册、JWT Bearer 鉴权、user/admin 角色和普通用户项目隔离。
3. 上传接口写入对象存储适配层；Docker 环境使用 MinIO，本机测试可使用本地文件后端。
4. 预览任务由 API 创建 DB task 并推送 Redis 队列，真实算法只在独立 worker 中运行。
5. worker 会物化真实上传文件，按输入帧数规则采样图片或抽取视频帧，调用 LiteVGGT -> EDGS -> Spark-SPZ。
6. Docker 预览镜像改为 Python 3.12 + CUDA devel，并在构建期自动 clone 算法仓库、下载 LiteVGGT 权重、安装依赖、生成算法 registry。
7. registry seed 已改为 upsert，Docker 生成的算法 commit、路径、权重和命令会同步到数据库。
8. 新增 `GET /api/admin/runtime/preflight` 和 `python -m backend.scripts.check_preview_runtime`，用于检查 GPU、torch、算法仓库、权重和命令。
9. 前端 Spark Viewer 增加自适应质量控制：目标 90 FPS，低于 90 降低画质，高于 105 并稳定后提升画质。
10. 管理页增加 Runtime Preflight 面板，上传页显示预览输入帧数规则。

## 当前预览规则

- 输入数据帧数和前端渲染 FPS 是两个独立概念。
- 图片项目至少 8 张图片；超过 800 张时均匀采样 800 张参与预览。
- 视频项目抽帧后至少 8 帧；最多抽取 800 帧。
- 前端查看 3DGS 模型时目标实时渲染为 90 FPS，优先保证速度，再提升清晰度。
- 任务失败时不创建假 `preview.spz` artifact。

## 已验证

- `docker compose -f deploy/docker-compose.preview.yml config` 可解析。
- 后端新增了输入校验、registry upsert 和 runtime preflight 的单元测试。
- 前端类型定义和 API 调用已覆盖 Runtime Preflight。

## 待真实环境验证

1. 在 Docker/WSL GPU 环境执行完整 `docker compose -f deploy/docker-compose.preview.yml build`。
2. 执行 `docker compose -f deploy/docker-compose.preview.yml run --rm worker python -m backend.scripts.check_preview_runtime`。
3. 用至少 8 张真实图片验证 LiteVGGT -> EDGS -> Spark-SPZ 成功路径。
4. 用真实视频验证 ffmpeg 抽帧、少帧失败路径和 800 帧上限。
5. 在前端确认 Spark Viewer 能加载非空 `preview.spz` 并自动调节 FPS/清晰度。

## 当前限制

1. 精细重建、Mesh、LOD 产物导出和实时摄像头仍未纳入本阶段可用范围。
2. EDGS 使用原仓库许可证，当前记录为非商业研究和个人用途。
3. Docker 构建依赖 GitHub、Hugging Face 镜像、PyPI 镜像和 npm 镜像可访问。
