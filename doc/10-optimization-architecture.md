# 系统优化点与架构融合创新

## 系统优化点

- **Docker 下载可恢复**：大权重不在 Docker build 临时层下载，统一进入共享 `model-cache`；下载脚本使用 `.part`、HTTP Range、文件锁和重试，避免网络中断后从头开始。
- **依赖隔离**：API、图片 Worker、视频/摄像头 Worker 使用不同镜像，避免 LiteVGGT/EDGS 与 LingBot-Map 的 Python、CUDA、PyTorch 依赖互相污染。
- **一键部署**：`make deploy` 先补齐权重缓存，再启动 Compose；没有 `make` 时也可直接 `docker compose up -d --build`，Worker 启动预检会兜底下载缺失权重。
- **渐进式预览**：摄像头分片和视频窗口可以生成 `preview_spz_segment`，Viewer 不等待完整重建完成即可加载已完成片段。
- **Viewer 预算控制**：前端以 90 FPS 为目标，根据 FPS、网络状况和 500 万 Gaussians 默认预算动态调节质量。

## 架构融合创新点

- **统一 LingBot-Map 视频/摄像头路径**：离线视频和实时摄像头都使用 LingBot-Map，区别只在输入来源和窗口粒度；精细重建仍可在长视频场景叠加 MASt3R/Pi3 做全局优化。
- **Artifact 兼容扩展**：不新增复杂产物表，复用 `artifacts.metadata` 表达 `segment_index`、时间窗口、LOD 和估算 splat 数；旧 `preview_spz` 单文件路径保持兼容。
- **SSE 驱动 Viewer 增量加载**：`preview_segment_ready` 事件只通知新增能力，真实数据仍从对象存储签名 URL 拉取，避免 API 承载大模型传输。
- **队列按任务类型拆分**：`preview_image_tasks`、`preview_video_tasks`、`preview_camera_tasks` 分别由对应 Worker 消费，是多 GPU 动态调度器之前的可运行调度基础。
- **可扩展算法注册**：新增算法按“Dockerfile/镜像 + Compose 服务块 + registry 命令 + Worker 队列”扩展，保持后端任务模型稳定。

## 渐进式渲染策略

- Worker 每完成一个摄像头分片或视频时间窗口，就输出一个独立 SPZ segment。
- API 将 segment 记录为 `preview_spz_segment` artifact，并通过 SSE 推送 `preview_segment_ready`。
- Viewer 获取新的 viewer-config 后追加加载 segment；时间线中已完成区间可浏览，未完成区间保留占位。

## LOD 加载策略

- 当前 MVP 先使用 Spark 的 LOD 能力和前端质量档位控制 pixel ratio、`lodSplatScale`、`maxStdDev`、`maxPixelRadius`。
- 后续 EcoSplat/RAP 产物接入后，segment metadata 中的 LOD、bounds 和 splat 数可用于视锥体剔除、方向预取和带宽自适应。
- 当总 splat 数超过预算或 FPS 低于阈值时，Viewer 优先降低旧 segment 或远处 segment 的质量，保证交互帧率。
