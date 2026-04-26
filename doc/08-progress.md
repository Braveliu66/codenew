# 当前实现进度

更新时间：2026-04-25

## 已完成

1. 后端数据层已从运行时 JSON 状态切到 SQLAlchemy 模型，覆盖 `users`、`projects`、`media_assets`、`tasks`、`artifacts`、`feedback`、`worker_heartbeats`、`algorithm_registry`。
2. 已增加 Alembic 基础配置和 `0001_initial` 迁移；Docker/WSL 环境启动时可执行 `alembic upgrade head`。
3. 已实现用户名/密码登录、注册、JWT Bearer 鉴权、`user/admin` 角色和普通用户项目隔离。
4. 上传接口已写入对象存储适配层：Docker 环境使用 MinIO；本机测试环境使用同一接口的本地文件后端，仍记录真实 `file_size` 和 `object_uri`。
5. 预览任务创建接口已改为只创建 DB task 并推送 Redis 队列，不在 API 请求内直接执行算法。
6. 已增加独立 preview worker：从 Redis 拉取任务，物化真实上传文件，调用现有 LiteVGGT → EDGS → Spark-SPZ 预览引擎，成功后才上传真实 `preview.spz` 并创建 artifact。
7. 未配置算法、缺权重、GPU 不可用、SPZ 转换器缺失等路径仍返回显式失败；失败路径不创建成功 artifact。
8. 前端已增加 `/login` 页面、token 存储、鉴权请求头、未登录跳转和管理员页面权限显示。
9. Docker Compose 已扩展为 `backend`、`worker`、`postgres`、`redis`、`minio`、`frontend`。

## 已验证

- `python -m unittest discover -s tests`：11 个测试通过。
- `npm run typecheck`：通过。
- `npm run build`：通过。
- UTF-8 读取检查：源码、配置和文档按 UTF-8 可读取。

## 当前限制

1. 当前 Windows CPU-only 环境没有真实 CUDA 算法运行条件，预览任务在未配置算法时会失败为明确错误。
2. MinIO、Postgres、Redis 的完整联调目标是 Docker/WSL；本机单元测试使用 SQLite 和本地对象存储适配层验证生命周期。
3. 实时摄像头、Stream3R、精细重建 4.1 和 Mesh 导出仍保持不可用状态，不伪装成功。
4. 大文件分片上传、SSE/WebSocket 事件推送和管理员用户用量统计尚未实现。

## 下一步

1. 在 Docker/WSL 中启动 Compose，验证 API、worker、Postgres、Redis、MinIO 联通。
2. 执行真实算法 bootstrap，写入 LiteVGGT、EDGS、Spark-SPZ 的实际 commit、路径和权重。
3. 用 2-5 张真实图片验证 LiteVGGT → EDGS → Spark-SPZ 成功路径，确认 MinIO 中存在非空 `preview.spz`。
4. 增加 Playwright 登录、上传、任务失败和项目详情 viewer unavailable 回归测试。
