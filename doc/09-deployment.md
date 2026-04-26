# 部署与运行说明

本文档记录当前 Docker/WSL 目标部署方式。所有源码、配置和文档均使用 UTF-8 编码。

## 1. 服务组成

`deploy/docker-compose.preview.yml` 当前包含：

- `backend`：FastAPI API，启动前执行 Alembic migration。
- `worker`：独立 preview worker，运行 `python3 -m backend.workers.preview_worker`。
- `postgres`：PostgreSQL 元数据库。
- `redis`：预览任务队列。
- `minio`：真实上传文件和算法产物对象存储。
- `frontend`：Next.js 前端。

## 2. 关键环境变量

| 变量 | 说明 |
| --- | --- |
| `DATABASE_URL` | SQLAlchemy 数据库连接，Compose 默认使用 PostgreSQL |
| `REDIS_URL` | Redis 队列连接 |
| `MINIO_ENDPOINT` | MinIO 服务地址，例如 `minio:9000` |
| `MINIO_ACCESS_KEY` | MinIO access key |
| `MINIO_SECRET_KEY` | MinIO secret key |
| `MINIO_BUCKET` | 对象存储 bucket |
| `JWT_SECRET` | JWT 签名密钥，部署时必须替换默认值 |
| `DEFAULT_ADMIN_USERNAME` | 默认管理员用户名 |
| `DEFAULT_ADMIN_PASSWORD` | 默认管理员密码，部署时必须替换 |
| `ALGORITHM_REGISTRY_PATH` | 算法登记 JSON 路径 |
| `THREE_DGS_STORAGE_ROOT` | worker 临时工作目录和本地测试存储根目录 |
| `NEXT_PUBLIC_API_BASE_URL` | 前端访问后端 API 的地址 |

## 3. 启动

在项目根目录执行：

```bash
docker compose -f deploy/docker-compose.preview.yml up --build
```

启动后：

- 前端：http://localhost:3000
- 后端健康检查：http://localhost:8000/health
- MinIO Console：http://localhost:9001

默认开发管理员为 `admin / admin123`。该账号只用于本地验证，正式部署必须通过环境变量替换。

## 4. 数据库迁移与种子数据

Compose 中 `backend` 和 `worker` 启动前都会执行：

```bash
alembic upgrade head
```

后端启动时会 seed：

- 默认管理员账号。
- `algorithm_registry` 初始记录，来源为 `backend/config/algorithm_registry.example.json`。

旧 JSON state 不再作为运行模式。若需要导入旧数据，只能使用只读迁移脚本：

```bash
python -m backend.scripts.import_runtime_state --state backend/storage/state.json
```

## 5. 真实算法部署

真实算法仓库和权重不在 API 启动时下载。部署人员需要单独执行算法 bootstrap 或手动配置：

1. clone LiteVGGT、EDGS、Spark/SPZ 转换器仓库。
2. 下载 LiteVGGT 权重，例如 `te_dict.pt`。
3. 校验并写入实际 commit hash、local path、weight paths、commands。
4. 将对应 `algorithm_registry` 记录设置为 enabled。
5. 在 CUDA Docker/WSL 环境中运行 preview worker。

未完成这些配置时，worker 必须返回明确失败，例如 `ALGORITHM_NOT_CONFIGURED`、`WEIGHTS_NOT_FOUND`、`GPU_RESOURCE_UNAVAILABLE` 或 `SPZ_CONVERTER_NOT_CONFIGURED`，不能生成假 `preview.spz`。

## 6. 本机 CPU-only 验证

当前 Windows CPU-only 环境可用于验证：

- 登录、注册、JWT 鉴权。
- 项目创建和用户隔离。
- 上传真实文件到存储适配层。
- 预览任务进入队列的 API 行为。
- 算法未配置时 worker 失败且 artifact 表为空。

本机没有 Redis/MinIO/PostgreSQL 时，单元测试使用 SQLite 和本地对象存储后端；这只用于开发测试，不作为最终运行架构。

## 7. 验证命令

```bash
python -m unittest discover -s tests
cd frontend
npm run typecheck
npm run build
```

当前验证状态见 `08-progress.md`。
