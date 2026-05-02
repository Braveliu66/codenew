# 精细重建融合状态与部署差距

更新时间：2026-05-03

## 1. 当前代码已经完成的部分

本阶段已经完成的是“平台编排层”和“算法融合调度层”，不是完整 GPU 训练能力。

已完成：

- `fine` task 创建后进入 `fine_tasks` 队列，由 `backend.workers.fine_worker` 消费执行。
- `FineSynthesisEngine` 改为线性产物链：输入素材落盘 -> `final.ply` -> `final_web.spz` -> `final_lod0.rad` 到 `final_lod3.rad` -> `metrics.json`。
- `FineSynthesisEngine` 成功结果支持按需产物字典 `artifact_paths`，同时保留 artifact list 供数据库落库使用。
- requested outputs 按配置校验，缺少任一必需产物则任务失败，不创建占位文件。
- 新增 `backend/scripts/run_fused3dgs_fine.py`，通过 `python -m fused3dgs.train` 调用真实训练入口。
- 新增 `backend/scripts/run_final_spz_convert.py`，只允许从真实 `final.ply` 生成 `final_web.spz`。
- 新增 `backend/scripts/run_lod_export.py`，只允许从真实 `final.ply` 或导出 checkpoint 生成 `.rad`，未配置真实 exporter 时失败。
- 新增 `fused3dgs` 算法包：
  - 嵌套配置结构：`fused3dgs`、`deblurring`、`lm_optimizer`、`vcd`、`outputs`。
  - LM 调度采用间隔式策略：默认 `start_iter=3000`、`interval=200`。
  - Deblurring MLP 只在训练态、达到 `start_iter` 后启用，eval/export 强制禁用。
  - CUDA 后端采用依赖注入：`GaussianRasterizerBackend`、`FasterGSBackend`、`LMBackend`，`FusedBackend` 只预留。
  - 新增 `FusedTrainingLoop`，可在同一主循环中按配置调度 SGD、FastGS VCD、Deblurring covariance modulation 和 3DGS-LM interval step。
  - FastGS VCD 已从“必须外部传入 scores”推进为“根据多视角 loss map、2D 投影坐标、半径和可见性聚合每个高斯的重要性分数”。
  - Deblurring MLP 已提供 `modulate_covariance`，可以对 scaling 和 rotation 做有界调制，并保持 quaternion 归一化。
- `viewer-config` 改为 final 优先：存在 `final_web_spz` 时返回 `source="final"`，否则回退 `preview_spz`。
- 前端项目详情页增加“精细重建”入口和 LOD 列表展示。
- `deploy/docker-compose.preview.yml` 增加 `fine-worker` 服务。

## 2. 当前还不是“可实际部署使用”的原因

当前系统已经能把精细重建任务从 API、队列、worker、产物校验、viewer 串起来，但真实部署还缺算法运行环境和真实命令。

关键缺口：

1. 真实 Fused3DGS 训练命令未配置  
   `fused3dgs.train` 当前要求环境变量 `FUSED3DGS_TRAIN_COMMAND` 指向真实训练命令。未配置时会失败，避免用假产物冒充成功。

2. Faster-GS / FastGS / Deblurring-3DGS / 3DGS-LM 还没有在 GPU runtime 中完成真实依赖安装和预检  
   当前已经有可注入训练循环、VCD 分数聚合、Deblur 协方差调制和 LM 调度调用面，但不编译 CUDA，不合并 kernel，也不运行真实训练。

3. 3DGS-LM 与 Faster-GS CUDA kernel 尚未融合  
   当前采用依赖注入降低风险。近期目标是能选择 Faster-GS 或 LM backend；远期目标才是合并 kernel 成 `FusedBackend`。

4. `final_web.spz` 转换需要真实 Spark/SPZ 命令  
   `run_final_spz_convert.py` 已经写好，但部署时需要保证 registry 中 `Spark-SPZ.compress_final` 可用，或配置 `SPZ_CONVERTER_COMMAND`。

5. `.rad` LOD 真实 exporter 未接入  
   `run_lod_export.py` 要求 `RAD_LOD_EXPORT_COMMAND`。未配置时会失败，不生成占位 LOD。

6. blur_detected 质量标记需要真实写入  
   `use_deblur="auto"` 已经会读取 task options 或 media quality flags，但实际上传/预处理阶段还需要可靠的 blur 检测或管理员显式配置。

7. fine runtime preflight 还没有独立脚本  
   现有 preflight 主要覆盖 preview runtime。精细重建需要新增 `check_fine_runtime`，检查 Faster-GS、FastGS、Deblurring、3DGS-LM、Spark final converter、RAD exporter。

8. 当前机器无独立显卡，未做 CUDA 编译和真实训练验证  
   所有 GPU 相关能力必须在带 NVIDIA GPU、NVIDIA Container Toolkit、CUDA runtime 可用的机器上重新验证。

9. 前端 typecheck 依赖未安装  
   本机 `npm run typecheck` 因 `tsc` 不存在失败，需要先在 `frontend/` 执行依赖安装。

10. 完整 Python 测试受当前沙箱临时目录权限影响  
    新增和相关目标测试通过，但全量测试中部分旧 runtime tests 因 `tempfile.TemporaryDirectory()` 无法写入随机目录失败，需要在正常权限环境重跑。

## 3. 距离系统设计目标的差距

### 3.1 已接近目标的部分

- 平台任务模型：已支持 preview 与 fine 分队列执行。
- 失败路径：未配置真实算法时会失败，不生成假 artifact。
- 产物链路：final / web spz / LOD / metrics 的依赖关系已明确。
- 配置结构：fine options 已从扁平字段升级为模块化嵌套配置。
- 算法融合策略：已采用依赖注入，避免初期强行合并 CUDA kernel。
- 算法训练调度：已新增 `FusedTrainingLoop`，能在无 GPU 环境下验证 SGD、VCD、Deblur、LM 的主循环调度。
- Viewer：已支持 final 优先和 LOD 元数据返回。

### 3.2 仍未达到目标的部分

- 真实 Faster-GS 训练尚未跑通。
- FastGS VCD 已具备基于投影的多视图 loss score 聚合，但还需要对齐 FastGS 原仓库的完整 clone/split/prune 策略和真实高斯参数表更新。
- Deblurring MLP 已具备 scaling/rotation 调制入口，但还需要接入真实 Faster-GS 高斯协方差构建和 rasterizer 参数路径。
- 3DGS-LM 当前完成调度、optimizer wrapper 和训练循环调用面，未接入真实 JVP/JTJ CUDA kernel。
- `.rad` LOD 导出缺真实 exporter。
- Fine 任务缺资源调度策略，例如 GPU 占用、并发限制、任务超时后的进程清理、失败重试和死信队列。
- 精细重建还缺小样本端到端验收数据集和指标基线。

## 4. 实际部署前必须完成的清单

最低可部署版本必须满足以下条件：

1. 安装前端依赖并通过类型检查  
   `cd frontend && npm install && npm run typecheck`

2. 在 GPU 机器上构建 runtime  
   `docker compose -f deploy/docker-compose.preview.yml build`

3. 配置真实算法仓库和命令  
   - `Faster-GS.fine_engine`
   - `Spark-SPZ.compress_final`
   - `RAD-LOD.export_rad`
   - FastGS / Deblurring / 3DGS-LM 的版本、license、commit hash

4. 配置真实训练命令  
   设置 `FUSED3DGS_TRAIN_COMMAND`，命令必须读取 stage spec，并输出真实非空 `final.ply`。

5. 配置 SPZ 转换命令  
   确认 `run_final_spz_convert.py` 能从 `final.ply` 输出真实非空 `final_web.spz`。

6. 配置 RAD LOD exporter  
   设置 `RAD_LOD_EXPORT_COMMAND`，并确保输出：
   - `final_lod0.rad`，目标最多 1,000,000 gaussians。
   - `final_lod1.rad`，目标最多 500,000 gaussians。
   - `final_lod2.rad`，目标最多 200,000 gaussians。
   - `final_lod3.rad`，目标最多 50,000 gaussians。

7. 新增并通过 fine runtime preflight  
   预检至少应覆盖 Python import、CUDA 可用性、算法仓库存在、关键命令存在、CUDA extension 可 import、模型权重存在、Spark/RAD exporter 可执行。

8. 用小样本跑通端到端 fine task  
   验证 API 创建任务、Redis 入队、fine worker 执行、MinIO artifact 落库、viewer 加载 final、LOD 列表返回。

9. 建立质量指标基线  
   至少记录训练耗时、GPU 显存峰值、Gaussian 数量、`final.ply` 大小、`final_web.spz` 大小、LOD actual count、PSNR/SSIM/LPIPS。

10. 补齐运维策略  
    包括 GPU worker 并发数、队列重试、失败清理、超时 kill、artifact retention、日志截断和管理员 runtime preflight 页面。

## 5. 推荐下一步实施顺序

1. 先完成 fine runtime preflight，不运行训练，只检查依赖是否具备。
2. 接入 `FUSED3DGS_TRAIN_COMMAND`，用一个真实小场景输出 `final.ply`。
3. 单独验证 `final.ply -> final_web.spz`。
4. 单独验证 `final.ply -> final_lod0..3.rad`。
5. 跑通完整 fine worker 端到端。
6. 再逐步打开 FastGS VCD、Deblurring MLP、3DGS-LM。
7. 最后评估是否需要进入 CUDA kernel 合并阶段。

## 6. 距离全部功能要求的差距

对照 `01-requirements.md`，当前系统距离完整设计目标还差以下能力。

### 6.1 项目与素材管理

- 项目搜索、标签筛选、训练状态筛选还需要补齐。
- 图片缩略图、大图预览、素材删除、补传后的增量任务策略还需要完善。
- 视频大文件分片上传和断点续传还需要实现或实测。
- 项目卡片封面策略还需要接入上传素材首帧、preview/final 截图或 viewer thumbnail。

### 6.2 极速预览

- LiteVGGT -> EDGS -> Spark-SPZ 图片预览链路仍需在真实 GPU Docker/WSL 环境验证。
- LingBot-Map 视频预览和实时摄像头 streaming 预览仍需真实算法环境验证。
- 预览质量提示还不完整，例如缺失视角、模糊素材、补拍方向建议。
- 预览任务预计剩余时间还需要基于历史耗时或阶段权重估算。

### 6.3 精细重建

- `FUSED3DGS_TRAIN_COMMAND` 未接入真实训练命令，尚不能实际输出 `final.ply`。
- Faster-GS、FastGS、Deblurring-3DGS、3DGS-LM 还未在 GPU runtime 中完成依赖安装、预检和小样本训练。
- FastGS VCD、Deblur covariance modulation、LM interval step 已进入主循环调度面，但还需要与真实 Gaussian model、真实 rasterizer、真实 optimizer state 对接。
- `final_web.spz` 和 `.rad` LOD 转换脚本已就位，但真实 converter/exporter 还需配置和验收。
- 精细重建阶段进度、预计剩余时间、失败重试、超时 kill 和资源回收还需要加强。

### 6.4 实时摄像头重建

- 当前已有 progressive segment 的平台接口基础，但还需要完整实时采集页面、重拍流程、结束录制后进入精细重建的端到端验证。
- 需要验证长时间录制时的帧率、缓存、对象存储写入、SSE 推送和 viewer 增量加载稳定性。

### 6.5 LOD 与 Viewer

- Viewer 已能 final 优先和返回 LOD 元数据，但还未实现真正的按性能自动选择 LOD、逐级加载高清层和移动端性能策略。
- `.rad` 文件的真实解析、加载、切换和回退策略还需要前后端联调。
- 全屏渲染、帧率监控、自动降级/升级清晰度还需要补齐。

### 6.6 导出与分享

- PLY 下载已有 artifact 基础，但 OBJ、GLB、Mesh 导出还未实现。
- 分享页面尚未完整实现，需要可交互 Web 模型、权限控制、签名 URL 和不暴露内部路径。
- 导出任务的队列、超时、失败清理和产物过期策略还未完善。

### 6.7 用户、反馈与管理

- 登录、角色和项目隔离已有基础；用户总览中的训练中数量、完成数量、总占用等统计还需要补齐。
- 反馈功能骨架已有，但截图、附件、项目关联后的管理查看流程还需要完善。
- 管理面板还需要补全队列深度、GPU 负载、用户存储占用、训练占用、任务耗时分布、失败原因聚合。
- Worker 心跳和日志已有基础，但管理员侧的完整可视化和筛选还需要实现。

### 6.8 合规与可审计性

- algorithm registry 已有基础，但关于页面仍需完整展示算法名称、许可证、仓库地址、commit hash、权重来源和非商业限制。
- 第三方算法真实接入后，需要逐项确认许可证允许的研究/毕业设计使用范围。

### 6.9 部署与运维

- 需要新增 fine runtime preflight，并把结果接入管理员页面。
- 需要真实 GPU 环境下的 Docker build、CUDA extension import、端到端小样本训练验收。
- 需要任务优先级、GPU 并发限制、失败重试、死信队列、日志截断、artifact retention 和磁盘清理策略。
- 当前本机前端依赖缺失，部署前必须安装依赖并通过 `npm run typecheck` 和生产构建。
