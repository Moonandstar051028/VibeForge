# 07：首版实现与本地验收报告

验证日期：2026-11-04（北京时间）；事件文件保存 UTC 时间。

## 1. 结论

| 验收项 | 结果 |
| --- | --- |
| 项目核心模块 | 已实现可运行首版，无仅抛出 NotImplemented 的业务占位 |
| 工程测试 | 43 项通过，真实目标测试独立执行 |
| CPU 完整闭环 | 通过：两轮修改，第一轮接受，第二轮正确性失败后恢复稳定版本 |
| 故障恢复 | 测试通过：提升意图写入后控制程序退出，新进程恢复保留已提交版本 |
| Thor 推理环境 | CUDA 实算、分词器、聊天模板、Qwen meta 结构及写入限制下运行通过 |
| Qwen BF16 完整部署 | 未通过：内存预检阻塞，完整权重未加载 |
| 真实 Qwen Agent 与自优化 | 尚未验证，受完整部署阻塞 |
| 可迁移优化范式 | 已有草案与证据管理机制，尚无经真实多配置实验验证的通用规则 |

工程通过与目标模型部署未通过分别保留结果，不合并为“系统全部通过”。

## 2. 模块与实现入口

| 模块 | 已实现职责 | 主要文件 |
| --- | --- | --- |
| CLI | 画像、启动、恢复、状态、停止、事件、报告、子进程入口 | `cli/main.py`、`campaign.py`、`status.py` |
| 控制 | 自举、有限迭代、固定预算、状态机、统计准入、恢复编排 | `controller/supervisor.py`、`state_machine.py`、`budgets.py`、`admission.py`、`recovery.py` |
| Agent | 本地聊天 JSON 工具循环、独立会话、历史上下文、只读证据、候选编辑、结果分析 | `agent/runner.py`、`tools.py`、`context.py`、`analysis.py` |
| 引擎 | Python 后端构建检查、HTTP 服务、健康检查、进程所有权、写入限制 | `engine/backend.py`、`builder.py`、`server.py`、`client.py`、`process.py`、`sandbox.py` |
| 画像 | 分片头部与索引检查、模型结构、设备/依赖探测、内存预检、资源采样 | `profiling/model.py`、`hardware.py`、`environment.py`、`collector.py`、`sampler.py` |
| 评测 | 冻结参考、确定性 token、抽样 logits、请求隔离、实际吞吐、配对 bootstrap | `evaluation/reference.py`、`correctness.py`、`performance.py`、`statistics.py` |
| 版本 | 隔离工作区、差异、内容哈希、冻结、持久化提升意图、提交和回退 | `versioning/` 全部业务模块 |
| 可观测 | SQLite outbox 导出、断尾修复、持续原始输出、工具审计、事件心跳、摘要 | `observability/events.py`、`logs.py`、`metrics.py`、`reports.py` |
| 知识 | 按轮抽取草案、校验状态和证据、按条件检索、索引 | `knowledge/extractor.py`、`validator.py`、`registry.py`、`retriever.py` |
| 存储 | 原子替换与 fsync、WAL 事务、锁、路径和标识符约束 | `storage/atomic.py`、`database.py`、`locks.py`、`paths.py` |
| 契约 | 配置验证、候选/画像/决策对象、事件时间标识 | `contracts/` 全部业务模块 |

文件路径均相对于 `src/vibeforge/`。配置、用例、引擎、提示词、报告和知识模板、运维脚本及服务模板已经提供；每个静态目录保留用途说明。

## 3. 工程测试证据

命令：

```bash
.venv/bin/python -m pytest -m 'not target' \
  --junitxml=var/campaigns/system-validation/tests.xml -q
```

覆盖内容：

| 类别 | 实际断言 |
| --- | --- |
| 正确性准入 | token/logit 错误先拒绝；不能靠性能收益掩盖错误 |
| 性能准入 | 明确收益接受、退化拒绝、置信区间不足不提升、无效样本拒绝 |
| 失败处理 | 语法错误、Agent 越界编辑、进程超时后恢复原稳定版本 |
| 存储 | 事务回滚、锁互斥、原子 JSON、路径穿越与符号链接限制 |
| 发布事务 | 幂等提交、意图取消、制品哈希损坏、提交前崩溃恢复 |
| 日志 | outbox 事件导出、重复导出与 JSONL 断尾恢复 |
| Agent | 模型动作编辑与提交、缺少上一轮分析时要求补充、会话不复用 |
| 写入限制 | 生成进程可写缓存，不能改控制状态；允许库访问 `/dev/null` |
| 画像与知识 | 分片缺失/头部越界检查、资源阻塞、配置校验、知识适用边界 |

43 项工程测试的 JUnit 和真实部署验收报告保存在本机 `var/campaigns/system-validation/`，属于忽略的运行产物，不纳入 Git。目标验收有 1 项失败，原因为 `BLOCKED: insufficient_memory`。

另已检查 79 个 Python 文件语法、20 个静态 JSON 文件、全部 shell 脚本语法、静态子目录说明和新增文档链接。活动退出后没有残留的引擎或 Agent 进程记录。

## 4. CPU 活动实测

活动：`system-validation-toy-final`，状态 `STOPPED`，完成 2 轮，`analysis_pending=false`，退出后 `active_ref=null`。

| 轮次 | 修改 | 结果 |
| --- | --- | --- |
| 1 | 把每个输出 token 的重复求和换成等价闭式算式 | 正确性通过；按实际计时接受候选 |
| 2 | 故意改变输出数值 | 正确性拒绝；稳定指针保持第一轮版本 |

该次第一轮相对吞吐收益 `gain=35.0648`，即约 +3506%；95% bootstrap 区间 `[31.4043, 39.9554]`。这是刻意设置冗余 CPU 运算的测试引擎，不能解释为 Qwen 加速。

活动摘要、各轮判定和完整事件流保存在本机 `var/campaigns/system-validation-toy-final/`，不会作为机器特定运行产物提交。

## 5. Qwen 本机部署结果

### 5.1 硬件与模型

| 项目 | 采集结果 |
| --- | --- |
| 平台 | aarch64，NVIDIA Thor，统一内存，Linux 6.8.12-1021-tegra |
| 驱动 / Torch | 595.78 / `2.13.0+cu132` |
| Transformers / Accelerate | `5.19.0` / `1.15.0` |
| 模型路径 | 本机路径由 `VIBEFORGE_MODEL_PATH` 提供，不进入版本库 |
| 模型类 | `Qwen3_5MoeForConditionalGeneration` |
| 模型结构 | 40 层混合注意力，256 专家、每 token 选择 8 个专家 |
| 分片 | 26 个 safetensors 分片，索引和头部检查通过 |
| 权重字节 | 71,903,645,408，约 66.96 GiB |
| BF16 需求估算 | 84,098,387,984 字节，约 78.32 GiB |
| 活动预检可用内存 | 40,259,145,728 字节，约 37.49 GiB |

本次没有重新计算全部约 67 GiB 权重的内容校验和；画像记录配置、tokenizer 等文件哈希及分片清单/大小/头部检查范围。此前下载校验材料不作为本次完整校验的替代。

### 5.2 已通过的轻量探针

- CUDA 上执行 8 个元素求和，结果 `8.0`。
- 本地聊天模板与分词器可用，样例得到 19 个 token。
- 在 meta 设备构造真实模型类，统计参数 `35,107,181,936`；不分配完整权重。
- 使用与引擎相同的 Landlock 写入限制后，上述检查仍通过。Thor 的 CUDA 驱动由受维护启动代码先初始化，再限制生成代码写入。

常规探针和受写入限制的探针 JSON 保存在本机 `var/`，均标记 `model_weights_loaded=false`，不会提交硬件运行产物。

### 5.3 部署阻塞

活动 `qwen36-bf16-final` 的结果：

```text
state = BLOCKED
stop_reason = insufficient_memory
completed_iterations = 0
stable_ref = null
missing_packages = []
```

完整部署预检和活动摘要保存在本机 `var/campaigns/qwen36-bf16-final/`。预检未通过后未终止其他工作负载、未修改权重、未自动改变精度。

下一步是提供足够空闲内存与满足评测约束的设备时段，再执行 `resume qwen36-bf16-final`。若选择量化，应使用单独活动，重新建立正确性参考并实测该后端兼容性。

## 6. 当前实现边界

| 方面 | 边界 |
| --- | --- |
| 真实推理 | BF16 Qwen 权重加载、生成、聊天 Agent、多轮优化尚未得到本机实测结论 |
| 生成范围 | 提供 Python 后端编辑与启动；C++/CUDA/Triton 专用构建、编译产物管理需扩展 |
| 正确性 | 抽样 32 个 logits、短续写、固定 token 和串行请求隔离；未覆盖全 logits、逐层误差、长上下文、并发和多模态 |
| 参考可信度 | 首版从优化前冻结的 Transformers 后端取得参考；独立实现的第二参考后端尚未接入 |
| 性能 | 主判据为控制侧实际请求吞吐；TTFT/TPOT 来自后端时间戳；未接入逐算子 profiler、能耗归因和完整显存峰值测量 |
| 资源 | 仅支持一个控制器数据根目录内的设备互斥；跨项目全局调度由外部负责 |
| 隔离 | Landlock 保护文件写入；CUDA 和标准字符设备可访问；不隔离网络、信号、驱动漏洞或共享缓存干扰 |
| 依赖 | 记录解释器和版本，未把整套 Python/CUDA 环境打包成不可变镜像；恢复时检查画像漂移 |
| 恢复 | 已测进程中断和提交窗口；未实际断电、重启机器或安装 systemd 服务进行验收 |
| 知识 | 只生成待审草案；跨配置规则仍需重复实测与人工审查，未自动宣称发现通用范式 |

设计文档中的更大目标继续作为研究路线。后续优先补齐真实 Qwen 部署与 Agent 验收，再按真实瓶颈引入原生内核和更强正确性评测。
