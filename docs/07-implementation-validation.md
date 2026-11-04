# 07：实现与本地验收报告

验证日期：2026-11-04（北京时间）；事件文件保存 UTC 时间。

## 0. 多样化系统增量验收（schema v2）

现行能力、字段和运行边界见 [09：多样化引擎系统](09-diverse-engine-system.md)。后续各节保留首版历史验收，旧测试数量和内存观测不代表当前实时状态。

| 本次增量 | 结果 |
| --- | --- |
| 完整工程回归 | `82 passed, 1 deselected`，约 98 秒；包含真实小型模型接口项，未选择 Qwen35B 目标测试 |
| 可组合配置 | 模型、设备、适配器、精度、需求、优化白名单独立配置；旧 dtype/quantization 字段归一化兼容 |
| 多格式画像 | 单文件/分片 safetensors、GGUF 头部、受限路径 manifest；目标精度参与内存估算与画像身份 |
| 精度搜索闭环 | 白名单内切换通过时提升；独立质量失败时恢复初始稳定版本；恢复上下文使用稳定版本精度 |
| 性能路径一致性 | 普通与固定长度生成都受差分检查，计时请求逐次检查长度与 token，错误快速路径被拒绝 |
| 个性化目标 | 吞吐最大化；请求 P95/RSS/启动/制品体积最小化；独立资源上限 |
| 构建及恢复 | 有界构建、制品冻结、越界写入拦截；提交前/后的真实进程崩溃恢复测试 |
| 真实 CPU 模型 | 离线随机小型 GPT-2、Llama，各 FP32/BF16 共四组合：引擎启动、生成、logits、固定长度及并发请求隔离通过 |
| 未验证范围 | 真实 Qwen35B 完整部署和 Agent、多种量化内核、CUDA/ROCm 完整模型优化、长上下文和多模态 |

新增真实小模型测试由本机已有推理解释器运行，无网络下载，不使用已有大模型权重。随机权重只验证软件接口，不衡量模型质量或自优化收益。

本次推理环境为 PyTorch `2.13.0+cu132`、Transformers `5.19.0`。Qwen35B 元数据预检重新读取了 1,045 个 tensor，`missing_packages=[]`；权重为 71,903,645,408 字节，BF16 需求估算仍为 84,098,387,984 字节（78.32 GiB），本次可用内存 31,699,218,432 字节（29.52 GiB），结果为 `insufficient_memory`。没有加载完整大模型权重或运行其 Agent。

复现：

```bash
# 框架工程测试；未设置推理解释器时，真实小模型接口项明确跳过
.venv/bin/python -m pytest -m 'not target' -q

# 将真实小模型接口项纳入完整工程检查
VIBEFORGE_TEST_INFERENCE_PYTHON=/path/to/transformers-env/bin/python \
  .venv/bin/python -m pytest -m 'not target' \
  --junitxml=var/validation-diversity/tests.xml -q
```

真实 CPU 单项 JUnit：`var/validation-diversity/transformers-cpu.xml`。
完整回归 JUnit：`var/validation-diversity/tests.xml`。另检查 93 个 Python 文件语法、35 个静态 JSON，以及 README/docs 中的 Markdown 文档链接；全部通过。

## 1. 首版结论（历史）

| 验收项 | 结果 |
| --- | --- |
| 项目核心模块 | 已实现可运行首版，无仅抛出 NotImplemented 的业务占位 |
| 工程测试 | 47 项通过，含 Qwen-Agent 模拟服务工具调用及控制器 worker 集成测试 |
| CPU 完整闭环 | 通过：`qwen-agent-toy-20261104` 两轮完成，第一轮接受，第二轮正确性拒绝并保持第一轮版本 |
| 故障恢复 | 测试通过：提升意图写入后控制程序退出，新进程恢复保留已提交版本 |
| Qwen-Agent 框架 | 通过：真实 `Assistant` 对接模拟 OpenAI SSE 服务，完成候选编辑、语法检查和提交 |
| Toy 引擎框架冒烟 | 通过：Qwen-Agent `Assistant` 经引擎 `/v1/chat/completions` 完成工具调用 |
| Qwen BF16 完整部署 | 未通过：配置现有推理环境后仍因内存不足阻塞，完整权重未加载 |
| 真实 Qwen Agent 与自优化 | 尚未验证；框架接入通过，真实 Qwen 权重推理受上述条件阻塞 |
| 可迁移优化范式 | 已有草案与证据管理机制，尚无经真实多配置实验验证的通用规则 |

工程通过与目标模型部署未通过分别保留结果，不合并为“系统全部通过”。

## 2. 模块与实现入口

| 模块 | 已实现职责 | 主要文件 |
| --- | --- | --- |
| CLI | 画像、启动、恢复、状态、停止、事件、报告、子进程入口 | `cli/main.py`、`campaign.py`、`status.py` |
| 控制 | 自举、有限迭代、固定预算、状态机、统计准入、恢复编排 | `controller/supervisor.py`、`state_machine.py`、`budgets.py`、`admission.py`、`recovery.py` |
| Agent | Qwen-Agent `Assistant` 本地模型配置、工具适配、独立会话、恢复上下文、候选编辑与结果分析 | `agent/runner.py`、`tools.py`、`context.py`、`analysis.py` |
| 引擎 | Python 后端构建检查、HTTP 服务、OpenAI SSE 兼容接口、健康检查、进程所有权、写入限制 | `engine/backend.py`、`builder.py`、`server.py`、`client.py`、`process.py`、`sandbox.py` |
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
| Agent | Qwen-Agent 框架实际解析并调用工具、候选编辑与提交；控制器启动独立 worker 和最终分析会话；工具越界审计；原生 Qwen 工具格式转换 |
| 写入限制 | 生成进程可写缓存，不能改控制状态；允许库访问 `/dev/null` |
| 画像与知识 | 分片缺失/头部越界检查、资源阻塞、配置校验、知识适用边界 |

工程测试命令 `.venv/bin/python -m pytest -m 'not target' -q` 结果为 `47 passed, 1 deselected`。JUnit 位于本机 `var/validation-qwen-agent-final/tests.xml`。真实目标命令启用 `VIBEFORGE_TARGET_TEST=1` 并配置现有推理环境后，目标测试有 1 项失败，原因为 `BLOCKED: insufficient_memory`；未进入引擎启动或完整模型加载，不按通过或跳过处理。

另已检查 79 个 Python 文件语法、20 个静态 JSON 文件、全部 shell 脚本语法、静态子目录说明和新增文档链接。活动退出后没有残留的引擎或 Agent 进程记录。

## 4. CPU 活动实测

活动：`qwen-agent-toy-20261104`，状态 `STOPPED`，完成 2 轮，`analysis_pending=false`，退出后 `active_ref=null`。控制器中的候选优化决策使用专门的 `scripted` 测试夹具；自举、候选激活和恢复检查中的 Assistant 冒烟通过 Qwen-Agent 完成。

| 轮次 | 修改 | 结果 |
| --- | --- | --- |
| 1 | 把每个输出 token 的重复求和换成等价闭式算式 | 正确性通过；按实际计时接受候选 |
| 2 | 故意改变输出数值 | 正确性拒绝；稳定指针保持第一轮版本 |

该次第一轮相对吞吐收益约 `gain=37.33`；95% bootstrap 区间约 `[34.55, 39.25]`。这是刻意设置冗余 CPU 运算的测试引擎，不能解释为 Qwen 加速。

活动摘要、各轮判定和完整事件流保存在本机 `var/validation-qwen-agent/campaigns/qwen-agent-toy-20261104/`。同样的两轮流程在最终活动 `var/validation-qwen-agent-final/campaigns/qwen-agent-final-20261104/` 再次通过。运行产物不提交到版本库。

## 5. Qwen 本机部署结果

### 5.1 硬件与模型

| 项目 | 采集结果 |
| --- | --- |
| 平台 | aarch64，NVIDIA Thor，统一内存，Linux 6.8.12-1021-tegra |
| 驱动 / 设备 | NVIDIA Thor，统一内存；驱动探测为 595.78 |
| 模型路径 | `/home/yxy/weights/Qwen3.6-35B-A3B`，通过 `VIBEFORGE_MODEL_PATH` 传入，不进入版本库 |
| 模型类 | 权重元数据声明 `Qwen3_5MoeForConditionalGeneration`；本次没有加载模型权重 |
| 模型结构 | 40 层混合注意力，256 专家、每 token 选择 8 个专家 |
| 分片 | 26 个 safetensors 分片，索引和头部检查通过 |
| 权重字节 | 71,903,645,408，约 66.96 GiB |
| BF16 需求估算 | 84,098,387,984 字节，约 78.32 GiB |
| 本次活动预检可用内存 | 39,984,177,152 字节，约 37.24 GiB |

本次没有重新计算全部约 67 GiB 权重的内容校验和；画像记录配置、tokenizer 等文件哈希及分片清单/大小/头部检查范围。此前下载校验材料不作为本次完整校验的替代。

### 5.2 预检与框架探针

- 本机默认 `runtime_python=python3` 缺少推理依赖；通过 `VIBEFORGE_RUNTIME_PYTHON` 指向现有推理环境后，预检的 `missing_packages=[]`。
- 本次真实活动可用内存约 37.24 GiB，BF16 估算约 78.32 GiB；预检因此在引擎子进程启动前停止。
- 使用现有推理环境、真实本地 tokenizer 和 `chat_template.jinja` 验证多轮工具历史：OpenAI 参数字符串转对象、原生工具渲染、代码缩进与末尾换行保留均通过；生成响应由测试数据代替，没有加载权重。
- Qwen-Agent 框架本身通过 `.venv` 中的本地模拟 SSE 服务和 Toy 后端完成集成测试；这不等同于真实模型推理。
- 独立于系统包的全新虚拟环境安装项目后，Qwen-Agent `Assistant` 与项目模块导入通过。

权重和预检运行数据留在本机忽略目录，不提交到版本库。

### 5.3 部署阻塞

活动 `qwen-agent-bf16-20261104` 的结果：

```text
state = BLOCKED
reasons = insufficient_memory
completed_iterations = 0
stable_ref = null
missing_packages = []
```

活动预检及画像位于本机 `var/validation-qwen-agent-final/`；真实目标测试也记录了同样的内存阻塞。预检未通过后未终止其他工作负载、未修改权重、未自动改变精度。

下一步是提供足够空闲内存与满足评测约束的设备时段，使用已配置的推理环境重新运行真实目标测试。若选择量化，应使用单独活动，重新建立正确性参考并实测该后端兼容性。

## 6. 首版实现边界（历史，现行边界见 09）

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
