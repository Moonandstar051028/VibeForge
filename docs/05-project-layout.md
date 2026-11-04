# 05：项目目录与文件放置规范

## 1. 组织原则

| 规则 | 作用 |
| --- | --- |
| 应用源码统一放入 `src/vibeforge/` | 按模块寻找实现，避免根目录堆积 Python 文件 |
| 按领域职责划分子包 | 控制、Agent、引擎、评测、版本管理有清晰入口 |
| 配置、模板、固定评测材料独立维护 | 实验输入可以审阅、版本化和冻结 |
| 候选工作区与发布制品独立 | Agent 修改候选，不覆盖稳定版本或控制程序 |
| 所有运行输出归入 `var/` | 日志、测量、状态、构建产物不会混入源码 |
| 每个目录保留 `README.md` | 进入目录即可看到用途和建议文件 |

下文目录及核心业务文件已实现。当前统一使用 JSON 配置；运行程序会创建真实活动、发布和测量文件。逐模块能力与验证边界见 [实现与验收报告](07-implementation-validation.md)。

## 2. 项目目录树

为便于阅读，树中省略各目录重复的 `README.md` 和各 Python 包的 `__init__.py`。

```text
VibeForge/
├── README.md                     # 项目入口、常用位置与当前阶段
├── pyproject.toml                # Python 包元信息、构建规则与后续依赖声明
├── .gitignore                    # 排除运行数据、环境、缓存与构建输出
├── src/                          # 可安装源码
│   └── vibeforge/
│       ├── cli/                  # 命令行入口
│       ├── controller/           # 监督器、状态机、准入与恢复
│       ├── agent/                # Agent 会话、工具与上下文
│       ├── engine/               # 引擎适配、构建、启动与推理客户端
│       ├── profiling/            # 模型、设备和软件环境画像
│       ├── evaluation/           # 正确性、性能、统计与参考适配
│       ├── versioning/           # 候选冻结、发布、提升和回退
│       ├── observability/        # 事件、日志、指标与报告
│       ├── knowledge/            # 知识抽取、验证与检索代码
│       ├── storage/              # 路径、事务存储、原子写入与锁
│       └── contracts/            # 跨模块类型、协议与数据校验
├── configs/                      # 声明式实验输入
│   ├── campaigns/                # 优化活动、预算、停止条件及配置引用
│   ├── models/                   # 模型、精度、权重引用与功能约束
│   ├── hardware/                 # 设备选择与硬件运行约束
│   ├── engines/                  # 人工初始引擎、适配器和构建命令
│   ├── precision/                # 运行精度与量化配置
│   ├── optimization/             # 精度变化授权与白名单
│   ├── requirements/             # 个性化主目标和资源上限
│   ├── workloads/                # 长度、批大小、并发、采样与负载权重
│   ├── evaluation/               # 正确性标准、指标、统计与复测协议
│   └── agent/                    # Agent 连接、工具、上下文与超时设置
├── benchmarks/                   # 正式推理评测的固定材料
│   ├── cases/                    # 数值、生成、性能与 Agent 用例
│   └── references/               # 可信参考版本、输出清单与哈希
├── tests/                        # 项目自身的自动化测试
│   ├── unit/                     # 单模块行为
│   ├── integration/              # 跨模块协作
│   ├── e2e/                      # 完整闭环与故障恢复
│   └── fixtures/                 # 假引擎、假 Agent 和小型固定材料
├── templates/                    # 可复用输入模板
│   ├── prompts/                  # 恢复、假设、实现、分析等提示词
│   ├── reports/                  # 每轮报告、活动总结、恢复资料模板
│   ├── engine/                   # 初始引擎源码与构建模板
│   └── knowledge/                # 规则、经验与反例模板
├── knowledge/                    # 经整理的优化知识内容
│   ├── rules/                    # 有条件、有证据的优化规则
│   └── counterexamples/          # 退化案例、失效条件和规则限制
├── scripts/                      # 调用应用能力的辅助入口
│   ├── dev/                      # 开发环境与检查脚本
│   ├── ops/                      # 启停、诊断、归档和恢复脚本
│   └── experiments/              # 批量实验、消融和迁移入口
├── deploy/                       # 部署源文件
│   └── systemd/                  # 开机启动与守护服务定义
├── docs/                         # 系统设计、目录规范与后续操作文档
└── var/                          # 持久化运行数据，默认不提交实际产物
    ├── state/                    # 权威状态、锁、租约和事务
    ├── profiles/                 # 采集后的画像快照
    ├── workspaces/               # 可写候选源码和临时构建输出
    ├── releases/                 # 不可变发布制品与恢复版本
    ├── campaigns/                # 完整活动、事件、测量、分析和报告
    └── cache/                    # 可重新生成的下载、编译和检索缓存
```

## 3. 业务源码放在哪里

| 目录 | 放置内容 | 当前主要文件 |
| --- | --- | --- |
| `src/vibeforge/cli/` | 参数解析、命令分发、状态展示 | `main.py`、`campaign.py`、`status.py` |
| `src/vibeforge/controller/` | 独立监督、状态迁移、设备租约、预算、准入和恢复 | `supervisor.py`、`state_machine.py`、`recovery.py`、`admission.py`、`budgets.py` |
| `src/vibeforge/agent/` | 会话生命周期、工具执行、恢复上下文和分析任务 | `runner.py`、`context.py`、`tools.py`、`analysis.py` |
| `src/vibeforge/engine/` | 后端声明、接口、构建、进程、健康检查、写入限制和推理调用 | `adapters.py`、`backend.py`、`builder.py`、`process.py`、`client.py`、`server.py`、`sandbox.py` |
| `src/vibeforge/profiling/` | 模型结构解析、硬件探测、软件环境与运行采样 | `model.py`、`hardware.py`、`environment.py`、`sampler.py`、`resources.py`、`collector.py` |
| `src/vibeforge/evaluation/` | 数值与功能验证、独立质量门禁、基准测量、汇总统计、参考适配 | `correctness.py`、`quality.py`、`performance.py`、`statistics.py`、`reference.py` |
| `src/vibeforge/versioning/` | 冻结候选、哈希、发布清单、提升事务、稳定指针与回退 | `candidate.py`、`artifacts.py`、`release.py`、`promotion.py`、`rollback.py` |
| `src/vibeforge/observability/` | 事件写入、输出采集、指标记录和报告渲染 | `events.py`、`logs.py`、`metrics.py`、`reports.py` |
| `src/vibeforge/knowledge/` | 经验抽取、规则验证、索引检索、证据关联 | `extractor.py`、`validator.py`、`retriever.py`、`registry.py` |
| `src/vibeforge/storage/` | 路径解析、数据库访问、原子写入、刷盘与锁 | `paths.py`、`database.py`、`atomic.py`、`locks.py` |
| `src/vibeforge/contracts/` | 画像、活动、精度、需求、候选、事件、结果与决策的数据契约 | `profile.py`、`campaign.py`、`precision.py`、`requirements.py`、`candidate.py`、`evaluation.py`、`event.py` |

### 3.1 容易混淆的边界

- **评测与准入**：`evaluation/` 产生结果；`controller/admission.py` 按活动规则决定是否接受。
- **决策与切换**：控制程序决定提升或回退；`versioning/` 执行可恢复的版本事务。
- **适配与生成代码**：`src/vibeforge/engine/` 是受维护的适配层；Agent 生成的引擎源码位于 `var/workspaces/`。
- **模板与候选**：`templates/engine/` 是版本化的初始输入；复制至候选工作区后再修改。
- **知识代码与内容**：`src/vibeforge/knowledge/` 是程序；根目录 `knowledge/` 是规则与反例。
- **通用存储与领域行为**：`storage/` 处理事务和文件可靠性；状态迁移、准入与回退规则留在对应业务模块。

### 3.2 依赖方向

1. `cli/` 调用 `controller/` 或公开的领域操作。
2. `controller/` 编排 Agent、引擎、评测、版本和日志等模块。
3. 领域模块使用 `contracts/` 和 `storage/`，不反向导入控制程序。
4. `contracts/` 保持轻量，不依赖推理框架或加速器运行时。
5. 引擎专用依赖留在对应候选和发布环境，控制程序不依赖候选引擎才能启动。

## 4. 配置、评测材料与测试

### 4.1 配置目录

| 目录 | 文件内容 | 命名建议 |
| --- | --- | --- |
| `configs/campaigns/` | 组合引用、预算、停止条件、数据根目录 | `<campaign-name>.json` |
| `configs/models/` | 模型路径、精度、功能声明 | `<model-name>.json` |
| `configs/hardware/` | 设备描述和运行约束 | `<device-name>.json` |
| `configs/engines/` | 人工初始引擎、适配器、能力、依赖与构建命令 | `<adapter-name>.json` |
| `configs/precision/` | 权重/计算/KV/累加精度与量化 | `<precision-name>.json` |
| `configs/optimization/` | 精度搜索授权及白名单 | `<policy-name>.json` |
| `configs/requirements/` | 主目标、资源上限及文字偏好 | `<requirements-name>.json` |
| `configs/workloads/` | 提示词、输出长度、并发和固定用例 | `<workload-name>.json` |
| `configs/evaluation/` | 正确性标准、指标、统计门槛和复测规则 | `<protocol-name>.json` |
| `configs/agent/` | Agent 模式、工具、提示词引用与会话预算 | `<agent-profile>.json` |

配置加载器只接受 JSON。模型权重实体通过外部路径引用；实际采集结果、执行状态与指标写入 `var/`。硬件配置可给出设备、架构与名称约束，实际能力以推理环境探测为准。`scripts/evaluation/` 保存受信任的任务质量评测程序；它的用例仍放在 `benchmarks/cases/`。

### 4.2 正式评测与项目测试

| 目录 | 放什么文件 | 主要验证对象 |
| --- | --- | --- |
| `benchmarks/cases/` | 固定输入、JSON/JSONL 用例、小型张量样例、数据集清单 | 生成的推理引擎 |
| `benchmarks/references/` | 参考版本、输出清单、哈希、生成参数、小型参考输出 | 正确性对照依据 |
| `tests/unit/` | 按源码模块分组的 `test_*.py` | 单个模块与规则 |
| `tests/integration/` | 事务、构建、日志、上下文等协作测试 | 多模块交互 |
| `tests/e2e/` | 完整迭代、候选拒绝、故障与重启恢复测试 | 系统闭环 |
| `tests/fixtures/` | 假引擎、假 Agent、最小配置、样例事件与清单 | 测试辅助材料 |

大型参考张量通过清单、哈希和外部路径管理；正式测量输出统一归档到活动目录。测试运行时生成的数据库和临时工作区由测试临时目录承载。

## 5. 模板、知识与辅助文件

| 目录 | 放置文件 |
| --- | --- |
| `templates/prompts/` | `resume.md`、`hypothesis.md`、`implementation.md`、`analysis.md` 等提示词 |
| `templates/reports/` | 单轮报告、活动总结、恢复资料包的 Markdown 或结构化模板 |
| `templates/engine/` | 初始引擎源码、启动脚本、依赖和原生构建模板 |
| `templates/knowledge/` | 规则、局部经验、反例的字段与编写模板 |
| `knowledge/rules/` | 以稳定 `rule_id` 标识的规则，包含条件、实现、验证、证据和状态 |
| `knowledge/counterexamples/` | 退化或失效案例摘要，关联规则版本和完整实验 |
| `scripts/dev/` | 环境检查、格式检查、开发辅助入口 |
| `scripts/ops/` | 服务启停、诊断、归档和恢复脚本 |
| `scripts/experiments/` | 批量活动、消融实验、迁移与最终确认评测入口 |
| `deploy/systemd/` | `vibeforge-controller@.service` 用户服务模板和部署说明 |
| `docs/` | 架构、流程、评测、日志、知识库、目录规范和后续操作说明 |

脚本只封装调用与批量操作，核心逻辑仍位于应用包。自动抽取的知识草案先保存在所属活动中，完成验证和整理后再进入根目录知识库。

## 6. 运行数据的放置规则

| 目录 | 运行时文件 | 保存性质 |
| --- | --- | --- |
| `var/state/` | 权威状态数据库、锁、设备租约、检查点、事务记录 | 持久化，恢复入口 |
| `var/profiles/` | 采集后的模型、设备、环境与组合画像 | 持久化，按哈希引用 |
| `var/workspaces/<campaign_id>/<iteration_id>/` | 可修改候选源码、配置和临时构建输出 | 可写；冻结后保留可追溯快照或差异 |
| `var/releases/<version_id>/` | 不可变发布清单、源码、依赖、运行配置与构建产物 | 持久化，支持回退 |
| `var/campaigns/<campaign_id>/` | 活动配置快照、事件、命令、测量、决策和分析 | 持久化，完整实验档案 |
| `var/cache/` | 下载与编译缓存、参考数据缓存、知识检索索引 | 可重建，不能保存唯一证据 |

### 6.1 单个活动的动态目录

下列子目录由运行程序按需创建，不预先生成虚构的活动或测量结果。

```text
var/campaigns/<campaign_id>/
├── manifest.json                 # 活动身份、目标、预算和配置哈希
├── configs/                      # 活动启动时冻结的配置快照
├── events/                       # 追加式结构化事件流
├── resume/                       # 当前上下文、最近结论与历史索引
├── knowledge-drafts/             # 待验证的局部经验和规则草案
├── summary.md                    # 活动结束后的总结
└── iterations/<iteration_id>/
    ├── hypothesis.md             # 修改前的瓶颈假设与验证方案
    ├── candidate.json            # 父版本、源码和制品哈希
    ├── change.patch              # 相对父稳定版本的代码差异
    ├── commands.jsonl            # 命令、工具调用、参数、退出码和时长
    ├── logs/                     # 构建、引擎和 Agent 原始输出
    ├── correctness/              # 误差报告、失败用例和最小复现
    ├── benchmarks/               # 原始测量、设备采样、统计与收益
    ├── decision.json             # 接受、拒绝、复测的原因与证据
    └── analysis.md               # 事实、原因解释、反例与下一步
```

### 6.2 候选引擎与发布制品

```text
模板或当前稳定版本
→ var/workspaces/<campaign_id>/<iteration_id>/ 中修改
→ 冻结候选并评测
→ var/releases/<version_id>/ 中封存不可变发布单元
→ 控制程序通过权威状态切换 stable_ref
```

发布目录名和工作区名都不能单独代表准入结果；是否稳定以权威状态中的 `stable_ref` 和已提交事务为准。失败候选的日志、差异和分析继续保存在活动档案中。

## 7. 命名与版本管理约定

- Python 包、模块使用 `snake_case`；测试文件使用 `test_<subject>.py`。
- 配置使用描述性名称，活动和运行目录使用唯一标识，避免反复覆盖 `latest.log`。
- 代码、配置、模板、目录说明、知识条目和小型固定材料纳入版本管理。
- `var/` 中的实际产物默认忽略，仅明确列出的目录说明随项目维护。
- 运行档案的长期保存通过独立存储或归档管理；当前稳定及恢复版本、研究证据需要保留。
- 引擎权重、依赖环境和大型原始数据通过路径、版本和哈希引用。
- 根目录保持入口文件和一级功能目录；跨模块公共代码应有明确职责，避免使用无边界的 `utils/`。

## 8. 新文件放置速查

| 准备新增的文件 | 放置位置 |
| --- | --- |
| 重启后检查未完成提升事务 | `src/vibeforge/controller/recovery.py` |
| 稳定指针切换和回退事务 | `src/vibeforge/versioning/` |
| 吞吐收益置信区间计算 | `src/vibeforge/evaluation/statistics.py` |
| 新硬件的探测逻辑 | `src/vibeforge/profiling/` |
| Agent 重启提示词 | `templates/prompts/resume.md` |
| 某次候选生成的设备内核 | `var/workspaces/<campaign_id>/<iteration_id>/` |
| 新增 KV cache 正确性用例 | `benchmarks/cases/` |
| 验证回退逻辑的测试 | `tests/integration/` 或 `tests/e2e/` |
| 已验证的算子融合知识 | `knowledge/rules/` |
| 某轮性能下降的原始数据和解释 | 对应活动的 `iterations/<iteration_id>/benchmarks/` 和 `analysis.md` |

架构与职责依据见 [总体架构](01-system-architecture.md)，运行档案的字段和持久化要求见 [评测、日志与上下文恢复](03-evaluation-and-observability.md)。
