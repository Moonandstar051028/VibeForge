# 06：本地运行与恢复

## 1. 当前可用范围

| 项目 | 首版行为 |
| --- | --- |
| 控制程序 | 控制与存储核心主要使用 Python 标准库，SQLite 保存权威状态；控制环境安装 Qwen-Agent，但不依赖 Torch 或候选推理环境 |
| 系统 | Linux aarch64 / x86_64，需要 Landlock；本机 ABI 4 已验证 |
| 测试后端 | 纯 Python CPU 引擎 + 明确标注的脚本 Agent |
| 模型后端 | 本地 Transformers 文本生成、聊天和抽样 logits |
| 优化方式 | Agent 使用稳定引擎推理，在独立候选工作区修改 Python 后端 |
| 评测 | 1—64 个客户端；确定性与近似模式；吞吐/时延/内存/启动/体积目标，详见 [09](09-diverse-engine-system.md) |
| Qwen 状态 | 35B/BF16 首版历史预检曾因内存阻塞；当前模型/精度须按活动独立检查，不能沿用历史资源结论 |

## 2. 创建控制环境

```bash
cd /path/to/VibeForge
bash scripts/dev/bootstrap.sh
.venv/bin/vibeforge --help
.venv/bin/python -m pytest -m 'not target' -q
```

`bootstrap.sh` 创建项目 `.venv`，复用系统测试工具，并以 editable 方式安装项目及 Qwen-Agent 等控制环境依赖。控制器和常规 CLI 不导入 Torch；Torch、Transformers、bitsandbytes 等推理依赖由活动中的 `runtime_python` 负责。

## 3. 先跑完整 CPU 闭环

```bash
.venv/bin/vibeforge run demo-001 --config configs/campaigns/toy.json
.venv/bin/vibeforge status demo-001
.venv/bin/vibeforge events demo-001 --after 0
.venv/bin/vibeforge report demo-001
```

每次新活动使用新 ID。该配置执行两轮：第一轮把重复 CPU 求和换成等价算式，第二轮故意改变数值输出。预期接受第一轮、拒绝第二轮、保留第一轮的稳定版本，最后补写分析并停止。速度由实际请求计时得到；脚本 Agent 只验证工程流程。

## 4. Qwen 环境与启动

### 4.1 配置入口

| 文件 | 关键字段 |
| --- | --- |
| `configs/campaigns/qwen36-35b.json` | 推理解释器、额外依赖目录、启动超时、活动预算 |
| `configs/models/qwen36-35b.json` | `weights/Qwen3.6-35B-A3B`（默认相对路径，可由环境变量覆盖）、BF16、CUDA、8192 上下文 |
| `configs/agent/local.json` | Qwen-Agent `Assistant`、本地 OpenAI 兼容接口、最多 6 次模型调用、每次最多 512 新 token |
| `configs/agent/codex.json` | 可选 Codex CLI 后端；Responses 请求仍发送给当前稳定引擎 |
| `configs/workloads/smoke.json` | 固定提示词、16 个输出 token、文本正确性用例 |
| `configs/campaigns/qwen35-27b-nf4.json` | 本机 27B NF4、BF16 compute、32K 上下文、代表性矩阵、thinking Agent 4096 输出上限；路径需按部署机器调整 |
| `configs/campaigns/qwen35-27b-thinking-study.json` | Qwen3.5-27B thinking 输出探测；128K 上下文、32768 探测上限、单次提案预算，不作为日常活动配置 |
| `configs/campaigns/qwen35-27b-codex-context-study.json` | Codex CLI 通过本地 Qwen3.5-27B 引擎执行三次提案；使用 Codex 默认自动压缩并保存原始事件和上下文摘要 |
| `configs/workloads/representative.json` | 256/2K/8K 输入、64/128/256 输出及并发 2 的五场景矩阵；正式多场景评测使用此配置 |
| `configs/evaluation/local.json` | 5 组重复、交替顺序、2% 最低收益、95% 区间、GPU 独占要求 |

设置本机权重和推理解释器路径：

```bash
export VIBEFORGE_MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export VIBEFORGE_RUNTIME_PYTHON=/path/to/inference-env/bin/python
```

模型权重和推理环境不随 Git 发布。`VIBEFORGE_MODEL_PATH` 与 `VIBEFORGE_RUNTIME_PYTHON` 覆盖配置里的通用默认值。缺少的 `accelerate==1.15.0` 可安装在 `var/cache/inference-packages/`，由 `inference_pythonpath` 注入。

选择 Agent 时修改活动配置的 `agent` 引用，或内联设置 `agent.framework`。例如，把 `"agent": "configs/agent/local.json"` 改成 `"agent": "configs/agent/codex.json"`。`qwen-agent` 是兼容默认值；`codex` 要求本机可执行 `codex`，但不需要 OpenAI API key，因为自定义 provider 指向本轮稳定 VibeForge 引擎。

```bash
bash scripts/dev/prepare_inference.sh
.venv/bin/vibeforge profile --config configs/campaigns/qwen36-35b.json
.venv/bin/vibeforge run qwen-local-001 --config configs/campaigns/qwen36-35b.json
```

`profile` 和 `run` 返回码：`0` 为通过/正常停止，`2` 为预检阻塞；其他异常返回非零。资源不满足时先记录 `preflight.json` 和 `deployment_blocked` 事件，再退出，不尝试分配完整权重。

### 4.2 不加载权重的环境探针

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:var/cache/inference-packages \
"${VIBEFORGE_RUNTIME_PYTHON:-python3}" \
  scripts/dev/probe_inference.py \
  --model "${VIBEFORGE_MODEL_PATH:?Set VIBEFORGE_MODEL_PATH first}" \
  --confined-cache "$PWD/var/cache" \
  --output "$PWD/var/cache/inference-confined-probe.json"
```

探针验证 CUDA 运算、聊天模板、模型类及 meta 结构。`model_weights_loaded=false` 表示没有验证真实权重推理。启用写入限制时，输出必须放在允许写入的缓存目录。

### 4.3 资源与精度边界

- 2026-11-04 的 35B/BF16 历史样本：权重约 66.96 GiB；按权重 × 1.08 + 6 GiB 预留，约需 78.32 GiB 可用内存。这不是 27B 或 4-bit 的估算。
- 历史验收采样约 37.5 GiB 可用；每次启动重新采样，其他 GPU 进程会使独占性能测量无效。
- 保持 BF16 时，需要先由资源所有者安排足够空闲内存和 GPU。完成后可恢复被阻塞活动，启动前会重新预检。
- 新精度契约支持 bnb NF4/FP4/INT8；较早的三模块更新仅做小模型验收，后续完整 NF4 模型验证见下文。默认精度固定；显式白名单搜索及质量约束见 [09](09-diverse-engine-system.md)。初始引擎必须先能部署。

后续观测矩阵增量已在本机 A100 40GB 的 Qwen3.5-27B NF4 上完成五场景生成验证，峰值 allocated 约 17—21 GiB，具体记录见 [07](07-implementation-validation.md)。这是加载与观测验证，不代表更高精度可直接部署，也不是候选优化的正式收益。

### 4.4 提案预算和诊断配置

新活动配置可显式包含以下片段；与模型、负载和评测协议一起冻结：

```json
{
  "budget": {
    "max_iterations": 10,
    "max_proposals": 30,
    "max_stalled_attempts": 3,
    "max_seconds": 3600,
    "agent_seconds": 240,
    "max_failures": 3
  },
  "profiling": {
    "max_new_tokens": 128,
    "max_profiles_per_session": 2,
    "max_age_seconds": 600
  }
}
```

`max_iterations` 约束完成实验数；`max_proposals` 约束全部提案尝试；`max_stalled_attempts` 限制连续没有有效候选。无修改、重复候选或证据不足会保留尝试档案，不能作为完成实验。摘要分别显示三个计数，旧字段 `completed_iterations` 仍表示尝试编号。

`profiling` 是推理阶段诊断配置，不是 CLI `vibeforge profile` 的部署预检。控制器在稳定引擎上调用 HTTP `/profile`，按 `coverage` 和 `max_matrix_profiles` 建立冻结场景/prompt 矩阵；缓存和单轮快照均绑定源码、画像与负载。Agent 可通过 `profile_inference` 有界追加诊断。输出上限同时受所选场景约束且最多 128 token；每会话次数由 `max_profiles_per_session` 冻结。内部 token-ready TTFT/TPOT 可观测，缓冲 HTTP 的客户端首 token 时间仍不可观测。旧活动的 `observation` 对象只作为加载迁移别名保留，新配置不要同时声明两者。

同一个稳定引擎也承载 Agent 的工具调用。新活动的引擎输出上限取初始模板、`agent.max_new_tokens` 和所有工作负载场景输出长度的最大值，避免 Agent 或某场景被模板默认值拒绝；正式性能测试使用各场景冻结长度，上下文总长度仍受 `model.max_context` 限制。

`agent.thinking.enabled` 控制 Agent 的 OpenAI 兼容端点是否使用模型原生 thinking 模板。thinking 模式的采样参数也在此对象中冻结；内部质量评测 `/chat` 始终使用非 thinking 确定性生成，避免短答案用例把输出预算耗在思考段。输出上限是模型和任务画像参数，不应跨模型照搬。`qwen35-27b-thinking-token-study-003` 在当前 Qwen3.5-27B/NF4/A100 流程中记录了 16 次调用，最大 1312、P95 1189，且没有长度截断，因此日常配置采用 4096，保留约三倍于实测最大值的余量。探测原始证据与摘要位于对应活动目录。

选择 `codex` 时必须使用 `templates/prompts/codex-implementation.md` 或另一个 Codex 专用提示；配置校验会拒绝 Qwen-Agent 的 `templates/prompts/implementation.md`。VibeForge 把有界摘要挂载为 `.vibeforge/brief.json`，完整、未删减的活动上下文仍临时挂载为 `.vibeforge/context.json`，但只允许在摘要缺字段时定向查询，不能整文件输出进消息历史。Codex 路径不调用 Qwen 的 `_fit_model_context` 或 `_model_context`；之后的消息历史、工具输出裁剪和自动摘要全部由 Codex CLI 管理，临时上下文不会同步进候选引擎。

正式 Codex 结果通过 VibeForge 专用的 stdio MCP `record_result` 提交。同一个宿主 MCP 还按会话权限提供 `list_evidence`、`read_evidence`、`search_history`、`read_telemetry` 和 `profile_inference`，与 Qwen-Agent 使用同源的有界回调和预算。MCP server 位于候选沙箱之外，`record_result` 按 `.vibeforge/submission-schema.json` 同源 schema 校验后原子写入活动证据；候选不能修改提交工具或伪造宿主结果。最终聊天文本只保留作审计，不能直接授权候选同步。分析会话的临时 workspace 可写，但编辑全部丢弃，只有结构化提交对象被宿主接收。

Codex 本地模型可能把“接下来检查”误当作最终回答，也可能忽略 Prompt 中的工具次数建议。适配器因此按 `agent.max_passes`（默认 3，范围 1—5）检查实际文件 diff，每阶段同时受宿主硬超时约束（最多 300 秒，并受总 `timeout_seconds` 均分限制），超时时终止整个 Codex/MCP 进程组。没有编辑也没有提交时，在同一临时候选上启动下一工作阶段；产生 diff 后不再要求弱模型自行决定提交时机。若 MCP 仍未被调用，宿主向同一稳定引擎发出一次关闭 thinking、强制指定 `record_result` 的函数调用请求；该工具使用扁平参数以兼容本地 Qwen 工具模板，宿主再还原并按权威 schema 校验。只有 MCP 或该强制工具阶段产生的非阻塞结果才允许同步候选并进入构建、正确性和 A/B 性能测试。

`analysis-only` 复盘不执行多阶段候选编辑：Codex 只运行一轮，未调用 MCP 时立即进入同一严格结构化提交阶段。候选门禁补全的 objective、direction、证据绑定、目标 profile 和运行时前置条件会被写回 `agent-result.json` 与数据库，后续性能验证只读取这份规范化计划。

强制结构化提交时，是否存在候选 diff 是宿主事实：有 diff 就进入候选门禁，不能因“尚无改后性能数据”标记为 blocked；无 diff 才是阻塞提案。复盘的目标 iteration、允许的 hypothesis status 和证据路径同样由宿主绑定，模型只负责解释实验结果。`brief.json.candidate_constraints` 提供当前控制器认可的 cache 与 attention 字面值，避免本地模型猜测配置名称。

VibeForge 不切割 Codex 消息，但会把真实 `model_context_window`、最大输出预算和工具输出预算转换成 Codex 自己的自动压缩阈值，使压缩请求在本地引擎硬上限前发生。每个会话的原始事件和解析摘要分别写入 `codex-events.jsonl` 与 `codex-context.json`。

不能直接使用 Codex 默认约 90% 的压缩点：32K 引擎在输入接近 28K 时还需为 4096 输出 token 留空间，实测默认点尚未触发，请求就被引擎拒绝。当前适配器将 Codex 的工具输出限制设为 `min(4096, agent.max_new_tokens)`，自动压缩点按“模型窗口 - 最大输出 - 工具结果 - 安全余量”计算。该值只触发 Codex 自己的摘要和历史替换，不调用任何 VibeForge 上下文裁剪函数。

`agent.max_turns` 对 Codex 表示提示内的工具执行上限；Codex CLI 当前没有由 VibeForge 强制终止第 N 个工具调用。超时仍由 `agent.timeout_seconds` 和活动 `agent_seconds` 硬性执行。研究配置使用 8 次工具执行，以覆盖一次自动压缩并限制单会话诊断成本。

thinking 响应只有在模型生成闭合的 `</think>` 后才算完成。若 thinking 段异常提前结束，Transformers 后端会把同一轮对话和工具重新渲染为普通 assistant 前缀，再自动生成一次；该回退仅影响本次响应，不改变活动中 thinking 配置。若重试仍无可见内容/工具调用，Responses 返回 `status=failed` 与标准 `server_error`，不会伪装成功。Responses 的 `incomplete_details.reason` 只使用官方枚举，因此这里不添加自定义 incomplete reason。Codex Prompt 也要求对 `.vibeforge/context.json` 做定向 `jq` 查询，不把完整源码和历史通过 shell 输出重复塞进对话。

Codex CLI 的工具沙箱也是运行依赖。本机扩展附带的 `0.155.0-alpha.16` 在存在 Docker/Snap `nsfs` mount 时会错误报告 `mountinfo path is not absolute`；研究配置暂时固定到本机已安装并通过 `codex sandbox -- /bin/true` 的 `0.162.1`。其他部署应把 `agent.executable` 改成其已验证的 Codex 路径，不能通过关闭沙箱掩盖该错误。

在本机使用新矩阵启动独立活动的命令：

```bash
# 先确保没有本项目旧活动占用 GPU 0；不终止其他用户的 GPU 进程
CUDA_VISIBLE_DEVICES=0 .venv/bin/vibeforge run qwen27-nf4-matrix-001 \
  --config configs/campaigns/qwen35-27b-nf4.json
```

活动 ID 必须唯一。该配置授权长时间自动迭代，但无有效候选、连续故障和用户停止仍按预算/状态机处理。`CUDA_VISIBLE_DEVICES=0` 将本活动限制到第一张物理卡；资源独占与 GPU 温度检查只检查可见设备，不能把另一张卡的进程当作本卡污染。运行前应核对本机路径和可见设备顺序。五组 A/B、每场景退化不超过 3% 与全部输出正确性检查在配置中冻结，耗时明显高于 smoke。

## 5. 查看证据

| 路径 | 内容 |
| --- | --- |
| `var/state/controller.sqlite3` | 活动状态、迭代、提升事务、事件 outbox；权威来源 |
| `var/profiles/<hash>.json` | 模型结构、设备、软件版本和负载画像 |
| `var/campaigns/<id>/manifest.json` | 启动时冻结的组合配置及哈希 |
| `configs/` | 此处指活动内目录：冻结的模型等配置、Agent 提示词、用例 |
| `preflight.json` | 资源、依赖和部署准入检查 |
| `events/events.jsonl` | 数据库事件的可读导出，带 UTC、单调时钟和 boot ID |
| `bootstrap/`、`reference.json`、`initial-quality.json` | 初始协议检查、重复回归参考、基线稳定性及独立任务质量最差分数 |
| `iterations/0001/commands.jsonl` | Agent 模型响应、工具调用、进程命令和退出状态 |
| `iterations/0001/agent-telemetry.json` | 本次 Agent 使用稳定引擎推理时的普通请求阶段摘要；诊断用途，不是准入样本 |
| `iterations/0001/logs/` | Agent 和各次引擎启动的原始输出 |
| `iterations/0001/logs/*.runtime.json` | 各次引擎 `/health` 的真实加载配置、后端、依赖及执行路径 |
| `iterations/0001/benchmarks/` | 原始样本、资源快照、区间和判定 |
| `observations/<stable>/`、`iterations/0001/observation.json` | 宿主性能诊断缓存、追加观测、本次提案的证据快照 |
| `iterations/0001/candidate-gate.json` | 实际改动、指纹去重与结构化计划检查结果 |
| `iterations/0001/probe/`、`iterations/0001/refinement/` | quick probe 的小样本反馈与各次独立修订证据；不进入正式 A/B 统计 |
| `iterations/0001/candidate-observation.json`、`verification.json` | 候选同条件诊断，目标瓶颈变化及事前预测幅度与配对实测区间的对照 |
| `iterations/0001/analysis.md` | 可核实的实验事实与 Agent 解释分开记录 |
| `memory/index.json`、`memory/experience.json`、`iterations/0001/memory.json` | 可重建的全量历史索引、已验证策略/反例台账和单轮计划、事实、解释、反思 |
| `final-analysis/<目标编号>/` | 各项只读复盘会话的上下文、工具轨迹和结果 |
| `resume/context.json` | 从权威状态和历史记录重建的上下文 |
| `knowledge-drafts/` | 带证据引用的草案，不自动变为通用优化规则 |
| `summary.md` | 当前状态、稳定版本和各轮结果 |

日志持续追加并刷盘；事件先随事务写入 SQLite，再导出 JSONL。引擎输出实时重定向到文件。终端可使用 `tail -f` 查看文件，或反复调用 `events <id> --after <sequence>` 读取新增事件。

## 6. 停止、恢复与版本切换

```bash
.venv/bin/vibeforge stop demo-001
.venv/bin/vibeforge resume demo-001
```

`stop` 请求在当前轮结束后停止；Agent、推理请求和启动过程各自受超时限制。`resume` 会：

1. 获取活动锁和数据根目录内的设备锁，禁止两个活动同时评测。
2. 按 PID、启动时刻、boot ID、随机标记清理本活动遗留进程。
3. 取消尚未提交的提升意图；未提交迭代记为中断拒绝，已经提交的提升保留接受判定；恢复待分析队列。
4. 重建画像和上下文，确认模型、硬件及软件环境没有漂移，再继续。

有待分析记录时，先逐项完成复盘，之后才规划下一候选；分析没有进展会进入 `HALTED` 并保留队列。用户主动停止会跳过后续复盘；键盘中断记录中断原因、清理本活动进程，之后保持停止。不会为了补齐分析自动重启已被用户要求暂停的实验。

预算从活动首次启动计算，包含停机时间；达到预算的活动恢复后会结束。模型、软件或验收协议改变时，应创建新活动；精度只有在原活动已授权的白名单内才可切换。历史 campaign schema v1/v2/v3 活动保留查看；统一闭环、telemetry、矩阵画像、quick probe 与修订配置属于 schema v4，新评测必须创建 v4 活动。

发布版本通过内容哈希冻结，启动前重新校验。候选不满足正确性、性能或执行要求时，重新启动原稳定引擎并进行协议检查；活动结束后清理本活动推理进程。

## 7. 开机恢复服务

`deploy/systemd/vibeforge-controller@.service` 是用户级服务模板，当前没有安装或启用。使用前先修改其中项目路径，并确保活动已创建：

```bash
mkdir -p "$HOME/.config/systemd/user"
cp deploy/systemd/vibeforge-controller@.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now vibeforge-controller@demo-001.service
journalctl --user -u vibeforge-controller@demo-001.service -f
```

用户服务随用户管理器启动；无人登录时自动运行还取决于管理员配置。`Restart=on-failure` 用于异常退出恢复；正常达到预算后保持停止。模板按具体活动实例化。
服务模板默认假设项目位于用户主目录下的 `VibeForge/`；若克隆到其他位置，先相应修改 `WorkingDirectory` 和 `ExecStart`。

## 8. 测试入口

```bash
# 工程测试，不启动真实大模型
.venv/bin/python -m pytest -m 'not target' -q

# 可选：加入离线随机 GPT-2/Llama 的 FP32/BF16 CPU 生成与 profiler 测试
VIBEFORGE_TEST_INFERENCE_PYTHON=/path/to/transformers-env/bin/python \
  .venv/bin/python -m pytest -m 'not target' -q

# 真实目标验收：资源不满足时应报告失败，不按跳过或通过处理
VIBEFORGE_TARGET_TEST=1 .venv/bin/python -m pytest \
  tests/e2e/test_target_deployment.py -q
```

底层 `serve`、`agent-worker` 是受管理子进程入口；日常使用 `run` / `resume`，由控制程序设置缓存、写入限制、进程身份和生命周期。
