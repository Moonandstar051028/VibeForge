# 08：Agent 接入

## 1. 组件职责

| 组件 | 职责 | 入口 |
| --- | --- | --- |
| Qwen-Agent `Assistant` | 模型交互、工具调用识别、工具结果反馈、多轮执行 | `src/vibeforge/agent/runner.py` |
| Codex CLI | 可选的上下文、压缩和工具执行框架 | `src/vibeforge/agent/codex.py` |
| 本地 Qwen 模型 | 基于上下文思考并选择优化动作 | 当前稳定引擎中的模型权重 |
| VibeForge 工具适配 | 为 Assistant 提供候选文件与实验记录的受限访问 | `src/vibeforge/agent/tools.py` |
| 引擎 HTTP 服务 | 提供 `/v1/chat/completions`、`/v1/responses` 和 `/tokenize` | `src/vibeforge/engine/server.py` |
| Transformers 模板 | 渲染 Qwen 原生工具提示、执行推理、转换工具输出 | `templates/engine/transformers/engine.py` |
| 独立控制器 | 启停引擎与 Agent、检查预算、评测、提升和回退 | `src/vibeforge/controller/supervisor.py` |

模型请求始终发到控制器为本轮启动的本地引擎。没有配置云端 Qwen 或 OpenAI 模型服务。

工具来自 VibeForge 的显式注册，而非 Qwen-Agent 自动扫描目录：`runner.py` 的 `TOOL_SCHEMAS` 定义名称、参数和说明，`_make_tool()` 将回调包装成 `BaseTool`，`_assistant_tools()` 按会话权限组装工具，再通过 `Assistant(function_list=...)` 注入。模型看到这些工具定义后选择调用，Qwen-Agent 执行回调并将结果反馈给模型。`agent/` 同时包含上下文、提交校验、实验记忆等控制代码，并非其中每个函数都是模型可调用工具。

本地实现与 Agent 框架的当前分工：

| 能力 | VibeForge 本地实现 | Qwen-Agent | Codex |
| --- | --- | --- | --- |
| 模型/硬件/精度/负载、观测与历史上下文 | 构建、裁剪重复原始数据、保存完整证据 | 作为首轮消息使用 | 作为首轮提示使用 |
| 会话循环、模型调用、工具调用识别 | 设置预算、超时和审计 | `Assistant` 执行 | Codex CLI 执行 |
| 候选文件工具 | 提供受限读写回调 | 使用 VibeForge `read_file/replace_text/write_file` | 使用 Codex 自带文件和命令工具操作临时候选副本 |
| 证据、历史、会话自测和受控 profiler | 提供 `read_evidence/list_evidence/search_history/read_telemetry/profile_inference` | 可在会话中调用 | 通过宿主 stdio MCP 调用同一组有界能力 |
| 上下文控制 | 精确计数并对 Qwen 首轮上下文做可检索裁剪 | 另有上游粗略消息截断，不做等价语义压缩 | 使用 Codex 自身多轮上下文和压缩机制；VibeForge 只做首轮去重裁剪 |
| 候选准入、构建、正确性、性能、晋升、回退 | 独立强制执行 | 无权绕过 | 无权绕过 |

因此，两种 Agent 共用实验控制平面，并获得同源的模型特征、硬件、软件环境、负载、稳定版 profile、会话 telemetry、历史和经验摘要。Qwen-Agent 通过 `BaseTool`，Codex 通过宿主管理的 stdio MCP 使用相同的证据、历史、自测和画像能力；文件编辑仍分别使用 VibeForge 受限工具与 Codex 自带沙箱工具。两条路线的序列化格式和上下文压缩机制不同，但候选契约与宿主准入完全相同。

Qwen 模型负责推理和判断，Qwen-Agent 负责会话与工具编排；两者都不能凭空获得 CPU/GPU 时间分布。仅有总 token/s、源码和错误日志时，关于“时间主要花在哪里”的回答只能是假设。现在由宿主提供独立计时和受控 profiler，再要求模型把观测、原因假设、修改、数值预测和验证结果关联起来；性能热点仍需实验才能建立因果解释。

## 2. 替换范围

- 删除原有逐轮 JSON 动作解析、手工反馈和自写工具循环。
- 使用 Qwen-Agent `Assistant` 和 `BaseTool`；固定依赖版本 `qwen-agent==0.0.34`。
- 保留候选目录约束、工具审计、持久化恢复上下文和提交校验。
- `scripted` 模式作为确定性控制器测试夹具保留；生产 `local` 模式可选 `qwen-agent` 或 `codex`。
- 优化会话以有效 `submit_candidate` 或明确的 `report_blocked` 结束；后者记录无可验证候选的原因，不创建有效实验。协议或上下文错误产生的控制器恢复记录不能冒充候选或上一轮分析。

## 3. 本地服务协议

1. Assistant 通过 `model_type=oai`、`use_raw_api=true` 向当前引擎发送工具定义和历史。
2. 引擎把历史中的工具参数由 JSON 字符串转成对象，交给权重自带的聊天模板。
3. 模型输出的 `<tool_call><function=...><parameter=...>` 转成 OpenAI `tool_calls`。
4. Qwen-Agent 执行受限工具并把结果放回下一次请求。
5. 提交工具检查实际修改、观测证据、结构化实验计划和待分析实验的复盘；worker 写盘后控制器独立复查，再决定是否构建和评测。

SSE 接口当前在一轮生成完成后发送完整内容与工具调用，不提供逐 token 实时流。性能评测继续使用独立生成接口及固定评测协议。

Codex 使用自定义 provider，把推理请求发送到当前稳定引擎的 `/v1/responses`。它不会通过 Ollama、LM Studio 或云端重新加载模型。Codex 的可写根目录是仓库外的临时候选副本；完成后宿主拒绝删除、符号链接、非 UTF-8 和超过 512 KiB 的文件，再把允许的文本修改原子同步到正式候选目录。结构化结果仍由控制器执行候选准入、构建、正确性、质量、性能、晋升和回退，Codex 无权修改稳定指针或活动证据。

## 4. 工具和约束

| 工具 | 作用 |
| --- | --- |
| `read_file` / `list_files` | 读取候选源码；读文件支持字符 `offset` / `limit` |
| `read_evidence` / `list_evidence` | 只读活动原始证据；文件默认 6,000、最多 12,000 字符；列表每页最多 100 项 |
| `search_history` | 检索历史假设、决策、分析和反思；按关键词相关性及时间返回，默认 5、最多 10 项 |
| `profile_inference` | 以 `profile_id`、`prompt_index` 选择冻结场景请求宿主剖析；每会话最多 2 次，输出最多 128 token，不允许任意脚本或路径 |
| `read_telemetry` | 读取本次 Agent 会话在当前稳定引擎上产生的普通推理摘要；区分排队、prefill、decode、分词和宿主开销，不作为准入样本 |
| `replace_text` / `write_file` | 修改候选目录内文件 |
| `check_python` | 检查修改后的 Python 语法 |
| `submit_candidate` | 提交 `hypothesis`、`summary`、`experiment_plan`；复盘时还需 `previous_analysis` 和 `reflection` |
| `report_blocked` | 说明证据或实现条件不足，结束提案尝试，不计作有效实验 |

- 分析会话只提供读取、检索和提交工具；冒烟会话只提供提交工具。
- `submit_candidate` 按会话生成参数声明：正式规划将 `experiment_plan` 列为必填嵌套对象，v2 进一步声明数值证据、目标画像、运行前提及瓶颈预测必填；待分析时显式要求分析与复盘字段。复盘/冒烟不会继承规划必填项。结构或类型错误返回带字段路径的简短反馈，不替模型搬移或填充字段；宿主候选准入继续独立校验实际修改和证据。
- 规划会话的普通文字回复结束不代表任务完成。若尚未提交候选或报告阻塞，且原有时间和工具预算仍有余量，宿主只给一次继续机会，保留原始用户上下文及完整工具轨迹，要求模型执行修改并提交，或调用 `report_blocked` 说明具体阻塞。两次调用共享原会话的时间、工具次数和每次输出上限；继续后仍无有效提交则按协议失败结束，不由宿主生成候选，也不跳过准入。
- `experiment_plan` 必须包含 `measured_bottleneck`、`evidence_paths`、`cause`、`change`、`metric`、`direction`、正数 `expected_change_pct`、`validation` 和 `history_refs`。存在历史时必须引用已记录的尝试；至少一条瓶颈证据必须绑定当前稳定版本源码、画像和活动观测。
- 新本地会话使用 v2 契约：再要求数值 `bottleneck_reference={evidence_path,metric,value,unit}`、`target_profile_ids`、真实配置的 `runtime_preconditions=[{field,equals}]`、`bottleneck_direction/expected_bottleneck_change_pct/expected_change_basis`。数值来自宿主 `measurements`，不是任意 prose；控制器独立核对。`scripted` 测试夹具兼容 v1。
- `runtime` 暴露已加载模型后端、包/模块、调度、精度和 cache 默认值；`normal_request` 提供实际生成参数与 cache 类。默认值显式化不算优化；paged attention 缺少执行路径时拒绝，不能仅检查名称白名单。kernel callable 也可能只是 fallback 包装器。
- `reflection` 包含目标 `iteration`、`hypothesis_status`（`supported/refuted/inconclusive`）、`explanation`、`next_action` 和该轮 `evidence_paths`。基础设施失败必须标记 `inconclusive`；候选被接受也不等于预测幅度成立，宣称 `supported` 还需宿主的预测验证支持。
- 进入新一轮规划前，控制器先完成全部 `analysis_pending`；无法完成则停止推进并保留证据。最近 6 条和最多 4 条相关记录保留详细计划、差异、结论和证据路径；所有更早轮次聚合为策略台账、结果计数和代表性反例，因而不会随轮数增长而完全消失。任意当前或归档记录仍可通过 `search_history` 与分页证据读取恢复。
- 新活动可以通过证据内的 `memory/archived-campaigns.json` 继续访问已结束活动的原始记忆与补丁。上下文提供有界归档索引，并把兼容活动中的已验证策略和反例合并到 `experience_knowledge`；`search_history` 同时搜索当前与归档活动。归档不是当前观测、当前待分析任务或本轮实验，不能替代当前瓶颈证据。配置、画像和稳定父版本严格匹配且已完成的候选进入宿主去重索引；完全相同的语义指纹返回 `duplicate_change`，不同实现或条件变化后的重试不因此被拦截。
- 历史详情挤占输入预算时，先把历史和上一轮详情压缩为带轮次、决策、假设及证据目录的索引；仍不足时由 `read_file` 读取当前源码。完整计划、解释、复盘和补丁继续保存在宿主证据，通过 `search_history/read_evidence` 分页获取。预算审核记录 `compaction_steps`，核心观测数值和工具历史预留不因压缩被放宽。
- 复盘使用独立系统提示和目标实验上下文，不复用“提出优化、修改源码”的规划指令，也不注入当前稳定版源码/诊断来替代历史目标。模型读取该轮证据并提交归属明确的分析；若只输出 prose 或出现可识别的工具格式/上下文错误，可在原会话时间预算内追加一次仅开放提交工具的模型总结。总结仍由模型撰写，不能绕过反思归属、证据路径和基础设施失败须为 `inconclusive` 的检查。复盘上下文显式提供目标轮次的证据目录及决策文件，避免猜测路径；`list_evidence(".")` 支持列出证据根目录，越界限制保持生效。占位分析检查针对完整占位回复，实质分析引用旧占位句不应被误拒。复盘另包含 `review_measurement_status`：本轮 `observation.json` 属于修改前的稳定父版本，候选诊断与正式 A/B 路径分开；正确性或基础设施失败时，不得将父版本数据写成候选数值提速/降速。提交工具和控制器均检查这类量化比较。`normal_request` 为父版本普通推理，`timing` 为同一父版本插桩推理，两者不能解释成版本对照；无 A/B 的候选也不能贴数值性能标签。已失效复盘从重新分析的模型输入中移除，原文仍保留在日志以供审计。该检查覆盖常见中英文比较表达，不等于对任意自然语言做完备事实核验。
- 长输入原文保留于冻结宿主上下文；初始提示和工具返回只提供场景、长度、prompt 索引及短预览，避免 8K 提示重复挤占 Agent 上下文。剖析仍使用完整冻结输入。
- profiler 注入内容保留完整数值索引、运行配置及模块分组，算子只展示前 5 行，超长 kernel 名保留 160 字符预览及原始行号；不重复注入插桩请求明细、负载原文和逐 token 数组。完整记录仍可按证据路径分页读取。
- 会话启动优先调用稳定引擎的 `/tokenize`，用宿主独立加载的目标模型 tokenizer 统计完整 chat template 和工具 schema，保存 `agent_context_budget` 审计事件，并为工具历史预留至多 4096 token。55% 仅作为提示词体量的审计参考，不是模型或 agent 框架的硬限制；硬门槛是精确初始请求加工具历史预留不得超过配置的输入预算。测试或旧第三方端点没有该能力时才回退 Qwen-Agent 估算。tokenizer 服务属于控制层，不来自可编辑候选 `engine.py`；同一模型的引擎版本变化不改变计数规则。
- 正确候选用同条件生成 `candidate-observation.json`，验证目标瓶颈是否移动；其单次诊断结果与正式配对统计分别记录并进入记忆。模块 inclusive 时间不可相加，内部 TTFT 不等于缓冲 SSE 的客户端首 token 时间。
- `max_turns` 传入 Qwen-Agent 官方调用预算环境变量，并限制未提交会话的工具次数。
- 控制器按活动与 Agent 超时中的较小值终止超时 worker。
- 未给 Assistant 注册 shell、联网搜索、代码解释器或任意执行工具。

## 5. 安装、运行和日志

```bash
bash scripts/dev/bootstrap.sh
.venv/bin/python -m pytest -m 'not target' -q

export VIBEFORGE_MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export VIBEFORGE_RUNTIME_PYTHON=/path/to/inference-env/bin/python
.venv/bin/vibeforge run qwen-agent-001 --config configs/campaigns/qwen36-35b.json
```

配置文件：`configs/agent/local.json` 或 `configs/agent/codex.json`。控制环境安装 Qwen-Agent；选择 Codex 时还要求 `agent.executable` 指向可用 Codex CLI。推理环境负责 Torch、Transformers 等模型依赖。

活动内也可以直接选择框架：

```json
{
  "agent": {
    "mode": "local",
    "framework": "codex",
    "executable": "codex",
    "model": "Qwen3.5-27B",
    "max_passes": 3,
    "max_new_tokens": 4096,
    "thinking": {
      "enabled": true,
      "temperature": 0.6,
      "top_p": 0.95,
      "top_k": 20,
      "min_p": 0.0
    },
    "timeout_seconds": 900,
    "prompt": "templates/prompts/codex-implementation.md"
  }
}
```

`model.max_context` 是活动选择的工作上限，可以省略。省略时配置加载会采用模型 `config.json` 的 `text_config.max_position_embeddings`；模型没有声明时回退 4096。显式值不能超过原生上限。增大工作上限还会增加 KV cache、预填充耗时和 Codex/Qwen-Agent 单轮输入成本，必须用目标硬件重新做部署与长上下文验证。

Codex 的压缩阈值由宿主根据模型上下文窗口，扣除 `agent.max_new_tokens`、工具结果预算和安全余量后自动计算，并通过 `model_auto_compact_token_limit` 传给 Codex CLI。活动配置不接受 `agent.compact_at_tokens`；配置该字段会被拒绝，避免人工阈值与实际输出及工具预算冲突。`agent.max_passes` 只控制一次候选会话最多启动多少个有界 Codex 工作阶段，不改变上下文窗口或压缩算法。

Codex 的普通本地 Responses 链路已经通过真实 CLI 冒烟；真实模型的多轮长上下文压缩尚未完成压力验收。本地兼容服务当前也没有实现 OpenAI `/responses/compact` 服务端压缩端点，因此这里的主动压缩指 Codex CLI 自身的上下文机制，不应与 OpenAI 托管 Responses 的服务端 compaction 混为一谈。

原有冻结引擎如果没有 `chat_completion`，不满足新后端契约。保留历史制品与日志，使用新活动从更新后的模板自举；不要直接修改历史发布目录。

| 日志文件 | 内容 |
| --- | --- |
| `iterations/<编号>/commands.jsonl` | Assistant 状态、工具参数、成功结果、错误与进程审计 |
| `iterations/<编号>/logs/agent.log` | Qwen-Agent 和 worker 标准输出 |
| `iterations/<编号>/context.json` | 重启后可重建的模型、硬件、历史和证据上下文 |
| `iterations/<编号>/agent-result.json` | 会话标识、框架、假设、分析与提交结果 |
| `iterations/<编号>/agent-telemetry.json` | Agent 在稳定引擎上推理时的有界请求计时和阶段汇总；诊断用途，不参与准入统计 |
| `iterations/<编号>/observation.json` / `candidate-observation.json` / `candidate-gate.json` / `verification.json` | 稳定版观测、候选同条件诊断、候选门槛、瓶颈变化与配对实测对照（按执行阶段生成） |
| `memory/index.json` / `iterations/<编号>/memory.json` | 从数据库重建的实验记忆索引与单轮条目 |
| `final-analysis/<目标编号>/` | 独立保存各目标的分析上下文、工具轨迹和结果 |
| `bootstrap/agent-smoke.jsonl` | 初始框架工具调用检查轨迹 |
| `activation-smoke.jsonl` / `restore-smoke.jsonl` | 候选激活或回退时的框架检查轨迹 |

## 6. 验收边界

- 工程测试覆盖真实 Qwen-Agent 框架对接本地模拟 SSE 服务、受限文件编辑、错误反馈、提交、控制器独立 worker 和最终分析。
- Toy 闭环覆盖评测、接受、正确性拒绝、恢复和退出清理。
- 2026-11-04 的历史 Qwen35B/BF16 验收：tokenizer 多轮工具模板通过；估算需求约 78.32 GiB，当时可用约 37.24 GiB，部署因 `insufficient_memory` 阻塞。这不是当前 27B/4-bit 活动的资源结论。
- 2026-11-05 三模块更新以受控工具、完整 CPU 闭环和离线随机小模型验证；27B 实验保持暂停，本次更新不能据此宣称完整 Qwen 推理或 CUDA 性能已通过。
- 同日后续真实配置/观测矩阵增量已在 A100 的 Qwen3.5-27B NF4 上验证五类完整长度生成、CUDA 模块/算子诊断、内部 token 计时、并发串行排队及重复 token 一致性；未运行五组正式候选 A/B，不能宣称优化提速或真实模型 Agent 自主决策已通过。详见验收报告当前增量。

具体结果见 [实现与验收报告](07-implementation-validation.md)。上游接口依据：[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)、[工具调用示例](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/function_calling.py)。
