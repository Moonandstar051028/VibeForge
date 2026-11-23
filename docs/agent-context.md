# Agent、工具与上下文

## 后端与身份

默认使用 Qwen-Agent；`agent.use_raw_api=true` 通过模型原生聊天模板和工具解析接口调用。`agent.backend=codex` 使用 Codex CLI；默认连接本地稳定引擎，只有显式 `agent.codex.inference=user-config` 才使用用户配置服务。项目不附带 CLI 安装包。

标准 CLI 将 Agent 连接到稳定目标引擎。Agent thinking 与目标评测 thinking 分别由 `agent.thinking`、`engine.thinking` 控制。不能把后端名称当作优化模型身份。

每轮提供唯一 `CURRENT_STABLE_ENGINE` 标记、实际运行状态、候选路径、接口契约、瓶颈摘要和历史反馈。`current/manifest.json` 是当前稳定身份依据；历史候选的 runtime 和说明不能代替当前状态。

## 工具与检查

Shell 用于读取源码、编辑候选、安装候选私有依赖和执行检查；写入范围限制在候选及私有路径。Shell 不用于加载完整 GPU 模型，GPU 构建与评测由宿主执行。单条命令默认 60 秒，可指定 `timeout_seconds`，上限默认 900 秒（`agent.shell_timeout_seconds`）。超时清理自有进程组并保存输出。

`edit_file` 用于精确文本替换；可选配置工具负责单字段 JSON 序列化、固定参数检查与回读。工具反馈包含实际差异、语法与运行路径检查。静态通过只允许继续评测，不证明优化有效。

宿主 CPU harness 用真实模型配置构造缩小随机模型，检查候选 load/generate/logits/chat 路径及近似数值等价性。GPU 专用实现可能跳过 CPU 路径，需明确记录并由 GPU 评测验证。

## 预算与工作记忆

原生 API 开启 `agent.exact_context=true` 时，以服务模型 tokenizer 和聊天模板计数，包括工具定义及推理内容；输入、输出预留和安全余量必须满足固定上下文窗口。

接近上限时整理旧的完整工具步骤，保留原任务、系统规则及最近内容。`extractive_memory=true` 使用抽取记忆；其它模式可用同一模型、关闭 thinking 的独立摘要请求，`structured_memory` 控制结构化字段。不同配置的记忆策略应分别核对，不假设所有后端行为相同。

`history_reasoning=last` 限制进入后续请求的历史思考，全量证据仍保存。上下文整理不是无损压缩；摘要失败、最近内容无法容纳或服务返回超限时保存错误并停止，不把不断削减预算当作恢复。

`agent.shell_output_chars` 限制单次工具返回文本，保留长输出头尾和完整日志路径；这是另一层限制。输出触及 `max_new_tokens` 可能导致思考或工具调用不完整，需要分析预算与原始生成，不能当作已完成候选。

## 证据

每轮 `agent/` 保存对话、实际工具调用、原始请求与响应（开启 `trace_requests` 时）、预算和工作记忆材料。`analysis.md` 为模型陈述，宿主 `decision.json`、真实 diff、加载状态与独立评测才是结论依据。

新实验关闭 `agent.related_history`，仅使用当次运行证据。日志可能包含实际命令和思考，审计工具只读分析，不生成或晋升候选。
