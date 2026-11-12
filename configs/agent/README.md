# Agent 配置

系统优化调试优先使用 `codex.json`；从已验证优化引擎继续迭代使用 `configs/campaigns/qwen35-27b-optimized-codex.json`；未优化起点的真实 Qwen3.5/A100 有界迭代入口为 `configs/campaigns/qwen35-27b-iterative-codex.json`。`local.json` 保留为 Qwen-Agent 对照配置，不修改已经冻结的历史活动。

- `local.json` 使用 Qwen-Agent 和 `templates/prompts/implementation.md`。
- `codex.json` 使用 Codex CLI 和独立的 `templates/prompts/codex-implementation.md`。Codex 通过宿主 stdio MCP 读取活动证据、检索历史、请求受预算约束的稳定引擎画像，并用 `record_result` 提交 schema 校验后的结果；文件编辑可由 Codex workspace 工具或候选目录内的 MCP 精确替换与语法检查完成。

- 放置稳定引擎连接方式、工具集合、生成长度、超时和提示词引用。
- `local.json` 使用 Qwen-Agent `Assistant`，通过当前稳定引擎的 Chat Completions 接口调用本地模型。
- `codex.json` 使用本机 Codex CLI，通过当前稳定引擎的 Responses 接口调用同一个本地模型；Codex 在仓库外的临时候选副本中工作，宿主校验后才同步文本修改。
- 两条 Agent 路线共享 schema v4 的模型特征、硬件、环境、workload、历史、knowledge 和 profiler 摘要首屏。完整包清单和大体积证据保留在宿主 context/evidence 中按需读取。
- 活动配置的 `agent` 可引用上述文件，也可内联配置；`agent.framework` 可选 `qwen-agent` 或 `codex`。
- `scripted.json` 仅用于可复现的接受/拒绝测试，不代表 Agent 推理能力。
- 提示词正文位于 `templates/prompts/`，会话与工具输出位于活动运行目录。
