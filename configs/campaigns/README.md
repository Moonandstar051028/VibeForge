# 优化活动配置

无人辅助验收入口：`qwen35-27b-autonomous-codex.json`，从原始引擎启动，使用 Codex、本活动知识范围和独立 `var/autonomous` 数据目录。首次完整验收 4 轮、0 次晋升，尚未达到自主收益目标。过程与限制见 `docs/10-continuous-improvement.md`。

`qwen35-27b-optimized-codex.json` 从人工辅助验证的 eager CUDA Graph 引擎继续运行，不能作为无提示自主发现的证据。`qwen35-27b-iterative-codex.json` 允许项目知识迁移；`qwen35-27b-graph-validation.json` 用于限定机制验证。

- `toy.json` 运行 CPU 测试闭环；`qwen36-35b.json` 配置 Thor 上的本地 Qwen BF16 活动。
- 引用模型、硬件、负载、评测和 Agent 配置，并定义预算、数据根目录与停止条件。
- `agent.framework` 可设为 `qwen-agent` 或 `codex`；两者都使用本轮当前稳定引擎，不会另行加载模型服务。两种框架使用不同 Prompt 和提交协议，不能把 Qwen-Agent 的 `implementation.md` 配给 Codex。
- 引用式切换分别写成 `"agent": "configs/agent/local.json"` 或 `"agent": "configs/agent/codex.json"`；也可以在活动配置中内联同样的字段。
- 单轮执行状态和结果保存于 `var/campaigns/<campaign_id>/`。
- `qwen35-27b-codex-context-study.json` 使用本地 Qwen3.5-27B 引擎和 Codex CLI，最多执行三次提案并保留 Codex 自己的 token/压缩事件；不配置 VibeForge 上下文裁剪阈值。
- `toy-qwen-agent.json` 与 `toy-codex.json` 分别用于单轮 Agent 编排验收。它们使用确定性的 CPU fixture，只验证协议、工具和恢复流程，性能数据不能代表真实模型优化收益。
- `toy-refinement.json` 验证 quick probe 与一次修订：初稿故意破坏正确性，修订恢复语义并应用 fixture 优化，再进入正式门禁。该脚本只验证控制流。
- `toy-scheduler.json` 验证受控调度器和执行适配层修改会经过 quick probe、正式门禁与稳定版本晋升；fixture 收益不代表真实模型收益。
