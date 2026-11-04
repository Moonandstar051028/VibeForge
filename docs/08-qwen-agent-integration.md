# 08：Qwen-Agent 接入

## 1. 组件职责

| 组件 | 职责 | 入口 |
| --- | --- | --- |
| Qwen-Agent `Assistant` | 模型交互、工具调用识别、工具结果反馈、多轮执行 | `src/vibeforge/agent/runner.py` |
| 本地 Qwen 模型 | 基于上下文思考并选择优化动作 | 当前稳定引擎中的模型权重 |
| VibeForge 工具适配 | 为 Assistant 提供候选文件与实验记录的受限访问 | `src/vibeforge/agent/tools.py` |
| 引擎 HTTP 服务 | 提供 `/v1/chat/completions` JSON / SSE 兼容接口 | `src/vibeforge/engine/server.py` |
| Transformers 模板 | 渲染 Qwen 原生工具提示、执行推理、转换工具输出 | `templates/engine/transformers/engine.py` |
| 独立控制器 | 启停引擎与 Agent、检查预算、评测、提升和回退 | `src/vibeforge/controller/supervisor.py` |

模型请求始终发到控制器为本轮启动的本地引擎。没有配置云端 Qwen 或 OpenAI 模型服务。

## 2. 替换范围

- 删除原有逐轮 JSON 动作解析、手工反馈和自写工具循环。
- 使用 Qwen-Agent `Assistant` 和 `BaseTool`；固定依赖版本 `qwen-agent==0.0.34`。
- 保留候选目录约束、工具审计、持久化恢复上下文和提交校验。
- `scripted` 模式作为确定性控制器测试夹具保留；生产 `local` 模式只允许 `qwen-agent` 框架。
- 完成条件是成功调用 `submit_candidate`；没有有效提交会话就失败。

## 3. 本地服务协议

1. Assistant 通过 `model_type=oai`、`use_raw_api=true` 向当前引擎发送工具定义和历史。
2. 引擎把历史中的工具参数由 JSON 字符串转成对象，交给权重自带的聊天模板。
3. 模型输出的 `<tool_call><function=...><parameter=...>` 转成 OpenAI `tool_calls`。
4. Qwen-Agent 执行受限工具并把结果放回下一次请求。
5. `submit_candidate` 校验假设和上一轮分析后，worker 将结构化结果写盘退出。

SSE 接口当前在一轮生成完成后发送完整内容与工具调用，不提供逐 token 实时流。性能评测继续使用独立生成接口及固定评测协议。

## 4. 工具和约束

| 工具 | 作用 |
| --- | --- |
| `read_file` / `list_files` | 读取候选源码 |
| `read_evidence` | 读取活动目录内的原始证据 |
| `replace_text` / `write_file` | 修改候选目录内文件 |
| `check_python` | 检查修改后的 Python 语法 |
| `submit_candidate` | 提交 `hypothesis`、`previous_analysis`、`summary` |

- 分析会话只提供读取和提交工具。
- 上一轮标记为 `analysis_pending` 时，提交必须包含分析。
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

配置文件：`configs/agent/local.json`。控制环境安装 Qwen-Agent；推理环境负责 Torch、Transformers 等模型依赖。

原有冻结引擎如果没有 `chat_completion`，不满足新后端契约。保留历史制品与日志，使用新活动从更新后的模板自举；不要直接修改历史发布目录。

| 日志文件 | 内容 |
| --- | --- |
| `iterations/<编号>/commands.jsonl` | Assistant 状态、工具参数、成功结果、错误与进程审计 |
| `iterations/<编号>/logs/agent.log` | Qwen-Agent 和 worker 标准输出 |
| `iterations/<编号>/context.json` | 重启后可重建的模型、硬件、历史和证据上下文 |
| `iterations/<编号>/agent-result.json` | 会话标识、框架、假设、分析与提交结果 |
| `bootstrap/agent-smoke.jsonl` | 初始框架工具调用检查轨迹 |
| `activation-smoke.jsonl` / `restore-smoke.jsonl` | 候选激活或回退时的框架检查轨迹 |

## 6. 验收边界

- 工程测试覆盖真实 Qwen-Agent 框架对接本地模拟 SSE 服务、受限文件编辑、错误反馈、提交、控制器独立 worker 和最终分析。
- Toy 闭环覆盖评测、接受、正确性拒绝、恢复和退出清理。
- 本地真实 Qwen tokenizer 的多轮工具模板已验证；未加载完整模型权重。
- 当前 BF16 需求约 78.32 GiB，本次可用约 37.24 GiB，真实部署验收因 `insufficient_memory` 未通过。

具体结果见 [实现与验收报告](07-implementation-validation.md)。上游接口依据：[Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)、[工具调用示例](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/function_calling.py)。
