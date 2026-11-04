# 推理引擎适配与构建

- 放置引擎后端适配、候选构建、进程启动、健康检查和推理客户端。
- 引擎后端提供 `generate`、`chat`、`chat_completion`、`logits`；`chat_completion` 实现 Qwen-Agent 所需的 OpenAI Chat Completions 流式接口。
- Transformers 模板将 OpenAI 工具定义传给 Qwen 原生聊天模板，并将原生工具调用映射回 OpenAI 响应。
- 实现文件：`adapters.py`（适配器声明与能力约束）、`backend.py`（后端协议）、`builder.py`（语法校验与受限构建）、`process.py`（进程）、`client.py`（评测调用）、`server.py`（HTTP 服务与 OpenAI 兼容路由）、`sandbox.py`（写入限制）。
- 生成的模型执行代码和原生内核放在 `var/workspaces/`；可复用的初始源码放在 `templates/engine/`。
- 本目录保存受维护的适配代码，已验证发布制品保存于 `var/releases/`。
