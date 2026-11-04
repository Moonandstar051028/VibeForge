# Agent 配置

- 放置稳定引擎连接方式、工具集合、生成长度、超时和提示词引用。
- `local.json` 使用 Qwen-Agent `Assistant`，通过本地推理引擎的 OpenAI 兼容接口调用 Qwen 模型。
- `scripted.json` 仅用于可复现的接受/拒绝测试，不代表 Agent 推理能力。
- 提示词正文位于 `templates/prompts/`，会话与工具输出位于活动运行目录。
