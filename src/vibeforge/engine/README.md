# 推理引擎适配与构建

- 放置引擎后端适配、候选构建、进程启动、健康检查和推理客户端。
- 实现文件：`backend.py`（后端协议）、`builder.py`（Python 构建检查）、`process.py`（进程）、`client.py`（调用）、`server.py`（HTTP 服务）、`sandbox.py`（写入限制）。
- 生成的模型执行代码和原生内核放在 `var/workspaces/`；可复用的初始源码放在 `templates/engine/`。
- 本目录保存受维护的适配代码，已验证发布制品保存于 `var/releases/`。
