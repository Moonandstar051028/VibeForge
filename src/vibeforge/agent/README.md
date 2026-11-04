# Agent 运行与上下文

- 放置 Qwen-Agent 会话启动、受限工具适配、恢复上下文组装和分析任务。
- `runner.py` 使用 Qwen-Agent `Assistant` 管理模型交互和工具调用；VibeForge 不再实现 Agent 推理循环。
- `tools.py` 继续约束候选文件读写并记录审计日志；`context.py` 重建恢复上下文；`analysis.py` 生成实验分析报告。
- 提示词正文放在 `templates/prompts/`；会话轨迹写入对应活动的运行目录。
- 日常修改目标是候选工作区，不直接更新控制程序或固定评测材料。
