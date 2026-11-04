# 开发辅助脚本

- 放置环境检查、格式检查、开发启动等辅助脚本。
- `bootstrap.sh` 安装控制程序；`check.sh` 执行工程测试；`check_environment.sh` 运行目标画像。
- `prepare_inference.sh` 将 Accelerate 安装到项目缓存；`probe_inference.py` 检查 CUDA、聊天模板与 meta 结构，不加载完整权重。
- 核心构建与评测逻辑不在脚本中重复实现。
