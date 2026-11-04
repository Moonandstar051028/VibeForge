# 人工初始引擎与适配器

保存 `initial_source`、模型格式、支持的设备/量化、依赖、加载器及构建命令。

- `transformers.json`：仓库提供的人工文本推理初始实现，可选择 causal LM 或 image-text 模型类的文本路径。
- `python` 适配器：由用户提供 `engine.py`、`engine.json`，封装其他运行库；详细契约见 `docs/09-diverse-engine-system.md`。
- 构建命令使用 argv 数组和超时，产物放 `runtime/`；`build/` 是临时目录，不会进入发布。
