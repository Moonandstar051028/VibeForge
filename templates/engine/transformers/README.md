# Transformers 自举引擎

加载本地权重并提供文本生成、聊天、OpenAI 兼容 Chat Completions 和采样 logits。Chat Completions 使用模型自带 Qwen 工具模板适配 Qwen-Agent；视觉和 MTP 尚不在此实现范围。

通过 engine.json 固定模型路径、精度、设备和上下文。BF16 与可选 NF4 配置分属独立活动，不能混用评测结论。

候选可修改该模板的副本。依赖版本和源文件哈希记录在画像与发布清单中。
