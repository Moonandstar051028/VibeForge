# Transformers 自举引擎

加载本地自回归模型并提供文本生成、聊天、OpenAI 兼容 Chat Completions 和采样 logits。支持 causal LM 或 image-text 模型类的文本路径；视觉和 MTP 尚不在此实现范围。

通过 engine.json 声明模型路径、精度、设备和上下文。支持 FP32/FP16/BF16 与 CUDA bnb NF4/FP4/INT8；实际支持取决于依赖及硬件。精度默认固定，允许白名单切换时必须经过独立质量门禁。预量化权重和显式 KV/累加精度需提供专用适配器。

聊天使用模型自带模板；提供 Qwen 工具标签和 JSON 工具调用转换。其他原生工具格式应在人工初始引擎中适配，并先通过 Agent 冒烟。

候选可修改该模板的副本。依赖版本和源文件哈希记录在画像与发布清单中。
