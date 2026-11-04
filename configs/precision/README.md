# 精度配置

保存运行时计算、权重、KV cache 和累加精度的 JSON 契约，与模型权重的存储格式分开。

- `fp32.json`、`fp16.json`、`bf16.json`：非量化计算配置。
- `bnb-nf4.json`、`bnb-fp4.json`、`bnb-int8.json`：内置 Transformers 的 CUDA 在线量化配置。
- KV/累加精度默认 `auto`；显式覆盖需人工提供支持它的 Python 适配器。
- 配置存在不代表当前 GPU/依赖已经验证兼容。用 `profile` 查看实际预检。
