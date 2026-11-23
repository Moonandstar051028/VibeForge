# Qwen3.8-27B 初始引擎

独立于旧 Transformers 引擎。使用 `/home/yxy/.venvs/vibeforge-qwen38/bin/python`，Transformers 5.20.2；完整 BF16 checkpoint，SDPA，文本请求。

保留通用引擎的 load/generate/logits/chat/chat_completion 接口、微批实现和工具解析。新增原生模板参数 preserve_thinking、reasoning_effort 的透传。上下文设为 32768，每次响应最多 8192 输出 tokens（含 thinking），预留 2048 tokens 给工具模板等开销，Qwen-Agent 输入预算为 22528 tokens。完整窗口的显存峰值须另行实测。

配置：configs/qwen3.8-27b-a800.json。只在 CUDA_VISIBLE_DEVICES=1、CUDA_DEVICE_ORDER=PCI_BUS_ID 下使用实验 A800。

本机 Python 环境：`/home/yxy/.venvs/vibeforge-qwen38/bin/python`。新运行使用空的输出目录，配置中的 `agent.related_history=false` 禁止自动加载其它运行的历史。本次运行自身的各轮反馈正常积累。

要另行启动以 Qwen3.8 为目标的 Pipeline，可从项目根目录运行（使用新的输出目录）：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  /home/yxy/.venvs/vibeforge-qwen38/bin/python -m vibeforge_v2 run \
  --config configs/qwen3.8-27b-a800.json --output runs/qwen38-new-run --rounds 1
```
