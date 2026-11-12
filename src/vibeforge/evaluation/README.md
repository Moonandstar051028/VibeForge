# 固定评测器

- 放置正确性评测、性能测量、统计汇总、资源检查和参考实现适配。
- 实现文件：`correctness.py`（严格/近似差分、固定长度与并发路径）、`quality.py`（独立质量命令与完整性门禁）、`performance.py`（客户端时延、吞吐和资源测量）、`statistics.py`（双向收益区间）、`reference.py`（冻结参考）。
- 协议来自 `configs/evaluation/`，用例和参考材料来自 `benchmarks/`。
- 输出原始样本和结果报告；是否提升版本由 `controller/` 的准入规则决定。
- v3 自生成参考按多次输出建立变体集合及有界 logit 包络，只作为回归基线；非测试模型还必须通过独立任务质量及相对初始质量退化门禁。质量重复执行取最差分数，原始证据保存在 `quality/runs/`。详见 [评测文档](../../../docs/03-evaluation-and-observability.md#当前实现v3-组合门禁)。
