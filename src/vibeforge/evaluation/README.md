# 固定评测器

- 放置正确性评测、性能测量、统计汇总、资源检查和参考实现适配。
- 实现文件：`correctness.py`（严格/近似差分、固定长度与并发路径）、`quality.py`（独立质量命令与完整性门禁）、`performance.py`（客户端时延、吞吐和资源测量）、`statistics.py`（双向收益区间）、`reference.py`（冻结参考）。
- 协议来自 `configs/evaluation/`，用例和参考材料来自 `benchmarks/`。
- 输出原始样本和结果报告；是否提升版本由 `controller/` 的准入规则决定。
