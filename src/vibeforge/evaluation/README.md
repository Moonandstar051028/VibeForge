# 固定评测器

- 放置正确性评测、性能测量、统计汇总、资源检查和参考实现适配。
- 实现文件：`correctness.py`（差分验证）、`performance.py`（测量）、`statistics.py`（区间）、`reference.py`（冻结参考）。
- 协议来自 `configs/evaluation/`，用例和参考材料来自 `benchmarks/`。
- 输出原始样本和结果报告；是否提升版本由 `controller/` 的准入规则决定。
