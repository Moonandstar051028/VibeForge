# 优化授权策略

- `fixed.json`：固定精度，只优化实现。
- `precision-search.json`：示例精度白名单。必须同时配置 `correctness_mode=approximate`、独立质量命令、`min_scores` 和 `max_regressions`。

白名单限制可尝试的完整精度配置；它不代表设备可运行所有配置。控制器仍逐候选预检。
