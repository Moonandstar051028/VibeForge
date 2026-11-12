# 优化授权策略

- `fixed.json`：固定精度，只优化实现。
- `precision-search.json`：示例精度白名单。必须同时配置 `correctness_mode=approximate`、独立质量命令、`min_scores` 和 `max_regressions`。

白名单限制可尝试的完整精度配置；它不代表设备可运行所有配置。控制器仍逐候选预检。

`allowed_surfaces` 冻结候选可触及的层级：运行配置、执行 adapter、cache policy、请求 scheduler 与 native kernel。`native_kernel` 还要求 engine 配置提供冻结的 argv build recipe；源文件、构建产物和加载代码全部进入候选 release 哈希并走完整门禁。
