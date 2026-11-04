# 独立任务质量评测

`quality_cases.py` 使用标准库请求候选引擎，按人工期望答案计算加权准确率。

- 输入：控制器生成的 `request.json`，以及通过 `--cases` 指定的固定用例。
- 输出：`passed`、`metrics.accuracy`、逐用例生成结果。
- 用例支持 `mode=chat|generate`、`match=exact|contains|regex`、`weight` 和输出长度。
- 活动中必须将脚本、用例及其依赖文件列入 `evaluation.quality.files`。
- 示例用例仅演示接线；研究结论需要有代表性的领域数据集。可通过同一命令接口接入第三方评测程序。
