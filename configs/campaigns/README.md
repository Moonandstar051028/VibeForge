# 优化活动配置

- `toy.json` 运行 CPU 测试闭环；`qwen36-35b.json` 配置 Thor 上的本地 Qwen BF16 活动。
- 引用模型、硬件、负载、评测和 Agent 配置，并定义预算、数据根目录与停止条件。
- 单轮执行状态和结果保存于 `var/campaigns/<campaign_id>/`。
