# 完整闭环与故障恢复测试

- 放置完整活动、候选接受/拒绝、Agent 重启和控制程序恢复测试。
- `test_optimization_loop.py` 验证两轮接受/拒绝；`test_restart_recovery.py` 验证真实控制进程崩溃后的恢复。
- `test_rollback_paths.py` 验证退化、语法错误和 Agent 错误；`test_target_deployment.py` 是需显式开启的真实模型验收。
- 优先使用轻量假引擎和受控故障验证流程；真实模型场景单独标明运行要求。
