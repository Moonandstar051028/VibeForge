# 模块协作测试

- 放置状态存储、子进程、构建、评测、版本切换和日志之间的协作测试。
- `test_promotion_and_events.py` 验证事务、恢复和事件；`test_engine_and_agent.py` 验证 HTTP 后端与模型动作执行。
- `test_backend_confinement.py` 验证写入限制；`test_process_timeouts.py` 验证超时清理。
- 需要硬件或环境能力的测试应明确标记前置条件。
