# 单元测试

- 放置独立模块的测试，例如状态迁移、准入规则、哈希、事件校验和统计计算。
- `test_storage.py`、`test_config_and_state.py`、`test_admission.py`、`test_profile.py`、`test_tools_and_knowledge.py` 按职责分组。
- 避免依赖真实大模型、GPU 或长期运行的服务。
