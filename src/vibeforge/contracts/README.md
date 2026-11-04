# 跨模块数据契约

- 放置画像、活动、候选、事件、评测结果和决策等类型及序列化约定。
- 实现文件：`profile.py`、`campaign.py`、`candidate.py`、`evaluation.py`、`event.py`；`precision.py` 定义精度及候选授权，`requirements.py` 定义主目标与资源限制。
- 仅保存数据结构、协议和校验规则，保持轻量，避免导入控制程序或推理依赖。
