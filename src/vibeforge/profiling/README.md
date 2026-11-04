# 模型与运行环境画像

- 放置模型结构解析、硬件能力探测、软件环境采集、设备状态采样和画像哈希。
- 实现文件：`model.py`（权重与结构）、`hardware.py`（设备）、`environment.py`（依赖）、`sampler.py`（资源与预检）、`collector.py`（组合画像）。
- 声明式目标配置来自 `configs/`；实际采集的画像写入 `var/profiles/`。
