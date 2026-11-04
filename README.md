# VibeForge

面向模型、硬件和工作负载的轻量推理引擎生成与自优化研究项目。

## 项目入口

- [项目目录说明](docs/05-project-layout.md)：目录树、每个文件夹的职责、文件放置和命名规则。
- [系统设计文档](docs/README.md)：总体架构、迭代恢复、评测日志和知识库设计。
- [本地运行指南](docs/06-local-operations.md)：环境、命令、配置、日志与恢复。
- [实现与验收报告](docs/07-implementation-validation.md)：逐模块实现、实测结果及当前限制。
- [Qwen-Agent 接入](docs/08-qwen-agent-integration.md)：组件边界、本地服务协议、工具与验证范围。
- [多样化引擎系统](docs/09-diverse-engine-system.md)：模型、设备、精度、构建、个性化目标、质量门禁与配置示例。

## 常用位置

| 要找的内容 | 位置 |
| --- | --- |
| 控制与恢复流程 | `src/vibeforge/controller/` |
| Agent 会话和上下文 | `src/vibeforge/agent/` |
| 引擎构建与运行适配 | `src/vibeforge/engine/` |
| 正确性和性能评测代码 | `src/vibeforge/evaluation/` |
| 配置与固定评测材料 | `configs/`、`benchmarks/` |
| 优化知识内容 | `knowledge/` |
| 候选源码、发布和实验日志 | `var/workspaces/`、`var/releases/`、`var/campaigns/` |

## 当前阶段

已实现人工初始引擎上的自优化闭环，并扩展为模型、设备、运行精度和需求可组合的活动。支持显式精度白名单、独立质量门禁、并发客户端评测、自定义限时构建、多个优化目标、事务提升与回退，以及持久化上下文。Qwen-Agent `Assistant` 通过当前稳定引擎调用所配置的本地模型。

本机已验证工程闭环，以及随机小型 GPT-2/Llama 的 FP32、BF16 CPU 推理接口。CPU 夹具、随机模型和模拟服务不代表真实模型优化能力；真实 Qwen35B、量化、CUDA/ROCm 完整验收范围见验收报告。

## 快速验证

```bash
cd /path/to/VibeForge
bash scripts/dev/bootstrap.sh
.venv/bin/python -m pytest -m 'not target' -q
.venv/bin/vibeforge capabilities
.venv/bin/vibeforge run my-toy-run --config configs/campaigns/toy.json
.venv/bin/vibeforge status my-toy-run
```

控制程序使用 Python 3.10 及以上版本，进程管理和文件写入限制要求 Linux Landlock。初始后端提供纯 Python 测试引擎及 Transformers 文本推理适配；其他运行库通过人工 `engine.py` 桥接。并发 HTTP 请求目前串行进入后端，没有动态批处理；多模态尚未验收。
