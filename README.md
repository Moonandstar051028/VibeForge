# VibeForge

面向模型、硬件和工作负载的轻量推理引擎生成与自优化研究项目。

## 项目入口

- [项目目录说明](docs/05-project-layout.md)：目录树、每个文件夹的职责、文件放置和命名规则。
- [系统设计文档](docs/README.md)：总体架构、迭代恢复、评测日志和知识库设计。
- [本地运行指南](docs/06-local-operations.md)：环境、命令、配置、日志与恢复。
- [实现与验收报告](docs/07-implementation-validation.md)：逐模块实现、实测结果及当前限制。
- [Qwen-Agent 接入](docs/08-qwen-agent-integration.md)：组件边界、本地服务协议、工具与验证范围。
- [多样化引擎系统](docs/09-diverse-engine-system.md)：模型、设备、精度、构建、个性化目标、质量门禁与配置示例。
- [持续迭代与收益复测](docs/10-continuous-improvement.md)：迭代可靠性、模型源码诊断、进度记录和初末版本比较。

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

无人辅助验收使用 `configs/campaigns/qwen35-27b-autonomous-codex.json`：从原始引擎启动，隔离人工配方和跨活动记录。首次完整验收 4 轮、0 次晋升，尚未证明自主收益，详见[持续迭代指南](docs/10-continuous-improvement.md)。

已有 Qwen3.5/A100 专用 eager CUDA Graph 引擎，入口为 `configs/campaigns/qwen35-27b-optimized-codex.json`；该引擎由人工辅助定位和修订，其收益不计入系统自主优化成果。

系统调试和真实优化实验优先使用 Codex CLI：本机配置见 `configs/campaigns/qwen35-27b-iterative-codex.json`，启动与收益复测见[持续迭代指南](docs/10-continuous-improvement.md)。当前接入仍使用本地模型推理，Qwen-Agent 配置保留用于对照。

已实现人工初始引擎上的自优化闭环，并扩展为模型、设备、运行精度和需求可组合的活动。Qwen-Agent 与 Codex 都通过当前稳定引擎调用本地模型，并获得同源的模型/硬件/环境摘要、多负载画像、会话 telemetry、分层历史和受控诊断工具。系统支持非准入 quick probe 的 `promising / inconclusive / revise` 三态反馈与一次或多次有界修订、Amdahl 收益上限、修改表面授权、显式精度白名单、独立质量门禁、自定义限时构建、事务提升与回退。

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

控制程序使用 Python 3.10 及以上版本，进程管理和文件写入限制要求 Linux Landlock。初始后端提供纯 Python 测试引擎及 Transformers 文本推理适配；其他运行库通过人工 `engine.py` 桥接。请求调度默认 `serialized_lock`，活动授权后候选可切换 `concurrent_requests` 并重新通过并发正确性与性能门禁；当前仍没有 continuous batching 或 paged-cache 调度器，多模态也尚未验收。
