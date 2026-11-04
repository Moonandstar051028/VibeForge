# VibeForge

面向模型、硬件和工作负载的轻量推理引擎生成与自优化研究项目。

## 项目入口

- [项目目录说明](docs/05-project-layout.md)：目录树、每个文件夹的职责、文件放置和命名规则。
- [系统设计文档](docs/README.md)：总体架构、迭代恢复、评测日志和知识库设计。
- [本地运行指南](docs/06-local-operations.md)：环境、命令、配置、日志与恢复。
- [实现与验收报告](docs/07-implementation-validation.md)：逐模块实现、实测结果及当前限制。
- [Qwen-Agent 接入](docs/08-qwen-agent-integration.md)：组件边界、本地服务协议、工具与验证范围。

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

已实现首版单设备、单请求自优化闭环：画像与部署预检、独立控制程序、Qwen-Agent `Assistant`、候选冻结、差分正确性检查、重复性能评测、事务提升与回退、崩溃恢复、持久化日志及知识草案。Qwen-Agent 通过本地推理引擎的 OpenAI 兼容接口调用所配置的 Qwen 模型。

47 项工程测试通过，覆盖 Qwen-Agent 工具调用、候选编辑、独立 worker、最终分析、评测与回退。CPU 测试引擎和模拟 OpenAI 服务不代表真实模型的优化能力。真实 Qwen BF16 部署验收仍因内存不足失败，详见验收报告。

## 快速验证

```bash
cd /path/to/VibeForge
bash scripts/dev/bootstrap.sh
.venv/bin/python -m pytest -m 'not target' -q
.venv/bin/vibeforge run my-toy-run --config configs/campaigns/toy.json
.venv/bin/vibeforge status my-toy-run
```

控制程序使用 Python 3.10 及以上版本，当前进程管理和文件写入限制要求 Linux Landlock。初始后端提供纯 Python 测试引擎及 Transformers 文本推理适配；原生编译、多模态和并发优化尚未实现。
