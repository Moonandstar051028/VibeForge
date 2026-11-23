# VibeForge v2

基于 Agent 的本地大模型推理引擎自优化项目。输入固定模型、硬件和负载，从初始引擎开始，逐轮生成候选并由宿主独立评测。

固定配置 → 启动稳定引擎 → 重新勘测 → Agent 修改候选副本 → 构建与快速探测 → 功能、质量、配对性能评测 → 接受或回退 → 下一轮。

## 快速开始

Python 3.10+。GPU 运行需要与本机 CUDA 及模型匹配的 PyTorch、Transformers、safetensors 和 accelerate。通用 `inference` extra 限定 Transformers 4；需要 Transformers 5 的模型应使用独立环境。已记录的通用环境见 [environment-tested.json](environment-tested.json)，Qwen3.8 环境见 [引擎说明](engines/transformers-qwen38/README.md)。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
# 通用 Transformers 4 引擎可安装：pip install -e '.[inference]'
```

本机已有独立推理环境。先核对配置中的权重路径和 `runtime_python`，再从项目根目录启动：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/yxy/.venvs/vibeforge-qwen38/bin/python -m vibeforge_v2 run \
  --config configs/qwen3-32b-a800.json --output runs/qwen32-new --rounds 1
```

连续运行将 `--rounds 1` 换为 `--continuous`，两者互斥。每次新实验使用新输出目录；恢复只能使用该实验自己的冻结配置与检查点。

```bash
PYTHONPATH=src python -m vibeforge_v2 status --output runs/qwen32-new
# GPU 恢复命令须沿用启动时的 Python 与 CUDA 环境
PYTHONPATH=src python -m vibeforge_v2 resume --output runs/qwen32-new --rounds 3
```

`--rounds` 表示包含已完成轮次的目标总数。普通候选失败后保留稳定版本并进入下一轮；完整性异常停止整次运行。SIGINT/SIGTERM 保留现场并清理自有模型服务。

## 配置与 Agent

| 配置 | 用途 |
| --- | --- |
| `configs/qwen3-32b-a800.json` | Qwen3-32B，Agent thinking 开启，目标评测关闭 |
| `configs/qwen3-coder-30b-a3b-a800.json` | Qwen3-Coder-30B-A3B-Instruct，thinking 关闭 |
| `configs/qwen3-coder-30b-a3b-a800-representative.json` | Coder 更长输入、输出及更高并发的验证矩阵 |
| `configs/qwen3.8-27b-a800.json` | Qwen3.8-27B，Agent thinking 开启 |
| `configs/qwen3.8-27b-a800-nothinking.json` | Qwen3.8-27B，Agent thinking 关闭 |
| `configs/toy.json` | CPU 流程测试 |

标准 CLI 默认让 Agent 连接本轮稳定目标引擎。支持 Qwen-Agent 与 Codex 后端；后端不等同于目标模型身份。跨模型优化需要显式配置独立服务，不会仅凭模型名称自动完成切换。

正式实验使用 A800（`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`）。真实模型配置关闭跨运行历史读取并每轮重新勘测；同一次运行的已验证结果与失败反馈仍会积累。Agent 自行生成候选，宿主不代写优化方案。

## 评测与证据

接受候选必须同时满足功能、独立质量、正式配对性能和激活检查。A800 配置默认使用固定 50 题质量筛查、5 组 AB/BA 性能配对、2% 最小收益和 3% 单场景退化保护。

性能主指标由 `evaluation.objective` 固定；常用 `request_output_tokens_per_second_mean` 为每请求输出 token 数除以客户端端到端耗时，再按请求取均值，多场景按固定权重取几何平均。小规模固定负载的收益不能直接推广到生产负载。完整规则见 [评测规则](docs/correctness-policy.md)。

运行根目录保存冻结配置、参考和状态；每轮保存真实 diff、工具调用、勘测和宿主判定。`current/manifest.json` 标识唯一当前稳定版本，`rounds/NNNN/decision.json` 是轮次结论，Agent 的 `analysis.md` 只是说明。旧实验产出已清理，新实验会重新生成 `runs/` 和 `workspace/`。

## 目录与文档

| 路径 | 内容 |
| --- | --- |
| `src/vibeforge_v2/` | 主循环、Agent、工具、上下文与准入 |
| `src/vf_runtime/` | 服务、勘测、评测、统计和存储 |
| `engines/` | 初始 Transformers 引擎与 CPU 测试引擎 |
| `configs/`、`prompts/` | 固定配置与提示词 |
| `benchmarks/` | 功能及质量评测输入 |
| `scripts/` | 评测辅助、审计、监控和恢复工具 |
| `tests/` | 回归测试 |
| `docs/` | 架构、上下文、评测与运维说明 |

- [架构与模块职责](docs/design.md)
- [Agent、工具与上下文](docs/agent-context.md)
- [功能、质量与性能规则](docs/correctness-policy.md)
- [运行、监控与异常恢复](docs/operations.md)
- [每轮 Agent 工作优化计划（待实施）](docs/round-optimization-plan.md)
- [版本来源与 Git 分支](docs/version-history.md)

## 开发检查

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src pytest -q -p no:cacheprovider
PYTHONPATH=src python scripts/demo_cpu.py --output /tmp/vibeforge-cpu-demo
```

CPU 示例使用明确的脚本改动验证接受/拒绝流程，不代表大模型自主优化或 GPU 收益。底层来源与有意改动哈希保存在 `vendor-manifest.json`。
