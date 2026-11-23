# 运行、监控与异常恢复

## 启动与停止

从项目根目录启动。核对权重、Python、精度、设备和负载；A800 使用 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`。新实验使用新的 `runs/<名称>`，关闭跨运行历史。完整命令见 [README](../README.md)。

`run --continuous` 无轮数上限；普通候选拒绝后继续下一轮。`--rounds N` 为目标总轮数，与连续模式互斥。中断时发送 SIGINT/SIGTERM，让 pipeline 保存现场并清理自有服务。

`status --output <运行目录>` 读取 `state.json`，不检查进程是否存活。判断正在运行须同时检查进程身份、模型 health、日志时间和 GPU；`workspace/active.json` 只是记录，不能单独作为存活证据。

## 保留的辅助脚本

| 脚本 | 用途 |
| --- | --- |
| `quality_cases.py` | 正式质量评测判分，配置直接引用 |
| `build_quality_suite.py` | 重建固定质量题集 |
| `build_representative_config.py` | 生成更广负载矩阵配置，不覆盖既有运行 |
| `demo_cpu.py` | CPU 接受/拒绝流程示例 |
| `audit_run.py` | 只读汇总宿主证据与实际工具结果 |
| `supervise_run.py` | 核验检查点后恢复自有连续实验 |
| `watch_run.py` | 监督器使用的阶段和工具事件观察器 |
| `monitor_live.py` | 汇总进程、GPU、Agent 和阶段状态 |
| `review_agent_context.py` | 由监控器调用，保存异常上下文核查材料 |

审计：`PYTHONPATH=src python scripts/audit_run.py --help`。

实时监控需要实际 GPU UUID，避免假定不同机器的编号相同：

```bash
nvidia-smi --query-gpu=index,name,uuid --format=csv
PYTHONPATH=src python scripts/monitor_live.py \
  --run runs/<名称> --workspace workspace/<名称> --gpu-uuid <A800_UUID>
```

## 恢复边界

正常恢复沿用实验启动时的 Python 与 CUDA 环境，使用 `resume --output <运行目录>`。已提交轮次不覆盖，中断轮保留证据；初始基准尚未建立时需新运行。

后台监督器需要已有稳定检查点，且 `workspace/active.json` 指向对应 `run` 与 `workspace`。用与 pipeline 相同的环境启动：

```bash
PYTHONPATH=src python scripts/supervise_run.py \
  --run runs/<名称> --workspace workspace/<名称>
```

监督器核验稳定源码、依赖和冻结输入，清理本次实验残留服务，再执行连续恢复。完整性异常或反复无进展退出进入 `needs_diagnosis`。主动停止时向 `supervisor-status.json` 中的监督器发送 SIGTERM，避免子进程退出被识别为意外故障。

共享依赖检查出现 `Permission denied` 时，表示无法验证文件身份，不足以直接认定文件被篡改。需保存错误、核查实际权限与依赖身份，再决定如何建立可信运行；不删除检查或强制清除失败标记。

## 产出管理

`runs/` 保存冻结输入、候选、依赖环境、勘测和判定；`workspace/` 保存监控、进程身份与运维日志。只有确认实验及自有服务退出后才清理。2026-11-22 已按用户要求删除所有旧实验产出及历史修复报告；当前项目保留源码、输入和通用工具。
