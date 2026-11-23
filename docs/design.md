# 架构与模块职责

## 优化循环

1. 固定模型、精度、硬件、负载和准入条件，在新运行目录冻结配置、输入与参考。
2. 构建初始引擎，建立功能和质量基线。
3. 启动稳定引擎，每轮重新采集性能勘测和并发诊断，复制稳定源码为候选。
4. 验证 Agent 的真实工具连接，提供源码路径、加载状态、瓶颈摘要与本次运行历史。
5. Agent 自主研究、修改和检查候选，用自然语言结束；宿主记录真实 diff。
6. 执行构建、CPU harness、快速探测、功能、独立质量和正式配对性能评测。
7. 合格后验证候选能承载 Agent 并更新稳定指针；失败保留稳定版本，继续下一轮。

真实模型配置设置 `profiling.reuse_stable_survey=false`。底层仍支持按稳定源码、依赖、负载、设备和采集设置缓存勘测；显式启用复用时，摘要记录来源与有效期。

## 模块

| 模块 | 职责 |
| --- | --- |
| `vibeforge_v2/loop.py` | 主循环、快照、提交、接受/回退与恢复 |
| `vibeforge_v2/backends.py`、`agent.py` | Qwen-Agent / Codex 后端、连接检查与工具对话 |
| `agent_context.py`、`agent_trace.py` | 精确预算、工作记忆与请求证据 |
| `edit_tool.py`、`config_tool.py`、`checks.py` | 候选编辑、配置序列化与静态检查 |
| `harness.py` | 缩小模型 CPU 路径等价性检查 |
| `environment.py`、`shared_guard.py` | 候选依赖环境与共享身份检查 |
| `evaluate.py`、`correctness.py`、`quality.py` | 构建、勘测及功能/质量评测编排 |
| `feedback.py`、`experience.py` | 基于宿主事实的反馈与经验台账 |
| `vf_runtime/` | 模型服务、请求、画像、性能测量、统计及存储 |

底层命名空间的来源和有意改动哈希保存在 `../vendor-manifest.json`。

## 状态与提交

`current/engine` 指向已接受引擎，`current/manifest.json` 保存来源和哈希。候选位于 `rounds/NNNN/engine`，拒绝不覆盖稳定入口。运行状态在 `state.json`，轮次决定在 `decision.json`。

先保存包含 `stable_after` 与哈希的轮次决定，再更新状态；恢复时重放已提交决定，保留中断轮证据，从稳定检查点进入后续轮次。完整性失败停止整次运行，不能直接恢复。

非 fixture 默认使用候选独立依赖环境。工具、构建和模型服务采用相应 Python。宿主验证固定配置、稳定源码、共享依赖和冻结评测材料；Shell 的写入限制不替代这些检查。进程清理核对所有权，只清理本次实验的服务。

Agent 与工具细节见 [上下文说明](agent-context.md)，准入见 [评测规则](correctness-policy.md)，运行方式见 [运维说明](operations.md)。
