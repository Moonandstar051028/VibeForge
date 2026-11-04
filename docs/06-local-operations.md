# 06：本地运行与恢复

## 1. 当前可用范围

| 项目 | 首版行为 |
| --- | --- |
| 控制程序 | Python 标准库；SQLite 保存权威状态；独立于推理环境 |
| 系统 | Linux aarch64 / x86_64，需要 Landlock；本机 ABI 4 已验证 |
| 测试后端 | 纯 Python CPU 引擎 + 明确标注的脚本 Agent |
| 模型后端 | 本地 Transformers 文本生成、聊天和抽样 logits |
| 优化方式 | Agent 使用稳定引擎推理，在独立候选工作区修改 Python 后端 |
| 评测 | 1—64 个客户端；确定性与近似模式；吞吐/时延/内存/启动/体积目标，详见 [09](09-diverse-engine-system.md) |
| Qwen 状态 | 环境探针通过；完整 BF16 部署因可用内存不足阻塞 |

## 2. 创建控制环境

```bash
cd /path/to/VibeForge
bash scripts/dev/bootstrap.sh
.venv/bin/vibeforge --help
.venv/bin/python -m pytest -m 'not target' -q
```

`bootstrap.sh` 创建项目 `.venv`，复用系统测试工具，并以 editable 方式安装项目。控制程序不导入 Torch；推理依赖由活动中的 `runtime_python` 负责。

## 3. 先跑完整 CPU 闭环

```bash
.venv/bin/vibeforge run demo-001 --config configs/campaigns/toy.json
.venv/bin/vibeforge status demo-001
.venv/bin/vibeforge events demo-001 --after 0
.venv/bin/vibeforge report demo-001
```

每次新活动使用新 ID。该配置执行两轮：第一轮把重复 CPU 求和换成等价算式，第二轮故意改变数值输出。预期接受第一轮、拒绝第二轮、保留第一轮的稳定版本，最后补写分析并停止。速度由实际请求计时得到；脚本 Agent 只验证工程流程。

## 4. Qwen 环境与启动

### 4.1 配置入口

| 文件 | 关键字段 |
| --- | --- |
| `configs/campaigns/qwen36-35b.json` | 推理解释器、额外依赖目录、启动超时、活动预算 |
| `configs/models/qwen36-35b.json` | `weights/Qwen3.6-35B-A3B`（默认相对路径，可由环境变量覆盖）、BF16、CUDA、8192 上下文 |
| `configs/agent/local.json` | Qwen-Agent `Assistant`、本地 OpenAI 兼容接口、最多 6 次模型调用、每次最多 512 新 token |
| `configs/workloads/smoke.json` | 固定提示词、16 个输出 token、文本正确性用例 |
| `configs/evaluation/local.json` | 5 组重复、交替顺序、2% 最低收益、95% 区间、GPU 独占要求 |

设置本机权重和推理解释器路径：

```bash
export VIBEFORGE_MODEL_PATH=/path/to/Qwen3.6-35B-A3B
export VIBEFORGE_RUNTIME_PYTHON=/path/to/inference-env/bin/python
```

模型权重和推理环境不随 Git 发布。`VIBEFORGE_MODEL_PATH` 与 `VIBEFORGE_RUNTIME_PYTHON` 覆盖配置里的通用默认值。缺少的 `accelerate==1.15.0` 可安装在 `var/cache/inference-packages/`，由 `inference_pythonpath` 注入。

```bash
bash scripts/dev/prepare_inference.sh
.venv/bin/vibeforge profile --config configs/campaigns/qwen36-35b.json
.venv/bin/vibeforge run qwen-local-001 --config configs/campaigns/qwen36-35b.json
```

`profile` 和 `run` 返回码：`0` 为通过/正常停止，`2` 为预检阻塞；其他异常返回非零。资源不满足时先记录 `preflight.json` 和 `deployment_blocked` 事件，再退出，不尝试分配完整权重。

### 4.2 不加载权重的环境探针

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:var/cache/inference-packages \
"${VIBEFORGE_RUNTIME_PYTHON:-python3}" \
  scripts/dev/probe_inference.py \
  --model "${VIBEFORGE_MODEL_PATH:?Set VIBEFORGE_MODEL_PATH first}" \
  --confined-cache "$PWD/var/cache" \
  --output "$PWD/var/cache/inference-confined-probe.json"
```

探针验证 CUDA 运算、聊天模板、模型类及 meta 结构。`model_weights_loaded=false` 表示没有验证真实权重推理。启用写入限制时，输出必须放在允许写入的缓存目录。

### 4.3 当前资源要求

- 权重约 66.96 GiB；BF16 预检按权重 × 1.08 + 6 GiB 预留，约需 78.32 GiB 可用内存。
- 历史验收采样约 37.5 GiB 可用；每次启动重新采样，其他 GPU 进程会使独占性能测量无效。
- 保持 BF16 时，需要先由资源所有者安排足够空闲内存和 GPU。完成后可恢复被阻塞活动，启动前会重新预检。
- 新精度契约支持 bnb NF4/FP4/INT8；量化实机兼容性尚未验收。默认精度固定；显式白名单搜索及质量约束见 [09](09-diverse-engine-system.md)。初始引擎必须先能部署。

## 5. 查看证据

| 路径 | 内容 |
| --- | --- |
| `var/state/controller.sqlite3` | 活动状态、迭代、提升事务、事件 outbox；权威来源 |
| `var/profiles/<hash>.json` | 模型结构、设备、软件版本和负载画像 |
| `var/campaigns/<id>/manifest.json` | 启动时冻结的组合配置及哈希 |
| `configs/` | 此处指活动内目录：冻结的模型等配置、Agent 提示词、用例 |
| `preflight.json` | 资源、依赖和部署准入检查 |
| `events/events.jsonl` | 数据库事件的可读导出，带 UTC、单调时钟和 boot ID |
| `bootstrap/`、`reference.json` | 初始协议检查、正确性和冻结参考输出 |
| `iterations/0001/commands.jsonl` | Agent 模型响应、工具调用、进程命令和退出状态 |
| `iterations/0001/logs/` | Agent 和各次引擎启动的原始输出 |
| `iterations/0001/benchmarks/` | 原始样本、资源快照、区间和判定 |
| `iterations/0001/analysis.md` | 可核实的实验事实与 Agent 解释分开记录 |
| `resume/context.json` | 从权威状态和历史记录重建的上下文 |
| `knowledge-drafts/` | 带证据引用的草案，不自动变为通用优化规则 |
| `summary.md` | 当前状态、稳定版本和各轮结果 |

日志持续追加并刷盘；事件先随事务写入 SQLite，再导出 JSONL。引擎输出实时重定向到文件。终端可使用 `tail -f` 查看文件，或反复调用 `events <id> --after <sequence>` 读取新增事件。

## 6. 停止、恢复与版本切换

```bash
.venv/bin/vibeforge stop demo-001
.venv/bin/vibeforge resume demo-001
```

`stop` 请求在当前轮结束后停止；Agent、推理请求和启动过程各自受超时限制。`resume` 会：

1. 获取活动锁和数据根目录内的设备锁，禁止两个活动同时评测。
2. 按 PID、启动时刻、boot ID、随机标记清理本活动遗留进程。
3. 取消尚未提交的提升意图；未提交迭代记为中断拒绝，已经提交的提升保留接受判定；恢复待分析队列。
4. 重建画像和上下文，确认模型、硬件及软件环境没有漂移，再继续。

预算从活动首次启动计算，包含停机时间；达到预算的活动恢复后会结束。模型、软件或验收协议改变时，应创建新活动；精度只有在原活动已授权的白名单内才可切换。历史 schema v1 活动保留查看，新评测契约需创建 v2 活动。

发布版本通过内容哈希冻结，启动前重新校验。候选不满足正确性、性能或执行要求时，重新启动原稳定引擎并进行协议检查；活动结束后清理本活动推理进程。

## 7. 开机恢复服务

`deploy/systemd/vibeforge-controller@.service` 是用户级服务模板，当前没有安装或启用。使用前先修改其中项目路径，并确保活动已创建：

```bash
mkdir -p "$HOME/.config/systemd/user"
cp deploy/systemd/vibeforge-controller@.service "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now vibeforge-controller@demo-001.service
journalctl --user -u vibeforge-controller@demo-001.service -f
```

用户服务随用户管理器启动；无人登录时自动运行还取决于管理员配置。`Restart=on-failure` 用于异常退出恢复；正常达到预算后保持停止。模板按具体活动实例化。
服务模板默认假设项目位于用户主目录下的 `VibeForge/`；若克隆到其他位置，先相应修改 `WorkingDirectory` 和 `ExecStart`。

## 8. 测试入口

```bash
# 工程测试，不启动真实大模型
.venv/bin/python -m pytest -m 'not target' -q

# 真实目标验收：资源不满足时应报告失败，不按跳过或通过处理
VIBEFORGE_TARGET_TEST=1 .venv/bin/python -m pytest \
  tests/e2e/test_target_deployment.py -q
```

底层 `serve`、`agent-worker` 是受管理子进程入口；日常使用 `run` / `resume`，由控制程序设置缓存、写入限制、进程身份和生命周期。
