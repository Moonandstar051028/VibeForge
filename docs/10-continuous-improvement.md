# 持续迭代与收益复测

系统的目标是持续提出、验证并积累有效改动。每轮可能得到收益、否证或证据不足；只有通过正确性、独立质量和统计性能门槛的候选才能成为稳定版本。不能承诺每轮都更快，也不能用局部算子加速代替整模型验收。

## 无人辅助验收入口

下文的 8.62% 来自人工辅助选题和修订，**不计入系统自主优化收益**。验证系统自主能力请使用独立入口：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vibeforge run autonomous-next \
  --config configs/campaigns/qwen35-27b-autonomous-codex.json
.venv/bin/python -m vibeforge --data-root var/autonomous status autonomous-next
# 活动停止后，独立复测原始版本与自动晋升后的稳定版本
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vibeforge --data-root var/autonomous compare autonomous-next
```

该入口使用原始 `templates/engine/transformers`，独立的 `var/autonomous` 数据目录，以及 `knowledge.scope=current_campaign`。该策略禁止注入项目知识配方、其他活动的迁移记录和历史归档，只保留本活动自身实验经验。中性任务提示不指定优化方案；Codex 通过 MCP 自行读取实际实现、提出假设、编辑、提交。隔离模式关闭 shell 工具，候选与证据访问受路径边界限制。`write_candidate` 支持创建辅助模块，与精确替换工具一样禁止修改上下文/证据、禁止分析阶段编辑、禁止提交后继续编辑。

验收运行开始后，不修改候选、提示词、知识或编排代码，不向 Agent 追加具体策略。保留失败和拒绝结果。4 次有效实验、6 次提案和 4 小时为本入口的上限，仍保留正确性、独立质量和统计性能门槛；预算耗尽时若没有自动晋升，只能报告自主运行尚未取得收益。项目知识模式仍可用于普通生产迭代，但不能与这个隔离验收混称。

### 首次完整隔离验收：没有达到自主收益目标

活动 `qwen35-autonomous-20261112` 从原始引擎启动，在没有主 Codex 追加策略或修改候选的情况下完成 4 次提案、4 次实验、1 次试跑后的自动修订和全部反思，随后因 `max_iterations` 自动停止。3 个候选未通过正式正确性，1 个通过正确性和独立质量但无显著收益，**自动晋升为 0，最终稳定版本与初始版本相同**。

| 轮次 | Agent 自选改动 | 系统结果 |
| --- | --- | --- |
| 1 | 微批调度配置 | 正式并发正确性失败，自动回退 |
| 2 | 静态缓存配置 | 正式生成/并发正确性失败，自动回退 |
| 3 | 混合缓存配置，试跑后自主修订 | 修订版正式正确性失败，自动回退 |
| 4 | 微批输出提取代码 | 正确性和独立质量通过；5 组配对收益 1.49%，95% 区间 −0.03%–2.92%，拒绝 |

第 4 轮未启用所修改的微批调度路径，不能将小幅均值差异归因于该代码修改。其统计结果也不支持性能晋升。这暴露了候选执行机制确认仍不充分的问题。第 2 轮提交说明混用了上轮假设和本轮改动，说明“能运行闭环”不等于“建议质量可靠”。这些结果不能用此前人工辅助的 8.62% 替代。

证据位于 `var/autonomous/campaigns/qwen35-autonomous-20261112/`，逐轮报告见 `summary.md`、`progress.json` 和 `iterations/`。`var/autonomous/launch-audit.json` 与 `completion-audit.json` 记录运行前后控制器、提示词和初始模板的哈希，结束时变化文件列表为空。各优化和修订会话的 `context.json` 均未注入项目知识或跨活动线索。这是流程审计记录，不是操作系统级隔离证明。

活动停止之后修复了三个通用问题，未改动上述历史候选和决定：修订简报仅保留试跑决定、正确性摘要、关键指标及报告路径，真实上下文从约 75 KB 压缩为 12.8 KB；结构化提交收尾获得当前测量和候选差异，明确区分当前假设与上一轮反思；快速并发预检改为优先搭配两个不同长度的冻结输入，避免只测相同短输入。完整正式门槛保持独立。这些修复已做工程回归，但尚未再完成一轮真实模型隔离收益验收，不能声称已经解决自主收益问题。

修复后完整工程回归：`.venv/bin/python -m pytest -m 'not target' -q`，332 项通过，1 项因未设置专用推理解释器而跳过，1 项目标环境测试排除。原始自主验收的可追溯摘要见 `docs/evidence/autonomous-20261112.json`。

### 第二次完整隔离验收与通用工具修复

活动 `qwen35-autonomous-rerun-20261112` 于 2026-11-12 再次从原始引擎运行，完成 5 次提案、4 次有效实验、4 次自主修订和全部复盘。4 个有效候选均未通过正确性；另一提案重写配置时遗漏固定模型路径，在启动候选之前失败。**自动晋升仍为 0，初始与最终稳定版本相同，没有正式性能收益。** 新的异长输入预检确实提前发现微批输出错误并触发修订，但 Agent 对错误原因的判断和修复仍不可靠。

此次活动没有混入人工候选修订。`var/autonomous/audits/qwen35-autonomous-rerun-20261112/` 保存开始、结束、15 个会话的知识隔离审计，以及活动结束后才实施的通用修复清单。运行中 154 个冻结文件的变化列表为空。可追溯摘要见 `docs/evidence/autonomous-rerun-20261112.json`。

活动结束后合入以下通用改进，未给 Agent 指定模型优化配方，也未修改原始引擎、工作负载或准入门槛：

- 候选简报包含实际文件清单、Python 定义及字符偏移。`search_candidate` 返回字面搜索结果和上下文，支持有界源码定位。
- `read_candidate`、`read_evidence` 支持 JSON Pointer（例如 `/mechanism`、`/correctness_precheck/checks`），只读取需要的字段。错误路径会区分候选和证据目录。
- CPU、自 GPU 算子归属和 GPU kernel 的排名分别保留，避免前五个设备算子遮蔽 CPU 开销。模型钩子同时采集位置参数和关键字参数中的张量形状；加载实现的符号索引可直接用于 `read_implementation`。
- 修订上下文明示稳定配置、待修候选配置、原始假设、已有补丁和对应试跑。试跑反馈标记为修订前候选的结果，不能当成新修订验证。
- 分析会话使用独立只读提示，不再拼接要求选题和编辑的优化指令。
- 配置写入先检查固定字段与精度约束，不合法则原文件保持不变，Agent 可在同一会话纠正。最终主机验证继续独立执行；非法候选配置单独记为提案失败，不冒充基础设施故障。
- MCP 桥接进程显式绑定当前控制器代码目录，避免 Codex 过滤环境变量后误加载另一处已安装的旧工具。

上述修复完整回归为 **352 项通过、1 项目标 GPU 测试排除**；已显式启用真实 CPU Transformers 测试。另使用实际 Codex CLI 验证 `search_candidate` 往返及隔离模式无 shell 工具。合入后针对性测试 47 项通过。代码和工具可运行不代表自主收益成立；后续独立活动 `qwen35-autonomous-tooling-20261112` 的结果需单独核验。

## 已完成的本机验收（2026-11-12）

最终引擎 `v-e92069e079523fe50e8f` 通过冻结准入标准：**加权吞吐提升 8.62%，95% 配对 bootstrap 区间 7.13%–10.17%**。5 组交替初版/候选测量均有效，三个负载都通过回归约束。完整逐 token、抽样 logits 与并发隔离检查通过；10 道独立任务重复 3 次，accuracy 均为 1.0。优化引擎上的真实 Codex CLI 就绪检查也通过，确认仍可为后续 Agent 会话提供本地 Responses 推理。

| 负载 | 初版平均 tokens/s | 优化版平均 tokens/s | 配对平均提升 | 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| 256-token 输入 | 11.10 | 12.24 | 10.30% | 6.92%–13.66% |
| 2048-token 输入 | 8.59 | 9.16 | 6.56% | 5.18%–8.46% |
| 512-token 输入、并发 2 | 10.59 | 11.77 | 11.26% | 8.92%–14.30% |

总体指标为各负载吞吐的加权几何均值，不能直接当作混合流量总 tokens/s。结果适用于本机 A100 40GB、Qwen3.5-27B NF4/BF16、Transformers 5.20.2，输出上限 32 token。测量含独立预热，不代表冷请求、长上下文、多卡或广泛任务质量。该方案由主 Codex 辅助定位和修订、本地 Codex 完成过候选实施，最终收益来自独立验证；不能声称本地 Agent 无提示自主发现，也不能保证每轮都提升。

原始报告：`var/campaigns/qwen35-codex-graph-focused-20261112/isolated-validation/1794422232602672337/report.json`。同目录保留正确性、质量、原始样本、源码差异和执行画像。可迁移摘要及报告哈希见 `knowledge/evidence/qwen35-eager-graph-full-model.json`。独立验收使用原准入函数得到 `accept/measured_gain`，没有改写原活动的稳定版本或失败记录。

从经过验证的引擎继续自动迭代：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vibeforge run optimized-next \
  --config configs/campaigns/qwen35-27b-optimized-codex.json
.venv/bin/python -m vibeforge status optimized-next
# 停止后复测后续迭代相对该优化起点的增益
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vibeforge compare optimized-next
```

此入口继续使用 Codex 和本地模型，带预算、正确性门槛、失败反思、恢复与回退。要验证整个系统从未优化起点再发现收益，使用下文通用迭代配置；两种起点的比较含义不同。

## 启动与观察

系统调试和真实优化实验优先使用 Codex。以下配置由 Codex CLI 执行工具和提交候选，推理仍由本机 Qwen3.5-27B 稳定引擎提供，评测在 A100 上运行：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vibeforge run my-iteration \
  --config configs/campaigns/qwen35-27b-iterative-codex.json
.venv/bin/python -m vibeforge status my-iteration
```

该配置的路径与推理 Python 针对本机，需要按环境修改后创建新的活动。负载覆盖约 256/2048 token 输入和并发 2，输出上限 32 token；它缩短实验反馈时间，不代表长上下文或完整部署负载的验收。正式部署应以目标负载另建活动并重新评测。

Codex 简报保留知识配方、迁移线索和当前测量依据，并可调用 `read_implementation`。本地 Responses 适配器保留 Codex 的命名空间工具及调用身份，避免 MCP 工具在转换成 Chat Completions 时丢失。Codex 会话使用项目临时工作区和本地 Responses 端点；这不是调用云端 Codex 模型。保留的 `qwen35-27b-iterative.json` 供 Qwen-Agent 对照使用。切换 Agent 时另建活动，保留各自冻结配置和证据。

Codex 还可通过 MCP 的 `read_candidate`、`replace_candidate`、`check_candidate` 完成候选读取、精确替换与语法检查。本机 bubblewrap 报错时，这条路径不依赖 shell，也不扩大 shell 权限；上下文和证据禁止修改，分析会话不提供编辑工具，提交后禁止继续编辑。每个 Codex 工作阶段保留独立 JSONL 和 stderr，便于定位工具或模型故障。对于至少 16K 的上下文窗口，按窗口大小限额（最多 32KB）的简报直接附在初始请求中，减少为读取已有证据而反复预填充；较大简报仍从文件按需读取。优化阶段的时间按配置分配并预留收尾时间，不再无条件截断到 300 秒。MCP 最多允许 8 次探索读取，并预留编辑、检查和提交能力；若首阶段只读未实施，第二阶段携带候选源码与简报，限制为候选编辑相关工具。就绪响应允许完整的外层 JSON 代码块，但仍必须通过 Schema 校验。收尾阶段没有改动时只报告实施未完成，不沿用历史环境故障作本轮原因。

活动目录 `var/campaigns/<id>/` 保存：

| 文件 | 用途 |
| --- | --- |
| `events/events.jsonl` | 运行中的状态变化与进程启动记录 |
| `iterations/<number>/` | 提案、修改、原始测量、决定与反思 |
| `progress.json` | 已落盘轮次的候选数、测量数、准入数与拒绝原因 |
| `summary.md` | 可读的逐轮结果和置信区间 |
| `cumulative-comparison.json` | 最近一次独立初末版本比较 |

`progress.json` 和汇总在控制器报告检查点更新；运行中的实时进展以事件和当轮文件为准。预算分别限制提案次数、有效实验次数、停滞次数和总耗时。默认分析失败最多重试两次，仍失败会保留分析屏障，避免跳过失败结论继续试验。

## 让后续轮次获得可用信息

- 同一实现和配置的已完成实验继续去重。中断、基础设施失败不被误记为已经验证过的优化。
- 单次无显著收益不再封禁整个优化方向，允许有证据支持的不同参数或实现；错误的显式瓶颈指标会被拒绝，不会悄悄改成 decode。
- 终结阶段检查整个候选文件清单，包含 `engine.json`、辅助文件以及较长的 `engine.py`，避免把已有修改误报为无修改。
- 上一活动同模型、同精度的实验可作为迁移线索，必须在当前环境复验。
- `read_implementation` 可列举并分页读取已加载模型模块的函数或 forward 源码；读取结果绑定稳定版本源哈希，只作实现依据，不充当性能证据。
- 编译缓存写入 `var/cache/inductor`。全局知识条目可以保存局部实验配方，但必须明确其验证范围；整模型候选仍须通过所有准入门槛。
- 上下文压缩保留检索到的知识，避免规则只在原始上下文中存在、却没有送入模型。依赖摘要合并完整环境清单和实际运行时清单，避免把运行时未列举的 Triton 误判成不存在。
- 快速试跑先做正确性预检。预检失败直接保留错误并进入有界修订，不再继续 warmup/profiling，以免后续异常覆盖失败证据。正式准入仍由独立门槛决定。
- Qwen-Agent 会话预留实施、语法检查和提交的调用预算。探索结束后仅提供编辑、检查和提交工具，同时保留知识和源码摘要，避免全部调用次数耗在重复分页上。分页结果明确给出下一页偏移量。

## 可比较的性能证据

测量协议版本 2 让吞吐/端到端目标的串行与并发请求统一使用轻量墙钟计时。需要 TTFT、TPOT 等阶段指标时使用详细观测；详细诊断不混入采用墙钟协议的正式吞吐比较。请求仍检查输出与参考结果是否一致。

如果准入依赖 P95/P99，要求每个负载、每次运行有足够实际请求样本，默认至少 100；否则判为证据无效。样本很少的吞吐实验不能据此声称尾延迟达标。

旧活动的冻结配置不能原地升级计时协议。应创建新活动，保留旧日志用于诊断。已完成活动可执行独立初末版本复测：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vibeforge compare my-iteration
```

复测要求活动停止、配置和环境一致，并重新交替运行初版与当前稳定版。结果写入 `comparisons/<timestamp>/report.json`，不修改稳定版本或迭代计数，使用独立的有界评测时间预算。报告引用该新测量的收益和置信区间，不累加各轮百分比。

## 局部编译配方的验证范围

`knowledge/rules/qwen35-recurrent-compile-component.json` 来自人工辅助定位后的局部实验，不是 Agent 无提示自主发现的结果。对应脚本为 `scripts/experiments/probe_qwen_delta.py`，在同机 A100 上测试 Qwen3.5 单 token recurrent gated delta 辅助函数。

已记录 batch 1/2 约 2.70/2.79 倍局部加速，BF16 输出最大绝对误差为 0，FP32 状态最大绝对误差约 4.77e-7。这组固定形状随机输入测试只支持局部结论。编译冷启动、完整模型 logits、逐 token 一致性、任务质量、不同负载及端到端吞吐都需另行验证；最终结果以真实活动证据为准。

首次整模型候选在活动 `qwen35-iterative-final-20261111` 的第 2 次提案中被拒绝：Transformers 对循环缓存状态标记固定地址，直接编译在跨层使用时触发 `FailOnRecompileLimitHit`。该失败说明单个随机状态的局部测试不足。

随后增加了 `--clone-static-state` 诊断模式，在编译区域外复制状态，并验证 48 个带地址标记的状态。包含复制成本后，batch 1/2 分别约加速 2.45/2.43 倍；BF16 输出最大绝对误差约 0.00098，FP32 状态最大误差约 9.54e-7。CPU 前端诊断还复现了直接编译的上限错误，并验证复制后的 48 个状态没有该错误。修订配方仍只是局部候选，不能据此宣称整模型逐 token 一致或吞吐提升。

## 工程回归记录

2026-11-12 执行 `.venv/bin/python -m pytest -m 'not target' -q`：326 项通过，1 项未设置专用推理解释器而跳过，1 项目标环境测试排除。包含中断候选可重试、不同参数可继续探索、错误瓶颈指标拒绝、尾分位样本门槛、计时路径一致、分析重试、源码读取边界、全文件修改识别、知识压缩保留、实施调用预算、试跑失败修订和初末版本复测闭环。随后 Codex 与迭代可靠性相关 37 项测试通过，覆盖命名空间传输、候选编辑边界、只读分析、阶段切换、提交约束、失败摘要与收尾；另用实际 Codex CLI 验证了命名空间 MCP 读取往返。真实 Qwen GPU 实验独立记录，不以这些测试数量替代模型收益证据。

活动 `qwen35-codex-iterative-20261112` 第 5 次提案由 Codex 完成状态复制编译包装，版本 `v-3cfbfcc9e5936818d4a4`。它通过快速预检，但正式正确性中 `benchmark-concurrent_2-0` 的生成 token 与参考不一致，并发请求检查也未通过，因此回退。短输入、2K 输入及抽样 logits 通过并不意味着全模型数值等价；不能将该候选报告为已获得端到端收益。

针对上述失败，`probe_qwen_delta.py --clone-static-state --preserve-eager-rounding` 仍出现非零局部误差，不能视为修复。新增 `--cuda-graph` 使用 `qwen_delta_graph.py` 工厂重放原始 eager 算子顺序；48 个缓存状态和 48 次跨流请求的局部输出、状态均为零误差，本次 batch 1/2 诊断约加速 1.27/1.83 倍。该配方仍待独立整模型验收，活动 `qwen35-codex-graph-20261112` 保留相同初始版本和冻结负载。

为区分 Agent 自由探索能力与具体实现的有效性，`configs/campaigns/qwen35-27b-graph-validation.json` 将 Codex 的任务限定为验证已测量的 eager CUDA Graph 工厂，仍保留全部正确性、质量、性能门槛。这是人工辅助选题的验证活动，不能报告成 Agent 无提示自主发现。通用迭代入口仍是 `qwen35-27b-iterative-codex.json`。

定向活动第 1 个候选 `v-01d6517190ae6680c279` 通过完整正确性和三次独立质量评测（accuracy=1.0），但未获性能准入。进一步检查发现实际模型传入 `cu_seqlens=None`，触发工厂对额外参数的 eager 回退；前后 cudaLaunchKernel 调用均为 100623，不能声称 Graph 已执行。实验工厂已修正为仅忽略这一已知空参数，未知或非空额外参数继续回退，并让局部脚本传入真实签名。独立验证脚本 `scripts/experiments/validate_delta_graph.py <stopped-campaign>` 使用原活动冻结门槛，报告不自动提升版本。

修订工厂通过源码 AST 确认原始函数不读取 `**kwargs`，再忽略这些透传元数据；不能仅处理 `cu_seqlens=None`，因为实际调用还带有其他未使用参数。候选 `v-e92069e079523fe50e8f` 完整正确性和独立质量通过，诊断记录 `cudaGraphLaunch=720`、`cudaLaunchKernel=81183`（初版为 100623）。`templates/engine/qwen35-eager-graph` 的运行文件与该候选一致；`configs/campaigns/qwen35-27b-optimized-codex.json` 从此引擎继续使用 Codex 迭代。性能最终结果见下方独立验证记录。
