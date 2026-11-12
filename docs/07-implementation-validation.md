# 07：实现与本地验收报告

最近实现与文档对照：2026-11-10；事件文件保存 UTC 时间。本报告按时间保留验收事实，不把活动的瞬时运行状态当作永久能力。下方 2026-11-04/05 的测试数量、软件版本、35B 内存和部署结论均为历史记录，不能当作当前 27B 实验状态；活动实时状态以 `vibeforge status <id>` 和 SQLite 为准。

## 统一自测调优闭环（2026-11-10，campaign schema v4）

本轮按 P0 到 P6 逐阶段实现并在每阶段运行系统验证；后续阶段只在前一阶段测试和活动结果符合预期后继续。

| 阶段 | 已实现与验收结果 |
| --- | --- |
| P0 | 统一 telemetry/profile/probe/refinement 循环契约和 `PROBING/REFINING` 状态；toy 活动验证接受、正确性拒绝和稳定版保留 |
| P1 | Qwen-Agent 与 Codex 首屏都包含模型特征、硬件、环境、历史、知识和 profile；两条路线共享证据、历史、会话 telemetry 与受控 profile 能力。Qwen/Codex fixture 活动及真实 stdio MCP 协议通过 |
| P2 | Agent 推理期间按会话采集普通请求 telemetry；稳定版按多个 profile/prompt 轮转建立画像矩阵，记录 KV 逻辑 tensor 字节及明确缺失项。双 prompt 活动通过并正常停机 |
| P3 | 最近 6 条保留详单，更早全部轮次进入策略聚合摘要；当前及归档活动可检索，兼容经验分为已验证策略与反例。两轮活动产生一条局部策略和一条正确性反例 |
| P4 | 候选先 quick probe，可将正确性/性能方向反馈给独立修订工作区；修订重新过候选门禁、构建和 probe，probe 不参与正式准入 |
| P5 | 可加阶段按普通请求墙钟占比计算 Amdahl 收益上限；数量级不可能的计划以 `implausible_expected_gain` 在构建前拒绝 |
| P6 | `allowed_surfaces` 控制运行配置、执行 adapter、cache、scheduler 和 native kernel；实现串行/并发请求调度，原生源码要求冻结 build recipe 并随 release 哈希和完整门禁验证 |

最终非目标/非推理模型回归为 `278 passed, 2 deselected`，耗时 219.45 秒。随后使用 `/home/yxy/agent-to-inferrence/.venv/bin/python` 单独运行离线 tiny Transformers 的两种架构 × 两种精度真实推理接口测试，结果 `1 passed`，耗时 27.68 秒。剩余 target 测试指向旧 Qwen3.6-35B/BF16 配置，未用它冒充当前 Qwen3.5-27B/NF4 验收；这些结果都不能代替真实目标模型的候选 A/B。

随后运行 `toy-probe-refinement-system-validation`：第一次 probe 正确性失败，执行 1 次修订后第二次 probe 通过；正式配对吞吐收益 14.46%，95% 区间为 11.49%–16.30%，候选晋升为 `v-f02210ee1ba2abd6802b`，最终状态 `STOPPED` 且引擎关闭。另运行 `toy-scheduler-surface-system-validation`，候选以 `request_scheduler=concurrent_requests` 启动并通过正式门禁，配对吞吐收益 15.29%，95% 区间为 12.15%–17.69%，晋升为 `v-399d9972616ff3e2e60c` 后正常停机。这些收益来自刻意设置固定算术开销的确定性 CPU fixture，只证明控制闭环、测量和晋升有效，不代表真实模型收益。

最终代码另运行 `final-p0-p6-system-validation-20261110`。首个错误候选被 quick probe 拒绝；修订结果与上述已接受的 14.46% 策略指纹一致，跨活动历史门禁返回 `duplicate_change` 并指向原活动及 `measured_gain` 决策，没有重复执行或重复累计同一收益。活动恢复初始稳定版、完成复盘并以 `STOPPED` 关机。这是长期去重的系统实测，不是一份新的性能提升。

新 schema v4 已在目标 Qwen 模型和实际 CUDA 环境运行一次完整规划活动 `qwen35-27b-v4-validation-20261110`。Qwen3.5-27B/NF4 在 GPU 0 实际加载，启动耗时 24.61 秒；初始 14 项正确性检查和独立质量门禁通过。系统对 5 个 profile × 2 个冻结 prompt 建立 10 项画像：interactive 256→32 约 3.23 秒、9.91 token/s；agent_2k 2048→32 约 4.35 秒、7.36 token/s；context_8k 8192→32 约 7.41 秒、4.32 token/s，内部 TTFT 约 4.20 秒。各项的第二 prompt 及 long-decode/concurrent 诊断也保存在矩阵证据中。

同一活动的 Agent 自推理会话记录 20 次普通 `chat_completion`、478,252 输入 token、2,267 输出 token及 496.32 秒后端墙钟；聚合输出速率 4.57 token/s，TTFT P50 12.78 秒，prefill P50 12.66 秒，decode P50/P95 为 5.88/24.59 秒。由此证明模型、环境、profile 矩阵和 Agent 自身推理性能均已进入可审计上下文，不再只是配置声明。

这次真实 Agent 最终调用 `report_blocked`，理由集中在 bitsandbytes GEMM 热点及缺少 flash-attn/paged cache，没有继续评估已经授权的 scheduler、execution adapter、cache policy 和受控 native kernel 路径。因此没有生成候选，没有 quick probe 或正式 A/B，稳定版本保持 `v-5f3bddfc445907d9027f`；活动按停止请求结束为 `STOPPED`，引擎已关闭。该结果证明目标闭环可运行和可诊断，但**尚未证明真实模型有调优收益**，不能与下方 toy 的 14%–15% 收益合并。

复盘同时发现 Transformers 5.20 的 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding` 时，宿主曾将字段数误记为 token 数，导致上下文预算失真。现从 `input_ids` 精确计数，并在超预算时压缩 prompt 预览和 profiler 冗余说明；完整 prompt、runtime 清单和画像仍保存在冻结证据中。Qwen/Codex 策略提示也要求在 `report_blocked` 前逐一审计所有授权修改面，缺少 flash-attn 只能排除该具体机制。修复后 Qwen 端到端 worker 及全量回归通过；真实 27B 活动发生在修复前，需以新活动验证 Agent 是否能形成候选并通过正式 A/B。

随后以相同配置哈希运行修复后活动 `qwen35-27b-v4-postfix-20261110`。真实首轮规划的静态上下文由模型 tokenizer 精确计为 13,797 token（输入预算 22,119，工具历史预留 4,096，剩余 8,322），证明不再出现固定 2 token 的错误计数。Agent 读取当前 10 项画像、会话 telemetry、源码和历史后提出 DynamicCache→StaticCache；期间实测暴露并修复两个 v3 接口问题：campaign objective 曾被模型的 measurement key 覆盖，且辅助证据排在画像前时数值绑定会选错文件。v3 现由宿主强制填充冻结 objective/direction、画像数值引用、目标 profile、runtime 前提和瓶颈方向；原始第 3 轮提交离线重放通过，绑定 decode 3.308 秒、时间占比 75.17%，8% 总体收益预测所需 decode 改善约 9.85%，未超过 Amdahl 上限。

第 4 轮 StaticCache 候选 `v-6c477c6e984410ce4159` 随后实际构建和加载，并进入 quick probe。结果为反例：最短冻结请求的串行/并发隔离检查失败；`agent_2k` 普通请求的 decode 从 3.244 秒增至 3.515 秒，退化 8.33%，预测未出现。系统关闭候选、保持稳定版 `v-5f3bddfc445907d9027f`，没有运行正式 A/B，也没有晋升错误候选。修订会话还暴露完整 probe 中逐 token timing 和算子表会挤爆上下文；现只注入检查结果、机制 before/after、实际 cache 类和关键请求指标，完整 `probe.json` 通过证据路径读取。旧活动中该修订错误一度把最终原因覆盖为 `execution_failed`，使复盘误称策略未测试；控制器现以 `candidate_probe_failed` 和 probe 路径作为主判定，修订传输错误只作为附加字段，确保该反例进入长期经验且不会被重复尝试。目标配置的 10 项画像实际约需 20 分钟，原 600 秒缓存期会在一次 Agent/复盘后必然过期，现提高为 3,600 秒；已冻结旧活动不被改写。

因此目前真实 27B 结论是：系统已经完成“自推理 telemetry→多 workload 画像→历史检索→候选生成→Amdahl 门禁→构建→quick probe→拒绝并恢复”的实际闭环，并获得一个可复用反例；仍没有通过正式正确性与配对 A/B 的真实提速候选。可运行性和坏策略淘汰已验证，目标模型上的良好调优收益尚不能宣称完成。

真实 DRAM 带宽、SM occupancy、continuous batching 和 paged-cache 执行路径仍未实现或测量；文档不把瞬时 GPU utilization 或逻辑 KV payload 冒充这些指标。

在 `CUDA_VISIBLE_DEVICES=0` 下对当前 `qwen35-27b-nf4.json` 执行配置级 profile 预检已通过：识别 Qwen3.5-27B 的 1,199 个 tensor、原始权重 55,562,872,800 字节，NF4 运行权重估算 18,335,742,448 字节，总需求估算 26,245,052,787 字节；目标 A100 可用 41,960,341,504 字节，未发现该可见设备上的其他 GPU 进程。该估算明确不覆盖 KV、激活和编译峰值；预检通过只说明具备启动条件，不等于实际加载、完整负载或性能调优已通过。

真实活动证据位于 `var/campaigns/qwen35-27b-v4-validation-20261110/`：初始门禁见 `bootstrap/correctness.json` 与 `initial-quality.json`，矩阵见 `observations/v-5f3bddfc445907d9027f/`，Agent 自推理见 `iterations/0001/agent-telemetry.json`，最终无候选判定见 `iterations/0001/decision.json` 和 `summary.md`。运行产物不作为源码提交内容。

## 本次正确性门禁增量（2026-11-05，schema v3）

自生成 `reference.json` 现在明确作为回归基线。真实模型使用自参考时必须配置独立任务质量下限和相对初始质量退化限制；初始引擎及每个候选都受检查。重复参考保存 token 变体、共识及有界 logit 包络，拒绝基线自身的不稳定、非有限数值与外部参考用例缺失。质量评测重复执行并取最差指标，保留逐次证据。设计与配置详见 [03：当前正确性实现](03-evaluation-and-observability.md#当前实现v3-组合门禁)。

本次非目标模型全套回归为 `208 passed, 1 skipped, 1 deselected`，耗时 236.72 秒。跳过的是未配置推理解释器的可选离线 Transformers CPU 测试，真实目标测试未选中。最后补充畸形评分结果和任务请求失败处理后，评测集成、错误初始引擎及精度搜索闭环另跑 `21 passed`，耗时 30.39 秒；两组有重叠，不能相加。证据分别为 `var/validation-correctness-v3/tests.xml` 和 `quality-followup.xml`。

新增关键断言包括：即使自参考差分全部通过，独立答案失败也不能建立稳定版本；基线低共识或数值异常不能扩大候选容差；多次评分中一次退化不能被均值掩盖；畸形指标保留失败报告；任务请求失败即使总准确率仍够高也不能通过；旧协议活动不能直接恢复为新协议。

```bash
.venv/bin/python -m pytest -m 'not target' -q \
  --junitxml=var/validation-correctness-v3/tests.xml
.venv/bin/python -m pytest tests/integration/test_evaluation_gates.py \
  tests/e2e/test_correctness_oracle.py tests/e2e/test_precision_search.py -q \
  --junitxml=var/validation-correctness-v3/quality-followup.xml
```

Qwen 27B、本地默认及精度搜索配置均通过配置加载校验，默认重复参考与质量各 3 次。本次没有执行真实 Qwen GPU 评测，也没有修改历史活动证据；新增的 10 条公开基础题和阈值尚待目标模型校准。下方历史 GPU 和活动记录不代表已通过 v3 门禁。旧 v1/v2 活动不能直接续跑新协议，需要新建活动重采参考。

## 0. 当前增量：真实配置、阶段观测和负载矩阵（2026-11-05）

按“真实生效配置 → 分阶段/模块观测 → 代表性负载 → 修改及验证”更新现有实现及文档。原活动 `qwen35-27b-nf4-loopcheck-20261105-124007` 已停止；没有把新协议混入旧冻结活动。此后已使用 `configs/campaigns/qwen35-27b-nf4.json` 建立独立持续优化活动，具体进展见本节后续时间线。

| 范围 | 本次验收 |
| --- | --- |
| 真实配置 | `/health`、启动 runtime 文件和诊断记录包含已加载 attention 配置、模块类型、NF4 精度校验、包版本、调度及执行路径；普通请求另记实际 generate 参数和 cache 类。拦截已生效默认后端的单独重复设置、缺失依赖和缺少 paged-cache/执行路径的选择 |
| 正常路径 | 分词/反分词、prefill/decode forward、内部 TTFT/TPOT 和 token 间隔、排队、CUDA allocated/reserved 峰值；缺失可信值不使用候选自报值替代 |
| 独立诊断 | CUDA/CPU 算子及 attention/MLP/lm_head 模块分组；修复同名 CPU/CUDA annotation 覆盖数值、lm_head 错归 linear attention 的问题，诊断 schema 升为 3 |
| 负载矩阵 | 五类输入/输出/并发组合，冻结实际 tokenizer 长度和完整输出参考；逐场景报告、加权几何吞吐得分及单场景退化保护 |
| 闭环 | v2 计划绑定数值/单位、运行前提和场景；候选同条件诊断、瓶颈预测对照、正式收益区间和实验记忆。确定性 Toy 测试验证完整编排，不能代替真实模型 Agent 的推理能力验收 |
| 资源条件 | 正式组前后记录可见 GPU 温度、频率、功率和利用率快照，GPU 温度门限使用 GPU 传感器；不冒充连续利用率、带宽、occupancy 或能耗 |

完整回归为 `165 passed, 1 deselected`，耗时 181.65 秒，包含离线随机小型 GPT-2/Llama 的四种模型/精度组合。后续调整完成后，契约、闭环、GPU 快照及兼容性用例另跑 `41 passed`，包括混合 root/text 配置防误拒和畸形计划字段的错误反馈。两次结果相互重叠，不能相加为测试总数。JUnit 在 `var/validation-observability-v2/tests.xml` 和 `followup.xml`。

文档与实现对照时再次执行未选择真实目标的当前测试集，结果为 `189 passed, 1 skipped, 1 deselected`，耗时 215.99 秒。跳过项是未设置 `VIBEFORGE_TEST_INFERENCE_PYTHON` 时的可选离线 Transformers CPU 测试；该结果不能替代下方已有的 GPU 实测，本次文档对照也没有占用 GPU 重跑目标测试。

```bash
VIBEFORGE_TEST_INFERENCE_PYTHON=/home/yxy/agent-to-inferrence/.venv/bin/python \
  .venv/bin/python -m pytest -m 'not target' -q \
  --junitxml=var/validation-observability-v2/tests.xml
```

真实 GPU 验证使用 GPU 0 的 A100 40GB、已有 Qwen3.5-27B 权重、NF4 double quant、BF16 compute。实际模型为 `Qwen3_5ForConditionalGeneration`，606 个量化层通过运行时精度检查；已加载 root/text attention 均为 `sdpa`，文本模型有 48 个 gated-delta 和 16 个 full-attention 层。`flash-attn`、`flash-linear-attention`、`causal-conv1d` 不在本次环境中；Transformers 内的可调用 fallback wrapper 不能当作已安装优化内核。

下表是完整长度的功能/观测验证，每个串行场景一次请求，并发场景两次请求；不是五组候选 A/B，也不提供可靠 P95/P99 结论：

| 场景 | 实际输入 → 输出 token | 内部 TTFT | 平均 TPOT | 后端墙钟 | 峰值 allocated |
| --- | --- | ---: | ---: | ---: | ---: |
| interactive | 256 → 128 | 0.815 秒 | 83.2 毫秒 | 11.381 秒 | 17.06 GiB |
| agent_2k | 2048 → 128 | 1.090 秒 | 80.6 毫秒 | 11.325 秒 | 17.69 GiB |
| context_8k | 8192 → 64 | 4.180 秒 | 82.3 毫秒 | 9.368 秒 | 20.16 GiB |
| long_decode | 512 → 256 | 0.338 秒 | 82.7 毫秒 | 21.434 秒 | 17.12 GiB |
| concurrent_2 | 每请求 512 → 64，并发 2 | 0.338 / 0.349 秒 | 82.9 / 81.6 毫秒 | 5.563 / 5.490 秒 | 17.12 GiB |

内部 TTFT 和后端墙钟不含排队及 HTTP。并发第二个请求额外排队约 5.563 秒，确认当前调度为串行锁，不能把并发提交误认为 continuous batching。8K 请求的正常 prefill forward 为 4.160 秒，2K 为 1.080 秒，可作为按输入长度定位瓶颈的证据。

独立 2K→32 诊断成功区分六类区间：prefill 1 次、decode 31 次、linear attention 1536 次、full attention 512 次、MLP 2048 次、lm_head 32 次；CPU/CUDA 同名记录没有互相覆盖。普通 2K→32 请求 prefill/decode forward 分别约 1.076 / 2.514 秒，不能将插桩模块时间相加当作这两个墙钟区间。剖析后重复生成 token ID 一致，测试引擎退出并清理。

证据：`gpu-report.json`、`gpu-health.json`、`gpu-workloads.json`、`gpu-diagnostic.json`、`gpu-engine.log`，均位于 `var/validation-observability-v2/`。本次证明完整 27B 的观测和长度矩阵可运行；尚未运行新矩阵的五组正式候选 A/B、未证明模型自主提出优化的质量或实际提速。合成提示词也不能代替真实业务分布和独立质量数据集。

**同日持续实验中的上下文修复。** 新活动 `qwen35-27b-nf4-matrix-20261105-174604` 的初始基线 14 项检查通过。第 1 轮出现重复 `list_files/read_file`，运行日志的初始上下文估算超过可用输入空间；用相同上下文和实际 Qwen-Agent 截断器重放，最新源码结果被替换成 `omit`，初始 JSON 也被截断。停止并保留中断证据后，压缩注入的 profiler 展示及重复明细，原始数值、单位、模块分组和宿主完整证据保留；增加静态上下文与工具历史空间检查。实际上下文估算从 22824 降到 15699 token；相同重放能完整保留 10194 字符的源码结果和初始 JSON。

修复相关单元/集成测试 `39 passed`，控制器闭环/集成测试 `13 passed`，两组有重叠。证据在 `var/monitoring/qwen35-27b-nf4-matrix-20261105-174604/context-replay-validation.json`、`context-fix-tests.xml`、`context-fix-e2e.xml`。修复后已恢复同一活动，保留冻结评测协议和初始稳定版本；第 1 轮中断没有候选性能结论。此处记录修复验收，不表示后续候选已提速或持续实验已完成。

随后复盘会话暴露指令冲突：通用规划提示和用户上下文仍要求提出修改，模型转向分析新优化，耗尽 2048 token 后未提交复盘，控制器按屏障停止。改为独立复盘提示与只含目标记录、协议和证据索引的上下文，移除当前引擎源码/观测及新候选指令；同一实际复盘上下文估算降到 628 token。仅输出 prose 或可识别协议错误时，原时间预算内最多追加一次模型提交，归属及证据检查保持生效。相关测试 `43 passed`，包括不能把中断实验归因为 `supported` 的反例，记录在同目录 `analysis-fix-tests.xml`。此项属于控制流程修复，不是模型引擎提速。

随后实际模型提交了有证据的 `inconclusive` 复盘，但因引用旧 `analysis.md` 的占位句而被误拒。占位检查现仅拒绝完整占位回复，允许实质分析引用该原句；复盘上下文给出目标目录和决策路径，`list_evidence(".")` 可列证据根目录，父目录越界仍被拦截。恢复稳定版时同时清空旧停止原因，避免 READY 状态显示上次错误。用原始模型提交、原始证据离线重放已成功，未修改活动数据库或编造分析。相关单元/集成 `46 passed`、闭环/集成 `13 passed`（两组有重叠），证据在同目录 `validation-replay.json`、`validation-fix-tests.xml`、`validation-fix-e2e.xml`；截至修复验收仍没有有效候选或正式 A/B 提速结论。

**同一活动的首个真实候选结果。** 第 3 次尝试由 27B Agent 自主修改默认注意力后端 SDPA→eager，提交前一次错误的运行前提被门槛拦截，模型自行修正后通过候选准入。加载后 `correctness.runtime.json` 确认实际后端为 eager，NF4/BF16 精度保持一致。正确性报告未通过：含数值差异超出冻结阈值，以及 8K 请求申请约 6 GiB 临时显存导致 OOM。系统自动以 `correctness_failed` 拒绝候选，未运行正式性能 A/B，恢复原 SDPA 稳定版本并进入模型复盘。此轮验证了候选错误可由系统闭环处理，不构成提速结果；测试标准未放宽。证据在该活动 `iterations/0003/` 的 `candidate-gate.json`、`change.patch`、`logs/correctness.runtime.json`、`correctness/report.json`、`decision.json` 和恢复日志。

**历史增长与复盘数据归属修复。** 第 4 轮初始静态输入 21117 token，仅余 3050，低于 4096 工具历史预留，预算检查阻止调用。加入按预算分层压缩：历史/上一轮详情变为证据索引，必要时源码改由工具读取，完整宿主证据保留。实际第 4 轮完整系统提示离线重放为 18276 token、余 5891，Qwen-Agent 截断器能同时保留用户上下文和最新源码结果。监控同时发现第 3 轮模型复盘误把父版本观测当候选性能；新上下文明示数据角色及 A/B 是否存在，工具和控制器拒绝常见无对照依据的量化比较，原始模型误写已在离线重放中被拒绝。原始提交保留为证据，须将该轮重新排队由模型复盘，不由宿主改写分析。单元/集成 `49 passed`、闭环/集成 `13 passed`（有重叠）；证据为同监控目录的 `memory-fix-tests.xml`、`memory-replay.json`、`memory-budget-replay.json`、`memory-fix-e2e.xml`。

复盘修复的进一步实测发现：模型虽声明无正式 A/B，仍把 `normal_request` 和 `timing` 分别贴为父版/候选。继续收紧候选数值标签校验，输入明确二者属于同一父版本的普通/插桩推理；失效旧复盘不再作为可信输入重复注入，原始提交在日志保留。实际错误提交重放被拒绝，相关单元/集成 `51 passed`、闭环/集成 `13 passed`（有重叠）；证据为 `attribution-replay.json`、`attribution-fix-tests.xml`，闭环回归另记录在 `attribution-fix-e2e.xml`。实际 27B 重新复盘后已明确普通/插桩两组数据均来自稳定父版本，候选无正式性能测量，10% 预测未验证，性能假设为 `inconclusive`；第 1–5 轮待分析队列清空并自动进入第 6 轮。该模型原始复盘备份在同监控目录 `corrected-analysis-0003.json`。

**规划回复结束但任务未完成的恢复。** 第 6、7 次尝试均只输出讨论，没有修改或提交，宿主随 `Assistant.run()` 迭代器结束直接记为 `agent_protocol_error`。修复为仅一次、有预算的规划继续：保留原始上下文与完整工具轨迹，剩余工具次数按前次已执行数扣除，截止时间不重置；模型必须自行修改并提交，或明确调用 `report_blocked`。回归验证真实改动仍需通过候选门槛，无改动不能通过，两次文字回复后停止继续，工具预算已耗尽时不再调用。相关单元/集成 `55 passed`、闭环/集成 `13 passed`（有重叠）；证据为同监控目录 `continuation-fix-tests.xml`、`continuation-fix-e2e.xml`。真实 27B 第 8 次尝试已触发继续，剩余约 394 秒、10 次工具调用；模型随后自行修改源码并检查语法。它的提交将计划字段错误地放在顶层，被 `missing_experiment_plan` 拒绝，最终耗尽原 900 秒预算并回退。因此已实测恢复工具动作，尚未获得有效候选或性能结果；证据为 `continuation-live-result.json`。本轮工具记录重放另确认原始用户上下文和最新工具结果保留，7 条较早返回被截断为省略标记，见 `iteration-0008-live-context-replay.json`。

**按会话声明提交结构。** 第 8 次尝试的模型提交把计划字段平铺为顶层参数；此前工具仅将 `hypothesis` 声明必填，运行时门槛却要求嵌套 `experiment_plan`，接口声明与执行要求不一致。现按规划/复盘/冒烟分别构造声明：规划显式要求计划对象和 v2 字段，待分析时声明分析/复盘字段必填；schema 独立复制，避免会话间污染。格式错误返回简短字段路径和修正说明，实际候选门槛保持生效。原始错误参数不改写重放，明确返回缺少 `experiment_plan`，见 `schema-replay.json`。单元/集成 `57 passed`、闭环/集成 `13 passed`（有重叠），证据为同监控目录 `schema-fix-tests.xml`、`schema-fix-e2e.xml`。旧活动已达到连续 3 次失败上限，保留该预算及失败记录，最终复盘后结束；后续活动使用相同评测配置重新验证。

**失败预算后的历史连续性。** 后续活动 `qwen35-27b-nf4-matrix-20261105-204724` 使用与旧活动相同的配置哈希 `aaf7caa0f5ee342d1473a91d96801e24a4b8ccacfd0aeec65e2abb274acdee9d`，旧活动已完成全部复盘并按 `max_failures` 结束。新活动证据内保留旧活动副本，`memory/archived-campaigns.json` 提供只读位置索引。回归验证归档分析可通过受限证据工具读取、不计入新轮次，且旧观测不能替代当前绑定观测；单元/集成 `58 passed`、闭环/集成 `13 passed`（有重叠）。证据在旧监控目录 `continuity-tests.xml`、`continuity-e2e.xml` 和新监控目录 `lineage.json`。真实新活动已完成自举正确性检查，14 项全部通过，覆盖 5–8192 输入 token；初始稳定制品为 `v-4094333fbbe12c8c3694`。第 1 次规划的实际上下文包含旧活动索引，静态估算 17577 token、剩余 6590，超过 4096 工具历史预留。随后该尝试因键盘中断记为 `interrupted_iteration`；本次文档对照时活动为 `STOPPED / keyboard_interrupt`、`proposal_attempts=1`、`completed_experiments=0`、`analysis_pending=true`。这证明自举和历史接续已运行，不构成有效候选、正式 A/B 或性能提升结论；恢复后应先完成该轮复盘。

**归档候选去重的实际缺口。** 新活动第 1 次规划没有读取归档，重复实施 SDPA→eager；语义指纹与旧活动第 3 次被正确性拒绝的候选完全相同，原新活动去重索引却为空。监控因此中断尝试，未手工修改候选。修复将同配置哈希、同环境画像、同稳定父版本且完成过评测的归档候选加入宿主去重，返回旧活动、轮次、决策和证据路径；模型收到有界候选结果索引。宿主扫描全部登记归档，模型仅接收最近 4 个归档位置和 6 条候选结果，二者的数量上限独立。真实候选离线重放已得到 `duplicate_change`，见新监控目录 `archive-dedup-replay.json`。相关单元/集成 `66 passed`、闭环/集成 `13 passed`（有重叠）；最后组合回归 `77 passed`，涵盖不同实现放行、条件变化/未完成尝试不拦截、证据路径限制，以及重建上下文后宿主去重实际生效、不增加当前历史计数，模型摘要截断也不会截断宿主去重。证据为 `archive-dedup-tests.xml`、`archive-dedup-e2e.xml`、`archive-dedup-final.xml`。此修复不产生性能收益结论。

## 0.1 三模块闭环验收（2026-11-05 较早阶段，历史）

本次修改覆盖有效候选门槛、可信性能观测、实验记忆闭环；复用并修改原有文档，不另建说明文档。

完整回归结果：`150 passed, 1 deselected`，耗时 165.94 秒；未选择真实目标测试，没有跳过本次离线小模型项。JUnit 保存在 `var/validation-experiment-loop/tests.xml`。另检查 103 个 Python 文件语法、32 个静态 JSON 和 29 条本地文档链接，全部通过；`git diff --check` 无空白错误。

| 范围 | 检查内容 |
| --- | --- |
| 有效候选 | 实际运行源码变化、Python AST/JSON 规范化、同父版本重复识别；缺少计划、绑定观测或历史对照不得进入正式实验；工具提交后控制器再次检查 |
| 可验证计划 | 瓶颈证据、原因、源码变化、指标/方向/预测幅度、验证方法、历史编号；接受候选与达到预测分别记录 |
| 可信计时 | 宿主墙钟和外部端到端计时；候选伪造 TTFT 不能替代宿主缺失值；诊断开销独立保存、不进入准入 |
| 性能剖析 | Toy 的 cProfile 路径；真实小型 GPT-2/Llama 各 FP32/BF16 的 torch.profiler CPU 热点、首个/后续 forward 分组、hook 清理和后续输出一致性 |
| 实验记忆 | 从 50 轮记录中检索早期分析；分页读取长日志尾部；近期/相关记录有界注入，完整去重及历史编号索引留在宿主、不进入模型提示 |
| 复盘约束 | 目标编号和证据归属校验；基础设施失败须标记证据不足；未达到预测幅度不能宣称预测成立；已完成复盘不会被下一次规划随意覆盖 |
| 完整编排 | CPU 接受/正确性拒绝/精度搜索/回退/恢复，Qwen-Agent 工具调用 worker，分析屏障失败停止推进，键盘中断清理及计数 |

复现命令及本次 JUnit：

```bash
VIBEFORGE_TEST_INFERENCE_PYTHON=/home/yxy/agent-to-inferrence/.venv/bin/python \
  .venv/bin/python -m pytest -m 'not target' -q \
  --junitxml=var/validation-experiment-loop/tests.xml
```

本次小模型推理解释器为 Python 3.11、PyTorch `2.7.1+cu128`、Transformers `5.20.2`；控制环境 Python 3.10、Qwen-Agent `0.0.34`。CPU 测试使用离线随机权重，仅验证软件行为，不代表 Qwen 模型质量或加速收益。

27B/4-bit 活动 `qwen35-27b-nf4-continuous-r4-20261105` 保持 `STOPPED / user_requested_pause`，未恢复迭代。此次未加载完整 27B 权重、未验收其 CUDA profiler 或新流程的实际优化收益。没有改写历史冻结发布和活动配置；使用新模板及观测配置需要新活动。

## 0.2 多样化系统增量验收（2026-11-04 历史，schema v2）

现行能力、字段和运行边界见 [09：多样化引擎系统](09-diverse-engine-system.md)。后续各节保留首版历史验收，旧测试数量和内存观测不代表当前实时状态。

| 本次增量 | 结果 |
| --- | --- |
| 完整工程回归 | `82 passed, 1 deselected`，约 98 秒；包含真实小型模型接口项，未选择 Qwen35B 目标测试 |
| 可组合配置 | 模型、设备、适配器、精度、需求、优化白名单独立配置；旧 dtype/quantization 字段归一化兼容 |
| 多格式画像 | 单文件/分片 safetensors、GGUF 头部、受限路径 manifest；目标精度参与内存估算与画像身份 |
| 精度搜索闭环 | 白名单内切换通过时提升；独立质量失败时恢复初始稳定版本；恢复上下文使用稳定版本精度 |
| 性能路径一致性 | 普通与固定长度生成都受差分检查，计时请求逐次检查长度与 token，错误快速路径被拒绝 |
| 个性化目标 | 吞吐最大化；请求 P95/RSS/启动/制品体积最小化；独立资源上限 |
| 构建及恢复 | 有界构建、制品冻结、越界写入拦截；提交前/后的真实进程崩溃恢复测试 |
| 真实 CPU 模型 | 离线随机小型 GPT-2、Llama，各 FP32/BF16 共四组合：引擎启动、生成、logits、固定长度及并发请求隔离通过 |
| 未验证范围 | 真实 Qwen35B 完整部署和 Agent、多种量化内核、CUDA/ROCm 完整模型优化、长上下文和多模态 |

新增真实小模型测试由本机已有推理解释器运行，无网络下载，不使用已有大模型权重。随机权重只验证软件接口，不衡量模型质量或自优化收益。

本次推理环境为 PyTorch `2.13.0+cu132`、Transformers `5.19.0`。Qwen35B 元数据预检重新读取了 1,045 个 tensor，`missing_packages=[]`；权重为 71,903,645,408 字节，BF16 需求估算仍为 84,098,387,984 字节（78.32 GiB），本次可用内存 31,699,218,432 字节（29.52 GiB），结果为 `insufficient_memory`。没有加载完整大模型权重或运行其 Agent。

复现：

```bash
# 框架工程测试；未设置推理解释器时，真实小模型接口项明确跳过
.venv/bin/python -m pytest -m 'not target' -q

# 将真实小模型接口项纳入完整工程检查
VIBEFORGE_TEST_INFERENCE_PYTHON=/path/to/transformers-env/bin/python \
  .venv/bin/python -m pytest -m 'not target' \
  --junitxml=var/validation-diversity/tests.xml -q
```

真实 CPU 单项 JUnit：`var/validation-diversity/transformers-cpu.xml`。
完整回归 JUnit：`var/validation-diversity/tests.xml`。另检查 93 个 Python 文件语法、35 个静态 JSON，以及 README/docs 中的 Markdown 文档链接；全部通过。

## 1. 首版结论（历史）

| 验收项 | 结果 |
| --- | --- |
| 项目核心模块 | 已实现可运行首版，无仅抛出 NotImplemented 的业务占位 |
| 工程测试 | 47 项通过，含 Qwen-Agent 模拟服务工具调用及控制器 worker 集成测试 |
| CPU 完整闭环 | 通过：`qwen-agent-toy-20261104` 两轮完成，第一轮接受，第二轮正确性拒绝并保持第一轮版本 |
| 故障恢复 | 测试通过：提升意图写入后控制程序退出，新进程恢复保留已提交版本 |
| Qwen-Agent 框架 | 通过：真实 `Assistant` 对接模拟 OpenAI SSE 服务，完成候选编辑、语法检查和提交 |
| Toy 引擎框架冒烟 | 通过：Qwen-Agent `Assistant` 经引擎 `/v1/chat/completions` 完成工具调用 |
| Qwen BF16 完整部署 | 未通过：配置现有推理环境后仍因内存不足阻塞，完整权重未加载 |
| 真实 Qwen Agent 与自优化 | 尚未验证；框架接入通过，真实 Qwen 权重推理受上述条件阻塞 |
| 可迁移优化范式 | 已有草案与证据管理机制，尚无经真实多配置实验验证的通用规则 |

工程通过与目标模型部署未通过分别保留结果，不合并为“系统全部通过”。

## 2. 模块与实现入口

| 模块 | 已实现职责 | 主要文件 |
| --- | --- | --- |
| CLI | 画像、启动、恢复、状态、停止、事件、报告、子进程入口 | `cli/main.py`、`campaign.py`、`status.py` |
| 控制 | 自举、有限迭代、固定预算、状态机、统计准入、恢复编排 | `controller/supervisor.py`、`state_machine.py`、`budgets.py`、`admission.py`、`recovery.py` |
| Agent | Qwen-Agent `Assistant` 本地模型配置、工具适配、候选门槛、有界历史检索、结构化实验计划与复盘 | `agent/runner.py`、`tools.py`、`context.py`、`candidate.py`、`memory.py`、`analysis.py` |
| 引擎 | Python 后端构建检查、HTTP 服务、OpenAI SSE 兼容接口、宿主计时与诊断剖析、进程所有权、写入限制 | `engine/backend.py`、`builder.py`、`server.py`、`client.py`、`observation.py`、`process.py`、`sandbox.py` |
| 画像 | 分片头部与索引检查、模型结构、设备/依赖探测、内存预检、资源采样 | `profiling/model.py`、`hardware.py`、`environment.py`、`collector.py`、`sampler.py` |
| 评测 | 重复参考与变体集合、有界抽样 logits、请求隔离、独立任务质量、实际吞吐、配对 bootstrap | `evaluation/reference.py`、`correctness.py`、`quality.py`、`performance.py`、`statistics.py` |
| 版本 | 隔离工作区、差异、内容哈希、冻结、持久化提升意图、提交和回退 | `versioning/` 全部业务模块 |
| 可观测 | SQLite outbox 导出、断尾修复、持续原始输出、工具审计、事件心跳、摘要 | `observability/events.py`、`logs.py`、`metrics.py`、`reports.py` |
| 知识 | 按轮抽取草案、校验状态和证据、按条件检索、索引 | `knowledge/extractor.py`、`validator.py`、`registry.py`、`retriever.py` |
| 存储 | 原子替换与 fsync、WAL 事务、锁、路径和标识符约束 | `storage/atomic.py`、`database.py`、`locks.py`、`paths.py` |
| 契约 | 配置验证、候选/画像/决策对象、事件时间标识 | `contracts/` 全部业务模块 |

文件路径均相对于 `src/vibeforge/`。配置、用例、引擎、提示词、报告和知识模板、运维脚本及服务模板已经提供；每个静态目录保留用途说明。

## 3. 工程测试证据

命令：

```bash
.venv/bin/python -m pytest -m 'not target' \
  --junitxml=var/campaigns/system-validation/tests.xml -q
```

覆盖内容：

| 类别 | 实际断言 |
| --- | --- |
| 正确性准入 | token/logit 错误先拒绝；不能靠性能收益掩盖错误 |
| 性能准入 | 明确收益接受、退化拒绝、置信区间不足不提升、无效样本拒绝 |
| 失败处理 | 语法错误、Agent 越界编辑、进程超时后恢复原稳定版本 |
| 存储 | 事务回滚、锁互斥、原子 JSON、路径穿越与符号链接限制 |
| 发布事务 | 幂等提交、意图取消、制品哈希损坏、提交前崩溃恢复 |
| 日志 | outbox 事件导出、重复导出与 JSONL 断尾恢复 |
| Agent | Qwen-Agent 框架实际解析并调用工具、候选编辑与提交；控制器启动独立 worker 和最终分析会话；工具越界审计；原生 Qwen 工具格式转换 |
| 写入限制 | 生成进程可写缓存，不能改控制状态；允许库访问 `/dev/null` |
| 画像与知识 | 分片缺失/头部越界检查、资源阻塞、配置校验、知识适用边界 |

工程测试命令 `.venv/bin/python -m pytest -m 'not target' -q` 结果为 `47 passed, 1 deselected`。JUnit 位于本机 `var/validation-qwen-agent-final/tests.xml`。真实目标命令启用 `VIBEFORGE_TARGET_TEST=1` 并配置现有推理环境后，目标测试有 1 项失败，原因为 `BLOCKED: insufficient_memory`；未进入引擎启动或完整模型加载，不按通过或跳过处理。

另已检查 79 个 Python 文件语法、20 个静态 JSON 文件、全部 shell 脚本语法、静态子目录说明和新增文档链接。活动退出后没有残留的引擎或 Agent 进程记录。

## 4. CPU 活动实测

活动：`qwen-agent-toy-20261104`，状态 `STOPPED`，完成 2 轮，`analysis_pending=false`，退出后 `active_ref=null`。控制器中的候选优化决策使用专门的 `scripted` 测试夹具；自举、候选激活和恢复检查中的 Assistant 冒烟通过 Qwen-Agent 完成。

| 轮次 | 修改 | 结果 |
| --- | --- | --- |
| 1 | 把每个输出 token 的重复求和换成等价闭式算式 | 正确性通过；按实际计时接受候选 |
| 2 | 故意改变输出数值 | 正确性拒绝；稳定指针保持第一轮版本 |

该次第一轮相对吞吐收益约 `gain=37.33`；95% bootstrap 区间约 `[34.55, 39.25]`。这是刻意设置冗余 CPU 运算的测试引擎，不能解释为 Qwen 加速。

活动摘要、各轮判定和完整事件流保存在本机 `var/validation-qwen-agent/campaigns/qwen-agent-toy-20261104/`。同样的两轮流程在最终活动 `var/validation-qwen-agent-final/campaigns/qwen-agent-final-20261104/` 再次通过。运行产物不提交到版本库。

## 5. Qwen 本机部署结果

### 5.1 硬件与模型

| 项目 | 采集结果 |
| --- | --- |
| 平台 | aarch64，NVIDIA Thor，统一内存，Linux 6.8.12-1021-tegra |
| 驱动 / 设备 | NVIDIA Thor，统一内存；驱动探测为 595.78 |
| 模型路径 | `/home/yxy/weights/Qwen3.6-35B-A3B`，通过 `VIBEFORGE_MODEL_PATH` 传入，不进入版本库 |
| 模型类 | 权重元数据声明 `Qwen3_5MoeForConditionalGeneration`；本次没有加载模型权重 |
| 模型结构 | 40 层混合注意力，256 专家、每 token 选择 8 个专家 |
| 分片 | 26 个 safetensors 分片，索引和头部检查通过 |
| 权重字节 | 71,903,645,408，约 66.96 GiB |
| BF16 需求估算 | 84,098,387,984 字节，约 78.32 GiB |
| 本次活动预检可用内存 | 39,984,177,152 字节，约 37.24 GiB |

本次没有重新计算全部约 67 GiB 权重的内容校验和；画像记录配置、tokenizer 等文件哈希及分片清单/大小/头部检查范围。此前下载校验材料不作为本次完整校验的替代。

### 5.2 预检与框架探针

- 本机默认 `runtime_python=python3` 缺少推理依赖；通过 `VIBEFORGE_RUNTIME_PYTHON` 指向现有推理环境后，预检的 `missing_packages=[]`。
- 本次真实活动可用内存约 37.24 GiB，BF16 估算约 78.32 GiB；预检因此在引擎子进程启动前停止。
- 使用现有推理环境、真实本地 tokenizer 和 `chat_template.jinja` 验证多轮工具历史：OpenAI 参数字符串转对象、原生工具渲染、代码缩进与末尾换行保留均通过；生成响应由测试数据代替，没有加载权重。
- Qwen-Agent 框架本身通过 `.venv` 中的本地模拟 SSE 服务和 Toy 后端完成集成测试；这不等同于真实模型推理。
- 独立于系统包的全新虚拟环境安装项目后，Qwen-Agent `Assistant` 与项目模块导入通过。

权重和预检运行数据留在本机忽略目录，不提交到版本库。

### 5.3 部署阻塞

活动 `qwen-agent-bf16-20261104` 的结果：

```text
state = BLOCKED
reasons = insufficient_memory
completed_iterations = 0
stable_ref = null
missing_packages = []
```

活动预检及画像位于本机 `var/validation-qwen-agent-final/`；真实目标测试也记录了同样的内存阻塞。预检未通过后未终止其他工作负载、未修改权重、未自动改变精度。

下一步是提供足够空闲内存与满足评测约束的设备时段，使用已配置的推理环境重新运行真实目标测试。若选择量化，应使用单独活动，重新建立正确性参考并实测该后端兼容性。

## 6. 首版实现边界（历史，现行边界见 09）

| 方面 | 边界 |
| --- | --- |
| 真实推理 | BF16 Qwen 权重加载、生成、聊天 Agent、多轮优化尚未得到本机实测结论 |
| 生成范围 | 提供 Python 后端编辑与启动；C++/CUDA/Triton 专用构建、编译产物管理需扩展 |
| 正确性 | 抽样 32 个 logits、短续写、固定 token 和串行请求隔离；未覆盖全 logits、逐层误差、长上下文、并发和多模态 |
| 参考可信度 | 首版从优化前冻结的 Transformers 后端取得参考；独立实现的第二参考后端尚未接入 |
| 性能 | 主判据为控制侧实际请求吞吐；TTFT/TPOT 来自后端时间戳；未接入逐算子 profiler、能耗归因和完整显存峰值测量 |
| 资源 | 仅支持一个控制器数据根目录内的设备互斥；跨项目全局调度由外部负责 |
| 隔离 | Landlock 保护文件写入；CUDA 和标准字符设备可访问；不隔离网络、信号、驱动漏洞或共享缓存干扰 |
| 依赖 | 记录解释器和版本，未把整套 Python/CUDA 环境打包成不可变镜像；恢复时检查画像漂移 |
| 恢复 | 已测进程中断和提交窗口；未实际断电、重启机器或安装 systemd 服务进行验收 |
| 知识 | 只生成待审草案；跨配置规则仍需重复实测与人工审查，未自动宣称发现通用范式 |

设计文档中的更大目标继续作为研究路线。后续优先补齐真实 Qwen 部署与 Agent 验收，再按真实瓶颈引入原生内核和更强正确性评测。
