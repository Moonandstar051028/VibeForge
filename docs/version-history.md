# 版本来源与 Git 分支

记录日期：2026-11-23。

## 本次保存的内容

`Vibeforge-v2` 原为独立本地目录，此次将其现有源码、配置、测试和文档作为一个可追溯快照，放入原 VibeForge GitHub 仓库的 `vibeforge-v2` 分支。仓库根目录即 v2 项目根目录，不再嵌套一个 v2 子目录。

远程仓库：`https://github.com/Moonandstar051028/VibeForge.git`。

该快照的 Git 父提交采用本次核对到的原仓库 `main` 提交：

```text
6b0b3b96233177b1d00de9314ef959e9003ce3d1
Update VibeForge v2 optimization workflows and validation
```

这是可核验的已发布版本参照，**不声称它是 v2 当初精确的开发分叉点**。本次导入前 v2 没有独立 Git 提交，无法补造此前逐次修改的历史。后续修改应在本分支正常提交，以积累真实历史。

原本地 `/home/yxy/VibeForge` 有尚未提交的修改，本次不将其提交、切分支或重置，也不把这些未提交内容作为新的祖先提交。Git 父链保留的是原仓库已提交的历史。

## 文件级来源

根目录 `vendor-manifest.json` 保存从 VibeForge 引入的运行时文件对应路径、原始 SHA-256、v2 文件 SHA-256 及部分有意修改记录。它提供文件级来源线索，与 Git 父链互补；不应把文件哈希记录误当成完整的历史提交链。

## 计划与实现的区分

`docs/round-optimization-plan.md` 记录 2026-11-23 讨论的每轮内部优化方案。状态为待实施；本次归档没有实施其中的 Agent 分阶段、假设状态或独立复核机制。

实验输出、候选工作目录、Python 环境与缓存按 `.gitignore` 排除。模型权重和项目外论文不属于此源码快照。

## 快照验证

在屏蔽 CUDA 设备的环境中执行全量测试，结果为 `284 passed`。另运行 `scripts/demo_cpu.py` 的三个脚本化 CPU 轮次，依次验证接受、正确性拒绝和语法错误拒绝。这些检查不代表真实本地模型的自主优化或 GPU 性能收益。
