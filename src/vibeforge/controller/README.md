# 独立控制程序

- 放置监督器、持久化状态机、设备租约、预算、超时、恢复与准入策略。
- 实现文件：`supervisor.py`（闭环）、`state_machine.py`（状态迁移）、`recovery.py`（恢复）、`admission.py`（准入）、`budgets.py`（停止条件）。
- 负责调度 Agent、引擎和评测；稳定指针的变更通过版本管理器执行。
