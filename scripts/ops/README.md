# 运维辅助脚本

- `run.sh`、`resume.sh`、`status.sh` 封装对应 CLI；停止活动使用 `vibeforge stop <id>`。
- 脚本调用控制程序，不能绕过版本管理器直接改写稳定指针。
- 系统服务定义位于 `deploy/systemd/`。
