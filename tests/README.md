# 项目自身的自动化测试

- 测试控制、恢复、版本管理、日志、Agent 协议和评测器自身的行为。
- `unit/`、`integration/`、`e2e/` 按测试范围划分；`fixtures/` 保存小型测试材料。
- 推理引擎的正式性能结果不存放于此；正式用例归属 `benchmarks/`。
- `test_precision_search.py` 验证白名单切换、独立质量失败回退和精度画像恢复；`test_evaluation_gates.py` 拦截固定长度性能路径作弊。
- `test_build_recipe.py` 验证构建产物冻结及写入限制。
- 设置 `VIBEFORGE_TEST_INFERENCE_PYTHON` 后，`test_transformers_cpu.py` 使用随机小型 GPT-2/Llama 权重验证 FP32/BF16 的真实 CPU 接口，不测试模型质量或 Agent 推理。
