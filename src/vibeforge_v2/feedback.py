"""Compact host-measured feedback, independent of the Agent's final prose."""
from vf_runtime.storage.atomic import read_json, atomic_write

from .performance_feedback import performance_attempts, performance_summary


def write_feedback(folder, decision):
    lines = ['# 本轮宿主实测反馈', '',
             f"HISTORICAL_CANDIDATE round={folder.name}; outcome={decision.get('outcome')}; "
             '此文件描述该轮候选，不是当前稳定引擎。当前身份只查运行根目录 current/manifest.json。',
             '', str(decision), '']
    harness_path = folder / 'agent/harness.json'
    if harness_path.exists():
        harness = read_json(harness_path)
        facts = harness.get('facts') or {}
        replaced = facts.get('replaced_forwards') or {}
        lines += [f'CPU 路径检查证据：{harness_path}',
                  str({key: harness.get(key) for key in ('status', 'passed', 'allow_gpu_evaluation')}),
                  'CPU 观测事实：' + str({key: facts.get(key) for key in (
                      'changed_lines_executed_on_request_path',
                      'unobserved_source_effect', 'gpu_only', 'cpu_config_overrides')}),
                  'CPU 替换函数：' + str({'count': len(replaced), 'first_four': dict(list(replaced.items())[:4])}),
                  'CPU 检查错误（完整见证据）：' + str([str(e)[:700] for e in (harness.get('errors') or [])[:2]]),
                  'CPU 检查说明（完整见证据）：' + str([str(w)[:800] for w in (harness.get('warnings') or [])[:3]])]
        if harness.get('status') == 'SKIP':
            lines.append('SKIP 不是通过。未观察到改动行执行，不等于证明 CUDA 分支无效；'
                         '须核对候选实现是否实际接入、GPU 实际路径及性能，不能将定义或写入配置当成激活。')
    lines += performance_summary(folder, decision) + ['']
    runtime_path = folder / 'logs/candidate.runtime.json'
    if runtime_path.exists():
        runtime = read_json(runtime_path).get('runtime') or {}
        stable_runtime_path = folder / 'logs/stable.runtime.json'
        if stable_runtime_path.exists():
            stable_runtime = read_json(stable_runtime_path).get('runtime') or {}
            lines += ['GPU 加载状态对照（稳定 → 候选；不等于算子执行或性能证明）：',
                      str({key: {'stable': stable_runtime.get(key), 'candidate': runtime.get(key)}
                           for key in ('attention_implementations', 'request_scheduler', 'module_classes')})]
        lines += [f'候选实际加载状态：{runtime_path}',
                  str({key: runtime.get(key) for key in ('request_scheduler', 'continuous_batching',
                       'batching_mode', 'attention_implementations', 'module_classes',
                       'optional_kernel_bindings', 'requested')}),
                  '以上来自宿主运行状态，不采信 Agent 自述；加载状态仍不等于算子执行或性能收益证明。']
    for name in ('correctness.json', 'quality/report.json', 'probe/probe.json'):
        path = folder / name
        if not path.exists():
            continue
        report = read_json(path)
        lines += [f'证据：{path}', f"passed={report.get('passed')}; errors={report.get('errors', [])}"]
        if name == 'probe/probe.json':
            lines.append('候选并发诊断：' + str(report.get('concurrency')))
            for survey in report.get('profiles', []):
                lines.append('候选实际并发/合批诊断：' + str(survey.get('concurrency')))
                for profile in survey.get('profiles', []):
                    timing = profile.get('normal_request') or {}
                    lines.append('候选普通请求：' + str({key: timing.get(key) for key in
                        ('backend_wall_seconds', 'input_tokens', 'output_tokens', 'batch_size', 'batching_mode')}))
        if name == 'correctness.json':
            lines += [str(row)[:1800] for row in report.get('checks', []) if not row.get('passed')][:12]
            lines += [f"failure_summary={report.get('failure_summary')}; fidelity_drift={report.get('fidelity_drift')}"]
        if name == 'quality/report.json':
            lines += [f"metrics={report.get('metrics')}", f"paired={report.get('paired')}"]
            losses = set((report.get('paired') or {}).get('losses', []))
            for run in report.get('runs', [report])[:1]:
                for row in run.get('cases', []):
                    if row.get('id') in losses:
                        lines.append('新增失败实际输出：' + str(row)[:1600])
            runs = report.get('runs', [report])
            rows = [row for run in runs for row in run.get('cases', [])]
            limits = [row for row in rows if 'reached_output_limit' in row]
            failures = [row for row in rows if 'failure_kind' in row]
            diagnostics = {'reached_output_limit': sum(bool(row['reached_output_limit']) for row in limits)
                           if limits else None,
                           'output_limit_observed_cases': len(limits),
                           'output_limit_unobserved_cases': len(rows) - len(limits),
                           'failed_exact_match': sum(row['failure_kind'] == 'exact_format_or_answer' for row in failures)
                           if failures else None,
                           'failure_kind_observed_cases': len(failures),
                           'failure_kind_unobserved_cases': len(rows) - len(failures),
                           'note': 'Counts cover observed cases only; missing diagnostics are not zero failures. '
                                   'Exact format or answer failures do not establish which of those two causes applies.'}
            lines.append('质量诊断（不改变评分规则）：' + str(diagnostics))
    attempts = performance_attempts(folder)
    if attempts:
        report = read_json(attempts[-1])
        lines += [f'正式性能：{attempts[-1]}', str(report.get('decision'))]
        lines.append('各场景 tokens/s、延迟、内部 TTFT、TPOT（描述均值，不替代配对置信判定）：' + str((report.get('comparison') or {}).get('profiles')))
        for label in ('baseline', 'candidate'):
            for row in report.get(label, []):
                lines.append(f"{label}: metrics={row.get('metrics')}; valid={row.get('valid')}; reasons={row.get('invalid_reasons', row.get('reasons'))}")
    diagnostic_path = folder / 'diagnostic-performance/report.json'
    if diagnostic_path.exists():
        diagnostic = read_json(diagnostic_path)
        lines += [f'探索性性能（质量未过关，禁止用于晋升）：{diagnostic_path}',
                  str({key: diagnostic.get(key) for key in
                       ('admission', 'valid', 'objective', 'gain', 'pairs', 'error', 'limitations')})]
        lines.append('各场景 tokens/s、延迟、内部 TTFT、TPOT：' + str((diagnostic.get('comparison') or {}).get('profiles')))
    atomic_write(folder / 'outcome.md', '\n'.join(lines) + '\n')
