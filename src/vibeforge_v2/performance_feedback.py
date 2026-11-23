"""Explicit units and profile comparisons, separate from admission statistics."""
import math
from pathlib import Path

from vf_runtime.storage.atomic import read_json


DEFINITIONS = {
    'output_tokens_per_second': 'Actual output tokens / profile measurement wall seconds; includes concurrency and queue effects.',
    'request_output_tokens_per_second_mean': 'Mean of each request output tokens / end-to-end client latency; not aggregate throughput.',
    'decode_tokens_per_second': '1 / mean internal TPOT; excludes prefill and queue; not HTTP streaming speed.',
    'latency_mean_seconds': 'Mean end-to-end client request latency, seconds.',
    'ttft_seconds_mean': 'Mean internal first-token readiness time, seconds; not first HTTP byte latency.',
    'tpot_seconds_mean': 'Mean internal time per output token after the first, seconds/token.',
    'composite_throughput_score': 'Weighted geometric mean of profile output tokens/s; not pooled system throughput.',
    'composite_request_speed_score': 'Weighted geometric mean of profile mean per-request end-to-end output tokens/s; the Qwen3.8 admission objective.',
}


def performance_attempts(folder):
    """Numbered completed attempts; lexical order puts attempt-9 after attempt-10."""
    paths = (Path(folder) / 'performance').glob('attempt-*.json')
    return sorted((path for path in paths if path.stem[8:].isdigit()),
                  key=lambda path: int(path.stem[8:]))


def performance_summary(folder, decision):
    """Keep measured rejection evidence ahead of verbose runtime/quality output."""
    attempts = performance_attempts(folder)
    if not attempts:
        samples = Path(folder) / 'performance/samples.jsonl'
        if samples.exists() and samples.stat().st_size:
            return ['正式性能已有部分采样，但无完成的配对评测报告，不能推断收益。']
        return ['正式性能未执行，不能推断收益。']
    report = read_json(attempts[-1])
    measured_decision = report.get('decision') or {}
    objective = decision.get('objective') or measured_decision.get('objective') or 'throughput'
    metric = 'output_tokens_per_second' if objective == 'throughput' else objective
    lines = [f'正式性能证据：{attempts[-1]}',
             f'已完成采样；配对准入结论：{measured_decision.get("outcome")}/{measured_decision.get("reason")}。',
             f'以下为各场景 {metric} 的描述均值（稳定 → 候选），不是汇总收益或配对置信区间：']
    records = list(report.get('baseline', [])) + list(report.get('candidate', []))
    if not records or any(row.get('valid') is not True for row in records):
        lines.append('注意：样本有效性未全部通过或未提供；描述数字不能作为有效收益。')
    profiles = (report.get('comparison') or {}).get('profiles') or {}
    rejected_profile = str(decision.get('reason', '')).partition('profile_regression:')[2]
    names = sorted(profiles, key=lambda name: (name != rejected_profile, name))
    def number(value, percent=False):
        if type(value) not in (float, int) or not math.isfinite(value):
            return '未知'
        return f'{value:+.2%}' if percent else f'{value:.6g}'
    for name in names[:6]:
        row = profiles[name]
        lines.append(f'- {name}: {number((row.get("baseline") or {}).get(metric))} → '
                     f'{number((row.get("candidate") or {}).get(metric))}；'
                     f'相对变化 {number((row.get("relative_change") or {}).get(metric), True)}')
    if len(names) > 6:
        lines.append(f'另有 {len(names) - 6} 个场景，见上述证据文件。')
    if not names:
        lines.append('该报告未提供逐场景描述统计；见原始样本和准入结论。')
    return lines


def _mean(values):
    return sum(values) / len(values) if values and all(
        type(x) in (float, int) and math.isfinite(x) for x in values) else None


def _metrics(record):
    raw = record.get('metrics', {})
    tpot = raw.get('tpot_seconds_mean')
    requests = record.get('requests', [])
    rates = [row['tokens'] / row['latency_seconds'] for row in requests
             if type(row.get('tokens')) is int and type(row.get('latency_seconds')) in (float, int)
             and row['latency_seconds'] > 0]
    return {'output_tokens_per_second': record.get('throughput', raw.get('throughput')),
            'request_output_tokens_per_second_mean': raw.get('request_output_tokens_per_second_mean',
                _mean(rates) if len(rates) == len(requests) else None),
            'decode_tokens_per_second': 1 / tpot if type(tpot) in (float, int) and tpot > 0 else None,
            **{name: raw.get(name) for name in
               ('latency_mean_seconds', 'ttft_seconds_mean', 'tpot_seconds_mean')}}


def compare_profiles(baseline, candidate):
    """Descriptive means only; the original paired confidence tests decide admission."""
    names = sorted(set().union(*(row.get('profile_records', {}) for row in baseline + candidate)))
    if not names:
        names = ['default']
    result = {}
    for name in names:
        sides = []
        for records in (baseline, candidate):
            profiles = [row.get('profile_records', {}).get(name, {}) if name != 'default' else row
                        for row in records]
            metrics = [_metrics(row) for row in profiles]
            sides.append({key: _mean([row[key] for row in metrics]) for key in DEFINITIONS
                          if not key.startswith('composite_')})
        result[name] = {'baseline': sides[0], 'candidate': sides[1],
                       'relative_change': {key: (sides[1][key] / value - 1)
                        if type(value) in (float, int) and value > 0 and sides[1][key] is not None else None
                        for key, value in sides[0].items()}}
    return {'scope': 'Descriptive per-profile means; use paired admission decision for confidence and acceptance.',
            'metric_definitions': DEFINITIONS, 'profiles': result}
