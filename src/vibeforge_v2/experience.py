"""Host-derived experience ledger: what each round actually changed and how it actually failed.

The Agent previously received only the last three history rows plus one raw
outcome dump. It could not see that the same configuration keys had been tried
and rejected many times, and it re-derived (often wrongly) why. This ledger is
computed from files the host already owns -- parent/candidate engine.json,
changes.patch and the decision/evidence JSON -- never from Agent prose.
"""
import json
import math
from collections import OrderedDict
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write, read_json

from .performance_feedback import performance_attempts

MICROBATCH_KEYS = ('microbatch_max_size', 'microbatch_wait_ms')
SOURCE_SUFFIXES = {'.py', '.c', '.cc', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.rs'}


def effective_config(config):
    """Drop values the unchanged template provably never reads (inert microbatch settings)."""
    if not isinstance(config, dict):
        return config
    result = dict(config)
    if result.get('request_scheduler', 'serialized_lock') != 'microbatch':
        for key in MICROBATCH_KEYS:
            result.pop(key, None)
    if result.get('request_scheduler') == 'serialized_lock':
        result.pop('request_scheduler')
    return result


def config_changes(before, after):
    before, after = effective_config(before or {}), effective_config(after or {})
    changes = OrderedDict()
    for key in sorted(before.keys() | after.keys()):
        if before.get(key) != after.get(key):
            changes[key] = (before.get(key, '<absent>'), after.get(key, '<absent>'))
    return changes


def _patch_files(patch):
    files = OrderedDict()
    current = None
    for line in patch.splitlines():
        if line.startswith('+++ '):
            current = line[4:].split('/', 1)[-1].strip()
            files[current] = [0, 0]
        elif current and line.startswith('+') and not line.startswith('+++'):
            files[current][0] += 1
        elif current and line.startswith('-') and not line.startswith('---'):
            files[current][1] += 1
    return files


def _value(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return text if len(text) <= 40 else text[:37] + '...'


def signature(changes, source_files):
    parts = [f'{key}={_value(after)}' for key, (_, after) in changes.items()]
    parts += [f'源码:{name}' for name in source_files]
    return ', '.join(parts) or '无有效运行时改动'


def _failure_detail(folder, decision):
    reason = decision.get('reason', '')
    if reason == 'correctness_failed' and (folder / 'correctness.json').exists():
        report = read_json(folder / 'correctness.json')
        details = []
        for row in report.get('checks', []):
            if row.get('passed'):
                continue
            if row.get('case') == 'candidate_concurrent_isolation':
                diverged = []
                for item in row.get('mismatched_outputs', [])[:2]:
                    own, other = item.get('isolated_tokens') or [], item.get('concurrent_tokens') or []
                    index = next((i for i, (a, b) in enumerate(zip(own, other)) if a != b), None)
                    diverged.append(f"{item.get('batching_mode')} 批大小 {item.get('batch_size')}，第 {index} 个 token 起分叉")
                text = '并发/合批输出与单请求不一致（' + '；'.join(diverged) + '）'
                if 'near_tie_divergences' not in row:
                    text += '［旧规则：要求逐 token 完全相等；现行规则允许分叉点 logit 差在阈值内的近似平局，此结论可能不再成立］'
                details.append(text)
            elif row.get('error'):
                details.append(f"{row.get('case')}: {str(row['error'])[:120]}")
            else:
                details.append(f"{row.get('case')} 未通过")
        return '；'.join(details[:3]) or '正确性未通过'
    if reason == 'quality_failed' and (folder / 'quality/report.json').exists():
        report = read_json(folder / 'quality/report.json')
        paired = report.get('paired') or {}
        text = (f"质量门槛：{', '.join(report.get('errors', [])) or '未通过'}；配对失分 {paired.get('losses')}，"
                f"得分 {paired.get('gains')}")
        errors = report.get('errors', [])
        if ('regression_limits' not in report and errors
                and all(error.startswith('quality_regression:accuracy_') for error in errors)):
            text += '［旧规则：类别 5% 门槛下单题翻转即失败；现行规则每类至少容忍 1 题，此结论可能不再成立］'
        diagnostic = decision.get('diagnostic_performance') or {}
        if diagnostic.get('gain') is not None:
            text += f"；探索性性能 {diagnostic['gain']:.1%}"
        return text
    if decision.get('gain') is not None:
        text = f"正式收益 {decision['gain']:.2%}"
        if reason.startswith('profile_regression'):
            text += f"，场景退化：{reason.split(':', 1)[-1]}"
        return text
    if reason.startswith('profile_regression'):
        profile = reason.split(':', 1)[-1]
        text = f'单场景退化超过阈值：{profile}'
        attempts = performance_attempts(folder)
        if attempts:
            report = read_json(attempts[-1])
            objective = decision.get('objective') or (report.get('decision') or {}).get('objective') or 'throughput'
            metric = 'output_tokens_per_second' if objective == 'throughput' else objective
            row = ((report.get('comparison') or {}).get('profiles') or {}).get(profile) or {}
            values = [(row.get(side) or {}).get(metric) for side in ('baseline', 'candidate', 'relative_change')]
            if all(type(value) in (float, int) and math.isfinite(value) for value in values):
                baseline, candidate, change = values
                text += (f'；{metric} 稳定 {baseline:.6g} → 候选 {candidate:.6g}，'
                         f'{change:+.2%}（描述均值，非汇总收益）')
            text += f'；证据 performance/{attempts[-1].name}'
        return text
    if reason == 'harness_failed':
        return 'CPU 等价性校验未通过：' + str(decision.get('error', ''))[:300]
    if decision.get('error'):
        error = str(decision['error']).splitlines()[0]
        if 'repeated the same command' in error:
            return 'Agent 陷入重复：同一命令得到同一输出达到上限，宿主终止本轮（' + error[:160] + '）'
        if 'same incomplete candidate' in error:
            return 'Agent 两次结束时都没有可评测的有效改动（静态检查未过或与已拒绝候选相同）'
        return error[:160]
    return reason


def round_entry(run, folder):
    decision = read_json(folder / 'decision.json')
    parent = None
    try:
        parent_source = read_json(folder / 'parent.json').get('source')
        parent = read_json(Path(run) / parent_source / 'engine.json') if parent_source else None
    except (OSError, ValueError):
        parent = None
    candidate = None
    try:
        candidate = read_json(folder / 'engine/engine.json')
    except (OSError, ValueError):
        candidate = None
    changes = config_changes(parent, candidate) if parent is not None and candidate is not None else OrderedDict()
    patch = folder / 'changes.patch'
    files = _patch_files(patch.read_text(errors='replace')) if patch.exists() else OrderedDict()
    source_files = [name for name in files if Path(name).suffix in SOURCE_SUFFIXES]
    return {'round': folder.name, 'outcome': decision.get('outcome'), 'reason': decision.get('reason'),
            'stage': decision.get('stage'), 'gain': decision.get('gain'),
            'config_changes': {key: list(value) for key, value in changes.items()},
            'source_files': {name: files[name] for name in source_files},
            'signature': signature(changes, source_files),
            'detail': _failure_detail(folder, decision)}


def collect(run):
    entries = []
    for folder in sorted((Path(run) / 'rounds').glob('[0-9]*')):
        if not (folder / 'decision.json').exists():
            continue
        try:
            entries.append(round_entry(run, folder))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def _unstable_quality_items(entries, run):
    counts = {}
    for entry in entries:
        path = Path(run) / 'rounds' / entry['round'] / 'quality/report.json'
        if not path.exists():
            continue
        try:
            paired = read_json(path).get('paired') or {}
        except (OSError, ValueError):
            continue
        for item in (paired.get('losses') or []) + (paired.get('gains') or []):
            counts[item] = counts.get(item, 0) + 1
    return sorted(item for item, count in counts.items() if count >= 2)


def render(entries, run, *, compact=False):
    lines = ['# 经验台账（宿主根据实际 diff 与判定生成，不采信 Agent 自述）', '']
    if not entries:
        return '\n'.join(lines + ['尚无已结束轮次。', ''])
    total = len(entries)
    with_source = sum(bool(entry['source_files']) for entry in entries)
    config_only = sum(bool(entry['config_changes']) and not entry['source_files'] for entry in entries)
    no_change = sum(not entry['config_changes'] and not entry['source_files'] for entry in entries)
    accepted = sum(entry['outcome'] == 'accept' for entry in entries)
    lines.append(f'已结束 {total} 轮：接受 {accepted}；只改 engine.json 配置 {config_only} 轮；修改源码 {with_source} 轮；'
                 f'没有有效运行时改动 {no_change} 轮。')
    lines.append('')
    groups = OrderedDict()
    for entry in entries:
        for key, (_, after) in entry['config_changes'].items():
            groups.setdefault(f'{key}={_value(after)}', []).append(entry)
        if entry['source_files']:
            groups.setdefault('源码改动:' + ','.join(entry['source_files']), []).append(entry)
    if groups:
        lines += ['## 按单项改动汇总（同一项出现在多轮时，前提未变就不要再提交）', '',
                  '| 改动 | 出现轮次 | 实际结论 |', '| --- | --- | --- |']
        for key, rows in sorted(groups.items(), key=lambda item: -len(item[1])):
            reasons = OrderedDict()
            for row in rows:
                reasons[row['reason']] = reasons.get(row['reason'], 0) + 1
            best = max((row['gain'] for row in rows if row.get('gain') is not None), default=None)
            summary = '；'.join(f'{reason}×{count}' for reason, count in reasons.items())
            if best is not None:
                summary += f'；最好正式收益 {best:.2%}'
            lines.append(f"| {key} | {', '.join(row['round'] for row in rows)} | {summary} |")
        lines.append('')
    unstable = _unstable_quality_items(entries, run)
    if unstable:
        lines += [f'质量题中在多轮里来回翻转的题（对数值扰动敏感，可作为近似平局题看待）：{", ".join(unstable)}', '']
    shown = entries[-8:] if compact else entries
    lines += ['## 逐轮记录' + ('（最近 8 轮；完整见运行根目录 experience.md）' if compact else ''), '',
              '| 轮次 | 结果 | 改动签名 | 失败阶段与实际原因 |', '| --- | --- | --- | --- |']
    for entry in shown:
        lines.append(f"| {entry['round']} | {entry['outcome']}/{entry['reason']} | {entry['signature']} | "
                     f"{entry.get('stage') or ''}: {entry['detail']} |")
    lines.append('')
    return '\n'.join(lines)


def write_experience(run):
    entries = collect(run)
    text = render(entries, run)
    atomic_write(Path(run) / 'experience.md', text)
    atomic_write(Path(run) / 'experience.json', json.dumps(entries, ensure_ascii=False, indent=1))
    return entries


def matching_rounds(entries, changes):
    """Earlier rounds whose effective config change includes any of the same key=value pairs."""
    wanted = {(key, json.dumps(after, sort_keys=True)) for key, (_, after) in changes.items()}
    matches = []
    for entry in entries:
        seen = {(key, json.dumps(after, sort_keys=True)) for key, (_, after) in entry['config_changes'].items()}
        overlap = wanted & seen
        if overlap:
            matches.append((entry, sorted(key for key, _ in overlap)))
    return matches
