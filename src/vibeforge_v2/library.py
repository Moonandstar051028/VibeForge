"""Same-run implementation evidence, with CPU checks and formal admission separated.

Keep each round's result even when replacement fingerprints overlap: integration,
configuration and host decisions can differ. Only formally accepted candidates
are recommended for reuse; rejected candidates remain failure evidence.
"""
import json
import shutil
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write, read_json

FP32_LIMIT = 2e-3


def _harness(run, folder):
    # A refreshed library copy (re-run with the current harness) takes precedence over the round's own record.
    for path in (Path(run) / 'library' / folder.name / 'harness.json', folder / 'agent/harness.json'):
        if path.exists():
            try:
                return read_json(path)
            except (OSError, ValueError):
                return None
    return None


def verified(report):
    facts = (report or {}).get('facts') or {}
    error = facts.get('float32_logit_relative_error')
    if any(info.get('fingerprint') and info.get('fingerprint') == info.get('original_fingerprint')
           for info in (facts.get('replaced_forwards') or {}).values()):
        return False
    return (isinstance(error, (int, float)) and error <= FP32_LIMIT and not facts.get('no_effect')
            and report.get('passed') is not False and not report.get('errors')
            and (facts.get('replaced_forwards') or facts.get('changed_lines_executed_on_request_path')))


def update_library(run, entries=None):
    """Copy changed files of verified rounds into run/library/NNNN/files; return rendered rows."""
    from .experience import collect
    run = Path(run)
    entries = {entry['round']: entry for entry in (entries if entries is not None else collect(run))}
    rows = []
    for folder in sorted((run / 'rounds').glob('[0-9]*')):
        report = _harness(run, folder)
        if not verified(report) or not (folder / 'engine').exists():
            continue
        target = run / 'library' / folder.name
        files = target / 'files'
        if not files.exists():
            stable = run / read_json(folder / 'parent.json')['source']
            for path in (folder / 'engine').rglob('*'):
                if not path.is_file() or '__pycache__' in path.parts:
                    continue
                relative = path.relative_to(folder / 'engine')
                original = stable / relative
                if original.exists() and original.read_bytes() == path.read_bytes():
                    continue
                destination = files / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            if (folder / 'changes.patch').exists():
                shutil.copy2(folder / 'changes.patch', target / 'changes.patch')
            if not (target / 'harness.json').exists():
                atomic_write(target / 'harness.json', json.dumps(report, ensure_ascii=False, indent=1))
        hardcoded = sorted(str(path.relative_to(files)) for path in files.rglob('*.py')
                           if str(folder / 'engine') in path.read_text(errors='replace')) if files.exists() else []
        entry = entries.get(folder.name, {})
        facts = report.get('facts') or {}
        replaced = ', '.join(f"{name} → {info.get('replacement')}"
                             for name, info in (facts.get('replaced_forwards') or {}).items()) or '请求路径上的源码改动'
        ratio = facts.get('cpu_decode_time_ratio')
        decision = read_json(folder / 'decision.json') if (folder / 'decision.json').exists() else {}
        gpu_gain = decision.get('gain', (decision.get('diagnostic_performance') or {}).get('gain'))
        row = {'round': folder.name, 'path': str(files), 'replaced': replaced,
               'flop_ratio': max((facts.get('module_flop_ratio') or {}).values(), default=None),
               'fingerprints': sorted(info.get('fingerprint') for info in (facts.get('replaced_forwards') or {}).values()
                                      if info.get('fingerprint')),
               'gpu_gain': gpu_gain,
               'decision_outcome': decision.get('outcome', entry.get('outcome')),
               'decision_reason': decision.get('reason', entry.get('reason')),
               'fp32_error': facts.get('float32_logit_relative_error'), 'cpu_ratio': ratio,
               'outcome': (f"{decision.get('outcome', entry.get('outcome'))}/"
                           f"{decision.get('reason', entry.get('reason'))}: {entry.get('detail', '')}")[:300],
               'hardcoded': hardcoded}
        atomic_write(target / 'info.md', render([row]))
        rows.append(row)
    return rows


def usable(row):
    return row.get('decision_outcome') == 'accept'


def render(rows, limit=5):
    accepted = [row for row in rows if usable(row)]
    unaccepted = [row for row in rows if not usable(row)]
    lines = ['CPU 数值等价检查不代表 GPU 提速，也不代表通过功能、质量和性能的正式准入。']
    if unaccepted:
        lines.append('未接受的历史实现（反例或待评测证据，不是推荐起点）：')
        for row in unaccepted[-limit:]:
            outcome = row.get('decision_outcome')
            conclusion = (f"{outcome}/{row.get('decision_reason')}" if outcome
                          else '尚无正式准入结论')
            gain = (f"；GPU 收益点估计 {row['gpu_gain']:.2%}" if isinstance(row.get('gpu_gain'), (int, float)) else '')
            lines.append(f"- 第 {row['round']} 轮：{row['replaced']}；宿主结论 {conclusion}{gain}。"
                         f"仅供核对失败原因，文件：{row['path']}。")
        lines.append('被拒绝的候选不得原样重复提交；再次尝试须针对原失败原因作实质修改。收益点估计为正也可能未达到准入门槛。')
    if not accepted:
        lines.append('（暂无通过正式准入、可作为起点的实现。）')
    else:
        lines.append('以下实现通过了宿主正式准入，可参考其已接受的改动：')
    for row in accepted[-limit:]:
        ratio = f"{row['cpu_ratio']:.2f}x" if isinstance(row.get('cpu_ratio'), (int, float)) else '未测'
        lines.append(f"- 第 {row['round']} 轮：{row['replaced']}；float32 误差 {row['fp32_error']:.2g}；CPU decode 为稳定版本的 "
                     f"{ratio}；宿主结论 {row['outcome']}\n  文件：{row['path']}（例如 `cp -r {row['path']}/. .` 后再修改）")
        if row.get('hardcoded'):
            lines.append(f"  注意：{', '.join(row['hardcoded'])} 写死了第 {row['round']} 轮目录的绝对路径，复制后必须改成相对当前文件的路径，"
                         '否则会加载旧目录的代码，宿主静态检查也会报错。')
    return '\n'.join(lines) + '\n'


def known_slow(run, fingerprints):
    """Library rows whose replacement source is identical and whose measured GPU change was negative."""
    wanted = {item for item in fingerprints if item}
    if not wanted:
        return []
    return [row for row in update_library(run)
            if wanted & set(row.get('fingerprints') or []) and isinstance(row.get('gpu_gain'), (int, float))
            and row['gpu_gain'] < 0]
