"""Plain source snapshots, diffs and a readable history; no database or proposal schema."""

import difflib
import shutil
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write, read_json
from vf_runtime.versioning.artifacts import inventory, source_ignore

from .performance_feedback import performance_attempts


def copy_engine(source, destination):
    shutil.copytree(source, destination, ignore=source_ignore(source))


def write_changes(parent, candidate, output):
    before, after = inventory(parent), inventory(candidate)
    parts = []
    changed = sorted(name for name in before.keys() | after.keys() if before.get(name) != after.get(name))
    for name in changed:
        a, b = Path(parent) / name, Path(candidate) / name
        try:
            parts.extend(difflib.unified_diff(a.read_text().splitlines(True) if a.exists() else [],
                                             b.read_text().splitlines(True) if b.exists() else [],
                                             fromfile='parent/' + name, tofile='candidate/' + name))
        except UnicodeError:
            parts.append(f'Binary file changed: {name}\n')
    atomic_write(output, ''.join(parts))
    return changed


def write_history(run):
    lines = ['# 运行历史', '', '| 轮次 | 结果 | 原因 | 正式测得相对父版本收益 |', '| --- | --- | --- | --- |']
    for folder in sorted((Path(run) / 'rounds').glob('*')):
        path = folder / 'decision.json'
        if not path.exists():
            continue
        row = read_json(path)
        gain = row.get('gain')
        if gain is not None:
            gain_text = f'{gain:.2%}'
        elif performance_attempts(folder):
            gain_text = '已采样，无汇总收益（各场景见 outcome.md）'
        else:
            samples = folder / 'performance/samples.jsonl'
            gain_text = ('正式采样未完成，无汇总收益' if samples.exists() and samples.stat().st_size
                         else '未测量')
        lines.append(f"| {folder.name} | {row['outcome']} | {row['reason']} | "
                     + gain_text + ' |')
    lines.extend(['', '逐轮收益不可相加；快速探测不是正式收益。详细错误、分析与样本在对应轮次目录。', ''])
    atomic_write(Path(run) / 'history.md', '\n'.join(lines))

