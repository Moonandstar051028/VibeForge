"""Discover nearby experiment evidence without reusing baselines or gating candidates."""
import json
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write


def _read(path):
    return json.loads(path.read_text())


def _deployment(config):
    model = config['model']
    return ({key: model.get(key) for key in ('name', 'path', 'dtype', 'quantization', 'max_context')},
            config.get('precision'), config.get('hardware', {}).get('name_contains'),
            config.get('engine', {}).get('adapter'))


def write_related_history(config, run, output, limit=4):
    """Only host decisions count as evidence; matching configuration is not identity."""
    if not config.get('agent', {}).get('related_history', True):
        text = '# 相关历史实验索引\n\n本次运行已禁用跨运行历史加载；仅使用本次运行产生的反馈。\n'
        atomic_write(output, text)
        return text
    run, rows = Path(run).resolve(), []
    for prior in (Path(config['project']) / 'runs').glob('*'):
        if not prior.is_dir() or prior.resolve() == run:
            continue
        try:
            old = _read(prior / 'config.json')
            if _deployment(old) != _deployment(config):
                continue
            differences = []
            if old.get('workload', {}).get('profiles') != config.get('workload', {}).get('profiles'):
                differences.append('负载场景不同')
            old_eval, current_eval = old.get('evaluation', {}), config.get('evaluation', {})
            oq, cq = old_eval.get('quality') or {}, current_eval.get('quality') or {}
            if sorted(oq.get('file_hashes', {}).values()) != sorted(cq.get('file_hashes', {}).values()):
                differences.append('质量评测输入/脚本哈希不同')
            ignored = {'command', 'files', 'file_hashes'}
            if ({key: value for key, value in old_eval.items() if key != 'quality'} !=
                    {key: value for key, value in current_eval.items() if key != 'quality'} or
                    {key: value for key, value in oq.items() if key not in ignored} !=
                    {key: value for key, value in cq.items() if key not in ignored}):
                differences.append('评测协议不同')
            for folder in (prior / 'rounds').glob('[0-9]*'):
                try:
                    decision = _read(folder / 'decision.json')
                    outcome = folder / 'outcome.md'
                    if decision.get('outcome') not in {'accept', 'reject'} or not outcome.is_file():
                        continue
                    rows.append((outcome.stat().st_mtime_ns, folder, decision, differences))
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    lines = ['# 相关历史实验索引', '',
             '仅按模型、精度、配置的硬件名称和引擎适配器筛选，不代表物理 GPU、模型权重或依赖完全相同。',
             '只用于查找失败机制；不得复用基线、冒充本轮测量或据此自动拒绝候选。运行环境与实际 diff 仍须核对。', '']
    for _, folder, decision, differences in sorted(rows, key=lambda row: row[0], reverse=True)[:limit]:
        lines += [f"- {folder}: {decision['outcome']} / {decision.get('reason')}",
                  f"  已知条件差异：{'；'.join(differences) or '所比较的负载/评测配置一致；依赖和物理设备未核验'}。",
                  f'  宿主证据：{folder / "outcome.md"}；实际修改：{folder / "changes.patch"}；依赖：{folder / "environment.json"}']
    if not rows:
        lines.append('未发现匹配配置且有宿主结论的历史实验。')
    text = '\n'.join(lines) + '\n'
    atomic_write(output, text)
    return text
