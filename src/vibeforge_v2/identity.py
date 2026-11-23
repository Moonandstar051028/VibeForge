"""Host-owned, unique pointer to the accepted engine; candidates never publish it."""
import json
import os
from pathlib import Path

from vf_runtime.storage.atomic import atomic_write, write_json
from vf_runtime.versioning.artifacts import source_hash


RUNTIME_BRIEF = ('model_class', 'execution_path', 'request_scheduler', 'continuous_batching', 'batching_mode',
                 'attention_implementations', 'packages', 'model_dtype', 'model_device', 'requested', 'limits')


def publish_current(run, state, *, health=None, round_number=None):
    run = Path(run)
    source = run / state['stable']
    if source_hash(source) != state['stable_hash']:
        raise ValueError('Cannot publish a current engine with a mismatched source hash')
    current = run / 'current'
    current.mkdir(exist_ok=True)
    pending = current / '.engine-next'
    pending.unlink(missing_ok=True)
    pending.symlink_to(Path('..') / state['stable'], target_is_directory=True)
    os.replace(pending, current / 'engine')
    manifest = {'role': 'CURRENT_STABLE_ENGINE', 'source': state['stable'],
                'source_hash': state['stable_hash'], 'engine_directory': str(current / 'engine'),
                'policy': 'Only the host updates this pointer after acceptance. Candidate edits and rejection do not update it.',
                'runtime_observed_round': round_number if health is not None else None,
                'runtime': health.get('runtime') if health is not None else None}
    write_json(current / 'manifest.json', manifest)
    config = (source / 'engine.json').read_text()
    # The manifest file keeps the full runtime; the prompt copy keeps the fields that decide behaviour.
    brief = {key: value for key, value in manifest.items() if key != 'runtime'}
    runtime = manifest.get('runtime') or {}
    if runtime:
        brief['runtime'] = {key: runtime.get(key) for key in RUNTIME_BRIEF if key in runtime}
        brief['runtime_full'] = str(current / 'manifest.json')
    text = ('<CURRENT_STABLE_ENGINE>\n' + json.dumps(brief, ensure_ascii=False, indent=2)
            + '\n当前稳定配置 engine.json：\n```json\n' + config + '\n```\n'
            + '完整源码通过 current/engine/engine.py 读取；候选文件在 rounds/NNNN/engine，未验证且不是当前版本。\n'
            + '</CURRENT_STABLE_ENGINE>\n')
    atomic_write(current / 'README.md', text)
    return text


def current_banner(workspace):
    """Reinforce identity after tools, including after official context folding."""
    workspace = Path(workspace).resolve()
    if len(workspace.parents) < 3:
        return ''
    path = workspace.parents[2] / 'current/manifest.json'
    if not path.exists():
        return ''
    manifest = json.loads(path.read_text())
    runtime = manifest.get('runtime') or {}
    return ('\nHOST CURRENT_STABLE_ENGINE: ' + str(path) + '\n'
            + f"source={manifest['source']}; source_hash={manifest['source_hash']}; "
            + f"attention={runtime.get('attention_implementations')}; scheduler={runtime.get('request_scheduler')}\n"
            + 'The working directory is an UNVERIFIED_CANDIDATE. Historical candidate runtime is not current stable runtime.\n')
