import json

from vf_runtime.storage.atomic import read_json, write_json
from vf_runtime.versioning.artifacts import source_hash
from vibeforge_v2.checks import CandidateCheck
from vibeforge_v2.identity import current_banner, publish_current


def test_current_identity_switches_only_when_host_publishes_accepted_source(tmp_path):
    stable = tmp_path / 'initial/engine'
    candidate = tmp_path / 'rounds/0001/engine'
    for path in (stable, candidate):
        path.mkdir(parents=True)
        write_json(path / 'engine.json', {'backend': 'toy'})
        (path / 'engine.py').write_text('def generate(): return 1\n')
    state = {'stable': 'initial/engine', 'stable_hash': source_hash(stable)}
    text = publish_current(tmp_path, state, health={'runtime': {
        'attention_implementations': {'model': 'sdpa'}, 'request_scheduler': 'serialized_lock'}}, round_number=1)
    checker = CandidateCheck(candidate, tmp_path / 'checks', protected=[tmp_path / 'current/manifest.json'])
    write_json(candidate / 'engine.json', {'backend': 'toy', 'attn_implementation': 'flash_attention_2'})
    assert (tmp_path / 'current/engine').resolve() == stable
    assert 'sdpa' in current_banner(candidate) and 'UNVERIFIED_CANDIDATE' in current_banner(candidate)
    assert text.count('<CURRENT_STABLE_ENGINE>') == 1
    assert checker.inspect()['protected_changed_files'] == []
    # A false current-version claim in a modified manifest is caught independently.
    manifest = tmp_path / 'current/manifest.json'
    data = read_json(manifest)
    data['source'] = 'rounds/0001/engine'
    write_json(manifest, data)
    assert checker.inspect()['protected_changed_files'] == [str(manifest)]
    # Promotion is explicit, and old runtime is cleared until measured for that source.
    publish_current(tmp_path, {'stable': 'rounds/0001/engine', 'stable_hash': source_hash(candidate)})
    assert (tmp_path / 'current/engine').resolve() == candidate
    assert json.loads(manifest.read_text())['runtime'] is None


def test_checkpoint_replay_repairs_an_old_current_pointer(tmp_path):
    from vibeforge_v2.config import ROOT, load
    from vibeforge_v2.loop import Pipeline
    for name, value in [('initial/engine', 1), ('rounds/0001/engine', 2)]:
        path = tmp_path / name
        path.mkdir(parents=True)
        write_json(path / 'engine.json', {'backend': 'toy'})
        (path / 'engine.py').write_text(f'def generate(): return {value}\n')
    old = {'stable': 'initial/engine', 'stable_hash': source_hash(tmp_path / 'initial/engine')}
    new = {'stable': 'rounds/0001/engine', 'stable_hash': source_hash(tmp_path / 'rounds/0001/engine')}
    publish_current(tmp_path, old)
    write_json(tmp_path / 'state.json', new)
    pipeline = Pipeline(load(ROOT / 'configs/toy.json'), tmp_path)
    pipeline.state = read_json(tmp_path / 'state.json')
    pipeline.save_state(status='running')
    assert (tmp_path / 'current/engine').resolve() == tmp_path / new['stable']
    assert read_json(tmp_path / 'current/manifest.json')['source_hash'] == new['stable_hash']
