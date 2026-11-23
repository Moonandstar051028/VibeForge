"""A CPU request trace cannot disprove effects outside its observed paths."""
from pathlib import Path

import pytest

from vibeforge_v2 import harness
from vibeforge_v2.library import verified


@pytest.mark.parametrize('setup', [
    "    if config['device'].startswith('cuda'):\n        configure_gpu_backend()\n",
    "    configure_model_execution()\n",
    # Removed startup work is also absent from the candidate's request trace.
    '',
])
@pytest.mark.parametrize('math_error', [False, True])
def test_untraced_setup_requires_gpu_evidence_but_does_not_hide_math_errors(
        tmp_path, monkeypatch, setup, math_error):
    stable, candidate = tmp_path / 'stable', tmp_path / 'candidate'
    request = '\ndef generate():\n    return 1\n'
    for folder in (stable, candidate):
        folder.mkdir()
        (folder / 'engine.json').write_text('{"device":"cuda:0"}')
    (stable / 'engine.py').write_text('def load(config):\n    prepare_slow_execution()\n' + request)
    (candidate / 'engine.py').write_text('def load(config):\n' + (setup or '    pass\n') + request)

    def cpu_result(directory, *args, **kwargs):
        is_candidate = Path(directory) == candidate
        source = (Path(directory) / 'engine.py').read_text().splitlines()
        # The worker starts tracing only after load(), then calls generate/logits.
        executed = [['engine.py', i + 1] for i, line in enumerate(source) if 'return 1' in line]
        return {'ok': True, 'logits': [1., 3. if math_error and is_candidate else 2.],
                'decode_seconds_per_token': .01, 'parameter_bytes': 1,
                'generate': {'short': {'token_ids': [1]}}, 'replaced_forwards': {},
                'executed_lines': executed}

    monkeypatch.setattr(harness, 'build_reduced_model', lambda *args: tmp_path)
    monkeypatch.setattr(harness, 'run_engine', cpu_result)
    report = harness.run(candidate, stable=stable, model_path='m', cache=tmp_path / 'cache')
    assert 'no_effect' not in report['facts']
    assert not verified(report)
    if math_error:
        assert report['status'] == 'FAIL' and not report['allow_gpu_evaluation']
        assert any('logits differ' in message for message in report['errors'])
    else:
        assert report['status'] == 'SKIP' and report['allow_gpu_evaluation']
        assert report['facts']['unobserved_source_effect']
        assert not report['errors'] and not report['passed']
