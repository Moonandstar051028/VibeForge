"""Regression tests for the 2026-11-17 analysis-loop fixes (history, digest, gates, context budget)."""
import json
import sys
import pytest
from pathlib import Path

from vf_runtime.storage.atomic import read_json, write_json
from vibeforge_v2.checks import CandidateCheck


def test_batched_divergence_at_near_tie_passes_but_contamination_fails():
    from vibeforge_v2.correctness import evaluate_functional
    isolated = [5, 6, 7, 8]

    class Client:
        def __init__(self, gap):
            self.gap, self.concurrent = gap, False
        def generate(self, prompt, max_new_tokens, **kwargs):
            tokens = [5, 6, 9, 9] if self.concurrent and prompt == 'p' else isolated
            return {'text': 'x', 'input_ids': [1, 2], 'token_ids': tokens,
                    'batch_size': 2 if self.concurrent else 1}
        def logits(self, input_ids, indices):
            # Only the divergence query asks for exactly two indices.
            return {'indices': indices, 'values': [3.0, 3.0 - self.gap][:len(indices)]}

    reference = {'cases': [{'id': 'c', 'prompt': 'p', 'max_new_tokens': 4, 'input_ids': [1, 2],
                            'token_ids': isolated, 'logits': {'indices': [0], 'values': [3.0]},
                            'continuation_logits': {'indices': [0], 'values': [3.0]}}]}
    protocol = {'concurrency': 2, 'atol': 10, 'rtol': 10, 'batch_divergence_max_logit_gap': 0.5}
    for gap, expected in ((0.1, True), (4.0, False)):
        client = Client(gap)
        original = client.generate
        calls = {'n': 0}
        def generate(prompt, max_new_tokens, _original=original, _client=client, **kwargs):
            calls['n'] += 1
            _client.concurrent = calls['n'] > 3  # first, sentinel, repeat are sequential
            return _original(prompt, max_new_tokens, **kwargs)
        client.generate = generate
        report = evaluate_functional(client, reference, protocol)
        check = next(row for row in report['checks'] if row['case'] == 'candidate_concurrent_isolation')
        assert check['passed'] is expected, check
        row = check['mismatched_outputs'][0]
        assert row['divergence_index'] == 2 and row['near_tie'] is expected


def test_unbatched_concurrent_divergence_still_requires_exact_match():
    from vibeforge_v2.correctness import batch_divergence
    row = batch_divergence(None, {'token_ids': [1, 2], 'input_ids': [0]}, {'token_ids': [1, 3], 'batch_size': 1}, 0.5)
    assert row['near_tie'] is False


def _quality_run(tmp_path, cases, baseline=None):
    from vf_runtime.evaluation.quality import run_quality
    script = tmp_path / 'fixture_quality.py'
    script.write_text('import json,sys\n'
                      'cases=json.load(open(sys.argv[2]))\n'
                      'cats={}\n'
                      'for c in cases: cats.setdefault(c["category"],[]).append(c["passed"])\n'
                      'metrics={"accuracy":sum(c["passed"] for c in cases)/len(cases)}\n'
                      'metrics.update({"accuracy_"+k:sum(v)/len(v) for k,v in cats.items()})\n'
                      'json.dump({"passed":True,"metrics":metrics,"cases":cases},open(sys.argv[1],"w"))\n')
    data = tmp_path / f'cases-{len(list(tmp_path.iterdir()))}.json'
    data.write_text(json.dumps(cases))
    settings = {'command': [sys.executable, str(script), '{output}', str(data)], 'files': [], 'file_hashes': {},
                'timeout_seconds': 30, 'min_scores': {},
                'max_regressions': {'accuracy': .05, 'accuracy_arithmetic': .05},
                'paired': {'min_cases': 1, 'max_loss_rate': .05, 'decision_mode': 'observed_loss'},
                'category_item_tolerance': 1}
    class Client:
        url = 'http://127.0.0.1:1'
    return run_quality(Client(), settings, tmp_path / f'q-{data.stem}',
                       {'project': str(tmp_path), 'python': sys.executable}, baseline)


def test_single_item_flip_in_small_category_is_within_granularity(tmp_path):
    good = [{'id': f'a{i}', 'category': 'arithmetic', 'passed': True} for i in range(12)]
    good += [{'id': f'r{i}', 'category': 'reading', 'passed': True} for i in range(38)]
    baseline = _quality_run(tmp_path, good)
    one_flip = [dict(row, passed=row['id'] != 'a0') for row in good]
    report = _quality_run(tmp_path, one_flip, baseline)
    assert report['passed'], report['errors']
    assert abs(report['regression_limits']['accuracy_arithmetic']['allowed_drop'] - 1 / 12) < 1e-9
    two_flips = [dict(row, passed=row['id'] not in ('a0', 'a1')) for row in good]
    report = _quality_run(tmp_path, two_flips, baseline)
    assert 'quality_regression:accuracy_arithmetic' in report['errors']


def test_bottleneck_digest_derives_ratios_from_saved_measurements():
    from vibeforge_v2.bottleneck import digest
    record = {'profile_id': 'agent_2k', 'output_tokens': 16,
              'normal_request': {'input_tokens': 1893, 'tpot_seconds': .1155, 'ttft_seconds': 1.36, 'output_tokens': 16},
              'phases': [{'name': 'vf_runtime.decode', 'cpu_total_seconds': 2.5, 'device_total_seconds': .52},
                         {'name': 'vf_runtime.module.mlp', 'cpu_total_seconds': 3.5},
                         {'name': 'vf_runtime.module.full_attention', 'cpu_total_seconds': .5}],
              'diagnostics': {'kernel_launches_per_output_token': 13730},
              'measurements': {'operator.cudaStreamSynchronize.calls': {'value': 23446}}}
    model = {'text_config': {'hidden_size': 2048, 'num_hidden_layers': 48, 'num_attention_heads': 32,
                             'num_key_value_heads': 4, 'head_dim': 128, 'vocab_size': 151936, 'num_experts': 128,
                             'num_experts_per_tok': 8, 'moe_intermediate_size': 768}}
    config = {'hardware': {'name_contains': 'A800', 'description': 'A800 80GB PCIe'}, 'precision': {'weight_dtype': 'bfloat16'}}
    text, facts = digest({'profiles': [record]}, {'model': model}, config)
    row = facts['profiles']['agent_2k']
    assert abs(row['decode_gpu_busy_fraction'] - .208) < .01 and abs(row['mlp_to_attention_cpu_ratio'] - 7) < 1e-9
    assert 2.5e-3 < facts['bandwidth_floor_tpot_seconds'] < 4e-3
    assert 'host/launch-bound' in text and '128 个专家' in text


def _round(run, number, parent_config, candidate_config, decision, extra=None):
    folder = run / 'rounds' / f'{number:04d}'
    write_json(folder / 'parent.json', {'source': 'initial/engine'})
    write_json(folder / 'engine/engine.json', candidate_config)
    write_json(folder / 'decision.json', {'round': number, **decision})
    (folder / 'changes.patch').write_text('--- parent/engine.json\n+++ candidate/engine.json\n+x\n')
    for name, value in (extra or {}).items():
        write_json(folder / name, value)
    return folder


def test_experience_ledger_groups_actual_changes_and_checker_reports_history(tmp_path):
    from vibeforge_v2.experience import collect, render
    run = tmp_path / 'run'
    stable = {'backend': 'toy', 'microbatch_max_size': 4}
    write_json(run / 'initial/engine/engine.json', stable)
    _round(run, 1, stable, {**stable, 'attn_implementation': 'flash_attention_2'},
           {'outcome': 'reject', 'reason': 'quality_failed', 'stage': 'quality'},
           {'quality/report.json': {'errors': ['quality_regression:accuracy_arithmetic'],
                                    'paired': {'losses': ['expression_0'], 'gains': []}}})
    _round(run, 2, stable, {'backend': 'toy'},  # removal of an unread key is not a change
           {'outcome': 'reject', 'reason': 'execution_failed', 'stage': 'agent',
            'error': 'RuntimeError: Agent ended again with the same incomplete candidate; x'})
    entries = collect(run)
    assert entries[0]['signature'] == 'attn_implementation=flash_attention_2'
    assert 'expression_0' in entries[0]['detail']
    assert entries[1]['signature'] == '无有效运行时改动'
    text = render(entries, run)
    assert '| attn_implementation=flash_attention_2 | 0001 | quality_failed×1 |' in text
    workspace = run / 'rounds/0003/engine'
    write_json(workspace / 'engine.json', stable)
    (workspace / 'engine.py').write_text('def generate(): return 1\n')
    checker = CandidateCheck(workspace, tmp_path / 'checks', history=entries)
    write_json(workspace / 'engine.json', {**stable, 'attn_implementation': 'flash_attention_2'})
    report = checker.inspect()
    assert any(w.startswith('HISTORY: attn_implementation was already submitted in round 0001') for w in report['warnings'])
    assert 'NOTE: HISTORY' in CandidateCheck.describe(report)


def test_long_shell_output_is_clipped_with_log_path_and_banner_only_once(tmp_path):
    from vibeforge_v2.agent import Shell
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    tool = Shell(workspace, tmp_path / 'agent', output_chars=300)
    result = tool.call(json.dumps({'command': "python3 -c \"print('A'*1000 + 'ZZZ')\""}))
    assert 'ZZZ' in result and 'characters omitted' in result and 'shell-001.log' in result
    assert len(result) < 900
    assert 'A' * 1000 in (tmp_path / 'agent/shell-001.log').read_text()
    assert tool.calls[0]['output'].count('A') == 1000  # the trace keeps everything


def test_static_check_catches_rewrite_that_drops_helpers_and_global_imports(tmp_path):
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    (workspace / 'engine.json').write_text('{"backend":"toy"}')
    (workspace / 'engine.py').write_text('import json\nTORCH = None\n\ndef load(config):\n    import torch\n'
                                         '    return torch\n\ndef parse_tool_calls(text):\n    return json.loads(text)\n\n'
                                         'def generate(prompt):\n    return TORCH\n')
    checker = CandidateCheck(workspace, tmp_path / 'checks')
    # A from-memory rewrite: helper removed, and a function-local import used from another function.
    (workspace / 'engine.py').write_text('def load(config):\n    import torch\n    return torch\n\n'
                                         'def generate(prompt):\n    with torch.inference_mode():\n        return 1\n')
    report = checker.inspect()
    assert not report['ready_for_host_evaluation']
    assert any('no longer defines parse_tool_calls' in error for error in report['errors'])
    assert any('torch (in generate' in error for error in report['errors'])


def _engine_report(logits, tokens=(1, 2), seconds=.01, memory=100, ok=True, error=None):
    if not ok:
        return {'ok': False, 'stage': 'generate()', 'error': error, 'traceback': error}
    return {'ok': True, 'logits': list(logits), 'decode_seconds_per_token': seconds, 'parameter_bytes': memory,
            'generate': {'short': {'token_ids': list(tokens)}}}


def test_harness_verdicts_separate_math_errors_runtime_errors_gpu_only_and_slowdowns():
    from vibeforge_v2.harness import compare
    stable = {'float32': _engine_report([10., -3., 2.]), 'bfloat16': _engine_report([10., -3., 2.])}
    exact = {'float32': _engine_report([10., -3., 2.0000001]), 'bfloat16': _engine_report([10.05, -3., 2.])}
    errors, warnings, facts = compare(stable, exact)
    assert not errors and facts['float32_logit_relative_error'] < 1e-6
    wrong = {'float32': _engine_report([7., -3., 2.]), 'bfloat16': _engine_report([7., -3., 2.])}
    errors, _, _ = compare(stable, wrong)
    assert any('math error' in error for error in errors)
    crash = {dtype: _engine_report([], ok=False, error='IndexError: index 6 is out of bounds') for dtype in stable}
    errors, _, _ = compare(stable, crash)
    assert len(errors) == 2 and 'IndexError' in errors[0]
    gpu = {dtype: _engine_report([], ok=False, error='RuntimeError: Found no NVIDIA driver; CUDA required') for dtype in stable}
    errors, warnings, facts = compare(stable, gpu)
    assert not errors and facts['gpu_only'] and any('GPU-only' in w for w in warnings)
    slow = {'float32': _engine_report([10., -3., 2.], seconds=.03, memory=200), 'bfloat16': _engine_report([10., -3., 2.])}
    errors, warnings, facts = compare(stable, slow)
    assert not errors and facts['cpu_decode_time_ratio'] == 3 and facts['parameter_memory_ratio'] == 2
    assert any('more work per token' in w for w in warnings) and any('duplicating weights' in w for w in warnings)


def test_harness_gate_is_skipped_for_fixtures_and_cached_per_source(tmp_path, monkeypatch):
    from vibeforge_v2 import harness, agent
    assert not agent.harness_enabled({'model': {'fixture': True}, 'engine': {'adapter': 'transformers'}, 'agent': {}})
    assert agent.harness_enabled({'model': {}, 'engine': {'adapter': 'transformers'}, 'agent': {}})
    assert not agent.harness_enabled({'model': {}, 'engine': {'adapter': 'transformers'}, 'agent': {'harness': False}})
    calls = []
    monkeypatch.setattr(harness, 'run', lambda *a, **k: calls.append(1) or
                        {'passed': False, 'errors': ['float32: math error'], 'warnings': [], 'facts': {}, 'scope': 's',
                         'wall_seconds': 1})
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    (workspace / 'engine.py').write_text('x = 1\n')
    config = {'agent': {}, 'runtime_python': sys.executable}
    first = agent.run_harness(config, workspace, tmp_path / 'agent', None)
    second = agent.run_harness(config, workspace, tmp_path / 'agent', None)
    assert not first['passed'] and 'HARNESS: FAIL' in first['text'] and len(calls) == 1 and second == first
    (workspace / 'engine.py').write_text('x = 2\n')
    agent.run_harness(config, workspace, tmp_path / 'agent', None)
    assert len(calls) == 2


def test_shared_guard_restores_in_place_library_edits_and_tells_the_agent(tmp_path):
    from vibeforge_v2.shared_guard import SharedGuard
    from vibeforge_v2.agent import Shell
    site = tmp_path / 'site-packages'
    (site / 'transformers/models').mkdir(parents=True)
    library = site / 'transformers/models/modeling.py'
    library.write_text('def forward():\n    return 1\n')
    guard = SharedGuard([site], tmp_path / 'cache')
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    tool = Shell(workspace, tmp_path / 'agent', guard=guard)
    result = tool.call(json.dumps({'command': f"sed -i 's/return 1/return 2/' {library} && touch {site}/injected.py"}))
    assert 'Permission denied' in result
    assert library.read_text() == 'def forward():\n    return 1\n'
    assert not (site / 'injected.py').exists()
    # The guard remains useful for edits by another, unconfined process.
    library.write_text('def forward():\n    return 2\n')
    (site / 'injected.py').touch()
    result = tool.call(json.dumps({'command': 'echo inspect external change'}))
    assert 'SHARED DEPENDENCY GUARD: ERROR' in result and 'NO effect' in result
    assert library.read_text() == 'def forward():\n    return 1\n'
    assert not (site / 'injected.py').exists()
    assert tool.calls[1]['shared_dependency_guard']['restored']
    clean = tool.call(json.dumps({'command': 'echo ok'}))
    assert 'SHARED DEPENDENCY GUARD' not in clean


def test_changed_lines_are_mapped_for_liveness_and_new_files(tmp_path):
    from vibeforge_v2.harness import changed_source_lines
    stable, candidate = tmp_path / 'stable', tmp_path / 'candidate'
    for root in (stable, candidate):
        root.mkdir()
    (stable / 'engine.py').write_text('a = 1\nb = 2\n')
    (candidate / 'engine.py').write_text('a = 1\nimport x\nb = 2\n')
    (candidate / 'helper.py').write_text('def f():\n\n    return 1\n')
    assert changed_source_lines(stable, candidate) == {('engine.py', 2), ('helper.py', 1), ('helper.py', 3)}


def test_unobserved_cpu_effect_defers_to_gpu_and_live_ones_pass(tmp_path, monkeypatch):
    from vibeforge_v2 import harness
    stable, candidate = tmp_path / 'run/current/engine', tmp_path / 'candidate'
    for root in (stable, candidate):
        root.mkdir(parents=True)
        (root / 'engine.json').write_text('{"backend": "transformers"}')
    (stable / 'engine.py').write_text('x = 1\n')
    (candidate / 'engine.py').write_text('x = 1\nimport json\n')
    same = {'ok': True, 'logits': [1., 2.], 'decode_seconds_per_token': .01, 'parameter_bytes': 1,
            'generate': {'short': {'token_ids': [1]}}, 'replaced_forwards': {}}
    def fake(directory, *args, **kwargs):
        lines = [] if Path(directory) == stable or not live else [['engine.py', 2]]
        return {**same, 'executed_lines': lines}
    monkeypatch.setattr(harness, 'build_reduced_model', lambda *a: tmp_path)
    monkeypatch.setattr(harness, 'run_engine', fake)
    live = False
    report = harness.run(candidate, stable=stable, model_path='m', cache=tmp_path / 'cache')
    assert report['status'] == 'SKIP' and report['allow_gpu_evaluation']
    assert not report['passed'] and not report['errors'] and 'no_effect' not in report['facts']
    assert report['facts']['unobserved_source_effect']
    assert any('import/load effects' in warning and 'CUDA-only' in warning for warning in report['warnings'])
    live = True
    assert harness.run(candidate, stable=stable, model_path='m', cache=tmp_path / 'cache')['passed']


def test_module_level_diff_pinpoints_the_wrong_output():
    from vibeforge_v2.harness import compare
    ok = {'ok': True, 'logits': [1., 2.], 'decode_seconds_per_token': .01, 'parameter_bytes': 1,
          'generate': {'short': {'token_ids': [1]}}}
    candidate = {**ok, 'logits': [1., 2.5], 'replaced_forwards': {'MoeBlock': {'module': 'layers.0.mlp'}},
                 'module_diff': [{'class': 'MoeBlock', 'module': 'layers.0.mlp', 'replacement': 'mine', 'input_shapes': [[1, 5, 8]],
                                  'structure': {'original': [['output[0]', [1, 5, 8]], ['output[1]', [5, 4]]],
                                                'candidate': [['output[0]', [1, 5, 8]], ['output[1]', [5, 4]]]},
                                  'outputs': [{'output': 'output[0]', 'shape': [1, 5, 8], 'max_relative_error': .4,
                                               'rows_wrong': 5, 'rows_total': 5, 'first_wrong_rows': [0, 1]},
                                              {'output': 'output[1]', 'shape': [5, 4], 'max_relative_error': 0.,
                                               'rows_wrong': 0, 'rows_total': 5, 'first_wrong_rows': []}]}]}
    errors, _, _ = compare({'float32': ok, 'bfloat16': ok}, {'float32': candidate, 'bfloat16': candidate})
    module = [e for e in errors if e.startswith('module check')]
    assert len(module) == 1 and 'output[0]' in module[0] and '5/5 token rows wrong' in module[0]


def test_candidate_may_not_load_code_from_another_round(tmp_path):
    workspace = tmp_path / 'run/rounds/0010/engine'
    workspace.mkdir(parents=True)
    (workspace / 'engine.json').write_text('{"backend":"toy"}')
    (workspace / 'engine.py').write_text('def generate():\n    return 1\n')
    checker = CandidateCheck(workspace, tmp_path / 'checks')
    (workspace / 'engine.py').write_text("import sys\nsys.path.insert(0, '/x/run/rounds/0001/engine')\n"
                                         'def generate():\n    return 1\n')
    report = checker.inspect()
    assert any("another round's directory" in error for error in report['errors'])


def test_library_keeps_only_verified_live_implementations():
    from vibeforge_v2.library import verified
    assert verified({'facts': {'float32_logit_relative_error': 0.0, 'replaced_forwards': {'A': {}}}, 'errors': []})
    assert not verified({'facts': {'float32_logit_relative_error': 0.0, 'replaced_forwards': {}, 'no_effect': True}, 'errors': []})
    assert not verified({'facts': {'float32_logit_relative_error': 0.13, 'replaced_forwards': {'A': {}}}, 'errors': []})


def test_resubmitting_an_implementation_measured_slower_on_gpu_is_blocked(tmp_path, monkeypatch):
    from vibeforge_v2 import agent, harness, library
    run = tmp_path / 'run'
    workspace = run / 'rounds/0010/engine'
    workspace.mkdir(parents=True)
    (workspace / 'engine.py').write_text('x = 1\n')
    report = {'passed': True, 'errors': [], 'warnings': [], 'scope': 's', 'wall_seconds': 1,
              'facts': {'replaced_forwards': {'Moe': {'fingerprint': 'abc'}}}}
    monkeypatch.setattr(harness, 'run', lambda *a, **k: dict(report))
    monkeypatch.setattr(library, 'update_library', lambda run: [{'round': '0001', 'fingerprints': ['abc'], 'gpu_gain': -.645}])
    result = agent.run_harness({'agent': {}, 'runtime_python': sys.executable}, workspace, tmp_path / 'agent', None)
    assert not result['passed'] and "round 0001" in result['errors'][0] and '-64.5%' in result['errors'][0]
    monkeypatch.setattr(library, 'update_library', lambda run: [{'round': '0001', 'fingerprints': ['zzz'], 'gpu_gain': -.645}])
    (workspace / 'engine.py').write_text('x = 2\n')
    assert agent.run_harness({'agent': {}, 'runtime_python': sys.executable}, workspace, tmp_path / 'agent', None)['passed']


def test_extra_flops_warns_but_leaves_performance_to_gpu_measurement():
    from vibeforge_v2.harness import compare
    ok = {'ok': True, 'logits': [1., 2.], 'decode_seconds_per_token': .01, 'parameter_bytes': 1,
          'generate': {'short': {'token_ids': [1]}}}
    def candidate(ratio):
        return {**ok, 'replaced_forwards': {'Moe': {'module': 'm'}}, 'module_diff': [{
            'class': 'Moe', 'module': 'm', 'replacement': 'r', 'input_shapes': [[1, 1, 8]],
            'structure': {'original': [['output[0]', [1]]], 'candidate': [['output[0]', [1]]]},
            'outputs': [{'output': 'output[0]', 'shape': [1], 'max_relative_error': 0., 'rows_wrong': 0,
                         'rows_total': 1, 'first_wrong_rows': []}],
            'flops': {'original': 100., 'candidate': 100. * ratio}}]}
    errors, warnings, facts = compare({'float32': ok, 'bfloat16': ok}, {'float32': candidate(8), 'bfloat16': candidate(8)})
    assert not errors
    assert any('8.0x the matmul FLOPs' in e for e in warnings)
    errors, _, facts = compare({'float32': ok, 'bfloat16': ok}, {'float32': candidate(1), 'bfloat16': candidate(1)})
    assert not errors and list(facts['module_flop_ratio'].values()) == [1.0]


def test_verbatim_copy_of_original_forward_is_no_effect_and_library_separates_negatives():
    from vibeforge_v2.harness import compare
    from vibeforge_v2.library import render
    ok = {'ok': True, 'logits': [1., 2.], 'decode_seconds_per_token': .01, 'parameter_bytes': 1,
          'generate': {'short': {'token_ids': [1]}}}
    copy = {**ok, 'replaced_forwards': {'Moe': {'replacement': 'mine', 'fingerprint': 'same', 'original_fingerprint': 'same'}}}
    errors, _, facts = compare({'float32': ok, 'bfloat16': ok}, {'float32': copy, 'bfloat16': copy})
    assert facts['no_effect'] and any('original library code under a new name' in e for e in errors)
    text = render([{'round': '0001', 'replaced': 'Moe → slow', 'gpu_gain': -.6, 'flop_ratio': 8., 'path': 'p',
                    'fp32_error': 0., 'cpu_ratio': 2., 'outcome': 'reject'}])
    assert '反例' in text and '可作为起点的实现' in text and 'cp -r' not in text


def test_runtime_failure_reports_local_tensor_shapes_of_candidate_frames(tmp_path):
    from vibeforge_v2.harness import run_engine, compare
    candidate, work = tmp_path / 'candidate', tmp_path / 'work'
    candidate.mkdir()
    work.mkdir()
    (candidate / 'engine.json').write_text('{}')
    (candidate / 'engine.py').write_text(
        'import torch\n'
        'def load(config):\n'
        '    hidden = torch.zeros(1, 4)\n'
        '    indices = torch.tensor([[0, 6]])\n'
        '    return hidden[indices[0]]\n')
    report = run_engine(candidate, tmp_path, 'float32', sys.executable, 60, work)
    assert not report['ok'] and report['stage'] == 'load()'
    frame = report['failure_frames'][-1]
    assert frame['locals']['hidden'] == 'Tensor[1, 4] float32' and frame['locals']['indices'] == 'Tensor[1, 2] int64'
    ok = {'ok': True, 'logits': [1.], 'decode_seconds_per_token': .01, 'parameter_bytes': 1, 'generate': {}}
    errors, _, _ = compare({'float32': ok, 'bfloat16': ok}, {'float32': report, 'bfloat16': report})
    assert 'SHAPES AT FAILURE' in errors[0] and 'indices = Tensor[1, 2] int64' in errors[0]


def test_only_the_latest_assistant_step_keeps_its_thinking_in_the_request():
    from vibeforge_v2.agent import strip_old_reasoning
    from qwen_agent.llm.schema import Message
    messages = [Message(role='user', content='task'),
                Message(role='assistant', content='', reasoning_content='old thinking'),
                Message(role='assistant', content='call 1'),
                Message(role='function', content='out 1', name='shell'),
                Message(role='assistant', content='', reasoning_content='latest thinking'),
                Message(role='assistant', content='call 2'),
                Message(role='function', content='out 2', name='shell')]
    trimmed = strip_old_reasoning(messages)
    assert [m.reasoning_content for m in trimmed if m.role == 'assistant'] == ['', None, 'latest thinking', None] or \
           [m.reasoning_content or '' for m in trimmed if m.role == 'assistant'] == ['', '', 'latest thinking', '']
    assert messages[1].reasoning_content == 'old thinking'  # the stored conversation is not modified


def test_repeated_identical_command_warns_without_stopping_the_round(tmp_path):
    from vibeforge_v2.agent import Shell
    from types import SimpleNamespace
    workspace = tmp_path / 'engine'
    workspace.mkdir()
    checker = SimpleNamespace(inspect=lambda: {'source_hash': 'unchanged', 'runtime_changed_files': []},
                              describe=lambda report: 'static_check: PASS')
    tool = Shell(workspace, tmp_path / 'agent', feedback=checker)
    first = tool.call(json.dumps({'command': "grep -n nothing_here /dev/null"}))
    assert 'NO LINE MATCHED' in first and 'HOST NOTE' not in first
    second = tool.call(json.dumps({'command': "grep -n nothing_here /dev/null"}))
    assert 'HOST NOTE' not in second
    tool.call(json.dumps({'command': 'echo different'}))
    tool.call(json.dumps({'command': "grep -n nothing_here /dev/null"}))
    tool.call(json.dumps({'command': "grep -n nothing_here /dev/null"}))
    fifth = tool.call(json.dumps({'command': "grep -n nothing_here /dev/null"}))
    assert 'possible local loop' in fifth and 'the round continues' in fifth
    sixth = tool.call(json.dumps({'command': "grep -n nothing_here /dev/null"}))
    assert 'possible local loop' in sixth
    recovered = tool.call(json.dumps({'command': 'echo recovered'}))
    assert 'recovered' in recovered and 'possible local loop' not in recovered
    records = [json.loads(line) for line in (tmp_path / 'agent/tools.jsonl').read_text().splitlines()]
    assert len(records) == len(tool.calls)
    assert records[5]['repeat_detection']['matching_calls'] == [1, 2, 4, 5, 6]
    assert records[5]['repeat_detection']['action'] == 'warn_only'


def test_agent_visible_guidance_names_no_specific_architecture():
    """The prompt and host messages must stay model-agnostic; measured facts may describe the loaded model."""
    from vibeforge_v2.config import ROOT
    import re
    prompt = (ROOT / 'prompts/engineer.md').read_text()
    assert not re.search(r'moe|expert|专家|qwen', prompt, re.IGNORECASE)
    harness_source = (ROOT / 'src/vibeforge_v2/harness.py').read_text()
    messages = re.findall(r"(?:errors|warnings)\.append\((.*?)\)\n", harness_source, re.DOTALL)
    assert messages and not any(re.search(r'MoE|top-k|selected_experts|Qwen3Moe', text) for text in messages)
