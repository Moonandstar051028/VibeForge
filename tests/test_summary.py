"""Large profiler detail remains on disk without bloating the initial context."""

import copy
import json

from vf_runtime.storage.atomic import write_json
from vf_runtime.versioning.artifacts import source_hash
from vibeforge_v2.config import ROOT, load
from vibeforge_v2.evaluate import Evaluator


def test_large_operator_shapes_do_not_expand_the_bottleneck_summary(tmp_path):
    config = load(ROOT / 'configs/toy.json')
    config['workload']['concurrency'] = 1
    source = tmp_path / 'engine'
    source.mkdir()
    (source / 'engine.py').write_text('pass\n')
    write_json(source / 'engine.json', {})
    details = 'UNABRIDGED_SHAPE_DETAIL_' * 5000
    record = {'source_hash': source_hash(source),
        'normal_request': {'backend_wall_seconds': .1, 'observed_generate_kwargs': details},
        'top_cpu_ops': [{'name': 'matmul', 'calls': 5, 'self_cpu_seconds': .08, 'input_shapes': details}],
        'top_device_ops': [{'name': 'kernel', 'calls': 5, 'self_device_seconds': .07, 'input_shapes': details}]}
    class Client:
        def bounded(self, *args):
            return self
        def profile(self, **kwargs):
            return copy.deepcopy(record)
    folder = tmp_path / 'observations'
    Evaluator(config, tmp_path / 'run').observe(Client(), source, folder)
    summary = (folder / 'summary.md').read_text()
    assert 'matmul' in summary and '0.08' in summary and 'kernel' in summary
    assert details not in summary and len(summary) < 8000
    assert details in (folder / 'observations.json').read_text()


def test_summary_exposes_moe_vs_attention_module_evidence(tmp_path):
    config = load(ROOT / 'configs/toy.json')
    config['workload']['concurrency'] = 1
    source = tmp_path / 'engine'
    source.mkdir()
    (source / 'engine.py').write_text('pass\n')
    write_json(source / 'engine.json', {})
    record = {'source_hash': source_hash(source), 'normal_request': {},
              'phases': [{'name': 'vf_runtime.module.mlp', 'device_total_seconds': .93},
                         {'name': 'vf_runtime.module.full_attention', 'device_total_seconds': .17}],
              'implementation_index': {'locations': {'Qwen3MoeSparseMoeBlock.forward': {'path': 'actual_model.py', 'line': 226}}}}
    class Client:
        def bounded(self, *args): return self
        def profile(self, **kwargs): return copy.deepcopy(record)
    folder = tmp_path / 'observations'
    Evaluator(config, tmp_path / 'run').observe(Client(), source, folder)
    summary = (folder / 'summary.md').read_text()
    assert 'vf_runtime.module.mlp' in summary and '0.93' in summary
    assert 'vf_runtime.module.full_attention' in summary and '0.17' in summary
    assert 'Qwen3MoeSparseMoeBlock.forward' in summary and 'actual_model.py' in summary
    assert '存在重叠' in summary
    digest = (folder / 'digest.md').read_text()
    assert '已加载的源码定义位置' in digest
    assert '实际被调用的实现位置' not in digest
    assert '不代表这些定义都被本次请求调用' in digest
