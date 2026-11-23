"""Load one human-owned configuration; the agent never submits configuration JSON."""

import copy
import hashlib
import math
import sys
from pathlib import Path

from vf_runtime.contracts.precision import normalize_precision
from vf_runtime.contracts.workload import normalize_workload, workload_profiles
from vf_runtime.engine.adapters import normalize_engine
from vf_runtime.storage.atomic import read_json

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    config = copy.deepcopy(read_json(path))
    config['project'] = str(ROOT)
    config.setdefault('runtime_python', sys.executable)
    config.setdefault('rounds', 3)
    if type(config['rounds']) is not int or config['rounds'] < 1:
        raise ValueError('rounds must be a positive integer')
    model = config['model']
    model.setdefault('device', 'cuda:0')
    model.setdefault('max_context', 27648)
    if model.get('path'):
        model['path'] = str((ROOT / model['path']).resolve())
    config.setdefault('hardware', {'device': model['device']})
    config['precision'] = normalize_precision(config.get('precision'), model)
    normalize_engine(config, ROOT)
    normalize_workload(config['workload'])
    config['workload']['cases'] = str((ROOT / config['workload']['cases']).resolve())
    protocol = config['evaluation']
    for key, value in {'repeats': 5, 'warmup': 1, 'min_gain': .02, 'confidence': .95,
                       'max_retests': 1, 'reference_repeats': 3, 'min_reference_consensus': 2/3,
                       'correctness_mode': 'strict', 'objective': 'throughput', 'limits': {}}.items():
        protocol.setdefault(key, value)
    if type(protocol['repeats']) is not int or protocol['repeats'] < 2:
        raise ValueError('Formal performance requires at least two paired measurements')
    if not 0 < protocol['confidence'] < 1 or not math.isfinite(protocol['min_gain']) or protocol['min_gain'] < 0:
        raise ValueError('Invalid confidence or minimum gain')
    protocol['concurrency'] = max(p['concurrency'] for p in workload_profiles(config['workload']))
    protocol['device_backend'] = model['device'].split(':')[0]
    protocol.setdefault('quality_failure_diagnostics', True)
    protocol.setdefault('collect_token_timing', False)
    if type(protocol['collect_token_timing']) is not bool:
        raise ValueError('collect_token_timing must be boolean')
    if type(protocol['quality_failure_diagnostics']) is not bool:
        raise ValueError('quality_failure_diagnostics must be boolean')
    protections = protocol.get('max_metric_regressions', {})
    if not isinstance(protections, dict):
        raise ValueError('max_metric_regressions must be an object')
    for name, maximum in protections.items():
        if name not in {'latency_mean_seconds', 'ttft_seconds_mean', 'tpot_seconds_mean'}:
            raise ValueError('Unsupported mean latency protection: ' + name)
        if type(maximum) not in (float, int) or not 0 <= maximum < math.inf:
            raise ValueError('Metric regression tolerance must be finite and non-negative')
    quality = protocol.get('quality')
    if protocol['correctness_mode'] not in ('strict', 'approximate', 'quality'):
        raise ValueError('correctness_mode must be strict, approximate or quality')
    if protocol['correctness_mode'] == 'quality':
        protocol.setdefault('distribution_diagnostics', True)
        if not quality or not quality.get('paired'):
            raise ValueError('Quality correctness requires an independent paired quality evaluator')
        paired = quality['paired']
        if paired.get('decision_mode', 'confidence_bound') not in ('confidence_bound', 'observed_loss'):
            raise ValueError('Invalid paired quality decision_mode')
        if (type(paired.get('min_cases', 100)) is not int or paired.get('min_cases', 100) < 1 or
                not 0 < paired.get('confidence', .95) < 1 or not 0 <= paired.get('max_loss_rate', .05) < 1):
            raise ValueError('Invalid paired quality parameters')
    if quality:
        quality.setdefault('timeout_seconds', 300)
        quality.setdefault('repeats', 1)
        quality.setdefault('baseline_repeats', quality['repeats'])
        if type(quality['baseline_repeats']) is not int or quality['baseline_repeats'] < 1:
            raise ValueError('quality.baseline_repeats must be a positive integer')
        quality['file_hashes'] = {str((ROOT / p).resolve()): hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                  for p in quality.get('files', [])}
    if protocol['correctness_mode'] != 'strict' and not quality:
        raise ValueError('Approximate correctness requires an independent quality evaluator')
    agent = config.setdefault('agent', {})
    agent.setdefault('shell_timeout_seconds', 900)
    agent.setdefault('isolate_dependencies', not model.get('fixture', False))
    if type(agent['isolate_dependencies']) is not bool:
        raise ValueError('agent.isolate_dependencies must be boolean')
    if type(agent['shell_timeout_seconds']) not in (int, float) or not 0 < agent['shell_timeout_seconds'] < math.inf:
        raise ValueError('agent.shell_timeout_seconds must be positive and finite')
    agent.setdefault('backend', 'qwen')
    agent.setdefault('config_tool', True)
    if agent['backend'] not in ('qwen', 'codex'):
        raise ValueError('agent.backend must be qwen or codex')
    if type(agent['config_tool']) is not bool:
        raise ValueError('agent.config_tool must be boolean')
    agent.setdefault('use_raw_api', False)
    if type(agent['use_raw_api']) is not bool:
        raise ValueError('agent.use_raw_api must be boolean')
    agent.setdefault('exact_context', agent['use_raw_api'] and bool(model.get('path')) and not model.get('fixture', False))
    if type(agent['exact_context']) is not bool:
        raise ValueError('agent.exact_context must be boolean')
    if agent['exact_context'] and (not agent['use_raw_api'] or not model.get('path')):
        raise ValueError('exact_context requires native API and the serving model tokenizer path')
    agent.setdefault('trace_requests', False)
    for key in ('fold_repeated_reads', 'structured_memory', 'repeat_recovery', 'fold_observed_repeats', 'extractive_memory'):
        agent.setdefault(key, False)
        if type(agent[key]) is not bool:
            raise ValueError(f'agent.{key} must be boolean')
        if agent[key] and not agent['exact_context']:
            raise ValueError(f'agent.{key} requires exact_context')
    if agent['repeat_recovery'] and not agent['fold_repeated_reads']:
        raise ValueError('agent.repeat_recovery requires fold_repeated_reads')
    if agent['fold_observed_repeats'] and not agent['fold_repeated_reads']:
        raise ValueError('agent.fold_observed_repeats requires fold_repeated_reads')
    agent.setdefault('memory_max_tokens', 1024)
    if type(agent['memory_max_tokens']) is not int or not 128 <= agent['memory_max_tokens'] <= 4096:
        raise ValueError('agent.memory_max_tokens must be an integer between 128 and 4096')
    if type(agent['trace_requests']) is not bool:
        raise ValueError('agent.trace_requests must be boolean')
    sampling = agent.get('sampling', {})
    allowed_sampling = {'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty'}
    if not isinstance(sampling, dict) or set(sampling) - allowed_sampling:
        raise ValueError('agent.sampling contains unsupported generation options')
    for name, value in sampling.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'agent.sampling.{name} must be a finite number')
        valid = (value >= 0 if name == 'temperature' else
                 type(value) is int and value >= 0 if name == 'top_k' else
                 0 <= value <= 1 if name == 'min_p' else
                 0 < value <= 1 if name == 'top_p' else value > 0)
        if not valid:
            raise ValueError(f'Invalid agent.sampling.{name}')
    agent.setdefault('max_new_tokens', 8192)
    # The framework counts message text approximately, before rendering tool schemas.
    agent.setdefault('context_reserve_tokens', min(2048, model['max_context'] // 16))
    if (type(agent['context_reserve_tokens']) is not int or agent['context_reserve_tokens'] < 0
            or agent['context_reserve_tokens'] + agent['max_new_tokens'] >= model['max_context']):
        raise ValueError('Agent context reserve and output must fit the model window')
    agent.setdefault('request_timeout', 1800)
    # Official Qwen-Agent switch: true for raw thinking in content, false for
    # servers returning separate reasoning_content. Do not guess from model name.
    agent.setdefault('thought_in_content', False)
    if type(agent['thought_in_content']) is not bool:
        raise ValueError('agent.thought_in_content must be boolean')
    agent.setdefault('thinking', {'enabled': True, 'temperature': .6, 'top_p': .95, 'top_k': 20})
    agent.setdefault('prompt', 'prompts/engineer.md')
    if not 0 < agent['max_new_tokens'] < model['max_context']:
        raise ValueError('Agent output must fit the configured model window')
    config.setdefault('profiling', {'coverage': 'all', 'max_matrix_profiles': 3, 'max_new_tokens': 16,
                                    'timeout_seconds': 900})
    config.setdefault('probe', {'repeats': 1, 'warmup': 0, 'timeout_seconds': 900})
    profiling, probe = config['profiling'], config['probe']
    # Reuse a stable survey while its source, dependencies, workload and settings are unchanged.
    profiling.setdefault('reuse_stable_survey', True)
    profiling.setdefault('reuse_max_age_seconds', 86400)
    if type(profiling['reuse_stable_survey']) is not bool:
        raise ValueError('profiling.reuse_stable_survey must be boolean')
    # Candidate probe profiles only the observation scenario unless configured otherwise.
    probe.setdefault('profile_coverage', 'selected')
    if probe['profile_coverage'] not in ('selected', 'all'):
        raise ValueError('probe.profile_coverage must be selected or all')
    agent.setdefault('history_reasoning', 'last')
    if agent['history_reasoning'] not in ('last', 'all'):
        raise ValueError('agent.history_reasoning must be last or all')
    agent.setdefault('related_history', True)
    if type(agent['related_history']) is not bool:
        raise ValueError('agent.related_history must be boolean')
    agent.setdefault('guard_shared_dependencies', True)
    agent.setdefault('harness', True)
    agent.setdefault('harness_timeout_seconds', 420)
    if type(agent['harness']) is not bool:
        raise ValueError('agent.harness must be boolean')
    agent.setdefault('shell_output_chars', 12000)
    if type(agent['shell_output_chars']) is not int or agent['shell_output_chars'] < 0:
        raise ValueError('agent.shell_output_chars must be a non-negative integer (0 disables clipping)')
    protocol.setdefault('batch_divergence_max_logit_gap', 0.5)
    if type(protocol['batch_divergence_max_logit_gap']) not in (int, float) or not 0 <= protocol['batch_divergence_max_logit_gap'] < math.inf:
        raise ValueError('evaluation.batch_divergence_max_logit_gap must be finite and non-negative')
    if quality:
        quality.setdefault('category_item_tolerance', 1)
        if type(quality['category_item_tolerance']) is not int or quality['category_item_tolerance'] < 0:
            raise ValueError('quality.category_item_tolerance must be a non-negative integer')
    config.setdefault('startup_timeout', 900)
    config.setdefault('request_timeout', 600)
    return config

