"""Coordinate the unchanged v1 measurements, with no agent proposal protocol."""

import os
import json
import math
import shutil
import time
from pathlib import Path
from .performance_feedback import compare_profiles
from types import SimpleNamespace

from vf_runtime.contracts.workload import profiling_cases, workload_profiles
from vf_runtime.controller.admission import decide
from vf_runtime.engine.builder import build
from vf_runtime.engine.process import EngineProcess
from vf_runtime.evaluation.correctness import evaluate_candidate_precheck, evaluate_correctness
from vf_runtime.evaluation.performance import diagnose_concurrency, measure_suite, measurement_mode
from vf_runtime.evaluation.quality import run_quality
from vf_runtime.observability.logs import run_command
from vf_runtime.storage.atomic import atomic_write, digest, read_json, write_json
from vf_runtime.versioning.artifacts import inventory, source_hash
from .environment import Environments, package_snapshot
from .bottleneck import write_digest


class Evaluator:
    def __init__(self, config, run):
        self.config, self.run = config, Path(run)
        paths = SimpleNamespace(project=Path(config['project']), data=self.run)
        self.engine = EngineProcess(paths, 'active', config['runtime_python'],
                                    config['startup_timeout'], config['request_timeout'])
        self.client = None
        self.environments = Environments(config, run)

    def stop(self):
        self.engine.stop()
        self.client = None

    def start(self, source, log):
        self.engine.python = self.environments.python(source)
        self.client = self.engine.start(Path(source), Path(log))
        health = self.client.health()
        write_json(Path(log).with_suffix('.runtime.json'), health)
        return self.client

    def build(self, source, folder):
        config = self.config
        try:
            report = build(source)
        except Exception as error:
            write_json(Path(folder) / 'build.json', {'passed': False, 'error': f'{type(error).__name__}: {error}'})
            raise
        cache = self.run / 'cache/build'
        cache.mkdir(parents=True, exist_ok=True)
        args = {'python': self.environments.python(source), 'workspace': str(source), 'cache': str(cache)}
        command = [part.format_map(args) for part in config['engine']['build_command']]
        if command:
            env = dict(os.environ, PYTHONPATH=str(Path(config['project']) / 'src'),
                       TMPDIR=str(cache), TORCH_EXTENSIONS_DIR=str(cache / 'extensions'),
                       HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
            execution = run_command(command, source, Path(folder) / 'build.log',
                                    config['engine']['build_timeout_seconds'], env=env)
            report['command'] = execution
            if execution['exit_code']:
                report['passed'] = False
                write_json(Path(folder) / 'build.json', report)
                raise RuntimeError(f'Build failed; see {Path(folder) / "build.log"}')
        report.update(build(source))
        write_json(Path(folder) / 'build.json', report)
        return report

    def survey_key(self, source, coverage):
        """Identity of everything a stable survey depends on; any change forces a fresh survey."""
        try:
            packages = read_json(self.environments.directory(source) / 'packages.json')['hash'] \
                if self.environments.enabled else package_snapshot(self.environments.python(source))['hash']
        except (OSError, ValueError, KeyError):
            return None
        settings = {key: value for key, value in self.config['profiling'].items()
                    if key not in ('reuse_stable_survey', 'reuse_max_age_seconds', 'timeout_seconds')}
        return digest({'source_hash': source_hash(source), 'packages': packages, 'coverage': coverage,
                       'profiling': settings, 'workload': self.config['workload'].get('profiles'),
                       'device': self.config['model'].get('device'), 'model': self.config['model'].get('path'),
                       'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})

    def _reuse_survey(self, key, folder):
        settings = self.config['profiling']
        cache = self.run / 'cache/surveys' / key
        if not key or not settings.get('reuse_stable_survey', True) or not (cache / 'observations.json').exists():
            return None
        report = read_json(cache / 'observations.json')
        age = time.time() - report.get('collected_at', 0)
        if age > settings.get('reuse_max_age_seconds', 86400):
            return None
        folder.mkdir(parents=True, exist_ok=True)
        for path in cache.iterdir():
            if path.is_file():
                shutil.copy2(path, folder / path.name)
        report['reused'] = {'from': report.get('origin_folder'), 'age_seconds': age, 'survey_key': key,
                            'reason': 'Stable source, dependencies, workload and profiling settings are unchanged.'}
        write_json(folder / 'observations.json', report)
        return report

    def observe(self, client, source, folder, label='stable', coverage=None):
        """Stable bottleneck survey; reused while its full identity is unchanged (see survey_key)."""
        folder = Path(folder)
        settings = self.config['profiling']
        coverage = coverage or settings['coverage']
        key = self.survey_key(source, coverage) if label == 'stable' else None
        reused = self._reuse_survey(key, folder) if key else None
        if reused is not None:
            print(f"survey {label}: reused {reused['reused']['from']} ({reused['reused']['age_seconds']:.0f}s old)", flush=True)
            write_survey_summary(reused, folder)
            self._digest(reused, folder)
            return reused
        client = client.bounded(time.monotonic() + settings['timeout_seconds'], 'bottleneck survey')
        survey_started = time.monotonic()
        records = []
        for case in profiling_cases(self.config['workload'], coverage, settings['max_matrix_profiles']):
            profile = case['profile']
            print(f"survey {label}: {profile['id']} prompt {case['prompt_index']}", flush=True)
            capture_started = time.monotonic()
            record = client.profile(prompt=case['prompt'], max_new_tokens=min(
                profile['max_new_tokens'], settings['max_new_tokens'], 128))
            if record.get('source_hash') != source_hash(source):
                raise RuntimeError('Profiler source identity differs from the running stable snapshot')
            record.update(profile_id=profile['id'], prompt_index=case['prompt_index'],
                          capture_wall_seconds=time.monotonic() - capture_started)
            write_json(folder / f"{profile['id']}-{case['prompt_index']}.json", record)
            records.append(record)
        concurrency = {p['id']: diagnose_concurrency(client, p)
                       for p in workload_profiles(self.config['workload']) if p['concurrency'] > 1}
        excerpts = []
        if callable(getattr(client, 'implementation', None)) and records:
            locations = (records[0].get('implementation_index') or {}).get('locations', {})
            selected = [name for name in locations if any(word in name.lower()
                        for word in ('sparsemoe', 'sparse_moe')) and name.endswith('.forward')]
            selected += [name for name in locations if name.lower().endswith('attention.forward')]
            for symbol in selected[:2]:
                try:
                    excerpt = client.implementation(symbol=symbol, limit=4000)
                    if excerpt.get('source_hash') != source_hash(source):
                        raise RuntimeError('Implementation source identity differs from stable snapshot')
                    excerpts.append(excerpt)
                except Exception as error:
                    excerpts.append({'symbol': symbol, 'error': str(error)})
        report = {'collected_at': time.time(), 'source_hash': source_hash(source), 'engine_role': label,
                  'admission': False, 'profiles': records, 'concurrency': concurrency,
                  'survey_wall_seconds': time.monotonic() - survey_started, 'implementation_excerpts': excerpts}
        report['origin_folder'] = str(folder)
        write_json(folder / 'observations.json', report)
        write_survey_summary(report, folder)
        self._digest(report, folder)
        if key:
            cache = self.run / 'cache/surveys' / key
            if cache.exists():
                shutil.rmtree(cache)
            shutil.copytree(folder, cache)
        return report

    def _digest(self, report, folder):
        try:
            profile = read_json(self.run / 'profile.json') if (self.run / 'profile.json').exists() else {}
            write_digest(report, profile, self.config, folder)
        except Exception as error:  # a derived aid; the raw survey is still complete
            atomic_write(Path(folder) / 'digest.md', f'瓶颈摘要生成失败：{type(error).__name__}: {error}\n')

    def probe(self, client, source, folder, reference):
        """Same precheck/profiler/concurrent load; no model-chosen JSON metric binding."""
        folder = Path(folder)
        settings = self.config['probe']
        client = client.bounded(time.monotonic() + settings['timeout_seconds'], 'quick probe')
        precheck = evaluate_candidate_precheck(client, reference, self.config['evaluation'])
        result = {'admission': False, 'correctness_precheck': precheck, 'profiles': [], 'concurrency': {}}
        # Keep mechanism evidence even if the conservative batch-equality check
        # fails; this diagnostic never permits admission on its own.
        for profile in workload_profiles(self.config['workload']):
            if profile['concurrency'] > 1:
                try:
                    result['concurrency'][profile['id']] = diagnose_concurrency(client, profile)
                except Exception as error:
                    result['concurrency'][profile['id']] = {'error': str(error)}
        if precheck['passed']:
            for _ in range(settings['warmup']):
                for p in workload_profiles(self.config['workload']):
                    client.generate(prompt=p['prompts'][0], max_new_tokens=min(
                        p['max_new_tokens'], self.config['profiling']['max_new_tokens'], 128), fixed_length=True)
            for i in range(settings['repeats']):
                result['profiles'].append(self.observe(client, source, folder / f'profile-{i+1}', label='candidate',
                                                       coverage=settings.get('profile_coverage', 'selected')))
        write_json(folder / 'probe.json', result)
        return result

    def correctness(self, client, reference, folder):
        report = evaluate_correctness(client, reference, self.config['evaluation'])
        write_json(Path(folder) / 'correctness.json', report)
        return report

    def quality(self, client, source, folder, baseline=None):
        settings = self.config['evaluation'].get('quality')
        if not settings:
            report = {'passed': True, 'enabled': False, 'metrics': {}}
            write_json(Path(folder) / 'report.json', report)
            return report
        if baseline is None:
            settings = {**settings, 'repeats': settings.get('baseline_repeats', settings.get('repeats', 1))}
        return run_quality(client, settings, folder, {
            'project': self.config['project'], 'python': self.environments.python(source), 'version': source_hash(source),
            'config_hash': digest(self.config), 'precision': read_json(Path(source) / 'engine.json').get('precision'),
        }, baseline=baseline)

    def performance(self, stable, candidate, reference, folder, correctness):
        protocol = self.config['evaluation']
        self.environments.verify(stable)
        self.environments.verify(candidate)
        for attempt in range(protocol['max_retests'] + 1):
            baseline, challenger = [], []
            for group in range(protocol['repeats']):
                order = [(stable, baseline, 'stable'), (candidate, challenger, 'candidate')]
                if group % 2:
                    order.reverse()
                for source, rows, label in order:
                    client = self.start(source, Path(folder) / 'logs' / f'{attempt}-{group}-{label}.log')
                    for _ in range(protocol['warmup']):
                        for p in workload_profiles(self.config['workload']):
                            for prompt in p['prompts']:
                                client.generate(prompt=prompt, max_new_tokens=p['max_new_tokens'], fixed_length=True,
                                                observation_mode=measurement_mode(protocol))
                    rows.append(measure_suite(client, self.config['workload'], Path(folder) / 'samples.jsonl',
                        source_hash(source), group, protocol, (self.engine.process.pid,), reference=reference,
                        startup_seconds=client.startup_seconds,
                        source_bytes=sum((Path(source) / name).stat().st_size for name in inventory(source)), attempt=attempt))
                    self.stop()
            decision = decide(correctness, baseline, challenger, protocol).to_dict()
            write_json(Path(folder) / f'attempt-{attempt}.json',
                       {'decision': decision, 'baseline': baseline, 'candidate': challenger,
                        'comparison': compare_profiles(baseline, challenger)})
            if decision['outcome'] != 'invalid':
                break
        return decision

    def diagnose_performance(self, stable, candidate, reference, folder):
        """Two AB/BA pairs for research feedback; cannot authorize promotion."""
        self.environments.verify(stable)
        self.environments.verify(candidate)
        protocol, folder = self.config['evaluation'], Path(folder)
        baseline, challenger = [], []
        for group in range(2):
            order = [(stable, baseline, 'stable'), (candidate, challenger, 'candidate')]
            if group % 2:
                order.reverse()
            for source, rows, label in order:
                print(f'exploratory performance: pair {group + 1}/2 {label}', flush=True)
                client = self.start(source, folder / 'logs' / f'{group}-{label}.log')
                try:
                    for p in workload_profiles(self.config['workload']):
                        for prompt in p['prompts']:
                            client.generate(prompt=prompt, max_new_tokens=p['max_new_tokens'], fixed_length=True,
                                            observation_mode=measurement_mode(protocol))
                    rows.append(measure_suite(client, self.config['workload'], folder / 'samples.jsonl',
                        source_hash(source), group, protocol, (self.engine.process.pid,), reference=reference,
                        startup_seconds=client.startup_seconds,
                        source_bytes=sum((Path(source) / name).stat().st_size for name in inventory(source))))
                finally:
                    self.stop()
        objective = protocol.get('objective', 'throughput')
        a = [row.get('metrics', {}).get(objective) for row in baseline]
        b = [row.get('metrics', {}).get(objective) for row in challenger]
        valid = (all(row['valid'] for row in baseline + challenger) and
                 all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in a + b))
        from vf_runtime.contracts.requirements import OBJECTIVES
        direction = OBJECTIVES[objective]
        gains = [(y / x - 1) if direction == 'maximize' else (1 - y / x) for x, y in zip(a, b)] if valid else []
        report = {'kind': 'exploratory_performance', 'admission': False, 'quality_passed': False,
                  'valid': valid, 'objective': objective, 'direction': direction,
                  'pairs': len(baseline), 'gain': sum(gains) / len(gains) if gains else None,
                  'paired_gains': gains, 'baseline': baseline, 'candidate': challenger,
                  'comparison': compare_profiles(baseline, challenger),
                  'limitations': ['Quality gate failed; this report cannot promote the candidate.',
                                  'Two pairs are research feedback, not formal significance or deployment acceptance.']}
        write_json(folder / 'report.json', report)
        return report


def write_survey_summary(report, folder):
    """Render saved measurements without rerunning or fabricating observations."""
    folder = Path(folder)
    records, concurrency = report['profiles'], report['concurrency']
    label = report.get('engine_role', 'stable')
    lines = [f'# 本轮 {label} 引擎性能勘测', '', '以下数据用于寻找瓶颈，不能当作正式收益。']
    if report.get('reused'):
        lines.append(f"本轮复用了相同稳定源码、依赖、负载与采集设置下的勘测（来源 {report['reused']['from']}，"
                     f"距今 {report['reused']['age_seconds']:.0f} 秒）；稳定版本变化后会重新采集。")
    for row in records:
        normal = row.get('normal_request') or {}
        concise = {key: normal.get(key) for key in ('backend_wall_seconds', 'input_tokens', 'output_tokens',
                   'output_tokens_per_second', 'ttft_seconds', 'tpot_seconds', 'queue_seconds', 'batch_size')}
        lines.extend(['', f"## {row['profile_id']} / prompt {row['prompt_index']}",
                      f"普通请求：{concise}", f"诊断采集总耗时（非推理收益）：{row.get('capture_wall_seconds')} 秒", 'CPU 热点（秒）：'])
        lines.extend(str({key: x.get(key) for key in ('name', 'calls', 'self_cpu_seconds')})
                     for x in row.get('top_cpu_ops', [])[:5])
        lines.append('模块与阶段归属（instrumented inclusive；存在重叠，不能相加或当墙钟占比）：')
        lines.extend(str({key: phase.get(key) for key in
                          ('name', 'calls', 'cpu_total_seconds', 'device_total_seconds')})
                     for phase in row.get('phases', [])[:8])
        locations = (row.get('implementation_index') or {}).get('locations', {})
        relevant = {name: location for name, location in locations.items()
                    if any(word in name.lower() for word in ('moe', 'attention', 'mlp', 'rmsnorm'))}
        lines.append('实际加载的模型实现位置（核对热点调用点，算子名本身不能区分注意力与 MoE）：')
        lines.append(str(dict(list(relevant.items())[:8])))
        lines.append('设备热点：')
        lines.extend(str({'name': x.get('name', '')[:180], 'calls': x.get('calls'),
                          'self_device_seconds': x.get('self_device_seconds')})
                     for x in row.get('top_device_ops', [])[:5])
    if report.get('implementation_excerpts'):
        lines.extend(['', '实际加载的关键实现摘录（来源哈希与后续分页在 observations.json；用于核对调用路径）：'])
        for excerpt in report['implementation_excerpts']:
            lines.extend([str({key: excerpt.get(key) for key in ('symbol', 'sha256', 'next_offset', 'error')}),
                          '```python', excerpt.get('source', ''), '```'])
    concise_concurrency = {name: {key: row.get(key) for key in ('throughput', 'latency_p95_seconds',
        'queue_p95_seconds', 'observed_batch_size_max', 'request_count')} for name, row in concurrency.items()}
    lines.extend(['', '并发诊断：', str(concise_concurrency), '',
                  '完整算子、形状与调用信息见本目录 JSON。原始文件很大，请按字段或数组元素读取。', ''])
    atomic_write(folder / 'summary.md', '\n'.join(lines))
