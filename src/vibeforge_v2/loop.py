"""The pipeline is one file-based loop; stable changes only after all gates pass."""

import fcntl
import hashlib
import json
import os
import shutil
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from vf_runtime.contracts.precision import precision_fields
from vf_runtime.contracts.workload import workload_profiles
from vf_runtime.evaluation.reference import capture_reference, load_reference
from vf_runtime.profiling.collector import collect
from vf_runtime.storage.atomic import atomic_write, digest, read_json, write_json
from vf_runtime.versioning.artifacts import inventory, source_hash

from .agent import check_connection, edit_candidate
from .environment import ExperimentIntegrityError, package_snapshot
from .feedback import write_feedback
from .evaluate import Evaluator
from .files import copy_engine, write_changes, write_history
from .related_history import write_related_history
from .identity import publish_current
from .experience import collect as collect_experience, render as render_experience, write_experience


@contextmanager
def device_lock(config):
    directory = Path(config['project']) / 'runs'
    directory.mkdir(exist_ok=True)
    key = digest([config['model']['device'], os.environ.get('CUDA_VISIBLE_DEVICES', 'all')])[:16]
    with (directory / f'.device-{key}.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another v2 run owns this device selection') from None
        yield


class Pipeline:
    def __init__(self, config, output, *, edit=edit_candidate, connect=check_connection):
        self.config, self.run = config, Path(output).resolve()
        self.edit, self.connect = edit, connect
        self.evaluator = Evaluator(config, self.run)
        self.state = {}

    def save_state(self, **updates):
        self.state.update(updates, updated_at=time.time())
        write_json(self.run / 'state.json', self.state)
        if self.state.get('stable') and self.state.get('stable_hash'):
            try:
                published = read_json(self.run / 'current/manifest.json')
            except (OSError, ValueError):
                published = {}
            if not isinstance(published, dict):
                published = {}
            if (published.get('source'), published.get('source_hash')) != (
                    self.state['stable'], self.state['stable_hash']) or (
                    self.run / 'current/engine').resolve() != (self.run / self.state['stable']).resolve():
                # Replay also repairs a crash between checkpoint and pointer writes.
                publish_current(self.run, self.state)

    def initialize(self):
        if self.run.exists():
            raise FileExistsError(f'Output already exists; choose a new directory or resume: {self.run}')
        self.run.mkdir(parents=True)
        (self.run / 'rounds').mkdir()
        config = self.config
        atomic_write(self.run / 'agent-prompt.md', (Path(config['project']) / config['agent']['prompt']).read_text())
        config['agent']['prompt'] = str(self.run / 'agent-prompt.md')
        shutil.copy2(config['workload']['cases'], self.run / 'cases.json')
        config['workload']['cases'] = str(self.run / 'cases.json')
        quality = config['evaluation'].get('quality')
        if quality:
            frozen = {}
            for original, hash_value in quality['file_hashes'].items():
                relative = Path(original).relative_to(config['project'])
                target = self.run / 'evaluation-inputs' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, target)
                quality['command'] = [part.replace('{project}/' + relative.as_posix(), str(target))
                                      for part in quality['command']]
                frozen[str(target)] = hash_value
            quality['file_hashes'] = frozen
        write_json(self.run / 'config.json', config)
        atomic_write(self.run / 'index.md', '''# 运行文件索引

- config.json：本次固定模型、硬件、精度、负载、评测和生成配置。
- profile.json / preflight.json：全局画像与本次启动检查。
- cases.json / reference.json / baseline-quality.json / evaluation-inputs/：冻结评测输入与初始基准。
- agent-prompt.md：本次系统提示。history.md：逐轮实际结论。state.json：已提交的稳定版本。
- initial/engine/：原始稳定引擎，initial/ 下为构建和基准检查。
- current/engine/、current/manifest.json、current/README.md：唯一的当前稳定引擎入口与身份标记；仅接受后切换。
- rounds/NNNN/observations/：该轮重新采集的稳定引擎性能、CPU/GPU 热点、形状和并发诊断。
- rounds/NNNN/observations/runtime.json：本轮稳定引擎的实际运行时状态。
- rounds/NNNN/agent/：输入上下文、官方框架对话、真实命令与 stdout/stderr。
- rounds/NNNN/agent/context-files.json：直接放入上下文的原始文件与 SHA256。
- rounds/NNNN/agent/checks/ / completion.json：真实磁盘静态检查与结束条件；不代表正确性或加速通过。
- rounds/NNNN/agent/context/：精确预算、语义摘要输入输出与工作记忆；context-error.json：服务端错误（发生时）。
- rounds/NNNN/engine/ / changes.patch：该轮实际候选与相对父版本修改。
- rounds/NNNN/analysis.md：Agent 的自然语言分析；不代表已经通过宿主验证。
- rounds/NNNN/probe/：候选快速预检和性能勘测，仅用于诊断。
- rounds/NNNN/correctness.json / quality/ / performance/：独立正确性、质量和配对性能样本。
- rounds/NNNN/decision.json / outcome.md：宿主实际结论和错误，下一轮可直接阅读。
- rounds/NNNN/connection/ / activation/：所选 Agent 的真实工具挑战；默认连接本地稳定引擎，Codex user-config 模式使用用户配置的独立服务。

文件中的 JSON 由宿主保存，Agent 无须提交任何结构化结果。直接编辑候选，以自然语言结束即可。
''')
        self.save_state(status='initializing', stable=None, completed_rounds=0, stage='deployment_preflight')
        profile, preflight = collect(config)
        write_json(self.run / 'profile.json', profile)
        write_json(self.run / 'preflight.json', preflight)
        if not preflight['passed']:
            raise RuntimeError(f'Deployment preflight failed: {preflight["reasons"]}')
        source = self.run / 'initial/engine'
        copy_engine(config['engine']['initial_source'], source)
        settings = read_json(source / 'engine.json')
        settings.update(precision_fields(config['precision']))
        settings.update(device=config['model']['device'], max_context=config['model']['max_context'],
                        model_loader=config['engine']['model_loader'],
                        max_new_tokens=max(config['agent']['max_new_tokens'],
                                           max(p['max_new_tokens'] for p in workload_profiles(config['workload']))),
                        thinking=config['engine'].get('thinking', config['agent']['thinking']))
        if not config['model'].get('fixture'):
            settings['model_path'] = config['model']['path']
        write_json(source / 'engine.json', settings)
        self.save_state(stage='initial_build')
        self.evaluator.environments.prepare(source)
        self.evaluator.build(source, self.run / 'initial')
        self.evaluator.environments.seal(source)
        client = self.evaluator.start(source, self.run / 'initial/engine.log')
        vocab = profile['model'].get('text_config', {}).get('vocab_size', profile['model'].get('vocab_size', 32))
        self.save_state(stage='initial_reference')
        reference = capture_reference(client, read_json(self.run / 'cases.json'), self.run / 'reference.json',
            {'source_hash': source_hash(source), 'profile_hash': profile['profile_hash'], 'vocab_size': vocab,
             'oracle': 'self_generated_regression_baseline'}, config['evaluation'], config['workload'])
        correctness = self.evaluator.correctness(client, reference, self.run / 'initial')
        self.save_state(stage='initial_quality')
        quality = self.evaluator.quality(client, source, self.run / 'initial/quality')
        write_json(self.run / 'baseline-quality.json', quality)
        if not correctness['passed'] or not quality['passed']:
            raise RuntimeError('Initial correctness or quality failed; see initial/')
        self.save_state(status='running', stable=str(source.relative_to(self.run)), stable_hash=source_hash(source))
        write_history(self.run)

    def context(self, number, folder, candidate, client):
        history_rows = [line for line in (self.run / 'history.md').read_text().splitlines()
                        if line.startswith('| ') and line.split('|')[1].strip().isdigit()]
        history = '\n'.join(history_rows[-3:]) or '尚无已结束轮次。'
        outcomes = sorted((self.run / 'rounds').glob('*/outcome.md'))
        last_outcome = outcomes[-1].read_text() if outcomes else '尚无上一轮实测结果。'
        previous_patch = outcomes[-1].with_name('changes.patch') if outcomes else None
        last_changes = previous_patch.read_text() if previous_patch and previous_patch.exists() else '尚无上一轮修改。'
        write_related_history(self.config, self.run, folder / 'agent/related-history.md')
        health = client.health()
        write_json(folder / 'observations/runtime.json', health)
        current = publish_current(self.run, self.state, health=health, round_number=number)
        files = [candidate / 'engine.json', candidate / 'engine.py']
        snapshots = {str(path): path.read_text() for path in files}
        from .interface_contract import describe_contract
        interface_contract = describe_contract(snapshots[str(candidate / 'engine.py')])
        atomic_write(folder / 'agent/interface-contract.md', interface_contract)
        write_json(folder / 'agent/context-files.json', {
            'included_files': {str(candidate / 'engine.json'): hashlib.sha256(snapshots[str(candidate / 'engine.json')].encode()).hexdigest()},
            'available_source_files': {path: hashlib.sha256(text.encode()).hexdigest() for path, text in snapshots.items()},
            'current_manifest': str(self.run / 'current/manifest.json'),
            'stable_source_hash': self.state.get('stable_hash'),
            'runtime_file': str(folder / 'observations/runtime.json'),
            'previous_outcome': str(outcomes[-1]) if outcomes else None,
            'previous_changes': str(previous_patch) if previous_patch and previous_patch.exists() else None,
        })
        engine_files = '\n'.join(f'- {name}' for name in sorted(inventory(candidate)))
        entries = collect_experience(self.run)
        experience = render_experience(entries, self.run, compact=True)
        from .library import update_library, render as render_library
        try:
            library = render_library(update_library(self.run, entries))
        except Exception as error:
            library = f'（实现库生成失败：{type(error).__name__}: {error}）\n'
        digest_path = folder / 'observations/digest.md'
        digest = digest_path.read_text() if digest_path.exists() else '（本轮没有生成瓶颈摘要；请读取 summary.md。）'
        def clip(text, limit):
            return text if len(text) <= limit else text[:limit] + f'\n...[已截断，完整内容见文件]\n'
        return f'''现在执行第 {number} 轮推理引擎优化。这是编码任务：读取源码、修改候选目录中的真实文件、做与改动相关的检查，然后用自然语言说明。

## 路径
- 候选工作目录（当前目录，UNVERIFIED_CANDIDATE，修改它不影响正在服务的模型）：{candidate}
- 当前稳定引擎固定入口（只读对照）：{self.run / 'current/engine'}；身份清单：{self.run / 'current/manifest.json'}
- 候选 Python（安装依赖用它的 python -m pip）：{self.evaluator.environments.python(candidate)}
- 稳定目标引擎 HTTP：{client.url}；Shell 默认不暴露 CUDA 设备，nvidia-smi 可用；整模型 GPU 验证由宿主完成。
- 本轮观测：{folder / 'observations/summary.md'}（概览）、{folder / 'observations/observations.json'}（完整，较大，请按字段读取）
- 经验台账全文：{self.run / 'experience.md'}；逐轮证据：{self.run / 'rounds'}/NNNN/（outcome.md、changes.patch、correctness.json、quality/report.json）
- 固定配置：{self.run / 'config.json'}；文件索引：{self.run / 'index.md'}
- 自检命令（在候选目录运行，约 1 分钟，CPU 上走真实 engine.py 路径并与稳定版本比较）：python -m vibeforge_v2.harness
- 候选文件：
{engine_files}

## 本轮接口契约（修改前已冻结）
{interface_contract}

## 固定性能目标
evaluation.objective = {self.config['evaluation'].get('objective', 'throughput')}
request_output_tokens_per_second_mean：每个请求的实际输出 token 数 / 客户端端到端耗时，包含预填充、服务端排队和生成，再取请求算术均值。多场景按固定权重取几何平均；本目标的单场景退化保护也使用该指标。throughput 则是并发总输出 token 数 / 测试总耗时，两者不得混用。

## 本轮瓶颈（宿主计算，先读这一节）
{digest}

## 历史实现与正式准入结论（CPU 检查通过不代表优化有效）
{library}
## 历史经验（宿主根据实际 diff 和判定生成）
{experience}
最近三轮（history.md）：
{history}

<HISTORICAL_CANDIDATES_NOT_CURRENT>
上一轮宿主实际反馈（节选）：
{clip(last_outcome, 2500)}

上一轮实际修改（相对该轮父版本；被拒绝时不在当前稳定源码中）：
```diff
{clip(last_changes, 2000)}
```
只使用本次运行历史；没有历史时直接依据本轮观测工作。
</HISTORICAL_CANDIDATES_NOT_CURRENT>

{current}
只有该标记块和固定入口代表当前稳定版本；加载状态不等于算子执行或加速证明。

结束时用自然语言说明：依据的瓶颈数字、改了哪段代码/配置及其机制、做了哪些检查（实际命令与结果）、尚未验证的风险。宿主随后构建、探测并独立评测。
'''

    def fixed_settings(self, stable, candidate):
        before, after = read_json(stable / 'engine.json'), read_json(candidate / 'engine.json')
        for key in ('backend', 'model_path', 'device', 'max_context', 'max_new_tokens',
                    'precision', 'dtype', 'quantization', 'thinking'):
            if before.get(key) != after.get(key):
                raise ValueError(f'Candidate changed the fixed model/deployment setting: {key}')

    def mark_stage(self, number, stage):
        self.save_state(current_round=number, stage=stage)
        print(f'round {number}: stage={stage}', flush=True)
        return stage

    def round(self, number):
        folder = self.run / 'rounds' / f'{number:04d}'
        folder.mkdir()
        stable = self.run / self.state['stable']
        if source_hash(stable) != self.state['stable_hash']:
            raise ExperimentIntegrityError('Stable source changed outside the pipeline')
        candidate = folder / 'engine'
        copy_engine(stable, candidate)
        write_json(folder / 'parent.json', {'source': self.state['stable'], 'source_hash': self.state['stable_hash']})
        decision = {'outcome': 'reject', 'reason': 'execution_failed'}
        stage = self.mark_stage(number, 'observations')
        fatal = None
        host_before = inventory(Path(self.config['project']) / 'src')
        shared_before = package_snapshot(self.config['runtime_python'])
        protected = [self.run / name for name in ('config.json', 'agent-prompt.md', 'cases.json', 'reference.json', 'baseline-quality.json')]
        protected += list((self.run / 'evaluation-inputs').rglob('*'))
        frozen = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected if path.is_file()}
        def integrity_violations():
            errors = []
            try:
                if inventory(Path(self.config['project']) / 'src') != host_before:
                    errors.append('Host source changed during round')
                if source_hash(stable) != self.state['stable_hash']:
                    errors.append('Stable source changed during round')
                if (self.run / 'current/engine').resolve() != stable.resolve():
                    errors.append('Current stable engine pointer changed during round')
                for name, expected in frozen.items():
                    if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                        errors.append('Frozen evaluation input changed: ' + name)
            except (OSError, ValueError) as error:
                errors.append('Cannot verify experiment identity: ' + str(error))
            return errors

        guard = None
        candidate_environment_ready = False
        try:
            self.evaluator.environments.verify(stable)
            self.evaluator.environments.prepare(candidate, parent=stable)
            candidate_environment_ready = True
            client = self.evaluator.start(stable, folder / 'logs/stable.log')
            self.evaluator.observe(client, stable, folder / 'observations')
            stage = self.mark_stage(number, 'connection')
            agent_config = {**self.config, 'runtime_python': self.evaluator.environments.python(candidate)}
            self.connect(client.url, agent_config, candidate, folder / 'connection')
            stage = self.mark_stage(number, 'agent')
            if self.config['agent'].get('guard_shared_dependencies', True) and not self.config['model'].get('fixture'):
                from .shared_guard import SharedGuard
                guard = SharedGuard.for_python(self.config['runtime_python'], self.run / 'cache')
                agent_config['_shared_guard'] = guard
            context = self.context(number, folder, candidate, client)
            for path in (self.run / 'current').glob('*'):
                if path.is_file():
                    frozen[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.edit(client.url, agent_config, candidate, folder / 'agent', context)
            agent_config.pop('_shared_guard', None)
            if guard is not None:
                # Also covers backends whose shell the host does not wrap (Codex).
                shared = guard.check()
                touched = shared['changed'] + shared['added'] + shared['removed']
                if touched:
                    write_json(folder / 'agent/shared-dependency-guard.json', shared)
                if shared['unrecoverable']:
                    raise ExperimentIntegrityError('Shared dependency files changed and could not be restored: '
                                                   + ', '.join(shared['unrecoverable'][:5]))
            violations = integrity_violations()
            if violations:
                raise ExperimentIntegrityError('; '.join(violations))
            self.evaluator.stop()
            stage = self.mark_stage(number, 'build')
            changed = write_changes(stable, candidate, folder / 'changes.patch')
            if not changed:
                decision = {'outcome': 'reject', 'reason': 'no_source_changes'}
                return decision
            self.fixed_settings(stable, candidate)
            from .agent import harness_enabled, run_harness
            if harness_enabled(agent_config):
                stage = self.mark_stage(number, 'cpu_harness')
                harness = run_harness(agent_config, candidate, folder / 'agent', source_hash(candidate))
                if not harness.get('allow_gpu_evaluation', harness['passed']):
                    decision = {'outcome': 'reject', 'reason': 'harness_failed',
                                'error': harness['errors'][0][:600] if harness['errors'] else 'harness failed'}
                    return decision
            self.evaluator.build(candidate, folder)
            self.fixed_settings(stable, candidate)
            violations = integrity_violations()
            if violations:
                raise ExperimentIntegrityError('; '.join(violations))
            self.evaluator.environments.seal(candidate)
            evaluated_hash = source_hash(candidate)
            reference = load_reference(self.run / 'reference.json')
            stage = self.mark_stage(number, 'quick_probe')
            client = self.evaluator.start(candidate, folder / 'logs/candidate.log')
            try:
                self.evaluator.probe(client, candidate, folder / 'probe', reference)
            except Exception as error:
                write_json(folder / 'probe/probe.json', {'admission': False, 'error': str(error)})
            stage = self.mark_stage(number, 'correctness')
            correctness = self.evaluator.correctness(client, reference, folder)
            if not correctness['passed']:
                decision = {'outcome': 'reject', 'reason': 'correctness_failed'}
                return decision
            stage = self.mark_stage(number, 'quality')
            quality = self.evaluator.quality(client, candidate, folder / 'quality', read_json(self.run / 'baseline-quality.json'))
            correctness['quality_passed'] = quality['passed']
            write_json(folder / 'correctness.json', correctness)
            if not quality['passed']:
                decision = {'outcome': 'reject', 'reason': 'quality_failed'}
                if self.config['evaluation'].get('quality_failure_diagnostics', True):
                    stage = self.mark_stage(number, 'diagnostic_performance')
                    try:
                        diagnostic = self.evaluator.diagnose_performance(stable, candidate, reference,
                                                                        folder / 'diagnostic-performance')
                        decision['diagnostic_performance'] = {
                            key: diagnostic[key] for key in ('admission', 'valid', 'gain', 'pairs')}
                    except ExperimentIntegrityError:
                        raise
                    except Exception as error:
                        write_json(folder / 'diagnostic-performance/report.json', {
                            'kind': 'exploratory_performance', 'admission': False,
                            'quality_passed': False, 'valid': False, 'error': str(error)})
                        decision['diagnostic_error'] = f'{type(error).__name__}: {error}'
                return decision
            stage = self.mark_stage(number, 'performance')
            decision = self.evaluator.performance(stable, candidate, reference, folder / 'performance', correctness)
            if decision['outcome'] == 'accept':
                stage = self.mark_stage(number, 'activation')
                client = self.evaluator.start(candidate, folder / 'logs/activation.log')
                self.connect(client.url, agent_config, candidate, folder / 'activation')
                self.evaluator.environments.verify(candidate)
                if source_hash(candidate) != evaluated_hash:
                    raise RuntimeError('Candidate changed after evaluation; refusing to promote untested files')
                decision.update(promoted_source=str(candidate.relative_to(self.run)), promoted_hash=evaluated_hash)
            return decision
        except Exception as error:
            atomic_write(folder / 'error.log', traceback.format_exc())
            if isinstance(error, ExperimentIntegrityError):
                fatal = str(error)
            decision = {'outcome': 'reject', 'reason': 'execution_failed', 'stage': stage,
                        'error': f'{type(error).__name__}: {error}'}
            return decision
        except BaseException:
            decision = {'outcome': 'interrupted', 'reason': 'interrupted_during_round', 'stage': stage}
            raise
        finally:
            self.evaluator.stop()
            try:
                write_changes(stable, candidate, folder / 'changes.patch')
            except Exception as error:
                decision['diff_error'] = str(error)
            violations = integrity_violations()
            if guard is not None:
                try:
                    shared = guard.check()
                    if shared['unrecoverable']:
                        violations.append('Shared dependency files changed and could not be restored: '
                                          + ', '.join(shared['unrecoverable'][:5]))
                except Exception as error:
                    violations.append('Cannot verify shared dependency files: ' + str(error))
            try:
                self.evaluator.environments.verify(stable)
            except Exception as error:
                violations.append('Cannot verify stable dependencies: ' + str(error))
            if candidate_environment_ready:
                try:
                    candidate_after = package_snapshot(self.evaluator.environments.python(candidate))
                except Exception as error:
                    candidate_after = {'error': str(error)}
                    violations.append('Cannot inspect candidate dependencies: ' + str(error))
            else:
                candidate_after = {'status': 'not_prepared',
                                   'scope': 'Candidate dependency preparation did not finish; no candidate was evaluated.'}
            shared_after = package_snapshot(self.config['runtime_python'])
            write_json(folder / 'environment.json', {'shared_before': shared_before, 'shared_after': shared_after,
                       'candidate_python': self.evaluator.environments.python(candidate), 'candidate_after': candidate_after})
            if shared_after['hash'] != shared_before['hash']:
                violations.append('Shared Python dependencies changed; source rollback cannot restore environment')
            if violations or fatal:
                fatal = '; '.join(violations + ([fatal] if fatal else []))
                decision.update(outcome='reject', reason='experiment_integrity_failed', error=fatal)
            promoted = decision.get('promoted_source') if decision['outcome'] == 'accept' else None
            decision.update(round=number, stage=stage, stable_after=promoted or self.state['stable'],
                            stable_after_hash=decision.get('promoted_hash') if promoted else self.state['stable_hash'])
            # The durable decision is the commit record. Recovery replays this record
            # if power is lost before the state checkpoint is updated.
            write_json(folder / 'decision.json', decision)
            write_feedback(folder, decision)
            self.save_state(stable=decision['stable_after'], stable_hash=decision['stable_after_hash'],
                            completed_rounds=number)
            write_history(self.run)
            try:
                write_experience(self.run)
            except Exception as error:  # diagnostic aid only; never blocks the commit record
                print(f'experience ledger not updated: {type(error).__name__}: {error}', flush=True)
            if fatal:
                raise ExperimentIntegrityError(fatal)

    def execute(self, rounds=None, resume=False, *, continuous=None):
        if rounds is not None and (type(rounds) is not int or rounds < 1):
            raise ValueError('rounds must be a positive integer')
        if continuous is not None and type(continuous) is not bool:
            raise ValueError('continuous must be boolean')
        if continuous and rounds is not None:
            raise ValueError('continuous and an explicit round limit are mutually exclusive')
        if rounds is not None and continuous is None:
            continuous = False
        with device_lock(self.config):
            try:
                if resume:
                    self.state = read_json(self.run / 'state.json')
                    if self.state.get('integrity_failed'):
                        raise ExperimentIntegrityError('Run previously failed integrity checks; inspect evidence and start a new run')
                    if not self.state.get('stable'):
                        raise RuntimeError('Initialization incomplete; start a new run')
                    # A killed round is preserved and marked interrupted; resume at its successor.
                    for folder in sorted((self.run / 'rounds').glob('*')):
                        number = int(folder.name)
                        if number > self.state['completed_rounds']:
                            path = folder / 'decision.json'
                            if path.exists():
                                decision = read_json(path)
                                stable = decision.get('stable_after', self.state['stable'])
                                expected = decision.get('stable_after_hash')
                                if not expected or source_hash(self.run / stable) != expected:
                                    raise ExperimentIntegrityError('Cannot replay committed round: missing or mismatched stable identity')
                                self.save_state(stable=stable, stable_hash=expected, completed_rounds=number)
                                if decision.get('reason') == 'experiment_integrity_failed':
                                    raise ExperimentIntegrityError(decision.get('error', 'Integrity failure requires inspection'))
                            else:
                                write_json(path, {'round': number, 'outcome': 'interrupted',
                                    'reason': 'not_committed_before_restart', 'stable_after': self.state['stable'],
                                    'stable_after_hash': self.state['stable_hash']})
                                self.save_state(completed_rounds=number)
                    write_history(self.run)
                else:
                    if continuous is None:
                        continuous = self.config.get('run_control', {}).get('continuous', False)
                    self.config['run_control'] = {
                        'continuous': continuous,
                        'round_limit': None if continuous else (rounds if rounds is not None else self.config['rounds'])}
                    self.initialize()
                if continuous is None:
                    continuous = self.state.get('continuous', self.config.get('run_control', {}).get('continuous', False))
                limit = None if continuous else (rounds if rounds is not None else self.config['rounds'])
                self.save_state(status='running', continuous=continuous, round_limit=limit)
                number = self.state['completed_rounds'] + 1
                while limit is None or number <= limit:
                    print(f'round {number}: survey -> agent -> build/probe -> correctness/quality/performance', flush=True)
                    decision = self.round(number)
                    self.save_state(completed_rounds=number)
                    print(f"round {number}: {decision['outcome']} / {decision['reason']}", flush=True)
                    number += 1
                self.save_state(status='completed', stage='completed')
                return self.state
            except BaseException as error:
                if isinstance(error, ExperimentIntegrityError):
                    self.save_state(integrity_failed=True)
                if self.state:
                    self.save_state(status='interrupted')
                raise
            finally:
                self.evaluator.stop()
