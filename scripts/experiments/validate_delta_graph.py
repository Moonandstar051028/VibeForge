"""Codex-assisted isolated validation of the measured eager graph recipe.

Uses an existing stopped campaign's frozen gates/reference without changing its
stable version or iteration records. Reports are explicitly not promotions.
"""
import argparse
import json
import time
from pathlib import Path

from vibeforge.contracts.workload import workload_profiles
from vibeforge.controller.admission import decide
from vibeforge.engine.process import EngineProcess
from vibeforge.evaluation.correctness import evaluate_correctness
from vibeforge.evaluation.performance import measure_suite, measurement_mode
from vibeforge.evaluation.quality import run_quality
from vibeforge.evaluation.reference import load_reference
from vibeforge.profiling.collector import collect
from vibeforge.storage.atomic import read_json, write_json
from vibeforge.storage.database import Database
from vibeforge.storage.locks import exclusive_lock
from vibeforge.storage.paths import Paths
from vibeforge.versioning.candidate import create_workspace, write_diff
from vibeforge.versioning.release import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    paths = Paths.create(project)
    directory = paths.campaign(args.campaign)
    config = read_json(directory / 'manifest.json')
    database = Database(paths.data / 'state/controller.sqlite3')
    engine = None
    try:
        with exclusive_lock(paths.data / 'state' / f'{args.campaign}.lock'), exclusive_lock(paths.data / 'state/device.lock'):
            state = database.get(args.campaign)
            if state['state'] not in {'STOPPED', 'HALTED'}:
                raise ValueError('Stop the source campaign before isolated validation')
            profile, preflight = collect(config)
            if not preflight['passed'] or profile['profile_hash'] != state['profile_hash']:
                raise ValueError('Frozen environment no longer matches')
            folder = directory / 'isolated-validation' / str(time.time_ns())
            workspace = folder / 'candidate'
            create_workspace(paths.release(state['initial_ref']), workspace)
            source = (workspace / 'engine.py').read_text()
            factory = (project / 'scripts/experiments/qwen_delta_graph.py').read_text()
            factory = factory[factory.index('def graph_delta('):]
            source = source.replace('def load(config):', factory + '\n\ndef load(config):', 1)
            install = ('    import inspect\n'
                       '    import transformers.models.qwen3_5.modeling_qwen3_5 as model_impl\n'
                       '    model_impl.torch_recurrent_gated_delta_rule = graph_delta(\n'
                       '        inspect.unwrap(model_impl.torch_recurrent_gated_delta_rule), torch)\n')
            marker = '    TOKENIZER = AutoTokenizer.from_pretrained('
            assert source.count(marker) == 1
            source = source.replace(marker, install + marker, 1)
            (workspace / 'engine.py').write_text(source)
            compile(source, 'engine.py', 'exec')
            write_diff(paths.release(state['initial_ref']), workspace, folder / 'change.patch')
            manifest = freeze(workspace, paths, state['initial_ref'], config['runtime_python'], config['config_hash'])
            version = manifest['version_id']
            write_json(folder / 'binding.json', {'candidate': version, 'initial_ref': state['initial_ref'],
                       'config_hash': config['config_hash'], 'profile_hash': state['profile_hash'],
                       'origin': 'Codex root-assisted recipe implementation; independent verification, no promotion'})
            print(json.dumps({'folder': str(folder), 'candidate': version}), flush=True)
            engine = EngineProcess(paths, args.campaign + '-isolated', config['runtime_python'],
                                   config['startup_timeout'], config['request_timeout'], config.get('inference_pythonpath', []))
            reference = load_reference(directory / 'reference.json')
            client = engine.start(paths.release(version), folder / 'correctness.log')
            correctness = evaluate_correctness(client, reference, config['evaluation'])
            write_json(folder / 'correctness.json', correctness)
            print('correctness', correctness['passed'], flush=True)
            if not correctness['passed']:
                return
            quality = run_quality(client, config['evaluation']['quality'], folder / 'quality', {
                'project': str(project), 'campaign_id': args.campaign, 'version': version,
                'source_hash': manifest['source_hash'], 'config_hash': config['config_hash'],
                'profile_hash': state['profile_hash'], 'precision': config['precision'],
            }, read_json(directory / 'initial-quality.json'))
            print('quality', quality['passed'], flush=True)
            if not quality['passed']:
                return
            profiles = workload_profiles(config['workload'])
            selected = next(p for p in profiles if p['id'] == config['workload']['observation_profile'])
            observation = client.profile(prompt=selected['prompts'][0], max_new_tokens=16)
            write_json(folder / 'candidate-observation.json', observation)
            print('profile', observation.get('diagnostics'), flush=True)
            engine.stop()
            baseline, candidate = [], []
            deadline = time.monotonic() + config['budget']['max_seconds']
            for group in range(config['evaluation']['repeats']):
                order = [(state['initial_ref'], baseline), (version, candidate)]
                if group % 2:
                    order.reverse()
                for ref, output in order:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Isolated validation budget exhausted')
                    client = engine.start(paths.release(ref), folder / f'benchmark-{group}-{ref}.log')
                    for _ in range(config['evaluation']['warmup']):
                        for p in profiles:
                            for prompt in p['prompts']:
                                client.generate(prompt=prompt, max_new_tokens=p['max_new_tokens'], fixed_length=True,
                                                observation_mode=measurement_mode(config['evaluation']))
                    m = read_json(paths.release(ref) / 'manifest.json')
                    sample = measure_suite(client, config['workload'], folder / 'samples.jsonl', ref, group,
                                           config['evaluation'], (engine.process.pid,), reference=reference,
                                           startup_seconds=client.startup_seconds,
                                           source_bytes=sum((paths.release(ref) / name).stat().st_size for name in m['files']))
                    output.append(sample)
                    print('measurement', group, ref, sample['metrics']['throughput'], sample['valid'], flush=True)
                    engine.stop()
            decision = decide(correctness, baseline, candidate, config['evaluation']).to_dict()
            write_json(folder / 'report.json', {'binding': read_json(folder / 'binding.json'), 'decision': decision,
                       'baseline': baseline, 'candidate': candidate, 'quality': quality,
                       'scope': 'Independent validation of a Codex-assisted candidate. No campaign promotion.'})
            print(json.dumps(decision), flush=True)
    finally:
        if engine:
            engine.stop()
        database.close()


if __name__ == '__main__':
    main()
