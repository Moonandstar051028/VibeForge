"""Summarize saved host evidence and real Agent tool outcomes; never execute candidates."""
import argparse
import json
from pathlib import Path


def read(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def audit(run):
    run = Path(run).resolve()
    rounds = []
    for folder in sorted((run / 'rounds').glob('[0-9]*')):
        records = []
        trace = folder / 'agent/tools.jsonl'
        if trace.exists():
            for line in trace.read_text().splitlines():
                records.append(json.loads(line))
        shell = [row for row in records if 'command' in row]
        quality = read(folder / 'quality/report.json', {})
        checks = read(folder / 'agent/completion.json', {})
        correctness = read(folder / 'correctness.json', {})
        observation = read(folder / 'observations/observations.json', {})
        runtime = read(folder / 'logs/candidate.runtime.json', {}).get('runtime', {})
        rounds.append({
            'round': folder.name, 'decision': read(folder / 'decision.json'),
            'parent': read(folder / 'parent.json'),
            'context_files': read(folder / 'agent/context-files.json'),
            'agent_trace_exists': trace.exists(), 'tool_records': len(records), 'shell_commands': len(shell),
            'failed_commands': [{'command': row['command'], 'exit_code': row.get('exit_code'),
                                 'timed_out': row.get('timed_out', False), 'output_tail': row.get('output','')[-1600:]}
                                for row in shell if row.get('exit_code') != 0],
            'completion': checks,
            'candidate_runtime': {key: runtime.get(key) for key in (
                'execution_path', 'request_scheduler', 'continuous_batching', 'batching_mode',
                'attention_implementations', 'requested')},
            'correctness': {key: correctness.get(key) for key in ('passed','failure_summary','fidelity_drift')},
            'failed_functional_checks': [row for row in correctness.get('checks',[]) if not row.get('passed')],
            'quality': {key: quality.get(key) for key in ('passed','repeat_count','planned_repeat_count','early_stop','metrics','paired','errors')},
            'survey': {'seconds': observation.get('survey_wall_seconds'),
                       'has_module_evidence': any(row.get('phases') for row in observation.get('profiles', [])),
                       'implementation_symbols': [row.get('symbol') for row in observation.get('implementation_excerpts',[])]},
            'formal_performance_attempts': len(list((folder / 'performance').glob('attempt-*.json'))),
            'exploratory_performance': read(folder / 'diagnostic-performance/report.json'),
            'analysis_path': str(folder / 'analysis.md'),
        })
    baseline = read(run / 'baseline-quality.json', {})
    return {'run': str(run), 'state': read(run / 'state.json'),
            'initial': {'correctness_passed': read(run / 'initial/correctness.json', {}).get('passed'),
                        'quality_passed': baseline.get('passed'),
                        'quality_repeats': baseline.get('repeat_count'),
                        'quality_metrics': baseline.get('metrics')},
            'rounds': rounds,
            'interpretation': 'Command success, static completion and model prose are not proof of mechanism correctness or measured gains. Read actual diff, assertions, independent reports and paired samples.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from vf_runtime.storage.atomic import write_json
    write_json(args.output, audit(args.run))
