"""Functional admission, candidate-local isolation, and separate fidelity evidence."""
import math
from concurrent.futures import ThreadPoolExecutor

from vf_runtime.evaluation.correctness import compare_generation, numeric_report


def distribution_report(actual, expected):
    """Full-vocabulary comparison only; sampled logits cannot define a KL."""
    if actual['indices'] != expected['indices']:
        raise ValueError('distribution_indices_mismatch')
    a, b = actual['values'], expected['values']
    if not a or len(a) != len(b) or not all(type(x) in (float, int) and math.isfinite(x) for x in a + b):
        raise ValueError('invalid_or_nonfinite_distribution')
    def logs(values):
        high = max(values)
        normalizer = high + math.log(sum(math.exp(x - high) for x in values))
        return [x - normalizer for x in values]
    la, lb = logs(a), logs(b)
    kl = max(0., sum(math.exp(y) * (y - x) for x, y in zip(la, lb)))
    tv = .5 * sum(abs(math.exp(x) - math.exp(y)) for x, y in zip(la, lb))
    ka = sorted(range(len(a)), key=a.__getitem__, reverse=True)[:10]
    kb = sorted(range(len(b)), key=b.__getitem__, reverse=True)[:10]
    return {'kl_reference_to_candidate': kl, 'total_variation': tv,
            'top1_equal': ka[0] == kb[0], 'top10_overlap': len(set(ka) & set(kb)) / len(ka),
            'candidate_top1': actual['indices'][ka[0]], 'reference_top1': expected['indices'][kb[0]],
            'scope': 'full vocabulary at identical teacher-forced prefix; diagnostic, no uncalibrated numeric gate'}


def batch_divergence(client, own, actual, margin):
    """Classify the first token where a batched row departs from the isolated output.

    bf16 kernels change reduction order with batch shape, so a greedy decode can
    flip between two almost equally likely tokens. That is numerically expected
    and must not be confused with cross-request contamination, which produces a
    token the isolated prefix considers clearly worse.
    """
    isolated, batched = own['token_ids'], actual.get('token_ids') or []
    if (actual.get('batch_size') or 1) <= 1:
        return {'near_tie': False, 'divergence_reason': 'not batched; concurrent requests must match exactly'}
    index = next((i for i, (a, b) in enumerate(zip(isolated, batched)) if a != b), None)
    if index is None:
        return {'near_tie': False, 'divergence_index': min(len(isolated), len(batched)),
                'divergence_reason': 'same prefix but different length'}
    prefix = list(own['input_ids']) + list(isolated[:index])
    values = client.logits(input_ids=prefix, indices=[isolated[index], batched[index]])['values']
    gap = float(values[0]) - float(values[1])
    return {'divergence_index': index, 'isolated_token': isolated[index], 'batched_token': batched[index],
            'logit_gap': gap, 'max_logit_gap': margin, 'near_tie': math.isfinite(gap) and abs(gap) <= margin}


def evaluate_functional(client, reference, protocol):
    checks, diagnostics, outputs = [], [], []
    vocab = reference.get('vocab_size')
    def generate(case):
        return client.generate(prompt=case['prompt'], max_new_tokens=case['max_new_tokens'],
                               **({'fixed_length': True} if case.get('fixed_length') else {}))
    def valid(output, case):
        return (isinstance(output.get('text'), str) and
                compare_generation(output, {**case, 'vocab_size': vocab}, {'correctness_mode': 'quality'}))
    for case in reference['cases']:
        try:
            first = generate(case)
            client.generate(prompt='request isolation sentinel', max_new_tokens=1)
            repeated = generate(case)
            structural = valid(first, case) and valid(repeated, case)
            isolated = structural and first['token_ids'] == repeated['token_ids']
            numerical = []
            for prefix, name in [(case['input_ids'], 'logits'),
                                 (case['input_ids'] + case['token_ids'][:2], 'continuation_logits')]:
                expected = case[name]
                actual = client.logits(input_ids=prefix, indices=expected['indices'])
                row = numeric_report(actual, expected, protocol)
                if row.get('error'):
                    raise ValueError(row['error'])
                numerical.append({'position': name, **row})
            row = {'case': case['id'], 'passed': structural and isolated,
                   'structure_passed': structural, 'sequential_isolation_passed': isolated,
                   'actual_tokens': first['token_ids'], 'repeat_tokens': repeated['token_ids']}
            checks.append(row)
            strict = compare_generation(first, case, {**protocol, 'correctness_mode': 'strict'})
            diag = {'case': case['id'], 'baseline_token_match': strict, 'sampled_numerical': numerical}
            if protocol.get('distribution_diagnostics') and vocab:
                full = []
                for suffix in ([], case['token_ids'][:2]):
                    prefix = case['input_ids'] + suffix
                    # Captured independently from the original engine before candidate evaluation.
                    key = 'continuation_distribution' if suffix else 'distribution'
                    if key not in case:
                        raise ValueError('Missing full-vocabulary baseline; recapture reference before quality-mode evaluation')
                    actual = client.logits(input_ids=prefix, indices=list(range(vocab)))
                    full.append(distribution_report(actual, case[key]))
                diag['full_distributions'] = full
            diagnostics.append(diag)
            outputs.append((case, first))
        except Exception as error:
            checks.append({'case': case['id'], 'passed': False, 'error': str(error)})
    if protocol.get('concurrency', 1) > 1 and outputs:
        try:
            # Compare with this candidate's isolated outputs, never baseline tokens.
            with ThreadPoolExecutor(max_workers=protocol['concurrency']) as pool:
                parallel = list(pool.map(lambda pair: generate(pair[0]), outputs * 2))
            margin = protocol.get('batch_divergence_max_logit_gap', 0.5)
            rows, hard = [], []
            for (case, own), actual in zip(outputs * 2, parallel):
                if valid(actual, case) and actual['token_ids'] == own['token_ids']:
                    continue
                row = {'case': case['id'], 'isolated_tokens': own['token_ids'],
                       'concurrent_tokens': actual.get('token_ids'),
                       'batch_size': actual.get('batch_size'), 'batching_mode': actual.get('batching_mode')}
                if valid(actual, case):
                    row.update(batch_divergence(client, own, actual, margin))
                else:
                    row.update(near_tie=False, divergence_reason='invalid structure or input_ids')
                rows.append(row)
                if not row.get('near_tie'):
                    hard.append(case['id'])
            checks.append({'case': 'candidate_concurrent_isolation', 'passed': not hard,
                           'mismatched_cases': hard,
                           'near_tie_divergences': [row['case'] for row in rows if row.get('near_tie')],
                           'mismatched_outputs': rows,
                           'policy': ('Batched rows (batch_size > 1) may diverge from the isolated output only at a '
                                      'numerical near-tie: at the first differing position the isolated prefix gives '
                                      f'|logit(isolated) - logit(batched)| <= {margin}. Unbatched concurrency must '
                                      'match exactly. Accepted divergences still go through the paired quality gate.')})
        except Exception as error:
            checks.append({'case': 'candidate_concurrent_isolation', 'passed': False, 'error': str(error)})
    drift = any(not d['baseline_token_match'] or any(not n['passed'] for n in d['sampled_numerical'])
                for d in diagnostics)
    return {'passed': bool(checks) and all(c['passed'] for c in checks), 'mode': 'quality',
            'requires_quality': True, 'checks': checks, 'fidelity_drift': drift,
            'diagnostics': diagnostics,
            'failure_summary': {'failed_checks': sum(not c['passed'] for c in checks),
                                'operational_errors': sum('error' in c for c in checks)},
            'not_covered': ['full layerwise numerics', 'direct cached-vs-uncached tensor comparison',
                            'unconfigured context lengths', 'long duration stability'],
            'interpretation': 'Functional pass only. Baseline drift is diagnostic; independent paired task quality must pass before performance or acceptance.'}
