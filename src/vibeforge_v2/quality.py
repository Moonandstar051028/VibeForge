"""Paired item-level quality evidence; repeated completions are not new tasks."""
import math


def loss_upper(losses, count, confidence):
    """Exact one-sided binomial upper bound (conservative: gains never offset losses)."""
    if not count or losses >= count:
        return 1.
    low, high = losses / count, 1.
    for _ in range(60):
        p = (low + high) / 2
        terms = [math.lgamma(count + 1) - math.lgamma(k + 1) - math.lgamma(count - k + 1)
                 + k * math.log(p) + (count - k) * math.log1p(-p) for k in range(losses + 1)]
        cdf = sum(math.exp(x) for x in terms)
        if cdf > 1 - confidence:
            low = p
        else:
            high = p
    return high


def paired_quality(baseline, candidate, settings):
    def collapse(report):
        runs = report.get('runs') or [report]
        groups = [run.get('cases', []) for run in runs]
        if not groups or any(not group for group in groups):
            raise ValueError('Missing item-level quality outcomes')
        identifiers = [row['id'] for row in groups[0]]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('Duplicate quality task IDs')
        if any([row['id'] for row in group] != identifiers for group in groups):
            raise ValueError('Repeated quality task coverage mismatch')
        return {name: all(group[i]['passed'] for group in groups) for i, name in enumerate(identifiers)}
    a, b = collapse(baseline), collapse(candidate)
    if set(a) != set(b):
        raise ValueError('Paired quality task coverage mismatch')
    losses = [key for key in a if a[key] and not b[key]]
    gains = [key for key in a if not a[key] and b[key]]
    upper = loss_upper(len(losses), len(a), settings.get('confidence', .95))
    mode = settings.get('decision_mode', 'confidence_bound')
    if mode not in {'confidence_bound', 'observed_loss'}:
        raise ValueError('Invalid paired quality decision_mode')
    observed = len(losses) / len(a)
    statistic = observed if mode == 'observed_loss' else upper
    return {'passed': len(a) >= settings.get('min_cases', 100) and statistic <= settings.get('max_loss_rate', .05),
            'unique_tasks': len(a), 'losses': losses, 'gains': gains,
            'decision_mode': mode, 'observed_loss_rate': observed,
            'confidence_bound_satisfied': upper <= settings.get('max_loss_rate', .05),
            'evidence_scope': ('Fixed-suite empirical screening; not a confidence-bound non-regression guarantee.'
                               if mode == 'observed_loss' else 'One-sided binomial confidence bound.'),
            'loss_rate_upper': upper, 'confidence': settings.get('confidence', .95),
            'max_loss_rate': settings.get('max_loss_rate', .05),
            'assumption': 'Binomial bound assumes representative independent tasks; generated template families limit generalization. Repeats collapse to one task, gains do not cancel losses.'}
