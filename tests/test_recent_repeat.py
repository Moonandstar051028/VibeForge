"""Distinguish local repetition from rereads and verification after edits."""
from vibeforge_v2.agent import recent_repeat


def call(command='read', output='same', source='a', code=0, timed_out=False):
    return {'command': command, 'output': output, 'exit_code': code, 'timed_out': timed_out,
            'candidate_check': {'source_hash': source}}


def test_spaced_lifetime_rereads_do_not_trigger():
    # Five lifetime occurrences, each separated by real evidence gathering.
    rows = []
    for i in range(5):
        rows.extend(call(command=f'inspect-{i}-{j}') for j in range(10))
        rows.append(call())
    assert recent_repeat(rows) is None


def test_consecutive_unchanged_errors_are_detected():
    rows = [call(output='dimension mismatch', code=1) for _ in range(5)]
    assert recent_repeat(rows)['matching_calls'] == [1, 2, 3, 4, 5]


def test_alternating_local_reads_are_detected_at_five_in_eight():
    rows = [call(command=c) for c in ['A', 'A', 'B', 'A', 'B', 'A', 'B', 'A']]
    assert recent_repeat(rows)['matching_calls'] == [1, 2, 4, 6, 8]


def test_edit_and_revert_reset_the_history():
    rows = [call() for _ in range(4)] + [call(command='edit', source='b'), call()]
    assert recent_repeat(rows) is None


def test_unknown_source_is_not_evidence_of_no_progress():
    assert recent_repeat([call(source=None) for _ in range(8)]) is None
    assert recent_repeat([call() for _ in range(4)] + [call(source=None), call()]) is None


def test_changed_results_or_status_are_not_identical():
    rows = [call() for _ in range(4)]
    assert recent_repeat(rows + [call(output='new evidence')]) is None
    assert recent_repeat(rows + [call(code=1)]) is None
    assert recent_repeat(rows + [call(timed_out=True)]) is None


def test_quoted_whitespace_is_not_normalized():
    rows = [call(command='printf "a b"') for _ in range(4)]
    assert recent_repeat(rows + [call(command='printf "a  b"')]) is None
