from vibeforge.evaluation.correctness import evaluate_candidate_precheck


class Client:
    def __init__(self):
        self.concurrent_calls = 0

    def generate(self, **arguments):
        # Sequential repeat agrees with the frozen output; batched execution
        # diverges after the first token as it did in the 27B probe.
        if arguments["prompt"] == "request isolation sentinel":
            tokens = [9]
        else:
            self.concurrent_calls += 1
            tokens = [1, 2, 3] if self.concurrent_calls <= 3 else [1, 8, 3]
        return {"input_ids": [10, 11], "token_ids": tokens,
                "host_observation": {"batch_size": 2, "batching_mode": "static_microbatch"}}

    def profile(self, **arguments):
        return {"normal_request": {"output_tokens": 2}}


def test_candidate_precheck_records_bounded_concurrent_token_difference():
    reference = {"cases": [{"prompt": "prompt", "input_ids": [10, 11],
                            "max_new_tokens": 3, "token_ids": [1, 2, 3]}],
                 "benchmark_cases": [{"prompt": "prompt", "input_ids": [10, 11],
                                      "max_new_tokens": 3, "token_ids": [1, 2, 3]}]}

    report = evaluate_candidate_precheck(Client(), reference, {"concurrency": 2})

    check = next(row for row in report["checks"] if row["case"] == "concurrent_request_isolation")
    assert not check["passed"]
    assert check["observed_batch_sizes"] == [2, 2]
    assert check["batching_modes"] == ["static_microbatch", "static_microbatch"]
    assert check["output_differences"] == [
        {"input_ids_match": True, "output_length": 3, "reference_length": 3,
         "first_mismatch_index": None, "actual_prefix": [1, 2, 3], "reference_prefix": [1, 2, 3]},
        {"input_ids_match": True, "output_length": 3, "reference_length": 3,
         "first_mismatch_index": 1, "actual_prefix": [1, 8, 3], "reference_prefix": [1, 2, 3]},
    ]


def test_precheck_exercises_distinct_frozen_inputs_and_reports_matching_reference():
    import threading
    barrier = threading.Barrier(2)
    short = {'prompt': 'short', 'input_ids': [1], 'token_ids': [2], 'max_new_tokens': 1}
    long = {'prompt': 'long', 'input_ids': [3] * 20, 'token_ids': [4], 'max_new_tokens': 1}
    class HeterogeneousClient:
        def generate(self, **arguments):
            if threading.current_thread() is not threading.main_thread():
                barrier.wait(timeout=2)
            case = long if arguments['prompt'] == 'long' else short
            return {'input_ids': case['input_ids'], 'token_ids': [9] if case is long else case['token_ids']}
        def profile(self, **arguments):
            return {}
    report = evaluate_candidate_precheck(HeterogeneousClient(),
        {'cases': [short, long], 'benchmark_cases': [short]}, {'concurrency': 2})
    check = next(c for c in report['checks'] if c['case'] == 'concurrent_request_isolation')
    assert not check['passed']
    assert check['input_lengths'] == [1, 20]
    assert check['output_differences'][1]['reference_prefix'] == [4]
    assert check['output_differences'][1]['input_ids_match'] is True
