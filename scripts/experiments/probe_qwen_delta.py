"""Qwen3.5 decode component diagnostic, not an end-to-end acceptance benchmark.

Run with the inference interpreter and CUDA_VISIBLE_DEVICES set to an idle GPU.
The final stdout JSON records component errors and warm timings; compilation and
model loading are deliberately outside those timings.
"""

import inspect
import json
import time
import argparse

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clone-static-state', action='store_true',
                        help='Validate 48 marked cache states and include state-copy cost in timing')
    parser.add_argument('--preserve-eager-rounding', action='store_true',
                        help='Preserve BF16/FP16 intermediate roundings in compiled fused operations')
    parser.add_argument('--cuda-graph', action='store_true', help='Replay original eager operations without compiler fusion')
    options = parser.parse_args()
    original = inspect.unwrap(modeling_qwen3_5.torch_recurrent_gated_delta_rule)
    compiled = torch.compile(original, fullgraph=True, dynamic=False,
                             options={"emulate_precision_casts": True} if options.preserve_eager_rounding else None)
    def reusable(query, key, value, g, beta, initial_state=None, **kwargs):
        state = initial_state.clone() if initial_state is not None else None
        return compiled(query, key, value, g, beta, initial_state=state, **kwargs)
    optimized = reusable if options.clone_static_state else compiled
    if options.cuda_graph:
        from qwen_delta_graph import graph_delta
        optimized = graph_delta(original, torch)
    records = []
    for batch in (1, 2):
        torch.manual_seed(123)
        query = torch.randn(batch, 1, 48, 128, device="cuda", dtype=torch.bfloat16)
        key, value = torch.randn_like(query), torch.randn_like(query)
        gate = -torch.rand(batch, 1, 48, device="cuda")
        beta = torch.rand_like(gate)
        state = torch.randn(batch, 48, 128, 128, device="cuda")
        args = (query, key, value, gate, beta)
        kwargs = {"initial_state": state, "output_final_state": True,
                  "use_qk_l2norm_in_kernel": True, "cu_seqlens": None}
        with torch.inference_mode():
            if options.clone_static_state:
                torch._dynamo.mark_static_address(state)
            expected, actual = original(*args, **kwargs), optimized(*args, **kwargs)
            torch.cuda.synchronize()
            errors = [(left - right).abs().max().item() for left, right in zip(expected, actual)]
            if options.clone_static_state:
                # Retain distinct marked states to exercise guards across layers.
                states = [torch.randn_like(state) for _ in range(48)]
                for cache_state in states:
                    torch._dynamo.mark_static_address(cache_state)
                    inputs = {**kwargs, 'initial_state': cache_state}
                    expected, actual = original(*args, **inputs), optimized(*args, **inputs)
                    errors = [max(error, (left - right).abs().max().item())
                              for error, left, right in zip(errors, expected, actual)]
                if options.cuda_graph:
                    streams = [torch.cuda.Stream() for _ in range(2)]
                    for stream in streams:
                        stream.wait_stream(torch.cuda.current_stream())
                    outputs = []
                    for index, cache_state in enumerate(states):
                        with torch.cuda.stream(streams[index % 2]):
                            outputs.append(optimized(*args, **{**kwargs, 'initial_state': cache_state}))
                    torch.cuda.synchronize()
                    for cache_state, actual in zip(states, outputs):
                        expected = original(*args, **{**kwargs, 'initial_state': cache_state})
                        errors = [max(error, (left - right).abs().max().item())
                                  for error, left, right in zip(errors, expected, actual)]
            timings = []
            for implementation in (original, optimized):
                for _ in range(5):
                    implementation(*args, **kwargs)
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(200):
                    implementation(*args, **kwargs)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) / 200)
        records.append({"batch": batch, "max_abs_errors": errors, "seconds": timings,
                        "speedup": timings[0] / timings[1]})
    print(json.dumps({"kind": "component_microbenchmark", "records": records,
                      "seed": 123, "iterations": 200,
                      "clone_static_state": options.clone_static_state,
                      "preserve_eager_rounding": options.preserve_eager_rounding,
                      "cuda_graph": options.cuda_graph,
                      "cross_stream_requests": 48 if options.cuda_graph and options.clone_static_state else 0,
                      "distinct_marked_states": 48 if options.clone_static_state else 0,
                      "end_to_end": "unverified"}, indent=2))


if __name__ == "__main__":
    main()
