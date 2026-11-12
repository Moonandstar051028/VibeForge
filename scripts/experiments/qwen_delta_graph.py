"""Candidate factory for replaying the unchanged eager single-token delta helper.

This is an experiment helper. Full-model gates remain required before adoption.
"""

import threading


def graph_delta(original, torch):
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    unused_kwargs = not any(isinstance(node, ast.Name) and node.id == 'kwargs' for node in ast.walk(tree))
    cache = {}
    lock = threading.Lock()

    def reusable(query, key, value, g, beta, initial_state=None,
                 output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if unused_kwargs:
            kwargs = {}
        if query.shape[1] != 1 or initial_state is None or kwargs or not query.is_cuda:
            return original(query, key, value, g, beta, initial_state=initial_state,
                            output_final_state=output_final_state,
                            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        inputs = (query, key, value, g, beta, initial_state)
        signature = (tuple((tuple(t.shape), t.dtype, t.device, tuple(t.stride())) for t in inputs),
                     output_final_state, use_qk_l2norm_in_kernel)
        with lock:
            if signature not in cache:
                if len(cache) >= 8:
                    return original(query, key, value, g, beta, initial_state=initial_state,
                                    output_final_state=output_final_state,
                                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
                buffers = tuple(t.detach().clone() for t in inputs)
                stream = torch.cuda.Stream(device=query.device)
                stream.wait_stream(torch.cuda.current_stream(query.device))
                options = {'initial_state': buffers[5], 'output_final_state': output_final_state,
                           'use_qk_l2norm_in_kernel': use_qk_l2norm_in_kernel}
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        original(*buffers[:5], **options)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    outputs = original(*buffers[:5], **options)
                done = torch.cuda.Event()
                done.record(stream)
                cache[signature] = buffers, graph, outputs, done
                print({'event': 'qwen_delta_graph_captured', 'signature': str(signature),
                       'original_ignores_kwargs': unused_kwargs}, flush=True)
            buffers, graph, outputs, done = cache[signature]
            current = torch.cuda.current_stream(query.device)
            current.wait_event(done)
            for target, source in zip(buffers, inputs):
                target.copy_(source)
            graph.replay()
            # Replay buffers belong to the cache, never to a layer/request.
            result = tuple(output.clone() if output is not None else None for output in outputs)
            # A second request may use a different CUDA stream. Order its
            # writes after these output copies without blocking the CPU.
            done.record(current)
            return result
    return reusable
