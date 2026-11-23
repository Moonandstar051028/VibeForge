"""Host-owned CPU equivalence harness: candidate vs stable on a reduced real-architecture model.

Why: the Agent's own self-checks used hand-written mock shapes and passed,
while the real request path then failed on the GPU (index rank, view/stride,
OOM, wrong math) only after ~20 minutes of build, load and correctness work,
and after the Agent's conversation had already ended. This harness runs the
actual candidate engine.py through the same import + load/generate/logits/
chat_completion path as the server, on CPU, in about a minute, and reports
the real exception or numeric error so the Agent can fix it in-conversation.

Scope (stated in every report): reduced model (2 layers, <=64 experts,
random weights, real hidden/expert widths, real tokenizer and vocabulary),
CPU only. It proves runtime validity and float32 numerical equivalence of
the patched code paths; it does not prove GPU speed or GPU-only kernels.

Usage (from the candidate directory):  python -m vibeforge_v2.harness
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GPU_ONLY_MARKERS = ('cuda', 'triton', 'nvidia', 'torch not compiled with', 'no kernel image', 'flash_attn',
                    'flashattention', 'cudnn', 'nccl')
SCOPE = ('CPU, reduced model: 2 layers (and at most 64 of any repeated sub-module such as experts), random weights, real widths/tokenizer/vocabulary. '
         'Checks the real engine.py load/generate/logits/chat_completion path. Not a GPU speed measurement.')


CHECK_VERSION = 6  # CPU substitutions do not verify the requested GPU implementation.


def forward_fingerprint(function):
    """Compare the callable body; inspect.getsource(function) follows __wrapped__.

    A decorated expert forward may dispatch to grouped-mm. Copying its eager
    definition bypasses that wrapper and is an actual runtime change.
    """
    import ast
    import inspect
    import textwrap
    function = getattr(function, '__func__', function)
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(function, '__code__', function))))
        for node in ast.walk(tree):
            if isinstance(node, ast.arg):
                node.annotation = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.name = 'f'
                node.returns = None
                node.decorator_list = [d for d in node.decorator_list
                                       if not (isinstance(d, ast.Call) and 'deprecate' in ast.unparse(d.func))]
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], 'value', None), ast.Constant):
                    node.body = node.body[1:] or [ast.Pass()]
        return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()[:16]
    except Exception:
        return None


def matmul_flop_counter():
    """Include arithmetic hidden inside the Transformers grouped-mm custom op.

    Torch's counter cannot see the inner mm calls of this opaque operator. Its
    final offset is the number of processed rows (input may contain sentinel
    rows), so 2 * rows * input_width * output_width matches the eager loop.
    The mapping is local to this counter, without changing the shared registry.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    mapping = {}
    if hasattr(torch.ops.transformers, 'grouped_mm_fallback'):
        def grouped_mm_flops(input, weight, offs, *, out_val=None):
            rows = int(offs[-1].item()) if offs.numel() else 0
            return 2 * rows * weight.shape[1] * weight.shape[2]
        grouped_mm_flops._get_raw = True
        mapping[torch.ops.transformers.grouped_mm_fallback] = grouped_mm_flops
    return FlopCounterMode(display=False, custom_mapping=mapping)


def sampling_contract(engine):
    """Observe effective sampling mode on real requests, not candidate helper declarations."""
    model = getattr(engine, 'MODEL', None)
    if model is None or not callable(getattr(model, 'generate', None)):
        return {}
    original = model.generate
    observed = []
    def generate(*args, **kwargs):
        config = kwargs.get('generation_config') or getattr(model, 'generation_config', None)
        observed.append(bool(kwargs.get('do_sample', getattr(config, 'do_sample', False))))
        return original(*args, **kwargs)
    model.generate = generate
    try:
        result = {}
        for name, options in [('default', None), ('greedy', {'temperature': 0}),
                              ('sampled', {'temperature': 0.7})]:
            observed.clear()
            engine.generate('Hello', max_new_tokens=1, generation_options=options)
            result[name] = sorted(set(observed))
        return result
    finally:
        model.generate = original


def _run_root(candidate):
    candidate = Path(candidate).resolve()
    return candidate.parents[2] if candidate.parent.parent.name == 'rounds' else None


def build_reduced_model(model_path, cache):
    """Random-weight reduced copy of the real architecture; cached by config hash."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    import torch
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    text = getattr(config, 'text_config', config)
    overrides = {'num_hidden_layers': 2}
    if getattr(text, 'num_experts', None):
        overrides['num_experts'] = min(text.num_experts, 64)
    if getattr(text, 'max_window_layers', None):
        overrides['max_window_layers'] = 2
    if hasattr(text, 'mlp_only_layers'):
        overrides['mlp_only_layers'] = []
    if isinstance(getattr(text, 'layer_types', None), list):
        overrides['layer_types'] = list(text.layer_types[:2])
    multimodal = any('ForConditionalGeneration' in name for name in (config.architectures or []))
    key = hashlib.sha256(json.dumps({'config': config.to_dict(), 'overrides': overrides, 'v': 2},
                                    sort_keys=True, default=str).encode()).hexdigest()[:16]
    target = Path(cache) / f'reduced-{key}'
    if (target / 'ready').exists():
        return target
    staging = Path(str(target) + f'.tmp-{os.getpid()}')
    shutil.rmtree(staging, ignore_errors=True)
    for name, value in overrides.items():
        setattr(text, name, value)
    torch.manual_seed(1234)
    loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM
    model = loader.from_config(config, torch_dtype=torch.float32)
    model.save_pretrained(staging, safe_serialization=True)
    for path in Path(model_path).iterdir():
        if path.is_file() and (path.name.startswith(('tokenizer', 'vocab', 'merges', 'special_tokens'))
                               or path.name in ('generation_config.json', 'chat_template.jinja')):
            shutil.copy2(path, staging / path.name)
    (staging / 'ready').write_text(json.dumps(overrides))
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)
    return target


WORKER = r'''
import importlib.util, json, sys, time, threading, traceback, copy
directory, model_dir, dtype, output = sys.argv[1:5]
report = {"dtype": dtype, "notes": []}
def finish():
    json.dump(report, open(output, "w"))
try:
    import torch
    torch.manual_seed(0)
    sys.path.insert(0, directory)
    config = json.load(open(directory + "/engine.json"))
    config.update(model_path=model_dir, device="cpu", dtype=dtype, num_threads=min(16, config.get("num_threads", 8) * 2),
                  precision={"weight_dtype": dtype, "compute_dtype": dtype, "quantization": "none",
                             "kv_cache_dtype": "auto", "accumulation_dtype": "auto", "double_quant": True},
                  quantization=None, max_context=4096, max_new_tokens=64)
    if config.get("attn_implementation") not in (None, "sdpa", "eager"):
        report["notes"].append("attn_implementation %s replaced by sdpa on CPU" % config["attn_implementation"])
        report["cpu_config_overrides"] = {"attn_implementation": {
            "requested": config["attn_implementation"], "tested": "sdpa"}}
        config["attn_implementation"] = "sdpa"
    import importlib, inspect
    originals = {}
    try:
        model_type = json.load(open(model_dir + "/config.json")).get("model_type", "")
        library = importlib.import_module("transformers.models.%s.modeling_%s" % (model_type, model_type))
        for name, value in vars(library).items():
            if inspect.isclass(value) and issubclass(value, torch.nn.Module) and "forward" in vars(value):
                originals[value] = vars(value)["forward"]
    except Exception as error:
        report["notes"].append("could not snapshot library forwards: %s" % error)
    spec = importlib.util.spec_from_file_location("candidate_engine", directory + "/engine.py")
    engine = importlib.util.module_from_spec(spec)
    stage = "import engine.py"
    spec.loader.exec_module(engine)
    stage = "load()"
    engine.load(copy.deepcopy(config))
    model = getattr(engine, "MODEL", None)
    if model is not None:
        storages = {}
        for tensor in list(model.parameters()) + list(model.buffers()):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()  # views sharing storage are counted once
        for module in model.modules():  # plain tensor attributes (e.g. stacked expert weights) also hold memory
            for value in vars(module).values():
                if torch.is_tensor(value):
                    storages[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
        report["parameter_bytes"] = sum(storages.values())
        report["moe_forward"] = sorted({type(m).forward.__module__ + "." + type(m).forward.__qualname__
                                        for m in model.modules() if "moe" in type(m).__name__.lower()
                                        and "sparse" in type(m).__name__.lower()})
    def original_forward(module):
        for base in type(module).__mro__:
            if base in originals:
                return base, originals[base]
        return None, None
    replaced, captures, hooks = {}, {}, []
    if model is not None:
        for name, module in model.named_modules():
            base, original = original_forward(module)
            if original is None:
                continue
            current = module.__dict__.get("forward") or type(module).forward
            current = getattr(current, "__func__", current)
            if current is not original and base.__name__ not in replaced:
                replaced[base.__name__] = {"module": name, "replacement": "%s.%s" % (
                    getattr(current, "__module__", "?"), getattr(current, "__qualname__", repr(current)))}
                captures[base.__name__] = []
                def hook(mod, args, kwargs, _key=base.__name__):
                    # Keep one prefill-shaped and one decode-shaped real input per replaced class.
                    first = args[0] if args else next((v for v in kwargs.values() if torch.is_tensor(v)), None)
                    shape = tuple(first.shape) if torch.is_tensor(first) else None
                    if shape is not None and len(captures[_key]) < 2 and shape not in [c[2] for c in captures[_key]]:
                        captures[_key].append((copy.deepcopy(args), copy.deepcopy(kwargs), shape))
                hooks.append(module.register_forward_pre_hook(hook, with_kwargs=True))
                replaced[base.__name__]["_module"] = module
                replaced[base.__name__]["_original"] = original
                replaced[base.__name__]["_current"] = current
    from vibeforge_v2.harness import forward_fingerprint as fingerprint
    report["replaced_forwards"] = {k: {"module": v["module"], "replacement": v["replacement"],
                                       "fingerprint": fingerprint(v["_current"]),
                                       "original_fingerprint": fingerprint(v["_original"])} for k, v in replaced.items()}
    prompts = {"short": "Hello", "medium": "def add(a, b):\n    return a + b\n" * 12,
               "long": "Record: attention, cache, sample, update. " * 60}
    report["generate"] = {}
    stage = "generate()"
    for name, prompt in prompts.items():
        out = engine.generate(prompt, max_new_tokens=6, fixed_length=True)
        report["generate"][name] = {"input_ids": out["input_ids"][:4] + [len(out["input_ids"])], "token_ids": out["token_ids"]}
    for handle in hooks:
        handle.remove()
    stage = "sampling contract"
    from vibeforge_v2.harness import sampling_contract
    report["sampling_contract"] = sampling_contract(engine)
    def flatten(value, path="output"):
        if torch.is_tensor(value):
            return [(path, value)]
        if isinstance(value, (tuple, list)):
            return [item for index, element in enumerate(value) for item in flatten(element, "%s[%d]" % (path, index))]
        if hasattr(value, "to_tuple"):
            return flatten(value.to_tuple(), path)
        return [(path, value)]
    report["module_diff"] = []
    stage = "module differential check"
    for key, info in replaced.items():
        for args, kwargs, _shape in captures.get(key, []):
            row = {"class": key, "module": info["module"], "replacement": info["replacement"],
                   "input_shapes": [list(a.shape) for a in args if torch.is_tensor(a)] or
                                   [list(v.shape) for v in kwargs.values() if torch.is_tensor(v)][:1]}
            try:
                with torch.no_grad():
                    expected = info["_original"](info["_module"], *copy.deepcopy(args), **copy.deepcopy(kwargs))
                    actual = info["_current"](info["_module"], *copy.deepcopy(args), **copy.deepcopy(kwargs))
            except Exception as error:
                row["error"] = "%s: %s" % (type(error).__name__, error)
                row["traceback"] = traceback.format_exc()[-1500:]
                report["module_diff"].append(row)
                continue
            try:
                from vibeforge_v2.harness import matmul_flop_counter
                counts = {}
                for label, function in (("original", info["_original"]), ("candidate", info["_current"])):
                    calls = {}
                    def counter_hook(m, a, o, _calls=calls):
                        name = type(m).__name__
                        _calls[name] = _calls.get(name, 0) + 1
                    handles = [child.register_forward_hook(counter_hook)
                               for _, child in info["_module"].named_modules()
                               if child is not info["_module"] and list(child.children())]
                    try:
                        with torch.no_grad(), matmul_flop_counter() as counter:
                            function(info["_module"], *copy.deepcopy(args), **copy.deepcopy(kwargs))
                    finally:
                        for handle in handles:
                            handle.remove()
                    counts[label] = counter.get_total_flops()
                    row.setdefault("child_calls", {})[label] = calls
                row["flops"] = counts
            except Exception as error:
                row["flops_error"] = "%s: %s" % (type(error).__name__, error)
            left, right = flatten(expected), flatten(actual)
            row["structure"] = {"original": [(p, list(v.shape) if torch.is_tensor(v) else type(v).__name__) for p, v in left],
                                "candidate": [(p, list(v.shape) if torch.is_tensor(v) else type(v).__name__) for p, v in right]}
            outputs = []
            for (path, a), (_, b) in zip(left, right):
                if not (torch.is_tensor(a) and torch.is_tensor(b)):
                    outputs.append({"output": path, "comparable": False})
                    continue
                if a.shape != b.shape:
                    outputs.append({"output": path, "shape_original": list(a.shape), "shape_candidate": list(b.shape)})
                    continue
                difference = (a.float() - b.float()).abs()
                scale = max(a.float().abs().max().item(), 1e-6)
                rows = difference.reshape(-1, a.shape[-1]) if a.dim() > 1 else difference.reshape(1, -1)
                bad = (rows.max(dim=-1).values / scale > 1e-3).nonzero().flatten().tolist()
                outputs.append({"output": path, "shape": list(a.shape), "max_relative_error": difference.max().item() / scale,
                                "rows_wrong": len(bad), "rows_total": rows.shape[0], "first_wrong_rows": bad[:8]})
            row["outputs"] = outputs
            report["module_diff"].append(row)
    stage = "line trace of request path"
    import os as _os
    root = _os.path.realpath(directory) + _os.sep
    executed = set()
    def tracer(frame, event, arg):
        filename = _os.path.realpath(frame.f_code.co_filename)
        if not filename.startswith(root):
            return None
        relative = filename[len(root):]
        def local(frame, event, arg):
            if event == "line":
                executed.add((relative, frame.f_lineno))
            return local
        executed.add((relative, frame.f_lineno))
        return local
    sys.settrace(tracer); threading.settrace(tracer)
    try:
        engine.generate(prompts["short"], max_new_tokens=2, fixed_length=True)
        engine.logits(engine.generate(prompts["short"], max_new_tokens=1)["input_ids"], indices=[0, 1])
    finally:
        sys.settrace(None); threading.settrace(None)
    report["executed_lines"] = sorted([list(item) for item in executed])
    stage = "logits()"
    ids = engine.generate(prompts["medium"], max_new_tokens=1)["input_ids"]
    values = engine.logits(ids, indices=list(range(0, 151000, 97)))["values"]
    report["logits"] = values
    if config.get("request_scheduler") in ("microbatch", "concurrent_requests"):
        stage = "concurrent generate()"
        results = {}
        def call(key, prompt):
            results[key] = engine.generate(prompt, max_new_tokens=6, fixed_length=True)["token_ids"]
        threads = [threading.Thread(target=call, args=(k, p)) for k, p in (("short", prompts["short"]), ("long", prompts["long"]))]
        [t.start() for t in threads]; [t.join(120) for t in threads]
        report["concurrent"] = results
    stage = "chat_completion() with tools"
    tools = [{"type": "function", "function": {"name": "shell", "description": "run",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    chat = engine.chat_completion([{"role": "user", "content": "list files"}], tools=tools, max_tokens=8, temperature=0)
    report["chat_completion_keys"] = sorted(chat)
    stage = "decode timing"
    timings = []
    for _ in range(2):
        started = time.perf_counter()
        engine.generate(prompts["short"], max_new_tokens=17, fixed_length=True)
        middle = time.perf_counter()
        engine.generate(prompts["short"], max_new_tokens=1, fixed_length=True)
        timings.append(((middle - started) - (time.perf_counter() - middle)) / 16)
    report["decode_seconds_per_token"] = min(timings)
    report["ok"] = True
except BaseException as error:
    report["ok"] = False
    report["stage"] = locals().get("stage", "startup")
    report["error"] = "%s: %s" % (type(error).__name__, error)
    report["traceback"] = traceback.format_exc()[-4000:]
    # Tensor shapes of local variables in the candidate's own frames at the moment of failure.
    try:
        import os as _os, linecache as _linecache
        root = _os.path.realpath(directory) + _os.sep
        frames = []
        tb = error.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            filename = _os.path.realpath(frame.f_code.co_filename)
            if filename.startswith(root):
                variables = {}
                for name, value in frame.f_locals.items():
                    if name.startswith("__"):
                        continue
                    if torch.is_tensor(value):
                        variables[name] = "Tensor%s %s" % (list(value.shape), str(value.dtype).replace("torch.", ""))
                    elif isinstance(value, (int, float, bool)) and not isinstance(value, type):
                        variables[name] = repr(value)
                    elif isinstance(value, (tuple, list)) and value and all(torch.is_tensor(v) for v in value[:4]):
                        variables[name] = "[" + ", ".join("Tensor%s" % list(v.shape) for v in value[:4]) + (", ..." if len(value) > 4 else "") + "]"
                frames.append({"file": filename[len(root):], "line": tb.tb_lineno, "function": frame.f_code.co_name,
                               "code": _linecache.getline(filename, tb.tb_lineno).strip(), "locals": variables})
            tb = tb.tb_next
        report["failure_frames"] = frames[-2:]
    except Exception as inner:
        report["notes"].append("shape capture failed: %s" % inner)
finish()
'''


def run_engine(directory, model_dir, dtype, python, timeout, workdir):
    from vf_runtime.engine.command_sandbox import command_argv
    output = Path(workdir) / f'{Path(directory).name}-{Path(directory).parent.name}-{dtype}.json'
    script = Path(workdir) / 'harness_worker.py'
    script.write_text(WORKER)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    scratch = Path(workdir) / 'scratch'
    scratch.mkdir(exist_ok=True)
    env.update(TMPDIR=str(scratch), TMP=str(scratch), TEMP=str(scratch), XDG_CACHE_HOME=str(scratch))
    started = time.monotonic()
    try:
        process = subprocess.run(command_argv(
                                     [python, str(script), str(directory), str(model_dir), dtype, str(output)],
                                     [workdir]),
                                 cwd=str(directory), env=env, capture_output=True, text=True, timeout=timeout)
        stderr = process.stderr[-3000:]
    except subprocess.TimeoutExpired:
        return {'ok': False, 'stage': 'timeout', 'error': f'harness worker exceeded {timeout}s', 'dtype': dtype}
    if not output.exists():
        return {'ok': False, 'stage': 'worker crashed', 'error': stderr or 'no output', 'dtype': dtype}
    report = json.loads(output.read_text())
    report['wall_seconds'] = time.monotonic() - started
    if not report.get('ok'):
        report['stderr_tail'] = stderr
    return report


def _relative_error(a, b):
    import math
    scale = max(max(abs(x) for x in b), 1e-6)
    return max(abs(x - y) for x, y in zip(a, b)) / scale if a and len(a) == len(b) and all(map(math.isfinite, a)) else float('inf')


def changed_source_lines(stable, candidate):
    """(relative path, line number) of candidate .py lines that differ from the stable source."""
    import difflib
    result = set()
    for path in Path(candidate).rglob('*.py'):
        relative = str(path.relative_to(candidate))
        if '__pycache__' in relative:
            continue
        new = path.read_text(errors='replace').splitlines()
        old_path = Path(stable) / relative
        old = old_path.read_text(errors='replace').splitlines() if old_path.exists() else []
        for tag, _, _, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag in ('replace', 'insert'):
                result.update((relative, line + 1) for line in range(j1, j2) if new[line].strip())
    return result


def compare(stable, candidate, *, fp32_tolerance=2e-3, bf16_tolerance=0.15):
    errors, warnings, facts = [], [], {}
    for side, reports in (('stable', stable), ('candidate', candidate)):
        for dtype, report in reports.items():
            for option, override in report.get('cpu_config_overrides', {}).items():
                facts.setdefault('cpu_config_overrides', []).append(
                    {'side': side, 'dtype': dtype, 'option': option, **override})
                warnings.append(
                    f"{side} {dtype}: CPU harness replaced {option}={override['requested']!r} "
                    f"with {override['tested']!r}. Only the substituted CPU path was tested; "
                    'the configured implementation and its dependencies, numerical behavior and speed '
                    'remain unverified and require host GPU evaluation.')
    for dtype in ('float32', 'bfloat16'):
        s, c = stable[dtype], candidate[dtype]
        if not s.get('ok'):
            warnings.append(f'stable engine itself failed on CPU in {dtype} ({s.get("stage")}: {s.get("error")}); '
                            'comparison skipped. This is a harness limitation, not evidence that the candidate is correct.')
            facts['harness_unavailable'] = True
            continue
        if not c.get('ok'):
            text = f"{dtype}: candidate failed at {c.get('stage')}: {c.get('error')}"
            if any(marker in (c.get('error', '') + c.get('traceback', '')).lower() for marker in GPU_ONLY_MARKERS):
                warnings.append(text + ' -- looks GPU-only, so it cannot be checked on CPU; the host will test it on GPU. '
                                'If possible keep a CPU fallback so the math can be verified here.')
                facts['gpu_only'] = True
            else:
                shapes = ''
                for frame in c.get('failure_frames') or []:
                    shapes += (f"\nAt {frame['file']}:{frame['line']} in {frame['function']}: `{frame['code']}`\n"
                               '  local variables at the failure: '
                               + ('; '.join(f'{k} = {v}' for k, v in frame['locals'].items()) or '(none)'))
                if shapes:
                    shapes = ('\nSHAPES AT FAILURE (real values on the reduced model, which keeps the real hidden sizes; tokens per call vary; '
                              'compare each with the shape your code expects):' + shapes)
                errors.append(text + shapes + '\n' + c.get('traceback', '')[-1200:])
            continue
        for name, modes in s.get('sampling_contract', {}).items():
            candidate_modes = c.get('sampling_contract', {}).get(name)
            if modes and candidate_modes:
                if modes != candidate_modes:
                    errors.append(f'{dtype}: sampling contract changed for {name} request: '
                                  f'stable do_sample={modes}, candidate do_sample={candidate_modes}. '
                                  'Changing greedy decoding to sampling (or the reverse) changes request semantics; '
                                  'restore the original sampling policy and retain a separate performance change.')
            else:
                warnings.append(f'{dtype}: sampling mode for {name} could not be observed; host validation required.')
        error = _relative_error(c['logits'], s['logits'])
        facts[f'{dtype}_logit_relative_error'] = error
        limit = fp32_tolerance if dtype == 'float32' else bf16_tolerance
        if error > limit:
            errors.append(f'{dtype}: logits differ from the stable engine by {error:.3g} (relative to max |logit|; limit '
                          f'{limit}). In float32 a correct re-implementation differs only by ~1e-6..1e-4; '
                          'this indicates a math error (wrong indexing or dimension order, a missing scale/weight factor, '
                          'normalisation, accumulation or residual).')
        if dtype == 'float32':
            different = [name for name in s['generate'] if s['generate'][name]['token_ids'] != c['generate'][name]['token_ids']]
            if different:
                warnings.append('float32 greedy tokens differ for ' + ', '.join(different)
                                + ' (random weights give near-flat distributions, so check the logit error above first).')
        if c.get('concurrent') is not None:
            for key, tokens in c['concurrent'].items():
                if tokens != c['generate'][key]['token_ids'] and dtype == 'float32':
                    warnings.append(f'{dtype}: concurrent request "{key}" differs from its isolated output')
    c32 = candidate.get('float32', {})
    facts['replaced_forwards'] = c32.get('replaced_forwards')
    for row in c32.get('module_diff') or []:
        label = f"{row['class']} ({row['module']}) input {row.get('input_shapes')}"
        if row.get('error'):
            errors.append(f"module check: {label}: the replacement raised {row['error']}\n{row.get('traceback', '')}")
            continue
        if [x[0] for x in row['structure']['original']] != [x[0] for x in row['structure']['candidate']]:
            text = (f"module check: {label}: output structure differs. original returns {row['structure']['original']}, "
                    f"replacement returns {row['structure']['candidate']}.")
            first = (row.get('outputs') or [{}])[0]
            if first.get('max_relative_error', 1) <= 1e-3:
                warnings.append(text + ' The hidden-state output matches and callers here accept it, but returning the '
                                'same layout as the original is safer.')
            else:
                errors.append(text + ' Return exactly the same tuple/tensor layout.')
            continue
        for output in row['outputs']:
            if 'shape_original' in output:
                errors.append(f"module check: {label}: {output['output']} shape {output['shape_candidate']} but the original "
                              f"returns {output['shape_original']}.")
            elif output.get('max_relative_error', 0) > 1e-3:
                errors.append(f"module check (float32, same real input, original vs your forward): {label}: "
                              f"{output['output']} shape {output['shape']} differs by {output['max_relative_error']:.3g} "
                              f"(relative); {output['rows_wrong']}/{output['rows_total']} token rows wrong, first wrong rows "
                              f"{output['first_wrong_rows']}. Outputs that match exactly are correct, so the bug is in how "
                              f"this output is computed. Re-check every tensor's dimension order against the original "
                              f"source (print .shape of each intermediate on a small input).")
    for row in c32.get('module_diff') or []:
        flops = row.get('flops') or {}
        if flops.get('original') and flops.get('candidate') is not None:
            ratio = flops['candidate'] / flops['original']
            facts.setdefault('module_flop_ratio', {})[f"{row['class']} {row.get('input_shapes')}"] = round(ratio, 3)
            if ratio > 2:
                calls = row.get('child_calls') or {}
                detail = ''
                for name in sorted(set(calls.get('original', {})) | set(calls.get('candidate', {}))):
                    a, b = calls.get('original', {}).get(name, 0), calls.get('candidate', {}).get(name, 0)
                    if a != b:
                        detail += f' {name} submodule called {b} times by your forward vs {a} times by the original.'
                warnings.append(
                    f"module check: {row['class']} ({row['module']}) input {row.get('input_shapes')}: your forward performs "
                    f"{ratio:.1f}x the matmul FLOPs of the original on the same real input "
                    f"({flops['candidate']:.3g} vs {flops['original']:.3g}).{detail} "
                    'Extra arithmetic is a performance risk, not a correctness failure. GPU measurements must '
                    'determine whether reduced launches or better utilization outweigh it.')
    for name, info in (c32.get('replaced_forwards') or {}).items():
        if info.get('fingerprint') and info.get('fingerprint') == info.get('original_fingerprint'):
            facts['no_effect'] = True
            errors.append(f'No effective change: the replacement for {name}.forward ({info.get("replacement")}) is the '
                          'original library code under a new name (identical after removing names, docstrings and '
                          'comments). Installing a copy cannot change speed.')
    s, c = stable.get('float32', {}), candidate.get('float32', {})
    if s.get('ok') and c.get('ok'):
        ratio = c['decode_seconds_per_token'] / max(s['decode_seconds_per_token'], 1e-9)
        facts['cpu_decode_time_ratio'] = ratio
        memory = c.get('parameter_bytes', 0) / max(s.get('parameter_bytes', 1), 1)
        facts['parameter_memory_ratio'] = memory
        if ratio > 1.25:
            warnings.append(f'CPU decode is {ratio:.2f}x the stable time on the reduced model. CPU is not the GPU, but a '
                            'large slowdown usually means more work per token (e.g. evaluating sub-modules the original skips).')
        if memory > 1.05:
            warnings.append(f'Model parameters/buffers take {memory:.2f}x the stable memory. The real model has '
                            'weights that already fill most of the GPU, so duplicating weights (e.g. stacking or re-laying-out weights while '
                            'keeping the originals) will run out of memory. Free or alias the originals.')
    return errors, warnings, facts


def run(candidate, *, stable=None, model_path=None, python=None, cache=None, timeout=420):
    candidate = Path(candidate).resolve()
    root = _run_root(candidate)
    if stable is None:
        stable = (root / 'current/engine') if root and (root / 'current/engine').exists() else None
    if stable is None:
        raise ValueError('Pass --stable; no current/engine found above the candidate')
    stable = Path(stable).resolve()
    if model_path is None:
        model_path = json.loads((stable / 'engine.json').read_text()).get('model_path')
    cache = Path(cache or ((root / 'cache/harness') if root else Path('/tmp/vibeforge-harness')))
    cache.mkdir(parents=True, exist_ok=True)
    python = python or sys.executable
    started = time.monotonic()
    model_dir = build_reduced_model(model_path, cache)
    workdir = cache / f'work-{os.getpid()}'
    workdir.mkdir(exist_ok=True)
    results = {'stable': {}, 'candidate': {}}
    for dtype in ('float32', 'bfloat16'):
        results['stable'][dtype] = run_engine(stable, model_dir, dtype, python, timeout, workdir)
        results['candidate'][dtype] = run_engine(candidate, model_dir, dtype, python, timeout, workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    errors, warnings, facts = compare(results['stable'], results['candidate'])
    history = cache / ('history-' + hashlib.sha256(str(candidate).encode()).hexdigest()[:12] + '.json')
    previous = json.loads(history.read_text()) if history.exists() else {}
    signature = [error.split('\n')[0] for error in errors]
    repeats = previous.get('repeats', 0) + 1 if errors and previous.get('signature') == signature else 0
    history.write_text(json.dumps({'signature': signature, 'repeats': repeats}))
    s32, c32 = results['stable'].get('float32', {}), results['candidate'].get('float32', {})
    if not errors and s32.get('ok') and c32.get('ok'):
        from .experience import config_changes
        config_delta = config_changes(json.loads((stable / 'engine.json').read_text()),
                                      json.loads((candidate / 'engine.json').read_text()))
        identical = (not c32.get('replaced_forwards') and facts.get('float32_logit_relative_error') == 0
                     and all(s32['generate'][k]['token_ids'] == c32['generate'][k]['token_ids'] for k in s32['generate'])
                     and not config_delta)
        if identical:
            changed_lines = changed_source_lines(stable, candidate)
            executed = {tuple(item) for item in c32.get('executed_lines') or []}
            live = sorted(f'{path}:{line}' for path, line in changed_lines if (path, line) in executed)
            facts['changed_lines_executed_on_request_path'] = len(live)
            if c32.get('executed_lines') is not None and not live:
                facts['unobserved_source_effect'] = True
                warnings.append(
                    f'The CPU request trace did not execute any changed source line ({len(changed_lines)} checked), '
                    'and the sampled CPU outputs match. This trace excludes import/load effects, CUDA-only '
                    'branches and paths outside these reduced-model requests; it cannot establish that the '
                    'candidate has no GPU effect. Runtime activation and speed remain unverified and require '
                    'host GPU evaluation. An unused definition or setting alone is not evidence of optimization.')
    if repeats:
        warnings.append(f'The same error has now appeared {repeats + 1} times in a row. Your edits did not change the '
                        'line that fails. Read the SHAPES AT FAILURE above, write down the shape every tensor on that '
                        'line must have, and change the code that produces the wrong one.')
    status = ('FAIL' if errors else 'SKIP' if facts.get('harness_unavailable') or facts.get('gpu_only')
              or facts.get('unobserved_source_effect') or facts.get('cpu_config_overrides') else 'PASS')
    return {'status': status, 'passed': status == 'PASS', 'allow_gpu_evaluation': not errors,
            'errors': errors, 'warnings': warnings, 'facts': facts, 'scope': SCOPE,
            'wall_seconds': time.monotonic() - started,
            'details': {side: {dtype: {k: v for k, v in report.items() if k not in ('logits',)}
                               for dtype, report in sides.items()} for side, sides in results.items()}}


def describe(report):
    lines = ['HOST CPU EQUIVALENCE HARNESS: ' + report.get('status', 'PASS' if report['passed'] else 'FAIL')
             + f" ({report['wall_seconds']:.0f}s)", 'Scope: ' + report['scope']]
    lines += ['ERROR: ' + error for error in report['errors']]
    lines += ['NOTE: ' + warning for warning in report['warnings']]
    facts = report.get('facts', {})
    if facts:
        lines.append('facts: ' + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in facts.items()},
                                           ensure_ascii=False))
    return '\n'.join(lines)


def engine_directory(value):
    """Reject a wrong CLI working directory before creating caches or loading weights."""
    directory = Path(value).resolve()
    missing = [name for name in ('engine.py', 'engine.json') if not (directory / name).is_file()]
    if missing:
        message = f'Not an engine directory: {directory}; missing {", ".join(missing)}.'
        child = directory / 'engine'
        if all((child / name).is_file() for name in ('engine.py', 'engine.json')):
            message += f' The engine files are in {child}. Run from that directory or pass --candidate {child}.'
        else:
            message += ' Run from the candidate directory containing engine.py and engine.json, or pass --candidate explicitly.'
        raise argparse.ArgumentTypeError(message + ' No model was loaded and no cache was written.')
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--candidate', default='.', type=engine_directory)
    parser.add_argument('--stable', type=engine_directory)
    parser.add_argument('--model-path')
    parser.add_argument('--output')
    parser.add_argument('--timeout', type=int, default=420)
    args = parser.parse_args(argv)
    report = run(args.candidate, stable=args.stable, model_path=args.model_path, timeout=args.timeout)
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(describe(report))
    raise SystemExit(0 if report.get('allow_gpu_evaluation', report['passed']) else 1)


if __name__ == '__main__':
    main()
