"""Check public call shapes from AST, without importing untrusted engine code."""
import ast
import inspect


PUBLIC = ('load', 'generate', 'chat', 'chat_completion', 'logits', 'synchronize',
          'run_on_inference_worker')
P = inspect.Parameter


def signatures(source):
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in PUBLIC:
            continue
        args = node.args
        positional = args.posonlyargs + args.args
        defaults = [P.empty] * (len(positional) - len(args.defaults)) + [ast.unparse(x) for x in args.defaults]
        parameters = [P(arg.arg, P.POSITIONAL_ONLY if i < len(args.posonlyargs) else P.POSITIONAL_OR_KEYWORD,
                        default=default) for i, (arg, default) in enumerate(zip(positional, defaults))]
        if args.vararg:
            parameters.append(P(args.vararg.arg, P.VAR_POSITIONAL))
        parameters.extend(P(arg.arg, P.KEYWORD_ONLY, default=P.empty if default is None else ast.unparse(default))
                          for arg, default in zip(args.kwonlyargs, args.kw_defaults))
        if args.kwarg:
            parameters.append(P(args.kwarg.arg, P.VAR_KEYWORD))
        result[node.name] = (inspect.Signature(parameters), isinstance(node, ast.AsyncFunctionDef),
                             ast.unparse(args))
    return result


def call_shapes(signature):
    """Minimal/full calls, every positional/keyword split, and variadic probes."""
    params = list(signature.parameters.values())
    positional = [p for p in params if p.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD)]
    for full in (False, True):
        for split in range(len(positional) + 1):
            args, kwargs = [], {}
            for i, param in enumerate(positional):
                if i < split:
                    args.append(None)
                elif full or param.default is P.empty:
                    kwargs[param.name] = None
            kwargs.update({p.name: None for p in params if p.kind == P.KEYWORD_ONLY
                           and (full or p.default is P.empty)})
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            yield args, kwargs
    args = [None] * len(positional)
    kwargs = {p.name: None for p in params if p.kind == P.KEYWORD_ONLY}
    if any(p.kind == P.VAR_POSITIONAL for p in params):
        yield args + [None], kwargs
    if any(p.kind == P.VAR_KEYWORD for p in params):
        key = '__host_extra_option__'
        while key in signature.parameters:
            key += '_'
        yield args, {**kwargs, key: None}


def check_contract(before, source):
    after = signatures(source)
    errors, warnings = [], []
    for name, (signature, was_async, display) in before.items():
        if name not in after:
            # Imports/aliases may be valid; existing symbol and runtime checks still apply.
            warnings.append(f'engine.py: {name} has no direct definition; call compatibility requires runtime validation.')
            continue
        candidate, is_async, _ = after[name]
        if is_async != was_async:
            errors.append(f'engine.py: {name} changed synchronous/asynchronous calling convention.')
        for args, kwargs in call_shapes(signature):
            try:
                candidate.bind(*args, **kwargs)
            except TypeError as error:
                call = ', '.join(['...'] * len(args) + [f'{key}=...' for key in kwargs])
                errors.append(f'engine.py: incompatible public call {name}({call}): {error}. '
                              f'Stable contract: {name}({display}). Repair this interface locally while retaining '
                              'your optimization; restoring the whole stable file removes the candidate change.')
                break
    return errors, warnings


def describe_contract(source):
    lines = ['Public engine call contract (from this round\'s stable source):']
    lines.extend(f'- {name}({display})' for name, (_, _, display) in signatures(source).items())
    lines.append('Preserve keyword names, optional arguments and return semantics. In particular, logits must accept '
                 'indices= and return indices/values. Static binding does not prove runtime correctness; '
                 'the CPU harness and host GPU evaluation remain required. Repair a broken interface locally; '
                 'keep the effective optimization diff.')
    return '\n'.join(lines)
