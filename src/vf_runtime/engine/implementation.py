"""Bounded read-only access to the loaded model's implementation, never arbitrary files."""

import hashlib
import inspect


def implementation_sources(backend):
    symbols = {}
    for name in ('load', 'generate', 'logits'):
        value = getattr(backend, name, None)
        if inspect.isfunction(value):
            symbols['backend.' + name] = value
    model = getattr(backend, 'MODEL', None)
    module = inspect.getmodule(type(model)) if model is not None else None
    if module is not None:
        for name, value in vars(module).items():
            if name.startswith('_') or getattr(value, '__module__', None) != module.__name__:
                continue
            if inspect.isfunction(value):
                symbols[name] = value
            elif inspect.isclass(value) and inspect.isfunction(getattr(value, 'forward', None)):
                symbols[name + '.forward'] = value.forward
    return symbols


def source_location(function):
    # getsourcelines follows __wrapped__, while getsourcefile does not.
    # Resolve once so the file, line and returned source refer to one definition.
    definition = inspect.unwrap(function)
    location = {'path': inspect.getsourcefile(definition),
                'line': inspect.getsourcelines(definition)[1]}
    if definition is not function:
        code = getattr(function, '__code__', None)
        if code is not None:
            location['callable_location'] = {'path': code.co_filename, 'line': code.co_firstlineno}
        location['source_role'] = 'Unwrapped definition; callable wrapper may dispatch to a different implementation'
    else:
        location['source_role'] = 'Loaded Python definition; not evidence of execution'
    return location


def read_implementation(backend, symbol='', offset=0, limit=6000):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
        raise ValueError('Invalid implementation page; limit must be 1..12000 characters')
    symbols = implementation_sources(backend)
    if not symbol:
        locations = {}
        for name, function in symbols.items():
            try:
                locations[name] = source_location(function)
            except (TypeError, OSError, ValueError):
                continue
        return {'symbols': sorted(symbols), 'locations': locations,
                'role': 'Loaded implementation source, not measured performance evidence'}
    if symbol not in symbols:
        raise ValueError('Select a symbol from the implementation index')
    function = symbols[symbol]
    source = inspect.getsource(inspect.unwrap(function))
    return {'symbol': symbol, 'module': symbols[symbol].__module__, **source_location(function),
            'sha256': hashlib.sha256(source.encode()).hexdigest(), 'total_characters': len(source),
            'offset': offset, 'source': source[offset:offset + limit],
            'next_offset': offset + limit if offset + limit < len(source) else None}
