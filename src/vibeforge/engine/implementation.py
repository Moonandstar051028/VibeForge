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


def read_implementation(backend, symbol='', offset=0, limit=6000):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 12000:
        raise ValueError('Invalid implementation page; limit must be 1..12000 characters')
    symbols = implementation_sources(backend)
    if not symbol:
        return {'symbols': sorted(symbols), 'role': 'Loaded implementation source, not measured performance evidence'}
    if symbol not in symbols:
        raise ValueError('Select a symbol from the implementation index')
    source = inspect.getsource(symbols[symbol])
    return {'symbol': symbol, 'module': symbols[symbol].__module__,
            'sha256': hashlib.sha256(source.encode()).hexdigest(), 'total_characters': len(source),
            'offset': offset, 'source': source[offset:offset + limit],
            'next_offset': offset + limit if offset + limit < len(source) else None}
