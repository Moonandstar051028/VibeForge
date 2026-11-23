from contextlib import contextmanager
import inspect
from pathlib import Path
from types import SimpleNamespace

from vf_runtime.engine import implementation


@contextmanager
def wrapped_generate():
    yield 'body is never called by source inspection'


def plain_generate():
    return 'plain'


def test_wrapped_source_path_line_and_excerpt_refer_to_same_definition():
    backend = SimpleNamespace(generate=wrapped_generate)
    location = implementation.read_implementation(backend)['locations']['backend.generate']
    original = inspect.unwrap(wrapped_generate)
    assert location['path'] == inspect.getsourcefile(original)
    assert location['line'] == inspect.getsourcelines(original)[1]
    assert location['callable_location']['path'] == wrapped_generate.__code__.co_filename
    assert location['callable_location']['line'] == wrapped_generate.__code__.co_firstlineno
    assert location['path'] != location['callable_location']['path']
    excerpt = implementation.read_implementation(backend, 'backend.generate')
    assert Path(location['path']).read_text().splitlines()[location['line'] - 1] == excerpt['source'].splitlines()[0]
    assert excerpt['path'] == location['path'] and excerpt['line'] == location['line']
    assert 'may dispatch' in excerpt['source_role']


def test_plain_function_has_one_location_and_paging_remains_consistent():
    backend = SimpleNamespace(generate=plain_generate)
    full = implementation.read_implementation(backend, 'backend.generate')
    page = implementation.read_implementation(backend, 'backend.generate', offset=5, limit=8)
    assert 'callable_location' not in full
    assert page['source'] == full['source'][5:13]
    assert page['sha256'] == full['sha256']
    assert page['next_offset'] == 13


def test_real_qwen_expert_wrapper_points_to_model_definition_and_dispatch_entry():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    function = Qwen3MoeExperts.forward
    location = implementation.source_location(function)
    assert location['path'].endswith('modeling_qwen3_moe.py')
    assert 'def forward(' in Path(location['path']).read_text().splitlines()[location['line'] - 1]
    assert location['callable_location']['path'].endswith('integrations/moe.py')
    wrapper_line = Path(location['callable_location']['path']).read_text().splitlines()[location['callable_location']['line'] - 1]
    assert 'wraps' in wrapper_line or 'def forward' in wrapper_line
