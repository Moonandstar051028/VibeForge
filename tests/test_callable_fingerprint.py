from functools import wraps
import inspect

from vibeforge_v2.harness import compare, forward_fingerprint


def eager(value: int) -> int:
    """Original calculation."""
    return value * 2


def eager_copy(value):
    # A rename and comment do not change this ordinary callable.
    return value * 2


def dispatch(original):
    @wraps(original)
    def wrapper(value):
        return original(value) + 1
    return wrapper


def test_decorated_dispatch_is_not_conflated_with_underlying_definition():
    wrapped = dispatch(eager)
    assert inspect.getsource(wrapped) == inspect.getsource(eager)
    assert wrapped(2) != eager_copy(2)
    assert forward_fingerprint(wrapped)
    assert forward_fingerprint(wrapped) != forward_fingerprint(eager_copy)


def test_ordinary_verbatim_copy_still_detected():
    assert forward_fingerprint(eager) == forward_fingerprint(eager_copy)


def test_replacing_dispatch_with_eager_is_not_rejected_as_no_effect():
    ok = {'ok': True, 'logits': [1., 2.], 'decode_seconds_per_token': .01,
          'parameter_bytes': 1, 'generate': {'short': {'token_ids': [1]}}}
    changed = {**ok, 'replaced_forwards': {'Moe': {
        'replacement': 'eager_copy', 'fingerprint': forward_fingerprint(eager_copy),
        'original_fingerprint': forward_fingerprint(dispatch(eager)),
    }}}
    errors, _, facts = compare({'float32': ok, 'bfloat16': ok}, {'float32': changed, 'bfloat16': changed})
    assert not errors
    assert not facts.get('no_effect')


def test_real_qwen_experts_wrapper_has_distinct_callable_fingerprint():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts
    wrapped = Qwen3MoeExperts.forward
    definition = inspect.unwrap(wrapped)
    assert wrapped is not definition
    assert forward_fingerprint(wrapped) != forward_fingerprint(definition)
    assert forward_fingerprint(wrapped) and forward_fingerprint(definition)
