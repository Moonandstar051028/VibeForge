import pytest

torch = pytest.importorskip('torch')

from vibeforge_v2.harness import matmul_flop_counter


@pytest.mark.parametrize('offsets', [[2, 4, 6, 8], [0, 3, 3, 5], [0, 0, 0, 0]])
def test_grouped_custom_op_and_eager_loop_count_the_same_arithmetic(offsets):
    from torch.utils.flop_counter import FlopCounterMode, flop_registry
    from transformers.integrations.moe import _grouped_mm_fallback

    op = torch.ops.transformers.grouped_mm_fallback
    original_mapping = flop_registry.get(op)
    inputs = torch.randn(8, 6)
    weights = torch.randn(4, 6, 10)
    offs = torch.tensor(offsets, dtype=torch.int32)
    with torch.no_grad(), FlopCounterMode(display=False) as eager:
        expected = _grouped_mm_fallback(inputs, weights, offs)
    with torch.no_grad(), matmul_flop_counter() as opaque:
        actual = op(inputs, weights, offs=offs)
    assert torch.equal(actual, expected)
    assert opaque.get_total_flops() == eager.get_total_flops() == 2 * offsets[-1] * 6 * 10
    assert flop_registry.get(op) is original_mapping


def test_grouped_moe_and_inlined_eager_experts_have_equal_matmul_counts():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeSparseMoeBlock

    torch.manual_seed(42)
    config = Qwen3MoeConfig(hidden_size=16, intermediate_size=32, moe_intermediate_size=12,
                          num_experts=4, num_experts_per_tok=2)
    config._experts_implementation = 'grouped_mm'
    module = Qwen3MoeSparseMoeBlock(config).eval()
    with torch.no_grad():
        for param in module.parameters():
            param.normal_(std=.1)
    hidden = torch.randn(1, 3, 16)
    with torch.no_grad(), matmul_flop_counter() as grouped:
        expected = module(hidden)
    with torch.no_grad(), matmul_flop_counter() as inlined:
        flat = hidden.view(-1, 16)
        _, routing, selected = module.gate(flat)
        actual = torch.zeros_like(flat)
        for expert in range(4):
            token, slot = torch.where(selected == expert)
            gate, up = torch.nn.functional.linear(flat[token], module.experts.gate_up_proj[expert]).chunk(2, dim=-1)
            value = torch.nn.functional.linear(module.experts.act_fn(gate) * up, module.experts.down_proj[expert])
            actual.index_add_(0, token, value * routing[token, slot, None])
    assert torch.allclose(actual.reshape_as(expected), expected, atol=1e-6)
    assert grouped.get_total_flops() == inlined.get_total_flops()
    # Router plus two expert projections, counting top-k routed rows.
    assert grouped.get_total_flops() == 2 * 3 * 16 * 4 + 2 * 3 * 2 * (16 * 24 + 12 * 16)
