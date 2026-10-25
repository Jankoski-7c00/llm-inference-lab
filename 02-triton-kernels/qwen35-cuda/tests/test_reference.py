import torch

from qwen35_lab import reference


def test_reference_attention_is_noncausal():
    q = torch.zeros(2, 1, 1)
    v = torch.tensor([[[2.]], [[6.]]])
    cu = torch.tensor([0, 2], dtype=torch.int32)
    torch.testing.assert_close(reference.attention(q, q, v, cu), torch.full_like(v, 4.))


def test_reference_delta_uses_decayed_state_for_prediction():
    q = k = torch.ones(1, 1, 1)
    v = torch.full_like(q, 3.)
    state = torch.tensor([[[[99.]]], [[[2.]]]])
    cu = torch.tensor([0, 1], dtype=torch.int32)
    idx = torch.tensor([1], dtype=torch.int32)
    # A_log=a=b=bias=0 -> decay=1/2, beta=1/2.
    out, final = reference.delta_rule(q, k, v, torch.zeros(1), torch.zeros(1, 1),
        torch.zeros(1, 1), torch.zeros(1), state, cu, idx, use_qk_norm=False)
    assert out.item() == 2.0  # 1 + (3-1)/2, not 1 + (3-2)/2.
    assert final[1].item() == 2.0
    assert final[0].item() == 99.0
