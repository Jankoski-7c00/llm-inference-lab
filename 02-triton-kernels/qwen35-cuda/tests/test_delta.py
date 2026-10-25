import pytest
import torch

from qwen35_lab import api, reference, upstream


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_delta_output_state_and_speculative_initial_index(cuda, dtype, norm, inplace):
    # V != K catches transposed-state bugs that square states cannot reveal.
    lengths = [1, 0, 3]
    t, h, hv, dk, dv = 4, 2, 4, 65, 33
    q = torch.randn(t, h, dk, device=cuda, dtype=dtype)*0.1
    k = torch.randn_like(q)*0.1
    v = torch.randn(t, hv, dv, device=cuda, dtype=dtype)*0.1
    alog = torch.randn(hv, device=cuda)*0.1
    bias = torch.randn(hv, device=cuda)*0.1
    a = torch.randn(t, hv, device=cuda, dtype=dtype)
    b = torch.randn_like(a)
    state = torch.randn(12, hv, dv, dk, device=cuda)*0.1
    cu = torch.tensor([0, 1, 1, 4], dtype=torch.int32, device=cuda)
    indices = torch.tensor([[1, 2, 3], [0, 0, 0], [6, 7, 8]], dtype=torch.int32, device=cuda)
    accepted = torch.tensor([1, 1, 2], dtype=torch.int32, device=cuda)
    args = (q, k, v, alog, a, b, bias)
    kwargs = dict(accepted_tokens=accepted, use_qk_norm=norm, inplace=inplace)
    ref_state = state.clone()
    expected = reference.delta_rule(*args, ref_state, cu, indices, **kwargs)
    for impl in ("baseline", "fused", "shared", "tiled", "optimized", "upstream"):
        actual_state = state.clone()
        if impl == "upstream":
            actual = upstream.delta_rule(*args, actual_state, cu, indices, **kwargs)
        else:
            actual = api.delta_rule(*args, actual_state, cu, indices, impl=impl, **kwargs)
        tol = 2e-4 if dtype == torch.float32 else (5e-3 if dtype == torch.float16 else 1e-2)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=tol, rtol=tol)
        torch.testing.assert_close(actual_state, ref_state, atol=tol, rtol=tol)


@pytest.mark.parametrize("state_dtype", [torch.float16, torch.bfloat16])
def test_single_token_low_precision_state_and_null_block(cuda, state_dtype):
    t, h, hv, dk, dv = 2, 1, 2, 128, 16
    q = torch.randn(t, h, dk, device=cuda, dtype=state_dtype)
    v = torch.randn(t, hv, dv, device=cuda, dtype=state_dtype)
    alog = torch.zeros(hv, device=cuda)
    a = torch.full((t, hv), 40., device=cuda, dtype=state_dtype)
    b = torch.full_like(a, -40)
    bias = torch.zeros_like(alog)
    state = torch.randn(3, hv, dv, dk, device=cuda, dtype=state_dtype)
    indices = torch.tensor([0, 2], dtype=torch.int32, device=cuda)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32, device=cuda)
    expected = reference.delta_rule(q, q, v, alog, a, b, bias, state.clone(), cu, indices)
    for impl in ("baseline", "fused", "shared", "tiled", "optimized"):
        actual_state = state.clone()
        actual = api.delta_rule(q, q, v, alog, a, b, bias, actual_state, cu, indices, impl=impl)
        assert actual[0][0].eq(0).all()
        torch.testing.assert_close(actual[1][0], state[0], atol=0, rtol=0)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=1e-2, rtol=1e-2)


def test_delta_debug_metadata_rejects_state_aliases_and_short_rows():
    cu = torch.tensor([0, 1, 3], dtype=torch.int32)
    with pytest.raises(ValueError, match="width"):
        api.validate_metadata(cu_seqlens=cu, tokens=3, state_indices=torch.tensor([[1], [2]]), state_count=5)
    with pytest.raises(ValueError, match="alias"):
        api.validate_metadata(cu_seqlens=cu, tokens=3, state_indices=torch.tensor([[1, 0], [1, 3]]), state_count=5)


def test_delta_full_key_width_and_custom_softplus_parameters(cuda):
    q = torch.randn(2, 1, 256, device=cuda)*0.1
    k = torch.randn_like(q)*0.1
    v = torch.randn(2, 2, 17, device=cuda)*0.1
    state = torch.randn(3, 2, 17, 256, device=cuda)*0.1
    alog = torch.zeros(2, device=cuda)
    a, b = torch.randn(2, 2, device=cuda), torch.randn(2, 2, device=cuda)
    bias = torch.ones(2, device=cuda)*0.1
    cu = torch.tensor([0, 2], dtype=torch.int32, device=cuda)
    idx = torch.tensor([[1, 2]], dtype=torch.int32, device=cuda)
    opts = dict(beta=0.5, threshold=10, scale=0.25)
    expected = reference.delta_rule(q, k, v, alog, a, b, bias, state.clone(), cu, idx, **opts)
    for impl in ("baseline", "fused", "shared", "tiled", "optimized"):
        actual = api.delta_rule(q, k, v, alog, a, b, bias, state.clone(), cu, idx, impl=impl, **opts)
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y, atol=2e-4, rtol=2e-4)
