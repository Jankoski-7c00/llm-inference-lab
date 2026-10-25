import pytest
import torch

from qwen35_lab import api, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [64, 72, 128])
def test_noncausal_varlen_padding_and_qkv_views(cuda, dtype, dim):
    lengths = [1, 0, 17, 35]
    cu = torch.tensor([0, 1, 1, 18, 53], device=cuda, dtype=torch.int32)
    qkv = torch.randn(sum(lengths), 2, 3, dim, device=cuda, dtype=dtype)
    q, k, v = (qkv[:, :, i] for i in range(3))
    expected = reference.attention(q, k, v, cu)
    impls = ["baseline", "simt"] if dtype == torch.float32 else ["baseline", "simt", "wmma", "async"]
    for impl in impls:
        actual = api.attention(q, k, v, cu, max(lengths), impl=impl)
        tol = 1e-4 if dtype == torch.float32 else (5e-3 if dtype == torch.float16 else 2e-2)
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("impl", ["baseline", "simt", "wmma", "async"])
def test_large_logits_are_stable_and_do_not_leak_between_sequences(cuda, impl):
    q = torch.randn(9, 1, 72, device=cuda, dtype=torch.float16)*25
    k = torch.randn_like(q)*25
    v = torch.randn_like(q)
    cu = torch.tensor([0, 4, 9], device=cuda, dtype=torch.int32)
    expected = reference.attention(q, k, v, cu)
    actual = api.attention(q, k, v, cu, 5, impl=impl)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=5e-3)


def test_float32_tail_dim_and_empty_sequences(cuda):
    q = torch.randn(3, 1, 33, device=cuda)
    cu = torch.tensor([0, 0, 3], device=cuda, dtype=torch.int32)
    expected = reference.attention(q, q, q, cu)
    for impl in ("baseline", "simt"):
        torch.testing.assert_close(api.attention(q, q, q, cu, 3, impl=impl), expected, atol=1e-4, rtol=1e-4)
    empty = q[:0]
    cu0 = torch.tensor([0, 0], device=cuda, dtype=torch.int32)
    assert api.attention(empty, empty, empty, cu0, 0, impl="simt").shape == empty.shape


def test_invalid_async_alignment_is_explicit(cuda):
    q = torch.zeros(2, 1, 65, dtype=torch.float16, device=cuda)
    cu = torch.tensor([0, 2], dtype=torch.int32, device=cuda)
    with pytest.raises(ValueError, match="16-byte"):
        api.attention(q, q, q, cu, 2, impl="async")


def test_async_rejects_misaligned_origin_but_wmma_accepts_it(cuda):
    q = torch.randn(2*72+1, dtype=torch.float16, device=cuda)[1:].view(2, 1, 72)
    cu = torch.tensor([0, 2], dtype=torch.int32, device=cuda)
    with pytest.raises(ValueError, match="aligned K/V pointers"):
        api.attention(q, q, q, cu, 2, impl="async")
    torch.testing.assert_close(api.attention(q, q, q, cu, 2, impl="wmma"), reference.attention(q, q, q, cu), atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("dim", [16, 80, 256])
def test_wmma_padding_and_dynamic_shared_memory_paths(cuda, dim):
    q = torch.randn(18, 1, dim, dtype=torch.float16, device=cuda)
    k, v = torch.randn_like(q), torch.randn_like(q)
    cu = torch.tensor([0, 1, 18], dtype=torch.int32, device=cuda)
    expected = reference.attention(q, k, v, cu)
    for impl in ("simt", "wmma", "async"):
        torch.testing.assert_close(api.attention(q, k, v, cu, 17, impl=impl), expected, atol=5e-3, rtol=5e-3)
