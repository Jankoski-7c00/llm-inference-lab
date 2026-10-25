"""Thin, stream-preserving interfaces for the teaching kernels.

Indices are metadata: valid slots must be unique; independent sequences must
not write the same state. Call validate_metadata once when constructing a case,
not inside timed regions. Inputs must be finite. KDA/INT4/backward are excluded.
"""

import math

import torch

from .extension import extension

INPUT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
FP8 = torch.float8_e4m3fn


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same_device(*tensors):
    _require(bool(tensors) and all(t.is_cuda for t in tensors), "all tensors must be CUDA tensors")
    _require(all(t.device == tensors[0].device for t in tensors), "all tensors must share a device")


def _vectors(*tensors):
    _require(all(t.ndim == 3 and t.stride(-1) == 1 for t in tensors), "expected [tokens, heads, dim] with contiguous last dimension")
    _require(tensors[0].dtype in INPUT_DTYPES, "inputs must be float32, float16 or bfloat16")
    _require(all(t.dtype == tensors[0].dtype for t in tensors), "input dtypes must match")
    _require(all(t.shape[1] > 0 and t.shape[2] > 0 for t in tensors), "head counts and dimensions must be positive")


def _slots(slots, tokens):
    _require(slots.dtype == torch.int64 and slots.ndim == 1 and slots.is_contiguous(), "slot_mapping must be contiguous int64 [tokens]")
    _require(slots.numel() == tokens, "slot_mapping length mismatch")


def _layout(k, v, kc, vc, dynamic=False):
    _vectors(k, v)
    _require(k.shape[:2] == v.shape[:2], "K/V token and head counts must match")
    _require(kc.dtype == vc.dtype, "cache dtypes must match")
    heads = k.shape[1]
    if kc.ndim == 4:
        _require(vc.ndim == 4 and kc.shape[:3] == vc.shape[:3], "expected NHD caches [blocks, slots, heads, dim]")
        _require(kc.shape[2:] == k.shape[1:] and vc.shape[2:] == v.shape[1:], "cache head/dim mismatch")
        _require(kc.stride(-1) == vc.stride(-1) == 1, "cache last dimension must be contiguous")
    else:
        _require(not dynamic and kc.ndim == 5 and vc.ndim == 4, "head-major only supports ordinary cache writes")
        nb, nh, split, block_size, x = kc.shape
        _require(nh == heads and split*x == k.shape[-1], "head-major K shape mismatch")
        _require(vc.shape == (nb, heads, v.shape[-1], block_size), "head-major V shape mismatch")
    if not dynamic:
        _require(k.shape == v.shape, "ordinary cache requires equal K/V dimensions")
        _require(kc.dtype in (k.dtype, FP8), "cache must use input dtype or FP8 E4M3")
    else:
        _require(kc.dtype in (torch.int8, FP8), "dynamic cache must be INT8 or FP8 E4M3")


def reshape_cache(key, value, key_cache, value_cache, slot_mapping, *,
                  k_scale=1.0, v_scale=1.0, impl="optimized"):
    """Write paged KV in place. Native-dtype copies ignore quantization scales."""
    _same_device(key, value, key_cache, value_cache, slot_mapping)
    _layout(key, value, key_cache, value_cache)
    _slots(slot_mapping, key.shape[0])
    _require(impl in ("baseline", "optimized"), "unknown cache implementation")
    _require(all(math.isfinite(s) and s > 0 for s in (k_scale, v_scale)), "scales must be positive finite Python numbers")
    extension().cache(key, value, key_cache, value_cache, slot_mapping,
                      float(k_scale), float(v_scale), impl == "optimized")


def dynamic_cache(key, value, key_cache, value_cache, k_scale_cache,
                  v_scale_cache, slot_mapping, *, impl="optimized"):
    """Per-token/head absmax quantization, followed by paged writes in place."""
    _same_device(key, value, key_cache, value_cache, k_scale_cache, v_scale_cache, slot_mapping)
    _layout(key, value, key_cache, value_cache, dynamic=True)
    _slots(slot_mapping, key.shape[0])
    _require(impl in ("baseline", "optimized"), "unknown dynamic implementation")
    for s in (k_scale_cache, v_scale_cache):
        _require(s.dtype == torch.float32 and s.shape == key_cache.shape[:3], "scale caches must be float32 [blocks, slots, heads]")
    extension().dynamic_cache(key, value, key_cache, value_cache, k_scale_cache,
                              v_scale_cache, slot_mapping, impl == "optimized")


def attention(q, k, v, cu_seqlens, max_seqlen, *, scale=None, impl="wmma"):
    """Non-causal, no-dropout encoder attention. Output: [tokens, heads, dim]."""
    _same_device(q, k, v, cu_seqlens)
    _vectors(q, k, v)
    _require(q.shape == k.shape == v.shape, "encoder Q/K/V shapes must match")
    _require(0 < q.shape[-1] <= 256 and q.shape[1] > 0, "attention requires heads > 0 and 1 <= dim <= 256")
    _require(cu_seqlens.ndim == 1 and cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous int32")
    _require(cu_seqlens.numel() >= 2 and isinstance(max_seqlen, int) and max_seqlen >= 0, "provide a nonnegative Python max_seqlen and at least one sequence")
    choices = {"baseline": 0, "simt": 1, "wmma": 2, "async": 3}
    _require(impl in choices, "unknown attention implementation")
    if impl in ("wmma", "async"):
        _require(q.dtype != torch.float32, "WMMA requires float16 or bfloat16; use simt for float32")
        _require(torch.cuda.get_device_capability(q.device)[0] >= 8, "teaching WMMA/async paths require compute capability >= 8.0")
    if impl == "async":
        _require(q.shape[-1] % 8 == 0 and all(t.stride(i) % 8 == 0 for t in (k, v) for i in (0, 1)) and all(t.data_ptr() % 16 == 0 for t in (k, v)), "async 16-byte copies require aligned K/V pointers, dim and outer strides divisible by 8; use wmma otherwise")
    scale = q.shape[-1]**-0.5 if scale is None else float(scale)
    _require(math.isfinite(scale) and scale > 0, "scale must be positive and finite")
    return extension().attention(q, k, v, cu_seqlens, max_seqlen, scale, choices[impl])


def delta_rule(q, k, v, A_log, a, b, dt_bias, initial_state, cu_seqlens,
               state_indices, *, accepted_tokens=None, scale=None, beta=1.0,
               threshold=20.0, use_qk_norm=True, inplace=True, impl="optimized"):
    """Return (output, state/trace); state layout is [pool, value_heads, V, K].

    Flat token inputs correspond to vLLM's packed batch-of-one form. A_log and
    dt_bias are float32. indices is [sequences, max_tokens_per_sequence]; a
    one-dimensional indices vector is the single-token decode shorthand.
    Index 0 is reserved. An invalid initial state produces zero output and no
    state update; the upstream oracle comparison masks this padding output.
    """
    _same_device(q, k, v, A_log, a, b, dt_bias, initial_state, cu_seqlens, state_indices)
    _vectors(q, k, v)
    _require(q.shape == k.shape and q.shape[0] == v.shape[0], "delta Q/K/V shape mismatch")
    t, h, dk = q.shape
    hv, dv = v.shape[1:]
    _require(h > 0 and hv > 0 and hv % h == 0 and 0 < dk <= 256 and dv > 0, "require HV divisible by H and 1 <= K <= 256, V > 0")
    _require(all(x.is_contiguous() for x in (q, k, v, a, b, A_log, dt_bias, initial_state, cu_seqlens)), "delta inputs must be contiguous; explicitly pack views before timing")
    _require(A_log.dtype == dt_bias.dtype == torch.float32 and A_log.shape == dt_bias.shape == (hv,), "A_log/dt_bias must be float32 [value_heads]")
    _require(a.dtype == b.dtype == q.dtype and a.shape == b.shape == (t, hv), "a/b must match input dtype and [tokens, value_heads]")
    _require(initial_state.ndim == 4 and initial_state.shape[1:] == (hv, dv, dk) and initial_state.dtype in INPUT_DTYPES, "state must be [pool, value_heads, V, K] with a floating dtype")
    _require(initial_state.shape[0] >= 1, "state pool must include reserved entry 0")
    _require(cu_seqlens.dtype == torch.int32 and cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, "cu_seqlens must be int32 [sequences+1]")
    state_indices = state_indices[:, None] if state_indices.ndim == 1 else state_indices
    _require(state_indices.dtype == torch.int32 and state_indices.ndim == 2 and state_indices.shape[0] == cu_seqlens.numel()-1 and state_indices.shape[1] > 0 and state_indices.is_contiguous(), "indices must be contiguous int32 [sequences, width]")
    if accepted_tokens is None:
        accepted_tokens = torch.empty(0, dtype=torch.int32, device=q.device)
    else:
        _same_device(q, accepted_tokens)
        _require(accepted_tokens.dtype == torch.int32 and accepted_tokens.shape == (state_indices.shape[0],) and accepted_tokens.is_contiguous(), "accepted_tokens must be contiguous int32 [sequences]")
    _require(impl in ("baseline", "fused", "shared", "tiled", "optimized"), "unknown delta implementation")
    scale = dk**-0.5 if scale is None else float(scale)
    _require(math.isfinite(scale) and scale > 0 and math.isfinite(beta) and beta > 0 and 0 < threshold <= 80, "invalid scale, softplus beta or threshold")
    return tuple(extension().delta(q, k, v, A_log, a, b, dt_bias, initial_state,
        cu_seqlens, state_indices, accepted_tokens, scale, float(beta),
        float(threshold), use_qk_norm, inplace, {"baseline": 0, "fused": 1, "shared": 2, "tiled": 3, "optimized": 4}[impl]))


def validate_metadata(*, slots=None, capacity=None, cu_seqlens=None, tokens=None,
                      max_seqlen=None, state_indices=None, state_count=None,
                      accepted_tokens=None):
    """Synchronizing debug check: run once during input construction, not timing."""
    if slots is not None:
        valid = slots[slots >= 0].cpu().tolist()
        _require(capacity is not None and all(s < capacity for s in valid), "slot exceeds capacity")
        _require(len(valid) == len(set(valid)), "duplicate nonnegative slots race")
    if cu_seqlens is not None:
        cu = cu_seqlens.cpu().tolist()
        _require(len(cu) >= 2 and cu[0] == 0 and cu[-1] == tokens, "invalid sequence endpoints")
        lengths = [e-s for s, e in zip(cu, cu[1:])]
        _require(min(lengths) >= 0, "cu_seqlens must be monotonic")
        if max_seqlen is not None:
            _require(max(lengths) <= max_seqlen, "max_seqlen is too small")
        if state_indices is not None:
            idx = state_indices[:, None] if state_indices.ndim == 1 else state_indices
            rows = idx.cpu().tolist()
            _require(idx.shape[0] == len(lengths) and idx.shape[1] >= max(lengths), "state index width is too small")
            used = set()
            for row, length in zip(rows, lengths):
                valid = {i for i in row[:length] if i > 0}
                _require(state_count is not None and all(i < state_count for i in row if i > 0), "state index exceeds pool")
                _require(not used.intersection(valid), "independent sequences alias state writes")
                used.update(valid)
            if accepted_tokens is not None:
                _require(all(1 <= i <= idx.shape[1] for i in accepted_tokens.cpu().tolist()), "invalid accepted-token count")
