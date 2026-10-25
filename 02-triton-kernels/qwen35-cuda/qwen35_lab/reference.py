"""Independent mathematical references. Host loops are intentionally untimed."""

import torch
import torch.nn.functional as F

from .api import FP8


def _write(cache, token, slot, head_major=False, key=False):
    if not head_major:
        block, off = divmod(slot, cache.shape[1])
        cache[block, off].copy_(token)
    else:
        block_size = cache.shape[3]
        block, off = divmod(slot, block_size)
        if key:
            cache[block, :, :, off, :].copy_(token.reshape(cache.shape[1], cache.shape[2], cache.shape[4]))
        else:
            cache[block, :, :, off].copy_(token)


def reshape_cache(key, value, key_cache, value_cache, slots, *, k_scale=1.0, v_scale=1.0):
    for t, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        for source, dest, scale, is_key in ((key, key_cache, k_scale, True), (value, value_cache, v_scale, False)):
            x = source[t]
            if dest.dtype == FP8:
                scale32 = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
                x = (x.double()/scale32.double()).float().clamp(-448, 448).to(FP8)
            _write(dest, x, slot, key_cache.ndim == 5, is_key)


def dynamic_cache(key, value, key_cache, value_cache, k_scale_cache, v_scale_cache, slots):
    integer = key_cache.dtype == torch.int8
    qmax, qmin = (127.0, -128.0) if integer else (448.0, -448.0)
    for t, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            continue
        block, off = divmod(slot, key_cache.shape[1])
        for source, dest, scales in ((key, key_cache, k_scale_cache), (value, value_cache, v_scale_cache)):
            x = source[t].float()
            # Round each division to FP32, rather than inheriting a tensor
            # scalar-divide implementation's reciprocal-multiply shortcut.
            scale = (x.abs().amax(-1).double()/qmax).float().clamp_min(1e-6)
            inverse = scale.double().reciprocal().float()
            y = x * inverse[:, None]
            if integer:
                y = torch.where(y >= 0, torch.trunc(y+0.5), torch.trunc(y-0.5))
            dest[block, off].copy_(y.clamp(qmin, qmax).to(dest.dtype))
            scales[block, off].copy_(scale)


def attention(q, k, v, cu_seqlens, *, scale=None):
    scale = q.shape[-1]**-0.5 if scale is None else scale
    out = torch.empty_like(q)
    cu = cu_seqlens.cpu().tolist()
    for s, e in zip(cu, cu[1:]):
        if s == e:
            continue
        qq, kk, vv = (x[s:e].double().transpose(0, 1) for x in (q, k, v))
        scores = (qq @ kk.transpose(-2, -1))*scale
        out[s:e].copy_((scores.softmax(-1) @ vv).transpose(0, 1).to(q.dtype))
    return out


def delta_rule(q, k, v, A_log, a, b, dt_bias, initial_state, cu_seqlens,
               state_indices, *, accepted_tokens=None, scale=None, beta=1.0,
               threshold=20.0, use_qk_norm=True, inplace=True):
    total, heads, dk = q.shape
    hv, dv = v.shape[1:]
    scale = dk**-0.5 if scale is None else scale
    indices = state_indices[:, None] if state_indices.ndim == 1 else state_indices
    indices = indices.cpu().tolist()
    cu = cu_seqlens.cpu().tolist()
    accepted = accepted_tokens.cpu().tolist() if accepted_tokens is not None else [1]*len(indices)
    out = torch.zeros_like(v)
    final = initial_state if inplace else initial_state.new_zeros(total, hv, dv, dk)
    head_map = torch.arange(hv, device=q.device)//(hv//heads)
    for n, (s, e) in enumerate(zip(cu, cu[1:])):
        if s == e:
            continue
        init = indices[n][accepted[n]-1]
        if init <= 0:
            continue
        state = initial_state[init].float().clone()
        for t in range(s, e):
            qq, kk = q[t].float(), k[t].float()
            if use_qk_norm:
                qq = qq*torch.rsqrt(qq.square().sum(-1, keepdim=True)+1e-6)
                kk = kk*torch.rsqrt(kk.square().sum(-1, keepdim=True)+1e-6)
            qq, kk = qq[head_map]*scale, kk[head_map]
            decay = (-A_log.float().exp()*F.softplus(a[t].float()+dt_bias.float(), beta=beta, threshold=threshold)).exp()
            weight = b[t].float().sigmoid()
            state = state*decay[:, None, None]
            prediction = torch.einsum("hvk,hk->hv", state, kk)
            residual = (v[t].float()-prediction)*weight[:, None]
            state = state+residual[:, :, None]*kk[:, None, :]
            out[t].copy_(torch.einsum("hvk,hk->hv", state, qq).to(out.dtype))
            dest = indices[n][t-s]
            if not inplace:
                final[t].copy_(state.to(final.dtype))
            elif dest > 0:
                final[dest].copy_(state.to(final.dtype))
    return out, final
