"""Load only the pinned upstream Triton functions, without installing vLLM.

The function bodies and decorators are copied verbatim into an ignored cache.
Their original copyright/license headers, source path and SHA256 are retained.
They are correctness oracles; their lightweight launch adapters are lab code.
"""

import ast
import functools
import hashlib
import importlib.util
import sys

import torch

from .extension import LAB_ROOT, REPO_ROOT

CACHE_SOURCE = "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"
DELTA_SOURCE = "vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py"


@functools.lru_cache(maxsize=None)
def _kernel(relative_path, name):
    path = REPO_ROOT / "third_party" / "vllm" / relative_path
    source = path.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])-1
    lines = source.splitlines(keepends=True)
    first_import = next(n.lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
    header = "".join(lines[:first_import-1])
    digest = hashlib.sha256(source.encode()).hexdigest()
    text = (header + f"\n# Source: {relative_path}\n# SHA256: {digest}\n"
            + "import torch\nimport triton\nimport triton.language as tl\n\n"
            + "".join(lines[start:node.end_lineno]))
    cache = LAB_ROOT / ".cache" / "upstream"
    cache.mkdir(parents=True, exist_ok=True)
    module_name = f"qwen35_upstream_{name}_{digest[:12]}"
    target = cache / f"{module_name}.py"
    if not target.exists() or target.read_text() != text:
        target.write_text(text)
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def reshape_cache(key, value, key_cache, value_cache, slot_mapping, *, k_scale=1.0, v_scale=1.0):
    import triton

    kernel = _kernel(CACHE_SOURCE, "reshape_and_cache_kernel_flash")
    hm = key_cache.ndim == 5
    d = key.shape[-1]
    block_size = key_cache.shape[3] if hm else key_cache.shape[1]
    x = key_cache.shape[4] if hm else 1
    tile = min(2048, triton.next_power_of_2(key.shape[1]*d))
    if torch.cuda.get_device_capability(key.device)[0] < 9:
        tile = min(512, tile)
    if not key.shape[0]:
        return
    # The upstream ordinary writer shares K/V block/head strides.
    if key_cache.stride(0) != value_cache.stride(0):
        raise ValueError("upstream oracle requires equal K/V block strides")
    kernel[(key.shape[0], triton.cdiv(key.shape[1]*d, tile))](
        key_ptr=key, value_ptr=value, key_cache_ptr=key_cache, value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=torch.tensor(k_scale, dtype=torch.float32, device=key.device),
        v_scale=torch.tensor(v_scale, dtype=torch.float32, device=key.device),
        key_stride=key.stride(0), value_stride=value.stride(0),
        block_stride=key_cache.stride(0),
        head_stride=key_cache.stride(1 if hm else 2),
        dim_stride_k=key_cache.stride(2) if hm else 0,
        dim_stride_v=value_cache.stride(2) if hm else 0,
        page_stride=key_cache.stride(1), num_heads=key.shape[1], head_size=d,
        block_size=block_size, x=x, USE_HEAD_MAJOR_LAYOUT=hm,
        FP8_KV_CACHE=key_cache.dtype == torch.float8_e4m3fn,
        TILE_SIZE=tile, num_warps=16, num_stages=10,
    )


def dynamic_cache(key, value, key_cache, value_cache, k_scale_cache, v_scale_cache, slots):
    import triton

    if not key.shape[0]:
        return
    kernel = _kernel(CACHE_SOURCE, "_reshape_cache_per_token_head")
    padded = triton.next_power_of_2(max(key.shape[-1], value.shape[-1]))
    integer = key_cache.dtype == torch.int8
    kernel[(key.shape[0], key.shape[1])](
        key_ptr=key, value_ptr=value, key_cache_ptr=key_cache, value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache, v_scale_cache_ptr=v_scale_cache, slot_mapping_ptr=slots,
        stride_key_tok=key.stride(0), stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0), stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0), stride_kc_slot=key_cache.stride(1), stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0), stride_vc_slot=value_cache.stride(1), stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0), stride_ks_slot=k_scale_cache.stride(1), stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0), stride_vs_slot=v_scale_cache.stride(1), stride_vs_head=v_scale_cache.stride(2),
        block_size=key_cache.shape[1], head_size=key.shape[-1], head_size_v=value.shape[-1],
        HEAD_SIZE_PADDED=padded, QUANT_MAX=127.0 if integer else 448.0,
        QUANT_MIN=-128.0 if integer else -448.0, IS_INT_QUANT=integer,
        num_warps=min(16, max(1, padded//32)),
    )


def delta_rule(q, k, v, A_log, a, b, dt_bias, initial_state, cu_seqlens,
               state_indices, *, accepted_tokens=None, scale=None, beta=1.0,
               threshold=20.0, use_qk_norm=True, inplace=True):
    import triton

    t, h, dk = q.shape
    hv, dv = v.shape[1:]
    idx = state_indices[:, None] if state_indices.ndim == 1 else state_indices
    out = torch.zeros_like(v)
    final = initial_state if inplace else initial_state.new_zeros(t, hv, dv, dk)
    if not t:
        return out, final
    kernel = _kernel(DELTA_SOURCE, "fused_sigmoid_gating_delta_rule_update_kernel")
    bk, bv = triton.next_power_of_2(dk), min(triton.next_power_of_2(dv), 32)
    kernel[(1, triton.cdiv(dv, bv), (cu_seqlens.numel()-1)*hv)](
        A_log=A_log, a=a, b=b, dt_bias=dt_bias, beta=beta, threshold=threshold,
        q=q, k=k, v=v, o=out, h0=initial_state, ht=final,
        cu_seqlens=cu_seqlens, ssm_state_indices=idx, num_accepted_tokens=accepted_tokens,
        scale=dk**-0.5 if scale is None else scale, N=cu_seqlens.numel()-1, T=t,
        B=1, H=h, HV=hv, K=dk, V=dv, BK=bk, BV=bv,
        stride_init_state_token=initial_state.stride(0), stride_final_state_token=final.stride(0),
        stride_indices_seq=idx.stride(0), stride_indices_tok=idx.stride(1),
        INPLACE_FINAL_STATE=inplace, USE_QK_L2NORM_IN_KERNEL=use_qk_norm,
        IS_KDA=False, num_warps=4, num_stages=3,
    )
    return out, final
