from dataclasses import dataclass, field

import torch

from . import api, reference, upstream

PRESETS = {
    "cache_decode": ("cache", "paged write, 32 tokens, 4 KV heads, dim=256"),
    "cache_prefill": ("cache", "paged write, 2048 tokens, 4 KV heads, dim=256"),
    "cache_head_major": ("cache", "split K layout and head-major V"),
    "dynamic_int8": ("dynamic", "256 tokens, per-head INT8 scales, K=128 V=64"),
    "dynamic_fp8": ("dynamic", "256 tokens, per-head FP8 scales, K=128 V=64"),
    "encoder_short": ("attention", "two sequences, lengths 33/72, 4 heads, dim=72"),
    "encoder_long": ("attention", "one sequence, length 1024, 4 heads, dim=72"),
    "delta_decode": ("delta", "16 single-token requests, H=16 HV=32 K=V=128"),
    "delta_spec": ("delta", "4 requests x 3 tokens, accepted-state lookup"),
}


@dataclass
class Case:
    name: str
    op: str
    args: dict
    kwargs: dict = field(default_factory=dict)
    mutable: tuple = ()
    initial: dict = field(default_factory=dict)

    def freeze(self):
        self.initial = {name: self.args[name].clone() for name in self.mutable}
        return self

    def reset(self):
        for name, value in self.initial.items():
            self.args[name].copy_(value)

    def run(self, impl="optimized"):
        names = {"cache": "reshape_cache", "dynamic": "dynamic_cache", "attention": "attention", "delta": "delta_rule"}
        if impl == "reference":
            args = dict(self.args)
            args.pop("max_seqlen", None)
            if self.op in ("cache", "dynamic"):
                args["slots"] = args.pop("slot_mapping")
            return getattr(reference, names[self.op])(**args, **self.kwargs)
        if impl == "upstream":
            if self.op == "attention":
                raise ValueError("use the mathematical attention reference; no full flash-attention package is installed by this lab")
            args = dict(self.args)
            if self.op in ("cache", "dynamic"):
                args["slots" if self.op == "dynamic" else "slot_mapping"] = args.pop("slot_mapping")
            return getattr(upstream, names[self.op])(**args, **self.kwargs)
        if self.op == "attention" and impl == "optimized":
            impl = "async" if self.args["q"].dtype != torch.float32 else "simt"
        return getattr(api, names[self.op])(**self.args, **self.kwargs, impl=impl)

    def check(self, impl):
        self.reset()
        expected_result = self.run("reference")
        expected_mutable = {name: self.args[name].clone() for name in self.mutable}
        self.reset()
        actual = self.run(impl)
        if actual is not None:
            expected_seq = expected_result if isinstance(expected_result, tuple) else (expected_result,)
            actual_seq = actual if isinstance(actual, tuple) else (actual,)
            for x, y in zip(actual_seq, expected_seq):
                tol = 2e-4 if x.dtype == torch.float32 else (5e-3 if x.dtype == torch.float16 else 2e-2)
                torch.testing.assert_close(x, y, rtol=tol, atol=tol)
        for name, y in expected_mutable.items():
            x = self.args[name]
            if x.dtype == api.FP8:
                if impl == "upstream":
                    diff = (x.view(torch.uint8).short()-y.view(torch.uint8).short()).abs()
                    assert ((diff <= 1) | ((x.float() == 0) & (y.float() == 0))).all()
                else:
                    torch.testing.assert_close(x.view(torch.uint8), y.view(torch.uint8), rtol=0, atol=0)
            elif self.op == "delta":
                torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-4)
            elif "scale" in name:
                torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-8)
            elif impl == "upstream" and x.dtype == torch.int8:
                assert (x.short()-y.short()).abs().le(1).all()
            else:
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.reset()


def make_case(name, *, dtype=torch.float16, device="cuda", seed=17):
    if name not in PRESETS:
        raise ValueError(f"unknown case {name}")
    torch.manual_seed(seed)
    op = PRESETS[name][0]
    if op in ("cache", "dynamic"):
        t = 2048 if name == "cache_prefill" else (256 if op == "dynamic" else 32)
        h, d = 4, (128 if op == "dynamic" or name == "cache_head_major" else 256)
        dv = 64 if op == "dynamic" else d
        block, nb = 16, max(32, (t*2+15)//16)
        key = torch.randn(t, h, d, dtype=dtype, device=device)
        value = torch.randn(t, h, dv, dtype=dtype, device=device)
        slots = torch.randperm(nb*block, device=device)[:t].to(torch.int64)
        slots[0] = -1
        if op == "dynamic":
            dest = api.FP8 if name.endswith("fp8") else torch.int8
            kc = torch.zeros(nb, block, h, d, device=device).to(dest)
            vc = torch.zeros(nb, block, h, dv, device=device).to(dest)
        elif name == "cache_head_major":
            kc = torch.zeros(nb, h, d//8, block, 8, dtype=dtype, device=device)
            vc = torch.zeros(nb, h, d, block, dtype=dtype, device=device)
        else:
            kc = torch.zeros(nb, block, h, d, dtype=dtype, device=device)
            vc = torch.zeros_like(kc)
        args = dict(key=key, value=value, key_cache=kc, value_cache=vc, slot_mapping=slots)
        mutable = ("key_cache", "value_cache")
        if op == "dynamic":
            args.update(k_scale_cache=torch.zeros(nb, block, h, device=device), v_scale_cache=torch.zeros(nb, block, h, device=device))
            mutable += ("k_scale_cache", "v_scale_cache")
        api.validate_metadata(slots=slots, capacity=nb*block)
        return Case(name, op, args, mutable=mutable).freeze()
    if op == "attention":
        lengths = [33, 72] if name == "encoder_short" else [1024]
        cu = torch.tensor([0]+list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device)
        qkv = torch.randn(sum(lengths), 4, 3, 72, dtype=dtype, device=device)
        q, k, v = (qkv[:, :, i] for i in range(3))
        api.validate_metadata(cu_seqlens=cu, tokens=q.shape[0], max_seqlen=max(lengths))
        return Case(name, op, dict(q=q, k=k, v=v, cu_seqlens=cu, max_seqlen=max(lengths)))
    n, length = (16, 1) if name == "delta_decode" else (4, 3)
    t, h, hv, dk, dv = n*length, 16, 32, 128, 128
    q = torch.randn(t, h, dk, dtype=dtype, device=device)*0.1
    k = torch.randn_like(q)*0.1
    v = torch.randn(t, hv, dv, dtype=dtype, device=device)*0.1
    state = torch.randn(t+1, hv, dv, dk, device=device)*0.1
    cu = torch.arange(0, t+1, length, dtype=torch.int32, device=device)
    idx = torch.arange(1, t+1, dtype=torch.int32, device=device).reshape(n, length)
    args = dict(q=q, k=k, v=v, A_log=torch.zeros(hv, device=device),
                a=torch.randn(t, hv, dtype=dtype, device=device),
                b=torch.randn(t, hv, dtype=dtype, device=device),
                dt_bias=torch.zeros(hv, device=device), initial_state=state,
                cu_seqlens=cu, state_indices=idx)
    kwargs = {} if length == 1 else dict(accepted_tokens=torch.full((n,), 2, dtype=torch.int32, device=device))
    api.validate_metadata(cu_seqlens=cu, tokens=t, state_indices=idx, state_count=t+1, **kwargs)
    return Case(name, op, args, kwargs, ("initial_state",)).freeze()
