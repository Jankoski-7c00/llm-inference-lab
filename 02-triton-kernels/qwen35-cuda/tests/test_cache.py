import pytest
import torch

from qwen35_lab import api, reference, upstream


def assert_cache(a, b, *, upstream_fp8=False):
    if a.dtype == api.FP8:
        if upstream_fp8:
            # Observed PTX: cvt.rz.f16.f32 followed by FP8 RNE. The lab uses
            # direct FP32->FP8 RNE. Only the upstream comparison permits one
            # adjacent finite code, never the lab-vs-mathematical reference.
            diff = (a.view(torch.uint8).short()-b.view(torch.uint8).short()).abs()
            both_zero = (a.float() == 0) & (b.float() == 0)
            assert torch.isfinite(a.float()).all() and torch.isfinite(b.float()).all()
            assert ((diff <= 1) | both_zero).all(), "upstream differs by more than one FP8 code"
        else:
            torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), atol=0, rtol=0)
    else:
        if upstream_fp8 and a.dtype == torch.int8:
            # Upstream div.full.f32 can move a value across an INT8 half tie.
            assert (a.short()-b.short()).abs().le(1).all()
        else:
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["nhd", "head_major"])
@pytest.mark.parametrize("quantized", [False, True])
def test_cache_random_slots_padding_and_untouched_entries(cuda, dtype, layout, quantized):
    t, h, d, block, nb = 7, 3, 72, 4, 4
    key = torch.randn(t, 3, h, d, device=cuda, dtype=dtype)[:, 0]
    value = torch.randn_like(key)
    slots = torch.tensor([15, 0, -1, 4, 7, 9, 3], device=cuda)
    dest_dtype = api.FP8 if quantized else dtype
    shape_k = (nb, block, h, d) if layout == "nhd" else (nb, h, d//8, block, 8)
    shape_v = (nb, block, h, d) if layout == "nhd" else (nb, h, d, block)
    initial_k = torch.full(shape_k, 3.0, device=cuda).to(dest_dtype)
    initial_v = torch.full(shape_v, -2.0, device=cuda).to(dest_dtype)
    expected_k, expected_v = initial_k.clone(), initial_v.clone()
    reference.reshape_cache(key, value, expected_k, expected_v, slots, k_scale=0.125, v_scale=0.25)
    for implementation in ("baseline", "optimized", "upstream"):
        kc, vc = initial_k.clone(), initial_v.clone()
        if implementation == "upstream":
            upstream.reshape_cache(key, value, kc, vc, slots, k_scale=0.125, v_scale=0.25)
        else:
            api.reshape_cache(key, value, kc, vc, slots, k_scale=0.125, v_scale=0.25, impl=implementation)
        assert_cache(kc, expected_k, upstream_fp8=implementation == "upstream")
        assert_cache(vc, expected_v, upstream_fp8=implementation == "upstream")


@pytest.mark.parametrize("d", [1, 33, 128, 256])
def test_cache_tail_dimension_and_empty_input(cuda, d):
    key = torch.randn(2, 2, d, device=cuda)
    value = torch.randn_like(key)
    slots = torch.tensor([5, 0], device=cuda)
    initial = torch.full((2, 3, 2, d), 99.0, device=cuda)
    expected_k, expected_v = initial.clone(), initial.clone()
    reference.reshape_cache(key, value, expected_k, expected_v, slots)
    for impl in ("baseline", "optimized"):
        kc, vc = initial.clone(), initial.clone()
        api.reshape_cache(key, value, kc, vc, slots, impl=impl)
        assert_cache(kc, expected_k)
        assert_cache(vc, expected_v)
        api.reshape_cache(key[:0], value[:0], kc, vc, slots[:0], impl=impl)
        assert_cache(kc, expected_k)


@pytest.mark.parametrize("dest_dtype", [torch.int8, api.FP8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dynamic_unequal_dimensions_strides_and_zero_scale(cuda, dest_dtype, dtype):
    t, h, dk, dv = 5, 3, 65, 33
    key = torch.randn(t, h, 2, dk, device=cuda, dtype=dtype)[:, :, 0]
    value = torch.randn(t, h, 2, dv, device=cuda, dtype=dtype)[:, :, 1]
    key[0].zero_()
    value[0].zero_()
    slots = torch.tensor([0, 7, -1, 3, 9], device=cuda)
    kc = torch.full((3, 4, h, dk), 7.0, device=cuda).to(dest_dtype)
    vc = torch.full((3, 4, h, dv), -7.0, device=cuda).to(dest_dtype)
    sk = torch.full((3, 4, h), 99.0, device=cuda)
    sv = sk.clone()
    expected = [x.clone() for x in (kc, vc, sk, sv)]
    reference.dynamic_cache(key, value, *expected, slots)
    for impl in ("baseline", "optimized", "upstream"):
        actual = [x.clone() for x in (kc, vc, sk, sv)]
        if impl == "upstream":
            upstream.dynamic_cache(key, value, *actual, slots)
        else:
            api.dynamic_cache(key, value, *actual, slots, impl=impl)
        assert_cache(actual[0], expected[0], upstream_fp8=impl == "upstream")
        assert_cache(actual[1], expected[1], upstream_fp8=impl == "upstream")
        for x, y in zip(actual[2:], expected[2:]):
            torch.testing.assert_close(x, y, atol=1e-8, rtol=1e-6)


def test_dynamic_int8_half_away_from_zero_and_extremes(cuda):
    # absmax=127 -> scale exactly 1; distinguish ties-away from torch.round.
    key = torch.tensor([[[127., 0.5, -0.5, 1.5, -1.5, 0, -127]]], device=cuda)
    value = key.clone()
    kc = torch.zeros(1, 1, 1, 7, dtype=torch.int8, device=cuda)
    vc = kc.clone()
    sk = torch.zeros(1, 1, 1, device=cuda)
    sv = sk.clone()
    slots = torch.zeros(1, dtype=torch.int64, device=cuda)
    for impl in ("baseline", "optimized"):
        api.dynamic_cache(key, value, kc, vc, sk, sv, slots, impl=impl)
        assert kc.flatten().tolist() == [127, 1, -1, 2, -2, 0, -127]
        assert sk.item() == 1


def test_fp8_native_conversion_avoids_intermediate_fp16_rounding(cuda):
    key = torch.tensor([[[-5.2522053718566895, 8.504573822021484]]], device=cuda)
    kc = torch.zeros(1, 1, 1, 2, device=cuda).to(api.FP8)
    vc = kc.clone()
    slots = torch.zeros(1, dtype=torch.int64, device=cuda)
    for impl in ("baseline", "optimized"):
        api.reshape_cache(key, key, kc, vc, slots, impl=impl)
        assert kc.float().flatten().tolist() == [-5.5, 9.0]


def test_static_fp8_nonbinary_scale_near_rounding_boundaries(cuda):
    boundaries = torch.tensor([1.0625, 1.1875, 1.3125, 1.4375, 3.25, 5.25, 8.5, 17.], device=cuda)
    slots = torch.zeros(1, dtype=torch.int64, device=cuda)
    for scale in (0.13, 0.21, 0.3, 0.7):
        scale32 = torch.tensor(scale, dtype=torch.float32, device=cuda)
        centers = (boundaries.double()*scale32.double()).float()
        values = torch.stack((torch.nextafter(centers, torch.zeros_like(centers)), centers,
                              torch.nextafter(centers, torch.full_like(centers, float("inf")))))
        key = values.reshape(1, 1, -1)
        kc = torch.zeros(1, 1, 1, key.numel(), device=cuda).to(api.FP8)
        vc, ek, ev = kc.clone(), kc.clone(), kc.clone()
        reference.reshape_cache(key, key, ek, ev, slots, k_scale=scale, v_scale=scale)
        for impl in ("baseline", "optimized"):
            api.reshape_cache(key, key, kc, vc, slots, k_scale=scale, v_scale=scale, impl=impl)
            assert_cache(kc, ek)
            assert_cache(vc, ev)


def test_debug_metadata_rejects_races_and_out_of_range_slots():
    with pytest.raises(ValueError, match="duplicate"):
        api.validate_metadata(slots=torch.tensor([0, 0]), capacity=4)
    with pytest.raises(ValueError, match="capacity"):
        api.validate_metadata(slots=torch.tensor([4]), capacity=4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quantized", [False, True])
def test_vector_path_falls_back_for_misaligned_view_origins(cuda, dtype, quantized):
    def offset_tensor(shape, dt):
        n = 1
        for size in shape:
            n *= size
        return torch.zeros(n+1, device=cuda, dtype=dt)[1:].view(shape)

    key = offset_tensor((2, 2, 8), dtype).normal_()
    value = offset_tensor((2, 2, 8), dtype).normal_()
    dest_dtype = api.FP8 if quantized else dtype
    kc = offset_tensor((2, 4, 2, 8), dest_dtype)
    vc = offset_tensor((2, 4, 2, 8), dest_dtype)
    slots = torch.tensor([0, 7], device=cuda)
    expected_k, expected_v = kc.clone(), vc.clone()
    reference.reshape_cache(key, value, expected_k, expected_v, slots)
    api.reshape_cache(key, value, kc, vc, slots)
    assert_cache(kc, expected_k)
    assert_cache(vc, expected_v)
