"""nvfp4_ds_mla KV_FP8_ROPE=1 record writer (SM120).

``concat_and_cache_nvfp4_mla_fp8_rope`` quantizes the MLA compressed latent
(512 16-bit dims) plus the decoupled RoPE key (64 dims) into the compact
368 B/token ``nvfp4_ds_mla`` record that this package's fp8-RoPE sparse-MLA
readers consume (``traits.kv_gmem_stride == 368``; offsets anchored by
``prefill_mg._NVFP4_FP8_ROPE_SCALE_OFFSET == 288`` and
``prefill_mg._NVFP4_ROPE_GMEM_OFFSET == 304``):

    [   0, 256)  packed E2M1 NoPE (512 x 4-bit, 32 group-16 blocks)
    [ 256, 288)  32 x E4M3 group scale bytes (group amax / 6.0)
    [ 288, 292)  fp32 RoPE scale (rope amax / 448.0)
    [ 292, 304)  zero pad
    [ 304, 368)  64 x E4M3 RoPE

This is the KV_FP8_ROPE=1 layout: the stock 432 B record's [288, 304) pad
carries the fp32 RoPE scale and the RoPE bytes stay at their stock offset,
re-encoded E4M3 (128 B BF16 -> 64 B E4M3), so the record shrinks in place.

One CTA per token; ``slot_mapping`` entries < 0 are skipped (padded CUDA
graph slots). The quantization recipe is the standard NVFP4 one at an
implicit global scale of 1.0, spelled with the same PTX conversions the
rest of this package uses: ``amax * rcp.approx.ftz(6.0)`` ->
``cvt.rn.satfinite.e4m3x2`` scale byte -> hardware-exact E4M3 decode
(``cvt.rn.f16x2.e4m3x2`` -- denormal-correct, the same decode the read
path applies) -> ``rcp.approx.ftz`` inverse ->
``cvt.rn.satfinite.e2m1x2`` packing (``quantize_and_pack_16_fast``).

The RoPE lane stores ``scale = amax / 448.0`` as fp32 at [288, 292) and
``cvt.rn.satfinite.e4m3x2(v / scale)`` bytes at [304, 368); the readers
reconstruct ``e4m3_decode(byte) * scale``
(``prefill_mg._ld_global_nvfp4_fp8_rope_bfloat2``).

``per_token_scale=True`` selects the inline-scale two-level variant: the
NoPE lane derives its own per-token second-level scale instead of assuming
the implicit global 1.0 (which parks small-magnitude tokens' group scales in
E4M3 subnormals -- the defect the static per-layer
``VLLM_NVFP4_MLA_SCALES_FILE`` calibration papers over).  Warp 0 reduces the
32 group amaxes to the token amax (butterfly shuffle), stores

    s_t = token_amax / (6.0 * 448.0)

as fp32 at [292, 296) (the first 4 bytes of the pad; [296, 304) stays zero),
and every group scale byte is encoded relative to ``s_t`` so the largest
group's E4M3 scale lands at the top of the E4M3 range by construction.  The
readers reconstruct ``e2m1 * e4m3_decode(scale_byte) * s_t`` -- the same
expression as the static path with ``latent_scale := s_t`` sourced from the
record instead of the launch.  The record width, RoPE lane, and all offsets
are unchanged; legacy records (zero at [292, 296)) are NOT readable in this
mode, so the mode is server-static and joins the kernel compile identity.
"""

from __future__ import annotations

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import (
    KernelCompileSpec,
    launch as b12x_launch,
    tensor_compile_fact,
)
from b12x._lib.intrinsics import (
    cvt_e4m3_to_f32_via_f16,
    cvt_f32_to_e4m3,
    cvt_f32x4_to_e4m3x4,
    f16x2_to_f32x2,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    max_abs_16,
    quantize_and_pack_16_fast,
    rcp_approx_ftz,
    st_global_f32,
    st_global_u64,
    st_global_u8,
)
from b12x._lib.utils import current_cuda_stream

_KV_LORA_RANK = 512
_PE_DIM = 64
_GROUP_SIZE = 16
_NUM_GROUPS = _KV_LORA_RANK // _GROUP_SIZE  # 32
_NOPE_BYTES = _KV_LORA_RANK // 2  # 256
_SCALE_BYTES = _NUM_GROUPS  # 32
# KV_FP8_ROPE=1 record geometry (matches the shipped readers: fp32 scale in
# the stock record's pad, E4M3 RoPE at the stock RoPE offset).
_ROPE_SCALE_OFFSET = _NOPE_BYTES + _SCALE_BYTES  # 288
_PAD_OFFSET = _ROPE_SCALE_OFFSET + 4  # 292
_PAD_BYTES = 12
_ROPE_OFFSET = _PAD_OFFSET + _PAD_BYTES  # 304
_RECORD_BYTES = _ROPE_OFFSET + _PE_DIM  # 368
_THREADS = 128
# E4M3 rope scale: exact compile-time f32 constant (double 1/448 rounded to
# f32 once), NOT a runtime rcp.approx -- torch references must mirror this.
_E4M3_MAX_RCP = 1.0 / 448.0
# Per-token second-level (NVFP4 two-level) scale: fp32 at [292, 296), value
# token_amax / (E2M1_MAX * E4M3_MAX).  Same exact-constant contract as
# _E4M3_MAX_RCP -- torch references must mirror this.
_LATENT_SCALE_OFFSET = _PAD_OFFSET  # 292
_LATENT_SCALE_BYTES = 4
_TWO_LEVEL_RCP = 1.0 / (6.0 * 448.0)


@dsl_user_op
def _ld_global_u32(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Plain (coherent) 32-bit global load."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_to_f32x2(bf2: Uint32, *, loc=None, ip=None):
    """Exact promotion of packed bfloat16x2 to two float32 (no scaling)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


class ConcatAndCacheNvfp4MlaFp8RopeKernel:
    """Per-token GLM-family nvfp4_ds_mla record writer.

    Thread mapping (128 threads/CTA): threads 0-31 quantize one 16-dim
    group each (eight coherent 32-bit loads -> exact f32 promote -> E2M1
    pack + E4M3 scale byte); threads 0-11 zero the [292, 304) pad; thread
    32 quantizes the RoPE lane to E4M3 with one fp32 per-token scale.
    """

    def __init__(
        self,
        block_size: int,
        is_bf16: bool,
        per_token_scale: bool = False,
        no_rope: bool = False,
    ):
        self.block_size = int(block_size)
        self.is_bf16 = bool(is_bf16)
        self.per_token_scale = bool(per_token_scale)
        self.no_rope = bool(no_rope)

    @cute.jit
    def __call__(
        self,
        kv_c: cute.Tensor,  # (num_tokens, 512) bf16/f16
        k_pe: cute.Tensor,  # (num_tokens, 64) bf16/f16
        kv_cache: cute.Tensor,  # (num_blocks, block_size, 368) u8
        slot_mapping: cute.Tensor,  # (num_tokens, 1) int64
        kv_c_stride: Int32,  # kv_c.stride(0), elements
        k_pe_stride: Int32,  # k_pe.stride(0), elements
        block_stride: Int64,  # kv_cache.stride(0), bytes
        entry_stride: Int32,  # kv_cache.stride(1), bytes
        slot_capacity: Int32,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            kv_c,
            k_pe,
            kv_cache,
            slot_mapping,
            kv_c_stride,
            k_pe_stride,
            block_stride,
            entry_stride,
            slot_capacity,
        ).launch(
            grid=(num_tokens, 1, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kv_c: cute.Tensor,
        k_pe: cute.Tensor,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        kv_c_stride: Int32,
        k_pe_stride: Int32,
        block_stride: Int64,
        entry_stride: Int32,
        slot_capacity: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        token = Int32(token_idx)

        slot = Int64(slot_mapping[token])
        if (slot >= Int64(0)) & (slot < slot_capacity.to(Int64)):
            # Capacity is host-asserted in (0, 2^31), so the block/offset
            # split is safe in Int32 after the Int64 bounds check.
            slot32 = slot.to(Int32)
            block_idx = slot32 // Int32(self.block_size)
            block_off = slot32 % Int32(self.block_size)
            dst = (
                get_ptr_as_int64(kv_cache, 0)
                + block_idx.to(Int64) * block_stride
                + (block_off * entry_stride).to(Int64)
            )

            # --- NoPE: one 16-dim group per thread -> 8 B E2M1 + 1 scale byte.
            if tid < Int32(_NUM_GROUPS):
                src_elem = token * kv_c_stride + tid * Int32(_GROUP_SIZE)
                vals = cute.make_rmem_tensor((_GROUP_SIZE,), Float32)
                for i in cutlass.range_constexpr(_GROUP_SIZE // 2):
                    pair = _ld_global_u32(
                        get_ptr_as_int64(kv_c, src_elem + Int32(2 * i))
                    )
                    if cutlass.const_expr(self.is_bf16):
                        f0, f1 = _bf16x2_to_f32x2(pair)
                    else:
                        f0, f1 = f16x2_to_f32x2(pair)
                    vals[2 * i] = f0
                    vals[2 * i + 1] = f1

                group_amax = max_abs_16(vals)
                if cutlass.const_expr(self.per_token_scale):
                    # Two-level NVFP4: warp-reduce the 32 group amaxes to the
                    # token amax (threads 0-31 are exactly warp 0), derive
                    # s_t = token_amax/(6*448) so the largest group's E4M3
                    # scale byte encodes 448 (top of range, no subnormals),
                    # and quantize every group relative to s_t.  Lane 0
                    # stores s_t fp32 at [292, 296) for the readers.
                    token_amax = group_amax
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=1)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=2)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=4)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=8)
                    token_amax = fmax_f32(token_amax, tmp)
                    tmp = cute.arch.shuffle_sync_bfly(token_amax, offset=16)
                    token_amax = fmax_f32(token_amax, tmp)
                    latent_scale = token_amax * Float32(_TWO_LEVEL_RCP)
                    if tid == Int32(0):
                        latent_scale_offset = (
                            _ROPE_SCALE_OFFSET
                            if cutlass.const_expr(self.no_rope)
                            else _LATENT_SCALE_OFFSET
                        )
                        st_global_f32(dst + Int64(latent_scale_offset), latent_scale)
                    scale_u32 = Uint32(0)
                    packed64 = Uint64(0)
                    if latent_scale != Float32(0.0):
                        inv_latent = rcp_approx_ftz(latent_scale)
                        scale_f32 = (group_amax * inv_latent) * rcp_approx_ftz(
                            Float32(6.0)
                        )
                        scale_u32 = cvt_f32_to_e4m3(scale_f32)
                        decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                        if decoded_scale != Float32(0.0):
                            packed64 = quantize_and_pack_16_fast(
                                vals, rcp_approx_ftz(decoded_scale) * inv_latent
                            )
                    st_global_u64(dst + (tid * Int32(8)).to(Int64), packed64)
                    st_global_u8(
                        dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                    )
                else:
                    # NVFP4 block quant at global scale 1.0: scale byte =
                    # e4m3(amax/6); values scaled by rcp.approx.ftz of the
                    # hardware-exact decode of that byte (what the reader
                    # multiplies back), then satfinite E2M1.
                    scale_f32 = group_amax * rcp_approx_ftz(Float32(6.0))
                    scale_u32 = cvt_f32_to_e4m3(scale_f32)
                    decoded_scale = cvt_e4m3_to_f32_via_f16(scale_u32)
                    packed64 = Uint64(0)
                    if decoded_scale != Float32(0.0):
                        packed64 = quantize_and_pack_16_fast(
                            vals, rcp_approx_ftz(decoded_scale)
                        )
                    st_global_u64(dst + (tid * Int32(8)).to(Int64), packed64)
                    st_global_u8(
                        dst + Int64(_NOPE_BYTES) + tid.to(Int64),
                        cutlass.Uint8(scale_u32 & Uint32(0xFF)),
                    )

            if cutlass.const_expr(self.no_rope):
                # Rope-less dynamic record: [288,292) latent scale and
                # [292,304) zero pad. Static records end exactly at byte 288.
                if cutlass.const_expr(self.per_token_scale):
                    if tid < Int32(_PAD_BYTES):
                        st_global_u8(
                            dst + Int64(_PAD_OFFSET) + tid.to(Int64),
                            cutlass.Uint8(0),
                        )
            else:
                # RoPE record: [292,304) pad in static mode; [296,304) when
                # the per-token latent scale occupies [292,296).
                if cutlass.const_expr(self.per_token_scale):
                    if tid < Int32(_PAD_BYTES - _LATENT_SCALE_BYTES):
                        st_global_u8(
                            dst
                            + Int64(_PAD_OFFSET + _LATENT_SCALE_BYTES)
                            + tid.to(Int64),
                            cutlass.Uint8(0),
                        )
                else:
                    if tid < Int32(_PAD_BYTES):
                        st_global_u8(
                            dst + Int64(_PAD_OFFSET) + tid.to(Int64),
                            cutlass.Uint8(0),
                        )

            # --- RoPE lane: amax -> fp32 scale at [288, 292) -> satfinite
            # E4M3 bytes at [304, 368).
            if cutlass.const_expr(not self.no_rope):
                if tid == Int32(_NUM_GROUPS):
                    rope_vals = cute.make_rmem_tensor((_PE_DIM,), Float32)
                    for i in cutlass.range_constexpr(_PE_DIM // 2):
                        pair = _ld_global_u32(
                            get_ptr_as_int64(
                                k_pe, token * k_pe_stride + Int32(2 * i)
                            )
                        )
                        if cutlass.const_expr(self.is_bf16):
                            f0, f1 = _bf16x2_to_f32x2(pair)
                        else:
                            f0, f1 = f16x2_to_f32x2(pair)
                        rope_vals[2 * i] = f0
                        rope_vals[2 * i + 1] = f1
                    rope_amax = Float32(0.0)
                    for i in cutlass.range_constexpr(_PE_DIM):
                        rope_amax = fmax_f32(rope_amax, fabs_f32(rope_vals[i]))
                    rope_scale = rope_amax * Float32(_E4M3_MAX_RCP)
                    st_global_f32(dst + Int64(_ROPE_SCALE_OFFSET), rope_scale)
                    for w in cutlass.range_constexpr(_PE_DIM // 8):
                        q8 = Uint64(0)
                        if rope_scale != Float32(0.0):
                            inv = rcp_approx_ftz(rope_scale)
                            lo = cvt_f32x4_to_e4m3x4(
                                rope_vals[8 * w + 0] * inv,
                                rope_vals[8 * w + 1] * inv,
                                rope_vals[8 * w + 2] * inv,
                                rope_vals[8 * w + 3] * inv,
                            )
                            hi = cvt_f32x4_to_e4m3x4(
                                rope_vals[8 * w + 4] * inv,
                                rope_vals[8 * w + 5] * inv,
                                rope_vals[8 * w + 6] * inv,
                                rope_vals[8 * w + 7] * inv,
                            )
                            q8 = lo.to(Uint64) | (hi.to(Uint64) << Uint64(32))
                        st_global_u64(dst + Int64(_ROPE_OFFSET + 8 * w), q8)


@lru_cache(maxsize=None)
def _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel(
    block_size: int,
    is_bf16: bool,
    per_token_scale: bool = False,
    no_rope: bool = False,
) -> ConcatAndCacheNvfp4MlaFp8RopeKernel:
    return ConcatAndCacheNvfp4MlaFp8RopeKernel(
        block_size, is_bf16, per_token_scale, no_rope
    )


def clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache() -> None:
    _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel.cache_clear()


def _torch_to_cutlass_dtype(dtype: torch.dtype) -> type[cutlass.Numeric]:
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.uint8:
        return cutlass.Uint8
    if dtype == torch.int64:
        return cutlass.Int64
    raise TypeError(f"unsupported dtype {dtype}")


def _to_kernel_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int,
    leading_dim: int,
) -> cute.Tensor:
    cute_tensor = from_dlpack(tensor, assumed_align=assumed_align)
    cute_tensor.element_type = _torch_to_cutlass_dtype(tensor.dtype)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool = False,
) -> None:
    num_tokens = int(slot_mapping.shape[0])
    if num_tokens == 0:
        return
    block_size = int(kv_cache.shape[1])
    slot_capacity = int(kv_cache.shape[0]) * block_size
    is_bf16 = kv_c.dtype == torch.bfloat16
    no_rope = int(k_pe.shape[1]) == 0
    kernel = _build_concat_and_cache_nvfp4_mla_fp8_rope_kernel(
        block_size, is_bf16, per_token_scale, no_rope
    )

    # A zero-width DLPack tensor has no usable device pointer. The NoPE
    # specialization never reads this argument, so provide an aligned alias.
    k_pe_kernel = kv_c[:, :1] if no_rope else k_pe

    args = (
        _to_kernel_tensor(kv_c, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(k_pe_kernel, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(kv_cache, assumed_align=16, leading_dim=2),
        _to_kernel_tensor(slot_mapping, assumed_align=8, leading_dim=0),
        Int32(int(kv_c.stride(0))),
        Int32(int(k_pe_kernel.stride(0))),
        Int64(int(kv_cache.stride(0))),
        Int32(int(kv_cache.stride(1))),
        Int32(slot_capacity),
        Int32(num_tokens),
        current_cuda_stream(),
    )
    cache_key = (
        tensor_compile_fact("kv_c", kv_c, dynamic_dims=(0,), dynamic_strides=(0,)),
        tensor_compile_fact(
            "k_pe", k_pe_kernel, dynamic_dims=(0,), dynamic_strides=(0,)
        ),
        tensor_compile_fact(
            "kv_cache",
            kv_cache,
            dynamic_dims=(0,),
            dynamic_strides=(0, 1),
        ),
        tensor_compile_fact("slot_mapping", slot_mapping, dynamic_dims=(0,)),
        str(kv_c.dtype),
        block_size,
        bool(per_token_scale),
        bool(no_rope),
    )
    # Version 4: rope-less 288/304-byte records joined the specialization key.
    spec = KernelCompileSpec.from_key(
        "attention.mla.nvfp4_fp8_rope_kv_cache",
        4,
        cache_key,
        labels=(
            "kv_c",
            "k_pe",
            "kv_cache",
            "slot_mapping",
            "kv_dtype",
            "block_size",
            "per_token_scale",
            "no_rope",
        ),
    )
    b12x_launch(
        kernel,
        compile_spec=spec,
        compile_args=args,
        runtime_args=args,
    )


@torch.library.custom_op(
    "b12x::concat_and_cache_nvfp4_mla_fp8_rope",
    mutates_args=("kv_cache",),
)
def _concat_and_cache_nvfp4_mla_fp8_rope_op(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool = False,
) -> None:
    _concat_and_cache_nvfp4_mla_fp8_rope_flat_launch(
        kv_c, k_pe, kv_cache, slot_mapping, per_token_scale
    )


@_concat_and_cache_nvfp4_mla_fp8_rope_op.register_fake
def _concat_and_cache_nvfp4_mla_fp8_rope_fake(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    per_token_scale: bool = False,
) -> None:
    return None


def concat_and_cache_nvfp4_mla_fp8_rope(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: torch.Tensor | None = None,
    per_token_scale: bool = False,
) -> None:
    """Write ``num_tokens`` GLM-family nvfp4_ds_mla records.

    :param kv_c: MLA compressed latent, ``(>= num_tokens, 512)`` bf16/f16.
    :param k_pe: decoupled RoPE key, ``(>= num_tokens, 64)``, or a zero-width
        ``(>= num_tokens, 0)`` tensor for GLM NoPE; same dtype as ``kv_c``.
    :param kv_cache: paged uint8 cache viewed with a 368-byte RoPE record, or
        a 288-byte NoPE record (304 bytes with ``per_token_scale``).
    :param slot_mapping: ``(num_tokens,)`` int64 flat slot ids; entries outside
        ``[0, num_blocks * block_size)`` are skipped.
    :param scale: accepted for signature parity with the fp8 cache-op
        family; the nvfp4_ds_mla record has an implicit global scale of 1.0
        (group scales carry all magnitude), so it is unused.
    :param per_token_scale: write inline-scale two-level records: the
        per-token second-level scale ``token_amax/(6*448)`` is stored fp32 at
        bytes [292,296) for RoPE or [288,292) for NoPE, and group scales are
        encoded relative to it.
        Readers must run in the matching ``latent_scale_per_token`` mode.
    """
    del scale
    if kv_c.ndim != 2 or int(kv_c.shape[1]) != _KV_LORA_RANK:
        raise ValueError(
            f"kv_c must be (num_tokens, {_KV_LORA_RANK}), got {tuple(kv_c.shape)}"
        )
    if k_pe.ndim != 2 or int(k_pe.shape[1]) not in (0, _PE_DIM):
        raise ValueError(
            f"k_pe must be (num_tokens, 0 or {_PE_DIM}), got {tuple(k_pe.shape)}"
        )
    if kv_c.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"kv_c must be bf16/f16, got {kv_c.dtype}")
    if k_pe.dtype != kv_c.dtype:
        raise TypeError(f"k_pe dtype {k_pe.dtype} must match kv_c dtype {kv_c.dtype}")
    no_rope = int(k_pe.shape[1]) == 0
    expected_record_bytes = (
        (304 if per_token_scale else 288) if no_rope else _RECORD_BYTES
    )
    if kv_cache.ndim != 3 or int(kv_cache.shape[2]) != expected_record_bytes:
        raise ValueError(
            "kv_cache must be (num_blocks, block_size, "
            f"{expected_record_bytes}) uint8, got {tuple(kv_cache.shape)}"
        )
    if int(kv_cache.shape[0]) <= 0 or int(kv_cache.shape[1]) <= 0:
        raise ValueError("kv_cache num_blocks and block_size must be positive")
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int64:
        raise TypeError(
            "slot_mapping must be a 1-D int64 tensor, got "
            f"{tuple(slot_mapping.shape)} {slot_mapping.dtype}"
        )
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    num_tokens = int(slot_mapping.shape[0])
    if int(kv_c.shape[0]) < num_tokens or int(k_pe.shape[0]) < num_tokens:
        raise ValueError(
            f"kv_c/k_pe must cover slot_mapping's {num_tokens} tokens, got "
            f"{int(kv_c.shape[0])}/{int(k_pe.shape[0])} rows"
        )
    if kv_c.stride(1) != 1 or k_pe.stride(1) != 1 or kv_cache.stride(2) != 1:
        raise ValueError(
            "kv_c/k_pe rows and kv_cache records must be innermost-contiguous"
        )
    if kv_c.stride(0) % 2 != 0 or kv_c.data_ptr() % 4 != 0:
        # The group loads are 32-bit (element pairs), same as the CUDA writer.
        raise ValueError("kv_c rows must be 4-byte aligned (even row stride)")
    if not no_rope and (k_pe.stride(0) % 2 != 0 or k_pe.data_ptr() % 4 != 0):
        raise ValueError("k_pe rows must be 4-byte aligned (even row stride)")
    if (
        kv_cache.data_ptr() % 16 != 0
        or kv_cache.stride(0) % 16 != 0
        or kv_cache.stride(1) % 16 != 0
    ):
        raise ValueError("kv_cache records must be 16-byte aligned")
    if int(kv_cache.shape[0]) * int(kv_cache.shape[1]) >= 2**31:
        raise ValueError("kv_cache slot capacity must fit in int32")
    if not (
        kv_c.is_cuda and k_pe.is_cuda and kv_cache.is_cuda and slot_mapping.is_cuda
    ):
        raise ValueError("all tensors must be on CUDA")
    if len({kv_c.device, k_pe.device, kv_cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")

    torch.ops.b12x.concat_and_cache_nvfp4_mla_fp8_rope(
        kv_c, k_pe, kv_cache, slot_mapping, per_token_scale
    )


# ---------------------------------------------------------------------------
# GLM_NOPE FP8 record writer (fp8_ds_mla, 528 B/token).
#
# The G3 gap: no writer existed for the validated 528B rope-less FP8 record
# (pooled_indexer.py:64 ``_MLA_RECORD_BYTES == 528``; the byte layout is
# pinned by opt-work/fp8-native/test_fp8_native.py T9). The stock
# ``ops.concat_and_cache_mla`` fp8 path is the rope-bearing 656B V3.2/NSA
# writer and would corrupt a NoPE pool; the NVFP4 writer above packs a
# different latent format. This writer emits the exact record the b12x GLM
# decode/prefill readers consume at ``traits.kv_gmem_stride == 528``
# (ScaleFormat.ARBITRARY_FP32):
#
#     [   0, 512)  e4m3 NoPE latent (kv_lora_rank == 512), 1 B/elem
#     [ 512, 528)  4 x fp32 LE group scales, group size 128 (quant_tile 128)
#
# Scale convention (T9 / SESSION_KNOWLEDGE fp8 notes): ``s = amax / 448``
# stored as fp32 -- the scale-INVERSE (multiplicative) convention, matching
# the NVFP4 RoPE lane's ``scale = amax / 448.0`` and the readers'
# ``e4m3_decode(byte) * s`` reconstruction. An all-zero group keeps a unit
# scale (T9's recipe) so no reader can ever divide by a zero scale; the
# e4m3 payload is zeros either way and dequantizes to 0.0 exactly.
# ``k_scale`` is accepted for signature parity with the fp8 cache-op family
# (ops.concat_and_cache_mla) and deliberately unused: the record is
# self-describing (dynamic per-group amax scaling) and the ARBITRARY_FP32
# read path applies no per-tensor scale -- the same reason the NVFP4 writer
# deletes ``scale`` (implicit global scale 1.0).
#
# Device dispatch: on CUDA the record is packed by the Triton kernel below
# (one program per token). The first cut used pure-torch masked advanced
# indexing, which host-syncs (nonzero/count) and is therefore ILLEGAL inside
# CUDA-graph capture -- the spec warmup capture dies with
# cudaErrorStreamCaptureUnsupported. The Triton launch is capture-safe
# (per-program bounds check replaces the mask, exactly the nvfp4 CUDA
# writer's skip contract) and is used for EVERY CUDA call, eager or
# captured, so there is no eager/captured behavior seam (and the kernel
# compiles on the eager warmup forwards that precede graph capture). The
# pure-torch tail is the CPU reference path (harness/tests); both paths
# produce BIT-IDENTICAL records (T10 asserts byte equality), so swapping a
# future CuTeDSL/CUDA kernel in cannot silently change the ABI.
# Validation mirrors ``concat_and_cache_nvfp4_mla_fp8_rope`` -- shapes,
# dtypes, record contiguity (stride(1:) == (528, 1), the T5 gate), int32
# slot capacity, and the slot-mapping contract (entries outside
# [0, num_blocks*block_size) are skipped: padded CUDA-graph slots are -1,
# DCP padding is >= capacity). The raw pointer-alignment asserts of the
# NVFP4 writer (16B records / 4B rows) are hardware contracts of its
# CuTeDSL 32-bit group loads and are intentionally NOT enforced here; the
# CUDA kernel twin of this writer must re-add them (it should copy this
# function's validation block verbatim).

# Kept inside this section so the staged diff stays a pure append: triton
# rides along with the b12x package in the image (route_pack.py imports it
# the same way); a CPU-only import context must not require it.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    _HAS_TRITON = False

_GLM_NOPE_FP8_RECORD_BYTES = 528  # 512 e4m3 latent + 16B inline fp32 scales
_GLM_NOPE_FP8_LATENT_BYTES = 512  # kv_lora_rank; == io._GLM_NOPE_SCALE_BYTES head
_GLM_NOPE_FP8_GROUP_SIZE = 128    # quant_tile: 4 groups of 128 -> 4 fp32 scales
_GLM_NOPE_FP8_NUM_GROUPS = 4
_GLM_NOPE_FP8_E4M3_MAX = 448.0    # float8_e4m3fn max finite


@triton.jit
def _glm_nope_fp8_concat_cache_kernel(
    kv_c_ptr,      # *bf16/f16 [num_tokens, 512], innermost-contiguous
    kv_cache_ptr,  # *u8 [num_blocks, block_size, 528], records contiguous
    slot_ptr,      # *i64 [num_tokens]
    n_records,
    page_stride,
    rec_stride,
    page_tokens,
    row_stride,    # kv_c elements per row
    NUM_GROUPS: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    slot = tl.load(slot_ptr + pid)
    # Bounds check per program -- the masked/nonzero host sync the torch
    # path needs is illegal under graph capture; this mirrors the nvfp4
    # CUDA writer's "entries outside [0, capacity) are skipped" contract.
    if (slot < 0) | (slot >= n_records):
        return
    page = slot // page_tokens
    rec = slot - page * page_tokens
    base = kv_cache_ptr + page * page_stride + rec * rec_stride
    offs_g = tl.arange(0, NUM_GROUPS)
    offs_e = tl.arange(0, GROUP)
    vals = tl.load(
        kv_c_ptr + pid * row_stride + offs_g[:, None] * GROUP + offs_e[None, :]
    ).to(tl.float32)
    amax = tl.max(tl.abs(vals), axis=1)  # fp32 max is exact (order-free)
    scale = tl.math.div_rn(amax, 448.0)  # correctly-rounded: bit-exact vs the torch reference
    # all-zero groups keep a unit scale so no reader divides by zero
    scale = tl.where(scale > 0.0, scale, 1.0)
    mant = tl.math.div_rn(vals, scale[:, None])
    mant = tl.minimum(tl.maximum(mant, -448.0), 448.0)
    payload = mant.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(base + offs_g[:, None] * GROUP + offs_e[None, :], payload)
    # fp32 scales at [512, 528), 4 B per group, LSB-first == little-endian
    # by construction (T9 pins struct.pack("<f", s)).
    bits = scale.to(tl.uint32, bitcast=True)
    offs_b = tl.arange(0, 4)
    scale_bytes = ((bits[:, None] >> (8 * offs_b)[None, :]) & 0xFF).to(tl.uint8)
    tl.store(base + 512 + offs_g[:, None] * 4 + offs_b[None, :], scale_bytes)


def concat_and_cache_glm_nope_fp8_mla(
    kv_c: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
) -> None:
    """Write ``num_tokens`` GLM-family fp8_ds_mla 528B NoPE records.

    :param kv_c: MLA compressed latent, ``(>= num_tokens, 512)`` bf16/f16.
    :param kv_cache: paged uint8 cache viewed with a 528-byte NoPE record
        ``(num_blocks, block_size, 528)``; records contiguous,
        ``stride(1:) == (528, 1)``.
    :param slot_mapping: ``(num_tokens,)`` int64 flat slot ids; entries
        outside ``[0, num_blocks * block_size)`` are skipped (padded
        CUDA-graph slots and DCP padding).
    :param k_scale: accepted for signature parity with the fp8 cache-op
        family; unused -- the 528B record is self-describing (4 inline fp32
        group-128 scales carry all magnitude, ``s = amax / 448``), and the
        ARBITRARY_FP32 read path applies no per-tensor scale.
    """
    del k_scale
    if kv_c.ndim != 2 or int(kv_c.shape[1]) != _GLM_NOPE_FP8_LATENT_BYTES:
        raise ValueError(
            f"kv_c must be (num_tokens, {_GLM_NOPE_FP8_LATENT_BYTES}), "
            f"got {tuple(kv_c.shape)}"
        )
    if kv_c.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"kv_c must be bf16/f16, got {kv_c.dtype}")
    if (
        kv_cache.ndim != 3
        or int(kv_cache.shape[2]) != _GLM_NOPE_FP8_RECORD_BYTES
    ):
        raise ValueError(
            "kv_cache must be (num_blocks, block_size, "
            f"{_GLM_NOPE_FP8_RECORD_BYTES}) uint8, got {tuple(kv_cache.shape)}"
        )
    if int(kv_cache.shape[0]) <= 0 or int(kv_cache.shape[1]) <= 0:
        raise ValueError("kv_cache num_blocks and block_size must be positive")
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int64:
        raise TypeError(
            "slot_mapping must be a 1-D int64 tensor, got "
            f"{tuple(slot_mapping.shape)} {slot_mapping.dtype}"
        )
    if not slot_mapping.is_contiguous():
        raise ValueError("slot_mapping must be contiguous")
    num_tokens = int(slot_mapping.shape[0])
    if int(kv_c.shape[0]) < num_tokens:
        raise ValueError(
            f"kv_c must cover slot_mapping's {num_tokens} tokens, got "
            f"{int(kv_c.shape[0])} rows"
        )
    if kv_c.stride(1) != 1 or kv_cache.stride(2) != 1:
        raise ValueError(
            "kv_c rows and kv_cache records must be innermost-contiguous"
        )
    if (
        int(kv_cache.shape[1]) > 1
        and int(kv_cache.stride(1)) != _GLM_NOPE_FP8_RECORD_BYTES
    ):
        raise ValueError(
            "kv_cache records must be contiguous: stride(1:) must be "
            f"({_GLM_NOPE_FP8_RECORD_BYTES}, 1), got {tuple(kv_cache.stride())}"
        )
    if int(kv_cache.shape[0]) * int(kv_cache.shape[1]) >= 2**31:
        raise ValueError("kv_cache slot capacity must fit in int32")
    if len({kv_c.device, kv_cache.device, slot_mapping.device}) != 1:
        raise ValueError("all tensors must be on the same device")
    if num_tokens == 0:
        return
    if kv_c.is_cuda:
        # Triton path for EVERY CUDA call: the pure-torch masked indexing
        # host-syncs and dies inside CUDA-graph capture
        # (cudaErrorStreamCaptureUnsupported); a Triton launch is
        # capture-safe. There is deliberately no eager-torch fallback on
        # CUDA so eager and captured behavior cannot seam apart.
        if not _HAS_TRITON:
            raise RuntimeError(
                "concat_and_cache_glm_nope_fp8_mla CUDA path requires "
                "triton (the b12x image ships it); the pure-torch fallback "
                "cannot run under CUDA-graph capture"
            )
        _glm_nope_fp8_concat_cache_kernel[(num_tokens,)](
            kv_c,
            kv_cache,
            slot_mapping,
            int(kv_cache.shape[0]) * int(kv_cache.shape[1]),
            kv_cache.stride(0),
            kv_cache.stride(1),
            int(kv_cache.shape[1]),
            kv_c.stride(0),
            NUM_GROUPS=_GLM_NOPE_FP8_NUM_GROUPS,
            GROUP=_GLM_NOPE_FP8_GROUP_SIZE,
            num_warps=4,
        )
        return
    # CPU reference path (harness/tests). Group-128 dynamic e4m3 with
    # inline fp32 scale_inv (T9 byte layout): s = amax / 448; payload =
    # satfinite e4m3(v / s); all-zero groups keep s = 1.0. Byte-identical
    # to the Triton CUDA path (T10 asserts equality).
    num_blocks = int(kv_cache.shape[0])
    block_size = int(kv_cache.shape[1])
    capacity = num_blocks * block_size
    slots = slot_mapping
    valid = (slots >= 0) & (slots < capacity)
    latent = kv_c[:num_tokens].to(torch.float32)
    groups = latent.reshape(
        num_tokens, _GLM_NOPE_FP8_NUM_GROUPS, _GLM_NOPE_FP8_GROUP_SIZE
    )
    amax = groups.abs().amax(dim=2)
    scale_inv = amax / _GLM_NOPE_FP8_E4M3_MAX
    safe_den = torch.where(scale_inv > 0.0, scale_inv, torch.ones_like(scale_inv))
    mant = (groups / safe_den.unsqueeze(2)).clamp(
        -_GLM_NOPE_FP8_E4M3_MAX, _GLM_NOPE_FP8_E4M3_MAX
    )
    payload = mant.to(torch.float8_e4m3fn)
    record = torch.empty(
        (num_tokens, _GLM_NOPE_FP8_RECORD_BYTES),
        dtype=torch.uint8,
        device=kv_c.device,
    )
    record[:, :_GLM_NOPE_FP8_LATENT_BYTES].copy_(
        payload.view(torch.uint8).reshape(num_tokens, _GLM_NOPE_FP8_LATENT_BYTES)
    )
    # fp32 scales at [512, 528), 4 B per group: the device's little-endian
    # fp32 representation (T9 pins struct.pack("<f", s); x86 hosts and all
    # current CUDA devices are little-endian).
    record[:, _GLM_NOPE_FP8_LATENT_BYTES:].copy_(
        safe_den.contiguous().view(torch.uint8).reshape(num_tokens, 16)
    )
    pages = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots - pages * block_size
    kv_cache[pages[valid], offsets[valid]] = record[valid]
