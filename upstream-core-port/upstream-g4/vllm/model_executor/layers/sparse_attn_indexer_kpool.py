# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.sparse_attn_indexer import (
    _merge_b12x_dcp_topk,
    _run_b12x_paged_topk,
)
from vllm.model_executor.layers.sparse_attn_indexer import (
    use_b12x_sparse_indexer as use_b12x_sparse_indexer_fn,
)
from vllm.models.glm5next.nvidia.ops.kpool_compress import (
    expand_pools_and_append_tail,
    expand_pools_to_tokens,
    kpool_compress_and_write_cache,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_paged_mqa_logits_32,
    kpool_seed_tail_cache,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

_KDA_INPUT_DEBUG: bool | None = None


def _kda_input_debug_enabled() -> bool:
    """VLLM_DEBUG_KDA_INPUTS=1 enables host-side validation of the kpool
    write inputs (slot mappings vs cache pools). Skipped during CUDA graph
    capture; diagnostics-only, one env read."""
    global _KDA_INPUT_DEBUG
    if _KDA_INPUT_DEBUG is None:
        _KDA_INPUT_DEBUG = bool(
            int(os.environ.get("VLLM_DEBUG_KDA_INPUTS", "0"))
        )
    if not _KDA_INPUT_DEBUG:
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        pass
    # [GUARD-WARMUP EXEMPTION] warmup/compile-warmup batches are synthetic
    # by design (spaced slot values vs dummy pools); never validate them.
    try:
        from vllm.v1.worker.gpu import warmup as _warmup_mod

        if getattr(_warmup_mod, "IN_WARMUP", False):
            return False
    except Exception:
        pass
    return _KDA_INPUT_DEBUG


def _kda_slot_guard(
    site: str,
    name: str,
    slot_mapping: torch.Tensor,
    slots_per_row: int,
    cache_rows: int,
    extra: dict | None = None,
) -> None:
    """Assert every valid (>=0) flat slot maps inside the cache pool.

    Slot granularities per cache layout (verified from the write kernels):
    the indexer K cache is ``[blocks, page_size, head_dim+4]`` and its
    ``loc`` is a flat token-granular slot (row = loc // page_size); the
    paged tail cache is ``[blocks, 2, kpool, head_dim]`` and its ``tslot``
    is ``block * kpool + pos % kpool`` (row = tslot // kpool). So the bound
    in both cases is ``slot < cache_rows * slots_per_row`` — NOT
    ``slot * kpool + kpool - 1 < rows`` (the earlier double-scaling
    arithmetic false-positived on every pool-granular slot past
    rows/kpool, e.g. slots 3019..3034 vs 12078 rows x kpool 4).
    """
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    sm = slot_mapping.to(torch.int64).cpu()
    valid = sm >= 0
    if not bool(valid.any()):
        return
    slots = sm[valid]
    bad = slots >= cache_rows * slots_per_row
    if bool(bad.any()):
        lines = [
            f"[KDA-INPUT-VIOLATION] site={site}: {name} flat slots "
            f"{slots[bad][:16].tolist()} map past the cache pool "
            f"(rows={cache_rows}, slots_per_row={slots_per_row})"
        ]
        lines.append(f"{name}_full={slots[:64].tolist()}")
        for k, v in (extra or {}).items():
            lines.append(f"{k}={v}")
        raise RuntimeError("\n".join(lines))

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32

# B12X's cross-rank row selector supports these widths. GLM selects 128 pools
# (2048 tokens / KPool16), so DCP ranks retain 512 local candidates for the
# exact merge and truncate the globally ranked result back to 128 pools.
_KPOOL_DCP_MERGE_TOPK = 512

# Direct DeepGEMM over a contiguous pool-K gather is materially faster for
# ordinary prompts.  Keep the paged B12X route for the large pooled histories
# where avoiding the transient logits/gather footprint matters.  GLM-5.3 uses
# IndexPool-4, so 65,536 pools corresponds to 262,144 source tokens.
_KPOOL_B12X_PAGED_PREFILL_MIN_POOLS = 65_536


def _can_use_b12x_kpool_prefill(
    *,
    use_b12x_sparse_indexer: bool,
    use_fp4_cache: bool,
    cache_page_size: int,
    num_reqs: int,
    total_seq_lens: int,
    local_total_seq_lens: int,
) -> bool:
    """Match the KPool metadata builder's replicated paged-prefill route."""
    return bool(
        use_b12x_sparse_indexer
        and not use_fp4_cache
        and cache_page_size == 64
        and num_reqs == 1
        and total_seq_lens >= _KPOOL_B12X_PAGED_PREFILL_MIN_POOLS
        and local_total_seq_lens == total_seq_lens
    )


# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts.
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def _build_decode_scatter_indices(
    decode_lens: torch.Tensor,
    num_requests: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (request id, intra-request index) for a non-uniform decode
    batch, with ``n == decode_lens.sum()`` as a host int (avoids a
    device sync and keeps both repeat_interleaves sync-free).

    Shared by every ``_scatter_decode_tokens_by_request`` call in a step:
    building it per call would repeat the same repeat_interleave/cumsum chain
    up to 5x per layer on the eager decode break.
    """
    device = decode_lens.device
    dl = decode_lens.to(torch.int64)
    req_id = torch.repeat_interleave(
        torch.arange(num_requests, device=device, dtype=torch.int64),
        dl,
        output_size=n,
    )
    req_starts = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=torch.int64), dl[:-1]]),
        dim=0,
    )
    # Broadcast the per-request start offsets to per-token (length n ==
    # dl.sum()) so each token's intra-request index subtracts its own
    # request's start.
    starts = torch.repeat_interleave(req_starts, dl, output_size=n)
    intra = torch.arange(n, device=device, dtype=torch.int64) - starts
    return req_id, intra


def _scatter_decode_tokens_by_request(
    tokens: torch.Tensor,
    pad_value,
    num_requests: int,
    lmax: int,
    scatter_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Group ``[N, ...]`` decode tokens into a padded ``[num_requests, lmax, ...]``
    layout: request ``r``'s tokens at row ``r`` in order; short requests padded.

    Unlike ``pack_seq_triton`` this is dtype-agnostic (needed for the int32
    slot/pos tensors) — it scatters with the shared per-step indices from
    ``_build_decode_scatter_indices``. Used only for the non-uniform
    (``requires_padding``) decode batch; uniform batches use a zero-copy
    reshape.
    """
    req_id, intra = scatter_indices
    out = torch.full(
        (num_requests, lmax, *tokens.shape[1:]),
        pad_value,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    # [APC-Xid43 fix] Bound the scatter to the padded (num_requests, lmax)
    # grid. A decode row carrying more tokens than the grid's width (e.g. a
    # short extend reclassified as a decode row under MTP verify) previously
    # drove ``intra`` past lmax and tripped ATen's IndexKernel bound assert
    # ("-sizes[i] <= index && index < sizes[i]") on a tiny per-request
    # write. Tokens beyond the representable width are dropped exactly like
    # pad rows, which the downstream consumers already skip.
    valid = intra < lmax
    linear = (req_id * lmax + intra.clamp(min=0, max=lmax - 1))[valid]
    flat_out = out.reshape(num_requests * lmax, *tokens.shape[1:])
    flat_out[linear] = tokens[valid]
    return out


def _decode_topk_seq_lens(
    positions: torch.Tensor,
    decode_lens: torch.Tensor,
    num_decode_tokens: int,
    batch_size: int,
    next_n: int,
    requires_padding: bool,
) -> torch.Tensor:
    """Token-granular seq_len (pos + 1) per pool-topk row, layout-aware.

    ``pool_topk`` (and the logits it comes from) follow the padded
    ``[batch_size, next_n]`` grid whenever ``requires_padding`` is set, so row
    ``(b, t)`` corresponds to flat decode token ``offset_b + t`` -- NOT
    ``b * next_n + t``. Slicing flat ``positions[: batch_size * next_n]``
    (the uniform-layout shortcut) misaligns every row after the first
    non-uniform request and, past the decode region, reads prefill tokens'
    positions; ``expand_pools_and_append_tail`` then anchors the tail at
    another request's length, dropping the row's real tail tokens or emitting
    indices past its sequence (out-of-bounds block-table reads). Padded rows
    get 0 (empty tail); they are dropped by ``unpack_seq_triton`` anyway.
    """
    n = batch_size * next_n
    if not requires_padding:
        return positions[:n].to(torch.int32) + 1
    scatter_idx = _build_decode_scatter_indices(
        decode_lens, batch_size, num_decode_tokens
    )
    padded = _scatter_decode_tokens_by_request(
        positions[:num_decode_tokens].to(torch.int32),
        -1,
        batch_size,
        next_n,
        scatter_idx,
    )
    return padded.reshape(n) + 1  # pad rows: -1 + 1 = 0 -> empty tail


def _normalize_native_paged_decode_rows(
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Flatten native MTP rows without scoring FULL-graph padding.

    Native MTP metadata stores context lengths as ``[B, next_n]`` while the
    model runner may pad the query tensor to a larger FULL-cudagraph token
    count.  The previous KPool32/B12X fast paths flattened only when
    ``num_decode_tokens == B * next_n``; a graph-padded ``3 -> 4`` MTP batch
    therefore retained rank-2 lengths and fatally rejected an otherwise valid
    request.  Padded query rows are scheduler-invisible, so score exactly the
    live metadata rows and leave the unused output-buffer tail untouched.
    """
    if seq_lens.dim() == 2:
        batch_size, next_n = seq_lens.shape
        seq_lens = seq_lens.reshape(-1).contiguous()
        block_table = block_table[:batch_size].repeat_interleave(
            next_n, dim=0
        ).contiguous()
    elif seq_lens.dim() != 1:
        raise RuntimeError(
            "Native paged KPool decode requires rank-1 or rank-2 seq_lens, "
            f"got shape={tuple(seq_lens.shape)}."
        )

    score_rows = min(int(num_decode_tokens), int(seq_lens.shape[0]))
    return seq_lens[:score_rows], block_table[:score_rows], score_rows


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


def _kpool_dcp_work_topk(select_k: int, dcp_world_size: int) -> int:
    return max(select_k, _KPOOL_DCP_MERGE_TOPK) if dcp_world_size > 1 else select_k


def _gather_selected_scores(
    logits: torch.Tensor,
    indices: torch.Tensor,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    safe_indices = indices.clamp_min(0).to(torch.int64)
    if row_starts is not None:
        safe_indices = safe_indices + row_starts[:, None].to(torch.int64)
    safe_indices.clamp_max_(max(0, logits.shape[1] - 1))
    scores = torch.gather(logits, 1, safe_indices)
    return scores.masked_fill(indices < 0, -float("inf")).contiguous()


def _merge_kpool_dcp_topk(
    *,
    logits: torch.Tensor | None,
    pool_topk: torch.Tensor,
    pool_scores: torch.Tensor | None,
    row_starts: torch.Tensor | None,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> None:
    if dcp_world_size <= 1:
        return
    if pool_scores is None:
        if logits is None:
            raise RuntimeError("DCP KPool merge requires logits or selected scores.")
        pool_scores = _gather_selected_scores(logits, pool_topk, row_starts)
    _merge_b12x_dcp_topk(
        topk_indices=pool_topk,
        topk_scores=pool_scores,
        topk_tokens=pool_topk.shape[1],
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    use_b12x_sparse_indexer: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Reserve profiler-visible memory for the worst-case decode logits,
        # whose shape is [B * next_n, max_model_len]. This profiling branch
        # returns before invoking the logits kernel itself.
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_kpool_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            # kpool prefill write: pool kpool consecutive prefill tokens via
            # softmax(gate+ape)-weighted sum -> Hadamard -> fp8 -> pool slots.
            # Decode tokens (the first num_decode_tokens in the batch) cannot be
            # pooled here — their pool's earlier tokens are not in this batch —
            # so they are deferred to the tail-buffer kernel in has_decode.
            # compress_ratio == index_kpool makes slot_mapping pool-granular.
            n_prefill = num_tokens - num_decode_tokens
            if n_prefill > 0:
                # decode tokens are batched first; prefill tokens follow.
                prefill_slice = slice(num_decode_tokens, num_tokens)
                if _kda_input_debug_enabled():
                    _kda_slot_guard(
                        "kpool_prefill_compress_insert",
                        "slot_mapping",
                        slot_mapping[prefill_slice],
                        int(kv_cache.shape[1]) if kv_cache.ndim >= 2 else 1,
                        int(kv_cache.shape[0]),
                        {"num_tokens": int(num_tokens)},
                    )
                _kpool_compress_insert(
                    k[prefill_slice],
                    gate_score[prefill_slice],
                    compress_ape,
                    kv_cache,
                    slot_mapping[prefill_slice],
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
                # Persist each request's incomplete prefill pool so decode can
                # finish it, including after PD transfer. Tail slots use
                # ``pos % kpool`` within the request's tail block. Processing
                # only the batch's trailing tokens would miss all but the last
                # request in a multi-request prefill.
                if (
                    tail_kv_cache is not None
                    and tail_prefix is not None
                    and os.environ.get("VLLM_KPOOL_SKIP_TAIL_CACHE") != "1"
                ):
                    tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                    if tail_meta is not None:
                        assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                        if _kda_input_debug_enabled():
                            _kda_slot_guard(
                                "kpool_prefill_seed_tail_cache",
                                "tail_slot_mapping",
                                tail_meta.slot_mapping[prefill_slice],
                                int(tail_kv_cache.shape[2]),
                                int(tail_kv_cache.shape[0]),
                            )
                        kpool_seed_tail_cache(
                            tail_kv_cache,
                            k[prefill_slice],
                            gate_score[prefill_slice],
                            tail_meta.slot_mapping[prefill_slice],
                            index_kpool,
                            head_dim,
                        )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            ops.indexer_k_quant_and_cache(
                k,
                kv_cache,
                slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool, so skip sparse scoring and fill
        # the top-k buffer with all causal token indices. The index-K cache was
        # already written above.
        n_prefill_sf = num_tokens - num_decode_tokens
        # Host-side short-prefill predicate: max_prefill_seq_len is computed
        # in the metadata builder (exact for prefill rows) and equals
        # positions[prefill_slice].max() + 1, so this replaces a
        # positions.max().item() device sync per layer. -1 (unknown metadata)
        # falls back to the device-side check.
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _arange = torch.arange(
                topk_indices_buffer.shape[1],
                device=topk_indices_buffer.device,
                dtype=torch.int32,
            )
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _buf[:] = _arange[None, :]
            _buf[_arange[None, :] > _pos[:, None]] = -1

        prefill_chunks = prefill_metadata.chunks if not short_prefill else ()
        use_b12x_prefill = bool(prefill_chunks) and all(
            _can_use_b12x_kpool_prefill(
                use_b12x_sparse_indexer=use_b12x_sparse_indexer_fn(
                    use_b12x_sparse_indexer
                ),
                use_fp4_cache=use_fp4_cache,
                cache_page_size=int(kv_cache.shape[1]),
                num_reqs=chunk.num_reqs,
                total_seq_lens=chunk.total_seq_lens,
                local_total_seq_lens=chunk.local_total_seq_lens,
            )
            for chunk in prefill_chunks
        )

        # The B12X route reads the paged pool cache directly. The fallback
        # gathers each rank's pool K rows into contiguous storage for DeepGEMM.
        k_quant_full = k_scale_full = None
        if prefill_chunks and not use_b12x_prefill:
            workspace_manager = current_workspace_manager()
            values_spec, scales_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
                values_spec,
                scales_spec,
            )

        for chunk in prefill_chunks:
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens

            if use_b12x_prefill:
                pool_topk = torch.empty(
                    (int(q_slice.shape[0]), select_k),
                    dtype=torch.int32,
                    device=q_slice.device,
                )
                pool_seq_lens = torch.where(
                    chunk.cu_seqlen_ke <= chunk.cu_seqlen_ks,
                    torch.zeros_like(chunk.cu_seqlen_ks),
                    chunk.cu_seqlen_ke - chunk.cu_seqlen_ks,
                )
                active_page_width = max(1, (int(chunk.total_seq_lens) + 63) // 64)
                active_page_width = min(
                    active_page_width, int(chunk.block_table.shape[1])
                )
                pool_block_table = chunk.block_table[:1, :active_page_width].expand(
                    int(q_slice.shape[0]), active_page_width
                )
                _run_b12x_paged_topk(
                    q_fp8=q_slice.contiguous(),
                    weights=weights[chunk.token_start : chunk.token_end].contiguous(),
                    kv_cache=kv_cache,
                    seq_lens=pool_seq_lens,
                    block_table=pool_block_table,
                    schedule_metadata=None,
                    topk_indices=pool_topk,
                    topk_tokens=select_k,
                    shared_page_table=True,
                )
                if positions is not None:
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expand_pools_and_append_tail(
                        pool_topk.to(torch.int64),
                        q_seq,
                        index_kpool,
                        out=topk_indices_buffer[chunk.token_start : chunk.token_end],
                    )
                else:
                    valid = pool_topk >= 0
                    expanded = expand_pools_to_tokens(
                        pool_topk.to(torch.int64),
                        valid,
                        topk_tokens,
                        index_kpool,
                    )
                    topk_indices_buffer[
                        chunk.token_start : chunk.token_end, : expanded.shape[-1]
                    ] = expanded
                continue

            assert k_quant_full is not None and k_scale_full is not None
            local_total_seq_lens = (
                chunk.local_total_seq_lens
                if dcp_world_size > 1
                else chunk.total_seq_lens
            )
            k_quant = k_quant_full[:local_total_seq_lens]
            k_scale = k_scale_full[:local_total_seq_lens]

            if not chunk.skip_kv_gather:
                gather_seq_lens = (
                    chunk.local_cu_seq_lens if dcp_world_size > 1 else chunk.cu_seq_lens
                )
                assert gather_seq_lens is not None
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    gather_seq_lens,
                )

            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            logits = fp8_fp4_mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            work_k = _kpool_dcp_work_topk(select_k, dcp_world_size)
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, work_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    work_k,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    work_k,
                )

            if index_kpool > 1:
                _merge_kpool_dcp_topk(
                    logits=logits,
                    pool_topk=pool_topk,
                    pool_scores=None,
                    row_starts=chunk.cu_seqlen_ks,
                    dcp_world_size=dcp_world_size,
                    dcp_rank=dcp_rank,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                )
                pool_ids = pool_topk[:, :select_k].to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = expand_pools_and_append_tail(
                        pool_ids,
                        q_seq,
                        index_kpool,
                        out=topk_indices_buffer[chunk.token_start : chunk.token_end],
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                if positions is None:
                    topk_indices_buffer[
                        chunk.token_start : chunk.token_end, : expanded.shape[-1]
                    ] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes

        # Update the tail before reading logits; completed pools are compressed
        # into the slot supplied by slot_mapping.
        # Spec verification groups tokens by request and preserves position
        # order so each token is stashed before the next completes its pool.
        # Positions must remain token-granular because the kernel derives the
        # pool phase and tail index from ``pos % kpool``.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
            and os.environ.get("VLLM_KPOOL_SKIP_DECODE_WRITE") != "1"
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # Kpool writes must recover the original request grouping after the
            # indexer's flattened decode path. Host metadata avoids a CUDA graph
            # sync when choosing the uniform or padded layout.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                write_num_decode_tokens = (
                    decode_metadata.write_num_decode_tokens or num_decode_tokens
                )
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and write_num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
                write_num_decode_tokens = num_decode_tokens
            if not use_uniform:
                # Non-uniform decode_lens (mixed plain-decode + spec-verify, or
                # a variable MTP-verify batch): scatter actual tokens into a
                # padded [B, lmax] layout. int32 tensors can't go through
                # pack_seq_triton (float/uint8 only). The scatter indices are
                # shared by all five scatters below (and the tail slot one).
                scatter_idx = _build_decode_scatter_indices(
                    group_lens, num_requests, write_num_decode_tokens
                )
                dec_k = _scatter_decode_tokens_by_request(
                    k[:write_num_decode_tokens], 0, num_requests, lmax, scatter_idx
                )
                dec_gate = _scatter_decode_tokens_by_request(
                    gate_score[:write_num_decode_tokens],
                    0,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_slot = _scatter_decode_tokens_by_request(
                    slot_mapping[:write_num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
                dec_pos = _scatter_decode_tokens_by_request(
                    positions[:write_num_decode_tokens].to(torch.int32),
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                next_n = num_decode_tokens // num_requests
                shape2 = (num_requests, next_n)
                dec_k = k[:num_decode_tokens].view(*shape2, head_dim)
                dec_gate = gate_score[:num_decode_tokens].view(*shape2, head_dim)
                dec_slot = slot_mapping[:num_decode_tokens].view(shape2)
                dec_pos = positions[:num_decode_tokens].to(torch.int32).view(shape2)
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            # Paged tail cache replaces the transient _DECODE_TAIL ring. Group
            # the tail group's token-granular slot_mapping per-request, mirroring
            # dec_slot / dec_pos, so the kernel gets each request's current-token
            # tail slot (block * kpool + pos % kpool).
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            if tail_meta is None or tail_kv_cache is None:
                dec_tail_slot = None
            elif not use_uniform:
                dec_tail_slot = _scatter_decode_tokens_by_request(
                    tail_meta.slot_mapping[:write_num_decode_tokens],
                    -1,
                    num_requests,
                    lmax,
                    scatter_idx,
                )
            else:
                dec_tail_slot = tail_meta.slot_mapping[:num_decode_tokens].view(shape2)
            # The compress kernel writes the raw fp8 cache (not the quant view);
            # pass the underlying kv_cache, not kv_cache_quant_view.
            if dec_tail_slot is not None:
                # Single batched launch over [num_requests, next_n] replaces the
                # per-token sequential loop. The kernel iterates each request's
                # tokens in position order internally, preserving the
                # pool-completion read-after-stash dependency that the loop
                # provided. Inputs are already grouped per request (uniform:
                # view; non-uniform: _scatter_decode_tokens_by_request padded to
                # [B, lmax]) — no per-token .contiguous() copies needed.
                if _kda_input_debug_enabled():
                    _kda_slot_guard(
                        "kpool_decode_update_batched",
                        "dec_slot",
                        dec_slot,
                        int(kv_cache_raw.shape[1]) if kv_cache_raw.ndim >= 2 else 1,
                        int(kv_cache_raw.shape[0]),
                    )
                    if dec_tail_slot is not None:
                        _kda_slot_guard(
                            "kpool_decode_update_batched",
                            "dec_tail_slot",
                            dec_tail_slot,
                            int(tail_kv_cache.shape[2]),
                            int(tail_kv_cache.shape[0]),
                        )
                kpool_decode_update_and_maybe_write_cache_batched(
                    kv_cache_raw,
                    tail_kv_cache,
                    dec_tail_slot,
                    dec_k,
                    dec_gate,
                    compress_ape,
                    dec_slot,
                    dec_pos,
                    index_kpool,
                    head_dim,
                    round_scale=(scale_fmt is not None),
                )
        use_b12x_indexer = (
            use_b12x_sparse_indexer_fn(use_b12x_sparse_indexer)
            and int(kv_cache_raw.shape[1]) == 64
        )
        use_kpool32_indexer = (
            current_platform.is_cuda()
            and not use_fp4_cache
            and int(kv_cache_raw.shape[1]) == 32
            and index_kpool > 1
        )
        if use_b12x_indexer and use_fp4_cache:
            raise RuntimeError(
                "B12X KPool sparse scoring requires the FP8 indexer cache; "
                "disable use_fp4_indexer_cache."
            )

        # KPool compresses the index cache to one row per pool.  A 64-row
        # physical page can use B12X's paged scorer directly; a 32-row GLM page
        # uses the dedicated Triton scorer below while B12X continues to own the
        # NoPE NVFP4 MLA attention path.  Both routes return pool-level winners
        # which are expanded to token ids before sparse attention.
        if use_b12x_indexer:
            b12x_seq_lens, b12x_block_table, score_rows = (
                _normalize_native_paged_decode_rows(
                    decode_metadata.seq_lens,
                    decode_metadata.block_table,
                    num_decode_tokens,
                )
            )
            if decode_metadata.requires_padding:
                raise RuntimeError(
                    "B12X KPool decode requires an unpadded rank-1 seq_lens "
                    "contract after native-spec normalization; "
                    f"requires_padding={decode_metadata.requires_padding}, "
                    f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                    f"normalized_seq_lens_shape={tuple(b12x_seq_lens.shape)}."
                )

            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            work_k = _kpool_dcp_work_topk(select_k, dcp_world_size)
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (score_rows, work_k),
                    dtype=torch.int32,
                    device=topk_indices_buffer.device,
                )
                pool_scores = (
                    torch.empty_like(pool_topk, dtype=torch.float32)
                    if dcp_world_size > 1
                    else None
                )
                b12x_topk = pool_topk
            else:
                b12x_topk = topk_indices_buffer[:score_rows, :select_k]
                pool_scores = None
            _run_b12x_paged_topk(
                q_fp8=q_quant[:score_rows].contiguous(),
                weights=weights[:score_rows].contiguous(),
                kv_cache=kv_cache_raw,
                seq_lens=b12x_seq_lens,
                block_table=b12x_block_table,
                schedule_metadata=decode_metadata.schedule_metadata,
                active_width=decode_metadata.active_width,
                topk_indices=b12x_topk,
                topk_tokens=work_k,
                topk_scores=pool_scores,
            )
            if index_kpool > 1:
                _merge_kpool_dcp_topk(
                    logits=None,
                    pool_topk=pool_topk,
                    pool_scores=pool_scores,
                    row_starts=None,
                    dcp_world_size=dcp_world_size,
                    dcp_rank=dcp_rank,
                    cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                )
                if positions is not None:
                    dec_seq = positions[:score_rows].to(torch.int32) + 1
                else:
                    # Metadata lengths are pool-granular for KPool.  Recover a
                    # conservative token length only for the legacy no-position
                    # path; GLM always supplies token positions.
                    dec_seq = b12x_seq_lens.to(torch.int32)
                    dec_seq *= index_kpool
                expand_pools_and_append_tail(
                    pool_topk[:, :select_k].to(torch.int64),
                    dec_seq,
                    index_kpool,
                    out=topk_indices_buffer[:score_rows],
                )
            return topk_indices_buffer

        if use_kpool32_indexer:
            pool_seq_lens, pool_block_table, score_rows = (
                _normalize_native_paged_decode_rows(
                    decode_metadata.seq_lens,
                    decode_metadata.block_table,
                    num_decode_tokens,
                )
            )
            if decode_metadata.requires_padding:
                raise RuntimeError(
                    "Native KPool32 decode requires an unpadded rank-1 "
                    "seq_lens contract after native-spec normalization; "
                    f"requires_padding={decode_metadata.requires_padding}, "
                    f"seq_lens_shape={tuple(decode_metadata.seq_lens.shape)}, "
                    f"normalized_seq_lens_shape={tuple(pool_seq_lens.shape)}."
                )

            select_k = topk_tokens // index_kpool
            work_k = _kpool_dcp_work_topk(select_k, dcp_world_size)
            max_pool_len = -(-max_model_len // index_kpool)
            logits = kpool_paged_mqa_logits_32(
                q_quant[:score_rows].contiguous(),
                kv_cache_raw,
                weights[:score_rows].contiguous(),
                pool_seq_lens,
                pool_block_table,
                max_pool_len,
            )
            pool_topk = torch.empty(
                (score_rows, work_k),
                dtype=torch.int32,
                device=logits.device,
            )
            if work_k in (512, 1024, 2048):
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    pool_seq_lens,
                    pool_topk,
                    topk_workspace,
                    work_k,
                    max_pool_len,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    1,
                    pool_seq_lens,
                    pool_topk,
                    score_rows,
                    logits.stride(0),
                    logits.stride(1),
                    work_k,
                )
            _merge_kpool_dcp_topk(
                logits=logits,
                pool_topk=pool_topk,
                pool_scores=None,
                row_starts=None,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
            if positions is not None:
                dec_seq = positions[:score_rows].to(torch.int32) + 1
            else:
                dec_seq = pool_seq_lens.to(torch.int32)
                dec_seq *= index_kpool
            expand_pools_and_append_tail(
                pool_topk[:, :select_k].to(torch.int64),
                dec_seq,
                index_kpool,
                out=topk_indices_buffer[:score_rows],
            )
            return topk_indices_buffer

        schedule_metadata = decode_metadata.schedule_metadata
        if schedule_metadata is None:
            raise RuntimeError(
                "DeepGEMM KPool decode requires schedule metadata; enable the "
                "B12X sparse indexer or check the metadata builder."
            )

        kv_cache = kv_cache_as_quant_view(kv_cache_raw, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Padding also covers short chunked prefills classified as decode.
            # MXFP4 uses zero-byte padding so padded slots dequantize to zero.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        logits = fp8_fp4_paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        work_k = _kpool_dcp_work_topk(select_k, dcp_world_size)
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, work_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and work_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                work_k,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    work_k,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_dst,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    work_k,
                )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            _merge_kpool_dcp_topk(
                logits=logits,
                pool_topk=pool_topk,
                pool_scores=None,
                row_starts=None,
                dcp_world_size=dcp_world_size,
                dcp_rank=dcp_rank,
                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            )
            pool_ids = pool_topk[:, :select_k].to(torch.int64)
            n = pool_topk.shape[0]
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = expand_pools_and_append_tail(
                pool_ids,
                dec_seq,
                index_kpool,
                out=(
                    None
                    if decode_metadata.requires_padding
                    else topk_indices_buffer[: pool_ids.shape[0]]
                ),
            )
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        if decode_metadata.requires_padding or index_kpool == 1:
            topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


def sparse_attn_indexer_kpool_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    use_b12x_sparse_indexer: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer_kpool",
    op_func=sparse_attn_indexer_kpool,
    # The indexer writes the index-K cache in place (prefill k-cache insert +
    # kpool decode write), so kv_cache must be declared as mutated — otherwise
    # under full-graph compile dynamo assumes it is unchanged across the
    # indexer→MLA boundary and the MLA reads stale/misaligned KV. The paged tail
    # cache is likewise written in place (prefill tail scatter + decode stash).
    mutates_args=["topk_indices_buffer", "kv_cache", "tail_kv_cache"],
    fake_impl=sparse_attn_indexer_kpool_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.use_b12x_sparse_indexer = use_b12x_sparse_indexer_fn()
        vllm_config = get_current_vllm_config_or_none()
        configured_dcp_world_size = (
            vllm_config.parallel_config.decode_context_parallel_size
            if vllm_config is not None
            else 1
        )
        self.dcp_world_size = int(configured_dcp_world_size)
        if self.dcp_world_size > 1:
            from vllm.distributed.parallel_state import get_indexer_dcp_group

            indexer_group = get_indexer_dcp_group(self.dcp_world_size)
            self.dcp_rank = int(indexer_group.rank_in_group)
        else:
            self.dcp_rank = 0
        self.cp_kv_cache_interleave_size = (
            int(vllm_config.parallel_config.cp_kv_cache_interleave_size)
            if vllm_config is not None
            else 1
        )
        if (
            current_platform.is_cuda()
            and not self.use_b12x_sparse_indexer
            and not has_deep_gemm()
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(
                hidden_states,
                q_quant,
                k,
                weights,
                gate_score=gate_score,
                compress_ape=compress_ape,
                index_kpool=index_kpool,
                positions=positions,
            )
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer_kpool(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
            self.use_b12x_sparse_indexer,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
