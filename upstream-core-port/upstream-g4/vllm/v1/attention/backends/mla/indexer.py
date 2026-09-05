# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_dcp_group,
    get_indexer_dcp_group,
    get_pcp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonPointerInputVariant,
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    native_next_n_supported,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_cache_dcp_shard_count,
)

logger = init_logger(__name__)

# [DIFF-DUMP] differential-harness instrumentation for the upstream-core port:
# gated by a marker file on the host-writable /cache volume (the compose env
# cannot carry a new variable), one existence check cached per process.
# Zero-cost when absent; skipped during CUDA-graph capture and warmup batches
# (same guards as VLLM_DEBUG_KDA_INPUTS).
_DIFF_DUMP_ENABLED: bool | None = None


def _diff_dump_enabled() -> bool:
    global _DIFF_DUMP_ENABLED
    if _DIFF_DUMP_ENABLED is None:
        try:
            _DIFF_DUMP_ENABLED = os.path.exists("/cache/.diff-dump")
        except Exception:
            _DIFF_DUMP_ENABLED = False
    if not _DIFF_DUMP_ENABLED:
        return False
    try:
        if torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        pass
    try:
        from vllm.v1.worker.gpu import warmup as _warmup_mod

        if getattr(_warmup_mod, "IN_WARMUP", False):
            return False
    except Exception:
        pass
    return True


# [APC guard] indexer page-validation gate, cached per process. Default OFF
# in production: the validation issues GPU kernels plus a bool(.item())
# DEVICE SYNC on every build() call (every decode step + prefill chunk),
# serializing the CPU/GPU pipeline. Enable with VLLM_APC_INDEXER_GUARD=1 to
# hunt APC lifecycle corruption (stale hash hits after eviction, CoW/evict
# misorders -- the Xid-31 precursor).
_APC_INDEXER_GUARD_ENABLED: bool | None = None


def _apc_indexer_guard_enabled() -> bool:
    global _APC_INDEXER_GUARD_ENABLED
    if _APC_INDEXER_GUARD_ENABLED is None:
        try:
            _APC_INDEXER_GUARD_ENABLED = (
                os.environ.get("VLLM_APC_INDEXER_GUARD", "0") == "1"
            )
        except Exception:
            _APC_INDEXER_GUARD_ENABLED = False
    return _APC_INDEXER_GUARD_ENABLED


def _diff_dump_int(tensor: torch.Tensor, limit: int = 16) -> str:
    """Compact int-tensor digest: count/sum/min/max plus the first values."""
    if tensor is None:
        return "None"
    t = tensor.detach().to(torch.int64).cpu().flatten()
    if t.numel() == 0:
        return "empty"
    valid = t[t >= 0]
    head = "[" + ",".join(str(int(v)) for v in t[:limit].tolist()) + "]"
    if valid.numel() == 0:
        return f"n={t.numel()} nvalid=0 head={head}"
    return (
        f"n={t.numel()} nvalid={int(valid.numel())} "
        f"vsum={int(valid.sum().item())} vmin={int(valid.min().item())} "
        f"vmax={int(valid.max().item())} head={head}"
    )


def _diff_step_counter(bucket: str, limit: int = 24) -> int:
    """Monotonic per-process step counter for dump gating; -1 when over limit."""
    counter = _DIFF_STEP_COUNTS.get(bucket, 0)
    _DIFF_STEP_COUNTS[bucket] = counter + 1
    return counter if counter < limit else -1


_DIFF_STEP_COUNTS: dict[str, int] = {}

# The DSA indexer K cache is always quantized; "auto" means fp8 (V3.2 layout)
# and mxfp4 is the opt-in Blackwell path.
DSA_INDEXER_KV_DTYPES = ("fp8", "mxfp4")


def dsa_indexer_uses_fp4(vllm_config: VllmConfig) -> bool:
    """Whether the DeepSeek sparse indexer should use the MXFP4 K cache."""
    # [FORK-COMPAT] v84 AttentionConfig lacks resolve_indexer_kv_dtype; the
    # stock v84 fork selects the indexer K-cache format via the
    # use_fp4_indexer_cache bool (bf16-by-default indexer_kv_dtype is never
    # consulted on this path), i.e. mxfp4 iff the fp4 indexer cache is on.
    _resolve_kv_dtype = getattr(
        vllm_config.attention_config, "resolve_indexer_kv_dtype", None
    )
    if _resolve_kv_dtype is not None:
        kv_dtype = _resolve_kv_dtype("fp8")
    else:
        kv_dtype = (
            "mxfp4"
            if getattr(
                vllm_config.attention_config, "use_fp4_indexer_cache", False
            )
            else "fp8"
        )
    if kv_dtype not in DSA_INDEXER_KV_DTYPES:
        raise ValueError(
            f"indexer_kv_dtype={kv_dtype!r} is not supported by the DeepSeek "
            f"sparse indexer (expected one of {DSA_INDEXER_KV_DTYPES})."
        )
    use_fp4 = kv_dtype == "mxfp4"
    if use_fp4 and not current_platform.is_device_capability_family(100):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
    return use_fp4


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # Compute number of KVs attended to by this token. Padding requests have
    # seq_len == 0, which would otherwise make the first token of each padded
    # request negative (e.g. next_n=2 gives 0-2+0+1 = -1). Downstream kernels
    # read these as uint32, turning -1 into ~4e9.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = tl.maximum(seq_len - max_decode_len + local_idx + 1, 0)
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class DeepseekV32IndexerBackend(AttentionBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Only the varlen paged MQA logits kernel takes per-request query
        # lengths from device tensors natively. Hopper can instead flatten each
        # query into a single-token row using device-built metadata.
        return _supports_varlen_paged_mqa_logits() or (
            _supports_flattened_device_query_lens()
        )

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, MultipleOf(16)] if current_platform.is_rocm() else [64]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # The sparse indexer packs its small pages beside the MLA latent
        # pages inside each block (same packing as DeepSeek-V4's indexer),
        # so the layer dim must sit inside the block dim.
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # DeepSeek-V4 packs the indexer pages beside the MLA latent pages inside
        # each block, so the layer dim must sit inside the block dim.
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class B12xNonCompressedIndexerBackend(DeepseekV32IndexerBackend):
    """[v84-fork compat] Fork-only backend alias.

    The fork's ``model_executor/models/deepseek_v2.py`` returns this backend
    for the indexer group when the B12X sparse indexer is selected
    (``use_b12x_sparse_indexer()``); upstream removed the class (B12X
    integration lives model-side) but the un-ported fork model files still
    import it, so the alias must survive the core swap. Behaviorally it is
    identical to DeepseekV32IndexerBackend.
    """

    @staticmethod
    def get_name() -> str:
        return "B12X_NON_COMPRESSED_INDEXER"


class KpoolTailBackend(DeepseekV32IndexerBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache.

    The tail cache holds the in-progress pool's raw K + gate score packed into
    one ``[num_blocks, 2, block_size, head_dim]`` bf16 tensor (K at index 0,
    gate score at index 1) and never runs attention. It reuses the indexer
    metadata builder (token-granular when ``compress_ratio == 1``) but exposes a
    4D K||score shape and unrestricted head/block sizes: the indexer backend
    hard-requires ``head_size in {32,64,128}`` and ``kernel_block_size 64``,
    which the tail's ``block_size == kpool`` (e.g. 4) violates. The 4D
    ``[2, block_size, head_dim]`` layout keeps K and score as separate block
    halves so the connectors' non-MLA K/V half-split transfers them correctly.
    """

    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []  # supports any (K+score packed into head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # block_size == kpool; no sub-block splitting (kernel_block_size ==
        # block_size), so the tensor stays [num_blocks, 2, kpool, head_dim].
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # head_size carries K+score packed (== 2 * head_dim); split into
        # [2, block_size, head_dim] so K (idx 0) and score (idx 1) are separate
        # block halves.
        assert num_kv_heads == 1
        assert head_size % 2 == 0, "KpoolTailSpec head_size must be 2*head_dim"
        return (num_blocks, 2, block_size, head_size // 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        return (0, 1, 2, 3)

    @staticmethod
    def get_builder_cls() -> type["KpoolTailMetadataBuilder"]:  # type: ignore[override]
        return KpoolTailMetadataBuilder


@dataclass(kw_only=True)
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    # Under DCP (dcp_world_size > 1) these hold this rank's local row bounds;
    # otherwise they hold the global bounds.
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False
    local_cu_seq_lens: torch.Tensor | None = None
    local_total_seq_lens: int = 0
    max_local_total_seq_lens: int = 0


_BUILD_PREFILL_CHUNK_METADATA_INPUT_VARIANTS = (
    TritonPointerInputVariant.from_alignment(uncompressed_seq_lens=True),
    TritonPointerInputVariant.from_alignment(uncompressed_seq_lens=False),
)


class BuildPrefillChunkMetadataKernel(
    VllmJitKernel["BuildPrefillChunkMetadataKernel.CompileKey"]
):
    BLOCK_SIZE = 1024

    @dataclass(frozen=True)
    class CompileKey:
        query_slice_start: int
        query_slice_stop: int
        DCP_RANK: int
        DCP_WORLD: int
        DCP_INTERLEAVE: int
        BLOCK_SIZE: int
        COMPRESS_RATIO: int
        input_variant: TritonPointerInputVariant

    @staticmethod
    @triton.jit
    def kernel(
        # Inputs
        query_start_loc_ptr,
        uncompressed_seq_lens_ptr,
        cu_compressed_seq_lens_ptr,
        # Row-start base for cu_seq_len_ks/ke: local cumulative lens under DCP,
        # aliases cu_compressed_seq_lens_ptr otherwise.
        row_start_cu_compressed_seq_lens_ptr,
        # Outputs
        token_to_seq_ptr,
        cu_compressed_seq_len_ks_ptr,
        cu_compressed_seq_len_ke_ptr,
        query_slice_start,
        query_slice_stop,
        DCP_RANK,
        DCP_WORLD,
        DCP_INTERLEAVE,
        BLOCK_SIZE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        query_len = query_end - query_start

        seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
        seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
        compressed_seq_len = seq_end - seq_start

        # Row start for the (possibly localized) cu_seq_len_ks/ke. Equals seq_start
        # when DCP is disabled (the pointer aliases cu_compressed_seq_lens_ptr).
        row_start = tl.load(row_start_cu_compressed_seq_lens_ptr + batch_idx)

        uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
        start_pos = uncompressed_seq_len - query_len

        for i in range(0, query_len, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            abs_pos = query_start + offset
            mask = (
                (offset < query_len)
                & (abs_pos >= query_slice_start)
                & (abs_pos < query_slice_stop)
            )
            out_pos = abs_pos - query_slice_start

            # cu_seq_len_ks: row start in the gathered K buffer.
            tl.store(cu_compressed_seq_len_ks_ptr + out_pos, row_start, mask=mask)

            # cu_seq_len_ke: row start + per-token context length. Under DCP the
            # global per-token length is sharded across ranks.
            global_ctx = start_pos + 1 + offset
            len_per_token = global_ctx // COMPRESS_RATIO
            if DCP_WORLD > 1:
                # Per-rank local context length under interleave-aware DCP, matching
                # get_dcp_local_seq_lens. K == 1 reduces to (len + world-1-rank)//world.
                base = (len_per_token // DCP_INTERLEAVE // DCP_WORLD) * DCP_INTERLEAVE
                remainder = len_per_token - base * DCP_WORLD
                remainder = tl.minimum(
                    tl.maximum(remainder - DCP_RANK * DCP_INTERLEAVE, 0), DCP_INTERLEAVE
                )
                len_per_token = base + remainder
            tl.store(
                cu_compressed_seq_len_ke_ptr + out_pos,
                row_start + len_per_token,
                mask=mask,
            )

        # Compute token_to_seq
        for i in range(0, compressed_seq_len, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < compressed_seq_len
            tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        query_slice_start: int,
        query_slice_stop: int,
        DCP_RANK: int,
        DCP_WORLD: int,
        DCP_INTERLEAVE: int,
        BLOCK_SIZE: int,
        COMPRESS_RATIO: int,
        input_variant: TritonPointerInputVariant,
    ) -> CompileKey:
        return self.CompileKey(
            query_slice_start=query_slice_start,
            query_slice_stop=query_slice_stop,
            DCP_RANK=DCP_RANK,
            DCP_WORLD=DCP_WORLD,
            DCP_INTERLEAVE=DCP_INTERLEAVE,
            BLOCK_SIZE=BLOCK_SIZE,
            COMPRESS_RATIO=COMPRESS_RATIO,
            input_variant=input_variant,
        )

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        max_tokens = max(1, min(vllm_config.scheduler_config.max_num_batched_tokens, 8))
        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        dcp_world = parallel_config.decode_context_parallel_size
        dcp_interleave = parallel_config.cp_kv_cache_interleave_size
        dcp_rank = get_dcp_group().rank_in_group if dcp_world > 1 else 0
        compress_ratios = tuple(
            max(1, int(ratio))
            for ratio in (getattr(hf_config, "compress_ratios", None) or (1,))
        )
        # The GLM-5.3-Flash kpool indexer sets COMPRESS_RATIO = index_kpool on
        # the indexer spec, which is not in compress_ratios. Include it so
        # warmup covers the runtime key instead of JIT-ing on first prefill.
        index_kpool = getattr(hf_config, "index_kpool", None)
        if index_kpool and index_kpool > 1 and index_kpool not in compress_ratios:
            compress_ratios = compress_ratios + (index_kpool,)
        return self._trace_dispatch(self.dispatch)(
            query_slice_start=WarmupIntRange(0, 2),
            query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),
            DCP_RANK=dcp_rank,
            DCP_WORLD=dcp_world,
            DCP_INTERLEAVE=dcp_interleave,
            BLOCK_SIZE=self.BLOCK_SIZE,
            COMPRESS_RATIO=list(compress_ratios),
            input_variant=_BUILD_PREFILL_CHUNK_METADATA_INPUT_VARIANTS,
        )

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            int32_ptr,
            compile_key.input_variant.pointer("uncompressed_seq_lens", torch.int32),
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            compile_key.query_slice_start,
            compile_key.query_slice_stop,
            compile_key.DCP_RANK,
            compile_key.DCP_WORLD,
            compile_key.DCP_INTERLEAVE,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            COMPRESS_RATIO=compile_key.COMPRESS_RATIO,
            grid=(1,),
        )

    def __call__(
        self,
        query_start_loc: torch.Tensor,
        uncompressed_seq_lens: torch.Tensor,
        cu_compressed_seq_lens: torch.Tensor,
        row_start_cu_compressed_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        cu_compressed_seq_len_ks: torch.Tensor,
        cu_compressed_seq_len_ke: torch.Tensor,
        query_slice_start: int,
        query_slice_stop: int,
        DCP_RANK: int,
        DCP_WORLD: int,
        DCP_INTERLEAVE: int,
        *,
        num_reqs: int,
        COMPRESS_RATIO: int,
    ) -> None:
        self.kernel[(num_reqs,)](
            query_start_loc,
            uncompressed_seq_lens,
            cu_compressed_seq_lens,
            row_start_cu_compressed_seq_lens,
            token_to_seq,
            cu_compressed_seq_len_ks,
            cu_compressed_seq_len_ke,
            query_slice_start,
            query_slice_stop,
            DCP_RANK,
            DCP_WORLD,
            DCP_INTERLEAVE,
            BLOCK_SIZE=self.BLOCK_SIZE,
            COMPRESS_RATIO=COMPRESS_RATIO,
        )


_BUILD_PREFILL_CHUNK_METADATA_KERNEL = BuildPrefillChunkMetadataKernel()


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]
    # Host-side max prefill seq len (== max token position + 1 across the
    # step's prefill tokens; exact for prefill rows per the assert in build()).
    # Lets the indexer layer's short-sequence check branch without the
    # device sync a positions.max().item() would cost on every layer.
    # -1 = unknown (other construction sites): the layer falls back to the
    # device-side check.
    max_prefill_seq_len: int = -1


_KPOOL_DENSE_MAX_LOGITS_MB = 384


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    global_seq_lens: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    # [GLM-5.3 kpool contract] The v84-fork's B12X decode scorer windowing
    # buffer; the upstream builder never sets it (no B12X path in this core --
    # B12X integration lives model-side). Kept as a None-default field so the
    # kpool indexer layer's ``decode_metadata.active_width`` access (passed to
    # _run_b12x_paged_topk, where None disables windowing) keeps resolving.
    active_width: torch.Tensor | None = None
    # Original per-request decode_lens (length num_decodes) captured before
    # _prepare_decode_tensors rewrites decode_lens. The kpool decode-write path
    # uses these to group tokens by request on a variable MTP-verify batch,
    # which the flatten path represents as all-1s with requires_padding=False.
    per_req_decode_lens: torch.Tensor | None = None
    # Host-side (build-time) values so the kpool decode-write path can branch
    # without a runtime .item() (which would break cudagraph capture).
    decode_is_uniform: bool = True
    write_max_decode_len: int = 0
    # Exact live token count for KPool state writes. num_decode_tokens may be
    # FULL-cudagraph padded and must remain padded for the scoring path.
    write_num_decode_tokens: int = 0


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    # [GLM-5.3 memory fix 2026-09-01] The 40x factor is flashmla-legacy sizing
    # (40 concurrent full-context gathers). This config runs max_num_seqs=4
    # with B12X, whose steady-state gather is at most max_num_seqs *
    # max_model_len/kpool rows (~262k rows = ~34MB); 8x (=2.1M rows) keeps
    # >8x headroom over the real bound while shrinking the per-lane profile
    # slot from 1321MB to ~264MB. Chunk splitting is unaffected (the splitter
    # only fires above the bound, which full 4-request batches never reach).
    return max_model_len * 4


def _supports_varlen_paged_mqa_logits() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_deep_gemm()
    )


def _supports_flattened_device_query_lens() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and has_deep_gemm()
    )


def _supports_native_decode(next_n: int) -> bool:
    """Whether decode can pass `next_n` Q rows per request to the kernel
    instead of flattening to one single-token row per query, which re-reads
    the KV tile once per row.
    """
    if not (current_platform.is_cuda() and has_deep_gemm()):
        return next_n in (1, 2)
    if current_platform.is_device_capability_family(100):
        return True
    if current_platform.is_device_capability_family(90):
        return native_next_n_supported(next_n)
    return next_n in (1, 2)


def _use_flattening(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    next_n = 1 + vllm_config.num_speculative_tokens
    return not _supports_native_decode(next_n) or (
        speculative_config is not None
        # [FORK-COMPAT] v84 SpeculativeConfig lacks
        # enable_adaptive_verification (same guard as rejection_sampler.py).
        and getattr(
            speculative_config, "enable_adaptive_verification", False
        )
        and _supports_flattened_device_query_lens()
    )


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    kpool: int,
) -> torch.Tensor:
    """Circular tail slots: every token of request r lands in r's own block.

    The generic per-group slot kernel maps ``pos -> bt[req][pos // bs] * bs +
    pos % bs`` (``_compute_slot_mappings_kernel``). The tail group's block
    table only ever has its FIRST column written -- ``KpoolTailManager``
    allocates exactly one block per request and never grows -- so for any
    token at ``pos >= kpool`` that kernel reads a zero column and the slot
    collapses onto physical tail block 0. All concurrently running requests
    then share one ``kpool``-slot ring and corrupt each other's pool
    compression (seed / stash / completion all target block 0). This maps
    ``slot = own_block * kpool + pos % kpool`` instead, which is the layout
    the tail kernels and ``KpoolTailManager`` are designed around.

    Pure torch (no Triton, no device sync): the indexer op consumes the tail
    slot mapping on its eager break, so the returned tensor need not be the
    persistent ``BlockTables`` buffer.
    """
    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
    device = slot_mapping.device
    tokens = torch.arange(num_actual_tokens, device=device)
    # searchsorted(right=True): token i in [qsl[r], qsl[r+1]) -> request r.
    req = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    req = req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)
    return out


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Lean metadata builder for the kpool tail cache.

    The tail is storage-only (no attention / MQA-logits), so it skips the
    DeepseekV32 indexer builder's DeepGEMM paged-MQA path -- which requires
    ``block_kv in {32,64}`` and asserts on the tail's ``block_size == kpool``.
    It exports only the group's token-granular ``slot_mapping`` plus the
    prefill/decode counts; the indexer op reads
    ``attn_metadata[tail_prefix].slot_mapping``. The slot mapping is
    recomputed with the circular per-request layout (see
    :func:`compute_kpool_tail_slot_mapping`) instead of reusing the generic
    per-group kernel output, which cannot express a 1-block-per-request
    circular buffer.
    """

    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # No indexer-builder buffers (expanded_block_table / scheduler_metadata /
        # compressed_slot_mapping) -- the tail is storage-only and exports only
        # slot_mapping, which is rebuilt per step from the group's block table.
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata)
        )
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=None,
            decode=None,
        )


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    # The indexer opts out of the shared reorder-threshold vote (see __init__),
    # so this is None; its own split uses self.decode_threshold.
    reorder_batch_threshold: int | None = None
    requires_block_table_width = True

    def _supports_native_decode(self, next_n: int) -> bool:
        return _supports_native_decode(next_n)

    def _split_prefill_chunks(
        self,
        compressed_seq_lens_cpu: torch.Tensor,
        prefill_query_lens_cpu: torch.Tensor,
        num_decodes: int,
        max_logits_bytes: int,
    ) -> list[tuple[slice, slice]]:
        return split_indexer_prefill_chunks(
            compressed_seq_lens_cpu[num_decodes:],
            prefill_query_lens_cpu,
            self.max_prefill_buffer_size,
            max_logits_bytes,
            request_offset=num_decodes,
        )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        if _supports_varlen_paged_mqa_logits() or _use_flattening(vllm_config):
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        parallel_config = self.vllm_config.parallel_config
        # [v84-fork port] The indexer's DCP layout is derived from the spec's
        # actual KV shard count (replicated / partially-sharded caches track
        # only their unique shards), and the rank comes from the indexer DCP
        # group -- not the raw configured DCP world. Ported verbatim from the
        # pristine v84 baseline; upstream had collapsed this to the configured
        # decode-context-parallel size and the general DCP group rank.
        self.dcp_replicated = bool(
            getattr(self.kv_cache_spec, "dcp_replicated", False)
        )
        configured_dcp_world_size = parallel_config.decode_context_parallel_size
        # PCP slot mappings are gathered independently below. This group tracks
        # only the unique sparse-indexer shards within DCP.
        self.dcp_world_size = get_kv_cache_dcp_shard_count(
            self.kv_cache_spec, configured_dcp_world_size
        )
        if self.dcp_world_size > 1:
            indexer_group = get_indexer_dcp_group(self.dcp_world_size)
            if int(indexer_group.world_size) != self.dcp_world_size:
                raise RuntimeError(
                    "Indexer metadata DCP group does not match its KV shard "
                    f"count: group={indexer_group.world_size}, "
                    f"shards={self.dcp_world_size}"
                )
            self.dcp_rank = int(indexer_group.rank_in_group)
        else:
            self.dcp_rank = 0
        self.pcp_world_size = parallel_config.prefill_context_parallel_size
        self.use_pcp = self.pcp_world_size > 1
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        # The DCP sparse-indexer code is parameterized by interleave size, but
        # interleave > 1 is not yet validated end-to-end (gsm8k parity fails),
        # so fail closed here rather than silently produce wrong output.
        if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size > 1:
            raise NotImplementedError(
                "DCP sparse indexer currently supports only "
                f"cp_kv_cache_interleave_size=1 (got "
                f"{self.cp_kv_cache_interleave_size})."
            )
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = dsa_indexer_uses_fp4(self.vllm_config)
        self.index_kpool = int(
            getattr(self.vllm_config.model_config.hf_text_config, "index_kpool", 1) or 1
        )

        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
        self.use_flattening = _use_flattening(self.vllm_config)
        self.supports_varlen = _supports_varlen_paged_mqa_logits()
        logger.info_once(
            "DSA indexer decode path: use_flattening=%s supports_varlen=%s "
            "(next_n=%d, use_fp4_cache=%s)",
            self.use_flattening,
            self.supports_varlen,
            next_n,
            self.use_fp4_indexer_cache,
        )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.global_decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_indices_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.expanded_block_table_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens, block_table_width),
            dtype=torch.int32,
            device=self.device,
        )
        # Persistent mirror of the original per-request decode_lens (length
        # num_decodes), captured in build() before _prepare_decode_tensors
        # flattens decode_lens_buffer to all-1s. Persistent so FULL-cudagraph
        # replay reads a stable pointer (a per-step clone would move address).
        self.per_req_decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # FULL CUDA graphs require the translated decode table at a stable
        # address; allocate it lazily at the fixed compressed width.
        self.indexer_decode_block_table_buffer: torch.Tensor | None = None
        self._max_num_batched_tokens = scheduler_config.max_num_batched_tokens

        # See: DeepGMM/csrc/apis/attention.hpp. Sized for one slot per SM;
        # build() narrows it to whatever the kernel actually schedules.
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # KV compression. Default to 1 for no compression.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            # MLA compression is a whole number of tokens per state (fractions
            # are whisper block pooling and never reach MLA).
            assert isinstance(self.kv_cache_spec.tokens_per_state, int)
            self.compress_ratio = self.kv_cache_spec.tokens_per_state
        # [FORK-COMPAT] upstream restricts DCP to compress_ratio=1; the v84
        # fork's production config runs DCP2 with the kpool-4 compressed
        # indexer (verified in long-running serving). Guard removed.

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Compressed slot mapping output buffer
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )

    def _dcp_localize_decode_seq_lens(
        self,
        seq_lens: torch.Tensor,
        num_decodes: int,
        seq_lens_is_buffer_view: bool,
    ) -> torch.Tensor:
        local_seq_lens = get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        if seq_lens_is_buffer_view:
            seq_lens.copy_(local_seq_lens)
            return seq_lens

        out = self.decode_seq_lens_buffer[:num_decodes]
        out.copy_(local_seq_lens)
        return out

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Prepare native or per-token flattened decode tensors."""
        spec_config = self.vllm_config.speculative_config
        # [FORK-COMPAT] v84 SpeculativeConfig lacks
        # enable_adaptive_verification (same guard as rejection_sampler.py).
        adaptive = bool(
            spec_config
            and getattr(spec_config, "enable_adaptive_verification", False)
        )
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native:
            assert self.decode_seq_lens_buffer.dim() == 1
            if (
                not self.supports_varlen
                and (num_decodes == 1 or not adaptive)
                and min_decode_len == max_decode_len
                and num_decodes * max_decode_len == num_decode_tokens
            ):
                # Uniform decode lengths with no cudagraph token padding.
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                # Clamp at 0: padding requests have seq_len == 0, which would
                # otherwise make token 0 negative (next_n=2 gives 0-2+1+0 = -1).
                # Downstream kernels read these as uint32, turning -1 into ~4e9.
                seq_lens_buffer[:] = (
                    seq_lens.unsqueeze(1)
                    - max_decode_len
                    + 1
                    + self.offsets_buffer[:max_decode_len]
                ).clamp_(min=0)
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def _prepare_global_decode_seq_lens(
        self,
        global_seq_lens: torch.Tensor | None,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decode_tokens: int,
        use_native: bool,
        max_decode_len: int,
    ) -> torch.Tensor | None:
        if global_seq_lens is None:
            return None
        if use_native or max_decode_len <= 1:
            return global_seq_lens

        actual_expanded = int(decode_lens_cpu.sum().item())
        if actual_expanded > 0:
            expanded_offsets = torch.repeat_interleave(
                global_seq_lens - decode_lens - query_start_loc,
                decode_lens,
                output_size=actual_expanded,
            )
            self.global_decode_seq_lens_buffer[:actual_expanded] = (
                expanded_offsets + self.arange_buffer[:actual_expanded] + 1
            )
        self.global_decode_seq_lens_buffer[actual_expanded:num_decode_tokens] = 0
        return self.global_decode_seq_lens_buffer[:num_decode_tokens]

    def _build_varlen_decode_indices(
        self,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        num_decode_tokens: int,
    ) -> torch.Tensor:
        """Build request ids for flattened SM100 varlen rows."""
        indices = self.decode_indices_buffer[:num_decode_tokens]
        actual_expanded = int(decode_lens_cpu.sum().item())
        num_decodes = decode_lens.shape[0]
        indices[:actual_expanded] = torch.repeat_interleave(
            self.arange_buffer[:num_decodes],
            decode_lens,
            output_size=actual_expanded,
        )
        if actual_expanded < num_decode_tokens:
            pad = num_decode_tokens - actual_expanded
            indices[actual_expanded:num_decode_tokens] = (
                num_decodes + self.arange_buffer[:pad]
            )
        return indices

    def _validate_indexer_pages(
        self, pages: torch.Tensor, seq_lens_tokens: torch.Tensor
    ) -> torch.Tensor:
        """[APC guard] Fail fast on invalid derived indexer pages.

        Prefix-cache lifecycle bugs (stale hash hits after eviction, gap-block
        frees, CoW redirects) can leave a NULL_BLOCK_ID (-1) inside a
        request's live page range. Downstream kernels then dereference it as an
        unsigned page id and fault the GPU (cudaErrorIllegalAddress / NVRM
        Xid 31), destroying the only evidence. Validate on the host and raise
        with the offending rows instead.

        Only each request row's live range is checked; unused tail entries are
        legitimately NULL. Row-count mismatch (flattened/expanded decode rows)
        skips the check -- the prefill builder always matches. DISABLED BY
        DEFAULT (VLLM_APC_INDEXER_GUARD=1 to enable): one validation costs
        GPU kernels plus a bool(.item()) device sync, and running it on every
        build() call (every decode step + prefill chunk) serialized the
        CPU/GPU pipeline in production.
        """
        if not _apc_indexer_guard_enabled():
            return pages
        if pages.numel() == 0 or pages.shape[0] != seq_lens_tokens.shape[0]:
            return pages
        # Tokens covered by one indexer page (one column of the translated
        # table). storage_block_size is in index rows; compress_ratio rows
        # share a page only when the spec is a compressed AttentionSpec -- for
        # uncompressed indexer specs both v84- and r7-style specs report
        # storage_block_size == block_size, so page_tokens == block_size.
        storage_block_size = getattr(self.kv_cache_spec, "storage_block_size", None)
        if storage_block_size is None:
            return pages
        page_tokens = storage_block_size * self.compress_ratio
        if page_tokens <= 0:
            return pages
        device = pages.device
        width = pages.shape[1]
        live = ((seq_lens_tokens.to(torch.int64) + page_tokens - 1) // page_tokens).to(
            device
        )
        live = torch.clamp(live, max=width)
        cols = torch.arange(width, device=device)
        within_live = cols[None, :] < live[:, None]
        invalid = (pages < 0) & within_live
        if bool(invalid.any().item()):
            bad_rows = invalid.any(dim=1).nonzero(as_tuple=True)[0]
            detail = []
            for r in bad_rows[:4].tolist():
                end = int(live[r].item())
                detail.append((r, pages[r, :end].tolist()))
            raise RuntimeError(
                "[APC guard] negative/null indexer page id inside a live "
                f"request range (row, live pages): {detail}. This is the "
                "Xid-31 precursor: capture scheduler state (block hashes, "
                "hit lengths, CoW/evict events) for this step."
            )
        return pages

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        # [v84-fork port] Under DCP, the compressed slot mapping must map
        # global compressed positions to this rank's local KV slots (the
        # pristine v84 kernel's DCP_WORLD/DCP_RANK remap), which upstream's
        # port had dropped alongside the write-path table translation below.
        use_dcp_local_kv = self.dcp_world_size > 1 and dcp_local_seq_lens is not None
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                require_uniform=not (self.use_flattening or self.supports_varlen),
                treat_short_extends_as_decodes=not self.use_pcp,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        # [v84-fork port] Default to the raw shared table; translated below
        # when the indexer is co-located with a virtually-split MLA
        # (compress_ratio > 1). This translated table is the ONE block_table
        # the indexer must use for every cache access (write via slot_mapping
        # AND the top-k reads), so the pool keys written at indexer-page
        # granularity are read back from the same physical pages. Upstream's
        # port had kept the translation only for the read paths; the write
        # slot mapping was built from the raw kernel-unit table, scrambling
        # every kpool K write (in-bounds, silently wrong).
        indexer_block_table = block_table
        if self.compress_ratio > 1:
            kbs = getattr(self, "kernel_block_size", None)
            if (
                kbs is not None
                and self.kv_cache_spec.block_size != kbs
                and self.kv_cache_spec.block_size % kbs == 0
            ):
                factor = self.kv_cache_spec.block_size // kbs
                # [PERF 2026-09-04] Persistent buffer instead of a fresh
                # .contiguous() allocation on every build() (every decode
                # step + prefill chunk). The translated table is consumed
                # within the step, so buffer reuse is safe; the decode side
                # already uses the identical pattern
                # (indexer_decode_block_table_buffer, indexer.py:1582).
                translated = block_table[:, ::factor] // factor
                rows, cols = translated.shape
                buf = getattr(self, "indexer_block_table_buffer", None)
                if buf is None or buf.shape[1] < cols:
                    buf = torch.zeros(
                        (self._max_num_batched_tokens, cols),
                        dtype=translated.dtype,
                        device=self.device,
                    )
                    self.indexer_block_table_buffer = buf
                buf[:rows, :cols].copy_(translated)
                indexer_block_table = buf[:rows, :cols]
            # [APC guard] host-side validation of the page table before any
            # kernel consumes it (see _validate_indexer_pages).
            indexer_block_table = self._validate_indexer_pages(
                indexer_block_table, seq_lens
            )
        if self.compress_ratio > 1 or use_dcp_local_kv:
            padded_num_tokens = num_tokens
            if self.pcp_world_size > 1:
                padded_num_tokens = slot_mapping.shape[0] // self.pcp_world_size
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                indexer_block_table,
                self.kv_cache_spec.num_states,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=self.dcp_world_size if use_dcp_local_kv else 1,
                dcp_rank=self.dcp_rank if use_dcp_local_kv else 0,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
            if self.pcp_world_size > 1:
                compressed_slot_mapping = get_pcp_group().all_gather(
                    self.compressed_slot_mapping_buffer[:padded_num_tokens],
                    dim=0,
                )
            compressed_seq_lens = seq_lens // self.compress_ratio
        if _diff_dump_enabled():
            step = _diff_step_counter("indexer-build")
            if step >= 0:
                print(
                    f"[DIFF-DUMP] IDX-BUILD step={step} "
                    f"compress_ratio={self.compress_ratio} "
                    f"spec_block={self.kv_cache_spec.block_size} "
                    f"num_states={self.kv_cache_spec.num_states} "
                    f"kernel_block_size={getattr(self, 'kernel_block_size', None)} "
                    f"dcp_world={self.dcp_world_size} dcp_rank={self.dcp_rank} "
                    f"dcp_replicated={self.dcp_replicated} "
                    f"use_dcp_local_kv={use_dcp_local_kv} "
                    f"num_tokens={num_tokens} num_reqs={num_reqs} "
                    f"num_decodes={num_decodes} num_prefills={num_prefills} "
                    f"interleave={self.cp_kv_cache_interleave_size}",
                    flush=True,
                )
                print(
                    f"[DIFF-DUMP] IDX-SLOTMAP step={step} "
                    f"csm={_diff_dump_int(compressed_slot_mapping[:num_tokens])}",
                    flush=True,
                )
                print(
                    f"[DIFF-DUMP] IDX-BT step={step} "
                    f"raw_bt={_diff_dump_int(block_table[0])} "
                    f"idx_bt={_diff_dump_int(indexer_block_table[0])}",
                    flush=True,
                )

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows.  It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            if self.index_kpool > 1:
                # GLM KPool prefill gathers compressed K rows and materializes
                # dense DeepGEMM logits; the uncompressed-path default admits
                # a max_model_len x topk_logits workspace that the kpool
                # gather (pool-granular rows) never needs.
                max_logits_bytes = min(
                    max_logits_bytes,
                    _KPOOL_DENSE_MAX_LOGITS_MB * 1024 * 1024,
                )
            chunk_specs = self._split_prefill_chunks(
                compressed_seq_lens_cpu,
                prefill_query_lens_cpu,
                num_decodes,
                max_logits_bytes,
            )
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below).
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound

            # [v84-fork port] The indexer block table was already translated
            # and validated at the top of build() (it is the ONE table every
            # indexer cache access uses: the write slot mapping above AND the
            # top-k reads here); the duplicate inline translation that the
            # port had left here is gone.


            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    indexer_block_table,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_rank=self.dcp_rank,
                    dcp_world_size=self.dcp_world_size,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(
                chunks,
                max_prefill_seq_len=(
                    # Uncompressed (token-granular) to match `positions`.
                    # seq_lens_cpu is exact for the prefill rows here, and the
                    # step's prefill tokens always include each request's last
                    # token (position seq_len-1), so this equals
                    # positions[prefill_slice].max() + 1.
                    int(seq_lens_cpu[num_decodes:].max().item())
                    if num_prefills > 0
                    else 0
                ),
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )
            # Stash the original per-request decode_lens before
            # _prepare_decode_tensors rewrites decode_lens_buffer (the flatten
            # path sets every entry to 1 for the logits read). The kpool
            # decode-write path uses these to scatter tokens by request on a
            # variable MTP-verify batch.
            self.per_req_decode_lens_buffer[:num_decodes].copy_(decode_lens)

            # Under DCP the per-token decode bounds must be localized AFTER the
            # per-token expansion below, not before. Expanding from a
            # request-level localized length subtracts decode offsets in local
            # space and yields too-short bounds (e.g. world=2, rank=1, global
            # per-token bounds [8, 9, 10] -> [3, 4, 5] instead of [4, 4, 5]), so
            # the first decode token would run top-k against too short a local KV
            # range and miss valid tokens. Keep the global seq_lens here and
            # localize the expanded bounds further down.
            global_seq_lens_for_decode: torch.Tensor | None = None
            if dcp_local_seq_lens is not None:
                global_seq_lens_for_decode = common_attn_metadata.seq_lens[:num_decodes]
            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            min_decode_len = int(decode_lens_cpu.min().item())
            # Host-side uniformity flag for the kpool decode-write path: the
            # flatten path below rewrites decode_lens to all-1s and reports
            # requires_padding=False even for a variable MTP-verify batch, so
            # the write path can't reuse requires_padding. Compute it here (on
            # the CPU copy, so no runtime .item() that would break cudagraph
            # capture) and hand down max_decode_len as the scatter lmax.
            write_is_uniform = min_decode_len == max_decode_len
            next_n = 1 + self.num_speculative_tokens
            # The kernel sees max_decode_len Q rows, not the configured next_n,
            # so legality is per-step: on SM90 a uniformly 3-deep batch has no
            # native kernel. max_decode_len <= 1 always has one.
            step_next_n_ok = max_decode_len <= 1 or self._supports_native_decode(
                max_decode_len
            )
            use_native = (
                not (self.use_flattening or self.supports_varlen)
                and max_decode_len <= next_n
                and step_next_n_ok
            )

            global_seq_lens_for_decode = self._prepare_global_decode_seq_lens(
                global_seq_lens=global_seq_lens_for_decode,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                max_decode_len=max_decode_len,
            )

            decode_indices = None
            if self.supports_varlen:
                decode_indices = self._build_varlen_decode_indices(
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    num_decode_tokens=num_decode_tokens,
                )

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

            # Translate the decode block_table to indexer-page granularity,
            # matching the prefill read and the write. Done AFTER
            # _prepare_decode_tensors because that copies raw MLA-width rows
            # (its expand kernel reads expanded_bt_stride columns per row).
            if self.compress_ratio > 1:
                _kbs = getattr(self, "kernel_block_size", None)
                if (
                    _kbs is not None
                    and self.kv_cache_spec.block_size != _kbs
                    and self.kv_cache_spec.block_size % _kbs == 0
                ):
                    _factor = self.kv_cache_spec.block_size // _kbs
                    # Copy into the persistent buffer (not a fresh
                    # .contiguous()) so FULL-mode cudagraph replay reads content
                    # overwritten each step by build(); a one-shot allocation
                    # would be frozen at capture-time values.
                    compressed = block_table[:, ::_factor] // _factor
                    rows, cols = compressed.shape
                    if self.indexer_decode_block_table_buffer is None:
                        self.indexer_decode_block_table_buffer = torch.zeros(
                            (self._max_num_batched_tokens, cols),
                            dtype=torch.int32,
                            device=self.device,
                        )
                    self.indexer_decode_block_table_buffer[:rows, :cols].copy_(
                        compressed
                    )
                    block_table = self.indexer_decode_block_table_buffer[:rows, :cols]
                    # [APC guard] same validation as the prefill table. The
                    # native path keeps one row per request; the flatten path
                    # has one row per decoded token (num_decode_tokens), which
                    # mismatches seq_lens[:num_decodes] rows and skips the
                    # check by design (prefill-style validation only).
                    if block_table.shape[0] == num_decodes:
                        block_table = self._validate_indexer_pages(
                            block_table, common_attn_metadata.seq_lens[:num_decodes]
                        )

            seq_lens_is_buffer_view = not use_native or next_n > 1

            # DCP: localize the now-expanded per-token global bounds to this
            # rank's owned KV. Done here (after expansion) so each token's global
            # causal length is localized individually; see the comment above.
            if dcp_local_seq_lens is not None:
                seq_lens = self._dcp_localize_decode_seq_lens(
                    seq_lens, num_decodes, seq_lens_is_buffer_view
                )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            # DeepGEMM is required for the paged MQA logits on CUDA devices
            schedule_metadata = self.scheduler_metadata_buffer
            if current_platform.is_cuda() and has_deep_gemm():
                metadata = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.num_states,
                    self.num_sms,
                    indices=decode_indices,
                )
                schedule_metadata = self.scheduler_metadata_buffer[: metadata.shape[0]]
                schedule_metadata[:] = metadata

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=schedule_metadata,
                indices=decode_indices,
                global_seq_lens=global_seq_lens_for_decode,
                per_req_decode_lens=self.per_req_decode_lens_buffer[:num_decodes],
                decode_is_uniform=write_is_uniform,
                write_max_decode_len=max_decode_len,
                write_num_decode_tokens=int(decode_lens_cpu.sum().item()),
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    local_cu_seq_lens = cu_seq_lens
    local_total_seq_lens = total_seq_lens
    max_local_total_seq_lens = total_seq_lens
    if dcp_world_size > 1:
        # Per-rank local KV length under interleave-aware DCP sharding, shape
        # [num_reqs, dcp_world_size]. Reuse the canonical CP helper so the
        # sharding matches the rest of the DCP pipeline (decode/prefill).
        local_seq_lens = get_dcp_local_seq_lens(
            compressed_seq_lens[start_idx:end_idx],
            dcp_world_size,
            None,
            cp_kv_cache_interleave_size,
        )
        this_rank_counts = local_seq_lens[:, dcp_rank].to(torch.int32)
        local_cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        torch.cumsum(this_rank_counts, dim=0, out=local_cu_seq_lens[1:])
        # [PREFILL-SYNC] Keep the device cumsum exact (it bounds the pool-K
        # gathers), but derive the HOST scalars from the CPU seq-lens copy
        # instead of syncing the device tensor twice per chunk. Each .item()
        # below stalls the CPU until every prior GPU op drains, serializing
        # the next chunk's metadata build against the current chunk's compute
        # (~50 chunks for a 100k prompt). The CPU copy carries the same
        # upper bound `total_seq_lens` already trusts: these scalars only
        # slice the gather workspace (k_quant_full[:local_total_seq_lens],
        # never read past the exact device-side cu bounds) and drive the
        # paged-prefill route test, where both operands come from the same
        # copy so `local == total` keeps its exact meaning. max_local_...
        # has no consumer on this stack (kept for dataclass compatibility).
        local_seq_lens_cpu = get_dcp_local_seq_lens(
            compressed_seq_lens_cpu[start_idx:end_idx],
            dcp_world_size,
            None,
            cp_kv_cache_interleave_size,
        )
        local_total_seq_lens = int(
            local_seq_lens_cpu[:, dcp_rank].to(torch.int32).sum().item()
        )
        max_local_total_seq_lens = int(
            local_seq_lens_cpu.sum(dim=0).max().item()
        )

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    # Under DCP the kernel writes this rank's local row bounds into
    # cu_seq_len_ks/ke; otherwise local_cu_seq_lens aliases cu_seq_lens.
    _BUILD_PREFILL_CHUNK_METADATA_KERNEL(
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        local_cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        dcp_rank,
        dcp_world_size,
        cp_kv_cache_interleave_size,
        num_reqs=num_reqs,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=local_total_seq_lens,
        max_local_total_seq_lens=max_local_total_seq_lens,
    )
