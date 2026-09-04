# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.buffer_utils import (
    FusedStagedWriter,
    StagedWriteTensor,
    UvaBackedTensor,
    _load_ptr,
)


_BLOCK_TABLE_AUDIT_STATE = {"step": 0}

_KDA_INPUTS_DEBUG_ENABLED: bool | None = None


def _log_cross_group_leak(
    group: int,
    req_index: int,
    ids: list,
    width: int,
    path: str,
) -> None:
    """Loud, always-on dump of a cross-group id stamp (never silent)."""
    import logging

    logging.getLogger(__name__).error(
        "[CROSS-GROUP-LEAK] group %d (width %d) row %d received "
        "out-of-pool ids via the %s path; ids (first 16): %s — the write "
        "is SKIPPED for this row; capture this step's per-group block-id "
        "channels (new_block_ids / rewrite_ids) to name the misordering.",
        group,
        width,
        req_index,
        path,
        list(ids[:16]),
    )


def _block_table_audit_due() -> bool:
    """VLLM_DEBUG_KDA_INPUTS gate (every-apply under the debug env);
    capture-skipped. Env read cached per process -- this runs once per
    apply_staged_writes (per step) and used to import os + environ.get
    on every call."""
    global _KDA_INPUTS_DEBUG_ENABLED
    if _KDA_INPUTS_DEBUG_ENABLED is None:
        try:
            import os

            _KDA_INPUTS_DEBUG_ENABLED = bool(
                int(os.environ.get("VLLM_DEBUG_KDA_INPUTS", "0"))
            )
        except Exception:
            _KDA_INPUTS_DEBUG_ENABLED = False
    if not _KDA_INPUTS_DEBUG_ENABLED:
        return False
    try:
        import torch as _torch

        if _torch.cuda.is_current_stream_capturing():
            return False
    except Exception:
        pass
    # [GUARD-WARMUP EXEMPTION] warmup stages synthetic writes; skip the audit.
    try:
        from vllm.v1.worker.gpu import warmup as _warmup_mod

        if getattr(_warmup_mod, "IN_WARMUP", False):
            return False
    except Exception:
        pass
    _BLOCK_TABLE_AUDIT_STATE["step"] += 1
    # [EVERY-STEP MODE] the corruption lands between 16-step audits and is
    # consumed before the next one fires; under the debug env, audit EVERY
    # apply_staged_writes to catch the cross-group scatter within one step.
    return True


class BlockTables:
    # [CROSS-GROUP-LEAK hardening] Per-group pool block counts (set by the
    # runner at initialize_kv_cache). ids >= the DESTINATION group's pool
    # are cross-group stamps (the genuine dump: MLA-pool ids 2072..2075 in
    # the 198-block mamba row) — rejected loudly, not silently staged.
    pool_num_blocks: list[int] | None = None

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
        group_cp_sizes: list[int] | None = None,
    ):
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device

        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave
        if group_cp_sizes is None:
            group_cp_sizes = [cp_size] * len(block_sizes)
        assert len(group_cp_sizes) == len(block_sizes)
        assert all(group_cp_size in (1, cp_size) for group_cp_size in group_cp_sizes)
        self.group_cp_sizes_list = list(group_cp_sizes)

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups

        self.blocks_per_kv_block = [
            bs // kbs for bs, kbs in zip(block_sizes, kernel_block_sizes)
        ]

        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.block_tables: list[StagedWriteTensor] = []
        for i in range(self.num_kv_cache_groups):
            max_num_blocks = max_num_blocks_per_group[i] * self.blocks_per_kv_block[i]
            block_table = StagedWriteTensor(
                (self.max_num_reqs, max_num_blocks), dtype=torch.int32, device=device
            )
            self.block_tables.append(block_table)

        self.num_blocks = UvaBackedTensor(
            (self.num_kv_cache_groups, self.max_num_reqs),
            dtype=torch.int32,
        )
        self.fused_writer: FusedStagedWriter | None = None
        if self.num_kv_cache_groups > 1:
            # Only the multi-group path uses the fused writer.
            self.fused_writer = FusedStagedWriter(
                self.device, self.num_kv_cache_groups * self.max_num_reqs
            )

        # Block tables used for model's forward pass.
        # num_kv_cache_groups x [max_num_reqs, max_num_blocks]
        self.input_block_tables: list[torch.Tensor] = [
            torch.zeros_like(b.gpu) for b in self.block_tables
        ]

        self.slot_mappings = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int64,
            device=self.device,
        )

        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        # NOTE(woosuk): Use uint64 instead of int64 to cover all possible addresses.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        # Called at init and after a CuMem kv_cache wake-up. The ptr tensors
        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; the size tensors need re-populating because
        # their storage lives under the kv_cache pool tag and comes back with
        # undefined contents.
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64,
            device=self.device,
        )
        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.kernel_block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.group_cp_sizes = torch.tensor(
            self.group_cp_sizes_list, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)

    def get_group_cp_parameters(self, group_id: int) -> tuple[int, int, int]:
        group_cp_size = self.group_cp_sizes_list[group_id]
        if group_cp_size == 1:
            return 0, 1, 1
        return self.cp_rank, group_cp_size, self.cp_interleave

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
        rewrite_ids: tuple[list[int] | None, ...] | None = None,
    ) -> None:
        for i in range(self.num_kv_cache_groups):
            bpk = self.blocks_per_kv_block[i]
            # [ALIGN-RESYNC] A group with a rewrite entry starts from 0 and
            # writes the scheduler's full manager truth (the manager
            # nulls/reorders align-state blocks mid-request, so the
            # append-only delta log drifts from manager truth). The stale
            # tail beyond the rewritten row is zeroed so the align state
            # gather (clamped to the row END for long sequences) reads null
            # (0) ids instead of freed/reused blocks.
            if rewrite_ids is not None and rewrite_ids[i] is not None:
                block_ids = rewrite_ids[i]
                real_ids = block_ids
                if bpk > 1:
                    real_ids = [
                        b * bpk + k for b in block_ids for k in range(bpk)
                    ]
                width = self.block_tables[i].gpu.shape[-1]
                # [CROSS-GROUP-LEAK check] Ids past the destination group's
                # own pool are cross-group stamps; SKIP the write entirely
                # (stamping wrong-group ids corrupts the state scatter).
                if self.pool_num_blocks is not None and any(
                    _id >= self.pool_num_blocks[i] or _id < 0
                    for _id in real_ids
                ):
                    _log_cross_group_leak(i, req_index, real_ids, width, "rewrite")
                    return
                # [WRITE-SIDE Xid-31/43 fix + COUNT-PARITY] The staged list
                # and its count must respect the row's OWN allocated width:
                # the genuine audit catch (align group, width 20, manager
                # shipped 23 = 3 = num_spec past) showed the full-truth
                # channel can over-ship the worker row. The kernel's
                # num_rows/num_cols mask already bounds the stores; this
                # clamp additionally keeps the COUNT inside the row so no
                # consumer's count-out-of-range path or audit fires, and
                # under VLLM_DEBUG_KDA_INPUTS the dropped ids are dumped so
                # the over-shipping producer is identifiable.
                if len(real_ids) > width:
                    try:
                        import os

                        if int(os.environ.get("VLLM_DEBUG_KDA_INPUTS", "0")):
                            import logging

                            logging.getLogger(__name__).warning(
                                "[BLOCK-TABLE-COUNT-PARITY] group %d row %d: "
                                "full-truth list of %d ids exceeds the row "
                                "width %d; clamping to the width and dropping "
                                "the last %d ids: %s",
                                i,
                                req_index,
                                len(real_ids),
                                width,
                                len(real_ids) - width,
                                real_ids[width:width + 16],
                            )
                    except Exception:
                        pass
                    real_ids = list(real_ids[:width])
                staged_ids = real_ids
                if len(staged_ids) < width:
                    staged_ids = list(staged_ids) + [0] * (width - len(staged_ids))
                self.block_tables[i].stage_write(req_index, 0, staged_ids)
                self.num_blocks.np[i, req_index] = len(real_ids)
            else:
                start = self.num_blocks.np[i, req_index] if not overwrite else 0
                block_ids = new_block_ids[i] if new_block_ids else []
                # [CROSS-GROUP-LEAK check] Same protection for the append
                # (delta) path — the path the MLA delta list took into the
                # mamba row without tripping any count-parity warning.
                if (
                    self.pool_num_blocks is not None
                    and block_ids
                    and any(
                        _id >= self.pool_num_blocks[i] or _id < 0
                        for _id in block_ids
                    )
                ):
                    _log_cross_group_leak(
                        i,
                        req_index,
                        block_ids,
                        self.block_tables[i].gpu.shape[-1],
                        "append",
                    )
                    block_ids = []
                    new_block_ids = None
                if bpk > 1:
                    block_ids = [
                        b * bpk + k for b in block_ids for k in range(bpk)
                    ]
                end = start + len(block_ids)
                row_capacity = self.block_tables[i].gpu.shape[1]
                if end > row_capacity:
                    raise RuntimeError(
                        f"Block table write for request {req_index}, group {i} "
                        f"exceeds row capacity ({end} > {row_capacity})"
                    )
                self.block_tables[i].stage_write(req_index, start, block_ids)
                self.num_blocks.np[i, req_index] = end

    def apply_staged_writes(self) -> None:
        if self.num_kv_cache_groups == 0:
            return
        if self.num_kv_cache_groups == 1:
            # Single group: write directly, skipping the per-write group lookup.
            self.block_tables[0].apply_write()
        elif self.num_kv_cache_groups > 1:
            # Multiple groups: apply all block tables with one fused kernel.
            assert self.fused_writer is not None
            self.fused_writer.apply(
                self.block_tables, self.block_table_ptrs, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()
        # [POST-WRITE AUDIT] Under VLLM_DEBUG_KDA_INPUTS (capture-skipped),
        # spot-validate every group's table right after the scatter: the
        # region beyond each row's num_blocks must be zero (the full-width
        # zeroing invariant; a stamp there is an OOB write from THIS step),
        # and each live id must look like a pool id (< 2**30 — block-id-scale,
        # not token-offset-scale). A violation names the staged-write as the
        # producer, one step before any consumer reads it.
        try:
            if _block_table_audit_due():
                self._debug_audit_tables()
                self._debug_audit_pool_buffers()
        except RuntimeError:
            raise
        except Exception:
            # The audit is diagnostics; never break serving on its own bugs.
            pass

    def _debug_audit_pool_buffers(self) -> None:
        """[POOL-METADATA AUDIT] Validate every small GPU bookkeeping buffer.

        The stale-chunk dump caught pointer-scale values (~1.6e13) and
        block-id runs stamped into the KDA metadata tensors — the
        fingerprint of the block pool's own GPU-resident metadata (free-queue
        ids + addresses) overflowing ITS arrays into the heap neighborhood.
        This audit validates the buffers the runner owns right here:
        num_blocks.gpu (per-group per-request counts) and the slot_mappings
        staging buffer. A value >= 1e9 in any of them is a heap stamp caught
        one step before any consumer reads it; the offending buffer names
        the overflowing kernel's input surface.
        """
        candidates: list[tuple[str, torch.Tensor]] = []
        try:
            candidates.append(("num_blocks.gpu", self.num_blocks.gpu))
        except Exception:
            pass
        try:
            candidates.append(("slot_mappings", self.slot_mappings))
        except Exception:
            pass
        for name, buf in candidates:
            if not isinstance(buf, torch.Tensor) or buf.numel() == 0:
                continue
            if buf.dtype not in (torch.int32, torch.int64):
                continue
            try:
                mx = int(buf.abs().max().item())
            except Exception:
                continue
            # num_blocks.gpu is counts (< row width ~2074); slot mappings are
            # pool-slot indices (< rows*page ~780k) or PAD (-1). Anything
            # >= 1e9 is a pointer/heap-scale stamp.
            if mx >= 10**9:
                flat = buf.abs().flatten()
                bad_idx = int((flat >= 10**9).nonzero()[0].item())
                raise RuntimeError(
                    f"[POOL-METADATA-AUDIT] {name} carries a heap-scale stamp "
                    f"(max={mx} at flat offset {bad_idx}) — the block pool's "
                    "GPU metadata overflowed into this buffer; capture this "
                    "step's pool bookkeeping writes."
                )

    def _debug_audit_tables(self) -> None:
        """[POST-WRITE AUDIT v2] Validate the LIVE PREFIX of every row.

        The tail beyond num_blocks is DON'T-CARE storage on append steps (a
        row that grew keeps dead ids from a previous step) — only a REWRITE
        (resync) zeroes it, so "tail must be zero" is not a steady-state
        invariant (audit v1 false-positived on legal appended rows). What
        every consumer actually reads is the live prefix rows[0:num_blocks]:
        the mamba select gathers near the live count, the compressor derives
        slots from the live range. The real producer detector is therefore:
        (a) every LIVE id is non-negative and block-id-scale (a token-scale
        or negative stamp in the live prefix is a corrupted write from THIS
        step); (b) counts are within the allocated width (a count past the
        width means the staging ran past the row).
        """
        for i in range(self.num_kv_cache_groups):
            table = self.block_tables[i].gpu
            counts = self.num_blocks.np[i]
            if table.ndim != 2 or counts.size != table.shape[0]:
                continue
            rows = int(table.shape[0])
            width = int(table.shape[1])
            for r in range(rows):
                c = int(counts[r])
                if c < 0 or c > width:
                    raise RuntimeError(
                        "[BLOCK-TABLE-AUDIT] num_blocks count out of range "
                        f"in group {i} row {r} (count={c}, width={width}) — "
                        "the staging ran past the row; capture the staged "
                        "write list for this step."
                    )
            # Vectorized live-prefix checks (one bounded host read).
            counts_t = torch.as_tensor(counts, device=table.device)
            live_mask = torch.arange(width, device=table.device)[None, :] < (
                counts_t.clamp(max=width)[:, None]
            )
            live = table[live_mask]
            if live.numel() == 0:
                continue
            live_min = int(live.min().item())
            live_max = int(live.max().item())
            pool = None
            try:
                pool = self.pool_num_blocks[i] if self.pool_num_blocks else None
            except Exception:
                pool = None
            if pool is not None and live_max >= int(pool):
                raise RuntimeError(
                    "[BLOCK-TABLE-AUDIT] cross-group stamp in the live prefix "
                    f"of group {i} (max={live_max} >= pool={pool}) — the fused "
                    "writer scattered another group's list into this table; "
                    "capture this step's staged lists and the fused writer's "
                    "pointer table."
                )
            if live_min < 0:
                # Locate a live negative stamp for the dump.
                first_bad = None
                for r in range(rows):
                    c = int(counts[r])
                    row_live = table[r, :c]
                    if row_live.numel() and int(row_live.min().item()) < 0:
                        first_bad = r
                        break
                raise RuntimeError(
                    "[BLOCK-TABLE-AUDIT] negative id in the live prefix of "
                    f"group {i} (min={live_min}, first bad row={first_bad}, "
                    f"rows[{max(0, (first_bad or 0) - 1)}:"
                    f"{min(rows, (first_bad or 0) + 2)}]=\n"
                    f"{table[max(0, (first_bad or 0) - 1):min(rows, (first_bad or 0) + 2)].tolist()[:256]}\n"
                    f"counts={counts.tolist()[:64]} — a corrupted write "
                    "stamped this step's live prefix; capture the staged "
                    "write list."
                )
            live_max = int(live.max().item())
            if live_max > 2**30:
                raise RuntimeError(
                    "[BLOCK-TABLE-AUDIT] token-scale value in the live "
                    f"prefix of group {i} (max={live_max}) — a grid/token "
                    "mix was stamped into the block table this step."
                )

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
        out: tuple[torch.Tensor, ...] | None = None,
        out_ptrs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.num_kv_cache_groups == 0:
            return ()
        if out is None:
            out = tuple(self.input_block_tables)
            out_ptrs = self.input_block_table_ptrs
        else:
            assert out_ptrs is not None
            assert len(out) == self.num_kv_cache_groups
        num_reqs = idx_mapping.shape[0]
        # Launch kernel with num_reqs_padded to fuse zeroing of padded rows.
        _gather_block_tables_kernel[(self.num_kv_cache_groups, num_reqs_padded)](
            idx_mapping,
            self.block_table_ptrs,
            out_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            num_reqs,
            BLOCK_SIZE=1024,  # type: ignore
        )
        return tuple(bt[:num_reqs_padded] for bt in out)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        #
        # Zero the rows so dummy runs write mamba state to the reserved null
        # block rather than through the previous real step's (stale) block
        # ids, which may point at blocks since freed and reallocated.
        return tuple(
            block_table[:num_reqs].zero_() for block_table in self.input_block_tables
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.num_kv_cache_groups == 0:
            return (self.slot_mappings if out is None else out)[:, :num_tokens_padded]
        num_reqs = idx_mapping.shape[0]
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.num_blocks.gpu,
            self.num_blocks.gpu.stride(0),
            self.block_sizes_tensor,
            self.kernel_block_sizes_tensor,
            self.group_cp_sizes,
            slot_mappings,
            slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,  # type: ignore
        )
        return slot_mappings[:, :num_tokens_padded]

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        # Fill the entire slot_mappings tensor, not just the first `num_tokens` entries.
        # This is because the padding logic is complex and kernels may access beyond
        # the requested range.
        self.slot_mappings.fill_(PAD_SLOT_ID)
        # NOTE(woosuk): The output may be used for CUDA graph capture.
        # Therefore, this method must return the persistent tensor
        # with the same memory address as that used during the model's forward pass,
        # rather than allocating a new tensor.
        return self.slot_mappings[:, :num_tokens]


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    max_num_blocks = stride  # stride equals max_num_blocks for this group.
    dst_block_table_ptr = _load_ptr(dst_block_table_ptrs + group_id, tl.int32)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr(src_block_table_ptrs + group_id, tl.int32)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)

    for i in tl.range(num_blocks, max_num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    block_sizes,  # [num_kv_cache_groups]
    kernel_block_sizes,  # [num_kv_cache_groups]
    group_cp_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    kv_block_size = tl.load(block_sizes + group_id)
    kernel_block_size = tl.load(kernel_block_sizes + group_id)
    group_cp_size = tl.load(group_cp_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    num_blocks = tl.load(group_num_blocks_ptr + req_state_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        token_mask = offset < end_idx
        positions = tl.load(pos + offset, mask=token_mask, other=0)

        if CP_SIZE == 1 or group_cp_size == 1:
            # Common case: Context parallelism is not used.
            local_positions = positions
            is_local = token_mask
        else:
            # Context parallelism is used.
            virtual_block_size = kv_block_size * CP_SIZE
            virtual_block_indices = positions // virtual_block_size
            virtual_block_offsets = positions % virtual_block_size
            is_local = virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = virtual_block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            local_positions = virtual_block_indices * kv_block_size + local_offsets

        block_indices = local_positions // kernel_block_size
        block_offsets = local_positions % kernel_block_size
        valid_block = token_mask & (block_indices < num_blocks)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local & valid_block,
            other=0,
        )
        slot_ids = block_numbers * kernel_block_size + block_offsets
        if CP_SIZE != 1 and group_cp_size != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        slot_ids = tl.where(valid_block, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
