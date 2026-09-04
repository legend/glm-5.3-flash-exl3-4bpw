# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from functools import partial

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import (
    async_tensor_h2d,
    get_accelerator_view_from_cpu_tensor,
)

# Default round-robin depth for the UVA buffer pools. Must be >= the number of
# concurrent in-flight steps (engine batch_queue_size).
_DEFAULT_MAX_CONCURRENCY = 2


def set_default_max_concurrency(n: int) -> None:
    global _DEFAULT_MAX_CONCURRENCY
    _DEFAULT_MAX_CONCURRENCY = max(2, n)


_ASYNC_H2D_INFLIGHT: list = []

_DIFF_H2D_DUMP_ENABLED: bool | None = None


def _diff_h2d_dump_enabled() -> bool:
    """[DIFF-DUMP] marker gate for the H2D-STAGE probe, cached per process
    (the raw /cache/.diff-dump exists() stat used to run on every small
    staged-metadata copy -- a per-step syscall in production)."""
    global _DIFF_H2D_DUMP_ENABLED
    if _DIFF_H2D_DUMP_ENABLED is None:
        try:
            import os as _os

            _DIFF_H2D_DUMP_ENABLED = _os.path.exists("/cache/.diff-dump")
        except Exception:
            _DIFF_H2D_DUMP_ENABLED = False
    return _DIFF_H2D_DUMP_ENABLED


def async_copy_to_gpu(
    x: torch.Tensor | np.ndarray,
    out: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    global _ASYNC_H2D_INFLIGHT
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu

    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)

    # pin_memory() is no-op if the memory is already pinned.
    pinned = x.pin_memory()
    nbytes = x.numel() * x.element_size()
    if nbytes <= 4096:
        # [H2D LIFETIME FIX, small-metadata arm] Per-step staged metadata
        # (idx_mapping, query grids, draft counts - a few dozen bytes) is
        # copied SYNCHRONOUSLY: the ~10us blocking cost per small copy is
        # negligible per step, and it closes the torn-read window
        # deterministically regardless of host-allocator subtleties. The
        # copy is complete at return, so the pinned source needs no
        # event-gated inflight hold (the async arm below keeps one).
        if _diff_h2d_dump_enabled():
            try:
                print(
                    f"[DIFF-DUMP] H2D-STAGE ptr={out.data_ptr():#x} "
                    f"vals={x.flatten()[:8].tolist()}",
                    flush=True,
                )
            except Exception:
                pass
        return out.copy_(pinned)
    # [H2D LIFETIME FIX - the Xid-31/43 producer] Hold the pinned source
    # alive until the async DMA completes. The previous version returned
    # with `pinned` dying at scope exit: the pinned block went back to the
    # host allocator while the copy was still in flight, the next
    # same-size pin reused it, and the GPU received torn/stale bytes -
    # measured live (2026-09-03 probe-run8, [DIFF-DUMP] DRAFT-IDX):
    # idx_mapping arrived on TP0 as [9143985375312] (a freed host-pointer
    # fragment) vs the correct [0] on TP1, driving the
    # IndexKernel.cu:111 device assert at the draft gather
    # req_states.draft_tokens[idx_mapping, :n], and - through wilder staged
    # metadata on other call sites (query grids, block tables, state
    # indices) - the whole Xid-31/43 fault family on BOTH engine cores
    # (run-1 token-index OOB, run-2 draft_tokens OOB, old-core garbage
    # state ids -> wild FLA/copy writes -> Xid 31 FAULT_PDE VIRT_WRITE).
    result = out.copy_(pinned, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    _ASYNC_H2D_INFLIGHT.append((pinned, ev))
    if len(_ASYNC_H2D_INFLIGHT) > 64:
        _ASYNC_H2D_INFLIGHT = [
            (t, e) for (t, e) in _ASYNC_H2D_INFLIGHT if not e.query()
        ]
    return result


class UvaBuffer:
    def __init__(self, size: int | Sequence[int], dtype: torch.dtype):
        if not is_uva_available():
            raise RuntimeError("UVA is not available")
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.uva = get_accelerator_view_from_cpu_tensor(self.cpu)


class UvaBufferPool:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        self.size = size
        self.dtype = dtype
        self.max_concurrency = max_concurrency

        # UVA buffers for concurrency
        self._uva_bufs = [UvaBuffer(size, dtype) for _ in range(max_concurrency)]
        # Current buffer index
        self._curr = 0
        # [COMPLETION-SYNC GUARD] Per-buffer CUDA events. A buffer may be
        # recycled while its previous GPU write (issued via non_blocking
        # copies reading this UVA memory) is still in flight; the round-robin
        # then overwrites UVA memory a pending kernel has not finished
        # reading, and recycled CPU staging memory gets re-allocated as small
        # GPU-adjacent metadata tensors whose heap neighborhood receives the
        # stale staged block-id lists (the [2072..2075]/[2168..2171] stamp
        # class). Recording an event per handout and waiting on reuse closes
        # the race without a full device sync (the event only waits for work
        # enqueued on the current stream at handout time).
        import torch as _torch_evt

        self._uva_events = [
            _torch_evt.cuda.Event() for _ in range(max_concurrency)
        ]

    def copy_to_uva(self, x: torch.Tensor | np.ndarray | list) -> torch.Tensor:
        # Round robin to the next buffer.
        self._curr = (self._curr + 1) % self.max_concurrency
        buf = self._uva_bufs[self._curr]
        # [COMPLETION-SYNC GUARD] wait for any prior stream work that was
        # reading this buffer before overwriting it (fail-safe: event waits
        # only for recorded stream progress; never blocks the host).
        try:
            self._uva_events[self._curr].synchronize()
        except Exception:
            pass
        # CPU-to-CPU copy
        dst = buf.cpu if isinstance(x, torch.Tensor) else buf.np
        n = len(x)
        dst[:n] = x
        try:
            self._uva_events[self._curr].record()
        except Exception:
            pass
        return buf.uva[:n]

    def record_current(self) -> None:
        """[COMPLETION-SYNC GUARD completion] Re-record the current slot's
        event after the consuming kernel/copy launch, so recycling waits for
        the consumer - not just the handout point."""
        try:
            self._uva_events[self._curr].record()
        except Exception:
            pass

    def copy_to_gpu(
        self,
        x: torch.Tensor | np.ndarray,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        uva = self.copy_to_uva(x)
        # CPU-to-GPU copy
        return uva.clone() if out is None else out.copy_(uva, non_blocking=True)


class UvaBackedTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        max_concurrency: int | None = None,
    ):
        self.dtype = dtype

        # Source of truth
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu", pin_memory=False)
        self.np = self.cpu.numpy()

        # Buffers for concurrency
        self.pool = UvaBufferPool(size, dtype, max_concurrency)
        self.gpu = self.pool.copy_to_uva(self.np)

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        # CPU-to-CPU copy
        self.gpu = self.pool.copy_to_uva(self.np[:n] if n is not None else self.np)
        return self.gpu


class StagedWriteTensor:
    def __init__(
        self,
        size: int | Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
        max_concurrency: int | None = None,
        uva_instead_of_gpu: bool = False,
    ):
        if max_concurrency is None:
            max_concurrency = _DEFAULT_MAX_CONCURRENCY
        supported_dtypes = [torch.int32, torch.int64, torch.float32]
        if dtype not in supported_dtypes:
            raise ValueError(
                f"Unsupported dtype {dtype}: should be one of {supported_dtypes}"
            )
        self.num_rows = size if isinstance(size, int) else size[0]
        self.dtype = dtype
        self.device = device
        self.max_concurrency = max_concurrency

        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
        else:
            # For a large but not-frequently-accessed tensor, we can use UVA instead of
            # GPU to save GPU memory
            self._uva_buf = UvaBuffer(size, dtype)
            self.gpu = self._uva_buf.uva

        self._staged_write_indices: list[int] = []
        self._staged_write_starts: list[int] = []
        self._staged_write_contents: list[int | float] = []
        self._staged_write_cu_lens: list[int] = []

        new_buffer = partial(UvaBufferPool, max_concurrency=max_concurrency)

        self.write_indices = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_starts = new_buffer(self.num_rows, dtype=torch.int32)
        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    def stage_write(
        self, index: int, start: int, x: Iterable[int] | Iterable[float]
    ) -> None:
        assert index >= 0
        assert start >= 0
        if not x:
            return
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(start)
        self._staged_write_contents.extend(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def stage_write_elem(self, index: int, x: int) -> None:
        assert index >= 0
        self._staged_write_indices.append(index)
        self._staged_write_starts.append(0)
        self._staged_write_contents.append(x)
        self._staged_write_cu_lens.append(len(self._staged_write_contents))

    def apply_write(self) -> None:
        n = len(self._staged_write_indices)
        if n == 0:
            return

        indices_uva = self.write_indices.copy_to_uva(self._staged_write_indices)
        starts_uva = self.write_starts.copy_to_uva(self._staged_write_starts)
        cu_lens_uva = self.write_cu_lens.copy_to_uva(self._staged_write_cu_lens)

        # Special handling for write_contents
        write_contents = async_tensor_h2d(
            self._staged_write_contents, device=self.device, dtype=self.dtype
        )

        # [WRITE-SIDE fix] rows/width for the in-kernel store clamp.
        num_rows_gpu = torch.full(
            (1,), self.gpu.shape[0], dtype=torch.int32, device=self.device
        )
        num_cols_gpu = torch.full(
            (1,), self.gpu.shape[-1], dtype=torch.int32, device=self.device
        )
        # [STREAM-SAFE LIFETIME] see FusedStagedWriter.apply.
        try:
            import torch as _torch_rs

            _stream = _torch_rs.cuda.current_stream()
            for _buf in (indices_uva, starts_uva, write_contents, cu_lens_uva):
                if isinstance(_buf, torch.Tensor):
                    _buf.record_stream(_stream)
        except Exception:
            pass

        # Write diffs to the GPU buffer
        _apply_write_kernel[(n,)](
            self.gpu,
            self.gpu.stride(0),
            indices_uva,
            starts_uva,
            write_contents,
            cu_lens_uva,
            None,
            num_rows_gpu,
            num_cols_gpu,
            BLOCK_SIZE=1024,
            MULTI_GROUP=False,
        )
        # [COMPLETION-SYNC GUARD completion] Re-record the pools' events
        # AFTER the consuming kernel launch: the handout-time record
        # predates the kernel, so recycling could previously overwrite a
        # UVA buffer this kernel was still reading (the round-2/3 tear).
        for _pool in (
            self.write_indices,
            self.write_starts,
            self.write_cu_lens,
        ):
            try:
                _pool.record_current()
            except Exception:
                pass
        # Clear the staged writes
        self.clear_staged_writes()

    def clear_staged_writes(self) -> None:
        self._staged_write_indices.clear()
        self._staged_write_starts.clear()
        self._staged_write_contents.clear()
        self._staged_write_cu_lens.clear()


class FusedStagedWriter:
    """Applies the staged writes of several `StagedWriteTensor`s at once."""

    def __init__(
        self, device: torch.device, max_writes: int, max_concurrency: int | None = None
    ):
        new_pool = partial(
            UvaBufferPool, dtype=torch.int32, max_concurrency=max_concurrency
        )
        self.group_ids = new_pool(max_writes)
        self.indices = new_pool(max_writes)
        self.starts = new_pool(max_writes)
        self.cu_lens = new_pool(max_writes)
        self.device = device

    def apply(
        self,
        tensors: Sequence[StagedWriteTensor],
        output_ptrs: torch.Tensor,
        output_strides: torch.Tensor,
    ) -> None:
        """Apply and clear the staged writes of `tensors` with one kernel."""
        group_ids: list[int] = []
        indices: list[int] = []
        starts: list[int] = []
        contents: list[int | float] = []
        cu_lens: list[int] = []
        # [WRITE-SIDE fix] per-group rows/width for the store clamp.
        num_rows: list[int] = []
        num_cols: list[int] = []

        for group_id, t in enumerate(tensors):
            n = len(t._staged_write_indices)
            if n == 0:
                num_rows.append(0)
                num_cols.append(0)
                continue

            group_ids.extend([group_id] * n)
            indices.extend(t._staged_write_indices)
            starts.extend(t._staged_write_starts)
            content_base = len(contents)
            contents.extend(t._staged_write_contents)
            cu_lens.extend(content_base + cu_len for cu_len in t._staged_write_cu_lens)
            num_rows.append(t.gpu.shape[0])
            num_cols.append(t.gpu.shape[-1])

        if not group_ids:
            return

        group_ids_uva = self.group_ids.copy_to_uva(group_ids)
        indices_uva = self.indices.copy_to_uva(indices)
        starts_uva = self.starts.copy_to_uva(starts)
        cu_lens_uva = self.cu_lens.copy_to_uva(cu_lens)
        contents_gpu = async_tensor_h2d(contents, device=self.device, dtype=torch.int32)

        # [WRITE-SIDE fix] per-group rows/width for the store clamp.
        num_rows_gpu = async_tensor_h2d(
            num_rows, device=self.device, dtype=torch.int32
        )
        num_cols_gpu = async_tensor_h2d(
            num_cols, device=self.device, dtype=torch.int32
        )

        # [STREAM-SAFE LIFETIME] The UVA buffers returned by copy_to_uva and
        # the async_tensor_h2d staging tensors are read by the kernel below
        # on the current stream, then their Python refs die at return — the
        # allocator can hand their memory to the next allocation (the small
        # metadata tensors) BEFORE the kernel finishes reading it: a
        # cross-stream use-after-free that stamps staged block-id lists into
        # freshly allocated metadata (the [2072..2075]/[2168..2171] class).
        # record_stream pins each buffer's lifetime to the consuming stream
        # so the allocator defers reuse until the kernel completes.
        try:
            import torch as _torch_rs

            _stream = _torch_rs.cuda.current_stream()
            for _buf in (
                group_ids_uva,
                indices_uva,
                starts_uva,
                cu_lens_uva,
                contents_gpu,
                num_rows_gpu,
                num_cols_gpu,
            ):
                if isinstance(_buf, torch.Tensor):
                    _buf.record_stream(_stream)
        except Exception:
            pass

        _apply_write_kernel[(len(group_ids),)](
            output_ptrs,
            output_strides,
            indices_uva,
            starts_uva,
            contents_gpu,
            cu_lens_uva,
            group_ids_uva,
            num_rows_gpu,
            num_cols_gpu,
            BLOCK_SIZE=1024,
            MULTI_GROUP=True,
        )
        # [COMPLETION-SYNC GUARD completion] Re-record the pools' events
        # AFTER the consuming kernel launch (see apply_write note).
        for _pool in (
            self.group_ids,
            self.indices,
            self.starts,
            self.cu_lens,
        ):
            try:
                _pool.record_current()
            except Exception:
                pass
        for t in tensors:
            t.clear_staged_writes()


@triton.jit
def _apply_write_kernel(
    output_ptr,  # MULTI_GROUP: ptr-to-ptrs [num_groups]; else: data ptr
    output_stride,  # MULTI_GROUP: ptr-to-strides [num_groups]; else: row stride
    write_indices_ptr,
    write_starts_ptr,
    write_contents_ptr,
    write_cu_lens_ptr,
    write_group_ids_ptr,  # [num_writes], used only when MULTI_GROUP
    write_num_rows_ptr,  # [num_groups] / scalar: rows of the target tensor
    write_num_cols_ptr,  # [num_groups] / scalar: row width of the target
    BLOCK_SIZE: tl.constexpr,
    MULTI_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    row_idx = tl.load(write_indices_ptr + pid)
    start_idx = tl.load(write_starts_ptr + pid)

    cu_start = tl.load(write_cu_lens_ptr + pid - 1) if pid > 0 else 0
    cu_end = tl.load(write_cu_lens_ptr + pid)
    content_len = cu_end - cu_start

    if MULTI_GROUP:
        # Each write targets a different output tensor (KV cache group);
        # resolve its base pointer and row stride per write.
        group_id = tl.load(write_group_ids_ptr + pid)
        row_ptr = _load_ptr(output_ptr + group_id, tl.int32)
        row_stride = tl.load(output_stride + group_id)
        num_cols = tl.load(write_num_cols_ptr + group_id)
        num_rows = tl.load(write_num_rows_ptr + group_id)
    else:
        row_ptr = output_ptr
        row_stride = output_stride
        num_cols = tl.load(write_num_cols_ptr)
        num_rows = tl.load(write_num_rows_ptr)
    row_ptr += row_idx * row_stride + start_idx

    # [WRITE-SIDE Xid-31/43 fix] Bound the scatter to the target tensor.
    # The read-side select kernel got this clamp hours ago; the write side
    # never did, so a staged write whose row + start + content_len ran past
    # the tensor's end scattered into heap-adjacent allocations (the fresh
    # KDA metadata tensors), corrupting whichever consumer kernel touched
    # them next. Lanes past the allocated width/height are dropped.
    in_row = row_idx < num_rows
    for i in range(0, content_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        col_ok = (start_idx + block) < num_cols
        mask = (block < content_len) & col_ok & in_row
        content = tl.load(write_contents_ptr + cu_start + block, mask=mask)
        tl.store(row_ptr + block, content, mask=mask)


@triton.jit
def _load_ptr(ptr_to_ptr, elem_dtype):
    ptr = tl.load(ptr_to_ptr)
    ptr = tl.cast(ptr, tl.pointer_type(elem_dtype))
    return tl.multiple_of(ptr, 16)
