# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from vllm import envs
from vllm.config.model import LogprobsMode
from vllm.distributed.parallel_state import is_global_first_rank
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

logger = init_logger(__name__)


def _should_capture_kld_batch(req_ids: list[str]) -> bool:
    if not os.environ.get("VLLM_KLD_CAPTURE_DIR"):
        return False
    synthetic_prefixes = ("_warmup_", "_dummy_req_")
    return not all(req_id.startswith(synthetic_prefixes) for req_id in req_ids)


def _maybe_capture_kld_prompt_logits(
    logits: torch.Tensor,
    *,
    req_id: str,
    start_idx: int,
    vocab_size: int,
) -> None:
    """Persist full-vocabulary prompt logits for an explicit KLD run."""
    capture_dir = os.environ.get("VLLM_KLD_CAPTURE_DIR")
    if not capture_dir or not is_global_first_rank():
        return

    safe_req_id = "".join(
        char if char.isalnum() or char in "-_." else "_" for char in req_id
    )
    request_dir = Path(capture_dir) / safe_req_id
    request_dir.mkdir(parents=True, exist_ok=True)
    end_idx = start_idx + logits.shape[0]
    output_path = request_dir / (
        f"logits.rows-{start_idx:06d}-{end_idx:06d}.safetensors"
    )
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite KLD capture chunk: {output_path}")

    logits_cpu = logits[:, :vocab_size].detach().to(device="cpu").float().contiguous()
    from safetensors.torch import save_file

    save_file(
        {"logits": logits_cpu},
        str(output_path),
        metadata={
            "request_id": req_id,
            "row_start": str(start_idx),
            "row_end": str(end_idx),
            "vocab_size": str(vocab_size),
        },
    )
    logger.info(
        "Saved KLD prompt-logit rows [%d, %d) shape=%s to %s",
        start_idx,
        end_idx,
        tuple(logits_cpu.shape),
        output_path,
    )


class PromptLogprobsWorker:
    def __init__(
        self,
        max_num_reqs: int,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        vocab_size: int | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.logprobs_mode = logprobs_mode
        self.vocab_size = vocab_size
        self.chunk_size = envs.VLLM_PROMPT_LOGPROBS_CHUNK_SIZE
        if self.chunk_size <= 0:
            raise ValueError(
                "VLLM_PROMPT_LOGPROBS_CHUNK_SIZE must be greater than zero, "
                f"got {self.chunk_size}"
            )

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_id -> CPU buffer containing all prompt logprobs accumulated so far.
        # Keeping chunk results on GPU until prompt completion can consume
        # unbounded device memory for long or concurrent prompts.
        self.in_progress_prompt_logprobs: dict[str, LogprobsTensors | None] = {}

    def add_request(self, req_id: str, req_idx: int, sampling_params: SamplingParams):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = None

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)

    def profile_run(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        max_num_logprobs: int,
    ) -> None:
        prompt_token_ids = torch.zeros(
            hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device
        )
        compute_prompt_logprobs_with_chunking(
            prompt_token_ids,
            hidden_states,
            logits_fn,
            max_num_logprobs,
            logprobs_mode=self.logprobs_mode,
            chunk_size=self.chunk_size,
        )

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens
        query_start_loc_np = input_batch.query_start_loc_np

        logits_capture: Callable[[torch.Tensor, int], None] | None = None
        if _should_capture_kld_batch(input_batch.req_ids):
            if len(input_batch.req_ids) != 1 or int(needs_prompt_logprobs.sum()) != 1:
                raise RuntimeError(
                    "VLLM_KLD_CAPTURE_DIR requires exactly one prompt-logprob "
                    "request in the batch"
                )
            req_id = input_batch.req_ids[0]
            capture_start = int(computed_prefill[0])
            capture_rows = int(query_start_loc_np[1] - query_start_loc_np[0])
            if not is_prompt_chunked[0]:
                capture_rows -= 1

            def logits_capture(logits: torch.Tensor, relative_start: int) -> None:
                if relative_start >= capture_rows:
                    return
                rows = min(logits.shape[0], capture_rows - relative_start)
                _maybe_capture_kld_prompt_logits(
                    logits[:rows],
                    req_id=req_id,
                    start_idx=capture_start + relative_start,
                    vocab_size=self.vocab_size or logits.shape[-1],
                )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
                logprobs_mode=self.logprobs_mode,
                chunk_size=self.chunk_size,
                logits_capture=logits_capture,
            )
        )
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            prompt_logprobs_cpu = self.in_progress_prompt_logprobs[req_id]
            if start_idx < end_idx:
                if prompt_logprobs_cpu is None:
                    prompt_logprobs_cpu = LogprobsTensors.empty_cpu(
                        int(prompt_lens[i]) - 1, width
                    )
                    self.in_progress_prompt_logprobs[req_id] = prompt_logprobs_cpu

                dst_start = int(computed_prefill[i])
                dst_end = dst_start + end_idx - start_idx
                prompt_logprobs_cpu.logprob_token_ids[dst_start:dst_end].copy_(
                    prompt_token_ids[start_idx:end_idx, :width]
                )
                prompt_logprobs_cpu.logprobs[dst_start:dst_end].copy_(
                    prompt_logprobs[start_idx:end_idx, :width]
                )
                prompt_logprobs_cpu.selected_token_ranks[dst_start:dst_end].copy_(
                    prompt_ranks[start_idx:end_idx]
                )

            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_cpu is None:
                continue

            prompt_logprobs_dict[req_id] = prompt_logprobs_cpu
            self.in_progress_prompt_logprobs[req_id] = None
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_state_rows,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    # [GATHER-BOUND FIX - Xid-31/43 amplifier] idx_mapping / num_computed_tokens
    # are staged metadata; a torn or stale value used to walk the UVA-backed
    # all_token_ids table out of its allocation (pinned host memory read over
    # PCIe), and the garbage token ids then became gather indices into the
    # logits rows downstream (OOB GPU reads -> device asserts). Clamp the row
    # index and the in-row position so a bad metadata word can only yield an
    # in-bounds (wrong but safe) token id.
    req_state_idx = tl.minimum(tl.maximum(req_state_idx, 0), num_state_rows - 1)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        pos_ok = (target_pos >= 0) & (target_pos < all_token_ids_stride)
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask & pos_ok,
            other=0,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        all_token_ids.shape[0],
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: LogprobsMode = "raw_logprobs",
    chunk_size: int = 1024,
    logits_capture: Callable[[torch.Tensor, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than zero, got {chunk_size}")
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    token_ids = []
    scores = []
    ranks = []
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], chunk_size):
        end_idx = start_idx + chunk_size
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        if logits_capture is not None:
            logits_capture(prompt_logits, start_idx)
        requested_num = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        del prompt_logits
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    scores = torch.cat(scores, dim=0) if len(scores) > 1 else scores[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, scores, ranks
