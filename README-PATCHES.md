# Patch Set — `upstream-core-port/` (release r1)

This fork carries the complete production patch set that the base image
(v84 runtime) does **not** include: an upstream engine-core port plus a family
of root-cause GPU fault and correctness fixes, differential-debugged against
the v84 runtime as the oracle. Everything lives in [`upstream-core-port/`](upstream-core-port/)
and is shipped as a plain file overlay — the Dockerfile for this fork `COPY`s it
over the same two trees the production compose bind-mounts into.

- **Base image (digest-pinned):** `verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`
- **Base vLLM lineage:** the `infernal-invocation` vLLM fork, PR-head `5f8e00d6c3` of [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
- **This overlay:** upstream PR-head engine core (the "r7 composition") + fork-compat
  bridges, ported **onto** the v84 model-side files
- **Provenance label in the patched image:** `local-inference.upstream-core-port.release=r1`

## Overlay layout

Each subdirectory is one port/fix wave; files keep their package-relative paths
so the whole tree can be copied straight over the vLLM package in the image.
[`upstream-core-port/MANIFEST.txt`](upstream-core-port/MANIFEST.txt) lists every
file with its exact in-image destination.

| Directory | Files | Contents | In-image target |
|---|---|---|---|
| `upstream-g1/vllm/` | 24 | scheduler, KV-cache utils/manager/coordinator/interface/pool, platforms, metrics, multimodal, model-executor interfaces | `/opt/infernal-invocation/vllm/vllm/` |
| `upstream-g2/vllm/` | 45 | worker chain: buffers, block tables, mamba hybrid state, cudagraph utils, speculators, rejection sampler, sampling, MLA attention layer, attention ops (DCP/PCP) | `/opt/infernal-invocation/vllm/vllm/` |
| `upstream-g3/vllm/` | 2 | `gpu/model_runner.py` (V2) + `gpu_model_runner.py` (V1) | `/opt/infernal-invocation/vllm/vllm/` |
| `upstream-g4/vllm/` | 7 | indexer, compressor utils, sparse kpool indexer, attention-backend utils, KV layout, deep_gemm, `config/cache.py` | `/opt/infernal-invocation/vllm/vllm/` (cache.py also into the venv's installed `vllm/config/`) |
| `r7/vllm/` | 9 | KEEP model-side files: glm5next `model.py`/`kda.py`, `kimi_gdn_linear_attn.py`, `abstract.py`, `gdn_attn.py`, `b12x.py`, `warmup.py`, `fused_recurrent.py`, `causal_conv1d.py` | `/opt/infernal-invocation/vllm/vllm/` |
| `r7-b12x/b12x/` | 2 | `attention/_shared/mla/{kernel.py, prefill.py}` — record-walk stride fix + storage validation gates | `/opt/infernal-invocation/b12x/b12x/` |
| `engine-core/` | 1 | `core.py` — the fork engine core with the prefill-throttle patch | `/opt/infernal-invocation/vllm/vllm/v1/engine/core.py` |

The production compose (`compose.sm120-tp2-ported.yaml` in this repo) mirrors
the bind-mount list these files were extracted from, for anyone who prefers the
overlay-mount style over the baked image.

## Patch history

### 1. Upstream core port (~85 files, ~60 fork-compat bridges)

The v84 runtime kept the model side (attention, MoE, speculators' kernels) but
ran a stale engine core. We ported the upstream engine core from the
PR-head lineage of `local-inference-lab/vllm` onto the v84 model-side files:
scheduler, KV-cache pool/manager/coordinator/interface, model runners
(V1 + V2), the full worker chain, and the speculator/rejection-sampler stack.
Bridging the two trees required ~60 fork-compat bridges — config attribute
differences, constructor signatures, `customize_spec` guards, profiling hooks,
encoder-manager wiring, grammar kwargs, and friends. This is the
`upstream-g1..g4` + `engine-core` portion of the overlay.

### 2. Sparse-MLA layer-view aliasing — the root-cause fix

The marquee bug. The ported per-bucket KV view emission gave the 12 sparse-MLA
layers views only one page (2.25 MB) apart, while each view claimed the full
310 MB pool — so every token past 7,808 silently read **another layer's KV**.
Symptoms: short answers correct, long contexts garbage, plus Xid 31 / Xid 43
GPU faults at the wild access.

Fix: layer-contiguous bucket emission — per-layer regions `num_blocks × P`
apart with extents that tile exactly — together with a collision-freeness
proof. The mamba/KpoolTail parasitic pairing was re-derived with byte-equal
twin proofs, and `KVBlockZeroer` per-layer coverage was fixed to match.

### 3. B12X kernel record-walk fix

The decode/prefill MG kernels derived the KV record stride from caller scalars
(a scratch-plan `page_size` × a format constant) instead of the emitted
per-layer view strides — a manager-level scalar could silently replace the
kernel-page pair (64/18432). The kernels are now view-faithful: stride comes
from `view.stride(0)`, page-block-size from `view.shape[1]`, with warmup guards.
Added storage-extent validation behind the `VLLM_B12X_STORAGE_CHECK` env gate
(`r7-b12x/b12x/attention/_shared/mla/`).

### 4. H2D lifetime fixes (~20 sites across 6 files)

Pinned host temporaries were dying at scope exit while `non_blocking` DMAs
were still in flight — torn metadata (the smoking gun: `idx_mapping` arriving
as a host-pointer fragment, proven at the value level). Fixes: synchronous
staging for KB-scale metadata, CUDA-event-held in-flight lists for large
transfers — `async_copy_to_gpu`, `async_tensor_h2d`, `CpuGpuBuffer.copy_to_uva`,
UVA pool recycle events, MLA chunk metadata (7 sites), and conv metadata.

### 5. Indexer fixes

The port had dropped the v84 write-path block-table translation (translated
table, factor 2, DCP remap, compressor DCP params, dcp-shard-count init), which
scrambled every kpool K write in-bounds — restored. Per-chunk `.item()` device
syncs were removed (2 per chunk) — the port now beats v84 on this path. Plus
the APC page-guard and a 4× right-sizing of the prefill buffers
(1321 MB → 132 MB per lane).

### 6. Gather bounds clamps

Logprobs UVA token-id reads, vocab gathers, and the drafter embedding gather
are now clamped (vocab clamp at the `_run_model` chokepoint), so torn metadata
degrades to a valid lookup instead of an out-of-bounds access / Xid.

### 7. Rejection-sampler padding mask (port bug)

The port dropped v84's `masked_fill` on verification rows, letting garbage
positive padded rows get "verified" — restored.

### 8. Perf / parity

v84-default cudagraph capture sizes (up to 64, including 24), scheduler
probe-gating (import/os.environ hoisted out of the hot loop), and all hunt
probes converted to env/marker-gated with zero cost when off.

## Measured outcomes

- A deterministic crash repro — warm 400k cache + 2×140k concurrent sessions
  that killed every pre-fix configuration in ≤ 90 s — now passes a clean
  5-round soak.
- 150k-token exact mid-document retrieval is correct.
- CC1 throughput: 147–157 tok/s (cudagraphs on, MTP-3).
- TTFT 8.4 s @ 33k context / 25.6 s @ 100k context (~4,700 tok/s prefill).

## What is *not* shipped here

- `_flashkda_C.abi3.so` (dormant ~4 MB extension binary) — not needed by the
  current code paths; deliberately left out of the public fork.
- Development/diagnostic harnesses, `__pycache__`, and port-time test scripts
  (`import_test.py`, `test_*.py`, run scripts) — excluded from the overlay.
- The `.diffbak-*` / backup composition trees (`r7-pooled` and friends) —
  intermediate work, not part of the live mount set.

## Credits

- **Upstream engine core:** [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) (PR-head lineage `5f8e00d6c3`)
- **Original v84 runtime:** brandonmmusic-max (the `verdictai/glm53-flash-exl3-k4:r19-...-v84-dflash2` image this fork builds on)
- **FlashKDA / vllm-project** — KDA attention kernels used by the model side

## Release r1.1 — performance batch (2026-09-05, peer-reviewed)

Applied on top of r1, from a 6-area optimization sweep with 3-agent peer review:

- **Scheduler Fix A**: `prefill_capacity_bound` latched to `bool(self.waiting)` on every
  non-deferred step, disarming `--prefill-schedule-interval` entirely under continuous
  submission (a 2048-token chunk ran in ~every window). The latch is removed; decode
  windows are guaranteed between prefill chunks.
- **kvcache 0001**: a non-participating KpoolTailManager vetoed partial hash hits for the
  whole engine, collapsing prefix-reuse granularity to 15,616 tokens (7808 × DCP2) — every
  sub-15k shared prefix got 0% cache hits. The veto now only counts participating groups:
  granularity back to 7,808.
- **kvcache 0002**: the mamba-align admission gate under-counted blocks (~2 vs ~14 for a
  100k no-hit request) → over-admission → measured 77 preemptions/3,467 requests. Honest count.
- **attn 0002**: the indexer's prefill path allocated a fresh `.contiguous()` block-table
  translation every step (2× per step, target + draft). Now a persistent buffer like the
  decode path.
- **Relay (tps-injector, separate repo)**: keepalive match (aiohttp 15s vs engine 5s caused
  a 36% request-failure + re-prefill amplification loop), safe one-shot send-retry, GATE
  journal-line removal, vision-image sizing without b64 decode.

Measurement corrections from the decode-speed investigation (opt-work/bench/RESULTS.md):
- vLLM streams one SSE chunk per MTP verify-step — chunk rates under-report tok/s by the
  acceptance factor. True CC1 was never below ~114 tok/s; the "36-46 tok/s crisis" was a
  chunk-counting artifact.
- Acceptance is content-driven (creative prose ~1.8, code ~2.9, enumeration 3.4-4.4
  tok/step); the historical 3.5-3.77 baseline was structured-content windows.
- True tok/s matrix (production shape, max-num-seqs 16, MTP3): CC1 122-142, CC2 184,
  CC4 253, CC8 339 (count-prompt CC8 aggregate 586).

## Release r2 — FP8 KV + scheduler/cache fixes (2026-09-06)

On top of r1.1, all peer-verified and benchmarked (llm-inference-bench A/B):

- **Native FP8 KV cache (fp8_ds_mla, 528B NoPE records)** — full read path
  (traits GLM_NOPE+ARBITRARY_FP32 incl. the infer_model_type 528-tie-break fix
  over the old DSV4 mis-inference, io/io_mg staging, kernel record validation,
  storage-check width parity, glm_nope_fp8 model version, width-authority
  replaces the four hardcoded 656s, NoPE-aware mla_attention spec) + the
  **Triton capture-safe 528-record writer** (torch path is CPU-reference only)
  + the prefill-MG admission for GLM_NOPE+ARBITRARY_FP32 (d_rope=0 rope staging
  is zero-iteration const_expr) + the DSL raise-in-const_expr fix (io_mg).
  Env-gated: `VLLM_B12X_FP8_KV=1`, default off = bit-identical nvfp4.
  Benched A/B: FP8 decode is 2.8x nvfp4 at 16k-32k/C8 (ITL p50 58->31ms),
  parity at ctx 0; prefill ~5% slower; KV pool 1.25M vs 1.75M tokens.
  Rollback/config portability: the scheduler block is dtype-coupled —
  retention interval must be 15,616 (nvfp4) / 17,920 (FP8).
- **Scheduler Fix A (throttle-latch)**: prefill_capacity_bound latched to
  bool(waiting) disarmed --prefill-schedule-interval under continuous load.
- **Fix B (decode-aware chunk cap)**: 512-token chunk budget when decodes run
  (ITL p95 421ms -> 27ms under concurrent whale prefill, 0 stalls >300ms).
- **kvcache 0001 (kpool veto)**: prefix granularity 15,616 -> 7,808.
- **kvcache 0002 (honest mamba admission)**: kills over-admission preemptions.
- **attn 0002 (indexer persistent buffer)**: per-step .contiguous() alloc gone.
- **Cache-wipe fix (VLLM_MAMBA_STATE_PROTECT=16)**: admission reserve so a
  concurrent whale's allocations cannot strip a fresh session's cached mamba/
  MLA/draft entries (free-ring exhaustion, NOT capacity — the 2-sessions-wipe-
  each-other root cause; hit rate ~20% -> 60-80% under churn).
- **Int64 Triton branch fixes** (required for max-num-seqs 16).
- **Fused draft-decode (P1/F6)**: complete + shadow-verified correct, shipped
  gated-off (performance-neutral: the MTP overhead is draft GPU-forward, not
  host metadata; enable via VLLM_FUSED_DRAFT_DECODE=1 if a case appears).
- **Prefix retention (VLLM_PREFIX_CACHE_RETENTION_INTERVAL)**: mamba boundary
  states survive request completion (15,616 legal max on nvfp4).

## Release r2.1 — admission deadlock fix (2026-09-07)

Community-reported (RunTime_Terror, r2 fork, FP8 KV, 1M window): engine
wedges when a >200k session arrives while another decodes — all requests
(including new ones) queue forever. Root cause: the align-mode admission
cap bills null slots as real demand (cdiv(prompt, block) + spec per mamba
group, ~10x inflation), so VLLM_MAMBA_STATE_PROTECT's gate refuses a
request that would actually fit; eviction only runs during allocation and
the scheduler breaks its waiting loop on first refusal — self-sustaining.

Fixes (all verified red/green on the real classes):
- 0004: the align cap bills the real footprint (spec+2+ckpt+partial per
  mamba group, prompt-length-independent). Kill-switch
  VLLM_MAMBA_ALIGN_CAP_LEGACY=1.
- 0005: starvation backstop — a request blocked solely by the reserve for
  VLLM_MAMBA_STATE_PROTECT_AGE attempts (default 128) is admitted on
  demand alone.
- 0006: deferred frees drain on 0-token steps too (a fully-refused
  scheduler can no longer hold the last request's blocks out of the pool).

Gate-off behavior is bit-identical (harness-verified both stacks). With the
patch the reserve can stay ON: cache-hit protection survives churn rounds.
