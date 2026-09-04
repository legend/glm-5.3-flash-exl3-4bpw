---
base_model: zai-org/GLM-5.3-Flash-BF16
library_name: transformers
pipeline_tag: image-text-to-text
license: other
license_name: shapleymcg-1.0
license_link: https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/main/LICENSE
tags:
  - glm
  - glm-5.3-flash
  - exl3
  - tr3
  - vllm
  - sm120
  - nvfp4
  - dflash2
  - multimodal
  - abliterated
---

# GLM-5.3-Flash TR3 4bpw — upstream-core port (r1)

Fork of [brandonmmusic-max/glm-5.3-flash-exl3-4bpw](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw)
carrying a **complete upstream engine-core port plus ~90 bug fixes**, a fixed
serving recipe, and the abliterated checkpoint option.

What this fork changes relative to the v84 baseline it is built on:

1. **Engine core replaced** — the scheduler, KV-cache pool/coordinator,
   interface, runners (V1+V2), worker chain, speculators, and rejection
   sampler are ported from the upstream
   [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)
   PR-head lineage (5f8e00d6c3 composition), with ~60 fork-compat bridges so
   they interoperate with the v84 in-image model-side files. The engine core
   itself is `engine-core/core.py` (the v84 core + the prefill-throttle
   patch, equivalent to upstream PR#546).
2. **~90 bugs fixed**, root-caused via differential debugging against the
   v84 oracle (full history in [README-PATCHES.md](README-PATCHES.md)). The
   headline item: the sparse-MLA layer-view aliasing bug — long contexts
   past ~7.8k tokens read another *layer's* KV (silent corruption or Xid
   31/43 GPU faults). Fixed with layer-contiguous bucket emission, a proven
   collision-free parasitic pairing (mamba/KpoolTail twin geometry), the
   b12x record-walk view-faithful stride fix, and ~20 H2D lifetime fixes.
3. **Abliterated checkpoint support** — the production recipe serves
   [lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated](https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated).
   Either checkpoint (censored or abliterated) works; set `GLM53_MODEL_PATH`.
4. **New serving recipe** — vision + tools + MTP3 + prefix caching, with the
   thinking-disable fixes (see credits) and v84-default CUDA-graph capture
   sizes.
5. **Thinking-disable fix** — `thinking:{type:disabled}` / `thinking:false` /
   `"disabled"` (top-level *and* `chat_template_kwargs`) now actually disables
   reasoning. Previously these were silently ignored and, when thinking ran
   past `max_tokens`, the response came back with `content: null` and
   `finish_reason: length` — which looks like a broken serve.

## Quick start

```bash
# Build the image (same recipe as the original: digest-pinned v84 base + overlay)
git clone https://github.com/legend/glm-5.3-flash-exl3-4bpw
cd glm-5.3-flash-exl3-4bpw
./build.sh     # -> legend/glm53-flash-exl3-k4:r1-upstream-core-port
# or: docker pull legend/glm53-flash-exl3-k4:r1-upstream-core-port

# Download a checkpoint (either):
#   hf download brandonmusic/GLM-5.3-Flash-tr3-4bpw --local-dir ...
#   hf download lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated --local-dir ...

GLM53_MODEL_PATH=/path/to/checkpoint \
GLM53_CACHE_PATH=/path/to/cache \
docker compose -f runtime/compose.sm120-tp2-ported.yaml up -d
curl http://127.0.0.1:8012/v1/models
```

Overlay-style (bind-mounts, matches production exactly) is also supported —
see `runtime/serve-glm53-sm120-tp2-ported.sh`.

## Current recipe (production serving profile)

| | |
|---|---|
| TP / EP / DCP | 2 / 2 / 2 (a2a) |
| Attention / MoE | B12X_MLA_SPARSE / b12x |
| KV cache | nvfp4_ds_mla (288 B/token) |
| Speculation | MTP3 (built-in head, probabilistic) |
| Vision / tools | on / on (glm45 reasoning + glm47 tool parsers) |
| Context | 262,144 |
| Batch / seqs | 2048 / 8 |
| CUDA-graph capture | [1,2,4,8,16,24,32,40,48,56,64] (v84 default) |
| KV pool | 1,747,626 tokens (6.67x @ full 262k requests) |

Measured on 2x RTX PRO 6000 Blackwell (96 GB), stock clocks:

| Measurement | Result |
|---|---:|
| C1 decode (graphed, MTP3) | 147–157 tok/s |
| Prefill | ~4,700 tok/s aggregate (8.4 s TTFT @ 33k, 25.6 s @ 100k) |
| Long-context retrieval | 150k tokens, exact mid-document quote — pass |
| Crash repro (was: Xid 31/43 in <=90 s) | clean, 5-round soak |

The deterministic crash shape (warm ~400k + two concurrent 140k multi-turn
sessions) killed every pre-fix configuration; it now passes a 5-round soak
with zero Xids. Full evidence trail: [README-PATCHES.md](README-PATCHES.md).

## Thinking control (all forms verified live)

```json
{"thinking": {"type": "disabled"}}   // top-level Zai form
{"thinking": false}                  // top-level short form
{"chat_template_kwargs": {"enable_thinking": false}}
```
All three disable reasoning and return clean content with `finish_reason:
stop`. Without the fix, these were silently dropped and produced
`content: null` at small `max_tokens`.

## What was fixed (summary)

See [README-PATCHES.md](README-PATCHES.md) for the full 8-category history
with evidence. The short list:

- **Sparse-MLA layer-view aliasing** (root cause of the Xid 31/43 family):
  12 sparse-MLA layer views were emitted one page apart while each claimed
  the full pool — every token past 7,808 read another layer's KV. Fixed via
  layer-contiguous bucket emission with a collision-freeness proof.
- **b12x record-walk stride** derived from caller scalars instead of the
  emitted view strides — a manager-level scalar silently replaced the
  kernel-page pair. Now view-faithful.
- **~20 H2D lifetime sites**: pinned host temps died at scope exit with
  non-blocking DMAs in flight → torn metadata (idx_mapping arrived as a
  host-pointer fragment — proven at value level).
- **Indexer write-path block-table translation** (dropped by the port,
  restored), per-chunk `.item()` device syncs removed (now beats v84), the
  1321 MB/lane prefill buffer right-sized to 132 MB.
- **Gather bounds clamps** (logprobs / vocab / drafter) — torn metadata now
  degrades to a valid lookup instead of an OOB access.
- **Rejection-sampler padding mask** (a port regression that "verified"
  garbage rows) restored.
- **Perf**: v84-default capture sizes, all debug probes env-gated
  zero-cost-when-off.

## Credits

- **Brandon Music (brandonmmusic-max)** — the original EXL3 4bpw quant, the
  v84 runtime base image this fork builds on, and the original serving
  profiles.
- **Chris (Local Inference Lab Discord)** — the chat template fix, the
  `chat_protocol.py` thinking-normalization patch (adopted here and extended
  to the top-level `thinking` field), the core update advisory (PR#546 —
  verified already carried by our engine core), and serving recipe advice
  (evaluated per-piece; the thinking fixes and template were adopted, the
  mixed-quant-only knobs were not applicable to uniform K4).
- **lovesenko** — the abliterated checkpoint
  ([GLM-5.3-Flash-tr3-4bpw-Abliterated](https://huggingface.co/lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated)).
- **local-inference-lab/vllm** — the upstream engine core this fork ports.
- **turboderp** — EXL3. **IncoAI** — DFlash2. The upstream vLLM project.

## Differences from upstream (brandonmmusic-max)

- Engine core replaced with the upstream port (upstream still ships v84's).
- ~90 additional bug fixes (upstream has none of these).
- The thinking-disable fix (upstream's v84 ignores these controls).
- Abliterated checkpoint support documented (upstream ships censored only).
- New measured datasheet (upstream's numbers are pre-port, prefix-cache-off,
  DFlash2-based; not comparable).

## Known limitations

- The crash fix family is verified by the deterministic repro + 5-round soak
  and 150k-retrieval, not by multi-day soak; upstream's v85 image (requested
  in [issue #3](https://github.com/brandonmmusic-max/glm-5.3-flash-exl3-4bpw/issues/3))
  remains the long-term cure.
- `_flashkda_C.abi3.so` is not shipped in this fork (dormant in production —
  see README-PATCHES.md exclusions).
- Prefix-cache reuse is weak (1.04–1.7x on repeats) — under investigation;
  shared prefixes below ~7.8k tokens get no reuse (structural: 7,808-token
  cache blocks).
- Mixed-K3/K4 recipes (satgeze) are NOT covered; this fork is uniform-K4 only.

## Provenance

The image embeds provenance labels (`local-inference.upstream-core-port.*`)
and an in-image self-check compiling all 91 overlay mappings. Base image:
`verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2@sha256:0f1cdcc8...`
(digest-pinned, unchanged from upstream). This checkpoint is distributed
under the ShapleyMCG License 1.0 in [LICENSE](LICENSE). Transparent
provenance only: no telemetry, callbacks, or inference modification.
