# legend/glm-5.3-flash-exl3-4bpw — r1 "upstream-core-port" release image.
#
# Same recipe as the original v84 release image (verdictai/...v84-dflash2):
# digest-pinned base + overlay files COPYed over the in-image source trees +
# provenance LABELs + a self-check RUN. ENTRYPOINT is inherited unchanged
# (/opt/venv/bin/vllm ... serve).
#
# Overlay: the upstream-core-port/ tree in this repo (see MANIFEST.txt and
# README-PATCHES.md for the full patch write-up). 89 files are copied over
# /opt/infernal-invocation/vllm/vllm/ (engine core + upstream g1-g4 + r7
# composition) and /opt/infernal-invocation/b12x/b12x/ (r7-b12x kernel fixes);
# config/cache.py is additionally copied over the installed vllm copy in
# /opt/venv/lib/python3.12/site-packages/vllm/ (production mounts it in both
# trees — the compose's venv line).
#
# The _flashkda_C.abi3.so kernel of the production overlay is NOT shipped in
# this fork (dormant binary; see README-PATCHES.md).
#
# Build context: repository root. Use ./build.sh.

FROM verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692

LABEL local-inference.upstream-core-port.release=r1 \
      local-inference.upstream-core-port.patch-date=2026-09-04 \
      local-inference.upstream-core-port.repo=https://github.com/legend/glm-5.3-flash-exl3-4bpw \
      local-inference.upstream-core-port.manifest=/opt/glm53/upstream-core-port-MANIFEST.txt \
      local-inference.base-image=verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692 \
      local-inference.base-image-label=verdictai/glm53-flash-exl3-k4:r19-sm120-tp2-ep2-dcp2-v84-dflash2 \
      org.opencontainers.image.title="GLM-5.3-Flash EXL3 4bpw v84 runtime — upstream-core-port r1" \
      org.opencontainers.image.description="v84 runtime with the upstream-core-port patch set: upstream engine core port, sparse-MLA layer-view aliasing fix, b12x record-walk stride fix, H2D lifetime fixes, indexer write-path restore, gather clamps, rejection-sampler padding mask, v84-perity perf." \
      org.opencontainers.image.source=https://github.com/legend/glm-5.3-flash-exl3-4bpw

# --- vllm source-tree overlay (group dirs are collision-free; see MANIFEST) ---
COPY upstream-core-port/upstream-g1/vllm/ /opt/infernal-invocation/vllm/vllm/
COPY upstream-core-port/upstream-g2/vllm/ /opt/infernal-invocation/vllm/vllm/
COPY upstream-core-port/upstream-g3/vllm/ /opt/infernal-invocation/vllm/vllm/
COPY upstream-core-port/upstream-g4/vllm/ /opt/infernal-invocation/vllm/vllm/
COPY upstream-core-port/r7/vllm/ /opt/infernal-invocation/vllm/vllm/

# --- b12x kernel fixes (record-walk stride + storage validation gates) ---
COPY upstream-core-port/r7-b12x/b12x/ /opt/infernal-invocation/b12x/b12x/

# --- installed vllm copy under /opt/venv needs the same config/cache.py ---
# (mirrors the production compose's site-packages mount)
COPY upstream-core-port/upstream-g4/vllm/config/cache.py /opt/venv/lib/python3.12/site-packages/vllm/config/cache.py

# --- fork engine core (prefill-throttle) over vllm/v1/engine/core.py ---
COPY upstream-core-port/engine-core/core.py /opt/infernal-invocation/vllm/vllm/v1/engine/core.py

# --- chat fixes: thinking-disable protocol patch + fixed chat template ---
COPY upstream-core-port/chat/protocol.py /opt/infernal-invocation/vllm/vllm/entrypoints/openai/chat_completion/protocol.py
COPY upstream-core-port/chat/chat_template.multimodal.jinja /opt/glm53/chat_template.multimodal.jinja

# --- parser fix: emit completed tool calls with undeclared names ---
COPY upstream-core-port/parser-fix/vllm/parser/engine/parser_engine.py /opt/infernal-invocation/vllm/vllm/parser/engine/parser_engine.py

# --- manifest: overlay file -> in-image destination (93 mappings / 92 files) ---
COPY upstream-core-port/MANIFEST.txt /opt/glm53/upstream-core-port-MANIFEST.txt

# Self-check, same pattern as the base image's provenance RUN: every mapped
# overlay destination must exist and byte-compile with the image's python.
RUN /bin/bash -o pipefail -c /opt/venv/bin/python - <<'PY'
from pathlib import Path

manifest = Path("/opt/glm53/upstream-core-port-MANIFEST.txt")
n = 0
for line in manifest.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "->" not in line:
        continue
    src, dst = (part.strip() for part in line.split("->", 1))
    path = Path(dst)
    assert path.is_file(), f"missing overlay destination: {dst} (from {src})"
    # in-memory byte-compile .py files only (the chat template is jinja)
    if path.suffix == ".py":
        compile(path.read_bytes(), str(path), "exec")
    n += 1
assert n >= 93, f"expected >=93 overlay mappings, compiled {n}"
print(f"UPSTREAM-CORE-PORT R2 SELF-CHECK PASSED ({n}/{n} overlay mappings compile)")
PY
