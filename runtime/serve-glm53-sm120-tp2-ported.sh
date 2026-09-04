#!/usr/bin/env bash
# GLM-5.3-Flash TR3 4bpw — v84 runtime + upstream-core-port r1 (fork image).
# Standalone serve script (runtime/ style): identical runtime configuration to
# runtime/compose.sm120-tp2-ported.yaml (MTP3, B12X_MLA_SPARSE, nvfp4_ds_mla,
# prefix caching ON, --max-model-len 262144, v84-default cudagraph sizes), run
# via docker run instead of compose. The image already carries the
# upstream-core-port overlay baked in — no bind mounts needed.
set -euo pipefail

IMAGE="${IMAGE:-legend/glm53-flash-exl3-k4:r1-upstream-core-port}"
MODEL="${MODEL:?set MODEL to the local EXL3 checkpoint directory}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
PORT="${PORT:-8012}"
NAME="${NAME:-glm53-flash-exl3-k4-ported}"
CACHE_PATH="${GLM53_CACHE_PATH:-${PWD}/glm53-vllm-cache}"

mkdir -p "${CACHE_PATH}"

exec docker run --rm --name "${NAME}" \
  --init --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_DEBUG_KDA_INPUTS=0 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_USE_B12X_DCP_A2A=1 \
  -e OMP_NUM_THREADS=2 \
  -e NCCL_P2P_LEVEL=4 \
  -v "${MODEL}:/model:ro" \
  -v "${CACHE_PATH}:/cache" \
  "${IMAGE}" serve /model \
  --served-model-name GLM-5.3-Flash-EXL3-4bpw \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --decode-context-parallel-size 2 \
  --dcp-comm-backend a2a \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --load-format safetensors \
  --moe-backend b12x \
  --attention-backend B12X_MLA_SPARSE \
  --kv-cache-dtype nvfp4_ds_mla \
  --max-model-len 262144 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.987 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --generation-config /model \
  --chat-template /opt/glm53/chat_template.multimodal.jinja \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --prefill-schedule-interval 8 \
  --kda-prefill-backend triton \
  --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  "$@"
