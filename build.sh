#!/usr/bin/env bash
# Build legend/glm53-flash-exl3-k4:r1-upstream-core-port from the digest-pinned
# v84 base image + the upstream-core-port overlay in this repo.
#
# CPU-only: the build is overlay COPYs + a python compile self-check; no GPU
# or model download happens at build time. Run from the repo root or anywhere
# (paths are repo-relative).
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-legend/glm53-flash-exl3-k4:r1-upstream-core-port}"

# 1) build-context gate: every overlay .py must byte-compile (local python,
#    no imports needed) and the manifest must map only shipped files.
bash scripts/check-overlay-compile.sh

# 2) build the image (in-image self-check RUN compiles every mapped overlay
#    destination with /opt/venv/bin/python and fails the build on any error).
docker build --tag "${TAG}" .

# 3) report the built image id.
docker image inspect "${TAG}" --format 'image: {{json .RepoTags}}
id: {{.Id}}'
