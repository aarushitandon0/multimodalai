#!/usr/bin/env bash
# Verification gate for the Docker image (`make docker-verify`).
#
# Three checks, fail-fast, all inside the container:
#   1. make test        - the whole suite must pass with 0 failures
#   2. make mfa-version - MFA version + the locally available model names
#   3. make smoke-align - one real forced alignment, monotonic timestamps
#
# The repo keeps no committed dataset_sample/smoke/<id>.wav pair, so the
# transcript for the smoke recording is written out of the sample text JSON
# (verbatim field) into a temp file and passed to scripts/smoke_align.py.
set -euo pipefail

WAV=${DOCKER_SMOKE_WAV:-dataset_sample/audio/s1_lv01.wav}
TEXT=${DOCKER_SMOKE_TEXT:-dataset_sample/transcripts/s1_gettysburg.json}
PY=${PY:-.venv/bin/python}

echo "=== [1/3] make test ==================================================="
make test

echo "=== [2/3] make mfa-version ============================================"
make mfa-version

echo "=== [3/3] make smoke-align ============================================"
TXT=$(mktemp /tmp/smoke_XXXX.txt)
"$PY" -c 'import json,sys; sys.stdout.write(json.load(open(sys.argv[1]))["verbatim"])' "$TEXT" > "$TXT"
make smoke-align SMOKE_ARGS="--wav $WAV --txt $TXT"

echo "=== all three checks passed ==========================================="
