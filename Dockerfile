# Rostrum - reproducible Linux image for the whole pipeline.
#
# Why this exists (one image fixes five blockers at once):
#   1. llvmlite.dll is blocked by Windows App Control on the dev machine - gone on Linux.
#   2. no conda on the dev machine            -> micromamba baked in.
#   3. no ffmpeg on the dev machine           -> apt ffmpeg baked in.
#   4. MFA + english_us_arpa models           -> baked in at build time.
#   5. "reproducible across deployment environments" -> one pinned image.
#
# Two environments, deliberately apart (see environment.mfa.yml):
#   /app/.venv        pip venv, everything except alignment (Makefile's $(PY))
#   micromamba "mfa"  Montreal Forced Aligner only, reached via `conda run -n mfa`
# A `conda` shim forwards to micromamba so the Makefile and
# src/rostrum/align/mfa.py need no Docker-specific branch.
FROM python:3.11-slim

# MFA_ROOT_DIR is set to a world-writable path so the models downloaded here as
# root stay usable by the non-root runtime user (MFA 3.x postgres refuses root).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=src \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    MFA_ROOT_DIR=/opt/mfa \
    TORCH_HOME=/opt/torch \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System packages: ffmpeg (decode/resample), build-essential (sed_scores_eval
# Cython extension), libsndfile1 (soundfile), make/git/curl for the targets.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libsndfile1 \
        libgomp1 \
        make \
        git \
        curl \
        bzip2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# micromamba + a `conda` shim (rostrum.align.mfa already accepts either name,
# but the Makefile calls `conda run -n mfa`).
# ---------------------------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local bin/micromamba \
    && printf '#!/bin/sh\nexec /usr/local/bin/micromamba "$@"\n' > /usr/local/bin/conda \
    && chmod +x /usr/local/bin/conda \
    && micromamba --version

# ---------------------------------------------------------------------------
# The `mfa` environment + the english_us_arpa models (invariant 4).
# ---------------------------------------------------------------------------
WORKDIR /app
COPY environment.mfa.yml ./
RUN micromamba create -y -n mfa -f environment.mfa.yml \
    && micromamba clean -a -y \
    && micromamba run -n mfa mfa version \
    && micromamba run -n mfa mfa model download acoustic english_us_arpa \
    && micromamba run -n mfa mfa model download dictionary english_us_arpa \
    && micromamba run -n mfa mfa model list acoustic \
    && chmod -R a+rwX /opt/mfa

# ---------------------------------------------------------------------------
# The pip venv. Copied separately so dependency layers survive source edits.
# ---------------------------------------------------------------------------
COPY requirements.txt ./
RUN python -m venv /app/.venv \
    && /app/.venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /app/.venv/bin/python -m pip install \
         --extra-index-url https://download.pytorch.org/whl/cpu \
         -r requirements.txt \
    && find /root/.cache -type d -name 'pip*' -prune -exec rm -rf {} + || true

# UTMOS for the injection QC gate, cached at build time so inference needs no
# network (invariant 13). Non-fatal: `make qc` reports the miss itself.
RUN /app/.venv/bin/python -c "import torch; torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)" \
      && chmod -R a+rwX /opt/torch \
      || echo "WARNING: UTMOS prefetch failed; 'make qc' will try again at runtime"

# ---------------------------------------------------------------------------
# The project itself.
# ---------------------------------------------------------------------------
COPY . .
RUN /app/.venv/bin/python -m pip install -e . --no-deps

# Non-root: MFA 3.x starts a local postgres, and postgres refuses to run as root.
RUN useradd --create-home --uid 1000 rostrum \
    && chown -R rostrum:rostrum /app /opt/micromamba \
    && chmod -R a+rwX /opt/mfa
USER rostrum

EXPOSE 8501 8000

CMD ["make", "app"]
