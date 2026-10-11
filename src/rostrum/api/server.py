"""FastAPI service over the analysis pipeline (spec 15.1).

Four endpoints, and a deliberately thin layer:

=====================  ===================================================
``POST /analyze``      multipart audio + transcript -> ``AnalysisResult``
``GET  /clip``         a window of the participant, or of a skilled reader
``GET  /library``      the texts and sample files this deployment holds
``GET  /health``       versions, config hash, limits, what is installed
=====================  ===================================================

What this module is allowed to do is check the request, call
:func:`rostrum.pipeline.analyze`, and serialise what comes back. What it must
not do is compute anything a score depends on: every number and every
timestamp in a response comes from the pipeline, so the API cannot become a
second, divergent implementation of the analysis.

Three consequences of that, which the code below enforces:

* **No fitting, no leakage.** The reference library is loaded once, read-only,
  and shared by every request (CLAUDE.md invariant 11).
* **Nothing is recomputed for presentation.** ``/clip`` cuts audio by the
  timestamps already in the result; it never re-measures anything.
* **No network and no LLM** (invariants 12, 13). A request is served entirely
  from the local dataset, cache and models.

Run it with ``make api``, or::

    uvicorn rostrum.api.server:app --port 8000
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from rostrum.api.models import (
    AnalysisResultModel,
    ClipInfoModel,
    HealthModel,
    LibraryModel,
    LibrarySampleModel,
    LibraryTextModel,
)
from rostrum.clips import (
    Clip,
    ClipError,
    ClipStore,
    participant_clip,
    reference_clip,
)
from rostrum.config import Config, code_version, load_config
from rostrum.io.audio import AudioError, duration_s, read_mono
from rostrum.pipeline import (
    AnalysisError,
    ReferenceLibrary,
    analyse_id,
    analyze,
    load_library,
    stages_of,
)
from rostrum.types import AnalysisResult

logger = logging.getLogger(__name__)

__all__ = ["Limits", "app", "get_config", "get_library", "limits_of"]

API_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Shared, read-only state
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def get_config() -> Config:
    """The configuration, loaded once per process."""
    return load_config()


@lru_cache(maxsize=1)
def get_library() -> ReferenceLibrary:
    """The reference library, loaded once per process.

    Loading is the expensive part of an analysis (envelopes, pooled norms, the
    classifier and every reference's cached features), and none of it depends
    on the request, so it happens once. The object is never mutated, which is
    also what keeps invariant 11 true across concurrent requests: there is no
    code path by which one upload could alter what the next is compared with.
    """
    library = load_library(get_config())
    for warning in library.warnings:
        logger.warning("library: %s", warning)
    return library


@lru_cache(maxsize=1)
def get_store() -> ClipStore:
    """Where uploads are kept so ``/clip`` can cut from them later."""
    return ClipStore.from_config(get_config())


@dataclass(frozen=True)
class Limits:
    """What a request is allowed to ask for, from ``configs/serving.yaml``.

    Attributes:
        max_upload_bytes: Largest upload accepted, checked before decoding.
        max_duration_s: Longest recording accepted, checked after decoding.
        accept_extensions: Extensions ``/analyze`` will take.
        clip_pad_s: Context added either side of a clip window.
        max_clip_s: Longest excerpt ``/clip`` will serve.
        clip_pick: Default reference-reader criterion; see :mod:`rostrum.clips`.
        cors_origins: Origins allowed to call the API from a browser.
    """

    max_upload_bytes: int
    max_duration_s: float
    accept_extensions: tuple[str, ...]
    clip_pad_s: float
    max_clip_s: float
    clip_pick: str
    cors_origins: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON form, for ``/health``."""
        return {
            "max_upload_bytes": self.max_upload_bytes,
            "max_upload_mb": round(self.max_upload_bytes / (1024 * 1024), 1),
            "max_duration_s": self.max_duration_s,
            "accept_extensions": list(self.accept_extensions),
            "max_clip_s": self.max_clip_s,
            "clip_pick": self.clip_pick,
        }


def limits_of(config: Config | None = None) -> Limits:
    """Read :class:`Limits` from the configuration."""
    block = dict((config or get_config()).serving["api"])
    return Limits(
        max_upload_bytes=int(block["max_upload_bytes"]),
        max_duration_s=float(block["max_duration_s"]),
        accept_extensions=tuple(str(e).lower() for e in block["accept_extensions"]),
        clip_pad_s=float(block["clip_pad_s"]),
        max_clip_s=float(block["max_clip_s"]),
        clip_pick=str(block["clip_pick"]),
        cors_origins=tuple(str(o) for o in block["cors_origins"]),
    )


# --------------------------------------------------------------------------- #
# The application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Rostrum",
    version=API_VERSION,
    summary="Contrastive speech analytics with temporally grounded flaw regions.",
    description=__doc__,
)

# CORS for the local dashboard only. The service reads the local dataset and
# writes to the local cache, so it is not something to expose to the internet;
# the origins are listed in configs/serving.yaml rather than opened with "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(limits_of(load_config()).cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# POST /analyze
# --------------------------------------------------------------------------- #


def _check_upload(data: bytes, filename: str, limits: Limits) -> None:
    """Refuse an upload that is too large or not audio, before decoding it.

    Raises:
        HTTPException: 413 when too large, 415 when the extension is not one
            the decoders handle, 400 when the body is empty.
    """
    if not data:
        raise HTTPException(status_code=400, detail="the uploaded file is empty")
    if len(data) > limits.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"the upload is {len(data) / 1048576:.1f} MB; the limit is "
                f"{limits.max_upload_bytes / 1048576:.0f} MB"
            ),
        )
    suffix = Path(filename or "").suffix.lower()
    if suffix not in limits.accept_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{suffix or 'that file'} is not a supported audio format; "
                f"expected one of {', '.join(limits.accept_extensions)}"
            ),
        )


def _check_duration(path: Path, limits: Limits, config: Config) -> float:
    """Decode the upload and refuse it if it is too long.

    The byte limit does not bound duration - 20 MB of MP3 is far more than five
    minutes - so the decoded signal is measured as well. The decode happens
    here rather than inside the pipeline so that a too-long or undecodable file
    is a clean 4xx instead of a 500 from somewhere deeper.

    Returns:
        The duration in seconds.

    Raises:
        HTTPException: 400 when the audio cannot be decoded, 413 when it is
            longer than the limit.
    """
    try:
        samples, sr = read_mono(path, sr=config.sample_rate)
    except (AudioError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"the audio could not be decoded: {exc}"
        ) from exc
    seconds = duration_s(samples, sr)
    if seconds > limits.max_duration_s:
        raise HTTPException(
            status_code=413,
            detail=(
                f"the recording is {seconds / 60:.1f} min; the limit is "
                f"{limits.max_duration_s / 60:.0f} min"
            ),
        )
    if seconds <= 0:
        raise HTTPException(status_code=400, detail="the recording contains no audio")
    return seconds


@app.post(
    "/analyze",
    response_model=AnalysisResultModel,
    summary="Analyse one recording against its transcript",
    responses={
        400: {"description": "Empty body, empty transcript, or undecodable audio"},
        413: {"description": "Over the size or duration limit"},
        415: {"description": "Unsupported audio format"},
        422: {"description": "No comparison could be formed"},
    },
)
async def analyze_endpoint(
    audio: Annotated[UploadFile, File(description="Audio: wav/mp3/m4a/ogg, <= 20 MB, <= 5 min.")],
    transcript: Annotated[
        str, Form(description="What the speaker was reading. Punctuation matters.")
    ],
    text_id: Annotated[
        str | None,
        Form(description="Force a library text instead of matching by similarity."),
    ] = None,
) -> AnalysisResultModel:
    """Analyse an upload and return the full :class:`AnalysisResult`.

    Both parts are required by spec 15.1: audio alone cannot be analysed,
    because every timestamp in the result is anchored to a word of the
    transcript.

    The response carries whichever operating mode the pipeline chose -
    ``reference``, ``norm`` or ``reject`` - and ``mode_reason`` says why. A
    rejected upload is a **200 with no score**, not an error: "this recording
    does not match this transcript" is a finding, and the client should show it.
    """
    config = get_config()
    limits = limits_of(config)

    data = await audio.read()
    _check_upload(data, audio.filename or "", limits)
    if not transcript.strip():
        raise HTTPException(
            status_code=400,
            detail="a transcript is required; audio alone cannot be analysed",
        )

    # The id is a pure function of what was sent, so the same upload reuses the
    # same stored audio and the same cached analysis.
    analysis_id = analyse_id(data, transcript, config.hash)
    store = get_store()
    directory = store.put(
        analysis_id,
        audio=data,
        filename=audio.filename or "upload.wav",
        transcript=transcript,
        text_id=text_id,
    )
    upload = directory / str(store.meta(analysis_id)["upload"])
    _check_duration(upload, limits, config)

    try:
        result = analyze(
            upload,
            transcript,
            text_id or None,
            config=config,
            library=get_library(),
            file_id=Path(audio.filename or "upload").stem or "upload",
            # The processed WAV lands here and stays, so /clip can cut from the
            # exact signal the timestamps refer to (invariant 2).
            work_dir=directory,
        )
    except AnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:  # pragma: no cover - the file was just written
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AnalysisResultModel.model_validate(result.to_dict())


@app.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisResultModel,
    summary="Re-read a cached analysis",
    responses={404: {"description": "No such analysis in the cache"}},
)
def analysis_endpoint(analysis_id: str) -> AnalysisResultModel:
    """Serve an analysis the pipeline has already run.

    The dashboard uses this to redraw a result - and the report to cite one -
    without paying for the analysis twice, which is also the cheapest proof
    that a result is reproducible from its id alone.
    """
    result = _cached_result(analysis_id)
    return AnalysisResultModel.model_validate(result.to_dict())


@app.get(
    "/stages/{analysis_id}",
    summary="Stage timings of a cached analysis",
)
def stages_endpoint(analysis_id: str) -> list[dict[str, Any]]:
    """What each pipeline stage cost, for the processing panel (spec 15.2).

    Timings are wall-clock numbers, so they are deliberately *not* part of the
    hashed result; they live beside it in the cache entry.
    """
    return stages_of(_safe_id(analysis_id), get_config())


def _safe_id(analysis_id: str) -> str:
    """Reject anything that is not a hex analysis id, before touching a path."""
    safe = "".join(c for c in str(analysis_id) if c.isalnum())
    if not safe or safe != str(analysis_id):
        raise HTTPException(status_code=400, detail="that is not a valid analysis id")
    return safe


def _cached_result(analysis_id: str) -> AnalysisResult:
    """Read one cached analysis, or 404.

    Raises:
        HTTPException: 404 when the analysis is not cached, 500 when the cache
            entry exists but cannot be parsed.
    """
    config = get_config()
    path = config.path("cache.analyses") / f"{_safe_id(analysis_id)}.json"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"no analysis {analysis_id} is cached; POST /analyze first, or the "
                "cache was cleared"
            ),
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return AnalysisResult.from_dict(payload["result"])
    except (OSError, KeyError, json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"the cached analysis is unreadable: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# GET /clip
# --------------------------------------------------------------------------- #


def _clip_response(clip: Clip) -> Response:
    """A WAV response carrying the clip's provenance in its headers.

    The body is audio, so the provenance - which reader, which words, why that
    reader - travels in ``X-Rostrum-Clip`` as JSON. A dashboard can show it
    under the player without a second request.
    """
    info = ClipInfoModel.model_validate(clip.to_dict())
    return Response(
        content=clip.to_wav_bytes(),
        media_type="audio/wav",
        headers={
            "X-Rostrum-Clip": json.dumps(info.model_dump(mode="json"), separators=(",", ":")),
            "Content-Disposition": (
                f'inline; filename="{clip.which}_{clip.file_id}_'
                f'{clip.t_start:.3f}-{clip.t_end:.3f}.wav"'
            ),
            "Cache-Control": "no-store",
        },
    )


@app.get(
    "/clip",
    summary="A WAV excerpt of the participant, or of a skilled reader",
    response_class=Response,
    responses={
        200: {"content": {"audio/wav": {}}, "description": "16 kHz mono 16-bit PCM"},
        404: {"description": "No such analysis, or no reference audio for it"},
        409: {"description": "The window maps to no words"},
    },
)
def clip_endpoint(
    analysis_id: Annotated[str, Query(description="The analysis the window belongs to.")],
    t_start: Annotated[float, Query(ge=0.0, description="Window onset, seconds.")],
    t_end: Annotated[float, Query(gt=0.0, description="Window offset, seconds.")],
    which: Annotated[
        Literal["participant", "reference"],
        Query(description="Whose audio to return."),
    ] = "participant",
    pick: Annotated[
        Literal["representative", "nearest"] | None,
        Query(description="Which reference reader; defaults to serving.yaml."),
    ] = None,
) -> Response:
    """Cut one window out of the participant or out of a skilled reading.

    A **participant** clip is cut at the timestamps the result reports, since
    those already refer to the processed WAV (invariant 2).

    A **reference** clip is cut by *word index*, not by clock time: the window
    is mapped to the words it covers, and those same words are then located in
    the chosen reader's own alignment. A participant who drags takes longer
    over a phrase than any reader does, so cutting the reader at the
    participant's seconds would clip them mid-word. See :mod:`rostrum.clips`
    for how the reader is chosen.
    """
    config = get_config()
    limits = limits_of(config)
    if t_end <= t_start:
        raise HTTPException(status_code=400, detail="t_end must be greater than t_start")

    result = _cached_result(analysis_id)
    try:
        if which == "reference":
            clip = reference_clip(
                result,
                t_start,
                t_end,
                pick=pick or limits.clip_pick,  # type: ignore[arg-type]
                pad_s=limits.clip_pad_s,
                sr=config.sample_rate,
                config=config,
            )
        else:
            wav = get_store().processed_wav(_safe_id(analysis_id), file_id=result.file_id)
            clip = participant_clip(
                wav,
                t_start,
                t_end,
                alignment=result.alignment,
                file_id=result.file_id,
                pad_s=limits.clip_pad_s,
                sr=config.sample_rate,
            )
    except ClipError as exc:
        # 404 when the audio simply is not here, 409 when it is here but the
        # window cannot be resolved to words.
        status = 409 if "maps to no words" in str(exc) or "no words at" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except (AudioError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return _clip_response(clip)


# --------------------------------------------------------------------------- #
# GET /library
# --------------------------------------------------------------------------- #


def _manifest_rows(config: Config) -> list[dict[str, str]]:
    """Rows of the sample manifest, or of the full one when the sample is absent."""
    for key in ("sample.dir", "data.dir"):
        try:
            path = config.path(key) / "MANIFEST.csv"
        except (KeyError, TypeError):  # pragma: no cover - keys always present
            continue
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
    return []


def _sample_audio(config: Config, file_id: str) -> Path | None:
    """Where a manifest row's audio actually is, if it is on disk at all."""
    candidates = [
        config.path("sample.audio") / f"{file_id}.wav",
        config.path("data.audio_clean") / f"{file_id}.wav",
        config.path("data.audio_injected") / f"{file_id}.wav",
        config.path("data.audio_human_flawed") / f"{file_id}.wav",
    ]
    return next((p for p in candidates if p.is_file()), None)


@app.get("/library", response_model=LibraryModel, summary="Texts and sample files available")
def library_endpoint() -> LibraryModel:
    """What this deployment can compare an upload against (spec 15.1).

    Each text says how many skilled readings stand behind it and therefore
    which mode an upload of it would be analysed in, so a client can tell a
    judge *before* they upload whether they will get a reference comparison or
    the lower-confidence pooled one.
    """
    config = get_config()
    library = get_library()
    min_references = int(config.detection["modes"]["reference"]["min_references"])

    texts: list[LibraryTextModel] = []
    for text_id, text in sorted(library.texts.items()):
        has_envelope = library.envelope_for(text_id) is not None
        enough = text.n_references >= min_references
        texts.append(
            LibraryTextModel(
                text_id=text_id,
                title=text.title,
                n_references=text.n_references,
                reference_ids=list(text.reference_ids),
                n_words=len(text.transcript.split()),
                has_envelope=has_envelope,
                mode="reference" if (has_envelope and enough) else "norm",
                transcript=text.transcript,
            )
        )

    samples: list[LibrarySampleModel] = []
    for row in _manifest_rows(config):
        file_id = str(row.get("file_id", ""))
        path = _sample_audio(config, file_id) if file_id else None
        if path is None:
            continue
        duration = row.get("duration_s") or ""
        samples.append(
            LibrarySampleModel(
                file_id=file_id,
                text_id=row.get("text_id") or None,
                path=str(path.relative_to(config.root).as_posix()),
                origin=str(row.get("origin", "")),
                tier=row.get("tier") or None,
                label=str(row.get("label", "")),
                level=str(row.get("level", "")),
                duration_s=float(duration) if duration else None,
            )
        )

    return LibraryModel(
        fold=library.fold,
        texts=texts,
        samples=sorted(samples, key=lambda s: s.file_id),
        min_references=min_references,
        pooled_references=len(library.pooled.reference_ids) if library.pooled else 0,
        has_classifier=library.classifier is not None,
        warnings=list(library.warnings),
    )


# --------------------------------------------------------------------------- #
# GET /health
# --------------------------------------------------------------------------- #


def _package_versions() -> dict[str, str]:
    """Versions of the libraries whose numbers end up in a result."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "praat-parselmouth",
        "librosa",
        "soundfile",
        "scikit-learn",
        "ruptures",
        "fastapi",
        "streamlit",
        "plotly",
    ):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


@app.get("/health", response_model=HealthModel, summary="Versions and config hash")
def health_endpoint() -> HealthModel:
    """Everything needed to say which build produced a given result (spec 15.1).

    ``config_hash`` and ``code_version`` are the two numbers a report cites;
    ``library_fingerprint`` is the third, because the same code and config over
    a different set of reference readings is a different system. ``status`` is
    ``degraded`` when something the analysis needs is missing, so a deployment
    problem shows up here rather than as a confusing result later.
    """
    config = get_config()
    warnings: list[str] = []

    try:
        library = get_library()
        fingerprint = library.fingerprint()
        warnings.extend(library.warnings)
        n_texts = len(library.texts)
        pooled = len(library.pooled.reference_ids) if library.pooled else 0
        if not library.texts:
            warnings.append("no library texts are loaded; every upload will be Norm or Reject")
        if library.classifier is None:
            warnings.append("no flaw-type classifier is loaded; regions carry the rule label only")
    except Exception as exc:  # pragma: no cover - a broken dataset, reported not raised
        fingerprint, n_texts, pooled = "", 0, 0
        warnings.append(f"the reference library could not be loaded: {exc}")

    packages = _package_versions()
    missing = [name for name, value in packages.items() if value == "not installed"]
    if missing:
        warnings.append(f"not installed: {', '.join(missing)}")

    return HealthModel(
        status="ok" if not warnings else "degraded",
        code_version=code_version(config.root),
        config_hash=config.hash,
        library_fingerprint=fingerprint,
        schema_version=AnalysisResult.SCHEMA_VERSION,
        python=platform.python_version(),
        packages=packages,
        aligner={
            "timing": "montreal-forced-aligner (invariant 4)",
            "env_name": str(config.paths["mfa"]["env_name"]),
            "acoustic_model": str(config.paths["mfa"]["acoustic_model"]),
            "dictionary": str(config.paths["mfa"]["dictionary"]),
            "asr_check_only": "faster-whisper",
        },
        limits=limits_of(config).to_dict(),
        texts=n_texts,
        pooled_references=pooled,
        warnings=warnings,
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    """Run the service with uvicorn. ``python -m rostrum.api.server --port 8000``."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Rostrum API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "rostrum.api.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
