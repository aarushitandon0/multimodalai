"""Where the dashboard gets its analyses from (prompt 22).

The dashboard calls the pipeline directly, **or** the FastAPI service when
``ROSTRUM_API_URL`` is set. Both are real deployments and both must behave
identically on screen, so the difference is confined to this module: the
screens only ever see a :class:`Backend`.

.. code-block:: text

    ROSTRUM_API_URL unset  ->  DirectBackend  ->  rostrum.pipeline.analyze
    ROSTRUM_API_URL set    ->  ApiBackend     ->  POST {url}/analyze

``DirectBackend`` is the default because it is the honest one for a judge
running ``make app`` on a laptop: one process, no network, nothing to start in
the right order. ``ApiBackend`` is what a split deployment uses - the Hugging
Face Space of spec 15.3 - and it exists here so that configuration is a
one-line change rather than a different dashboard.

Both backends return the same two things: an
:class:`~rostrum.types.AnalysisResult` and the stage timings, because the
processing panel is part of the interface and a remote analysis must be able to
report what it spent its time on.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rostrum.clips import Clip, ClipError, ClipStore, participant_clip, reference_clip
from rostrum.config import Config, load_config
from rostrum.types import AnalysisResult

logger = logging.getLogger(__name__)

__all__ = [
    "ApiBackend",
    "Backend",
    "BackendError",
    "DirectBackend",
    "make_backend",
]


class BackendError(RuntimeError):
    """The analysis could not be obtained, with a message fit to show a user."""


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #


class Backend(Protocol):
    """What the screens need from whatever is doing the analysing."""

    name: str

    def analyze(
        self,
        audio: Path,
        transcript: str,
        text_id: str | None = None,
    ) -> tuple[AnalysisResult, list[dict[str, Any]]]:
        """Analyse one recording; return the result and the stage timings."""
        ...

    def library(self) -> dict[str, Any]:
        """The texts and sample files available."""
        ...

    def health(self) -> dict[str, Any]:
        """Versions, config hash and limits."""
        ...

    def clip(
        self,
        result: AnalysisResult,
        t_start: float,
        t_end: float,
        which: str = "participant",
    ) -> Clip | bytes:
        """Audio for the A/B players: a :class:`Clip`, or raw WAV bytes."""
        ...


# --------------------------------------------------------------------------- #
# In-process
# --------------------------------------------------------------------------- #


@dataclass
class DirectBackend:
    """Calls :func:`rostrum.pipeline.analyze` in this process.

    The reference library is loaded once and reused, which is both the fast
    path and the correct one: it is read-only, fitted on training folds, and
    nothing an upload does can alter what the next upload is compared against
    (CLAUDE.md invariant 11).

    Attributes:
        config: Configuration.
        name: Shown in the dashboard's footer.
    """

    config: Config = field(default_factory=load_config)
    name: str = "in-process pipeline"
    _library: Any = field(default=None, repr=False)

    # -- lazily shared state ----------------------------------------------

    @property
    def library_object(self) -> Any:
        """The :class:`~rostrum.pipeline.ReferenceLibrary`, loaded on first use."""
        if self._library is None:
            from rostrum.pipeline import load_library

            self._library = load_library(self.config)
            for warning in self._library.warnings:
                logger.warning("library: %s", warning)
        return self._library

    @property
    def store(self) -> ClipStore:
        """Where uploads are kept, so a clip can be cut after the fact."""
        return ClipStore.from_config(self.config)

    # -- the interface ----------------------------------------------------

    def analyze(
        self,
        audio: Path,
        transcript: str,
        text_id: str | None = None,
    ) -> tuple[AnalysisResult, list[dict[str, Any]]]:
        """Analyse one recording with the local pipeline.

        The upload is copied into the clip store first and analysed *there*, so
        the processed WAV every timestamp refers to survives the request and
        the A/B players have something to cut (see :mod:`rostrum.clips`).

        Raises:
            BackendError: The analysis failed, with the pipeline's own reason.
        """
        from rostrum.pipeline import AnalysisError, analyse_id, analyze, stages_of

        source = Path(audio)
        data = source.read_bytes()
        analysis_id = analyse_id(data, transcript, self.config.hash)
        directory = self.store.put(
            analysis_id,
            audio=data,
            filename=source.name,
            transcript=transcript,
            text_id=text_id,
        )
        upload = directory / str(self.store.meta(analysis_id)["upload"])

        dashboard = dict(self.config.serving["dashboard"])
        try:
            result = analyze(
                upload,
                transcript,
                text_id,
                config=self.config,
                library=self.library_object,
                file_id=source.stem or "upload",
                refine=bool(dashboard.get("refine", True)),
                with_curves=bool(dashboard.get("with_curves", True)),
                with_band=bool(dashboard.get("with_band", True)),
                check_text_match=bool(dashboard.get("check_text_match", True)),
                work_dir=directory,
            )
        except (AnalysisError, FileNotFoundError, ValueError) as exc:
            raise BackendError(str(exc)) from exc

        return result, stages_of(result.analysis_id, self.config)

    def library(self) -> dict[str, Any]:
        """The same payload ``GET /library`` returns, built locally."""
        from rostrum.api.server import library_endpoint

        return _as_dict(library_endpoint())

    def health(self) -> dict[str, Any]:
        """The same payload ``GET /health`` returns, built locally."""
        from rostrum.api.server import health_endpoint

        return _as_dict(health_endpoint())

    def clip(
        self,
        result: AnalysisResult,
        t_start: float,
        t_end: float,
        which: str = "participant",
    ) -> Clip:
        """Cut one window locally.

        Raises:
            BackendError: No audio is available for that window, with the
                reason - most often that the dataset's reference WAVs or
                TextGrids have not been built on this machine.
        """
        pad = float(dict(self.config.serving["dashboard"]).get("clip_pad_s", 0.15))
        try:
            if which == "reference":
                return reference_clip(
                    result,
                    t_start,
                    t_end,
                    pad_s=pad,
                    sr=self.config.sample_rate,
                    config=self.config,
                )
            wav = self.store.processed_wav(result.analysis_id, file_id=result.file_id)
            return participant_clip(
                wav,
                t_start,
                t_end,
                alignment=result.alignment,
                file_id=result.file_id,
                pad_s=pad,
                sr=self.config.sample_rate,
            )
        except (ClipError, FileNotFoundError, OSError) as exc:
            raise BackendError(str(exc)) from exc


def _as_dict(model: Any) -> dict[str, Any]:
    """A pydantic model, or anything dict-like, as a plain dict."""
    if hasattr(model, "model_dump"):
        return dict(model.model_dump(mode="json"))
    return dict(model)


# --------------------------------------------------------------------------- #
# Over HTTP
# --------------------------------------------------------------------------- #


@dataclass
class ApiBackend:
    """Calls the FastAPI service at ``ROSTRUM_API_URL``.

    Attributes:
        url: Base URL, without a trailing slash.
        timeout: Seconds to wait. The default is generous because a five-minute
            recording is allowed and a cold analysis of one is not quick; spec
            15.3 targets 60 s for two minutes of audio on CPU.
        name: Shown in the dashboard's footer.
    """

    url: str
    timeout: float = 600.0
    name: str = "remote API"

    def __post_init__(self) -> None:
        """Normalise the URL and name the backend after it."""
        self.url = str(self.url).rstrip("/")
        self.name = f"remote API at {self.url}"

    def _client(self) -> Any:
        import httpx

        return httpx.Client(base_url=self.url, timeout=self.timeout)

    def analyze(
        self,
        audio: Path,
        transcript: str,
        text_id: str | None = None,
    ) -> tuple[AnalysisResult, list[dict[str, Any]]]:
        """POST the upload to ``/analyze`` and read the result back.

        Raises:
            BackendError: The service refused the request or is unreachable.
            The service's own ``detail`` message is passed through unchanged,
            because it is already written for a person.
        """
        import httpx

        source = Path(audio)
        files = {"audio": (source.name, source.read_bytes(), _media_type(source))}
        data = {"transcript": transcript}
        if text_id:
            data["text_id"] = text_id

        try:
            with self._client() as client:
                response = client.post("/analyze", files=files, data=data)
                if response.status_code >= 400:
                    raise BackendError(_detail(response))
                result = AnalysisResult.from_dict(response.json())
                stages = client.get(f"/stages/{result.analysis_id}")
                timings = stages.json() if stages.status_code == 200 else []
        except httpx.HTTPError as exc:
            raise BackendError(f"the API at {self.url} could not be reached: {exc}") from exc
        return result, list(timings)

    def library(self) -> dict[str, Any]:
        """``GET /library``."""
        return self._get("/library")

    def health(self) -> dict[str, Any]:
        """``GET /health``."""
        return self._get("/health")

    def _get(self, path: str) -> dict[str, Any]:
        """One GET returning JSON, or a :class:`BackendError`."""
        import httpx

        try:
            with self._client() as client:
                response = client.get(path)
            if response.status_code >= 400:
                raise BackendError(_detail(response))
            return dict(response.json())
        except httpx.HTTPError as exc:
            raise BackendError(f"the API at {self.url} could not be reached: {exc}") from exc

    def clip(
        self,
        result: AnalysisResult,
        t_start: float,
        t_end: float,
        which: str = "participant",
    ) -> bytes:
        """``GET /clip`` - raw WAV bytes, ready for ``st.audio``."""
        import httpx

        try:
            with self._client() as client:
                response = client.get(
                    "/clip",
                    params={
                        "analysis_id": result.analysis_id,
                        "t_start": round(float(t_start), 3),
                        "t_end": round(float(t_end), 3),
                        "which": which,
                    },
                )
            if response.status_code >= 400:
                raise BackendError(_detail(response))
            return bytes(response.content)
        except httpx.HTTPError as exc:
            raise BackendError(f"the API at {self.url} could not be reached: {exc}") from exc


def _media_type(path: Path) -> str:
    """A reasonable content type from the extension."""
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }.get(path.suffix.lower(), "application/octet-stream")


def _detail(response: Any) -> str:
    """The service's error message, or a generic one with the status code."""
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return f"the API returned HTTP {response.status_code}"
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, list) and detail:
        # FastAPI's 422 validation shape.
        first = detail[0]
        return str(first.get("msg", first))
    return str(detail or f"the API returned HTTP {response.status_code}")


# --------------------------------------------------------------------------- #
# Choosing one
# --------------------------------------------------------------------------- #


def make_backend(config: Config | None = None) -> Backend:
    """The backend this deployment is configured for.

    ``ROSTRUM_API_URL`` selects the remote service; with it unset the pipeline
    runs in this process.

    Args:
        config: Configuration, passed to :class:`DirectBackend`.

    Returns:
        A :class:`Backend`.
    """
    url = os.environ.get("ROSTRUM_API_URL", "").strip()
    if url:
        logger.info("dashboard: using the API at %s", url)
        return ApiBackend(url=url)
    return DirectBackend(config=config or load_config())
