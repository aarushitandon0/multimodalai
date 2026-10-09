"""ASR-vs-transcript mismatch check (spec 8, 13).

The pipeline must refuse to analyse audio whose transcript is not the text
being spoken: every number downstream is anchored to a word index, so a wrong
transcript does not degrade the analysis, it invalidates it. Spec 8 settles the
test - ``faster-whisper`` transcribes the audio, the word error rate against
the supplied transcript is computed, and WER above
``detection.modes.reject.max_wer`` sends the analysis to Reject mode (spec 13).

Two project rules shape this module.

* **ASR is never used for timing** (CLAUDE.md invariant 4). The recogniser's
  output is compared with the transcript and then thrown away; forced
  alignment is MFA's job alone.
* **No network at inference** (invariant 13). The model is loaded with
  ``local_files_only`` by default, so a machine that has not pre-fetched the
  weights fails loudly instead of quietly downloading them mid-analysis.

Transcriptions are cached by audio hash under ``paths.cache.asr``, so a second
analysis of the same file - the determinism check runs two - costs nothing and
cannot drift.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rostrum.align.mfa import normalise_for_alignment
from rostrum.io.hashing import sha256_file

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "ASRError",
    "ASRNotAvailableError",
    "ASRSettings",
    "Transcription",
    "WordErrors",
    "check_transcript",
    "max_wer",
    "normalise_tokens",
    "transcribe",
    "wer",
]

CACHE_VERSION: int = 1
"""Bumped when the cached payload changes shape."""


class ASRError(RuntimeError):
    """Transcription failed."""


class ASRNotAvailableError(ASRError):
    """``faster-whisper`` or its weights are not installed on this machine."""


@dataclass(frozen=True)
class ASRSettings:
    """How the mismatch check runs (``detection.asr``).

    Attributes:
        model: Model size, e.g. ``tiny.en`` or ``base.en``. Small is enough:
            the test is "is this the same text", not "transcribe it well".
        device: ``cpu``; the dashboard runs on CPU.
        compute_type: CTranslate2 quantisation.
        beam_size: Fixed beam, so two runs agree.
        language: Forced language, or ``None`` to detect.
        local_files_only: Refuse to download weights (invariant 13).
        enabled: Whether the check runs at all.
    """

    model: str = "tiny.en"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5
    language: str | None = "en"
    local_files_only: bool = True
    enabled: bool = True

    @classmethod
    def from_config(cls, config: Config | None) -> ASRSettings:
        """Read ``detection.asr``; code defaults without a config."""
        if config is None:
            return cls()
        block = dict(config.detection.get("asr", {}))
        if not block:
            return cls()
        language = block.get("language", "en")
        return cls(
            model=str(block.get("model", "tiny.en")),
            device=str(block.get("device", "cpu")),
            compute_type=str(block.get("compute_type", "int8")),
            beam_size=int(block.get("beam_size", 5)),
            language=None if language in (None, "", "auto") else str(language),
            local_files_only=bool(block.get("local_files_only", True)),
            enabled=bool(block.get("enabled", True)),
        )

    @property
    def key(self) -> str:
        """Cache key component: everything that changes the output text."""
        return f"{self.model}|{self.device}|{self.compute_type}|{self.beam_size}|{self.language}"


@dataclass(frozen=True)
class Transcription:
    """What the recogniser heard.

    Attributes:
        text: The transcript, segments joined by single spaces.
        model: Model that produced it.
        duration: Audio duration the recogniser reported, in seconds.
        from_cache: Whether this came from the on-disk cache.
    """

    text: str
    model: str = ""
    duration: float = 0.0
    from_cache: bool = False


@dataclass(frozen=True)
class WordErrors:
    """A word error rate with the edits behind it.

    Attributes:
        wer: ``(S + D + I) / N``, unbounded above.
        substitutions: Substituted words.
        deletions: Reference words the recogniser did not produce.
        insertions: Words the recogniser added.
        n_reference: Reference word count, ``N``.
    """

    wer: float
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    n_reference: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON form, for the result's warnings and the report."""
        return {
            "wer": round(float(self.wer), 4),
            "substitutions": int(self.substitutions),
            "deletions": int(self.deletions),
            "insertions": int(self.insertions),
            "n_reference": int(self.n_reference),
        }


def normalise_tokens(text: str) -> list[str]:
    """Tokenise for comparison: lower-case, punctuation-free, numerals spelt out.

    The same normalisation the aligner applies to its ``.lab`` input, so "1863"
    against "eighteen sixty three" is not counted as three errors and a missing
    comma is not counted at all.
    """
    return list(normalise_for_alignment(text).lab_tokens)


def wer(reference: str, hypothesis: str) -> WordErrors:
    """Word error rate of ``hypothesis`` against ``reference`` (spec 8).

    A plain Levenshtein alignment over normalised tokens. It is implemented
    here rather than taken from a dependency because the normalisation has to
    be the aligner's own, and because the pipeline must not acquire a package
    just to count three integers.

    Args:
        reference: The supplied transcript.
        hypothesis: What the recogniser produced.

    Returns:
        A :class:`WordErrors`. An empty reference gives ``wer`` 0.0 for an
        empty hypothesis and 1.0 otherwise, rather than dividing by zero.
    """
    ref = normalise_tokens(reference)
    hyp = normalise_tokens(hypothesis)
    if not ref:
        return WordErrors(wer=0.0 if not hyp else 1.0, insertions=len(hyp), n_reference=0)

    # Edit distance with back-pointers; O(len(ref) * len(hyp)) is fine for a
    # five-minute recording and keeps the counts exact.
    n, m = len(ref), len(hyp)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    ops: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0], ops[i][0] = i, "D"
    for j in range(1, m + 1):
        cost[0][j], ops[0][j] = j, "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost[i][j], ops[i][j] = cost[i - 1][j - 1], "="
                continue
            choices = (
                (cost[i - 1][j - 1] + 1, "S"),
                (cost[i - 1][j] + 1, "D"),
                (cost[i][j - 1] + 1, "I"),
            )
            cost[i][j], ops[i][j] = min(choices, key=lambda pair: pair[0])

    counts = {"S": 0, "D": 0, "I": 0}
    i, j = n, m
    while i > 0 or j > 0:
        op = ops[i][j]
        if op in counts:
            counts[op] += 1
        if op in ("=", "S"):
            i, j = i - 1, j - 1
        elif op == "D":
            i -= 1
        else:
            j -= 1

    errors = counts["S"] + counts["D"] + counts["I"]
    return WordErrors(
        wer=errors / float(n),
        substitutions=counts["S"],
        deletions=counts["D"],
        insertions=counts["I"],
        n_reference=n,
    )


def _cache_path(cache_dir: Path, audio_sha256: str, settings: ASRSettings) -> Path:
    """Where one transcription is cached."""
    tag = settings.model.replace("/", "_").replace(".", "-")
    return cache_dir / f"{audio_sha256[:32]}.{tag}.json"


def transcribe(
    audio_path: Path | str,
    *,
    settings: ASRSettings | None = None,
    config: Config | None = None,
    cache_dir: Path | str | None = None,
    audio_sha256: str = "",
    refresh: bool = False,
) -> Transcription:
    """Transcribe one file with faster-whisper, caching by audio hash.

    Args:
        audio_path: The processed WAV.
        settings: Model settings; read from ``config`` when omitted.
        config: Configuration, for the settings and the cache directory.
        cache_dir: Override the cache directory. ``None`` with no config
            disables caching.
        audio_sha256: Hash of the file, if already known.
        refresh: Recompute and overwrite the cache entry.

    Returns:
        A :class:`Transcription`.

    Raises:
        FileNotFoundError: The audio does not exist.
        ASRNotAvailableError: ``faster_whisper`` is not importable, or the
            weights are absent and downloads are disabled.
        ASRError: The recogniser failed.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"audio does not exist: {path}")
    settings = settings or ASRSettings.from_config(config)

    directory: Path | None = None
    if cache_dir is not None:
        directory = Path(cache_dir)
    elif config is not None:
        directory = config.path("cache.asr")

    digest = audio_sha256 or sha256_file(path)
    entry = _cache_path(directory, digest, settings) if directory is not None else None
    if entry is not None and entry.is_file() and not refresh:
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
            if payload.get("key") == settings.key and payload.get("version") == CACHE_VERSION:
                return Transcription(
                    text=str(payload["text"]),
                    model=settings.model,
                    duration=float(payload.get("duration", 0.0)),
                    from_cache=True,
                )
        except (OSError, ValueError, KeyError) as exc:  # pragma: no cover - corrupt cache
            logger.warning("ignoring unreadable ASR cache %s: %s", entry, exc)

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ASRNotAvailableError(
            "faster-whisper is not installed; the transcript mismatch check cannot run"
        ) from exc

    try:
        model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
            local_files_only=settings.local_files_only,
        )
        segments, info = model.transcribe(
            str(path),
            beam_size=settings.beam_size,
            language=settings.language,
            # Determinism (invariant 13): a fixed beam, no sampling and no
            # carry-over of previous text, so two runs produce one string.
            temperature=0.0,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        duration = float(getattr(info, "duration", 0.0) or 0.0)
    except OSError as exc:
        raise ASRNotAvailableError(
            f"could not load the {settings.model!r} weights"
            f"{' (downloads are disabled)' if settings.local_files_only else ''}: {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - recogniser internals
        raise ASRError(f"faster-whisper failed on {path.name}: {exc}") from exc

    if entry is not None:
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "key": settings.key,
                    "audio_sha256": digest,
                    "text": text,
                    "duration": duration,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    logger.info("asr: %d character(s) from %s with %s", len(text), path.name, settings.model)
    return Transcription(text=text, model=settings.model, duration=duration)


def check_transcript(
    audio_path: Path | str,
    transcript: str,
    *,
    settings: ASRSettings | None = None,
    config: Config | None = None,
    audio_sha256: str = "",
) -> tuple[WordErrors, Transcription]:
    """Transcribe and score in one call; see :func:`transcribe` and :func:`wer`."""
    heard = transcribe(audio_path, settings=settings, config=config, audio_sha256=audio_sha256)
    return wer(transcript, heard.text), heard


def max_wer(config: Config | None) -> float:
    """The Reject threshold, ``detection.modes.reject.max_wer`` (spec 13)."""
    if config is None:
        return 0.25
    return float(config.detection["modes"]["reject"]["max_wer"])
