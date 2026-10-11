"""Audio excerpts for A/B playback: the participant, and a skilled reader (spec 15.2).

A region card makes a numeric claim - "you took 1.9 syll/s here where the
readers took 3.8". The two players under the card are how a judge checks that
claim with their ears, so the two excerpts must cover **the same words**, not
the same clock time: a participant who drags takes longer over "four score"
than any reference reader does, and cutting the reference at the participant's
timestamps would clip it mid-word.

So a reference excerpt is cut by *word index*:

.. code-block:: text

    participant seconds -> participant word indices  (this analysis's alignment)
                        -> reference word indices    (same text, same indices)
                        -> reference seconds         (that reader's TextGrid)

Which reader
------------
Spec 15.2 asks for the "closest reference reader", which can mean two
different things, so both are available and the default is stated plainly:

``representative`` (default)
    The reader closest to the envelope's own median tempo over those words -
    the most *typical* skilled reading. This is the default because the player
    exists to demonstrate the target, and a judge asking "what should this have
    sounded like?" wants the middle of the envelope, not its edge.

``nearest``
    The reader closest to **this participant** over those words. Use it to
    control for tempo: if the fastest reader is picked against a slow
    participant, every difference is audible and none of it is the flaw.

Both criteria are computed from word durations in the TextGrids alone, so
selecting a clip costs two file reads and no feature extraction, and neither
one can reach into the detection path.

Nothing here is fitted, cached or random: the same analysis and window always
pick the same reader, ties broken by ``file_id``.
"""

from __future__ import annotations

import io
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from rostrum.align import mfa as M
from rostrum.config import Config, load_config
from rostrum.io.audio import SAMPLE_RATE, preprocess, read_mono, trim_excerpt
from rostrum.io.hashing import sha256_bytes

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.types import Alignment, AnalysisResult

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PAD_S",
    "Clip",
    "ClipError",
    "ORIGINAL_STEM",
    "ClipStore",
    "Which",
    "choose_reference",
    "participant_clip",
    "reference_clip",
    "window_to_words",
]

Which = Literal["participant", "reference"]
Pick = Literal["representative", "nearest"]

DEFAULT_PAD_S: float = 0.15
"""Context added either side of a window, so a clip does not start mid-plosive."""

MAX_CLIP_S: float = 30.0
"""Longest excerpt served; a card is about a region, not about the speech."""

ORIGINAL_STEM = "__original__"
"""Reserved stem for the archived upload, so no user filename can collide."""


class ClipError(RuntimeError):
    """No excerpt could be produced, with the reason stated."""


# --------------------------------------------------------------------------- #
# The excerpt
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Clip:
    """One audio excerpt and where it came from.

    Attributes:
        which: ``participant`` or ``reference``.
        file_id: The recording the excerpt was cut from.
        t_start: Onset in that recording, seconds.
        t_end: Offset in that recording, seconds.
        samples: Mono float32 at ``sample_rate``.
        sample_rate: 16 kHz (invariant 1).
        words: The words the excerpt covers, for the player's caption.
        word_idx: ``(first, last)`` word indices covered, inclusive.
        note: One line of provenance for the dashboard to show.
    """

    which: Which
    file_id: str
    t_start: float
    t_end: float
    samples: np.ndarray
    sample_rate: int = SAMPLE_RATE
    words: str = ""
    word_idx: tuple[int, int] | None = None
    note: str = ""

    @property
    def duration(self) -> float:
        """Length of the excerpt in seconds."""
        return float(self.samples.size) / float(self.sample_rate)

    def to_wav_bytes(self) -> bytes:
        """16-bit PCM WAV bytes (invariant 1), ready to serve or to play."""
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(
            buffer,
            np.asarray(self.samples, dtype=np.float32),
            int(self.sample_rate),
            subtype="PCM_16",
            format="WAV",
        )
        return buffer.getvalue()

    def to_dict(self) -> dict[str, Any]:
        """JSON form, without the samples."""
        return {
            "which": self.which,
            "file_id": self.file_id,
            "t_start": round(float(self.t_start), 3),
            "t_end": round(float(self.t_end), 3),
            "duration": round(self.duration, 3),
            "sample_rate": int(self.sample_rate),
            "words": self.words,
            "word_idx": list(self.word_idx) if self.word_idx else None,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Where the uploads live
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClipStore:
    """The audio behind an analysis, kept so ``/clip`` can cut from it later.

    :func:`~rostrum.pipeline.analyze` preprocesses its input into a temporary
    directory and does not keep it, which is right for the analysis and wrong
    for the dashboard: a card drawn from a cached result still needs the audio
    to play. The store is that audio, keyed by ``analysis_id``, which is a hash
    of the bytes, the transcript and the config - so the same upload reuses the
    same directory and re-uploading is free.

    Both the original upload and the processed WAV are kept. The processed WAV
    is what every timestamp in the result refers to (invariant 2); the original
    is what lets the processed one be rebuilt if the cache was cleared.

    Attributes:
        root: Directory holding one subdirectory per analysis.
    """

    root: Path

    @classmethod
    def from_config(cls, config: Config | None = None) -> ClipStore:
        """The store under ``cache.dir/uploads``."""
        config = config or load_config()
        return cls(root=config.path("cache.dir") / "uploads")

    def directory(self, analysis_id: str) -> Path:
        """Where one analysis keeps its audio, created on demand."""
        safe = "".join(c for c in str(analysis_id) if c.isalnum())
        if not safe:
            raise ClipError("an analysis id is required")
        path = self.root / safe
        path.mkdir(parents=True, exist_ok=True)
        return path

    def put(
        self,
        analysis_id: str,
        *,
        audio: bytes,
        filename: str,
        transcript: str,
        text_id: str | None = None,
    ) -> Path:
        """Write one upload to the store and return its directory."""
        directory = self.directory(analysis_id)
        suffix = Path(filename).suffix.lower() or ".bin"
        # A reserved stem, so a user file genuinely called "upload.wav" cannot
        # collide with the processed WAV the analysis writes beside it.
        (directory / f"{ORIGINAL_STEM}{suffix}").write_bytes(audio)
        (directory / "transcript.txt").write_text(transcript, encoding="utf-8")
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "analysis_id": str(analysis_id),
                    "filename": str(filename),
                    "text_id": text_id,
                    "sha256": sha256_bytes(audio),
                    "upload": f"{ORIGINAL_STEM}{suffix}",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return directory

    def meta(self, analysis_id: str) -> dict[str, Any]:
        """What was stored with this analysis, or ``{}``."""
        path = self.directory(analysis_id) / "meta.json"
        if not path.is_file():
            return {}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt store
            return {}

    def processed_wav(self, analysis_id: str, *, file_id: str = "") -> Path:
        """The processed WAV every timestamp refers to, rebuilt if it is missing.

        Args:
            analysis_id: The analysis the audio belongs to.
            file_id: Preferred stem, when several WAVs are present.

        Returns:
            Path to a 16 kHz mono 16-bit WAV at the project invariants.

        Raises:
            ClipError: Neither a processed WAV nor the original upload is in
                the store, so there is nothing to cut.
        """
        directory = self.directory(analysis_id)

        # The analysis wrote its processed WAV here as `<file_id>.wav`; prefer
        # that exact name, then any other WAV in the directory. The upload
        # itself is excluded: it may be an MP3, or a WAV at the wrong rate, and
        # the result's timestamps refer to the *processed* signal.
        named = directory / f"{file_id}.wav"
        if file_id and named.is_file():
            return named
        others = sorted(p for p in directory.glob("*.wav") if p.stem != ORIGINAL_STEM)
        if others:
            return others[0]

        # Nothing processed survives, so rebuild it from the original upload.
        # preprocess is deterministic, so the rebuilt WAV is sample-identical
        # to the one the analysis measured.
        meta = self.meta(analysis_id)
        upload = directory / str(meta.get("upload", ""))
        if not upload.is_file():
            found = sorted(directory.glob(f"{ORIGINAL_STEM}.*"))
            upload = found[0] if found else upload
        if not upload.is_file():
            raise ClipError(
                f"no audio is stored for analysis {analysis_id}; re-run the analysis "
                "before asking for a clip"
            )
        target = directory / f"{file_id or 'participant'}.wav"
        processed, _ = preprocess(upload, target)
        return processed


# --------------------------------------------------------------------------- #
# Time window -> words -> time window
# --------------------------------------------------------------------------- #


def window_to_words(
    alignment: Alignment,
    t_start: float,
    t_end: float,
) -> tuple[int, int] | None:
    """The inclusive word-index span a time window overlaps.

    A window that falls entirely inside a pause overlaps no word; the nearest
    word on each side is used instead, so a ``missing_pause`` region - which
    is a gap, not a word - still resolves to something playable.

    Args:
        alignment: The recording's alignment.
        t_start: Window onset, seconds.
        t_end: Window offset, seconds.

    Returns:
        ``(first_idx, last_idx)``, or ``None`` when the alignment has no words.
    """
    words = sorted(alignment.words, key=lambda w: int(w.idx))
    if not words:
        return None
    lo, hi = (float(t_start), float(t_end)) if t_end >= t_start else (float(t_end), float(t_start))

    inside = [w for w in words if float(w.t_end) > lo and float(w.t_start) < hi]
    if inside:
        return int(inside[0].idx), int(inside[-1].idx)

    before = [w for w in words if float(w.t_end) <= lo]
    after = [w for w in words if float(w.t_start) >= hi]
    first = int(before[-1].idx) if before else int(after[0].idx)
    last = int(after[0].idx) if after else int(before[-1].idx)
    return min(first, last), max(first, last)


def _words_text(alignment: Alignment, span: tuple[int, int] | None) -> str:
    """The display text of a word-index span."""
    if span is None:
        return ""
    lo, hi = span
    return " ".join(
        (w.display or w.text)
        for w in sorted(alignment.words, key=lambda w: int(w.idx))
        if lo <= int(w.idx) <= hi
    )


def _span_seconds(
    alignment: Alignment,
    span: tuple[int, int],
) -> tuple[float, float] | None:
    """``(onset, offset)`` of a word-index span in one alignment."""
    lo, hi = span
    covered = [w for w in alignment.words if lo <= int(w.idx) <= hi]
    if not covered:
        return None
    return (
        min(float(w.t_start) for w in covered),
        max(float(w.t_end) for w in covered),
    )


def _durations(alignment: Alignment, span: tuple[int, int]) -> dict[int, float]:
    """Per-word durations over a span, dropping anything non-positive."""
    lo, hi = span
    out: dict[int, float] = {}
    for word in alignment.words:
        idx = int(word.idx)
        if lo <= idx <= hi:
            length = float(word.t_end) - float(word.t_start)
            if length > 0:
                out[idx] = length
    return out


def _log_distance(a: dict[int, float], b: dict[int, float]) -> float:
    """Mean ``|log a - log b|`` over the words both sides measured.

    Log duration, because tempo is multiplicative: a word taking twice as long
    is the same deviation whether the word is short or long.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return math.inf
    return float(np.mean([abs(math.log(a[i]) - math.log(b[i])) for i in shared]))


# --------------------------------------------------------------------------- #
# Cutting
# --------------------------------------------------------------------------- #


def _cut(
    wav: Path,
    t_start: float,
    t_end: float,
    *,
    pad_s: float,
    sr: int,
) -> tuple[np.ndarray, float, float]:
    """Read ``wav`` and return ``(samples, t_start, t_end)`` of a padded window."""
    samples, rate = read_mono(wav, sr=int(sr))
    duration = float(samples.size) / float(rate)
    lo = max(0.0, float(t_start) - float(pad_s))
    hi = min(duration, float(t_end) + float(pad_s))
    if hi <= lo:
        raise ClipError(
            f"the window {t_start:.3f}-{t_end:.3f} s lies outside {wav.name} "
            f"({duration:.3f} s long)"
        )
    hi = min(hi, lo + MAX_CLIP_S)
    return trim_excerpt(samples, int(rate), lo, hi), lo, hi


def participant_clip(
    wav: Path | str,
    t_start: float,
    t_end: float,
    *,
    alignment: Alignment | None = None,
    file_id: str = "",
    pad_s: float = DEFAULT_PAD_S,
    sr: int = SAMPLE_RATE,
) -> Clip:
    """Cut a window out of the analysed recording.

    Args:
        wav: The processed WAV the result's timestamps refer to.
        t_start: Window onset, seconds.
        t_end: Window offset, seconds.
        alignment: Supplies the words the window covers, for the caption.
        file_id: Identifier for the clip; the WAV's stem by default.
        pad_s: Context added either side.
        sr: Output sample rate.

    Returns:
        A :class:`Clip`.

    Raises:
        ClipError: The window lies outside the recording.
        FileNotFoundError: The WAV does not exist.
    """
    path = Path(wav)
    samples, lo, hi = _cut(path, t_start, t_end, pad_s=pad_s, sr=sr)
    span = window_to_words(alignment, t_start, t_end) if alignment is not None else None
    return Clip(
        which="participant",
        file_id=file_id or path.stem,
        t_start=round(lo, 3),
        t_end=round(hi, 3),
        samples=samples,
        sample_rate=int(sr),
        words=_words_text(alignment, span) if alignment is not None else "",
        word_idx=span,
        note="the uploaded recording, at the timestamps the analysis reports",
    )


# --------------------------------------------------------------------------- #
# Choosing a reference reader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Candidate:
    """One skilled reading that could be played against the participant."""

    file_id: str
    wav: Path
    alignment: Alignment
    durations: dict[int, float]


def _candidates(
    reference_ids: list[str],
    text_id: str | None,
    span: tuple[int, int],
    config: Config,
) -> list[_Candidate]:
    """Readers of this text whose audio *and* alignment are both on disk."""
    out: list[_Candidate] = []
    audio_dir = config.path("data.audio_clean")
    grid_dir = config.path("data.alignments")
    for file_id in sorted(set(reference_ids)):
        wav = audio_dir / f"{file_id}.wav"
        grid = grid_dir / f"{file_id}.TextGrid"
        if not wav.is_file() or not grid.is_file():
            continue
        try:
            alignment = M.read_textgrid(grid, file_id=file_id, text_id=text_id)
        except (OSError, ValueError) as exc:
            logger.warning("%s: unusable alignment for a clip (%s)", file_id, exc)
            continue
        durations = _durations(alignment, span)
        if not durations:
            continue
        out.append(_Candidate(file_id=file_id, wav=wav, alignment=alignment, durations=durations))
    return out


def choose_reference(
    result: AnalysisResult,
    span: tuple[int, int],
    *,
    pick: Pick = "representative",
    config: Config | None = None,
) -> tuple[str, Alignment, Path, str]:
    """Pick the reference reader to play against the participant.

    See the module docstring for what ``representative`` and ``nearest`` mean
    and why the default is the former.

    Args:
        result: The analysis; supplies ``reference_ids`` and the participant's
            own alignment.
        span: Inclusive word-index span to compare over.
        pick: Selection criterion.
        config: Supplies the dataset paths.

    Returns:
        ``(file_id, alignment, wav_path, note)`` - the note explains the choice
        in one line, for the dashboard to print under the player.

    Raises:
        ClipError: This analysis has no reference readers, or none of them has
            both audio and an alignment on disk.
    """
    config = config or load_config()
    if result.mode != "reference" or not result.reference_ids:
        raise ClipError(
            "this analysis has no reference readers to play: " f"it ran in {result.mode} mode"
        )

    pool = _candidates(list(result.reference_ids), result.text_id, span, config)
    if not pool:
        raise ClipError(
            f"none of the {len(result.reference_ids)} reference reading(s) for "
            f"{result.text_id!r} has both audio in {config.path('data.audio_clean')} and a "
            f"TextGrid in {config.path('data.alignments')}; run `make data` to build them"
        )

    if pick == "nearest" and result.alignment is not None:
        target = _durations(result.alignment, span)
        basis = "closest to this participant over these words"
    else:
        indices = sorted({i for c in pool for i in c.durations})
        target = {}
        for idx in indices:
            values = [c.durations[idx] for c in pool if idx in c.durations]
            if values:
                target[idx] = float(np.median(values))
        basis = "closest to the readers' median tempo over these words"

    if not target:
        chosen = pool[0]
        return (
            chosen.file_id,
            chosen.alignment,
            chosen.wav,
            f"{chosen.file_id}: first available reader (no tempo to compare over)",
        )

    ranked = sorted(pool, key=lambda c: (_log_distance(c.durations, target), c.file_id))
    chosen = ranked[0]
    distance = _log_distance(chosen.durations, target)
    return (
        chosen.file_id,
        chosen.alignment,
        chosen.wav,
        f"{chosen.file_id}, {basis} "
        f"({distance:.3f} mean |log duration| apart, of {len(pool)} reader(s))",
    )


def reference_clip(
    result: AnalysisResult,
    t_start: float,
    t_end: float,
    *,
    pick: Pick = "representative",
    pad_s: float = DEFAULT_PAD_S,
    sr: int = SAMPLE_RATE,
    config: Config | None = None,
) -> Clip:
    """Cut the same **words** out of a skilled reading of the same text.

    Args:
        result: The analysis whose window is being played.
        t_start: Window onset in the participant's recording, seconds.
        t_end: Window offset in the participant's recording, seconds.
        pick: Which reader to use; see the module docstring.
        pad_s: Context added either side.
        sr: Output sample rate.
        config: Supplies the dataset paths.

    Returns:
        A :class:`Clip` cut from the chosen reader at *their* timestamps for
        the words the window covers.

    Raises:
        ClipError: No reader is available, or the window maps to no word.
    """
    config = config or load_config()
    if result.alignment is None:
        raise ClipError("this analysis has no alignment, so a window maps to no words")

    span = window_to_words(result.alignment, t_start, t_end)
    if span is None:
        raise ClipError("this analysis aligned no words")

    file_id, alignment, wav, note = choose_reference(result, span, pick=pick, config=config)
    seconds = _span_seconds(alignment, span)
    if seconds is None:
        raise ClipError(
            f"{file_id} has no words at indices {span[0]}-{span[1]}; the two "
            "alignments do not describe the same text"
        )

    samples, lo, hi = _cut(wav, seconds[0], seconds[1], pad_s=pad_s, sr=sr)
    return Clip(
        which="reference",
        file_id=file_id,
        t_start=round(lo, 3),
        t_end=round(hi, 3),
        samples=samples,
        sample_rate=int(sr),
        words=_words_text(alignment, span),
        word_idx=span,
        note=note,
    )
