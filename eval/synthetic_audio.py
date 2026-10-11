"""Synthetic audio with a known alignment, for pipeline tests (spec 13).

:mod:`eval.synthetic` generates *feature tables*, which is the right level for
testing the envelope, the detector and the rubric. The pipeline needs a layer
below that: a WAV on disk, a transcript, and an alignment that really describes
that WAV, so that preprocessing, Praat pitch extraction, the frame grid, the
feature cache, the curves and the cards can all be exercised end to end.

Forced alignment is the one stage that cannot be faked usefully, and MFA needs
its own conda environment (spec 8), which CI and most laptops do not have.
These recordings therefore come **with** their alignment: the generator knows
exactly where each word starts and ends because it put it there, and
:func:`rostrum.pipeline.analyze` accepts that alignment instead of calling MFA.

What the signal is: one voiced burst per word - a sine at the word's F0 with a
little vibrato and a raised-cosine amplitude envelope - separated by low-level
room tone. It is not speech, and nothing here measures how the system performs
on speech; it is a signal whose rate, pitch, loudness and pauses are *known*,
which is what a pipeline test needs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.features.extract_all import FeatureBundle
    from rostrum.pipeline import ReferenceLibrary
    from rostrum.types import FeatureTable

from rostrum.features.pauses import gaps_from_words
from rostrum.io.audio import write_wav16
from rostrum.types import Alignment, Word

logger = logging.getLogger(__name__)

__all__ = [
    "DEMO_TEXT",
    "OTHER_TEXT",
    "SAMPLE_RATE",
    "SpokenText",
    "Utterance",
    "demo_library",
    "make_utterance",
    "parse_text",
    "tables_for",
]

SAMPLE_RATE: int = 16000
"""The project's working rate (CLAUDE.md invariant 1)."""

DEMO_TEXT: str = (
    "We can not dedicate, we can not consecrate. "
    "We can not hallow this ground, we can not forget."
)
"""18 words, four clauses, one anaphora - enough text for every rule to fire."""

OTHER_TEXT: str = (
    "The world will little note what we say here, but it can never forget what they did."
)
"""A second text, for the Norm-mode example: no library envelope exists for it."""

_BOUNDARY_PAUSE_S: float = 0.52
_MINOR_PAUSE_S: float = 0.24
_NONE_PAUSE_S: float = 0.03
_ROOM_TONE_DBFS: float = -62.0


@dataclass(frozen=True)
class SpokenText:
    """A transcript split into words with their syllable counts.

    Attributes:
        display: The transcript as written.
        words: Display form per word.
        syllables: Syllables per word, counted by vowel groups.
        pause_after: Seconds of silence the *skilled* reading leaves after each
            word, from the punctuation.
    """

    display: str
    words: tuple[str, ...]
    syllables: tuple[int, ...]
    pause_after: tuple[float, ...]


def _syllables(word: str) -> int:
    """Vowel-group count, floored at one. Enough for a synthetic rate."""
    letters = [c for c in word.lower() if c.isalpha()]
    count, previous_vowel = 0, False
    for char in letters:
        vowel = char in "aeiouy"
        if vowel and not previous_vowel:
            count += 1
        previous_vowel = vowel
    return max(1, count)


def parse_text(display: str) -> SpokenText:
    """Split a transcript into words, syllables and the pauses its text invites."""
    from rostrum.text.gaps import gap_labels, tokenise_display

    tokens = [t for t in tokenise_display(display) if t.word]
    labels = [label for label, _ in gap_labels(display)]
    pauses = []
    for i in range(len(tokens)):
        label = labels[i] if i < len(labels) else "boundary"
        pauses.append(
            {"boundary": _BOUNDARY_PAUSE_S, "minor": _MINOR_PAUSE_S}.get(label, _NONE_PAUSE_S)
        )
    return SpokenText(
        display=display,
        words=tuple(t.word for t in tokens),
        syllables=tuple(_syllables(t.word) for t in tokens),
        pause_after=tuple(pauses),
    )


@dataclass
class Utterance:
    """One generated recording.

    Attributes:
        file_id: Identifier.
        path: The WAV written to disk.
        alignment: Its true alignment, as the generator laid it down.
        samples: The signal.
        text: The transcript it reads.
        word_rate: Syllables per second per word, as generated.
    """

    file_id: str
    path: Path
    alignment: Alignment
    samples: np.ndarray
    text: SpokenText
    word_rate: list[float] = field(default_factory=list)


def make_utterance(
    file_id: str,
    directory: Path | str,
    *,
    text: SpokenText | str = DEMO_TEXT,
    seed: int = 0,
    rate_syll_s: float = 4.0,
    f0_hz: float = 120.0,
    f0_range_st: float = 5.0,
    level_db: float = -20.0,
    rate_scale: Sequence[float] | None = None,
    level_scale: Sequence[float] | None = None,
    pause_scale: Sequence[float] | None = None,
    monotone: bool = False,
    sample_rate: int = SAMPLE_RATE,
) -> Utterance:
    """Generate one recording and the alignment that describes it.

    Args:
        file_id: Identifier; also the WAV's stem.
        directory: Where to write the WAV.
        text: The transcript, or a parsed :class:`SpokenText`.
        seed: Per-reader jitter seed. Two readers with different seeds differ
            the way skilled readers differ: a little, everywhere.
        rate_syll_s: Base speaking rate.
        f0_hz: Base pitch.
        f0_range_st: Peak-to-peak pitch movement across a clause, in semitones.
        level_db: Base level, dBFS, before loudness normalisation.
        rate_scale: Per-word multiplier on the local rate - ``2.0`` at word 5
            is a word read twice as fast. ``None`` is a uniform reading.
        level_scale: Per-word multiplier on amplitude, in dB.
        pause_scale: Per-word multiplier on the pause *after* that word, so
            ``0.05`` at a boundary is a cut pause and ``10.0`` mid-clause is a
            hesitation.
        monotone: Flatten the pitch contour to one note.
        sample_rate: Output rate.

    Returns:
        An :class:`Utterance`.
    """
    spoken = parse_text(text) if isinstance(text, str) else text
    rng = np.random.default_rng(seed)
    n = len(spoken.words)
    rate_scale = list(rate_scale or [1.0] * n)
    level_scale = list(level_scale or [0.0] * n)
    pause_scale = list(pause_scale or [1.0] * n)

    pieces: list[np.ndarray] = []
    words: list[Word] = []
    rates: list[float] = []
    cursor = 0.0

    # A little leading room tone: the preprocessor keeps 200 ms and the frame
    # grid needs something before the first word.
    pieces.append(_room_tone(0.3, sample_rate, rng))
    cursor += 0.3

    for i, word in enumerate(spoken.words):
        syllables = spoken.syllables[i]
        rate = rate_syll_s * float(rate_scale[i]) * float(rng.normal(1.0, 0.04))
        duration = max(0.08, syllables / max(rate, 0.5))
        # Pitch: a falling contour across each clause, so a phrase has range.
        position = (i % 6) / 5.0
        semitones = 0.0 if monotone else f0_range_st * (0.5 - position)
        f0 = f0_hz * 2.0 ** (semitones / 12.0) * float(rng.normal(1.0, 0.01))
        amplitude = 10.0 ** ((level_db + float(level_scale[i])) / 20.0)
        pieces.append(_voiced(duration, f0, amplitude, sample_rate, rng))
        words.append(
            Word(
                idx=i,
                text=word.lower(),
                t_start=round(cursor, 6),
                t_end=round(cursor + duration, 6),
                # Two phones per syllable is enough for every consumer that
                # looks at the phone list; the syllable count is explicit.
                phones=[f"P{j}" for j in range(2 * syllables)],
                n_syllables=int(syllables),
                display=word,
            )
        )
        rates.append(syllables / duration)
        cursor += duration

        # Readers vary their pauses as much as they vary their rate; without
        # that jitter every reading has the identical silence and the gap
        # envelope has no spread to standardise against.
        pause = spoken.pause_after[i] * float(pause_scale[i]) * float(rng.normal(1.0, 0.12))
        if i == n - 1:
            pause = max(pause, 0.3)
        if pause > 0:
            pieces.append(_room_tone(pause, sample_rate, rng))
            cursor += pause

    samples = np.concatenate(pieces).astype(np.float32)
    path = Path(directory) / f"{file_id}.wav"
    write_wav16(path, samples, sample_rate)

    alignment = Alignment(
        file_id=file_id,
        words=words,
        gaps=gaps_from_words(words),
        duration=len(samples) / float(sample_rate),
        sample_rate=sample_rate,
        aligner="eval.synthetic_audio (generated, not aligned)",
    )
    logger.debug("%s: %d words, %.2f s", file_id, len(words), alignment.duration)
    return Utterance(
        file_id=file_id,
        path=path,
        alignment=alignment,
        samples=samples,
        text=spoken,
        word_rate=rates,
    )


def _voiced(
    duration: float, f0: float, amplitude: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """One voiced burst: a sine with vibrato under a raised-cosine envelope."""
    n = max(2, round(duration * sr))
    t = np.arange(n, dtype=np.float64) / sr
    vibrato = 1.0 + 0.01 * np.sin(2.0 * np.pi * 5.0 * t)
    phase = 2.0 * np.pi * f0 * np.cumsum(vibrato) / sr
    # A couple of harmonics so the spectrum is not a single line; Praat's pitch
    # tracker and the spectral features both want more than a pure tone.
    wave = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    wave /= np.abs(wave).max() or 1.0
    ramp = max(2, int(0.01 * sr))
    window = np.ones(n)
    fade = 0.5 * (1 - np.cos(np.linspace(0.0, np.pi, ramp)))
    window[:ramp] = fade
    window[-ramp:] = fade[::-1]
    noise = rng.normal(0.0, 0.004, n)
    return (amplitude * (wave * window + noise)).astype(np.float64)


def _room_tone(duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Low-level noise, never digital silence (CLAUDE.md invariant 8)."""
    n = max(1, round(duration * sr))
    return rng.normal(0.0, 10.0 ** (_ROOM_TONE_DBFS / 20.0), n)


# --------------------------------------------------------------------------- #
# A whole library of generated readings
# --------------------------------------------------------------------------- #


def tables_for(
    utterance: Utterance,
    *,
    config: Config,
    work: Path,
    apply_gap_labels: bool = True,
) -> tuple[dict[str, FeatureTable], FeatureBundle]:
    """Preprocess one generated recording and extract its features.

    The audio goes through the same :func:`rostrum.io.audio.preprocess` chain
    as an upload, so the reference tables and the participant's are measured on
    equally normalised audio (CLAUDE.md invariant 1). The generated alignment
    stays valid because the chain shifts no time here: the room tone is above
    the trim threshold, so nothing is cut from the head.

    Args:
        utterance: The generated recording.
        config: Configuration.
        work: Directory for the processed WAV.
        apply_gap_labels: Label the gaps from the transcript's punctuation, as
            the pipeline does, so phrases and pause rules see the same text
            expectation on both sides of the comparison.

    Returns:
        ``({unit: FeatureTable}, FeatureBundle)``.
    """
    from rostrum.features import extract_all as X
    from rostrum.features.pauses import apply_gap_labels as label_gaps
    from rostrum.io.audio import preprocess, read_mono
    from rostrum.io.hashing import sha256_file
    from rostrum.text.gaps import gap_label_map

    processed, _ = preprocess(utterance.path, Path(work) / f"{utterance.file_id}_16k.wav")
    digest = sha256_file(processed)
    if apply_gap_labels:
        utterance.alignment.gaps = label_gaps(
            utterance.alignment.gaps,
            gap_label_map(
                utterance.text.display,
                n_gaps=len(utterance.alignment.gaps),
                config=config,
            ),
        )
    samples, sr = read_mono(processed, sr=config.sample_rate)
    bundle, _ = X.extract_all_cached(
        np.asarray(samples, dtype=np.float32),
        int(sr),
        utterance.alignment,
        config=config,
        audio_sha256=digest,
        crosscheck=False,
    )
    return {unit: bundle.table(unit) for unit in ("word", "gap", "phrase")}, bundle


def demo_library(
    config: Config,
    work: Path | str,
    *,
    text_id: str = "demo_anaphora",
    text: str = DEMO_TEXT,
    n_readers: int = 5,
    fold: str = "demo",
) -> tuple[ReferenceLibrary, list[Utterance]]:
    """Build a :class:`~rostrum.pipeline.ReferenceLibrary` from generated readings.

    Every skilled reader reads the same text at a slightly different rate,
    pitch and level, which is exactly what an envelope is a model of. The
    readers all come from this generator, so they share a channel; the
    envelope therefore permits the channel-sensitive rules, and the demo says
    so rather than pretending the comparison is cross-speaker.

    Args:
        config: Configuration.
        work: Directory for the generated WAVs.
        text_id: Identifier for the generated text.
        text: The transcript every reader reads.
        n_readers: Readings in the envelope. Reference mode needs at least
            ``detection.modes.reference.min_references``.
        fold: Fold tag recorded on the envelopes.

    Returns:
        ``(library, utterances)`` - the library and the generated readings.
    """
    from rostrum.envelope import norms as NO
    from rostrum.envelope.build import Reference, build_fold_envelopes
    from rostrum.pipeline import LibraryText, ReferenceLibrary

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    spoken = parse_text(text)

    utterances: list[Utterance] = []
    references: list[Reference] = []
    reference_tables: list[dict[str, Any]] = []
    frames: dict[str, Any] = {}
    word_spans: dict[str, dict[int, tuple[float, float]]] = {}
    for reader in range(int(n_readers)):
        utterance = make_utterance(
            f"{text_id}_r{reader:02d}",
            work,
            text=spoken,
            seed=1000 + reader,
            rate_syll_s=4.0 + 0.12 * reader,
            f0_hz=112.0 + 4.0 * reader,
            level_db=-20.0 + 0.5 * reader,
        )
        tables, bundle = tables_for(utterance, config=config, work=work)
        utterances.append(utterance)
        reference_tables.append(dict(tables))
        references.append(
            Reference(
                file_id=utterance.file_id,
                channel_id="generated",
                tables=tables,
            )
        )
        frames[utterance.file_id] = bundle.frames
        word_spans[utterance.file_id] = {
            int(w.idx): (float(w.t_start), float(w.t_end)) for w in utterance.alignment.words
        }
        logger.debug("reference %s: %s", utterance.file_id, bundle.summary())

    envelopes = build_fold_envelopes(
        {text_id: [r.file_id for r in references]},
        references,
        fold=fold,
        train_texts=[text_id],
        config=config,
        # With the frame curves the envelope also carries the median +/- sigma
        # band the dashboard draws behind the participant (spec 15.2).
        frames=frames,
        word_spans=word_spans,
    )
    pooled = NO.build_pooled_norms(references, config=config)

    library = ReferenceLibrary(
        fold=fold,
        texts={
            text_id: LibraryText(
                text_id=text_id,
                display=spoken.display,
                verbatim=spoken.display,
                title="Generated anaphora",
                reference_ids=tuple(r.file_id for r in references),
            )
        },
        envelopes=envelopes,
        reference_tables={text_id: reference_tables},
        pooled=pooled,
        classifier=None,
        warnings=[
            "Demo library: the readings are generated signals, not speech, so every "
            "number below exercises the pipeline and none of them is a dataset result."
        ],
    )
    return library, utterances
