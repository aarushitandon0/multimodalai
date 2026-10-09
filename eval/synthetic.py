"""Synthetic feature tables with known flaw spans, for exercising 10.2-10.5.

The envelope and the detector consume *feature tables*, not audio. That makes a
synthetic corpus at the table level a genuine end-to-end exercise of spec
10.2-10.5: a text, several skilled readers who differ from one another by a
realistic jitter, and participants whose tables have a known perturbation over a
known word span. Everything the detector sees - canonical indices, gap labels,
phrase spans, frame curves - has the same shape it would have after MFA and
feature extraction.

What it is **not** is a substitute for the dataset. The numbers it produces say
that region formation recovers a deviation this generator put in; they say
nothing about whether the features recover a deviation a *speaker* put in. Only
the real S1 injected files answer that, and they need Montreal Forced Aligner
(``make data``). Any report built from this module must label it as synthetic.

The generator is seeded from ``configs/detection.yaml`` (``seed: 1337``) and
every reader's jitter derives from ``(seed, file_id)``, so two runs produce
identical tables.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover

    from rostrum.types import FeatureTable

logger = logging.getLogger(__name__)

__all__ = [
    "FLAW_DELTAS",
    "Flaw",
    "Recording",
    "build_corpus",
    "make_recording",
    "truth_events",
]


WORD_DURATION_S: float = 0.350
MINOR_GAP_S: float = 0.060
BOUNDARY_GAP_S: float = 0.520
PHRASE_WORDS: int = 6
HOP_S: float = 0.010

BASE: dict[str, float] = {
    "rate_local_syll_per_s": 4.60,
    "rate_relative": 1.00,
    "word_duration_s": WORD_DURATION_S,
    "duration_per_syllable_s": 0.175,
    "duration_rel_median": 1.00,
    "f0_mean_st": 0.60,
    "f0_max_st": 3.40,
    "f0_min_st": -2.60,
    "f0_range_st": 6.00,
    "f0_std_st": 1.90,
    "f0_slope_st_per_s": 0.00,
    "energy_mean_z": 0.00,
    "energy_peak_z": 0.90,
    "energy_mean_db_rel": 0.00,
    "energy_slope_db_per_s": 0.00,
    "prominence": 1.00,
    "prominence_peak_rel": 0.50,
}
"""Population means per word feature; a reader's own values jitter around these."""

JITTER: dict[str, float] = {
    "rate_local_syll_per_s": 0.26,
    "rate_relative": 0.06,
    "word_duration_s": 0.030,
    "duration_per_syllable_s": 0.014,
    "duration_rel_median": 0.07,
    "f0_mean_st": 0.45,
    "f0_max_st": 0.60,
    "f0_min_st": 0.55,
    "f0_range_st": 0.80,
    "f0_std_st": 0.26,
    "f0_slope_st_per_s": 0.70,
    "energy_mean_z": 0.24,
    "energy_peak_z": 0.26,
    "energy_mean_db_rel": 0.80,
    "energy_slope_db_per_s": 1.40,
    "prominence": 0.22,
    "prominence_peak_rel": 0.07,
}
"""Reader-to-reader standard deviation; the spread the envelope measures."""

GAP_JITTER_S: float = 0.035

STRESS_SHIFT: float = 1.80
"""Prominence moved off the accented word and onto its neighbour."""

FLAW_DELTAS: dict[str, dict[str, float]] = {
    # Each entry is {column: delta}, applied over the flaw's span and scaled by
    # the severity. The sizes are chosen to sit a few sigma out, which is what
    # an L2-L3 injection does to these same columns.
    "rushing": {"rate_local_syll_per_s": +1.60, "duration_per_syllable_s": -0.055},
    "dragging": {"rate_local_syll_per_s": -1.50, "duration_per_syllable_s": +0.060},
    "monotone": {"f0_range_st": -3.60, "f0_std_st": -1.10, "f0_max_st": -2.40},
    "low_energy": {"energy_mean_z": -1.30, "energy_mean_db_rel": -4.20},
    "fading": {"energy_slope_db_per_s": -7.50},
    # misplaced_stress is deliberately NOT a uniform prominence boost. The rule
    # is a within-phrase rank test (spec 10.3), and a block of words all raised
    # together leaves the ranking intact - correctly, because that is
    # over-emphasis, not misplacement. The injection therefore MOVES the accent:
    # it raises the first word of the span and suppresses the last, which is
    # what the PSOLA operation does to real audio (spec 5.2).
    "misplaced_stress": {},
    "mumbling": {
        "rate_local_syll_per_s": -0.80,
        "energy_mean_z": -0.80,
        "duration_per_syllable_s": -0.040,
        "cpps_db": -3.20,
        "spectral_tilt_db_per_khz": -2.60,
    },
}
"""Per-label feature perturbation, before the severity multiplier."""

CHANNEL_BASE: dict[str, float] = {
    "spectral_tilt_db_per_khz": -8.0,
    "spectral_centroid_hz": 1800.0,
    "spectral_flux": 0.40,
    "band_energy_ratio_db": 6.0,
    "hnr_db": 14.0,
    "jitter_local": 0.016,
    "shimmer_local": 0.075,
    "cpps_db": 15.0,
}
CHANNEL_JITTER: dict[str, float] = {
    "spectral_tilt_db_per_khz": 0.55,
    "spectral_centroid_hz": 70.0,
    "spectral_flux": 0.04,
    "band_energy_ratio_db": 0.60,
    "hnr_db": 0.90,
    "jitter_local": 0.0022,
    "shimmer_local": 0.008,
    "cpps_db": 0.85,
}
"""Channel-sensitive block, only used when a corpus is built same-channel."""


@dataclass(frozen=True)
class Flaw:
    """One perturbation to inject, with the span it occupies.

    Attributes:
        label: Flaw label, or ``"fading"`` for the ``low_energy`` variant.
        first_word: First word index affected, inclusive.
        last_word: Last word index affected, inclusive.
        severity: Multiplier on :data:`FLAW_DELTAS`; 1.0 is roughly an L3.
        gap_idx: For the two pause labels, the gap to perturb instead of words.
    """

    label: str
    first_word: int = 0
    last_word: int = 0
    severity: float = 1.0
    gap_idx: int | None = None

    @property
    def event_label(self) -> str:
        """The dataset label; ``fading`` scores as ``low_energy``."""
        return "low_energy" if self.label == "fading" else self.label


@dataclass
class Recording:
    """One synthetic recording: three unit tables, frame curves and the truth.

    Attributes:
        file_id: Recording id.
        text_id: Text id.
        channel_id: Channel tag, for the channel rule.
        tables: ``{unit: FeatureTable}``.
        frames: Frame curves with ``time_s``.
        word_spans: ``{word_idx: (t_start, t_end)}``.
        flaws: The flaws injected, empty for a clean reader.
    """

    file_id: str
    text_id: str
    channel_id: str
    tables: dict[str, FeatureTable] = field(default_factory=dict)
    frames: FeatureTable | None = None
    word_spans: dict[int, tuple[float, float]] = field(default_factory=dict)
    flaws: tuple[Flaw, ...] = ()


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _layout(
    n_words: int, gap_durations: Sequence[float]
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Lay words and gaps out on a timeline, returning both span lists."""
    words: list[tuple[float, float]] = []
    gaps: list[tuple[float, float]] = []
    t = 0.250  # a short run-in, as a processed WAV has
    for i in range(n_words):
        words.append((t, t + WORD_DURATION_S))
        t += WORD_DURATION_S
        if i < n_words - 1:
            gap = float(gap_durations[i])
            gaps.append((t, t + gap))
            t += gap
    return words, gaps


def _prominence_contour(text_id: str, n_words: int) -> np.ndarray:
    """A per-text prominence contour every reader of that text shares.

    Spread over 1.5 units so the ordering survives the per-reader jitter of
    :data:`JITTER`, with the phrase-final word carrying the nuclear accent.
    """
    rng = np.random.default_rng(zlib.crc32(f"contour:{text_id}".encode()))
    contour = rng.uniform(0.0, 1.5, size=n_words)
    for i in range(n_words):
        if (i + 1) % PHRASE_WORDS == 0:
            contour[i] += 1.2
    return contour


def _gap_label(i: int) -> str:
    """Boundary every :data:`PHRASE_WORDS` words, a comma halfway between."""
    if (i + 1) % PHRASE_WORDS == 0:
        return "boundary"
    if (i + 1) % PHRASE_WORDS == PHRASE_WORDS // 2:
        return "minor"
    return "none"


# --------------------------------------------------------------------------- #
# One recording
# --------------------------------------------------------------------------- #


def make_recording(
    file_id: str,
    text_id: str,
    *,
    n_words: int = 24,
    seed: int = 1337,
    flaws: Sequence[Flaw] = (),
    channel_id: str = "ch_a",
    with_channel_features: bool = False,
    reader_id: str | None = None,
) -> Recording:
    """Build one synthetic recording's tables, frames and ground truth.

    Args:
        file_id: Recording id; mixed into the RNG so each reader differs,
            unless ``reader_id`` overrides which identity the RNG keys on.
        text_id: Text id. Every recording of one text shares the word count and
            the gap labels, which is what makes word *i* comparable across
            readers (spec 10.1).
        n_words: Words in the text.
        seed: Master seed; the effective seed is ``(seed, reader_id or file_id)``.
        flaws: Perturbations to inject.
        channel_id: Channel tag.
        with_channel_features: Add the channel-sensitive block, needed for
            ``mumbling``.
        reader_id: Identity the reader's jitter is drawn from, when it must be
            shared by several recordings. Spec 5.4 builds the single-flaw files
            by editing *one* clean file, so an L1-L4 ladder has to be one
            reading with a graded edit rather than four different readers;
            passing the same ``reader_id`` with different ``file_id``s is how
            that is expressed here. Defaults to ``file_id``, which gives every
            recording its own reader.

    Returns:
        A :class:`Recording`.
    """
    import pandas as pd

    # crc32 rather than hash(): str.__hash__ is salted, so a PYTHONHASHSEED
    # change would silently alter every table (CLAUDE.md invariant 13).
    rng = np.random.default_rng(
        (int(seed) * 2_654_435_761 + zlib.crc32((reader_id or file_id).encode())) % 2**32
    )

    gap_labels = [_gap_label(i) for i in range(n_words - 1)]
    gap_base = [BOUNDARY_GAP_S if label == "boundary" else MINOR_GAP_S for label in gap_labels]
    gap_durations = [max(0.015, base + float(rng.normal(0.0, GAP_JITTER_S))) for base in gap_base]

    # Pause flaws change the timeline, so they are applied before the layout.
    for flaw in flaws:
        if flaw.label == "missing_pause" and flaw.gap_idx is not None:
            gap_durations[flaw.gap_idx] = 0.040
        elif flaw.label == "hesitation_pause" and flaw.gap_idx is not None:
            gap_durations[flaw.gap_idx] = 0.620

    word_spans_list, gap_spans = _layout(n_words, gap_durations)

    columns: dict[str, np.ndarray] = {
        name: BASE[name] + rng.normal(0.0, JITTER[name], size=n_words) for name in BASE
    }
    if with_channel_features:
        for name, base in CHANNEL_BASE.items():
            columns[name] = base + rng.normal(0.0, CHANNEL_JITTER[name], size=n_words)

    # A real reading has a prominence *contour*, and skilled readers of one text
    # largely agree on it. misplaced_stress is detected as a change in the
    # within-phrase prominence ORDERING (spec 10.3), so that ordering has to be
    # a stable property of the text rather than per-reader noise - otherwise
    # every reader disagrees with the envelope and the rule fires everywhere.
    # The contour is therefore derived from the text id, not from the reader.
    contour = _prominence_contour(text_id, n_words)
    columns["prominence"] = columns["prominence"] + contour

    for flaw in flaws:
        if flaw.gap_idx is not None:
            continue
        if flaw.label == "misplaced_stress":
            lo, hi = int(flaw.first_word), int(flaw.last_word)
            shift = STRESS_SHIFT * float(flaw.severity)
            if 0 <= lo < n_words:
                columns["prominence"][lo] += shift
            if 0 <= hi < n_words and hi != lo:
                columns["prominence"][hi] -= shift
            continue
        deltas = FLAW_DELTAS.get(flaw.label, {})
        span = range(int(flaw.first_word), int(flaw.last_word) + 1)
        for column, delta in deltas.items():
            if column not in columns:
                continue
            for i in span:
                if 0 <= i < n_words:
                    columns[column][i] += float(delta) * float(flaw.severity)

    words = pd.DataFrame(
        {
            "text": [f"w{i:02d}" for i in range(n_words)],
            "t_start": [s for s, _ in word_spans_list],
            "t_end": [e for _, e in word_spans_list],
            **columns,
        },
        index=pd.Index(range(n_words), name="unit_idx", dtype="int64"),
    )
    words.attrs.update(unit="word", file_id=file_id, channel_id=channel_id)

    gaps = pd.DataFrame(
        {
            "t_start": [s for s, _ in gap_spans],
            "t_end": [e for _, e in gap_spans],
            "gap_duration_s": [e - s for s, e in gap_spans],
            "gap_label": gap_labels,
            "gap_is_boundary": [label == "boundary" for label in gap_labels],
            "gap_prev_word_idx": list(range(n_words - 1)),
            "gap_next_word_idx": list(range(1, n_words)),
        },
        index=pd.Index(range(n_words - 1), name="unit_idx", dtype="int64"),
    )
    gaps.attrs.update(unit="gap", file_id=file_id, channel_id=channel_id)

    # Phrases run between boundary gaps (spec 9).
    bounds = [i for i, label in enumerate(gap_labels) if label == "boundary"]
    starts = [0] + [i + 1 for i in bounds]
    ends = [*bounds, n_words - 1]
    rows: list[dict[str, float]] = []
    for first, last in zip(starts, ends, strict=True):
        if first > last:
            continue
        inside = words.loc[first:last]
        rows.append(
            {
                "t_start": float(inside["t_start"].min()),
                "t_end": float(inside["t_end"].max()),
                "first_word_idx": int(first),
                "last_word_idx": int(last),
                "phrase_duration_s": float(inside["t_end"].max() - inside["t_start"].min()),
                "phrase_f0_max_st": float(inside["f0_max_st"].max()),
                "phrase_f0_range_st": float(inside["f0_range_st"].mean()),
                "phrase_rate_syll_per_s": float(inside["rate_local_syll_per_s"].mean()),
            }
        )
    phrases = pd.DataFrame(rows)
    phrases.index = pd.Index(range(len(rows)), name="unit_idx", dtype="int64")
    phrases.attrs.update(unit="phrase", file_id=file_id, channel_id=channel_id)

    frames = _frames(words, gaps, columns, n_words)
    frames.attrs.update(unit="frame", file_id=file_id, channel_id=channel_id, hop_s=HOP_S)

    return Recording(
        file_id=file_id,
        text_id=text_id,
        channel_id=channel_id,
        tables={"word": words, "gap": gaps, "phrase": phrases},
        frames=frames,
        word_spans={i: (s, e) for i, (s, e) in enumerate(word_spans_list)},
        flaws=tuple(flaws),
    )


def _frames(
    words: FeatureTable, gaps: FeatureTable, columns: Mapping[str, np.ndarray], n_words: int
) -> FeatureTable:
    """Frame curves consistent with the per-word values, on the 10 ms grid.

    Each word's frames hold that word's own value plus a little frame noise, and
    the gaps hold silence-like values, so a change point exists at every word
    and gap edge for the refinement to find. The curve is *derived* from the
    word table rather than generated independently, which is what makes a
    refined boundary land near the word boundary it belongs to.
    """
    import pandas as pd

    total = float(words["t_end"].max()) + 0.250
    n = int(np.ceil(total / HOP_S))
    time = np.arange(n, dtype=np.float64) * HOP_S
    rng = np.random.default_rng(7)

    f0 = np.full(n, np.nan)
    energy = np.full(n, np.nan)
    energy_rel = np.full(n, -45.0)
    prominence_response = np.zeros(n)
    prominence_composite = np.zeros(n)
    tilt = np.full(n, np.nan)
    voiced = np.zeros(n, dtype=np.int8)

    for i in range(n_words):
        t0, t1 = float(words.loc[i, "t_start"]), float(words.loc[i, "t_end"])
        idx = np.flatnonzero((time >= t0) & (time < t1))
        if idx.size == 0:
            continue
        noise = rng.normal(0.0, 0.12, size=idx.size)
        f0[idx] = float(columns["f0_mean_st"][i]) + noise * float(columns["f0_std_st"][i])
        energy[idx] = float(columns["energy_mean_z"][i]) + noise * 0.3
        energy_rel[idx] = float(columns["energy_mean_db_rel"][i]) + noise
        prominence_response[idx] = float(columns["prominence"][i])
        prominence_composite[idx] = (
            float(columns["prominence"][i]) + float(columns["rate_local_syll_per_s"][i]) / 5.0
        )
        voiced[idx] = 1
        if "spectral_tilt_db_per_khz" in columns:
            tilt[idx] = float(columns["spectral_tilt_db_per_khz"][i]) + noise

    frames = pd.DataFrame(
        {
            "time_s": time,
            "f0_st": f0,
            "voiced": voiced,
            "energy_z": energy,
            "energy_db_rel": energy_rel,
            "prominence_response": prominence_response,
            "prominence_composite": prominence_composite,
            "spectral_tilt_db_per_khz": tilt,
        }
    )
    frames.index = pd.Index(range(n), name="frame_idx", dtype="int64")
    return frames


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


def truth_events(recording: Recording) -> list[Any]:
    """Ground-truth :class:`~rostrum.detect.tune.Event` list for a recording.

    Spans come from the recording's own timeline: word-based flaws span the
    first affected word's onset to the last affected word's offset, and pause
    flaws span the gap. These are the same conventions the dataset label JSON
    uses (spec 7.1), which is why the same event-F1 code scores both.
    """
    from rostrum.detect.tune import Event

    out: list[Event] = []
    words = recording.tables["word"]
    gaps = recording.tables["gap"]
    for flaw in recording.flaws:
        if flaw.gap_idx is not None:
            row = gaps.loc[int(flaw.gap_idx)]
            out.append(
                Event(
                    flaw.event_label,
                    float(row["t_start"]),
                    float(row["t_end"]),
                    file_id=recording.file_id,
                )
            )
            continue
        out.append(
            Event(
                flaw.event_label,
                float(words.loc[int(flaw.first_word), "t_start"]),
                float(words.loc[int(flaw.last_word), "t_end"]),
                file_id=recording.file_id,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# A corpus
# --------------------------------------------------------------------------- #


def build_corpus(
    *,
    texts: Sequence[str] = ("syn_a", "syn_b", "syn_c"),
    n_readers: int = 6,
    n_words: int = 24,
    seed: int = 1337,
    participants_per_text: Sequence[Sequence[Flaw]] | None = None,
    with_channel_features: bool = True,
) -> tuple[dict[str, list[Recording]], dict[str, list[Recording]]]:
    """Build clean readers and flawed participants for several texts.

    Args:
        texts: Text ids.
        n_readers: Clean readers per text; they form the envelope.
        n_words: Words per text.
        seed: Master seed.
        participants_per_text: One flaw list per participant. ``None`` uses a
            default set covering seven of the eight labels - every label except
            the human-only ``mumbling``, which is added when
            ``with_channel_features`` is set.
        with_channel_features: Add the channel-sensitive block throughout, so
            the same-channel path (and ``mumbling``) can be exercised.

    Returns:
        ``(references, participants)``, each ``{text_id: [Recording, ...]}``.
    """
    if participants_per_text is None:
        participants_per_text = [
            [Flaw("rushing", 7, 11)],
            [Flaw("dragging", 13, 16)],
            [Flaw("monotone", 6, 11)],
            [Flaw("low_energy", 12, 17)],
            [Flaw("missing_pause", gap_idx=5)],
            [Flaw("hesitation_pause", gap_idx=13)],
            [Flaw("misplaced_stress", 4, 5)],
        ]
        if with_channel_features:
            participants_per_text = [*participants_per_text, [Flaw("mumbling", 8, 13)]]

    references: dict[str, list[Recording]] = {}
    participants: dict[str, list[Recording]] = {}

    for text_id in texts:
        references[text_id] = [
            make_recording(
                f"{text_id}_ref{r:02d}",
                text_id,
                n_words=n_words,
                seed=seed,
                with_channel_features=with_channel_features,
            )
            for r in range(n_readers)
        ]
        participants[text_id] = [
            make_recording(
                f"{text_id}_p{p:02d}_{'_'.join(f.label for f in flaws)}",
                text_id,
                n_words=n_words,
                seed=seed + 1000 + p,
                flaws=flaws,
                with_channel_features=with_channel_features,
            )
            for p, flaws in enumerate(participants_per_text)
        ]

    logger.info(
        "synthetic corpus: %d text(s) x (%d reader(s) + %d participant(s)), %d words",
        len(texts),
        n_readers,
        len(participants_per_text),
        n_words,
    )
    return references, participants
