"""Speech-rate features (spec 9.1 RATE, 9.2 rate normalisation).

Rate is the feature most likely to be computed wrongly, in three ways this
module avoids:

* **Speaking rate is not articulation rate.** Counting pauses as speaking time
  makes a hesitant speaker look slow when their articulation is normal. Pauses
  longer than ``exclude_pauses_over_s`` are removed from the denominator, which
  is the fluency-research convention.
* **Syllables are vowel nuclei from the phone tier**, not orthographic guesses:
  the MFA ``phones`` tier gives ARPAbet vowels, one per syllable.
* **A rate needs a window.** A per-word rate over a 200 ms word is dominated by
  that word's own length, so the reported rate is computed over a +/- 3-word
  window (``configs/features.yaml``: ``local_window_words``).

As an independent witness, :func:`syllable_nuclei` ports the de Jong & Wempe
syllable-nuclei detector [R14] - intensity peaks above a relative threshold,
separated by a dip of at least 2 dB, at a voiced moment - and
:func:`nuclei_crosscheck` reports how its count compares with the aligner's.
It is reported, never substituted: the alignment is the measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from rostrum.features.duration import count_syllables
from rostrum.features.frames import FrameGrid

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "RATE_WORD_COLUMNS",
    "NucleiReport",
    "articulation_rate",
    "articulation_time",
    "count_syllables",
    "local_rate",
    "nuclei_crosscheck",
    "nuclei_rate",
    "relative_rate",
    "syllable_nuclei",
    "word_rate",
]

RATE_WORD_COLUMNS: tuple[str, ...] = (
    "rate_local_syll_per_s",
    "rate_relative",
    "rate_window_syllables",
    "rate_window_span_s",
)
"""Word-level rate columns (spec 9.1: local rate over a +/- 3-word window)."""

DEFAULT_EXCLUDE_PAUSES_OVER_S = 0.250
"""Gaps longer than this are pauses, not articulation time (spec 9.1)."""

DEFAULT_WINDOW_WORDS = 3
"""Words each side of the target word in the local-rate window."""

DEFAULT_MIN_DIP_DB = 2.0
"""de Jong & Wempe's minimum intensity dip between two syllable nuclei."""

DEFAULT_SILENCE_DB = 25.0
"""de Jong & Wempe's peak threshold, in dB below the recording's maximum."""


# --------------------------------------------------------------------------- #
# Rate from the alignment
# --------------------------------------------------------------------------- #


def _syllables(words: list[Word]) -> int:
    """Total vowel nuclei across ``words``."""
    return sum(w.n_syllables or count_syllables(list(w.phones)) for w in words)


def _articulation_time(
    words: list[Word], exclude_pauses_over_s: float = DEFAULT_EXCLUDE_PAUSES_OVER_S
) -> float:
    """Seconds spanned by ``words``, minus every gap longer than the threshold."""
    if not words:
        return 0.0
    ordered = sorted(words, key=lambda w: w.t_start)
    total = ordered[-1].t_end - ordered[0].t_start
    for prev, nxt in pairwise(ordered):
        gap = nxt.t_start - prev.t_end
        if gap > exclude_pauses_over_s:
            total -= gap
    return max(total, 0.0)


def articulation_time(
    words: list[Word], *, exclude_pauses_over_s: float = DEFAULT_EXCLUDE_PAUSES_OVER_S
) -> float:
    """Seconds of articulation spanned by ``words`` (pauses removed).

    The public form of the span used by :func:`articulation_rate` and
    :func:`local_rate`, so a consumer that needs the denominator - the rate
    cross-check in ``eval/validate_rate.py``, for instance - computes it the
    same way the feature does rather than approximating it.
    """
    return _articulation_time(words, exclude_pauses_over_s)


def articulation_rate(
    words: list[Word], *, exclude_pauses_over_s: float = DEFAULT_EXCLUDE_PAUSES_OVER_S
) -> float:
    """Syllables per second of articulation over the whole recording.

    Args:
        words: Aligned words.
        exclude_pauses_over_s: Gaps longer than this leave the denominator.

    Returns:
        Syllables per second, or ``NaN`` when there is no syllable or no time -
        never 0.0, which would read as "spoke infinitely slowly".
    """
    syllables = _syllables(words)
    seconds = _articulation_time(words, exclude_pauses_over_s)
    if syllables == 0 or seconds <= 0:
        return float("nan")
    return syllables / seconds


def local_rate(
    words: list[Word],
    idx: int,
    *,
    window_words: int = DEFAULT_WINDOW_WORDS,
    exclude_pauses_over_s: float = DEFAULT_EXCLUDE_PAUSES_OVER_S,
) -> float:
    """Articulation rate over the ``+/- window_words`` window around ``idx``.

    The window is clipped at the edges of the utterance rather than padded, so a
    rate near the start is computed from fewer words instead of from invented
    silence.
    """
    if not words:
        return float("nan")
    lo = max(0, idx - window_words)
    hi = min(len(words), idx + window_words + 1)
    window = words[lo:hi]
    syllables = _syllables(window)
    seconds = _articulation_time(window, exclude_pauses_over_s)
    if syllables == 0 or seconds <= 0:
        return float("nan")
    return syllables / seconds


def relative_rate(local: float, global_rate: float) -> float:
    """``local / global``: the speaker's own tempo divided out (spec 9.2).

    1.0 means "this stretch is at this speaker's normal rate", which is the
    comparable quantity across speakers; absolute syll/s is reported too.
    """
    if not np.isfinite(local) or not np.isfinite(global_rate) or global_rate <= 0:
        return float("nan")
    return float(local) / float(global_rate)


def word_rate(
    words: list[Word],
    *,
    config: Config | None = None,
    window_words: int | None = None,
    exclude_pauses_over_s: float | None = None,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """Local and relative rate for every word.

    The recording's articulation rate travels in ``attrs["articulation_rate"]``,
    so a consumer can report "1.6x this speaker's normal pace" without
    recomputing it.
    """
    import pandas as pd

    window, exclude = _rate_conf(config, window_words, exclude_pauses_over_s)
    global_rate = articulation_rate(words, exclude_pauses_over_s=exclude)

    rows = []
    for i, _word in enumerate(words):
        lo = max(0, i - window)
        hi = min(len(words), i + window + 1)
        local = local_rate(words, i, window_words=window, exclude_pauses_over_s=exclude)
        rows.append(
            {
                "rate_local_syll_per_s": local,
                "rate_relative": relative_rate(local, global_rate),
                "rate_window_syllables": _syllables(words[lo:hi]),
                "rate_window_span_s": _articulation_time(words[lo:hi], exclude),
            }
        )
    table = pd.DataFrame(rows, columns=list(RATE_WORD_COLUMNS))
    table.index = pd.Index([w.idx for w in words], name="unit_idx", dtype="int64")
    table.attrs.update(
        unit="word",
        file_id=file_id,
        channel_id=channel_id,
        articulation_rate=global_rate,
        window_words=window,
    )
    return table


def _rate_conf(
    config: Config | None, window_words: int | None, exclude_pauses_over_s: float | None
) -> tuple[int, float]:
    """``(window_words, exclude_pauses_over_s)`` from the config or defaults."""
    window, exclude = DEFAULT_WINDOW_WORDS, DEFAULT_EXCLUDE_PAUSES_OVER_S
    if config is not None:
        r = config.section("features").get("rate", {})
        window = int(r.get("local_window_words", window))
        exclude = float(r.get("exclude_pauses_over_s", exclude))
    if window_words is not None:
        window = int(window_words)
    if exclude_pauses_over_s is not None:
        exclude = float(exclude_pauses_over_s)
    return window, exclude


# --------------------------------------------------------------------------- #
# de Jong & Wempe syllable nuclei [R14]
# --------------------------------------------------------------------------- #


def syllable_nuclei(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    silence_db: float = DEFAULT_SILENCE_DB,
    min_dip_db: float = DEFAULT_MIN_DIP_DB,
    min_pitch_hz: float = 50.0,
    require_voiced: bool = True,
) -> list[float]:
    """Detect syllable nuclei the de Jong & Wempe way, in seconds [R14].

    The ported logic, in the original's order:

    1. take the intensity contour;
    2. keep intensity peaks above ``max_intensity - silence_db``;
    3. require a dip of at least ``min_dip_db`` between consecutive peaks, so a
       long steady vowel counts once;
    4. require the peak to fall at a voiced moment, so a loud fricative or a
       door slam is not a syllable.

    Args:
        samples: Mono audio.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        silence_db: Peak threshold, dB below the maximum intensity.
        min_dip_db: Minimum dip between two accepted peaks.
        min_pitch_hz: Pitch floor for both intensity and the voicing check.
        require_voiced: Apply step 4.

    Returns:
        Nucleus times in seconds, ascending.
    """
    from rostrum.features.energy import praat_intensity_db
    from rostrum.features.pitch import praat_pitch_hz

    grid = grid or FrameGrid(sample_rate=sr)
    db = praat_intensity_db(samples, sr, grid=grid, min_pitch_hz=min_pitch_hz)
    if db.size == 0 or not np.any(np.isfinite(db)):
        return []

    finite = np.isfinite(db)
    peak_db = float(np.nanmax(db))
    floor_db = float(np.nanmin(db[finite]))
    threshold = max(peak_db - float(silence_db), floor_db)
    curve = np.where(finite, db, floor_db)

    peaks = _local_maxima(curve)
    peaks = [i for i in peaks if curve[i] >= threshold]
    peaks = _enforce_dips(curve, peaks, min_dip_db=float(min_dip_db))

    times = grid.centres(curve.size)
    if require_voiced and peaks:
        f0 = praat_pitch_hz(
            samples, sr, floor_hz=min_pitch_hz, ceiling_hz=700.0, grid=grid, n_frames=curve.size
        )
        peaks = [i for i in peaks if np.isfinite(f0[i])]
    return [float(times[i]) for i in peaks]


def _local_maxima(curve: np.ndarray) -> list[int]:
    """Indices of local maxima, plateaux reported at their first frame."""
    if curve.size < 3:
        return [int(np.argmax(curve))] if curve.size else []
    out: list[int] = []
    for i in range(1, curve.size - 1):
        if curve[i] > curve[i - 1] and curve[i] >= curve[i + 1]:
            out.append(i)
    if not out:  # a monotonic contour still has one maximum
        out.append(int(np.argmax(curve)))
    return out


def _enforce_dips(curve: np.ndarray, peaks: list[int], *, min_dip_db: float) -> list[int]:
    """Keep peaks separated from the previous kept peak by a dip of ``min_dip_db``."""
    kept: list[int] = []
    for i in peaks:
        if not kept:
            kept.append(i)
            continue
        valley = float(np.min(curve[kept[-1] : i + 1]))
        if min(curve[kept[-1]], curve[i]) - valley >= min_dip_db:
            kept.append(i)
        elif curve[i] > curve[kept[-1]]:
            kept[-1] = i  # same syllable, better peak
    return kept


def nuclei_rate(nuclei: list[float], *, speech_duration_s: float) -> float:
    """Nuclei per second of speech, or ``NaN`` without speech time."""
    if speech_duration_s <= 0:
        return float("nan")
    return len(nuclei) / float(speech_duration_s)


@dataclass(frozen=True)
class NucleiReport:
    """Agreement between the aligner's syllables and the nuclei detector.

    Attributes:
        n_mfa_syllables: Vowel phones in the alignment.
        n_nuclei: Peaks the detector accepted.
        ratio: ``n_nuclei / n_mfa_syllables``, or ``None`` without syllables.
        correlation: Pearson correlation of the per-word counts, or ``None``
            when fewer than three words or either count is constant.
        per_word: ``(mfa, nuclei)`` counts per word, in word order.
    """

    n_mfa_syllables: int
    n_nuclei: int
    ratio: float | None
    correlation: float | None
    per_word: tuple[tuple[int, int], ...]


def nuclei_crosscheck(words: list[Word], nuclei: list[float]) -> NucleiReport:
    """Compare the aligner's syllable count with the detector's (spec 9.1 RATE).

    Reported in the technical report as evidence that the rate feature is not an
    artefact of one tool; never used to override the alignment.
    """
    times = np.asarray(sorted(nuclei), dtype=np.float64)
    per_word: list[tuple[int, int]] = []
    for w in words:
        mfa = w.n_syllables or count_syllables(list(w.phones))
        inside = int(np.sum((times >= w.t_start) & (times < w.t_end)))
        per_word.append((mfa, inside))

    n_mfa = _syllables(words)
    n_nuclei = int(times.size)
    ratio = (n_nuclei / n_mfa) if n_mfa else None

    correlation: float | None = None
    if len(per_word) >= 3:
        a = np.array([m for m, _ in per_word], dtype=np.float64)
        b = np.array([n for _, n in per_word], dtype=np.float64)
        if np.std(a) > 0 and np.std(b) > 0:
            correlation = float(np.corrcoef(a, b)[0, 1])
    return NucleiReport(
        n_mfa_syllables=n_mfa,
        n_nuclei=n_nuclei,
        ratio=ratio,
        correlation=correlation,
        per_word=tuple(per_word),
    )
