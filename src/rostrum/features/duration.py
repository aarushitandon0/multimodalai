"""Duration features from the forced alignment (spec 9.1 DUR).

Word duration on its own confounds two things: how long the word *is* (a
monosyllable against a five-syllable word) and how long the speaker *took*.
Dividing by the phone count and by the syllable count separates them, and the
ratio to the recording's median word duration makes a word comparable to its
counterpart in another rendition of the same text.

Pure functions over :class:`~rostrum.types.Word`; no audio is read here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from rostrum.align.mfa import ARPA_VOWELS

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "DURATION_WORD_COLUMNS",
    "count_syllables",
    "duration_per_phone",
    "duration_per_syllable",
    "word_duration",
    "word_duration_features",
]

DURATION_WORD_COLUMNS: tuple[str, ...] = (
    "word_duration_s",
    "duration_per_phone_s",
    "duration_per_syllable_s",
    "duration_rel_median",
    "n_phones",
    "n_syllables",
)
"""Word-level duration columns (spec 9.1: duration, duration / phone count)."""


def count_syllables(phones: list[str]) -> int:
    """Count vowel nuclei in an ARPAbet phone sequence.

    Stress digits are stripped, so ``AH0`` and ``AH1`` both count. Silence and
    ``spn`` tokens carry no vowel and are ignored.
    """
    return sum(1 for p in phones if p.rstrip("012").upper() in ARPA_VOWELS)


def word_duration(word: Word) -> float:
    """Word duration in seconds."""
    return float(word.t_end - word.t_start)


def duration_per_phone(word: Word) -> float:
    """Mean seconds per phone, or ``NaN`` when the word has no phones.

    A word with no phone intervals is not a fast word - it is an unaligned one,
    and must not enter a rate statistic as a small number.
    """
    n = len(word.phones)
    return word_duration(word) / n if n else float("nan")


def duration_per_syllable(word: Word) -> float:
    """Mean seconds per syllable, or ``NaN`` when no vowel was aligned."""
    n = word.n_syllables or count_syllables(list(word.phones))
    return word_duration(word) / n if n else float("nan")


def word_duration_features(
    words: list[Word], *, file_id: str = "", channel_id: str = ""
) -> FeatureTable:
    """Aggregate durations to one row per word.

    ``duration_rel_median`` divides each word by the recording's median word
    duration, which removes the speaker's overall tempo and leaves the local
    lengthening or shortening that ``dragging`` and ``rushing`` consist of.
    """
    import pandas as pd

    rows = []
    durations = np.array([word_duration(w) for w in words], dtype=np.float64)
    positive = durations[durations > 0]
    median = float(np.median(positive)) if positive.size else float("nan")

    for word, dur in zip(words, durations, strict=True):
        n_phones = len(word.phones)
        n_syllables = word.n_syllables or count_syllables(list(word.phones))
        rows.append(
            {
                "word_duration_s": float(dur),
                "duration_per_phone_s": duration_per_phone(word),
                "duration_per_syllable_s": duration_per_syllable(word),
                "duration_rel_median": float(dur / median) if median > 0 else float("nan"),
                "n_phones": n_phones,
                "n_syllables": n_syllables,
            }
        )
    table = pd.DataFrame(rows, columns=list(DURATION_WORD_COLUMNS))
    table.index = pd.Index([w.idx for w in words], name="unit_idx", dtype="int64")
    table.attrs.update(unit="word", file_id=file_id, channel_id=channel_id)
    return table
