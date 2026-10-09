"""Speech-rate features (spec 9.1 RATE, 9.2 rate normalisation)."""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import rate as R
from rostrum.features.frames import FrameGrid
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()


def word(idx: int, t0: float, t1: float, phones: list[str]) -> Word:
    return Word(
        idx=idx,
        text=f"w{idx}",
        t_start=t0,
        t_end=t1,
        phones=phones,
        n_syllables=R.count_syllables(phones),
    )


def even_words(n: int, *, duration: float = 0.2, gap: float = 0.0) -> list[Word]:
    """``n`` one-syllable words of ``duration`` seconds, separated by ``gap``."""
    out = []
    t = 0.0
    for i in range(n):
        out.append(word(i, t, t + duration, ["D", "AH1", "T"]))
        t += duration + gap
    return out


# --------------------------------------------------------------------------- #
# Syllables from the phone tier
# --------------------------------------------------------------------------- #


def test_vowel_phones_are_the_syllable_nuclei() -> None:
    assert R.count_syllables(["DH", "AH0", "K", "AE1", "T"]) == 2


def test_stress_digits_do_not_hide_a_vowel() -> None:
    assert R.count_syllables(["AH0"]) == R.count_syllables(["AH1"]) == 1


def test_a_word_of_consonants_has_no_syllable() -> None:
    assert R.count_syllables(["S", "T", "R"]) == 0


# --------------------------------------------------------------------------- #
# Articulation rate (global)
# --------------------------------------------------------------------------- #


def test_ten_syllables_in_two_seconds_is_five_per_second() -> None:
    assert R.articulation_rate(even_words(10, duration=0.2)) == pytest.approx(5.0)


def test_short_gaps_count_as_articulation_time() -> None:
    words = even_words(4, duration=0.2, gap=0.1)
    # 4 syllables over 0.2*4 + 0.1*3 = 1.1 s of articulation.
    assert R.articulation_rate(words, exclude_pauses_over_s=0.25) == pytest.approx(
        4.0 / 1.1, rel=1e-6
    )


def test_a_long_pause_is_excluded_from_articulation_time() -> None:
    words = [word(0, 0.0, 0.2, ["D", "AH1"]), word(1, 2.2, 2.4, ["D", "AH1"])]
    assert R.articulation_rate(words, exclude_pauses_over_s=0.25) == pytest.approx(5.0)


def test_articulation_rate_of_an_empty_alignment_is_nan() -> None:
    assert np.isnan(R.articulation_rate([]))


def test_articulation_rate_without_a_syllable_is_nan() -> None:
    assert np.isnan(R.articulation_rate([word(0, 0.0, 0.2, ["S", "T"])]))


# --------------------------------------------------------------------------- #
# Local rate (+/- 3 words)
# --------------------------------------------------------------------------- #


def test_local_rate_uses_a_window_of_three_words_each_side() -> None:
    words = even_words(21, duration=0.2)
    # Words 7..13 inclusive: 7 syllables over 1.4 s.
    assert R.local_rate(words, 10, window_words=3) == pytest.approx(5.0)


def test_local_window_is_clipped_at_the_start_of_the_utterance() -> None:
    words = even_words(21, duration=0.2)
    assert R.local_rate(words, 0, window_words=3) == pytest.approx(5.0)


def test_local_rate_excludes_a_long_pause_inside_the_window() -> None:
    words = [
        word(0, 0.0, 0.2, ["D", "AH1"]),
        word(1, 2.2, 2.4, ["D", "AH1"]),
        word(2, 2.4, 2.6, ["D", "AH1"]),
    ]
    assert R.local_rate(words, 1, window_words=3, exclude_pauses_over_s=0.25) == pytest.approx(5.0)


def test_local_rate_of_a_rushed_stretch_is_higher_than_the_global_rate() -> None:
    words = even_words(10, duration=0.4)
    words += [word(10 + i, 4.0 + 0.1 * i, 4.1 + 0.1 * i, ["D", "AH1"]) for i in range(7)]
    assert R.local_rate(words, 13) > R.articulation_rate(words)


def test_relative_rate_is_one_when_local_matches_the_global_rate() -> None:
    assert R.relative_rate(5.0, 5.0) == pytest.approx(1.0)


def test_relative_rate_is_nan_without_a_global_rate() -> None:
    assert np.isnan(R.relative_rate(5.0, float("nan")))


# --------------------------------------------------------------------------- #
# Word table
# --------------------------------------------------------------------------- #


def test_word_rate_table_is_indexed_by_word_index() -> None:
    table = R.word_rate(even_words(8, duration=0.2))
    assert list(table.index) == list(range(8))
    assert table.attrs["unit"] == "word"


def test_word_rate_table_reports_local_absolute_and_relative_rate() -> None:
    table = R.word_rate(even_words(9, duration=0.2))
    row = table.loc[4]
    assert row["rate_local_syll_per_s"] == pytest.approx(5.0)
    assert row["rate_relative"] == pytest.approx(1.0, abs=0.01)


def test_word_rate_table_carries_the_global_rate_in_its_attrs() -> None:
    table = R.word_rate(even_words(9, duration=0.2))
    assert table.attrs["articulation_rate"] == pytest.approx(5.0)


def test_word_rate_of_an_empty_alignment_is_an_empty_table() -> None:
    table = R.word_rate([])
    assert table.empty and list(table.columns) == list(R.RATE_WORD_COLUMNS)


# --------------------------------------------------------------------------- #
# de Jong & Wempe syllable nuclei cross-check [R14]
# --------------------------------------------------------------------------- #


def bursts(n: int, *, burst_s: float = 0.15, gap_s: float = 0.15, f_hz: float = 150.0):
    """``n`` vowel-like tone bursts separated by silence."""
    out = []
    for _ in range(n):
        t = np.arange(int(burst_s * SR), dtype=np.float64) / SR
        envelope = np.hanning(t.size)
        out.append(0.5 * envelope * np.sin(2.0 * np.pi * f_hz * t))
        out.append(np.zeros(int(gap_s * SR)))
    return np.concatenate(out).astype(np.float32)


def test_nuclei_count_matches_the_number_of_bursts() -> None:
    nuclei = R.syllable_nuclei(bursts(5), SR, grid=GRID)
    assert len(nuclei) == 5


def test_nuclei_times_fall_inside_their_bursts() -> None:
    nuclei = R.syllable_nuclei(bursts(3), SR, grid=GRID)
    # Burst k spans [k*0.3, k*0.3 + 0.15).
    for k, t in enumerate(sorted(nuclei)):
        assert k * 0.3 <= t <= k * 0.3 + 0.16


def test_silence_has_no_syllable_nuclei() -> None:
    assert R.syllable_nuclei(np.zeros(SR, dtype=np.float32), SR, grid=GRID) == []


def test_a_single_plateau_without_a_dip_counts_once() -> None:
    t = np.arange(SR, dtype=np.float64) / SR
    steady = (0.5 * np.sin(2.0 * np.pi * 150.0 * t)).astype(np.float32)
    assert len(R.syllable_nuclei(steady, SR, grid=GRID)) == 1


def test_raising_the_dip_requirement_merges_adjacent_peaks() -> None:
    audio = bursts(5, gap_s=0.03)
    permissive = R.syllable_nuclei(audio, SR, grid=GRID, min_dip_db=2.0)
    strict = R.syllable_nuclei(audio, SR, grid=GRID, min_dip_db=40.0)
    assert len(strict) < len(permissive)


def test_nuclei_rate_is_nuclei_per_second_of_speech() -> None:
    assert R.nuclei_rate([0.1, 0.3, 0.5], speech_duration_s=1.0) == pytest.approx(3.0)


def test_nuclei_rate_without_speech_time_is_nan() -> None:
    assert np.isnan(R.nuclei_rate([0.1], speech_duration_s=0.0))


def test_crosscheck_correlates_the_two_syllable_counts() -> None:
    words = [
        word(0, 0.0, 0.15, ["D", "AH1"]),
        word(1, 0.3, 0.45, ["D", "AH1"]),
        word(2, 0.6, 0.75, ["D", "AH1"]),
        word(3, 0.9, 1.05, ["D", "AH1"]),
    ]
    report = R.nuclei_crosscheck(words, R.syllable_nuclei(bursts(4), SR, grid=GRID))
    assert report.n_mfa_syllables == 4
    assert report.n_nuclei == 4
    assert report.correlation is None or -1.0 <= report.correlation <= 1.0


def test_crosscheck_reports_the_count_ratio() -> None:
    words = [word(0, 0.0, 0.15, ["D", "AH1"]), word(1, 0.3, 0.45, ["D", "AH1"])]
    report = R.nuclei_crosscheck(words, [0.07, 0.37, 0.67, 0.97])
    assert report.ratio == pytest.approx(2.0)


def test_crosscheck_of_an_empty_alignment_reports_no_ratio() -> None:
    assert R.nuclei_crosscheck([], [0.1]).ratio is None
