"""Duration features (spec 9.1 DUR)."""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import duration as D
from rostrum.types import Word


def word(idx: int, t0: float, t1: float, phones: list[str] | None = None) -> Word:
    phones = ["DH", "AH"] if phones is None else phones
    return Word(
        idx=idx,
        text=f"w{idx}",
        t_start=t0,
        t_end=t1,
        phones=phones,
        n_syllables=sum(1 for p in phones if p.rstrip("012") in {"AH", "IY", "AA"}),
    )


def test_duration_is_the_interval_length() -> None:
    assert D.word_duration(word(0, 1.0, 1.25)) == pytest.approx(0.25)


def test_duration_per_phone_divides_by_the_phone_count() -> None:
    assert D.duration_per_phone(word(0, 0.0, 0.3, ["DH", "AH", "T"])) == pytest.approx(0.1)


def test_duration_per_phone_of_a_word_without_phones_is_nan() -> None:
    assert np.isnan(D.duration_per_phone(word(0, 0.0, 0.3, [])))


def test_duration_per_syllable_divides_by_the_vowel_count() -> None:
    w = word(0, 0.0, 0.4, ["DH", "AH", "T", "AH"])
    assert D.duration_per_syllable(w) == pytest.approx(0.2)


def test_duration_per_syllable_of_a_syllable_free_word_is_nan() -> None:
    assert np.isnan(D.duration_per_syllable(word(0, 0.0, 0.3, ["S", "T"])))


def test_table_has_one_row_per_word_indexed_by_word_index() -> None:
    table = D.word_duration_features([word(0, 0.0, 0.2), word(1, 0.3, 0.6)])
    assert list(table.index) == [0, 1]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "word"


def test_table_reports_duration_phones_and_syllables() -> None:
    row = D.word_duration_features([word(0, 0.0, 0.4, ["DH", "AH", "T", "AH"])]).loc[0]
    assert row["word_duration_s"] == pytest.approx(0.4)
    assert row["n_phones"] == 4
    assert row["n_syllables"] == 2
    assert row["duration_per_phone_s"] == pytest.approx(0.1)
    assert row["duration_per_syllable_s"] == pytest.approx(0.2)


def test_duration_is_also_reported_relative_to_the_recording_median() -> None:
    table = D.word_duration_features([word(0, 0.0, 0.1), word(1, 0.2, 0.4), word(2, 0.5, 0.9)])
    assert table.loc[1, "duration_rel_median"] == pytest.approx(1.0)
    assert table.loc[2, "duration_rel_median"] > 1.0


def test_relative_duration_is_nan_for_an_empty_alignment() -> None:
    table = D.word_duration_features([])
    assert table.empty
    assert list(table.columns) == list(D.DURATION_WORD_COLUMNS)


def test_a_zero_length_word_is_reported_not_silently_dropped() -> None:
    row = D.word_duration_features([word(0, 1.0, 1.0)]).loc[0]
    assert row["word_duration_s"] == pytest.approx(0.0)


def test_table_carries_file_and_channel_for_the_channel_rule() -> None:
    table = D.word_duration_features([word(0, 0.0, 0.2)], file_id="f", channel_id="c")
    assert (table.attrs["file_id"], table.attrs["channel_id"]) == ("f", "c")
