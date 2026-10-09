"""Pause features (spec 9.1 PAU): MFA gaps refined by Silero VAD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rostrum.features import pauses as PA
from rostrum.features.frames import FrameGrid
from rostrum.types import Gap, Word

SR = 16000
GRID = FrameGrid()
CLEAN_DIR = Path(__file__).resolve().parents[1] / "dataset" / "audio" / "clean"


def words(*spans: tuple[float, float]) -> list[Word]:
    return [
        Word(idx=i, text=f"w{i}", t_start=a, t_end=b, phones=["D", "AH1"], n_syllables=1)
        for i, (a, b) in enumerate(spans)
    ]


# --------------------------------------------------------------------------- #
# Gaps from the alignment
# --------------------------------------------------------------------------- #


def test_one_gap_between_each_pair_of_words() -> None:
    gaps = PA.gaps_from_words(words((0.0, 0.5), (1.0, 1.5), (2.0, 2.5)))
    assert [(g.idx, g.prev_word_idx, g.next_word_idx) for g in gaps] == [(0, 0, 1), (1, 1, 2)]


def test_gap_spans_the_silence_between_the_words() -> None:
    gap = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5)))[0]
    assert (gap.t_start, gap.t_end) == (1.0, 1.5)
    assert gap.duration == pytest.approx(0.5)


def test_touching_words_still_produce_a_zero_length_gap() -> None:
    gaps = PA.gaps_from_words(words((0.0, 1.0), (1.0, 2.0)))
    assert len(gaps) == 1 and gaps[0].duration == pytest.approx(0.0)


def test_a_single_word_has_no_gaps() -> None:
    assert PA.gaps_from_words(words((0.0, 1.0))) == []


def test_leading_and_trailing_silence_is_not_a_gap() -> None:
    # Words start at 2.0 s and end at 3.0 s; neither edge becomes a gap.
    gaps = PA.gaps_from_words(words((2.0, 2.4), (2.6, 3.0)))
    assert len(gaps) == 1 and gaps[0].t_start == pytest.approx(2.4)


def test_gap_labels_are_applied_from_the_text_module() -> None:
    gaps = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5), (3.0, 4.0)))
    labelled = PA.apply_gap_labels(gaps, {0: ("boundary", ".")})
    assert (labelled[0].label, labelled[0].punctuation) == ("boundary", ".")
    assert labelled[1].label == "none"


def test_an_unknown_gap_label_is_rejected() -> None:
    gaps = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5)))
    with pytest.raises(ValueError, match="gap label"):
        PA.apply_gap_labels(gaps, {0: ("comma", ",")})


# --------------------------------------------------------------------------- #
# Speech intervals from VAD probabilities
# --------------------------------------------------------------------------- #


def probs_with_silence(
    total_s: float = 2.0, silence: tuple[float, float] = (1.0, 1.5)
) -> tuple[np.ndarray, np.ndarray]:
    """Frame probabilities: 1.0 everywhere except a silent span."""
    n = GRID.n_frames(total_s)
    times = GRID.centres(n)
    p = np.ones(n)
    p[(times >= silence[0]) & (times < silence[1])] = 0.0
    return times, p


def test_speech_intervals_bracket_the_silence() -> None:
    times, p = probs_with_silence()
    intervals = PA.speech_intervals(times, p, threshold=0.5, speech_pad_s=0.0)
    assert len(intervals) == 2
    assert intervals[0][1] == pytest.approx(1.0, abs=0.02)
    assert intervals[1][0] == pytest.approx(1.5, abs=0.02)


def test_a_silence_shorter_than_the_minimum_does_not_split_speech() -> None:
    times, p = probs_with_silence(silence=(1.0, 1.03))
    intervals = PA.speech_intervals(times, p, threshold=0.5, min_silence_s=0.1, speech_pad_s=0.0)
    assert len(intervals) == 1


def test_padding_extends_each_speech_interval() -> None:
    times, p = probs_with_silence()
    padded = PA.speech_intervals(times, p, threshold=0.5, speech_pad_s=0.03)
    assert padded[0][1] == pytest.approx(1.03, abs=0.02)


def test_all_silence_yields_no_speech_intervals() -> None:
    times = GRID.centres(100)
    assert PA.speech_intervals(times, np.zeros(100), threshold=0.5) == []


def test_all_speech_yields_one_interval() -> None:
    times = GRID.centres(100)
    assert len(PA.speech_intervals(times, np.ones(100), threshold=0.5)) == 1


# --------------------------------------------------------------------------- #
# Refinement (spec 9.1 PAU acceptance: a 0.5 s silence measures 0.50 s)
# --------------------------------------------------------------------------- #


def test_an_inserted_half_second_silence_measures_half_a_second() -> None:
    gap = PA.gaps_from_words(words((0.0, 1.02), (1.47, 2.5)))[0]
    times, p = probs_with_silence(total_s=2.5, silence=(1.0, 1.5))
    refined = PA.refine_gaps([gap], PA.speech_intervals(times, p, threshold=0.5, speech_pad_s=0.0))[
        0
    ]
    assert refined.duration == pytest.approx(0.50, abs=0.02)
    assert refined.vad_refined is True


def test_refinement_leaves_the_gap_alone_when_vad_hears_speech_in_it() -> None:
    gap = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5)))[0]
    refined = PA.refine_gaps([gap], [(0.0, 2.5)])[0]
    assert (refined.t_start, refined.t_end) == (1.0, 1.5)
    assert refined.vad_refined is False


def test_refinement_never_moves_a_boundary_further_than_the_limit() -> None:
    gap = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5)))[0]
    refined = PA.refine_gaps([gap], [(0.0, 0.2), (1.49, 2.5)], max_shift_s=0.1)[0]
    assert refined.t_start == pytest.approx(1.0)  # 0.2 is 0.8 s away: rejected
    assert refined.t_end == pytest.approx(1.49)


def test_refinement_without_speech_intervals_changes_nothing() -> None:
    gaps = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5)))
    assert PA.refine_gaps(gaps, [])[0].vad_refined is False


def test_refinement_keeps_the_label_and_the_word_indices() -> None:
    gaps = PA.apply_gap_labels(
        PA.gaps_from_words(words((0.0, 1.02), (1.47, 2.5))), {0: ("boundary", ".")}
    )
    times, p = probs_with_silence(total_s=2.5, silence=(1.0, 1.5))
    refined = PA.refine_gaps(gaps, PA.speech_intervals(times, p, speech_pad_s=0.0))[0]
    assert refined.label == "boundary"
    assert (refined.prev_word_idx, refined.next_word_idx) == (0, 1)


# --------------------------------------------------------------------------- #
# Gap table
# --------------------------------------------------------------------------- #


def table_gaps() -> list[Gap]:
    gaps = PA.gaps_from_words(words((0.0, 1.0), (1.5, 2.5), (2.6, 3.0)))
    return PA.apply_gap_labels(gaps, {0: ("boundary", "."), 1: ("minor", ",")})


def test_gap_table_is_indexed_by_gap_index() -> None:
    table = PA.gap_features(table_gaps())
    assert list(table.index) == [0, 1]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "gap"


def test_gap_table_reports_duration_label_and_boundary_flag() -> None:
    table = PA.gap_features(table_gaps())
    assert table.loc[0, "gap_duration_s"] == pytest.approx(0.5)
    assert table.loc[0, "gap_label"] == "boundary"
    assert bool(table.loc[0, "gap_is_boundary"]) is True
    assert bool(table.loc[1, "gap_is_boundary"]) is False


def test_a_gap_over_the_threshold_counts_as_a_pause() -> None:
    table = PA.gap_features(table_gaps(), min_pause_s=0.25)
    assert bool(table.loc[0, "gap_is_pause"]) is True
    assert bool(table.loc[1, "gap_is_pause"]) is False


def test_gap_table_records_whether_vad_refined_the_boundaries() -> None:
    table = PA.gap_features(table_gaps())
    assert not table["gap_vad_refined"].any()


def test_gap_table_of_a_single_word_is_empty_but_typed() -> None:
    table = PA.gap_features([])
    assert table.empty and list(table.columns) == list(PA.PAUSE_GAP_COLUMNS)


# --------------------------------------------------------------------------- #
# Silero, on real speech
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not CLEAN_DIR.is_dir(), reason="needs dataset/audio/clean")
def test_silero_measures_a_half_second_silence_inserted_into_real_speech() -> None:
    import soundfile as sf

    path = sorted(CLEAN_DIR.glob("*.wav"))[0]
    audio, sr = sf.read(path, dtype="float32")
    audio = audio[: sr * 20]
    cut = len(audio) // 2
    spliced = np.concatenate([audio[:cut], np.zeros(sr // 2, dtype=np.float32), audio[cut:]])

    times, probs = PA.speech_probs(spliced, sr)
    intervals = PA.speech_intervals(times, probs, threshold=0.5, speech_pad_s=0.0)
    measured = PA.silence_containing(intervals, cut / sr + 0.25)
    assert measured is not None
    assert (measured[1] - measured[0]) == pytest.approx(0.50, abs=0.05)


@pytest.mark.slow
def test_speech_probabilities_cover_the_frame_grid() -> None:
    rng = np.random.default_rng(0)
    audio = (0.05 * rng.standard_normal(SR)).astype(np.float32)
    times, probs = PA.speech_probs(audio, SR)
    assert times.shape == probs.shape == (GRID.n_frames(1.0),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
