"""Pitch features (spec 9.1 F0/F0x, 9.2 speaker normalisation).

Synthetic signals only: a tone of known frequency, a glide of known slope, and
silence. No alignment, no MFA, no dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import pitch as P
from rostrum.features.frames import FrameGrid
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()


def tone(f_hz: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A steady sine at ``f_hz``."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return (0.5 * np.sin(2.0 * np.pi * f_hz * t)).astype(np.float32)


def glide(f0: float, f1: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A linear frequency sweep from ``f0`` to ``f1``."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t**2)
    return (0.5 * np.sin(phase)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Semitone conversion (spec 9.2)
# --------------------------------------------------------------------------- #


def test_semitones_of_double_the_reference_is_plus_twelve() -> None:
    assert P.semitones(np.array([400.0]), 200.0)[0] == pytest.approx(12.0)


def test_semitones_of_half_the_reference_is_minus_twelve() -> None:
    assert P.semitones(np.array([100.0]), 200.0)[0] == pytest.approx(-12.0)


def test_semitones_of_the_reference_itself_is_zero() -> None:
    assert P.semitones(np.array([200.0]), 200.0)[0] == pytest.approx(0.0)


def test_unvoiced_frames_stay_nan_in_semitones() -> None:
    out = P.semitones(np.array([200.0, np.nan, 0.0]), 200.0)
    assert out[0] == pytest.approx(0.0)
    assert np.isnan(out[1]) and np.isnan(out[2])


def test_semitones_needs_a_positive_reference() -> None:
    with pytest.raises(ValueError, match="reference"):
        P.semitones(np.array([200.0]), 0.0)


def test_cents_difference_of_an_octave_is_1200() -> None:
    assert P.cents_difference(np.array([400.0]), np.array([200.0]))[0] == pytest.approx(1200.0)


def test_cents_difference_is_nan_where_either_track_is_unvoiced() -> None:
    out = P.cents_difference(np.array([200.0, np.nan]), np.array([np.nan, 200.0]))
    assert np.all(np.isnan(out))


# --------------------------------------------------------------------------- #
# Praat extraction and the two-pass range
# --------------------------------------------------------------------------- #


def test_200_hz_tone_yields_f0_of_200_hz() -> None:
    f0 = P.praat_pitch_hz(tone(200.0), SR, floor_hz=50.0, ceiling_hz=700.0, grid=GRID)
    voiced = f0[np.isfinite(f0)]
    assert voiced.size > 50
    assert abs(float(np.median(voiced)) - 200.0) < 1.0


def test_pitch_array_matches_the_frame_grid() -> None:
    f0 = P.praat_pitch_hz(tone(200.0), SR, floor_hz=50.0, ceiling_hz=700.0, grid=GRID)
    assert f0.shape == (GRID.n_frames(1.0),)


def test_silence_has_no_voiced_frames() -> None:
    f0 = P.praat_pitch_hz(
        np.zeros(SR, dtype=np.float32), SR, floor_hz=50.0, ceiling_hz=700.0, grid=GRID
    )
    assert not np.any(np.isfinite(f0))


def test_two_pass_range_narrows_to_the_observed_quartiles() -> None:
    f0 = np.concatenate([np.full(50, 100.0), np.full(50, 200.0)])
    floor, ceiling = P.two_pass_range(f0, q1_scale=0.75, q3_scale=1.5)
    assert floor == pytest.approx(0.75 * 100.0)
    assert ceiling == pytest.approx(1.5 * 200.0)


def test_two_pass_range_is_clamped_to_the_configured_bounds() -> None:
    floor, ceiling = P.two_pass_range(
        np.full(100, 60.0),
        q1_scale=0.1,
        q3_scale=100.0,
        min_floor_hz=40.0,
        max_ceiling_hz=800.0,
    )
    assert (floor, ceiling) == (40.0, 800.0)


def test_two_pass_range_returns_none_when_too_few_voiced_frames() -> None:
    assert P.two_pass_range(np.full(3, 200.0), min_voiced_frames=50) is None


def test_two_pass_range_ignores_unvoiced_frames() -> None:
    f0 = np.concatenate([np.full(60, 200.0), np.full(60, np.nan)])
    out = P.two_pass_range(f0, min_voiced_frames=50)
    assert out is not None and out[0] == pytest.approx(150.0)


# --------------------------------------------------------------------------- #
# Cross-check and agreement (spec 9.1 F0x)
# --------------------------------------------------------------------------- #


def test_agreement_mask_accepts_frames_within_fifty_cents() -> None:
    praat = np.array([200.0, 200.0])
    crepe = np.array([200.0 * 2 ** (40 / 1200), 200.0 * 2 ** (60 / 1200)])
    assert list(P.agreement_mask(praat, crepe, agreement_cents=50.0)) == [True, False]


def test_a_frame_the_crosscheck_did_not_estimate_does_not_agree() -> None:
    assert not P.agreement_mask(np.array([200.0]), np.array([np.nan]))[0]


def test_agreement_rate_counts_only_frames_both_trackers_voiced() -> None:
    praat = np.array([200.0, 200.0, np.nan])
    crepe = np.array([200.0, 400.0, 200.0])
    assert P.agreement_rate(praat, crepe) == pytest.approx(0.5)


def test_agreement_rate_is_none_when_nothing_is_comparable() -> None:
    assert P.agreement_rate(np.array([np.nan]), np.array([200.0])) is None


# --------------------------------------------------------------------------- #
# PitchTrack
# --------------------------------------------------------------------------- #


def test_track_of_a_200_hz_tone_has_a_200_hz_median() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    assert abs(track.median_hz - 200.0) < 1.0


def test_track_of_a_steady_tone_is_zero_semitones_re_its_own_median() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    assert float(np.nanmedian(track.f0_st)) == pytest.approx(0.0, abs=0.05)


def test_track_uses_the_second_pass_range() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    assert track.floor_hz > 50.0
    assert track.ceiling_hz < 700.0
    assert track.two_pass is True


def test_track_keeps_the_first_pass_range_when_voicing_is_too_sparse() -> None:
    quiet = np.zeros(SR, dtype=np.float32)
    quiet[: SR // 50] = tone(200.0, 0.02)
    track = P.pitch_track(quiet, SR, grid=GRID, crosscheck=False)
    assert (track.floor_hz, track.ceiling_hz) == (50.0, 700.0)
    assert track.two_pass is False


def test_track_without_a_crosscheck_treats_every_voiced_frame_as_usable() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    assert np.array_equal(track.usable, track.voiced)
    assert track.agreement_rate is None


def test_track_of_silence_has_no_median_and_all_nan_semitones() -> None:
    track = P.pitch_track(np.zeros(SR, dtype=np.float32), SR, grid=GRID, crosscheck=False)
    assert track.median_hz is None
    assert np.all(np.isnan(track.f0_st))


def test_a_disagreeing_frame_is_nan_for_statistics() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    bad = np.zeros_like(track.voiced)
    bad[:5] = True
    masked = P.apply_agreement(track, agree=~bad)
    assert np.all(np.isnan(masked.f0_st[:5]))
    assert np.isfinite(masked.f0_st[track.voiced & ~bad]).all()
    assert masked.usable[:5].sum() == 0


@pytest.mark.slow
def test_torchcrepe_agrees_with_praat_on_a_pure_tone() -> None:
    track = P.pitch_track(tone(200.0, 0.5), SR, grid=GRID, crosscheck=True, crosscheck_model="tiny")
    assert track.agreement_rate is not None
    assert track.agreement_rate > 0.9


# --------------------------------------------------------------------------- #
# Word aggregates (spec 9.1: mean st, max st, slope st/s, range st)
# --------------------------------------------------------------------------- #


def words(*spans: tuple[float, float]) -> list[Word]:
    return [
        Word(idx=i, text=f"w{i}", t_start=a, t_end=b, phones=["AH"], n_syllables=1)
        for i, (a, b) in enumerate(spans)
    ]


def test_word_table_is_indexed_by_word_index() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    table = P.word_pitch(track, words((0.1, 0.4), (0.5, 0.9)))
    assert list(table.index) == [0, 1]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "word"


def test_steady_tone_has_zero_mean_semitones_and_no_range() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    row = P.word_pitch(track, words((0.2, 0.8))).loc[0]
    assert row["f0_mean_st"] == pytest.approx(0.0, abs=0.1)
    assert row["f0_range_st"] == pytest.approx(0.0, abs=0.2)
    assert row["f0_slope_st_per_s"] == pytest.approx(0.0, abs=0.5)


def test_rising_glide_has_a_positive_slope_in_semitones_per_second() -> None:
    track = P.pitch_track(glide(150.0, 300.0), SR, grid=GRID, crosscheck=False)
    row = P.word_pitch(track, words((0.1, 0.9))).loc[0]
    assert row["f0_slope_st_per_s"] > 5.0
    assert row["f0_max_st"] > row["f0_mean_st"]


def test_glide_range_is_about_an_octave_for_a_doubling_sweep() -> None:
    track = P.pitch_track(glide(150.0, 300.0), SR, grid=GRID, crosscheck=False)
    row = P.word_pitch(track, words((0.05, 0.95))).loc[0]
    assert row["f0_range_st"] == pytest.approx(12.0, abs=2.0)


def test_word_with_no_voiced_frames_is_nan_not_zero() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    row = P.word_pitch(track, words((0.2, 0.3), (5.0, 5.2))).loc[1]
    assert np.isnan(row["f0_mean_st"])
    assert row["f0_n_voiced"] == 0


def test_voiced_fraction_is_reported_per_word() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    row = P.word_pitch(track, words((0.2, 0.8))).loc[0]
    assert row["f0_voiced_frac"] == pytest.approx(1.0, abs=0.1)


def test_word_table_carries_the_file_and_channel_for_the_channel_rule() -> None:
    track = P.pitch_track(tone(200.0), SR, grid=GRID, crosscheck=False)
    table = P.word_pitch(track, words((0.2, 0.8)), file_id="f1", channel_id="c1")
    assert (table.attrs["file_id"], table.attrs["channel_id"]) == ("f1", "c1")
