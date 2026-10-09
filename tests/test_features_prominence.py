"""CWT prominence (spec 9.1 PROM).

The acceptance test from the build prompt: a synthetic utterance of equal
syllables with one word made louder, longer and higher must rank that word
first. Around it, the pieces - gap interpolation, the duration component, the
scale bank and the scale-summed response - are checked on analytic inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import energy as E
from rostrum.features import pitch as P
from rostrum.features import prominence as PR
from rostrum.features.frames import FrameGrid
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()


def syllable(f_hz: float, seconds: float, amplitude: float, sr: int = SR) -> np.ndarray:
    """A harmonic buzz with a short raised onset and offset."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    out = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        out += weight * np.sin(2.0 * np.pi * f_hz * harmonic * t)
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.03)
    return amplitude * out * envelope / 1.9


def utterance(
    stressed: int = 2,
    n_words: int = 5,
    base_hz: float = 140.0,
    base_s: float = 0.30,
    base_amp: float = 0.25,
) -> tuple[np.ndarray, list[Word]]:
    """An utterance of equal syllables with one louder, longer, higher word.

    Returns:
        ``(samples, words)``; ``words[stressed]`` is the prominent one.
    """
    parts = [np.zeros(int(0.2 * SR))]
    words: list[Word] = []
    t = 0.2
    for i in range(n_words):
        emphatic = i == stressed
        f_hz = base_hz * 1.26 if emphatic else base_hz  # +4 semitones
        seconds = base_s * 1.6 if emphatic else base_s
        amplitude = base_amp * 2.2 if emphatic else base_amp
        parts.append(syllable(f_hz, seconds, amplitude))
        words.append(
            Word(
                idx=i,
                text=f"w{i}",
                t_start=round(t, 3),
                t_end=round(t + seconds, 3),
                phones=["N", "AW1"],
                n_syllables=1,
            )
        )
        t += seconds
        parts.append(np.zeros(int(0.08 * SR)))
        t += 0.08
    return np.concatenate(parts).astype(np.float32), words


def tracks(samples: np.ndarray) -> tuple[P.PitchTrack, E.EnergyTrack]:
    """Pitch and energy on one frame count, with no CREPE cross-check."""
    pitch = P.pitch_track(samples, SR, grid=GRID, crosscheck=False)
    energy = E.energy_track(samples, SR, grid=GRID, voiced=pitch.voiced, n_frames=pitch.n_frames)
    return pitch, energy


# --------------------------------------------------------------------------- #
# The acceptance test
# --------------------------------------------------------------------------- #


def test_the_louder_longer_higher_word_is_the_most_prominent() -> None:
    samples, words = utterance(stressed=2)
    pitch, energy = tracks(samples)
    track = PR.prominence_track(pitch, energy, words, grid=GRID)
    table = PR.word_prominence(track, words, grid=GRID)
    assert int(table["prominence"].idxmax()) == 2
    assert table.loc[2, "prominence_z"] > 1.0
    assert (table.drop(index=2)["prominence_z"] < table.loc[2, "prominence_z"]).all()


@pytest.mark.parametrize("stressed", [0, 1, 3, 4])
def test_the_ranking_follows_whichever_word_is_emphasised(stressed: int) -> None:
    samples, words = utterance(stressed=stressed)
    pitch, energy = tracks(samples)
    table = PR.word_prominence(
        PR.prominence_track(pitch, energy, words, grid=GRID), words, grid=GRID
    )
    assert int(table["prominence"].idxmax()) == stressed


def test_the_prominence_peak_lands_inside_the_emphasised_word() -> None:
    samples, words = utterance(stressed=2)
    pitch, energy = tracks(samples)
    table = PR.word_prominence(
        PR.prominence_track(pitch, energy, words, grid=GRID), words, grid=GRID
    )
    peak = float(table.loc[2, "prominence_peak_t"])
    assert words[2].t_start <= peak <= words[2].t_end
    assert 0.0 <= float(table.loc[2, "prominence_peak_rel"]) <= 1.0


# --------------------------------------------------------------------------- #
# The composite signal
# --------------------------------------------------------------------------- #


def test_gap_interpolation_fills_a_run_and_holds_the_edges() -> None:
    curve = np.array([np.nan, 1.0, np.nan, np.nan, 4.0, np.nan])
    out = PR.interpolate_gaps(curve)
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(4.0)
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)


def test_interpolation_of_an_all_nan_curve_is_zeros() -> None:
    assert PR.interpolate_gaps(np.full(5, np.nan)) == pytest.approx(0.0)


def test_zscore_of_a_flat_curve_is_zeros_not_nan() -> None:
    assert PR.zscore(np.full(10, 3.0)) == pytest.approx(0.0)


def test_duration_signal_is_higher_over_the_longer_word() -> None:
    words = [
        Word(idx=0, text="a", t_start=0.0, t_end=0.20, phones=["AH1"], n_syllables=1),
        Word(idx=1, text="b", t_start=0.25, t_end=0.75, phones=["AH1"], n_syllables=1),
    ]
    curve = PR.duration_signal(words, grid=GRID, n_frames=100)
    short = curve[GRID.frame_indices(0.0, 0.20, 100)].mean()
    long = curve[GRID.frame_indices(0.25, 0.75, 100)].mean()
    assert long > short


def test_duration_signal_divides_by_the_syllable_count() -> None:
    """Two syllables in twice the time is the same tempo, so the same value."""
    words = [
        Word(idx=0, text="a", t_start=0.0, t_end=0.20, phones=["AH1"], n_syllables=1),
        Word(idx=1, text="bc", t_start=0.20, t_end=0.60, phones=["AH1", "IY1"], n_syllables=2),
    ]
    curve = PR.duration_signal(words, grid=GRID, n_frames=80)
    assert curve == pytest.approx(0.0)


def test_composite_applies_the_spec_weights() -> None:
    f0 = np.array([1.0, -1.0, 1.0, -1.0])
    energy = np.array([1.0, -1.0, 1.0, -1.0])
    duration = np.zeros(4)
    out = PR.composite_signal(f0, energy, duration, weights=(1.0, 0.5, 1.0), normalise=True)
    assert out.weights == (1.0, 0.5, 1.0)
    # Both non-zero components z-score to the same unit-variance curve, so the
    # sum is (1.0 + 0.5) times it; the flat duration curve contributes nothing.
    assert out.values == pytest.approx(1.5 * np.array([1.0, -1.0, 1.0, -1.0]))
    assert set(out.components) == set(PR.COMPONENT_ORDER)


def test_composite_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        PR.composite_signal(np.zeros(5), np.zeros(4), np.zeros(5))


# --------------------------------------------------------------------------- #
# The scale bank
# --------------------------------------------------------------------------- #


def test_scale_bank_spans_the_documented_periods() -> None:
    scales, periods = PR.cwt_scales(hop_s=0.010)
    assert scales.size == PR.DEFAULT_N_OCTAVES * PR.DEFAULT_SCALES_PER_OCTAVE + 1
    assert periods[0] == pytest.approx(0.100, rel=1e-9)
    assert periods[-1] == pytest.approx(3.200, rel=1e-9)
    assert np.all(np.diff(scales) > 0)


def test_scale_bank_rejects_nonsense_parameters() -> None:
    with pytest.raises(ValueError, match="positive"):
        PR.cwt_scales(hop_s=0.0)
    with pytest.raises(ValueError, match="positive"):
        PR.cwt_scales(n_octaves=0)


def test_summed_response_is_positive_on_a_bump_and_negative_in_a_dip() -> None:
    signal = np.zeros(400)
    signal[200:220] = 1.0
    scales, _ = PR.cwt_scales(hop_s=0.010)
    _, summed = PR.cwt_response(signal, scales, hop_s=0.010)
    assert summed[210] > 0.0
    assert summed[summed.size // 8] <= 0.0 or abs(summed[summed.size // 8]) < summed[210]


def test_response_of_empty_input_is_empty() -> None:
    scales, _ = PR.cwt_scales(hop_s=0.010)
    coefficients, summed = PR.cwt_response(np.zeros(0), scales, hop_s=0.010)
    assert coefficients.shape == (scales.size, 0)
    assert summed.size == 0


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_table_shape_attrs_and_scale_provenance() -> None:
    samples, words = utterance()
    pitch, energy = tracks(samples)
    track = PR.prominence_track(pitch, energy, words, grid=GRID)
    table = PR.word_prominence(track, words, grid=GRID, file_id="f1", channel_id="ch1")
    assert list(table.columns) == list(PR.PROMINENCE_WORD_COLUMNS)
    assert list(table.index) == [w.idx for w in words]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "word"
    assert table.attrs["file_id"] == "f1"
    assert table.attrs["channel_id"] == "ch1"
    assert table.attrs["scale_range_s"] == pytest.approx((0.1, 3.2))
    assert track.scale_range_s == pytest.approx((0.1, 3.2))


def test_mismatched_track_lengths_are_rejected() -> None:
    import dataclasses

    samples, words = utterance()
    pitch, energy = tracks(samples)
    half = pitch.n_frames // 2
    short = dataclasses.replace(
        energy,
        times=energy.times[:half],
        db=energy.db[:half],
        db_rel=energy.db_rel[:half],
        z=energy.z[:half],
        voiced=energy.voiced[:half],
    )
    with pytest.raises(ValueError, match="frames"):
        PR.prominence_track(pitch, short, words, grid=GRID)


def test_a_word_with_no_frames_is_nan() -> None:
    samples, words = utterance()
    pitch, energy = tracks(samples)
    track = PR.prominence_track(pitch, energy, words, grid=GRID)
    outside = [Word(idx=0, text="x", t_start=99.0, t_end=99.3, phones=["AH1"], n_syllables=1)]
    table = PR.word_prominence(track, outside, grid=GRID)
    assert np.isnan(table.loc[0, "prominence"])
    assert table.loc[0, "prominence_n_frames"] == 0
