"""Energy features (spec 9.1 EN, 9.2 energy normalisation)."""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import energy as E
from rostrum.features.frames import FrameGrid
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()


def tone(amplitude: float = 0.5, seconds: float = 1.0, f_hz: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    return (amplitude * np.sin(2.0 * np.pi * f_hz * t)).astype(np.float32)


def ramp(a0: float, a1: float, seconds: float = 1.0, f_hz: float = 200.0) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n, dtype=np.float64) / SR
    envelope = np.linspace(a0, a1, n)
    return (envelope * np.sin(2.0 * np.pi * f_hz * t)).astype(np.float32)


def words(*spans: tuple[float, float]) -> list[Word]:
    return [
        Word(idx=i, text=f"w{i}", t_start=a, t_end=b, phones=["AH"], n_syllables=1)
        for i, (a, b) in enumerate(spans)
    ]


# --------------------------------------------------------------------------- #
# Praat intensity
# --------------------------------------------------------------------------- #


def test_intensity_array_matches_the_frame_grid() -> None:
    db = E.praat_intensity_db(tone(), SR, grid=GRID)
    assert db.shape == (GRID.n_frames(1.0),)


def test_halving_the_amplitude_drops_intensity_by_six_db() -> None:
    loud = np.nanmedian(E.praat_intensity_db(tone(0.5), SR, grid=GRID))
    quiet = np.nanmedian(E.praat_intensity_db(tone(0.25), SR, grid=GRID))
    assert loud - quiet == pytest.approx(6.0, abs=0.5)


def test_tenfold_amplitude_is_twenty_db() -> None:
    loud = np.nanmedian(E.praat_intensity_db(tone(0.5), SR, grid=GRID))
    quiet = np.nanmedian(E.praat_intensity_db(tone(0.05), SR, grid=GRID))
    assert loud - quiet == pytest.approx(20.0, abs=0.5)


def test_digital_silence_is_clamped_to_the_floor() -> None:
    db = E.praat_intensity_db(np.zeros(SR, dtype=np.float32), SR, grid=GRID, floor_db=-70.0)
    assert np.nanmax(db) <= -70.0 + 1e-6


# --------------------------------------------------------------------------- #
# RMS cross-check
# --------------------------------------------------------------------------- #


def test_rms_of_a_full_scale_sine_is_minus_three_dbfs() -> None:
    db = E.rms_db(tone(1.0), SR, grid=GRID)
    assert float(np.median(db)) == pytest.approx(-3.01, abs=0.1)


def test_rms_follows_the_same_grid_as_intensity() -> None:
    assert (
        E.rms_db(tone(), SR, grid=GRID).shape == E.praat_intensity_db(tone(), SR, grid=GRID).shape
    )


def test_rms_tracks_praat_intensity_on_a_ramp() -> None:
    audio = ramp(0.05, 0.5)
    r = E.track_correlation(
        E.praat_intensity_db(audio, SR, grid=GRID), E.rms_db(audio, SR, grid=GRID)
    )
    assert r is not None and r > 0.95


def test_correlation_is_none_when_a_track_is_constant() -> None:
    assert E.track_correlation(np.ones(10), np.arange(10.0)) is None


# --------------------------------------------------------------------------- #
# Normalisation (spec 9.2: dB re voiced median, then z-score)
# --------------------------------------------------------------------------- #


def test_relative_energy_is_zero_at_the_voiced_median() -> None:
    db = np.array([60.0, 65.0, 70.0])
    voiced = np.array([True, True, True])
    rel, _z, median = E.normalise_db(db, voiced)
    assert median == pytest.approx(65.0)
    assert rel[1] == pytest.approx(0.0)


def test_unvoiced_frames_do_not_set_the_reference() -> None:
    db = np.array([60.0, 65.0, 70.0, -70.0, -70.0])
    voiced = np.array([True, True, True, False, False])
    _rel, _z, median = E.normalise_db(db, voiced)
    assert median == pytest.approx(65.0)


def test_z_scores_have_unit_spread_over_voiced_frames() -> None:
    rng = np.random.default_rng(0)
    db = 65.0 + 3.0 * rng.standard_normal(500)
    voiced = np.ones(500, dtype=bool)
    _rel, z, _median = E.normalise_db(db, voiced)
    assert float(np.std(z[voiced])) == pytest.approx(1.0, abs=0.05)
    assert float(np.mean(z[voiced])) == pytest.approx(0.0, abs=0.05)


def test_normalisation_without_a_voiced_frame_yields_nan() -> None:
    rel, z, median = E.normalise_db(np.array([-70.0, -70.0]), np.zeros(2, dtype=bool))
    assert median is None
    assert np.all(np.isnan(rel)) and np.all(np.isnan(z))


def test_a_flat_recording_gives_zero_z_rather_than_dividing_by_zero() -> None:
    _rel, z, _median = E.normalise_db(np.full(10, 65.0), np.ones(10, dtype=bool))
    assert np.all(np.isfinite(z))


# --------------------------------------------------------------------------- #
# EnergyTrack
# --------------------------------------------------------------------------- #


def test_track_reports_the_voiced_median_in_db() -> None:
    track = E.energy_track(tone(), SR, grid=GRID)
    assert track.voiced_median_db is not None
    assert 40.0 < track.voiced_median_db < 100.0


def test_track_of_a_steady_tone_is_near_zero_relative_db() -> None:
    track = E.energy_track(tone(), SR, grid=GRID)
    assert float(np.nanmedian(track.db_rel)) == pytest.approx(0.0, abs=0.5)


def test_track_marks_silence_as_unvoiced() -> None:
    audio = np.concatenate([tone(0.5, 0.5), np.zeros(SR // 2, dtype=np.float32)])
    track = E.energy_track(audio, SR, grid=GRID)
    assert track.voiced[:40].all()
    assert not track.voiced[-20:].any()


def test_track_accepts_an_externally_supplied_voicing_mask() -> None:
    audio = tone()
    mask = np.zeros(GRID.n_frames(1.0), dtype=bool)
    mask[:50] = True
    track = E.energy_track(audio, SR, grid=GRID, voiced=mask)
    assert np.array_equal(track.voiced, mask)


def test_track_carries_the_rms_crosscheck_correlation() -> None:
    track = E.energy_track(ramp(0.05, 0.5), SR, grid=GRID, rms_crosscheck=True)
    assert track.rms_correlation is not None and track.rms_correlation > 0.95


# --------------------------------------------------------------------------- #
# Word aggregates (spec 9.1: mean, peak, slope dB/s)
# --------------------------------------------------------------------------- #


def test_word_table_is_indexed_by_word_index() -> None:
    track = E.energy_track(tone(), SR, grid=GRID)
    table = E.word_energy(track, words((0.1, 0.4), (0.5, 0.9)))
    assert list(table.index) == [0, 1]
    assert table.attrs["unit"] == "word"


def test_louder_word_has_the_higher_mean_z() -> None:
    audio = np.concatenate([tone(0.05, 0.5), tone(0.5, 0.5)])
    track = E.energy_track(audio, SR, grid=GRID)
    table = E.word_energy(track, words((0.05, 0.45), (0.55, 0.95)))
    assert table.loc[1, "energy_mean_z"] > table.loc[0, "energy_mean_z"]


def test_fading_word_has_a_negative_slope_in_db_per_second() -> None:
    track = E.energy_track(ramp(0.5, 0.05), SR, grid=GRID)
    row = E.word_energy(track, words((0.1, 0.9))).loc[0]
    assert row["energy_slope_db_per_s"] < -5.0


def test_peak_is_at_least_the_mean() -> None:
    track = E.energy_track(ramp(0.05, 0.5), SR, grid=GRID)
    row = E.word_energy(track, words((0.1, 0.9))).loc[0]
    assert row["energy_peak_z"] >= row["energy_mean_z"]


def test_word_outside_the_audio_is_nan() -> None:
    track = E.energy_track(tone(), SR, grid=GRID)
    row = E.word_energy(track, words((5.0, 5.2))).loc[0]
    assert np.isnan(row["energy_mean_z"]) and row["energy_n_frames"] == 0


def test_word_table_reports_absolute_relative_db_as_well_as_z() -> None:
    track = E.energy_track(tone(), SR, grid=GRID)
    row = E.word_energy(track, words((0.1, 0.9))).loc[0]
    assert row["energy_mean_db_rel"] == pytest.approx(0.0, abs=1.0)
