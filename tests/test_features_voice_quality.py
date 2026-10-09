"""Voice quality (spec 9.1 VQ).

Synthetic phonation only. A clean harmonic buzz is the "healthy" voice; adding
broadband noise to the same buzz must lower HNR and CPPS and raise jitter and
shimmer. A word too short to hold three glottal periods must come back ``NaN``
rather than as a number computed from one or two.
"""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import voice_quality as VQ
from rostrum.features.frames import FrameGrid
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()
SEED = 20261006


def buzz(
    f_hz: float = 140.0,
    seconds: float = 0.40,
    amplitude: float = 0.5,
    noise: float = 0.0,
    sr: int = SR,
    seed: int = SEED,
) -> np.ndarray:
    """A harmonic buzz, optionally with additive broadband noise."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    out = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        out += weight * np.sin(2.0 * np.pi * f_hz * harmonic * t)
    rng = np.random.default_rng(seed)
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.02)
    return amplitude * (out / 1.9 + noise * rng.standard_normal(t.size)) * envelope


def utterance(specs: list[tuple[float, float]]) -> tuple[np.ndarray, list[Word]]:
    """Build words from ``(duration_s, noise)`` pairs separated by silence."""
    parts = [np.zeros(int(0.2 * SR))]
    words: list[Word] = []
    t = 0.2
    for i, (seconds, noise) in enumerate(specs):
        parts.append(buzz(seconds=seconds, noise=noise))
        words.append(
            Word(
                idx=i,
                text=f"w{i}",
                t_start=round(t, 3),
                t_end=round(t + seconds, 3),
                phones=["AH1"],
                n_syllables=1,
            )
        )
        t += seconds
        parts.append(np.zeros(int(0.10 * SR)))
        t += 0.10
    return np.concatenate(parts).astype(np.float32), words


# --------------------------------------------------------------------------- #
# Harmonicity track
# --------------------------------------------------------------------------- #


def test_harmonicity_track_is_on_the_project_grid() -> None:
    x = buzz(seconds=1.0)
    track = VQ.harmonicity_track(x, SR, grid=GRID, floor_hz=100.0, ceiling_hz=250.0)
    assert track.n_frames == GRID.n_frames_for(x.size)
    assert track.times.size == track.n_frames
    assert track.floor_hz == pytest.approx(100.0)


def test_harmonicity_is_high_for_a_clean_buzz_and_low_with_noise() -> None:
    clean = VQ.harmonicity_track(buzz(seconds=1.0), SR, grid=GRID, floor_hz=100.0, ceiling_hz=250.0)
    noisy = VQ.harmonicity_track(
        buzz(seconds=1.0, noise=0.5), SR, grid=GRID, floor_hz=100.0, ceiling_hz=250.0
    )
    assert np.nanmedian(clean.hnr_db) > np.nanmedian(noisy.hnr_db) + 10.0


def test_praat_unvoiced_sentinel_does_not_leak_into_the_curve() -> None:
    track = VQ.harmonicity_track(
        np.zeros(SR, dtype=np.float32), SR, grid=GRID, floor_hz=100.0, ceiling_hz=250.0
    )
    finite = track.hnr_db[np.isfinite(track.hnr_db)]
    assert finite.size == 0 or finite.min() > -100.0


def test_empty_audio_yields_an_empty_track() -> None:
    track = VQ.harmonicity_track(np.zeros(0, dtype=np.float32), SR, grid=GRID)
    assert track.n_frames == 0


# --------------------------------------------------------------------------- #
# CPPS
# --------------------------------------------------------------------------- #


def test_cpps_is_higher_for_the_cleaner_voice() -> None:
    clean = VQ.cpps_db(buzz(seconds=0.4), SR)
    noisy = VQ.cpps_db(buzz(seconds=0.4, noise=0.4), SR)
    assert np.isfinite(clean) and np.isfinite(noisy)
    assert clean > noisy


def test_cpps_of_a_segment_shorter_than_the_minimum_is_nan() -> None:
    assert np.isnan(VQ.cpps_db(buzz(seconds=0.02), SR))


# --------------------------------------------------------------------------- #
# The per-word table
# --------------------------------------------------------------------------- #


def test_table_shape_columns_and_attrs() -> None:
    samples, words = utterance([(0.4, 0.0), (0.4, 0.3)])
    table = VQ.word_voice_quality(
        samples,
        SR,
        words,
        grid=GRID,
        floor_hz=100.0,
        ceiling_hz=250.0,
        file_id="f1",
        channel_id="ch1",
    )
    assert list(table.columns) == list(VQ.VQ_WORD_COLUMNS)
    assert list(table.index) == [0, 1]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "word"
    assert table.attrs["file_id"] == "f1"
    assert table.attrs["channel_sensitive"] is True


def test_noise_lowers_hnr_and_cpps_and_raises_jitter_and_shimmer() -> None:
    samples, words = utterance([(0.4, 0.0), (0.4, 0.25)])
    table = VQ.word_voice_quality(samples, SR, words, grid=GRID, floor_hz=100.0, ceiling_hz=250.0)
    clean, noisy = table.loc[0], table.loc[1]
    assert clean["hnr_db"] > noisy["hnr_db"]
    assert clean["cpps_db"] > noisy["cpps_db"]
    assert clean["jitter_local"] < noisy["jitter_local"]
    assert clean["shimmer_local"] < noisy["shimmer_local"]


def test_a_word_with_too_few_periods_is_nan_everywhere() -> None:
    """Spec 9.1: NaN below three voiced periods, never a number from nothing."""
    samples, words = utterance([(0.4, 0.0), (0.015, 0.0)])
    table = VQ.word_voice_quality(samples, SR, words, grid=GRID, floor_hz=100.0, ceiling_hz=250.0)
    short = table.loc[1]
    assert short["vq_n_periods"] < VQ.MIN_PERIODS
    for column in ("hnr_db", "jitter_local", "shimmer_local", "cpps_db"):
        assert np.isnan(short[column]), column


def test_silence_has_no_periods_and_no_measurements() -> None:
    words = [Word(idx=0, text="x", t_start=0.1, t_end=0.6, phones=["AH1"], n_syllables=1)]
    table = VQ.word_voice_quality(
        np.zeros(SR, dtype=np.float32),
        SR,
        words,
        grid=GRID,
        floor_hz=100.0,
        ceiling_hz=250.0,
    )
    assert table.loc[0, "vq_n_periods"] == 0
    assert np.isnan(table.loc[0, "hnr_db"])


def test_period_count_scales_with_f0_and_word_length() -> None:
    """A 140 Hz buzz of 0.4 s holds about 56 periods; the count must be near it."""
    samples, words = utterance([(0.4, 0.0)])
    table = VQ.word_voice_quality(samples, SR, words, grid=GRID, floor_hz=100.0, ceiling_hz=250.0)
    assert 40 <= table.loc[0, "vq_n_periods"] <= 60


def test_switching_a_measure_off_in_the_config_leaves_it_nan() -> None:
    from rostrum.config import load_config

    cfg = load_config(overrides={"features": {"voice_quality": {"cpp": False}}})
    samples, words = utterance([(0.4, 0.0)])
    table = VQ.word_voice_quality(
        samples, SR, words, grid=GRID, config=cfg, floor_hz=100.0, ceiling_hz=250.0
    )
    assert np.isnan(table.loc[0, "cpps_db"])
    assert np.isfinite(table.loc[0, "hnr_db"])
