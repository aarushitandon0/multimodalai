"""Spectral features (spec 9.1 SPEC).

Synthetic signals with known spectra: white noise (flat), pink noise (-3 dB per
octave), a pure tone, and silence. One test cross-checks the STFT itself against
``librosa``, which is the reference implementation this module deliberately does
not depend on.
"""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import spectral as S
from rostrum.features.frames import FrameGrid
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()
SEED = 20261006


def white_noise(seconds: float = 2.0, sr: int = SR, seed: int = SEED) -> np.ndarray:
    """Gaussian noise: a flat expected magnitude spectrum."""
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal(int(seconds * sr))).astype(np.float64)


def pink_noise(seconds: float = 2.0, sr: int = SR, seed: int = SEED) -> np.ndarray:
    """Noise with 1/sqrt(f) magnitude: -3 dB per octave, so a negative tilt."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(int(seconds * sr)))
    bins = np.arange(spectrum.size, dtype=np.float64)
    bins[0] = 1.0
    out = np.fft.irfft(spectrum / np.sqrt(bins), n=int(seconds * sr))
    return (0.1 * out / np.std(out)).astype(np.float64)


def tone(f_hz: float, seconds: float = 1.0, sr: int = SR) -> np.ndarray:
    """A steady sine."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    return 0.5 * np.sin(2.0 * np.pi * f_hz * t)


# --------------------------------------------------------------------------- #
# The STFT
# --------------------------------------------------------------------------- #


def test_stft_shape_follows_the_frame_grid() -> None:
    x = white_noise(1.0)
    magnitude, freqs = S.stft_magnitude(x, SR, grid=GRID)
    assert magnitude.shape == (GRID.n_frames_for(x.size), S.DEFAULT_N_FFT // 2 + 1)
    assert freqs[0] == 0.0
    assert freqs[-1] == pytest.approx(SR / 2)


def test_stft_honours_an_explicit_frame_count() -> None:
    magnitude, _ = S.stft_magnitude(white_noise(1.0), SR, grid=GRID, n_frames=50)
    assert magnitude.shape[0] == 50


def test_stft_rejects_an_fft_shorter_than_the_window() -> None:
    with pytest.raises(ValueError, match="n_fft"):
        S.stft_magnitude(white_noise(0.5), SR, grid=GRID, n_fft=128)


def test_stft_of_a_tone_peaks_at_the_tone_frequency() -> None:
    magnitude, freqs = S.stft_magnitude(tone(1000.0), SR, grid=GRID)
    peak_hz = freqs[int(np.argmax(magnitude[10]))]
    assert abs(peak_hz - 1000.0) <= SR / S.DEFAULT_N_FFT


def test_stft_matches_librosa_once_the_framing_offset_is_removed() -> None:
    """Cross-check against the reference implementation (see module docstring).

    ``librosa.stft(center=False)`` pads the 400-sample window symmetrically
    inside the 512-point FFT, so its first frame starts ``(512 - 400) // 2``
    samples in. This project anchors frame ``i`` at ``i * hop`` instead, because
    every reported timestamp is anchored there. Shifting the input by that
    offset makes the two identical.
    """
    librosa = pytest.importorskip("librosa")
    x = white_noise(1.0)
    offset = (S.DEFAULT_N_FFT - GRID.window_samples) // 2
    ours, _ = S.stft_magnitude(x[offset:], SR, grid=GRID)
    theirs = np.abs(
        librosa.stft(
            x,
            n_fft=S.DEFAULT_N_FFT,
            hop_length=GRID.hop_samples,
            win_length=GRID.window_samples,
            center=False,
        )
    ).T
    n = min(ours.shape[0], theirs.shape[0])
    assert np.abs(ours[:n] - theirs[:n]).max() < 1e-9


# --------------------------------------------------------------------------- #
# Tilt: the acceptance test from the prompt
# --------------------------------------------------------------------------- #


def test_white_noise_tilt_is_about_zero_and_pink_noise_tilt_is_clearly_negative() -> None:
    white = S.spectral_track(white_noise(), SR, grid=GRID)
    pink = S.spectral_track(pink_noise(), SR, grid=GRID)
    white_tilt = float(np.nanmedian(white.tilt_db_per_khz))
    pink_tilt = float(np.nanmedian(pink.tilt_db_per_khz))
    assert abs(white_tilt) < 0.5, f"white noise should be flat, got {white_tilt:.3f} dB/kHz"
    assert pink_tilt < white_tilt - 1.0, f"pink {pink_tilt:.3f} vs white {white_tilt:.3f}"


def test_pink_noise_has_the_lower_centroid_and_the_higher_band_ratio() -> None:
    white = S.spectral_track(white_noise(), SR, grid=GRID)
    pink = S.spectral_track(pink_noise(), SR, grid=GRID)
    assert np.nanmedian(pink.centroid_hz) < np.nanmedian(white.centroid_hz)
    assert np.nanmedian(pink.band_ratio_db) > np.nanmedian(white.band_ratio_db)


# --------------------------------------------------------------------------- #
# Frame features in isolation
# --------------------------------------------------------------------------- #


def test_centroid_of_a_single_bin_is_that_bin() -> None:
    freqs = np.array([0.0, 100.0, 200.0, 300.0])
    magnitude = np.array([[0.0, 0.0, 1.0, 0.0]])
    assert S.spectral_centroid_hz(magnitude, freqs)[0] == pytest.approx(200.0)


def test_silence_has_no_centroid_rather_than_zero_hz() -> None:
    freqs = np.array([0.0, 100.0, 200.0])
    assert np.isnan(S.spectral_centroid_hz(np.zeros((1, 3)), freqs)[0])


def test_flux_is_nan_in_the_first_frame_and_zero_for_a_steady_spectrum() -> None:
    magnitude = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    flux = S.spectral_flux(magnitude)
    assert np.isnan(flux[0])
    assert flux[1:] == pytest.approx(0.0, abs=1e-12)


def test_flux_ignores_a_pure_gain_change() -> None:
    magnitude = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    assert S.spectral_flux(magnitude)[1] == pytest.approx(0.0, abs=1e-12)


def test_tilt_recovers_a_known_slope() -> None:
    freqs = np.linspace(0.0, 4000.0, 81)
    # 20*log10(mag) = -6 dB per kHz  =>  mag = 10 ** (-6 * khz / 20)
    magnitude = (10.0 ** (-6.0 * (freqs / 1000.0) / 20.0))[None, :]
    assert S.spectral_tilt_db_per_khz(magnitude, freqs)[0] == pytest.approx(-6.0, abs=1e-6)


def test_band_energy_ratio_of_equal_bands_is_zero_db() -> None:
    freqs = np.array([500.0, 2000.0])
    magnitude = np.array([[1.0, 1.0]])
    bands = ((0.0, 1000.0), (1000.0, 4000.0))
    assert S.band_energy_ratio_db(magnitude, freqs, bands=bands)[0] == pytest.approx(0.0)


def test_band_energy_ratio_doubling_low_power_is_plus_six_db() -> None:
    freqs = np.array([500.0, 2000.0])
    magnitude = np.array([[2.0, 1.0]])
    bands = ((0.0, 1000.0), (1000.0, 4000.0))
    out = S.band_energy_ratio_db(magnitude, freqs, bands=bands)[0]
    assert out == pytest.approx(10.0 * np.log10(4.0))


def test_rolloff_of_a_tone_sits_at_the_tone() -> None:
    track = S.spectral_track(tone(1000.0), SR, grid=GRID)
    assert np.nanmedian(track.rolloff_hz) == pytest.approx(1000.0, abs=100.0)


def test_empty_audio_yields_empty_arrays() -> None:
    track = S.spectral_track(np.zeros(0, dtype=np.float32), SR, grid=GRID)
    assert track.n_frames == 0
    assert track.times.size == 0


# --------------------------------------------------------------------------- #
# Per-word aggregation
# --------------------------------------------------------------------------- #


def words_of(spans: list[tuple[float, float]]) -> list[Word]:
    """Minimal words covering ``spans``."""
    return [
        Word(idx=i, text=f"w{i}", t_start=t0, t_end=t1, phones=["AH1"], n_syllables=1)
        for i, (t0, t1) in enumerate(spans)
    ]


def test_word_spectral_has_one_row_per_word_and_the_documented_columns() -> None:
    track = S.spectral_track(white_noise(1.0), SR, grid=GRID, channel_id="ch1")
    words = words_of([(0.1, 0.4), (0.5, 0.9)])
    table = S.word_spectral(track, words, grid=GRID, file_id="f1")
    assert list(table.columns) == list(S.SPECTRAL_WORD_COLUMNS)
    assert list(table.index) == [0, 1]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "word"
    assert table.attrs["file_id"] == "f1"
    assert table.attrs["channel_id"] == "ch1"
    assert table.attrs["channel_sensitive"] is True


def test_a_word_outside_the_audio_is_nan_not_zero() -> None:
    track = S.spectral_track(white_noise(0.5), SR, grid=GRID)
    table = S.word_spectral(track, words_of([(5.0, 5.3)]), grid=GRID)
    assert np.isnan(table.loc[0, "spectral_centroid_hz"])
    assert table.loc[0, "spectral_n_frames"] == 0


def test_a_louder_word_does_not_change_the_band_ratio() -> None:
    """Band ratio is a shape measure: scaling the whole word must not move it."""
    x = white_noise(1.0)
    loud = x.copy()
    loud[int(0.5 * SR) : int(0.8 * SR)] *= 4.0
    quiet_table = S.word_spectral(
        S.spectral_track(x, SR, grid=GRID), words_of([(0.5, 0.8)]), grid=GRID
    )
    loud_table = S.word_spectral(
        S.spectral_track(loud, SR, grid=GRID), words_of([(0.5, 0.8)]), grid=GRID
    )
    assert loud_table.loc[0, "band_energy_ratio_db"] == pytest.approx(
        quiet_table.loc[0, "band_energy_ratio_db"], abs=0.2
    )
