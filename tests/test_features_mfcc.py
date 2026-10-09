"""MFCCs (spec 9.1 MFCC, 9.2 CMVN).

The filterbank, the DCT and the derivative window are checked on analytic
inputs, and the whole static chain is cross-checked against ``librosa``
configured to match - which is the evidence that the hand-written filterbank is
the standard one and not an approximation of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features import mfcc as M
from rostrum.features.frames import FrameGrid
from rostrum.features.spectral import DEFAULT_N_FFT
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()
SEED = 20261006


def voiced(f_hz: float = 150.0, seconds: float = 1.5, sr: int = SR) -> np.ndarray:
    """A harmonic-rich buzz: something with real cepstral structure."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    out = np.zeros_like(t)
    for harmonic, amplitude in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)):
        out += amplitude * np.sin(2.0 * np.pi * f_hz * harmonic * t)
    rng = np.random.default_rng(SEED)
    return 0.3 * (out / 1.93 + 0.01 * rng.standard_normal(t.size))


# --------------------------------------------------------------------------- #
# Mel scale and filterbank
# --------------------------------------------------------------------------- #


def test_mel_scale_round_trips() -> None:
    hz = np.array([20.0, 100.0, 1000.0, 4000.0, 8000.0])
    assert M.mel_to_hz(M.hz_to_mel(hz)) == pytest.approx(hz, rel=1e-9)


def test_mel_of_zero_hz_is_zero() -> None:
    assert float(M.hz_to_mel(0.0)) == pytest.approx(0.0)


def test_filterbank_has_the_requested_shape_and_unit_peaks() -> None:
    freqs = np.fft.rfftfreq(DEFAULT_N_FFT, d=1.0 / SR)
    bank = M.mel_filterbank(freqs, n_mels=40)
    assert bank.shape == (40, freqs.size)
    assert bank.min() >= 0.0
    # Unit height, not unit area: every filter must reach close to 1.0.
    assert bank.max(axis=1).min() > 0.5


def test_filterbank_rejects_an_inverted_range() -> None:
    freqs = np.fft.rfftfreq(DEFAULT_N_FFT, d=1.0 / SR)
    with pytest.raises(ValueError, match="fmax_hz"):
        M.mel_filterbank(freqs, fmin_hz=4000.0, fmax_hz=20.0)


def test_filterbank_matches_librosa_with_matched_settings() -> None:
    librosa = pytest.importorskip("librosa")
    freqs = np.fft.rfftfreq(DEFAULT_N_FFT, d=1.0 / SR)
    ours = M.mel_filterbank(freqs, n_mels=40, fmin_hz=20.0, fmax_hz=8000.0)
    theirs = librosa.filters.mel(
        sr=SR, n_fft=DEFAULT_N_FFT, n_mels=40, fmin=20.0, fmax=8000.0, htk=True, norm=None
    )
    assert np.abs(ours - theirs).max() < 1e-6


def test_static_mfcc_matches_librosa_with_matched_settings() -> None:
    """The reference cross-check for the whole static chain.

    ``librosa`` is matched on every choice that differs by convention: HTK mel
    scale, unit-height filters (``norm=None``), power spectrum, no dB clipping,
    and the framing offset ``librosa`` introduces by centring the window inside
    the FFT (see ``tests/test_features_spectral.py``).
    """
    librosa = pytest.importorskip("librosa")
    from scipy.fft import dct

    from rostrum.features.spectral import stft_magnitude

    x = voiced(seconds=1.0)
    offset = (DEFAULT_N_FFT - GRID.window_samples) // 2
    magnitude, freqs = stft_magnitude(x[offset:], SR, grid=GRID)
    bank = M.mel_filterbank(freqs, n_mels=40, fmin_hz=20.0, fmax_hz=8000.0)
    ours = dct(
        np.log10(np.maximum((magnitude**2) @ bank.T, M.LOG_FLOOR)), type=2, axis=1, norm="ortho"
    )[:, : M.DEFAULT_N_MFCC]

    mel = librosa.feature.melspectrogram(
        y=x,
        sr=SR,
        n_fft=DEFAULT_N_FFT,
        hop_length=GRID.hop_samples,
        win_length=GRID.window_samples,
        center=False,
        n_mels=40,
        fmin=20.0,
        fmax=8000.0,
        htk=True,
        norm=None,
        power=2.0,
    )
    theirs = (
        dct(
            librosa.power_to_db(mel, ref=1.0, amin=M.LOG_FLOOR, top_db=None).T,
            type=2,
            axis=1,
            norm="ortho",
        )[:, : M.DEFAULT_N_MFCC]
        / 10.0
    )
    n = min(ours.shape[0], theirs.shape[0])
    assert np.abs(ours[:n] - theirs[:n]).max() < 1e-6


# --------------------------------------------------------------------------- #
# CMVN
# --------------------------------------------------------------------------- #


def test_cmvn_centres_and_scales_each_column() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(loc=[5.0, -3.0], scale=[2.0, 7.0], size=(500, 2))
    out = M.cmvn(x)
    assert out.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert out.std(axis=0) == pytest.approx([1.0, 1.0], abs=1e-9)


def test_cmvn_uses_only_the_active_frames() -> None:
    x = np.concatenate([np.full((10, 1), 100.0), np.zeros((10, 1))])
    active = np.concatenate([np.ones(10, dtype=bool), np.zeros(10, dtype=bool)])
    out = M.cmvn(x, active)
    # The active block is constant, so it is centred on itself and the silence
    # lands far away - the opposite of what using every frame would give.
    assert out[:10] == pytest.approx(0.0)
    assert out[10:].min() < -1.0


def test_cmvn_leaves_a_constant_column_finite() -> None:
    out = M.cmvn(np.full((20, 1), 7.0))
    assert np.isfinite(out).all()
    assert out == pytest.approx(0.0)


def test_cmvn_of_empty_input_is_empty() -> None:
    assert M.cmvn(np.empty((0, 13))).shape == (0, 13)


# --------------------------------------------------------------------------- #
# Derivatives
# --------------------------------------------------------------------------- #


def test_delta_of_a_constant_is_zero() -> None:
    assert M.deltas(np.full((10, 3), 2.0)) == pytest.approx(0.0)


def test_delta_of_a_ramp_is_the_slope_per_frame() -> None:
    ramp = np.arange(20, dtype=np.float64)[:, None] * 3.0
    out = M.deltas(ramp, window=2)
    # Away from the replicated edges the regression recovers the exact slope.
    assert out[3:-3, 0] == pytest.approx(3.0)


def test_delta_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="window"):
        M.deltas(np.zeros((5, 2)), window=0)


def test_second_derivative_of_a_ramp_is_zero() -> None:
    ramp = np.arange(30, dtype=np.float64)[:, None]
    assert M.deltas(M.deltas(ramp))[5:-5, 0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# The track and the table
# --------------------------------------------------------------------------- #


def test_track_has_thirteen_static_coefficients_and_both_derivatives() -> None:
    track = M.mfcc_track(voiced(), SR, grid=GRID)
    assert track.static.shape == (track.n_frames, 13)
    assert track.delta is not None and track.delta.shape == track.static.shape
    assert track.delta_delta is not None
    assert track.stacked().shape == (track.n_frames, 39)
    assert len(track.column_names()) == 39
    assert track.column_names()[0] == "mfcc_00"
    assert track.column_names()[-1] == "mfcc_dd_12"


def test_track_is_normalised_over_its_active_frames() -> None:
    track = M.mfcc_track(voiced(), SR, grid=GRID)
    active = track.static[track.active]
    assert active.mean(axis=0) == pytest.approx(np.zeros(13), abs=1e-9)
    assert active.std(axis=0) == pytest.approx(np.ones(13), abs=1e-9)


def test_empty_audio_yields_an_empty_track() -> None:
    track = M.mfcc_track(np.zeros(0, dtype=np.float32), SR, grid=GRID)
    assert track.n_frames == 0
    assert track.stacked().shape[0] == 0


def test_word_means_have_one_row_per_word_and_the_expected_columns() -> None:
    track = M.mfcc_track(voiced(), SR, grid=GRID, channel_id="ch1")
    words = [
        Word(idx=0, text="a", t_start=0.1, t_end=0.5, phones=["AH1"], n_syllables=1),
        Word(idx=1, text="b", t_start=0.6, t_end=1.0, phones=["AH1"], n_syllables=1),
    ]
    table = M.word_mfcc(track, words, grid=GRID, file_id="f1")
    assert list(table.columns) == [*M.mfcc_column_names(), "mfcc_n_frames"]
    assert list(table.index) == [0, 1]
    assert table.attrs["unit"] == "word"
    assert table.attrs["channel_id"] == "ch1"
    assert table.attrs["channel_sensitive"] is True
    assert (table["mfcc_n_frames"] > 0).all()


def test_a_word_with_no_frames_is_nan() -> None:
    track = M.mfcc_track(voiced(seconds=0.5), SR, grid=GRID)
    words = [Word(idx=0, text="x", t_start=9.0, t_end=9.3, phones=["AH1"], n_syllables=1)]
    table = M.word_mfcc(track, words, grid=GRID)
    assert np.isnan(table.loc[0, "mfcc_01"])
    assert table.loc[0, "mfcc_n_frames"] == 0


def test_cmvn_makes_a_constant_gain_change_invisible() -> None:
    """A stationary level offset is exactly what per-utterance CMVN removes."""
    x = voiced()
    quiet = M.mfcc_track(x, SR, grid=GRID)
    loud = M.mfcc_track(x * 0.25, SR, grid=GRID)
    assert np.abs(quiet.static[quiet.active] - loud.static[loud.active]).max() < 1e-6
