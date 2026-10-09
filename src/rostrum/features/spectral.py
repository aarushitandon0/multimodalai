"""Spectral features from the short-time Fourier transform (spec 9.1 SPEC).

Four quantities, all computed on the project frame grid (10 ms hop, 25 ms
window, 512-point FFT; CLAUDE.md invariant 3):

* **centroid** - the magnitude-weighted mean frequency, the "brightness" of the
  frame;
* **flux** - how far the normalised magnitude spectrum moved since the previous
  frame, which rises at consonantal onsets and falls in sustained vowels;
* **tilt** - the slope of a least-squares line through the log-magnitude
  spectrum from 0 to 4 kHz, in dB/kHz. Mumbled or off-axis speech loses high
  frequencies and the slope steepens;
* **band-energy ratio** - energy in 0-1 kHz over energy in 1-4 kHz, in dB. The
  same effect as tilt, measured without assuming the spectrum is a straight
  line.

**Channel rule (CLAUDE.md invariant 7).** Every feature here is channel
sensitive: a different microphone, room or encoder moves all four. They may
only be compared within one ``channel_id`` - a clean file against its own
injected edit, or a speaker's clean take against their own flawed take - never
against the multi-speaker envelope.

Pure functions: audio and an alignment in, frame arrays and a per-word table
out. :func:`stft_magnitude` is the single STFT in the project and is reused by
:mod:`rostrum.features.mfcc`, so both see exactly the same frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features.frames import FrameGrid

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_N_FFT",
    "DEFAULT_ROLLOFF_PERCENT",
    "DEFAULT_TILT_MAX_HZ",
    "SPECTRAL_WORD_COLUMNS",
    "SpectralTrack",
    "band_energy_ratio_db",
    "spectral_centroid_hz",
    "spectral_flux",
    "spectral_rolloff_hz",
    "spectral_tilt_db_per_khz",
    "spectral_track",
    "stft_magnitude",
    "word_spectral",
]

DEFAULT_N_FFT: int = 512
"""FFT length: the 25 ms window (400 samples at 16 kHz) zero-padded to 512."""

DEFAULT_BANDS: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1000.0), (1000.0, 4000.0))
"""Numerator and denominator bands of the band-energy ratio (spec 9.1)."""

DEFAULT_TILT_MAX_HZ: float = 4000.0
"""Upper edge of the tilt fit; above 4 kHz channels differ too wildly to compare."""

DEFAULT_ROLLOFF_PERCENT: float = 0.85
"""Fraction of total magnitude below the roll-off frequency."""

EPS: float = 1e-12
"""Floor that keeps ``log10`` finite on digital silence."""

SPECTRAL_WORD_COLUMNS: tuple[str, ...] = (
    "spectral_centroid_hz",
    "spectral_flux",
    "spectral_tilt_db_per_khz",
    "band_energy_ratio_db",
    "spectral_rolloff_hz",
    "spectral_n_frames",
)
"""Word-level spectral columns; all channel sensitive (invariant 7)."""


# --------------------------------------------------------------------------- #
# The one STFT
# --------------------------------------------------------------------------- #


def _periodic_hann(n: int) -> np.ndarray:
    """Periodic (DFT-even) Hann window of length ``n``.

    ``np.hanning`` is the symmetric variant, which is meant for filter design;
    spectral analysis wants the periodic one, as ``scipy.signal.get_window`` and
    ``librosa`` use. The difference is one sample but it biases the DC bin.
    """
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def stft_magnitude(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    n_fft: int = DEFAULT_N_FFT,
    n_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrogram on the project frame grid.

    Frame ``i`` is the window of :attr:`FrameGrid.window_samples` samples
    starting at ``i * hop_samples``, tapered by a *periodic* Hann window and
    zero-padded to ``n_fft``. No centring and no reflection padding, so a word's
    frames are exactly the frames inside the word
    (``configs/features.yaml``: ``center: false``).

    The zero padding sits after the window rather than around it. A shift inside
    a zero-padded buffer only rotates the phase, so the magnitude is unchanged -
    verified against ``librosa.stft(center=False)``, which pads the window
    symmetrically and therefore starts its first frame
    ``(n_fft - window) // 2`` samples in: correcting for that offset, the two
    magnitude spectrograms agree to 7e-15 (see
    ``tests/test_features_spectral.py``). This project keeps the frame at
    ``i * hop`` because the whole time model, including every reported
    timestamp, is anchored there.

    Args:
        samples: Mono audio.
        sr: Sample rate; must match the grid.
        grid: Frame grid; defaults to the project grid at ``sr``.
        n_fft: FFT length, at least the window length.
        n_frames: Output length. Defaults to the number of whole windows that
            fit. A longer request is zero-padded, a shorter one truncated, so
            every feature of a recording can share one frame count.

    Returns:
        ``(magnitude, freqs)`` where ``magnitude`` has shape
        ``(n_frames, n_fft // 2 + 1)`` and ``freqs`` is the bin centre
        frequencies in Hz.

    Raises:
        ValueError: ``n_fft`` is shorter than the analysis window.
    """
    grid = grid or FrameGrid(sample_rate=sr)
    window_n = grid.window_samples
    if n_fft < window_n:
        raise ValueError(f"n_fft ({n_fft}) must be at least the window ({window_n} samples)")

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    count = grid.n_frames_for(x.size) if n_frames is None else int(n_frames)
    count = max(count, 0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    if count == 0:
        return np.empty((0, freqs.size), dtype=np.float64), freqs

    taper = _periodic_hann(window_n)
    frames = np.zeros((count, n_fft), dtype=np.float64)
    hop = grid.hop_samples
    for i in range(count):
        start = i * hop
        chunk = x[start : start + window_n]
        if chunk.size == 0:
            continue
        frames[i, : chunk.size] = chunk * taper[: chunk.size]
    return np.abs(np.fft.rfft(frames, n=n_fft, axis=1)), freqs


# --------------------------------------------------------------------------- #
# Frame-level features
# --------------------------------------------------------------------------- #


def spectral_centroid_hz(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Magnitude-weighted mean frequency per frame, Hz.

    A frame with no energy has no centroid, and gets ``NaN`` rather than 0 Hz -
    a silent frame is not a dark frame.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    if mag.size == 0:
        return np.empty(0, dtype=np.float64)
    total = mag.sum(axis=1)
    out = np.full(mag.shape[0], np.nan, dtype=np.float64)
    ok = total > EPS
    out[ok] = (mag[ok] @ np.asarray(freqs, dtype=np.float64)) / total[ok]
    return out


def spectral_flux(magnitude: np.ndarray) -> np.ndarray:
    """Euclidean distance between successive L1-normalised spectra.

    Normalising each spectrum before differencing makes flux a measure of
    spectral *shape* change rather than of loudness change, so a gain ramp does
    not register as articulation. Frame 0 has no predecessor and is ``NaN``;
    frames adjacent to silence are ``NaN`` for the same reason.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    if mag.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    out = np.full(mag.shape[0], np.nan, dtype=np.float64)
    total = mag.sum(axis=1)
    ok = total > EPS
    norm = np.zeros_like(mag)
    norm[ok] = mag[ok] / total[ok, None]
    both = ok[1:] & ok[:-1]
    diff = np.linalg.norm(norm[1:] - norm[:-1], axis=1)
    out[1:][both] = diff[both]
    return out


def spectral_tilt_db_per_khz(
    magnitude: np.ndarray, freqs: np.ndarray, *, max_hz: float = DEFAULT_TILT_MAX_HZ
) -> np.ndarray:
    """Least-squares slope of the log-magnitude spectrum, dB per kHz.

    The fit runs from DC to ``max_hz``. The DC bin itself is dropped: it carries
    any residual offset rather than spectral shape, and loudness-normalised
    audio can leave a large one.

    Args:
        magnitude: Magnitude spectrogram, ``(n_frames, n_bins)``.
        freqs: Bin frequencies, Hz.
        max_hz: Upper edge of the fit.

    Returns:
        One slope per frame, ``NaN`` for frames without energy.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    if mag.size == 0:
        return np.empty(0, dtype=np.float64)
    band = (f > 0.0) & (f <= float(max_hz))
    out = np.full(mag.shape[0], np.nan, dtype=np.float64)
    if band.sum() < 2:
        return out

    khz = f[band] / 1000.0
    ok = mag.sum(axis=1) > EPS
    if not ok.any():
        return out
    db = 20.0 * np.log10(np.maximum(mag[np.ix_(ok, band)], EPS))
    # One shared design matrix: every usable frame is solved in a single lstsq.
    design = np.column_stack([khz, np.ones_like(khz)])
    coefficients, *_ = np.linalg.lstsq(design, db.T, rcond=None)
    out[ok] = coefficients[0]
    return out


def band_energy_ratio_db(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    *,
    bands: tuple[tuple[float, float], tuple[float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    """Energy ratio between two bands, in dB (spec 9.1: 0-1 kHz over 1-4 kHz).

    Positive values mean the low band dominates, which is the signature of
    muffled or distant speech. Expressed in dB so the feature is symmetric
    about 0 and additive in level.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    if mag.size == 0:
        return np.empty(0, dtype=np.float64)
    (lo0, lo1), (hi0, hi1) = bands
    low = (f >= lo0) & (f < lo1)
    high = (f >= hi0) & (f < hi1)
    out = np.full(mag.shape[0], np.nan, dtype=np.float64)
    if not low.any() or not high.any():
        return out
    power = mag**2
    e_low = power[:, low].sum(axis=1)
    e_high = power[:, high].sum(axis=1)
    ok = (e_low > EPS) & (e_high > EPS)
    out[ok] = 10.0 * np.log10(e_low[ok] / e_high[ok])
    return out


def spectral_rolloff_hz(
    magnitude: np.ndarray, freqs: np.ndarray, *, percent: float = DEFAULT_ROLLOFF_PERCENT
) -> np.ndarray:
    """Frequency below which ``percent`` of the frame's magnitude lies, Hz."""
    mag = np.asarray(magnitude, dtype=np.float64)
    if mag.size == 0:
        return np.empty(0, dtype=np.float64)
    f = np.asarray(freqs, dtype=np.float64)
    total = mag.sum(axis=1)
    out = np.full(mag.shape[0], np.nan, dtype=np.float64)
    ok = total > EPS
    if not ok.any():
        return out
    cumulative = np.cumsum(mag[ok], axis=1)
    target = float(percent) * total[ok][:, None]
    index = np.argmax(cumulative >= target, axis=1)
    out[ok] = f[index]
    return out


# --------------------------------------------------------------------------- #
# The track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectralTrack:
    """Frame-level spectral features for one recording.

    Attributes:
        times: Frame centre times, seconds.
        centroid_hz: Spectral centroid per frame.
        flux: Spectral flux per frame; frame 0 is ``NaN``.
        tilt_db_per_khz: Log-magnitude slope per frame.
        band_ratio_db: Low-over-high band energy ratio per frame, dB.
        rolloff_hz: Roll-off frequency per frame.
        channel_id: Channel these values belong to; comparisons outside it are
            forbidden (invariant 7).
    """

    times: np.ndarray
    centroid_hz: np.ndarray
    flux: np.ndarray
    tilt_db_per_khz: np.ndarray
    band_ratio_db: np.ndarray
    rolloff_hz: np.ndarray
    channel_id: str = ""

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.centroid_hz.size)


def _spectral_conf(config: Config | None) -> dict[str, Any]:
    """Spectral parameters, from ``configs/features.yaml`` or the spec defaults."""
    conf: dict[str, Any] = {
        "n_fft": DEFAULT_N_FFT,
        "bands": DEFAULT_BANDS,
        "rolloff_percent": DEFAULT_ROLLOFF_PERCENT,
        "tilt_max_hz": DEFAULT_TILT_MAX_HZ,
    }
    if config is None:
        return conf
    frame = config.features.get("frame", {})
    conf["n_fft"] = int(frame.get("n_fft", DEFAULT_N_FFT))
    section = config.features.get("spectral", {})
    bands = section.get("band_energy_ratios")
    if bands and len(bands) >= 2:
        conf["bands"] = (
            (float(bands[0][0]), float(bands[0][1])),
            (float(bands[1][0]), float(bands[1][1])),
        )
        # The tilt fit stops where the comparison bands stop.
        conf["tilt_max_hz"] = float(bands[1][1])
    conf["rolloff_percent"] = float(section.get("rolloff_percent", DEFAULT_ROLLOFF_PERCENT))
    return conf


def spectral_track(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    config: Config | None = None,
    n_frames: int | None = None,
    channel_id: str = "",
) -> SpectralTrack:
    """Compute every spectral feature of one recording from a single STFT.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        config: Loaded config; ``configs/features.yaml`` ``spectral:`` supplies
            the bands and the roll-off percentile.
        n_frames: Output length, so this track can be made the same length as
            the pitch and energy tracks of the same file.
        channel_id: Channel tag carried into the table for the channel rule.

    Returns:
        A :class:`SpectralTrack`.
    """
    grid = grid or FrameGrid(sample_rate=sr)
    conf = _spectral_conf(config)
    magnitude, freqs = stft_magnitude(
        samples, sr, grid=grid, n_fft=conf["n_fft"], n_frames=n_frames
    )
    return SpectralTrack(
        times=grid.centres(magnitude.shape[0]),
        centroid_hz=spectral_centroid_hz(magnitude, freqs),
        flux=spectral_flux(magnitude),
        tilt_db_per_khz=spectral_tilt_db_per_khz(magnitude, freqs, max_hz=conf["tilt_max_hz"]),
        band_ratio_db=band_energy_ratio_db(magnitude, freqs, bands=conf["bands"]),
        rolloff_hz=spectral_rolloff_hz(magnitude, freqs, percent=conf["rolloff_percent"]),
        channel_id=channel_id,
    )


def word_spectral(
    track: SpectralTrack,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    file_id: str = "",
    channel_id: str | None = None,
) -> FeatureTable:
    """Aggregate a spectral track to one row per word.

    Each column is the mean over the word's frames, ignoring ``NaN``. Columns
    are :data:`SPECTRAL_WORD_COLUMNS`.
    """
    import pandas as pd

    grid = grid or FrameGrid()
    curves = {
        "spectral_centroid_hz": track.centroid_hz,
        "spectral_flux": track.flux,
        "spectral_tilt_db_per_khz": track.tilt_db_per_khz,
        "band_energy_ratio_db": track.band_ratio_db,
        "spectral_rolloff_hz": track.rolloff_hz,
    }
    rows = []
    for word in words:
        idx = grid.frame_indices(word.t_start, word.t_end, track.n_frames)
        row: dict[str, float] = {}
        for name, curve in curves.items():
            values = curve[idx] if idx.size else np.empty(0)
            ok = np.isfinite(values)
            row[name] = float(values[ok].mean()) if ok.any() else np.nan
        centroid = track.centroid_hz[idx] if idx.size else np.empty(0)
        row["spectral_n_frames"] = int(np.isfinite(centroid).sum())
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(SPECTRAL_WORD_COLUMNS))
    table.index = pd.Index([w.idx for w in words], name="unit_idx", dtype="int64")
    table.attrs.update(
        unit="word",
        file_id=file_id,
        channel_id=track.channel_id if channel_id is None else channel_id,
        channel_sensitive=True,
    )
    return table
