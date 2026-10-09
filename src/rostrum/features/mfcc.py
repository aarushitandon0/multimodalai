"""Mel-frequency cepstral coefficients (spec 9.1 MFCC).

13 coefficients including :math:`c_0`, plus first and second time derivatives,
on the project frame grid (10 ms hop, 25 ms window; CLAUDE.md invariant 3).
The magnitude spectrogram comes from :func:`rostrum.features.spectral.stft_magnitude`,
so MFCCs and the spectral features of a file are computed from exactly the same
frames.

The chain is the textbook one, written out here rather than delegated so that
the filterbank, the normalisation and the derivative window are all visible and
pinned:

1. 40 triangular mel filters between 20 Hz and 8 kHz, HTK mel scale
   :math:`m = 2595\\log_{10}(1 + f/700)`, unit-height (not unit-area), applied
   to the power spectrum;
2. :math:`\\log_{10}` of the filter outputs, floored so silence stays finite;
3. orthonormal DCT-II, first 13 coefficients kept;
4. **per-utterance CMVN** (CLAUDE.md invariant 6): each coefficient's mean and
   standard deviation over the recording's *active* frames are removed, so a
   level or channel offset that is constant over the file cannot look like a
   flaw;
5. derivatives by the HTK regression formula over a two-frame window.

CMVN is applied to the static coefficients, before differencing. A constant
offset makes no difference to a derivative anyway, and normalising first keeps
the delta of coefficient *k* in the same units as the static coefficient *k*.

**Channel rule (CLAUDE.md invariant 7).** MFCCs are channel sensitive. CMVN
removes a *stationary* channel, not a different microphone's resonances or a
different codec, so these columns may only be compared within one
``channel_id``. They never trigger a flaw against the multi-speaker envelope.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features.frames import FrameGrid
from rostrum.features.spectral import stft_magnitude

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DELTA_WINDOW",
    "DEFAULT_FMAX_HZ",
    "DEFAULT_FMIN_HZ",
    "DEFAULT_N_MELS",
    "DEFAULT_N_MFCC",
    "MfccTrack",
    "cmvn",
    "deltas",
    "hz_to_mel",
    "mel_filterbank",
    "mel_to_hz",
    "mfcc_column_names",
    "mfcc_track",
    "word_mfcc",
]

DEFAULT_N_MFCC: int = 13
"""Coefficients kept, including ``c0`` (spec 9.1)."""

DEFAULT_N_MELS: int = 40
"""Triangular mel filters."""

DEFAULT_FMIN_HZ: float = 20.0
"""Low edge of the filterbank."""

DEFAULT_FMAX_HZ: float = 8000.0
"""High edge of the filterbank; Nyquist at the project's 16 kHz."""

DEFAULT_DELTA_WINDOW: int = 2
"""Half-width of the HTK derivative regression window, in frames."""

LOG_FLOOR: float = 1e-10
"""Floor on filterbank power before the logarithm."""


# --------------------------------------------------------------------------- #
# Mel filterbank
# --------------------------------------------------------------------------- #


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """HTK mel scale: ``2595 * log10(1 + hz / 700)``."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    freqs: np.ndarray,
    *,
    n_mels: int = DEFAULT_N_MELS,
    fmin_hz: float = DEFAULT_FMIN_HZ,
    fmax_hz: float = DEFAULT_FMAX_HZ,
) -> np.ndarray:
    """Unit-height triangular mel filters on the FFT bin grid.

    Args:
        freqs: Bin centre frequencies in Hz, ascending.
        n_mels: Number of filters.
        fmin_hz: Low edge of the lowest filter.
        fmax_hz: High edge of the highest filter.

    Returns:
        Weight matrix of shape ``(n_mels, freqs.size)``.

    Raises:
        ValueError: ``fmax_hz`` is not above ``fmin_hz``, or ``n_mels`` < 1.
    """
    f = np.asarray(freqs, dtype=np.float64)
    if n_mels < 1:
        raise ValueError(f"n_mels must be at least 1, got {n_mels}")
    if fmax_hz <= fmin_hz:
        raise ValueError(f"fmax_hz ({fmax_hz}) must be above fmin_hz ({fmin_hz})")

    edges_mel = np.linspace(hz_to_mel(fmin_hz), hz_to_mel(fmax_hz), n_mels + 2)
    edges_hz = mel_to_hz(edges_mel)
    bank = np.zeros((n_mels, f.size), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]
        if mid > lo:
            rising = (f >= lo) & (f <= mid)
            bank[m, rising] = (f[rising] - lo) / (mid - lo)
        if hi > mid:
            falling = (f > mid) & (f <= hi)
            bank[m, falling] = (hi - f[falling]) / (hi - mid)
    return bank


# --------------------------------------------------------------------------- #
# Normalisation and derivatives
# --------------------------------------------------------------------------- #


def cmvn(coefficients: np.ndarray, active: np.ndarray | None = None) -> np.ndarray:
    """Per-utterance cepstral mean and variance normalisation (invariant 6).

    Args:
        coefficients: ``(n_frames, n_coefficients)``.
        active: Boolean mask of the frames the statistics are taken over -
            normally the frames with speech energy, so a long leading silence
            cannot drag the mean. ``None`` uses every frame.

    Returns:
        The input with each column centred and scaled to unit standard
        deviation. A column with no spread is centred only; dividing by a zero
        spread would turn a constant into ``inf``.
    """
    x = np.asarray(coefficients, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    mask = np.ones(x.shape[0], dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if mask.sum() < 2:
        mask = np.ones(x.shape[0], dtype=bool)
    stats = x[mask]
    mean = np.nanmean(stats, axis=0)
    std = np.nanstd(stats, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    return (x - mean) / std


def deltas(coefficients: np.ndarray, *, window: int = DEFAULT_DELTA_WINDOW) -> np.ndarray:
    """Time derivative by the HTK regression formula.

    .. math::

        d_t = \\frac{\\sum_{n=1}^{N} n\\,(c_{t+n} - c_{t-n})}{2\\sum_{n=1}^{N} n^2}

    Edge frames are handled by replicating the first and last frame, which is
    what HTK does: it keeps the derivative defined everywhere without inventing
    a trend at the boundaries.

    Args:
        coefficients: ``(n_frames, n_coefficients)``.
        window: Half-width ``N`` in frames, at least 1.

    Returns:
        An array shaped like the input.

    Raises:
        ValueError: ``window`` is below 1.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1 frame, got {window}")
    x = np.asarray(coefficients, dtype=np.float64)
    if x.shape[0] == 0:
        return x.copy()
    padded = np.pad(x, ((window, window), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for n in range(1, window + 1):
        ahead = padded[window + n : window + n + x.shape[0]]
        behind = padded[window - n : window - n + x.shape[0]]
        out += n * (ahead - behind)
    return out / (2.0 * sum(n * n for n in range(1, window + 1)))


def mfcc_column_names(n_mfcc: int = DEFAULT_N_MFCC, *, derivatives: bool = True) -> list[str]:
    """Column names for a word-level MFCC table, in matrix order."""
    names = [f"mfcc_{k:02d}" for k in range(n_mfcc)]
    if derivatives:
        names += [f"mfcc_d_{k:02d}" for k in range(n_mfcc)]
        names += [f"mfcc_dd_{k:02d}" for k in range(n_mfcc)]
    return names


# --------------------------------------------------------------------------- #
# The track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MfccTrack:
    """Frame-level cepstral features for one recording.

    Attributes:
        times: Frame centre times, seconds.
        static: ``(n_frames, n_mfcc)`` CMVN-normalised coefficients.
        delta: First derivative, same shape, or ``None``.
        delta_delta: Second derivative, same shape, or ``None``.
        active: Frames with speech energy, the ones CMVN was fitted on.
        n_mfcc: Number of static coefficients.
        channel_id: Channel tag; comparisons outside it are forbidden
            (invariant 7).
    """

    times: np.ndarray
    static: np.ndarray
    delta: np.ndarray | None
    delta_delta: np.ndarray | None
    active: np.ndarray
    n_mfcc: int = DEFAULT_N_MFCC
    channel_id: str = ""

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.static.shape[0])

    def stacked(self) -> np.ndarray:
        """Static, delta and delta-delta side by side, ``(n_frames, n_columns)``."""
        blocks = [self.static]
        if self.delta is not None:
            blocks.append(self.delta)
        if self.delta_delta is not None:
            blocks.append(self.delta_delta)
        return np.concatenate(blocks, axis=1)

    def column_names(self) -> list[str]:
        """Names matching :meth:`stacked`."""
        names = [f"mfcc_{k:02d}" for k in range(self.n_mfcc)]
        if self.delta is not None:
            names += [f"mfcc_d_{k:02d}" for k in range(self.n_mfcc)]
        if self.delta_delta is not None:
            names += [f"mfcc_dd_{k:02d}" for k in range(self.n_mfcc)]
        return names


def _mfcc_conf(config: Config | None) -> dict[str, Any]:
    """MFCC parameters, from ``configs/features.yaml`` or the spec defaults."""
    conf: dict[str, Any] = {
        "n_mfcc": DEFAULT_N_MFCC,
        "n_mels": DEFAULT_N_MELS,
        "fmin_hz": DEFAULT_FMIN_HZ,
        "fmax_hz": DEFAULT_FMAX_HZ,
        "deltas": True,
        "delta_deltas": True,
        "n_fft": 512,
    }
    if config is None:
        return conf
    conf["n_fft"] = int(config.features.get("frame", {}).get("n_fft", 512))
    section = config.features.get("mfcc", {})
    conf["n_mfcc"] = int(section.get("n_mfcc", DEFAULT_N_MFCC))
    conf["n_mels"] = int(section.get("n_mels", DEFAULT_N_MELS))
    conf["fmin_hz"] = float(section.get("fmin_hz", DEFAULT_FMIN_HZ))
    conf["fmax_hz"] = float(section.get("fmax_hz", DEFAULT_FMAX_HZ))
    conf["deltas"] = bool(section.get("deltas", True))
    conf["delta_deltas"] = bool(section.get("delta_deltas", True))
    return conf


def mfcc_track(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    config: Config | None = None,
    n_frames: int | None = None,
    channel_id: str = "",
    active_floor_db: float = 40.0,
) -> MfccTrack:
    """Extract CMVN-normalised MFCCs and their derivatives for one recording.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        config: Loaded config; ``configs/features.yaml`` ``mfcc:`` supplies the
            filterbank and which derivatives to compute.
        n_frames: Output length, so this track matches the other tracks of the
            same file.
        channel_id: Channel tag carried into the table.
        active_floor_db: A frame counts as active, and so enters the CMVN
            statistics, when its broadband energy is within this many dB of the
            recording's loudest frame.

    Returns:
        An :class:`MfccTrack`.
    """
    from scipy.fft import dct

    grid = grid or FrameGrid(sample_rate=sr)
    conf = _mfcc_conf(config)
    magnitude, freqs = stft_magnitude(
        samples, sr, grid=grid, n_fft=conf["n_fft"], n_frames=n_frames
    )
    n_rows = magnitude.shape[0]
    if n_rows == 0:
        empty = np.empty((0, conf["n_mfcc"]), dtype=np.float64)
        return MfccTrack(
            times=np.empty(0, dtype=np.float64),
            static=empty,
            delta=empty.copy() if conf["deltas"] else None,
            delta_delta=empty.copy() if conf["delta_deltas"] else None,
            active=np.empty(0, dtype=bool),
            n_mfcc=conf["n_mfcc"],
            channel_id=channel_id,
        )

    bank = mel_filterbank(
        freqs, n_mels=conf["n_mels"], fmin_hz=conf["fmin_hz"], fmax_hz=conf["fmax_hz"]
    )
    mel_power = (magnitude**2) @ bank.T
    log_mel = np.log10(np.maximum(mel_power, LOG_FLOOR))
    static = dct(log_mel, type=2, axis=1, norm="ortho")[:, : conf["n_mfcc"]]

    frame_power = (magnitude**2).sum(axis=1)
    peak = float(frame_power.max()) if frame_power.size else 0.0
    if peak > 0.0:
        active = frame_power >= peak * 10.0 ** (-active_floor_db / 10.0)
    else:
        active = np.zeros(n_rows, dtype=bool)

    static = cmvn(static, active)
    delta = deltas(static) if conf["deltas"] else None
    delta_delta = deltas(delta) if (conf["delta_deltas"] and delta is not None) else None
    return MfccTrack(
        times=grid.centres(n_rows),
        static=static,
        delta=delta,
        delta_delta=delta_delta,
        active=active,
        n_mfcc=conf["n_mfcc"],
        channel_id=channel_id,
    )


def word_mfcc(
    track: MfccTrack,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    file_id: str = "",
    channel_id: str | None = None,
) -> FeatureTable:
    """Mean of every cepstral coefficient over each word's frames (spec 9.1).

    A word with no frames gets ``NaN`` in every column rather than a zero: an
    unmeasured word is not an average one.
    """
    import pandas as pd

    grid = grid or FrameGrid()
    matrix = track.stacked()
    names = track.column_names()
    rows = np.full((len(words), len(names)), np.nan, dtype=np.float64)
    counts = np.zeros(len(words), dtype=np.int64)
    for row, word in enumerate(words):
        idx = grid.frame_indices(word.t_start, word.t_end, track.n_frames)
        if idx.size == 0:
            continue
        counts[row] = int(idx.size)
        rows[row] = matrix[idx].mean(axis=0)

    table = pd.DataFrame(rows, columns=names)
    table["mfcc_n_frames"] = counts
    table.index = pd.Index([w.idx for w in words], name="unit_idx", dtype="int64")
    table.attrs.update(
        unit="word",
        file_id=file_id,
        channel_id=track.channel_id if channel_id is None else channel_id,
        channel_sensitive=True,
    )
    return table
