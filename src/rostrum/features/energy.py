"""Energy features: Praat intensity, RMS cross-check, speaker normalisation.

Spec 9.1 (EN) and 9.2. Two points of care:

* **Absolute dB is meaningless across recordings.** Every file is loudness
  normalised to -23 LUFS, but microphone and room still differ, so energy is
  expressed as dB **re the speaker's own voiced median** and then z-scored per
  recording (invariant 6). Only those relative units reach the envelope.
* **Praat intensity is the measurement; RMS is the witness.** Praat's intensity
  uses a Kaiser-windowed, pitch-tied analysis window, which is what the
  prosody literature reports; a plain RMS curve is computed alongside and their
  correlation is reported, so a broken extractor shows up as a number rather
  than as a silently odd feature.

Pure functions throughout: audio and parameters in, arrays out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rostrum.features.frames import FrameGrid

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "ENERGY_WORD_COLUMNS",
    "EnergyTrack",
    "energy_track",
    "normalise_db",
    "praat_intensity_db",
    "rms_db",
    "track_correlation",
    "word_energy",
]

ENERGY_WORD_COLUMNS: tuple[str, ...] = (
    "energy_mean_z",
    "energy_peak_z",
    "energy_mean_db_rel",
    "energy_peak_db_rel",
    "energy_slope_db_per_s",
    "energy_n_frames",
)
"""Word-level energy columns (spec 9.1: mean, peak, slope dB/s)."""

DEFAULT_FLOOR_DB = -70.0
"""Intensity floor; below this a frame is silence, not quiet speech."""

VOICED_MARGIN_DB = 15.0
"""A frame counts as voiced when it is within this much of the recording's peak.

Used only when the caller supplies no voicing mask. Pitch-derived voicing is
better and :func:`energy_track` takes it when offered.
"""


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def praat_intensity_db(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    min_pitch_hz: float = 50.0,
    floor_db: float = DEFAULT_FLOOR_DB,
    n_frames: int | None = None,
) -> np.ndarray:
    """Praat intensity in dB, mapped onto the frame grid.

    Args:
        samples: Mono audio.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        min_pitch_hz: Praat ties its analysis window to this minimum pitch.
        floor_db: Finite values below this are clamped, so silence becomes the
            floor rather than ``-inf``. Frames Praat did not measure at all stay
            ``NaN``.
        n_frames: Output length; defaults to the grid's count for this audio.

    Returns:
        Intensity per frame in dB, clamped at ``floor_db``, ``NaN`` where Praat
        produced no estimate within one analysis window.
    """
    import parselmouth

    grid = grid or FrameGrid(sample_rate=sr)
    audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float64).ravel())
    frames = grid.n_frames_for(audio.size) if n_frames is None else int(n_frames)
    if frames <= 0:
        return np.empty(0, dtype=np.float64)

    sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
    intensity = sound.to_intensity(minimum_pitch=float(min_pitch_hz), time_step=grid.hop_s)
    values = np.asarray(intensity.values, dtype=np.float64).ravel()
    times = np.asarray(intensity.xs(), dtype=np.float64)

    # Praat computes intensity over a 3.2 / min_pitch window (64 ms at 50 Hz)
    # and emits no frame for the first and last half-window. Accept the nearest
    # estimate within one such window, so the few frames at each edge hold the
    # neighbouring value instead of reading as digital silence; frames further
    # out than that stay NaN rather than being invented.
    tolerance = max(grid.hop_s / 2.0, 3.2 / float(min_pitch_hz))
    out = grid.resample_nearest(times, values, frames, tolerance_s=tolerance)
    return np.where(np.isfinite(out), np.maximum(out, float(floor_db)), np.nan)


def rms_db(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    floor_db: float = DEFAULT_FLOOR_DB,
) -> np.ndarray:
    """Frame RMS in dBFS on the frame grid, as a cross-check on Praat.

    Computed with numpy framing rather than a library call so the grid is
    exactly the project's: frame ``i`` is ``samples[i*hop : i*hop + window]``.
    """
    grid = grid or FrameGrid(sample_rate=sr)
    audio = np.asarray(samples, dtype=np.float64).ravel()
    frames = grid.n_frames_for(audio.size)
    if frames <= 0:
        return np.empty(0, dtype=np.float64)

    hop, win = grid.hop_samples, grid.window_samples
    idx = np.arange(frames)[:, None] * hop + np.arange(win)[None, :]
    windowed = audio[idx] * np.hanning(win)[None, :]
    power = np.mean(windowed**2, axis=1)
    # A Hann window halves the RMS of a stationary signal; undo that so the
    # scale is comparable to an unwindowed RMS (a full-scale sine reads -3 dBFS).
    power = power / np.mean(np.hanning(win) ** 2)
    with np.errstate(divide="ignore"):
        db = 10.0 * np.log10(np.maximum(power, 1e-20))
    return np.maximum(db, float(floor_db))


def track_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation between two frame tracks, ignoring ``NaN``.

    Returns ``None`` when fewer than three frames are comparable or either track
    is constant - a correlation is undefined there, and reporting 0.0 would read
    as disagreement.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    xs, ys = x[ok], y[ok]
    if np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


# --------------------------------------------------------------------------- #
# Normalisation (spec 9.2)
# --------------------------------------------------------------------------- #


def normalise_db(db: np.ndarray, voiced: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Normalise intensity: dB re the voiced median, then z-score (invariant 6).

    Args:
        db: Intensity per frame, dB.
        voiced: Which frames are voiced speech; only these set the reference and
            the spread.

    Returns:
        ``(db_rel, z, voiced_median_db)``. With no voiced frame, both arrays are
        ``NaN`` and the median is ``None`` - a silent recording has no energy
        reference and must not be compared to anything.
    """
    values = np.asarray(db, dtype=np.float64)
    mask = np.asarray(voiced, dtype=bool)
    if not mask.any():
        nan = np.full(values.shape, np.nan, dtype=np.float64)
        return nan, nan.copy(), None

    median = float(np.median(values[mask]))
    rel = values - median
    spread = float(np.std(rel[mask], ddof=0))
    z = rel / spread if spread > 0 else np.zeros_like(rel)
    return rel, z, median


# --------------------------------------------------------------------------- #
# The track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EnergyTrack:
    """Frame-level energy for one recording.

    Attributes:
        times: Frame centre times, seconds.
        db: Praat intensity, dB.
        db_rel: dB re the speaker's voiced median.
        z: :attr:`db_rel` z-scored over voiced frames.
        voiced: Frames counted as voiced speech.
        voiced_median_db: The reference, or ``None`` for a silent recording.
        rms_db: The RMS cross-check track, or ``None``.
        rms_correlation: Correlation between intensity and RMS, or ``None``.
    """

    times: np.ndarray
    db: np.ndarray
    db_rel: np.ndarray
    z: np.ndarray
    voiced: np.ndarray
    voiced_median_db: float | None
    rms_db: np.ndarray | None = None
    rms_correlation: float | None = None

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.db.size)


def energy_track(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    config: Config | None = None,
    voiced: np.ndarray | None = None,
    rms_crosscheck: bool | None = None,
    n_frames: int | None = None,
) -> EnergyTrack:
    """Extract and normalise the energy track of one recording.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        config: Loaded config; ``configs/features.yaml`` ``energy:`` supplies the
            Praat minimum pitch, the floor and the cross-check switch.
        voiced: Voicing mask, normally :attr:`PitchTrack.voiced`. Without it a
            level gate :data:`VOICED_MARGIN_DB` below the recording's peak is
            used, which is adequate for a reference but less precise.
        rms_crosscheck: Compute the RMS witness track. Defaults to the config.
        n_frames: Output length; defaults to the grid's count for this audio.

    Returns:
        An :class:`EnergyTrack` whose ``z`` is what the envelope compares.
    """
    grid = grid or FrameGrid(sample_rate=sr)
    min_pitch, floor_db, want_rms = _energy_conf(config, rms_crosscheck)

    db = praat_intensity_db(
        samples, sr, grid=grid, min_pitch_hz=min_pitch, floor_db=floor_db, n_frames=n_frames
    )
    mask = (
        np.asarray(voiced, dtype=bool) if voiced is not None else _level_gate(db, floor_db=floor_db)
    )
    if mask.shape != db.shape:
        raise ValueError(f"voiced mask has shape {mask.shape}, expected {db.shape}")

    rel, z, median = normalise_db(db, mask)
    rms = rms_db(samples, sr, grid=grid, floor_db=floor_db) if want_rms else None
    correlation = None
    if rms is not None and rms.size == db.size:
        correlation = track_correlation(db, rms)
    return EnergyTrack(
        times=grid.centres(db.size),
        db=db,
        db_rel=rel,
        z=z,
        voiced=mask,
        voiced_median_db=median,
        rms_db=rms,
        rms_correlation=correlation,
    )


def _level_gate(db: np.ndarray, *, floor_db: float) -> np.ndarray:
    """Frames within :data:`VOICED_MARGIN_DB` of the peak, excluding the floor."""
    if db.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.nanmax(db))
    if not np.isfinite(peak) or peak <= floor_db:
        return np.zeros(db.shape, dtype=bool)
    return (db > floor_db) & (db >= peak - VOICED_MARGIN_DB)


def _energy_conf(config: Config | None, rms_crosscheck: bool | None) -> tuple[float, float, bool]:
    """``(min_pitch_hz, floor_db, rms_crosscheck)`` from the config or defaults."""
    min_pitch, floor_db, want_rms = 50.0, DEFAULT_FLOOR_DB, False
    if config is not None:
        e = config.section("features").get("energy", {})
        min_pitch = float(e.get("min_pitch_hz", min_pitch))
        floor_db = float(e.get("floor_db", floor_db))
        want_rms = bool(e.get("rms_crosscheck", want_rms))
    if rms_crosscheck is not None:
        want_rms = bool(rms_crosscheck)
    return min_pitch, floor_db, want_rms


# --------------------------------------------------------------------------- #
# Word aggregates (spec 9.1)
# --------------------------------------------------------------------------- #


def word_energy(
    track: EnergyTrack,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """Aggregate an energy track to one row per word.

    Columns are :data:`ENERGY_WORD_COLUMNS`: mean and peak in z units (what the
    envelope compares) and in dB re the voiced median (what an explanation
    quotes), plus the slope in dB/s, which is what ``fading`` is measured on.
    """
    import pandas as pd

    grid = grid or FrameGrid()
    rows = []
    for word in words:
        idx = grid.frame_indices(word.t_start, word.t_end, track.n_frames)
        z = track.z[idx] if idx.size else np.empty(0)
        rel = track.db_rel[idx] if idx.size else np.empty(0)
        times = track.times[idx] if idx.size else np.empty(0)
        ok = np.isfinite(z) & np.isfinite(rel)
        n = int(ok.sum())
        rows.append(
            {
                "energy_mean_z": float(z[ok].mean()) if n else np.nan,
                "energy_peak_z": float(z[ok].max()) if n else np.nan,
                "energy_mean_db_rel": float(rel[ok].mean()) if n else np.nan,
                "energy_peak_db_rel": float(rel[ok].max()) if n else np.nan,
                "energy_slope_db_per_s": _slope(times[ok], rel[ok]),
                "energy_n_frames": n,
            }
        )
    table = pd.DataFrame(rows, columns=list(ENERGY_WORD_COLUMNS))
    table.index = pd.Index([w.idx for w in words], name="unit_idx")
    table.attrs.update(unit="word", file_id=file_id, channel_id=channel_id)
    return table


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of ``values`` against ``times``, or ``NaN``."""
    if values.size < 2 or np.ptp(times) <= 0:
        return float("nan")
    return float(np.polyfit(times, values, 1)[0])
