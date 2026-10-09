"""Voice-quality features from Praat (spec 9.1 VQ).

Four measurements per word, all via ``parselmouth``:

* **HNR (cc)** - harmonics-to-noise ratio in dB from the cross-correlation
  method, the breathiness measure;
* **jitter (local)** - mean absolute difference between consecutive glottal
  periods over the mean period, the cycle-to-cycle frequency instability;
* **shimmer (local)** - the same for peak amplitude;
* **CPPS** - Praat's smoothed cepstral peak prominence, which tracks how
  periodic the spectrum is overall and is the single best correlate of
  perceived dysphonia and of mumbling.

**Three periods minimum.** Jitter and shimmer are defined on *differences*
between consecutive periods, so two periods give one difference and no spread.
Any word with fewer than :data:`MIN_PERIODS` detected glottal periods in its
interval is reported as ``NaN`` across the board rather than as a small number
computed from nothing (spec 9.1: "NaN if < 3 voiced periods").

**Channel rule (CLAUDE.md invariant 7).** Every feature here is channel
sensitive. HNR and CPPS in particular move several dB with microphone distance
and with codec; a LibriVox volunteer's home recording and a 1941 radio
broadcast cannot be compared on them. They may only be compared within one
``channel_id`` - a clean file against its own injected edit, or a speaker's
clean take against their own flawed take.

The pitch floor and ceiling should be the ones the two-pass pitch extractor
settled on for this speaker (:class:`~rostrum.features.pitch.PitchTrack`), not
a global default: period detection with a wrong range is the main source of
spurious jitter.
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
    "MIN_PERIODS",
    "VQ_WORD_COLUMNS",
    "VoiceQualityTrack",
    "cpps_db",
    "harmonicity_track",
    "period_process",
    "word_voice_quality",
]

MIN_PERIODS: int = 3
"""Fewest glottal periods a word needs before jitter and shimmer are reported."""

PERIOD_FLOOR_S: float = 0.0001
"""Praat's shortest admissible period (1/10 kHz)."""

PERIOD_CEILING_S: float = 0.02
"""Praat's longest admissible period (1/50 Hz)."""

MAX_PERIOD_FACTOR: float = 1.3
"""Praat's maximum ratio between consecutive periods before the pair is skipped."""

MAX_AMPLITUDE_FACTOR: float = 1.6
"""Praat's maximum ratio between consecutive amplitudes, for shimmer."""

HARMONICITY_SILENCE_THRESHOLD: float = 0.1
"""Praat ``To Harmonicity (cc)`` silence threshold."""

HARMONICITY_PERIODS_PER_WINDOW: float = 1.0
"""Praat ``To Harmonicity (cc)`` periods per analysis window."""

CPPS_QUEFRENCY_HZ: tuple[float, float] = (60.0, 330.0)
"""Search range for the cepstral peak, as F0 in Hz (quefrency 3.0-16.7 ms)."""

MIN_CPPS_DURATION_S: float = 0.060
"""Shortest word a cepstrogram is attempted on; below this it returns ``NaN``."""

VQ_WORD_COLUMNS: tuple[str, ...] = (
    "hnr_db",
    "jitter_local",
    "shimmer_local",
    "cpps_db",
    "vq_n_periods",
)
"""Word-level voice-quality columns; all channel sensitive (invariant 7)."""


# --------------------------------------------------------------------------- #
# Praat objects
# --------------------------------------------------------------------------- #


def _sound(samples: np.ndarray, sr: int) -> Any:
    """Wrap mono audio in a ``parselmouth.Sound``."""
    import parselmouth

    return parselmouth.Sound(
        np.asarray(samples, dtype=np.float64).reshape(-1), sampling_frequency=sr
    )


def period_process(
    samples: np.ndarray,
    sr: int,
    *,
    floor_hz: float = 50.0,
    ceiling_hz: float = 700.0,
) -> Any:
    """Glottal period marks for the whole recording (``PointProcess``).

    One point process is built per recording and queried per word interval, so
    period detection never depends on where a word boundary happens to fall.

    Args:
        samples: Mono audio.
        sr: Sample rate.
        floor_hz: Pitch floor, normally :attr:`PitchTrack.floor_hz`.
        ceiling_hz: Pitch ceiling, normally :attr:`PitchTrack.ceiling_hz`.

    Returns:
        A Praat ``PointProcess`` object.
    """
    from parselmouth.praat import call

    return call(_sound(samples, sr), "To PointProcess (periodic, cc)", floor_hz, ceiling_hz)


@dataclass(frozen=True)
class VoiceQualityTrack:
    """Frame-level harmonicity for one recording.

    Attributes:
        times: Frame centre times, seconds.
        hnr_db: HNR per frame in dB, ``NaN`` where Praat found no periodicity.
        floor_hz: Pitch floor used.
        ceiling_hz: Pitch ceiling used.
        channel_id: Channel tag; comparisons outside it are forbidden.
    """

    times: np.ndarray
    hnr_db: np.ndarray
    floor_hz: float
    ceiling_hz: float
    channel_id: str = ""

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.hnr_db.size)


def harmonicity_track(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    floor_hz: float = 50.0,
    ceiling_hz: float = 700.0,
    n_frames: int | None = None,
    channel_id: str = "",
) -> VoiceQualityTrack:
    """Frame HNR from Praat's cross-correlation harmonicity, on the project grid.

    Praat chooses its own time step for harmonicity; the result is mapped onto
    the project grid with :meth:`FrameGrid.resample_nearest`, which leaves
    ``NaN`` wherever Praat produced nothing within half a hop.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        floor_hz: Pitch floor; sets the analysis window length.
        ceiling_hz: Pitch ceiling, carried for provenance.
        n_frames: Output length, so this track matches the others.
        channel_id: Channel tag.

    Returns:
        A :class:`VoiceQualityTrack`.
    """
    from parselmouth.praat import call

    grid = grid or FrameGrid(sample_rate=sr)
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    count = grid.n_frames_for(x.size) if n_frames is None else int(n_frames)
    if count <= 0:
        return VoiceQualityTrack(
            times=np.empty(0, dtype=np.float64),
            hnr_db=np.empty(0, dtype=np.float64),
            floor_hz=float(floor_hz),
            ceiling_hz=float(ceiling_hz),
            channel_id=channel_id,
        )

    harmonicity = call(
        _sound(x, sr),
        "To Harmonicity (cc)",
        grid.hop_s,
        floor_hz,
        HARMONICITY_SILENCE_THRESHOLD,
        HARMONICITY_PERIODS_PER_WINDOW,
    )
    values = np.asarray(harmonicity.values, dtype=np.float64).reshape(-1)
    times = np.asarray(harmonicity.xs(), dtype=np.float64).reshape(-1)
    # Praat marks unvoiced frames with -200 dB; that is a sentinel, not a value.
    values = np.where(values <= -100.0, np.nan, values)
    mapped = grid.resample_nearest(times, values, count, tolerance_s=grid.hop_s)
    return VoiceQualityTrack(
        times=grid.centres(count),
        hnr_db=mapped,
        floor_hz=float(floor_hz),
        ceiling_hz=float(ceiling_hz),
        channel_id=channel_id,
    )


def cpps_db(
    samples: np.ndarray,
    sr: int,
    *,
    floor_hz: float = CPPS_QUEFRENCY_HZ[0],
    ceiling_hz: float = CPPS_QUEFRENCY_HZ[1],
) -> float:
    """Smoothed cepstral peak prominence of one segment, dB.

    Praat's ``PowerCepstrogram -> Get CPPS`` with the published defaults: 2 ms
    cepstrogram time step, 5 kHz analysis ceiling, peak searched between
    ``floor_hz`` and ``ceiling_hz``, parabolic interpolation, and a robust
    straight-line fit as the baseline the peak is measured above.

    ``Get CPPS`` has no time argument, so the segment is extracted first; this
    function is therefore called once per word rather than once per recording.

    Args:
        samples: Mono audio of one segment.
        sr: Sample rate.
        floor_hz: Low edge of the peak search.
        ceiling_hz: High edge of the peak search.

    Returns:
        CPPS in dB, or ``NaN`` for a segment too short to analyse.
    """
    from parselmouth.praat import call

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size / float(sr) < MIN_CPPS_DURATION_S:
        return float("nan")
    try:
        cepstrogram = call(_sound(x, sr), "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        value = call(
            cepstrogram,
            "Get CPPS",
            False,  # subtract trend before smoothing
            0.02,  # time averaging window, s
            0.0005,  # quefrency averaging window, s
            floor_hz,
            ceiling_hz,
            0.05,  # tolerance
            "Parabolic",
            0.001,  # fit range start, s
            0.05,  # fit range end, s
            "Exponential decay",
            "Robust",
        )
    except Exception as exc:  # pragma: no cover - Praat rejects odd segments
        logger.debug("CPPS unavailable for a %.3f s segment: %s", x.size / sr, exc)
        return float("nan")
    return float(value)


# --------------------------------------------------------------------------- #
# Per-word table
# --------------------------------------------------------------------------- #


def _vq_conf(config: Config | None) -> dict[str, bool]:
    """Which voice-quality measures to compute, from ``configs/features.yaml``."""
    conf = {"hnr": True, "jitter_local": True, "shimmer_local": True, "cpp": True}
    if config is None:
        return conf
    section = config.features.get("voice_quality", {})
    for key in conf:
        conf[key] = bool(section.get(key, True))
    return conf


def word_voice_quality(
    samples: np.ndarray,
    sr: int,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    config: Config | None = None,
    floor_hz: float = 50.0,
    ceiling_hz: float = 700.0,
    track: VoiceQualityTrack | None = None,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """Voice quality per voiced word (spec 9.1 VQ).

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        words: Aligned words.
        grid: Frame grid; defaults to the project grid.
        config: Loaded config; ``configs/features.yaml`` ``voice_quality:``
            switches individual measures off.
        floor_hz: Pitch floor for period detection - pass the speaker's
            two-pass floor, not a default.
        ceiling_hz: Pitch ceiling for period detection.
        track: A harmonicity track to take word HNR means from; computed here
            when omitted.
        file_id: File tag for the table.
        channel_id: Channel tag for the table.

    Returns:
        A table with columns :data:`VQ_WORD_COLUMNS`, ``NaN`` in every measure
        for a word with fewer than :data:`MIN_PERIODS` glottal periods.
    """
    import pandas as pd
    from parselmouth.praat import call

    grid = grid or FrameGrid(sample_rate=sr)
    conf = _vq_conf(config)
    x = np.asarray(samples, dtype=np.float64).reshape(-1)

    if track is None and conf["hnr"]:
        track = harmonicity_track(
            x, sr, grid=grid, floor_hz=floor_hz, ceiling_hz=ceiling_hz, channel_id=channel_id
        )

    sound = _sound(x, sr)
    points = call(sound, "To PointProcess (periodic, cc)", floor_hz, ceiling_hz)

    rows: list[dict[str, float]] = []
    for word in words:
        t0, t1 = float(word.t_start), float(word.t_end)
        n_periods = int(
            call(
                points,
                "Get number of periods",
                t0,
                t1,
                PERIOD_FLOOR_S,
                PERIOD_CEILING_S,
                MAX_PERIOD_FACTOR,
            )
        )
        row: dict[str, float] = dict.fromkeys(VQ_WORD_COLUMNS, float("nan"))
        row["vq_n_periods"] = float(n_periods)
        if n_periods < MIN_PERIODS:
            rows.append(row)
            continue

        if conf["hnr"] and track is not None:
            idx = grid.frame_indices(t0, t1, track.n_frames)
            hnr = track.hnr_db[idx] if idx.size else np.empty(0)
            ok = np.isfinite(hnr)
            row["hnr_db"] = float(hnr[ok].mean()) if ok.any() else float("nan")
        if conf["jitter_local"]:
            row["jitter_local"] = float(
                call(
                    points,
                    "Get jitter (local)",
                    t0,
                    t1,
                    PERIOD_FLOOR_S,
                    PERIOD_CEILING_S,
                    MAX_PERIOD_FACTOR,
                )
            )
        if conf["shimmer_local"]:
            row["shimmer_local"] = float(
                call(
                    [sound, points],
                    "Get shimmer (local)",
                    t0,
                    t1,
                    PERIOD_FLOOR_S,
                    PERIOD_CEILING_S,
                    MAX_PERIOD_FACTOR,
                    MAX_AMPLITUDE_FACTOR,
                )
            )
        if conf["cpp"]:
            lo = max(0, round(t0 * sr))
            hi = min(x.size, round(t1 * sr))
            row["cpps_db"] = cpps_db(x[lo:hi], sr) if hi > lo else float("nan")
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(VQ_WORD_COLUMNS))
    table.index = pd.Index([w.idx for w in words], name="unit_idx", dtype="int64")
    table.attrs.update(
        unit="word",
        file_id=file_id,
        channel_id=channel_id,
        channel_sensitive=True,
        floor_hz=float(floor_hz),
        ceiling_hz=float(ceiling_hz),
    )
    return table
