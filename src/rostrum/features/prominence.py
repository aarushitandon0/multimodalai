"""Continuous-wavelet prominence (spec 9.1 PROM), after Suni et al. [R25, R26].

Prominence is not a single acoustic measurement. A word stands out because it is
higher, louder, longer, or some combination of the three, and listeners trade
the cues off against one another. The wavelet method models that directly:

1. build one **composite prosodic signal** on the project frame grid from three
   z-scored components, weighted as the spec fixes them
   (``configs/features.yaml``: ``component_weights``)::

       composite = 1.0 * f0_z + 0.5 * energy_z + 1.0 * duration_z

   F0 is interpolated across unvoiced frames (a voiceless stop inside a word is
   not a pitch valley), energy comes from the already speaker-normalised track,
   and the duration component is each word's normalised duration held constant
   over its own frames;
2. take a **continuous wavelet transform** of that signal with a Mexican-hat
   wavelet over scales from roughly one syllable to roughly one phrase;
3. **sum the scales** into one response curve, so a bump that is a peak at the
   syllable scale *and* at the word scale scores higher than one that is a peak
   at a single scale;
4. a word's prominence is the **maximum of the summed response inside the
   word**, and the time of that maximum is where the stress landed - which is
   what the ``misplaced_stress`` label needs.

**Scales (documented as the prompt requires).** With a 10 ms hop and the
Mexican hat's centre frequency :math:`f_c = 0.25`, a scale :math:`a` responds to
features of period :math:`T = a \\cdot \\mathrm{hop} / f_c`. The bank starts at
:data:`DEFAULT_MIN_PERIOD_S` = 0.10 s (a short syllable), spans
``n_octaves`` = 5 octaves at ``scales_per_octave`` = 12 scales per octave, and
so ends at 3.2 s (a short phrase): 61 scales from :math:`a = 2.5` to
:math:`a = 80`. Each scale is divided by its own standard deviation before the
sum, so the long scales - which carry far more energy - cannot drown the
syllable scales.

**Reference implementation.** The prompt asks for a cross-check against
``wavelet_prosody_toolkit``. That package is not on PyPI (``pip install
wavelet_prosody_toolkit`` reports "no matching distribution", verified
2026-10-06), so there is nothing to pin or install cleanly and this module is
the project's own implementation of the published method. It differs from the
toolkit in two ways worth stating in the report: the composite is built from the
*already speaker-normalised* curves this project uses everywhere, and no
"lines of maximum amplitude" tree is built - the per-word maximum of the
scale-summed response is taken directly, because the unit of analysis here is
always the aligned word, never a discovered one.

Everything in this module is a pure function of a signal and the alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features.frames import FrameGrid

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.features.energy import EnergyTrack
    from rostrum.features.pitch import PitchTrack
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MIN_PERIOD_S",
    "DEFAULT_N_OCTAVES",
    "DEFAULT_SCALES_PER_OCTAVE",
    "DEFAULT_WAVELET",
    "DEFAULT_WEIGHTS",
    "PROMINENCE_WORD_COLUMNS",
    "CompositeSignal",
    "ProminenceTrack",
    "composite_signal",
    "cwt_response",
    "cwt_scales",
    "duration_signal",
    "interpolate_gaps",
    "prominence_track",
    "word_prominence",
    "zscore",
]

DEFAULT_WAVELET: str = "mexh"
"""Mexican hat (Ricker): real, zero-mean, symmetric - a peak detector."""

DEFAULT_MIN_PERIOD_S: float = 0.100
"""Shortest period the scale bank responds to: a short syllable."""

DEFAULT_N_OCTAVES: int = 5
"""Octaves spanned, so the longest period is 0.1 * 2**5 = 3.2 s."""

DEFAULT_SCALES_PER_OCTAVE: int = 12
"""Scales per octave."""

DEFAULT_WEIGHTS: tuple[float, float, float] = (1.0, 0.5, 1.0)
"""Composite weights for (F0, energy, duration) - spec 9.1."""

COMPONENT_ORDER: tuple[str, str, str] = ("f0_semitones", "energy_db", "word_duration_norm")
"""Component names, matching ``configs/features.yaml``: ``prominence.components``."""


# --------------------------------------------------------------------------- #
# Building the composite signal
# --------------------------------------------------------------------------- #


def interpolate_gaps(curve: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``NaN`` runs, holding the end values at the edges.

    Used only for the wavelet input (``configs/features.yaml``:
    ``interpolate_unvoiced.for_cwt: true``). Statistics are never computed on an
    interpolated curve - invariant 5 forbids inventing F0 where there was none,
    and the interpolated values exist here solely so the wavelet does not see a
    cliff at every voiceless consonant.
    """
    y = np.asarray(curve, dtype=np.float64).copy()
    if y.size == 0:
        return y
    ok = np.isfinite(y)
    if not ok.any():
        return np.zeros_like(y)
    index = np.arange(y.size, dtype=np.float64)
    y[~ok] = np.interp(index[~ok], index[ok], y[ok])
    return y


def zscore(values: np.ndarray) -> np.ndarray:
    """Centre and scale to unit standard deviation; a flat curve becomes zeros."""
    y = np.asarray(values, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.zeros_like(y)
    mean = float(y[ok].mean())
    std = float(y[ok].std())
    if not np.isfinite(std) or std <= 1e-12:
        return np.zeros_like(y)
    return (y - mean) / std


def duration_signal(
    words: list[Word], *, grid: FrameGrid | None = None, n_frames: int
) -> np.ndarray:
    """A frame curve of normalised word duration (the third composite component).

    Each word contributes its duration per syllable, z-scored across the
    recording's words, held constant over its own frames. Dividing by the
    syllable count is what makes the component a *lengthening* signal rather
    than a word-length signal: without it every polysyllabic word would look
    stressed.

    Frames between words are filled by interpolating between the neighbouring
    word values, so a pause does not inject a step into the wavelet input.

    Args:
        words: Aligned words, ordered by index.
        grid: Frame grid; defaults to the project grid.
        n_frames: Length of the output curve.

    Returns:
        A curve of length ``n_frames``.
    """
    from rostrum.features.duration import duration_per_syllable

    grid = grid or FrameGrid()
    out = np.full(int(n_frames), np.nan, dtype=np.float64)
    if out.size == 0 or not words:
        return np.zeros_like(out)

    per_syllable = np.array([duration_per_syllable(w) for w in words], dtype=np.float64)
    if not np.isfinite(per_syllable).any():
        per_syllable = np.array([w.duration for w in words], dtype=np.float64)
    normalised = zscore(per_syllable)
    for word, value in zip(words, normalised, strict=True):
        idx = grid.frame_indices(word.t_start, word.t_end, out.size)
        if idx.size:
            out[idx] = value
    return interpolate_gaps(out)


@dataclass(frozen=True)
class CompositeSignal:
    """The weighted sum of the three prosodic components, on the frame grid.

    Attributes:
        values: The composite curve, one value per frame.
        components: Each z-scored component curve by name, before weighting, so
            an explanation can say which cue carried the prominence.
        weights: The weights applied, in :data:`COMPONENT_ORDER`.
    """

    values: np.ndarray
    components: dict[str, np.ndarray]
    weights: tuple[float, float, float]


def composite_signal(
    f0_st: np.ndarray,
    energy_z: np.ndarray,
    duration: np.ndarray,
    *,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
    normalise: bool = True,
) -> CompositeSignal:
    """Combine F0, energy and duration into one curve (spec 9.1 PROM).

    Args:
        f0_st: F0 in semitones re the speaker median, ``NaN`` where unvoiced.
        energy_z: Speaker-normalised energy, ``NaN`` where unmeasured.
        duration: The duration component from :func:`duration_signal`.
        weights: ``(f0, energy, duration)``; the spec fixes 1.0 / 0.5 / 1.0.
        normalise: z-score each component before weighting, so that the weights
            mean what they say regardless of each cue's native spread.

    Returns:
        A :class:`CompositeSignal`.

    Raises:
        ValueError: The three curves are not the same length.
    """
    curves = [np.asarray(c, dtype=np.float64) for c in (f0_st, energy_z, duration)]
    lengths = {c.size for c in curves}
    if len(lengths) > 1:
        raise ValueError(f"composite components must be the same length, got {sorted(lengths)}")

    prepared: dict[str, np.ndarray] = {}
    for name, curve in zip(COMPONENT_ORDER, curves, strict=True):
        filled = interpolate_gaps(curve)
        prepared[name] = zscore(filled) if normalise else filled

    total = np.zeros(curves[0].size, dtype=np.float64)
    for weight, name in zip(weights, COMPONENT_ORDER, strict=True):
        total = total + float(weight) * prepared[name]
    return CompositeSignal(values=total, components=prepared, weights=tuple(map(float, weights)))


# --------------------------------------------------------------------------- #
# The wavelet transform
# --------------------------------------------------------------------------- #


def cwt_scales(
    *,
    hop_s: float = 0.010,
    wavelet: str = DEFAULT_WAVELET,
    min_period_s: float = DEFAULT_MIN_PERIOD_S,
    n_octaves: int = DEFAULT_N_OCTAVES,
    scales_per_octave: int = DEFAULT_SCALES_PER_OCTAVE,
) -> tuple[np.ndarray, np.ndarray]:
    """The dyadic scale bank and the period each scale responds to.

    Args:
        hop_s: Frame hop of the composite signal, seconds.
        wavelet: PyWavelets continuous wavelet name.
        min_period_s: Period of the shortest scale (syllable end of the bank).
        n_octaves: Octaves spanned above ``min_period_s``.
        scales_per_octave: Scales per octave.

    Returns:
        ``(scales, periods_s)``: the scale values to hand to ``pywt.cwt`` and
        the period in seconds each one is centred on.

    Raises:
        ValueError: A parameter is non-positive.
    """
    import pywt

    if hop_s <= 0 or min_period_s <= 0:
        raise ValueError(f"hop ({hop_s}) and min_period ({min_period_s}) must be positive")
    if n_octaves <= 0 or scales_per_octave <= 0:
        raise ValueError(
            f"n_octaves ({n_octaves}) and scales_per_octave ({scales_per_octave}) "
            "must be positive"
        )

    centre = float(pywt.central_frequency(wavelet))
    # period T = scale * hop / centre_frequency  =>  scale = T * centre / hop
    smallest = min_period_s * centre / hop_s
    steps = int(n_octaves) * int(scales_per_octave) + 1
    exponents = np.arange(steps, dtype=np.float64) / float(scales_per_octave)
    scales = smallest * 2.0**exponents
    periods = scales * hop_s / centre
    return scales, periods


def cwt_response(
    signal: np.ndarray,
    scales: np.ndarray,
    *,
    wavelet: str = DEFAULT_WAVELET,
    hop_s: float = 0.010,
) -> tuple[np.ndarray, np.ndarray]:
    """Wavelet coefficients and the scale-summed response.

    Each scale is divided by its own standard deviation before summing: the long
    scales carry far more energy than the short ones, and without that
    normalisation the sum would be a low-pass version of the composite instead
    of a multi-scale peak detector.

    The sum keeps the **sign** of the coefficients. For a zero-mean symmetric
    wavelet a positive coefficient means "locally higher than the surround", so
    a signed sum ranks peaks and leaves valleys negative - which is what
    prominence means. Taking ``abs`` first would make a dip as prominent as a
    rise.

    Args:
        signal: The composite curve.
        scales: Scale bank from :func:`cwt_scales`.
        wavelet: PyWavelets continuous wavelet name.
        hop_s: Frame hop, seconds; sets the sampling period.

    Returns:
        ``(coefficients, summed)`` with shapes ``(n_scales, n_frames)`` and
        ``(n_frames,)``.
    """
    import pywt

    x = np.asarray(signal, dtype=np.float64)
    bank = np.asarray(scales, dtype=np.float64)
    if x.size == 0 or bank.size == 0:
        return np.zeros((bank.size, x.size), dtype=np.float64), np.zeros(x.size, dtype=np.float64)

    coefficients, _ = pywt.cwt(x, bank, wavelet, sampling_period=float(hop_s))
    coefficients = np.asarray(coefficients, dtype=np.float64)
    spread = coefficients.std(axis=1, keepdims=True)
    spread = np.where(np.isfinite(spread) & (spread > 1e-12), spread, 1.0)
    summed = (coefficients / spread).sum(axis=0)
    return coefficients, summed


# --------------------------------------------------------------------------- #
# The track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProminenceTrack:
    """Frame-level prominence response for one recording.

    Attributes:
        times: Frame centre times, seconds.
        composite: The composite prosodic signal.
        response: The scale-summed wavelet response.
        scales: The scale bank used.
        periods_s: Period each scale is centred on, seconds.
        components: The z-scored component curves, by name.
        wavelet: Wavelet name.
    """

    times: np.ndarray
    composite: np.ndarray
    response: np.ndarray
    scales: np.ndarray
    periods_s: np.ndarray
    components: dict[str, np.ndarray]
    wavelet: str = DEFAULT_WAVELET

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.response.size)

    @property
    def scale_range_s(self) -> tuple[float, float]:
        """Shortest and longest period in the bank, seconds - for the report."""
        if self.periods_s.size == 0:
            return (float("nan"), float("nan"))
        return (float(self.periods_s[0]), float(self.periods_s[-1]))


PROMINENCE_WORD_COLUMNS: tuple[str, ...] = (
    "prominence",
    "prominence_z",
    "prominence_mean",
    "prominence_peak_t",
    "prominence_peak_rel",
    "prominence_n_frames",
)
"""Word-level prominence columns.

``prominence`` is the raw maximum of the scale-summed response inside the word;
``prominence_z`` is that value z-scored over the recording's words, which is the
quantity the envelope compares; ``prominence_peak_t`` is the absolute time of
the maximum and ``prominence_peak_rel`` its position within the word on 0-1,
which is what distinguishes a well-placed stress from a late one.
"""


def _prominence_conf(config: Config | None) -> dict[str, Any]:
    """Prominence parameters, from ``configs/features.yaml`` or the spec defaults."""
    conf: dict[str, Any] = {
        "wavelet": DEFAULT_WAVELET,
        "scales_per_octave": DEFAULT_SCALES_PER_OCTAVE,
        "n_octaves": DEFAULT_N_OCTAVES,
        "min_period_s": DEFAULT_MIN_PERIOD_S,
        "weights": DEFAULT_WEIGHTS,
        "normalise": True,
    }
    if config is None:
        return conf
    section = config.features.get("prominence", {})
    conf["wavelet"] = str(section.get("wavelet", DEFAULT_WAVELET))
    conf["scales_per_octave"] = int(section.get("scales_per_octave", DEFAULT_SCALES_PER_OCTAVE))
    conf["n_octaves"] = int(section.get("n_octaves", DEFAULT_N_OCTAVES))
    conf["min_period_s"] = float(section.get("min_period_s", DEFAULT_MIN_PERIOD_S))
    weights = section.get("component_weights")
    if weights and len(weights) == 3:
        conf["weights"] = (float(weights[0]), float(weights[1]), float(weights[2]))
    conf["normalise"] = bool(section.get("normalise_components", True))
    return conf


def prominence_track(
    pitch: PitchTrack,
    energy: EnergyTrack,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    config: Config | None = None,
) -> ProminenceTrack:
    """Build the composite signal and its wavelet response for one recording.

    Args:
        pitch: Pitch track; ``f0_st`` supplies the first component.
        energy: Energy track; ``z`` supplies the second.
        words: Aligned words; their durations supply the third.
        grid: Frame grid; defaults to the project grid.
        config: Loaded config; ``configs/features.yaml`` ``prominence:``.

    Returns:
        A :class:`ProminenceTrack`.

    Raises:
        ValueError: The pitch and energy tracks have different lengths.
    """
    grid = grid or FrameGrid()
    conf = _prominence_conf(config)
    if pitch.n_frames != energy.n_frames:
        raise ValueError(
            f"pitch has {pitch.n_frames} frames and energy {energy.n_frames}; "
            "extract both with the same n_frames before combining them"
        )
    n_frames = pitch.n_frames

    composite = composite_signal(
        pitch.f0_st,
        energy.z,
        duration_signal(words, grid=grid, n_frames=n_frames),
        weights=conf["weights"],
        normalise=conf["normalise"],
    )
    scales, periods = cwt_scales(
        hop_s=grid.hop_s,
        wavelet=conf["wavelet"],
        min_period_s=conf["min_period_s"],
        n_octaves=conf["n_octaves"],
        scales_per_octave=conf["scales_per_octave"],
    )
    _, response = cwt_response(composite.values, scales, wavelet=conf["wavelet"], hop_s=grid.hop_s)
    return ProminenceTrack(
        times=grid.centres(n_frames),
        composite=composite.values,
        response=response,
        scales=scales,
        periods_s=periods,
        components=composite.components,
        wavelet=conf["wavelet"],
    )


def word_prominence(
    track: ProminenceTrack,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """Per-word prominence: the maximum of the summed response inside the word.

    Columns are :data:`PROMINENCE_WORD_COLUMNS`. A word with no frames gets
    ``NaN``, never zero - zero is a real prominence value here, meaning "as
    prominent as the local surround".
    """
    import pandas as pd

    grid = grid or FrameGrid()
    peaks = np.full(len(words), np.nan, dtype=np.float64)
    means = np.full(len(words), np.nan, dtype=np.float64)
    peak_t = np.full(len(words), np.nan, dtype=np.float64)
    peak_rel = np.full(len(words), np.nan, dtype=np.float64)
    counts = np.zeros(len(words), dtype=np.int64)

    for row, word in enumerate(words):
        idx = grid.frame_indices(word.t_start, word.t_end, track.n_frames)
        if idx.size == 0:
            continue
        values = track.response[idx]
        ok = np.isfinite(values)
        if not ok.any():
            continue
        counts[row] = int(ok.sum())
        local = np.argmax(np.where(ok, values, -np.inf))
        peaks[row] = float(values[local])
        means[row] = float(values[ok].mean())
        peak_t[row] = float(track.times[idx[local]])
        duration = word.duration
        if duration > 0:
            peak_rel[row] = float(np.clip((peak_t[row] - word.t_start) / duration, 0.0, 1.0))

    table = pd.DataFrame(
        {
            "prominence": peaks,
            "prominence_z": zscore(peaks),
            "prominence_mean": means,
            "prominence_peak_t": peak_t,
            "prominence_peak_rel": peak_rel,
            "prominence_n_frames": counts,
        },
        columns=list(PROMINENCE_WORD_COLUMNS),
    )
    table.index = pd.Index([w.idx for w in words], name="unit_idx", dtype="int64")
    table.attrs.update(
        unit="word",
        file_id=file_id,
        channel_id=channel_id,
        wavelet=track.wavelet,
        scale_range_s=track.scale_range_s,
    )
    return table
