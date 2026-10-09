"""Pitch features: two-pass Praat F0, semitones, CREPE cross-check (spec 9.1-9.2).

Four decisions carry the project's invariants:

* **Praat is the extractor** (autocorrelation, via parselmouth), run twice: a
  wide pass (50-700 Hz) measures the speaker's quartiles, then a narrow pass
  (floor ``0.75*Q1``, ceiling ``1.5*Q3``) re-tracks within that range. The
  narrow pass is what removes most octave errors (invariant 5).
* **Comparisons are in semitones re the speaker's own median** over the
  recording, never in Hz (invariant 5, spec 9.2): a 20 Hz excursion means
  something different for a 100 Hz voice and a 220 Hz one.
* **A neural tracker cross-checks every frame.** ``torchcrepe`` with Viterbi
  decoding runs on the same grid; frames where the two trackers disagree by more
  than 50 cents are set to ``NaN`` *for statistics* rather than averaged, since
  a disagreement means one of them is wrong and we cannot say which.
* **Unvoiced is ``NaN``, never 0 and never interpolated.** Interpolation is a
  plotting convenience and is done by the caller, after statistics.

Every function here is pure: audio and parameters in, arrays out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features.frames import FrameGrid

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable, Word

logger = logging.getLogger(__name__)

__all__ = [
    "PITCH_WORD_COLUMNS",
    "PitchTrack",
    "agreement_mask",
    "agreement_rate",
    "apply_agreement",
    "cents_difference",
    "crepe_pitch_hz",
    "pitch_track",
    "praat_pitch_hz",
    "semitones",
    "two_pass_range",
    "word_pitch",
]

PITCH_WORD_COLUMNS: tuple[str, ...] = (
    "f0_mean_st",
    "f0_max_st",
    "f0_min_st",
    "f0_range_st",
    "f0_slope_st_per_s",
    "f0_std_st",
    "f0_n_voiced",
    "f0_voiced_frac",
)
"""Word-level pitch columns (spec 9.1: mean, max, slope, range in semitones)."""

DEFAULT_PASS1 = (50.0, 700.0)
"""Pass-1 range in Hz (invariant 5)."""


# --------------------------------------------------------------------------- #
# Unit conversions (spec 9.2)
# --------------------------------------------------------------------------- #


def semitones(f0_hz: np.ndarray, reference_hz: float) -> np.ndarray:
    """Convert F0 in Hz to semitones relative to ``reference_hz``.

    ``st(t) = 12 * log2(f0(t) / reference)``, so a doubling is +12 st.

    Args:
        f0_hz: F0 per frame; ``NaN`` or non-positive means unvoiced.
        reference_hz: The speaker's median F0 over the recording.

    Returns:
        Semitones per frame, ``NaN`` where unvoiced.

    Raises:
        ValueError: ``reference_hz`` is not positive.
    """
    if not np.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError(f"reference frequency must be positive, got {reference_hz}")
    f0 = np.asarray(f0_hz, dtype=np.float64)
    out = np.full(f0.shape, np.nan, dtype=np.float64)
    voiced = np.isfinite(f0) & (f0 > 0)
    out[voiced] = 12.0 * np.log2(f0[voiced] / float(reference_hz))
    return out


def cents_difference(a_hz: np.ndarray, b_hz: np.ndarray) -> np.ndarray:
    """Signed difference between two F0 tracks in cents (1200 per octave)."""
    a = np.asarray(a_hz, dtype=np.float64)
    b = np.asarray(b_hz, dtype=np.float64)
    out = np.full(np.broadcast(a, b).shape, np.nan, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out[ok] = 1200.0 * np.log2(a[ok] / b[ok])
    return out


# --------------------------------------------------------------------------- #
# Praat extraction
# --------------------------------------------------------------------------- #


def praat_pitch_hz(
    samples: np.ndarray,
    sr: int,
    *,
    floor_hz: float,
    ceiling_hz: float,
    grid: FrameGrid | None = None,
    silence_threshold: float = 0.03,
    voicing_threshold: float = 0.45,
    octave_cost: float = 0.01,
    octave_jump_cost: float = 0.35,
    voiced_unvoiced_cost: float = 0.14,
    n_frames: int | None = None,
) -> np.ndarray:
    """Track F0 with Praat autocorrelation, mapped onto the frame grid.

    Args:
        samples: Mono audio.
        sr: Sample rate.
        floor_hz: Pitch floor for this pass.
        ceiling_hz: Pitch ceiling for this pass.
        grid: Frame grid; defaults to the project grid.
        silence_threshold: Praat ``silence threshold``.
        voicing_threshold: Praat ``voicing threshold``.
        octave_cost: Praat ``octave cost``.
        octave_jump_cost: Praat ``octave-jump cost``.
        voiced_unvoiced_cost: Praat ``voiced / unvoiced cost``.
        n_frames: Output length; defaults to the grid's count for this audio.

    Returns:
        F0 per frame in Hz, ``NaN`` where Praat found no voicing.
    """
    import parselmouth

    grid = grid or FrameGrid(sample_rate=sr)
    audio = np.ascontiguousarray(np.asarray(samples, dtype=np.float64).ravel())
    frames = grid.n_frames_for(audio.size) if n_frames is None else int(n_frames)
    if frames <= 0:
        return np.empty(0, dtype=np.float64)

    sound = parselmouth.Sound(audio, sampling_frequency=float(sr))
    pitch = sound.to_pitch_ac(
        time_step=grid.hop_s,
        pitch_floor=float(floor_hz),
        pitch_ceiling=float(ceiling_hz),
        silence_threshold=float(silence_threshold),
        voicing_threshold=float(voicing_threshold),
        octave_cost=float(octave_cost),
        octave_jump_cost=float(octave_jump_cost),
        voiced_unvoiced_cost=float(voiced_unvoiced_cost),
    )
    values = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    values[values <= 0.0] = np.nan
    return grid.resample_nearest(np.asarray(pitch.xs(), dtype=np.float64), values, frames)


def two_pass_range(
    f0_hz: np.ndarray,
    *,
    q1_scale: float = 0.75,
    q3_scale: float = 1.5,
    min_floor_hz: float = 40.0,
    max_ceiling_hz: float = 800.0,
    min_voiced_frames: int = 50,
) -> tuple[float, float] | None:
    """Derive the pass-2 pitch range from a pass-1 track (invariant 5).

    Args:
        f0_hz: Pass-1 F0 per frame, ``NaN`` where unvoiced.
        q1_scale: Multiplier on Q1 giving the new floor.
        q3_scale: Multiplier on Q3 giving the new ceiling.
        min_floor_hz: Hard lower clamp on the floor.
        max_ceiling_hz: Hard upper clamp on the ceiling.
        min_voiced_frames: Below this many voiced frames the quartiles are not
            trustworthy and the caller should keep the pass-1 range.

    Returns:
        ``(floor_hz, ceiling_hz)``, or ``None`` when voicing is too sparse.
    """
    voiced = np.asarray(f0_hz, dtype=np.float64)
    voiced = voiced[np.isfinite(voiced) & (voiced > 0)]
    if voiced.size < int(min_voiced_frames):
        return None
    q1, q3 = (float(x) for x in np.percentile(voiced, [25.0, 75.0]))
    floor = max(float(min_floor_hz), q1_scale * q1)
    ceiling = min(float(max_ceiling_hz), q3_scale * q3)
    if ceiling <= floor:
        return None
    return floor, ceiling


# --------------------------------------------------------------------------- #
# Neural cross-check (spec 9.1 F0x)
# --------------------------------------------------------------------------- #


def crepe_pitch_hz(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    floor_hz: float = DEFAULT_PASS1[0],
    ceiling_hz: float = DEFAULT_PASS1[1],
    model: str = "full",
    periodicity_threshold: float = 0.5,
    batch_size: int = 512,
    device: str = "cpu",
    n_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Track F0 with ``torchcrepe`` (Viterbi decoding) on the frame grid.

    Frames whose periodicity falls below ``periodicity_threshold`` are returned
    as ``NaN``: CREPE always emits a frequency, so without this gate silence
    would look voiced and disagree with Praat everywhere.

    Args:
        samples: Mono audio.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        floor_hz: Search floor, Hz.
        ceiling_hz: Search ceiling, Hz.
        model: ``"full"`` or ``"tiny"``.
        periodicity_threshold: Minimum periodicity to keep a frame.
        batch_size: Frames per forward pass.
        device: Torch device.
        n_frames: Output length; defaults to the grid's count for this audio.

    Returns:
        ``(f0_hz, periodicity)``, both on the frame grid, F0 ``NaN`` where the
        periodicity gate rejected the frame.
    """
    import torch
    import torchcrepe

    grid = grid or FrameGrid(sample_rate=sr)
    audio = np.asarray(samples, dtype=np.float32).ravel()
    frames = grid.n_frames_for(audio.size) if n_frames is None else int(n_frames)
    if frames <= 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    with torch.inference_mode():
        f0_t, periodicity_t = torchcrepe.predict(
            torch.from_numpy(audio)[None],
            sr,
            hop_length=grid.hop_samples,
            fmin=float(floor_hz),
            fmax=float(ceiling_hz),
            model=model,
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=True,
            batch_size=int(batch_size),
            device=device,
        )
    f0 = f0_t[0].double().numpy()
    periodicity = periodicity_t[0].double().numpy()
    f0[periodicity < float(periodicity_threshold)] = np.nan

    # torchcrepe centres its frames (pad=True), so its frame i sits at i*hop.
    src_times = np.arange(f0.size, dtype=np.float64) * grid.hop_s
    tolerance = grid.window_s / 2.0 + grid.hop_s / 2.0
    return (
        grid.resample_nearest(src_times, f0, frames, tolerance_s=tolerance),
        grid.resample_nearest(src_times, periodicity, frames, tolerance_s=tolerance),
    )


def agreement_mask(
    praat_hz: np.ndarray, crepe_hz: np.ndarray, *, agreement_cents: float = 50.0
) -> np.ndarray:
    """Frames where the two trackers agree within ``agreement_cents``.

    A frame either tracker left unvoiced does not agree: there is nothing to
    corroborate it.
    """
    diff = np.abs(cents_difference(praat_hz, crepe_hz))
    return np.isfinite(diff) & (diff <= float(agreement_cents))


def agreement_rate(
    praat_hz: np.ndarray, crepe_hz: np.ndarray, *, agreement_cents: float = 50.0
) -> float | None:
    """Fraction of commonly-voiced frames that agree, or ``None`` if there are none."""
    comparable = np.isfinite(cents_difference(praat_hz, crepe_hz))
    n = int(comparable.sum())
    if n == 0:
        return None
    agree = agreement_mask(praat_hz, crepe_hz, agreement_cents=agreement_cents)
    return float(agree.sum()) / float(n)


# --------------------------------------------------------------------------- #
# The track
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PitchTrack:
    """Frame-level pitch for one recording.

    Attributes:
        times: Frame centre times, seconds.
        f0_hz: F0 per frame in Hz, ``NaN`` where unvoiced or unusable.
        f0_st: F0 in semitones re :attr:`median_hz`, ``NaN`` likewise.
        voiced: Frames Praat voiced.
        usable: Frames usable for statistics - voiced and, when a cross-check
            ran, agreeing with it.
        median_hz: Speaker median over the voiced frames, or ``None`` if silent.
        floor_hz: Pitch floor actually used.
        ceiling_hz: Pitch ceiling actually used.
        two_pass: Whether the narrow second pass was applied.
        agreement_rate: Fraction of comparable frames agreeing with the
            cross-check; ``None`` when no cross-check ran.
        crosscheck: Name of the cross-check tracker, or ``""``.
        periodicity: Cross-check periodicity per frame, or ``None``.
    """

    times: np.ndarray
    f0_hz: np.ndarray
    f0_st: np.ndarray
    voiced: np.ndarray
    usable: np.ndarray
    median_hz: float | None
    floor_hz: float
    ceiling_hz: float
    two_pass: bool
    agreement_rate: float | None = None
    crosscheck: str = ""
    periodicity: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.f0_hz.size)

    @property
    def voiced_frac(self) -> float:
        """Fraction of frames Praat voiced."""
        return float(self.voiced.mean()) if self.n_frames else 0.0


def pitch_track(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    config: Config | None = None,
    crosscheck: bool | None = None,
    crosscheck_model: str | None = None,
) -> PitchTrack:
    """Extract the two-pass Praat pitch track of one recording.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        grid: Frame grid; defaults to the project grid.
        config: Loaded config; ``configs/features.yaml`` ``pitch:`` supplies the
            Praat parameters and the cross-check settings when given.
        crosscheck: Run the neural cross-check. Defaults to the config value,
            or ``False`` without a config - CREPE on CPU costs seconds per
            second of audio, so the caller opts in.
        crosscheck_model: ``"full"`` or ``"tiny"``, overriding the config.

    Returns:
        A :class:`PitchTrack` whose ``f0_st`` is ready for comparison against an
        envelope: speaker-normalised, with unusable frames already ``NaN``.
    """
    grid = grid or FrameGrid(sample_rate=sr)
    conf = _pitch_conf(config)
    praat_kwargs: dict[str, Any] = {
        "silence_threshold": conf["silence_threshold"],
        "voicing_threshold": conf["voicing_threshold"],
        "octave_cost": conf["octave_cost"],
        "octave_jump_cost": conf["octave_jump_cost"],
        "voiced_unvoiced_cost": conf["voiced_unvoiced_cost"],
    }

    pass1 = praat_pitch_hz(
        samples,
        sr,
        floor_hz=conf["pass1"][0],
        ceiling_hz=conf["pass1"][1],
        grid=grid,
        **praat_kwargs,
    )
    narrowed = two_pass_range(
        pass1,
        q1_scale=conf["q1_scale"],
        q3_scale=conf["q3_scale"],
        min_floor_hz=conf["min_floor_hz"],
        max_ceiling_hz=conf["max_ceiling_hz"],
        min_voiced_frames=conf["min_voiced_frames"],
    )
    if narrowed is None:
        floor_hz, ceiling_hz = conf["pass1"]
        f0 = pass1
        two_pass = False
        logger.debug("pitch: keeping the pass-1 range; too few voiced frames for quartiles")
    else:
        floor_hz, ceiling_hz = narrowed
        f0 = praat_pitch_hz(
            samples,
            sr,
            floor_hz=floor_hz,
            ceiling_hz=ceiling_hz,
            grid=grid,
            n_frames=pass1.size,
            **praat_kwargs,
        )
        two_pass = True

    voiced = np.isfinite(f0) & (f0 > 0)
    median_hz = float(np.median(f0[voiced])) if voiced.any() else None

    want_crosscheck = conf["crosscheck_enabled"] if crosscheck is None else bool(crosscheck)
    agree = voiced.copy()
    rate: float | None = None
    name = ""
    periodicity: np.ndarray | None = None
    if want_crosscheck and voiced.any():
        try:
            crepe_hz, periodicity = crepe_pitch_hz(
                samples,
                sr,
                grid=grid,
                floor_hz=conf["pass1"][0],
                ceiling_hz=conf["pass1"][1],
                model=crosscheck_model or conf["crosscheck_model"],
                periodicity_threshold=conf["periodicity_threshold"],
                n_frames=f0.size,
            )
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            logger.warning("pitch cross-check unavailable (%s); using Praat alone", exc)
        else:
            name = "torchcrepe"
            rate = agreement_rate(f0, crepe_hz, agreement_cents=conf["agreement_cents"])
            if conf["exclude_disagreeing"]:
                agree = agreement_mask(f0, crepe_hz, agreement_cents=conf["agreement_cents"])

    usable = voiced & agree
    f0_usable = np.where(usable, f0, np.nan)
    f0_st = (
        semitones(f0_usable, median_hz)
        if median_hz is not None
        else np.full(f0.shape, np.nan, dtype=np.float64)
    )
    return PitchTrack(
        times=grid.centres(f0.size),
        f0_hz=f0_usable,
        f0_st=f0_st,
        voiced=voiced,
        usable=usable,
        median_hz=median_hz,
        floor_hz=float(floor_hz),
        ceiling_hz=float(ceiling_hz),
        two_pass=two_pass,
        agreement_rate=rate,
        crosscheck=name,
        periodicity=periodicity,
    )


def apply_agreement(track: PitchTrack, agree: np.ndarray) -> PitchTrack:
    """Return ``track`` with frames outside ``agree`` removed from statistics.

    Used when a cross-check is computed separately (for instance cached, or run
    on a different device) and applied afterwards.
    """
    usable = track.voiced & np.asarray(agree, dtype=bool)
    f0 = np.where(usable, track.f0_hz, np.nan)
    f0_st = np.where(usable, track.f0_st, np.nan)
    return replace(track, f0_hz=f0, f0_st=f0_st, usable=usable)


def _pitch_conf(config: Config | None) -> dict[str, Any]:
    """Pitch parameters, from ``configs/features.yaml`` or the spec defaults."""
    conf: dict[str, Any] = {
        "pass1": DEFAULT_PASS1,
        "q1_scale": 0.75,
        "q3_scale": 1.5,
        "min_floor_hz": 40.0,
        "max_ceiling_hz": 800.0,
        "min_voiced_frames": 50,
        "silence_threshold": 0.03,
        "voicing_threshold": 0.45,
        "octave_cost": 0.01,
        "octave_jump_cost": 0.35,
        "voiced_unvoiced_cost": 0.14,
        "crosscheck_enabled": False,
        "crosscheck_model": "full",
        "agreement_cents": 50.0,
        "exclude_disagreeing": True,
        "periodicity_threshold": 0.5,
    }
    if config is None:
        return conf
    p = config.section("features").get("pitch", {})
    if not p:
        return conf
    pass1 = p.get("pass1", {})
    pass2 = p.get("pass2", {})
    cross = p.get("crosscheck", {})
    conf.update(
        pass1=(
            float(pass1.get("floor_hz", DEFAULT_PASS1[0])),
            float(pass1.get("ceiling_hz", DEFAULT_PASS1[1])),
        ),
        q1_scale=float(pass2.get("q1_scale", conf["q1_scale"])),
        q3_scale=float(pass2.get("q3_scale", conf["q3_scale"])),
        min_floor_hz=float(pass2.get("min_floor_hz", conf["min_floor_hz"])),
        max_ceiling_hz=float(pass2.get("max_ceiling_hz", conf["max_ceiling_hz"])),
        min_voiced_frames=int(p.get("min_voiced_frames_for_pass2", conf["min_voiced_frames"])),
        silence_threshold=float(p.get("silence_threshold", conf["silence_threshold"])),
        voicing_threshold=float(p.get("voicing_threshold", conf["voicing_threshold"])),
        octave_cost=float(p.get("octave_cost", conf["octave_cost"])),
        octave_jump_cost=float(p.get("octave_jump_cost", conf["octave_jump_cost"])),
        voiced_unvoiced_cost=float(p.get("voiced_unvoiced_cost", conf["voiced_unvoiced_cost"])),
        crosscheck_enabled=bool(cross.get("enabled", conf["crosscheck_enabled"])),
        crosscheck_model=str(cross.get("model", conf["crosscheck_model"])),
        agreement_cents=float(cross.get("agreement_cents", conf["agreement_cents"])),
        exclude_disagreeing=bool(
            cross.get("exclude_disagreeing_frames", conf["exclude_disagreeing"])
        ),
        periodicity_threshold=float(
            cross.get("periodicity_threshold", conf["periodicity_threshold"])
        ),
    )
    return conf


# --------------------------------------------------------------------------- #
# Word aggregates (spec 9.1)
# --------------------------------------------------------------------------- #


def word_pitch(
    track: PitchTrack,
    words: list[Word],
    *,
    grid: FrameGrid | None = None,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """Aggregate a pitch track to one row per word.

    Columns are :data:`PITCH_WORD_COLUMNS`: mean, max, min, range, slope and
    standard deviation in semitones re the speaker median, plus the voiced frame
    count and fraction so a consumer can tell "flat" from "unmeasured".

    Only :attr:`PitchTrack.usable` frames contribute - a frame the cross-check
    contradicted is absent from every statistic, not down-weighted.
    """
    import pandas as pd

    grid = grid or FrameGrid()
    rows = []
    for word in words:
        idx = grid.frame_indices(word.t_start, word.t_end, track.n_frames)
        st = track.f0_st[idx] if idx.size else np.empty(0)
        ok = np.isfinite(st)
        n_voiced = int(ok.sum())
        values = st[ok]
        times = track.times[idx][ok] if idx.size else np.empty(0)
        rows.append(
            {
                "f0_mean_st": float(values.mean()) if n_voiced else np.nan,
                "f0_max_st": float(values.max()) if n_voiced else np.nan,
                "f0_min_st": float(values.min()) if n_voiced else np.nan,
                "f0_range_st": float(values.max() - values.min()) if n_voiced else np.nan,
                "f0_slope_st_per_s": _slope(times, values),
                "f0_std_st": float(values.std(ddof=0)) if n_voiced > 1 else np.nan,
                "f0_n_voiced": n_voiced,
                "f0_voiced_frac": (float(n_voiced) / idx.size) if idx.size else np.nan,
            }
        )
    table = pd.DataFrame(rows, columns=list(PITCH_WORD_COLUMNS))
    table.index = pd.Index([w.idx for w in words], name="unit_idx")
    table.attrs.update(unit="word", file_id=file_id, channel_id=channel_id)
    return table


def _slope(times: np.ndarray, values: np.ndarray) -> float:
    """Least-squares slope of ``values`` against ``times``, or ``NaN``."""
    if values.size < 2 or np.ptp(times) <= 0:
        return float("nan")
    return float(np.polyfit(times, values, 1)[0])
