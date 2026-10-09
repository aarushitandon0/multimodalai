"""Boundary refinement with ruptures PELT (spec 10.5, [R33]).

A region's initial boundaries come from the alignment: the first word's onset
and the last word's offset. That is honest but coarse. The audible onset of
rushing rarely coincides with a word boundary - it is wherever the speaker
accelerated, which can be mid-word - and the evaluation measures onset/offset
error in milliseconds, so a word-quantised boundary costs real metric points.

So within **+-300 ms** of each boundary, this module runs change-point
detection on the frame-level deviation of the region's primary feature:

* **PELT** search with an **RBF cost** and a **fixed penalty**
  (``ruptures.Pelt(model="rbf")``, verified against ruptures v1.1.10). RBF is a
  kernel cost on the *distribution* of the signal, so it detects a change in
  variability - a pitch track flattening out - as readily as a change in level,
  which an ``l2`` cost would miss entirely for ``monotone``.
* The refined point is accepted **only if it moves less than 300 ms**
  (``refinement.max_shift_s``). A larger jump means PELT latched onto something
  other than this region's edge, and the alignment-derived boundary is the
  better estimate.
* When PELT finds **no** change point in the window, the raw boundary is kept.
  Nudging the boundary to the window edge in that case would manufacture
  precision the signal does not support.

**Both boundaries are kept.** ``Region.t_start_unrefined`` /
``t_end_unrefined`` hold the alignment-derived values and ``Region.refined``
records whether anything moved, because spec 10.5 requires the evaluation to
report metrics with *and* without refinement (ablation 3, spec 14.2).

**Which frame curve?** The primary features of ``rushing``/``dragging`` (local
rate) and of the pause labels (gap duration) are per-unit quantities with no
frame-rate measurement of their own, so ``detection.refinement.frame_signal``
maps each to the closest frame-level correlate this project computes: the
prominence composite for rate, since it carries F0, energy and normalised
duration together, and the relative-energy curve for a pause edge, where the
change point genuinely *is* the speech/silence transition. Features that do
have a frame curve - F0 in semitones, energy z, the prominence response,
spectral tilt - map to their own.

The deviation is formed inside the window by subtracting the window median and
dividing by its MAD, so PELT sees a scale-free curve and one fixed penalty is
meaningful across features. This module does not alter any score: a refined
boundary changes only *where* the region is said to start and stop.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.envelope.build import MAD_SCALE

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.types import FeatureTable, Region

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MAX_SHIFT_S",
    "DEFAULT_PENALTY",
    "DEFAULT_WINDOW_S",
    "RefinementSettings",
    "frame_signal_for",
    "refine_boundary",
    "refine_regions",
    "window_deviation",
]


DEFAULT_WINDOW_S: float = 0.300
DEFAULT_MAX_SHIFT_S: float = 0.300
DEFAULT_PENALTY: float = 3.0
DEFAULT_FALLBACK_SIGNAL: str = "energy_z"


@dataclass(frozen=True)
class RefinementSettings:
    """The refinement parameters (``detection.refinement``).

    Attributes:
        enabled: Whether to refine at all.
        window_s: Half-width of the search window around each boundary.
        max_shift_s: Reject a refinement that moves further than this.
        penalty: Fixed PELT penalty.
        min_size_frames: ``ruptures`` ``min_size``; 3 frames = 30 ms.
        jump_frames: ``ruptures`` ``jump``; 1 evaluates every frame.
        require_change_point: Keep the raw boundary when PELT finds nothing.
        frame_signal: Canonical primary feature to frame-column map.
        fallback_signal: Frame column used when the map has no entry.
    """

    enabled: bool = True
    window_s: float = DEFAULT_WINDOW_S
    max_shift_s: float = DEFAULT_MAX_SHIFT_S
    penalty: float = DEFAULT_PENALTY
    min_size_frames: int = 3
    jump_frames: int = 1
    require_change_point: bool = True
    frame_signal: Mapping[str, str] = field(default_factory=dict)
    fallback_signal: str = DEFAULT_FALLBACK_SIGNAL

    @classmethod
    def from_config(cls, config: Config, **overrides: Any) -> RefinementSettings:
        """Read ``detection.refinement``, then apply overrides.

        Raises:
            ValueError: The config asks for a search or cost this module does
                not implement; spec 10.5 fixes them at PELT and RBF.
        """
        block = config.detection["refinement"]
        search = str(block.get("search", "pelt")).lower()
        cost = str(block.get("cost", "rbf")).lower()
        if search != "pelt" or cost != "rbf":
            raise ValueError(
                f"spec 10.5 fixes refinement at PELT with an RBF cost; config asks for "
                f"{search!r}/{cost!r}"
            )
        values: dict[str, Any] = {
            "enabled": bool(block["enabled"]),
            "window_s": float(block["window_s"]),
            "max_shift_s": float(block["max_shift_s"]),
            "penalty": float(block["penalty"]),
            "min_size_frames": int(block.get("min_size_frames", 3)),
            "jump_frames": int(block.get("jump_frames", 1)),
            "require_change_point": bool(block.get("require_exit_threshold_crossing", True)),
            "frame_signal": {
                str(k): str(v) for k, v in dict(block.get("frame_signal", {})).items()
            },
            "fallback_signal": str(block.get("frame_signal_fallback", DEFAULT_FALLBACK_SIGNAL)),
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def frame_signal_for(
    primary_feature: str, frames: FeatureTable, settings: RefinementSettings
) -> str | None:
    """Return the frame column to refine ``primary_feature`` on, or ``None``.

    Tries, in order: the configured mapping, the feature name itself (so a
    feature whose frame column shares its name needs no entry), and the
    configured fallback. ``None`` means nothing usable is in ``frames``, and the
    caller must keep the raw boundary.
    """
    columns = set(frames.columns)
    mapped = settings.frame_signal.get(primary_feature)
    for candidate in (mapped, primary_feature, settings.fallback_signal):
        if candidate and candidate in columns:
            return str(candidate)
    logger.debug(
        "no frame curve for primary feature %r (frames have %d columns); boundary kept raw",
        primary_feature,
        len(columns),
    )
    return None


def window_deviation(
    frames: FeatureTable, signal: str, t_centre: float, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, deviation)`` inside ``+-window_s`` of ``t_centre``.

    The deviation is the signal minus the window's own median, divided by
    ``1.4826 * MAD`` of the window, so one fixed penalty means the same thing
    whether the curve is in semitones, dB or dimensionless prominence. Frames
    with a non-finite value are dropped, and their timestamps with them, so the
    change point is reported at a real instant.
    """
    times = np.asarray(frames["time_s"], dtype=np.float64)
    values = np.asarray(frames[signal], dtype=np.float64)
    inside = np.flatnonzero(
        (times >= t_centre - window_s) & (times <= t_centre + window_s) & np.isfinite(values)
    )
    if inside.size == 0:
        return np.empty(0), np.empty(0)
    local_t = times[inside]
    local_v = values[inside]
    centre = float(np.median(local_v))
    spread = MAD_SCALE * float(np.median(np.abs(local_v - centre)))
    if not np.isfinite(spread) or spread <= 0:
        spread = 1.0
    return local_t, (local_v - centre) / spread


def refine_boundary(
    frames: FeatureTable,
    signal: str,
    t_raw: float,
    settings: RefinementSettings,
) -> tuple[float, bool, str]:
    """Refine one boundary, returning ``(t, moved, reason)``.

    Args:
        frames: The frame table of the recording being analysed.
        signal: Frame column to search.
        t_raw: The alignment-derived boundary.
        settings: Window, penalty and shift limit.

    Returns:
        ``(t, moved, reason)``. ``t`` is ``t_raw`` unchanged whenever the
        refinement was rejected, and ``reason`` says which guard rejected it:
        ``"accepted"``, ``"no_frames"``, ``"too_short"``, ``"no_change_point"``
        or ``"shift_too_large"``. The guarantee the tests rely on is that
        ``abs(t - t_raw) < settings.max_shift_s`` always holds.
    """
    import ruptures as rpt

    times, deviation = window_deviation(frames, signal, float(t_raw), settings.window_s)
    if times.size == 0:
        return float(t_raw), False, "no_frames"
    # PELT needs at least two admissible segments to have anything to find.
    if times.size < 2 * max(1, settings.min_size_frames):
        return float(t_raw), False, "too_short"

    algo = rpt.Pelt(
        model="rbf",
        min_size=max(1, int(settings.min_size_frames)),
        jump=max(1, int(settings.jump_frames)),
    ).fit(deviation.reshape(-1, 1))
    # ruptures returns breakpoints as end indices, the last of which is always
    # the series length and is therefore not a change point.
    breaks = [b for b in algo.predict(pen=float(settings.penalty)) if 0 < b < times.size]
    if not breaks:
        return float(t_raw), False, "no_change_point"

    candidate = float(times[min(breaks, key=lambda b: abs(times[b] - float(t_raw)))])
    shift = abs(candidate - float(t_raw))
    if shift >= settings.max_shift_s:
        return float(t_raw), False, "shift_too_large"
    return candidate, candidate != float(t_raw), "accepted"


def refine_regions(
    regions: Sequence[Region],
    frames: FeatureTable | None,
    *,
    config: Config | None = None,
    settings: RefinementSettings | None = None,
    primary_features: Mapping[int, str] | None = None,
) -> list[Region]:
    """Refine every region's boundaries in place and return the list (spec 10.5).

    The raw boundaries are preserved in ``t_start_unrefined`` /
    ``t_end_unrefined`` before anything moves, and ``refined`` is set only when
    at least one boundary actually changed, so the +-refinement ablation can be
    run from a single analysis.

    Args:
        regions: Regions from :func:`rostrum.detect.regions.build_regions`.
        frames: The recording's frame table. ``None`` disables refinement and
            the raw boundaries are kept, which is the honest behaviour when a
            caller has no frame curves rather than an error.
        config: Supplies ``detection.refinement``.
        settings: Explicit settings, overriding the config.
        primary_features: ``{region_id: canonical primary feature}``. Without
            it the first evidence row's feature is used, which is the primary
            feature by construction (:func:`rostrum.detect.regions.evidence_rows`
            forces it first).

    Returns:
        The same region objects, mutated, in input order.
    """
    if settings is None:
        settings = (
            RefinementSettings.from_config(config) if config is not None else RefinementSettings()
        )

    if not settings.enabled or frames is None or "time_s" not in getattr(frames, "columns", ()):
        for region in regions:
            region.t_start_unrefined = region.t_start
            region.t_end_unrefined = region.t_end
            region.refined = False
        if settings.enabled and frames is None:
            logger.info("refinement skipped: no frame curves supplied")
        return list(regions)

    accepted = 0
    for region in regions:
        raw_start, raw_end = float(region.t_start), float(region.t_end)
        region.t_start_unrefined = raw_start
        region.t_end_unrefined = raw_end

        primary = (primary_features or {}).get(region.region_id)
        if primary is None:
            primary = region.evidence[0].feature if region.evidence else ""
        column = frame_signal_for(str(primary), frames, settings)
        if column is None:
            region.refined = False
            continue

        t0, moved_start, reason_start = refine_boundary(frames, column, raw_start, settings)
        t1, moved_end, reason_end = refine_boundary(frames, column, raw_end, settings)
        # A refinement must not invert or collapse the region; if it would, the
        # raw pair is kept rather than half-applied.
        if t1 <= t0:
            logger.debug(
                "region %d: refined boundaries %.3f/%.3f would invert the region; keeping raw",
                region.region_id,
                t0,
                t1,
            )
            region.refined = False
            continue

        region.t_start, region.t_end = t0, t1
        region.refined = bool(moved_start or moved_end)
        accepted += int(region.refined)
        logger.debug(
            "region %d (%s on %s): start %s (%+.0f ms), end %s (%+.0f ms)",
            region.region_id,
            region.label,
            column,
            reason_start,
            (t0 - raw_start) * 1000,
            reason_end,
            (t1 - raw_end) * 1000,
        )

    logger.info(
        "refinement: %d of %d region(s) moved (penalty=%g, window=%.0f ms, max shift=%.0f ms)",
        accepted,
        len(regions),
        settings.penalty,
        settings.window_s * 1000,
        settings.max_shift_s * 1000,
    )
    return list(regions)
