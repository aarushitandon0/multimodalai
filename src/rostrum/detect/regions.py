"""From a per-unit score to a timestamped region (spec 10.4).

This is where the project earns the word *grounding*: a label with no interval
is an opinion, and an interval taken straight from whichever single word
happened to cross a threshold is a jittery one. Spec 10.4 fixes the pipeline,
and it runs in exactly this order:

1. **3-unit median smoothing** of the signed score. A single word can cross
   2.5 sigma because MFA put a boundary 30 ms late; three consecutive words
   cannot. The median (not the mean) is used so one extreme unit cannot drag
   its neighbours over the line.
2. **Hysteresis**: open a region where the smoothed score exceeds
   ``tau_in = 2.5``, and keep it open while the score stays above
   ``tau_out = 1.5``. One threshold would chatter - a score oscillating around
   2.5 would produce five regions where a listener hears one - and the gap
   between the two thresholds is what makes the extent stable.
3. **Merge** regions of the same label less than 200 ms apart, then **drop**
   regions shorter than 250 ms. The two pause labels are exempt from the drop,
   because a pause flaw legitimately occupies a single short gap: a 120 ms
   silence where every reference reader took 500 ms *is* the flaw.
4. **Boundaries** come from the alignment, never from the score: the first
   word's onset and the last word's offset, or the gap's own edges for the two
   pause labels. Refinement (:mod:`rostrum.detect.refine`) may then move them
   by at most 300 ms.

Each surviving region carries its **top-3 evidence rows by** ``|z|``, so the
explanation is built from the same numbers the detection used, and
``peak_z`` / ``mean_excess_z`` so the rubric can convert a region into a
deviation mass without re-reading the tables (spec 11).

Everything here is a pure function of a :class:`~rostrum.detect.zscore.Signal`
and the tables behind it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features import names as N
from rostrum.types import Evidence, Region

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations

    from rostrum.config import Config
    from rostrum.detect.zscore import Signal, ZResult
    from rostrum.types import RubricCategory

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MERGE_GAP_S",
    "DEFAULT_MIN_DURATION_S",
    "DEFAULT_TAU_IN",
    "DEFAULT_TAU_OUT",
    "PAUSE_LABELS",
    "RegionThresholds",
    "Segment",
    "build_regions",
    "drop_short",
    "evidence_rows",
    "hysteresis_segments",
    "label_category",
    "label_counts",
    "median_smooth",
    "merge_adjacent",
    "segments_for_signal",
]


DEFAULT_TAU_IN: float = 2.5
DEFAULT_TAU_OUT: float = 1.5
DEFAULT_MERGE_GAP_S: float = 0.200
DEFAULT_MIN_DURATION_S: float = 0.250
DEFAULT_SMOOTH_UNITS: int = 3

PAUSE_LABELS: frozenset[str] = frozenset({"missing_pause", "hesitation_pause"})
"""Labels exempt from the minimum-duration rule (spec 10.4(3))."""

EXTEND_ON_CHOICES: frozenset[str] = frozenset({"smoothed", "raw"})
"""Which score the hysteresis stay condition may read."""

DEFAULT_EXTEND_ON: str = "smoothed"
"""Spec 10.4 smooths (step 1) before the hysteresis (step 2), so this is the
literal reading. See :func:`hysteresis_segments` for its measured cost."""

_CATEGORY_FALLBACK: dict[str, str] = {
    "rushing": "pace",
    "dragging": "pace",
    "missing_pause": "pausing",
    "hesitation_pause": "pausing",
    "monotone": "intonation",
    "misplaced_stress": "emphasis",
    "low_energy": "energy",
    "mumbling": "clarity",
}
"""Used only when no config is supplied; ``scoring.categories`` is the source."""


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegionThresholds:
    """The tunable part of region formation (spec 10.4(5)).

    Fitted per LOTO fold on the fold's training texts only
    (:mod:`rostrum.detect.tune`), never on the held-out text.

    Attributes:
        tau_in: Open a region above this smoothed score.
        tau_out: Keep it open above this.
        merge_gap_s: Merge same-label regions closer than this.
        min_duration_s: Drop regions shorter than this.
        smooth_units: Width of the median smoother, in units.
        smooth_exempt_labels: Labels that are never smoothed.
        smooth_exempt_units: Unit kinds that are never smoothed.
        extend_on: ``"smoothed"`` or ``"raw"``; which score the stay condition
            reads. See :func:`hysteresis_segments`.
    """

    tau_in: float = DEFAULT_TAU_IN
    tau_out: float = DEFAULT_TAU_OUT
    merge_gap_s: float = DEFAULT_MERGE_GAP_S
    min_duration_s: float = DEFAULT_MIN_DURATION_S
    smooth_units: int = DEFAULT_SMOOTH_UNITS
    smooth_exempt_labels: frozenset[str] = PAUSE_LABELS
    smooth_exempt_units: frozenset[str] = frozenset({"phrase"})
    extend_on: str = DEFAULT_EXTEND_ON

    def width_for(self, label: str, unit: str) -> int:
        """Smoother width for one signal; 1 means "do not smooth".

        Spec 10.4(1) prescribes a 3-unit median filter to absorb a single-unit
        alignment artefact. Two signals must opt out or the filter destroys
        them outright, and both exemptions are configured, not assumed:

        * a **pause** flaw lives at exactly one gap, so a 3-gap median replaces
          the deviation with its own neighbours - the identical reason spec
          10.4(3) exempts these labels from the minimum-duration rule;
        * a **phrase** is already about six words, so a 3-phrase median smooths
          across eighteen words, which is nothing like the one-word artefact
          the filter is for.
        """
        if str(label) in self.smooth_exempt_labels or str(unit) in self.smooth_exempt_units:
            return 1
        return int(self.smooth_units)

    def __post_init__(self) -> None:
        if not self.tau_out <= self.tau_in:
            raise ValueError(
                f"hysteresis needs tau_out <= tau_in, got tau_out={self.tau_out} "
                f"> tau_in={self.tau_in}"
            )
        if self.smooth_units < 1 or self.smooth_units % 2 == 0:
            raise ValueError(f"smooth_units must be a positive odd number, got {self.smooth_units}")
        if self.extend_on not in EXTEND_ON_CHOICES:
            raise ValueError(
                f"extend_on must be one of {sorted(EXTEND_ON_CHOICES)}, got {self.extend_on!r}"
            )

    @classmethod
    def from_config(cls, config: Config, **overrides: Any) -> RegionThresholds:
        """Read the defaults from ``detection.regions``, then apply overrides."""
        regions = config.detection["regions"]
        smoothing = config.detection["deviation"]["smoothing"]
        values: dict[str, Any] = {
            "tau_in": float(regions["tau_in"]),
            "tau_out": float(regions["tau_out"]),
            "merge_gap_s": float(regions["merge_gap_s"]),
            "min_duration_s": float(regions["min_duration_s"]),
            "smooth_units": int(smoothing["median_filter_words"]),
            "extend_on": str(regions.get("extend_on", DEFAULT_EXTEND_ON)),
        }
        values["smooth_exempt_labels"] = frozenset(
            str(x) for x in smoothing.get("exempt_labels", PAUSE_LABELS)
        )
        values["smooth_exempt_units"] = frozenset(
            str(x) for x in smoothing.get("exempt_units", ("phrase",))
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


def label_category(label: str, config: Config | None = None) -> RubricCategory:
    """Return the rubric category ``label`` belongs to (spec 11).

    The map lives in ``configs/scoring.yaml``; the literal fallback is only for
    unit tests that construct regions without a config.

    Raises:
        KeyError: The label belongs to no category.
    """
    if config is not None:
        for category, block in config.scoring["categories"].items():
            if label in {str(x) for x in block.get("labels", ())}:
                return str(category)  # type: ignore[return-value]
    if label in _CATEGORY_FALLBACK:
        return _CATEGORY_FALLBACK[label]  # type: ignore[return-value]
    raise KeyError(f"no rubric category for label {label!r}")


# --------------------------------------------------------------------------- #
# Smoothing and hysteresis
# --------------------------------------------------------------------------- #


def median_smooth(score: Any, width: int = DEFAULT_SMOOTH_UNITS) -> Any:
    """Median-filter a per-unit score over ``width`` units (spec 10.4(1)).

    ``NaN`` is treated as missing rather than as a value: a window's median is
    taken over whatever is finite in it, and a window with nothing finite stays
    ``NaN``.

    **The edges shrink symmetrically**, so the window stays centred and stays
    odd: unit ``i`` uses a half-width of ``min(half, i, n - 1 - i)``, which
    leaves the first and last units untouched. Two alternatives were rejected:

    * *padding* with the edge value would invent evidence at the start of the
      utterance, which is exactly where the onset metric is measured;
    * *truncating* on one side only leaves an even-length window at the edge,
      and the median of two values is their mean - so a score profile of
      ``[0, 9, 9, 9, 0]`` would come back with ``4.5`` at unit 0 and open a
      spurious region there. Dilating a region is worse than failing to smooth
      a spike on the very first word, because a false onset can land anywhere
      a region begins while an edge spike can only land at two units.

    Args:
        score: Signed score per unit, indexed by unit index.
        width: Odd window width in units; 1 returns a copy unchanged.

    Returns:
        A Series on the same index.

    Raises:
        ValueError: ``width`` is even or below 1.
    """
    import pandas as pd

    if width < 1 or width % 2 == 0:
        raise ValueError(f"median_smooth needs an odd width >= 1, got {width}")
    if width == 1:
        return score.copy()

    values = np.asarray(score.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
    n = values.size
    half = width // 2
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for i in range(n):
        reach = min(half, i, n - 1 - i)
        window = values[i - reach : i + reach + 1]
        finite = window[np.isfinite(window)]
        if finite.size:
            out[i] = float(np.median(finite))
    return pd.Series(out, index=score.index, name=getattr(score, "name", None))


@dataclass(frozen=True)
class Segment:
    """A run of consecutive units above the hysteresis thresholds.

    Attributes:
        first: First unit index in the run, inclusive.
        last: Last unit index in the run, inclusive.
        peak: Largest smoothed score inside the run.
    """

    first: int
    last: int
    peak: float


def hysteresis_segments(
    score: Any,
    *,
    tau_in: float = DEFAULT_TAU_IN,
    tau_out: float = DEFAULT_TAU_OUT,
    gate: Any | None = None,
    extend_score: Any | None = None,
) -> list[Segment]:
    """Open runs above ``tau_in``, extend them while above ``tau_out`` (spec 10.4(2)).

    The score must already be oriented so that positive means "towards the
    flaw" (see :mod:`rostrum.detect.zscore`), so this function never looks at
    the sign of a z and the same code serves ``rushing`` and ``dragging``.

    A run extends both *backwards* and forwards: a score profile of
    ``[1.8, 2.9, 1.7]`` is one region spanning all three units, because the
    first unit was already above the stay-threshold when the trigger fired.
    Taking only the forward direction would clip every onset, which is the
    metric the evaluation reports.

    **``extend_score`` separates the two thresholds' sources.** Spec 10.4 runs
    the median smoother (step 1) before the hysteresis (step 2), so by default
    both the trigger and the extension read the smoothed score. That has a
    measurable cost at a region edge: the 3-point window there holds one
    in-flaw value, the edge unit and the next unit, so its median is
    ``max(edge, next)`` and a unit whose own measurement is far below
    ``tau_out`` can be carried above it by its neighbour. The region then
    dilates by one unit - about 0.4 s for a word, past both the 300 ms
    refinement limit and the offset collar, so the event is scored as a miss
    *and* a false positive.

    Passing the unsmoothed score as ``extend_score`` keeps the smoother where
    its stated purpose lies - stopping a single-unit artefact from *opening* a
    region - while letting extent follow measurements each unit actually
    produced. ``RegionThresholds.extend_on`` selects between the two and the
    difference is reported as an ablation rather than silently adopted.

    Args:
        score: Smoothed, oriented score per unit index; drives the trigger.
        tau_in: Trigger threshold.
        tau_out: Extension threshold; must not exceed ``tau_in``.
        gate: Optional boolean mask per unit. A unit gated ``False`` can neither
            trigger a region nor be part of one - that is how a gap whose text
            expects no pause is kept out of ``missing_pause``.
        extend_score: Score the stay condition reads. ``None`` uses ``score``.

    Returns:
        Segments in index order, non-overlapping. ``peak`` is measured on
        ``score``, so it is comparable across both extension modes.

    Raises:
        ValueError: ``tau_out > tau_in``, or ``extend_score`` has a different
            index from ``score``.
    """
    if tau_out > tau_in:
        raise ValueError(f"hysteresis needs tau_out <= tau_in, got {tau_out} > {tau_in}")

    index = [int(i) for i in score.index]
    values = np.asarray(score.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
    if extend_score is None:
        extend_values = values
    else:
        if list(extend_score.index) != list(score.index):
            raise ValueError("extend_score must share the index of score")
        extend_values = np.asarray(
            extend_score.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64
        )

    if gate is None:
        allowed = np.ones(values.shape, dtype=bool)
    else:
        allowed = np.asarray(
            gate.reindex(score.index).fillna(False).to_numpy(dtype=bool), dtype=bool
        )

    above_in = np.isfinite(values) & (values > tau_in) & allowed
    above_out = np.isfinite(extend_values) & (extend_values > tau_out) & allowed

    segments: list[Segment] = []
    i = 0
    n = values.size
    while i < n:
        if not above_in[i]:
            i += 1
            continue
        start = i
        while start > 0 and above_out[start - 1]:
            start -= 1
        if segments and start <= segments[-1].last:
            start = segments[-1].last + 1
        end = i
        while end + 1 < n and above_out[end + 1]:
            end += 1
        run = values[start : end + 1]
        segments.append(
            Segment(
                first=index[start],
                last=index[end],
                peak=float(np.nanmax(run)) if np.isfinite(run).any() else float("nan"),
            )
        )
        i = end + 1

    return segments


def segments_for_signal(signal: Signal, thresholds: RegionThresholds) -> tuple[list[Segment], Any]:
    """Smooth ``signal``'s score and return its hysteresis segments.

    ``thresholds.extend_on`` decides whether the stay condition reads the
    smoothed score (spec 10.4 literal) or the unsmoothed one; see
    :func:`hysteresis_segments`.

    Returns:
        ``(segments, smoothed_score)`` - the smoothed score is returned because
        the region's ``peak_z`` and ``mean_excess_z`` are measured on it, not on
        the raw score, and the caller would otherwise recompute it.
    """
    smoothed = median_smooth(signal.score, thresholds.width_for(str(signal.label), signal.unit))
    segments = hysteresis_segments(
        smoothed,
        tau_in=thresholds.tau_in,
        tau_out=thresholds.tau_out,
        gate=signal.gate,
        extend_score=signal.score if thresholds.extend_on == "raw" else None,
    )
    return segments, smoothed


# --------------------------------------------------------------------------- #
# Spans and evidence
# --------------------------------------------------------------------------- #


def _span_and_indices(
    segment: Segment, result: ZResult, *, unit: str
) -> tuple[float, float, list[int], int | None, str]:
    """Resolve a segment of units to ``(t0, t1, word_idx, gap_idx, words)``.

    Spec 10.4(4): word-based flaws take the first word's onset and the last
    word's offset; pause flaws take the gap's own edges; a phrase-based flaw
    takes the phrase span and reports the words it covers.
    """
    table = result.table
    lo, hi = int(segment.first), int(segment.last)
    rows = table.loc[(table.index >= lo) & (table.index <= hi)]
    if rows.empty:  # pragma: no cover - a segment always comes from the index
        raise KeyError(f"no {unit} rows between {lo} and {hi}")

    t0 = float(rows["t_start"].min())
    t1 = float(rows["t_end"].max())

    if unit == "gap":
        prev_idx = rows.get("gap_prev_word_idx")
        next_idx = rows.get("gap_next_word_idx")
        word_idx = (
            [int(prev_idx.iloc[0]), int(next_idx.iloc[-1])]
            if prev_idx is not None and next_idx is not None
            else [lo, hi]
        )
        return t0, t1, word_idx, lo, ""

    if unit == "phrase":
        first = rows.get("first_word_idx")
        last = rows.get("last_word_idx")
        word_idx = (
            [int(first.iloc[0]), int(last.iloc[-1])]
            if first is not None and last is not None
            else [lo, hi]
        )
        return t0, t1, word_idx, None, ""

    words = " ".join(str(w) for w in rows["text"]) if "text" in rows.columns else ""
    return t0, t1, [lo, hi], None, words


def evidence_rows(
    segment: Segment,
    signal: Signal,
    result: ZResult,
    *,
    top_k: int = 3,
    extra_results: Mapping[str, ZResult] | None = None,
) -> list[Evidence]:
    """The ``top_k`` features behind a region, ranked by mean ``|z|`` (spec 12).

    The ranking is over the mean ``|z|`` *inside the region*, not over the peak:
    a feature that is 3 sigma out across the whole region explains it better
    than one that spiked once. The signal's own
    :attr:`~rostrum.detect.zscore.Signal.evidence_features` are always
    considered first and the primary feature is forced into the list, so the
    headline number of a card is never crowded out by a correlate.

    Args:
        segment: The unit run.
        signal: The signal that produced it.
        result: The z comparison the signal was built from.
        top_k: How many rows to keep.
        extra_results: Other units' comparisons, so a word region may quote the
            gap beside it. Unused for now; accepted so callers need not change.

    Returns:
        Up to ``top_k`` :class:`~rostrum.types.Evidence` rows, strongest first.
    """
    lo, hi = int(segment.first), int(segment.last)
    inside = (result.z.index >= lo) & (result.z.index <= hi)
    if not inside.any():  # pragma: no cover
        return []

    try:
        primary = N.resolve(signal.primary_feature, result.features)
    except KeyError:
        # A derived primary such as `clarity_composite` is not a column at all;
        # its components are already in `evidence_features`.
        primary = []
    # dict.fromkeys keeps the first occurrence: the primary feature usually
    # appears in `evidence_features` too, and a column listed twice would be
    # ranked twice and printed twice on the card.
    preferred = list(
        dict.fromkeys(c for c in (*primary, *signal.evidence_features) if c in result.z.columns)
    )
    candidates = preferred + [c for c in result.features if c not in preferred]

    scored: list[tuple[float, str]] = []
    for column in candidates:
        values = result.z.loc[inside, column].to_numpy(dtype=np.float64, na_value=np.nan)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        rank = float(np.abs(finite).mean())
        # Keep the primary feature first whatever its magnitude: the card's
        # headline must be the quantity the rule actually fired on.
        scored.append((rank + 1e6 if column in primary else rank, column))

    scored.sort(key=lambda pair: -pair[0])
    rows: list[Evidence] = []
    for _, column in scored[: max(0, top_k)]:
        z_vals = result.z.loc[inside, column].to_numpy(dtype=np.float64, na_value=np.nan)
        finite_mask = np.isfinite(z_vals)
        # Quote the unit where the deviation was largest; that is the instant a
        # listener would point at.
        local = int(np.argmax(np.where(finite_mask, np.abs(z_vals), -np.inf)))
        unit_idx = int(result.z.index[inside][local])
        gap_label = None
        if result.unit == "gap" and "gap_label" in result.table.columns:
            gap_label = str(result.table.loc[unit_idx, "gap_label"])
        rows.append(
            Evidence(
                feature=N.canonical(column),
                value=float(result.value.loc[unit_idx, column]),
                env_median=float(result.median.loc[unit_idx, column]),
                env_sigma=float(result.sigma.loc[unit_idx, column]),
                z=float(result.z.loc[unit_idx, column]),
                unit=N.display_unit(column),
                gap_label=gap_label,  # type: ignore[arg-type]
                n_references=int(result.n_references.loc[unit_idx]),
                channel_matched=bool(result.channel_matched),
            )
        )
    return rows


def _excess(values: np.ndarray, tau_out: float) -> float:
    """Mean of ``(score - tau_out)_+`` over the finite values (spec 11)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.maximum(finite - tau_out, 0.0).mean())


# --------------------------------------------------------------------------- #
# Merge and drop
# --------------------------------------------------------------------------- #


def merge_adjacent(regions: Sequence[Region], merge_gap_s: float) -> list[Region]:
    """Merge same-label regions less than ``merge_gap_s`` apart (spec 10.4(3)).

    Two regions merge only when they carry the same label *and* the same
    variant: a ``low_energy`` level region and a ``fading`` slope region 100 ms
    apart are two different statements about the delivery and stay separate.

    The merged region keeps the union of the spans and word indices, the larger
    ``peak_z``, the duration-weighted ``mean_excess_z``, and the evidence of
    whichever part deviated more - so the card still quotes a real measurement
    at a real instant rather than an average of two.
    """
    if not regions:
        return []

    out: list[Region] = []
    for region in sorted(regions, key=lambda r: (r.label, r.context, r.t_start)):
        previous = out[-1] if out else None
        same = (
            previous is not None
            and previous.label == region.label
            and previous.context == region.context
        )
        if same and region.t_start - previous.t_end < merge_gap_s:
            d_prev = max(previous.duration, 1e-9)
            d_this = max(region.duration, 1e-9)
            stronger = (
                region if _abs_or_zero(region.peak_z) > _abs_or_zero(previous.peak_z) else previous
            )
            previous.t_end = max(previous.t_end, region.t_end)
            previous.t_start = min(previous.t_start, region.t_start)
            previous.word_idx = [
                min(previous.word_idx[0], region.word_idx[0]),
                max(previous.word_idx[-1], region.word_idx[-1]),
            ]
            previous.peak_z = stronger.peak_z
            previous.evidence = stronger.evidence
            previous.mean_excess_z = float(
                np.nansum([previous.mean_excess_z * d_prev, region.mean_excess_z * d_this])
                / (d_prev + d_this)
            )
            previous.envelope_n = max(previous.envelope_n, region.envelope_n)
            if previous.gap_idx is None:
                previous.gap_idx = region.gap_idx
            continue
        out.append(region)

    out.sort(key=lambda r: (r.t_start, r.label, r.context))
    return out


def _abs_or_zero(value: float | None) -> float:
    """``|value|``, or 0 for ``None``/``NaN``; used only for comparisons."""
    if value is None or not np.isfinite(value):
        return 0.0
    return abs(float(value))


def drop_short(
    regions: Sequence[Region],
    min_duration_s: float,
    *,
    exempt_labels: Iterable[str] = PAUSE_LABELS,
) -> list[Region]:
    """Drop regions shorter than ``min_duration_s``, pause labels exempt.

    The exemption is in spec 10.4(3) and is not a convenience: a pause flaw is
    located at one gap, and a gap that is *too short* is by definition a short
    region. Applying the floor to it would make ``missing_pause`` undetectable
    exactly when it is most severe.
    """
    exempt = {str(label) for label in exempt_labels}
    kept: list[Region] = []
    for region in regions:
        if region.duration >= min_duration_s or region.label in exempt:
            kept.append(region)
        else:
            logger.debug(
                "dropped %s at %.3f-%.3f s (%.0f ms < %.0f ms)",
                region.label,
                region.t_start,
                region.t_end,
                region.duration * 1000,
                min_duration_s * 1000,
            )
    return kept


# --------------------------------------------------------------------------- #
# The whole of 10.4
# --------------------------------------------------------------------------- #


def build_regions(
    signals: Sequence[Signal],
    z_by_unit: Mapping[str, ZResult],
    *,
    config: Config | None = None,
    thresholds: RegionThresholds | None = None,
    top_k_evidence: int = 3,
) -> list[Region]:
    """Run the whole of spec 10.4 over every signal and return the regions.

    Args:
        signals: Oriented per-unit scores from
            :func:`rostrum.detect.zscore.trigger_signals`.
        z_by_unit: ``{unit: ZResult}``, the comparisons the signals came from.
        config: Supplies thresholds and the label-to-category map.
        thresholds: Explicit thresholds, overriding the config. This is what a
            LOTO fold passes after tuning on its training texts.
        top_k_evidence: Evidence rows per region.

    Returns:
        Regions sorted by onset, ``region_id`` assigned in that order, with
        ``t_start_unrefined`` / ``t_end_unrefined`` already set to the
        alignment-derived boundaries so the refinement ablation has a baseline
        even when refinement is switched off.

    Raises:
        KeyError: A signal names a unit absent from ``z_by_unit``.
    """
    if thresholds is None:
        thresholds = (
            RegionThresholds.from_config(config) if config is not None else RegionThresholds()
        )

    exempt: Iterable[str] = PAUSE_LABELS
    if config is not None:
        exempt = [str(x) for x in config.detection["regions"]["min_duration_exempt_labels"]]

    raw: list[Region] = []
    for signal in signals:
        result = z_by_unit.get(signal.unit)
        if result is None:
            raise KeyError(
                f"signal {signal.key!r} is indexed on {signal.unit!r} but no such comparison "
                f"was supplied; have {sorted(z_by_unit)}"
            )
        segments, smoothed = segments_for_signal(signal, thresholds)
        for segment in segments:
            t0, t1, word_idx, gap_idx, words = _span_and_indices(segment, result, unit=signal.unit)
            inside = (smoothed.index >= segment.first) & (smoothed.index <= segment.last)
            run = smoothed.loc[inside].to_numpy(dtype=np.float64, na_value=np.nan)
            n_refs = result.n_references.loc[inside]
            raw.append(
                Region(
                    region_id=-1,  # assigned after sorting
                    t_start=t0,
                    t_end=t1,
                    label=signal.label,
                    category=label_category(str(signal.label), config),
                    rule_label=signal.label,
                    envelope_n=int(n_refs.max()) if len(n_refs) else 0,
                    word_idx=word_idx,
                    gap_idx=gap_idx,
                    words=words,
                    evidence=evidence_rows(segment, signal, result, top_k=top_k_evidence),
                    # The variant travels in `context` so merging keeps a level
                    # region and a fading region apart, and the card can say
                    # which of the two fired.
                    context=signal.variant,
                    peak_z=float(segment.peak),
                    mean_excess_z=_excess(run, thresholds.tau_out),
                    t_start_unrefined=t0,
                    t_end_unrefined=t1,
                )
            )

    merged = merge_adjacent(raw, thresholds.merge_gap_s)
    kept = drop_short(merged, thresholds.min_duration_s, exempt_labels=exempt)
    kept.sort(key=lambda r: (r.t_start, str(r.label), r.context))
    for region_id, region in enumerate(kept):
        region.region_id = region_id
        region.t_start_unrefined = region.t_start
        region.t_end_unrefined = region.t_end

    logger.info(
        "regions: %d raw -> %d after merge -> %d kept (tau_in=%.2f, tau_out=%.2f)",
        len(raw),
        len(merged),
        len(kept),
        thresholds.tau_in,
        thresholds.tau_out,
    )
    return kept


def label_counts(regions: Sequence[Region]) -> dict[str, int]:
    """Regions per label key, for a build log or an acceptance print-out."""
    counts: dict[str, int] = {}
    for region in regions:
        key = f"{region.label}:{region.context}" if region.context else str(region.label)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
