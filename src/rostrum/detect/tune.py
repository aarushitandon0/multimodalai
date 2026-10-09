"""Threshold tuning on training folds only (spec 10.4(5), 14.1).

``tau_in = 2.5`` and ``tau_out = 1.5`` are the spec's defaults, not its final
answer: spec 10.4(5) says the thresholds are "tuned per LOTO fold on training
texts only (maximise event-F1)". This module is that search, and the leakage
guard is structural rather than advisory - :func:`grid_search` takes the fold's
*training* cases and the held-out text simply has no way in. Tuning on the test
fold would make every number in the report meaningless, so the chosen values
are written per fold to ``results/tuning.json`` and the evaluation reads them
back rather than re-deriving them.

**The objective** is the event-based F1 of DCASE/``sed_eval`` convention
(spec 14.1): a predicted event matches a reference event of the same label when
their onsets are within a 200 ms collar *and* their offsets are within
``max(200 ms, 20% of the reference length)``. The offset collar scales because a
3-second ``monotone`` region and a 300 ms ``missing_pause`` cannot reasonably be
held to the same absolute tolerance.

:func:`event_f1` is implemented here rather than called from ``sed_eval``,
because the tuning loop scores thousands of grid cells and the matching is 30
lines of greedy assignment that is easier to unit-test, and far cheaper, than a
``dcase_util`` container per cell. The convention it implements is the published
one, and it is not taken on trust: :func:`eval.metrics.crosscheck_event_f1`
scores the same event lists with ``sed_eval`` itself and the LOTO report carries
both numbers plus their difference.

**The grid** covers ``tau_in`` and ``tau_out`` from
``detection.threshold_search`` plus the refinement penalty, because the penalty
changes the boundaries and the boundaries are what the collar measures. Ties are
broken deterministically - highest F1, then the smaller ``tau_in`` (fewer
false positives on clean speech, which spec 14.3 also budgets), then the larger
``tau_out``, then the smaller penalty - so two runs on one machine and two runs
on two machines choose the same cell (CLAUDE.md invariant 13).
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.detect.refine import RefinementSettings, refine_regions
from rostrum.detect.regions import RegionThresholds, build_regions

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.detect.zscore import Signal, ZResult
    from rostrum.types import FeatureTable, Region

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OFFSET_COLLAR_FRAC",
    "DEFAULT_ONSET_COLLAR_S",
    "Event",
    "EventMatch",
    "MatchCounts",
    "TuningCase",
    "TuningResult",
    "event_f1",
    "events_from_regions",
    "grid_search",
    "match_events",
    "read_tuning",
    "write_tuning",
]


DEFAULT_ONSET_COLLAR_S: float = 0.200
DEFAULT_OFFSET_COLLAR_S: float = 0.200
DEFAULT_OFFSET_COLLAR_FRAC: float = 0.20


# --------------------------------------------------------------------------- #
# Events and event-based F1
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Event:
    """One labelled interval, reference or predicted.

    Attributes:
        label: Flaw label. Matching is within a label, never across.
        t_start: Onset in seconds.
        t_end: Offset in seconds.
        file_id: Recording the event belongs to; events never match across files.
    """

    label: str
    t_start: float
    t_end: float
    file_id: str = ""

    @property
    def duration(self) -> float:
        """Event duration in seconds."""
        return float(self.t_end) - float(self.t_start)


def events_from_regions(regions: Sequence[Region], *, file_id: str = "") -> list[Event]:
    """Convert regions to events, dropping the variant tag and coalescing overlaps.

    Two things happen here, both forced by the taxonomy:

    * ``low_energy``/``fading`` becomes plain ``low_energy``. The dataset labels
      eight flaws (CLAUDE.md taxonomy), so the variant must not create a ninth
      class that no reference event can ever match.
    * **overlapping events of one label are merged.** Dropping the variant can
      leave the level region and the fading region of the same passage as two
      ``low_energy`` events over the same seconds; the dataset has one event per
      flaw per span, so scoring them separately would charge a false positive
      for agreeing with itself. The merged event spans their union.

    The regions keep both forms internally - the card still says whether the
    level or the slope fired - so nothing explanatory is lost by merging here.
    """
    by_label: dict[str, list[list[float]]] = {}
    for region in sorted(regions, key=lambda r: (str(r.label), float(r.t_start))):
        spans = by_label.setdefault(str(region.label), [])
        if spans and float(region.t_start) <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], float(region.t_end))
        else:
            spans.append([float(region.t_start), float(region.t_end)])

    return sorted(
        (
            Event(label=label, t_start=t0, t_end=t1, file_id=file_id)
            for label, spans in by_label.items()
            for t0, t1 in spans
        ),
        key=lambda e: (e.t_start, e.label),
    )


@dataclass
class MatchCounts:
    """True positives, false positives and false negatives for one label.

    Attributes:
        tp: Matched reference events.
        fp: Predicted events with no match.
        fn: Reference events with no match.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        """TP / (TP + FP); 0 when nothing was predicted."""
        denominator = self.tp + self.fp
        return float(self.tp / denominator) if denominator else 0.0

    @property
    def recall(self) -> float:
        """TP / (TP + FN); 0 when there is nothing to find."""
        denominator = self.tp + self.fn
        return float(self.tp / denominator) if denominator else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall; 0 when either is 0."""
        p, r = self.precision, self.recall
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, float | int]:
        """JSON form, with the three derived rates rounded for readability."""
        return {
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def _offset_collar(reference: Event, collar_s: float, collar_frac: float) -> float:
    """``max(collar_s, collar_frac * reference length)`` (spec 14.1)."""
    return max(float(collar_s), float(collar_frac) * abs(reference.duration))


def _matches(
    reference: Event, predicted: Event, onset_collar: float, collar_s: float, collar_frac: float
) -> bool:
    """Whether ``predicted`` is an event-based match for ``reference``."""
    if reference.label != predicted.label or reference.file_id != predicted.file_id:
        return False
    if abs(predicted.t_start - reference.t_start) > onset_collar:
        return False
    return abs(predicted.t_end - reference.t_end) <= _offset_collar(
        reference, collar_s, collar_frac
    )


@dataclass(frozen=True)
class EventMatch:
    """One reference event and the predicted event assigned to it.

    Attributes:
        reference: The ground-truth event.
        predicted: The prediction that matched it within the collars.
    """

    reference: Event
    predicted: Event

    @property
    def onset_error(self) -> float:
        """Signed onset error in seconds, positive when the prediction is late."""
        return float(self.predicted.t_start) - float(self.reference.t_start)

    @property
    def offset_error(self) -> float:
        """Signed offset error in seconds, positive when the prediction is late."""
        return float(self.predicted.t_end) - float(self.reference.t_end)


def match_events(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
) -> dict[str, tuple[list[EventMatch], list[Event], list[Event]]]:
    """Assign predictions to references per label, greedily by onset distance.

    This is the single matching implementation the project uses: :func:`event_f1`
    counts its output and :mod:`eval.metrics` measures boundary error on the
    pairs it returns, so the counts and the errors can never come from two
    different notions of "matched".

    Args:
        reference: Ground-truth events.
        predicted: Detected events.
        onset_collar_s: Onset tolerance.
        offset_collar_s: Absolute part of the offset tolerance.
        offset_collar_frac: Relative part, as a fraction of the reference
            event's length.

    Returns:
        ``{label: (matches, unmatched_predictions, unmatched_references)}`` over
        every label present in either list.
    """
    labels = sorted({e.label for e in reference} | {e.label for e in predicted})
    out: dict[str, tuple[list[EventMatch], list[Event], list[Event]]] = {}

    for label in labels:
        refs = sorted(
            (e for e in reference if e.label == label), key=lambda e: (e.file_id, e.t_start)
        )
        preds = sorted(
            (e for e in predicted if e.label == label), key=lambda e: (e.file_id, e.t_start)
        )
        taken = [False] * len(preds)
        matches: list[EventMatch] = []
        missed: list[Event] = []
        for ref in refs:
            best: int | None = None
            best_distance = float("inf")
            for i, pred in enumerate(preds):
                if taken[i] or not _matches(
                    ref, pred, onset_collar_s, offset_collar_s, offset_collar_frac
                ):
                    continue
                distance = abs(pred.t_start - ref.t_start)
                if distance < best_distance:
                    best, best_distance = i, distance
            if best is None:
                missed.append(ref)
            else:
                taken[best] = True
                matches.append(EventMatch(reference=ref, predicted=preds[best]))
        spurious = [pred for i, pred in enumerate(preds) if not taken[i]]
        out[label] = (matches, spurious, missed)

    return out


def event_f1(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
) -> dict[str, Any]:
    """Event-based precision, recall and F1 with a tolerance collar (spec 14.1).

    Matching is **one-to-one within a label and within a file**, assigned
    greedily by onset distance. Greedy is the right choice here and not merely
    the easy one: the collars are small relative to the gaps between events of
    one label, so the assignment is almost always forced, and a greedy pass is
    deterministic where an optimal assignment would need a tie-break policy of
    its own.

    Args:
        reference: Ground-truth events.
        predicted: Detected events.
        onset_collar_s: Onset tolerance.
        offset_collar_s: Absolute part of the offset tolerance.
        offset_collar_frac: Relative part, as a fraction of the reference
            event's length.

    Returns:
        A dict with ``micro`` (pooled counts and rates), ``macro_f1`` (mean of
        the per-label F1 over labels present in either list) and ``per_label``.
    """
    assigned = match_events(
        reference,
        predicted,
        onset_collar_s=onset_collar_s,
        offset_collar_s=offset_collar_s,
        offset_collar_frac=offset_collar_frac,
    )
    per_label: dict[str, MatchCounts] = {
        label: MatchCounts(tp=len(matches), fp=len(spurious), fn=len(missed))
        for label, (matches, spurious, missed) in assigned.items()
    }

    micro = MatchCounts(
        tp=sum(c.tp for c in per_label.values()),
        fp=sum(c.fp for c in per_label.values()),
        fn=sum(c.fn for c in per_label.values()),
    )
    macro = float(np.mean([c.f1 for c in per_label.values()])) if per_label else 0.0
    return {
        "micro": micro.to_dict(),
        "f1": round(micro.f1, 4),
        "macro_f1": round(macro, 4),
        "per_label": {label: counts.to_dict() for label, counts in sorted(per_label.items())},
        "collars": {
            "onset_s": float(onset_collar_s),
            "offset_s": float(offset_collar_s),
            "offset_frac": float(offset_collar_frac),
        },
    }


# --------------------------------------------------------------------------- #
# Grid search
# --------------------------------------------------------------------------- #


@dataclass
class TuningCase:
    """One training recording, ready to be re-detected at any threshold.

    Holding the z-scores rather than the audio is what makes the grid search
    affordable: the expensive half of the pipeline (alignment, feature
    extraction, envelope) runs once per recording, and each grid cell only
    re-runs region formation.

    Attributes:
        file_id: Recording id; events never match across files.
        text_id: Text, for the per-text breakdown.
        signals: Oriented per-unit scores for this recording.
        z_by_unit: The comparisons those signals came from.
        truth: Ground-truth events from the dataset label JSON (spec 7.1).
        frames: Frame curves, for boundary refinement. ``None`` tunes without it.
    """

    file_id: str
    text_id: str
    signals: Sequence[Signal]
    z_by_unit: Mapping[str, ZResult]
    truth: Sequence[Event] = field(default_factory=tuple)
    frames: FeatureTable | None = None

    def predict(
        self,
        thresholds: RegionThresholds,
        *,
        config: Config | None = None,
        refinement: RefinementSettings | None = None,
    ) -> list[Region]:
        """Run region formation (and optional refinement) at these thresholds."""
        regions = build_regions(self.signals, self.z_by_unit, config=config, thresholds=thresholds)
        if refinement is not None and refinement.enabled:
            regions = refine_regions(regions, self.frames, settings=refinement)
        return regions


@dataclass
class TuningResult:
    """The chosen cell of the grid for one fold.

    Attributes:
        fold: LOTO fold id.
        train_texts: Texts the search was fitted on; the audit trail for the
            leakage guard.
        tau_in: Chosen trigger threshold.
        tau_out: Chosen stay threshold.
        penalty: Chosen PELT penalty.
        event_f1: Micro event-F1 at the chosen cell.
        metrics: The full metric dict at the chosen cell, including per-label
            precision and recall.
        grid: Every cell evaluated, with its F1; the tuning curve for the report.
        n_cases: Training recordings used.
        n_reference_events: Ground-truth events in those recordings.
    """

    fold: str
    train_texts: tuple[str, ...]
    tau_in: float
    tau_out: float
    penalty: float
    event_f1: float
    metrics: dict[str, Any] = field(default_factory=dict)
    grid: list[dict[str, float]] = field(default_factory=list)
    n_cases: int = 0
    n_reference_events: int = 0

    def thresholds(
        self, *, config: Config | None = None, base: RegionThresholds | None = None
    ) -> RegionThresholds:
        """The chosen thresholds as a :class:`RegionThresholds`.

        Args:
            config: Supplies every field the search did not tune.
            base: Use these as the untuned fields instead of the config. Pass
                the same base the search ran with, or a setting such as
                ``extend_on`` that the caller overrode is silently lost.
        """
        if base is None:
            base = (
                RegionThresholds.from_config(config) if config is not None else RegionThresholds()
            )
        return replace(base, tau_in=self.tau_in, tau_out=self.tau_out)

    def to_dict(self) -> dict[str, Any]:
        """JSON form written to ``results/tuning.json``."""
        return {
            "fold": self.fold,
            "train_texts": list(self.train_texts),
            "tau_in": float(self.tau_in),
            "tau_out": float(self.tau_out),
            "penalty": float(self.penalty),
            "event_f1": round(float(self.event_f1), 4),
            "n_cases": int(self.n_cases),
            "n_reference_events": int(self.n_reference_events),
            "metrics": self.metrics,
            "grid": self.grid,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TuningResult:
        """Rebuild from the JSON form."""
        return cls(
            fold=str(data["fold"]),
            train_texts=tuple(str(t) for t in data.get("train_texts", ())),
            tau_in=float(data["tau_in"]),
            tau_out=float(data["tau_out"]),
            penalty=float(data.get("penalty", 0.0)),
            event_f1=float(data.get("event_f1", 0.0)),
            metrics=dict(data.get("metrics", {})),
            grid=[dict(cell) for cell in data.get("grid", [])],
            n_cases=int(data.get("n_cases", 0)),
            n_reference_events=int(data.get("n_reference_events", 0)),
        )


def _search_conf(config: Config | None) -> dict[str, Any]:
    """Grid and collars from ``detection.threshold_search``, with code defaults."""
    defaults: dict[str, Any] = {
        "tau_in_grid": (2.0, 2.5, 3.0),
        "tau_out_grid": (1.0, 1.5, 2.0),
        "penalty_grid": (3.0,),
        "constraint_tau_out_lt_tau_in": True,
        "onset_collar_s": DEFAULT_ONSET_COLLAR_S,
        "offset_collar_s": DEFAULT_OFFSET_COLLAR_S,
        "offset_collar_frac": DEFAULT_OFFSET_COLLAR_FRAC,
    }
    if config is None:
        return defaults
    block = config.detection["threshold_search"]
    defaults.update(
        tau_in_grid=tuple(float(x) for x in block["tau_in_grid"]),
        tau_out_grid=tuple(float(x) for x in block["tau_out_grid"]),
        penalty_grid=tuple(
            float(x)
            for x in block.get("penalty_grid", (float(config.detection["refinement"]["penalty"]),))
        ),
        constraint_tau_out_lt_tau_in=bool(block.get("constraint_tau_out_lt_tau_in", True)),
        onset_collar_s=float(block.get("onset_collar_s", DEFAULT_ONSET_COLLAR_S)),
        offset_collar_s=float(block.get("offset_collar_s", DEFAULT_OFFSET_COLLAR_S)),
        offset_collar_frac=float(block.get("offset_collar_frac", DEFAULT_OFFSET_COLLAR_FRAC)),
    )
    return defaults


def _evaluate(
    regions_by_file: Sequence[tuple[str, list[Region]]],
    cases: Sequence[TuningCase],
    truth: Sequence[Event],
    conf: Mapping[str, Any],
    settings: RefinementSettings | None,
) -> dict[str, Any]:
    """Score one configuration: refine copies of the regions, then event-F1."""
    predicted: list[Event] = []
    for case, (file_id, regions) in zip(cases, regions_by_file, strict=True):
        if settings is not None and settings.enabled:
            regions = refine_regions(deepcopy(regions), case.frames, settings=settings)
        predicted += events_from_regions(regions, file_id=file_id)
    return event_f1(
        truth,
        predicted,
        onset_collar_s=conf["onset_collar_s"],
        offset_collar_s=conf["offset_collar_s"],
        offset_collar_frac=conf["offset_collar_frac"],
    )


def _settings_for(penalty: float, config: Config | None) -> RefinementSettings:
    """Refinement settings at one penalty."""
    return (
        RefinementSettings.from_config(config, penalty=penalty)
        if config is not None
        else RefinementSettings(penalty=penalty)
    )


def grid_search(
    cases: Sequence[TuningCase],
    *,
    fold: str,
    train_texts: Sequence[str],
    config: Config | None = None,
    tau_in_grid: Sequence[float] | None = None,
    tau_out_grid: Sequence[float] | None = None,
    penalty_grid: Sequence[float] | None = None,
    refine: bool = True,
    staged: bool = True,
    base_thresholds: RegionThresholds | None = None,
) -> TuningResult:
    """Maximise event-F1 over the threshold grid on training cases (spec 10.4(5)).

    **Staged by default.** Spec 10.4(5) tunes the hysteresis thresholds and
    spec 10.5 fixes a refinement penalty. Searching all three jointly costs
    ``|tau_in| x |tau_out| x |penalty|`` full detector passes, and every
    refinement pass runs ``ruptures`` twice per region. The staged search picks
    ``(tau_in, tau_out)`` with refinement off, then picks the penalty at those
    thresholds. That is sound rather than merely cheap: refinement moves a
    boundary by **less than 300 ms** and can neither create nor remove a
    region, so it cannot reorder the threshold grid by more than the collar,
    while the joint cost is two orders of magnitude higher. Pass
    ``staged=False`` for the full product when a report needs the joint
    surface.

    Args:
        cases: Training recordings only. A case whose ``text_id`` is not in
            ``train_texts`` is dropped and logged - the leakage guard
            (CLAUDE.md invariant 11), enforced here rather than trusted to the
            caller.
        fold: Fold id, recorded in the result.
        train_texts: The fold's training texts.
        config: Supplies the grid, the collars and the region defaults.
        tau_in_grid: Override the configured trigger grid.
        tau_out_grid: Override the configured stay grid.
        penalty_grid: Override the configured penalty grid. Ignored when
            ``refine`` is ``False``.
        refine: Whether refinement is part of the pipeline being tuned. Tuning
            without it is what ablation 3 needs.
        staged: Two-stage search as described above; ``False`` searches the
            full product.
        base_thresholds: The untuned fields (merge gap, duration floor,
            smoothing, ``extend_on``). ``None`` reads them from ``config``.
            Pass this whenever the caller has overridden one of them, so the
            search evaluates the pipeline that will actually run.

    Returns:
        A :class:`TuningResult`; ``grid`` holds every cell evaluated, so the
        report can show the tuning surface rather than just the argmax.

    Raises:
        ValueError: No training case survived the filter, or the grid is empty
            after the ``tau_out < tau_in`` constraint.
    """
    conf = _search_conf(config)
    wanted = set(train_texts)
    usable = [c for c in cases if c.text_id in wanted]
    dropped = [c.file_id for c in cases if c.text_id not in wanted]
    if dropped:
        logger.warning(
            "fold %s: %d case(s) excluded from tuning, text not in the training set: %s",
            fold,
            len(dropped),
            sorted(dropped),
        )
    if not usable:
        raise ValueError(
            f"fold {fold}: no tuning case belongs to the training texts {sorted(wanted)}"
        )

    taus_in = tuple(float(x) for x in (tau_in_grid or conf["tau_in_grid"]))
    taus_out = tuple(float(x) for x in (tau_out_grid or conf["tau_out_grid"]))
    penalties = (
        tuple(float(x) for x in (penalty_grid or conf["penalty_grid"])) if refine else (0.0,)
    )
    strict = bool(conf["constraint_tau_out_lt_tau_in"])

    tau_cells = [
        (ti, to)
        for ti, to in itertools.product(taus_in, taus_out)
        if (to < ti if strict else to <= ti)
    ]
    if not tau_cells:
        raise ValueError(
            f"empty grid: tau_in {taus_in} and tau_out {taus_out} leave no cell with "
            f"tau_out below tau_in"
        )

    base = base_thresholds
    if base is None:
        base = RegionThresholds.from_config(config) if config is not None else RegionThresholds()
    truth = [
        Event(e.label, e.t_start, e.t_end, file_id=e.file_id or case.file_id)
        for case in usable
        for e in case.truth
    ]

    # Region formation depends only on (tau_in, tau_out); the penalty moves the
    # boundaries afterwards. Building the regions once per threshold pair and
    # refining copies of them keeps the detector pass out of the inner loop.
    region_cache: dict[tuple[float, float], list[tuple[str, list[Region]]]] = {}

    def regions_at(tau_in: float, tau_out: float) -> list[tuple[str, list[Region]]]:
        key = (tau_in, tau_out)
        if key not in region_cache:
            thresholds = replace(base, tau_in=tau_in, tau_out=tau_out)
            region_cache[key] = [
                (
                    case.file_id,
                    build_regions(
                        case.signals, case.z_by_unit, config=config, thresholds=thresholds
                    ),
                )
                for case in usable
            ]
        return region_cache[key]

    grid_log: list[dict[str, float]] = []
    # Deterministic tie-break: best F1, then the stricter trigger (fewer false
    # positives on clean speech, spec 14.3), then the looser stay threshold,
    # then the smaller penalty.
    scored: list[
        tuple[tuple[float, float, float, float], dict[str, Any], tuple[float, float, float]]
    ] = []

    def record(tau_in: float, tau_out: float, penalty: float, metrics: Mapping[str, Any]) -> None:
        f1 = float(metrics["f1"])
        grid_log.append(
            {
                "tau_in": tau_in,
                "tau_out": tau_out,
                "penalty": penalty,
                "event_f1": round(f1, 4),
                "precision": float(metrics["micro"]["precision"]),
                "recall": float(metrics["micro"]["recall"]),
                "n_predicted": int(metrics["micro"]["tp"] + metrics["micro"]["fp"]),
            }
        )
        scored.append(((-f1, tau_in, -tau_out, penalty), dict(metrics), (tau_in, tau_out, penalty)))

    if staged:
        stage1: list[tuple[tuple[float, float, float], float, float]] = []
        for tau_in, tau_out in tau_cells:
            metrics = _evaluate(regions_at(tau_in, tau_out), usable, truth, conf, None)
            record(tau_in, tau_out, 0.0, metrics)
            stage1.append(((-float(metrics["f1"]), tau_in, -tau_out), tau_in, tau_out))
        stage1.sort(key=lambda item: item[0])
        _, best_in, best_out = stage1[0]
        logger.info(
            "fold %s stage 1: tau_in=%.2f tau_out=%.2f over %d cell(s)",
            fold,
            best_in,
            best_out,
            len(tau_cells),
        )

        if refine:
            # Stage 2 replaces stage 1's unrefined scores: the pipeline being
            # tuned includes refinement, so the reported F1 must as well.
            grid_log.clear()
            scored.clear()
            for penalty in penalties:
                metrics = _evaluate(
                    regions_at(best_in, best_out),
                    usable,
                    truth,
                    conf,
                    _settings_for(penalty, config),
                )
                record(best_in, best_out, penalty, metrics)
    else:
        for (tau_in, tau_out), penalty in itertools.product(tau_cells, penalties):
            settings = _settings_for(penalty, config) if refine else None
            metrics = _evaluate(regions_at(tau_in, tau_out), usable, truth, conf, settings)
            record(tau_in, tau_out, penalty, metrics)

    scored.sort(key=lambda item: item[0])
    _, best_metrics, (tau_in, tau_out, penalty) = scored[0]

    result = TuningResult(
        fold=fold,
        train_texts=tuple(sorted(wanted)),
        tau_in=tau_in,
        tau_out=tau_out,
        penalty=penalty,
        event_f1=float(best_metrics["f1"]),
        metrics=best_metrics,
        grid=sorted(grid_log, key=lambda c: (c["tau_in"], c["tau_out"], c["penalty"])),
        n_cases=len(usable),
        n_reference_events=len(truth),
    )
    logger.info(
        "fold %s: tau_in=%.2f tau_out=%.2f penalty=%g -> event-F1 %.3f "
        "(P %.3f / R %.3f) on %d training file(s), %d reference event(s)",
        fold,
        tau_in,
        tau_out,
        penalty,
        result.event_f1,
        float(best_metrics["micro"]["precision"]),
        float(best_metrics["micro"]["recall"]),
        result.n_cases,
        result.n_reference_events,
    )
    return result


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def write_tuning(
    results: Sequence[TuningResult],
    path: Path | str,
    *,
    config: Config | None = None,
    source: str = "dataset",
) -> Path:
    """Write the chosen thresholds per fold to ``results/tuning.json``.

    The file is written with sorted keys, a trailing newline and no timestamp,
    so two identical runs produce identical bytes (CLAUDE.md invariant 13).

    Args:
        results: One result per fold.
        path: Destination file.
        config: When given, its ``hash`` is recorded alongside, so a tuning
            file can be traced to the parameters that produced it.
        source: Which corpus the thresholds were fitted on. Anything other than
            ``"dataset"`` is recorded in the file *and* logged as a warning:
            thresholds fitted on generated feature tables must never silently
            become the ones a reported metric depends on.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source != "dataset":
        logger.warning(
            "writing %s with source=%r; these thresholds are NOT fitted on the dataset "
            "and must be refitted before any reported metric uses them",
            destination,
            source,
        )
    payload: dict[str, Any] = {
        "objective": "event_f1",
        "fit_scope": "training_folds_only",
        "source": source,
        "folds": {r.fold: r.to_dict() for r in sorted(results, key=lambda r: r.fold)},
    }
    if config is not None:
        payload["config_hash"] = config.hash
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("wrote %d fold(s) of tuning to %s", len(results), destination)
    return destination


def read_tuning(path: Path | str) -> dict[str, TuningResult]:
    """Read ``results/tuning.json`` back as ``{fold: TuningResult}``.

    Returns an empty dict when the file does not exist, so a caller can fall
    back to the configured defaults rather than crashing before tuning has run.
    """
    source = Path(path)
    if not source.is_file():
        logger.info("%s does not exist yet; using the configured default thresholds", source)
        return {}
    data = json.loads(source.read_text(encoding="utf-8"))
    origin = str(data.get("source", "dataset"))
    if origin != "dataset":
        logger.warning(
            "%s holds thresholds fitted on %r, not on the dataset; refit before "
            "any reported metric depends on them",
            source,
            origin,
        )
    return {
        fold: TuningResult.from_dict(block) for fold, block in dict(data.get("folds", {})).items()
    }
