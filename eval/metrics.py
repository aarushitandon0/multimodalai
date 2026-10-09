"""Evaluation metrics for the LOTO protocol (spec 14.1, 14.3).

Every function here is pure: events and numbers in, numbers out. The harness
that reads the dataset, fits a fold and writes JSON is :mod:`eval.run_loto`;
keeping the two apart is what makes the metrics unit-testable against hand-built
event lists whose answer is known by inspection.

**Event-based F1** is the DCASE convention of spec 14.1: a prediction matches a
reference of the same label when their onsets fall within a 200 ms collar *and*
their offsets within ``max(200 ms, 20% of the reference length)``. Two
implementations score it and the report carries both:

* :func:`rostrum.detect.tune.event_f1`, the project's own greedy matcher, which
  the threshold search also optimises, so the tuned cell and the reported score
  are the same quantity;
* ``sed_eval.sound_event.EventBasedMetrics``, the published reference
  implementation, via :func:`sed_eval_metrics`.

:func:`crosscheck_event_f1` runs both and reports the difference. They are not
expected to agree to the last digit - ``sed_eval`` resolves one-to-one
assignment its own way - and a visible delta is worth more than a single number
whose provenance is a matter of trust.

**Boundary error** is reported twice for one reason worth stating in the report:
measured over *collar-matched* pairs it cannot exceed the collar, so a 200 ms
collar makes the "MAE ≤ 150 ms" target of spec 14.3 nearly free. The headline
number here is therefore measured over **IoU-matched** pairs
(:func:`boundary_errors` with ``min_iou``), which admits the badly-placed
regions the collar throws away, and the collar-matched value is reported beside
it as ``*_collar_matched``.

**Threshold-free PR** (spec 14.1, PSDS [R10]) sweeps the region score rather
than the operating point, in :func:`pr_curve`. True PSDS wants
``sed_scores_eval``; that package has no wheel and needs a C toolchain, so where
it is importable :func:`psds_crosscheck` adds its collar-based PSD-ROC AUC and
where it is not the report says so instead of quietly substituting this curve
for it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from rostrum.detect.classifier import iou
from rostrum.detect.tune import (
    DEFAULT_OFFSET_COLLAR_FRAC,
    DEFAULT_OFFSET_COLLAR_S,
    DEFAULT_ONSET_COLLAR_S,
    Event,
    event_f1,
    match_events,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TARGET_DIRECTIONS",
    "TARGET_LABELS",
    "BoundaryErrors",
    "ScoredEvent",
    "TargetRow",
    "boundary_errors",
    "check_targets",
    "crosscheck_event_f1",
    "event_based_metrics",
    "event_diagnostics",
    "false_positives_per_minute",
    "mean_sd",
    "pr_curve",
    "psds_crosscheck",
    "sed_eval_available",
    "sed_eval_metrics",
    "sed_scores_eval_available",
    "severity_spearman",
    "sex_score_gap",
    "targets_markdown",
    "typing_metrics",
]


DEFAULT_MIN_IOU: float = 0.3
"""IoU an unmatched-by-collar pair needs before its boundary error is measured."""


# --------------------------------------------------------------------------- #
# Optional third-party scorers
# --------------------------------------------------------------------------- #


def sed_eval_available() -> bool:
    """Whether ``sed_eval`` can be imported in this environment."""
    try:
        import sed_eval  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.debug("sed_eval unavailable: %s", exc)
        return False
    return True


def sed_scores_eval_available() -> bool:
    """Whether ``sed_scores_eval`` can be imported in this environment.

    It ships no wheel for Windows and its build needs MSVC, so this is routinely
    ``False`` on a developer machine and ``True`` in the Docker image.
    """
    try:
        import sed_scores_eval  # noqa: F401
    except Exception as exc:
        logger.debug("sed_scores_eval unavailable: %s", exc)
        return False
    return True


# --------------------------------------------------------------------------- #
# Event-based F1
# --------------------------------------------------------------------------- #


def event_based_metrics(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
) -> dict[str, Any]:
    """Event-F1 by the project's own matcher (thin pass-through to tuning).

    The indirection is deliberate: the eval harness imports every metric from
    this module, and this is the one that must stay identical to the threshold
    search's objective.
    """
    return event_f1(
        reference,
        predicted,
        onset_collar_s=onset_collar_s,
        offset_collar_s=offset_collar_s,
        offset_collar_frac=offset_collar_frac,
    )


def _as_sed_eval_rows(events: Sequence[Event]) -> dict[str, list[dict[str, Any]]]:
    """Group events into ``sed_eval``'s per-file dicts, keyed by file id."""
    out: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        out.setdefault(str(event.file_id), []).append(
            {
                "event_label": str(event.label),
                "onset": float(event.t_start),
                "offset": float(event.t_end),
                "filename": str(event.file_id),
            }
        )
    return out


def sed_eval_metrics(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
    labels: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Event-F1 from ``sed_eval.sound_event.EventBasedMetrics`` (spec 14.1).

    ``sed_eval`` takes one ``t_collar`` for both onset and offset and adds
    ``percentage_of_length`` to the offset side, which is exactly the spec's
    ``max(200 ms, 20%)`` rule, and it accumulates over files, so the events are
    grouped by ``file_id`` and fed one file per call.

    Args:
        reference: Ground-truth events.
        predicted: Detected events.
        onset_collar_s: ``t_collar``.
        offset_collar_frac: ``percentage_of_length``.
        labels: Label vocabulary; defaults to the labels present.

    Returns:
        A dict shaped like :func:`event_based_metrics`'s, or ``None`` when
        ``sed_eval`` cannot be imported.
    """
    try:
        import sed_eval
    except Exception as exc:
        logger.info("sed_eval cross-check skipped: %s", exc)
        return None

    vocabulary = sorted(
        set(labels)
        if labels is not None
        else ({e.label for e in reference} | {e.label for e in predicted})
    )
    if not vocabulary:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "per_label": {}}

    scorer = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=list(vocabulary),
        t_collar=float(onset_collar_s),
        percentage_of_length=float(offset_collar_frac),
        evaluate_onset=True,
        evaluate_offset=True,
    )
    ref_by_file = _as_sed_eval_rows(reference)
    est_by_file = _as_sed_eval_rows(predicted)
    for file_id in sorted(set(ref_by_file) | set(est_by_file)):
        scorer.evaluate(
            reference_event_list=ref_by_file.get(file_id, []),
            estimated_event_list=est_by_file.get(file_id, []),
        )

    results = scorer.results()
    overall = results["overall"]["f_measure"]
    class_wise = results.get("class_wise", {})
    return {
        "f1": round(float(overall["f_measure"]), 4),
        "precision": round(float(overall["precision"]), 4),
        "recall": round(float(overall["recall"]), 4),
        "macro_f1": round(
            float(results["class_wise_average"]["f_measure"]["f_measure"]),
            4,
        ),
        "per_label": {
            label: {
                "f1": round(float(row["f_measure"]["f_measure"]), 4),
                "precision": round(float(row["f_measure"]["precision"]), 4),
                "recall": round(float(row["f_measure"]["recall"]), 4),
                "support": int(row["count"]["Nref"]),
            }
            for label, row in sorted(class_wise.items())
        },
        "engine": f"sed_eval {getattr(sed_eval, '__version__', '?')}",
    }


def crosscheck_event_f1(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Score one event list with both implementations and report the gap.

    Args:
        reference: Ground-truth events.
        predicted: Detected events.
        onset_collar_s: Onset tolerance for both engines.
        offset_collar_s: Absolute offset tolerance (native engine only;
            ``sed_eval`` derives it from ``onset_collar_s``).
        offset_collar_frac: Relative offset tolerance for both engines.
        tolerance: Log a warning when the two F1 values differ by more than
            this. Not an assertion: a genuine difference belongs in the report,
            not in a traceback.

    Returns:
        ``{"rostrum": {...}, "sed_eval": {...} | None, "delta_f1": float | None,
        "agrees": bool | None}``.
    """
    native = event_based_metrics(
        reference,
        predicted,
        onset_collar_s=onset_collar_s,
        offset_collar_s=offset_collar_s,
        offset_collar_frac=offset_collar_frac,
    )
    reference_impl = sed_eval_metrics(
        reference,
        predicted,
        onset_collar_s=onset_collar_s,
        offset_collar_frac=offset_collar_frac,
    )
    if reference_impl is None:
        return {
            "rostrum": native,
            "sed_eval": None,
            "delta_f1": None,
            "agrees": None,
            "note": "sed_eval not importable in this environment",
        }

    delta = abs(float(native["f1"]) - float(reference_impl["f1"]))
    agrees = delta <= float(tolerance)
    if not agrees:
        logger.warning(
            "event-F1 engines disagree: rostrum %.4f vs sed_eval %.4f (delta %.4f > %.4f)",
            native["f1"],
            reference_impl["f1"],
            delta,
            tolerance,
        )
    return {
        "rostrum": native,
        "sed_eval": reference_impl,
        "delta_f1": round(delta, 4),
        "agrees": bool(agrees),
    }


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BoundaryErrors:
    """Onset/offset error and overlap over matched events (spec 14.1).

    Attributes:
        onset_mae_ms: Mean absolute onset error in milliseconds.
        offset_mae_ms: Mean absolute offset error in milliseconds.
        mae_ms: Mean of the two, the number spec 14.3 budgets.
        onset_bias_ms: Signed mean onset error; positive means late.
        offset_bias_ms: Signed mean offset error; positive means late.
        iou_mean: Mean temporal IoU of the matched pairs.
        n: Pairs measured.
        matching: ``"iou"`` or ``"collar"``, so a number cannot be quoted
            without saying which pairs it came from.
    """

    onset_mae_ms: float = 0.0
    offset_mae_ms: float = 0.0
    mae_ms: float = 0.0
    onset_bias_ms: float = 0.0
    offset_bias_ms: float = 0.0
    iou_mean: float = 0.0
    n: int = 0
    matching: str = "iou"

    def to_dict(self) -> dict[str, Any]:
        """Rounded JSON form."""
        return {
            "onset_mae_ms": round(self.onset_mae_ms, 2),
            "offset_mae_ms": round(self.offset_mae_ms, 2),
            "mae_ms": round(self.mae_ms, 2),
            "onset_bias_ms": round(self.onset_bias_ms, 2),
            "offset_bias_ms": round(self.offset_bias_ms, 2),
            "iou_mean": round(self.iou_mean, 4),
            "n": int(self.n),
            "matching": self.matching,
        }


def _errors_from_pairs(pairs: Sequence[tuple[Event, Event]], matching: str) -> BoundaryErrors:
    """Aggregate signed and absolute boundary errors over reference/prediction pairs."""
    if not pairs:
        return BoundaryErrors(n=0, matching=matching)
    onset = np.array([float(p.t_start) - float(r.t_start) for r, p in pairs], dtype=float)
    offset = np.array([float(p.t_end) - float(r.t_end) for r, p in pairs], dtype=float)
    overlaps = np.array(
        [iou(r.t_start, r.t_end, p.t_start, p.t_end) for r, p in pairs], dtype=float
    )
    onset_mae = float(np.mean(np.abs(onset))) * 1000.0
    offset_mae = float(np.mean(np.abs(offset))) * 1000.0
    return BoundaryErrors(
        onset_mae_ms=onset_mae,
        offset_mae_ms=offset_mae,
        mae_ms=(onset_mae + offset_mae) / 2.0,
        onset_bias_ms=float(np.mean(onset)) * 1000.0,
        offset_bias_ms=float(np.mean(offset)) * 1000.0,
        iou_mean=float(np.mean(overlaps)),
        n=len(pairs),
        matching=matching,
    )


def _iou_pairs(
    reference: Sequence[Event], predicted: Sequence[Event], min_iou: float
) -> list[tuple[Event, Event]]:
    """Pair each reference with its best-overlapping prediction of the same label.

    One-to-one and deterministic: candidate pairs are sorted by descending IoU
    with ties broken by onset, then taken greedily. A reference with no
    prediction reaching ``min_iou`` contributes no pair - it is a miss, which
    :func:`event_based_metrics` already counts, and inventing a boundary error
    for it would mix detection failure into a localisation number.
    """
    candidates: list[tuple[float, float, int, int]] = []
    for i, ref in enumerate(reference):
        for j, pred in enumerate(predicted):
            if ref.label != pred.label or ref.file_id != pred.file_id:
                continue
            overlap = iou(ref.t_start, ref.t_end, pred.t_start, pred.t_end)
            if overlap >= min_iou:
                candidates.append((-overlap, float(ref.t_start), i, j))
    candidates.sort()

    used_ref: set[int] = set()
    used_pred: set[int] = set()
    pairs: list[tuple[Event, Event]] = []
    for _, _, i, j in candidates:
        if i in used_ref or j in used_pred:
            continue
        used_ref.add(i)
        used_pred.add(j)
        pairs.append((reference[i], predicted[j]))
    return pairs


def boundary_errors(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
) -> dict[str, Any]:
    """Onset/offset MAE and temporal IoU, over both notions of "matched".

    Args:
        reference: Ground-truth events.
        predicted: Detected events.
        min_iou: Overlap a pair needs for the headline (IoU-matched) figure.
        onset_collar_s: Onset collar for the collar-matched figure.
        offset_collar_s: Absolute offset collar for it.
        offset_collar_frac: Relative offset collar for it.

    Returns:
        ``{"iou_matched": {...}, "collar_matched": {...}, "min_iou": float}``;
        both inner dicts are :meth:`BoundaryErrors.to_dict` forms. Spec 14.3's
        ``onset_offset_mae_ms`` is ``iou_matched.mae_ms``.
    """
    assigned = match_events(
        reference,
        predicted,
        onset_collar_s=onset_collar_s,
        offset_collar_s=offset_collar_s,
        offset_collar_frac=offset_collar_frac,
    )
    collar_pairs = [
        (match.reference, match.predicted)
        for matches, _, _ in assigned.values()
        for match in matches
    ]
    return {
        "iou_matched": _errors_from_pairs(
            _iou_pairs(list(reference), list(predicted), min_iou), "iou"
        ).to_dict(),
        "collar_matched": _errors_from_pairs(collar_pairs, "collar").to_dict(),
        "min_iou": float(min_iou),
    }


def event_diagnostics(
    reference: Sequence[Event],
    predicted: Sequence[Event],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
    scores: Mapping[tuple[str, float, float], float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The individual matches and misses behind the aggregate numbers.

    :func:`boundary_errors` returns means, which is what the targets table
    needs and what a histogram and a failure analysis cannot be built from
    (prompt 20). This returns the per-event records instead: one row per
    IoU-matched pair with its signed errors, one per reference the detector
    missed, and one per prediction that matched nothing.

    Matching is :func:`~rostrum.detect.tune.match_events`'s, so the misses
    counted here are exactly the FN and FP of the reported event-F1.

    Args:
        reference: Ground-truth events.
        predicted: Detected events.
        min_iou: Overlap a pair needs to appear in ``pairs``.
        onset_collar_s: Onset collar for deciding FN and FP.
        offset_collar_s: Absolute part of the offset collar.
        offset_collar_frac: Relative part of the offset collar.
        scores: ``{(file_id, t_start, t_end): detection score}``, attached to
            each false positive so the worst one can be found. Spans are
            matched on 3-decimal rounding, as every output time is.

    Returns:
        ``{"pairs": [...], "false_negatives": [...], "false_positives": [...]}``.
        Times are seconds to 3 decimals, errors milliseconds to 1 decimal.
    """
    lookup = {
        (str(file_id), round(float(start), 3), round(float(end), 3)): float(value)
        for (file_id, start, end), value in (scores or {}).items()
    }

    def score_of(event: Event) -> float | None:
        key = (str(event.file_id), round(float(event.t_start), 3), round(float(event.t_end), 3))
        return lookup.get(key)

    pairs = [
        {
            "file_id": ref.file_id,
            "label": ref.label,
            "ref_start": round(float(ref.t_start), 3),
            "ref_end": round(float(ref.t_end), 3),
            "pred_start": round(float(pred.t_start), 3),
            "pred_end": round(float(pred.t_end), 3),
            "onset_error_ms": round((float(pred.t_start) - float(ref.t_start)) * 1000.0, 1),
            "offset_error_ms": round((float(pred.t_end) - float(ref.t_end)) * 1000.0, 1),
            "iou": round(iou(ref.t_start, ref.t_end, pred.t_start, pred.t_end), 4),
            "score": score_of(pred),
        }
        for ref, pred in _iou_pairs(list(reference), list(predicted), min_iou)
    ]

    assigned = match_events(
        reference,
        predicted,
        onset_collar_s=onset_collar_s,
        offset_collar_s=offset_collar_s,
        offset_collar_frac=offset_collar_frac,
    )
    misses: list[dict[str, Any]] = []
    spurious: list[dict[str, Any]] = []
    for _, unmatched_predictions, unmatched_references in assigned.values():
        for event in unmatched_references:
            misses.append(
                {
                    "file_id": event.file_id,
                    "label": event.label,
                    "t_start": round(float(event.t_start), 3),
                    "t_end": round(float(event.t_end), 3),
                }
            )
        for event in unmatched_predictions:
            spurious.append(
                {
                    "file_id": event.file_id,
                    "label": event.label,
                    "t_start": round(float(event.t_start), 3),
                    "t_end": round(float(event.t_end), 3),
                    "score": score_of(event),
                }
            )

    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row["file_id"], row["t_start"], row["label"])

    return {
        "pairs": sorted(pairs, key=lambda row: (row["file_id"], row["ref_start"], row["label"])),
        "false_negatives": sorted(misses, key=key),
        "false_positives": sorted(spurious, key=key),
    }


# --------------------------------------------------------------------------- #
# Typing
# --------------------------------------------------------------------------- #


def typing_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Macro-F1, per-label rates and the confusion matrix for flaw typing.

    Macro-F1 averages over the labels that **occur in the truth**: averaging in
    a zero for a label never asked about would move the headline number for
    reasons that have nothing to do with the model. That is also the convention
    :func:`rostrum.detect.classifier.evaluate` uses, so the two agree.

    Args:
        y_true: True labels.
        y_pred: Predicted labels, same length.
        labels: Row/column order of the matrix; defaults to every label seen.

    Returns:
        ``{"macro_f1", "accuracy", "labels", "confusion", "per_label", "n"}``.
        ``confusion[i][j]`` counts truth ``labels[i]`` predicted as
        ``labels[j]``.

    Raises:
        ValueError: The two sequences differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} item(s), y_pred has {len(y_pred)}")
    vocabulary = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    index = {label: i for i, label in enumerate(vocabulary)}
    confusion = [[0 for _ in vocabulary] for _ in vocabulary]
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if truth in index and prediction in index:
            confusion[index[truth]][index[prediction]] += 1

    per_label: dict[str, dict[str, float]] = {}
    for label in vocabulary:
        i = index[label]
        tp = confusion[i][i]
        fp = sum(confusion[r][i] for r in range(len(vocabulary))) - tp
        fn = sum(confusion[i][c] for c in range(len(vocabulary))) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(tp + fn),
        }

    supported = [label for label in vocabulary if per_label[label]["support"] > 0]
    macro = float(np.mean([per_label[label]["f1"] for label in supported])) if supported else 0.0
    correct = sum(confusion[i][i] for i in range(len(vocabulary)))
    total = sum(sum(row) for row in confusion)
    return {
        "macro_f1": round(macro, 4),
        "accuracy": round(correct / total, 4) if total else 0.0,
        "labels": vocabulary,
        "confusion": confusion,
        "per_label": per_label,
        "n": total,
    }


# --------------------------------------------------------------------------- #
# Threshold-free PR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoredEvent:
    """A predicted event with the detection score the sweep thresholds on.

    Attributes:
        event: The event itself.
        score: Its detection score; :attr:`rostrum.types.Region.peak_z` is what
            the harness passes, being the quantity region formation triggers on.
    """

    event: Event
    score: float


def pr_curve(
    reference: Sequence[Event],
    predicted: Sequence[ScoredEvent],
    *,
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_s: float = DEFAULT_OFFSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
    max_points: int = 64,
) -> dict[str, Any]:
    """Collar-based precision/recall as a function of the region score.

    The operating point of spec 10.4 is one cell of this curve; sweeping the
    score shows whether a disappointing F1 is a detector that cannot separate
    flaws from clean speech or a threshold set in the wrong place. The curve is
    the threshold-independent view spec 14.1 asks for; it is **not** PSDS, which
    needs ``sed_scores_eval`` - see :func:`psds_crosscheck`.

    Args:
        reference: Ground-truth events.
        predicted: Scored predictions.
        onset_collar_s: Onset collar.
        offset_collar_s: Absolute offset collar.
        offset_collar_frac: Relative offset collar.
        max_points: Thresholds to evaluate. The distinct scores are subsampled
            at even quantiles when there are more, so the curve costs the same
            on 40 events and on 40 000.

    Returns:
        ``{"points": [{"threshold", "precision", "recall", "f1"}, ...],
        "best_f1", "best_threshold", "auc", "n_thresholds"}``. ``auc`` is the
        step integral of precision over recall, 0 when the curve has one point.
    """
    scores = sorted({round(float(item.score), 6) for item in predicted})
    if not scores:
        return {
            "points": [],
            "best_f1": 0.0,
            "best_threshold": None,
            "auc": 0.0,
            "n_thresholds": 0,
        }
    if len(scores) > max_points:
        picks = np.linspace(0, len(scores) - 1, num=max_points)
        scores = sorted({scores[round(float(i))] for i in picks})

    points: list[dict[str, float]] = []
    for threshold in scores:
        kept = [item.event for item in predicted if float(item.score) >= threshold]
        metrics = event_f1(
            reference,
            kept,
            onset_collar_s=onset_collar_s,
            offset_collar_s=offset_collar_s,
            offset_collar_frac=offset_collar_frac,
        )
        micro = metrics["micro"]
        points.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": float(micro["precision"]),
                "recall": float(micro["recall"]),
                "f1": float(micro["f1"]),
            }
        )

    best = max(points, key=lambda row: (row["f1"], -row["threshold"]))
    ordered = sorted(points, key=lambda row: (row["recall"], row["precision"]))
    auc = 0.0
    previous_recall = 0.0
    for row in ordered:
        auc += (row["recall"] - previous_recall) * row["precision"]
        previous_recall = row["recall"]
    return {
        "points": points,
        "best_f1": round(float(best["f1"]), 4),
        "best_threshold": round(float(best["threshold"]), 4),
        "auc": round(float(auc), 4),
        "n_thresholds": len(points),
    }


def psds_crosscheck(
    reference: Sequence[Event],
    predicted: Sequence[ScoredEvent],
    *,
    durations: Mapping[str, float],
    onset_collar_s: float = DEFAULT_ONSET_COLLAR_S,
    offset_collar_frac: float = DEFAULT_OFFSET_COLLAR_FRAC,
) -> dict[str, Any]:
    """Collar-based PSD-ROC AUC from ``sed_scores_eval``, when it is installed.

    Args:
        reference: Ground-truth events.
        predicted: Scored predictions.
        durations: ``{file_id: duration_s}`` for every evaluated file;
            threshold-free metrics need the clean seconds to rate false alarms.
        onset_collar_s: Onset collar.
        offset_collar_frac: Relative offset collar.

    Returns:
        ``{"available": False, "reason": str}`` when the package is missing,
        otherwise ``{"available": True, "psds": float, ...}``.
    """
    if not sed_scores_eval_available():
        return {
            "available": False,
            "reason": "sed_scores_eval is not installed (no wheel; its build needs a C toolchain)",
        }
    import sed_scores_eval  # pragma: no cover - exercised only where installed

    ground_truth = {
        file_id: [
            (float(e.t_start), float(e.t_end), str(e.label))
            for e in reference
            if e.file_id == file_id
        ]
        for file_id in sorted(durations)
    }
    scores = {
        file_id: [
            (
                float(item.event.t_start),
                float(item.event.t_end),
                str(item.event.label),
                float(item.score),
            )
            for item in predicted
            if item.event.file_id == file_id
        ]
        for file_id in sorted(durations)
    }
    result = sed_scores_eval.collar_based.auroc(  # pragma: no cover
        scores,
        ground_truth,
        onset_collar=float(onset_collar_s),
        offset_collar=float(onset_collar_s),
        offset_collar_rate=float(offset_collar_frac),
        max_fpr=0.1,
    )
    return {  # pragma: no cover
        "available": True,
        "auroc": round(float(result[0]), 4),
        "engine": f"sed_scores_eval {getattr(sed_scores_eval, '__version__', '?')}",
    }


# --------------------------------------------------------------------------- #
# Gradient, false alarms, fairness
# --------------------------------------------------------------------------- #


def severity_spearman(levels: Sequence[float], scores: Sequence[float]) -> dict[str, Any]:
    """Spearman rho of injected severity against the category score (spec 14.1).

    The target is **negative** (spec 14.3: rho <= -0.85): a worse flaw must score
    lower. ``None`` comes back when one side has no variance, which is the
    honest answer - a constant has no rank correlation - rather than 0.

    Args:
        levels: Injected severity per file, e.g. ``s`` from the ladder.
        scores: The category score of the same files, in the same order.

    Returns:
        ``{"rho": float | None, "p_value": float | None, "n": int}``.

    Raises:
        ValueError: The two sequences differ in length.
    """
    if len(levels) != len(scores):
        raise ValueError(f"levels has {len(levels)} item(s), scores has {len(scores)}")
    if len(levels) < 3:
        return {"rho": None, "p_value": None, "n": len(levels), "reason": "fewer than 3 points"}
    x = np.asarray(levels, dtype=float)
    y = np.asarray(scores, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"rho": None, "p_value": None, "n": int(x.size), "reason": "no variance"}

    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    return {
        "rho": round(rho, 4) if math.isfinite(rho) else None,
        "p_value": round(p_value, 6) if math.isfinite(p_value) else None,
        "n": int(x.size),
    }


def false_positives_per_minute(
    predicted: Sequence[Event], durations: Mapping[str, float]
) -> dict[str, Any]:
    """False-alarm rate on files with no flaw in them (spec 14.3).

    Every prediction on a clean file is a false positive by construction, so no
    matching is needed - only the minutes to divide by.

    Args:
        predicted: Predictions on clean files only.
        durations: ``{file_id: duration_s}`` of those files. Files with no
            prediction must still appear, or the rate is computed over the
            wrong denominator.

    Returns:
        ``{"fp": int, "minutes": float, "fp_per_min": float | None,
        "per_file": {...}}``.
    """
    minutes = float(sum(float(v) for v in durations.values())) / 60.0
    counted = [e for e in predicted if str(e.file_id) in durations]
    stray = sorted({str(e.file_id) for e in predicted} - set(durations))
    if stray:
        logger.warning(
            "%d prediction(s) on file(s) with no duration given: %s", len(stray), stray[:5]
        )
    per_file: dict[str, int] = dict.fromkeys(sorted(durations), 0)
    for event in counted:
        per_file[str(event.file_id)] += 1
    return {
        "fp": len(counted),
        "minutes": round(minutes, 4),
        "fp_per_min": round(len(counted) / minutes, 4) if minutes > 0 else None,
        "n_files": len(durations),
        "per_file": per_file,
    }


def sex_score_gap(
    scores_by_file: Mapping[str, float], sex_by_file: Mapping[str, str]
) -> dict[str, Any]:
    """Male-female gap in overall score for identically injected flaws (spec 14.3).

    The caller is responsible for passing files that carry the *same* flaw at
    the *same* level - the gap is only interpretable as bias when the thing
    being scored is held fixed.

    Args:
        scores_by_file: ``{file_id: overall score}``.
        sex_by_file: ``{file_id: "m" | "f"}``; files missing here are skipped
            and counted in ``n_unknown``.

    Returns:
        ``{"gap_points": float | None, "male": {...}, "female": {...},
        "n_unknown": int}``. ``gap_points`` is ``None`` when either side is
        empty, which is what a dataset with no sex register reports.
    """
    buckets: dict[str, list[float]] = {"m": [], "f": []}
    unknown = 0
    for file_id, score in sorted(scores_by_file.items()):
        sex = str(sex_by_file.get(file_id, "")).strip().lower()[:1]
        if sex in buckets:
            buckets[sex].append(float(score))
        else:
            unknown += 1

    def side(values: Sequence[float]) -> dict[str, Any]:
        return {
            "n": len(values),
            "mean": round(float(np.mean(values)), 4) if values else None,
            "sd": round(float(np.std(values, ddof=0)), 4) if values else None,
        }

    male, female = side(buckets["m"]), side(buckets["f"])
    gap = (
        round(abs(float(male["mean"]) - float(female["mean"])), 4)
        if male["mean"] is not None and female["mean"] is not None
        else None
    )
    if gap is None:
        logger.info(
            "male-female gap not computable: %d male, %d female, %d without a sex",
            male["n"],
            female["n"],
            unknown,
        )
    return {"gap_points": gap, "male": male, "female": female, "n_unknown": unknown}


# --------------------------------------------------------------------------- #
# Aggregation and the pre-registered targets
# --------------------------------------------------------------------------- #


def mean_sd(values: Sequence[float | None]) -> dict[str, Any]:
    """Mean ± sd across folds, ignoring folds that could not report (spec 7.3).

    The population sd (``ddof=0``) is used: the folds are the complete set of
    LOTO folds this corpus admits, not a sample from a larger pool.

    Returns:
        ``{"mean": float | None, "sd": float | None, "n": int, "values": [...]}``.
    """
    usable = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return {
        "mean": round(float(np.mean(usable)), 4) if usable else None,
        "sd": round(float(np.std(usable, ddof=0)), 4) if usable else None,
        "n": len(usable),
        "values": [round(float(v), 4) for v in usable],
    }


TARGET_DIRECTIONS: dict[str, str] = {
    "event_f1_200ms": "higher",
    "onset_offset_mae_ms": "lower",
    "flaw_type_macro_f1": "higher",
    "severity_spearman_rho": "lower",
    "false_positives_per_min_on_clean": "lower",
    "male_female_score_gap_points": "lower",
    "utmos_drop_at_l1": "lower",
}
"""Which way is better for each pre-registered target of spec 14.3.

``severity_spearman_rho`` is ``lower`` because the target is rho <= -0.85: the
correlation must be strongly *negative*, so a less negative rho is a miss.
"""

TARGET_LABELS: dict[str, str] = {
    "event_f1_200ms": "Event-F1 @ 200 ms",
    "onset_offset_mae_ms": "Onset/offset MAE (ms)",
    "flaw_type_macro_f1": "Macro-F1 (type)",
    "severity_spearman_rho": "Spearman rho",
    "false_positives_per_min_on_clean": "FP on clean (per min)",
    "male_female_score_gap_points": "M-F gap (points)",
    "utmos_drop_at_l1": "UTMOS drop at L1",
}
"""Row names for ``results/metrics.md``, in the wording of spec 14.3."""


@dataclass
class TargetRow:
    """One pre-registered target beside what the run actually achieved.

    Attributes:
        key: Target key, as in ``configs/scoring.yaml``.
        label: Human-readable row name.
        target: The pre-registered value, or ``None`` if the split has none.
        actual: What this run achieved, or ``None`` if it could not be measured.
        direction: ``"higher"`` or ``"lower"``.
        verdict: ``PASS``, ``MISS`` or ``N/A``.
    """

    key: str
    label: str
    target: float | None
    actual: float | None
    direction: str = "higher"
    verdict: str = "N/A"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "key": self.key,
            "label": self.label,
            "target": self.target,
            "actual": self.actual,
            "direction": self.direction,
            "verdict": self.verdict,
            "note": self.note,
        }


def check_targets(
    actual: Mapping[str, float | None],
    targets: Mapping[str, float] | None,
    *,
    keys: Sequence[str] | None = None,
) -> list[TargetRow]:
    """Compare achieved numbers with the pre-registered targets (spec 14.3).

    A target with no measurement, or a measurement with no target, is reported
    ``N/A`` with the reason attached. Nothing is silently dropped: a row that
    disappears from the table is a row nobody has to explain.

    Args:
        actual: ``{target key: value}``; ``None`` for "not measurable".
        targets: ``{target key: pre-registered value}``, e.g.
            ``config.scoring["targets"]["synthetic"]``.
        keys: Rows to emit and their order; defaults to
            :data:`TARGET_DIRECTIONS`'s order.

    Returns:
        One :class:`TargetRow` per key.
    """
    targets = dict(targets or {})
    rows: list[TargetRow] = []
    for key in keys if keys is not None else TARGET_DIRECTIONS:
        direction = TARGET_DIRECTIONS.get(key, "higher")
        target = targets.get(key)
        value = actual.get(key)
        row = TargetRow(
            key=key,
            label=TARGET_LABELS.get(key, key),
            target=None if target is None else round(float(target), 4),
            actual=None if value is None else round(float(value), 4),
            direction=direction,
        )
        if target is None:
            row.note = "no pre-registered target for this split"
        elif value is None:
            row.note = "not measurable in this run"
        else:
            ok = (
                float(value) >= float(target)
                if direction == "higher"
                else (float(value) <= float(target))
            )
            row.verdict = "PASS" if ok else "MISS"
        rows.append(row)
    return rows


def targets_markdown(rows: Sequence[TargetRow], *, title: str = "") -> str:
    """Render :func:`check_targets`'s rows as a markdown table.

    Args:
        rows: Rows to render.
        title: Optional ``###`` heading above the table.

    Returns:
        The markdown, ending in a newline.
    """
    lines: list[str] = []
    if title:
        lines += [f"### {title}", ""]
    lines += ["| Metric | Target | Actual | Verdict | Note |", "|---|---|---|---|---|"]
    for row in rows:
        target = "-" if row.target is None else f"{row.target:g}"
        actual = "-" if row.actual is None else f"{row.actual:g}"
        arrow = ">=" if row.direction == "higher" else "<="
        lines.append(f"| {row.label} | {arrow} {target} | {actual} | {row.verdict} | {row.note} |")
    return "\n".join(lines) + "\n"
