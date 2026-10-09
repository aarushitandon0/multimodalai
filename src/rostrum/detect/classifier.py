"""Flaw-type classifier and severity calibrator (spec 10.6).

The detector of :mod:`rostrum.detect.regions` is a *rule*: a label fires when
the signed deviation of its primary feature crosses the hysteresis thresholds.
That is the part of the system that can be explained, and spec 10.6 is explicit
that it stays primary. This module adds a second opinion on top of it, for two
jobs the rule cannot do on its own:

1. **Typing.** The rules are per-label and independent, so two of them can fire
   over the same words (a rushed passage is often also a quiet one). A learned
   model sees *all* the deviations of a region at once, which is exactly the
   information needed to say which label the evidence best supports.
2. **Severity.** The rule produces a region, not a rung of the severity ladder.
   An isotonic regression maps the region's deviation mass onto ``s`` in
   [0, 1], monotonically by construction, and the nearest rung of the
   ``injection.severity`` ladder becomes :attr:`~rostrum.types.Region.level`.

**The label policy** (spec 10.6, ``detection.classifier.on_disagreement``) is
the whole point of the module's restraint: :attr:`~rostrum.types.Region.label`
keeps the rule's answer whatever the classifier says. The classifier's label and
probability travel beside it, and when the two disagree *and* the classifier is
confident (``clf_prob > 0.8``) the region is tagged :data:`UNCERTAIN_TAG`, so a
card can say the type is uncertain instead of quietly asserting the wrong one.

**The leakage guard** is structural, as in :mod:`rostrum.detect.tune`: a
classifier is fitted from :class:`ClassifierCase` objects of one fold's
*training* texts, and the held-out text has no way in. Human-recorded takes are
test-only (CLAUDE.md invariant 11), which ``detection.classifier.train_on``
enforces by listing only ``injected``; :func:`fit_fold` applies both filters
itself and records in the model what it was shown.

**The design matrix** (spec 10.6, "features per region") is built once as a
:class:`RegionFeatureSpec` so that a model saved for fold *A* can never be fed a
differently-ordered vector in fold *B*: the column order is stored with the
model and checked on load. Per feature *k* a region contributes

* ``mean_z__k`` - the mean *signed* z inside the region, and
* ``max_z__k`` - the signed z of the unit where ``|z|`` was largest,

signed rather than absolute because the sign is what separates ``rushing`` from
``dragging``; the magnitude alone cannot. A region's words, the gaps between
them and the phrases it overlaps are all read, so a gap region still carries the
rate of the words around it. Features no unit could supply stay ``NaN``:
``HistGradientBoostingClassifier`` splits on missingness natively, so a
channel-sensitive column in a cross-channel comparison is absent rather than
imputed to a lie.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features import names as N

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.detect.zscore import ZResult
    from rostrum.types import Region, SeverityLevel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MIN_IOU",
    "DEFAULT_UNCERTAIN_PROB",
    "GAP_LABEL_COLUMNS",
    "PHRASE_POSITION_COLUMNS",
    "SCALAR_COLUMNS",
    "UNCERTAIN_TAG",
    "ClassifierCase",
    "ClassifierError",
    "FlawClassifier",
    "Prediction",
    "RegionFeatureSpec",
    "SeverityCalibrator",
    "TrainingExample",
    "TruthFlaw",
    "apply_label_policy",
    "build_examples",
    "evaluate",
    "evidence_scalar",
    "fit_fold",
    "fit_severity",
    "iou",
    "load_model",
    "match_to_truth",
    "model_path",
    "region_vector",
    "save_model",
    "truth_flaws_from_label_json",
]


DEFAULT_MIN_IOU: float = 0.30
"""Spec 10.6: a predicted region trains on the truth it overlaps this much."""

DEFAULT_UNCERTAIN_PROB: float = 0.80
"""A disagreement above this classifier probability is surfaced, not hidden."""

UNCERTAIN_TAG: str = "uncertain_type"
"""Appended to :attr:`~rostrum.types.Region.context` on a confident disagreement."""

SCALAR_COLUMNS: tuple[str, ...] = ("duration_s", "n_words")
"""Non-feature columns of the design matrix (spec 10.6: duration, n words)."""

GAP_LABEL_COLUMNS: tuple[str, ...] = (
    "gap_label__boundary",
    "gap_label__minor",
    "gap_label__none",
)
"""One-hot of the text's pause expectation; all zero for a non-gap region."""

PHRASE_POSITION_COLUMNS: tuple[str, ...] = (
    "phrase_pos__start",
    "phrase_pos__mid",
    "phrase_pos__end",
)
"""One-hot of where in its phrase the region starts (spec 10.6)."""

_UNITS: tuple[str, ...] = ("word", "gap", "phrase")


class ClassifierError(ValueError):
    """Raised when a classifier cannot be fitted or applied as specified."""


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TruthFlaw:
    """One ground-truth flaw from the dataset label JSON (spec 7.1).

    :class:`~rostrum.detect.tune.Event` carries only a label and a span, because
    event-F1 needs nothing else. Training needs the severity too, so this is the
    same interval with ``level`` and ``s`` attached.

    Attributes:
        label: Flaw label, as written in the label JSON.
        t_start: Onset in seconds.
        t_end: Offset in seconds.
        s: Continuous severity in [0, 1] from the ladder.
        level: Severity rung, ``"L0"`` to ``"L4"``.
        file_id: Recording the flaw belongs to.
    """

    label: str
    t_start: float
    t_end: float
    s: float = 0.0
    level: str = "L0"
    file_id: str = ""

    @property
    def duration(self) -> float:
        """Flaw duration in seconds."""
        return float(self.t_end) - float(self.t_start)


def truth_flaws_from_label_json(data: Mapping[str, Any]) -> list[TruthFlaw]:
    """Read the ``flaws`` array of one label JSON into :class:`TruthFlaw` rows.

    Args:
        data: A parsed ``dataset/labels/<file_id>.json`` (spec 7.1).

    Returns:
        One row per flaw, in file order.
    """
    file_id = str(data.get("file_id", ""))
    return [
        TruthFlaw(
            label=str(flaw["label"]),
            t_start=float(flaw["t_start"]),
            t_end=float(flaw["t_end"]),
            s=float(flaw.get("s", 0.0)),
            level=str(flaw.get("level", "L0")),
            file_id=file_id,
        )
        for flaw in data.get("flaws", ())
    ]


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Temporal intersection over union of two intervals.

    Returns ``0.0`` when the intervals do not overlap or either is empty, rather
    than dividing by zero.
    """
    lo = max(float(a_start), float(b_start))
    hi = min(float(a_end), float(b_end))
    overlap = max(0.0, hi - lo)
    if overlap <= 0.0:
        return 0.0
    union = (float(a_end) - float(a_start)) + (float(b_end) - float(b_start)) - overlap
    if union <= 0.0:
        return 0.0
    return overlap / union


def match_to_truth(
    regions: Sequence[Region],
    truth: Sequence[TruthFlaw],
    *,
    min_iou: float = DEFAULT_MIN_IOU,
) -> dict[int, TruthFlaw]:
    """Assign each region the ground-truth flaw it overlaps most (spec 10.6).

    The matching is deliberately **label-agnostic**. The classifier's job is to
    learn which label the evidence supports, so training it only on regions the
    rule already labelled correctly would teach it nothing except the rule. A
    region is matched to the truth interval with the largest IoU provided that
    IoU reaches ``min_iou``, and each truth interval is used at most once; ties
    go to the earlier region and then the earlier truth interval, so the
    assignment is identical on every machine (CLAUDE.md invariant 13).

    Args:
        regions: Predicted regions from one recording.
        truth: Ground-truth flaws of that same recording.
        min_iou: Overlap floor; spec 10.6 uses 0.3.

    Returns:
        ``{region_id: TruthFlaw}``, omitting regions that matched nothing.
    """
    pairs: list[tuple[float, int, int]] = []
    for r_i, region in enumerate(regions):
        for t_i, flaw in enumerate(truth):
            score = iou(region.t_start, region.t_end, flaw.t_start, flaw.t_end)
            if score >= float(min_iou):
                pairs.append((score, r_i, t_i))
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))

    used_regions: set[int] = set()
    used_truth: set[int] = set()
    out: dict[int, TruthFlaw] = {}
    for _score, r_i, t_i in pairs:
        if r_i in used_regions or t_i in used_truth:
            continue
        used_regions.add(r_i)
        used_truth.add(t_i)
        out[int(regions[r_i].region_id)] = truth[t_i]
    return out


# --------------------------------------------------------------------------- #
# The design matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegionFeatureSpec:
    """The column layout of the region feature vector (spec 10.6).

    Stored with every saved model, so a vector can never be fed to a model
    fitted on a different layout.

    Attributes:
        features: Canonical feature names contributing ``mean_z__`` and
            ``max_z__`` columns, in a fixed sorted order.
    """

    features: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Every column of the design matrix, in order."""
        z_columns = [
            f"{prefix}__{name}" for name in self.features for prefix in ("mean_z", "max_z")
        ]
        return (*z_columns, *SCALAR_COLUMNS, *GAP_LABEL_COLUMNS, *PHRASE_POSITION_COLUMNS)

    @property
    def n_columns(self) -> int:
        """Width of the design matrix."""
        return len(self.columns)

    @classmethod
    def from_config(cls, config: Config) -> RegionFeatureSpec:
        """Every envelope feature of ``configs/detection.yaml``, channel-sensitive included.

        Taking the vocabulary from the config rather than from one recording
        means a fold whose comparison happened to be cross-channel still
        produces vectors of the right width, with the channel-sensitive columns
        left ``NaN``.
        """
        envelope = config.detection["envelope"]
        names: set[str] = set()
        for block in ("features", "channel_sensitive_features"):
            for unit_features in envelope[block].values():
                names.update(str(feature) for feature in unit_features)
        return cls(features=tuple(sorted(names)))

    @classmethod
    def from_results(cls, z_by_unit: Mapping[str, ZResult]) -> RegionFeatureSpec:
        """The canonical features present in one comparison; for tests without a config."""
        names: set[str] = set()
        for result in z_by_unit.values():
            names.update(N.canonical(column) for column in result.features)
        return cls(features=tuple(sorted(names)))


def _column_for(result: ZResult, feature: str) -> str | None:
    """The table column carrying one canonical feature, or ``None``."""
    for column in result.features:
        if N.canonical(column) == feature:
            return column
    return None


def _unit_mask(region: Region, result: ZResult, unit: str) -> Any:
    """Boolean mask over ``result``'s index for the units this region covers.

    * **word** - the inclusive word span the region reports;
    * **gap** - the region's own gap, or every gap *between* its words;
    * **phrase** - every phrase overlapping the region in time, read from the
      table rather than from indices, because a phrase index means nothing to a
      word-based region.
    """
    index = result.z.index
    if unit == "gap":
        if region.gap_idx is not None:
            return index == int(region.gap_idx)
        if len(region.word_idx) == 2:
            first, last = int(region.word_idx[0]), int(region.word_idx[1])
            # Gap i follows word i, so the gaps strictly inside [first, last]
            # are first .. last-1.
            return (index >= first) & (index <= last - 1)
        return index != index  # all False
    if unit == "phrase":
        table = result.table
        if "t_start" not in table.columns or "t_end" not in table.columns:
            return index != index
        return (table["t_end"] > region.t_start) & (table["t_start"] < region.t_end)
    if len(region.word_idx) == 2:
        return (index >= int(region.word_idx[0])) & (index <= int(region.word_idx[1]))
    return index != index


def _phrase_position(region: Region, z_by_unit: Mapping[str, ZResult]) -> str | None:
    """``"start"``, ``"mid"`` or ``"end"`` for the region's place in its phrase.

    The phrase table supplies ``first_word_idx``/``last_word_idx``. A region
    whose first word sits in the first third of its phrase is at the start, in
    the last third at the end, otherwise mid; a one-word phrase counts as
    ``"start"``. ``None`` means no phrase contains the region, which the one-hot
    encodes as three zeros rather than inventing a position.
    """
    result = z_by_unit.get("phrase")
    if result is None or not len(region.word_idx):
        return None
    table = result.table
    if "first_word_idx" not in table.columns or "last_word_idx" not in table.columns:
        return None
    first_word = int(region.word_idx[0])
    inside = (table["first_word_idx"] <= first_word) & (table["last_word_idx"] >= first_word)
    rows = table.loc[inside]
    if rows.empty:
        return None
    row = rows.iloc[0]
    lo, hi = int(row["first_word_idx"]), int(row["last_word_idx"])
    if hi <= lo:
        return "start"
    position = (first_word - lo) / (hi - lo)
    if position <= 1.0 / 3.0:
        return "start"
    if position >= 2.0 / 3.0:
        return "end"
    return "mid"


def _gap_label(region: Region, z_by_unit: Mapping[str, ZResult]) -> str | None:
    """The text's pause expectation at a gap region, or ``None`` elsewhere."""
    if region.gap_idx is None:
        return None
    result = z_by_unit.get("gap")
    if result is None or "gap_label" not in result.table.columns:
        return None
    try:
        return str(result.table.loc[int(region.gap_idx), "gap_label"])
    except KeyError:
        return None


def region_vector(
    region: Region,
    z_by_unit: Mapping[str, ZResult],
    spec: RegionFeatureSpec,
) -> np.ndarray:
    """Build one region's feature vector in ``spec``'s column order (spec 10.6).

    Args:
        region: The region to describe.
        z_by_unit: ``{unit: ZResult}`` for the recording the region came from.
        spec: The column layout.

    Returns:
        A ``float64`` vector of length ``spec.n_columns``. Features no unit
        could supply are ``NaN``, which the gradient-boosting model treats as
        missing rather than as zero.
    """
    values: dict[str, float] = {}
    for feature in spec.features:
        mean_z = math.nan
        max_z = math.nan
        for unit in _UNITS:
            result = z_by_unit.get(unit)
            if result is None:
                continue
            column = _column_for(result, feature)
            if column is None:
                continue
            series = result.z.loc[_unit_mask(region, result, unit), column]
            if series.empty:
                continue
            raw = series.to_numpy(dtype=np.float64, na_value=np.nan)
            finite = raw[np.isfinite(raw)]
            if finite.size == 0:
                continue
            mean_z = float(finite.mean())
            # The signed value where |z| was largest: the sign is what
            # separates rushing from dragging, so an absolute maximum would
            # discard the only thing that distinguishes them.
            max_z = float(finite[int(np.argmax(np.abs(finite)))])
            break  # first unit carrying the feature wins: word, then gap, then phrase
        values[f"mean_z__{feature}"] = mean_z
        values[f"max_z__{feature}"] = max_z

    values["duration_s"] = float(region.duration)
    values["n_words"] = (
        float(int(region.word_idx[1]) - int(region.word_idx[0]) + 1)
        if len(region.word_idx) == 2
        else math.nan
    )

    gap_label = _gap_label(region, z_by_unit)
    for column in GAP_LABEL_COLUMNS:
        values[column] = 1.0 if gap_label == column.split("__", 1)[1] else 0.0

    position = _phrase_position(region, z_by_unit)
    for column in PHRASE_POSITION_COLUMNS:
        values[column] = 1.0 if position == column.split("__", 1)[1] else 0.0

    return np.array([values[column] for column in spec.columns], dtype=np.float64)


def evidence_scalar(region: Region) -> float:
    """The one-dimensional deviation the severity calibrator reads.

    :attr:`~rostrum.types.Region.mean_excess_z` is the region's mean of
    :math:`(\\mathrm{score} - \\tau_{out})_+` - the same quantity the rubric
    integrates into the deviation mass (spec 11), and the natural monotone
    stand-in for "how far outside the envelope this stretch was". When it could
    not be formed the peak ``|z|`` is used instead, so a region always has an
    input.
    """
    mass = float(region.mean_excess_z)
    if math.isfinite(mass):
        return mass
    peak = float(region.peak_z)
    return abs(peak) if math.isfinite(peak) else 0.0


# --------------------------------------------------------------------------- #
# Training examples
# --------------------------------------------------------------------------- #


@dataclass
class ClassifierCase:
    """One recording's regions, comparisons and ground truth.

    Attributes:
        file_id: Recording id.
        text_id: Text id; the fold's training texts are selected on this.
        regions: Regions the rule produced at this fold's thresholds.
        z_by_unit: The comparisons those regions were built from.
        truth: Ground-truth flaws, from the label JSON.
        origin: Provenance. ``detection.classifier.train_on`` restricts which
            origins may be fitted on, which is how human takes stay test-only.
    """

    file_id: str
    text_id: str
    regions: Sequence[Region]
    z_by_unit: Mapping[str, ZResult]
    truth: Sequence[TruthFlaw] = field(default_factory=tuple)
    origin: str = "injected"


@dataclass(frozen=True)
class TrainingExample:
    """One matched region, ready for the model.

    Attributes:
        x: The feature vector.
        label: Ground-truth label; the classification target.
        s: Ground-truth severity; the isotonic target.
        level: Ground-truth rung.
        evidence: The scalar the severity calibrator reads.
        rule_label: What the rule said, so the report can compare the two.
        file_id: Recording id.
        text_id: Text id.
        region_id: Region id within that recording.
    """

    x: np.ndarray
    label: str
    s: float
    level: str
    evidence: float
    rule_label: str
    file_id: str
    text_id: str
    region_id: int


def build_examples(
    cases: Sequence[ClassifierCase],
    spec: RegionFeatureSpec,
    *,
    min_iou: float = DEFAULT_MIN_IOU,
) -> list[TrainingExample]:
    """Match every case's regions to its truth and vectorise the matches.

    Regions that match nothing are dropped and counted in the log: a false
    positive has no ground-truth type, so a *type* classifier has nothing to
    learn from it. Nothing is being hidden - event-F1 in
    :mod:`rostrum.detect.tune` already counts it against precision.

    Args:
        cases: Recordings, already detected.
        spec: Column layout.
        min_iou: Overlap floor for a match.

    Returns:
        Examples in case order, region order within a case.
    """
    out: list[TrainingExample] = []
    n_regions = 0
    for case in cases:
        matched = match_to_truth(case.regions, case.truth, min_iou=min_iou)
        n_regions += len(case.regions)
        for region in case.regions:
            flaw = matched.get(int(region.region_id))
            if flaw is None:
                continue
            out.append(
                TrainingExample(
                    x=region_vector(region, case.z_by_unit, spec),
                    label=str(flaw.label),
                    s=float(flaw.s),
                    level=str(flaw.level),
                    evidence=evidence_scalar(region),
                    rule_label=str(region.rule_label or region.label),
                    file_id=case.file_id,
                    text_id=case.text_id,
                    region_id=int(region.region_id),
                )
            )
    logger.info(
        "classifier: %d of %d region(s) matched ground truth at IoU >= %.2f",
        len(out),
        n_regions,
        min_iou,
    )
    return out


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #


@dataclass
class SeverityCalibrator:
    """Isotonic map from a region's deviation onto the severity ladder (spec 10.6).

    Monotonicity is the reason for isotonic regression rather than anything
    richer: "further outside the envelope" must never mean "less severe", and
    that is a constraint of the method, not a property one hopes the fit happens
    to have.

    A per-label calibrator is fitted wherever a label has at least
    ``min_per_label`` examples, because the same deviation mass means different
    things for a four-second ``monotone`` stretch and a single missing pause.
    The global fit is the fallback for everything else.

    Attributes:
        global_model: Isotonic regression over all examples, or ``None`` when
            there were too few to fit.
        per_label: Per-label isotonic regressions.
        levels: The ladder, ``(("L0", 0.0), ...)``, from ``injection.severity``.
        min_per_label: Examples a label needs before it gets its own fit.
    """

    global_model: Any = None
    per_label: dict[str, Any] = field(default_factory=dict)
    levels: tuple[tuple[str, float], ...] = (
        ("L0", 0.00),
        ("L1", 0.15),
        ("L2", 0.35),
        ("L3", 0.60),
        ("L4", 0.90),
    )
    min_per_label: int = 5

    def predict(self, evidence: float, label: str | None = None) -> float:
        """Severity ``s`` in [0, 1] for one region.

        Falls back to the global fit when the label has none of its own, and to
        ``0.0`` when nothing was fitted at all - an uncalibrated model reports
        no severity rather than a guessed one.
        """
        model = self.per_label.get(str(label)) if label is not None else None
        if model is None:
            model = self.global_model
        if model is None:
            return 0.0
        value = float(np.asarray(model.predict([float(evidence)]), dtype=np.float64)[0])
        return float(min(1.0, max(0.0, value)))

    def level(self, s: float) -> SeverityLevel:
        """The nearest rung of the ladder to ``s``; ties go to the lower rung.

        The distance is rounded before comparison. Without that, ``s = 0.25`` -
        exactly between L1 (0.15) and L2 (0.35) - resolves by the last bit of
        two floating-point subtractions rather than by the stated rule, which is
        both arbitrary and a determinism hazard (CLAUDE.md invariant 13).
        """
        rung = min(self.levels, key=lambda item: (round(abs(float(s) - item[1]), 9), item[1]))
        return rung[0]  # type: ignore[return-value]

    def to_meta(self) -> dict[str, Any]:
        """Small description of the fit, for the model card and the report."""
        return {
            "fitted": self.global_model is not None,
            "per_label": sorted(self.per_label),
            "min_per_label": int(self.min_per_label),
            "levels": [[name, value] for name, value in self.levels],
        }


def _levels_from_config(config: Config | None) -> tuple[tuple[str, float], ...]:
    """The severity ladder from ``configs/injection.yaml``, ordered by ``s``."""
    if config is None:
        return SeverityCalibrator().levels
    ladder = config.injection["severity"]
    return tuple(sorted(((str(k), float(v)) for k, v in ladder.items()), key=lambda r: r[1]))


def fit_severity(
    examples: Sequence[TrainingExample],
    *,
    config: Config | None = None,
    min_per_label: int = 5,
) -> SeverityCalibrator:
    """Fit the global and per-label isotonic regressions.

    Args:
        examples: Matched training regions.
        config: Supplies the severity ladder and the ``out_of_bounds`` policy.
        min_per_label: Examples a label needs before it gets its own fit.

    Returns:
        A :class:`SeverityCalibrator`; unfitted when there is no usable example.
    """
    from sklearn.isotonic import IsotonicRegression

    out_of_bounds = "clip"
    if config is not None:
        out_of_bounds = str(
            config.detection["classifier"]["severity_calibration"].get("out_of_bounds", "clip")
        )

    def fit(subset: Sequence[TrainingExample]) -> Any | None:
        x = np.array([e.evidence for e in subset], dtype=np.float64)
        y = np.array([e.s for e in subset], dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(y)
        # Isotonic regression needs at least two distinct x values; with one it
        # would be a constant dressed up as a calibration.
        if keep.sum() < 2 or np.unique(x[keep]).size < 2:
            return None
        model = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds=out_of_bounds
        )
        model.fit(x[keep], y[keep])
        return model

    per_label: dict[str, Any] = {}
    for label in sorted({e.label for e in examples}):
        subset = [e for e in examples if e.label == label]
        if len(subset) < int(min_per_label):
            continue
        model = fit(subset)
        if model is not None:
            per_label[label] = model

    calibrator = SeverityCalibrator(
        global_model=fit(examples),
        per_label=per_label,
        levels=_levels_from_config(config),
        min_per_label=int(min_per_label),
    )
    logger.info(
        "severity: global fit %s, per-label fits for %s",
        "ok" if calibrator.global_model is not None else "unavailable",
        sorted(per_label) or "none",
    )
    return calibrator


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Prediction:
    """What the classifier says about one region.

    Attributes:
        label: The classifier's most probable label.
        prob: Its probability for that label.
        s: Calibrated severity in [0, 1].
        level: The nearest rung of the ladder to ``s``.
    """

    label: str
    prob: float
    s: float
    level: str


@dataclass
class FlawClassifier:
    """A fitted per-fold flaw-type model and its severity calibrator.

    Attributes:
        fold: LOTO fold this was fitted for.
        spec: Column layout of the design matrix.
        classes: Labels the model can emit, in the model's own order.
        model: The fitted ``HistGradientBoostingClassifier``.
        severity: The fitted :class:`SeverityCalibrator`.
        train_texts: Texts the fit was allowed to see; the audit trail for the
            leakage guard.
        train_origins: Provenances the fit was allowed to see.
        n_train: Training examples used.
        config_hash: ``Config.hash`` at fit time, checked on load.
        sklearn_version: Recorded, so a model that may not reproduce under a
            different scikit-learn is visible rather than silently reused.
    """

    fold: str
    spec: RegionFeatureSpec
    classes: tuple[str, ...]
    model: Any
    severity: SeverityCalibrator
    train_texts: tuple[str, ...] = ()
    train_origins: tuple[str, ...] = ()
    n_train: int = 0
    config_hash: str = ""
    sklearn_version: str = ""

    def predict_one(self, region: Region, z_by_unit: Mapping[str, ZResult]) -> Prediction:
        """Classify one region and calibrate its severity."""
        x = region_vector(region, z_by_unit, self.spec).reshape(1, -1)
        probabilities = np.asarray(self.model.predict_proba(x), dtype=np.float64)[0]
        best = int(np.argmax(probabilities))
        label = str(self.classes[best])
        s = self.severity.predict(evidence_scalar(region), label)
        return Prediction(
            label=label, prob=float(probabilities[best]), s=float(s), level=self.severity.level(s)
        )

    def predict(
        self, regions: Sequence[Region], z_by_unit: Mapping[str, ZResult]
    ) -> list[Prediction]:
        """Classify several regions of one recording."""
        return [self.predict_one(region, z_by_unit) for region in regions]

    def annotate(
        self,
        regions: Sequence[Region],
        z_by_unit: Mapping[str, ZResult],
        *,
        uncertain_prob: float = DEFAULT_UNCERTAIN_PROB,
    ) -> list[Region]:
        """Attach predictions to regions under the spec 10.6 label policy.

        The regions are modified in place and returned, so a caller can treat
        this as one pass over its own list.
        """
        for region in regions:
            apply_label_policy(
                region, self.predict_one(region, z_by_unit), uncertain_prob=uncertain_prob
            )
        return list(regions)

    def to_meta(self) -> dict[str, Any]:
        """Everything about the fit except the fitted objects themselves."""
        return {
            "fold": self.fold,
            "classes": list(self.classes),
            "columns": list(self.spec.columns),
            "n_train": int(self.n_train),
            "train_texts": list(self.train_texts),
            "train_origins": list(self.train_origins),
            "config_hash": self.config_hash,
            "sklearn_version": self.sklearn_version,
            "severity": self.severity.to_meta(),
        }


def _model_params(config: Config | None) -> dict[str, Any]:
    """``HistGradientBoostingClassifier`` keywords from ``detection.classifier``.

    ``random_state=0`` is spec 10.6 and ``early_stopping=False`` is CLAUDE.md
    invariant 13: early stopping carves a validation split out of the training
    set, which would make two runs on the same data differ.
    """
    defaults: dict[str, Any] = {
        "random_state": 0,
        "max_iter": 200,
        "learning_rate": 0.1,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "early_stopping": False,
    }
    if config is None:
        return defaults
    block = config.detection["classifier"]
    return {
        "random_state": int(block.get("random_state", 0)),
        "max_iter": int(block.get("max_iter", 200)),
        "learning_rate": float(block.get("learning_rate", 0.1)),
        "max_leaf_nodes": int(block.get("max_leaf_nodes", 15)),
        "l2_regularization": float(block.get("l2_regularization", 1.0)),
        "early_stopping": bool(block.get("early_stopping", False)),
    }


def _fit_matrix(examples: Sequence[TrainingExample], spec: RegionFeatureSpec) -> Any:
    """Stack the training vectors into a design matrix scikit-learn can bin.

    A column no training example could supply is ``NaN`` all the way down - the
    channel-sensitive columns of a cross-channel fold, or any feature the
    envelope of these texts never carried (see the module docstring). Partially
    missing columns are exactly what ``HistGradientBoostingClassifier`` handles
    natively, but a *fully* missing one leaves its binner with no distinct value
    to place a threshold between, and scikit-learn >= 1.9 raises ``ValueError:
    window shape cannot be larger than input array shape`` out of
    ``sliding_window_view``.

    Such a column carries no information by definition, so it is filled with a
    single constant: the binner then sees one distinct value, produces no
    threshold, and no tree can split on it. The layout is left untouched -
    :attr:`RegionFeatureSpec.columns` is stored with the model and checked on
    load - and because the column is unsplittable the fitted model is the one
    the informative columns alone would have produced.
    """
    x = np.vstack([e.x for e in examples])
    empty = np.flatnonzero(np.all(np.isnan(x), axis=0))
    if empty.size:
        columns = spec.columns
        logger.info(
            "%d of %d column(s) are missing from every training example and cannot be "
            "split on: %s",
            empty.size,
            x.shape[1],
            [columns[i] for i in empty] if len(columns) == x.shape[1] else list(empty),
        )
        x[:, empty] = 0.0
    return x


def fit_fold(
    cases: Sequence[ClassifierCase],
    *,
    fold: str,
    train_texts: Sequence[str],
    config: Config | None = None,
    spec: RegionFeatureSpec | None = None,
    min_iou: float = DEFAULT_MIN_IOU,
    min_per_label: int = 5,
) -> FlawClassifier:
    """Fit one fold's classifier on its training texts only (spec 10.6, 7.3).

    The held-out text cannot leak in, because this function *selects* the
    training cases itself: anything whose ``text_id`` is not in ``train_texts``,
    or whose ``origin`` is not in ``detection.classifier.train_on``, is dropped
    before a single vector is built. Human takes are test-only by that second
    filter (CLAUDE.md invariant 11).

    Args:
        cases: Every detected recording available; the selection happens here.
        fold: Fold id, recorded in the model.
        train_texts: The fold's training texts.
        config: Supplies the column vocabulary, model parameters and the ladder.
        spec: Explicit column layout, overriding the config.
        min_iou: Overlap floor for matching a region to the truth.
        min_per_label: Examples a label needs for its own severity fit.

    Returns:
        A fitted :class:`FlawClassifier`.

    Raises:
        ClassifierError: No training example survived the filters, or fewer than
            two labels are present - a classifier over one class is not a
            classifier, and returning one would make that fold's macro-F1
            meaningless.
    """
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    allowed_origins = ("injected",)
    if config is not None:
        allowed_origins = tuple(str(o) for o in config.detection["classifier"]["train_on"])

    wanted = {str(text) for text in train_texts}
    training = [c for c in cases if c.text_id in wanted and c.origin in allowed_origins]
    withheld = len(cases) - len(training)
    if withheld:
        logger.info(
            "fold %s: %d case(s) withheld from the fit (held-out text, or origin outside %s)",
            fold,
            withheld,
            list(allowed_origins),
        )

    if spec is None:
        spec = (
            RegionFeatureSpec.from_config(config)
            if config is not None
            else RegionFeatureSpec.from_results(training[0].z_by_unit if training else {})
        )

    examples = build_examples(training, spec, min_iou=min_iou)
    if not examples:
        raise ClassifierError(
            f"fold {fold!r}: no region matched ground truth at IoU >= {min_iou} on the "
            f"training texts {sorted(wanted)}; nothing to fit"
        )
    labels = sorted({e.label for e in examples})
    if len(labels) < 2:
        raise ClassifierError(
            f"fold {fold!r}: only the label {labels[0]!r} appears in {len(examples)} training "
            "example(s); a flaw-type classifier needs at least two"
        )

    model = HistGradientBoostingClassifier(**_model_params(config))
    model.fit(
        _fit_matrix(examples, spec),
        np.array([e.label for e in examples], dtype=object),
    )

    classifier = FlawClassifier(
        fold=str(fold),
        spec=spec,
        classes=tuple(str(c) for c in model.classes_),
        model=model,
        severity=fit_severity(examples, config=config, min_per_label=min_per_label),
        train_texts=tuple(sorted(wanted)),
        train_origins=tuple(allowed_origins),
        n_train=len(examples),
        config_hash=config.hash if config is not None else "",
        sklearn_version=sklearn.__version__,
    )
    logger.info(
        "fold %s: fitted %s on %d example(s) over %d class(es): %s",
        fold,
        type(model).__name__,
        len(examples),
        len(labels),
        labels,
    )
    return classifier


# --------------------------------------------------------------------------- #
# The label policy
# --------------------------------------------------------------------------- #


def apply_label_policy(
    region: Region,
    prediction: Prediction,
    *,
    uncertain_prob: float = DEFAULT_UNCERTAIN_PROB,
) -> Region:
    """Attach a prediction to a region without overruling the rule (spec 10.6).

    * :attr:`~rostrum.types.Region.label` is left alone. The rule is what the
      card explains, and an unexplainable relabelling would break the chain from
      measurement to claim.
    * ``clf_label`` and ``clf_prob`` are recorded, so a disagreement is visible
      in the JSON rather than discarded.
    * ``s`` and ``level`` come from the calibrator; the rule produces no
      severity of its own.
    * On a **confident** disagreement (``clf_prob > uncertain_prob``) the tag
      :data:`UNCERTAIN_TAG` is appended to ``context``, which the card template
      reads to say the type is uncertain instead of asserting one. ``context``
      already carries the signal variant (``"fading"``), so the tags are joined
      with ``";"`` and the existing value is preserved.

    Returns:
        The same region, modified in place.
    """
    region.clf_label = prediction.label  # type: ignore[assignment]
    region.clf_prob = float(prediction.prob)
    region.s = float(prediction.s)
    region.level = prediction.level  # type: ignore[assignment]

    rule_label = str(region.rule_label or region.label)
    if prediction.label != rule_label and float(prediction.prob) > float(uncertain_prob):
        tags = [tag for tag in str(region.context).split(";") if tag]
        if UNCERTAIN_TAG not in tags:
            tags.append(UNCERTAIN_TAG)
        region.context = ";".join(tags)
        logger.debug(
            "region %d: rule says %s, classifier says %s at p=%.2f -> %s",
            region.region_id,
            rule_label,
            prediction.label,
            prediction.prob,
            UNCERTAIN_TAG,
        )
    return region


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    classifier: FlawClassifier,
    examples: Sequence[TrainingExample],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Macro-F1, per-label rates and the confusion matrix on held-out examples.

    The *rule's* macro-F1 over the same examples is reported alongside, because
    "the classifier types flaws better than the rule" is a claim the report
    should have to back with both numbers.

    Macro-F1 averages over the labels that actually occur in the truth:
    including a label with no support would average in a zero for a question
    never asked, which flatters or punishes the model at random depending on the
    fold.

    Args:
        classifier: The fitted model.
        examples: Held-out matched regions, from :func:`build_examples`.
        labels: Label order for the matrix; defaults to every label appearing in
            the truth, the predictions or the model's classes.

    Returns:
        ``macro_f1``, ``accuracy``, ``per_label``, ``confusion`` (``labels``
        plus a row-per-truth ``matrix``), ``rule_macro_f1`` and ``n``. Empty but
        valid when ``examples`` is empty, so a fold with no held-out match does
        not crash the report.
    """
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    if not examples:
        logger.warning("fold %s: no held-out example to evaluate", classifier.fold)
        return {
            "n": 0,
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "per_label": {},
            "confusion": {"labels": [], "matrix": []},
            "rule_macro_f1": 0.0,
        }

    probabilities = np.asarray(
        classifier.model.predict_proba(np.vstack([e.x for e in examples])), dtype=np.float64
    )
    predicted = [str(classifier.classes[int(i)]) for i in np.argmax(probabilities, axis=1)]
    truth = [e.label for e in examples]
    rule = [e.rule_label for e in examples]

    order = (
        list(labels)
        if labels is not None
        else sorted(set(truth) | set(predicted) | set(classifier.classes))
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=order, zero_division=0
    )
    _, _, rule_f1, _ = precision_recall_fscore_support(truth, rule, labels=order, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=order)

    present = [i for i, _label in enumerate(order) if support[i] > 0]
    return {
        "n": len(examples),
        "macro_f1": float(np.mean([f1[i] for i in present])) if present else 0.0,
        "accuracy": float(np.mean([t == p for t, p in zip(truth, predicted, strict=True)])),
        "per_label": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(order)
        },
        "confusion": {"labels": list(order), "matrix": [[int(v) for v in row] for row in matrix]},
        "rule_macro_f1": float(np.mean([rule_f1[i] for i in present])) if present else 0.0,
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def model_path(fold: str, config: Config) -> Path:
    """``.cache/models/clf_<fold>.joblib``, the per-fold model file (spec 10.6)."""
    return config.path("cache.models") / f"clf_{fold}.joblib"


def save_model(classifier: FlawClassifier, path: Path | str) -> Path:
    """Write one fold's model with ``joblib``, carrying its config hash.

    The payload holds the fitted model, the severity calibrator, the column
    layout and the metadata, so a loaded model can be *checked* against the
    config about to use it rather than trusted.

    Returns:
        The path written.
    """
    import joblib

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema": 1,
            "fold": classifier.fold,
            "columns": list(classifier.spec.columns),
            "features": list(classifier.spec.features),
            "classes": list(classifier.classes),
            "model": classifier.model,
            "severity": classifier.severity,
            "meta": classifier.to_meta(),
        },
        destination,
    )
    logger.info("wrote fold %s classifier to %s", classifier.fold, destination)
    return destination


def load_model(path: Path | str, *, config: Config | None = None) -> FlawClassifier:
    """Read a model back, warning when it was fitted under a different config.

    A config-hash mismatch is a warning rather than an error: it is legitimate
    to inspect an old model and illegitimate to report a metric from one without
    noticing. The warning names both hashes so the difference can be traced.

    Raises:
        ClassifierError: The file is not a model payload this version wrote, or
            its stored column layout contradicts its feature list.
    """
    import joblib

    source = Path(path)
    payload = joblib.load(source)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ClassifierError(f"{source} is not a Rostrum classifier payload")
    meta = dict(payload.get("meta", {}))
    classifier = FlawClassifier(
        fold=str(payload.get("fold", meta.get("fold", ""))),
        spec=RegionFeatureSpec(features=tuple(str(f) for f in payload.get("features", ()))),
        classes=tuple(str(c) for c in payload.get("classes", ())),
        model=payload["model"],
        severity=payload.get("severity", SeverityCalibrator()),
        train_texts=tuple(str(t) for t in meta.get("train_texts", ())),
        train_origins=tuple(str(o) for o in meta.get("train_origins", ())),
        n_train=int(meta.get("n_train", 0)),
        config_hash=str(meta.get("config_hash", "")),
        sklearn_version=str(meta.get("sklearn_version", "")),
    )
    stored = list(payload.get("columns", ()))
    if stored and stored != list(classifier.spec.columns):
        raise ClassifierError(
            f"{source}: the stored column layout does not match the saved feature list; "
            "the model was written by an incompatible version"
        )
    if config is not None and classifier.config_hash and classifier.config_hash != config.hash:
        logger.warning(
            "%s was fitted under config %s but is being used with %s; refit before any "
            "reported metric depends on it",
            source,
            classifier.config_hash[:12],
            config.hash[:12],
        )
    return classifier
