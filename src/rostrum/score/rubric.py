"""The six-category rubric, its calibration and its confidence band (spec 11).

Four things live here, in the order the pipeline uses them.

**Deviation mass.** A category's score is driven by how much of the speech sat
outside the envelope, not by how many flags were raised:

.. math::

    M_c = \\frac{1}{T_{speech}} \\sum_{j \\in c}
          \\mathrm{dur}_j \\cdot \\overline{(|z| - \\tau_{out})_+}_{\\,j},
    \\qquad S_c = 10\\,e^{-k_c M_c}, \\qquad S = \\tfrac16 \\sum_c S_c

Counting regions would make a detector that splits one flaw into two look twice
as bad; integrating *duration times mean excess deviation* is invariant to that,
which is the property a score needs. The per-region factor is already on the
region as :attr:`~rostrum.types.Region.mean_excess_z`, computed by
:mod:`rostrum.detect.regions` with the :math:`\\tau_{out}` that formed it, so
scoring never re-derives a number the detector already measured. Under a tuned
fold that baseline is the fold's own :math:`\\tau_{out}` rather than the 1.5 in
``configs/scoring.yaml``; :func:`deviation_mass` logs a warning if asked to
assume otherwise.

The exponential maps :math:`[0, \\infty)` onto :math:`(0, 10]` with no clipping,
so a very bad reading degrades smoothly instead of piling up at zero, and a
reading with no flagged region scores exactly 10.

**Calibration.** :math:`k_c` is fixed per fold so that the *median* single-flaw
**L2** file of that category scores 6.0 (spec 11). No search is needed:
:math:`S_c` is a strictly decreasing function of :math:`M_c`, so the median of
the scores is the score of the median mass, and

.. math::

    10 e^{-k_c \\tilde M_c} = 6 \\iff k_c = \\frac{\\ln(10/6)}{\\tilde M_c}

exactly, with :math:`\\tilde M_c` the median mass over those files. The fit uses
training texts only (CLAUDE.md invariant 11) and :func:`calibrate_k` selects
them itself rather than trusting the caller.

**Descriptors.** The four raw numbers of spec 11 - articulation rate, pitch
variation, mean length of runs and pause rate split boundary vs mid-clause -
are reported beside the score because a number a speaker can act on is worth
more than a category total [R21]. They are measurements, never scored.

**Confidence band.** Every member of the envelope is dropped in turn, the
analysis is recomputed, and the score is reported as the mean and standard
deviation across those recomputations (spec 11). A band of :math:`\\pm 1.5`
points is itself the finding: it says the references disagree about this text,
and the dashboard must not print a single confident number over it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.types import CategoryScore, RubricScore

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.detect.zscore import SkippedSignal
    from rostrum.types import FeatureTable, Region

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_ORDER",
    "DEFAULT_EXCESS_BASELINE",
    "DEFAULT_K",
    "DEFAULT_MAX_SCORE",
    "DEFAULT_RUN_PAUSE_S",
    "DEFAULT_TARGET_LEVEL",
    "DEFAULT_TARGET_SCORE",
    "CalibrationCase",
    "build_descriptors",
    "calibrate_k",
    "category_order",
    "category_score",
    "confidence_band",
    "descriptors",
    "deviation_mass",
    "k_from_config",
    "label_to_category",
    "rubric_score",
    "suppressed_from_skipped",
]


CATEGORY_ORDER: tuple[str, ...] = (
    "pace",
    "pausing",
    "intonation",
    "emphasis",
    "energy",
    "clarity",
)
"""Spec 11's category order; ``configs/scoring.yaml`` is the source of truth."""

CATEGORY_LABELS: dict[str, tuple[str, ...]] = {
    "pace": ("rushing", "dragging"),
    "pausing": ("missing_pause", "hesitation_pause"),
    "intonation": ("monotone",),
    "emphasis": ("misplaced_stress",),
    "energy": ("low_energy",),
    "clarity": ("mumbling",),
}
"""Fallback label map for tests that build a score without a config."""

DEFAULT_MAX_SCORE: float = 10.0
DEFAULT_K: float = 1.0
DEFAULT_EXCESS_BASELINE: float = 1.5
"""The :math:`\\tau_{out}` in :math:`(|z| - \\tau_{out})_+`; spec 10.4's default."""

DEFAULT_TARGET_LEVEL: str = "L2"
DEFAULT_TARGET_SCORE: float = 6.0
"""Spec 11: a single-flaw L2 file scores about 6/10 in its own category."""

DEFAULT_RUN_PAUSE_S: float = 0.250
"""A run of speech ends at a pause longer than this (spec 11, [R31])."""

_PAUSE_RATE_KEYS: tuple[str, ...] = ("all", "boundary", "mid_clause")


# --------------------------------------------------------------------------- #
# Category bookkeeping
# --------------------------------------------------------------------------- #


def category_order(config: Config | None = None) -> tuple[str, ...]:
    """The six categories in report order, from ``scoring.category_order``."""
    if config is None:
        return CATEGORY_ORDER
    return tuple(str(c) for c in config.scoring["category_order"])


def label_to_category(config: Config | None = None) -> dict[str, str]:
    """``{flaw label: rubric category}`` from ``scoring.categories``.

    The config is the source of truth; the literal fallback exists only for unit
    tests that score regions without loading one.
    """
    if config is None:
        return {label: category for category, labels in CATEGORY_LABELS.items() for label in labels}
    out: dict[str, str] = {}
    for category, block in config.scoring["categories"].items():
        for label in block["labels"]:
            out[str(label)] = str(category)
    return out


def k_from_config(config: Config | None = None) -> dict[str, float]:
    """The ``k_c`` of ``configs/scoring.yaml``, the pre-calibration starting point."""
    if config is None:
        return dict.fromkeys(CATEGORY_ORDER, DEFAULT_K)
    return {str(c): float(k) for c, k in config.scoring["k_c"].items()}


def suppressed_from_skipped(
    skipped: Sequence[SkippedSignal],
    *,
    config: Config | None = None,
) -> dict[str, str]:
    """Categories that cannot be scored because every one of their labels was skipped.

    ``clarity`` is the case this exists for: its only label is ``mumbling``,
    which needs a same-channel reference (CLAUDE.md invariant 7). Spec 11 and
    ``scoring.score.suppressed_category_policy`` say such a category is
    *excluded from the mean*, never scored 10 - a flaw that could not be looked
    for is not a flaw that was absent.

    Args:
        skipped: The skipped signals from
            :func:`rostrum.detect.zscore.trigger_signals`.
        config: Supplies the label-to-category map.

    Returns:
        ``{category: reason}`` for every category all of whose labels are
        missing from the evaluation.
    """
    mapping = label_to_category(config)
    reasons: dict[str, list[str]] = {}
    for item in skipped:
        category = mapping.get(str(item.label))
        if category is None:
            continue
        reasons.setdefault(category, []).append(f"{item.label}: {item.detail or item.reason}")

    labels_by_category: dict[str, set[str]] = {}
    for label, category in mapping.items():
        labels_by_category.setdefault(category, set()).add(label)

    out: dict[str, str] = {}
    for category, notes in sorted(reasons.items()):
        skipped_labels = {str(i.label) for i in skipped}
        if labels_by_category.get(category, set()) <= skipped_labels:
            out[category] = "; ".join(sorted(notes))
    return out


# --------------------------------------------------------------------------- #
# Deviation mass and the category score
# --------------------------------------------------------------------------- #


def _region_excess(region: Region) -> float:
    """The region's mean excess deviation, falling back to its peak.

    ``mean_excess_z`` is ``NaN`` only when every unit inside the region was
    unmeasurable, which cannot happen for a region the detector emitted; the
    fallback is there so a hand-built or round-tripped region still scores
    rather than poisoning the sum with ``NaN``.
    """
    excess = float(region.mean_excess_z)
    if math.isfinite(excess):
        return max(0.0, excess)
    peak = float(region.peak_z)
    if not math.isfinite(peak):
        return 0.0
    return max(0.0, abs(peak) - DEFAULT_EXCESS_BASELINE)


def deviation_mass(
    regions: Iterable[Region],
    speech_duration: float,
    *,
    category: str | None = None,
    config: Config | None = None,
    excess_baseline: float | None = None,
) -> float:
    """:math:`M_c`: duration-weighted mean excess deviation per second of speech.

    Args:
        regions: Regions to integrate. When ``category`` is given, regions of
            other categories are ignored.
        speech_duration: :math:`T_{speech}` in seconds, excluding leading and
            trailing silence (``scoring.score.t_speech``).
        category: Restrict to one rubric category.
        config: Only used to resolve a region's category when the region itself
            does not carry one.
        excess_baseline: The :math:`\\tau_{out}` the caller believes the regions
            were built with. Passing a value that differs from
            :data:`DEFAULT_EXCESS_BASELINE` logs a warning: the excess is
            measured once, by the detector, and cannot be re-based here without
            the per-unit scores.

    Returns:
        :math:`M_c \\ge 0`; ``0.0`` when no region qualifies. Raises nothing for
        an empty region list - a clean reading is the common case, not an error.

    Raises:
        ValueError: ``speech_duration`` is not positive.
    """
    if float(speech_duration) <= 0.0:
        raise ValueError(f"speech_duration must be positive, got {speech_duration}")
    if excess_baseline is not None and not math.isclose(
        float(excess_baseline), DEFAULT_EXCESS_BASELINE
    ):
        logger.warning(
            "deviation_mass was given excess_baseline=%s, but the excess on each region was "
            "already measured at the tau_out that formed it; the value is recorded, not applied",
            excess_baseline,
        )

    mapping = label_to_category(config) if category is not None else {}
    total = 0.0
    for region in regions:
        if category is not None:
            region_category = str(region.category) or mapping.get(str(region.label), "")
            if region_category != category:
                continue
        total += float(region.duration) * _region_excess(region)
    return total / float(speech_duration)


def category_score(mass: float, k: float, *, max_score: float = DEFAULT_MAX_SCORE) -> float:
    """:math:`S_c = 10 e^{-k_c M_c}`, clamped to :math:`[0, 10]`.

    Raises:
        ValueError: ``mass`` is negative or ``k`` is not positive - either would
            make the score increase with the deviation.
    """
    if float(mass) < 0.0:
        raise ValueError(f"deviation mass must be non-negative, got {mass}")
    if float(k) <= 0.0:
        raise ValueError(f"k must be positive or the score would rise with the deviation, got {k}")
    return float(min(max_score, max(0.0, float(max_score) * math.exp(-float(k) * float(mass)))))


# --------------------------------------------------------------------------- #
# Calibration of k_c
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationCase:
    """One scored training file, as :func:`calibrate_k` needs it.

    Attributes:
        file_id: Recording id.
        text_id: Text id; only the fold's training texts are used.
        category: The rubric category of the file's single injected flaw.
        level: Severity rung of that flaw.
        mass: :math:`M_c` measured for that category on that file.
        n_flaws: Flaws injected into the file. Only single-flaw files calibrate:
            with two flaws the mass of a category is contaminated by the other.
        origin: Provenance; human takes are test-only (CLAUDE.md invariant 11).
    """

    file_id: str
    text_id: str
    category: str
    level: str
    mass: float
    n_flaws: int = 1
    origin: str = "injected"


def calibrate_k(
    train_results: Sequence[CalibrationCase],
    *,
    train_texts: Sequence[str] | None = None,
    config: Config | None = None,
    target_level: str = DEFAULT_TARGET_LEVEL,
    target_score: float = DEFAULT_TARGET_SCORE,
    max_score: float = DEFAULT_MAX_SCORE,
    k_bounds: tuple[float, float] = (0.01, 50.0),
) -> dict[str, float]:
    """Choose :math:`k_c` per category so the median L2 single-flaw file scores 6.0.

    Closed form, not a search. :math:`S_c = 10 e^{-k M}` is strictly decreasing
    in :math:`M`, so the median score is the score of the median mass and

    .. math:: k_c = \\frac{\\ln(S_{max} / S_{target})}{\\tilde M_c}

    is exact. A 1-D root find would return the same number to within its own
    tolerance and hide the fact that the problem is analytic.

    Categories with no usable case keep the configured ``k_c`` and are logged -
    inventing a calibration from nothing would make a score that looks
    calibrated and is not.

    Args:
        train_results: Scored files from this fold's training texts.
        train_texts: Restrict to these texts; ``None`` accepts whatever the
            caller supplied (it has already selected), which is logged.
        config: Supplies the fallback ``k_c``, the target level and score.
        target_level: Rung the calibration targets.
        target_score: Score that rung should receive.
        max_score: :math:`S_{max}`, normally 10.
        k_bounds: Clamp; a near-zero median mass would otherwise send
            :math:`k_c` to infinity on one unlucky file.

    Returns:
        ``{category: k_c}`` for every category of the rubric.

    Raises:
        ValueError: ``target_score`` is not in ``(0, max_score)``, which no
            :math:`k > 0` could ever produce.
    """
    if config is not None:
        calibration = config.scoring["calibration"]
        target_level = str(calibration.get("target_level", target_level))
        target_score = float(calibration.get("target_score", target_score))
        bounds = calibration.get("k_bounds")
        if bounds is not None:
            k_bounds = (float(bounds[0]), float(bounds[1]))
    if not 0.0 < float(target_score) < float(max_score):
        raise ValueError(
            f"target_score must lie strictly between 0 and {max_score}, got {target_score}"
        )

    wanted = {str(t) for t in train_texts} if train_texts is not None else None
    if wanted is None:
        logger.info("calibrate_k: no train_texts given; trusting the caller's selection")

    out = k_from_config(config)
    constant = math.log(float(max_score) / float(target_score))
    for category in category_order(config):
        masses = [
            float(case.mass)
            for case in train_results
            if case.category == category
            and case.level == target_level
            and case.n_flaws == 1
            and case.origin == "injected"
            and (wanted is None or case.text_id in wanted)
            and math.isfinite(case.mass)
            and case.mass > 0.0
        ]
        if not masses:
            logger.warning(
                "calibrate_k: no single-flaw %s file for %s on the training texts; keeping "
                "k=%.4g from the config",
                target_level,
                category,
                out.get(category, DEFAULT_K),
            )
            continue
        median_mass = float(np.median(masses))
        k = constant / median_mass
        clamped = float(min(max(k, k_bounds[0]), k_bounds[1]))
        if not math.isclose(clamped, k):
            logger.warning(
                "calibrate_k: %s wanted k=%.4g from a median mass of %.4g; clamped to %.4g",
                category,
                k,
                median_mass,
                clamped,
            )
        out[category] = clamped
        logger.info(
            "calibrate_k: %s k=%.4g from %d %s file(s), median M=%.4g",
            category,
            clamped,
            len(masses),
            target_level,
            median_mass,
        )
    return out


# --------------------------------------------------------------------------- #
# Raw descriptors
# --------------------------------------------------------------------------- #


def _column(table: FeatureTable, *candidates: str) -> Any | None:
    """The first of ``candidates`` present in ``table``, as a Series."""
    for name in candidates:
        if name in table.columns:
            return table[name]
    return None


def _finite(series: Any) -> np.ndarray:
    """A series' finite values as float64."""
    values = np.asarray(series.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)
    return values[np.isfinite(values)]


def descriptors(
    words: FeatureTable,
    gaps: FeatureTable | None = None,
    *,
    run_pause_s: float = DEFAULT_RUN_PAUSE_S,
) -> dict[str, Any]:
    """The four raw descriptors of spec 11, measured from one recording's tables.

    * ``articulation_rate_syll_s`` - syllables per second of *phonation*: the
      syllable count over the summed word durations, so pauses do not flatter a
      rushed delivery. Without a syllable count the mean of ``rate_local`` is
      used, which is the same quantity measured per word.
    * ``pitch_variation_st`` - the standard deviation across words of the mean
      F0 in semitones. Utterance-level variation is the liveliness correlate of
      [R21]; the within-word ``f0_std_st`` measures something else and is not a
      substitute.
    * ``mean_length_of_runs_syll`` - mean syllables between pauses longer than
      ``run_pause_s``.
    * ``pause_rate_per_min`` - pauses per minute, split ``boundary`` (rhetorical)
      vs ``mid_clause`` (every other gap label), because the same rate means
      opposite things in the two places [R32].

    Args:
        words: Word-unit feature table.
        gaps: Gap-unit table. Without it the run length and pause rate cannot be
            measured and come back as ``None`` rather than as zero.
        run_pause_s: Pause length that ends a run of speech.

    Returns:
        ``{name: value}``, with ``pause_rate_per_min`` a dict over
        ``all``/``boundary``/``mid_clause``. A descriptor that could not be
        measured is ``None``.
    """
    out: dict[str, Any] = {}

    durations = _column(words, "word_duration_s")
    syllables = _column(words, "n_syllables")
    rate = _column(words, "rate_local_syll_per_s")
    phonation = float(_finite(durations).sum()) if durations is not None else 0.0
    if syllables is not None and phonation > 0.0:
        out["articulation_rate_syll_s"] = float(_finite(syllables).sum()) / phonation
    elif rate is not None and _finite(rate).size:
        out["articulation_rate_syll_s"] = float(_finite(rate).mean())
    else:
        out["articulation_rate_syll_s"] = None

    f0 = _column(words, "f0_mean_st")
    values = _finite(f0) if f0 is not None else np.empty(0)
    out["pitch_variation_st"] = float(values.std(ddof=1)) if values.size > 1 else None

    if gaps is None or gaps.empty:
        out["mean_length_of_runs_syll"] = None
        out["pause_rate_per_min"] = dict.fromkeys(_PAUSE_RATE_KEYS)
        return out

    gap_durations = _column(gaps, "gap_duration_s")
    labels = _column(gaps, "gap_label")
    is_pause = (
        np.asarray(gap_durations.to_numpy(dtype=np.float64, na_value=0.0)) > float(run_pause_s)
        if gap_durations is not None
        else np.zeros(len(gaps), dtype=bool)
    )

    # Runs of speech, in syllables, separated by the pauses just identified.
    per_word = (
        _finite(syllables)
        if syllables is not None and _finite(syllables).size == len(words)
        else np.ones(len(words), dtype=np.float64)
    )
    pause_after_word = {
        int(gaps["gap_prev_word_idx"].iloc[i]) if "gap_prev_word_idx" in gaps.columns else int(i)
        for i in range(len(gaps))
        if bool(is_pause[i])
    }
    runs: list[float] = []
    current = 0.0
    for position, word_idx in enumerate(words.index):
        current += float(per_word[position]) if position < per_word.size else 1.0
        if int(word_idx) in pause_after_word:
            runs.append(current)
            current = 0.0
    if current > 0.0:
        runs.append(current)
    out["mean_length_of_runs_syll"] = float(np.mean(runs)) if runs else None

    span_s = float(words["t_end"].max()) - float(words["t_start"].min())
    minutes = span_s / 60.0 if span_s > 0.0 else 0.0
    if minutes <= 0.0:
        out["pause_rate_per_min"] = dict.fromkeys(_PAUSE_RATE_KEYS)
        return out
    label_values = [str(v) for v in labels.tolist()] if labels is not None else ["none"] * len(gaps)
    n_boundary = sum(1 for i, lab in enumerate(label_values) if is_pause[i] and lab == "boundary")
    n_total = int(is_pause.sum())
    out["pause_rate_per_min"] = {
        "all": n_total / minutes,
        "boundary": n_boundary / minutes,
        "mid_clause": (n_total - n_boundary) / minutes,
    }
    return out


def build_descriptors(
    tables: Mapping[str, FeatureTable],
    *,
    reference_tables: Sequence[Mapping[str, FeatureTable]] = (),
    run_pause_s: float = DEFAULT_RUN_PAUSE_S,
) -> dict[str, Any]:
    """The participant's descriptors with the skilled readers' range beside them.

    ``configs/scoring.yaml`` asks for each descriptor ``with_envelope_range``.
    The range is computed from the references' *own* tables rather than from the
    envelope: the envelope holds per-word medians, and the median of a
    per-word quantity is not the descriptor of any actual reading - a range built
    from it would be narrower than the readers really were.

    Args:
        tables: ``{unit: FeatureTable}`` for the participant.
        reference_tables: The same for each envelope member.
        run_pause_s: Pause length that ends a run of speech.

    Returns:
        ``{name: {"value": ..., "reference_low": ..., "reference_median": ...,
        "reference_high": ..., "n_references": n}}``. The reference keys are
        absent when no reference supplied that descriptor.
    """
    mine = descriptors(tables["word"], tables.get("gap"), run_pause_s=run_pause_s)
    theirs = [
        descriptors(block["word"], block.get("gap"), run_pause_s=run_pause_s)
        for block in reference_tables
        if "word" in block
    ]

    def band(extract: Any) -> dict[str, Any]:
        values = [v for v in (extract(d) for d in theirs) if v is not None and math.isfinite(v)]
        if not values:
            return {}
        array = np.asarray(values, dtype=np.float64)
        return {
            "reference_low": float(array.min()),
            "reference_median": float(np.median(array)),
            "reference_high": float(array.max()),
            "n_references": int(array.size),
        }

    out: dict[str, Any] = {}
    for name, value in mine.items():
        if name == "pause_rate_per_min":
            block: dict[str, Any] = {"value": value.get("all") if value else None}
            block.update({key: (value or {}).get(key) for key in ("boundary", "mid_clause")})
            block.update(band(lambda d: (d.get("pause_rate_per_min") or {}).get("all")))
            out[name] = block
            continue
        entry: dict[str, Any] = {"value": value}
        entry.update(band(lambda d, key=name: d.get(key)))
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# The rubric score
# --------------------------------------------------------------------------- #


def rubric_score(
    regions: Sequence[Region],
    speech_duration: float,
    *,
    k_by_category: Mapping[str, float] | None = None,
    config: Config | None = None,
    suppressed: Mapping[str, str] | None = None,
    descriptor_block: Mapping[str, Any] | None = None,
    max_score: float = DEFAULT_MAX_SCORE,
) -> RubricScore:
    """Score all six categories and the overall mean (spec 11).

    Args:
        regions: Every region of the analysis, any category.
        speech_duration: :math:`T_{speech}` in seconds.
        k_by_category: The fold's calibrated ``k_c``; the config's starting
            values are used for any category missing from it.
        config: Supplies the category order, the label map and the fallback
            ``k_c``.
        suppressed: ``{category: reason}`` for categories that could not be
            evaluated - see :func:`suppressed_from_skipped`. They are excluded
            from the overall mean rather than scored 10.
        descriptor_block: Output of :func:`build_descriptors`, attached as-is.
        max_score: :math:`S_{max}`, normally 10.

    Returns:
        A :class:`~rostrum.types.RubricScore`. With no regions every category
        scores exactly ``max_score``.
    """
    k_values = dict(k_from_config(config))
    if k_by_category:
        k_values.update({str(c): float(k) for c, k in k_by_category.items()})
    suppressed = dict(suppressed or {})

    scores: list[CategoryScore] = []
    for category in category_order(config):
        mass = deviation_mass(regions, speech_duration, category=category, config=config)
        k = float(k_values.get(category, DEFAULT_K))
        n_regions = sum(1 for r in regions if str(r.category) == category)
        reason = suppressed.get(category, "")
        scores.append(
            CategoryScore(
                category=category,  # type: ignore[arg-type]
                # A suppressed category carries the score it would have had, so
                # the UI can show it greyed out, but it is excluded from the
                # mean and flagged - never silently counted as a 10.
                score=category_score(mass, k, max_score=max_score),
                deviation_mass=mass,
                k=k,
                n_regions=n_regions,
                suppressed=bool(reason),
                suppressed_reason=reason,
            )
        )

    counted = [c.score for c in scores if not c.suppressed]
    if not counted:
        logger.warning("every rubric category was suppressed; the overall score is undefined")
    return RubricScore(
        categories=scores,
        overall=float(np.mean(counted)) if counted else 0.0,
        descriptors=dict(descriptor_block or {}),
    )


# --------------------------------------------------------------------------- #
# Confidence band
# --------------------------------------------------------------------------- #


def confidence_band(
    base: RubricScore,
    variants: Sequence[RubricScore],
    *,
    replace_score: bool = True,
) -> RubricScore:
    """Mean ± std over leave-one-reference-out recomputations (spec 11).

    Each variant is the same analysis with one envelope member dropped. The
    standard deviation across them is the band, and with ``replace_score`` the
    reported score becomes the mean of the variants - which is what
    ``scoring.confidence.statistic: mean_and_std`` asks for, and is more honest
    than quoting the full-envelope score with someone else's error bar.

    The population standard deviation (``ddof=0``) is used: the recomputations
    are not a sample from a wider population, they are the complete set of
    leave-one-out analyses this envelope admits.

    Args:
        base: The score computed with the full envelope.
        variants: One score per dropped reference. Fewer than two leaves the
            band at zero, and ``base`` is returned unchanged except that the
            zero band is explicit.
        replace_score: Report the variant mean instead of ``base``'s score.

    Returns:
        A new :class:`~rostrum.types.RubricScore`; ``base`` is not modified.
    """
    if len(variants) < 2:
        logger.info(
            "confidence_band: %d recomputation(s) is too few for a band; reporting zero",
            len(variants),
        )
        return RubricScore(
            categories=[CategoryScore(**{**vars(c), "band": 0.0}) for c in base.categories],
            overall=base.overall,
            overall_band=0.0,
            descriptors=dict(base.descriptors),
        )

    categories: list[CategoryScore] = []
    for category in base.categories:
        values = np.asarray(
            [
                v.get(category.category).score
                for v in variants
                if v.get(category.category) is not None
            ],
            dtype=np.float64,
        )
        fields = dict(vars(category))
        fields["band"] = float(values.std(ddof=0)) if values.size > 1 else 0.0
        if replace_score and values.size:
            fields["score"] = float(values.mean())
        categories.append(CategoryScore(**fields))

    overalls = np.asarray([v.overall for v in variants], dtype=np.float64)
    return RubricScore(
        categories=categories,
        overall=float(overalls.mean()) if replace_score else base.overall,
        overall_band=float(overalls.std(ddof=0)),
        descriptors=dict(base.descriptors),
    )
