"""Standardised deviation from the envelope, and the per-label trigger signals.

Two layers live here, and the separation matters for explainability.

**Layer 1, :func:`zscores`** turns a participant's feature table into signed
z-scores against a :class:`~rostrum.envelope.build.TextEnvelope`:

.. math::

    z_{wk} = \\frac{x^{(p)}_{wk} - \\mu_{wk}}{\\hat\\sigma_{wk}}, \\qquad
    D_w = \\sqrt{\\sum_k \\lambda_k z_{wk}^2}

Nothing is judged at this layer; it is pure arithmetic over the envelope, and
every number a card later quotes comes from here.

**Layer 2, :func:`trigger_signals`** applies the spec 10.3 trigger table. Each
label becomes one :class:`Signal`: a *signed score per unit, oriented so that
positive always means "further towards this flaw"*. ``dragging`` fires on
:math:`z_{RATE} < -\\tau`, so its score is :math:`-z_{RATE}`; ``rushing`` fires
on :math:`z_{RATE} > +\\tau`, so its score is :math:`+z_{RATE}`. With that one
convention the whole of region formation (:mod:`rostrum.detect.regions`) becomes
label-independent - enter while ``score > tau_in``, stay while
``score > tau_out`` - and the sign convention is stated once, here, instead of
being rediscovered per label.

The trigger table, verbatim from spec 10.3, and how each row is realised:

======================  ====================================================
Label                   Score and gate
======================  ====================================================
``rushing``             ``+z[rate_local]``
``dragging``            ``-z[rate_local]``
``missing_pause``       ``-z[gap_duration]``, gated to ``gap_label ==
                        "boundary"``
``hesitation_pause``    ``+z[gap_duration]``, gated to ``gap_label == "none"``
                        and an absolute gap of at least 0.25 s [R31, R32]
``monotone``            ``min(-z[phrase range], -z[F0 std])`` per phrase: the
                        ``min`` is how the spec's *and* survives an arbitrary
                        threshold, since ``min(a, b) > tau`` iff both exceed it
``misplaced_stress``    per phrase, Spearman rho between the participant's
                        prominence ranking and the envelope median ranking;
                        below ``rho_min`` the phrase fails and the score is
                        ``|z[prominence]|`` on the ``n`` words with the largest
                        ``|delta prominence|``
``low_energy``          ``-z[energy_rel_z]``
``low_energy`` /fading  ``-z[energy_slope_db_s]``, reported as the ``fading``
                        variant of the same label (the taxonomy has eight
                        labels; spec 11 names the variant in the card)
``mumbling``            weighted mean of the five clarity components, each
                        oriented towards "less clear"; **same channel only**
======================  ====================================================

**The channel rule is enforced here, not downstream** (CLAUDE.md invariant 7).
``trigger_signals`` takes ``same_channel`` explicitly: when it is ``False``
every channel-sensitive feature is withheld, ``mumbling`` is not evaluated at
all, and the reason is returned as a :class:`SkippedSignal` so the dashboard can
say *why* clarity carries no score instead of showing a silent 10/10.

Everything in this module is a pure function of a table and an envelope.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features import names as N

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations

    from rostrum.config import Config
    from rostrum.envelope.build import TextEnvelope, UnitEnvelope, WordIndexMap
    from rostrum.types import FeatureTable, FlawLabel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CLIP_ABS_Z",
    "LABEL_UNITS",
    "Signal",
    "SkippedSignal",
    "ZResult",
    "combined_deviation",
    "envelope_z",
    "signal_z_frame",
    "trigger_signals",
    "zscores",
]


DEFAULT_CLIP_ABS_Z: float = 12.0
"""``|z|`` is clipped here so one pathological unit cannot dominate a score."""

LABEL_UNITS: dict[str, str] = {
    "rushing": "word",
    "dragging": "word",
    "missing_pause": "gap",
    "hesitation_pause": "gap",
    "monotone": "phrase",
    "misplaced_stress": "word",
    "low_energy": "word",
    "mumbling": "word",
}
"""Unit each label's score is indexed on.

``monotone`` is measured per phrase (spec 10.3) and ``misplaced_stress`` is
*detected* per phrase but *localised* to words, which is why it is a word
signal here.
"""


# --------------------------------------------------------------------------- #
# Layer 1: z-scores
# --------------------------------------------------------------------------- #


@dataclass
class ZResult:
    """Signed z-scores of one recording against one unit envelope.

    Attributes:
        text_id: Text compared against.
        fold: LOTO fold of the envelope.
        unit: ``"word"``, ``"gap"`` or ``"phrase"``.
        z: Signed z, indexed by canonical unit index, one column per feature.
        value: The participant's own values, same shape as ``z``.
        median: Envelope medians, same shape.
        sigma: Effective sigma (already floored), same shape.
        n_references: Readers behind each unit, indexed like ``z``.
        table: The participant's full table for this unit, canonically indexed,
            so region formation can read spans and gap labels from one place.
        channel_matched: Whether the comparison was within one channel.
        n_clipped: How many z values hit the clip; reported, not hidden.
        no_scale: Features x units where the effective sigma was zero or
            non-finite, so no z could be formed. Counted, never turned into an
            infinite z.
    """

    text_id: str
    fold: str
    unit: str
    z: FeatureTable
    value: FeatureTable
    median: FeatureTable
    sigma: FeatureTable
    n_references: Any
    table: FeatureTable
    channel_matched: bool = True
    n_clipped: int = 0
    no_scale: int = 0

    @property
    def features(self) -> list[str]:
        """Feature columns, in envelope order."""
        return list(self.z.columns)

    def has(self, *features: str) -> bool:
        """Whether every named feature is present."""
        return all(f in self.z.columns for f in features)

    def series(self, feature: str) -> Any:
        """Signed z for one feature as a Series.

        Raises:
            KeyError: The feature is not in this comparison.
        """
        if feature not in self.z.columns:
            raise KeyError(f"{feature!r} is not in the {self.unit} comparison for {self.text_id}")
        return self.z[feature]

    def span(self, unit_idx: int) -> tuple[float, float]:
        """``(t_start, t_end)`` of one unit, from the participant's own table."""
        row = self.table.loc[int(unit_idx)]
        return float(row["t_start"]), float(row["t_end"])


def _effective_sigma_frame(unit_envelope: UnitEnvelope, index: Sequence[int]) -> FeatureTable:
    """Build the floored-sigma frame for the requested units."""
    import pandas as pd

    inner = unit_envelope.envelope
    features = list(inner.features)
    floors = np.array([float(inner.sigma_floor.get(f, 0.0)) for f in features], dtype=np.float64)
    rows = []
    for idx in index:
        raw = inner.sigma.get(int(idx))
        if raw is None:
            rows.append(np.full(len(features), np.nan))
            continue
        rows.append(np.maximum(np.asarray(raw, dtype=np.float64), floors))
    return pd.DataFrame(
        np.asarray(rows, dtype=np.float64),
        index=pd.Index([int(i) for i in index], name="unit_idx", dtype="int64"),
        columns=features,
    )


def _median_frame(unit_envelope: UnitEnvelope, index: Sequence[int]) -> FeatureTable:
    """Build the median frame for the requested units."""
    import pandas as pd

    inner = unit_envelope.envelope
    features = list(inner.features)
    rows = [
        np.asarray(inner.median.get(int(i), [np.nan] * len(features)), dtype=np.float64)
        for i in index
    ]
    return pd.DataFrame(
        np.asarray(rows, dtype=np.float64),
        index=pd.Index([int(i) for i in index], name="unit_idx", dtype="int64"),
        columns=features,
    )


def zscores(
    table: FeatureTable,
    unit_envelope: UnitEnvelope,
    *,
    text_id: str = "",
    index_map: WordIndexMap | None = None,
    clip_abs_z: float | None = None,
    config: Config | None = None,
    channel_matched: bool = True,
) -> ZResult:
    """Standardise one recording's units against the envelope (spec 10.3).

    Only the units the envelope knows about are compared, and only the features
    it carries: a participant column with no envelope counterpart has no scale
    and is therefore not a deviation, it is just a number.

    Where the effective sigma is zero - every reader produced the identical
    value *and* the fitted floor for that feature is also zero - the z is
    ``NaN``, not infinity. A zero scale means the envelope has no information
    about acceptable variation there, which is a different statement from "this
    participant is infinitely wrong", and letting an infinity through would
    propagate into every score downstream.

    Args:
        table: The participant's per-unit table.
        unit_envelope: The envelope for the same unit kind, floors applied.
        text_id: Text id for the result; defaults to the envelope's.
        index_map: Canonical-index override for this recording, if any.
        clip_abs_z: Clip ``|z|`` here. ``None`` reads
            ``detection.deviation.clip_abs_z``, else :data:`DEFAULT_CLIP_ABS_Z`.
        config: Configuration, for ``clip_abs_z``.
        channel_matched: Whether this comparison is within one channel. Recorded
            on the result and on every :class:`~rostrum.types.Evidence` row
            built from it.

    Returns:
        A :class:`ZResult`.

    Raises:
        ValueError: ``table`` and ``unit_envelope`` describe different units.
    """
    import pandas as pd

    unit = str(table.attrs.get("unit", unit_envelope.unit))
    if unit != unit_envelope.unit:
        raise ValueError(
            f"table is a {unit!r} table but the envelope is for {unit_envelope.unit!r}"
        )

    if clip_abs_z is None:
        clip_abs_z = (
            float(config.detection["deviation"]["clip_abs_z"])
            if config is not None
            else DEFAULT_CLIP_ABS_Z
        )

    participant = table if index_map is None else index_map.apply(table)
    index = [i for i in unit_envelope.unit_indices if i in set(participant.index)]
    features = [f for f in unit_envelope.features if f in participant.columns]
    missing = [f for f in unit_envelope.features if f not in participant.columns]
    if missing:
        logger.debug(
            "%s/%s: %d envelope feature(s) absent from the participant table: %s",
            text_id or unit_envelope.envelope.text_id,
            unit,
            len(missing),
            missing,
        )

    value = participant.reindex(index=index, columns=features).astype("float64")
    median = _median_frame(unit_envelope, index)[features]
    sigma = _effective_sigma_frame(unit_envelope, index)[features]

    scale = sigma.to_numpy(dtype=np.float64, na_value=np.nan)
    usable = np.isfinite(scale) & (scale > 0.0)
    no_scale = int((~usable & np.isfinite(value.to_numpy(dtype=np.float64, na_value=np.nan))).sum())

    raw = np.where(
        usable,
        (
            value.to_numpy(dtype=np.float64, na_value=np.nan)
            - median.to_numpy(dtype=np.float64, na_value=np.nan)
        )
        / np.where(usable, scale, 1.0),
        np.nan,
    )
    n_clipped = int((np.isfinite(raw) & (np.abs(raw) > clip_abs_z)).sum())
    clipped = np.clip(raw, -clip_abs_z, clip_abs_z)

    z = pd.DataFrame(clipped, index=value.index, columns=features)
    n_references = pd.Series(
        [unit_envelope.n_at(i) for i in index],
        index=value.index,
        name="n_references",
        dtype="int64",
    )

    if n_clipped:
        logger.debug("%s/%s: %d z value(s) clipped at +/-%g", text_id, unit, n_clipped, clip_abs_z)
    if no_scale:
        logger.debug(
            "%s/%s: %d measurement(s) had no usable scale (sigma = 0 and no floor); z is NaN",
            text_id,
            unit,
            no_scale,
        )

    return ZResult(
        text_id=text_id or unit_envelope.envelope.text_id,
        fold=unit_envelope.envelope.fold,
        unit=unit,
        z=z,
        value=value,
        median=median,
        sigma=sigma,
        n_references=n_references,
        table=participant,
        channel_matched=channel_matched,
        n_clipped=n_clipped,
        no_scale=no_scale,
    )


def combined_deviation(result: ZResult, config: Config | None = None) -> Any:
    """Return :math:`D_w = \\sqrt{\\sum_k \\lambda_k z_{wk}^2}` per unit.

    The weights come from ``detection.deviation.lambda``, keyed by canonical
    feature name; a feature with no weight does not enter ``D``. Units where no
    weighted feature has a finite z get ``NaN`` rather than 0, so "nothing was
    measurable here" is distinguishable from "this unit was exactly typical".

    Args:
        result: A z-score comparison.
        config: Supplies the lambda weights; ``None`` weights every feature 1.

    Returns:
        A Series indexed like ``result.z``.
    """
    import pandas as pd

    if config is None:
        weights = dict.fromkeys(result.features, 1.0)
    else:
        configured = dict(config.detection["deviation"]["lambda"])
        weights = {}
        for name, weight in configured.items():
            for column in N.resolve(str(name), result.features):
                weights[column] = float(weight)

    columns = [c for c in result.features if c in weights]
    if not columns:
        return pd.Series(np.nan, index=result.z.index, name="deviation")

    z = result.z[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    lam = np.array([weights[c] for c in columns], dtype=np.float64)
    finite = np.isfinite(z)
    total = np.where(finite, lam * np.square(np.where(finite, z, 0.0)), 0.0).sum(axis=1)
    any_finite = finite.any(axis=1)
    return pd.Series(
        np.where(any_finite, np.sqrt(total), np.nan), index=result.z.index, name="deviation"
    )


# --------------------------------------------------------------------------- #
# Layer 2: trigger signals
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Signal:
    """One label's signed score per unit, oriented so positive means "worse".

    Attributes:
        label: The flaw label.
        variant: ``""``, or a named variant of the same label (``"fading"``).
        unit: Unit the score is indexed on.
        primary_feature: Canonical name of the feature the score is built from;
            drives boundary refinement (spec 10.5) and the card's headline.
        score: Signed score per unit index. ``NaN`` where nothing is measurable.
        gate: Boolean per unit index; ``False`` means the rule may not fire
            there whatever the score (e.g. a gap whose text expects no pause).
        evidence_features: Columns to consider for the region's evidence rows,
            most important first.
        channel_matched: Whether this signal came from a same-channel
            comparison.
        note: Free text for the card, e.g. which phrases failed the rank test.
    """

    label: FlawLabel
    unit: str
    primary_feature: str
    score: Any
    gate: Any
    variant: str = ""
    evidence_features: tuple[str, ...] = ()
    channel_matched: bool = True
    note: str = ""

    @property
    def key(self) -> str:
        """``label`` or ``label:variant``; the merge key in region formation."""
        return f"{self.label}:{self.variant}" if self.variant else str(self.label)


@dataclass(frozen=True)
class SkippedSignal:
    """A label that could not be evaluated, with the reason the UI should show.

    Attributes:
        label: The flaw label.
        reason: Short machine-readable reason, e.g. ``"no_same_channel_reference"``.
        detail: Sentence for the dashboard.
    """

    label: FlawLabel
    reason: str
    detail: str = ""


def _directed(series: Any, direction: str) -> Any:
    """Orient a z series so that positive means "further towards the flaw".

    ``"above"`` keeps the sign, ``"below"`` and ``"steeper"`` flip it. "Steeper"
    is the spectral-tilt case: a steeper tilt is a *more negative* slope, so the
    flaw direction is the negative one.
    """
    if direction == "above":
        return series
    if direction in {"below", "steeper"}:
        return -series
    raise ValueError(f"unknown rule direction {direction!r}")


def _true_like(frame: Any) -> Any:
    """An all-``True`` boolean Series on ``frame``'s index."""
    import pandas as pd

    return pd.Series(True, index=frame.index, dtype=bool)


def _rate_signals(word: ZResult, rules: Mapping[str, Any]) -> list[Signal]:
    """``rushing`` and ``dragging``: the two signs of the rate deviation."""
    out: list[Signal] = []
    for label in ("rushing", "dragging"):
        rule = rules[label]
        columns = N.resolve(str(rule["feature"]), word.features)
        if not columns:
            continue
        z = word.series(columns[0])
        out.append(
            Signal(
                label=label,  # type: ignore[arg-type]
                unit="word",
                primary_feature=str(rule["feature"]),
                score=_directed(z, str(rule["direction"])),
                gate=_true_like(z),
                evidence_features=tuple(
                    c for c in (columns[0], *N.resolve("duration_per_syllable", word.features)) if c
                ),
                channel_matched=word.channel_matched,
            )
        )
    return out


def _pause_signals(gap: ZResult, rules: Mapping[str, Any]) -> list[Signal]:
    """``missing_pause`` and ``hesitation_pause``, gated by the text expectation.

    The gate is the whole point of the pair: a 0.6 s silence after a full stop
    is skilled rhetoric and a 0.6 s silence mid-clause is a hesitation, and the
    acoustics are identical. Only the text tells them apart (spec 10.3,
    [R31, R32]).
    """
    out: list[Signal] = []
    columns = N.resolve("gap_duration", gap.features)
    if not columns:
        return out
    column = columns[0]
    z = gap.series(column)
    labels = gap.table["gap_label"].astype("string") if "gap_label" in gap.table.columns else None
    duration_column = "gap_duration_s"

    for label in ("missing_pause", "hesitation_pause"):
        rule = rules[label]
        gate = _true_like(z)
        wanted = [str(x) for x in rule.get("require_gap_label", ())]
        if wanted:
            if labels is None:
                logger.warning(
                    "gap table has no gap_label column; %s cannot be gated and is skipped", label
                )
                continue
            gate &= labels.reindex(z.index).isin(wanted).fillna(False)
        minimum = rule.get("min_absolute_gap_s")
        if minimum is not None and duration_column in gap.table.columns:
            absolute = gap.table[duration_column].reindex(z.index).astype("float64")
            gate &= absolute >= float(minimum)
        out.append(
            Signal(
                label=label,  # type: ignore[arg-type]
                unit="gap",
                primary_feature="gap_duration",
                score=_directed(z, str(rule["direction"])),
                gate=gate,
                evidence_features=(column,),
                channel_matched=gap.channel_matched,
            )
        )
    return out


def _monotone_signal(
    phrase: ZResult | None, word: ZResult, rules: Mapping[str, Any]
) -> Signal | SkippedSignal:
    """``monotone``: phrase pitch range **and** phrase F0 variation both low.

    Spec 10.3 requires both conditions, so the score is the ``min`` of the two
    oriented z values: ``min(a, b) > tau`` holds exactly when both do, which
    keeps the *and* intact under any threshold the tuner picks rather than
    hard-coding one.

    The phrase table carries ``phrase_f0_range_st`` but no phrase-level F0
    standard deviation, so the variation term is the mean of the per-word
    ``f0_std_st`` z over the phrase's own words. That is a phrase-level
    statement built from word-level measurements, and it is the only place in
    the detector where a unit's score aggregates another unit's z - stated here
    so the report can say so.
    """
    rule = rules["monotone"]
    if phrase is None:
        return SkippedSignal(
            label="monotone",
            reason="no_phrase_envelope",
            detail="This text has no phrase-level envelope, so pitch variety is not scored.",
        )

    range_columns = [
        c
        for c in (
            *N.resolve("phrase_f0_range_st", phrase.features),
            *N.resolve(str(rule["feature"]), phrase.features),
        )
        if c
    ]
    if not range_columns:
        return SkippedSignal(
            label="monotone",
            reason="no_phrase_pitch_range",
            detail="The phrase envelope carries no pitch range, so pitch variety is not scored.",
        )
    range_z = _directed(phrase.series(range_columns[0]), str(rule["direction"]))

    also = dict(rule.get("also_require", {}))
    std_columns = N.resolve(str(also.get("feature", "f0_std_st")), word.features)
    if not std_columns or not {"first_word_idx", "last_word_idx"} <= set(phrase.table.columns):
        logger.debug("monotone: no per-word F0 std available; using the pitch range alone")
        score = range_z
        note = "pitch range only; no F0 variation available"
    else:
        std_z = _directed(word.series(std_columns[0]), str(also.get("direction", "below")))
        per_phrase = []
        for idx in phrase.z.index:
            row = phrase.table.loc[int(idx)]
            lo, hi = int(row["first_word_idx"]), int(row["last_word_idx"])
            inside = std_z.loc[(std_z.index >= lo) & (std_z.index <= hi)]
            per_phrase.append(float(inside.mean()) if len(inside) else np.nan)
        import pandas as pd

        std_phrase = pd.Series(per_phrase, index=phrase.z.index, dtype="float64")
        # NaN-tolerant min: a phrase where only one term is measurable is
        # judged on that term rather than dropped entirely.
        score = range_z.combine(std_phrase, lambda a, b: np.nanmin([a, b]))
        note = "pitch range and F0 variation, both relative to skilled readings"

    return Signal(
        label="monotone",
        unit="phrase",
        primary_feature=str(rule["feature"]),
        score=score,
        gate=_true_like(range_z),
        evidence_features=tuple(range_columns[:1]),
        channel_matched=phrase.channel_matched,
        note=note,
    )


def _stress_signal(
    word: ZResult, phrase: ZResult | None, rules: Mapping[str, Any]
) -> Signal | SkippedSignal:
    """``misplaced_stress``: the prominence *ranking* inside a phrase is wrong.

    Emphasis is relational. A speaker who is uniformly 20% more prominent than
    the references has not misplaced any stress, and a per-word z on prominence
    would flag every word of that phrase. So the test is a Spearman rank
    correlation between the participant's prominence ordering within a phrase
    and the envelope median ordering (spec 10.3). Only when the ordering itself
    disagrees - ``rho < rho_min`` - does the phrase fail, and the region is then
    localised to the ``n_localise_words`` words whose prominence moved furthest,
    because that is where a listener hears the misplacement.
    """
    from scipy import stats

    rule = rules["misplaced_stress"]
    columns = N.resolve(str(rule["feature"]), word.features)
    if not columns:
        return SkippedSignal(
            label="misplaced_stress",
            reason="no_prominence",
            detail="Prominence was not computed, so emphasis placement is not scored.",
        )
    if phrase is None or not {"first_word_idx", "last_word_idx"} <= set(phrase.table.columns):
        return SkippedSignal(
            label="misplaced_stress",
            reason="no_phrases",
            detail="Emphasis placement is judged within a phrase; this text has no phrases.",
        )

    import pandas as pd

    column = columns[0]
    rho_min = float(rule["rho_min"])
    n_localise = int(rule.get("n_localise_words", 2))
    z = word.series(column)
    observed = word.value[column]
    expected = word.median[column]

    score = pd.Series(np.nan, index=z.index, dtype="float64")
    gate = pd.Series(False, index=z.index, dtype=bool)
    failed: list[str] = []

    for idx in phrase.table.index:
        row = phrase.table.loc[int(idx)]
        lo, hi = int(row["first_word_idx"]), int(row["last_word_idx"])
        members = [i for i in z.index if lo <= i <= hi]
        pair = [
            i
            for i in members
            if np.isfinite(float(observed.get(i, np.nan)))
            and np.isfinite(float(expected.get(i, np.nan)))
        ]
        # Spearman on fewer than three points is either undefined or can only
        # return +/-1, so such a phrase is left unjudged rather than guessed at.
        if len(pair) < 3:
            continue
        rho = stats.spearmanr(
            [float(observed[i]) for i in pair], [float(expected[i]) for i in pair]
        ).statistic
        if not np.isfinite(rho) or rho >= rho_min:
            continue
        failed.append(f"phrase {int(idx)} (rho={rho:.2f})")
        deltas = sorted(pair, key=lambda i: -abs(float(z.get(i, 0.0) or 0.0)))
        for i in deltas[: max(1, n_localise)]:
            gate.loc[i] = True
            score.loc[i] = abs(float(z.loc[i]))

    return Signal(
        label="misplaced_stress",
        unit="word",
        primary_feature=str(rule["feature"]),
        score=score,
        gate=gate,
        evidence_features=(column,),
        channel_matched=word.channel_matched,
        note=("; ".join(failed) if failed else "prominence ranking matched the envelope"),
    )


def _energy_signals(word: ZResult, rules: Mapping[str, Any]) -> list[Signal]:
    """``low_energy`` and its ``fading`` variant (a falling slope, not a low level)."""
    out: list[Signal] = []
    rule = rules["low_energy"]
    level = N.resolve(str(rule["feature"]), word.features)
    if level:
        z = word.series(level[0])
        out.append(
            Signal(
                label="low_energy",
                unit="word",
                primary_feature=str(rule["feature"]),
                score=_directed(z, str(rule["direction"])),
                gate=_true_like(z),
                evidence_features=tuple(level[:1]),
                channel_matched=word.channel_matched,
            )
        )
    variant = dict(rule.get("variant_fading", {}))
    slope = N.resolve(str(variant.get("feature", "energy_slope_db_s")), word.features)
    if variant and slope:
        z = word.series(slope[0])
        out.append(
            Signal(
                label="low_energy",
                unit="word",
                primary_feature=str(variant["feature"]),
                score=_directed(z, str(variant.get("direction", "below"))),
                gate=_true_like(z),
                variant=str(variant.get("label_suffix", "fading")),
                evidence_features=tuple(slope[:1]),
                channel_matched=word.channel_matched,
                note="falling energy within the word, not a low level",
            )
        )
    return out


def _mumbling_signal(
    word: ZResult, rules: Mapping[str, Any], *, same_channel: bool
) -> Signal | SkippedSignal:
    """``mumbling``: the five-component clarity composite, same channel only.

    Clarity rests on spectral tilt, cepstral peak prominence and the rest of the
    channel-sensitive block, and those measure the microphone as much as the
    mouth. Comparing a participant's tilt against an envelope built from other
    people's recordings would report the recording chain, not the delivery, so
    the rule is simply not evaluated unless the envelope is same-channel
    (CLAUDE.md invariant 7). The caller is told why (:class:`SkippedSignal`)
    rather than being handed a silent zero.
    """
    import pandas as pd

    rule = rules["mumbling"]
    if not same_channel:
        return SkippedSignal(
            label="mumbling",
            reason="no_same_channel_reference",
            detail=(
                "Clarity is measured from spectral and voice-quality features, which are "
                "comparable only within one recording channel. No same-channel reference "
                "was available, so clarity is reported but never flagged."
            ),
        )

    components = dict(rule["composite_features"])
    weights = [float(w) for w in rule.get("composite_weights", ())]
    if len(weights) != len(components):
        weights = [1.0] * len(components)

    parts: list[Any] = []
    used: list[float] = []
    names: list[str] = []
    for (name, direction), weight in zip(components.items(), weights, strict=True):
        resolved = N.resolve(str(name), word.features)
        if not resolved:
            logger.debug("mumbling: component %r is not in the envelope; dropped", name)
            continue
        parts.append(_directed(word.series(resolved[0]), str(direction)))
        used.append(float(weight))
        names.append(resolved[0])

    if len(parts) < 2:
        return SkippedSignal(
            label="mumbling",
            reason="too_few_clarity_components",
            detail=(
                f"Only {len(parts)} of {len(components)} clarity components were available; "
                "clarity is not scored."
            ),
        )

    stack = np.stack([p.to_numpy(dtype=np.float64, na_value=np.nan) for p in parts], axis=1)
    lam = np.asarray(used, dtype=np.float64)
    finite = np.isfinite(stack)
    denom = np.where(finite, lam, 0.0).sum(axis=1)
    numer = np.where(finite, lam * np.where(finite, stack, 0.0), 0.0).sum(axis=1)
    composite = np.where(denom > 0, numer / np.where(denom > 0, denom, 1.0), np.nan)
    score = pd.Series(composite, index=word.z.index, dtype="float64")

    return Signal(
        label="mumbling",
        unit="word",
        primary_feature="clarity_composite",
        score=score,
        gate=_true_like(score),
        evidence_features=tuple(names),
        channel_matched=True,
        note=f"weighted mean of {len(parts)} clarity components ({', '.join(names)})",
    )


def trigger_signals(
    z_by_unit: Mapping[str, ZResult],
    *,
    config: Config,
    same_channel: bool = False,
    labels: Sequence[str] | None = None,
) -> tuple[list[Signal], list[SkippedSignal]]:
    """Build every label's signed score from the z-scores (spec 10.3).

    Args:
        z_by_unit: ``{unit: ZResult}``; ``"word"`` is required, ``"gap"`` and
            ``"phrase"`` enable the pause and phrase rules.
        config: Supplies ``detection.rules``.
        same_channel: Whether the envelope and the participant share a channel.
            Gates every channel-sensitive rule (CLAUDE.md invariant 7).
        labels: Restrict to these labels; ``None`` evaluates all eight.

    Returns:
        ``(signals, skipped)``. ``signals`` may hold two entries for one label
        when it has a variant (``low_energy`` and ``low_energy``/``fading``).

    Raises:
        KeyError: ``z_by_unit`` has no ``"word"`` entry.
    """
    if "word" not in z_by_unit:
        raise KeyError("trigger_signals needs at least the word-level comparison")

    rules = dict(config.detection["rules"])
    word = z_by_unit["word"]
    gap = z_by_unit.get("gap")
    phrase = z_by_unit.get("phrase")
    wanted = set(labels) if labels is not None else set(LABEL_UNITS)

    signals: list[Signal] = []
    skipped: list[SkippedSignal] = []

    def collect(item: Signal | SkippedSignal) -> None:
        (signals if isinstance(item, Signal) else skipped).append(item)  # type: ignore[arg-type]

    if {"rushing", "dragging"} & wanted:
        signals += [s for s in _rate_signals(word, rules) if s.label in wanted]

    if {"missing_pause", "hesitation_pause"} & wanted:
        if gap is None:
            for label in sorted({"missing_pause", "hesitation_pause"} & wanted):
                skipped.append(
                    SkippedSignal(
                        label=label,  # type: ignore[arg-type]
                        reason="no_gap_envelope",
                        detail="This text has no gap-level envelope, so pausing is not scored.",
                    )
                )
        else:
            signals += [s for s in _pause_signals(gap, rules) if s.label in wanted]

    if "monotone" in wanted:
        collect(_monotone_signal(phrase, word, rules))

    if "misplaced_stress" in wanted:
        collect(_stress_signal(word, phrase, rules))

    if "low_energy" in wanted:
        signals += _energy_signals(word, rules)

    if "mumbling" in wanted:
        collect(_mumbling_signal(word, rules, same_channel=same_channel))

    logger.debug(
        "trigger signals: %d built (%s), %d skipped (%s)",
        len(signals),
        ", ".join(s.key for s in signals),
        len(skipped),
        ", ".join(f"{s.label}:{s.reason}" for s in skipped),
    )
    return signals, skipped


def signal_z_frame(signal: Signal, result: ZResult) -> FeatureTable:
    """Return the z columns relevant to ``signal``, for evidence selection.

    Falls back to every feature of the comparison when the signal names none,
    so the top-3-by-``|z|`` rule in :mod:`rostrum.detect.regions` still has
    something to rank.
    """
    columns = [c for c in signal.evidence_features if c in result.z.columns]
    return result.z[columns] if columns else result.z


def envelope_z(
    tables: Mapping[str, FeatureTable],
    envelope: TextEnvelope,
    *,
    index_maps: Mapping[str, WordIndexMap] | None = None,
    config: Config | None = None,
    channel_matched: bool | None = None,
    file_id: str = "",
) -> dict[str, ZResult]:
    """Standardise every unit of one recording against a text envelope.

    Args:
        tables: ``{unit: participant table}``.
        envelope: The text's envelope, floors applied.
        index_maps: ``{unit: WordIndexMap}`` for this recording.
        config: Configuration.
        channel_matched: Override the channel decision. ``None`` compares the
            participant's own ``channel_id`` against the envelope's references.
        file_id: Recording id, for log messages.

    Returns:
        ``{unit: ZResult}`` for every unit present in both inputs.
    """
    maps = index_maps or {}
    out: dict[str, ZResult] = {}
    for unit, table in tables.items():
        if unit not in envelope.units:
            logger.debug("%s: envelope has no %s unit; skipped", file_id or "participant", unit)
            continue
        if channel_matched is None:
            channel = str(table.attrs.get("channel_id", ""))
            matched = bool(channel) and envelope.same_channel and channel in envelope.channel_ids
        else:
            matched = bool(channel_matched)
        out[unit] = zscores(
            table,
            envelope.unit(unit),
            text_id=envelope.text_id,
            index_map=maps.get(unit),
            config=config,
            channel_matched=matched,
        )
    return out
