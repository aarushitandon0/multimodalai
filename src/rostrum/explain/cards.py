"""Explanation cards: evidence to sentences to JSON and Markdown (spec 12).

A flag is not feedback. Spec 12 asks every region to state four things - the
error, where it is, how big it is and what to do instead - and [R39] is the
reason: feedback that names the fix is the feedback a speaker can act on.

This module is the last step of that chain and the only one that writes prose:

    features -> z -> regions (+ evidence rows) -> **card** -> dashboard / PDF

Three guarantees hold here.

1. **No LLM** (CLAUDE.md invariant 12). Every sentence is a template from
   ``configs/templates.yaml`` filled with numbers DSP computed. The optional
   ``--polish`` rephrasing of spec 12 is not implemented; when it is, it may
   only rewrite an already-rendered string.
2. **No new measurement.** A card reads :class:`~rostrum.types.Region`,
   its :class:`~rostrum.types.Evidence` rows, the text annotation and the
   envelope. It never computes a feature, so nothing it says can disagree with
   what was detected.
3. **Strict rendering.** A placeholder with no value raises
   :class:`~rostrum.explain.templates.TemplateError`, so a half-rendered
   sentence cannot reach a user.

The JSON form is :class:`Card`, a pydantic model carrying the spec 12 schema
in its field order, followed by a few additions the dashboard needs
(``headline``, ``duration``, ``variant``, ``uncertain``, ``confidence_basis``,
``context_detail``). A reader of the spec's example will find every key it
shows, with the same meaning.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rostrum.detect.classifier import UNCERTAIN_TAG
from rostrum.explain.context import locate, region_context
from rostrum.explain.templates import Templates
from rostrum.features import names as N
from rostrum.types import TIME_DECIMALS, Evidence, Region, round_time, round_value

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.detect.zscore import ZResult
    from rostrum.envelope.build import TextEnvelope
    from rostrum.text.rhetoric import TextAnnotation

logger = logging.getLogger(__name__)

__all__ = [
    "Card",
    "EvidenceRow",
    "build_card",
    "build_cards",
    "detection_confidence",
    "region_fields",
    "render_markdown",
]

_VARIANTS: frozenset[str] = frozenset({"fading"})
"""Signal variants that :mod:`rostrum.detect.regions` parks in ``Region.context``."""

ConfidenceBasis = Literal["empirical_exceedance", "detector", "unavailable"]
"""How :attr:`Card.confidence` was arrived at; see :func:`detection_confidence`."""


# --------------------------------------------------------------------------- #
# The JSON schema (spec 12)
# --------------------------------------------------------------------------- #


class EvidenceRow(BaseModel):
    """One measured feature behind a card, as spec 12 serialises it."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float
    env_median: float
    env_sigma: float
    z: float
    unit: str = ""
    gap_label: str | None = None
    n_references: int = 0
    channel_matched: bool = True

    @classmethod
    def from_evidence(cls, evidence: Evidence) -> EvidenceRow:
        """Build from a :class:`~rostrum.types.Evidence`, rounding as types.py does."""
        return cls(**evidence.to_dict())


class Card(BaseModel):
    """The explanation of one region (spec 12).

    The first block of fields is the spec's schema, in its order. The second
    block is what the dashboard and the PDF need on top of it and is additive:
    no spec field changes meaning.

    Attributes:
        region_id: Index of the region in the analysis.
        t_start: Onset in seconds from the start of the processed WAV.
        t_end: Offset in seconds.
        words: The words the region covers, from the transcript.
        category: Rubric category.
        label: The rule's label - what the card explains.
        level: Severity rung, ``L0`` to ``L4``.
        confidence: How certain the flag is; see :func:`detection_confidence`.
        rule_label: The label the rule produced.
        clf_prob: The classifier's probability for its own label, if it ran.
        evidence: The measurements, strongest first.
        context: Where this sits in the text and what the readers did there.
        rationale: The deviation stated in words and numbers.
        fix: What to do instead, with the target value.
        envelope_n: Skilled readers behind the comparison.
        mode: ``reference``, ``norm`` or ``reject``.
        headline: One line for the top of the card.
        duration: ``t_end - t_start``.
        variant: Named variant of the label (``fading``), or empty.
        clf_label: The classifier's label, when it ran.
        uncertain: Whether rule and classifier disagree at high probability.
        confidence_basis: How ``confidence`` was computed.
        context_detail: The facts the context sentence was built from.
        s: Continuous severity behind ``level``.
    """

    model_config = ConfigDict(extra="forbid")

    # -- spec 12 schema ----------------------------------------------------
    region_id: int
    t_start: float
    t_end: float
    words: str = ""
    category: str = ""
    label: str = ""
    level: str = "L0"
    confidence: float = 0.0
    rule_label: str | None = None
    clf_prob: float | None = None
    evidence: list[EvidenceRow] = Field(default_factory=list)
    context: str = ""
    rationale: str = ""
    fix: str = ""
    envelope_n: int = 0
    mode: str = "reference"

    # -- additions ---------------------------------------------------------
    headline: str = ""
    duration: float = 0.0
    variant: str = ""
    clf_label: str | None = None
    uncertain: bool = False
    confidence_basis: ConfidenceBasis = "unavailable"
    context_detail: dict[str, Any] = Field(default_factory=dict)
    s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON form, field order preserved."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


def detection_confidence(
    region: Region,
    *,
    feature: str,
    envelope: TextEnvelope | None = None,
    z_by_unit: Mapping[str, ZResult] | None = None,
) -> tuple[float, ConfidenceBasis, int, int]:
    """How certain the flag is, as a fraction of the skilled readers.

    The statistic is the **empirical exceedance**: of the readers who measured
    this unit, how many does the participant lie beyond, in the direction the
    rule fired? With add-one smoothing,

    .. math:: c = \\frac{k + 0.5}{n + 1}

    so a participant outside every one of 19 readers scores 0.95 and the same
    reading against 4 readers scores 0.90 - a thin envelope cannot produce a
    confident claim. The quantity is non-parametric, bounded, and means exactly
    one thing a speaker can check ("further out than *k* of *n* readers").

    It deliberately does **not** grow with the size of the deviation: how far
    outside skilled practice a reading is belongs to the severity
    (:attr:`~rostrum.types.Region.level`), not to the certainty that it is
    outside at all.

    Args:
        region: The region.
        feature: Canonical name of its primary feature.
        envelope: The text envelope, carrying the raw reader values.
        z_by_unit: The comparisons detection ran on.

    Returns:
        ``(confidence, basis, k, n)``. Without an envelope the region's own
        confidence is passed through with basis ``"detector"``, or zero with
        basis ``"unavailable"``.
    """
    fallback: tuple[float, ConfidenceBasis, int, int] = (
        (float(region.confidence), "detector", 0, 0)
        if float(region.confidence) > 0.0
        else (0.0, "unavailable", 0, 0)
    )
    if envelope is None:
        return fallback
    location = locate(region, feature=feature, envelope=envelope, z_by_unit=z_by_unit)
    if location is None or location.unit not in envelope.units:
        return fallback
    unit_envelope = envelope.unit(location.unit)
    if not unit_envelope.has(location.column) or location.unit_idx not in unit_envelope.values:
        return fallback

    values = unit_envelope.reference_values(location.unit_idx, location.column)
    measured = values[np.isfinite(values)]
    if measured.size == 0:
        return fallback

    row = next((e for e in region.evidence if e.feature == N.canonical(location.column)), None)
    if row is None:
        return fallback
    beyond = (
        int((measured < float(row.value)).sum())
        if float(row.z) >= 0
        else int((measured > float(row.value)).sum())
    )
    confidence = (beyond + 0.5) / (measured.size + 1.0)
    return float(confidence), "empirical_exceedance", beyond, int(measured.size)


# --------------------------------------------------------------------------- #
# Fields
# --------------------------------------------------------------------------- #


def _variant_of(region: Region) -> str:
    """The signal variant parked in ``Region.context`` by region formation."""
    return next((tag for tag in str(region.context).split(";") if tag in _VARIANTS), "")


def _is_uncertain(region: Region) -> bool:
    """Whether the classifier disagreed with the rule at high probability."""
    return UNCERTAIN_TAG in str(region.context).split(";")


def _evidence_fields(row: Evidence, templates: Templates) -> dict[str, str]:
    """Formatted fields of one evidence row."""
    fmt = templates.formatting
    unit = row.unit or N.display_unit(row.feature)
    delta = abs(float(row.value) - float(row.env_median))
    return {
        "feature": str(row.feature),
        "value": fmt.value(row.value, unit or row.feature),
        "median": fmt.value(row.env_median, unit or row.feature),
        "target": fmt.value(row.env_median, unit or row.feature),
        "sigma": fmt.value(row.env_sigma, unit or row.feature),
        "delta": fmt.value(delta, unit or row.feature),
        "z": fmt.z(row.z),
        "abs_z": fmt.z(abs(float(row.z))),
        "unit": unit,
        "direction": "above" if float(row.z) >= 0 else "below",
        "gap_label": str(row.gap_label or "none"),
    }


def region_fields(
    region: Region,
    *,
    templates: Templates,
    words: str = "",
    word_before: str = "",
    word_after: str = "",
    mode: str = "reference",
    n_readers: int = 0,
    confidence: float = 0.0,
) -> dict[str, str]:
    """Every placeholder a ``rationale``/``fix``/``headline`` template may use.

    Keys are listed in the header of ``configs/templates.yaml``; all of them
    are always present, which is what lets a template fail loudly on a typo
    instead of silently on a missing measurement.

    Args:
        region: The region being explained.
        templates: Loaded templates, for the number formatting.
        words: The words to quote; the region's own when empty.
        word_before: For a pause flaw, the word the gap follows.
        word_after: For a pause flaw, the word the gap precedes.
        mode: Operating mode of the analysis.
        n_readers: Readers behind the comparison.
        confidence: The confidence the card will report.

    Returns:
        ``{placeholder: formatted value}``.
    """
    fmt = templates.formatting
    primary = region.evidence[0] if region.evidence else None
    fields: dict[str, str] = {
        "region_id": str(region.region_id),
        "t_start": fmt.time(region.t_start),
        "t_end": fmt.time(region.t_end),
        "duration": fmt.time(region.duration),
        "words": words or region.words,
        "word_before": word_before,
        "word_after": word_after,
        "label": str(region.label),
        "category": str(region.category),
        "level": str(region.level),
        "s": fmt.probability(region.s),
        "confidence": fmt.probability(confidence),
        "mode": str(mode),
        "n_readers": str(int(n_readers or region.envelope_n)),
        "envelope_n": str(int(region.envelope_n)),
        "variant": _variant_of(region),
        "clf_label": str(region.clf_label or region.label),
        "clf_prob": fmt.probability(region.clf_prob if region.clf_prob is not None else 0.0),
        "peak_z": fmt.z(region.peak_z),
    }
    if primary is not None:
        fields.update(_evidence_fields(primary, templates))
    else:
        fields.update(
            {
                "feature": "",
                "value": "n/a",
                "median": "n/a",
                "target": "n/a",
                "sigma": "n/a",
                "delta": "n/a",
                "z": fmt.z(region.peak_z),
                "abs_z": fmt.z(abs(float(region.peak_z))),
                "unit": "",
                "direction": "above" if float(region.peak_z) >= 0 else "below",
                "gap_label": "none",
            }
        )
    return fields


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_card(
    region: Region,
    *,
    annotation: TextAnnotation | None = None,
    envelope: TextEnvelope | None = None,
    z_by_unit: Mapping[str, ZResult] | None = None,
    mode: str = "reference",
    templates: Templates | None = None,
    config: Config | None = None,
    attach: bool = True,
) -> Card:
    """Explain one region (spec 12).

    Args:
        region: The region to explain.
        annotation: Transcript annotation from
            :func:`rostrum.text.rhetoric.annotate`; without it the card still
            renders, with no text context.
        envelope: The text envelope, for the readers' behaviour and the
            confidence.
        z_by_unit: The comparisons detection ran on, to locate the unit the
            card quotes.
        mode: Operating mode (spec 13).
        templates: Loaded templates; built from ``config`` when omitted.
        config: Configuration.
        attach: Write ``rationale``, ``fix`` and ``confidence`` back onto the
            region, so an :class:`~rostrum.types.AnalysisResult` carries the
            same sentences the card shows. ``Region.context`` is left alone: it
            holds the variant and classifier tags, not prose.

    Returns:
        A :class:`Card`.

    Raises:
        TemplateError: A template needed a field that does not exist, or the
            label has no template at all.
    """
    templates = templates or Templates.from_config(config)
    label = str(region.label)
    variant = _variant_of(region)
    primary_feature = (
        N.canonical(region.evidence[0].feature) if region.evidence else str(region.label)
    )

    confidence, basis, _, _ = detection_confidence(
        region, feature=primary_feature, envelope=envelope, z_by_unit=z_by_unit
    )
    context = region_context(
        region,
        feature=primary_feature,
        annotation=annotation,
        envelope=envelope,
        z_by_unit=z_by_unit,
        templates=templates,
        config=config,
    )

    words = region.words
    word_before, word_after = "", ""
    if region.word_idx:
        first, last = int(region.word_idx[0]), int(region.word_idx[-1])
        if annotation is not None:
            words = annotation.word_text(first, last) or words
            word_before, word_after = annotation.word_text(first), annotation.word_text(last)
        elif region.gap_idx is not None and region.words:
            # A gap region quotes the two words around it; without a transcript
            # annotation, split what region formation recorded.
            parts = region.words.split()
            word_before, word_after = (parts[0], parts[-1]) if parts else ("", "")

    n_readers = int(region.envelope_n) or (envelope.n_references if envelope else 0)
    fields = region_fields(
        region,
        templates=templates,
        words=words,
        word_before=word_before,
        word_after=word_after,
        mode=mode,
        n_readers=n_readers,
        confidence=confidence,
    )

    rationale_key = f"rationale_variant.{variant}" if variant else f"rationale.{label}"
    if variant and not templates.has(rationale_key):
        rationale_key = f"rationale.{label}"
    fix_key = f"fix_variant.{variant}" if variant else f"fix.{label}"
    if variant and not templates.has(fix_key):
        fix_key = f"fix.{label}"

    rationale = templates.render_key(rationale_key, fields)
    extras = [
        templates.render_key(
            "evidence_phrase.gap" if row.gap_label else "evidence_phrase.default",
            _evidence_fields(row, templates),
            required=False,
        )
        for row in region.evidence[1:]
    ]
    extras = [part for part in extras if part]
    if extras:
        rationale = f"{rationale.rstrip('.')}; {'; '.join(extras)}."
    uncertain = _is_uncertain(region)
    if uncertain and templates.has("uncertain"):
        rationale = f"{rationale} {templates.render_key('uncertain', fields)}"

    card = Card(
        region_id=int(region.region_id),
        t_start=float(round_time(region.t_start) or 0.0),
        t_end=float(round_time(region.t_end) or 0.0),
        words=words,
        category=str(region.category),
        label=label,
        level=str(region.level),
        confidence=float(round(confidence, 3)),
        rule_label=str(region.rule_label) if region.rule_label else label,
        clf_prob=(None if region.clf_prob is None else float(round(region.clf_prob, 3))),
        evidence=[EvidenceRow.from_evidence(row) for row in region.evidence],
        context=context.text,
        rationale=rationale,
        fix=templates.render_key(fix_key, fields),
        envelope_n=int(region.envelope_n),
        mode=str(mode),
        headline=templates.render_key("headline", fields, required=False),
        duration=round(float(region.duration), TIME_DECIMALS),
        variant=variant,
        clf_label=(str(region.clf_label) if region.clf_label else None),
        uncertain=uncertain,
        confidence_basis=basis,
        context_detail=context.to_dict(),
        s=float(round_value(region.s) or 0.0),
    )

    if attach:
        region.rationale = card.rationale
        region.fix = card.fix
        region.words = words or region.words
        if not float(region.confidence):
            region.confidence = card.confidence
    return card


def build_cards(
    regions: Sequence[Region],
    *,
    annotation: TextAnnotation | None = None,
    envelope: TextEnvelope | None = None,
    z_by_unit: Mapping[str, ZResult] | None = None,
    mode: str = "reference",
    templates: Templates | None = None,
    config: Config | None = None,
    attach: bool = True,
) -> list[Card]:
    """Explain every region, in time order. See :func:`build_card`."""
    templates = templates or Templates.from_config(config)
    cards = [
        build_card(
            region,
            annotation=annotation,
            envelope=envelope,
            z_by_unit=z_by_unit,
            mode=mode,
            templates=templates,
            config=config,
            attach=attach,
        )
        for region in sorted(regions, key=lambda r: (r.t_start, str(r.label)))
    ]
    logger.info("explained %d region(s) in %s mode", len(cards), mode)
    return cards


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_markdown(card: Card, *, heading_level: int = 3) -> str:
    """Render a card as Markdown for the dashboard and the PDF.

    The order is the order a reader needs it in: what and where, then why
    (prose), then the measurements that back the prose, then the fix.

    Args:
        card: The card.
        heading_level: Markdown heading depth for the headline.

    Returns:
        A Markdown block, no trailing newline.
    """
    hashes = "#" * max(1, int(heading_level))
    headline = card.headline or (
        f"[{card.t_start:.2f}-{card.t_end:.2f} s] {card.category} - {card.label}"
    )
    lines = [f"{hashes} {headline}", ""]
    if card.words:
        lines += [f"> {card.words}", ""]
    lines += [card.rationale, ""]
    if card.context:
        lines += [f"*{card.context}*", ""]
    if card.evidence:
        lines += [
            "| feature | you | readers (median +/- sigma) | z |",
            "| --- | --- | --- | --- |",
        ]
        for row in card.evidence:
            unit = f" {row.unit}" if row.unit else ""
            lines.append(
                f"| {row.feature} | {row.value:g}{unit} | "
                f"{row.env_median:g} +/- {row.env_sigma:g}{unit} | {row.z:+.1f} |"
            )
        lines.append("")
    lines += [f"**Fix.** {card.fix}", ""]
    footer = [f"severity {card.level}", f"confidence {card.confidence:.2f}"]
    if card.envelope_n:
        footer.append(f"{card.envelope_n} reference readers")
    if card.clf_prob is not None:
        footer.append(f"classifier {card.clf_label} p={card.clf_prob:.2f}")
    footer.append(f"mode: {card.mode}")
    lines.append(f"<sub>{' | '.join(footer)}</sub>")
    return "\n".join(lines).rstrip()
