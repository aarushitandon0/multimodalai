"""Why this spot: text context and reference behaviour for a card (spec 12).

A number is not yet an explanation. Spec 12 asks each card to say where in the
*text* the region sits and what the skilled readers did there, because that is
what turns "rate z = +4.1" into "you rushed a repeated phrase and cut the pause
the readers all took".

Two sources, both already computed elsewhere:

* :class:`rostrum.text.rhetoric.TextAnnotation` - sentences, repeated
  sequences, comma lists and the gap labels, derived from the transcript alone;
* :class:`rostrum.envelope.build.UnitEnvelope` - the raw per-reader values
  behind the envelope, so "17 of 19 readers paused here" is a count over
  measurements rather than a claim.

Nothing here judges. It selects a clause from ``configs/templates.yaml`` and
fills it with counts and medians, and it is a pure function of the region, the
annotation and the envelope.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.explain.templates import Templates
from rostrum.features import names as N

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.detect.zscore import ZResult
    from rostrum.envelope.build import TextEnvelope, UnitEnvelope
    from rostrum.text.rhetoric import TextAnnotation
    from rostrum.types import Region

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PAUSE_THRESHOLD_S",
    "RegionContext",
    "UnitLocation",
    "locate",
    "reference_behaviour",
    "region_context",
    "text_context",
]

DEFAULT_PAUSE_THRESHOLD_S: float = 0.250
"""Silence counted as a pause, from ``detection.rules.hesitation_pause``.

The fluency literature's threshold, and the one the hesitation rule already
uses, so the context line and the detector agree on what a pause is.
"""

_GAP_COLUMN = "gap_duration_s"
_RATE_FEATURES = frozenset({"rate_local", "phrase_rate", "rate_relative"})


@dataclass(frozen=True)
class UnitLocation:
    """Where a region's primary feature was measured.

    Attributes:
        unit: ``"word"``, ``"gap"`` or ``"phrase"``.
        unit_idx: Canonical index of the unit with the largest deviation.
        column: Feature-table column of the primary feature.
    """

    unit: str
    unit_idx: int
    column: str


@dataclass(frozen=True)
class RegionContext:
    """The context sentence of one card, with the parts it was built from.

    Attributes:
        text: The rendered sentence: text clause plus reference behaviour.
        clause: The text clause alone.
        reference: The reference-behaviour clause alone.
        kind: Which context template was chosen (``"anaphora"``, ``"none"``).
        position: Sentence position of the region's first word.
        gap_label: Gap label at the region, for a pause flaw.
        punctuation: Punctuation at the region's boundary, if any.
        repeat_text: The repeated sequence, when one covers the region.
        n_occurrences: How often that sequence occurs in the text.
        readers_matching: Readers satisfying the reference-behaviour test.
        readers_measured: Readers with a measurement at all.
    """

    text: str = ""
    clause: str = ""
    reference: str = ""
    kind: str = "none"
    position: str = ""
    gap_label: str = ""
    punctuation: str = ""
    repeat_text: str = ""
    n_occurrences: int = 0
    readers_matching: int = 0
    readers_measured: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON form, for the card's ``context_detail`` block."""
        return {
            "kind": self.kind,
            "position": self.position,
            "gap_label": self.gap_label,
            "punctuation": self.punctuation,
            "repeat_text": self.repeat_text,
            "n_occurrences": int(self.n_occurrences),
            "readers_matching": int(self.readers_matching),
            "readers_measured": int(self.readers_measured),
        }


# --------------------------------------------------------------------------- #
# Locating the region's primary measurement
# --------------------------------------------------------------------------- #


def _overlapping_rows(result: ZResult, region: Region) -> list[int]:
    """Unit indices of ``result`` whose span overlaps the region."""
    table = result.table
    if "t_start" not in table.columns or "t_end" not in table.columns:
        return []
    inside = (table["t_start"] < float(region.t_end)) & (table["t_end"] > float(region.t_start))
    return [int(i) for i in table.index[inside]]


def locate(
    region: Region,
    *,
    feature: str,
    envelope: TextEnvelope | None = None,
    z_by_unit: Mapping[str, ZResult] | None = None,
) -> UnitLocation | None:
    """Find the unit whose deviation the card should quote.

    A pause flaw is its own gap. Anything else is resolved by overlap in time
    with the comparison tables, and the unit with the largest ``|z|`` wins -
    the instant a listener would point at.

    Args:
        region: The region to locate.
        feature: Canonical name of its primary feature.
        envelope: The text envelope, to check which units carry the feature.
        z_by_unit: ``{unit: ZResult}`` from the detection run.

    Returns:
        A :class:`UnitLocation`, or ``None`` when the feature cannot be found
        in any available unit.
    """
    z_by_unit = z_by_unit or {}
    units = list(z_by_unit) or (list(envelope.units) if envelope is not None else [])
    # Gap first: a pause flaw is indexed on exactly one gap.
    if region.gap_idx is not None and "gap" in units:
        columns = (
            list(z_by_unit["gap"].features)
            if "gap" in z_by_unit
            else (envelope.unit("gap").features if envelope is not None else [])
        )
        resolved = [c for c in N.resolve(feature, columns) if c in columns] or [_GAP_COLUMN]
        return UnitLocation(unit="gap", unit_idx=int(region.gap_idx), column=resolved[0])

    order = [u for u in ("word", "phrase", "gap") if u in units]
    for unit in order:
        result = z_by_unit.get(unit)
        columns = (
            list(result.features)
            if result is not None
            else (envelope.unit(unit).features if envelope is not None else [])
        )
        resolved = [c for c in N.resolve(feature, columns) if c in columns]
        if not resolved:
            continue
        column = resolved[0]
        if result is None:
            # No comparison to scan; the word indices are the canonical ones.
            if unit != "word" or not region.word_idx:
                continue
            return UnitLocation(unit=unit, unit_idx=int(region.word_idx[0]), column=column)
        indices = _overlapping_rows(result, region)
        if not indices:
            continue
        z_values = result.z.loc[[i for i in indices if i in result.z.index], column]
        unit_idx = int(z_values.abs().idxmax()) if z_values.notna().any() else int(indices[0])
        return UnitLocation(unit=unit, unit_idx=unit_idx, column=column)
    return None


# --------------------------------------------------------------------------- #
# What the readers did
# --------------------------------------------------------------------------- #


def _pause_threshold(config: Config | None) -> float:
    """Silence counted as a pause, from ``detection.rules.hesitation_pause``."""
    if config is None:
        return DEFAULT_PAUSE_THRESHOLD_S
    rules = config.detection["rules"]["hesitation_pause"]
    return float(rules.get("min_absolute_gap_s", DEFAULT_PAUSE_THRESHOLD_S))


def reference_behaviour(
    location: UnitLocation,
    unit_envelope: UnitEnvelope,
    *,
    templates: Templates,
    config: Config | None = None,
) -> tuple[str, int, int]:
    """One sentence on what the skilled readers did at this spot (spec 12).

    For a gap this is a count - "17 of 19 skilled readers paused at least
    0.25 s here" - because a count is what a reader can act on. For everything
    else it is the envelope's median and spread at that unit.

    Args:
        location: The unit and column the card quotes.
        unit_envelope: That unit's envelope, carrying the raw reader values.
        templates: Loaded templates.
        config: Supplies the pause threshold.

    Returns:
        ``(sentence, matching, measured)``. The sentence is empty when the
        envelope has no values for this unit.
    """
    fmt = templates.formatting
    column = location.column
    if not unit_envelope.has(column) or location.unit_idx not in unit_envelope.values:
        return "", 0, 0

    unit_label = N.display_unit(column)
    values = unit_envelope.reference_values(location.unit_idx, column)
    finite = values[np.isfinite(values)]
    median = float(np.median(finite)) if finite.size else float("nan")
    spread = float(np.std(finite)) if finite.size else float("nan")

    if location.unit == "gap" and column == _GAP_COLUMN:
        threshold = _pause_threshold(config)
        matching, measured = unit_envelope.count_where(
            location.unit_idx, column, at_least=threshold
        )
        key = "reference_behaviour.pause" if matching else "reference_behaviour.pause_none"
        fields = {
            "k": str(matching),
            "n": str(measured),
            "threshold": fmt.value(threshold, "s"),
            "unit": unit_label or "s",
            "feature": N.canonical(column),
            "median": fmt.value(median, column),
            "sigma": fmt.value(spread, column),
            "words": "",
        }
        return templates.render_key(key, fields), matching, measured

    measured = int(finite.size)
    key = (
        "reference_behaviour.rate"
        if N.canonical(column) in _RATE_FEATURES and templates.has("reference_behaviour.rate")
        else "reference_behaviour.default"
    )
    fields = {
        "k": str(measured),
        "n": str(measured),
        "threshold": "",
        "unit": unit_label,
        "feature": N.canonical(column),
        "median": fmt.value(median, column),
        "sigma": fmt.value(spread, column),
        "words": "",
    }
    return templates.render_key(key, fields), measured, measured


# --------------------------------------------------------------------------- #
# What the text says
# --------------------------------------------------------------------------- #


def text_context(
    region: Region,
    annotation: TextAnnotation | None,
    *,
    templates: Templates,
) -> RegionContext:
    """Choose and render the text-context clause for a region (spec 12).

    Precedence, strongest first: a repeated sequence (anaphora, then
    parallelism), a list item, the gap's own punctuation for a pause flaw, then
    the plain sentence position. The strongest fact is the one a reader needs:
    that the phrase is the third "we can not" matters more than that it sits
    mid-sentence.

    Args:
        region: The region to describe.
        annotation: The transcript annotation, or ``None`` for an unknown text.
        templates: Loaded templates.

    Returns:
        A :class:`RegionContext` with the clause but no reference behaviour.
    """
    if annotation is None or not annotation.words:
        return RegionContext()

    first = int(region.word_idx[0]) if region.word_idx else 0
    last = int(region.word_idx[-1]) if region.word_idx else first
    position = annotation.position(first)
    gap_label, punctuation = (
        annotation.gap(int(region.gap_idx)) if region.gap_idx is not None else ("", "")
    )
    n_sentences = len(annotation.sentences)
    sentence = annotation.sentence_of(first)
    base = {
        "words": annotation.word_text(first, last),
        "repeat_text": "",
        "n_occurrences": "0",
        "punctuation": punctuation,
        "position": position,
        "gap_label": gap_label or "none",
        "sentence_idx": str((sentence.idx + 1) if sentence is not None else 0),
        "n_sentences": str(n_sentences),
    }
    base["position_phrase"] = templates.render_key(
        f"position_phrase.{position}", base, required=False
    )

    repeats = annotation.repeats_at(first, last)
    chosen = next((r for r in repeats if r.kind == "anaphora"), repeats[0] if repeats else None)

    if chosen is not None:
        fields = {**base, "repeat_text": chosen.text, "n_occurrences": str(chosen.n_occurrences)}
        kind = chosen.kind
        clause = templates.render_key(f"context.{kind}", fields, required=False)
        if clause:
            return RegionContext(
                clause=clause,
                kind=kind,
                position=position,
                gap_label=gap_label,
                punctuation=punctuation,
                repeat_text=chosen.text,
                n_occurrences=chosen.n_occurrences,
            )

    if annotation.in_list(first, last):
        clause = templates.render_key("context.list_item", base, required=False)
        if clause:
            return RegionContext(
                clause=clause,
                kind="list_item",
                position=position,
                gap_label=gap_label,
                punctuation=punctuation,
            )

    kind = position
    if region.gap_idx is not None and punctuation:
        kind = "clause_boundary"
    clause = templates.render_key(f"context.{kind}", base, required=False)
    return RegionContext(
        clause=clause,
        kind=kind,
        position=position,
        gap_label=gap_label,
        punctuation=punctuation,
    )


def region_context(
    region: Region,
    *,
    feature: str,
    annotation: TextAnnotation | None = None,
    envelope: TextEnvelope | None = None,
    z_by_unit: Mapping[str, ZResult] | None = None,
    templates: Templates | None = None,
    config: Config | None = None,
) -> RegionContext:
    """The full context sentence: text clause plus reference behaviour.

    Args:
        region: The region to describe.
        feature: Canonical name of its primary feature, i.e. the feature of its
            strongest evidence row.
        annotation: Transcript annotation, or ``None`` for an unknown text.
        envelope: The text envelope, for the readers' behaviour.
        z_by_unit: The comparisons detection ran on, to locate the unit.
        templates: Loaded templates; built from ``config`` when omitted.
        config: Configuration.

    Returns:
        A :class:`RegionContext`; ``text`` is the sentence the card prints.
    """
    templates = templates or Templates.from_config(config)
    context = text_context(region, annotation, templates=templates)

    reference, matching, measured = "", 0, 0
    if envelope is not None:
        location = locate(region, feature=feature, envelope=envelope, z_by_unit=z_by_unit)
        if location is not None and location.unit in envelope.units:
            reference, matching, measured = reference_behaviour(
                location,
                envelope.unit(location.unit),
                templates=templates,
                config=config,
            )

    parts = [part for part in (context.clause, reference) if part]
    return RegionContext(
        text=" ".join(parts),
        clause=context.clause,
        reference=reference,
        kind=context.kind,
        position=context.position,
        gap_label=context.gap_label,
        punctuation=context.punctuation,
        repeat_text=context.repeat_text,
        n_occurrences=context.n_occurrences,
        readers_matching=matching,
        readers_measured=measured,
    )
