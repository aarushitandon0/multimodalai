"""Tests for the explanation cards and their templates (spec 12).

The acceptance criterion of this phase is that *every* label renders with no
missing placeholder and that the card JSON validates, so the parametrised test
over the eight labels of the taxonomy is the centre of this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from rostrum.config import load_config
from rostrum.detect.classifier import UNCERTAIN_TAG
from rostrum.detect.regions import label_category
from rostrum.explain.cards import (
    Card,
    build_card,
    build_cards,
    detection_confidence,
    region_fields,
    render_markdown,
)
from rostrum.explain.context import locate, region_context, text_context
from rostrum.explain.templates import Formatting, TemplateError, Templates, collapse
from rostrum.text.rhetoric import annotate
from rostrum.types import Evidence, Region

LABELS: tuple[str, ...] = (
    "rushing",
    "dragging",
    "missing_pause",
    "hesitation_pause",
    "monotone",
    "misplaced_stress",
    "low_energy",
    "mumbling",
)

TEXT = (
    "We can not dedicate, we can not consecrate. "
    "We can not hallow this ground, we can not forget."
)

PRIMARY: dict[str, tuple[str, float, float, float, str]] = {
    # label -> (feature, value, median, z, unit)
    "rushing": ("rate_local", 6.42, 3.91, 4.1, "syll/s"),
    "dragging": ("rate_local", 2.30, 3.91, -3.4, "syll/s"),
    "missing_pause": ("gap_duration", 0.06, 0.82, -3.6, "s"),
    "hesitation_pause": ("gap_duration", 0.71, 0.04, 4.5, "s"),
    "monotone": ("phrase_f0_range_st", 1.80, 7.40, -3.9, "st"),
    "misplaced_stress": ("prominence", 4.10, 2.40, 3.2, ""),
    "low_energy": ("energy_rel_z", -1.90, 0.10, -3.1, "z"),
    "mumbling": ("cpp", 9.80, 14.20, -2.9, "dB"),
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def templates(cfg):
    return Templates.from_config(cfg)


@pytest.fixture(scope="module")
def annotation(cfg):
    return annotate(TEXT, text_id="anaphora_18w", config=cfg)


def make_region(
    label: str,
    *,
    cfg=None,
    region_id: int = 0,
    variant: str = "",
    extra_evidence: bool = True,
) -> Region:
    """A region of one label with the evidence rows a card would receive."""
    feature, value, median, z, unit = PRIMARY[label]
    is_gap = label in {"missing_pause", "hesitation_pause"}
    evidence = [
        Evidence(
            feature=feature,
            value=value,
            env_median=median,
            env_sigma=abs(value - median) / max(abs(z), 1e-9),
            z=z,
            unit=unit,
            gap_label=("boundary" if label == "missing_pause" else "none") if is_gap else None,
            n_references=19,
        )
    ]
    if extra_evidence:
        evidence.append(
            Evidence(
                feature="f0_std_st",
                value=1.1,
                env_median=3.4,
                env_sigma=0.7,
                z=-3.2,
                unit="st",
                n_references=19,
            )
        )
    return Region(
        region_id=region_id,
        t_start=41.274,
        t_end=45.831,
        label=label,  # type: ignore[arg-type]
        category=label_category(label, cfg),
        level="L3",
        s=0.6,
        rule_label=label,  # type: ignore[arg-type]
        envelope_n=19,
        word_idx=[4, 6] if not is_gap else [3, 4],
        gap_idx=3 if is_gap else None,
        words="we can not" if not is_gap else "dedicate, we",
        evidence=evidence,
        context=variant,
        peak_z=z,
    )


# --------------------------------------------------------------------------- #
# Every label renders
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", LABELS)
def test_every_label_renders_without_a_missing_placeholder(cfg, annotation, label):
    card = build_card(make_region(label, cfg=cfg), annotation=annotation, config=cfg)
    for text in (card.headline, card.rationale, card.fix):
        assert text
        assert "{" not in text and "}" not in text
        assert "n/a" not in text


@pytest.mark.parametrize("label", LABELS)
def test_every_label_has_a_rationale_and_a_fix_template(templates, label):
    assert templates.has(f"rationale.{label}")
    assert templates.has(f"fix.{label}")


@pytest.mark.parametrize("label", LABELS)
def test_every_card_validates_as_json(cfg, annotation, label):
    card = build_card(make_region(label, cfg=cfg), annotation=annotation, config=cfg)
    payload = card.to_dict()
    assert Card.model_validate(payload) == card
    # The spec 12 schema, key by key.
    for key in (
        "region_id",
        "t_start",
        "t_end",
        "words",
        "category",
        "label",
        "level",
        "confidence",
        "rule_label",
        "clf_prob",
        "evidence",
        "context",
        "rationale",
        "fix",
        "envelope_n",
        "mode",
    ):
        assert key in payload


def test_a_card_rejects_an_unknown_field():
    with pytest.raises(ValidationError):
        Card(region_id=0, t_start=0.0, t_end=1.0, nonsense=True)  # type: ignore[call-arg]


def test_the_variant_renders_its_own_sentence(cfg, annotation):
    region = make_region("low_energy", cfg=cfg, variant="fading")
    card = build_card(region, annotation=annotation, config=cfg)
    assert card.variant == "fading"
    assert "fade" in card.rationale or "fades" in card.rationale


def test_a_missing_template_is_an_error(cfg, annotation):
    region = make_region("rushing", cfg=cfg)
    region.label = "no_such_label"  # type: ignore[assignment]
    with pytest.raises(TemplateError):
        build_card(region, annotation=annotation, config=cfg)


# --------------------------------------------------------------------------- #
# What the sentences say
# --------------------------------------------------------------------------- #


def test_the_rationale_quotes_the_primary_number(cfg, annotation):
    card = build_card(make_region("rushing", cfg=cfg), annotation=annotation, config=cfg)
    assert "6.4 syll/s" in card.rationale  # rates to 1 dp
    assert "3.9 syll/s" in card.rationale
    assert "4.1 sigma" in card.rationale


def test_the_second_evidence_row_is_appended_with_its_own_numbers(cfg, annotation):
    card = build_card(make_region("rushing", cfg=cfg), annotation=annotation, config=cfg)
    assert "f0_std_st" in card.rationale
    assert "-3.2 sigma" in card.rationale


def test_a_pause_card_names_the_words_either_side(cfg, annotation):
    card = build_card(make_region("missing_pause", cfg=cfg), annotation=annotation, config=cfg)
    assert "dedicate," in card.rationale
    assert "0.06 s" in card.rationale  # seconds to 2 dp
    assert "boundary" in card.rationale


def test_the_fix_names_the_target_value(cfg, annotation):
    card = build_card(make_region("missing_pause", cfg=cfg), annotation=annotation, config=cfg)
    assert "0.82 s" in card.fix


def test_the_context_reports_the_anaphora(cfg, annotation):
    card = build_card(make_region("rushing", cfg=cfg), annotation=annotation, config=cfg)
    assert "we can not" in card.context
    assert card.context_detail["kind"] == "anaphora"
    assert card.context_detail["n_occurrences"] == 4


def test_the_words_come_from_the_transcript_not_the_region(cfg, annotation):
    region = make_region("rushing", cfg=cfg)
    region.words = "whatever the detector recorded"
    card = build_card(region, annotation=annotation, config=cfg)
    assert card.words == annotation.word_text(4, 6)


def test_without_an_annotation_the_card_still_renders(cfg):
    card = build_card(make_region("rushing", cfg=cfg), config=cfg)
    assert card.rationale
    assert card.context == ""
    assert card.words == "we can not"


def test_an_uncertain_type_is_stated_rather_than_hidden(cfg, annotation):
    region = make_region("rushing", cfg=cfg, variant="")
    region.context = UNCERTAIN_TAG
    region.clf_label = "dragging"  # type: ignore[assignment]
    region.clf_prob = 0.91
    card = build_card(region, annotation=annotation, config=cfg)
    assert card.uncertain
    assert "uncertain" in card.rationale
    assert "0.91" in card.rationale


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #


def test_confidence_is_unavailable_without_an_envelope(cfg):
    value, basis, _, _ = detection_confidence(make_region("rushing", cfg=cfg), feature="rate_local")
    assert basis == "unavailable"
    assert value == 0.0


def test_the_detectors_own_confidence_is_passed_through(cfg):
    region = make_region("rushing", cfg=cfg)
    region.confidence = 0.42
    value, basis, _, _ = detection_confidence(region, feature="rate_local")
    assert (value, basis) == (0.42, "detector")


def test_attach_writes_the_sentences_back_onto_the_region(cfg, annotation):
    region = make_region("rushing", cfg=cfg)
    card = build_card(region, annotation=annotation, config=cfg)
    assert region.rationale == card.rationale
    assert region.fix == card.fix


def test_attach_can_be_switched_off(cfg, annotation):
    region = make_region("rushing", cfg=cfg)
    build_card(region, annotation=annotation, config=cfg, attach=False)
    assert region.rationale == ""


def test_the_region_context_tag_is_never_overwritten_with_prose(cfg, annotation):
    """Region.context carries the variant and the classifier tags, not text."""
    region = make_region("low_energy", cfg=cfg, variant="fading")
    build_card(region, annotation=annotation, config=cfg)
    assert region.context == "fading"


# --------------------------------------------------------------------------- #
# Several regions
# --------------------------------------------------------------------------- #


def test_build_cards_sorts_by_onset(cfg, annotation):
    early = make_region("rushing", cfg=cfg, region_id=1)
    early.t_start, early.t_end = 1.0, 2.0
    late = make_region("monotone", cfg=cfg, region_id=0)
    cards = build_cards([late, early], annotation=annotation, config=cfg)
    assert [c.t_start for c in cards] == [1.0, 41.274]


def test_times_are_rounded_to_milliseconds(cfg, annotation):
    region = make_region("rushing", cfg=cfg)
    region.t_start, region.t_end = 1.23456, 2.98765
    card = build_card(region, annotation=annotation, config=cfg)
    assert card.t_start == 1.235
    assert card.t_end == 2.988


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def test_render_markdown_holds_the_whole_card(cfg, annotation):
    card = build_card(make_region("rushing", cfg=cfg), annotation=annotation, config=cfg)
    text = render_markdown(card)
    assert text.startswith("### ")
    assert card.rationale in text
    assert card.fix in text
    assert "| feature | you |" in text
    assert "rate_local" in text
    assert "mode: reference" in text


def test_render_markdown_is_ascii_only(cfg, annotation):
    """The PDF and the Windows terminal both get the same bytes."""
    card = build_card(make_region("monotone", cfg=cfg), annotation=annotation, config=cfg)
    render_markdown(card).encode("ascii")


def test_render_markdown_survives_an_empty_card():
    card = Card(region_id=0, t_start=0.0, t_end=1.0, label="rushing", category="pace")
    text = render_markdown(card)
    assert "rushing" in text


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def test_the_formatter_uses_the_unit_to_choose_decimals(templates):
    fmt = templates.formatting
    assert fmt.value(3.14159, "syll/s") == "3.1"
    assert fmt.value(3.14159, "s") == "3.14"
    assert fmt.value(3.14159, "st") == "3.1"
    assert fmt.z(-3.14159) == "-3.1"


def test_the_formatter_accepts_a_feature_name_as_well_as_a_unit(templates):
    assert templates.formatting.value(3.14159, "rate_local") == "3.1"


def test_a_non_finite_value_is_marked_missing(templates):
    assert templates.formatting.value(float("nan"), "s") == "n/a"
    assert templates.formatting.value(None, "s") == "n/a"


def test_an_unknown_unit_falls_back_to_the_default(templates):
    assert templates.formatting.value(3.14159, "parsecs") == "3.14"


def test_collapse_closes_up_an_empty_placeholder():
    assert collapse("spread 0.23 , n = 6") == "spread 0.23, n = 6"
    assert collapse("one\n  two") == "one two"


def test_render_raises_on_an_unknown_field():
    templates = Templates(data={"rationale": {"x": "{nope}"}}, formatting=Formatting())
    with pytest.raises(TemplateError):
        templates.render_key("rationale.x", {"other": "1"})


def test_an_optional_template_may_be_absent():
    templates = Templates(data={}, formatting=Formatting())
    assert templates.render_key("context.none", {}, required=False) == ""


def test_region_fields_are_complete_even_without_evidence(cfg, templates):
    region = Region(region_id=0, t_start=0.0, t_end=1.0, label="rushing", category="pace")
    fields = region_fields(region, templates=templates)
    for key in ("value", "median", "sigma", "z", "unit", "target", "delta"):
        assert key in fields


# --------------------------------------------------------------------------- #
# Locating a region in its comparison
# --------------------------------------------------------------------------- #


def _z_result(cfg, unit: str, column: str):
    from rostrum.detect.zscore import ZResult

    index = pd.Index(range(4), name="unit_idx", dtype="int64")
    table = pd.DataFrame(
        {
            "t_start": [0.0, 1.0, 2.0, 3.0],
            "t_end": [1.0, 2.0, 3.0, 4.0],
            column: [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )
    frame = pd.DataFrame({column: [0.2, 0.4, 5.0, 0.1]}, index=index)
    return ZResult(
        unit=unit,
        text_id="t",
        fold="all",
        table=table,
        value=frame,
        median=frame,
        sigma=frame,
        z=frame,
        n_references=pd.Series([4, 4, 4, 4], index=index),
        channel_matched=True,
    )


def test_locate_picks_the_unit_with_the_largest_deviation(cfg):
    region = make_region("rushing", cfg=cfg)
    region.t_start, region.t_end = 0.0, 4.0
    result = _z_result(cfg, "word", "rate_local_syll_per_s")
    location = locate(region, feature="rate_local", z_by_unit={"word": result})
    assert location is not None
    assert (location.unit, location.unit_idx) == ("word", 2)


def test_locate_uses_the_gap_index_for_a_pause_flaw(cfg):
    region = make_region("missing_pause", cfg=cfg)
    result = _z_result(cfg, "gap", "gap_duration_s")
    location = locate(region, feature="gap_duration", z_by_unit={"gap": result})
    assert location is not None
    assert (location.unit, location.unit_idx) == ("gap", 3)


def test_locate_returns_none_when_the_feature_is_absent(cfg):
    region = make_region("rushing", cfg=cfg)
    result = _z_result(cfg, "word", "energy_mean_z")
    assert locate(region, feature="rate_local", z_by_unit={"word": result}) is None


def test_text_context_without_an_annotation_is_empty(cfg):
    assert (
        text_context(
            make_region("rushing", cfg=cfg), None, templates=Templates.from_config(cfg)
        ).clause
        == ""
    )


def test_region_context_needs_no_envelope(cfg, annotation):
    context = region_context(
        make_region("rushing", cfg=cfg), feature="rate_local", annotation=annotation, config=cfg
    )
    assert context.clause
    assert context.reference == ""
    assert context.text == context.clause


def test_confidence_counts_the_readers_beyond_the_participant(cfg, annotation):
    """The exceedance is over the envelope's own raw reader values."""
    from rostrum.envelope.build import TextEnvelope, UnitEnvelope
    from rostrum.types import Envelope

    readers = np.array([[3.8], [3.9], [4.0], [4.1], [6.6]], dtype=float)
    unit = UnitEnvelope(
        envelope=Envelope(
            text_id="t",
            unit="word",
            features=["rate_local_syll_per_s"],
            median={4: [3.95]},
            sigma={4: [0.2]},
            n_references={4: 5},
            reference_ids=[f"r{i}" for i in range(5)],
        ),
        values={4: readers},
    )
    envelope = TextEnvelope(
        text_id="t", units={"word": unit}, reference_ids=[f"r{i}" for i in range(5)]
    )
    region = make_region("rushing", cfg=cfg)
    region.word_idx = [4, 4]
    value, basis, beyond, measured = detection_confidence(
        region, feature="rate_local", envelope=envelope
    )
    assert basis == "empirical_exceedance"
    assert (beyond, measured) == (4, 5)  # 6.42 syll/s is past four of the five
    assert value == pytest.approx((4 + 0.5) / 6.0)
