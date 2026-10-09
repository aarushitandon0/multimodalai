"""Tests for the human-annotation path: TextGrid in, label JSON and agreement out.

Nothing here needs MFA, conda or audio. The two fixtures under
``tests/fixtures/annotation/done/`` are the ``tiny_long.TextGrid`` alignment
("four score and seven years ago", with one 150 ms gap after "and") plus a
``flaws`` tier drawn by two imaginary annotators, so every expected number is
known by construction:

``ann1``
    ``rushing:2`` 0.20-0.99, ``hesitation_pause:1`` 0.99-1.14 (the gap),
    ``monotone:3`` 1.14-2.19.
``ann2``
    ``rushing:3`` 0.28-1.05, ``dragging:2`` 1.14-2.19, and on a second tier
    (because it sits *inside* the dragging span) ``low_energy:1`` 1.52-1.87.

Which gives, at the configured collars: 2 of 3 events matched, so event-F1
2/3; and on those two matched events the labels agree once out of twice, with
one marginal each for ``monotone`` and ``dragging``, so kappa = 1/3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# `scripts/` is not a package; the import scripts live there.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import annotate_import
import annotate_prepare
import annotator_agreement
from rostrum.align.mfa import TextGridTier, format_textgrid, parse_textgrid, write_textgrid
from rostrum.config import load_config
from rostrum.data.annotation import (
    AnnotationError,
    FlawAnnotation,
    agreement,
    blank_flaw_tier,
    build_label_json,
    cohens_kappa,
    double_annotated,
    event_f1,
    flaw_tier,
    format_agreement_md,
    label_categories,
    match_events,
    parse_interval_label,
    read_annotations,
    settings_from_config,
)
from rostrum.inject import engine

FIXTURES = Path(__file__).parent / "fixtures"
DONE = FIXTURES / "annotation" / "done"
ANN1 = DONE / "ann1" / "tiny.TextGrid"
ANN2 = DONE / "ann2" / "tiny.TextGrid"


@pytest.fixture(scope="module")
def settings():
    """Annotation settings from the real ``configs/``."""
    return settings_from_config(load_config(apply_seeds=False))


@pytest.fixture(scope="module")
def cfg():
    """The loaded project config."""
    return load_config(apply_seeds=False)


# --------------------------------------------------------------------------- #
# Taxonomy and settings
# --------------------------------------------------------------------------- #


def test_taxonomy_is_the_eight_labels_of_the_spec(cfg):
    categories = label_categories(cfg)
    assert set(categories) == {
        "rushing",
        "dragging",
        "missing_pause",
        "hesitation_pause",
        "monotone",
        "misplaced_stress",
        "low_energy",
        "mumbling",
    }
    # Six rubric categories, and mumbling is the clarity one (CLAUDE.md).
    assert set(categories.values()) == {
        "pace",
        "pausing",
        "intonation",
        "emphasis",
        "energy",
        "clarity",
    }
    assert categories["mumbling"] == "clarity"


def test_hand_severity_maps_onto_the_ladder(settings):
    # 1-3 is what an annotator can hear and rate; L0 is "no flaw" (an empty
    # interval) and L1 is "barely perceptible", so neither is reachable by hand.
    assert [settings.level_for(s) for s in (1, 2, 3)] == ["L2", "L3", "L4"]
    assert settings.s_for(1) == pytest.approx(0.35)
    assert settings.s_for(3) == pytest.approx(0.90)
    with pytest.raises(AnnotationError, match="not on the annotation scale"):
        settings.level_for(4)


# --------------------------------------------------------------------------- #
# Interval text
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("rushing:2", ("rushing", 2)),
        ("  monotone:1  ", ("monotone", 1)),
        ("RUSHING:3", ("rushing", 3)),
        ("hesitation_pause:2", ("hesitation_pause", 2)),
        ("mumbling:3:", ("mumbling", 3)),
    ],
)
def test_parse_interval_label_accepts_the_guide_format(text, expected, settings):
    assert parse_interval_label(text, settings) == expected


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("", "empty"),
        ("rushing", "missing the severity"),
        ("rushing:0", "off the scale"),
        ("rushing:4", "off the scale"),
        ("rushing:high", "not an integer"),
        ("too_fast:2", "not a flaw label"),
        ("fading:2", "not a flaw label"),
    ],
)
def test_parse_interval_label_rejects_rather_than_guesses(text, match, settings):
    # A label quietly repaired into something plausible becomes wrong ground
    # truth, and every metric downstream is then measured against it.
    with pytest.raises(AnnotationError, match=match):
        parse_interval_label(text, settings)


def test_the_error_names_the_valid_labels(settings):
    with pytest.raises(AnnotationError) as exc:
        parse_interval_label("too_fast:2", settings)
    for label in settings.labels:
        assert label in str(exc.value)


# --------------------------------------------------------------------------- #
# Reading a completed grid
# --------------------------------------------------------------------------- #


def test_read_annotations_reads_the_fixture(settings):
    annotations, problems = read_annotations(ANN1, settings)
    assert problems == []
    assert [(a.label, a.severity, a.t_start, a.t_end) for a in annotations] == [
        ("rushing", 2, 0.20, 0.99),
        ("hesitation_pause", 1, 0.99, 1.14),
        ("monotone", 3, 1.14, 2.19),
    ]


def test_read_annotations_merges_the_overflow_tier(settings):
    # ann2's low_energy sits inside its dragging span, so it had to go on a
    # second tier; both tiers are one annotation set.
    annotations, _ = read_annotations(ANN2, settings)
    assert [a.label for a in annotations] == ["rushing", "dragging", "low_energy"]
    assert len(parse_textgrid(ANN2)) == 4  # words, phones, flaws, flaws_2


def test_read_annotations_sorts_by_onset(settings, tmp_path):
    # One flaw on each tier, the later one first: the reader still returns
    # them in time order, so a label JSON does not depend on tier order.
    grid = _write_flaws(
        tmp_path / "x.TextGrid",
        [(1.14, 2.19, "monotone:3")],
        extra=[(0.20, 0.99, "rushing:2")],
    )
    annotations, _ = read_annotations(grid, settings)
    assert [a.label for a in annotations] == ["rushing", "monotone"]


def test_read_annotations_needs_a_flaw_tier(settings):
    with pytest.raises(AnnotationError, match="no 'flaws' tier"):
        read_annotations(FIXTURES / "tiny_long.TextGrid", settings)


def test_a_misclick_is_not_an_annotation(settings, tmp_path):
    grid = _write_flaws(tmp_path / "short.TextGrid", [(0.20, 0.21, "rushing:2")])
    with pytest.raises(AnnotationError, match="only 10 ms long"):
        read_annotations(grid, settings)


def test_two_intervals_of_one_label_may_not_overlap(settings, tmp_path):
    grid = _write_flaws(
        tmp_path / "overlap.TextGrid",
        [(0.20, 0.99, "rushing:2")],
        extra=[(0.50, 1.40, "rushing:3")],
    )
    with pytest.raises(AnnotationError, match="overlap"):
        read_annotations(grid, settings)


def test_different_labels_may_overlap(settings, tmp_path):
    # Mumbling through a hesitation pause is one moment with two flaws;
    # splitting it would lose that.
    grid = _write_flaws(
        tmp_path / "cooccur.TextGrid",
        [(0.99, 1.14, "hesitation_pause:2")],
        extra=[(0.90, 1.30, "mumbling:1")],
    )
    annotations, problems = read_annotations(grid, settings)
    assert problems == []
    assert {a.label for a in annotations} == {"hesitation_pause", "mumbling"}


def test_non_strict_collects_every_problem(settings, tmp_path):
    grid = _write_flaws(
        tmp_path / "bad.TextGrid",
        [(0.20, 0.99, "too_fast:2"), (1.14, 2.19, "monotone:9")],
    )
    annotations, problems = read_annotations(grid, settings, strict=False)
    assert annotations == []
    assert len(problems) == 2  # one pass reports both, not one per run


def _write_flaws(path, intervals, extra=None):
    """Write ``tiny_long.TextGrid`` with a ``flaws`` tier holding ``intervals``."""
    base = parse_textgrid(FIXTURES / "tiny_long.TextGrid")
    tiers = [*base, TextGridTier("flaws", 0.0, 2.4, tuple(intervals))]
    if extra:
        tiers.append(TextGridTier("flaws_2", 0.0, 2.4, tuple(extra)))
    return write_textgrid(path, tiers)


# --------------------------------------------------------------------------- #
# Tiers and the TextGrid writer
# --------------------------------------------------------------------------- #


def test_blank_tier_is_one_empty_interval(settings):
    tier = blank_flaw_tier(0.0, 2.4, settings)
    assert tier.name == "flaws"
    assert tier.intervals == ((0.0, 2.4, ""),)


def test_writer_round_trips_an_mfa_grid(tmp_path):
    original = parse_textgrid(FIXTURES / "tiny_long.TextGrid")
    back = parse_textgrid(write_textgrid(tmp_path / "rt.TextGrid", original))
    assert [(t.name, t.intervals) for t in back] == [(t.name, t.intervals) for t in original]


def test_writer_fills_holes_so_praat_can_open_the_file(settings):
    # A tier with a hole in it is not a legal TextGrid; the gaps become
    # explicitly empty intervals.
    tier = flaw_tier(
        [FlawAnnotation("rushing", 2, 0.5, 1.0)],
        0.0,
        2.0,
        settings,
    )
    text = format_textgrid([tier])
    assert text.count("intervals [") == 3  # [0, 0.5], [0.5, 1.0], [1.0, 2.0]
    assert 'text = "rushing:2"' in text


def test_writer_rejects_overlapping_intervals(settings):
    with pytest.raises(ValueError, match="overlap"):
        flaw_tier(
            [FlawAnnotation("rushing", 2, 0.5, 1.5), FlawAnnotation("monotone", 2, 1.0, 2.0)],
            0.0,
            2.4,
            settings,
        )


def test_annotation_round_trip_through_a_textgrid(settings, tmp_path):
    """The acceptance criterion: fixture -> annotations -> grid -> annotations.

    Writing the tier back out and reading it again has to return exactly the
    same annotations, because that is what ``annotate_import.py`` does when it
    publishes ``dataset/alignments/<file_id>.TextGrid``.
    """
    original, _ = read_annotations(ANN1, settings)
    base = [t for t in parse_textgrid(ANN1) if t.name != "flaws"]
    grid = write_textgrid(
        tmp_path / "rt.TextGrid",
        [*base, flaw_tier(original, 0.0, 2.4, settings)],
    )
    again, problems = read_annotations(grid, settings)
    assert problems == []
    assert again == original


# --------------------------------------------------------------------------- #
# Label JSON (spec 7.1)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def label(settings, cfg):
    """The label JSON built from ann1's fixture annotation."""
    from rostrum.align.mfa import read_textgrid

    annotations, _ = read_annotations(ANN1, settings)
    alignment = read_textgrid(ANN1, file_id="tiny", text_id="s0_tiny")
    return build_label_json(
        "tiny",
        annotations,
        alignment,
        settings,
        annotator="ann1",
        annotators=["ann1", "ann2"],
        source_textgrid="tests/fixtures/annotation/done/ann1/tiny.TextGrid",
        config_hash=cfg.hash,
    )


def test_label_json_is_human_and_test_only(label):
    # CLAUDE.md invariant 11: human flawed takes are test-only, and nothing
    # was generated, so there is no seed to regenerate the file from.
    assert label["origin"] == "human"
    assert label["split"] == "test"
    assert label["seed"] is None
    assert label["text_id"] == "s0_tiny"


def test_label_json_records_who_annotated_it(label, settings):
    assert label["annotation"]["annotator"] == "ann1"
    assert label["annotation"]["annotators"] == ["ann1", "ann2"]
    assert label["annotation"]["guide_version"] == settings.guide_version
    assert label["annotation"]["source_textgrid"].endswith("ann1/tiny.TextGrid")


def test_label_json_flaws_carry_level_category_and_words(label):
    flaws = label["flaws"]
    assert [f["label"] for f in flaws] == ["rushing", "hesitation_pause", "monotone"]
    assert [f["level"] for f in flaws] == ["L3", "L2", "L4"]
    assert [f["severity"] for f in flaws] == [2, 1, 3]
    assert [f["s"] for f in flaws] == [0.6, 0.35, 0.9]
    assert [f["category"] for f in flaws] == ["pace", "pausing", "intonation"]
    # "four score and" / "seven years ago"; the pause sits between them and is
    # anchored to the word before it.
    assert [f["word_idx"] for f in flaws] == [[0, 2], [2, 2], [3, 5]]


def test_a_pause_annotation_names_its_gap(label):
    pause = next(f for f in label["flaws"] if f["label"] == "hesitation_pause")
    assert pause["gap_idx"] == 2  # the gap after "and", the only non-empty one
    assert "gap_idx" not in label["flaws"][0]  # not a pause label


def test_label_json_times_are_three_decimals(label):
    for flaw in label["flaws"]:
        assert flaw["t_start"] == round(flaw["t_start"], 3)
        assert flaw["t_end"] == round(flaw["t_end"], 3)


def test_label_json_qc_does_not_claim_numbers_it_has_not_got(label):
    # There is no edit to verify on a real recording, so UTMOS and the
    # re-alignment error are null rather than invented.
    assert label["qc"]["utmos_clean"] is None
    assert label["qc"]["realign_max_err_ms"] is None
    assert label["qc"]["passed"] is True


def test_label_json_is_json_serialisable(label):
    assert json.loads(json.dumps(label)) == label


def test_an_annotation_past_the_end_is_a_desynced_file(settings, cfg):
    from rostrum.align.mfa import read_textgrid

    alignment = read_textgrid(ANN1, file_id="tiny")
    with pytest.raises(AnnotationError, match="out of sync"):
        build_label_json(
            "tiny",
            [FlawAnnotation("rushing", 2, 1.0, 99.0)],
            alignment,
            settings,
        )


def test_label_json_renders_events_tsv_rows(label):
    rows = engine.events_rows(label)
    assert rows[0] == ("tiny.wav", 0.20, 0.99, "rushing")
    text = engine.format_events_tsv([label])
    assert text.splitlines()[0] == "tiny.wav\t0.200\t0.990\trushing"
    assert len(text.splitlines()) == 3


# --------------------------------------------------------------------------- #
# Event matching
# --------------------------------------------------------------------------- #


def test_match_events_on_the_fixtures(settings):
    a, _ = read_annotations(ANN1, settings)
    b, _ = read_annotations(ANN2, settings)
    matches, unmatched_a, unmatched_b = match_events(a, b, settings)

    assert [(m.a_idx, m.b_idx, m.same_label) for m in matches] == [(0, 0, True), (2, 1, False)]
    assert unmatched_a == [1]  # ann2 did not mark the hesitation pause
    assert unmatched_b == [2]  # ann1 did not mark the low energy
    assert matches[0].onset_err == pytest.approx(0.08)
    assert matches[0].offset_err == pytest.approx(0.06)


def test_matching_is_one_to_one(settings):
    a = [FlawAnnotation("rushing", 2, 1.0, 2.0)]
    b = [FlawAnnotation("rushing", 2, 1.02, 2.0), FlawAnnotation("monotone", 2, 1.05, 2.0)]
    matches, _, unmatched_b = match_events(a, b, settings)
    assert len(matches) == 1
    assert matches[0].b_idx == 0  # the nearer onset wins
    assert unmatched_b == [1]


def test_matching_does_not_depend_on_input_order(settings):
    a = [FlawAnnotation("rushing", 2, 1.0, 2.0), FlawAnnotation("monotone", 2, 3.0, 4.0)]
    b = [FlawAnnotation("monotone", 2, 3.05, 4.0), FlawAnnotation("rushing", 2, 1.05, 2.0)]
    forward = match_events(a, b, settings)[0]
    reverse = match_events(a, list(reversed(b)), settings)[0]
    assert [(m.a_idx, a[m.a_idx].label) for m in forward] == [
        (m.a_idx, a[m.a_idx].label) for m in reverse
    ]


def test_onset_outside_the_collar_is_not_a_match(settings):
    a = [FlawAnnotation("rushing", 2, 1.0, 2.0)]
    b = [FlawAnnotation("rushing", 2, 1.0 + settings.onset_collar_s + 0.01, 2.0)]
    assert match_events(a, b, settings)[0] == []


def test_the_offset_collar_grows_with_event_length(settings):
    # collar = max(200 ms, 20% of the A event's length): 300 ms is outside it
    # for a 1 s event and inside it for a 3 s one.
    short_a = [FlawAnnotation("rushing", 2, 1.0, 2.0)]
    short_b = [FlawAnnotation("rushing", 2, 1.0, 2.3)]
    assert match_events(short_a, short_b, settings)[0] == []

    long_a = [FlawAnnotation("rushing", 2, 1.0, 4.0)]
    long_b = [FlawAnnotation("rushing", 2, 1.0, 4.3)]
    assert len(match_events(long_a, long_b, settings)[0]) == 1


def test_require_same_label_only_matches_agreeing_events(settings):
    a = [FlawAnnotation("monotone", 2, 1.0, 2.0)]
    b = [FlawAnnotation("dragging", 2, 1.0, 2.0)]
    assert len(match_events(a, b, settings)[0]) == 1
    assert match_events(a, b, settings, require_same_label=True)[0] == []


# --------------------------------------------------------------------------- #
# F1 and kappa
# --------------------------------------------------------------------------- #


def test_event_f1_is_symmetric_under_swapping_the_annotators():
    _, _, f1 = event_f1(2, 3, 5)
    _, _, swapped = event_f1(2, 5, 3)
    assert f1 == pytest.approx(swapped)
    # Which annotator is called the reference must not change the headline
    # number, or it would not be *inter*-annotator agreement.
    assert f1 == pytest.approx(0.5)


def test_event_f1_edge_cases():
    assert event_f1(0, 0, 0) == (1.0, 1.0, 1.0)  # both found nothing: agreement
    assert event_f1(0, 3, 0)[2] == 0.0


def test_cohens_kappa_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    a = ["rushing", "rushing", "monotone", "monotone", "low_energy", "rushing"]
    b = ["rushing", "monotone", "monotone", "monotone", "rushing", "rushing"]
    assert cohens_kappa(a, b) == pytest.approx(sklearn_metrics.cohen_kappa_score(a, b))


def test_cohens_kappa_degenerate_cases():
    import math

    assert math.isnan(cohens_kappa([], []))
    # Both annotators used one label and agreed on all of it: chance agreement
    # is 1 and kappa is formally undefined, but perfect agreement is less
    # misleading reported as perfect than as nan.
    assert cohens_kappa(["rushing"] * 4, ["rushing"] * 4) == 1.0
    assert cohens_kappa(["rushing", "monotone"], ["monotone", "rushing"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="not the same events"):
        cohens_kappa(["rushing"], ["rushing", "monotone"])


# --------------------------------------------------------------------------- #
# The agreement report
# --------------------------------------------------------------------------- #


@pytest.fixture()
def report(settings):
    """Agreement between the two fixture annotators."""
    a, _ = read_annotations(ANN1, settings)
    b, _ = read_annotations(ANN2, settings)
    return agreement(
        {"tiny": (a, b)}, settings, annotator_a="ann1", annotator_b="ann2", n_human_files=3
    )


def test_agreement_numbers_are_the_hand_computed_ones(report):
    assert (report.n_events_a, report.n_events_b, report.n_matched) == (3, 3, 2)
    assert report.f1 == pytest.approx(2 / 3)
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)
    assert report.kappa == pytest.approx(1 / 3)
    assert report.label_agreement == pytest.approx(0.5)
    assert report.onset_mae == pytest.approx(0.04)
    assert report.offset_mae == pytest.approx(0.03)


def test_agreement_reports_the_confusion_and_the_coverage(report, settings):
    assert report.confusion == {("rushing", "rushing"): 1, ("monotone", "dragging"): 1}
    assert report.coverage == pytest.approx(1 / 3)
    assert report.coverage >= settings.min_double_annotated_frac


def test_agreement_keeps_per_file_rows(report):
    assert [row["file_id"] for row in report.per_file] == ["tiny"]
    assert report.per_file[0]["n_matched"] == 2


def test_agreement_to_dict_is_serialisable(report):
    assert json.loads(json.dumps(report.to_dict()))["f1"] == pytest.approx(2 / 3, abs=1e-3)


def test_agreement_markdown_states_the_collars_and_the_numbers(report, settings):
    text = format_agreement_md(report, settings)
    assert "# Inter-annotator agreement" in text
    assert "onset collar 200 ms" in text
    assert "20% of the A event's length" in text
    assert "0.667" in text and "0.333" in text
    assert "monotone" in text and "dragging" in text


def test_empty_agreement_does_not_crash(settings):
    empty = agreement({"tiny": ([], [])}, settings, annotator_a="a", annotator_b="b")
    assert empty.n_matched == 0
    assert empty.f1 == 1.0  # both annotators found nothing, which is agreement
    assert "n/a" in format_agreement_md(empty, settings)


def test_double_annotated_finds_the_pairs():
    pairs = double_annotated({"a": ["f1", "f2"], "b": ["f2", "f3"], "c": ["f9"]})
    assert pairs == {("a", "b"): ["f2"]}


# --------------------------------------------------------------------------- #
# scripts/annotate_prepare.py
# --------------------------------------------------------------------------- #


def test_assign_gives_the_primary_everything():
    assignment = annotate_prepare.assign(["c", "a", "b"], ["ann1"], 0.30)
    assert assignment == {"ann1": ["a", "b", "c"]}


def test_assign_meets_the_double_annotation_quota():
    files = [f"f{i:02d}" for i in range(10)]
    assignment = annotate_prepare.assign(files, ["ann1", "ann2"], 0.30)
    assert assignment["ann1"] == files
    assert len(assignment["ann2"]) == 3  # ceil(0.30 * 10), spec 6.2
    assert set(assignment["ann2"]) <= set(files)


def test_assign_is_deterministic_and_order_independent():
    files = [f"f{i:02d}" for i in range(10)]
    first = annotate_prepare.assign(files, ["ann1", "ann2"], 0.30)
    second = annotate_prepare.assign(list(reversed(files)), ["ann1", "ann2"], 0.30)
    assert first == second


def test_assign_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one annotator"):
        annotate_prepare.assign(["a"], [], 0.3)
    with pytest.raises(ValueError, match="duplicate annotator"):
        annotate_prepare.assign(["a"], ["x", "x"], 0.3)
    with pytest.raises(ValueError, match="not a fraction"):
        annotate_prepare.assign(["a"], ["x"], 1.5)


def test_write_todo_appends_an_empty_flaw_tier(settings, tmp_path):
    dest = annotate_prepare.write_todo(
        FIXTURES / "tiny_long.TextGrid", tmp_path / "t.TextGrid", settings
    )
    tiers = parse_textgrid(dest)
    assert [t.name for t in tiers] == ["words", "phones", "flaws"]
    assert all(not text for _, _, text in tiers[-1].intervals)
    # The words tier is passed through untouched: the annotator has to see the
    # same alignment the labels will be grounded against.
    assert tiers[0].intervals == parse_textgrid(FIXTURES / "tiny_long.TextGrid")[0].intervals


def test_write_todo_refuses_to_blank_an_existing_annotation(settings, tmp_path):
    with pytest.raises(ValueError, match="already has a 'flaws' tier"):
        annotate_prepare.write_todo(ANN1, tmp_path / "t.TextGrid", settings)


# --------------------------------------------------------------------------- #
# scripts/annotate_import.py
# --------------------------------------------------------------------------- #


def test_find_finished_reads_both_layouts(tmp_path):
    (tmp_path / "ann1").mkdir()
    (tmp_path / "ann1" / "b.TextGrid").write_text("x", encoding="utf-8")
    (tmp_path / "a.TextGrid").write_text("x", encoding="utf-8")
    found = annotate_import.find_finished(tmp_path)
    assert [(f.file_id, f.annotator) for f in found] == [("a", ""), ("b", "ann1")]


def test_find_finished_honours_only(tmp_path):
    (tmp_path / "a.TextGrid").write_text("x", encoding="utf-8")
    (tmp_path / "b.TextGrid").write_text("x", encoding="utf-8")
    assert [f.file_id for f in annotate_import.find_finished(tmp_path, {"b"})] == ["b"]


def test_find_finished_on_a_missing_directory(tmp_path):
    assert annotate_import.find_finished(tmp_path / "nope") == []


def test_choose_prefers_the_primary():
    grids = [
        annotate_import.Finished("tiny", "ann1", Path("a")),
        annotate_import.Finished("tiny", "ann2", Path("b")),
    ]
    chosen, others = annotate_import.choose(grids, "ann2")
    assert (chosen.annotator, others) == ("ann2", ["ann1"])
    # No primary named: the choice is arbitrary but stable, not order-dependent.
    assert annotate_import.choose(grids, "")[0].annotator == "ann1"
    assert annotate_import.choose(list(reversed(grids)), "")[0].annotator == "ann1"


def test_worst_level_is_the_file_headline():
    assert annotate_import.worst_level([{"level": "L2"}, {"level": "L4"}, {"level": "L3"}]) == "L4"
    assert annotate_import.worst_level([]) == "L0"


def test_manifest_row_follows_the_injected_convention(label):
    from rostrum.data.manifest import ManifestRow

    existing = ManifestRow(
        file_id="tiny", text_id="s0_tiny", tier="bronze", origin="human", split_fold="fold_s0"
    )
    row = annotate_import.manifest_row(label, existing)
    assert row.origin == "human"
    assert row.label == "composite"  # three flaws, so the labels live in the JSON
    assert row.level == "L4"
    assert row.split_fold == "fold_s0"

    single = dict(label, flaws=[label["flaws"][0]])
    assert annotate_import.manifest_row(single, existing).label == "rushing"


def test_import_writes_label_json_and_a_flaws_alignment(settings, cfg, tmp_path, monkeypatch):
    """End-to-end import of one file, with the dataset redirected to tmp_path."""
    import dataclasses

    local = dataclasses.replace(cfg, root=tmp_path)
    chosen = annotate_import.Finished("tiny", "ann1", ANN1)
    label, annotations = annotate_import.import_one(
        local, chosen, ["ann2"], settings, "s0_tiny", dry_run=False
    )

    written = json.loads((tmp_path / "dataset" / "labels" / "tiny.json").read_text("utf-8"))
    assert written == label
    assert written["origin"] == "human" and written["split"] == "test"

    grid = tmp_path / "dataset" / "alignments" / "tiny.TextGrid"
    tiers = parse_textgrid(grid)
    assert [t.name for t in tiers] == ["words", "phones", "flaws"]
    # Round trip: the published grid re-reads to the same annotations.
    again, problems = read_annotations(grid, settings)
    assert problems == []
    assert again == annotations


def test_import_splits_overlapping_labels_across_tiers(settings, cfg, tmp_path):
    import dataclasses

    local = dataclasses.replace(cfg, root=tmp_path)
    chosen = annotate_import.Finished("tiny", "ann2", ANN2)
    _, annotations = annotate_import.import_one(
        local, chosen, [], settings, "s0_tiny", dry_run=False
    )
    tiers = parse_textgrid(tmp_path / "dataset" / "alignments" / "tiny.TextGrid")
    # low_energy sits inside dragging, so one interval tier cannot hold both.
    assert [t.name for t in tiers] == [
        "words",
        "phones",
        "flaws_dragging",
        "flaws_low_energy",
        "flaws_rushing",
    ]
    again, _ = read_annotations(tmp_path / "dataset" / "alignments" / "tiny.TextGrid", settings)
    assert sorted(again, key=lambda a: (a.t_start, a.label)) == sorted(
        annotations, key=lambda a: (a.t_start, a.label)
    )


def test_import_dry_run_writes_nothing(settings, cfg, tmp_path):
    import dataclasses

    local = dataclasses.replace(cfg, root=tmp_path)
    chosen = annotate_import.Finished("tiny", "ann1", ANN1)
    annotate_import.import_one(local, chosen, [], settings, "s0_tiny", dry_run=True)
    assert not (tmp_path / "dataset").exists()


def test_import_rejects_a_bad_grid(settings, cfg, tmp_path):
    import dataclasses

    local = dataclasses.replace(cfg, root=tmp_path)
    bad = _write_flaws(tmp_path / "bad.TextGrid", [(0.2, 0.99, "too_fast:2")])
    chosen = annotate_import.Finished("bad", "ann1", bad)
    with pytest.raises(AnnotationError, match="not a flaw label"):
        annotate_import.import_one(local, chosen, [], settings, "s0_tiny", dry_run=True)


def test_rebuild_events_pools_every_label_json(label, cfg, tmp_path):
    import dataclasses

    local = dataclasses.replace(cfg, root=tmp_path)
    labels_dir = tmp_path / "dataset" / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "tiny.json").write_text(json.dumps(label), encoding="utf-8")
    (labels_dir / "other.json").write_text(
        json.dumps(dict(label, file_id="other")), encoding="utf-8"
    )
    assert annotate_import.rebuild_events(local) == 6
    rows = (tmp_path / "dataset" / "labels" / "events.tsv").read_text("utf-8").splitlines()
    # sed_eval format, sorted by filename then onset so the file is byte-stable.
    assert rows[0].split("\t") == ["other.wav", "0.200", "0.990", "rushing"]
    assert rows == sorted(rows)


# --------------------------------------------------------------------------- #
# scripts/annotator_agreement.py
# --------------------------------------------------------------------------- #


def test_collect_reads_every_annotator(settings):
    by_annotator, problems = annotator_agreement.collect(DONE, settings)
    assert problems == []
    assert sorted(by_annotator) == ["ann1", "ann2"]
    assert sorted(by_annotator["ann1"]) == ["tiny"]


def test_collect_can_restrict_to_some_annotators(settings):
    by_annotator, _ = annotator_agreement.collect(DONE, settings, keep=["ann1"])
    assert sorted(by_annotator) == ["ann1"]


def test_collect_reports_a_bad_grid_without_hiding_the_good_ones(settings, tmp_path):
    (tmp_path / "ann1").mkdir()
    (tmp_path / "ann2").mkdir()
    _write_flaws(tmp_path / "ann1" / "tiny.TextGrid", [(0.2, 0.99, "rushing:2")])
    _write_flaws(tmp_path / "ann2" / "tiny.TextGrid", [(0.2, 0.99, "too_fast:2")])
    by_annotator, problems = annotator_agreement.collect(tmp_path, settings)
    assert sorted(by_annotator) == ["ann1"]
    assert len(problems) == 1 and "too_fast" in problems[0]


def test_collect_ignores_grids_with_no_annotator(settings, tmp_path):
    _write_flaws(tmp_path / "tiny.TextGrid", [(0.2, 0.99, "rushing:2")])
    by_annotator, _ = annotator_agreement.collect(tmp_path, settings)
    assert by_annotator == {}


def test_agreement_script_runs_on_the_fixtures(tmp_path, capsys):
    """The acceptance criterion: the agreement script runs on the fixtures."""
    out = tmp_path / "agreement.md"
    code = annotator_agreement.main(["--done-dir", str(DONE), "--out", str(out)])
    assert code == 0
    assert "event-F1 0.667" in capsys.readouterr().out

    text = out.read_text(encoding="utf-8")
    assert "Event-F1 @ 200 ms | **0.667**" in text
    assert "Cohen's kappa" in text

    numbers = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert numbers[0]["n_matched"] == 2
    assert numbers[0]["annotator_a"] == "ann1"


def test_agreement_script_without_a_second_annotator(tmp_path):
    (tmp_path / "ann1").mkdir()
    _write_flaws(tmp_path / "ann1" / "tiny.TextGrid", [(0.2, 0.99, "rushing:2")])
    assert (
        annotator_agreement.main(["--done-dir", str(tmp_path), "--out", str(tmp_path / "a.md")])
        == 2
    )


# --------------------------------------------------------------------------- #
# The guide itself
# --------------------------------------------------------------------------- #


def test_the_guide_documents_every_label(cfg, settings):
    guide = cfg.path("data.annotation_guide")
    assert guide.is_file(), f"{guide} is a spec 6.2 deliverable"
    text = guide.read_text(encoding="utf-8")
    for name in settings.labels:
        assert f"`{name}`" in text, f"{name} is not defined in ANNOTATION_GUIDE.md"
    for severity in settings.severity_levels:
        assert f"{settings.separator}{severity}" in text
    assert f"version {settings.guide_version}" in text.lower()
