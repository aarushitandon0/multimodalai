"""Dataset packaging helpers: counts, sample selection, datasheet blocks (spec 7.4)."""

from __future__ import annotations

import pytest

from rostrum.data.manifest import ManifestRow
from rostrum.data.package import (
    AUTOBLOCK_END,
    AUTOBLOCK_START,
    clean_label,
    composition_table,
    label_table,
    manifest_row_from_label,
    render_autoblocks,
    sample_coverage,
    select_sample,
    speaker_slug,
)


def row(
    file_id: str,
    text_id: str,
    origin: str = "clean",
    label: str = "",
    level: str = "",
    tier: str = "silver",
    duration_s: float = 100.0,
) -> ManifestRow:
    return ManifestRow(
        file_id=file_id,
        text_id=text_id,
        tier=tier,
        origin=origin,
        label=label,
        level=level,
        duration_s=duration_s,
    )


@pytest.fixture
def rows() -> list[ManifestRow]:
    out = [row(f"s1_lv0{i}", "s1") for i in range(1, 4)]
    out += [row(f"s2_lv0{i}", "s2") for i in range(1, 4)]
    for lab in ("rushing", "monotone"):
        for lvl in ("L1", "L3"):
            out.append(row(f"s1_lv01_{lab}_{lvl}", "s1", "injected", lab, lvl, duration_s=100.0))
    out.append(row("s1_tm01_mumbling_L2", "s1", "human", "mumbling", "L2", tier="bronze"))
    out.append(row("s2_tm01_rushing_L3", "s2", "human", "rushing", "L3", tier="bronze"))
    return out


# --------------------------------------------------------------------------- #
# manifest_row_from_label
# --------------------------------------------------------------------------- #


def test_single_flaw_label_json_becomes_one_manifest_row() -> None:
    label = {
        "file_id": "s1_lv07_rushing_L3",
        "text_id": "s1_gettysburg",
        "origin": "injected",
        "tier": "silver",
        "flaws": [{"label": "rushing", "level": "L3", "t_start": 1.0, "t_end": 2.0}],
    }
    got = manifest_row_from_label(label, duration_s=120.5)
    assert got == ManifestRow(
        file_id="s1_lv07_rushing_L3",
        text_id="s1_gettysburg",
        tier="silver",
        origin="injected",
        label="rushing",
        level="L3",
        duration_s=120.5,
    )


def test_multi_flaw_label_json_is_recorded_as_composite() -> None:
    label = {
        "file_id": "s1_lv07_composite_L4",
        "text_id": "s1_gettysburg",
        "origin": "injected",
        "tier": "silver",
        "flaws": [
            {"label": "rushing", "level": "L4"},
            {"label": "monotone", "level": "L4"},
        ],
    }
    assert manifest_row_from_label(label, duration_s=1.0).label == "composite"


def test_clean_label_json_has_no_label_or_level() -> None:
    label = {
        "file_id": "s1_lv07",
        "text_id": "s1",
        "origin": "clean",
        "tier": "silver",
        "flaws": [],
    }
    got = manifest_row_from_label(label, duration_s=1.0)
    assert (got.label, got.level) == ("", "")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #


def test_composition_table_counts_files_and_duration_per_origin(rows: list[ManifestRow]) -> None:
    table = composition_table(rows)
    assert "| `audio/clean/` | 6 |" in table
    assert "| `audio/injected/` | 4 |" in table
    assert "| `audio/human_flawed/` | 2 |" in table


def test_composition_table_totals_every_row(rows: list[ManifestRow]) -> None:
    assert "| **Total** | 12 |" in composition_table(rows)


def test_composition_table_marks_an_empty_dataset() -> None:
    assert "no files" in composition_table([]).lower()


def test_label_table_is_a_label_by_level_grid(rows: list[ManifestRow]) -> None:
    table = label_table(rows)
    assert "| rushing | 1 | 0 | 1 | 0 | 1 |" in table  # L1..L4 then human
    assert "| mumbling | 0 | 0 | 0 | 0 | 1 |" in table


# --------------------------------------------------------------------------- #
# Sample selection
# --------------------------------------------------------------------------- #


def test_sample_covers_every_label_and_level_pair(rows: list[ManifestRow]) -> None:
    chosen = set(select_sample(rows, max_files=20))
    for lab in ("rushing", "monotone"):
        for lvl in ("L1", "L3"):
            assert f"s1_lv01_{lab}_{lvl}" in chosen


def test_sample_includes_two_human_files(rows: list[ManifestRow]) -> None:
    chosen = select_sample(rows, max_files=20)
    humans = [r.file_id for r in rows if r.origin == "human" and r.file_id in chosen]
    assert len(humans) == 2


def test_sample_includes_enough_clean_references_to_build_an_envelope(
    rows: list[ManifestRow],
) -> None:
    chosen = select_sample(rows, max_files=20, min_clean_per_text=2)
    for text_id in ("s1", "s2"):
        clean = [
            r.file_id
            for r in rows
            if r.origin == "clean" and r.text_id == text_id and r.file_id in chosen
        ]
        assert len(clean) >= 2, text_id


def test_sample_is_sorted_and_order_independent(rows: list[ManifestRow]) -> None:
    forward = select_sample(rows, max_files=20)
    backward = select_sample(list(reversed(rows)), max_files=20)
    assert forward == backward == sorted(forward)


def test_sample_respects_the_file_cap(rows: list[ManifestRow]) -> None:
    assert len(select_sample(rows, max_files=8)) <= 8


def test_cap_keeps_flaw_coverage_before_extra_clean_files(rows: list[ManifestRow]) -> None:
    chosen = set(select_sample(rows, max_files=8))
    assert "s1_lv01_rushing_L1" in chosen


def test_clean_only_dataset_samples_every_text(rows: list[ManifestRow]) -> None:
    clean_only = [r for r in rows if r.origin == "clean"]
    chosen = select_sample(clean_only, max_files=20)
    assert {r.text_id for r in clean_only if r.file_id in chosen} == {"s1", "s2"}


def test_coverage_report_lists_missing_pairs(rows: list[ManifestRow]) -> None:
    chosen = select_sample(rows, max_files=20)
    report = sample_coverage(rows, chosen)
    assert report.missing_pairs == ()
    assert report.n_human == 2
    assert report.complete is False  # L2 and L4 were never injected at all


def test_coverage_is_complete_when_every_label_and_level_exists() -> None:
    labels = (
        "rushing",
        "dragging",
        "missing_pause",
        "hesitation_pause",
        "monotone",
        "misplaced_stress",
        "low_energy",
    )
    rows = [row(f"s{i}_lv01", f"s{i}") for i in (1, 2)]
    rows += [
        row(f"s1_lv01_{lab}_{lvl}", "s1", "injected", lab, lvl)
        for lab in labels
        for lvl in ("L1", "L2", "L3", "L4")
    ]
    rows += [row(f"s{i}_tm01_mumbling_L2", f"s{i}", "human", "mumbling", "L2") for i in (1, 2)]
    chosen = select_sample(rows, max_files=40)
    assert sample_coverage(rows, chosen).complete is True


# --------------------------------------------------------------------------- #
# Datasheet auto-blocks
# --------------------------------------------------------------------------- #


def test_autoblock_content_is_replaced_in_place() -> None:
    doc = f"before\n{AUTOBLOCK_START('counts')}\nstale\n{AUTOBLOCK_END('counts')}\nafter\n"
    out = render_autoblocks(doc, {"counts": "fresh"})
    assert "stale" not in out
    assert "fresh" in out
    assert out.startswith("before\n") and out.endswith("after\n")


def test_rendering_is_idempotent() -> None:
    doc = f"{AUTOBLOCK_START('counts')}\nstale\n{AUTOBLOCK_END('counts')}\n"
    once = render_autoblocks(doc, {"counts": "fresh"})
    assert render_autoblocks(once, {"counts": "fresh"}) == once


def test_unknown_block_name_is_an_error() -> None:
    doc = f"{AUTOBLOCK_START('counts')}\n{AUTOBLOCK_END('counts')}\n"
    with pytest.raises(KeyError, match="qc"):
        render_autoblocks(doc, {"qc": "x"})


def test_missing_marker_in_the_document_is_an_error() -> None:
    with pytest.raises(ValueError, match="counts"):
        render_autoblocks("no markers here\n", {"counts": "x"})


# --------------------------------------------------------------------------- #
# Size budget
# --------------------------------------------------------------------------- #

MB = 1024 * 1024


def test_budget_drops_clean_files_until_the_sample_fits(rows: list[ManifestRow]) -> None:
    sizes = dict.fromkeys((r.file_id for r in rows), 4 * MB)
    chosen = select_sample(rows, max_files=20, sizes=sizes, budget_bytes=30 * MB)
    assert sum(sizes[f] for f in chosen) <= 30 * MB


def test_budget_keeps_flaw_coverage_and_human_takes(rows: list[ManifestRow]) -> None:
    sizes = dict.fromkeys((r.file_id for r in rows), 4 * MB)
    chosen = set(select_sample(rows, max_files=20, sizes=sizes, budget_bytes=30 * MB))
    assert "s1_lv01_rushing_L1" in chosen
    assert "s1_tm01_mumbling_L2" in chosen


def test_budget_too_small_for_coverage_still_returns_the_pairs(
    rows: list[ManifestRow],
) -> None:
    """An impossible budget yields the flawed files and their parents, nothing else."""
    sizes = dict.fromkeys((r.file_id for r in rows), 4 * MB)
    chosen = select_sample(rows, max_files=20, sizes=sizes, budget_bytes=1 * MB)
    flawed = {r.file_id for r in rows if r.origin in ("injected", "human")}
    assert set(chosen) == flawed | {"s1_lv01"}


def test_unknown_file_size_is_not_charged_against_the_budget(rows: list[ManifestRow]) -> None:
    chosen = select_sample(rows, max_files=20, sizes={}, budget_bytes=1)
    assert len(chosen) == 12


def test_no_budget_means_no_trimming(rows: list[ManifestRow]) -> None:
    sizes = dict.fromkeys((r.file_id for r in rows), 4 * MB)
    assert len(select_sample(rows, max_files=20, sizes=sizes)) == 12


# --------------------------------------------------------------------------- #
# Pairing: a flawed file is only useful next to the reading it was cut from
# --------------------------------------------------------------------------- #


def test_pair_representative_prefers_the_smallest_file() -> None:
    """Two files cover rushing/L1; the sample takes the one that costs less."""
    rows = [
        row("s1_lv01", "s1", duration_s=400.0),
        row("s5_lv01", "s5", duration_s=20.0),
        row("s1_lv01_rushing_L1", "s1", "injected", "rushing", "L1", duration_s=400.0),
        row("s5_lv01_rushing_L1", "s5", "injected", "rushing", "L1", duration_s=20.0),
    ]
    sizes = {"s1_lv01": 40 * MB, "s5_lv01": 2 * MB}
    sizes |= {"s1_lv01_rushing_L1": 40 * MB, "s5_lv01_rushing_L1": 2 * MB}
    chosen = select_sample(rows, max_files=20, sizes=sizes, budget_bytes=45 * MB)
    assert "s5_lv01_rushing_L1" in chosen
    assert "s1_lv01_rushing_L1" not in chosen


def test_sample_includes_the_clean_parent_of_every_chosen_injected_file(
    rows: list[ManifestRow],
) -> None:
    """Every injected file in the sample is shipped next to its clean reading."""
    chosen = set(select_sample(rows, max_files=20))
    clean = {r.file_id for r in rows if r.origin == "clean"}
    injected = [r.file_id for r in rows if r.origin == "injected" and r.file_id in chosen]
    assert injected
    for file_id in injected:
        parents = [c for c in clean if file_id.startswith(c)]
        assert max(parents, key=len) in chosen


def test_parents_are_protected_from_the_budget_trim() -> None:
    """A budget too small for the clean round-robin still keeps the pair whole.

    ``s1_lv01`` sorts first, so the unprotected round-robin would spend the
    budget on it and drop ``s1_lv02`` - the reading the flawed file came from.
    """
    rows = [
        row("s1_lv01", "s1"),
        row("s1_lv02", "s1"),
        row("s1_lv02_rushing_L1", "s1", "injected", "rushing", "L1"),
    ]
    sizes = dict.fromkeys((r.file_id for r in rows), 4 * MB)
    chosen = set(select_sample(rows, max_files=20, sizes=sizes, budget_bytes=9 * MB))
    assert {"s1_lv02", "s1_lv02_rushing_L1"} <= chosen


# --------------------------------------------------------------------------- #
# clean_label: a good reading needs a label record that says "no flaws"
# --------------------------------------------------------------------------- #


def test_clean_label_declares_an_empty_flaw_list() -> None:
    label = clean_label(row("s5_lv01", "s5_im_nobody", duration_s=39.925))
    assert label["flaws"] == []
    assert label["origin"] == "clean"
    assert label["file_id"] == "s5_lv01"
    assert label["text_id"] == "s5_im_nobody"


def test_clean_label_is_its_own_channel() -> None:
    """A clean reading is the channel other files are compared against (invariant 7)."""
    label = clean_label(row("s5_lv01", "s5_im_nobody"))
    assert label["channel_id"] == "s5_lv01"
    assert label["source_file"] == ""


def test_clean_label_carries_the_speaker_and_digest_when_known() -> None:
    label = clean_label(
        row("s5_lv01", "s5_im_nobody"), speaker_id="algy_pug", audio_sha256="abc123"
    )
    assert label["speaker_id"] == "algy_pug"
    assert label["audio_sha256"] == "abc123"


def test_clean_label_round_trips_through_the_manifest() -> None:
    """``manifest_row_from_label`` must read back what ``clean_label`` wrote."""
    original = row("s5_lv01", "s5_im_nobody", tier="gold", duration_s=39.925)
    back = manifest_row_from_label(clean_label(original), 39.925)
    assert (back.file_id, back.text_id, back.origin, back.tier) == (
        "s5_lv01",
        "s5_im_nobody",
        "clean",
        "gold",
    )
    assert (back.label, back.level) == ("", "")


# --------------------------------------------------------------------------- #
# speaker_slug
# --------------------------------------------------------------------------- #


def test_speaker_slug_lowercases_and_underscores_a_display_name() -> None:
    assert speaker_slug("Alan Davis-Drake") == "alan_davis-drake"
    assert speaker_slug("Lee Ann Howlett") == "lee_ann_howlett"


def test_speaker_slug_drops_abbreviation_dots() -> None:
    """``Franklin D. Roosevelt`` must not become ``franklin_d._roosevelt``."""
    assert speaker_slug("Franklin D. Roosevelt") == "franklin_d_roosevelt"
    assert speaker_slug("Lucretia B.") == "lucretia_b"


def test_speaker_slug_drops_parenthesised_dates() -> None:
    assert speaker_slug("Ray Kasper (1938-2017)") == "ray_kasper"


@pytest.mark.parametrize(
    "name",
    ["MaryAnn", "Franklin D. Roosevelt", "Ray Kasper (1938-2017)", "Alan Davis-Drake"],
)
def test_speaker_slug_is_idempotent(name: str) -> None:
    """Re-slugging a slug must not grow separators: ``franklin_d__roosevelt``."""
    once = speaker_slug(name)
    assert speaker_slug(once) == once


def test_speaker_slug_never_doubles_a_separator() -> None:
    assert speaker_slug("franklin_d._roosevelt") == "franklin_d_roosevelt"


def test_speaker_slug_of_nothing_is_empty() -> None:
    assert speaker_slug("") == ""
    assert speaker_slug("   ") == ""


def test_clean_label_stores_the_speaker_as_a_slug() -> None:
    label = clean_label(row("s5_lv01", "s5"), speaker_id="Alan Davis-Drake")
    assert label["speaker_id"] == "alan_davis-drake"


# --------------------------------------------------------------------------- #
# composition_table speaker count
# --------------------------------------------------------------------------- #


def test_composition_counts_distinct_speakers_not_recordings() -> None:
    """Two readings by one reader are one speaker (the label JSONs know this)."""
    rows = [row("s1_lv01", "s1"), row("s6_lv01", "s6")]
    speakers = {"s1_lv01": "algy_pug", "s6_lv01": "algy_pug"}
    table = composition_table(rows, speakers=speakers)
    header, _, clean, *_ = table.splitlines()
    assert "Speakers" in header
    assert clean.split("|")[4].strip() == "1"


def test_composition_falls_back_to_the_recording_prefix() -> None:
    """Without a speaker map the column still reports something defensible."""
    rows = [row("s1_lv01", "s1"), row("s6_lv01", "s6")]
    clean = composition_table(rows).splitlines()[2]
    assert clean.split("|")[4].strip() == "2"


def test_injected_speaker_comes_from_its_parent() -> None:
    rows = [
        row("s1_lv01", "s1"),
        row("s1_lv01_rushing_L1", "s1", "injected", "rushing", "L1"),
    ]
    speakers = {"s1_lv01": "algy_pug", "s1_lv01_rushing_L1": "algy_pug"}
    total = composition_table(rows, speakers=speakers).splitlines()[-1]
    assert total.split("|")[4].strip() == "1"
