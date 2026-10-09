"""Leave-one-text-out split builder (spec 7.3)."""

from __future__ import annotations

import json

import pytest

from rostrum.data.manifest import ManifestRow
from rostrum.data.splits import (
    LeakageError,
    build_loto,
    check_no_leakage,
    fold_id,
    load_loto,
    with_split_folds,
    write_loto,
)


def row(file_id: str, text_id: str, origin: str = "clean", **kw: object) -> ManifestRow:
    return ManifestRow(
        file_id=file_id,
        text_id=text_id,
        tier=str(kw.pop("tier", "silver")),
        origin=origin,
        label=str(kw.pop("label", "")),
        level=str(kw.pop("level", "")),
        duration_s=kw.pop("duration_s", 10.0),  # type: ignore[arg-type]
    )


@pytest.fixture
def rows() -> list[ManifestRow]:
    return [
        row("s1_a", "s1"),
        row("s1_a_rushing_L3", "s1", origin="injected", label="rushing", level="L3"),
        row("s1_h01", "s1", origin="human", label="mumbling", level="L2", tier="bronze"),
        row("s2_a", "s2"),
        row("s2_h01", "s2", origin="human", label="mumbling", level="L2", tier="bronze"),
        row("s3_a", "s3"),
    ]


def test_one_fold_per_text(rows: list[ManifestRow]) -> None:
    splits = build_loto(rows)
    assert [f.held_out_text for f in splits.folds] == ["s1", "s2", "s3"]


def test_fold_trains_on_every_other_text(rows: list[ManifestRow]) -> None:
    fold = build_loto(rows).fold("s1")
    assert fold.train_texts == ("s2", "s3")
    assert fold.test_texts == ("s1",)


def test_test_set_is_exactly_the_held_out_text(rows: list[ManifestRow]) -> None:
    fold = build_loto(rows).fold("s1")
    assert fold.test == ("s1_a", "s1_a_rushing_L3", "s1_h01")


def test_human_files_never_appear_in_train(rows: list[ManifestRow]) -> None:
    for fold in build_loto(rows).folds:
        assert "s1_h01" not in fold.train
        assert "s2_h01" not in fold.train


def test_human_file_appears_only_in_its_own_test_fold(rows: list[ManifestRow]) -> None:
    splits = build_loto(rows)
    appears_in = [f.held_out_text for f in splits.folds if "s2_h01" in f.test]
    assert appears_in == ["s2"]


def test_train_holds_the_non_human_files_of_the_other_texts(rows: list[ManifestRow]) -> None:
    fold = build_loto(rows).fold("s1")
    assert fold.train == ("s2_a", "s3_a")


def test_train_and_test_never_share_a_file(rows: list[ManifestRow]) -> None:
    for fold in build_loto(rows).folds:
        assert not set(fold.train) & set(fold.test)


def test_fold_id_is_derived_from_the_held_out_text(rows: list[ManifestRow]) -> None:
    assert fold_id("s1_gettysburg") == "loto_s1_gettysburg"
    assert build_loto(rows).fold("s1").fold_id == "loto_s1"


def test_single_text_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="at least two texts"):
        build_loto([row("s1_a", "s1")])


def test_rows_without_a_text_id_are_rejected() -> None:
    with pytest.raises(ValueError, match="text_id"):
        build_loto([row("x", ""), row("s2_a", "s2")])


def test_split_fold_column_is_the_fold_the_file_is_tested_in(rows: list[ManifestRow]) -> None:
    updated = with_split_folds(rows)
    assert {r.file_id: r.split_fold for r in updated} == {
        "s1_a": "loto_s1",
        "s1_a_rushing_L3": "loto_s1",
        "s1_h01": "loto_s1",
        "s2_a": "loto_s2",
        "s2_h01": "loto_s2",
        "s3_a": "loto_s3",
    }


def test_leakage_check_rejects_a_human_file_in_train(rows: list[ManifestRow]) -> None:
    splits = build_loto(rows)
    bad = splits.folds[0]
    object.__setattr__(bad, "train", (*bad.train, "s2_h01"))
    with pytest.raises(LeakageError, match="human"):
        check_no_leakage(splits, rows)


def test_leakage_check_rejects_a_file_in_both_sides(rows: list[ManifestRow]) -> None:
    splits = build_loto(rows)
    bad = splits.folds[0]
    object.__setattr__(bad, "train", (*bad.train, "s1_a"))
    with pytest.raises(LeakageError, match="both"):
        check_no_leakage(splits, rows)


def test_clean_split_passes_the_leakage_check(rows: list[ManifestRow]) -> None:
    splits = build_loto(rows)
    check_no_leakage(splits, rows)  # does not raise


def test_round_trips_through_json(tmp_path, rows: list[ManifestRow]) -> None:
    path = tmp_path / "loto.json"
    splits = build_loto(rows)
    write_loto(path, splits)
    assert load_loto(path).to_dict() == splits.to_dict()


def test_written_json_is_byte_identical_across_runs(tmp_path, rows: list[ManifestRow]) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    write_loto(a, build_loto(rows))
    write_loto(b, build_loto(list(reversed(rows))))
    assert a.read_bytes() == b.read_bytes()


def test_json_records_the_test_only_rule(tmp_path, rows: list[ManifestRow]) -> None:
    path = tmp_path / "loto.json"
    write_loto(path, build_loto(rows))
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["protocol"] == "leave_one_text_out"
    assert doc["test_only_origins"] == ["human"]
