"""Cross-file consistency of the shipped dataset (spec 7.4).

Every check here failed at some point during the build and nothing caught it:
clean readings with no label record, 266 alignments the packager dropped
because it only matched ``*.TextGrid``, a ``rejections.csv`` whose rows named
files that are in the dataset, and speaker ids that were ``Alan Davis-Drake``
in one file and ``alan_davis-drake`` in the next. They are consistency
assertions over whatever is on disk, so they also pass in a fresh clone, where
only ``dataset_sample/`` and the text sidecars exist.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from rostrum.data.manifest import parent_file_id, read_manifest
from rostrum.data.package import speaker_slug

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "dataset"
SAMPLE = ROOT / "dataset_sample"
TAB = "\t"


def _roots() -> list[Path]:
    return [d for d in (FULL, SAMPLE) if (d / "MANIFEST.csv").is_file()]


@pytest.fixture(params=_roots(), ids=lambda d: d.name)
def dataset(request: pytest.FixtureRequest) -> Path:
    return request.param


def _label(dataset: Path, file_id: str) -> dict:
    path = dataset / "labels" / f"{file_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Manifest vs disk
# --------------------------------------------------------------------------- #


def test_manifest_and_audio_agree(dataset: Path) -> None:
    rows = read_manifest(dataset / "MANIFEST.csv")
    assert {r.file_id for r in rows} == {p.stem for p in (dataset / "audio").rglob("*.wav")}


def test_file_ids_are_unique(dataset: Path) -> None:
    ids = [r.file_id for r in read_manifest(dataset / "MANIFEST.csv")]
    assert len(ids) == len(set(ids))


def test_every_audio_file_has_a_label_record(dataset: Path) -> None:
    """A clean reading asserting "no flaws" is a claim; a missing file is not."""
    rows = read_manifest(dataset / "MANIFEST.csv")
    missing = [r.file_id for r in rows if not (dataset / "labels" / f"{r.file_id}.json").is_file()]
    assert missing == []


def test_clean_rows_declare_no_flaw_in_their_label_json(dataset: Path) -> None:
    for row in read_manifest(dataset / "MANIFEST.csv"):
        if row.origin != "clean":
            continue
        data = _label(dataset, row.file_id)
        assert data["flaws"] == [], row.file_id
        assert data["origin"] == "clean", row.file_id


def test_injected_rows_have_at_least_one_flaw(dataset: Path) -> None:
    for row in read_manifest(dataset / "MANIFEST.csv"):
        if row.origin == "injected":
            assert _label(dataset, row.file_id)["flaws"], row.file_id


def test_every_derived_file_has_an_alignment(dataset: Path) -> None:
    """Clean files carry a TextGrid, injected files a JSON word table (spec 7.2)."""
    align = dataset / "alignments"
    stems = {p.stem for p in align.glob("*") if p.suffix in (".TextGrid", ".json")}
    if not stems:
        pytest.skip("no alignments shipped in this root")
    rows = read_manifest(dataset / "MANIFEST.csv")
    clean = sorted(r.file_id for r in rows if r.origin == "clean")
    derived = [r.file_id for r in rows if r.origin != "clean"]
    assert set(derived) <= stems
    # Only the clean readings something was injected from must be aligned; the
    # other 53 are an open task, recorded in dataset/texts/README.md.
    parents = {parent_file_id(f, clean) for f in derived} - {""}
    assert parents <= stems


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #


def test_every_derived_file_names_a_parent_that_ships(dataset: Path) -> None:
    rows = read_manifest(dataset / "MANIFEST.csv")
    clean = sorted(r.file_id for r in rows if r.origin == "clean")
    for row in rows:
        if row.origin != "clean":
            assert parent_file_id(row.file_id, clean), row.file_id


def test_label_json_source_file_matches_the_name_based_parent(dataset: Path) -> None:
    rows = read_manifest(dataset / "MANIFEST.csv")
    clean = sorted(r.file_id for r in rows if r.origin == "clean")
    for row in rows:
        if row.origin != "injected":
            continue
        got = _label(dataset, row.file_id)["source_file"]
        assert got == parent_file_id(row.file_id, clean), row.file_id


def test_a_derived_file_shares_its_parents_text(dataset: Path) -> None:
    rows = {r.file_id: r for r in read_manifest(dataset / "MANIFEST.csv")}
    clean = sorted(f for f, r in rows.items() if r.origin == "clean")
    for file_id, row in rows.items():
        if row.origin == "clean":
            continue
        assert row.text_id == rows[parent_file_id(file_id, clean)].text_id, file_id


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def test_flaw_spans_are_inside_the_file(dataset: Path) -> None:
    for row in read_manifest(dataset / "MANIFEST.csv"):
        for flaw in _label(dataset, row.file_id)["flaws"]:
            assert 0.0 <= flaw["t_start"] <= flaw["t_end"], row.file_id
            assert flaw["t_end"] <= float(row.duration_s) + 0.001, (row.file_id, flaw)


def test_flaw_times_are_rounded_to_three_decimals(dataset: Path) -> None:
    """Invariant 2: seconds, float, 3 decimals, or the output hash will not reproduce."""
    for path in sorted((dataset / "labels").glob("*.json")):
        for flaw in json.loads(path.read_text(encoding="utf-8"))["flaws"]:
            for key in ("t_start", "t_end"):
                assert round(flaw[key], 3) == flaw[key], (path.name, key, flaw[key])


def test_manifest_label_and_level_match_the_label_json(dataset: Path) -> None:
    for row in read_manifest(dataset / "MANIFEST.csv"):
        flaws = _label(dataset, row.file_id)["flaws"]
        if not flaws:
            assert (row.label, row.level) == ("", ""), row.file_id
        elif len({f["label"] for f in flaws}) == 1:
            assert row.label == flaws[0]["label"], row.file_id
        else:
            assert row.label == "composite", row.file_id


def test_events_tsv_covers_every_labelled_flaw(dataset: Path) -> None:
    events = dataset / "labels" / "events.tsv"
    if not events.is_file():
        pytest.skip("no events.tsv in this root")
    lines = [ln.split(TAB) for ln in events.read_text(encoding="utf-8").splitlines() if ln.strip()]
    seen = {(r[0].removesuffix(".wav"), r[3]) for r in lines}
    for path in sorted((dataset / "labels").glob("*.json")):
        for flaw in json.loads(path.read_text(encoding="utf-8"))["flaws"]:
            assert (path.stem, flaw["label"]) in seen, (path.stem, flaw["label"])


# --------------------------------------------------------------------------- #
# Speakers and splits
# --------------------------------------------------------------------------- #


def test_speaker_ids_are_slugs(dataset: Path) -> None:
    for path in sorted((dataset / "labels").glob("*.json")):
        speaker = json.loads(path.read_text(encoding="utf-8")).get("speaker_id") or ""
        assert speaker == speaker_slug(speaker), (path.name, speaker)


def test_a_derived_file_shares_its_parents_speaker(dataset: Path) -> None:
    rows = read_manifest(dataset / "MANIFEST.csv")
    clean = sorted(r.file_id for r in rows if r.origin == "clean")
    for row in rows:
        if row.origin != "injected":
            continue
        mine = _label(dataset, row.file_id).get("speaker_id") or ""
        theirs = _label(dataset, parent_file_id(row.file_id, clean)).get("speaker_id") or ""
        assert mine == theirs, row.file_id


def test_every_file_is_tested_in_exactly_its_own_texts_fold(dataset: Path) -> None:
    splits = dataset / "splits" / "loto.json"
    if not splits.is_file():
        pytest.skip("no splits in this root")
    folds = json.loads(splits.read_text(encoding="utf-8"))["folds"]
    by_text = {f["held_out_text"]: f["fold_id"] for f in folds}
    for row in read_manifest(dataset / "MANIFEST.csv"):
        if row.text_id in by_text:
            assert row.split_fold == by_text[row.text_id], row.file_id


def test_no_file_is_on_both_sides_of_a_fold(dataset: Path) -> None:
    splits = dataset / "splits" / "loto.json"
    if not splits.is_file():
        pytest.skip("no splits in this root")
    for fold in json.loads(splits.read_text(encoding="utf-8"))["folds"]:
        assert not set(fold["train"]) & set(fold["test"]), fold["fold_id"]


# --------------------------------------------------------------------------- #
# Provenance and QC bookkeeping
# --------------------------------------------------------------------------- #


def test_every_clean_file_has_a_licence_row(dataset: Path) -> None:
    path = dataset / "LICENSES.csv"
    if not path.is_file():
        pytest.skip("no LICENSES.csv in this root")
    with path.open(encoding="utf-8") as fh:
        licensed = {r["file_id"] for r in csv.DictReader(fh)}
    clean = {r.file_id for r in read_manifest(dataset / "MANIFEST.csv") if r.origin == "clean"}
    assert clean <= licensed


def test_rejections_resolve_against_what_shipped() -> None:
    """``retried_ok`` must name a file that ships and ``dropped`` one that does not."""
    path = FULL / "qc" / "rejections.csv"
    if not path.is_file():
        pytest.skip("no rejections.csv")
    shipped = {r.file_id for r in read_manifest(FULL / "MANIFEST.csv")}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "rejections.csv is empty"
    for row in rows:
        assert row["final_status"] in ("retried_ok", "dropped"), row
        assert (row["file_id"] in shipped) == (row["final_status"] == "retried_ok"), row


def test_every_clean_file_has_a_build_row() -> None:
    path = FULL / "qc" / "clean_build.csv"
    if not path.is_file():
        pytest.skip("no clean_build.csv")
    with path.open(encoding="utf-8") as fh:
        built = {r["file_id"] for r in csv.DictReader(fh)}
    clean = {r.file_id for r in read_manifest(FULL / "MANIFEST.csv") if r.origin == "clean"}
    assert clean <= built
