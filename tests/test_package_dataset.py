"""Tests for ``scripts/package_dataset.py`` (spec 7.4 acceptance).

A miniature dataset is built in ``tmp_path`` - two texts, clean plus injected
plus human audio, label JSONs, alignments - so the packager is covered without
MFA, without UTMOS and without the real dataset.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import package_dataset as pkg
from rostrum.io.audio import write_wav16

SR = 16000


def _wav(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    write_wav16(path, 0.1 * np.sin(2 * np.pi * 200.0 * t).astype(np.float32), SR)


def _label_json(path: Path, file_id: str, text_id: str, origin: str, flaws: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "file_id": file_id,
                "text_id": text_id,
                "speaker_id": "sp",
                "origin": origin,
                "tier": "bronze" if origin == "human" else "silver",
                "flaws": flaws,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A two-text dataset: 4 clean, 2 injected, 1 human."""
    d = tmp_path / "dataset"
    for text_id, stems in (("s1", ("s1_lv01", "s1_lv02")), ("s2", ("s2_lv01", "s2_lv02"))):
        for stem in stems:
            _wav(d / "audio" / "clean" / f"{stem}.wav", 2.0)
            _label_json(d / "labels" / f"{stem}.json", stem, text_id, "clean", [])
        (d / "texts").mkdir(parents=True, exist_ok=True)
        (d / "texts" / f"{text_id}.json").write_text(
            json.dumps({"text_id": text_id, "verbatim": "hello world", "display": "Hello world."}),
            encoding="utf-8",
        )
    for stem, lab, lvl in (
        ("s1_lv01_rushing_L3", "rushing", "L3"),
        ("s1_lv01_monotone_L1", "monotone", "L1"),
    ):
        _wav(d / "audio" / "injected" / f"{stem}.wav", 2.0)
        _label_json(
            d / "labels" / f"{stem}.json",
            stem,
            "s1",
            "injected",
            [{"label": lab, "level": lvl, "t_start": 0.5, "t_end": 1.0, "s": 0.6}],
        )
        (d / "alignments").mkdir(parents=True, exist_ok=True)
        (d / "alignments" / f"{stem}.TextGrid").write_text("fake textgrid", encoding="utf-8")
    _wav(d / "audio" / "human_flawed" / "s2_tm01_mumbling_L2.wav", 2.0)
    _label_json(
        d / "labels" / "s2_tm01_mumbling_L2.json",
        "s2_tm01_mumbling_L2",
        "s2",
        "human",
        [{"label": "mumbling", "level": "L2", "t_start": 0.2, "t_end": 1.2, "s": 0.35}],
    )
    (d / "LICENSES.csv").write_text(
        "file_id,title,speaker,source_url,licence,retrieved_on,sha256\n"
        + "".join(
            f"{fid},T,S,http://x,Public domain,2026-01-01,deadbeef\n"
            for fid in ("s1_lv01", "s1_lv02", "s2_lv01", "s2_lv02", "s2_tm01_mumbling_L2")
        ),
        encoding="utf-8",
    )
    return d


# --------------------------------------------------------------------------- #
# Manifest regeneration
# --------------------------------------------------------------------------- #


def test_manifest_has_one_row_per_audio_file(dataset: Path) -> None:
    rows = pkg.collect_rows(dataset)
    assert len(rows) == 7


def test_origin_comes_from_the_audio_directory(dataset: Path) -> None:
    by_id = {r.file_id: r for r in pkg.collect_rows(dataset)}
    assert by_id["s1_lv01"].origin == "clean"
    assert by_id["s1_lv01_rushing_L3"].origin == "injected"
    assert by_id["s2_tm01_mumbling_L2"].origin == "human"


def test_label_and_level_come_from_the_label_json(dataset: Path) -> None:
    by_id = {r.file_id: r for r in pkg.collect_rows(dataset)}
    assert (by_id["s1_lv01_rushing_L3"].label, by_id["s1_lv01_rushing_L3"].level) == (
        "rushing",
        "L3",
    )
    assert by_id["s1_lv01"].label == ""


def test_duration_is_measured_from_the_wav(dataset: Path) -> None:
    by_id = {r.file_id: r for r in pkg.collect_rows(dataset)}
    assert float(by_id["s1_lv01"].duration_s) == pytest.approx(2.0, abs=0.01)


def test_text_id_falls_back_to_the_previous_manifest_row(dataset: Path) -> None:
    (dataset / "labels" / "s1_lv02.json").unlink()
    (dataset / "MANIFEST.csv").write_text(
        "file_id,text_id,tier,origin,label,level,duration_s,split_fold\n"
        "s1_lv02,s1,silver,clean,,,2.000,\n",
        encoding="utf-8",
    )
    by_id = {r.file_id: r for r in pkg.collect_rows(dataset)}
    assert (by_id["s1_lv02"].text_id, by_id["s1_lv02"].tier) == ("s1", "silver")


def test_file_with_no_text_id_anywhere_is_reported(dataset: Path) -> None:
    _wav(dataset / "audio" / "clean" / "orphan.wav", 1.0)
    with pytest.raises(pkg.PackagingError, match="orphan"):
        pkg.collect_rows(dataset, strict=True)


def test_split_fold_is_filled_in_the_manifest(dataset: Path) -> None:
    rows = pkg.collect_rows(dataset)
    folded = {r.file_id: r.split_fold for r in pkg.with_split_folds(rows)}
    assert folded["s1_lv01"] == "loto_s1"
    assert folded["s2_tm01_mumbling_L2"] == "loto_s2"


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def test_checksums_cover_every_shipped_file(dataset: Path) -> None:
    n = pkg.regenerate_checksums(dataset)
    lines = (dataset / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    assert n == len(lines)
    paths = {line.split("  ", 1)[1] for line in lines}
    assert "audio/clean/s1_lv01.wav" in paths
    assert "labels/s1_lv01.json" in paths
    assert "checksums.sha256" not in paths


def test_regenerated_checksums_verify(dataset: Path) -> None:
    pkg.regenerate_checksums(dataset)
    assert pkg.verify_checksums(dataset) == []


def test_verification_reports_a_changed_file(dataset: Path) -> None:
    pkg.regenerate_checksums(dataset)
    _wav(dataset / "audio" / "clean" / "s1_lv01.wav", 3.0)
    bad = pkg.verify_checksums(dataset)
    assert bad == ["audio/clean/s1_lv01.wav"]


def test_verification_reports_a_missing_file(dataset: Path) -> None:
    pkg.regenerate_checksums(dataset)
    (dataset / "labels" / "s1_lv01.json").unlink()
    assert "labels/s1_lv01.json" in pkg.verify_checksums(dataset)


# --------------------------------------------------------------------------- #
# dataset_sample/
# --------------------------------------------------------------------------- #


def test_sample_holds_the_audio_and_its_sidecars(dataset: Path, tmp_path: Path) -> None:
    rows = pkg.collect_rows(dataset)
    out = tmp_path / "sample"
    pkg.build_sample(dataset, out, ["s1_lv01", "s1_lv01_rushing_L3"])
    assert (out / "audio" / "s1_lv01.wav").is_file()
    assert (out / "labels" / "s1_lv01_rushing_L3.json").is_file()
    assert (out / "alignments" / "s1_lv01_rushing_L3.TextGrid").is_file()
    assert (out / "transcripts" / "s1.json").is_file()
    assert rows  # fixture sanity


def test_sample_copies_json_alignments_too(dataset: Path, tmp_path: Path) -> None:
    """Injected files carry ``alignments/*.json``, not a TextGrid (spec 7.2)."""
    (dataset / "alignments" / "s1_lv01_rushing_L3.TextGrid").unlink()
    (dataset / "alignments" / "s1_lv01_rushing_L3.json").write_text(
        '{"file_id": "s1_lv01_rushing_L3", "words": []}', encoding="utf-8"
    )
    out = tmp_path / "sample"
    pkg.build_sample(dataset, out, ["s1_lv01", "s1_lv01_rushing_L3"])
    assert (out / "alignments" / "s1_lv01_rushing_L3.json").is_file()


def test_sample_manifest_and_licences_are_filtered_to_the_subset(
    dataset: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sample"
    pkg.build_sample(dataset, out, ["s1_lv01", "s2_tm01_mumbling_L2"])
    manifest = (out / "MANIFEST.csv").read_text(encoding="utf-8")
    assert "s1_lv01" in manifest and "s1_lv02" not in manifest
    licences = (out / "LICENSES.csv").read_text(encoding="utf-8")
    assert "s2_tm01_mumbling_L2" in licences and "s2_lv01" not in licences


def test_sample_carries_its_own_checksums(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample"
    pkg.build_sample(dataset, out, ["s1_lv01"])
    assert pkg.verify_checksums(out) == []


def test_rebuilding_the_sample_drops_files_no_longer_selected(
    dataset: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sample"
    pkg.build_sample(dataset, out, ["s1_lv01", "s1_lv02"])
    pkg.build_sample(dataset, out, ["s1_lv01"])
    assert not (out / "audio" / "s1_lv02.wav").exists()


def test_sample_keeps_hand_written_notes(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample"
    (out / "smoke").mkdir(parents=True)
    (out / "smoke" / "README.md").write_text("keep me", encoding="utf-8")
    pkg.build_sample(dataset, out, ["s1_lv01"])
    assert (out / "smoke" / "README.md").read_text(encoding="utf-8") == "keep me"


def test_sample_size_is_reported_in_bytes(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "sample"
    result = pkg.build_sample(dataset, out, ["s1_lv01"])
    assert result.n_bytes == sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# Zip
# --------------------------------------------------------------------------- #


def test_zip_holds_dataset_relative_paths(dataset: Path, tmp_path: Path) -> None:
    pkg.regenerate_checksums(dataset)
    zip_path = pkg.make_zip(dataset, tmp_path / "dist" / "rostrum_dataset_v1.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "audio/clean/s1_lv01.wav" in names
    assert "checksums.sha256" in names


def test_zip_is_written_atomically_over_an_old_one(dataset: Path, tmp_path: Path) -> None:
    target = tmp_path / "dist" / "rostrum_dataset_v1.zip"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")
    pkg.make_zip(dataset, target)
    assert zipfile.is_zipfile(target)
    assert not list(target.parent.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# Clean label backfill
# --------------------------------------------------------------------------- #


def test_missing_clean_label_json_is_written(dataset: Path) -> None:
    """A clean reading without a label record gets one that asserts no flaws."""
    (dataset / "MANIFEST.csv").write_text(
        "file_id,text_id,tier,origin,label,level,duration_s,split_fold\n"
        "s1_lv01,s1,silver,clean,,,2.000,\n",
        encoding="utf-8",
    )
    (dataset / "labels" / "s1_lv01.json").unlink()
    rows = pkg.collect_rows(dataset)
    written = pkg.write_missing_clean_labels(dataset, rows)
    assert written == ["s1_lv01"]
    data = json.loads((dataset / "labels" / "s1_lv01.json").read_text(encoding="utf-8"))
    assert data["origin"] == "clean"
    assert data["flaws"] == []
    assert data["text_id"] == "s1"


def test_existing_label_json_is_never_overwritten(dataset: Path) -> None:
    before = (dataset / "labels" / "s1_lv01.json").read_text(encoding="utf-8")
    assert pkg.write_missing_clean_labels(dataset, pkg.collect_rows(dataset)) == []
    assert (dataset / "labels" / "s1_lv01.json").read_text(encoding="utf-8") == before


def test_clean_label_takes_the_speaker_from_licences(dataset: Path) -> None:
    (dataset / "labels" / "s1_lv01.json").unlink()
    pkg.write_missing_clean_labels(dataset, pkg.collect_rows(dataset))
    data = json.loads((dataset / "labels" / "s1_lv01.json").read_text(encoding="utf-8"))
    assert data["speaker_id"] == "s"  # speaker_slug of the licence credit "S"


# --------------------------------------------------------------------------- #
# split_fold survives a run that does not rebuild the splits
# --------------------------------------------------------------------------- #


def test_collect_rows_keeps_the_previous_split_fold(dataset: Path) -> None:
    """``--only sample`` must not blank the fold column the split builder filled."""
    (dataset / "MANIFEST.csv").write_text(
        "file_id,text_id,tier,origin,label,level,duration_s,split_fold\n"
        "s1_lv01,s1,silver,clean,,,2.000,loto_s1\n",
        encoding="utf-8",
    )
    by_id = {r.file_id: r for r in pkg.collect_rows(dataset)}
    assert by_id["s1_lv01"].split_fold == "loto_s1"


def test_sample_ships_a_filtered_events_tsv(dataset: Path, tmp_path: Path) -> None:
    """Event-F1 on the sample alone needs the sed_eval file, cut to the subset."""
    tab = chr(9)
    rows = [
        tab.join(["s1_lv01_rushing_L3.wav", "0.500", "1.000", "rushing"]),
        tab.join(["s1_lv01_monotone_L1.wav", "0.200", "0.800", "monotone"]),
    ]
    (dataset / "labels" / "events.tsv").write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
    out = tmp_path / "sample"
    pkg.build_sample(dataset, out, ["s1_lv01", "s1_lv01_rushing_L3"])
    text = (out / "labels" / "events.tsv").read_text(encoding="utf-8")
    assert "s1_lv01_rushing_L3.wav" in text
    assert "s1_lv01_monotone_L1.wav" not in text
