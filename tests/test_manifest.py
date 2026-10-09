"""Tests for LICENSES.csv / MANIFEST.csv upserts and the provenance checker."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import yaml

from rostrum.data.manifest import (
    LICENSES_FIELDS,
    MANIFEST_FIELDS,
    LicenseRow,
    ManifestRow,
    append_licenses,
    append_manifest,
    read_licenses,
    write_checksums,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_provenance

PROV_FILE = "S9.pdf"


def _row(
    file_id: str = "s9_a", *, licence: str = "Public Domain", sha: str = "a" * 64
) -> LicenseRow:
    return LicenseRow(
        file_id=file_id,
        title="A Title",
        speaker="A Reader",
        source_url="https://example.org/page",
        licence=licence,
        retrieved_on="2026-10-02",
        sha256=sha,
    )


# --------------------------------------------------------------------------- #
# Column contract
# --------------------------------------------------------------------------- #


def test_license_dataclass_matches_spec_column_order() -> None:
    assert tuple(LicenseRow.__dataclass_fields__) == LICENSES_FIELDS


def test_manifest_dataclass_matches_spec_column_order() -> None:
    assert tuple(ManifestRow.__dataclass_fields__) == MANIFEST_FIELDS


# --------------------------------------------------------------------------- #
# Upsert behaviour
# --------------------------------------------------------------------------- #


def test_append_creates_file_with_header(tmp_path: Path) -> None:
    path = tmp_path / "LICENSES.csv"
    assert append_licenses(path, [_row()]) == 1
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        assert tuple(next(reader)) == LICENSES_FIELDS


def test_append_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "LICENSES.csv"
    append_licenses(path, [_row()])
    first = path.read_bytes()
    append_licenses(path, [_row()])
    assert path.read_bytes() == first


def test_append_updates_existing_row(tmp_path: Path) -> None:
    path = tmp_path / "LICENSES.csv"
    append_licenses(path, [_row(sha="a" * 64)])
    append_licenses(path, [_row(sha="b" * 64)])
    rows = read_licenses(path)
    assert len(rows) == 1
    assert rows[0].sha256 == "b" * 64


def test_rows_are_sorted_by_file_id(tmp_path: Path) -> None:
    """A canonical order keeps the CSV diffable and its hash stable (spec 16)."""
    path = tmp_path / "LICENSES.csv"
    append_licenses(path, [_row("s9_c"), _row("s9_a"), _row("s9_b")])
    assert [r.file_id for r in read_licenses(path)] == ["s9_a", "s9_b", "s9_c"]


def test_read_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_licenses(tmp_path / "nope.csv") == []


def test_wrong_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "LICENSES.csv"
    path.write_text("file_id,licence\nx,Public Domain\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_licenses(path)


def test_manifest_optional_columns_default_empty(tmp_path: Path) -> None:
    path = tmp_path / "MANIFEST.csv"
    append_manifest(path, [ManifestRow("s9_a", "t9", "silver", "clean")])
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[1] == "s9_a,t9,silver,clean,,,,"


def test_unknown_column_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown column"):
        append_manifest(tmp_path / "MANIFEST.csv", [{"file_id": "x", "bogus": 1}])


def test_write_checksums_is_sorted(tmp_path: Path) -> None:
    path = tmp_path / "checksums.sha256"
    assert write_checksums(path, {"b.wav": "b" * 64, "a.wav": "a" * 64}) == 2
    assert path.read_text(encoding="utf-8").splitlines()[0].endswith("a.wav")


# --------------------------------------------------------------------------- #
# Provenance checker
# --------------------------------------------------------------------------- #


def _dataset(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Build an empty dataset skeleton and return the paths check() needs."""
    root = tmp_path / "dataset"
    (root / "audio" / "clean").mkdir(parents=True)
    (root / "provenance").mkdir()
    return (
        root,
        root / "LICENSES.csv",
        root / "SOURCES.yaml",
        root / "provenance",
        root / "CONSENT.md",
    )


def _sources_yaml(path: Path, file_ids: list[str]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "id": "S9",
                        "text_id": "t9",
                        "title": "A Title",
                        "tier": "silver",
                        "page_url": "https://example.org/page",
                        "licence": "Public Domain",
                        "provenance": PROV_FILE,
                        "files": [
                            {"file_id": fid, "download_url": f"https://example.org/{fid}.mp3"}
                            for fid in file_ids
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_empty_dataset_passes(tmp_path: Path) -> None:
    """Acceptance: the checker passes on an empty dataset."""
    root, lic, src, prov, consent = _dataset(tmp_path)
    assert check_provenance.check(root, lic, src, prov, consent) == []


def test_audio_without_licence_row_fails(tmp_path: Path) -> None:
    root, lic, src, prov, consent = _dataset(tmp_path)
    (root / "audio" / "clean" / "s9_a.wav").write_bytes(b"RIFF")
    _sources_yaml(src, ["s9_a"])
    problems = check_provenance.check(root, lic, src, prov, consent)
    assert any("no LICENSES.csv row" in p.reason for p in problems)


def test_audio_without_provenance_artefact_fails(tmp_path: Path) -> None:
    root, lic, src, prov, consent = _dataset(tmp_path)
    (root / "audio" / "clean" / "s9_a.wav").write_bytes(b"RIFF")
    _sources_yaml(src, ["s9_a"])
    append_licenses(lic, [_row("s9_a")])
    problems = check_provenance.check(root, lic, src, prov, consent)
    assert any(PROV_FILE in p.reason for p in problems)


def test_fully_documented_audio_passes(tmp_path: Path) -> None:
    root, lic, src, prov, consent = _dataset(tmp_path)
    (root / "audio" / "clean" / "s9_a.wav").write_bytes(b"RIFF")
    _sources_yaml(src, ["s9_a"])
    append_licenses(lic, [_row("s9_a")])
    (prov / PROV_FILE).write_bytes(b"%PDF-1.4")
    assert check_provenance.check(root, lic, src, prov, consent) == []


def test_injected_file_inherits_its_source_row(tmp_path: Path) -> None:
    """`s9_a_rushing_L3` is covered by the row for `s9_a` (spec 7.1 naming)."""
    root, lic, src, prov, consent = _dataset(tmp_path)
    injected = root / "audio" / "injected"
    injected.mkdir()
    (injected / "s9_a_rushing_L3.wav").write_bytes(b"RIFF")
    _sources_yaml(src, ["s9_a"])
    append_licenses(lic, [_row("s9_a")])
    (prov / PROV_FILE).write_bytes(b"%PDF-1.4")
    assert check_provenance.check(root, lic, src, prov, consent) == []


def test_nd_licence_in_csv_is_rejected(tmp_path: Path) -> None:
    """A hand-edited ND licence must not slip past the checker."""
    root, lic, src, prov, consent = _dataset(tmp_path)
    (root / "audio" / "clean" / "s9_a.wav").write_bytes(b"RIFF")
    _sources_yaml(src, ["s9_a"])
    append_licenses(lic, [_row("s9_a", licence="CC BY-NC-ND 4.0")])
    (prov / PROV_FILE).write_bytes(b"%PDF-1.4")
    problems = check_provenance.check(root, lic, src, prov, consent)
    assert any("rejected" in p.reason for p in problems)


def test_undeclared_audio_needs_consent_file(tmp_path: Path) -> None:
    """A team recording not in SOURCES.yaml is covered by CONSENT.md (spec 4.4)."""
    root, lic, src, prov, consent = _dataset(tmp_path)
    (root / "audio" / "clean" / "team_01.wav").write_bytes(b"RIFF")
    _sources_yaml(src, ["s9_a"])
    append_licenses(lic, [_row("team_01")])
    problems = check_provenance.check(root, lic, src, prov, consent)
    assert any("CONSENT.md" in p.reason for p in problems)

    consent.write_text("I, A Person, release my recordings under CC BY 4.0.\n", encoding="utf-8")
    assert check_provenance.check(root, lic, src, prov, consent) == []


def test_longest_prefix_wins(tmp_path: Path) -> None:
    """`s1_lv1` must not claim `s1_lv16`'s derivatives."""
    rows = {fid: _row(fid) for fid in ("s1_lv1", "s1_lv16")}
    assert check_provenance.resolve_row("s1_lv16_rushing_L3", rows).file_id == "s1_lv16"


def test_provenance_dirs_are_not_scanned_for_audio(tmp_path: Path) -> None:
    root, _lic, _src, prov, _consent = _dataset(tmp_path)
    (prov / "licence_recording.mp3").write_bytes(b"ID3")
    assert check_provenance.find_audio(root) == []
