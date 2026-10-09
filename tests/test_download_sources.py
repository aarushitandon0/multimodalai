"""Tests for the source downloader (spec 4.2).

``fetch`` is exercised over ``file://`` URLs: real behaviour of the atomic
write, the empty-body guard and the digest, with no network (CLAUDE.md
invariant 13).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_sources as ds
from rostrum.data.sources import TODO_SENTINEL, parse_sources


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://x/Roosevelt_Pearl_Harbor.ogg", ".ogg"),
        ("https://x/a.ogg?download=1", ".ogg"),
        ("https://x/gettysburg_64kb.MP3", ".mp3"),
        ("https://x/take.flac", ".flac"),
        ("https://x/Some%20Name.m4a", ".m4a"),
        ("https://x/stream", ".bin"),
        ("https://x/page.html", ".bin"),
    ],
)
def test_guess_suffix(url: str, expected: str) -> None:
    assert ds.guess_suffix(url) == expected


def _source_doc(url: str, licence: str = "Public Domain") -> dict:
    return {
        "sources": [
            {
                "id": "S9",
                "text_id": "t9",
                "title": "A Title",
                "tier": "silver",
                "page_url": "https://example.org/page",
                "licence": licence,
                "provenance": "S9.pdf",
                "files": [{"file_id": "s9_a", "reader": "A Reader", "download_url": url}],
            }
        ]
    }


def test_fetch_writes_file_and_leaves_no_part(tmp_path: Path) -> None:
    src = tmp_path / "src.ogg"
    src.write_bytes(b"OggS" * 100)
    dest = tmp_path / "out" / "s9_a.ogg"
    assert ds.fetch(src.as_uri(), dest) == 400
    assert dest.read_bytes() == src.read_bytes()
    assert not list(dest.parent.glob("*.part"))


def test_fetch_rejects_empty_body(tmp_path: Path) -> None:
    src = tmp_path / "empty.ogg"
    src.write_bytes(b"")
    with pytest.raises(ds.DownloadError):
        ds.fetch(src.as_uri(), tmp_path / "out.ogg", retries=0)


def test_fetch_raises_after_retries(tmp_path: Path) -> None:
    missing = (tmp_path / "nope.ogg").as_uri()
    with pytest.raises(ds.DownloadError, match="could not download"):
        ds.fetch(missing, tmp_path / "out.ogg", retries=0)


def test_handle_file_downloads_then_caches(tmp_path: Path) -> None:
    payload = tmp_path / "src.ogg"
    payload.write_bytes(b"OggS")
    (source,) = parse_sources(_source_doc(payload.as_uri()))
    (sf,) = source.files
    raw = tmp_path / "raw"

    first = ds.handle_file(source, sf, raw, dry_run=False, force=False)
    assert first.status == "downloaded"
    assert first.path == raw / "S9" / "s9_a.ogg"
    assert len(first.sha256) == 64

    second = ds.handle_file(source, sf, raw, dry_run=False, force=False)
    assert second.status == "cached"
    assert second.sha256 == first.sha256


def test_handle_file_reports_todo_url(tmp_path: Path) -> None:
    (source,) = parse_sources(_source_doc(TODO_SENTINEL))
    (sf,) = source.files
    outcome = ds.handle_file(source, sf, tmp_path, dry_run=False, force=False)
    assert outcome.status == "todo"
    assert not list(tmp_path.rglob("*.ogg"))


def test_dry_run_downloads_nothing(tmp_path: Path) -> None:
    payload = tmp_path / "src.ogg"
    payload.write_bytes(b"OggS")
    (source,) = parse_sources(_source_doc(payload.as_uri()))
    (sf,) = source.files
    raw = tmp_path / "raw"
    assert ds.handle_file(source, sf, raw, dry_run=True, force=False).status == "todo"
    assert not raw.exists()


def test_rows_for_builds_both_registers(tmp_path: Path) -> None:
    (source,) = parse_sources(_source_doc("https://example.org/a.ogg"))
    (sf,) = source.files
    outcome = ds.Outcome("s9_a", "S9", "downloaded", sha256="c" * 64)
    lic, man = ds.rows_for(source, sf, outcome, "2026-10-02")
    assert lic.file_id == "s9_a"
    assert lic.source_url == "https://example.org/page"  # the licence page, not the asset
    assert lic.sha256 == "c" * 64
    assert lic.retrieved_on == "2026-10-02"
    assert (man.text_id, man.tier, man.origin) == ("t9", "silver", "clean")
    assert man.duration_s == "" and man.split_fold == ""


def test_rows_for_falls_back_to_unknown_reader() -> None:
    doc = _source_doc("https://example.org/a.ogg")
    doc["sources"][0]["files"][0]["reader"] = TODO_SENTINEL
    (source,) = parse_sources(doc)
    (sf,) = source.files
    lic, _ = ds.rows_for(source, sf, ds.Outcome("s9_a", "S9", "downloaded"), "2026-10-02")
    assert lic.speaker == "unknown"


def test_report_todos_lists_pending_files_even_when_licence_is_unfilled() -> None:
    """A source stopped at the licence gate must still show its missing URLs."""
    doc = _source_doc(TODO_SENTINEL, licence=f"{TODO_SENTINEL} - verify the page")
    sources = list(parse_sources(doc))
    lines = "\n".join(ds.report_todos(sources, []))
    assert "licence" in lines
    assert "s9_a" in lines
    assert "S9.pdf" in lines


def test_main_exits_2_when_everything_is_todo(tmp_path: Path, capsys) -> None:
    """Acceptance: a file with nothing filled in yields a to-do list, not a crash.

    Built from a fixture rather than the committed ``SOURCES.yaml``: that file
    gains real URLs as sources are resolved, so asserting against its current
    state would test how far dataset collection has got, not the behaviour.
    """
    import yaml

    doc = _source_doc(TODO_SENTINEL, licence=f"{TODO_SENTINEL} - copy from the page")
    path = tmp_path / "SOURCES.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    assert ds.main(["--sources", str(path), "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "HUMAN STEPS REQUIRED" in err
    assert "s9_a" in err


def test_committed_sources_parse_and_report_what_is_left(capsys) -> None:
    """The committed ``SOURCES.yaml`` loads, and still names its unfilled sources.

    Pinned to S2/S4 because those are the entries no machine here can fill:
    the JFK Library blocks automated requests and S4 is an optional source
    whose page has not been identified. Delete the matching assertion when one
    is resolved.
    """
    code = ds.main(["--dry-run"])
    assert code in {0, 2}, "the committed SOURCES.yaml must not crash the downloader"
    err = capsys.readouterr().err
    assert "HUMAN STEPS REQUIRED" in err
    assert "s2_jfk_gold" in err
    assert "s4_lv01" in err


def test_main_exits_1_on_forbidden_licence(tmp_path: Path) -> None:
    import yaml

    doc = _source_doc("https://example.org/a.ogg", licence="CC BY-NC-ND 4.0")
    path = tmp_path / "SOURCES.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert ds.main(["--sources", str(path), "--dry-run"]) == 1
