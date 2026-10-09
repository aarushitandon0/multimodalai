"""Tests for the alignment QC report (spec 8) and the clean-build helpers.

``summarise`` is pure, so the flag rule is tested on hand-built alignments
rather than on MFA output - the suite must pass without the ``mfa`` conda env.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import align_report as ar
import build_clean as bc
from align_clean import MissingTranscript, transcript_for
from rostrum.data.sources import parse_sources
from rostrum.io.audio import AudioStats
from rostrum.types import Alignment, Gap, Word

THRESHOLD = 0.02


def _alignment(words: list[tuple[str, float, float]], file_id: str = "f1") -> Alignment:
    """Build an Alignment from ``(text, t_start, t_end)`` triples, gaps included."""
    objs = [Word(idx=i, text=t, t_start=a, t_end=b) for i, (t, a, b) in enumerate(words)]
    gaps = [
        Gap(
            idx=i,
            prev_word_idx=i,
            next_word_idx=i + 1,
            t_start=objs[i].t_end,
            t_end=objs[i + 1].t_start,
        )
        for i in range(len(objs) - 1)
    ]
    return Alignment(
        file_id=file_id,
        words=objs,
        gaps=gaps,
        duration=objs[-1].t_end if objs else 0.0,
    )


def _clean(n: int = 100) -> Alignment:
    return _alignment([(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(n)])


# --------------------------------------------------------------------------- #
# summarise
# --------------------------------------------------------------------------- #


def test_summarise_counts_words_and_durations() -> None:
    report = ar.summarise(
        _alignment([("four", 0.2, 0.5), ("score", 0.6, 1.0), ("and", 1.5, 1.6)]),
        n_expected_tokens=3,
        max_unaligned_frac=THRESHOLD,
        text_id="s1_gettysburg",
    )
    assert report.n_words == 3
    assert report.text_id == "s1_gettysburg"
    assert report.mean_word_dur_s == pytest.approx((0.3 + 0.4 + 0.1) / 3, abs=1e-3)
    assert report.longest_gap_s == pytest.approx(0.5, abs=1e-3)
    assert report.longest_gap_after_word == 1
    assert not report.flagged


def test_summarise_passes_a_fully_aligned_file() -> None:
    report = ar.summarise(_clean(), n_expected_tokens=100, max_unaligned_frac=THRESHOLD)
    assert report.n_unaligned == 0
    assert report.unaligned_frac == 0.0
    assert not report.flagged and report.reason == ""


def test_summarise_flags_unknown_tokens() -> None:
    words = [(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(100)]
    words[7] = ("<unk>", 3.5, 3.8)
    words[8] = ("<unk>", 4.0, 4.3)
    words[9] = ("<unk>", 4.5, 4.8)
    report = ar.summarise(_alignment(words), n_expected_tokens=100, max_unaligned_frac=THRESHOLD)
    assert report.n_unk == 3
    assert report.n_unaligned == 3
    assert report.flagged
    assert "dictionary" in report.reason


def test_summarise_flags_missing_transcript_tokens() -> None:
    report = ar.summarise(_clean(90), n_expected_tokens=100, max_unaligned_frac=THRESHOLD)
    assert report.n_missing == 10
    assert report.flagged
    assert "overrides" in report.reason


def test_summarise_flags_zero_length_words() -> None:
    words = [(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(100)]
    words[3] = ("ghost", 1.5, 1.5)
    words[4] = ("ghost", 2.0, 2.0)
    words[5] = ("ghost", 2.5, 2.5)
    report = ar.summarise(_alignment(words), n_expected_tokens=100, max_unaligned_frac=THRESHOLD)
    assert report.n_zero_length == 3
    assert report.flagged


def test_summarise_tolerates_one_bad_word_in_a_hundred() -> None:
    # 1/100 = 1%, under the 2% budget: reported, not flagged.
    words = [(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(100)]
    words[0] = ("<unk>", 0.0, 0.3)
    report = ar.summarise(_alignment(words), n_expected_tokens=100, max_unaligned_frac=THRESHOLD)
    assert report.n_unaligned == 1
    assert not report.flagged


def test_summarise_flags_an_empty_grid() -> None:
    empty = Alignment(file_id="f1", words=[], gaps=[], duration=0.0)
    report = ar.summarise(empty, n_expected_tokens=42, max_unaligned_frac=THRESHOLD)
    assert report.flagged
    assert report.n_words == 0
    assert "no aligned words" in report.reason


def test_summarise_ignores_the_token_count_when_it_is_unknown() -> None:
    report = ar.summarise(_clean(50), n_expected_tokens=0, max_unaligned_frac=THRESHOLD)
    assert report.n_missing == 0
    assert not report.flagged


# --------------------------------------------------------------------------- #
# Report writing
# --------------------------------------------------------------------------- #


def test_write_report_round_trips(tmp_path: Path) -> None:
    import csv

    reports = [
        ar.summarise(_clean(), n_expected_tokens=100, max_unaligned_frac=THRESHOLD),
        ar.summarise(_clean(90), n_expected_tokens=100, max_unaligned_frac=THRESHOLD),
    ]
    out = tmp_path / "align_report.csv"
    ar.write_report(out, reports)
    with out.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(ar.REPORT_FIELDS)
    assert len(rows) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_format_table_marks_flagged_files() -> None:
    text = ar.format_table(
        [
            ar.summarise(_clean(), n_expected_tokens=100, max_unaligned_frac=THRESHOLD),
            ar.summarise(_clean(80), n_expected_tokens=100, max_unaligned_frac=THRESHOLD),
        ]
    )
    assert "FLAG" in text
    assert "ok" in text


# --------------------------------------------------------------------------- #
# Transcript lookup
# --------------------------------------------------------------------------- #


def test_transcript_prefers_a_per_file_override(tmp_path: Path) -> None:
    (tmp_path / "overrides").mkdir()
    (tmp_path / "overrides" / "f1.txt").write_text("this reading's own words", encoding="utf-8")
    (tmp_path / "t1.json").write_text(json.dumps({"verbatim": "library text"}), encoding="utf-8")
    text, source = transcript_for("f1", "t1", tmp_path)
    assert text == "this reading's own words"
    assert source.name == "f1.txt"


def test_transcript_falls_back_to_the_text_json(tmp_path: Path) -> None:
    (tmp_path / "t1.json").write_text(json.dumps({"verbatim": "library text"}), encoding="utf-8")
    assert transcript_for("f1", "t1", tmp_path)[0] == "library text"


def test_transcript_skips_a_todo_placeholder(tmp_path: Path) -> None:
    (tmp_path / "t1.json").write_text(json.dumps({"verbatim": "TODO(human)"}), encoding="utf-8")
    with pytest.raises(MissingTranscript) as exc:
        transcript_for("f1", "t1", tmp_path)
    assert "t1.json" in str(exc.value)


def test_transcript_missing_names_every_path_tried(tmp_path: Path) -> None:
    with pytest.raises(MissingTranscript) as exc:
        transcript_for("f1", "t1", tmp_path)
    message = str(exc.value)
    assert "t1.json" in message and "t1.txt" in message and "overrides" in message


# --------------------------------------------------------------------------- #
# build_clean helpers
# --------------------------------------------------------------------------- #


def _stats(peak: float = -1.5, lufs: float | None = -23.0, sr: int = 16_000) -> AudioStats:
    return AudioStats(sample_rate=sr, n_samples=sr * 2, duration_s=2.0, lufs=lufs, peak_dbfs=peak)


def test_check_invariants_accepts_a_conforming_wav() -> None:
    assert bc.check_invariants(_stats(), target_lufs=-23.0, peak_ceiling=-1.0) == []


def test_check_invariants_rejects_the_wrong_sample_rate() -> None:
    problems = bc.check_invariants(_stats(sr=44_100), target_lufs=-23.0, peak_ceiling=-1.0)
    assert any("sample rate" in p for p in problems)


def test_check_invariants_rejects_a_hot_peak() -> None:
    problems = bc.check_invariants(_stats(peak=-0.2), target_lufs=-23.0, peak_ceiling=-1.0)
    assert any("peak" in p for p in problems)


def test_check_invariants_rejects_an_unmeasurable_clip() -> None:
    problems = bc.check_invariants(_stats(lufs=None), target_lufs=-23.0, peak_ceiling=-1.0)
    assert any("loudness" in p for p in problems)


def test_excerpt_bounds_passes_numbers_through() -> None:
    (source,) = parse_sources(
        {
            "sources": [
                {
                    "id": "S9",
                    "text_id": "t1",
                    "page_url": "https://x",
                    "licence": "Public Domain",
                    "provenance": "p.pdf",
                    "files": [
                        {
                            "file_id": "f1",
                            "download_url": "https://x/a.mp3",
                            "excerpt_start_s": 1.5,
                            "excerpt_end_s": 9.0,
                        }
                    ],
                }
            ]
        }
    )
    assert bc.excerpt_bounds(source.files[0]) == (1.5, 9.0, "")


def test_excerpt_bounds_reports_a_todo_bound() -> None:
    (source,) = parse_sources(
        {
            "sources": [
                {
                    "id": "S9",
                    "text_id": "t1",
                    "page_url": "https://x",
                    "licence": "Public Domain",
                    "provenance": "p.pdf",
                    "files": [
                        {
                            "file_id": "f1",
                            "download_url": "https://x/a.mp3",
                            "excerpt_start_s": 0,
                            "excerpt_end_s": "TODO(human)",
                        }
                    ],
                }
            ]
        }
    )
    start, end, problem = bc.excerpt_bounds(source.files[0])
    assert (start, end) == (None, None)
    assert "excerpt_end_s" in problem


def test_find_raw_ignores_sidecars(tmp_path: Path) -> None:
    (tmp_path / "team").mkdir()
    (tmp_path / "team" / "tm01_s1_clean.md").write_text("notes", encoding="utf-8")
    (tmp_path / "team" / "tm01_s1_clean.wav").write_bytes(b"RIFF")
    found = bc.find_raw(tmp_path, "team", "tm01_s1_clean")
    assert found is not None and found.suffix == ".wav"


def test_find_raw_returns_none_when_nothing_is_there(tmp_path: Path) -> None:
    assert bc.find_raw(tmp_path, "team", "tm01_s1_clean") is None


def test_refresh_checksums_keeps_unrelated_live_entries(tmp_path: Path) -> None:
    dataset = tmp_path
    (dataset / "audio" / "clean").mkdir(parents=True)
    (dataset / "audio" / "clean" / "a.wav").write_bytes(b"one")
    (dataset / "other.bin").write_bytes(b"two")
    checksums = dataset / "checksums.sha256"
    checksums.write_text(f"{'0' * 64}  other.bin\n{'1' * 64}  gone.bin\n", encoding="utf-8")

    n = bc.refresh_checksums(dataset, [dataset / "audio" / "clean"], checksums)
    lines = dict(reversed(line.split("  ", 1)) for line in checksums.read_text().splitlines())
    assert n == 2
    assert "audio/clean/a.wav" in lines
    assert lines["other.bin"] == "0" * 64  # preserved
    assert "gone.bin" not in lines  # dropped: the file no longer exists


# --------------------------------------------------------------------------- #
# raw_subdir (team recordings share one folder)
# --------------------------------------------------------------------------- #


def test_raw_dirname_defaults_to_the_source_id() -> None:
    (source,) = parse_sources(
        {
            "sources": [
                {
                    "id": "S9",
                    "text_id": "t1",
                    "page_url": "https://x",
                    "licence": "Public Domain",
                    "provenance": "p.pdf",
                }
            ]
        }
    )
    assert source.raw_dirname == "S9"


def test_raw_subdir_overrides_the_folder() -> None:
    (source,) = parse_sources(
        {
            "sources": [
                {
                    "id": "team_s1",
                    "text_id": "t1",
                    "raw_subdir": "team",
                    "page_url": "https://x",
                    "licence": "CC BY 4.0 (consent in CONSENT.md)",
                    "provenance": "team_consent.md",
                }
            ]
        }
    )
    assert source.raw_dirname == "team"
