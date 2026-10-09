"""Tests for the QC pass over the injected dataset (spec 5.4).

MFA and UTMOS are injected (``align_fn``, ``scorer``), so the whole pass -
planning, gating, CSVs, label marking, MANIFEST exclusion - is covered on a
miniature dataset built in ``tmp_path``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qc_dataset as qd
from rostrum.data.manifest import ManifestRow, append_manifest, read_manifest
from rostrum.inject import qc
from rostrum.types import Alignment, Gap, Word

WORDS = [("one", 0.5, 1.0), ("two", 1.2, 1.8), ("three", 2.0, 2.6)]


def _alignment(words: list[tuple[str, float, float]], file_id: str) -> Alignment:
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
    return Alignment(file_id=file_id, words=objs, gaps=gaps, duration=3.0, text_id="s1_gettysburg")


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tone = 0.1 * np.sin(2 * np.pi * 200 * np.arange(16000 * 3) / 16000)
    sf.write(str(path), tone.astype("float32"), 16000, subtype="PCM_16")


@pytest.fixture
def ds(tmp_path: Path) -> qd.Layout:
    """A miniature dataset: one clean source, two injected files, one manifest."""
    root = tmp_path / "dataset"
    (root / "texts").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    (root / "alignments").mkdir(parents=True)

    (root / "texts" / "s1_gettysburg.json").write_text(
        json.dumps({"verbatim": "one two three", "display": "One two three."}), encoding="utf-8"
    )
    _wav(root / "audio" / "clean" / "f1.wav")

    manifest_rows = [ManifestRow("f1", "s1_gettysburg", "silver", "clean", "", "L0", 3.0, "fold_1")]
    for level in ("L1", "L4"):
        file_id = f"f1_rushing_{level}"
        _wav(root / "audio" / "injected" / f"{file_id}.wav")
        (root / "alignments" / f"{file_id}.json").write_text(
            json.dumps(_alignment(WORDS, file_id).to_dict()), encoding="utf-8"
        )
        (root / "labels" / f"{file_id}.json").write_text(
            json.dumps(
                {
                    "file_id": file_id,
                    "source_file": "f1",
                    "text_id": "s1_gettysburg",
                    "origin": "injected",
                    "flaws": [{"label": "rushing", "level": level, "t_start": 1.0, "t_end": 2.0}],
                    "qc": None,
                }
            ),
            encoding="utf-8",
        )
        manifest_rows.append(
            ManifestRow(
                file_id, "s1_gettysburg", "silver", "injected", "rushing", level, 3.0, "fold_1"
            )
        )
    append_manifest(root / "MANIFEST.csv", manifest_rows)

    return qd.Layout(
        labels=root / "labels",
        alignments=root / "alignments",
        texts=root / "texts",
        clean_audio=root / "audio" / "clean",
        injected_audio=root / "audio" / "injected",
        qc_dir=root / "qc",
        manifest=root / "MANIFEST.csv",
    )


def _perfect_aligner(wav_path, transcript, file_id, work_dir):
    """An aligner that reproduces the expected times exactly."""
    return _alignment(WORDS, file_id)


def _sloppy_aligner(wav_path, transcript, file_id, work_dir):
    """An aligner that puts the second word 90 ms late - past the 50 ms gate."""
    return _alignment([("one", 0.5, 1.0), ("two", 1.29, 1.8), ("three", 2.0, 2.6)], file_id)


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def test_plan_finds_every_injected_file_with_its_inputs(ds: qd.Layout) -> None:
    jobs, skipped = qd.plan(ds)

    assert [j.file_id for j in jobs] == ["f1_rushing_L1", "f1_rushing_L4"]
    assert skipped == []
    job = jobs[0]
    assert job.level == "L1"
    assert job.label == "rushing"
    assert job.transcript == "one two three"
    assert job.edited_wav.is_file()
    assert job.clean_wav.is_file()
    assert job.expected.words[1].t_start == pytest.approx(1.2)


def test_plan_skips_a_file_whose_expected_alignment_is_missing(ds: qd.Layout) -> None:
    (ds.alignments / "f1_rushing_L4.json").unlink()

    jobs, skipped = qd.plan(ds)

    assert [j.file_id for j in jobs] == ["f1_rushing_L1"]
    assert [s[0] for s in skipped] == ["f1_rushing_L4"]
    assert "alignment" in skipped[0][1]


def test_plan_skips_a_file_whose_audio_is_missing(ds: qd.Layout) -> None:
    (ds.injected_audio / "f1_rushing_L1.wav").unlink()

    jobs, skipped = qd.plan(ds)

    assert [j.file_id for j in jobs] == ["f1_rushing_L4"]
    assert "WAV" in skipped[0][1]


def test_plan_honours_an_only_filter(ds: qd.Layout) -> None:
    jobs, _ = qd.plan(ds, only=["f1_rushing_L4"])
    assert [j.file_id for j in jobs] == ["f1_rushing_L4"]


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def test_a_clean_pass_marks_both_labels_and_writes_the_csvs(ds: qd.Layout) -> None:
    summary = qd.run(ds, align_fn=_perfect_aligner, scorer=lambda p: 4.0)

    assert summary.checked == 2
    assert summary.rejected == 0
    for level in ("L1", "L4"):
        label = json.loads((ds.labels / f"f1_rushing_{level}.json").read_text(encoding="utf-8"))
        assert label["qc"]["passed"] is True
        assert label["qc"]["realign_max_err_ms"] == pytest.approx(0.0)
        assert label["qc"]["utmos_edited"] == pytest.approx(4.0)
    assert (ds.qc_dir / "realign.csv").is_file()
    assert set(qc.read_utmos_cache(ds.qc_dir / "utmos.csv")) == {
        "f1",
        "f1_rushing_L1",
        "f1_rushing_L4",
    }


def test_the_clean_source_is_scored_once_for_all_its_edits(ds: qd.Layout) -> None:
    scored: list[str] = []

    qd.run(ds, align_fn=_perfect_aligner, scorer=lambda p: scored.append(p.stem) or 4.0)

    assert scored.count("f1") == 1


def test_a_boundary_error_over_50_ms_is_rejected_and_logged(ds: qd.Layout) -> None:
    summary = qd.run(ds, align_fn=_sloppy_aligner, scorer=lambda p: 4.0)

    assert summary.rejected == 2
    label = json.loads((ds.labels / "f1_rushing_L1.json").read_text(encoding="utf-8"))
    assert label["qc"]["passed"] is False
    rows = list(
        csv.DictReader((ds.qc_dir / "rejections.csv").read_text(encoding="utf-8").splitlines())
    )
    assert {r["file_id"] for r in rows} == {"f1_rushing_L1", "f1_rushing_L4"}
    assert all("realign" in r["reason"] for r in rows)
    assert all(r["stage"] == "qc" for r in rows)


def test_an_l1_utmos_collapse_is_rejected_but_l4_survives_it(ds: qd.Layout) -> None:
    scores = {"f1": 4.0, "f1_rushing_L1": 3.5, "f1_rushing_L4": 3.5}

    summary = qd.run(ds, align_fn=_perfect_aligner, scorer=lambda p: scores[p.stem])

    assert summary.rejected == 1
    l1 = json.loads((ds.labels / "f1_rushing_L1.json").read_text(encoding="utf-8"))
    l4 = json.loads((ds.labels / "f1_rushing_L4.json").read_text(encoding="utf-8"))
    assert l1["qc"]["passed"] is False
    assert "utmos" in l1["qc"]["reasons"][0]
    assert l4["qc"]["passed"] is True


def test_rejected_files_are_excluded_from_the_manifest_splits(ds: qd.Layout) -> None:
    qd.run(ds, align_fn=_sloppy_aligner, scorer=lambda p: 4.0)

    folds = {r.file_id: r.split_fold for r in read_manifest(ds.manifest)}
    assert folds["f1_rushing_L1"] == qc.EXCLUDED_FOLD
    assert folds["f1_rushing_L4"] == qc.EXCLUDED_FOLD
    assert folds["f1"] == "fold_1"  # the clean source is untouched


def test_a_re_segmented_realignment_is_a_rejection_not_a_crash(ds: qd.Layout) -> None:
    def dropped_word(wav_path, transcript, file_id, work_dir):
        return _alignment(WORDS[:2], file_id)

    summary = qd.run(ds, align_fn=dropped_word, scorer=lambda p: 4.0)

    assert summary.rejected == 2
    label = json.loads((ds.labels / "f1_rushing_L1.json").read_text(encoding="utf-8"))
    assert label["qc"]["passed"] is False
    assert "not comparable" in label["qc"]["reasons"][0]


def test_skipping_realignment_leaves_only_the_utmos_gate(ds: qd.Layout) -> None:
    def exploding(wav_path, transcript, file_id, work_dir):
        raise AssertionError("MFA must not run with realign disabled")

    summary = qd.run(ds, align_fn=exploding, scorer=lambda p: 4.0, realign=False)

    assert summary.checked == 2
    label = json.loads((ds.labels / "f1_rushing_L1.json").read_text(encoding="utf-8"))
    assert label["qc"]["realign_max_err_ms"] is None
    assert label["qc"]["passed"] is True


def test_rerunning_the_pass_is_idempotent(ds: qd.Layout) -> None:
    qd.run(ds, align_fn=_perfect_aligner, scorer=lambda p: 4.0)
    realign_once = (ds.qc_dir / "realign.csv").read_text(encoding="utf-8")

    qd.run(ds, align_fn=_perfect_aligner, scorer=lambda p: 4.0)

    assert (ds.qc_dir / "realign.csv").read_text(encoding="utf-8") == realign_once
