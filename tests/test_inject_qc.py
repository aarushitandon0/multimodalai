"""Tests for the injection realism gates (spec 5.1, 5.4).

Everything here runs without the ``mfa`` conda env and without downloading the
UTMOS checkpoint: the alignment comparison and the gate arithmetic are pure
functions over hand-built :class:`~rostrum.types.Alignment` objects, and the
two outside-world dependencies are injected (``align_fn``, ``scorer``) so the
glue around them is still covered.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from rostrum.align.mfa import MFAError
from rostrum.config import load_config
from rostrum.data.manifest import ManifestRow, append_manifest, read_manifest
from rostrum.inject import qc
from rostrum.inject.timemap import TimeMap
from rostrum.types import Alignment, Gap, Word

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _alignment(
    words: list[tuple[str, float, float]],
    file_id: str = "f1",
    duration: float | None = None,
) -> Alignment:
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
        duration=float(duration if duration is not None else objs[-1].t_end + 0.3),
    )


EXPECTED = _alignment([("one", 0.5, 1.0), ("two", 1.2, 1.8), ("three", 2.0, 2.6)])


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# 1. Boundary errors
# --------------------------------------------------------------------------- #


def test_identical_alignments_have_zero_boundary_error() -> None:
    errors = qc.boundary_errors_ms(EXPECTED, EXPECTED)
    assert errors.shape == (3,)
    assert errors.tolist() == [0.0, 0.0, 0.0]


def test_boundary_error_is_the_worse_of_onset_and_offset_in_ms() -> None:
    realigned = _alignment([("one", 0.5, 1.0), ("two", 1.23, 1.79), ("three", 2.0, 2.6)])

    errors = qc.boundary_errors_ms(EXPECTED, realigned)

    assert errors[0] == pytest.approx(0.0)
    assert errors[1] == pytest.approx(30.0)  # onset 30 ms, offset 10 ms
    assert errors[2] == pytest.approx(0.0)


def test_boundary_errors_reject_a_different_word_count() -> None:
    with pytest.raises(ValueError, match="word count"):
        qc.boundary_errors_ms(EXPECTED, _alignment([("one", 0.5, 1.0)]))


def test_boundary_errors_reject_mismatched_word_text() -> None:
    realigned = _alignment([("one", 0.5, 1.0), ("deux", 1.2, 1.8), ("three", 2.0, 2.6)])
    with pytest.raises(ValueError, match="word 1"):
        qc.boundary_errors_ms(EXPECTED, realigned)


def test_boundary_errors_ignore_word_case() -> None:
    """MFA can echo a token in a different case; that is not a re-segmentation."""
    realigned = _alignment([("one", 0.5, 1.0), ("TWO", 1.2, 1.8), ("three", 2.0, 2.6)])
    assert qc.boundary_errors_ms(EXPECTED, realigned).tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# 2. Realign report
# --------------------------------------------------------------------------- #


def test_realign_report_summarises_max_mean_and_p95() -> None:
    realigned = _alignment([("one", 0.5, 1.0), ("two", 1.21, 1.8), ("three", 2.1, 2.6)])

    report = qc.realign_report(EXPECTED, realigned, file_id="f1_rushing_L2")

    assert report.file_id == "f1_rushing_L2"
    assert report.n_words == 3
    assert report.max_err_ms == pytest.approx(100.0)
    assert report.mean_err_ms == pytest.approx((0.0 + 10.0 + 100.0) / 3)
    assert report.p95_err_ms == pytest.approx(91.0)  # numpy linear interpolation


def test_realign_report_counts_the_20_ms_agreement_target() -> None:
    realigned = _alignment([("one", 0.5, 1.0), ("two", 1.21, 1.8), ("three", 2.1, 2.6)])

    report = qc.realign_report(EXPECTED, realigned)

    assert report.target_ms == pytest.approx(20.0)
    assert report.frac_within_target == pytest.approx(2 / 3)


def test_realign_report_passes_under_50_ms_and_fails_over_it() -> None:
    ok = qc.realign_report(
        EXPECTED, _alignment([("one", 0.5, 1.0), ("two", 1.249, 1.8), ("three", 2.0, 2.6)])
    )
    bad = qc.realign_report(
        EXPECTED, _alignment([("one", 0.5, 1.0), ("two", 1.26, 1.8), ("three", 2.0, 2.6)])
    )

    assert ok.max_err_ms == pytest.approx(49.0)
    assert ok.passed is True
    assert bad.passed is False


def test_realign_report_row_is_csv_ready() -> None:
    report = qc.realign_report(
        EXPECTED, EXPECTED, file_id="f1_rushing_L2", label="rushing", level="L2"
    )
    row = report.to_row()
    assert set(row) == set(qc.REALIGN_COLUMNS)
    assert row["label"] == "rushing"
    assert row["passed"] == "true"


# --------------------------------------------------------------------------- #
# 3. realign_check glue (MFA injected)
# --------------------------------------------------------------------------- #


def test_realign_check_compares_the_expected_alignment_with_the_aligner_output(tmp_path) -> None:
    wav = tmp_path / "f1_rushing_L2.wav"
    wav.write_bytes(b"RIFF")  # the fake aligner never opens it
    calls: list[tuple[Path, str, str]] = []

    def fake_align(wav_path: Path, transcript: str, file_id: str, work_dir: Path) -> Alignment:
        calls.append((wav_path, transcript, file_id))
        return _alignment([("one", 0.5, 1.0), ("two", 1.22, 1.8), ("three", 2.0, 2.6)])

    report = qc.realign_check(
        wav, "one two three", EXPECTED, align_fn=fake_align, work_dir=tmp_path
    )

    assert calls == [(wav, "one two three", "f1_rushing_L2")]
    assert report.file_id == "f1_rushing_L2"
    assert report.max_err_ms == pytest.approx(20.0)


def test_realign_check_uses_the_expected_alignment_from_a_time_map(tmp_path) -> None:
    """The expectation is the clean alignment pushed through the TimeMap."""
    wav = tmp_path / "f1_dragging_L3.wav"
    wav.write_bytes(b"RIFF")
    clean = EXPECTED
    tmap = TimeMap.replacement(
        clean.duration,
        1.2,
        1.8,
        1.2,  # the 0.6 s span is doubled in length
        op="psola_lengthen",
    )
    from rostrum.inject.engine import remap_alignment

    expected = remap_alignment(clean, tmap)

    def perfect(wav_path, transcript, file_id, work_dir):
        return expected

    report = qc.realign_check(wav, "one two three", expected, align_fn=perfect, work_dir=tmp_path)

    assert report.max_err_ms == pytest.approx(0.0)
    # The edited expectation really did move: the third word shifted by +0.6 s.
    assert expected.words[2].t_start == pytest.approx(2.6)


# --------------------------------------------------------------------------- #
# 4. UTMOS cache
# --------------------------------------------------------------------------- #


def test_utmos_cache_round_trips(tmp_path) -> None:
    path = tmp_path / "utmos.csv"

    qc.append_utmos(path, {"b_file": 3.5, "a_file": 4.125})

    assert qc.read_utmos_cache(path) == {"a_file": 4.125, "b_file": 3.5}
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert [r["file_id"] for r in rows] == ["a_file", "b_file"]  # sorted, for determinism
    assert rows[0]["model"] == qc.UTMOS_MODEL


def test_appending_utmos_merges_with_what_is_already_cached(tmp_path) -> None:
    path = tmp_path / "utmos.csv"
    qc.append_utmos(path, {"a_file": 4.0})

    qc.append_utmos(path, {"b_file": 3.0})

    assert qc.read_utmos_cache(path) == {"a_file": 4.0, "b_file": 3.0}


def test_read_utmos_cache_of_a_missing_file_is_empty(tmp_path) -> None:
    assert qc.read_utmos_cache(tmp_path / "nope.csv") == {}


def test_utmos_for_files_scores_only_what_is_not_cached(tmp_path) -> None:
    cache = tmp_path / "utmos.csv"
    qc.append_utmos(cache, {"cached": 4.2})
    (tmp_path / "cached.wav").write_bytes(b"RIFF")
    (tmp_path / "fresh.wav").write_bytes(b"RIFF")
    scored: list[str] = []

    def scorer(path: Path) -> float:
        scored.append(path.stem)
        return 3.75

    scores = qc.utmos_for_files(
        [tmp_path / "cached.wav", tmp_path / "fresh.wav"], cache_path=cache, scorer=scorer
    )

    assert scored == ["fresh"]
    assert scores == {"cached": 4.2, "fresh": 3.75}
    assert qc.read_utmos_cache(cache) == {"cached": 4.2, "fresh": 3.75}


def test_utmos_for_files_never_builds_a_scorer_when_everything_is_cached(tmp_path) -> None:
    cache = tmp_path / "utmos.csv"
    qc.append_utmos(cache, {"cached": 4.2})
    (tmp_path / "cached.wav").write_bytes(b"RIFF")

    def exploding_scorer(path: Path) -> float:
        raise AssertionError("the UTMOS model must not be loaded when the cache is complete")

    assert qc.utmos_for_files(
        [tmp_path / "cached.wav"], cache_path=cache, scorer=exploding_scorer
    ) == {"cached": 4.2}


def test_the_pinned_utmos_model_is_the_published_one() -> None:
    assert qc.UTMOS_HUB_REPO == "tarepan/SpeechMOS:v1.2.0"
    assert qc.UTMOS_MODEL == "utmos22_strong"


# --------------------------------------------------------------------------- #
# 5. Gates (spec 5.4)
# --------------------------------------------------------------------------- #


def _report(max_err_ms: float) -> qc.RealignReport:
    realigned = _alignment(
        [("one", 0.5, 1.0), ("two", 1.2 + max_err_ms / 1000.0, 1.8), ("three", 2.0, 2.6)]
    )
    return qc.realign_report(EXPECTED, realigned)


def test_a_clean_file_with_a_small_drop_passes(cfg) -> None:
    verdict = qc.gate(
        realign=_report(14.0), utmos_clean=3.91, utmos_edited=3.62, level="L1", cfg=cfg
    )
    assert verdict.passed is True
    assert verdict.reasons == ()
    assert verdict.utmos_drop == pytest.approx(0.29)


def test_a_boundary_error_over_50_ms_is_rejected(cfg) -> None:
    verdict = qc.gate(realign=_report(60.0), utmos_clean=3.9, utmos_edited=3.9, level="L1", cfg=cfg)
    assert verdict.passed is False
    assert len(verdict.reasons) == 1
    assert "realign" in verdict.reasons[0]
    assert "60" in verdict.reasons[0]


def test_an_l1_utmos_drop_over_0_3_is_rejected(cfg) -> None:
    verdict = qc.gate(realign=_report(5.0), utmos_clean=4.0, utmos_edited=3.6, level="L1", cfg=cfg)
    assert verdict.passed is False
    assert "utmos" in verdict.reasons[0]


def test_the_same_drop_is_allowed_at_l4(cfg) -> None:
    verdict = qc.gate(realign=_report(5.0), utmos_clean=4.0, utmos_edited=3.6, level="L4", cfg=cfg)
    assert verdict.passed is True


def test_both_gates_can_fail_at_once(cfg) -> None:
    verdict = qc.gate(realign=_report(80.0), utmos_clean=4.0, utmos_edited=3.0, level="L1", cfg=cfg)
    assert verdict.passed is False
    assert len(verdict.reasons) == 2


def test_a_missing_utmos_score_leaves_only_the_realign_gate(cfg) -> None:
    verdict = qc.gate(
        realign=_report(5.0), utmos_clean=None, utmos_edited=None, level="L1", cfg=cfg
    )
    assert verdict.passed is True
    assert verdict.utmos_drop is None


def test_an_improved_utmos_is_never_a_rejection(cfg) -> None:
    verdict = qc.gate(realign=_report(5.0), utmos_clean=3.0, utmos_edited=4.0, level="L1", cfg=cfg)
    assert verdict.passed is True
    assert verdict.utmos_drop == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# 6. Label JSON (spec 7.1)
# --------------------------------------------------------------------------- #


def test_the_qc_block_carries_the_spec_7_1_fields(cfg) -> None:
    verdict = qc.gate(
        realign=_report(14.0), utmos_clean=3.91, utmos_edited=3.62, level="L1", cfg=cfg
    )

    block = verdict.to_dict()

    assert block["utmos_clean"] == pytest.approx(3.91)
    assert block["utmos_edited"] == pytest.approx(3.62)
    assert block["realign_max_err_ms"] == pytest.approx(14.0)
    assert block["passed"] is True
    assert block["reasons"] == []


def test_applying_a_failed_verdict_marks_the_label(cfg) -> None:
    label = {"file_id": "f1_rushing_L1", "qc": None}
    verdict = qc.gate(realign=_report(90.0), utmos_clean=4.0, utmos_edited=4.0, level="L1", cfg=cfg)

    marked = qc.apply_verdict(label, verdict)

    assert marked["qc"]["passed"] is False
    assert qc.label_passed(marked) is False
    assert label["qc"] is None  # the input is not mutated


def test_an_unverified_label_has_not_passed() -> None:
    assert qc.label_passed({"file_id": "f1", "qc": None}) is False
    assert qc.label_passed({"file_id": "f1"}) is False


# --------------------------------------------------------------------------- #
# 7. Rejections CSV and MANIFEST exclusion
# --------------------------------------------------------------------------- #


def test_rejections_are_logged_with_their_reasons(tmp_path, cfg) -> None:
    path = tmp_path / "rejections.csv"
    label = {
        "file_id": "f1_rushing_L1",
        "source_file": "f1",
        "flaws": [{"label": "rushing", "level": "L1"}],
    }
    verdict = qc.gate(realign=_report(90.0), utmos_clean=4.0, utmos_edited=3.0, level="L1", cfg=cfg)

    n = qc.append_rejections(path, [qc.rejection_row(label, verdict)])

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert n == 1
    assert rows[0]["file_id"] == "f1_rushing_L1"
    assert rows[0]["label"] == "rushing"
    assert rows[0]["level"] == "L1"
    assert rows[0]["stage"] == "qc"
    assert "realign" in rows[0]["reason"] and "utmos" in rows[0]["reason"]


def test_appending_rejections_writes_one_header(tmp_path, cfg) -> None:
    path = tmp_path / "rejections.csv"
    verdict = qc.gate(
        realign=_report(90.0), utmos_clean=None, utmos_edited=None, level="L2", cfg=cfg
    )
    qc.append_rejections(path, [qc.rejection_row({"file_id": "a"}, verdict)])
    qc.append_rejections(path, [qc.rejection_row({"file_id": "b"}, verdict)])

    text = path.read_text(encoding="utf-8")
    assert text.count("file_id,source_file") == 1
    assert len(list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))) == 2


def test_rejection_columns_match_the_build_script() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_injected

    assert build_injected.REJECTION_COLUMNS == qc.REJECTION_COLUMNS


def test_a_rejected_file_is_excluded_from_the_manifest_splits(tmp_path, cfg) -> None:
    manifest = tmp_path / "MANIFEST.csv"
    append_manifest(
        manifest,
        [
            ManifestRow(
                "f1_rushing_L1", "s1", "silver", "injected", "rushing", "L1", 12.0, "fold_1"
            ),
            ManifestRow(
                "f1_rushing_L2", "s1", "silver", "injected", "rushing", "L2", 12.0, "fold_1"
            ),
        ],
    )
    failed = {"file_id": "f1_rushing_L1", "qc": {"passed": False}}
    passed = {"file_id": "f1_rushing_L2", "qc": {"passed": True}}

    n = qc.exclude_rejected(manifest, [failed, passed])

    folds = {r.file_id: r.split_fold for r in read_manifest(manifest)}
    assert n == 1
    assert folds["f1_rushing_L1"] == qc.EXCLUDED_FOLD
    assert folds["f1_rushing_L2"] == "fold_1"


def test_excluding_nothing_leaves_the_manifest_alone(tmp_path) -> None:
    manifest = tmp_path / "MANIFEST.csv"
    append_manifest(manifest, [ManifestRow("f1", "s1", "gold", "clean", "", "L0", 10.0, "fold_1")])
    before = manifest.read_text(encoding="utf-8")

    assert qc.exclude_rejected(manifest, [{"file_id": "f1", "qc": {"passed": True}}]) == 0
    assert manifest.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
# 8. Writing the realign CSV
# --------------------------------------------------------------------------- #


def test_realign_rows_are_written_sorted_by_file_id(tmp_path) -> None:
    path = tmp_path / "realign.csv"
    qc.append_realign(
        path,
        [
            qc.realign_report(EXPECTED, EXPECTED, file_id="b_file"),
            qc.realign_report(EXPECTED, EXPECTED, file_id="a_file"),
        ],
    )

    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    assert [r["file_id"] for r in rows] == ["a_file", "b_file"]


def test_label_flaw_label_and_level_are_read_off_the_label_json(tmp_path) -> None:
    label = json.loads(
        '{"file_id": "f1_composite1_L4", "source_file": "f1", '
        '"flaws": [{"label": "rushing", "level": "L4"}, {"label": "monotone", "level": "L4"}]}'
    )
    assert qc.label_of(label) == "composite"
    assert qc.level_of(label) == "L4"


# --------------------------------------------------------------------------- #
# 9. UTMOS scoring (predictor stubbed; the real model is verified by hand)
# --------------------------------------------------------------------------- #


class _StubPredictor:
    """Stands in for the torch.hub predictor: records its call, returns a MOS."""

    def __init__(self, value: float = 3.87) -> None:
        self.value = value
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def __call__(self, wave, sr):
        import torch

        self.calls.append((tuple(wave.shape), sr))
        return torch.tensor([self.value])


def test_utmos_score_batches_one_mono_signal_at_16_khz() -> None:
    predictor = _StubPredictor(3.87)
    samples = np.zeros(16_000, dtype=np.float32)

    score = qc.utmos_score(samples, 16_000, predictor=predictor)

    assert score == pytest.approx(3.87)
    assert predictor.calls == [((1, 16_000), 16_000)]


def test_utmos_score_rejects_audio_that_is_not_16_khz() -> None:
    with pytest.raises(ValueError, match="16000 Hz"):
        qc.utmos_score(np.zeros(100, dtype=np.float32), 22_050, predictor=_StubPredictor())


def test_utmos_score_rejects_an_empty_or_stereo_signal() -> None:
    with pytest.raises(ValueError, match="mono"):
        qc.utmos_score(np.zeros((2, 100), dtype=np.float32), 16_000, predictor=_StubPredictor())
    with pytest.raises(ValueError, match="mono"):
        qc.utmos_score(np.zeros(0, dtype=np.float32), 16_000, predictor=_StubPredictor())


def test_the_utmos_scorer_reads_a_wav_off_disk(tmp_path) -> None:
    import soundfile as sf

    wav = tmp_path / "tone.wav"
    tone = 0.1 * np.sin(2 * np.pi * 200 * np.arange(16_000) / 16_000)
    sf.write(str(wav), tone.astype("float32"), 16_000, subtype="PCM_16")
    predictor = _StubPredictor(4.01)

    score = qc.utmos_scorer(predictor)(wav)

    assert score == pytest.approx(4.01)
    assert predictor.calls == [((1, 16_000), 16_000)]


# --------------------------------------------------------------------------- #
# Batched re-alignment (one MFA invocation for the whole QC pass)
# --------------------------------------------------------------------------- #


def test_mfa_realign_batch_empty_never_shells_out() -> None:
    """An empty batch is a no-op: MFA must not be invoked to align nothing."""
    assert qc.mfa_realign_batch([], Path("unused")) == {}


def test_cached_aligner_serves_from_the_batch() -> None:
    alignment = _alignment([("hello", 0.0, 0.5)], file_id="f1_rushing_L2")
    aligner = qc.cached_aligner({"f1_rushing_L2": alignment})
    got = aligner(Path("f1_rushing_L2.wav"), "hello", "f1_rushing_L2", Path("unused"))
    assert got is alignment


def test_cached_aligner_raises_mfa_error_for_a_file_mfa_dropped() -> None:
    """A variant MFA produced no TextGrid for is gated as a failure, not crashed on."""
    aligner = qc.cached_aligner({})
    with pytest.raises(MFAError, match="no batch re-alignment for f1_rushing_L2"):
        aligner(Path("f1_rushing_L2.wav"), "hello", "f1_rushing_L2", Path("unused"))
