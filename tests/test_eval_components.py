"""Component-validity checks (spec 14.1 "Components").

The shared plumbing in :mod:`eval.components` is tested on its own, then each
validator is run twice: once with its inputs missing, where the contract is that
it writes a ``pending`` section with a usable reason rather than failing
silently, and once against synthetic inputs where the right answer is known.

Pitch agreement is exercised on a synthetic tone, where Praat and CREPE must
both find the same frequency, so the check does not need the dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import components as C
from eval import validate_alignment as VA
from eval import validate_pitch as VP
from eval import validate_rate as VR

from rostrum.align.mfa import TextGridTier, write_textgrid
from rostrum.config import load_config
from rostrum.io.audio import write_wav16

SR = 16000


# --------------------------------------------------------------------------- #
# Shared plumbing
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg(tmp_path):
    """A config whose results and figures land in a temporary directory."""
    return load_config(
        overrides={
            "paths": {
                "results": {"components": str(tmp_path / "results" / "components.json")},
                "docs": {"figures": str(tmp_path / "figures")},
            }
        }
    )


def test_a_missing_components_file_reads_as_empty(cfg) -> None:
    assert C.read_components(cfg) == {}


def test_sections_are_merged_not_overwritten(cfg) -> None:
    C.write_section(cfg, "pitch", {"status": C.STATUS_OK, "overall": {"agreement_pct": 93.2}})
    C.write_section(cfg, "rate", {"status": C.STATUS_PENDING, "reason": "no alignment"})
    data = C.read_components(cfg)
    assert set(data) == {"pitch", "rate", "meta"}
    assert data["pitch"]["overall"]["agreement_pct"] == pytest.approx(93.2)
    assert data["meta"]["sections"] == ["pitch", "rate"]
    assert data["meta"]["config_hash"] == cfg.hash


def test_an_unknown_section_is_refused(cfg) -> None:
    with pytest.raises(KeyError, match="unknown component section"):
        C.write_section(cfg, "loudness", {})


def test_non_finite_values_become_null_and_floats_are_rounded(cfg) -> None:
    C.write_section(cfg, "pitch", {"a": float("nan"), "b": 1.23456789, "c": np.float64(2.5)})
    section = C.read_components(cfg)["pitch"]
    assert section["a"] is None
    assert section["b"] == pytest.approx(1.235)
    assert section["c"] == pytest.approx(2.5)


def test_absolute_paths_are_written_relative_to_the_repository(cfg) -> None:
    C.write_section(cfg, "rate", {"where": str(cfg.root / "dataset" / "qc" / "x.csv")})
    assert C.read_components(cfg)["rate"]["where"] == "dataset/qc/x.csv"


def test_a_corrupt_components_file_is_replaced_not_crashed_on(cfg) -> None:
    path = cfg.path("results.components")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert C.read_components(cfg) == {}
    C.write_section(cfg, "pitch", {"status": C.STATUS_OK})
    assert C.read_components(cfg)["pitch"]["status"] == C.STATUS_OK


def test_error_summary_counts_the_tolerance_bands() -> None:
    # 10 ms, 30 ms and 100 ms errors: one within 20, two within 50.
    out = C.summarise_errors_ms([0.010, -0.030, 0.100])
    assert out["n"] == 3
    assert out["mae_ms"] == pytest.approx((10 + 30 + 100) / 3)
    assert out["within_20ms_pct"] == pytest.approx(100 / 3)
    assert out["within_50ms_pct"] == pytest.approx(200 / 3)
    assert out["bias_ms"] == pytest.approx((10 - 30 + 100) / 3)


def test_error_summary_of_nothing_is_not_an_error() -> None:
    out = C.summarise_errors_ms([])
    assert out["n"] == 0
    assert np.isnan(out["mae_ms"])


def test_describe_ignores_non_finite_values() -> None:
    out = C.describe([1.0, 2.0, 3.0, float("nan"), float("inf")])
    assert out["n"] == 3
    assert out["median"] == pytest.approx(2.0)


def test_percentile_of_nothing_is_nan() -> None:
    assert np.isnan(C.percentile([], 95))


# --------------------------------------------------------------------------- #
# Pitch agreement
# --------------------------------------------------------------------------- #


def test_agreement_summary_classifies_the_bands() -> None:
    cents = np.array([0.0, 20.0, -49.0, 80.0, 1200.0])
    out = VP.summarise(cents, agreement_cents=50.0)
    assert out["agreement_pct"] == pytest.approx(60.0)
    assert out["octave_slip_pct"] == pytest.approx(20.0)
    assert out["median_abs_cents"] == pytest.approx(49.0)


def test_agreement_summary_of_nothing_is_nan() -> None:
    out = VP.summarise(np.empty(0), agreement_cents=50.0)
    assert np.isnan(out["agreement_pct"])


@pytest.mark.slow
def test_both_trackers_find_a_synthetic_tone(tmp_path) -> None:
    """A 200 Hz tone is unambiguous, so agreement must be essentially total."""
    from rostrum.features.frames import FrameGrid

    t = np.arange(int(1.5 * SR), dtype=np.float64) / SR
    tone = 0.4 * np.sin(2.0 * np.pi * 200.0 * t)
    cfg = load_config()
    cents, praat, n_crepe = VP.compare_one(
        tone, SR, cfg=cfg, grid=FrameGrid.from_config(cfg), model="tiny"
    )
    assert praat.voiced.sum() > 50
    assert n_crepe > 50
    assert cents.size > 50
    stats = VP.summarise(cents, agreement_cents=50.0)
    assert stats["agreement_pct"] > 95.0


def test_pitch_section_is_pending_without_audio(tmp_path, monkeypatch) -> None:
    cfg = load_config(
        overrides={
            "paths": {
                "results": {"components": str(tmp_path / "components.json")},
                "data": {
                    "audio_clean": str(tmp_path / "none"),
                    "audio_injected": str(tmp_path / "none"),
                    "audio_human_flawed": str(tmp_path / "none"),
                    "manifest": str(tmp_path / "none.csv"),
                },
            }
        }
    )
    monkeypatch.setattr(VP, "load_config", lambda *a, **k: cfg)
    assert VP.main([]) == 2
    section = json.loads((tmp_path / "components.json").read_text(encoding="utf-8"))["pitch"]
    assert section["status"] == C.STATUS_PENDING
    assert "make data" in section["reason"]


# --------------------------------------------------------------------------- #
# Alignment accuracy
# --------------------------------------------------------------------------- #


def _write_alignment(path: Path, spans: list[tuple[float, float, str]]) -> None:
    """A words-only TextGrid at ``path``."""
    write_textgrid(
        path,
        [TextGridTier("words", 0.0, spans[-1][1] + 0.1, tuple(spans))],
        xmin=0.0,
        xmax=spans[-1][1] + 0.1,
    )


def _write_hand_labels(path: Path, rows: list[tuple[str, int, float, float, str]]) -> None:
    lines = [",".join(VA.HAND_FIELDS)]
    lines += [f"{f},{i},{on},{off},{w}" for f, i, on, off, w in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def alignment_setup(tmp_path, monkeypatch):
    """A config with one alignment on disk and a place for hand labels."""
    cfg = load_config(
        overrides={
            "paths": {
                "results": {"components": str(tmp_path / "components.json")},
                "docs": {"figures": str(tmp_path / "figures")},
                "data": {
                    "alignments": str(tmp_path / "alignments"),
                    "qc_hand_boundaries": str(tmp_path / "qc" / "hand_boundaries.csv"),
                },
            }
        }
    )
    _write_alignment(
        tmp_path / "alignments" / "f1.TextGrid",
        [(0.100, 0.400, "now"), (0.450, 0.700, "is"), (0.750, 1.000, "the")],
    )
    monkeypatch.setattr(VA, "load_config", lambda *a, **k: cfg)
    return cfg, tmp_path


def test_write_template_creates_the_expected_header(alignment_setup) -> None:
    cfg, _ = alignment_setup
    assert VA.main(["--write-template"]) == 0
    path = cfg.path("data.qc_hand_boundaries")
    assert path.read_text(encoding="utf-8").strip() == ",".join(VA.HAND_FIELDS)
    assert VA.read_hand_labels(path) == []


def test_alignment_section_is_pending_without_hand_labels(alignment_setup) -> None:
    cfg, _ = alignment_setup
    assert VA.main([]) == 2
    section = json.loads(cfg.path("results.components").read_text(encoding="utf-8"))["alignment"]
    assert section["status"] == C.STATUS_PENDING
    assert section["expects"]["n_boundaries"] == VA.TARGET_BOUNDARIES
    assert section["expects"]["columns"] == list(VA.HAND_FIELDS)


def test_a_header_with_no_rows_is_still_pending(alignment_setup) -> None:
    cfg, _ = alignment_setup
    VA.write_template(cfg.path("data.qc_hand_boundaries"))
    assert VA.main([]) == 2
    section = json.loads(cfg.path("results.components").read_text(encoding="utf-8"))["alignment"]
    assert "no rows" in section["reason"]


def test_missing_columns_are_an_error(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("file_id,word_idx\nf1,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        VA.read_hand_labels(path)


def test_malformed_rows_are_skipped_not_fatal(tmp_path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "file_id,word_idx,true_onset,true_offset,word\n"
        "f1,0,0.100,0.400,now\n"
        ",,,,\n"
        "f1,oops,0.1,0.2,is\n",
        encoding="utf-8",
    )
    labels = VA.read_hand_labels(path)
    assert [label.word_idx for label in labels] == [0]


def test_known_offsets_produce_the_known_error_statistics(alignment_setup) -> None:
    """Hand labels 20 ms early on every boundary must give a 20 ms MAE."""
    cfg, _ = alignment_setup
    _write_hand_labels(
        cfg.path("data.qc_hand_boundaries"),
        [
            ("f1", 0, 0.080, 0.380, "now"),
            ("f1", 1, 0.430, 0.680, "is"),
            ("f1", 2, 0.730, 0.980, "the"),
        ],
    )
    assert VA.main([]) == 0
    section = json.loads(cfg.path("results.components").read_text(encoding="utf-8"))["alignment"]
    assert section["status"] == C.STATUS_OK
    assert section["n_boundaries"] == 3
    assert section["pooled"]["n"] == 6
    assert section["pooled"]["mae_ms"] == pytest.approx(20.0, abs=0.5)
    assert section["pooled"]["bias_ms"] == pytest.approx(20.0, abs=0.5)
    assert section["pooled"]["within_20ms_pct"] == pytest.approx(100.0)
    assert section["meets_spec_sample_size"] is False
    for name in section["figures"]:
        assert (cfg.path("docs.figures") / name).is_file()


def test_a_word_index_outside_the_alignment_is_skipped(alignment_setup) -> None:
    cfg, _ = alignment_setup
    _write_hand_labels(
        cfg.path("data.qc_hand_boundaries"),
        [("f1", 0, 0.100, 0.400, "now"), ("f1", 99, 1.0, 1.2, "nope")],
    )
    assert VA.main(["--no-figures"]) == 0
    section = json.loads(cfg.path("results.components").read_text(encoding="utf-8"))["alignment"]
    assert section["n_boundaries"] == 1


def test_a_text_mismatch_is_reported_not_hidden(alignment_setup) -> None:
    cfg, _ = alignment_setup
    _write_hand_labels(cfg.path("data.qc_hand_boundaries"), [("f1", 0, 0.100, 0.400, "different")])
    assert VA.main(["--no-figures"]) == 0
    section = json.loads(cfg.path("results.components").read_text(encoding="utf-8"))["alignment"]
    assert section["n_text_mismatches"] == 1


def test_labels_for_a_file_with_no_alignment_are_pending(alignment_setup) -> None:
    cfg, _ = alignment_setup
    _write_hand_labels(cfg.path("data.qc_hand_boundaries"), [("nosuch", 0, 0.1, 0.4, "now")])
    assert VA.main([]) == 2
    section = json.loads(cfg.path("results.components").read_text(encoding="utf-8"))["alignment"]
    assert "make align" in section["reason"]


# --------------------------------------------------------------------------- #
# Rate cross-check
# --------------------------------------------------------------------------- #


def test_correlation_needs_enough_varying_points() -> None:
    assert np.isnan(VR.correlate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))["pearson_r"])
    constant = VR.correlate(np.ones(10), np.arange(10.0))
    assert np.isnan(constant["pearson_r"])


def test_correlation_of_identical_series_is_one() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = VR.correlate(x, x)
    assert out["pearson_r"] == pytest.approx(1.0)
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["mean_absolute_difference"] == pytest.approx(0.0)


def test_rate_section_is_pending_without_alignments(tmp_path, monkeypatch) -> None:
    audio_dir = tmp_path / "clean"
    t = np.arange(SR, dtype=np.float64) / SR
    write_wav16(audio_dir / "f1.wav", (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), SR)
    cfg = load_config(
        overrides={
            "paths": {
                "results": {"components": str(tmp_path / "components.json")},
                "data": {
                    "audio_clean": str(audio_dir),
                    "alignments": str(tmp_path / "alignments"),
                    "manifest": str(tmp_path / "none.csv"),
                },
            }
        }
    )
    monkeypatch.setattr(VR, "load_config", lambda *a, **k: cfg)
    assert VR.main([]) == 2
    section = json.loads((tmp_path / "components.json").read_text(encoding="utf-8"))["rate"]
    assert section["status"] == C.STATUS_PENDING
    assert "make align" in section["reason"]


def test_rate_section_is_pending_without_audio(tmp_path, monkeypatch) -> None:
    cfg = load_config(
        overrides={
            "paths": {
                "results": {"components": str(tmp_path / "components.json")},
                "data": {
                    "audio_clean": str(tmp_path / "none"),
                    "alignments": str(tmp_path / "none"),
                    "manifest": str(tmp_path / "none.csv"),
                },
            }
        }
    )
    monkeypatch.setattr(VR, "load_config", lambda *a, **k: cfg)
    assert VR.main([]) == 2
    section = json.loads((tmp_path / "components.json").read_text(encoding="utf-8"))["rate"]
    assert "make data" in section["reason"]


def test_local_nuclei_rate_counts_only_the_window(tmp_path) -> None:
    from rostrum.types import Word

    words = [
        Word(
            idx=i,
            text=f"w{i}",
            t_start=i * 0.3,
            t_end=i * 0.3 + 0.25,
            phones=["AH1"],
            n_syllables=1,
        )
        for i in range(7)
    ]
    # One nucleus per word, at the middle of each.
    nuclei = np.array([w.t_start + 0.1 for w in words])
    rate = VR.local_nuclei_rate(words, 3, nuclei, window=3, exclude_s=0.25)
    assert rate > 0
    # The seven-word window holds all seven nuclei over its articulation time.
    span = VR.R.articulation_time(words, exclude_pauses_over_s=0.25)
    assert rate == pytest.approx(7.0 / span)


@pytest.mark.slow
def test_rate_cross_check_runs_on_synthetic_speech(tmp_path, monkeypatch) -> None:
    """Five equal syllables: both estimators must see roughly the same rate."""
    audio_dir = tmp_path / "clean"
    alignment_dir = tmp_path / "alignments"
    parts = [np.zeros(int(0.2 * SR))]
    spans: list[tuple[float, float, str]] = []
    t = 0.2
    for i in range(5):
        tt = np.arange(int(0.30 * SR), dtype=np.float64) / SR
        wave = np.zeros_like(tt)
        for harmonic, weight in ((1, 1.0), (2, 0.5), (3, 0.25)):
            wave += weight * np.sin(2.0 * np.pi * 150.0 * harmonic * tt)
        envelope = np.minimum(1.0, np.minimum(tt, 0.30 - tt) / 0.03)
        parts.append(0.4 * wave * envelope / 1.75)
        spans.append((round(t, 3), round(t + 0.30, 3), f"w{i}"))
        t += 0.30
        parts.append(np.zeros(int(0.10 * SR)))
        t += 0.10
    samples = np.concatenate(parts).astype(np.float32)
    write_wav16(audio_dir / "f1.wav", samples, SR)
    write_textgrid(
        alignment_dir / "f1.TextGrid",
        [
            TextGridTier("words", 0.0, len(samples) / SR, tuple(spans)),
            TextGridTier(
                "phones",
                0.0,
                len(samples) / SR,
                tuple((a, b, "AH1") for a, b, _ in spans),
            ),
        ],
        xmin=0.0,
        xmax=len(samples) / SR,
    )
    cfg = load_config(
        overrides={
            "paths": {
                "results": {"components": str(tmp_path / "components.json")},
                "docs": {"figures": str(tmp_path / "figures")},
                "data": {
                    "audio_clean": str(audio_dir),
                    "alignments": str(alignment_dir),
                    "manifest": str(tmp_path / "none.csv"),
                },
            }
        }
    )
    monkeypatch.setattr(VR, "load_config", lambda *a, **k: cfg)
    assert VR.main(["--no-figures"]) == 0
    section = json.loads((tmp_path / "components.json").read_text(encoding="utf-8"))["rate"]
    assert section["status"] == C.STATUS_OK
    assert section["n_files"] == 1
    assert section["files"][0]["n_mfa_syllables"] == 5
    assert section["files"][0]["n_nuclei"] == pytest.approx(5, abs=2)
