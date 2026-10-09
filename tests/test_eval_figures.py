"""Tests for ``eval/figures.py`` (prompt 20).

Each figure is rendered from a hand-built metrics payload into ``tmp_path`` and
checked for the two things that can be asserted about a plot without looking at
it: that both formats were written and are non-empty, and that the figure was
built from the data it claims - the manifest's ``n`` and caption. The figure
that *cannot* be built must write nothing and say why; that is asserted too,
because a silently missing figure in a report is worse than an absent one.
"""

from __future__ import annotations

import json

import pytest
from eval.figures import (
    DPI,
    FIGURES,
    FORMATS,
    confusion_matrix,
    event_f1_by_label,
    example_overlay,
    onset_error_hist,
    render_all,
    severity_curves,
    utmos_by_level,
)

from rostrum.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def _fold(
    *,
    cases: dict | None = None,
    categories: dict | None = None,
    per_label: dict | None = None,
    pairs: tuple = (),
    confusion: list | None = None,
    labels: list | None = None,
    human: dict | None = None,
) -> dict:
    """One fold payload with only the blocks the figures read."""
    injected = {
        "available": True,
        "events": {"rostrum": {"per_label": per_label or {}}},
        "diagnostics": {"pairs": list(pairs), "false_negatives": [], "false_positives": []},
        "typing": {"rule": {"confusion": confusion or [], "labels": labels or []}},
    }
    return {
        "fold": "loto_a",
        "variant": {"name": "full", "refine": True},
        "cases": cases or {},
        "scores": {"categories": categories or {}, "overall": {}},
        "splits": {"injected": injected, "human": human or {"available": False}},
    }


def _payload(fold: dict | None = None, *, corpus: str = "synthetic") -> dict:
    return {
        "corpus": corpus,
        "config_hash": "deadbeefcafe",
        "code_version": "testing",
        "folds": ["loto_a"],
        "variant": {"name": "full", "description": "the system as specified"},
        "targets": {
            "synthetic": [{"key": "event_f1_200ms", "target": 0.75, "actual": 0.6}],
            "human": [{"key": "event_f1_200ms", "target": 0.50, "actual": None}],
        },
        "splits": {"injected": {"available": True}, "human": {"available": False}},
        "per_fold": {"loto_a": fold or _fold()},
    }


def _assert_written(result, tmp_path):
    """Both formats exist in ``tmp_path``, are non-empty, and are listed PNG first."""
    assert result.built is True, result.reason
    assert [p.rsplit(".", 1)[-1] for p in result.paths] == list(FORMATS)
    for suffix in FORMATS:
        path = tmp_path / f"{result.name}.{suffix}"
        assert path.is_file() and path.stat().st_size > 0
    assert result.caption


# --------------------------------------------------------------------------- #
# Severity curves
# --------------------------------------------------------------------------- #


def test_severity_curves_use_each_files_own_category_score(tmp_path):
    fold = _fold(
        cases={
            f"f_{level}": {
                "label": "rushing",
                "level": level,
                "category": "pace",
                "n_truth": 1,
                "severity_s": s,
            }
            for level, s in (("L1", 0.15), ("L2", 0.35), ("L3", 0.60), ("L4", 0.90))
        },
        categories={
            "f_L1": {"pace": 9.0, "energy": 2.0},
            "f_L2": {"pace": 7.0},
            "f_L3": {"pace": 5.0},
            "f_L4": {"pace": 2.0},
        },
    )
    result = severity_curves(_payload(fold), tmp_path)
    _assert_written(result, tmp_path)
    assert result.n == 4  # one file per rung, and no others counted


def test_severity_curves_say_why_when_no_file_has_a_category_score(tmp_path):
    result = severity_curves(_payload(), tmp_path)
    assert result.built is False
    assert "category score" in result.reason
    assert not list(tmp_path.glob("severity_curves.*"))


# --------------------------------------------------------------------------- #
# Event-F1 per label
# --------------------------------------------------------------------------- #


def test_event_f1_bars_skip_a_label_the_fold_never_asked_about(tmp_path):
    fold = _fold(
        per_label={
            "rushing": {"f1": 0.8, "tp": 4, "fp": 1, "fn": 1},
            "mumbling": {"f1": 0.0, "tp": 0, "fp": 2, "fn": 0},  # no truth: not plotted
        }
    )
    result = event_f1_by_label(_payload(fold), tmp_path)
    _assert_written(result, tmp_path)
    assert result.n == 1


def test_event_f1_bars_refuse_when_no_fold_reported_a_label(tmp_path):
    result = event_f1_by_label(_payload(), tmp_path)
    assert result.built is False and "per-label" in result.reason


# --------------------------------------------------------------------------- #
# Onset error histogram
# --------------------------------------------------------------------------- #


def _pair(onset: float, offset: float = 0.0, file_id: str = "f1") -> dict:
    return {
        "file_id": file_id,
        "label": "rushing",
        "ref_start": 1.0,
        "ref_end": 2.0,
        "pred_start": 1.0 + onset / 1000.0,
        "pred_end": 2.0 + offset / 1000.0,
        "onset_error_ms": onset,
        "offset_error_ms": offset,
        "iou": 0.8,
    }


def test_the_onset_histogram_reports_the_mae_it_plots(tmp_path):
    fold = _fold(pairs=(_pair(-120.0), _pair(60.0), _pair(30.0)))
    result = onset_error_hist(_payload(fold), tmp_path)
    _assert_written(result, tmp_path)
    assert result.n == 3
    assert "MAE 70 ms" in result.caption  # (120 + 60 + 30) / 3


def test_the_onset_histogram_refuses_without_a_matched_pair(tmp_path):
    result = onset_error_hist(_payload(), tmp_path)
    assert result.built is False and "matched pair" in result.reason


# --------------------------------------------------------------------------- #
# UTMOS
# --------------------------------------------------------------------------- #


def test_utmos_names_the_files_it_needs_when_the_dataset_is_absent(tmp_path, config):
    local = load_config(
        overrides={
            "paths": {
                "data": {
                    "qc_utmos": str(tmp_path / "nope" / "utmos.csv"),
                    "manifest": str(tmp_path / "nope" / "MANIFEST.csv"),
                }
            }
        }
    )
    result = utmos_by_level(_payload(), tmp_path, config=local)
    assert result.built is False
    assert "utmos.csv" in result.reason and "make data" in result.reason


def test_utmos_plots_each_label_against_the_rungs(tmp_path, config):
    root = tmp_path / "ds"
    root.mkdir()
    (root / "utmos.csv").write_text(
        "file_id,utmos,model,sr\n"
        "clean1,4.10,m,16000\n"
        "clean1_rushing_L1,3.95,m,16000\n"
        "clean1_rushing_L3,3.40,m,16000\n",
        encoding="utf-8",
    )
    (root / "MANIFEST.csv").write_text(
        "file_id,text_id,tier,origin,label,level,duration_s,split_fold\n"
        "clean1,t1,silver,clean,,,10.0,loto_t1\n"
        "clean1_rushing_L1,t1,silver,injected,rushing,L1,10.0,loto_t1\n"
        "clean1_rushing_L3,t1,silver,injected,rushing,L3,10.0,loto_t1\n",
        encoding="utf-8",
    )
    local = load_config(
        overrides={
            "paths": {
                "data": {
                    "qc_utmos": str(root / "utmos.csv"),
                    "manifest": str(root / "MANIFEST.csv"),
                }
            }
        }
    )
    result = utmos_by_level(_payload(), tmp_path, config=local)
    _assert_written(result, tmp_path)
    assert result.n == 2  # the two edited files; the clean one is the baseline


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #


def test_the_confusion_matrix_sums_the_folds(tmp_path):
    fold = _fold(confusion=[[3, 1], [0, 2]], labels=["rushing", "dragging"])
    payload = _payload(fold)
    payload["per_fold"]["loto_b"] = _fold(
        confusion=[[1, 0], [1, 1]], labels=["rushing", "dragging"]
    )
    result = confusion_matrix(payload, tmp_path)
    _assert_written(result, tmp_path)
    assert result.n == 9  # 3+1+0+2 and 1+0+1+1


def test_the_confusion_matrix_refuses_when_no_fold_typed_anything(tmp_path):
    result = confusion_matrix(_payload(), tmp_path)
    assert result.built is False and "confusion matrix" in result.reason


# --------------------------------------------------------------------------- #
# Overlay and the whole set
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_overlay_plots_a_flawed_held_out_file_against_the_band(tmp_path, config):
    result = example_overlay(tmp_path, config=config, corpus="synthetic")
    _assert_written(result, tmp_path)
    assert "against the envelope of" in result.caption


def test_the_overlay_reports_a_missing_dataset_instead_of_raising(tmp_path, config):
    from eval.dataset_corpus import is_available

    if is_available(config):  # pragma: no cover - only on a built checkout
        pytest.skip("this checkout has a built dataset")
    result = example_overlay(tmp_path, config=config, corpus="dataset")
    assert result.built is False and result.reason


def test_render_all_writes_a_manifest_naming_every_figure(tmp_path, config):
    metrics = tmp_path / "metrics.json"
    fold = _fold(
        cases={
            "f_L2": {
                "label": "rushing",
                "level": "L2",
                "category": "pace",
                "n_truth": 1,
                "severity_s": 0.35,
            }
        },
        categories={"f_L2": {"pace": 6.0}},
        per_label={"rushing": {"f1": 0.7, "tp": 3, "fp": 1, "fn": 1}},
        pairs=(_pair(80.0, -40.0),),
        confusion=[[3]],
        labels=["rushing"],
    )
    metrics.write_text(json.dumps(_payload(fold)), encoding="utf-8")
    manifest = render_all(
        config=config,
        metrics=metrics,
        out_dir=tmp_path,
        only=["severity_curves", "event_f1_by_label", "onset_error_hist", "confusion_matrix"],
    )
    assert manifest["dpi"] == DPI
    names = {figure["name"]: figure for figure in manifest["figures"]}
    assert set(names) <= set(FIGURES)
    assert all(names[name]["built"] for name in names), names
    assert (tmp_path / "figures.json").is_file()
    assert manifest["config_hash"] == config.hash


def test_render_all_refuses_without_a_metrics_file(tmp_path, config):
    with pytest.raises(FileNotFoundError, match="make eval"):
        render_all(config=config, metrics=tmp_path / "nope.json", out_dir=tmp_path)
