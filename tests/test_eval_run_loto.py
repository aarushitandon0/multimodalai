"""Unit tests for eval/run_loto.py (spec 14, prompt 19).

The leakage guard, the aggregation and the markdown are tested on hand-built
payloads so they run in milliseconds; one end-to-end fold is exercised too,
because a harness whose parts all pass separately can still be wired up wrong.
"""

from __future__ import annotations

import json

import pytest
from eval.metrics import TargetRow
from eval.run_loto import (
    CorpusUnavailable,
    FoldReport,
    LeakageError,
    Variant,
    aggregate,
    assert_no_leakage,
    metrics_markdown,
    run_fold,
    run_loto,
)

from rostrum.config import load_config


@pytest.fixture(scope="module")
def config():
    """The real config; the harness reads targets and thresholds from it."""
    return load_config()


# --------------------------------------------------------------------------- #
# Leakage guard (prompt 19, item 3)
# --------------------------------------------------------------------------- #


def test_disjoint_fitting_and_test_ids_pass():
    audit = assert_no_leakage(["a", "b"], ["c", "d"], fold="loto_x")
    assert audit["overlap"] == []
    assert audit["n_fitted_on"] == 2
    assert audit["n_test"] == 2


def test_a_test_file_in_the_fitting_inputs_fails_the_run():
    with pytest.raises(LeakageError, match="reached a fitting call"):
        assert_no_leakage(["a", "c"], ["c", "d"], fold="loto_x")


def test_the_message_names_the_leaked_file():
    with pytest.raises(LeakageError, match="s1_lv07"):
        assert_no_leakage(["s1_lv07"], ["s1_lv07"])


def test_a_human_take_may_never_be_fitted_on():
    with pytest.raises(LeakageError, match="test-only origin"):
        assert_no_leakage(["h1"], ["t1"], origins={"h1": "human"}, test_only_origins=("human",))


def test_an_injected_file_may_be_fitted_on():
    audit = assert_no_leakage(["i1"], ["t1"], origins={"i1": "injected"})
    assert audit["origins_checked"] is True


def test_without_an_origin_map_the_origin_check_is_reported_as_skipped():
    assert assert_no_leakage(["a"], ["b"])["origins_checked"] is False


# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #


def test_the_default_variant_is_the_full_system():
    variant = Variant()
    assert (variant.refine, variant.classifier, variant.tune) == (True, True, True)
    assert variant.single_reference is False
    assert variant.speaker_normalisation is True


def test_a_variant_round_trips_through_its_dict():
    variant = Variant(name="no_refine", refine=False)
    assert variant.to_dict()["refine"] is False
    assert variant.to_dict()["name"] == "no_refine"


# --------------------------------------------------------------------------- #
# Aggregation and the report
# --------------------------------------------------------------------------- #


def _fold_payload(fold: str, *, f1: float, mae: float, macro: float, rho: float) -> dict:
    """A minimal fold payload with only the fields the aggregator reads."""
    return {
        "splits": {
            "injected": {
                "available": True,
                "event_f1": f1,
                "onset_offset_mae_ms": mae,
                "typing": {"macro_f1": macro},
                "events": {"delta_f1": 0.0},
            },
            "human": {"available": False, "reason": "no human files"},
        },
        "clean": {"fp_per_min": 0.5, "minutes": 1.0},
        "severity": {"pooled": {"rho": rho}},
        "fairness": {"gap_points": None},
        "fold": fold,
    }


def _reports() -> list[FoldReport]:
    return [
        FoldReport(
            fold="loto_a",
            held_out="a",
            train_texts=("b", "c"),
            payload=_fold_payload("loto_a", f1=0.80, mae=100.0, macro=0.9, rho=-0.9),
        ),
        FoldReport(
            fold="loto_b",
            held_out="b",
            train_texts=("a", "c"),
            payload=_fold_payload("loto_b", f1=0.70, mae=120.0, macro=0.8, rho=-0.8),
        ),
    ]


def test_the_aggregate_reports_mean_and_sd_across_folds(config):
    payload = aggregate(_reports(), config=config, corpus="synthetic", variant=Variant())
    metrics = payload["splits"]["injected"]["metrics"]
    assert metrics["event_f1_200ms"]["mean"] == pytest.approx(0.75)
    assert metrics["event_f1_200ms"]["sd"] == pytest.approx(0.05)
    assert metrics["onset_offset_mae_ms"]["mean"] == pytest.approx(110.0)
    assert payload["n_folds"] == 2


def test_the_aggregate_carries_the_pre_registered_targets(config):
    payload = aggregate(_reports(), config=config, corpus="synthetic", variant=Variant())
    rows = {row["key"]: row for row in payload["targets"]["synthetic"]}
    # 0.75 meets the 0.75 target exactly, and the mean MAE of 110 ms beats 150.
    assert rows["event_f1_200ms"]["verdict"] == "PASS"
    assert rows["onset_offset_mae_ms"]["verdict"] == "PASS"
    # rho of -0.85 exactly meets the -0.85 target.
    assert rows["severity_spearman_rho"]["verdict"] == "PASS"


def test_a_split_with_no_files_is_reported_as_unavailable_not_as_zero(config):
    payload = aggregate(_reports(), config=config, corpus="synthetic", variant=Variant())
    assert payload["splits"]["human"]["available"] is False
    assert payload["splits"]["human"]["metrics"]["event_f1_200ms"]["mean"] is None


def test_the_markdown_states_the_corpus_the_caveat_and_every_target_row(config):
    payload = aggregate(_reports(), config=config, corpus="synthetic", variant=Variant())
    text = metrics_markdown(payload)
    assert "corpus: **synthetic**" in text
    assert "not a dataset result" in text
    assert "PASS" in text
    for row in payload["targets"]["synthetic"]:
        assert TargetRow(**row).label in text
    assert "Not evaluated" in text  # the human section


def test_the_markdown_warns_when_the_clean_denominator_is_tiny(config):
    payload = aggregate(_reports(), config=config, corpus="synthetic", variant=Variant())
    payload["clean_minutes"] = 0.5
    assert "0.5 minute(s)" in metrics_markdown(payload)


def test_a_fold_report_writes_one_json_per_fold(tmp_path):
    report = _reports()[0]
    path = report.write(tmp_path)
    assert path.name == "fold_a.json"
    assert json.loads(path.read_text(encoding="utf-8"))["fold"] == "loto_a"


# --------------------------------------------------------------------------- #
# Corpus guard
# --------------------------------------------------------------------------- #


def test_the_dataset_corpus_refuses_with_what_is_missing(config):
    from eval.dataset_corpus import is_available

    if is_available(config):  # pragma: no cover - only on a built checkout
        pytest.skip("this checkout has a built dataset")
    with pytest.raises(CorpusUnavailable) as caught:
        run_fold(config=config, held_out="s1_gettysburg", corpus="dataset")
    message = str(caught.value)
    # The refusal is actionable: every missing piece is named with its real
    # path and the target that produces it (prompt 19).
    assert "not built in this checkout" in message
    assert "data.alignments" in message and "make align" in message
    assert "--corpus synthetic" in message


def test_the_dataset_refusal_lists_real_paths_not_config_keys(config):
    from eval.dataset_corpus import describe_missing, is_available

    if is_available(config):  # pragma: no cover - only on a built checkout
        pytest.skip("this checkout has a built dataset")
    message = describe_missing(config)
    assert str(config.path("data.alignments")) in message


def test_a_dataset_variant_without_speaker_normalisation_is_refused(config):
    from eval.dataset_corpus import is_available

    if not is_available(config):  # pragma: no cover - only on a built checkout
        pytest.skip("needs a built dataset; eval/dataset_corpus has the full test")
    with pytest.raises(NotImplementedError):
        run_fold(
            config=config,
            held_out="s1_gettysburg",
            corpus="dataset",
            variant=Variant(speaker_normalisation=False),
        )


# --------------------------------------------------------------------------- #
# End to end, one fold
# --------------------------------------------------------------------------- #


def test_one_fold_runs_end_to_end_and_fits_only_on_the_training_text(config, tmp_path):
    report = run_fold(config=config, held_out="syn_a", texts=("syn_a", "syn_b"))
    payload = report.payload

    assert payload["held_out"] == "syn_a"
    assert payload["train_texts"] == ["syn_b"]
    assert payload["leakage_check"]["overlap"] == []
    assert payload["fit"]["thresholds"]["tuned"] is True
    assert payload["fit"]["n_train_files"] > 0
    # Every number the targets table needs is present for the injected split.
    injected = payload["splits"]["injected"]
    assert injected["available"] is True
    assert 0.0 <= injected["event_f1"] <= 1.0
    assert injected["boundaries"]["iou_matched"]["n"] >= 0
    assert injected["typing"]["rule_macro_f1"] >= 0.0
    # The synthetic corpus has no human takes, and says so.
    assert payload["splits"]["human"]["available"] is False
    # Severity is graded over the ladder, so a rank correlation exists.
    assert payload["severity"]["n"] > 3


def test_the_fold_carries_the_per_event_records_the_figures_need(config):
    report = run_fold(
        config=config,
        held_out="syn_a",
        texts=("syn_a", "syn_b"),
        variant=Variant(name="smoke", tune=False),
    )
    injected = report.payload["splits"]["injected"]
    diagnostics = injected["diagnostics"]
    assert set(diagnostics) == {"pairs", "false_negatives", "false_positives"}
    # The per-event records and the aggregates come from the same matching, so
    # the pair count is the IoU-matched n that the MAE was measured over.
    assert len(diagnostics["pairs"]) == injected["boundaries"]["iou_matched"]["n"]
    for pair in diagnostics["pairs"]:
        assert {"file_id", "label", "onset_error_ms", "offset_error_ms", "iou"} <= set(pair)
    # Every held-out file is described, so a figure can group by label and rung.
    cases = report.payload["cases"]
    assert cases
    assert all(
        {"label", "level", "origin", "severity_s", "n_regions"} <= set(c) for c in cases.values()
    )
    assert all(case["origin"] == "injected" for case in cases.values())


def test_the_synthetic_corpus_is_compared_within_one_channel(config):
    # The synthetic readers share a channel, which is what lets `mumbling` fire
    # at all; on the dataset the same check turns the spectral triggers off.
    report = run_fold(
        config=config,
        held_out="syn_a",
        texts=("syn_a", "syn_b"),
        variant=Variant(name="smoke", tune=False, classifier=False),
    )
    assert all(case["same_channel"] for case in report.payload["cases"].values())


def test_a_rules_only_fold_reports_no_classifier(config):
    report = run_fold(
        config=config,
        held_out="syn_a",
        texts=("syn_a", "syn_b"),
        variant=Variant(name="rules_only", classifier=False, tune=False),
    )
    assert report.payload["fit"]["classifier"]["fitted"] is False
    assert report.payload["splits"]["injected"]["typing"]["classifier"] is None
    assert report.payload["fit"]["thresholds"]["tuned"] is False


def test_run_loto_writes_metrics_json_markdown_and_one_file_per_fold(config, tmp_path):
    out = tmp_path / "metrics.json"
    payload = run_loto(
        config=config,
        texts=("syn_a", "syn_b"),
        variant=Variant(name="smoke", tune=False, classifier=False),
        out=out,
    )
    assert out.is_file()
    assert (tmp_path / "metrics.md").is_file()
    assert {p.name for p in tmp_path.glob("fold_*.json")} == {"fold_syn_a.json", "fold_syn_b.json"}
    assert payload["n_folds"] == 2
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == payload["schema"]
