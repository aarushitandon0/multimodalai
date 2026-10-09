"""Unit tests for eval/metrics.py (spec 14.1, 14.3).

Every case here is hand-built so the expected number can be read off the
fixture: two events that obviously match, a boundary error that is obviously
300 ms, a ladder that is obviously monotone.
"""

from __future__ import annotations

import pytest
from eval.metrics import (
    TARGET_DIRECTIONS,
    ScoredEvent,
    boundary_errors,
    check_targets,
    crosscheck_event_f1,
    event_based_metrics,
    false_positives_per_minute,
    mean_sd,
    pr_curve,
    psds_crosscheck,
    sed_eval_available,
    sed_eval_metrics,
    severity_spearman,
    sex_score_gap,
    targets_markdown,
    typing_metrics,
)

from rostrum.detect.tune import Event


def ev(label: str, t0: float, t1: float, file_id: str = "f1") -> Event:
    """One event, with the file id defaulted so the fixtures stay readable."""
    return Event(label=label, t_start=t0, t_end=t1, file_id=file_id)


# --------------------------------------------------------------------------- #
# Event-based F1
# --------------------------------------------------------------------------- #


def test_an_exact_prediction_scores_one():
    reference = [ev("rushing", 1.0, 2.0)]
    assert event_based_metrics(reference, list(reference))["f1"] == 1.0


def test_a_prediction_inside_the_collar_still_matches():
    # 150 ms onset error, inside the 200 ms collar; offset error 150 ms, inside
    # max(200 ms, 20% of 1 s) = 200 ms.
    metrics = event_based_metrics([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.15, 2.15)])
    assert metrics["f1"] == 1.0


def test_a_prediction_outside_the_onset_collar_is_a_miss_and_a_false_positive():
    metrics = event_based_metrics([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.5, 2.5)])
    assert metrics["micro"]["tp"] == 0
    assert metrics["micro"]["fn"] == 1
    assert metrics["micro"]["fp"] == 1


def test_the_offset_collar_scales_with_the_reference_length():
    # 20% of a 5 s event is 1 s, so a 600 ms offset error matches here and would
    # not match a 1 s event.
    assert event_based_metrics([ev("monotone", 0.0, 5.0)], [ev("monotone", 0.0, 5.6)])["f1"] == 1.0
    assert event_based_metrics([ev("monotone", 0.0, 1.0)], [ev("monotone", 0.0, 1.6)])["f1"] == 0.0


def test_events_never_match_across_files():
    metrics = event_based_metrics(
        [ev("rushing", 1.0, 2.0, "a")], [ev("rushing", 1.0, 2.0, "b")]
    )
    assert metrics["micro"]["tp"] == 0


def test_events_never_match_across_labels():
    metrics = event_based_metrics([ev("rushing", 1.0, 2.0)], [ev("dragging", 1.0, 2.0)])
    assert metrics["micro"]["tp"] == 0


# --------------------------------------------------------------------------- #
# sed_eval cross-check
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not sed_eval_available(), reason="sed_eval is not importable here")
def test_sed_eval_agrees_with_the_project_matcher_on_a_clear_case():
    reference = [ev("rushing", 1.0, 2.0), ev("monotone", 5.0, 8.0, "f2")]
    predicted = [ev("rushing", 1.05, 2.05), ev("monotone", 5.1, 8.2, "f2")]
    both = crosscheck_event_f1(reference, predicted)
    assert both["sed_eval"] is not None
    assert both["rostrum"]["f1"] == 1.0
    assert both["sed_eval"]["f1"] == 1.0
    assert both["delta_f1"] == 0.0
    assert both["agrees"] is True


@pytest.mark.skipif(not sed_eval_available(), reason="sed_eval is not importable here")
def test_sed_eval_also_rejects_a_prediction_outside_the_collar():
    result = sed_eval_metrics([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.9, 2.9)])
    assert result is not None
    assert result["f1"] == 0.0


def test_the_crosscheck_degrades_to_a_note_when_sed_eval_is_missing(monkeypatch):
    monkeypatch.setattr("eval.metrics.sed_eval_metrics", lambda *a, **k: None)
    both = crosscheck_event_f1([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.0, 2.0)])
    assert both["sed_eval"] is None
    assert both["agrees"] is None
    assert "sed_eval" in both["note"]


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #


def test_boundary_error_is_measured_in_milliseconds():
    errors = boundary_errors([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.1, 2.2)])
    iou_matched = errors["iou_matched"]
    assert iou_matched["onset_mae_ms"] == pytest.approx(100.0, abs=1e-6)
    assert iou_matched["offset_mae_ms"] == pytest.approx(200.0, abs=1e-6)
    assert iou_matched["mae_ms"] == pytest.approx(150.0, abs=1e-6)
    assert iou_matched["n"] == 1


def test_a_late_prediction_has_a_positive_bias():
    errors = boundary_errors([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.1, 2.1)])
    assert errors["iou_matched"]["onset_bias_ms"] == pytest.approx(100.0, abs=1e-6)


def test_the_iou_matching_sees_errors_the_collar_throws_away():
    # 500 ms late: outside the 200 ms collar, so collar matching reports
    # nothing, while the overlap is 0.5/1.5 = 0.33 so IoU matching still
    # measures the real error. This is why the headline MAE uses IoU matching -
    # spec 14.3's target would otherwise be bounded by its own collar.
    errors = boundary_errors([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.5, 2.5)])
    assert errors["collar_matched"]["n"] == 0
    assert errors["iou_matched"]["n"] == 1
    assert errors["iou_matched"]["onset_mae_ms"] == pytest.approx(500.0, abs=1e-6)


def test_a_non_overlapping_prediction_contributes_no_boundary_error():
    errors = boundary_errors([ev("rushing", 1.0, 2.0)], [ev("rushing", 9.0, 10.0)])
    assert errors["iou_matched"]["n"] == 0
    assert errors["iou_matched"]["mae_ms"] == 0.0


def test_matching_is_one_to_one():
    reference = [ev("rushing", 1.0, 2.0)]
    predicted = [ev("rushing", 1.0, 2.0), ev("rushing", 1.02, 2.02)]
    assert boundary_errors(reference, predicted)["iou_matched"]["n"] == 1


# --------------------------------------------------------------------------- #
# Typing
# --------------------------------------------------------------------------- #


def test_perfect_typing_has_macro_f1_one():
    result = typing_metrics(["rushing", "monotone"], ["rushing", "monotone"])
    assert result["macro_f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_the_confusion_matrix_is_truth_by_prediction():
    result = typing_metrics(["rushing", "rushing"], ["rushing", "monotone"])
    i = result["labels"].index("rushing")
    j = result["labels"].index("monotone")
    assert result["confusion"][i][i] == 1
    assert result["confusion"][i][j] == 1
    assert result["per_label"]["rushing"]["support"] == 2
    assert result["per_label"]["monotone"]["support"] == 0


def test_macro_f1_ignores_labels_with_no_support():
    # "dragging" is predicted but never true; averaging a zero for it would
    # punish the model for a question nobody asked.
    only_truth = typing_metrics(["rushing"], ["rushing"], labels=["rushing", "dragging"])
    assert only_truth["macro_f1"] == 1.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="y_pred"):
        typing_metrics(["rushing"], [])


# --------------------------------------------------------------------------- #
# Threshold-free PR
# --------------------------------------------------------------------------- #


def test_the_pr_sweep_finds_the_threshold_that_drops_the_false_positive():
    reference = [ev("rushing", 1.0, 2.0)]
    predicted = [
        ScoredEvent(ev("rushing", 1.0, 2.0), score=4.0),  # the true one, strong
        ScoredEvent(ev("rushing", 8.0, 9.0), score=1.0),  # a weak false alarm
    ]
    curve = pr_curve(reference, predicted)
    assert curve["best_f1"] == 1.0
    assert curve["best_threshold"] == pytest.approx(4.0)
    # At the low threshold both survive, so precision is 1/2.
    low = min(curve["points"], key=lambda row: row["threshold"])
    assert low["precision"] == pytest.approx(0.5)


def test_an_empty_prediction_list_gives_an_empty_curve():
    curve = pr_curve([ev("rushing", 1.0, 2.0)], [])
    assert curve["points"] == []
    assert curve["best_threshold"] is None


def test_the_sweep_is_subsampled_to_max_points():
    reference = [ev("rushing", 1.0, 2.0)]
    predicted = [ScoredEvent(ev("rushing", float(i), float(i) + 0.5), i / 10) for i in range(50)]
    assert pr_curve(reference, predicted, max_points=8)["n_thresholds"] <= 8


def test_psds_reports_its_own_absence_rather_than_substituting_a_number():
    result = psds_crosscheck(
        [ev("rushing", 1.0, 2.0)],
        [ScoredEvent(ev("rushing", 1.0, 2.0), 3.0)],
        durations={"f1": 10.0},
    )
    assert "available" in result
    if not result["available"]:
        assert "sed_scores_eval" in result["reason"]


# --------------------------------------------------------------------------- #
# Gradient, false alarms, fairness
# --------------------------------------------------------------------------- #


def test_a_monotone_ladder_has_rho_minus_one():
    result = severity_spearman([0.15, 0.35, 0.60, 0.90], [9.0, 7.0, 5.0, 2.0])
    assert result["rho"] == pytest.approx(-1.0)
    assert result["n"] == 4


def test_a_flat_score_has_no_rank_correlation_at_all():
    result = severity_spearman([0.15, 0.35, 0.60], [5.0, 5.0, 5.0])
    assert result["rho"] is None
    assert result["reason"] == "no variance"


def test_two_points_are_too_few_for_a_correlation():
    assert severity_spearman([0.15, 0.35], [9.0, 7.0])["rho"] is None


def test_mismatched_ladder_lengths_are_rejected():
    with pytest.raises(ValueError, match="scores"):
        severity_spearman([0.15], [9.0, 7.0])


def test_false_positives_are_counted_per_minute_of_clean_speech():
    result = false_positives_per_minute(
        [ev("rushing", 1.0, 2.0, "c1"), ev("monotone", 5.0, 6.0, "c1")],
        durations={"c1": 60.0, "c2": 60.0},
    )
    assert result["fp"] == 2
    assert result["minutes"] == pytest.approx(2.0)
    assert result["fp_per_min"] == pytest.approx(1.0)
    assert result["per_file"]["c2"] == 0


def test_a_clean_file_with_no_detection_still_counts_towards_the_minutes():
    result = false_positives_per_minute([], durations={"c1": 120.0})
    assert result["fp_per_min"] == 0.0
    assert result["minutes"] == pytest.approx(2.0)


def test_the_sex_gap_is_the_absolute_difference_of_the_means():
    gap = sex_score_gap({"a": 7.0, "b": 6.0}, {"a": "m", "b": "f"})
    assert gap["gap_points"] == pytest.approx(1.0)
    assert gap["male"]["n"] == 1
    assert gap["female"]["n"] == 1


def test_the_sex_gap_is_none_when_the_dataset_has_no_sex_register():
    gap = sex_score_gap({"a": 7.0, "b": 6.0}, {})
    assert gap["gap_points"] is None
    assert gap["n_unknown"] == 2


# --------------------------------------------------------------------------- #
# Aggregation and targets
# --------------------------------------------------------------------------- #


def test_mean_sd_skips_folds_that_could_not_report():
    result = mean_sd([0.8, None, 0.6])
    assert result["mean"] == pytest.approx(0.7)
    assert result["n"] == 2


def test_mean_sd_of_nothing_is_none():
    assert mean_sd([None])["mean"] is None


def test_a_higher_is_better_target_passes_when_met():
    rows = check_targets({"event_f1_200ms": 0.80}, {"event_f1_200ms": 0.75})
    row = next(r for r in rows if r.key == "event_f1_200ms")
    assert row.verdict == "PASS"


def test_a_lower_is_better_target_misses_when_exceeded():
    rows = check_targets({"onset_offset_mae_ms": 200.0}, {"onset_offset_mae_ms": 150.0})
    row = next(r for r in rows if r.key == "onset_offset_mae_ms")
    assert row.verdict == "MISS"


def test_spearman_is_judged_as_lower_is_better():
    # rho = -0.9 beats the -0.85 target; rho = -0.5 does not.
    assert TARGET_DIRECTIONS["severity_spearman_rho"] == "lower"
    passing = check_targets({"severity_spearman_rho": -0.9}, {"severity_spearman_rho": -0.85})
    missing = check_targets({"severity_spearman_rho": -0.5}, {"severity_spearman_rho": -0.85})
    assert passing[3].verdict == "PASS"
    assert missing[3].verdict == "MISS"


def test_an_unmeasurable_metric_is_reported_not_dropped():
    rows = check_targets({"event_f1_200ms": None}, {"event_f1_200ms": 0.75})
    row = next(r for r in rows if r.key == "event_f1_200ms")
    assert row.verdict == "N/A"
    assert row.note == "not measurable in this run"


def test_a_split_without_a_target_says_so():
    rows = check_targets({"utmos_drop_at_l1": 0.1}, {})
    row = next(r for r in rows if r.key == "utmos_drop_at_l1")
    assert row.verdict == "N/A"
    assert "no pre-registered target" in row.note


def test_the_markdown_table_has_one_row_per_target():
    rows = check_targets({"event_f1_200ms": 0.8}, {"event_f1_200ms": 0.75})
    table = targets_markdown(rows, title="Synthetic")
    assert "### Synthetic" in table
    assert table.count("\n") == len(rows) + 4  # heading, blank, header, rule, rows
    assert "PASS" in table
