"""Tests for event-F1 and the training-fold-only threshold search (spec 10.4(5), 14.1)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.detect.regions import RegionThresholds, label_category
from rostrum.detect.tune import (
    Event,
    MatchCounts,
    TuningCase,
    TuningResult,
    event_f1,
    events_from_regions,
    grid_search,
    read_tuning,
    write_tuning,
)
from rostrum.detect.zscore import Signal, ZResult
from rostrum.types import Region


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def ev(label: str, t0: float, t1: float, file_id: str = "f1") -> Event:
    return Event(label=label, t_start=t0, t_end=t1, file_id=file_id)


# --------------------------------------------------------------------------- #
# Counts
# --------------------------------------------------------------------------- #


def test_match_counts_rates():
    counts = MatchCounts(tp=3, fp=1, fn=1)
    assert counts.precision == pytest.approx(0.75)
    assert counts.recall == pytest.approx(0.75)
    assert counts.f1 == pytest.approx(0.75)


def test_match_counts_degenerate_cases_are_zero_not_nan():
    assert MatchCounts().f1 == 0.0
    assert MatchCounts(fp=2).precision == 0.0
    assert MatchCounts(fn=2).recall == 0.0


# --------------------------------------------------------------------------- #
# Event-F1
# --------------------------------------------------------------------------- #


def test_an_exact_match_is_a_true_positive():
    metrics = event_f1([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.0, 2.0)])
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["micro"]["tp"] == 1


def test_an_onset_inside_the_collar_matches():
    metrics = event_f1([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.19, 2.0)])
    assert metrics["micro"]["tp"] == 1


def test_an_onset_outside_the_collar_does_not_match():
    metrics = event_f1([ev("rushing", 1.0, 2.0)], [ev("rushing", 1.25, 2.0)])
    assert metrics["micro"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def test_the_offset_collar_scales_with_the_reference_length():
    """max(200 ms, 20% of length): a 3 s region gets 600 ms, a 0.3 s one 200 ms."""
    long_ref = [ev("monotone", 0.0, 3.0)]
    assert event_f1(long_ref, [ev("monotone", 0.0, 3.55)])["micro"]["tp"] == 1
    assert event_f1(long_ref, [ev("monotone", 0.0, 3.65)])["micro"]["tp"] == 0

    short_ref = [ev("missing_pause", 0.0, 0.3)]
    assert event_f1(short_ref, [ev("missing_pause", 0.0, 0.49)])["micro"]["tp"] == 1
    assert event_f1(short_ref, [ev("missing_pause", 0.0, 0.55)])["micro"]["tp"] == 0


def test_labels_never_match_across_each_other():
    metrics = event_f1([ev("rushing", 1.0, 2.0)], [ev("dragging", 1.0, 2.0)])
    assert metrics["micro"]["tp"] == 0
    assert metrics["per_label"]["rushing"]["fn"] == 1
    assert metrics["per_label"]["dragging"]["fp"] == 1


def test_events_never_match_across_files():
    metrics = event_f1([ev("rushing", 1.0, 2.0, "a")], [ev("rushing", 1.0, 2.0, "b")])
    assert metrics["micro"]["tp"] == 0


def test_matching_is_one_to_one():
    """Two predictions over one reference give one TP and one FP, not two TPs."""
    metrics = event_f1(
        [ev("rushing", 1.0, 2.0)],
        [ev("rushing", 1.0, 2.0), ev("rushing", 1.05, 2.05)],
    )
    assert metrics["micro"]["tp"] == 1
    assert metrics["micro"]["fp"] == 1


def test_the_nearest_onset_wins_the_assignment():
    metrics = event_f1(
        [ev("rushing", 1.0, 2.0), ev("rushing", 5.0, 6.0)],
        [ev("rushing", 1.15, 2.0), ev("rushing", 1.02, 2.0)],
    )
    # Both predictions are near the first reference; one matches, one is a FP,
    # and the second reference is a FN.
    assert metrics["micro"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_empty_inputs_score_zero_without_crashing():
    assert event_f1([], [])["f1"] == 0.0
    assert event_f1([], [ev("rushing", 1.0, 2.0)])["micro"]["fp"] == 1
    assert event_f1([ev("rushing", 1.0, 2.0)], [])["micro"]["fn"] == 1


def test_macro_f1_averages_over_labels():
    metrics = event_f1(
        [ev("rushing", 1.0, 2.0), ev("dragging", 5.0, 6.0)],
        [ev("rushing", 1.0, 2.0)],
    )
    assert metrics["per_label"]["rushing"]["f1"] == pytest.approx(1.0)
    assert metrics["per_label"]["dragging"]["f1"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == pytest.approx(0.5)


def test_the_collars_are_reported_with_the_metrics():
    metrics = event_f1([ev("rushing", 1.0, 2.0)], [])
    assert metrics["collars"] == {"onset_s": 0.200, "offset_s": 0.200, "offset_frac": 0.20}


# --------------------------------------------------------------------------- #
# Regions to events
# --------------------------------------------------------------------------- #


def region(label, t0, t1, *, context="", region_id=0) -> Region:
    return Region(
        region_id=region_id,
        t_start=t0,
        t_end=t1,
        label=label,
        category=label_category(label),
        context=context,
    )


def test_the_fading_variant_scores_as_low_energy():
    events = events_from_regions([region("low_energy", 1.0, 2.0, context="fading")])
    assert [e.label for e in events] == ["low_energy"]


def test_overlapping_same_label_regions_coalesce_into_one_event():
    """A level region and a fading region over one passage are one flaw."""
    events = events_from_regions(
        [
            region("low_energy", 1.0, 2.0),
            region("low_energy", 1.5, 2.4, context="fading", region_id=1),
        ]
    )
    assert len(events) == 1
    assert (events[0].t_start, events[0].t_end) == (1.0, 2.4)


def test_disjoint_same_label_regions_stay_separate():
    events = events_from_regions(
        [region("rushing", 1.0, 2.0), region("rushing", 5.0, 6.0, region_id=1)]
    )
    assert len(events) == 2


def test_the_file_id_travels_with_the_events():
    events = events_from_regions([region("rushing", 1.0, 2.0)], file_id="rec7")
    assert events[0].file_id == "rec7"


# --------------------------------------------------------------------------- #
# Grid search
# --------------------------------------------------------------------------- #


def make_case(file_id: str, text_id: str, *, peak: float, span=(3, 6), n: int = 12) -> TuningCase:
    """A case whose rate z is ``peak`` over ``span`` and ~0 elsewhere."""
    index = pd.Index(range(n), name="unit_idx", dtype="int64")
    feature = "rate_local_syll_per_s"
    z = np.zeros(n)
    z[span[0] : span[1] + 1] = peak
    # A neighbour just above a loose tau_out, so the choice of tau_out changes
    # the extent and therefore the offset error - which is what the search must
    # discover for itself.
    if span[1] + 1 < n:
        z[span[1] + 1] = 1.7

    starts = np.arange(n) * 0.40
    table = pd.DataFrame(
        {
            "text": [f"w{i}" for i in range(n)],
            "t_start": starts,
            "t_end": starts + 0.35,
            feature: 5.0 + z * 0.3,
        },
        index=index,
    )
    table.attrs.update(unit="word", file_id=file_id, channel_id="ch_a")
    frame = pd.DataFrame({feature: z}, index=index)
    result = ZResult(
        text_id=text_id,
        fold="f",
        unit="word",
        z=frame,
        value=pd.DataFrame({feature: 5.0 + z * 0.3}, index=index),
        median=pd.DataFrame({feature: np.full(n, 5.0)}, index=index),
        sigma=pd.DataFrame({feature: np.full(n, 0.3)}, index=index),
        n_references=pd.Series(np.full(n, 6), index=index, dtype="int64"),
        table=table,
    )
    signal = Signal(
        label="rushing",
        unit="word",
        primary_feature="rate_local",
        score=frame[feature],
        gate=pd.Series(True, index=index, dtype=bool),
        evidence_features=(feature,),
    )
    truth = [
        Event(
            "rushing",
            float(table.loc[span[0], "t_start"]),
            float(table.loc[span[1], "t_end"]),
            file_id=file_id,
        )
    ]
    return TuningCase(
        file_id=file_id,
        text_id=text_id,
        signals=[signal],
        z_by_unit={"word": result},
        truth=truth,
        frames=None,
    )


def test_grid_search_finds_thresholds_that_beat_the_defaults(cfg):
    cases = [make_case(f"train{i}", "A", peak=5.0) for i in range(3)]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)

    default = RegionThresholds.from_config(cfg)
    baseline = next(
        c for c in tuned.grid if c["tau_in"] == default.tau_in and c["tau_out"] == default.tau_out
    )
    assert tuned.event_f1 >= baseline["event_f1"]
    assert tuned.tau_out <= tuned.tau_in
    assert tuned.n_cases == 3
    assert tuned.n_reference_events == 3


def test_grid_search_excludes_cases_from_non_training_texts(cfg):
    """The leakage guard of CLAUDE.md invariant 11, enforced in the function."""
    cases = [
        make_case("train0", "A", peak=5.0),
        make_case("train1", "A", peak=5.0),
        make_case("held0", "B", peak=5.0),
    ]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)
    assert tuned.n_cases == 2, "the held-out text must not reach the search"
    assert tuned.train_texts == ("A",)
    assert tuned.n_reference_events == 2


def test_grid_search_refuses_to_run_with_no_training_case(cfg):
    cases = [make_case("held0", "B", peak=5.0)]
    with pytest.raises(ValueError, match="no tuning case belongs"):
        grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg)


def test_an_empty_grid_is_an_error(cfg):
    cases = [make_case("train0", "A", peak=5.0)]
    with pytest.raises(ValueError, match="empty grid"):
        grid_search(
            cases,
            fold="loto_B",
            train_texts=["A"],
            config=cfg,
            tau_in_grid=[1.0],
            tau_out_grid=[2.0],
            refine=False,
        )


def test_grid_search_is_deterministic(cfg):
    cases = [make_case(f"train{i}", "A", peak=5.0) for i in range(3)]
    runs = [
        grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False).to_dict()
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


def test_ties_are_broken_towards_the_stricter_trigger(cfg):
    """Fewer false positives on clean speech is the documented preference."""
    cases = [make_case(f"train{i}", "A", peak=50.0) for i in range(2)]
    tuned = grid_search(
        cases,
        fold="loto_B",
        train_texts=["A"],
        config=cfg,
        tau_in_grid=[2.0, 3.0],
        tau_out_grid=[1.0],
        refine=False,
    )
    best = max(c["event_f1"] for c in tuned.grid)
    tied = [c for c in tuned.grid if c["event_f1"] == best]
    if len(tied) > 1:
        assert tuned.tau_in == min(c["tau_in"] for c in tied)


def test_the_grid_log_records_every_cell(cfg):
    cases = [make_case("train0", "A", peak=5.0)]
    tuned = grid_search(
        cases,
        fold="loto_B",
        train_texts=["A"],
        config=cfg,
        tau_in_grid=[2.0, 2.5, 3.0],
        tau_out_grid=[1.0, 1.5],
        refine=False,
    )
    assert len(tuned.grid) == 6
    for cell in tuned.grid:
        assert set(cell) == {
            "tau_in",
            "tau_out",
            "penalty",
            "event_f1",
            "precision",
            "recall",
            "n_predicted",
        }


def test_thresholds_round_trip_through_the_result(cfg):
    cases = [make_case("train0", "A", peak=5.0)]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)
    thresholds = tuned.thresholds(config=cfg)
    assert thresholds.tau_in == pytest.approx(tuned.tau_in)
    assert thresholds.tau_out == pytest.approx(tuned.tau_out)
    # The non-tuned fields keep their configured values.
    default = RegionThresholds.from_config(cfg)
    assert thresholds.merge_gap_s == pytest.approx(default.merge_gap_s)
    assert thresholds.min_duration_s == pytest.approx(default.min_duration_s)
    assert thresholds.smooth_exempt_labels == default.smooth_exempt_labels


def test_the_staged_and_joint_searches_agree_on_the_thresholds(cfg):
    cases = [make_case(f"train{i}", "A", peak=5.0) for i in range(3)]
    staged = grid_search(
        cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False, staged=True
    )
    joint = grid_search(
        cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False, staged=False
    )
    # With refinement off the two searches are the same grid, so they must
    # choose the same cell.
    assert (staged.tau_in, staged.tau_out) == (joint.tau_in, joint.tau_out)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_tuning_round_trips_through_json(tmp_path, cfg):
    cases = [make_case("train0", "A", peak=5.0)]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)
    path = write_tuning([tuned], tmp_path / "tuning.json", config=cfg)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["objective"] == "event_f1"
    assert data["fit_scope"] == "training_folds_only"
    assert data["config_hash"] == cfg.hash

    back = read_tuning(path)
    assert set(back) == {"loto_B"}
    assert back["loto_B"].tau_in == pytest.approx(tuned.tau_in)
    assert back["loto_B"].train_texts == ("A",)


def test_writing_tuning_twice_gives_identical_bytes(tmp_path, cfg):
    """Determinism: no timestamp, sorted keys (CLAUDE.md invariant 13)."""
    cases = [make_case("train0", "A", peak=5.0)]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)
    first = write_tuning([tuned], tmp_path / "a.json", config=cfg).read_bytes()
    second = write_tuning([tuned], tmp_path / "b.json", config=cfg).read_bytes()
    assert first == second


def test_reading_a_missing_tuning_file_is_empty_not_an_error(tmp_path):
    assert read_tuning(tmp_path / "absent.json") == {}


def test_tuning_result_from_dict_is_the_inverse_of_to_dict():
    result = TuningResult(
        fold="loto_X",
        train_texts=("A", "B"),
        tau_in=2.75,
        tau_out=1.25,
        penalty=5.0,
        event_f1=0.8123,
        metrics={"f1": 0.8123},
        grid=[{"tau_in": 2.75, "tau_out": 1.25, "penalty": 5.0, "event_f1": 0.8123}],
        n_cases=4,
        n_reference_events=9,
    )
    back = TuningResult.from_dict(result.to_dict())
    assert back.fold == result.fold
    assert back.train_texts == result.train_texts
    assert back.tau_in == pytest.approx(result.tau_in)
    assert back.penalty == pytest.approx(result.penalty)
    assert back.n_reference_events == 9


def test_a_non_dataset_source_is_recorded_in_the_file(tmp_path, cfg):
    """Synthetic thresholds must never masquerade as dataset-fitted ones."""
    cases = [make_case("train0", "A", peak=5.0)]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)

    path = write_tuning([tuned], tmp_path / "t.json", config=cfg, source="synthetic")
    assert json.loads(path.read_text(encoding="utf-8"))["source"] == "synthetic"

    default = write_tuning([tuned], tmp_path / "d.json", config=cfg)
    assert json.loads(default.read_text(encoding="utf-8"))["source"] == "dataset"


def test_reading_a_synthetic_tuning_file_warns(tmp_path, cfg, caplog):
    cases = [make_case("train0", "A", peak=5.0)]
    tuned = grid_search(cases, fold="loto_B", train_texts=["A"], config=cfg, refine=False)
    path = write_tuning([tuned], tmp_path / "t.json", config=cfg, source="synthetic")

    import logging

    with caplog.at_level(logging.WARNING, logger="rostrum.detect.tune"):
        read_tuning(path)
    assert any("not on the dataset" in r.message for r in caplog.records)
