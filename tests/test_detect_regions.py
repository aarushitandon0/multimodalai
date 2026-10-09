"""Tests for smoothing, hysteresis, merge/drop and evidence (spec 10.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.detect.regions import (
    PAUSE_LABELS,
    RegionThresholds,
    build_regions,
    drop_short,
    hysteresis_segments,
    label_category,
    label_counts,
    median_smooth,
    merge_adjacent,
)
from rostrum.detect.zscore import Signal, ZResult
from rostrum.types import Evidence, Region


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def series(values) -> pd.Series:
    return pd.Series(
        np.asarray(values, dtype=float),
        index=pd.Index(range(len(values)), name="unit_idx", dtype="int64"),
    )


# --------------------------------------------------------------------------- #
# Median smoothing
# --------------------------------------------------------------------------- #


def test_median_smooth_removes_a_single_unit_spike():
    # The whole point of 10.4(1): one word crossing the threshold alone is an
    # alignment artefact, and a 3-unit median removes it.
    smoothed = median_smooth(series([0, 0, 9, 0, 0]), 3)
    assert list(smoothed) == [0, 0, 0, 0, 0]


def test_median_smooth_keeps_a_three_unit_run():
    smoothed = median_smooth(series([0, 9, 9, 9, 0]), 3)
    assert list(smoothed) == [0.0, 9.0, 9.0, 9.0, 0.0]


def test_median_smooth_does_not_dilate_a_run():
    """A median filter must not widen a region; that would break the collar."""
    smoothed = median_smooth(series([0, 0, 5, 5, 5, 5, 0, 0]), 3)
    assert smoothed.iloc[1] == 0.0
    assert smoothed.iloc[6] == 0.0


def test_median_smooth_treats_nan_as_missing():
    smoothed = median_smooth(series([1.0, np.nan, 3.0]), 3)
    assert smoothed.iloc[1] == pytest.approx(2.0)
    assert median_smooth(series([np.nan, np.nan, np.nan]), 3).isna().all()


def test_median_smooth_shrinks_the_edges_symmetrically():
    # The window stays centred and stays odd, so the first unit keeps its own
    # value. Truncating on one side only would leave an even-length window at
    # the edge, whose median is a mean, and that dilates a region: see
    # test_median_smooth_does_not_dilate_a_run_at_the_edges.
    assert median_smooth(series([4.0, 0.0, 0.0]), 3).iloc[0] == pytest.approx(4.0)
    assert median_smooth(series([0.0, 0.0, 4.0]), 3).iloc[-1] == pytest.approx(4.0)


def test_median_smooth_does_not_dilate_a_run_at_the_edges():
    # A one-sided truncation would give 4.5 at units 0 and 4 here - above
    # tau_in, so a spurious region would open at the utterance onset.
    smoothed = median_smooth(series([0, 9, 9, 9, 0]), 3)
    assert smoothed.iloc[0] == pytest.approx(0.0)
    assert smoothed.iloc[-1] == pytest.approx(0.0)


def test_median_smooth_width_one_is_a_passthrough():
    raw = series([1.0, 9.0, 1.0])
    assert list(median_smooth(raw, 1)) == list(raw)


def test_even_width_is_rejected():
    with pytest.raises(ValueError, match="odd width"):
        median_smooth(series([1.0, 2.0, 3.0]), 2)


# --------------------------------------------------------------------------- #
# Hysteresis
# --------------------------------------------------------------------------- #


def test_hysteresis_opens_above_tau_in_and_stays_above_tau_out():
    segments = hysteresis_segments(series([0.0, 1.8, 2.9, 1.7, 0.5]), tau_in=2.5, tau_out=1.5)
    assert len(segments) == 1
    # It extends backwards too: unit 1 was already above the stay threshold.
    assert (segments[0].first, segments[0].last) == (1, 3)
    assert segments[0].peak == pytest.approx(2.9)


def test_a_score_below_tau_in_never_opens_a_region():
    assert hysteresis_segments(series([2.4, 2.4, 2.4]), tau_in=2.5, tau_out=1.5) == []


def test_hysteresis_does_not_chatter_around_one_threshold():
    """A single threshold would split this into three regions; hysteresis gives one."""
    score = series([2.6, 2.4, 2.6, 2.4, 2.6])
    assert len(hysteresis_segments(score, tau_in=2.5, tau_out=1.5)) == 1
    assert len(hysteresis_segments(score, tau_in=2.5, tau_out=2.5)) == 3


def test_two_separated_runs_stay_separate():
    segments = hysteresis_segments(series([3.0, 0.0, 0.0, 3.0]), tau_in=2.5, tau_out=1.5)
    assert [(s.first, s.last) for s in segments] == [(0, 0), (3, 3)]


def test_a_gate_keeps_a_unit_out_of_every_region():
    score = series([3.0, 3.0, 3.0])
    gate = pd.Series([True, False, True], index=score.index)
    segments = hysteresis_segments(score, tau_in=2.5, tau_out=1.5, gate=gate)
    assert [(s.first, s.last) for s in segments] == [(0, 0), (2, 2)]


def test_nan_breaks_a_run():
    segments = hysteresis_segments(series([3.0, np.nan, 3.0]), tau_in=2.5, tau_out=1.5)
    assert [(s.first, s.last) for s in segments] == [(0, 0), (2, 2)]


def test_tau_out_above_tau_in_is_rejected():
    with pytest.raises(ValueError, match="tau_out <= tau_in"):
        hysteresis_segments(series([3.0]), tau_in=1.0, tau_out=2.0)
    with pytest.raises(ValueError, match="tau_out <= tau_in"):
        RegionThresholds(tau_in=1.0, tau_out=2.0)


# --------------------------------------------------------------------------- #
# Smoothing exemptions
# --------------------------------------------------------------------------- #


def test_pause_labels_and_phrases_are_not_smoothed(cfg):
    thresholds = RegionThresholds.from_config(cfg)
    assert thresholds.width_for("missing_pause", "gap") == 1
    assert thresholds.width_for("hesitation_pause", "gap") == 1
    assert thresholds.width_for("monotone", "phrase") == 1
    assert thresholds.width_for("rushing", "word") == thresholds.smooth_units == 3


def test_a_single_gap_flaw_survives_because_pauses_opt_out():
    """Without the exemption, a 3-gap median would erase every pause flaw."""
    score = series([0.0, 0.0, 12.0, 0.0, 0.0])
    assert hysteresis_segments(median_smooth(score, 3), tau_in=2.5, tau_out=1.5) == []
    assert len(hysteresis_segments(median_smooth(score, 1), tau_in=2.5, tau_out=1.5)) == 1


# --------------------------------------------------------------------------- #
# Merge and drop
# --------------------------------------------------------------------------- #


def region(label, t0, t1, *, context="", peak=3.0, excess=1.0, words=(0, 0)) -> Region:
    return Region(
        region_id=-1,
        t_start=t0,
        t_end=t1,
        label=label,
        category=label_category(label),
        context=context,
        peak_z=peak,
        mean_excess_z=excess,
        word_idx=list(words),
    )


def test_regions_closer_than_the_merge_gap_become_one():
    merged = merge_adjacent([region("rushing", 1.0, 1.5), region("rushing", 1.6, 2.0)], 0.200)
    assert len(merged) == 1
    assert (merged[0].t_start, merged[0].t_end) == (1.0, 2.0)


def test_regions_further_apart_than_the_merge_gap_stay_separate():
    merged = merge_adjacent([region("rushing", 1.0, 1.5), region("rushing", 1.75, 2.0)], 0.200)
    assert len(merged) == 2


def test_different_labels_never_merge():
    merged = merge_adjacent([region("rushing", 1.0, 1.5), region("low_energy", 1.55, 2.0)], 0.200)
    assert len(merged) == 2


def test_a_variant_does_not_merge_with_its_own_label():
    """A low level and a falling slope are two statements about the delivery."""
    merged = merge_adjacent(
        [
            region("low_energy", 1.0, 1.5),
            region("low_energy", 1.55, 2.0, context="fading"),
        ],
        0.200,
    )
    assert len(merged) == 2


def test_merging_keeps_the_stronger_peak_and_weights_the_excess():
    merged = merge_adjacent(
        [
            region("rushing", 0.0, 1.0, peak=3.0, excess=1.0),
            region("rushing", 1.1, 1.5, peak=8.0, excess=5.0),
        ],
        0.200,
    )
    assert len(merged) == 1
    assert merged[0].peak_z == pytest.approx(8.0)
    expected = (1.0 * 1.0 + 5.0 * 0.4) / 1.4
    assert merged[0].mean_excess_z == pytest.approx(expected)


def test_short_regions_are_dropped():
    kept = drop_short([region("rushing", 0.0, 0.1), region("rushing", 1.0, 1.5)], 0.250)
    assert [(r.t_start, r.t_end) for r in kept] == [(1.0, 1.5)]


def test_pause_labels_are_exempt_from_the_duration_floor():
    """A too-short pause IS a short region; the floor would make it invisible."""
    for label in sorted(PAUSE_LABELS):
        kept = drop_short([region(label, 1.0, 1.04)], 0.250)
        assert len(kept) == 1, label


def test_the_exempt_list_comes_from_the_config(cfg):
    assert set(cfg.detection["regions"]["min_duration_exempt_labels"]) == set(PAUSE_LABELS)


# --------------------------------------------------------------------------- #
# The whole of 10.4, end to end on a crafted z sequence
# --------------------------------------------------------------------------- #


def word_z(values, *, feature="rate_local_syll_per_s", word_s=0.35, gap_s=0.05) -> ZResult:
    """A ZResult over equally spaced words, carrying one feature."""
    n = len(values)
    index = pd.Index(range(n), name="unit_idx", dtype="int64")
    starts = np.arange(n) * (word_s + gap_s)
    table = pd.DataFrame(
        {
            "text": [f"w{i}" for i in range(n)],
            "t_start": starts,
            "t_end": starts + word_s,
            feature: np.asarray(values, dtype=float),
        },
        index=index,
    )
    table.attrs.update(unit="word", file_id="p", channel_id="ch_a")
    z = pd.DataFrame({feature: np.asarray(values, dtype=float)}, index=index)
    return ZResult(
        text_id="T",
        fold="f",
        unit="word",
        z=z,
        value=pd.DataFrame({feature: np.asarray(values, dtype=float) * 0.1 + 5.0}, index=index),
        median=pd.DataFrame({feature: np.full(n, 5.0)}, index=index),
        sigma=pd.DataFrame({feature: np.full(n, 0.1)}, index=index),
        n_references=pd.Series(np.full(n, 6), index=index, dtype="int64"),
        table=table,
        channel_matched=True,
    )


def rushing_signal(result: ZResult) -> Signal:
    feature = result.features[0]
    return Signal(
        label="rushing",
        unit="word",
        primary_feature="rate_local",
        score=result.z[feature],
        gate=pd.Series(True, index=result.z.index, dtype=bool),
        evidence_features=(feature,),
    )


def test_a_crafted_z_sequence_gives_the_expected_region(cfg):
    # Words 3-6 are 4 sigma out; nothing else crosses.
    values = [0.2, -0.4, 0.3, 4.0, 4.2, 3.9, 4.1, 0.1, -0.2, 0.0]
    result = word_z(values)
    regions = build_regions([rushing_signal(result)], {"word": result}, config=cfg)

    assert len(regions) == 1
    found = regions[0]
    assert found.label == "rushing"
    assert found.category == "pace"
    assert found.word_idx == [3, 6]
    assert found.t_start == pytest.approx(result.table.loc[3, "t_start"])
    assert found.t_end == pytest.approx(result.table.loc[6, "t_end"])
    assert found.words == "w3 w4 w5 w6"
    assert found.region_id == 0
    assert found.peak_z == pytest.approx(4.1)
    assert found.envelope_n == 6


def test_a_lone_spike_produces_no_region(cfg):
    result = word_z([0.0, 0.0, 9.0, 0.0, 0.0, 0.0])
    assert build_regions([rushing_signal(result)], {"word": result}, config=cfg) == []


def test_a_two_word_run_survives_the_duration_floor(cfg):
    # 2 words x 0.35 s = 0.70 s, comfortably over the 250 ms floor.
    result = word_z([0.0, 0.0, 4.0, 4.0, 0.0, 0.0])
    regions = build_regions([rushing_signal(result)], {"word": result}, config=cfg)
    assert len(regions) == 1
    assert regions[0].word_idx == [2, 3]


def test_evidence_puts_the_primary_feature_first_with_its_numbers(cfg):
    result = word_z([0.0, 0.0, 4.0, 4.0, 0.0, 0.0])
    regions = build_regions([rushing_signal(result)], {"word": result}, config=cfg)
    evidence = regions[0].evidence
    assert evidence, "a region must carry its evidence"
    assert isinstance(evidence[0], Evidence)
    assert evidence[0].feature == "rate_local"
    assert evidence[0].unit == "syll/s"
    assert evidence[0].z == pytest.approx(4.0)
    assert evidence[0].env_median == pytest.approx(5.0)
    assert evidence[0].env_sigma == pytest.approx(0.1)
    assert evidence[0].n_references == 6
    assert evidence[0].channel_matched is True


def test_at_most_three_evidence_rows(cfg):
    n = 8
    index = pd.Index(range(n), name="unit_idx", dtype="int64")
    columns = {f"f{k}": np.full(n, 4.0 if k == 0 else 3.0) for k in range(6)}
    table = pd.DataFrame(
        {
            "text": [f"w{i}" for i in range(n)],
            "t_start": np.arange(n) * 0.4,
            "t_end": np.arange(n) * 0.4 + 0.35,
            **columns,
        },
        index=index,
    )
    table.attrs.update(unit="word", file_id="p", channel_id="ch_a")
    frame = pd.DataFrame(columns, index=index)
    result = ZResult(
        text_id="T",
        fold="f",
        unit="word",
        z=frame,
        value=frame + 5.0,
        median=pd.DataFrame({k: np.full(n, 5.0) for k in columns}, index=index),
        sigma=pd.DataFrame({k: np.full(n, 0.1) for k in columns}, index=index),
        n_references=pd.Series(np.full(n, 5), index=index, dtype="int64"),
        table=table,
    )
    signal = Signal(
        label="rushing",
        unit="word",
        primary_feature="f0",
        score=frame["f0"],
        gate=pd.Series(True, index=index, dtype=bool),
        evidence_features=tuple(columns),
    )
    regions = build_regions([signal], {"word": result}, config=cfg)
    assert len(regions[0].evidence) == 3


def test_unrefined_boundaries_are_recorded_before_any_refinement(cfg):
    result = word_z([0.0, 4.0, 4.0, 4.0, 0.0])
    regions = build_regions([rushing_signal(result)], {"word": result}, config=cfg)
    found = regions[0]
    assert found.refined is False
    assert found.t_start_unrefined == pytest.approx(found.t_start)
    assert found.t_end_unrefined == pytest.approx(found.t_end)


def test_a_signal_on_a_missing_unit_is_an_error(cfg):
    result = word_z([4.0, 4.0, 4.0])
    signal = Signal(
        label="monotone",
        unit="phrase",
        primary_feature="f0_range_st",
        score=result.z[result.features[0]],
        gate=pd.Series(True, index=result.z.index, dtype=bool),
    )
    with pytest.raises(KeyError, match="no such comparison"):
        build_regions([signal], {"word": result}, config=cfg)


def test_region_ids_follow_onset_order(cfg):
    result = word_z([4.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0, 4.0, 4.0])
    regions = build_regions([rushing_signal(result)], {"word": result}, config=cfg)
    assert [r.region_id for r in regions] == list(range(len(regions)))
    assert regions == sorted(regions, key=lambda r: r.t_start)


def test_label_counts_separates_variants():
    counts = label_counts(
        [
            region("low_energy", 0.0, 1.0),
            region("low_energy", 2.0, 3.0, context="fading"),
            region("rushing", 4.0, 5.0),
        ]
    )
    assert counts == {"low_energy": 1, "low_energy:fading": 1, "rushing": 1}


def test_label_category_reads_the_config(cfg):
    assert label_category("rushing", cfg) == "pace"
    assert label_category("missing_pause", cfg) == "pausing"
    assert label_category("mumbling", cfg) == "clarity"
    with pytest.raises(KeyError):
        label_category("not_a_label", cfg)


# --------------------------------------------------------------------------- #
# extend_on: the smoothing-dilation ablation (spec 10.4(1) vs 10.4(2))
# --------------------------------------------------------------------------- #


def test_the_default_extend_on_is_the_spec_literal_reading(cfg):
    # Spec 10.4 smooths (step 1) before the hysteresis (step 2).
    assert RegionThresholds().extend_on == "smoothed"
    assert RegionThresholds.from_config(cfg).extend_on == "smoothed"


def test_an_unknown_extend_on_is_rejected():
    with pytest.raises(ValueError, match="extend_on must be one of"):
        RegionThresholds(extend_on="whatever")


def test_the_smoother_dilates_a_region_by_one_unit_at_its_edge():
    """The defect `extend_on` exists to measure.

    At a region edge the 3-point window holds one in-flaw value, the edge unit
    and the next unit, so its median is ``max(edge, next)``. Unit 4 measured
    -1.9 - far below tau_out - but its smoothed value is 2.1, carried there by
    unit 5, so the region swallows a unit whose own measurement says stop.
    """
    raw = series([0.3, 8.0, 5.0, 6.0, -1.9, 2.1, -0.2])
    smoothed = median_smooth(raw, 3)
    assert smoothed.iloc[4] == pytest.approx(2.1)

    # One unit of dilation: unit 4 joins although it measured -1.9.
    dilated = hysteresis_segments(smoothed, tau_in=3.5, tau_out=2.0)
    assert [(s.first, s.last) for s in dilated] == [(1, 4)]

    honest = hysteresis_segments(smoothed, tau_in=3.5, tau_out=2.0, extend_score=raw)
    assert [(s.first, s.last) for s in honest] == [(1, 3)]


def test_extend_on_raw_still_uses_the_smoothed_score_to_trigger():
    """The smoother keeps its job: a single-unit spike must not open a region."""
    raw = series([0.0, 0.0, 9.0, 0.0, 0.0])
    smoothed = median_smooth(raw, 3)
    assert hysteresis_segments(smoothed, tau_in=2.5, tau_out=1.5, extend_score=raw) == []


def test_extend_score_must_share_the_index():
    raw = series([1.0, 2.0, 3.0])
    other = pd.Series([1.0, 2.0], index=pd.Index([0, 1], name="unit_idx"))
    with pytest.raises(ValueError, match="share the index"):
        hysteresis_segments(raw, tau_in=2.5, tau_out=1.5, extend_score=other)


def test_peak_is_measured_on_the_trigger_score_in_both_modes():
    raw = series([0.3, 8.0, 5.0, 6.0, -1.9, 2.1, -0.2])
    smoothed = median_smooth(raw, 3)
    a = hysteresis_segments(smoothed, tau_in=3.5, tau_out=2.0)[0]
    b = hysteresis_segments(smoothed, tau_in=3.5, tau_out=2.0, extend_score=raw)[0]
    assert a.peak == pytest.approx(b.peak)


def test_extend_on_changes_the_region_extent_end_to_end(cfg):
    values = [0.2, 6.0, 5.5, 6.2, -1.8, 2.2, -0.3, 0.1]
    result = word_z(values)
    signal = rushing_signal(result)

    spec_literal = build_regions(
        [signal],
        {"word": result},
        config=cfg,
        thresholds=RegionThresholds(tau_in=3.5, tau_out=2.0, extend_on="smoothed"),
    )
    on_raw = build_regions(
        [signal],
        {"word": result},
        config=cfg,
        thresholds=RegionThresholds(tau_in=3.5, tau_out=2.0, extend_on="raw"),
    )
    assert spec_literal[0].word_idx == [1, 4]
    assert on_raw[0].word_idx == [1, 3]
    assert on_raw[0].t_end < spec_literal[0].t_end
