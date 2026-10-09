"""Tests for the rubric, its calibration and its confidence band (spec 11)."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.detect.zscore import SkippedSignal
from rostrum.score.rubric import (
    DEFAULT_TARGET_SCORE,
    CalibrationCase,
    build_descriptors,
    calibrate_k,
    category_order,
    category_score,
    confidence_band,
    descriptors,
    deviation_mass,
    k_from_config,
    label_to_category,
    rubric_score,
    suppressed_from_skipped,
)
from rostrum.types import CategoryScore, Region, RubricScore


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def region(
    label: str = "rushing",
    *,
    category: str = "pace",
    t_start: float = 1.0,
    t_end: float = 2.0,
    excess: float = 1.0,
    peak: float = 3.0,
    region_id: int = 0,
) -> Region:
    return Region(
        region_id=region_id,
        t_start=t_start,
        t_end=t_end,
        label=label,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        rule_label=label,  # type: ignore[arg-type]
        mean_excess_z=excess,
        peak_z=peak,
    )


# --------------------------------------------------------------------------- #
# Category bookkeeping
# --------------------------------------------------------------------------- #


def test_the_six_categories_come_from_the_config(cfg):
    assert category_order(cfg) == (
        "pace",
        "pausing",
        "intonation",
        "emphasis",
        "energy",
        "clarity",
    )


def test_every_flaw_label_maps_to_exactly_one_category(cfg):
    mapping = label_to_category(cfg)
    assert mapping["rushing"] == "pace"
    assert mapping["mumbling"] == "clarity"
    assert len(mapping) == 8  # the eight labels of the taxonomy


def test_the_fallback_map_agrees_with_the_config(cfg):
    assert label_to_category(None) == label_to_category(cfg)


def test_the_starting_k_comes_from_the_config(cfg):
    assert set(k_from_config(cfg)) == set(category_order(cfg))


# --------------------------------------------------------------------------- #
# Deviation mass and the category score
# --------------------------------------------------------------------------- #


def test_no_regions_means_no_deviation_mass():
    assert deviation_mass([], 10.0) == 0.0


def test_the_mass_is_duration_times_excess_over_speech_time():
    # 1 s long, mean excess 2.0, over 10 s of speech
    mass = deviation_mass([region(excess=2.0, t_start=0.0, t_end=1.0)], 10.0)
    assert mass == pytest.approx(0.2)


def test_the_mass_only_counts_the_named_category():
    regions = [region("rushing", category="pace"), region("monotone", category="intonation")]
    assert deviation_mass(regions, 10.0, category="pace") == pytest.approx(0.1)


def test_a_longer_flaw_weighs_more_than_a_shorter_one():
    short = deviation_mass([region(t_start=0.0, t_end=0.5)], 10.0)
    long = deviation_mass([region(t_start=0.0, t_end=2.0)], 10.0)
    assert long > short


def test_splitting_one_flaw_into_two_does_not_change_the_mass():
    """The reason the mass integrates duration rather than counting regions."""
    whole = deviation_mass([region(t_start=0.0, t_end=2.0, excess=1.5)], 10.0)
    halves = deviation_mass(
        [
            region(t_start=0.0, t_end=1.0, excess=1.5, region_id=0),
            region(t_start=1.0, t_end=2.0, excess=1.5, region_id=1),
        ],
        10.0,
    )
    assert whole == pytest.approx(halves)


def test_a_nan_excess_falls_back_to_the_peak():
    mass = deviation_mass([region(excess=float("nan"), peak=3.5, t_start=0.0, t_end=1.0)], 10.0)
    assert mass == pytest.approx((3.5 - 1.5) / 10.0)


def test_a_non_positive_speech_duration_is_an_error():
    with pytest.raises(ValueError, match="must be positive"):
        deviation_mass([], 0.0)


def test_a_rebased_excess_is_recorded_not_applied(caplog):
    with caplog.at_level(logging.WARNING):
        deviation_mass([region()], 10.0, excess_baseline=2.5)
    assert "already measured" in caplog.text


def test_no_deviation_scores_a_perfect_ten():
    assert category_score(0.0, 1.0) == pytest.approx(10.0)


def test_the_score_decreases_monotonically_with_the_mass():
    scores = [category_score(mass, 1.0) for mass in (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)]
    assert all(b < a for a, b in itertools.pairwise(scores))


def test_the_score_never_leaves_the_zero_to_ten_range():
    assert 0.0 <= category_score(1000.0, 5.0) <= 10.0


def test_a_larger_k_punishes_the_same_mass_harder():
    assert category_score(0.5, 2.0) < category_score(0.5, 1.0)


def test_a_negative_mass_or_non_positive_k_is_an_error():
    with pytest.raises(ValueError, match="non-negative"):
        category_score(-0.1, 1.0)
    with pytest.raises(ValueError, match="k must be positive"):
        category_score(0.1, 0.0)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def calibration_case(mass: float, *, category="pace", level="L2", text_id="A", n_flaws=1):
    return CalibrationCase(
        file_id=f"{text_id}_{level}_{mass}",
        text_id=text_id,
        category=category,
        level=level,
        mass=mass,
        n_flaws=n_flaws,
    )


def test_calibration_puts_the_median_l2_file_at_six(cfg):
    cases = [calibration_case(m) for m in (0.4, 0.5, 0.6)]
    k = calibrate_k(cases, train_texts=("A",), config=cfg)
    # The median mass is 0.5, so that file must score exactly the target.
    assert category_score(0.5, k["pace"]) == pytest.approx(DEFAULT_TARGET_SCORE)


def test_calibration_is_the_closed_form(cfg):
    k = calibrate_k([calibration_case(0.5)], train_texts=("A",), config=cfg)
    assert k["pace"] == pytest.approx(np.log(10.0 / 6.0) / 0.5)


def test_calibration_ignores_other_levels_and_composite_files(cfg):
    cases = [
        calibration_case(0.5),
        calibration_case(5.0, level="L4"),
        calibration_case(5.0, n_flaws=3),
    ]
    k = calibrate_k(cases, train_texts=("A",), config=cfg)
    assert k["pace"] == pytest.approx(np.log(10.0 / 6.0) / 0.5)


def test_calibration_never_sees_the_held_out_text(cfg):
    """The leakage guard of CLAUDE.md invariant 11, enforced by the selection."""
    cases = [calibration_case(0.5, text_id="A"), calibration_case(10.0, text_id="HELD")]
    k = calibrate_k(cases, train_texts=("A",), config=cfg)
    assert k["pace"] == pytest.approx(np.log(10.0 / 6.0) / 0.5)


def test_calibration_ignores_human_takes(cfg):
    human = CalibrationCase("h", "A", "pace", "L2", 10.0, n_flaws=1, origin="human")
    k = calibrate_k([calibration_case(0.5), human], train_texts=("A",), config=cfg)
    assert k["pace"] == pytest.approx(np.log(10.0 / 6.0) / 0.5)


def test_a_category_with_no_case_keeps_the_configured_k(cfg, caplog):
    with caplog.at_level(logging.WARNING):
        k = calibrate_k([calibration_case(0.5)], train_texts=("A",), config=cfg)
    assert k["clarity"] == pytest.approx(float(cfg.scoring["k_c"]["clarity"]))
    assert "keeping k" in caplog.text


def test_a_tiny_median_mass_is_clamped(cfg, caplog):
    with caplog.at_level(logging.WARNING):
        k = calibrate_k([calibration_case(1e-9)], train_texts=("A",), config=cfg)
    assert k["pace"] == pytest.approx(float(cfg.scoring["calibration"]["k_bounds"][1]))
    assert "clamped" in caplog.text


def test_an_unreachable_target_score_is_an_error(cfg):
    with pytest.raises(ValueError, match="strictly between"):
        calibrate_k([], target_score=10.0, max_score=10.0)


# --------------------------------------------------------------------------- #
# Suppression
# --------------------------------------------------------------------------- #


def test_clarity_is_suppressed_when_mumbling_cannot_be_evaluated(cfg):
    skipped = [
        SkippedSignal(
            label="mumbling",  # type: ignore[arg-type]
            reason="no_same_channel_reference",
            detail="no same-channel reference for this speaker",
        )
    ]
    assert "clarity" in suppressed_from_skipped(skipped, config=cfg)


def test_a_category_with_one_label_still_evaluated_is_not_suppressed(cfg):
    # pace has two labels; skipping only `rushing` leaves `dragging` live.
    skipped = [SkippedSignal(label="rushing", reason="no_feature")]  # type: ignore[arg-type]
    assert suppressed_from_skipped(skipped, config=cfg) == {}


def test_nothing_skipped_suppresses_nothing(cfg):
    assert suppressed_from_skipped([], config=cfg) == {}


# --------------------------------------------------------------------------- #
# The rubric score
# --------------------------------------------------------------------------- #


def test_a_clean_reading_scores_ten_everywhere(cfg):
    score = rubric_score([], 30.0, config=cfg)
    assert score.overall == pytest.approx(10.0)
    assert [c.score for c in score.categories] == [pytest.approx(10.0)] * 6
    assert all(c.n_regions == 0 for c in score.categories)


def test_the_overall_is_the_unweighted_mean_of_six(cfg):
    score = rubric_score([region(excess=2.0)], 10.0, k_by_category={"pace": 1.0}, config=cfg)
    pace = score.get("pace")
    assert pace is not None
    assert score.overall == pytest.approx((pace.score + 5 * 10.0) / 6.0)


def test_only_the_flawed_category_loses_points(cfg):
    score = rubric_score([region("monotone", category="intonation")], 10.0, config=cfg)
    assert score.get("intonation").score < 10.0
    assert score.get("pace").score == pytest.approx(10.0)


def test_the_score_decreases_as_the_deviation_grows(cfg):
    scores = [
        rubric_score([region(excess=e)], 10.0, k_by_category={"pace": 1.0}, config=cfg)
        .get("pace")
        .score
        for e in (0.5, 1.0, 2.0, 4.0)
    ]
    assert all(b < a for a, b in itertools.pairwise(scores))


def test_a_suppressed_category_is_excluded_from_the_mean_not_scored_ten(cfg):
    score = rubric_score([], 10.0, config=cfg, suppressed={"clarity": "no same-channel reference"})
    clarity = score.get("clarity")
    assert clarity.suppressed is True
    assert clarity.suppressed_reason == "no same-channel reference"
    # The mean is over five categories; it would be identical here either way,
    # so the n_regions/suppressed flags are what the UI must read.
    assert score.overall == pytest.approx(10.0)
    assert sum(1 for c in score.categories if not c.suppressed) == 5


def test_a_fully_suppressed_analysis_warns(cfg, caplog):
    everything = dict.fromkeys(category_order(cfg), "unavailable")
    with caplog.at_level(logging.WARNING):
        score = rubric_score([], 10.0, config=cfg, suppressed=everything)
    assert score.overall == 0.0
    assert "every rubric category was suppressed" in caplog.text


def test_the_calibrated_k_is_recorded_on_every_category(cfg):
    score = rubric_score([], 10.0, k_by_category={"pace": 2.5}, config=cfg)
    assert score.get("pace").k == pytest.approx(2.5)


def test_the_region_count_is_reported_per_category(cfg):
    regions = [region(region_id=0), region(region_id=1, t_start=5.0, t_end=6.0)]
    assert rubric_score(regions, 10.0, config=cfg).get("pace").n_regions == 2


def test_the_score_round_trips_through_json(cfg):
    score = rubric_score([region()], 10.0, config=cfg)
    # `to_dict` rounds to VALUE_SIGFIGS on the way out (types.py), so the
    # round-trip is exact only to that precision - which is the point: the
    # determinism hash must not depend on the last bits of a float.
    restored = RubricScore.from_dict(score.to_dict())
    assert restored.overall == pytest.approx(score.overall, rel=1e-3)
    assert restored.get("pace").deviation_mass == pytest.approx(
        score.get("pace").deviation_mass, rel=1e-3
    )


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #


def tables(
    *, n_words: int = 8, syllables: int = 2, long_gap_at: int | None = 3
) -> dict[str, pd.DataFrame]:
    """A simple reading: 0.4 s words, 0.1 s gaps, one optional long pause."""
    index = pd.Index(range(n_words), name="unit_idx", dtype="int64")
    starts = [i * 0.5 for i in range(n_words)]
    words = pd.DataFrame(
        {
            "t_start": starts,
            "t_end": [s + 0.4 for s in starts],
            "word_duration_s": [0.4] * n_words,
            "n_syllables": [syllables] * n_words,
            "rate_local_syll_per_s": [syllables / 0.4] * n_words,
            "f0_mean_st": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0][:n_words],
        },
        index=index,
    )
    gap_index = pd.Index(range(n_words - 1), name="unit_idx", dtype="int64")
    durations = [0.1] * (n_words - 1)
    labels = ["none"] * (n_words - 1)
    if long_gap_at is not None:
        durations[long_gap_at] = 0.6
        labels[long_gap_at] = "boundary"
    gaps = pd.DataFrame(
        {
            "t_start": [i * 0.5 + 0.4 for i in range(n_words - 1)],
            "t_end": [i * 0.5 + 0.4 + d for i, d in enumerate(durations)],
            "gap_duration_s": durations,
            "gap_label": labels,
            "gap_prev_word_idx": list(range(n_words - 1)),
            "gap_next_word_idx": list(range(1, n_words)),
        },
        index=gap_index,
    )
    return {"word": words, "gap": gaps}


def test_the_articulation_rate_excludes_pauses():
    block = tables()
    # 8 words x 2 syllables over 8 x 0.4 s of phonation = 5 syll/s
    assert descriptors(block["word"], block["gap"])["articulation_rate_syll_s"] == pytest.approx(
        5.0
    )


def test_the_articulation_rate_falls_back_to_the_per_word_rate():
    block = tables()
    words = block["word"].drop(columns=["n_syllables"])
    assert descriptors(words, block["gap"])["articulation_rate_syll_s"] == pytest.approx(5.0)


def test_the_pitch_variation_is_the_spread_of_word_f0():
    block = tables()
    expected = float(np.std(block["word"]["f0_mean_st"].to_numpy(), ddof=1))
    assert descriptors(block["word"], block["gap"])["pitch_variation_st"] == pytest.approx(expected)


def test_a_run_ends_at_a_pause_over_the_threshold():
    block = tables(long_gap_at=3)
    # 8 words of 2 syllables, split after word 3: runs of 8 and 8 syllables
    assert descriptors(block["word"], block["gap"])["mean_length_of_runs_syll"] == pytest.approx(
        8.0
    )


def test_without_a_long_pause_the_whole_utterance_is_one_run():
    block = tables(long_gap_at=None)
    assert descriptors(block["word"], block["gap"])["mean_length_of_runs_syll"] == pytest.approx(
        16.0
    )


def test_the_pause_rate_splits_boundary_from_mid_clause():
    block = tables(long_gap_at=3)
    rate = descriptors(block["word"], block["gap"])["pause_rate_per_min"]
    assert rate["all"] == pytest.approx(rate["boundary"])
    assert rate["mid_clause"] == pytest.approx(0.0)


def test_a_mid_clause_pause_is_counted_as_such():
    block = tables(long_gap_at=3)
    block["gap"].loc[3, "gap_label"] = "none"
    rate = descriptors(block["word"], block["gap"])["pause_rate_per_min"]
    assert rate["boundary"] == pytest.approx(0.0)
    assert rate["mid_clause"] > 0.0


def test_descriptors_without_a_gap_table_are_none_not_zero():
    block = tables()
    out = descriptors(block["word"], None)
    assert out["mean_length_of_runs_syll"] is None
    assert out["pause_rate_per_min"]["all"] is None


def test_build_descriptors_adds_the_readers_range():
    mine = tables()
    mine["word"]["n_syllables"] = 3  # faster than the references
    block = build_descriptors(mine, reference_tables=[tables(), tables(syllables=2)])
    rate = block["articulation_rate_syll_s"]
    assert rate["value"] == pytest.approx(7.5)
    assert rate["reference_low"] == pytest.approx(5.0)
    assert rate["reference_high"] == pytest.approx(5.0)
    assert rate["n_references"] == 2


def test_build_descriptors_without_references_omits_the_range():
    block = build_descriptors(tables())
    assert "reference_low" not in block["articulation_rate_syll_s"]
    assert block["articulation_rate_syll_s"]["value"] == pytest.approx(5.0)


def test_build_descriptors_keeps_the_pause_split():
    block = build_descriptors(tables(), reference_tables=[tables()])
    assert "boundary" in block["pause_rate_per_min"]
    assert "mid_clause" in block["pause_rate_per_min"]


# --------------------------------------------------------------------------- #
# Confidence band
# --------------------------------------------------------------------------- #


def score_with(pace: float, overall: float | None = None) -> RubricScore:
    categories = [
        CategoryScore(category=c, score=pace if c == "pace" else 10.0)  # type: ignore[arg-type]
        for c in category_order(None)
    ]
    return RubricScore(
        categories=categories,
        overall=overall if overall is not None else float(np.mean([c.score for c in categories])),
    )


def test_the_band_is_the_spread_across_recomputations():
    base = score_with(6.0)
    variants = [score_with(5.0), score_with(6.0), score_with(7.0)]
    banded = confidence_band(base, variants)
    pace = banded.get("pace")
    assert pace.score == pytest.approx(6.0)  # the mean of the recomputations
    assert pace.band == pytest.approx(np.std([5.0, 6.0, 7.0], ddof=0))


def test_agreeing_references_give_a_zero_band():
    banded = confidence_band(score_with(6.0), [score_with(6.0)] * 4)
    assert banded.get("pace").band == pytest.approx(0.0)


def test_the_overall_gets_its_own_band():
    banded = confidence_band(score_with(6.0), [score_with(5.0), score_with(7.0)])
    assert banded.overall_band > 0.0


def test_the_full_envelope_score_can_be_kept():
    banded = confidence_band(
        score_with(6.0), [score_with(2.0), score_with(4.0)], replace_score=False
    )
    assert banded.get("pace").score == pytest.approx(6.0)
    assert banded.get("pace").band > 0.0


def test_too_few_recomputations_report_no_band(caplog):
    with caplog.at_level(logging.INFO):
        banded = confidence_band(score_with(6.0), [score_with(2.0)])
    assert banded.get("pace").score == pytest.approx(6.0)
    assert banded.overall_band == 0.0
    assert "too few for a band" in caplog.text


def test_the_band_does_not_mutate_the_base_score():
    base = score_with(6.0)
    confidence_band(base, [score_with(1.0), score_with(2.0)])
    assert base.get("pace").score == pytest.approx(6.0)
    assert base.get("pace").band == pytest.approx(0.0)


def test_the_descriptors_survive_the_band():
    base = RubricScore(categories=list(score_with(6.0).categories), descriptors={"a": 1})
    assert confidence_band(base, [score_with(5.0), score_with(7.0)]).descriptors == {"a": 1}
