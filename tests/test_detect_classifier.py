"""Tests for the flaw-type classifier and the severity calibrator (spec 10.6)."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.detect.classifier import (
    DEFAULT_MIN_IOU,
    GAP_LABEL_COLUMNS,
    PHRASE_POSITION_COLUMNS,
    SCALAR_COLUMNS,
    UNCERTAIN_TAG,
    ClassifierCase,
    ClassifierError,
    FlawClassifier,
    Prediction,
    RegionFeatureSpec,
    SeverityCalibrator,
    TrainingExample,
    TruthFlaw,
    apply_label_policy,
    build_examples,
    evaluate,
    evidence_scalar,
    fit_fold,
    fit_severity,
    iou,
    load_model,
    match_to_truth,
    model_path,
    region_vector,
    save_model,
    truth_flaws_from_label_json,
)
from rostrum.detect.zscore import ZResult
from rostrum.types import Region


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# Fixtures: a tiny three-unit comparison
# --------------------------------------------------------------------------- #


def _index(n: int) -> pd.Index:
    return pd.Index(range(n), name="unit_idx", dtype="int64")


def word_result(z_by_feature: dict[str, list[float]], n: int = 6) -> ZResult:
    """A word comparison with explicit z values and a 0.5 s word grid."""
    index = _index(n)
    z = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in z_by_feature.items()}, index=index)
    table = pd.DataFrame(
        {
            "text": [f"w{i}" for i in range(n)],
            "t_start": [i * 0.5 for i in range(n)],
            "t_end": [i * 0.5 + 0.4 for i in range(n)],
        },
        index=index,
    )
    return ZResult(
        text_id="T",
        fold="f",
        unit="word",
        z=z,
        value=z + 5.0,
        median=pd.DataFrame({k: np.full(n, 5.0) for k in z.columns}, index=index),
        sigma=pd.DataFrame({k: np.full(n, 1.0) for k in z.columns}, index=index),
        n_references=pd.Series(np.full(n, 6), index=index, dtype="int64"),
        table=table,
    )


def gap_result(durations_z: list[float], labels: list[str]) -> ZResult:
    """A gap comparison; gap ``i`` follows word ``i``."""
    n = len(durations_z)
    index = _index(n)
    z = pd.DataFrame({"gap_duration_s": np.asarray(durations_z, dtype=float)}, index=index)
    table = pd.DataFrame(
        {
            "t_start": [i * 0.5 + 0.4 for i in range(n)],
            "t_end": [i * 0.5 + 0.5 for i in range(n)],
            "gap_label": labels,
            "gap_prev_word_idx": list(range(n)),
            "gap_next_word_idx": list(range(1, n + 1)),
        },
        index=index,
    )
    return ZResult(
        text_id="T",
        fold="f",
        unit="gap",
        z=z,
        value=z + 0.2,
        median=pd.DataFrame({"gap_duration_s": np.full(n, 0.2)}, index=index),
        sigma=pd.DataFrame({"gap_duration_s": np.full(n, 0.05)}, index=index),
        n_references=pd.Series(np.full(n, 6), index=index, dtype="int64"),
        table=table,
    )


def phrase_result(n_words: int = 6) -> ZResult:
    """One phrase covering every word."""
    index = _index(1)
    z = pd.DataFrame({"phrase_f0_range_st": [-2.0]}, index=index)
    table = pd.DataFrame(
        {
            "t_start": [0.0],
            "t_end": [n_words * 0.5],
            "first_word_idx": [0],
            "last_word_idx": [n_words - 1],
        },
        index=index,
    )
    return ZResult(
        text_id="T",
        fold="f",
        unit="phrase",
        z=z,
        value=z + 6.0,
        median=pd.DataFrame({"phrase_f0_range_st": [6.0]}, index=index),
        sigma=pd.DataFrame({"phrase_f0_range_st": [1.0]}, index=index),
        n_references=pd.Series([6], index=index, dtype="int64"),
        table=table,
    )


@pytest.fixture
def comparison() -> dict[str, ZResult]:
    return {
        "word": word_result(
            {
                "rate_local_syll_per_s": [0.0, 0.0, 3.0, 4.0, 1.0, 0.0],
                "energy_mean_z": [0.0, 0.0, -1.0, -0.5, 0.0, 0.0],
            }
        ),
        "gap": gap_result(
            [-3.0, 0.0, 0.0, 0.0, 0.0], ["boundary", "none", "none", "minor", "none"]
        ),
        "phrase": phrase_result(),
    }


def region(
    label: str = "rushing",
    *,
    region_id: int = 0,
    t_start: float = 1.0,
    t_end: float = 1.9,
    word_idx: list[int] | None = None,
    gap_idx: int | None = None,
    excess: float = 1.5,
    peak: float = 4.0,
    context: str = "",
    category: str = "pace",
) -> Region:
    return Region(
        region_id=region_id,
        t_start=t_start,
        t_end=t_end,
        label=label,  # type: ignore[arg-type]
        category=category,  # type: ignore[arg-type]
        rule_label=label,  # type: ignore[arg-type]
        word_idx=[2, 3] if word_idx is None else word_idx,
        gap_idx=gap_idx,
        mean_excess_z=excess,
        peak_z=peak,
        context=context,
    )


# --------------------------------------------------------------------------- #
# IoU and matching
# --------------------------------------------------------------------------- #


def test_identical_intervals_have_iou_one():
    assert iou(1.0, 2.0, 1.0, 2.0) == pytest.approx(1.0)


def test_half_overlap_has_the_expected_iou():
    # overlap 0.5, union 1.5
    assert iou(0.0, 1.0, 0.5, 1.5) == pytest.approx(1.0 / 3.0)


def test_disjoint_intervals_have_zero_iou():
    assert iou(0.0, 1.0, 1.0, 2.0) == 0.0
    assert iou(0.0, 1.0, 5.0, 6.0) == 0.0


def test_an_empty_interval_is_zero_not_a_division_error():
    assert iou(1.0, 1.0, 1.0, 1.0) == 0.0


def test_a_region_matches_the_truth_it_overlaps():
    regions = [region(region_id=7, t_start=1.0, t_end=2.0)]
    truth = [TruthFlaw("rushing", 1.0, 2.0, s=0.6, level="L3")]
    assert match_to_truth(regions, truth)[7].label == "rushing"


def test_an_overlap_below_the_floor_does_not_match():
    regions = [region(t_start=0.0, t_end=1.0)]
    truth = [TruthFlaw("rushing", 0.8, 3.0)]  # IoU = 0.2/3.0
    assert match_to_truth(regions, truth, min_iou=DEFAULT_MIN_IOU) == {}


def test_matching_is_label_agnostic():
    """The classifier must be able to learn that the rule mistyped a region."""
    regions = [region("rushing", region_id=3, t_start=1.0, t_end=2.0)]
    truth = [TruthFlaw("dragging", 1.0, 2.0)]
    assert match_to_truth(regions, truth)[3].label == "dragging"


def test_each_truth_interval_is_used_at_most_once():
    regions = [
        region(region_id=0, t_start=1.0, t_end=2.0),
        region(region_id=1, t_start=1.05, t_end=1.95),
    ]
    truth = [TruthFlaw("rushing", 1.0, 2.0)]
    matched = match_to_truth(regions, truth)
    assert len(matched) == 1
    assert 0 in matched  # the better IoU wins


def test_matching_is_deterministic_under_an_exact_tie():
    regions = [region(region_id=0, t_start=1.0, t_end=2.0)]
    truth = [TruthFlaw("rushing", 1.0, 2.0), TruthFlaw("dragging", 1.0, 2.0)]
    first = match_to_truth(regions, truth)
    assert first[0].label == "rushing"
    assert match_to_truth(regions, truth)[0].label == first[0].label


def test_truth_flaws_are_read_from_a_label_json():
    flaws = truth_flaws_from_label_json(
        {
            "file_id": "s1_lv07_rushing_L3",
            "flaws": [
                {"label": "rushing", "t_start": 41.212, "t_end": 45.884, "level": "L3", "s": 0.60}
            ],
        }
    )
    assert len(flaws) == 1
    assert flaws[0].file_id == "s1_lv07_rushing_L3"
    assert flaws[0].duration == pytest.approx(4.672)
    assert flaws[0].s == pytest.approx(0.60)


def test_a_label_json_without_flaws_is_a_clean_file_not_an_error():
    assert truth_flaws_from_label_json({"file_id": "clean", "flaws": []}) == []


# --------------------------------------------------------------------------- #
# The design matrix
# --------------------------------------------------------------------------- #


def labelled(spec: RegionFeatureSpec, vector: np.ndarray) -> dict[str, float]:
    """A region vector as ``{column: value}``, so a test names what it asserts."""
    return dict(zip(spec.columns, vector, strict=True))


def test_the_column_layout_covers_the_spec_fields(cfg):
    spec = RegionFeatureSpec.from_config(cfg)
    columns = spec.columns
    assert spec.n_columns == len(columns) == 2 * len(spec.features) + 8
    for column in (*SCALAR_COLUMNS, *GAP_LABEL_COLUMNS, *PHRASE_POSITION_COLUMNS):
        assert column in columns
    assert "mean_z__rate_local" in columns
    assert "max_z__rate_local" in columns


def test_the_column_layout_is_stable_across_calls(cfg):
    assert RegionFeatureSpec.from_config(cfg).columns == RegionFeatureSpec.from_config(cfg).columns


def test_the_layout_includes_channel_sensitive_features(cfg):
    # They are NaN in a cross-channel comparison, but the width must not change
    # with it or a saved model could not be reused.
    assert "mean_z__cpp" in RegionFeatureSpec.from_config(cfg).columns


def test_a_spec_can_be_derived_from_a_comparison(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    assert "rate_local" in spec.features
    assert "gap_duration" in spec.features
    assert "phrase_f0_range_st" in spec.features


def test_region_vector_has_the_spec_width(comparison, cfg):
    spec = RegionFeatureSpec.from_config(cfg)
    assert region_vector(region(), comparison, spec).shape == (spec.n_columns,)


def test_region_vector_reports_the_mean_and_signed_peak_z(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    vector = labelled(spec, region_vector(region(word_idx=[2, 3]), comparison, spec))
    assert vector["mean_z__rate_local"] == pytest.approx(3.5)  # mean of 3.0 and 4.0
    assert vector["max_z__rate_local"] == pytest.approx(4.0)
    # The sign is kept: a dragging region must not look like a rushing one.
    assert vector["max_z__energy_rel_z"] == pytest.approx(-1.0)


def test_region_vector_leaves_an_absent_feature_missing(comparison, cfg):
    spec = RegionFeatureSpec.from_config(cfg)
    vector = labelled(spec, region_vector(region(), comparison, spec))
    assert np.isnan(vector["mean_z__cpp"])  # no channel-sensitive column supplied


def test_region_vector_reports_the_duration_and_word_count(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    vector = labelled(
        spec, region_vector(region(t_start=1.0, t_end=2.5, word_idx=[2, 4]), comparison, spec)
    )
    assert vector["duration_s"] == pytest.approx(1.5)
    assert vector["n_words"] == pytest.approx(3.0)


def test_a_gap_region_one_hots_its_text_expectation(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    vector = labelled(
        spec,
        region_vector(
            region("missing_pause", gap_idx=0, word_idx=[0, 1], category="pausing"),
            comparison,
            spec,
        ),
    )
    assert vector["gap_label__boundary"] == 1.0
    assert vector["gap_label__none"] == 0.0
    assert vector["mean_z__gap_duration"] == pytest.approx(-3.0)


def test_a_word_region_has_no_gap_label_set(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    vector = labelled(spec, region_vector(region(), comparison, spec))
    assert [vector[c] for c in GAP_LABEL_COLUMNS] == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("word_idx", "expected"),
    [([0, 1], "start"), ([2, 3], "mid"), ([5, 5], "end")],
)
def test_the_phrase_position_is_one_hot_encoded(comparison, word_idx, expected):
    spec = RegionFeatureSpec.from_results(comparison)
    vector = labelled(spec, region_vector(region(word_idx=word_idx), comparison, spec))
    assert vector[f"phrase_pos__{expected}"] == 1.0
    assert sum(vector[c] for c in PHRASE_POSITION_COLUMNS) == 1.0


def test_a_region_outside_every_phrase_has_no_position(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    vector = labelled(spec, region_vector(region(word_idx=[40, 41]), comparison, spec))
    assert [vector[c] for c in PHRASE_POSITION_COLUMNS] == [0.0, 0.0, 0.0]


def test_the_evidence_scalar_is_the_mean_excess(comparison):
    assert evidence_scalar(region(excess=2.25)) == pytest.approx(2.25)


def test_the_evidence_scalar_falls_back_to_the_peak():
    assert evidence_scalar(region(excess=float("nan"), peak=-3.5)) == pytest.approx(3.5)


def test_the_evidence_scalar_is_never_negative():
    assert evidence_scalar(region(excess=float("nan"), peak=float("nan"))) == 0.0


# --------------------------------------------------------------------------- #
# Training examples
# --------------------------------------------------------------------------- #


def case(regions, truth, *, text_id="T", origin="injected") -> ClassifierCase:
    return ClassifierCase(
        file_id=f"{text_id}_f",
        text_id=text_id,
        regions=regions,
        z_by_unit={},
        truth=truth,
        origin=origin,
    )


def test_build_examples_keeps_only_matched_regions(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    regions = [
        region(region_id=0, t_start=1.0, t_end=2.0),
        region(region_id=1, t_start=8.0, t_end=9.0),  # a false positive
    ]
    bundle = ClassifierCase("f", "T", regions, comparison, [TruthFlaw("dragging", 1.0, 2.0)])
    examples = build_examples([bundle], spec)
    assert [(e.region_id, e.label, e.rule_label) for e in examples] == [(0, "dragging", "rushing")]


def test_build_examples_carries_the_severity(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    bundle = ClassifierCase(
        "f", "T", [region()], comparison, [TruthFlaw("rushing", 1.0, 1.9, s=0.35, level="L2")]
    )
    example = build_examples([bundle], spec)[0]
    assert (example.s, example.level) == (pytest.approx(0.35), "L2")


def test_build_examples_on_a_clean_file_is_empty(comparison):
    spec = RegionFeatureSpec.from_results(comparison)
    assert build_examples([ClassifierCase("f", "T", [], comparison, [])], spec) == []


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #


def example(label: str, evidence: float, s: float) -> TrainingExample:
    return TrainingExample(
        x=np.zeros(3),
        label=label,
        s=s,
        level="L2",
        evidence=evidence,
        rule_label=label,
        file_id="f",
        text_id="T",
        region_id=0,
    )


def test_the_severity_calibration_is_monotone(cfg):
    ladder = [(0.2, 0.15), (1.0, 0.35), (2.0, 0.60), (4.0, 0.90)]
    calibrator = fit_severity(
        [example("rushing", x, s) for x, s in ladder], config=cfg, min_per_label=99
    )
    predicted = [calibrator.predict(x) for x in (0.0, 0.5, 1.5, 3.0, 10.0)]
    assert all(b >= a for a, b in itertools.pairwise(predicted))


def test_the_severity_calibration_recovers_the_ladder(cfg):
    ladder = [(0.2, 0.15), (1.0, 0.35), (2.0, 0.60), (4.0, 0.90)]
    calibrator = fit_severity(
        [example("rushing", x, s) for x, s in ladder], config=cfg, min_per_label=99
    )
    assert calibrator.level(calibrator.predict(0.2)) == "L1"
    assert calibrator.level(calibrator.predict(4.0)) == "L4"


def test_a_label_with_enough_examples_gets_its_own_calibrator(cfg):
    examples = [example("rushing", x, x / 10.0) for x in (0.5, 1.0, 1.5, 2.0, 2.5)]
    examples += [example("monotone", 1.0, 0.9)]
    calibrator = fit_severity(examples, config=cfg, min_per_label=5)
    assert "rushing" in calibrator.per_label
    assert "monotone" not in calibrator.per_label  # falls back to the global fit


def test_a_single_point_cannot_be_calibrated(cfg):
    calibrator = fit_severity([example("rushing", 1.0, 0.35)], config=cfg)
    assert calibrator.global_model is None
    assert calibrator.predict(5.0) == 0.0  # no fit means no severity claim


def test_the_nearest_rung_wins_and_ties_go_down():
    calibrator = SeverityCalibrator()
    assert calibrator.level(0.0) == "L0"
    assert calibrator.level(0.34) == "L2"
    assert calibrator.level(0.95) == "L4"
    # Exactly halfway between L1 (0.15) and L2 (0.35):
    assert calibrator.level(0.25) == "L1"


def test_the_ladder_comes_from_the_config(cfg):
    calibrator = fit_severity([], config=cfg)
    assert dict(calibrator.levels)["L2"] == pytest.approx(0.35)


def test_the_calibrator_describes_itself(cfg):
    meta = fit_severity([example("rushing", x, x / 10.0) for x in (1.0, 2.0)], config=cfg).to_meta()
    assert meta["fitted"] is True
    assert meta["min_per_label"] == 5


# --------------------------------------------------------------------------- #
# The label policy
# --------------------------------------------------------------------------- #


def prediction(label="rushing", prob=0.9, s=0.35, level="L2") -> Prediction:
    return Prediction(label=label, prob=prob, s=s, level=level)


def test_the_rule_label_survives_a_confident_disagreement():
    target = region("rushing")
    apply_label_policy(target, prediction("dragging", 0.95))
    assert target.label == "rushing"
    assert target.clf_label == "dragging"
    assert target.clf_prob == pytest.approx(0.95)


def test_a_confident_disagreement_is_tagged_uncertain():
    target = region("rushing")
    apply_label_policy(target, prediction("dragging", 0.95))
    assert UNCERTAIN_TAG in target.context


def test_a_hesitant_disagreement_is_not_tagged():
    target = region("rushing")
    apply_label_policy(target, prediction("dragging", 0.55))
    assert UNCERTAIN_TAG not in target.context
    assert target.clf_label == "dragging"  # still recorded


def test_agreement_is_never_tagged():
    target = region("rushing")
    apply_label_policy(target, prediction("rushing", 0.99))
    assert target.context == ""


def test_the_variant_tag_is_preserved_alongside_the_uncertainty_tag():
    target = region("low_energy", context="fading", category="energy")
    apply_label_policy(target, prediction("mumbling", 0.95))
    assert target.context.split(";") == ["fading", UNCERTAIN_TAG]


def test_the_tag_is_not_duplicated_on_a_second_pass():
    target = region("rushing")
    apply_label_policy(target, prediction("dragging", 0.95))
    apply_label_policy(target, prediction("dragging", 0.95))
    assert target.context.count(UNCERTAIN_TAG) == 1


def test_the_policy_sets_the_severity_and_level():
    target = region("rushing")
    apply_label_policy(target, prediction("rushing", 0.9, s=0.6, level="L3"))
    assert (target.s, target.level) == (pytest.approx(0.6), "L3")


# --------------------------------------------------------------------------- #
# Evaluation, with a stand-in model
# --------------------------------------------------------------------------- #


class _AlwaysRushing:
    """A two-class model that always answers ``rushing``."""

    def predict_proba(self, X):
        rows = np.asarray(X).shape[0]
        return np.tile(np.array([0.9, 0.1]), (rows, 1))


def stub_classifier(cfg) -> FlawClassifier:
    return FlawClassifier(
        fold="f",
        spec=RegionFeatureSpec.from_config(cfg),
        classes=("rushing", "dragging"),
        model=_AlwaysRushing(),
        severity=SeverityCalibrator(),
        train_texts=("A", "B"),
        n_train=4,
        config_hash=cfg.hash,
    )


def test_evaluate_scores_a_perfect_model(cfg):
    classifier = stub_classifier(cfg)
    examples = [example("rushing", 1.0, 0.35), example("rushing", 2.0, 0.6)]
    metrics = evaluate(classifier, examples, labels=["rushing", "dragging"])
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["confusion"]["matrix"] == [[2, 0], [0, 0]]


def test_evaluate_reports_the_rules_own_macro_f1(cfg):
    classifier = stub_classifier(cfg)
    wrong_rule = TrainingExample(
        x=np.zeros(3),
        label="rushing",
        s=0.35,
        level="L2",
        evidence=1.0,
        rule_label="dragging",
        file_id="f",
        text_id="T",
        region_id=0,
    )
    metrics = evaluate(classifier, [wrong_rule], labels=["rushing", "dragging"])
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["rule_macro_f1"] == pytest.approx(0.0)


def test_macro_f1_ignores_labels_with_no_support(cfg):
    # `dragging` never appears in the truth, so averaging a zero for it in would
    # punish the model for a question it was never asked.
    metrics = evaluate(stub_classifier(cfg), [example("rushing", 1.0, 0.35)])
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_evaluate_on_no_examples_is_empty_not_an_error(cfg, caplog):
    with caplog.at_level(logging.WARNING):
        metrics = evaluate(stub_classifier(cfg), [])
    assert metrics == {
        "n": 0,
        "macro_f1": 0.0,
        "accuracy": 0.0,
        "per_label": {},
        "confusion": {"labels": [], "matrix": []},
        "rule_macro_f1": 0.0,
    }
    assert "no held-out example" in caplog.text


def test_predict_one_uses_the_calibrated_severity(cfg, comparison):
    ladder = [(0.2, 0.15), (1.0, 0.35), (2.0, 0.60), (4.0, 0.90)]
    classifier = stub_classifier(cfg)
    classifier.severity = fit_severity(
        [example("rushing", x, s) for x, s in ladder], config=cfg, min_per_label=99
    )
    predicted = classifier.predict_one(region(excess=4.0), comparison)
    assert predicted.label == "rushing"
    assert predicted.level == "L4"


def test_annotate_applies_the_policy_to_every_region(cfg, comparison):
    classifier = stub_classifier(cfg)
    regions = [region("dragging", region_id=0), region("rushing", region_id=1)]
    classifier.annotate(regions, comparison)
    assert UNCERTAIN_TAG in regions[0].context  # rule says dragging, model says rushing
    assert regions[1].context == ""
    assert [r.clf_label for r in regions] == ["rushing", "rushing"]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_the_model_path_is_per_fold(cfg):
    assert model_path("loto_s1", cfg).name == "clf_loto_s1.joblib"
    assert model_path("loto_s1", cfg).parent == cfg.path("cache.models")


def test_a_model_round_trips_through_joblib(tmp_path, cfg):
    classifier = stub_classifier(cfg)
    path = save_model(classifier, tmp_path / "clf.joblib")
    loaded = load_model(path, config=cfg)
    assert loaded.fold == classifier.fold
    assert loaded.classes == classifier.classes
    assert loaded.spec.columns == classifier.spec.columns
    assert loaded.config_hash == cfg.hash
    assert loaded.train_texts == ("A", "B")


def test_loading_a_model_from_another_config_warns(tmp_path, cfg, caplog):
    classifier = stub_classifier(cfg)
    classifier.config_hash = "0" * 64
    path = save_model(classifier, tmp_path / "clf.joblib")
    with caplog.at_level(logging.WARNING):
        load_model(path, config=cfg)
    assert "refit before any reported metric" in caplog.text


def test_loading_something_that_is_not_a_model_is_an_error(tmp_path):
    import joblib

    path = tmp_path / "junk.joblib"
    joblib.dump({"hello": "world"}, path)
    with pytest.raises(ClassifierError, match="not a Rostrum classifier payload"):
        load_model(path)


# --------------------------------------------------------------------------- #
# Fitting
#
# HistGradientBoostingClassifier lives in sklearn.ensemble, which imports
# sklearn.tree. On a machine whose Windows Application Control policy blocks
# that extension module the import fails outright - an environment limitation,
# not a behaviour to be papered over, so these tests skip loudly instead.
# --------------------------------------------------------------------------- #


def _requires_ensemble() -> None:
    # pytest.importorskip re-raises an ImportError that is not a
    # ModuleNotFoundError (pytest >= 8.2), and a blocked DLL raises the former,
    # so the guard is written out.
    try:
        import sklearn.ensemble  # noqa: F401
    except ImportError as error:
        pytest.skip(f"sklearn.ensemble is unavailable here: {error}")


def fit_cases(comparison) -> list[ClassifierCase]:
    """Two training texts with two labels each, plus a held-out text."""
    cases: list[ClassifierCase] = []
    for text in ("A", "B", "HELD"):
        cases.append(
            ClassifierCase(
                f"{text}_rush",
                text,
                [region("rushing", region_id=0, t_start=1.0, t_end=2.0)],
                comparison,
                [TruthFlaw("rushing", 1.0, 2.0, s=0.6, level="L3")],
            )
        )
        cases.append(
            ClassifierCase(
                f"{text}_drag",
                text,
                [region("dragging", region_id=0, t_start=1.0, t_end=2.0)],
                comparison,
                [TruthFlaw("dragging", 1.0, 2.0, s=0.35, level="L2")],
            )
        )
    return cases


def test_fit_fold_uses_the_training_texts_only(cfg, comparison):
    _requires_ensemble()
    classifier = fit_fold(
        fit_cases(comparison), fold="loto_held", train_texts=("A", "B"), config=cfg
    )
    assert classifier.train_texts == ("A", "B")
    assert classifier.n_train == 4  # the held-out text contributed nothing
    assert sorted(classifier.classes) == ["dragging", "rushing"]


def test_fit_fold_refuses_an_origin_outside_train_on(cfg, comparison):
    _requires_ensemble()
    human = [
        ClassifierCase(
            "h",
            "A",
            [region("rushing", region_id=0, t_start=1.0, t_end=2.0)],
            comparison,
            [TruthFlaw("rushing", 1.0, 2.0)],
            origin="human",
        )
    ]
    with pytest.raises(ClassifierError, match="nothing to fit"):
        fit_fold(human, fold="f", train_texts=("A",), config=cfg)


def test_fit_fold_refuses_a_single_class(cfg, comparison):
    _requires_ensemble()
    one = [
        ClassifierCase(
            "a",
            "A",
            [region("rushing", region_id=0, t_start=1.0, t_end=2.0)],
            comparison,
            [TruthFlaw("rushing", 1.0, 2.0)],
        )
    ]
    with pytest.raises(ClassifierError, match="needs at least two"):
        fit_fold(one, fold="f", train_texts=("A",), config=cfg)


def test_fit_fold_survives_a_column_no_training_example_supplies(cfg, comparison):
    """A fully missing column is unsplittable, not a fit failure.

    The config vocabulary is wider than any one comparison (channel-sensitive
    columns of a cross-channel fold, for instance), so some columns are NaN in
    every training row. scikit-learn's binner cannot place a threshold in such
    a column, so the fit must neutralise it - and the model must still ignore
    it, which is what feeding two different values for it proves.
    """
    _requires_ensemble()
    spec = RegionFeatureSpec.from_config(cfg)
    classifier = fit_fold(fit_cases(comparison), fold="f", train_texts=("A", "B"), config=cfg)

    x = region_vector(region("rushing"), comparison, spec).reshape(1, -1)
    empty = np.flatnonzero(np.isnan(x[0]))
    assert empty.size, "fixture should leave some config columns unsupplied"

    baseline = classifier.model.predict_proba(x)
    for filler in (0.0, 7.5):
        probed = x.copy()
        probed[0, empty] = filler
        assert classifier.model.predict_proba(probed) == pytest.approx(baseline)


def test_fit_fold_is_deterministic(cfg, comparison):
    _requires_ensemble()
    cases = fit_cases(comparison)
    first = fit_fold(cases, fold="f", train_texts=("A", "B"), config=cfg)
    second = fit_fold(cases, fold="f", train_texts=("A", "B"), config=cfg)
    examples = build_examples(
        [c for c in cases if c.text_id == "HELD"], RegionFeatureSpec.from_config(cfg)
    )
    assert evaluate(first, examples)["macro_f1"] == evaluate(second, examples)["macro_f1"]
