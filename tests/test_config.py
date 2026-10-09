"""Tests for :mod:`rostrum.config` and for the consistency of ``configs/*.yaml``.

The consistency checks are deliberately strict. Rostrum spreads one set of
parameters across five YAML files - a label defined in ``injection.yaml`` has
to be detectable per ``detection.yaml`` and scorable per ``scoring.yaml`` - and
a mismatch between them would not raise anywhere; it would just quietly drop a
flaw type from the rubric.
"""

from __future__ import annotations

import random

import pytest

from rostrum.config import (
    CONFIG_FILES,
    Config,
    FrozenDict,
    code_version,
    config_hash,
    load_config,
    repo_root,
    set_seeds,
)

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


def test_repo_root_contains_the_configs(cfg: Config):
    assert (cfg.root / "configs").is_dir()
    assert (cfg.root / "pyproject.toml").is_file()
    assert repo_root() == cfg.root


def test_every_section_loads(cfg: Config):
    for name in CONFIG_FILES:
        assert isinstance(cfg.section(name), FrozenDict)


def test_unknown_section_raises(cfg: Config):
    with pytest.raises(KeyError):
        cfg.section("nope")


def test_dotted_and_item_access_agree(cfg: Config):
    assert cfg.features.frame.hop_s == cfg.features["frame"]["hop_s"]


def test_frame_grid_matches_the_invariant(cfg: Config):
    # 10 ms hop / 25 ms window / 16 kHz everywhere (CLAUDE.md invariants 1, 3).
    assert cfg.hop_s == 0.010
    assert cfg.window_s == 0.025
    assert cfg.sample_rate == 16000
    assert cfg.features.audio.sample_rate == cfg.sample_rate


def test_config_is_frozen(cfg: Config):
    with pytest.raises(TypeError):
        cfg.features.frame.hop_s = 0.02
    with pytest.raises(TypeError):
        cfg.features["frame"]["hop_s"] = 0.02


def test_lists_are_frozen_to_tuples(cfg: Config):
    assert isinstance(cfg.scoring.category_order, tuple)


def test_path_resolution_is_absolute(cfg: Config):
    p = cfg.path("results.metrics")
    assert p.is_absolute()
    assert p == cfg.root / "results" / "metrics.json"


def test_path_rejects_a_non_path_key(cfg: Config):
    with pytest.raises(TypeError):
        cfg.path("mfa.num_jobs")  # an int, not a path


def test_to_dict_is_plain_and_json_shaped(cfg: Config):
    plain = cfg.to_dict()
    assert set(plain) == set(CONFIG_FILES)
    assert isinstance(plain["scoring"]["category_order"], list)
    plain["scoring"]["category_order"].append("mutable now")  # proves it is a copy
    assert len(cfg.scoring.category_order) == 6


# --------------------------------------------------------------------------- #
# Overrides and hashing
# --------------------------------------------------------------------------- #


def test_overrides_are_deep_merged():
    cfg = load_config(overrides={"detection": {"regions": {"tau_in": 3.0}}})
    assert cfg.detection.regions.tau_in == 3.0
    # Sibling keys survive the merge.
    assert cfg.detection.regions.tau_out == 1.5
    assert cfg.detection.refinement.enabled is True


def test_override_for_unknown_section_raises():
    with pytest.raises(KeyError):
        load_config(overrides={"nonexistent": {"a": 1}})


def test_config_hash_is_stable_and_order_independent():
    a = config_hash({"x": {"b": 1, "a": [1, 2]}})
    b = config_hash({"x": {"a": [1, 2], "b": 1}})
    assert a == b
    assert len(a) == 64


def test_config_hash_changes_with_a_threshold(cfg: Config):
    other = load_config(overrides={"detection": {"regions": {"tau_in": 3.0}}})
    assert other.hash != cfg.hash


def test_seed_override_changes_the_hash(cfg: Config):
    # The recorded seed is folded into the hashed config, so two runs at
    # different seeds cannot be mistaken for the same configuration.
    other = load_config(seed=99)
    assert other.seed == 99
    assert other.injection.seed == 99
    assert other.hash != cfg.hash


def test_code_version_never_raises():
    assert isinstance(code_version(), str)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_set_seeds_makes_random_reproducible():
    set_seeds(1337)
    first = [random.random() for _ in range(5)]
    set_seeds(1337)
    assert [random.random() for _ in range(5)] == first


def test_set_seeds_makes_numpy_reproducible():
    np = pytest.importorskip("numpy")
    set_seeds(1337)
    first = np.random.rand(5).tolist()
    set_seeds(1337)
    assert np.random.rand(5).tolist() == first


# --------------------------------------------------------------------------- #
# Cross-file consistency of the shipped configs
# --------------------------------------------------------------------------- #

EXPECTED_LABELS = {
    "rushing",
    "dragging",
    "missing_pause",
    "hesitation_pause",
    "monotone",
    "misplaced_stress",
    "low_energy",
    "mumbling",
}
EXPECTED_CATEGORIES = {"pace", "pausing", "intonation", "emphasis", "energy", "clarity"}


def test_the_taxonomy_is_eight_labels(cfg: Config):
    assert set(cfg.injection.labels) == EXPECTED_LABELS


def test_the_rubric_is_six_categories(cfg: Config):
    assert set(cfg.scoring.categories) == EXPECTED_CATEGORIES
    assert set(cfg.scoring.category_order) == EXPECTED_CATEGORIES
    assert len(cfg.scoring.category_order) == 6


def test_every_label_is_scored_exactly_once(cfg: Config):
    mapped: list[str] = []
    for spec in cfg.scoring.categories.values():
        mapped.extend(spec["labels"])
    assert sorted(mapped) == sorted(EXPECTED_LABELS), "a label is unscored or double-counted"


def test_every_label_is_detectable(cfg: Config):
    assert set(cfg.detection.rules) == EXPECTED_LABELS


def test_injection_and_scoring_agree_on_each_label_category(cfg: Config):
    label_to_category = {
        label: cat for cat, spec in cfg.scoring.categories.items() for label in spec["labels"]
    }
    for label, spec in cfg.injection.labels.items():
        assert spec["category"] == label_to_category[label], label


def test_every_k_c_exists(cfg: Config):
    assert set(cfg.scoring.k_c) == EXPECTED_CATEGORIES


def test_severity_ladder_is_monotonic(cfg: Config):
    sev = cfg.injection.severity
    assert [sev[k] for k in ("L0", "L1", "L2", "L3", "L4")] == [0.00, 0.15, 0.35, 0.60, 0.90]


def test_mumbling_is_not_synthesisable(cfg: Config):
    # A low-pass filter models a bad microphone, not bad articulation.
    mumbling = cfg.injection.labels.mumbling
    assert mumbling["synthesisable"] is False
    assert mumbling["method"] == "human_recorded_only"


def test_only_mumbling_is_human_only(cfg: Config):
    for label, spec in cfg.injection.labels.items():
        if label == "mumbling":
            continue
        assert spec.get("synthesisable", True) is True, label


def test_hysteresis_thresholds_are_ordered(cfg: Config):
    r = cfg.detection.regions
    assert r.tau_in == 2.5
    assert r.tau_out == 1.5
    assert r.tau_out < r.tau_in, "hysteresis requires an exit threshold below the entry one"


def test_region_timing_parameters_match_the_spec(cfg: Config):
    r = cfg.detection.regions
    assert r.merge_gap_s == 0.200
    assert r.min_duration_s == 0.250
    assert set(r.min_duration_exempt_labels) == {"missing_pause", "hesitation_pause"}
    assert cfg.detection.refinement.window_s == 0.300


def test_scoring_excess_baseline_tracks_tau_out(cfg: Config):
    # M_c sums (|z| - tau_out)_+, so these two numbers must stay in step.
    assert cfg.scoring.score.excess_z_baseline == cfg.detection.regions.tau_out


def test_threshold_grids_respect_the_hysteresis_constraint(cfg: Config):
    search = cfg.detection.threshold_search
    assert search.constraint_tau_out_lt_tau_in is True
    assert min(search.tau_in_grid) > min(search.tau_out_grid)


def test_mad_scale_is_the_consistency_constant(cfg: Config):
    assert cfg.detection.envelope.mad_scale == 1.4826


def test_leakage_guards_are_declared(cfg: Config):
    # Nothing fitted may see the held-out text (CLAUDE.md invariants 10-11).
    assert cfg.detection.envelope.sigma_floor_scope == "training_texts_only"
    assert cfg.detection.threshold_search.fit_scope == "training_folds_only"
    assert cfg.scoring.calibration.fit_scope == "training_texts_only"
    assert tuple(cfg.detection.classifier.train_on) == ("injected",)


def test_channel_rule_partitions_the_features(cfg: Config):
    rule = cfg.features.channel_rule
    robust, sensitive = set(rule.channel_robust), set(rule.channel_sensitive)
    assert robust and sensitive
    assert not robust & sensitive, "a feature cannot be both channel-robust and channel-sensitive"
    assert rule.enforce is True


def test_clarity_is_channel_sensitive(cfg: Config):
    clarity = cfg.scoring.categories.clarity
    assert clarity["channel_sensitive"] is True
    assert cfg.detection.rules.mumbling["channel_sensitive"] is True


def test_pitch_two_pass_range(cfg: Config):
    p = cfg.features.pitch
    assert (p.pass1.floor_hz, p.pass1.ceiling_hz) == (50.0, 700.0)
    assert (p.pass2.q1_scale, p.pass2.q3_scale) == (0.75, 1.50)


def test_f0_is_compared_in_semitones(cfg: Config):
    assert cfg.features.normalisation.f0.transform == "semitones_re_speaker_median"
    assert cfg.features.normalisation.pitch_variation.method == "std_semitones"


def test_statistics_never_use_interpolated_f0(cfg: Config):
    assert cfg.features.pitch.interpolate_unvoiced.for_statistics is False


def test_injected_silence_is_room_tone_not_zeros(cfg: Config):
    assert cfg.injection.room_tone.source == "self"
    assert cfg.injection.splice.crossfade_ms == 10.0


def test_qc_gates_are_armed(cfg: Config):
    qc = cfg.injection.qc
    assert qc.realign.enabled is True
    assert qc.realign.max_boundary_err_ms == 50.0
    assert qc.utmos.max_drop_by_level["L1"] == 0.30


def test_pause_threshold_is_consistent(cfg: Config):
    # The 250 ms silent-pause threshold comes from fluency research; the
    # detector and the feature extractor must use the same number.
    assert cfg.features.pauses.min_pause_s == 0.250
    assert cfg.detection.rules.hesitation_pause["min_absolute_gap_s"] == 0.250


def test_templates_are_placeholders_for_every_label(cfg: Config):
    # Phase 6 fills these; the keys must already cover the taxonomy so a label
    # cannot be forgotten when the strings are written.
    assert set(cfg.templates.rationale) == EXPECTED_LABELS
    assert set(cfg.templates.fix) == EXPECTED_LABELS


def test_targets_are_pre_registered(cfg: Config):
    syn = cfg.scoring.targets.synthetic
    assert syn.event_f1_200ms == 0.75
    assert syn.onset_offset_mae_ms == 150.0
    assert cfg.scoring.targets.human.event_f1_200ms == 0.50


# --------------------------------------------------------------------------- #
# Spec-specific parameter values
#
# These pin the numbers where MASTER_SPEC_PLAN.md corrects PROJECT_PLAN.md. Each
# was wrong in an earlier draft of the configs, so each gets a test.
# --------------------------------------------------------------------------- #


def test_prominence_cwt_weights_are_not_equal(cfg: Config):
    # spec 9.1: F0 1.0, energy 0.5, duration 1.0 - energy is deliberately
    # down-weighted, since intensity alone is a weak prominence cue.
    assert tuple(cfg.features.prominence.component_weights) == (1.0, 0.5, 1.0)
    assert len(cfg.features.prominence.components) == 3


def test_band_energy_ratio_is_two_bands(cfg: Config):
    # spec 9.1: 0-1 kHz over 1-4 kHz.
    bands = [tuple(b) for b in cfg.features.spectral.band_energy_ratios]
    assert bands == [(0, 1000), (1000, 4000)]


def test_feature_units_are_word_gap_phrase(cfg: Config):
    assert tuple(cfg.features.units) == ("word", "gap", "phrase")
    assert cfg.features.phrase_definition == "between_boundary_gaps"


def test_hash_rounding_follows_the_spec(cfg: Config):
    # spec 16: times 3 dp, feature values 4 significant figures.
    assert cfg.features.cache.round_significant_figures == 4
    assert cfg.features.cache.round_time_decimals == 3


def test_missing_pause_needs_a_300ms_boundary_pause(cfg: Config):
    # spec 5.2: the clean file's boundary pause must be at least 300 ms.
    assert cfg.injection.labels.missing_pause.valid_span["min_gap_s"] == 0.30


def test_injection_span_sizes_match_the_spec(cfg: Config):
    labels = cfg.injection.labels
    assert tuple(labels.rushing.span_words) == (3, 10)
    assert tuple(labels.dragging.span_words) == (3, 10)
    assert labels.monotone.valid_span["min_phrase_words"] == 5
    assert labels.monotone.valid_span["max_phrase_words"] == 15
    assert tuple(labels.low_energy.valid_span["span_words"]) == (4, 12)


def test_misplaced_stress_pairs_words_within_one_phrase(cfg: Config):
    span = cfg.injection.labels.misplaced_stress.valid_span
    assert span["same_phrase"] is True
    assert span["content_word_selection"] == "most_prominent_in_phrase"


def test_composite_generation_plan(cfg: Config):
    # spec 5.4: 2 files at L2 and 2 at L4, 3-5 flaws, at least 1 s apart.
    comp = cfg.injection.composite
    assert tuple(comp.levels) == ("L2", "L4")
    assert comp.files_per_level == 2
    assert tuple(comp.n_flaws) == (3, 5)
    assert comp.min_separation_s == 1.00


def test_realign_target_and_rejection_are_different_numbers(cfg: Config):
    # spec 5.1 sets a 20 ms agreement target; spec 5.4 rejects above 50 ms.
    realign = cfg.injection.qc.realign
    assert realign.target_boundary_err_ms == 20.0
    assert realign.max_boundary_err_ms == 50.0
    assert realign.target_boundary_err_ms < realign.max_boundary_err_ms
    assert realign.log_rejections is True


def test_monotone_requires_both_range_and_std(cfg: Config):
    # spec 10.3: phrase pitch-range z < -tau AND phrase F0-std z < -tau.
    rule = cfg.detection.rules.monotone
    assert rule["feature"] == "f0_range_st"
    assert rule["direction"] == "below"
    assert rule["also_require"]["feature"] == "f0_std_st"
    assert rule["also_require"]["direction"] == "below"


def test_misplaced_stress_uses_rank_correlation(cfg: Config):
    # spec 10.3: a within-phrase prominence-rank correlation, not a single z.
    rule = cfg.detection.rules.misplaced_stress
    assert rule["method"] == "prominence_rank_correlation"
    assert rule["unit"] == "phrase"
    assert 0.0 < rule["rho_min"] <= 1.0
    assert rule["localise_to_largest_delta"] is True


def test_mumbling_is_a_same_channel_composite(cfg: Config):
    # spec 10.3: rate down, intensity down, vowel duration down, CPP down,
    # spectral tilt steeper - not spectral tilt on its own.
    rule = cfg.detection.rules.mumbling
    assert rule["method"] == "clarity_composite"
    assert rule["channel_sensitive"] is True
    assert set(rule["composite_features"]) == {
        "rate_local",
        "energy_rel_z",
        "vowel_duration_norm",
        "cpp",
        "spectral_tilt",
    }
    assert len(rule["composite_weights"]) == len(rule["composite_features"])


def test_fading_is_a_variant_of_low_energy_not_a_ninth_label(cfg: Config):
    # spec 11 writes the Energy category as "low_energy, fading", but the
    # taxonomy is eight labels throughout, so fading is a detected variant.
    variant = cfg.detection.rules.low_energy["variant_fading"]
    assert variant["feature"] == "energy_slope_db_s"
    assert variant["direction"] == "below"
    assert tuple(cfg.scoring.categories.energy["variants"]) == ("fading",)
    assert "fading" not in cfg.injection.labels


def test_pause_flaw_regions_take_gap_edges(cfg: Config):
    # spec 10.4(4): word-based flaws use word edges, pause flaws use gap edges.
    assert cfg.detection.regions.boundary_from == "word_edges"
    assert cfg.detection.regions.boundary_from_gap_labels == "gap_edges"


def test_classifier_matches_the_spec(cfg: Config):
    clf = cfg.detection.classifier
    assert clf.random_state == 0  # spec 10.6
    assert "mean_z_per_feature" in clf.features
    assert "max_z_per_feature" in clf.features
    assert "gap_label_one_hot" in clf.features
    # spec 10.6: the explainable rule wins; the classifier only adds a number.
    assert clf.on_disagreement == "show_rule_label_with_clf_prob"


def test_reference_mode_needs_four_readers_and_a_text_match(cfg: Config):
    # spec 13: R >= 4 and a normalised string match of at least 0.95.
    ref = cfg.detection.modes.reference
    assert ref.min_references == 4
    assert ref.min_transcript_match == 0.95


def test_norm_mode_never_fires_channel_sensitive_flaws(cfg: Config):
    # spec 9.3/13: without a same-channel reference these are informational.
    assert cfg.detection.modes.norm.channel_sensitive_informational_only is True


def test_k_c_is_calibrated_per_fold(cfg: Config):
    assert cfg.scoring.calibration.per_fold is True  # spec 11


def test_confidence_band_reports_mean_and_std(cfg: Config):
    assert cfg.scoring.confidence.statistic == "mean_and_std"  # spec 11


def test_dataset_layout_matches_spec_7_4(cfg: Config):
    assert cfg.paths.data["splits"].endswith("loto.json")
    for key in ("manifest", "consent", "annotation_guide", "provenance"):
        assert key in cfg.paths.data, key
    for key in ("qc_utmos", "qc_realign", "qc_rejections"):
        assert cfg.paths.data[key].startswith("dataset/qc/"), key
