"""Tests for the canonical-name registry (spec 9.1, 9.3)."""

from __future__ import annotations

import pytest

from rostrum.config import load_config
from rostrum.features import names as N
from rostrum.features.duration import DURATION_WORD_COLUMNS
from rostrum.features.energy import ENERGY_WORD_COLUMNS
from rostrum.features.pauses import PAUSE_GAP_COLUMNS
from rostrum.features.pitch import PITCH_WORD_COLUMNS
from rostrum.features.prominence import PROMINENCE_WORD_COLUMNS
from rostrum.features.rate import RATE_WORD_COLUMNS
from rostrum.features.spectral import SPECTRAL_WORD_COLUMNS
from rostrum.features.voice_quality import VQ_WORD_COLUMNS


@pytest.fixture(scope="module")
def cfg():
    return load_config()


ALL_COLUMNS = (
    *PITCH_WORD_COLUMNS,
    *ENERGY_WORD_COLUMNS,
    *DURATION_WORD_COLUMNS,
    *RATE_WORD_COLUMNS,
    *PAUSE_GAP_COLUMNS,
    *PROMINENCE_WORD_COLUMNS,
    *SPECTRAL_WORD_COLUMNS,
    *VQ_WORD_COLUMNS,
)


def test_every_alias_target_is_a_real_feature_column():
    """The registry must not drift away from what the extractors emit."""
    known = set(ALL_COLUMNS) | {
        "phrase_f0_range_st",
        "phrase_f0_max_st",
        "phrase_rate_syll_per_s",
        "phrase_duration_s",
    }
    for name, targets in N.ALIASES.items():
        for target in targets:
            if target == N.MFCC_PREFIX:
                continue
            assert target in known, f"alias {name!r} points at unknown column {target!r}"


def test_every_config_feature_name_resolves(cfg):
    """Nothing in the configs may name a feature the registry cannot resolve."""
    referenced: set[str] = set()
    rule = cfg.features["channel_rule"]
    referenced |= {str(x) for x in rule["channel_robust"]}
    referenced |= {str(x) for x in rule["channel_sensitive"]}
    referenced |= {str(k) for k in cfg.detection["deviation"]["lambda"]}
    for block in cfg.scoring["categories"].values():
        referenced |= {str(x) for x in block.get("primary_features", ())}
    envelope = cfg.detection["envelope"]
    for per_unit in (envelope.get("features", {}), envelope.get("channel_sensitive_features", {})):
        for features in dict(per_unit).values():
            referenced |= {str(x) for x in features}

    unresolvable = sorted(n for n in referenced if n not in N.ALIASES)
    assert unresolvable == [], f"config names features the registry does not know: {unresolvable}"


def test_resolve_maps_a_canonical_name_to_its_column():
    assert N.resolve("rate_local") == ["rate_local_syll_per_s"]
    assert N.resolve("gap_duration") == ["gap_duration_s"]
    assert N.resolve("energy_rel_z") == ["energy_mean_z"]
    assert N.resolve("cpp") == ["cpps_db"]
    assert N.resolve("spectral_tilt") == ["spectral_tilt_db_per_khz"]


def test_resolve_drops_columns_a_table_does_not_have():
    assert N.resolve("rate_local", ["f0_mean_st"]) == []
    assert N.resolve("rate_local", ["rate_local_syll_per_s"]) == ["rate_local_syll_per_s"]


def test_resolve_accepts_a_column_name_directly():
    assert N.resolve("f0_mean_st", ["f0_mean_st"]) == ["f0_mean_st"]


def test_an_unknown_name_raises():
    with pytest.raises(KeyError, match="unknown feature name"):
        N.resolve("not_a_feature")
    with pytest.raises(KeyError, match="unknown feature name"):
        N.resolve("not_a_feature", ["f0_mean_st"])


def test_mfcc_expands_to_the_whole_cepstral_block():
    columns = ["mfcc_00", "mfcc_01", "mfcc_d_00", "mfcc_n_frames", "f0_mean_st"]
    assert N.resolve("mfcc", columns) == ["mfcc_00", "mfcc_01", "mfcc_d_00"]
    # The frame count is bookkeeping, not a cepstral coefficient.
    assert "mfcc_n_frames" not in N.resolve("mfcc", columns)


def test_mfcc_without_columns_is_an_error():
    with pytest.raises(KeyError, match="needs the table's columns"):
        N.resolve("mfcc")


def test_resolve_many_deduplicates_and_keeps_order():
    columns = ["rate_local_syll_per_s", "f0_mean_st", "energy_mean_z"]
    out = N.resolve_many(["energy_rel_z", "rate_local", "f0_semitones", "f0_mean_st"], columns)
    assert out == ["energy_mean_z", "rate_local_syll_per_s", "f0_mean_st"]


def test_resolve_many_skips_absent_features_unless_strict():
    assert N.resolve_many(["rate_local", "cpp"], ["rate_local_syll_per_s"]) == [
        "rate_local_syll_per_s"
    ]
    with pytest.raises(KeyError):
        N.resolve_many(["rate_local", "cpp"], ["rate_local_syll_per_s"], strict=True)


def test_canonical_is_the_inverse_and_never_raises():
    assert N.canonical("rate_local_syll_per_s") == "rate_local"
    assert N.canonical("cpps_db") == "cpp"
    assert N.canonical("mfcc_07") == "mfcc"
    # Unregistered columns pass through: a card must not lose a measurement.
    assert N.canonical("something_new") == "something_new"


def test_display_units_are_present_for_every_alias():
    for name in N.ALIASES:
        assert name in N.DISPLAY_UNITS, f"{name!r} has no display unit"
    assert N.display_unit("rate_local_syll_per_s") == "syll/s"
    assert N.display_unit("gap_duration_s") == "s"
    assert N.display_unit("f0_range_st") == "st"


def test_the_channel_rule_comes_from_the_config(cfg):
    assert N.is_channel_sensitive("cpps_db", cfg)
    assert N.is_channel_sensitive("spectral_tilt_db_per_khz", cfg)
    assert N.is_channel_sensitive("mfcc_03", cfg)
    assert not N.is_channel_sensitive("rate_local_syll_per_s", cfg)
    assert not N.is_channel_sensitive("gap_duration_s", cfg)


def test_channel_robust_and_sensitive_sets_are_disjoint(cfg):
    robust = set(N.channel_robust_names(cfg))
    sensitive = set(N.channel_sensitive_names(cfg))
    assert robust & sensitive == set()


def test_numeric_feature_columns_drops_identity_and_bookkeeping():
    columns = [
        "text",
        "t_start",
        "t_end",
        "gap_label",
        "rate_local_syll_per_s",
        "f0_n_voiced",
        "n_syllables",
        "f0_mean_st",
    ]
    assert N.numeric_feature_columns(columns) == ["rate_local_syll_per_s", "f0_mean_st"]
