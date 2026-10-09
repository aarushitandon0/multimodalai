"""Tests for ruptures PELT boundary refinement (spec 10.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.detect.refine import (
    RefinementSettings,
    frame_signal_for,
    refine_boundary,
    refine_regions,
    window_deviation,
)
from rostrum.detect.regions import label_category
from rostrum.types import Evidence, Region

HOP_S = 0.010


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def frames(step_at: float | None = None, *, n: int = 300, signal: str = "energy_z") -> pd.DataFrame:
    """A frame table whose ``signal`` steps from 0 to 5 at ``step_at`` seconds."""
    time = np.arange(n) * HOP_S
    values = np.zeros(n)
    if step_at is not None:
        values[time >= step_at] = 5.0
    table = pd.DataFrame({"time_s": time, signal: values})
    table.attrs.update(unit="frame", file_id="p", channel_id="ch_a", hop_s=HOP_S)
    return table


def flat_frames(n: int = 300, *, signal: str = "energy_z") -> pd.DataFrame:
    time = np.arange(n) * HOP_S
    table = pd.DataFrame({"time_s": time, signal: np.zeros(n)})
    table.attrs.update(unit="frame", file_id="p", channel_id="ch_a", hop_s=HOP_S)
    return table


def region(t0: float, t1: float, *, feature: str = "energy_rel_z", region_id: int = 0) -> Region:
    return Region(
        region_id=region_id,
        t_start=t0,
        t_end=t1,
        label="low_energy",
        category=label_category("low_energy"),
        word_idx=[0, 1],
        evidence=[Evidence(feature=feature, value=-3.0, env_median=0.0, env_sigma=1.0, z=-3.0)],
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_settings_come_from_the_config(cfg):
    settings = RefinementSettings.from_config(cfg)
    assert settings.window_s == pytest.approx(0.300)
    assert settings.max_shift_s == pytest.approx(0.300)
    assert settings.enabled is True
    assert settings.frame_signal["rate_local"] == "prominence_composite"
    assert settings.frame_signal["gap_duration"] == "energy_db_rel"


def test_a_non_pelt_or_non_rbf_config_is_refused():
    # Spec 10.5 fixes the search and the cost; a config drift must fail loudly
    # rather than silently refine on an l2 cost, which would miss a flattening
    # pitch track entirely.
    bad = load_config(overrides={"detection": {"refinement": {"cost": "l2"}}})
    with pytest.raises(ValueError, match="PELT with an RBF cost"):
        RefinementSettings.from_config(bad)


def test_overrides_win_over_the_config(cfg):
    settings = RefinementSettings.from_config(cfg, penalty=42.0)
    assert settings.penalty == pytest.approx(42.0)


# --------------------------------------------------------------------------- #
# Signal selection
# --------------------------------------------------------------------------- #


def test_the_frame_signal_map_is_consulted_first(cfg):
    settings = RefinementSettings.from_config(cfg)
    table = frames(signal="prominence_composite")
    assert frame_signal_for("rate_local", table, settings) == "prominence_composite"


def test_a_feature_named_like_its_own_column_resolves(cfg):
    settings = RefinementSettings.from_config(cfg)
    table = frames(signal="f0_st")
    assert frame_signal_for("f0_st", table, settings) == "f0_st"


def test_an_unmappable_feature_falls_back_then_gives_up(cfg):
    settings = RefinementSettings.from_config(cfg)
    assert frame_signal_for("something_unknown", frames(), settings) == "energy_z"
    only_time = pd.DataFrame({"time_s": np.arange(10) * HOP_S})
    assert frame_signal_for("something_unknown", only_time, settings) is None


# --------------------------------------------------------------------------- #
# The window deviation
# --------------------------------------------------------------------------- #


def test_window_deviation_is_scale_free():
    """One fixed penalty must mean the same thing in semitones and in dB."""
    time = np.arange(120) * HOP_S
    small = pd.DataFrame({"time_s": time, "x": np.where(time >= 0.6, 1.0, 0.0)})
    large = pd.DataFrame({"time_s": time, "x": np.where(time >= 0.6, 1000.0, 0.0)})
    _, dev_small = window_deviation(small, "x", 0.6, 0.3)
    _, dev_large = window_deviation(large, "x", 0.6, 0.3)
    assert np.allclose(dev_small, dev_large)


def test_window_deviation_drops_non_finite_frames():
    time = np.arange(120) * HOP_S
    values = np.zeros(120)
    values[30:40] = np.nan
    table = pd.DataFrame({"time_s": time, "x": values})
    times, dev = window_deviation(table, "x", 0.35, 0.3)
    assert np.isfinite(dev).all()
    assert times.size == dev.size
    assert times.size < 61


def test_window_deviation_outside_the_signal_is_empty():
    assert window_deviation(frames(), "energy_z", 99.0, 0.3)[0].size == 0


# --------------------------------------------------------------------------- #
# One boundary
# --------------------------------------------------------------------------- #


def test_refinement_snaps_to_a_real_change_point(cfg):
    settings = RefinementSettings.from_config(cfg)
    # The region claims the boundary is at 1.00 s; the signal steps at 1.12 s.
    t, moved, reason = refine_boundary(frames(step_at=1.12), "energy_z", 1.00, settings)
    assert reason == "accepted"
    assert moved
    assert t == pytest.approx(1.12, abs=0.02)


def test_refinement_never_moves_further_than_max_shift(cfg):
    """The guarantee the whole of 10.5 rests on."""
    settings = RefinementSettings.from_config(cfg)
    rng = np.random.default_rng(0)
    for trial in range(40):
        step = float(rng.uniform(0.0, 2.9))
        raw = float(rng.uniform(0.3, 2.6))
        t, _, _ = refine_boundary(frames(step_at=step), "energy_z", raw, settings)
        assert abs(t - raw) < settings.max_shift_s, f"trial {trial}: moved {t - raw:.3f} s"


def test_a_change_point_beyond_the_window_is_not_reachable(cfg):
    settings = RefinementSettings.from_config(cfg)
    # The step is 1.0 s away, far outside the +/-0.3 s window.
    t, moved, reason = refine_boundary(frames(step_at=2.00), "energy_z", 1.00, settings)
    assert t == pytest.approx(1.00)
    assert not moved
    assert reason in {"no_change_point", "shift_too_large"}


def test_a_flat_signal_keeps_the_raw_boundary(cfg):
    settings = RefinementSettings.from_config(cfg)
    t, moved, reason = refine_boundary(flat_frames(), "energy_z", 1.00, settings)
    assert t == pytest.approx(1.00)
    assert not moved
    assert reason == "no_change_point"


def test_too_few_frames_keeps_the_raw_boundary(cfg):
    settings = RefinementSettings.from_config(cfg)
    tiny = pd.DataFrame({"time_s": np.array([0.0, 0.01, 0.02]), "energy_z": np.zeros(3)})
    t, _moved, reason = refine_boundary(tiny, "energy_z", 0.01, settings)
    assert t == pytest.approx(0.01)
    assert reason == "too_short"


def test_a_larger_penalty_detects_fewer_change_points(cfg):
    """The penalty is the knob the tuner turns; it must actually do something."""
    time = np.arange(120) * HOP_S
    noisy = np.where(time >= 0.6, 0.35, 0.0) + np.random.default_rng(1).normal(0, 0.1, 120)
    table = pd.DataFrame({"time_s": time, "energy_z": noisy})
    lenient = RefinementSettings.from_config(cfg, penalty=0.5)
    strict = RefinementSettings.from_config(cfg, penalty=200.0)
    _, _, lenient_reason = refine_boundary(table, "energy_z", 0.60, lenient)
    _, _, strict_reason = refine_boundary(table, "energy_z", 0.60, strict)
    assert lenient_reason == "accepted"
    assert strict_reason == "no_change_point"


# --------------------------------------------------------------------------- #
# Whole regions
# --------------------------------------------------------------------------- #


def test_refine_regions_keeps_both_boundary_sets(cfg):
    settings = RefinementSettings.from_config(cfg)
    found = region(1.00, 2.00)
    refine_regions([found], frames(step_at=1.12), settings=settings)
    assert found.t_start_unrefined == pytest.approx(1.00)
    assert found.t_end_unrefined == pytest.approx(2.00)
    # The raw pair survives for the +/-refinement ablation (spec 14.2 #3).
    assert found.t_start != found.t_start_unrefined or found.t_end != found.t_end_unrefined


def test_no_frames_means_no_refinement_not_an_error(cfg):
    settings = RefinementSettings.from_config(cfg)
    found = region(1.00, 2.00)
    refine_regions([found], None, settings=settings)
    assert found.refined is False
    assert found.t_start == pytest.approx(1.00)
    assert found.t_start_unrefined == pytest.approx(1.00)


def test_disabled_refinement_leaves_the_boundaries_alone(cfg):
    settings = RefinementSettings.from_config(cfg, enabled=False)
    found = region(1.00, 2.00)
    refine_regions([found], frames(step_at=1.12), settings=settings)
    assert found.refined is False
    assert found.t_start == pytest.approx(1.00)


def test_a_refinement_that_would_invert_the_region_is_rejected(cfg):
    settings = RefinementSettings.from_config(cfg)
    # A region shorter than the search window, over a signal with one strong
    # change point: both boundaries would snap to the same instant.
    found = region(1.00, 1.05)
    refine_regions([found], frames(step_at=1.02), settings=settings)
    assert found.t_end > found.t_start


def test_every_region_keeps_its_shift_within_the_limit(cfg):
    settings = RefinementSettings.from_config(cfg)
    rng = np.random.default_rng(5)
    regions = [
        region(float(t0), float(t0) + 0.8, region_id=i)
        for i, t0 in enumerate(rng.uniform(0.4, 1.8, size=12))
    ]
    table = frames(step_at=1.25)
    refine_regions(regions, table, settings=settings)
    for found in regions:
        assert abs(found.t_start - found.t_start_unrefined) < settings.max_shift_s
        assert abs(found.t_end - found.t_end_unrefined) < settings.max_shift_s


def test_the_primary_feature_can_be_passed_explicitly(cfg):
    settings = RefinementSettings.from_config(cfg)
    found = region(1.00, 2.00, feature="not_a_feature")
    table = frames(step_at=1.12, signal="prominence_composite")
    # Without the override, `not_a_feature` falls back to energy_z, which this
    # table does not have, so nothing moves.
    refine_regions([found], table, settings=settings)
    assert found.refined is False

    found = region(1.00, 2.00, feature="not_a_feature")
    refine_regions([found], table, settings=settings, primary_features={0: "rate_local"})
    assert found.refined is True


def test_refinement_is_deterministic(cfg):
    settings = RefinementSettings.from_config(cfg)
    table = frames(step_at=1.17)
    results = []
    for _ in range(3):
        found = region(1.00, 2.00)
        refine_regions([found], table, settings=settings)
        results.append((found.t_start, found.t_end, found.refined))
    assert len(set(results)) == 1
