"""The global frame grid: 10 ms hop, 25 ms window (CLAUDE.md invariant 3)."""

from __future__ import annotations

import numpy as np
import pytest

from rostrum.features.frames import FrameGrid


@pytest.fixture
def grid() -> FrameGrid:
    return FrameGrid(hop_s=0.010, window_s=0.025, sample_rate=16000)


def test_grid_defaults_to_the_project_invariant(grid: FrameGrid) -> None:
    assert (grid.hop_s, grid.window_s, grid.sample_rate) == (0.010, 0.025, 16000)


def test_hop_and_window_in_samples(grid: FrameGrid) -> None:
    assert (grid.hop_samples, grid.window_samples) == (160, 400)


def test_frame_count_fits_whole_windows_only(grid: FrameGrid) -> None:
    # 1.000 s = 975 ms of window starts at 10 ms spacing -> 98 frames.
    assert grid.n_frames(1.0) == 98


def test_audio_shorter_than_one_window_yields_no_frames(grid: FrameGrid) -> None:
    assert grid.n_frames(0.02) == 0


def test_frame_starts_step_by_the_hop(grid: FrameGrid) -> None:
    assert grid.starts(3) == pytest.approx([0.0, 0.010, 0.020])


def test_frame_centres_sit_half_a_window_in(grid: FrameGrid) -> None:
    assert grid.centres(2) == pytest.approx([0.0125, 0.0225])


def test_frames_of_a_span_are_those_whose_centre_is_inside(grid: FrameGrid) -> None:
    idx = grid.frame_indices(0.0225, 0.0425, n_frames=10)
    assert list(idx) == [1, 2, 3]


def test_span_outside_the_audio_selects_nothing(grid: FrameGrid) -> None:
    assert grid.frame_indices(5.0, 6.0, n_frames=10).size == 0


def test_empty_span_selects_nothing(grid: FrameGrid) -> None:
    assert grid.frame_indices(0.05, 0.05, n_frames=10).size == 0


def test_resampling_takes_the_nearest_source_frame(grid: FrameGrid) -> None:
    src_times = np.array([0.0125, 0.0225, 0.0325])
    values = np.array([1.0, 2.0, 3.0])
    out = grid.resample_nearest(src_times, values, n_frames=3)
    assert out == pytest.approx([1.0, 2.0, 3.0])


def test_resampling_leaves_nan_where_the_source_has_no_frame_nearby(grid: FrameGrid) -> None:
    out = grid.resample_nearest(np.array([0.5]), np.array([1.0]), n_frames=3)
    assert np.all(np.isnan(out))


def test_resampling_tolerance_is_half_a_hop_by_default(grid: FrameGrid) -> None:
    # 0.0125 + 0.004 is inside half a hop; + 0.006 is not.
    near = grid.resample_nearest(np.array([0.0165]), np.array([7.0]), n_frames=1)
    far = grid.resample_nearest(np.array([0.0185]), np.array([7.0]), n_frames=1)
    assert near[0] == 7.0
    assert np.isnan(far[0])


def test_resampling_an_empty_source_is_all_nan(grid: FrameGrid) -> None:
    out = grid.resample_nearest(np.array([]), np.array([]), n_frames=4)
    assert out.shape == (4,) and np.all(np.isnan(out))


def test_resampling_rejects_mismatched_lengths(grid: FrameGrid) -> None:
    with pytest.raises(ValueError, match="same length"):
        grid.resample_nearest(np.array([0.0, 0.01]), np.array([1.0]), n_frames=2)


def test_grid_is_built_from_the_feature_config() -> None:
    from rostrum.config import load_config

    cfg = load_config()
    built = FrameGrid.from_config(cfg)
    assert (built.hop_s, built.window_s, built.sample_rate) == (
        cfg.hop_s,
        cfg.window_s,
        cfg.sample_rate,
    )


def test_a_non_positive_hop_is_rejected() -> None:
    with pytest.raises(ValueError, match="hop"):
        FrameGrid(hop_s=0.0, window_s=0.025, sample_rate=16000)


def test_a_window_shorter_than_the_hop_is_rejected() -> None:
    with pytest.raises(ValueError, match="window"):
        FrameGrid(hop_s=0.010, window_s=0.005, sample_rate=16000)
