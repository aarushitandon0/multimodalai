"""Tests for ``scripts/feature_demo.py`` (spec 9 acceptance).

The alignment branch is covered with the fixture TextGrid, so the word-boundary
path is exercised even though MFA has not been run in this checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import feature_demo as demo
from rostrum.align.mfa import read_textgrid
from rostrum.features.frames import FrameGrid
from rostrum.io.audio import write_wav16
from rostrum.types import Word

SR = 16000
GRID = FrameGrid()
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_long.TextGrid"


def words(n: int, duration: float = 0.2) -> list[Word]:
    return [
        Word(
            idx=i,
            text=f"w{i}",
            t_start=i * duration,
            t_end=(i + 1) * duration,
            phones=["D", "AH1"],
            n_syllables=1,
        )
        for i in range(n)
    ]


def test_word_rate_curve_is_constant_for_evenly_spaced_words() -> None:
    times = GRID.centres(GRID.n_frames(2.0))
    curve = demo.rate_curve_from_words(words(10), times, window_words=3)
    inside = curve[np.isfinite(curve)]
    assert inside.size > 0
    assert np.allclose(inside, 5.0)


def test_word_rate_curve_is_nan_outside_every_word() -> None:
    times = GRID.centres(GRID.n_frames(2.0))
    curve = demo.rate_curve_from_words(words(2), times, window_words=3)
    assert np.isnan(curve[-1])


def test_nuclei_rate_curve_counts_nuclei_in_the_window() -> None:
    times = np.array([1.0])
    curve = demo.rate_curve_from_nuclei([0.1, 0.5, 0.9, 1.4, 1.9], times, window_s=2.0)
    # Window [0.0, 2.0) holds all five nuclei -> 2.5 per second.
    assert curve[0] == pytest.approx(2.5)


def test_nuclei_rate_curve_without_nuclei_is_all_nan() -> None:
    curve = demo.rate_curve_from_nuclei([], np.array([0.5, 1.0]))
    assert np.all(np.isnan(curve))


def test_demo_writes_a_png_without_an_alignment(tmp_path: Path) -> None:
    wav = tmp_path / "demo01.wav"
    t = np.arange(int(2.0 * SR), dtype=np.float64) / SR
    write_wav16(wav, (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))

    out = tmp_path / "demo.png"
    rc = demo.main(["--audio", str(wav), "--seconds", "2", "--out", str(out)])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0


def test_demo_reports_a_missing_audio_file(tmp_path: Path) -> None:
    assert demo.main(["--audio", str(tmp_path / "nope.wav")]) == 2


def test_demo_uses_word_boundaries_when_the_alignment_exists(tmp_path: Path) -> None:
    wav = tmp_path / "tiny_long.wav"
    t = np.arange(int(2.4 * SR), dtype=np.float64) / SR
    write_wav16(wav, (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))

    out = tmp_path / "aligned.png"
    rc = demo.main(
        ["--audio", str(wav), "--alignment", str(FIXTURE), "--seconds", "2.4", "--out", str(out)]
    )
    assert rc == 0
    assert out.is_file()


def test_fixture_alignment_yields_a_usable_rate_curve() -> None:
    alignment = read_textgrid(FIXTURE, file_id="tiny_long")
    times = GRID.centres(GRID.n_frames(2.4))
    curve = demo.rate_curve_from_words(alignment.words, times, window_words=3)
    assert np.isfinite(curve).any()
