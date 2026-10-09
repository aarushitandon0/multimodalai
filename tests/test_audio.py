"""Tests for the preprocessing chain (spec 4.6, ``CLAUDE.md`` invariant 1).

Everything here runs on synthetic signals written to ``tmp_path``, so the suite
needs no dataset audio and no network.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from rostrum.io.audio import (
    DEFAULT_PEAK_DBFS,
    DEFAULT_TARGET_LUFS,
    SAMPLE_RATE,
    AudioError,
    describe,
    duration_s,
    integrated_lufs,
    keep_padding,
    load_any,
    loudness_normalize,
    peak_dbfs,
    preprocess,
    to_mono_16k,
    trim_excerpt,
    write_wav16,
)

TONE_HZ = 200.0


def tone(seconds: float, sr: int = SAMPLE_RATE, freq: float = TONE_HZ, amp: float = 0.5):
    """A deterministic sine, the stand-in for speech in these tests."""
    t = np.arange(round(seconds * sr), dtype=np.float64) / sr
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------------------- #
# load_any
# --------------------------------------------------------------------------- #


def test_load_any_round_trips_a_wav(tmp_path: Path) -> None:
    src = tmp_path / "x.wav"
    sf.write(src, tone(1.0, 44_100), 44_100, subtype="PCM_16")
    data, sr = load_any(src)
    assert sr == 44_100
    assert data.dtype == np.float32
    assert data.shape == (44_100,)


def test_load_any_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_any(tmp_path / "nope.wav")


def test_load_any_rejects_undecodable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio at all")
    with pytest.raises(AudioError):
        load_any(bad, allow_ffmpeg=False)


# --------------------------------------------------------------------------- #
# to_mono_16k
# --------------------------------------------------------------------------- #


def test_to_mono_16k_resamples_to_16k() -> None:
    out = to_mono_16k(tone(2.0, 44_100), 44_100)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(out.shape[0] - 2 * SAMPLE_RATE) <= SAMPLE_RATE // 100


def test_to_mono_16k_downmixes_channels() -> None:
    left = tone(0.5)
    right = -left
    stereo = np.stack([left, right], axis=1)
    out = to_mono_16k(stereo, SAMPLE_RATE)
    # Mean of a signal and its inverse is silence: proves the mean, not a pick.
    assert np.allclose(out, 0.0, atol=1e-6)


def test_to_mono_16k_is_a_no_op_at_target_rate() -> None:
    x = tone(0.5)
    assert np.array_equal(to_mono_16k(x, SAMPLE_RATE), x)


def test_to_mono_16k_is_deterministic() -> None:
    x = tone(1.0, 44_100)
    assert np.array_equal(to_mono_16k(x, 44_100), to_mono_16k(x, 44_100))


@pytest.mark.parametrize("bad", [np.zeros((0,), np.float32), np.zeros((4, 2, 2), np.float32)])
def test_to_mono_16k_rejects_bad_shapes(bad: np.ndarray) -> None:
    with pytest.raises(AudioError):
        to_mono_16k(bad, SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# trim_excerpt
# --------------------------------------------------------------------------- #


def test_trim_excerpt_cuts_exact_sample_counts() -> None:
    x = tone(3.0)
    out = trim_excerpt(x, SAMPLE_RATE, 1.0, 2.5)
    assert out.shape[0] == round(1.5 * SAMPLE_RATE)
    assert np.array_equal(out, x[SAMPLE_RATE : int(2.5 * SAMPLE_RATE)])


def test_trim_excerpt_none_bounds_keep_everything() -> None:
    x = tone(1.0)
    assert np.array_equal(trim_excerpt(x, SAMPLE_RATE, None, None), x)


def test_trim_excerpt_clamps_an_end_past_the_file() -> None:
    x = tone(1.0)
    assert trim_excerpt(x, SAMPLE_RATE, 0.5, 99.0).shape[0] == int(0.5 * SAMPLE_RATE)


@pytest.mark.parametrize(("start", "end"), [(2.0, 1.0), (-1.0, 0.5), (5.0, 6.0)])
def test_trim_excerpt_rejects_impossible_bounds(start: float, end: float) -> None:
    with pytest.raises(AudioError):
        trim_excerpt(tone(1.0), SAMPLE_RATE, start, end)


# --------------------------------------------------------------------------- #
# keep_padding
# --------------------------------------------------------------------------- #


def test_keep_padding_keeps_300_ms_of_each_side() -> None:
    quiet = np.full(SAMPLE_RATE, 1e-5, dtype=np.float32)  # 1 s of noise floor
    signal = np.concatenate([quiet, tone(1.0), quiet])
    out = keep_padding(signal, SAMPLE_RATE, keep_s=0.300)
    # 1 s of speech plus 300 ms either side, within one 25 ms analysis frame.
    assert out.shape[0] == pytest.approx(1.6 * SAMPLE_RATE, abs=0.05 * SAMPLE_RATE)


def test_keep_padding_keeps_what_little_silence_exists() -> None:
    signal = np.concatenate([np.full(800, 1e-5, dtype=np.float32), tone(1.0)])
    out = keep_padding(signal, SAMPLE_RATE, keep_s=0.300)
    assert out.shape[0] <= signal.shape[0]
    assert out.shape[0] >= SAMPLE_RATE  # the speech itself survives


def test_keep_padding_returns_an_all_silent_clip_unchanged() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert np.array_equal(keep_padding(silence, SAMPLE_RATE), silence)


def test_keep_padding_rejects_negative_keep() -> None:
    with pytest.raises(AudioError):
        keep_padding(tone(1.0), SAMPLE_RATE, keep_s=-0.1)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def test_peak_dbfs_of_full_scale_is_zero() -> None:
    assert peak_dbfs(np.array([1.0, -1.0], dtype=np.float32)) == pytest.approx(0.0)


def test_peak_dbfs_of_half_scale_is_minus_six() -> None:
    assert peak_dbfs(np.array([0.5], dtype=np.float32)) == pytest.approx(-6.02, abs=0.01)


def test_peak_dbfs_of_silence_is_floored() -> None:
    assert peak_dbfs(np.zeros(10, dtype=np.float32)) == -120.0


def test_integrated_lufs_is_none_for_a_too_short_clip() -> None:
    assert integrated_lufs(tone(0.2), SAMPLE_RATE) is None


def test_duration_s_is_rounded_to_milliseconds() -> None:
    assert duration_s(np.zeros(16_007, dtype=np.float32), SAMPLE_RATE) == 1.0


# --------------------------------------------------------------------------- #
# loudness_normalize
# --------------------------------------------------------------------------- #


def test_loudness_normalize_hits_the_target() -> None:
    out = loudness_normalize(tone(3.0), SAMPLE_RATE)
    assert integrated_lufs(out, SAMPLE_RATE) == pytest.approx(DEFAULT_TARGET_LUFS, abs=0.1)
    assert peak_dbfs(out) <= DEFAULT_PEAK_DBFS + 1e-6


def test_loudness_normalize_raises_the_level_of_a_quiet_clip() -> None:
    quiet = tone(3.0, amp=0.01)
    out = loudness_normalize(quiet, SAMPLE_RATE)
    assert peak_dbfs(out) > peak_dbfs(quiet)
    assert integrated_lufs(out, SAMPLE_RATE) == pytest.approx(DEFAULT_TARGET_LUFS, abs=0.1)


def test_loudness_normalize_prefers_the_peak_ceiling_over_the_target() -> None:
    # A spiky signal: reaching -23 LUFS would need a gain that clips, so the
    # peak rule wins and the result ends up quieter than the target.
    spiky = tone(3.0, amp=0.02).copy()
    spiky[::4000] = 0.9
    out = loudness_normalize(spiky, SAMPLE_RATE)
    assert peak_dbfs(out) == pytest.approx(DEFAULT_PEAK_DBFS, abs=0.01)
    assert integrated_lufs(out, SAMPLE_RATE) <= DEFAULT_TARGET_LUFS + 0.1


def test_loudness_normalize_rejects_silence() -> None:
    with pytest.raises(AudioError):
        loudness_normalize(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)


def test_loudness_normalize_rejects_a_clip_shorter_than_one_block() -> None:
    with pytest.raises(AudioError):
        loudness_normalize(tone(0.2), SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# write_wav16
# --------------------------------------------------------------------------- #


def test_write_wav16_writes_16_bit_mono_pcm(tmp_path: Path) -> None:
    dest = write_wav16(tmp_path / "out" / "x.wav", tone(1.0), SAMPLE_RATE)
    info = sf.info(dest)
    assert info.subtype == "PCM_16"
    assert info.channels == 1
    assert info.samplerate == SAMPLE_RATE
    assert not list(dest.parent.glob("*.tmp"))


def test_write_wav16_rejects_over_full_scale(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        write_wav16(tmp_path / "x.wav", np.array([1.5], dtype=np.float32), SAMPLE_RATE)


def test_write_wav16_rejects_empty_and_multichannel(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        write_wav16(tmp_path / "a.wav", np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(AudioError):
        write_wav16(tmp_path / "b.wav", np.zeros((10, 2), dtype=np.float32), SAMPLE_RATE)


# --------------------------------------------------------------------------- #
# preprocess (the whole chain)
# --------------------------------------------------------------------------- #


def test_preprocess_meets_every_invariant(tmp_path: Path) -> None:
    quiet = np.full(44_100, 1e-5, dtype=np.float32)
    src = tmp_path / "raw.flac"
    sf.write(src, np.concatenate([quiet, tone(4.0, 44_100), quiet]), 44_100)

    out, stats = preprocess(src, tmp_path / "clean" / "x.wav")

    info = sf.info(out)
    assert info.samplerate == SAMPLE_RATE and info.channels == 1
    assert info.subtype == "PCM_16"
    assert stats.sample_rate == SAMPLE_RATE
    assert stats.peak_dbfs <= DEFAULT_PEAK_DBFS + 0.01
    assert stats.lufs == pytest.approx(DEFAULT_TARGET_LUFS, abs=0.5)
    assert stats.duration_s == pytest.approx(4.6, abs=0.1)
    assert stats.duration_s == round(stats.duration_s, 3)


def test_preprocess_applies_the_excerpt_before_measuring(tmp_path: Path) -> None:
    src = tmp_path / "raw.wav"
    sf.write(src, np.concatenate([tone(2.0), tone(2.0, freq=880.0)]), SAMPLE_RATE)
    _, stats = preprocess(
        src, tmp_path / "x.wav", excerpt_start_s=0.0, excerpt_end_s=2.0, keep_s=0.0
    )
    assert stats.duration_s == pytest.approx(2.0, abs=0.05)


def test_preprocess_is_byte_identical_across_runs(tmp_path: Path) -> None:
    src = tmp_path / "raw.wav"
    sf.write(src, tone(3.0, 44_100), 44_100)
    a, _ = preprocess(src, tmp_path / "a.wav")
    b, _ = preprocess(src, tmp_path / "b.wav")
    assert a.read_bytes() == b.read_bytes()


def test_describe_matches_the_file_on_disk(tmp_path: Path) -> None:
    dest = write_wav16(tmp_path / "x.wav", loudness_normalize(tone(3.0), SAMPLE_RATE))
    data, sr = sf.read(dest, dtype="float32")
    stats = describe(data, sr)
    assert stats.n_samples == data.shape[0]
    assert stats.duration_s == pytest.approx(3.0, abs=0.01)
