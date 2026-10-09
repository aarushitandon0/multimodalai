"""Unit tests for the injection primitives (``rostrum.inject.ops``).

Everything is checked against **synthetic signals** whose ground truth is known
by construction (``CLAUDE.md`` testing expectations), so no dataset audio is
needed and the numbers are exact rather than plausible:

* a word is a harmonic stack on a known F0 contour, so its pitch statistics are
  predictable;
* the gaps between words are low-level pink-ish noise, which stands in for room
  tone and lets the room-tone donor logic be exercised;
* each test asserts a tolerance taken from the prompt/spec (1 ms on timestamps,
  10 ms on span duration, 3 dB on inserted level, 0.5 dB on ramp level).
"""

from __future__ import annotations

import math

import numpy as np
import parselmouth
import pytest
from parselmouth.praat import call

from rostrum.inject import ops
from rostrum.inject.timemap import Segment, TimeMap
from rostrum.types import Alignment, Gap, Word

SR = 16_000
"""Working sample rate (``CLAUDE.md`` invariant 1)."""

NOISE_RMS = 2e-3
"""Room-tone level of the synthetic gaps, about -54 dBFS."""


# --------------------------------------------------------------------------- #
# Synthetic utterance
# --------------------------------------------------------------------------- #


def _voiced(duration: float, f0: float, depth_st: float, rng: np.random.Generator) -> np.ndarray:
    """A harmonic stack whose F0 sweeps ``+/- depth_st`` semitones around ``f0``.

    Eleven harmonics with 1/k amplitude approximate a glottal source closely
    enough for Praat's autocorrelation tracker to lock on, which is all the ops
    need.
    """
    n = round(duration * SR)
    t = np.arange(n) / SR
    contour = f0 * np.power(2.0, (depth_st / 12.0) * np.sin(2.0 * np.pi * 1.3 * t))
    phase = 2.0 * np.pi * np.cumsum(contour) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 12))
    sig = 0.35 * sig / np.max(np.abs(sig))
    # A gentle envelope so each word starts and ends without a step.
    edge = int(0.01 * SR)
    env = np.ones(n)
    env[:edge] = np.linspace(0.0, 1.0, edge)
    env[-edge:] = np.linspace(1.0, 0.0, edge)
    return sig * env + rng.normal(0.0, NOISE_RMS, n)


def _silence(duration: float, rng: np.random.Generator) -> np.ndarray:
    """Noise-floor silence: never digital zeros, matching the room-tone rule."""
    return rng.normal(0.0, NOISE_RMS, round(duration * SR))


def make_utterance(
    word_durations=(0.40, 0.50, 0.45),
    gap_durations=(0.50, 0.12),
    *,
    lead: float = 0.30,
    tail: float = 0.30,
    f0: float = 150.0,
    depth_st: float = 3.0,
    seed: int = 7,
) -> tuple[np.ndarray, Alignment]:
    """Build a synthetic utterance and its exact alignment.

    The default is a three-word utterance whose first gap is 500 ms - long
    enough to donate room tone (``room_tone.min_donor_gap_s`` is 400 ms) - and
    whose second gap is 120 ms, a plausible ``hesitation_pause`` target.

    Args:
        word_durations: Duration of each word in seconds.
        gap_durations: Duration of each inter-word gap; one fewer than words.
        lead: Leading silence.
        tail: Trailing silence.
        f0: Centre pitch in Hz.
        depth_st: Peak F0 excursion in semitones, so the contour has range to
            compress.
        seed: Noise seed, so the fixture is deterministic.

    Returns:
        ``(audio, alignment)`` with timings that match the audio to the sample.
    """
    assert len(gap_durations) == len(word_durations) - 1
    rng = np.random.default_rng(seed)

    pieces = [_silence(lead, rng)]
    words: list[Word] = []
    gaps: list[Gap] = []
    cursor = lead
    for i, wd in enumerate(word_durations):
        pieces.append(_voiced(wd, f0, depth_st, rng))
        words.append(Word(idx=i, text=f"w{i}", t_start=cursor, t_end=cursor + wd, display=f"w{i}"))
        cursor += wd
        if i < len(gap_durations):
            gd = gap_durations[i]
            pieces.append(_silence(gd, rng))
            gaps.append(
                Gap(
                    idx=i,
                    prev_word_idx=i,
                    next_word_idx=i + 1,
                    t_start=cursor,
                    t_end=cursor + gd,
                    label="boundary" if gd >= 0.30 else "none",
                )
            )
            cursor += gd
    pieces.append(_silence(tail, rng))

    audio = np.concatenate(pieces).astype(np.float64)
    alignment = Alignment(
        file_id="synthetic",
        text_id="synthetic_text",
        speaker_id="synth",
        channel_id="synth",
        words=words,
        gaps=gaps,
        duration=audio.size / SR,
        sample_rate=SR,
        aligner="synthetic",
    )
    return audio, alignment


@pytest.fixture
def utterance() -> tuple[np.ndarray, Alignment]:
    """The default synthetic three-word utterance."""
    return make_utterance()


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, as the engine will pass (invariant 13)."""
    return np.random.default_rng(1337)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _db(x: np.ndarray) -> float:
    """RMS level of ``x`` in dB, floored for silence."""
    return 20.0 * math.log10(max(ops._rms(x), 1e-12))


def _region(audio: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """Samples of ``audio`` between two times in seconds."""
    return audio[round(t0 * SR) : round(t1 * SR)]


def _median_f0(x: np.ndarray, ceiling_hz: float = 900.0) -> float:
    """Median voiced F0 of ``x`` in Hz, measured straight from Praat.

    The ceiling is deliberately generous so an upward-shifted word is not
    clipped by the tracker and the test measures the edit, not the analysis.
    """
    pitch = call(
        parselmouth.Sound(np.ascontiguousarray(x, dtype=np.float64), sampling_frequency=SR),
        "To Pitch (ac)...",
        0.0,
        50.0,
        15,
        "no",
        0.03,
        0.45,
        0.01,
        0.35,
        0.14,
        ceiling_hz,
    )
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced > 0.0]
    assert voiced.size > 0, "no voiced frames to measure"
    return float(np.median(voiced))


# --------------------------------------------------------------------------- #
# TimeMap
# --------------------------------------------------------------------------- #


def test_timemap_identity_is_the_identity():
    tmap = TimeMap.identity(3.0)
    assert tmap.delta == 0.0
    assert tmap.map_time(1.234) == pytest.approx(1.234)
    assert tmap.edited_spans == ()


def test_timemap_replacement_shifts_everything_after_the_cut():
    tmap = TimeMap.replacement(5.0, 1.0, 2.0, 0.7, op="x")
    assert tmap.delta == pytest.approx(-0.3)
    assert tmap.map_time(0.5) == pytest.approx(0.5)  # before: identity
    assert tmap.map_time(1.5) == pytest.approx(1.35)  # inside: scaled
    assert tmap.map_time(3.0) == pytest.approx(2.7)  # after: shifted by delta
    assert tmap.unmap_time(tmap.map_time(3.0)) == pytest.approx(3.0)
    assert tmap.edited_spans == ((1.0, 2.0),)


def test_timemap_insertion_is_a_step_and_intervals_contain_it():
    tmap = TimeMap.replacement(5.0, 2.5, 2.5, 0.8, op="ins", kind="insert")
    assert tmap.new_duration == pytest.approx(5.8)
    # A word ending exactly at the insertion keeps its offset...
    assert tmap.map_time(2.5) == pytest.approx(2.5)
    # ...while the next word onset is pushed past the inserted room tone.
    assert tmap.map_time(2.6) == pytest.approx(3.4)
    # A region bracketing the insertion grows to contain it.
    assert tmap.map_interval(2.0, 3.0) == pytest.approx((2.0, 3.8))


def test_timemap_composition_is_exact():
    first = TimeMap.replacement(5.0, 1.0, 2.0, 0.7, op="a")
    second = TimeMap.replacement(first.new_duration, 3.0, 3.5, 1.0, op="b")
    both = first.then(second)
    assert both.new_duration == pytest.approx(5.0 - 0.3 + 0.5)
    for t in np.linspace(0.0, 5.0, 101):
        assert both.map_time(t) == pytest.approx(second.map_time(first.map_time(t)), abs=1e-9)


def test_timemap_composition_preserves_an_insertion():
    first = TimeMap.replacement(5.0, 1.0, 2.0, 0.7, op="a")
    second = TimeMap.replacement(first.new_duration, 3.0, 3.0, 0.8, op="b", kind="insert")
    both = first.then(second)
    assert both.new_duration == pytest.approx(5.5)
    assert any(seg.kind == "insert" for seg in both.segments)


def test_timemap_rejects_a_discontinuous_map():
    with pytest.raises(ValueError, match="gap in old time"):
        TimeMap(segments=(Segment(0.0, 1.0, 0.0, 1.0), Segment(2.0, 3.0, 1.0, 2.0)))


def test_timemap_round_trips_through_json_shape():
    tmap = TimeMap.replacement(5.0, 1.0, 2.0, 0.7, op="x")
    assert TimeMap.from_dict(tmap.to_dict()).to_dict() == tmap.to_dict()


# --------------------------------------------------------------------------- #
# splice
# --------------------------------------------------------------------------- #


def test_splice_length_is_exactly_the_replacement_length(utterance):
    audio, _ = utterance
    rep = np.ones(1000)
    out = ops.splice(audio, 5000, 6000, rep, fade=160)
    assert out.size == audio.size  # same-length replacement
    out = ops.splice(audio, 5000, 6000, np.ones(2500), fade=160)
    assert out.size == audio.size + 1500


def test_splice_leaves_the_untouched_tail_bit_identical(utterance):
    audio, _ = utterance
    out = ops.splice(audio, 5000, 6000, np.ones(1000), fade=160)
    assert np.array_equal(out[6000:], audio[6000:])
    assert np.array_equal(out[:5000], audio[:5000])


def test_splice_rejects_an_invalid_region(utterance):
    audio, _ = utterance
    with pytest.raises(ops.InjectionError, match="invalid splice region"):
        ops.splice(audio, 100, 50, np.ones(10), fade=16)


# --------------------------------------------------------------------------- #
# Pitch range
# --------------------------------------------------------------------------- #


def test_two_pass_pitch_range_brackets_a_known_f0():
    rng = np.random.default_rng(3)
    tone = _voiced(1.5, 200.0, 0.0, rng)
    floor_hz, ceiling_hz = ops.two_pass_pitch_range(tone, SR)
    assert floor_hz < 200.0 < ceiling_hz
    # 0.75*Q1 and 1.5*Q3 of a steady 200 Hz tone.
    assert floor_hz == pytest.approx(150.0, rel=0.05)
    assert ceiling_hz == pytest.approx(300.0, rel=0.05)


def test_two_pass_pitch_range_falls_back_without_enough_voiced_frames():
    rng = np.random.default_rng(4)
    assert ops.two_pass_pitch_range(_silence(0.5, rng), SR) == (
        ops.PASS1_FLOOR_HZ,
        ops.PASS1_CEILING_HZ,
    )


# --------------------------------------------------------------------------- #
# 1. psola_lengthen
# --------------------------------------------------------------------------- #


def test_psola_lengthen_shifts_later_boundaries_by_the_exact_delta(utterance, rng):
    """Prompt acceptance test 1: factor 0.7 on a synthetic 3-word signal."""
    audio, alignment = utterance
    factor = 0.7
    target = alignment.words[1]

    out, tmap = ops.psola_lengthen(audio, SR, alignment, (1, 1), {"factor": factor}, rng)

    # The span really scales by `factor` (+/- 10 ms).
    new_t0, new_t1 = tmap.map_time(target.t_start), tmap.map_time(target.t_end)
    assert (new_t1 - new_t0) == pytest.approx(factor * target.duration, abs=0.010)

    # Every later boundary shifts by exactly the computed delta (+/- 1 ms).
    delta = tmap.delta
    assert delta == pytest.approx((factor - 1.0) * target.duration, abs=0.010)
    for word in alignment.words[2:]:
        assert tmap.map_time(word.t_start) == pytest.approx(word.t_start + delta, abs=0.001)
        assert tmap.map_time(word.t_end) == pytest.approx(word.t_end + delta, abs=0.001)

    # The audio is the length the map claims, to the sample.
    assert out.size == pytest.approx(round(tmap.new_duration * SR), abs=1)
    # Nothing before the span moved.
    assert tmap.map_time(alignment.words[0].t_end) == pytest.approx(
        alignment.words[0].t_end, abs=1e-9
    )


def test_psola_lengthen_dragging_grows_the_span(utterance, rng):
    audio, alignment = utterance
    out, tmap = ops.psola_lengthen(audio, SR, alignment, (1, 1), {"factor": 1.48}, rng)
    assert tmap.delta > 0.0
    assert out.size > audio.size
    assert tmap.edited_spans[0] == pytest.approx(
        (alignment.words[1].t_start, alignment.words[1].t_end), abs=0.003
    )


def test_psola_lengthen_preserves_pitch(utterance, rng):
    """Time-scaling must not transpose: a rushed speaker is not a higher one."""
    audio, alignment = utterance
    word = alignment.words[1]
    before = _region(audio, word.t_start, word.t_end)
    out, tmap = ops.psola_lengthen(audio, SR, alignment, (1, 1), {"factor": 0.7}, rng)
    after = _region(out, tmap.map_time(word.t_start), tmap.map_time(word.t_end))
    assert _median_f0(after) == pytest.approx(_median_f0(before), rel=0.03)


def test_psola_lengthen_factor_one_is_a_no_op(utterance, rng):
    audio, alignment = utterance
    out, tmap = ops.psola_lengthen(audio, SR, alignment, (1, 1), {"factor": 1.0}, rng)
    assert tmap.delta == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(out, audio)


def test_psola_lengthen_rejects_a_bad_factor_or_span(utterance, rng):
    audio, alignment = utterance
    with pytest.raises(ops.InjectionError, match="must be positive"):
        ops.psola_lengthen(audio, SR, alignment, (1, 1), {"factor": 0.0}, rng)
    with pytest.raises(ops.InjectionError, match="outside"):
        ops.psola_lengthen(audio, SR, alignment, (0, 9), {"factor": 0.7}, rng)


# --------------------------------------------------------------------------- #
# 2. shorten_gap
# --------------------------------------------------------------------------- #


def test_shorten_gap_keeps_the_requested_fraction(utterance, rng):
    audio, alignment = utterance
    gap = alignment.gaps[0]
    keep_frac = 0.43  # L3: 1 - 0.95 * 0.60

    out, tmap = ops.shorten_gap(audio, SR, alignment, gap.idx, {"keep_frac": keep_frac}, rng)

    kept = tmap.map_time(gap.t_end) - tmap.map_time(gap.t_start)
    assert kept == pytest.approx(keep_frac * gap.duration, abs=0.001)
    assert tmap.delta == pytest.approx(-(1.0 - keep_frac) * gap.duration, abs=0.001)
    assert out.size == pytest.approx(round(tmap.new_duration * SR), abs=1)
    # The word after the gap moves earlier by exactly the removed silence.
    nxt = alignment.words[gap.next_word_idx]
    assert tmap.map_time(nxt.t_start) == pytest.approx(nxt.t_start + tmap.delta, abs=0.001)


def test_shorten_gap_never_cuts_below_the_floor(utterance, rng):
    audio, alignment = utterance
    gap = alignment.gaps[0]
    _, tmap = ops.shorten_gap(audio, SR, alignment, gap.idx, {"keep_frac": 0.0}, rng)
    kept = tmap.map_time(gap.t_end) - tmap.map_time(gap.t_start)
    assert kept == pytest.approx(ops.MIN_KEPT_PAUSE_S, abs=0.001)


def test_shorten_gap_rejects_an_unknown_gap_or_fraction(utterance, rng):
    audio, alignment = utterance
    with pytest.raises(ops.InjectionError, match="no gap with idx"):
        ops.shorten_gap(audio, SR, alignment, 99, {"keep_frac": 0.5}, rng)
    with pytest.raises(ops.InjectionError, match=r"keep_frac must be in"):
        ops.shorten_gap(audio, SR, alignment, 0, {"keep_frac": 1.5}, rng)
    with pytest.raises(ops.InjectionError, match="cannot be shortened"):
        ops.shorten_gap(audio, SR, alignment, 0, {"keep_frac": 1.0}, rng)


# --------------------------------------------------------------------------- #
# 3. insert_roomtone
# --------------------------------------------------------------------------- #


def test_insert_roomtone_adds_exactly_the_requested_duration(utterance, rng):
    """Prompt acceptance test 2: exact duration and room-tone-matched level."""
    audio, alignment = utterance
    gap = alignment.gaps[1]  # the 120 ms, non-boundary gap
    donor = alignment.gaps[0]  # the 500 ms boundary gap
    dur_s = 1.32  # L3: 0.3 + 1.7 * 0.60

    out, tmap = ops.insert_roomtone(audio, SR, alignment, gap.idx, {"dur_s": dur_s}, rng)

    # Exactly dur_s longer (+/- 1 ms).
    assert (out.size - audio.size) / SR == pytest.approx(dur_s, abs=0.001)
    assert tmap.delta == pytest.approx(dur_s, abs=0.001)
    assert out.size == pytest.approx(round(tmap.new_duration * SR), abs=1)

    # The inserted level matches the donor pause within 3 dB.
    at = 0.5 * (gap.t_start + gap.t_end)
    inserted = _region(out, at + 0.030, at + dur_s - 0.030)
    donor_tone = _region(audio, donor.t_start + 0.050, donor.t_end - 0.050)
    assert _db(inserted) == pytest.approx(_db(donor_tone), abs=3.0)

    # It is room tone, not digital silence.
    assert np.count_nonzero(inserted) == inserted.size

    # Words on either side keep their own timing.
    prev_word = alignment.words[gap.prev_word_idx]
    next_word = alignment.words[gap.next_word_idx]
    assert tmap.map_time(prev_word.t_end) == pytest.approx(prev_word.t_end, abs=1e-9)
    assert tmap.map_time(next_word.t_start) == pytest.approx(next_word.t_start + dur_s, abs=0.001)


def test_room_tone_loops_a_short_donor_up_to_length(utterance, rng):
    audio, alignment = utterance
    want = int(3.0 * SR)  # much longer than the 500 ms donor
    tone = ops.room_tone(audio, SR, alignment, want, rng=rng)
    assert tone.size == want
    donor = _region(audio, alignment.gaps[0].t_start + 0.05, alignment.gaps[0].t_end - 0.05)
    assert _db(tone) == pytest.approx(_db(donor), abs=1.0)


def test_room_tone_needs_a_long_enough_donor_pause(rng):
    audio, alignment = make_utterance(gap_durations=(0.10, 0.12))
    with pytest.raises(ops.InjectionError, match="no pause of at least"):
        ops.room_tone(audio, SR, alignment, 1000, rng=rng)


def test_insert_roomtone_rejects_a_bad_duration(utterance, rng):
    audio, alignment = utterance
    with pytest.raises(ops.InjectionError, match="must be positive"):
        ops.insert_roomtone(audio, SR, alignment, 1, {"dur_s": 0.0}, rng)


# --------------------------------------------------------------------------- #
# 4. psola_pitch_compress
# --------------------------------------------------------------------------- #


def test_pitch_compress_removes_most_of_the_f0_range(utterance, rng):
    """Prompt acceptance test 3: alpha=0.9 cuts semitone F0 std by >= 70%."""
    audio, alignment = utterance
    span = (0, 2)
    t0, t1 = alignment.words[0].t_start, alignment.words[2].t_end

    out, tmap = ops.psola_pitch_compress(audio, SR, alignment, span, {"alpha": 0.9}, rng)

    floor_hz, ceiling_hz = ops.two_pass_pitch_range(_region(audio, t0, t1), SR)
    before = ops._f0_semitone_std(_region(audio, t0, t1), SR, floor_hz, ceiling_hz)
    after = ops._f0_semitone_std(_region(out, t0, t1), SR, floor_hz, ceiling_hz)
    assert before > 0.5, "the fixture must have pitch range to remove"
    assert after <= 0.30 * before, f"std {before:.3f} st -> {after:.3f} st is under 70%"

    # Length-preserving, so the time map is the identity.
    assert out.size == audio.size
    assert tmap.delta == 0.0
    assert tmap.map_time(1.0) == pytest.approx(1.0)


def test_pitch_compress_accepts_the_config_exponent_form(utterance, rng):
    audio, alignment = utterance
    by_alpha, _ = ops.psola_pitch_compress(audio, SR, alignment, (0, 2), {"alpha": 0.54}, rng)
    by_exp, _ = ops.psola_pitch_compress(
        audio, SR, alignment, (0, 2), {"exponent": 1.0 - 0.54}, rng
    )
    assert np.allclose(by_alpha, by_exp)


def test_pitch_compress_leaves_audio_outside_the_span_alone(utterance, rng):
    audio, alignment = utterance
    out, _ = ops.psola_pitch_compress(audio, SR, alignment, (1, 1), {"alpha": 0.9}, rng)
    before_span = round(alignment.words[1].t_start * SR)
    assert np.array_equal(out[:before_span], audio[:before_span])


def test_pitch_compress_rejects_a_bad_alpha(utterance, rng):
    audio, alignment = utterance
    with pytest.raises(ops.InjectionError, match="alpha must be in"):
        ops.psola_pitch_compress(audio, SR, alignment, (0, 2), {"alpha": 1.4}, rng)


# --------------------------------------------------------------------------- #
# 5. stress_swap
# --------------------------------------------------------------------------- #


def test_stress_swap_boosts_one_word_and_damps_the_other(utterance, rng):
    audio, alignment = utterance
    fn, ct = 0, 2  # pretend word 0 is a function word, word 2 the content word
    s = 0.60

    out, tmap = ops.stress_swap(audio, SR, alignment, (fn, ct), {"s": s}, rng)

    fn_word, ct_word = alignment.words[fn], alignment.words[ct]
    fn_new = tmap.map_interval(fn_word.t_start, fn_word.t_end)
    ct_new = tmap.map_interval(ct_word.t_start, ct_word.t_end)

    # Duration: x(1 + 0.3s) and x(1 - 0.2s).
    assert (fn_new[1] - fn_new[0]) == pytest.approx((1.0 + 0.3 * s) * fn_word.duration, abs=0.010)
    assert (ct_new[1] - ct_new[0]) == pytest.approx((1.0 - 0.2 * s) * ct_word.duration, abs=0.010)
    assert out.size == pytest.approx(round(tmap.new_duration * SR), abs=2)

    # Level: +4s dB and -4s dB. Lengthening is energy-preserving per sample, so
    # the RMS change is the gain change.
    fn_before = _db(_region(audio, fn_word.t_start + 0.02, fn_word.t_end - 0.02))
    fn_after = _db(_region(out, fn_new[0] + 0.02, fn_new[1] - 0.02))
    ct_before = _db(_region(audio, ct_word.t_start + 0.02, ct_word.t_end - 0.02))
    ct_after = _db(_region(out, ct_new[0] + 0.02, ct_new[1] - 0.02))
    assert fn_after - fn_before == pytest.approx(+4.0 * s, abs=1.0)
    assert ct_after - ct_before == pytest.approx(-4.0 * s, abs=1.0)


def test_stress_swap_f0_moves_in_opposite_directions(utterance, rng):
    """The function word rises and the content word falls, by about ``4s`` st."""
    audio, alignment = utterance
    s = 0.90
    out, tmap = ops.stress_swap(audio, SR, alignment, (0, 2), {"s": s}, rng)

    for word_idx, sign in ((0, +1.0), (2, -1.0)):
        word = alignment.words[word_idx]
        before = _median_f0(_region(audio, word.t_start + 0.02, word.t_end - 0.02))
        new0, new1 = tmap.map_interval(word.t_start, word.t_end)
        after = _median_f0(_region(out, new0 + 0.03, new1 - 0.03))
        shift_st = 12.0 * math.log2(after / before)
        assert sign * shift_st > 1.5, f"word {word_idx} moved {shift_st:+.2f} st"
        assert shift_st == pytest.approx(sign * 4.0 * s, abs=1.0)


def test_stress_swap_rejects_a_degenerate_pair(utterance, rng):
    audio, alignment = utterance
    with pytest.raises(ops.InjectionError, match="two distinct word indices"):
        ops.stress_swap(audio, SR, alignment, (1, 1), {"s": 0.6}, rng)


def test_stress_swap_time_map_covers_the_whole_recording(utterance, rng):
    audio, alignment = utterance
    _, tmap = ops.stress_swap(audio, SR, alignment, (0, 2), {"s": 0.6}, rng)
    assert tmap.segments[0].old_start == pytest.approx(0.0)
    assert tmap.old_duration == pytest.approx(audio.size / SR, abs=1e-9)
    assert tmap.op == "stress_swap"


# --------------------------------------------------------------------------- #
# 6. gain_ramp
# --------------------------------------------------------------------------- #


def test_gain_ramp_reaches_the_target_end_level(utterance, rng):
    """Prompt acceptance test 4: end level within 0.5 dB of target."""
    audio, alignment = utterance
    db_end = -10.8  # L3: -18 * 0.60
    span = (1, 2)
    t1 = alignment.words[2].t_end

    out, tmap = ops.gain_ramp(audio, SR, alignment, span, {"db_end": db_end}, rng)

    # Compare the last 20 ms of the span against the same samples before.
    tail_before = _region(audio, t1 - 0.020, t1)
    tail_after = _region(out, t1 - 0.020, t1)
    assert _db(tail_after) - _db(tail_before) == pytest.approx(db_end, abs=0.5)

    # The very last sample of the span carries exactly the target gain.
    last = round(t1 * SR) - 1
    assert 20.0 * math.log10(abs(out[last] / audio[last])) == pytest.approx(db_end, abs=0.01)

    assert out.size == audio.size
    assert tmap.delta == 0.0


def test_gain_ramp_starts_at_unity_and_leaves_the_rest_alone(utterance, rng):
    audio, alignment = utterance
    out, _ = ops.gain_ramp(audio, SR, alignment, (1, 2), {"db_end": -10.8}, rng)
    start = round(alignment.words[1].t_start * SR)
    end = round(alignment.words[2].t_end * SR)
    assert out[start] == pytest.approx(audio[start], abs=1e-12)
    assert np.array_equal(out[:start], audio[:start])
    assert np.array_equal(out[end:], audio[end:])


def test_gain_ramp_linear_db_reaches_the_same_end_level(utterance, rng):
    audio, alignment = utterance
    out, _ = ops.gain_ramp(
        audio, SR, alignment, (1, 2), {"db_end": -10.8, "ramp": "linear_db"}, rng
    )
    last = round(alignment.words[2].t_end * SR) - 1
    assert 20.0 * math.log10(abs(out[last] / audio[last])) == pytest.approx(-10.8, abs=0.01)


def test_gain_ramp_rejects_an_unknown_shape(utterance, rng):
    audio, alignment = utterance
    with pytest.raises(ops.InjectionError, match="unknown ramp shape"):
        ops.gain_ramp(audio, SR, alignment, (1, 2), {"db_end": -6.0, "ramp": "sine"}, rng)


# --------------------------------------------------------------------------- #
# Cross-cutting invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "span", "params"),
    [
        ("psola_lengthen", (1, 1), {"factor": 0.73}),
        ("shorten_gap", 0, {"keep_frac": 0.43}),
        ("insert_roomtone", 1, {"dur_s": 1.32}),
        ("psola_pitch_compress", (0, 2), {"alpha": 0.54}),
        ("stress_swap", (0, 2), {"s": 0.60}),
        ("gain_ramp", (1, 2), {"db_end": -10.8}),
    ],
)
def test_every_op_is_deterministic_and_leaves_the_input_untouched(name, span, params):
    """Invariant 13: same seed, same bytes. And ops never mutate their input."""
    audio, alignment = make_utterance()
    original = audio.copy()
    op = getattr(ops, name)

    first, map_a = op(audio, SR, alignment, span, params, np.random.default_rng(1337))
    second, map_b = op(audio, SR, alignment, span, params, np.random.default_rng(1337))

    assert np.array_equal(first, second)
    assert map_a.to_dict() == map_b.to_dict()
    assert np.array_equal(audio, original), f"{name} mutated its input"


@pytest.mark.parametrize(
    ("name", "span", "params"),
    [
        ("psola_lengthen", (1, 1), {"factor": 0.73}),
        ("shorten_gap", 0, {"keep_frac": 0.43}),
        ("insert_roomtone", 1, {"dur_s": 1.32}),
        ("psola_pitch_compress", (0, 2), {"alpha": 0.54}),
        ("stress_swap", (0, 2), {"s": 0.60}),
        ("gain_ramp", (1, 2), {"db_end": -10.8}),
    ],
)
def test_every_time_map_matches_the_audio_it_describes(name, span, params):
    """The map is only useful if the audio really is the length it claims."""
    audio, alignment = make_utterance()
    out, tmap = getattr(ops, name)(audio, SR, alignment, span, params, np.random.default_rng(1))
    assert tmap.old_duration == pytest.approx(audio.size / SR, abs=1e-9)
    assert out.size == pytest.approx(round(tmap.new_duration * SR), abs=2)
    assert tmap.op != ""
