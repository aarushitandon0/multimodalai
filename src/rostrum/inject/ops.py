"""Primitive flaw-injection operations (spec 5.1-5.2).

Each public op has the same shape::

    op(audio, sr, alignment, span, params, rng) -> (new_audio, TimeMap)

``audio`` is mono float, ``alignment`` is the forced alignment of the *clean*
recording, ``span`` identifies what to edit in alignment terms (word indices or
a gap index, never samples), ``params`` is the resolved numeric parameter set
for one severity level, and ``rng`` is a seeded :class:`numpy.random.Generator`.
The returned :class:`~rostrum.inject.timemap.TimeMap` is the exact old->new
piecewise-linear time map (``CLAUDE.md`` invariant 8); every downstream label
timestamp is derived from it rather than re-measured.

Ops are pure: they never touch the filesystem and never mutate ``audio``.

Splicing
--------
Every length-changing op goes through :func:`splice`, which cross-fades over
10 ms raised-cosine windows **without consuming length**. The outgoing side of
a join is the material that *would* have played there in the clean file, so
there is always real neighbouring audio to fade against and never digital
silence (invariant 8). Because the join costs no samples, the new length of the
edited region is exactly the length of the replacement, which is what makes the
time maps exact instead of approximate.

Praat commands used (verified against the installed ``parselmouth`` 0.4.7 /
Praat build with :func:`parselmouth.praat.call`, October 2025 - each was called
and its return type checked before this module was written):

=========================================  ===================================
Command                                    Arguments
=========================================  ===================================
``Lengthen (overlap-add)...`` (Sound)      min pitch (Hz), max pitch (Hz), factor
``To Pitch (ac)...`` (Sound)               time step, pitch floor, max candidates,
                                           very accurate, silence thr., voicing thr.,
                                           octave cost, octave-jump cost,
                                           voiced/unvoiced cost, pitch ceiling
``Get quantile...`` (Pitch)                tmin, tmax, quantile, unit
``To Manipulation...`` (Sound)             time step, pitch floor, pitch ceiling
``Extract pitch tier`` (Manipulation)      -
``Get number of points`` (PitchTier)       -
``Get time from index...`` (PitchTier)     index
``Get value at index...`` (PitchTier)      index
``Remove points between...`` (PitchTier)   tmin, tmax
``Add point...`` (PitchTier)               time, frequency (Hz)
``Replace pitch tier``                     called on ``[manipulation, tier]``
``Get resynthesis (overlap-add)``          - (no trailing ellipsis; the
                                           ``...`` form is rejected)
=========================================  ===================================

``Lengthen (overlap-add)...`` and ``Get resynthesis (overlap-add)`` were both
measured to be sample-exact: output length is ``round(factor * n)`` and ``n``
respectively, which the time-map tests rely on.

They are **not** deterministic by default - overlap-add draws random numbers
for unvoiced material, and two identical calls on real speech were measured to
differ by up to 0.5 full scale on a few hundred samples. Every randomised Praat
command is therefore preceded by :func:`seed_praat`, without which
``CLAUDE.md`` invariant 13 cannot hold.

Defaults mirror ``configs/injection.yaml``; callers holding a
:class:`~rostrum.config.Config` should pass the resolved values rather than
rely on them, so one config edit moves the whole dataset (``CLAUDE.md`` coding
conventions). Parameter *names* here are the op-level ones; the
``injection.yaml`` key each corresponds to is named in the op's docstring.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

import numpy as np
import parselmouth
from parselmouth.praat import call
from parselmouth.praat import run as run_praat

from rostrum.inject.timemap import TimeMap
from rostrum.types import Alignment

__all__ = [
    "DEFAULT_CROSSFADE_MS",
    "DEFAULT_LOOP_CROSSFADE_MS",
    "DEFAULT_PRAAT_SEED",
    "DEFAULT_ROOM_TONE_MARGIN_MS",
    "DEFAULT_ROOM_TONE_MIN_GAP_S",
    "DEFAULT_ZERO_CROSSING_MS",
    "MANIPULATION_TIME_STEP_S",
    "MIN_KEPT_PAUSE_S",
    "InjectionError",
    "gain_ramp",
    "insert_roomtone",
    "pitch_track",
    "psola_lengthen",
    "psola_pitch_compress",
    "room_tone",
    "seed_praat",
    "shorten_gap",
    "splice",
    "stress_swap",
    "two_pass_pitch_range",
]

logger = logging.getLogger(__name__)

Params: TypeAlias = Mapping[str, float]
"""Resolved numeric parameters for one severity level."""

OpResult: TypeAlias = tuple[np.ndarray, TimeMap]
"""What every op returns: the edited audio and the exact old->new time map."""

# --------------------------------------------------------------------------- #
# Defaults, mirroring configs/injection.yaml
# --------------------------------------------------------------------------- #

DEFAULT_CROSSFADE_MS: float = 10.0
"""Raised-cosine crossfade at every splice join (``splice.crossfade_ms``)."""

DEFAULT_ZERO_CROSSING_MS: float = 2.0
"""How far a cut may be nudged to find a zero crossing
(``splice.zero_crossing_search_ms``)."""

DEFAULT_ROOM_TONE_MIN_GAP_S: float = 0.40
"""Shortest pause that may donate room tone (``room_tone.min_donor_gap_s``)."""

DEFAULT_ROOM_TONE_MARGIN_MS: float = 50.0
"""Trimmed off each end of the donor pause (``room_tone.donor_margin_ms``)."""

DEFAULT_LOOP_CROSSFADE_MS: float = 20.0
"""Crossfade between room-tone loop repetitions (``room_tone.loop_crossfade_ms``)."""

MIN_KEPT_PAUSE_S: float = 0.020
"""A shortened pause is never cut below 20 ms: a butt splice of two words
produces a click that a detector could trivially exploit."""

MANIPULATION_TIME_STEP_S: float = 0.01
"""``To Manipulation...`` time step; 10 ms, matching the frame hop
(``CLAUDE.md`` invariant 3)."""

DEFAULT_PRAAT_SEED: int = 1337
"""Fallback seed for Praat's global RNG, mirroring ``injection.yaml`` ``seed``."""

PITCH_ANALYSIS_MARGIN_S: float = 0.050
"""Context given to Praat around a span so pitch tracking has something to
latch onto. The margin is analysed but discarded, never spliced back."""

# Two-pass pitch range (``CLAUDE.md`` invariant 5, ``configs/features.yaml``).
PASS1_FLOOR_HZ: float = 50.0
PASS1_CEILING_HZ: float = 700.0
PASS2_Q1_SCALE: float = 0.75
PASS2_Q3_SCALE: float = 1.50
PASS2_MIN_FLOOR_HZ: float = 40.0
PASS2_MAX_CEILING_HZ: float = 800.0
PASS2_MIN_VOICED_FRAMES: int = 50
# Praat's autocorrelation window is 3 / floor long, so a clip shorter than that
# cannot be analysed at that floor at all. Above this floor the "speaker range"
# is a fiction (a 10 ms clip would need 300 Hz), so such a span is refused.
MAX_ANALYSABLE_FLOOR_HZ: float = 180.0

_PITCH_AC_ARGS: tuple[float, int, str, float, float, float, float, float] = (
    0.0,  # time step: 0 = Praat's default (ceiling-dependent)
    15,  # max candidates
    "no",  # very accurate
    0.03,  # silence threshold
    0.45,  # voicing threshold
    0.01,  # octave cost
    0.35,  # octave-jump cost
    0.14,  # voiced / unvoiced cost
)
"""``To Pitch (ac)...`` arguments other than floor and ceiling, from
``configs/features.yaml`` (``pitch.*``)."""

_SILENT = 1e-12
"""Floor for RMS/level arithmetic, so a digitally silent block cannot divide."""


class InjectionError(RuntimeError):
    """An op cannot be applied to this recording at this span.

    Raised rather than returning degraded audio: the generation plan (spec 5.4)
    selects a different span or drops the file, and a silently-weakened edit
    would be mislabelled at full severity.
    """


# --------------------------------------------------------------------------- #
# Array helpers
# --------------------------------------------------------------------------- #


def _as_mono(audio: np.ndarray) -> np.ndarray:
    """Return ``audio`` as a 1-D float64 array (Praat's native precision)."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=0) if x.shape[0] < x.shape[1] else x.mean(axis=1)
    if x.ndim != 1:
        raise InjectionError(f"expected mono audio, got shape {np.shape(audio)}")
    return x


def _rising_cosine(n: int) -> np.ndarray:
    """A raised-cosine ramp of ``n`` samples going 0 -> 1.

    This is the Hann half-window named by ``splice.crossfade_window``; paired
    with its complement it is a constant-amplitude (not constant-power)
    crossfade, which is the right choice for correlated material - the two
    sides of a join are the same voice saying the same thing.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.array([1.0])
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(n, dtype=np.float64) / (n - 1)))


def _ms_to_samples(ms: float, sr: int) -> int:
    """Convert milliseconds to a whole number of samples (round half up)."""
    return round(float(ms) * 1e-3 * sr)


def _rms(x: np.ndarray) -> float:
    """Root-mean-square level of ``x``; ``0.0`` for an empty array."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _snap_to_zero_crossing(x: np.ndarray, n: int, search: int) -> int:
    """Nudge sample index ``n`` to the nearest zero crossing within ``search``.

    Reduces the residual click a crossfade cannot fully hide. Returns ``n``
    unchanged when ``search <= 0`` or no crossing is in reach; the caller
    records whatever index comes back, so the time map stays exact either way.

    Args:
        x: The signal being cut.
        n: Candidate cut index.
        search: Half-width of the search, in samples.

    Returns:
        The chosen cut index, clamped to ``[0, len(x)]``.
    """
    if search <= 0 or x.size == 0:
        return int(min(max(n, 0), x.size))
    lo = int(max(n - search, 1))
    hi = int(min(n + search, x.size - 1))
    if hi <= lo:
        return int(min(max(n, 0), x.size))
    window = x[lo - 1 : hi]
    crossings = np.nonzero(np.signbit(window[:-1]) != np.signbit(window[1:]))[0]
    if crossings.size == 0:
        return int(min(max(n, 0), x.size))
    candidates = crossings + lo
    return int(candidates[np.argmin(np.abs(candidates - n))])


def splice(
    audio: np.ndarray,
    n0: int,
    n1: int,
    replacement: np.ndarray,
    *,
    fade: int,
) -> np.ndarray:
    """Replace samples ``[n0, n1)`` with ``replacement``, cross-faded in place.

    The join does **not** consume length, so the edited region occupies exactly
    ``len(replacement)`` samples and the caller's time map is exact. That is
    achieved by cross-fading against *displaced* clean material rather than
    overlapping the two pieces:

    * at the opening join the clean file's continuation ``audio[n0:n0+fade]``
      fades out while ``replacement[:fade]`` fades in;
    * at the closing join the end of ``replacement`` fades out while
      ``audio[n1-fade:n1]`` - the material that used to precede the tail -
      fades in, after which the untouched tail resumes.

    Both sides are therefore real speech or real room tone, never zeros
    (``CLAUDE.md`` invariant 8). With ``n0 == n1`` this is a pure insertion.

    Args:
        audio: Clean mono signal.
        n0: First sample of the region to replace.
        n1: One past the last sample of the region; ``n1 == n0`` inserts.
        replacement: New material for the region.
        fade: Crossfade length in samples; clamped to what is available.

    Returns:
        A new array of length ``len(audio) - (n1 - n0) + len(replacement)``.

    Raises:
        InjectionError: The region is not a valid half-open range of ``audio``.
    """
    x = _as_mono(audio)
    rep = _as_mono(replacement)
    n = x.size
    if not (0 <= n0 <= n1 <= n):
        raise InjectionError(f"invalid splice region [{n0}, {n1}) for {n} samples")

    out = np.concatenate([x[:n0], rep, x[n1:]])
    m = rep.size
    if m == 0 or fade <= 0:
        return out

    f1 = int(min(fade, m, n - n0))
    if f1 > 0:
        w = _rising_cosine(f1)
        out[n0 : n0 + f1] = x[n0 : n0 + f1] * (1.0 - w) + rep[:f1] * w

    f2 = int(min(fade, m - f1, n1))
    if f2 > 0:
        w = _rising_cosine(f2)
        end = n0 + m
        out[end - f2 : end] = rep[m - f2 :] * (1.0 - w) + x[n1 - f2 : n1] * w

    return out


# --------------------------------------------------------------------------- #
# Alignment -> sample indices
# --------------------------------------------------------------------------- #


def _word_span_samples(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: Sequence[int] | int,
    *,
    snap_ms: float,
) -> tuple[int, int, int, int]:
    """Resolve a word-index span to cut indices at word boundaries.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``(first_word_idx, last_word_idx)`` inclusive, or a single index.
        snap_ms: Zero-crossing search half-width in milliseconds.

    Returns:
        ``(n0, n1, i0, i1)``: the cut indices and the resolved word indices.

    Raises:
        InjectionError: The span is empty, out of range, or not ordered.
    """
    if isinstance(span, int):
        i0 = i1 = int(span)
    else:
        idx = [int(v) for v in span]
        if len(idx) != 2:
            raise InjectionError(f"word span must be (first, last), got {span!r}")
        i0, i1 = idx
    if not alignment.words:
        raise InjectionError("alignment has no words")
    if not (0 <= i0 <= i1 < len(alignment.words)):
        raise InjectionError(f"word span ({i0}, {i1}) outside 0..{len(alignment.words) - 1}")

    x = _as_mono(audio)
    search = _ms_to_samples(snap_ms, sr)
    n0 = _snap_to_zero_crossing(x, round(alignment.words[i0].t_start * sr), search)
    n1 = _snap_to_zero_crossing(x, round(alignment.words[i1].t_end * sr), search)
    if n1 <= n0:
        raise InjectionError(f"word span ({i0}, {i1}) resolved to an empty region")
    return n0, n1, i0, i1


def _gap(alignment: Alignment, gap_idx: int):
    """Return gap ``gap_idx``, looked up by its ``idx`` field.

    Gaps are keyed by ``Gap.idx`` rather than list position so a span chosen
    from a filtered candidate list still refers to the same silence.

    Raises:
        InjectionError: No gap carries that index.
    """
    for gap in alignment.gaps:
        if gap.idx == int(gap_idx):
            return gap
    raise InjectionError(f"no gap with idx {gap_idx} in {alignment.file_id!r}")


# --------------------------------------------------------------------------- #
# Pitch
# --------------------------------------------------------------------------- #


def _to_sound(x: np.ndarray, sr: int) -> parselmouth.Sound:
    """Wrap a mono float array as a Praat ``Sound`` starting at t = 0."""
    return parselmouth.Sound(_as_mono(x), sampling_frequency=float(sr))


def seed_praat(seed: int) -> None:
    """Pin Praat's global RNG so overlap-add resynthesis is bit-reproducible.

    This is not optional. ``Lengthen (overlap-add)`` draws random numbers for
    the unvoiced stretches of a span, so two identical calls on real speech
    return audio differing by up to ~0.5 full scale on a few hundred samples -
    measured, not assumed. That would break ``CLAUDE.md`` invariant 13 and the
    determinism test, and would mean a regenerated dataset did not match its
    published checksums.

    Praat exposes the seed as a scripting *function*, not a menu command, so it
    cannot be reached through :func:`parselmouth.praat.call`; the one-line
    script below is the route that works. The seed is global interpreter state,
    so it is set immediately before each randomised command rather than once at
    import.

    Verified on ``parselmouth`` 0.4.7: the same seed gives bit-identical
    output, a different seed gives different output.

    Args:
        seed: Non-negative integer seed.
    """
    run_praat(f"random_initializeWithSeedUnsafelyButPredictably({int(seed)})")


def _draw_praat_seed(params: Params, rng: np.random.Generator | None) -> int:
    """Pick the Praat seed for one op.

    Priority: an explicit ``praat_seed`` in ``params``, else a draw from the
    caller's seeded ``rng`` (so two spans in one file do not share a seed),
    else :data:`DEFAULT_PRAAT_SEED`.
    """
    if "praat_seed" in params:
        return int(params["praat_seed"])
    if rng is None:
        return DEFAULT_PRAAT_SEED
    return int(rng.integers(0, 2**31 - 1))


def min_analysable_floor_hz(n_samples: int, sr: int, *, margin: float = 1.02) -> float:
    """Lowest pitch floor Praat can use on a clip of ``n_samples`` samples.

    Praat's autocorrelation window is ``3 / floor`` seconds long and it refuses
    a sound shorter than one window, so the floor is bounded below by
    ``3 / duration``. ``margin`` keeps the result off the exact boundary, where
    float rounding decides whether Praat accepts the call.

    Args:
        n_samples: Length of the clip.
        sr: Sample rate.
        margin: Safety factor on the boundary value.

    Returns:
        The floor in Hz, or ``0.0`` for an empty clip.
    """
    if n_samples <= 0 or sr <= 0:
        return 0.0
    return margin * 3.0 * sr / float(n_samples)


def two_pass_pitch_range(
    audio: np.ndarray,
    sr: int,
    *,
    floor_hz: float = PASS1_FLOOR_HZ,
    ceiling_hz: float = PASS1_CEILING_HZ,
    q1_scale: float = PASS2_Q1_SCALE,
    q3_scale: float = PASS2_Q3_SCALE,
    min_floor_hz: float = PASS2_MIN_FLOOR_HZ,
    max_ceiling_hz: float = PASS2_MAX_CEILING_HZ,
    min_voiced_frames: int = PASS2_MIN_VOICED_FRAMES,
) -> tuple[float, float]:
    """Return the speaker-adapted Praat pitch range ``(floor, ceiling)``.

    The two-pass procedure of ``CLAUDE.md`` invariant 5: a wide first pass
    (50-700 Hz by default) measures the quartiles of the voiced F0, and the
    second range is ``0.75 * Q1`` to ``1.5 * Q3``, clamped to
    ``[min_floor_hz, max_ceiling_hz]``. A clip with too few voiced frames keeps
    the pass-1 range, because quartiles of a handful of frames are noise.

    Every op that calls Praat passes this range rather than Praat's defaults:
    TD-PSOLA segments the signal on pitch periods, so a wrong range is audible
    as doubling or halving artefacts in the edited span.

    Note:
        ``features/pitch.py`` does not exist yet. When it lands it should call
        this function rather than re-deriving the range, so the injector and
        the analyser agree on the speaker's range by construction.

    Args:
        audio: Mono float signal to measure.
        sr: Sample rate.
        floor_hz: Pass-1 floor.
        ceiling_hz: Pass-1 ceiling.
        q1_scale: Multiplier on Q1 giving the pass-2 floor.
        q3_scale: Multiplier on Q3 giving the pass-2 ceiling.
        min_floor_hz: Hard lower clamp on the returned floor.
        max_ceiling_hz: Hard upper clamp on the returned ceiling.
        min_voiced_frames: Below this many voiced frames, keep the pass-1 range.

    Returns:
        ``(floor_hz, ceiling_hz)`` for the second pass.
    """
    # A span shorter than Praat's 3 / floor analysis window makes "To Pitch (ac)"
    # raise rather than return, and pass 2 would then be estimated from nothing
    # anyway. Raise the floor to the shortest window the clip can carry, and
    # report that floor back: the caller passes it straight to Praat.
    floor_hz = max(float(floor_hz), min_analysable_floor_hz(_as_mono(audio).size, sr))
    if floor_hz >= float(ceiling_hz):
        return float(floor_hz), float(ceiling_hz)

    sound = _to_sound(audio, sr)
    pitch = call(
        sound,
        "To Pitch (ac)...",
        _PITCH_AC_ARGS[0],
        float(floor_hz),
        *_PITCH_AC_ARGS[1:],
        float(ceiling_hz),
    )
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced > 0.0]
    if voiced.size < int(min_voiced_frames):
        logger.debug(
            "pass-2 pitch range skipped: %d voiced frames < %d",
            voiced.size,
            min_voiced_frames,
        )
        return float(floor_hz), float(ceiling_hz)

    q1 = float(call(pitch, "Get quantile...", 0.0, 0.0, 0.25, "Hertz"))
    q3 = float(call(pitch, "Get quantile...", 0.0, 0.0, 0.75, "Hertz"))
    if not (math.isfinite(q1) and math.isfinite(q3)) or q1 <= 0.0 or q3 <= 0.0:
        return float(floor_hz), float(ceiling_hz)

    new_floor = max(float(min_floor_hz), q1_scale * q1, float(floor_hz))
    new_ceiling = min(float(max_ceiling_hz), q3_scale * q3)
    if new_ceiling <= new_floor * 1.5:
        # A degenerate range would make PSOLA worse, not better.
        return float(floor_hz), float(ceiling_hz)
    return new_floor, new_ceiling


def pitch_track(
    audio: np.ndarray,
    sr: int,
    floor_hz: float,
    ceiling_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(times, freqs)`` from Praat ``To Pitch (ac)...``.

    Unvoiced frames carry ``0.0`` in ``freqs``, as Praat reports them, so a
    caller can compute a voiced *fraction* and not merely a voiced count - the
    ``monotone`` span gate needs the fraction
    (``labels.monotone.valid_span.min_voiced_fraction``).

    The analysis arguments are the shared :data:`_PITCH_AC_ARGS`, so a span gate
    and the edit that follows it see the same contour.

    Args:
        audio: Mono signal.
        sr: Sample rate.
        floor_hz: Pitch floor, usually from :func:`two_pass_pitch_range`.
        ceiling_hz: Pitch ceiling.

    Returns:
        ``(times, freqs)``, both 1-D float arrays of one entry per frame.
    """
    pitch = call(
        _to_sound(audio, sr),
        "To Pitch (ac)...",
        _PITCH_AC_ARGS[0],
        float(floor_hz),
        *_PITCH_AC_ARGS[1:],
        float(ceiling_hz),
    )
    return (
        np.asarray(pitch.xs(), dtype=np.float64),
        np.asarray(pitch.selected_array["frequency"], dtype=np.float64),
    )


def _f0_semitone_std(audio: np.ndarray, sr: int, floor_hz: float, ceiling_hz: float) -> float:
    """Standard deviation of voiced F0 in semitones re the clip's own median.

    Semitones re the median, never Hz: a Hz-domain spread is sex-biased
    (``configs/features.yaml``, ``pitch_variation.method``).
    """
    sound = _to_sound(audio, sr)
    pitch = call(
        sound,
        "To Pitch (ac)...",
        _PITCH_AC_ARGS[0],
        float(floor_hz),
        *_PITCH_AC_ARGS[1:],
        float(ceiling_hz),
    )
    voiced = pitch.selected_array["frequency"]
    voiced = voiced[voiced > 0.0]
    if voiced.size < 2:
        return 0.0
    median = float(np.median(voiced))
    if median <= 0.0:
        return 0.0
    return float(np.std(12.0 * np.log2(voiced / median)))


def _transform_pitch_tier(
    chunk: np.ndarray,
    sr: int,
    transform,
    *,
    floor_hz: float,
    ceiling_hz: float,
    region: tuple[float, float] | None = None,
    seed: int = DEFAULT_PRAAT_SEED,
) -> np.ndarray:
    """Resynthesise ``chunk`` with every pitch-tier point passed through ``transform``.

    The Praat route is ``To Manipulation...`` -> ``Extract pitch tier`` ->
    rewrite the points -> ``Replace pitch tier`` -> ``Get resynthesis
    (overlap-add)``, i.e. TD-PSOLA, never a vocoder (``CLAUDE.md`` invariant 8).
    Resynthesis was measured to return exactly as many samples as it was given,
    so this function is length-preserving and the caller's time map is
    unaffected.

    Args:
        chunk: Mono float audio to resynthesise.
        sr: Sample rate.
        transform: ``f(hz) -> hz`` applied to each pitch-tier point.
        floor_hz: Manipulation pitch floor.
        ceiling_hz: Manipulation pitch ceiling.
        region: Optional ``(t0, t1)`` in chunk-local seconds; points outside it
            are left untouched, which keeps an analysis margin unmodified and
            the pitch contour continuous at the region edge.
        seed: Praat RNG seed; see :func:`seed_praat`.

    Returns:
        The resynthesised signal, same length as ``chunk``.

    Raises:
        InjectionError: Praat found no pitch points to manipulate.
    """
    x = _as_mono(chunk)
    sound = _to_sound(x, sr)
    manip = call(
        sound, "To Manipulation...", MANIPULATION_TIME_STEP_S, float(floor_hz), float(ceiling_hz)
    )
    tier = call(manip, "Extract pitch tier")
    n_points = int(call(tier, "Get number of points"))
    if n_points <= 0:
        raise InjectionError("no pitch-tier points: the span is unvoiced")

    points = [
        (
            float(call(tier, "Get time from index...", i)),
            float(call(tier, "Get value at index...", i)),
        )
        for i in range(1, n_points + 1)
    ]
    t0, t1 = region if region is not None else (-math.inf, math.inf)
    new_points = [(t, float(transform(f)) if t0 <= t <= t1 else f) for t, f in points if f > 0.0]
    if not new_points:
        raise InjectionError("no voiced pitch-tier points in the span")

    call(tier, "Remove points between...", sound.xmin, sound.xmax)
    for t, f in new_points:
        call(tier, "Add point...", t, f)
    call([manip, tier], "Replace pitch tier")
    seed_praat(seed)
    resynth = call(manip, "Get resynthesis (overlap-add)")

    y = np.asarray(resynth.values[0], dtype=np.float64)
    if y.size != x.size:  # defensive: measured exact, but do not trust silently
        logger.warning("resynthesis returned %d samples for %d; trimming", y.size, x.size)
        y = np.resize(y, x.size) if y.size < x.size else y[: x.size]
    return y


def _span_with_margin(
    x: np.ndarray, sr: int, n0: int, n1: int, margin_s: float
) -> tuple[np.ndarray, int, float, float]:
    """Extract ``[n0, n1)`` plus analysis margin.

    Returns:
        ``(chunk, offset, t0_local, t1_local)`` where ``offset`` is the sample
        index in ``x`` at which ``chunk`` begins and the local times bracket
        the real span inside ``chunk``.
    """
    margin = _ms_to_samples(margin_s * 1e3, sr)
    a = int(max(0, n0 - margin))
    b = int(min(x.size, n1 + margin))
    return x[a:b].copy(), a, (n0 - a) / sr, (n1 - a) / sr


# --------------------------------------------------------------------------- #
# Room tone
# --------------------------------------------------------------------------- #


def room_tone(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    n_samples: int,
    *,
    min_donor_gap_s: float = DEFAULT_ROOM_TONE_MIN_GAP_S,
    donor_margin_ms: float = DEFAULT_ROOM_TONE_MARGIN_MS,
    loop_crossfade_ms: float = DEFAULT_LOOP_CROSSFADE_MS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesise ``n_samples`` of this recording's own noise floor.

    The donor is the **longest** pause in the alignment, trimmed by
    ``donor_margin_ms`` at each end so no breath or word edge leaks in. Shorter
    than the request, it is looped with ``loop_crossfade_ms`` raised-cosine
    joins, and the result's RMS is restored to the donor's - crossfading
    correlated noise loses a little energy, and the inserted silence has to sit
    at the same level as the surrounding silence to be believable.

    Inserted silence is never digital zeros (``CLAUDE.md`` invariant 8): an
    exact-zero block is a giveaway a detector could learn instead of learning
    prosody.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment, used to find pauses.
        n_samples: How much room tone to produce.
        min_donor_gap_s: Shortest pause allowed to donate.
        donor_margin_ms: Trim at each end of the donor pause.
        loop_crossfade_ms: Crossfade between loop repetitions.
        rng: Seeded generator; used only to pick the loop start phase, so that
            repeated insertions in one file are not bit-identical.

    Returns:
        ``n_samples`` of looped room tone, or an empty array if
        ``n_samples <= 0``.

    Raises:
        InjectionError: No pause is long enough to donate room tone.
    """
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    x = _as_mono(audio)
    margin = _ms_to_samples(donor_margin_ms, sr)

    donors: list[tuple[int, int]] = []
    for gap in alignment.gaps:
        if gap.duration < min_donor_gap_s:
            continue
        a = round(gap.t_start * sr) + margin
        b = round(gap.t_end * sr) - margin
        if b - a > 2 * _ms_to_samples(loop_crossfade_ms, sr):
            donors.append((max(0, a), min(x.size, b)))
    if not donors:
        raise InjectionError(
            f"{alignment.file_id!r} has no pause of at least {min_donor_gap_s}s "
            f"(plus {donor_margin_ms}ms margins) to harvest room tone from"
        )

    a, b = max(donors, key=lambda ab: ab[1] - ab[0])
    donor = x[a:b]
    target_rms = _rms(donor)

    xf = int(min(_ms_to_samples(loop_crossfade_ms, sr), donor.size // 2))
    phase = 0 if rng is None else int(rng.integers(0, donor.size))
    tone = _loop_with_crossfade(np.roll(donor, -phase), n_samples, xf)

    tone_rms = _rms(tone)
    if tone_rms > _SILENT and target_rms > _SILENT:
        tone *= target_rms / tone_rms
    return tone


def _loop_with_crossfade(donor: np.ndarray, n_samples: int, xf: int) -> np.ndarray:
    """Tile ``donor`` to ``n_samples``, cross-fading ``xf`` samples per seam."""
    donor = _as_mono(donor)
    if donor.size == 0:
        raise InjectionError("empty room-tone donor")
    if n_samples <= donor.size:
        return donor[:n_samples].copy()

    xf = int(max(0, min(xf, donor.size // 2)))
    out = donor.copy()
    w = _rising_cosine(xf)
    while out.size < n_samples:
        if xf > 0:
            seam = out[-xf:] * (1.0 - w) + donor[:xf] * w
            out = np.concatenate([out[:-xf], seam, donor[xf:]])
        else:
            out = np.concatenate([out, donor])
    return out[:n_samples]


# --------------------------------------------------------------------------- #
# 1. Pace: rushing / dragging
# --------------------------------------------------------------------------- #


def psola_lengthen(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: Sequence[int] | int,
    params: Params,
    rng: np.random.Generator | None = None,
) -> OpResult:
    """Time-scale a span of words with Praat TD-PSOLA (spec 5.2, pace).

    ``factor < 1`` produces ``rushing`` and ``factor > 1`` produces
    ``dragging``; ``injection.yaml`` derives it from severity as
    ``1 - 0.45 * s`` and ``1 + 0.8 * s`` respectively. Pitch is preserved - the
    speaker gets faster, not higher - because ``Lengthen (overlap-add)`` is
    given the speaker's own two-pass pitch range.

    The cut is made at the first word's onset and the last word's offset, so
    the surrounding pauses keep their clean duration: this changes articulation
    rate, not the pause structure.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``(first_word_idx, last_word_idx)`` inclusive, or one word index.
        params: ``factor`` (``labels.rushing.params.factor``); optional
            ``crossfade_ms``, ``snap_ms``.
        rng: Seeded generator; one draw supplies Praat's RNG seed, which
            :func:`seed_praat` explains is required for reproducibility.
            Pass ``praat_seed`` in ``params`` to fix it explicitly.

    Returns:
        ``(new_audio, TimeMap)``. The span occupies ``factor`` times its clean
        duration to within half a sample, and every later timestamp shifts by
        exactly ``TimeMap.delta``.

    Raises:
        InjectionError: ``factor`` is not positive, or the span is invalid.
    """
    factor = float(params["factor"])
    if factor <= 0.0:
        raise InjectionError(f"lengthen factor must be positive, got {factor}")
    fade = _ms_to_samples(float(params.get("crossfade_ms", DEFAULT_CROSSFADE_MS)), sr)
    snap_ms = float(params.get("snap_ms", DEFAULT_ZERO_CROSSING_MS))

    x = _as_mono(audio)
    n0, n1, _, _ = _word_span_samples(x, sr, alignment, span, snap_ms=snap_ms)
    stretched = _psola_time_scale(x[n0:n1], sr, factor, _draw_praat_seed(params, rng))

    out = splice(x, n0, n1, stretched, fade=fade)
    tmap = TimeMap.replacement(
        x.size / sr, n0 / sr, n1 / sr, stretched.size / sr, op="psola_lengthen"
    )
    logger.debug(
        "psola_lengthen factor=%.3f span=[%.3f, %.3f]s delta=%+.4fs",
        factor,
        n0 / sr,
        n1 / sr,
        tmap.delta,
    )
    return out.astype(np.float64, copy=False), tmap


def _psola_time_scale(chunk: np.ndarray, sr: int, factor: float, seed: int) -> np.ndarray:
    """Scale ``chunk`` in time by ``factor`` with ``Lengthen (overlap-add)...``.

    Praat returns ``round(factor * n)`` samples exactly. A factor of 1 is
    short-circuited so an L0-ish edit is bit-identical rather than merely
    close. ``seed`` pins Praat's RNG - see :func:`seed_praat` for why that
    is mandatory here.
    """
    x = _as_mono(chunk)
    if abs(factor - 1.0) < 1e-12:
        return x.copy()
    floor_hz, ceiling_hz = two_pass_pitch_range(x, sr)
    if floor_hz > MAX_ANALYSABLE_FLOOR_HZ:
        raise InjectionError(
            f"span is {1000.0 * x.size / sr:.0f} ms, too short for TD-PSOLA: it would need a "
            f"{floor_hz:.0f} Hz pitch floor (max {MAX_ANALYSABLE_FLOOR_HZ:.0f} Hz)"
        )
    sound = _to_sound(x, sr)
    seed_praat(seed)
    out = call(
        sound, "Lengthen (overlap-add)...", float(floor_hz), float(ceiling_hz), float(factor)
    )
    return np.asarray(out.values[0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 2. Pausing: missing_pause
# --------------------------------------------------------------------------- #


def shorten_gap(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: int,
    params: Params,
    rng: np.random.Generator | None = None,
) -> OpResult:
    """Cut a pause down to ``keep_frac`` of its length (spec 5.2, ``missing_pause``).

    The **first** ``keep_frac`` of the silence is kept, not the middle or a
    resampled version: the trailing edge of a pause carries the onset of the
    next word's breath, and keeping the head leaves the previous word's offset
    untouched. The pause is never cut below :data:`MIN_KEPT_PAUSE_S`, so the
    two words never butt-splice into a click.

    Spec 5.2 restricts this to a *boundary* pause of at least 300 ms in the
    clean file; selecting such a gap is the engine's job
    (``labels.missing_pause.valid_span``), not this op's.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``Gap.idx`` of the pause to shorten.
        params: ``keep_frac`` (``labels.missing_pause.params.keep_fraction``);
            optional ``crossfade_ms``, ``min_keep_s``.
        rng: Unused; present so every op has one signature.

    Returns:
        ``(new_audio, TimeMap)``.

    Raises:
        InjectionError: ``keep_frac`` is outside ``[0, 1]``, or the gap is
            already at or below the floor so there is nothing to remove.
    """
    del rng
    keep_frac = float(params["keep_frac"])
    if not 0.0 <= keep_frac <= 1.0:
        raise InjectionError(f"keep_frac must be in [0, 1], got {keep_frac}")
    fade = _ms_to_samples(float(params.get("crossfade_ms", DEFAULT_CROSSFADE_MS)), sr)
    min_keep_s = float(params.get("min_keep_s", MIN_KEPT_PAUSE_S))

    x = _as_mono(audio)
    gap = _gap(alignment, span)
    n0 = round(gap.t_start * sr)
    n1 = int(min(round(gap.t_end * sr), x.size))
    old_len = n1 - n0
    if old_len <= 0:
        raise InjectionError(f"gap {span} has no duration in {alignment.file_id!r}")

    keep = int(max(round(keep_frac * old_len), _ms_to_samples(min_keep_s * 1e3, sr)))
    keep = min(keep, old_len)
    if keep >= old_len:
        raise InjectionError(
            f"gap {span} ({old_len / sr:.3f}s) cannot be shortened to "
            f"{keep_frac:.3f} without falling below the {min_keep_s * 1e3:.0f}ms floor"
        )

    out = splice(x, n0, n1, x[n0 : n0 + keep], fade=fade)
    tmap = TimeMap.replacement(x.size / sr, n0 / sr, n1 / sr, keep / sr, op="shorten_gap")
    logger.debug(
        "shorten_gap idx=%d %.3fs -> %.3fs (keep_frac=%.3f)",
        gap.idx,
        old_len / sr,
        keep / sr,
        keep_frac,
    )
    return out, tmap


# --------------------------------------------------------------------------- #
# 3. Pausing: hesitation_pause
# --------------------------------------------------------------------------- #


def insert_roomtone(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: int,
    params: Params,
    rng: np.random.Generator | None = None,
) -> OpResult:
    """Insert room tone into a gap, making a hesitation (spec 5.2).

    The insertion point is the gap's **midpoint**, so whatever short silence
    already existed is split rather than displaced and both neighbouring words
    keep their own offsets and onsets. ``injection.yaml`` derives the duration
    from severity as ``0.3 + 1.7 * s`` and restricts the op to non-boundary
    gaps - a long silence at a clause boundary is good rhetoric, not a flaw.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``Gap.idx`` of the gap to lengthen.
        params: ``dur_s`` (``labels.hesitation_pause.params.duration_s``);
            optional ``crossfade_ms`` and the :func:`room_tone` keys
            ``min_donor_gap_s``, ``donor_margin_ms``, ``loop_crossfade_ms``.
        rng: Seeded generator, passed to :func:`room_tone` for the loop phase.

    Returns:
        ``(new_audio, TimeMap)``. The new duration exceeds the old by exactly
        ``round(dur_s * sr)`` samples, and the map records a zero-old-width
        ``insert`` segment at the midpoint.

    Raises:
        InjectionError: ``dur_s`` is not positive, or no pause can donate
            room tone.
    """
    dur_s = float(params["dur_s"])
    if dur_s <= 0.0:
        raise InjectionError(f"insertion duration must be positive, got {dur_s}")
    fade = _ms_to_samples(float(params.get("crossfade_ms", DEFAULT_CROSSFADE_MS)), sr)

    x = _as_mono(audio)
    gap = _gap(alignment, span)
    at = int(min(max(round(0.5 * (gap.t_start + gap.t_end) * sr), 0), x.size))
    n_insert = _ms_to_samples(dur_s * 1e3, sr)

    tone = room_tone(
        x,
        sr,
        alignment,
        n_insert,
        min_donor_gap_s=float(params.get("min_donor_gap_s", DEFAULT_ROOM_TONE_MIN_GAP_S)),
        donor_margin_ms=float(params.get("donor_margin_ms", DEFAULT_ROOM_TONE_MARGIN_MS)),
        loop_crossfade_ms=float(params.get("loop_crossfade_ms", DEFAULT_LOOP_CROSSFADE_MS)),
        rng=rng,
    )
    out = splice(x, at, at, tone, fade=fade)
    tmap = TimeMap.replacement(
        x.size / sr, at / sr, at / sr, n_insert / sr, op="insert_roomtone", kind="insert"
    )
    logger.debug("insert_roomtone gap=%d at=%.3fs dur=%.3fs", gap.idx, at / sr, n_insert / sr)
    return out, tmap


# --------------------------------------------------------------------------- #
# 4. Intonation: monotone
# --------------------------------------------------------------------------- #


def psola_pitch_compress(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: Sequence[int] | int,
    params: Params,
    rng: np.random.Generator | None = None,
) -> OpResult:
    """Flatten the F0 contour of a phrase towards its median (spec 5.2, ``monotone``).

    Every pitch-tier point in the span becomes ``f' = m * (f / m) ** (1 - alpha)``
    with ``m`` the span's median F0 in Hz. In semitones re ``m`` that is exactly
    a multiplication by ``1 - alpha``, so ``alpha`` *is* the fraction of pitch
    range removed and the semitone standard deviation of the span drops by the
    same fraction. ``injection.yaml`` stores the exponent instead
    (``exponent = 1 - 0.9 * s``, so ``alpha = 0.9 * s``).

    Only the span is replaced; the rest of the recording is untouched, and the
    op is length-preserving, so the time map is the identity.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``(first_word_idx, last_word_idx)`` inclusive - one phrase.
        params: ``alpha`` in ``[0, 1]`` (or ``exponent = 1 - alpha``); optional
            ``crossfade_ms``, ``snap_ms``, ``margin_s``.
        rng: Seeded generator; one draw supplies Praat's RNG seed, which
            :func:`seed_praat` explains is required for reproducibility.
            Pass ``praat_seed`` in ``params`` to fix it explicitly.

    Returns:
        ``(new_audio, TimeMap)`` with an identity time map.

    Raises:
        InjectionError: ``alpha`` is outside ``[0, 1]``, or the span is
            unvoiced so there is no contour to flatten.
    """
    praat_seed = _draw_praat_seed(params, rng)
    # `alpha` is the op-level knob; `exponent` is how injection.yaml stores it.
    alpha = float(params["alpha"]) if "alpha" in params else 1.0 - float(params["exponent"])
    if not 0.0 <= alpha <= 1.0:
        raise InjectionError(f"alpha must be in [0, 1], got {alpha}")
    exponent = 1.0 - alpha
    fade = _ms_to_samples(float(params.get("crossfade_ms", DEFAULT_CROSSFADE_MS)), sr)
    snap_ms = float(params.get("snap_ms", DEFAULT_ZERO_CROSSING_MS))
    margin_s = float(params.get("margin_s", PITCH_ANALYSIS_MARGIN_S))

    x = _as_mono(audio)
    n0, n1, _, _ = _word_span_samples(x, sr, alignment, span, snap_ms=snap_ms)
    chunk, offset, t0, t1 = _span_with_margin(x, sr, n0, n1, margin_s)

    floor_hz, ceiling_hz = two_pass_pitch_range(chunk, sr)
    median_hz = _span_median_f0(chunk, sr, floor_hz, ceiling_hz, (t0, t1))

    def flatten(f: float) -> float:
        return median_hz * (f / median_hz) ** exponent

    processed = _transform_pitch_tier(
        chunk,
        sr,
        flatten,
        floor_hz=floor_hz,
        ceiling_hz=ceiling_hz,
        region=(t0, t1),
        seed=praat_seed,
    )
    replacement = processed[n0 - offset : n1 - offset]
    out = splice(x, n0, n1, replacement, fade=fade)
    tmap = TimeMap.identity(x.size / sr, op="psola_pitch_compress")
    logger.debug(
        "psola_pitch_compress alpha=%.3f span=[%.3f, %.3f]s median=%.1fHz",
        alpha,
        n0 / sr,
        n1 / sr,
        median_hz,
    )
    return out, tmap


def _span_median_f0(
    chunk: np.ndarray,
    sr: int,
    floor_hz: float,
    ceiling_hz: float,
    region: tuple[float, float],
) -> float:
    """Median voiced F0 in Hz over ``region`` of ``chunk``.

    Raises:
        InjectionError: The region has no voiced frames.
    """
    sound = _to_sound(chunk, sr)
    pitch = call(
        sound,
        "To Pitch (ac)...",
        _PITCH_AC_ARGS[0],
        float(floor_hz),
        *_PITCH_AC_ARGS[1:],
        float(ceiling_hz),
    )
    freqs = pitch.selected_array["frequency"]
    times = np.asarray(pitch.xs(), dtype=np.float64)
    inside = (times >= region[0]) & (times <= region[1]) & (freqs > 0.0)
    if not inside.any():
        raise InjectionError(f"no voiced frames in span {region} - nothing to flatten")
    return float(np.median(freqs[inside]))


# --------------------------------------------------------------------------- #
# 5. Emphasis: misplaced_stress
# --------------------------------------------------------------------------- #


def stress_swap(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: Sequence[int],
    params: Params,
    rng: np.random.Generator | None = None,
) -> OpResult:
    """Move prominence from a content word onto a function word (spec 5.2).

    All three prominence cues are moved together, because moving one alone is
    not what a misplaced stress sounds like:

    ============  ================  ==================
    Cue           Function word     Content word
    ============  ================  ==================
    F0            ``+4s`` st        ``-4s`` st
    Level         ``+4s`` dB        ``-4s`` dB
    Duration      ``x(1 + 0.3s)``   ``x(1 - 0.2s)``
    ============  ================  ==================

    The two lengthenings are separate splices, so their time maps are composed
    (:meth:`TimeMap.then`) and the returned map is still in clean-file time.
    The lengthenings are applied **right to left** so the earlier edit cannot
    invalidate the later word's sample indices before they are used.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``(function_word_idx, content_word_idx)``; the two need not be in
            textual order.
        params: ``s`` in ``[0, 1]``; or the explicit
            ``f0_shift_st`` / ``gain_db`` / ``duration_factor`` from
            ``labels.misplaced_stress.params``. Optional ``crossfade_ms``,
            ``snap_ms``, ``margin_s``, ``content_duration_factor``.
        rng: Seeded generator; one draw supplies Praat's RNG seed, which
            :func:`seed_praat` explains is required for reproducibility.
            Pass ``praat_seed`` in ``params`` to fix it explicitly.

    Returns:
        ``(new_audio, TimeMap)``.

    Raises:
        InjectionError: The word pair is invalid or identical, or either word
            is unvoiced so its F0 cannot be shifted.
    """
    praat_seed = _draw_praat_seed(params, rng)
    idx = [int(v) for v in span]
    if len(idx) != 2 or idx[0] == idx[1]:
        raise InjectionError(f"stress_swap needs two distinct word indices, got {span!r}")
    fn_idx, ct_idx = idx

    s = float(params.get("s", 0.0))
    f0_st = float(params.get("f0_shift_st", 4.0 * s))
    gain_db = float(params.get("gain_db", 4.0 * s))
    fn_factor = float(params.get("duration_factor", 1.0 + 0.3 * s))
    ct_factor = float(params.get("content_duration_factor", 1.0 - 0.2 * s))
    if ct_factor <= 0.0:
        raise InjectionError(f"content-word duration factor must be positive, got {ct_factor}")

    fade = _ms_to_samples(float(params.get("crossfade_ms", DEFAULT_CROSSFADE_MS)), sr)
    snap_ms = float(params.get("snap_ms", DEFAULT_ZERO_CROSSING_MS))
    margin_s = float(params.get("margin_s", PITCH_ANALYSIS_MARGIN_S))

    x = _as_mono(audio)
    # Step 1: pitch and level, both length-preserving, so no time map yet.
    for word_idx, st, db in ((fn_idx, +f0_st, +gain_db), (ct_idx, -f0_st, -gain_db)):
        x = _shift_word_pitch(
            x,
            sr,
            alignment,
            word_idx,
            st,
            fade=fade,
            snap_ms=snap_ms,
            margin_s=margin_s,
            seed=praat_seed,
        )
        x = _gain_word(x, sr, alignment, word_idx, db, fade=fade, snap_ms=snap_ms)

    # Step 2: duration, right to left, composing the two maps.
    edits = sorted(((fn_idx, fn_factor), (ct_idx, ct_factor)), key=lambda p: p[0], reverse=True)
    tmap: TimeMap | None = None
    for word_idx, factor in edits:
        x, step = psola_lengthen(
            x,
            sr,
            alignment,
            word_idx,
            {
                "factor": factor,
                "crossfade_ms": fade * 1e3 / sr,
                "snap_ms": snap_ms,
                "praat_seed": praat_seed,
            },
        )
        # The later word is edited first, in clean time; the earlier word is
        # then edited on the timeline that produced, so the composition runs
        # clean -> after-later -> after-earlier in that order.
        tmap = step if tmap is None else tmap.then(step)

    assert tmap is not None  # two edits always run
    logger.debug(
        "stress_swap fn=%d ct=%d f0=%+.2fst gain=%+.2fdB dur=%.3f/%.3f delta=%+.4fs",
        fn_idx,
        ct_idx,
        f0_st,
        gain_db,
        fn_factor,
        ct_factor,
        tmap.delta,
    )
    return x, TimeMap.from_segments(tmap.segments, op="stress_swap")


def _shift_word_pitch(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    word_idx: int,
    semitones: float,
    *,
    fade: int,
    snap_ms: float,
    margin_s: float,
    seed: int = DEFAULT_PRAAT_SEED,
) -> np.ndarray:
    """Multiply one word's pitch tier by ``2 ** (semitones / 12)``.

    Length-preserving: the resynthesised word is spliced back over itself.
    """
    if abs(semitones) < 1e-12:
        return audio
    x = _as_mono(audio)
    n0, n1, _, _ = _word_span_samples(x, sr, alignment, word_idx, snap_ms=snap_ms)
    chunk, offset, t0, t1 = _span_with_margin(x, sr, n0, n1, margin_s)
    floor_hz, ceiling_hz = two_pass_pitch_range(chunk, sr)
    ratio = 2.0 ** (float(semitones) / 12.0)
    processed = _transform_pitch_tier(
        chunk,
        sr,
        lambda f: f * ratio,
        floor_hz=floor_hz,
        ceiling_hz=ceiling_hz * max(ratio, 1.0),
        region=(t0, t1),
        seed=seed,
    )
    return splice(x, n0, n1, processed[n0 - offset : n1 - offset], fade=fade)


def _gain_word(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    word_idx: int,
    db: float,
    *,
    fade: int,
    snap_ms: float,
) -> np.ndarray:
    """Apply a flat ``db`` gain to one word, tapered at its edges.

    The taper is the splice crossfade itself: the word is scaled and spliced
    back over its own position, so the level change ramps in and out over 10 ms
    instead of stepping.
    """
    if abs(db) < 1e-12:
        return audio
    x = _as_mono(audio)
    n0, n1, _, _ = _word_span_samples(x, sr, alignment, word_idx, snap_ms=snap_ms)
    return splice(x, n0, n1, x[n0:n1] * (10.0 ** (float(db) / 20.0)), fade=fade)


# --------------------------------------------------------------------------- #
# 6. Energy: low_energy / fading
# --------------------------------------------------------------------------- #


def gain_ramp(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    span: Sequence[int] | int,
    params: Params,
    rng: np.random.Generator | None = None,
) -> OpResult:
    """Fade a span from 0 dB down (or up) to ``db_end`` (spec 5.2, energy).

    The ramp is a raised cosine **in the dB domain**: it leaves 0 dB with zero
    slope, so the fade has no audible onset, and arrives at exactly ``db_end``
    on the span's last sample. ``injection.yaml`` derives it from severity as
    ``-18 * s`` dB and prefers phrase-final spans, where trailing off is what
    speakers actually do.

    Pass ``ramp="linear_db"`` for the straight-line interpolation named in
    ``labels.low_energy.params.ramp``; both reach the same end level.

    No samples are added or removed and the ramp starts at unity, so no splice
    is needed and the time map is the identity.

    Args:
        audio: Clean mono signal.
        sr: Sample rate.
        alignment: Forced alignment of the clean recording.
        span: ``(first_word_idx, last_word_idx)`` inclusive, or one word index.
        params: ``db_end`` (``labels.low_energy.params.end_gain_db``); optional
            ``ramp`` (``"raised_cosine"``, the default, or ``"linear_db"``),
            ``snap_ms``, and ``hold`` - when true the post-span audio keeps the
            end gain instead of returning to 0 dB.
        rng: Unused; present so every op has one signature.

    Returns:
        ``(new_audio, TimeMap)`` with an identity time map.

    Raises:
        InjectionError: ``ramp`` is not a known shape, or the span is invalid.
    """
    del rng
    db_end = float(params["db_end"])
    shape = str(params.get("ramp", "raised_cosine"))
    snap_ms = float(params.get("snap_ms", DEFAULT_ZERO_CROSSING_MS))
    hold = bool(params.get("hold", False))

    x = _as_mono(audio)
    n0, n1, _, _ = _word_span_samples(x, sr, alignment, span, snap_ms=snap_ms)
    n = n1 - n0

    if shape == "raised_cosine":
        progress = _rising_cosine(n)
    elif shape == "linear_db":
        progress = np.linspace(0.0, 1.0, n) if n > 1 else np.ones(1)
    else:
        raise InjectionError(f"unknown ramp shape {shape!r}")

    out = x.copy()
    out[n0:n1] *= np.power(10.0, (db_end * progress) / 20.0)
    if hold:
        out[n1:] *= 10.0 ** (db_end / 20.0)

    tmap = TimeMap.identity(x.size / sr, op="gain_ramp")
    logger.debug("gain_ramp %s to %+.2fdB over [%.3f, %.3f]s", shape, db_end, n0 / sr, n1 / sr)
    return out, tmap
