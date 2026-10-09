"""Pause features: alignment gaps refined by Silero VAD (spec 9.1 PAU).

A gap is the silence between two aligned words. Two things make it a *feature*
rather than a number:

* **Where it really starts and ends.** The aligner places a word boundary to fit
  its acoustic model; a final stop release or a breath can land on either side
  of it. Silero VAD, run on the same frame grid, tightens each gap to the
  silence actually heard - but never by more than ``max_shift_s``, because a
  large disagreement means the VAD and the aligner are describing different
  things and the alignment is the one the word indices rest on.
* **What was expected there.** ``Gap.label`` (``boundary`` / ``minor`` /
  ``none``) comes from the display transcript's punctuation, via
  :mod:`rostrum.text`, and is applied here with :func:`apply_gap_labels`. The
  same 700 ms silence is good rhetoric after a full stop and a hesitation in the
  middle of a clause; without the label a pause cannot be judged at all.

Only :func:`speech_probs` touches a model; everything else is a pure function
over times and intervals, so the refinement logic is tested without Silero.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from rostrum.features.frames import FrameGrid
from rostrum.types import Gap

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable, GapLabel, Word

logger = logging.getLogger(__name__)

__all__ = [
    "GAP_LABELS",
    "PAUSE_GAP_COLUMNS",
    "apply_gap_labels",
    "gap_features",
    "gaps_from_words",
    "refine_gaps",
    "silence_containing",
    "speech_intervals",
    "speech_probs",
]

PAUSE_GAP_COLUMNS: tuple[str, ...] = (
    "gap_duration_s",
    "gap_label",
    "gap_is_boundary",
    "gap_is_pause",
    "gap_vad_refined",
    "gap_prev_word_idx",
    "gap_next_word_idx",
)
"""Gap-level columns (spec 9.1: gap duration + gap label + boundary flag)."""

GAP_LABELS: frozenset[str] = frozenset({"boundary", "minor", "none"})
"""The three text-derived gap expectations (spec 7.1)."""

DEFAULT_MIN_PAUSE_S = 0.250
"""Below this a gap is co-articulation, not a pause (fluency convention)."""

DEFAULT_MAX_SHIFT_S = 0.100
"""How far VAD may move an aligner boundary before the move is rejected."""

VAD_WINDOW_SAMPLES = 512
"""Silero's 16 kHz frame size; the model requires exactly this many samples."""


# --------------------------------------------------------------------------- #
# Gaps from the alignment
# --------------------------------------------------------------------------- #


def gaps_from_words(words: Sequence[Word]) -> list[Gap]:
    """Build the inter-word gaps of an alignment.

    One gap per adjacent pair, in word order, including zero-length ones so gap
    ``i`` always follows word ``i``. Leading and trailing silence is excluded:
    neither is a pause the speaker chose to make inside the text.
    """
    return [
        Gap(
            idx=i,
            prev_word_idx=words[i].idx,
            next_word_idx=words[i + 1].idx,
            t_start=float(words[i].t_end),
            t_end=float(words[i + 1].t_start),
        )
        for i in range(len(words) - 1)
    ]


def apply_gap_labels(gaps: Sequence[Gap], labels: Mapping[int, tuple[str, str]]) -> list[Gap]:
    """Attach text-derived labels to gaps (spec 7.1).

    Args:
        gaps: Gaps from the alignment.
        labels: Gap index to ``(label, punctuation)``, as produced by
            :mod:`rostrum.text` from the display transcript. A gap missing from
            the mapping keeps its default ``"none"``.

    Returns:
        New gaps; the inputs are not mutated.

    Raises:
        ValueError: A label is not one of :data:`GAP_LABELS`.
    """
    unknown = {lab for lab, _ in labels.values()} - GAP_LABELS
    if unknown:
        raise ValueError(
            f"unknown gap label(s) {sorted(unknown)}; expected one of {sorted(GAP_LABELS)}"
        )
    out = []
    for gap in gaps:
        label, punctuation = labels.get(gap.idx, ("none", ""))
        out.append(replace(gap, label=label, punctuation=punctuation))  # type: ignore[arg-type]
    return out


# --------------------------------------------------------------------------- #
# Silero VAD
# --------------------------------------------------------------------------- #


def speech_probs(
    samples: np.ndarray,
    sr: int,
    *,
    grid: FrameGrid | None = None,
    model: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Silero speech probability per frame, on the project frame grid.

    Silero works on fixed 512-sample windows (32 ms at 16 kHz), so its native
    grid is coarser than ours; the probabilities are mapped onto our centres with
    nearest-neighbour, holding each window's value across the frames it covers.

    Args:
        samples: Mono audio.
        sr: Sample rate; Silero supports 8 k and 16 k.
        grid: Frame grid; defaults to the project grid.
        model: A loaded Silero model, to avoid re-loading per file.

    Returns:
        ``(times, probs)``, both on the frame grid.
    """
    import torch

    grid = grid or FrameGrid(sample_rate=sr)
    audio = np.asarray(samples, dtype=np.float32).ravel()
    frames = grid.n_frames_for(audio.size)
    times = grid.centres(frames)
    if frames <= 0:
        return times, np.empty(0, dtype=np.float64)

    if model is None:
        from silero_vad import load_silero_vad

        model = load_silero_vad()
    model.reset_states()  # type: ignore[attr-defined]

    window = VAD_WINDOW_SAMPLES
    starts = np.arange(0, max(audio.size - window + 1, 1), window)
    values = np.empty(starts.size, dtype=np.float64)
    with torch.inference_mode():
        for i, start in enumerate(starts):
            chunk = audio[start : start + window]
            if chunk.size < window:
                chunk = np.pad(chunk, (0, window - chunk.size))
            values[i] = float(model(torch.from_numpy(chunk), sr))  # type: ignore[operator]

    src_times = (starts + window / 2.0) / float(sr)
    probs = grid.resample_nearest(src_times, values, frames, tolerance_s=window / float(sr))
    return times, np.nan_to_num(probs, nan=0.0)


def speech_intervals(
    times: np.ndarray,
    probs: np.ndarray,
    *,
    threshold: float = 0.5,
    min_silence_s: float = 0.100,
    min_speech_s: float = 0.0,
    speech_pad_s: float = 0.030,
) -> list[tuple[float, float]]:
    """Turn frame probabilities into speech intervals.

    Args:
        times: Frame times, seconds.
        probs: Speech probability per frame.
        threshold: Frames at or above this are speech.
        min_silence_s: Silences shorter than this do not split an interval.
        min_speech_s: Intervals shorter than this are dropped.
        speech_pad_s: Widen every interval by this much at each end, as Silero's
            own helper does, so a soft onset is not clipped.

    Returns:
        Ascending, non-overlapping ``(start, end)`` pairs in seconds.
    """
    t = np.asarray(times, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    if t.size == 0 or t.shape != p.shape:
        return []

    speech = p >= float(threshold)
    if not speech.any():
        return []

    step = float(np.median(np.diff(t))) if t.size > 1 else 0.010
    runs: list[list[float]] = []
    start_idx: int | None = None
    for i, flag in enumerate(speech):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            runs.append([t[start_idx] - step / 2.0, t[i - 1] + step / 2.0])
            start_idx = None
    if start_idx is not None:
        runs.append([t[start_idx] - step / 2.0, t[-1] + step / 2.0])

    merged: list[list[float]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] < float(min_silence_s):
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    out = []
    for start, end in merged:
        if end - start < float(min_speech_s):
            continue
        out.append((max(0.0, start - speech_pad_s), end + speech_pad_s))
    return out


def silence_containing(
    intervals: Sequence[tuple[float, float]], t: float
) -> tuple[float, float] | None:
    """The silence between speech intervals that contains ``t``, if any.

    Used to measure a pause independently of any alignment - the acceptance check
    for an inserted silence.
    """
    for (_, end), (next_start, _) in pairwise(intervals):
        if end <= t <= next_start:
            return float(end), float(next_start)
    return None


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #


def refine_gaps(
    gaps: Sequence[Gap],
    intervals: Sequence[tuple[float, float]],
    *,
    max_shift_s: float = DEFAULT_MAX_SHIFT_S,
) -> list[Gap]:
    """Tighten each gap to the silence the VAD actually heard.

    A boundary moves only when the VAD places it within ``max_shift_s`` of the
    aligner's: a larger disagreement is evidence the two tools are describing
    different events, and the alignment wins because every word index depends on
    it. A gap the VAD considers speech throughout is left untouched.

    Args:
        gaps: Gaps from the alignment.
        intervals: Speech intervals from :func:`speech_intervals`.
        max_shift_s: Largest accepted boundary move, seconds.

    Returns:
        New gaps, with ``vad_refined`` set where a boundary moved.
    """
    speech = sorted(intervals)
    out = []
    for gap in gaps:
        mid = (gap.t_start + gap.t_end) / 2.0
        if any(start <= mid <= end for start, end in speech):
            out.append(gap)
            continue

        left = [end for _, end in speech if end <= mid]
        right = [start for start, _ in speech if start >= mid]
        new_start = gap.t_start
        new_end = gap.t_end
        if left and abs(left[-1] - gap.t_start) <= max_shift_s:
            new_start = float(max(left))
        if right and abs(right[0] - gap.t_end) <= max_shift_s:
            new_end = float(min(right))

        moved = (new_start, new_end) != (gap.t_start, gap.t_end)
        if not moved or new_end <= new_start:
            out.append(gap)
            continue
        out.append(replace(gap, t_start=new_start, t_end=new_end, vad_refined=True))
    return out


# --------------------------------------------------------------------------- #
# Gap table
# --------------------------------------------------------------------------- #


def gap_features(
    gaps: Sequence[Gap],
    *,
    config: Config | None = None,
    min_pause_s: float | None = None,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """One row per gap: duration, label, boundary and pause flags.

    ``gap_is_pause`` applies the ``min_pause_s`` threshold, and
    ``gap_is_boundary`` reports whether the *text* expected a pause there - the
    pair is what distinguishes a rhetorical pause from a hesitation, and a
    missing pause from a well-run-on phrase.
    """
    import pandas as pd

    threshold = _min_pause(config, min_pause_s)
    rows = [
        {
            "gap_duration_s": float(g.duration),
            "gap_label": g.label,
            "gap_is_boundary": g.label == "boundary",
            "gap_is_pause": float(g.duration) >= threshold,
            "gap_vad_refined": bool(g.vad_refined),
            "gap_prev_word_idx": int(g.prev_word_idx),
            "gap_next_word_idx": int(g.next_word_idx),
        }
        for g in gaps
    ]
    table = pd.DataFrame(rows, columns=list(PAUSE_GAP_COLUMNS))
    table.index = pd.Index([g.idx for g in gaps], name="unit_idx", dtype="int64")
    table.attrs.update(unit="gap", file_id=file_id, channel_id=channel_id, min_pause_s=threshold)
    return table


def _min_pause(config: Config | None, override: float | None) -> float:
    """The pause threshold, from the config or the fluency-research default."""
    value = DEFAULT_MIN_PAUSE_S
    if config is not None:
        value = float(config.section("features").get("pauses", {}).get("min_pause_s", value))
    return float(override) if override is not None else value


def gap_label_of(gap: Gap) -> GapLabel:
    """The gap's text-derived label, for callers that want the narrow type."""
    return gap.label
