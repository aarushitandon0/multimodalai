"""Exact old -> new time maps for injection edits (spec 5.1).

``CLAUDE.md`` invariant 8: *every operation returns an exact old->new time map*,
and invariant 9: *labels are generated from time maps*. This module is that
contract.

A :class:`TimeMap` is a **monotone piecewise-linear** function from a time on
the clean recording to the corresponding time on the edited recording. It is
stored as an ordered list of :class:`Segment` objects whose old intervals tile
``[0, old_duration]`` without gaps, so nothing is interpolated between
*unknown* points: every kink in the function is an explicit, recorded segment.

Two segment shapes matter:

* a **scale** segment, where ``old_end > old_start`` - material that was kept,
  possibly time-scaled by ``Segment.rate``;
* an **insert** segment, where ``old_end == old_start`` - material that did not
  exist in the original (room tone). One old instant maps to a whole new
  interval, so the map steps there; :meth:`TimeMap.map_time` resolves the
  ambiguity by returning the *earliest* new time, which keeps a word offset
  sitting exactly on the insertion point from being pushed past the inserted
  silence. :meth:`TimeMap.map_interval` takes the other convention for an
  interval end, so a region that brackets an insertion grows to contain it.

Segment arithmetic is done in **samples** by the callers in
:mod:`rostrum.inject.ops` and only then converted to seconds, which is why the
maps are exact rather than merely close: the audio really is the length the map
says it is.

Composition (:meth:`TimeMap.then`) exists because one flaw can be two edits -
``stress_swap`` lengthens two different words - and the label must still refer
to clean-file time.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from rostrum.types import round_time, round_value

__all__ = ["Segment", "TimeMap"]

_EPS = 1e-9
"""Tolerance for float comparisons on second-valued times (1 ns)."""


@dataclass(frozen=True)
class Segment:
    """One linear piece of a :class:`TimeMap`.

    Attributes:
        old_start: Start of the interval on the clean recording, in seconds.
        old_end: End of that interval; equal to ``old_start`` for an insertion.
        new_start: Where ``old_start`` lands on the edited recording.
        new_end: Where ``old_end`` lands.
        kind: ``"identity"`` (untouched), ``"scale"`` (time-scaled) or
            ``"insert"`` (new material). Informational: the arithmetic depends
            only on the four times.
    """

    old_start: float
    old_end: float
    new_start: float
    new_end: float
    kind: str = "scale"

    @property
    def old_duration(self) -> float:
        """Length of the source interval in seconds; ``0.0`` for an insertion."""
        return self.old_end - self.old_start

    @property
    def new_duration(self) -> float:
        """Length of the destination interval in seconds."""
        return self.new_end - self.new_start

    @property
    def rate(self) -> float:
        """New duration divided by old duration.

        Returns ``inf`` for an insertion, which has no old duration. A
        ``rushing`` edit has ``rate < 1``, ``dragging`` has ``rate > 1``.
        """
        if self.old_duration <= _EPS:
            return float("inf")
        return self.new_duration / self.old_duration

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_start": round_time(self.old_start),
            "old_end": round_time(self.old_end),
            "new_start": round_time(self.new_start),
            "new_end": round_time(self.new_end),
            "kind": self.kind,
            "rate": round_value(self.rate),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Segment:
        return cls(
            old_start=float(data["old_start"]),
            old_end=float(data["old_end"]),
            new_start=float(data["new_start"]),
            new_end=float(data["new_end"]),
            kind=str(data.get("kind", "scale")),
        )


@dataclass(frozen=True)
class TimeMap:
    """A monotone piecewise-linear map from clean time to edited time.

    The segments tile ``[0, old_duration]`` in order and share endpoints, which
    is checked on construction: a map that does not account for every instant
    of the original cannot produce trustworthy labels.

    Attributes:
        segments: Ordered, contiguous pieces of the map.
        op: Name of the operation that produced it, for the label JSON.
    """

    segments: tuple[Segment, ...]
    op: str = ""

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a TimeMap needs at least one segment")
        first = self.segments[0]
        if first.old_start < -_EPS or first.new_start < -_EPS:
            raise ValueError("a TimeMap may not start before zero")
        for seg in self.segments:
            if seg.old_end < seg.old_start - _EPS or seg.new_end < seg.new_start - _EPS:
                raise ValueError(f"non-monotone segment: {seg}")
        for a, b in pairwise(self.segments):
            if abs(b.old_start - a.old_end) > _EPS:
                raise ValueError(f"gap in old time between {a} and {b}")
            if abs(b.new_start - a.new_end) > _EPS:
                raise ValueError(f"gap in new time between {a} and {b}")

    # ----------------------------------------------------------------- #
    # Constructors
    # ----------------------------------------------------------------- #

    @classmethod
    def identity(cls, duration: float, *, op: str = "identity") -> TimeMap:
        """Return the map of an edit that changes no timing (e.g. a gain ramp).

        Args:
            duration: Length of the recording in seconds.
            op: Operation name to record.
        """
        return cls(
            segments=(Segment(0.0, float(duration), 0.0, float(duration), kind="identity"),),
            op=op,
        )

    @classmethod
    def from_segments(cls, segments: Iterable[Segment], *, op: str = "") -> TimeMap:
        """Build a map from an iterable of already-contiguous segments."""
        return cls(segments=tuple(segments), op=op)

    @classmethod
    def replacement(
        cls,
        duration: float,
        old_start: float,
        old_end: float,
        new_duration: float,
        *,
        op: str = "",
        kind: str = "scale",
    ) -> TimeMap:
        """Map for replacing ``[old_start, old_end]`` with ``new_duration`` seconds.

        This is the shape every splice-based op produces: identity before the
        cut, one scaled (or inserted) piece, then a rigid shift of everything
        after it by ``new_duration - (old_end - old_start)``.

        Args:
            duration: Length of the clean recording in seconds.
            old_start: Start of the replaced region on the clean recording.
            old_end: End of the replaced region; equal to ``old_start`` for a
                pure insertion.
            new_duration: Length the region occupies after the edit.
            op: Operation name to record.
            kind: Segment kind for the middle piece.

        Returns:
            The exact :class:`TimeMap`.

        Raises:
            ValueError: The region does not lie inside ``[0, duration]``.
        """
        duration = float(duration)
        old_start, old_end = float(old_start), float(old_end)
        new_duration = float(new_duration)
        if not (-_EPS <= old_start <= old_end <= duration + _EPS):
            raise ValueError(f"region [{old_start}, {old_end}] outside [0, {duration}]")

        segments: list[Segment] = []
        if old_start > _EPS:
            segments.append(Segment(0.0, old_start, 0.0, old_start, kind="identity"))
        segments.append(Segment(old_start, old_end, old_start, old_start + new_duration, kind=kind))
        if duration - old_end > _EPS:
            shift = new_duration - (old_end - old_start)
            segments.append(
                Segment(old_end, duration, old_end + shift, duration + shift, kind="identity")
            )
        return cls(segments=tuple(segments), op=op)

    # ----------------------------------------------------------------- #
    # Properties
    # ----------------------------------------------------------------- #

    @property
    def old_duration(self) -> float:
        """Length of the clean recording this map was built for."""
        return self.segments[-1].old_end

    @property
    def new_duration(self) -> float:
        """Length of the edited recording."""
        return self.segments[-1].new_end

    @property
    def delta(self) -> float:
        """Total length change in seconds, ``new_duration - old_duration``."""
        return self.new_duration - self.old_duration

    @property
    def edited_spans(self) -> tuple[tuple[float, float], ...]:
        """Clean-time spans of every non-identity segment.

        These are the regions a label may claim as flawed: outside them the
        audio is sample-identical to the clean file.
        """
        return tuple((s.old_start, s.old_end) for s in self.segments if s.kind != "identity")

    @property
    def edited_spans_new(self) -> tuple[tuple[float, float], ...]:
        """The same regions expressed in edited-file time."""
        return tuple((s.new_start, s.new_end) for s in self.segments if s.kind != "identity")

    # ----------------------------------------------------------------- #
    # Evaluation
    # ----------------------------------------------------------------- #

    def map_time(self, t: float) -> float:
        """Map a clean-file time to its edited-file time.

        Linear inside the containing segment; clamped to the ends outside
        ``[0, old_duration]``. At an insertion point the *earliest* new time is
        returned, so a word that ended exactly where room tone was inserted
        keeps its offset instead of jumping past the inserted silence.

        Args:
            t: Time in seconds on the clean recording.

        Returns:
            Time in seconds on the edited recording.
        """
        t = float(t)
        if t <= self.segments[0].old_start + _EPS:
            return self.segments[0].new_start
        if t >= self.segments[-1].old_end - _EPS:
            return self.segments[-1].new_end
        seg = self._segment_at(t)
        if seg.old_duration <= _EPS:
            return seg.new_start
        frac = (t - seg.old_start) / seg.old_duration
        return seg.new_start + frac * seg.new_duration

    def unmap_time(self, t_new: float) -> float:
        """Map an edited-file time back to clean-file time.

        The inverse is exact except inside an insertion, which has no preimage;
        there the insertion point itself is returned.

        Args:
            t_new: Time in seconds on the edited recording.

        Returns:
            Time in seconds on the clean recording.
        """
        t_new = float(t_new)
        if t_new <= self.segments[0].new_start + _EPS:
            return self.segments[0].old_start
        if t_new >= self.segments[-1].new_end - _EPS:
            return self.segments[-1].old_end
        for seg in self.segments:
            if seg.new_start - _EPS <= t_new <= seg.new_end + _EPS:
                if seg.new_duration <= _EPS:
                    return seg.old_start
                frac = (t_new - seg.new_start) / seg.new_duration
                return seg.old_start + frac * seg.old_duration
        return self.segments[-1].old_end

    def map_interval(self, t0: float, t1: float) -> tuple[float, float]:
        """Map ``[t0, t1]`` to the edited file, keeping inserted material inside.

        ``t1`` is mapped with the *latest* new time when it coincides with an
        insertion point, the opposite convention to :meth:`map_time`, so an
        interval that brackets inserted room tone grows to contain it rather
        than silently excluding it.

        Args:
            t0: Interval start on the clean recording, in seconds.
            t1: Interval end, in seconds.

        Returns:
            ``(new_start, new_end)`` in seconds.
        """
        new0 = self.map_time(t0)
        new1 = self.map_time(t1)
        for seg in self.segments:
            if seg.old_duration <= _EPS and t0 - _EPS <= seg.old_start <= t1 + _EPS:
                new1 = max(new1, seg.new_end)
        return new0, new1

    def then(self, other: TimeMap, *, op: str | None = None) -> TimeMap:
        """Compose with a second map applied to the already-edited audio.

        ``self`` maps clean -> intermediate and ``other`` maps intermediate ->
        final; the result maps clean -> final. The composition is exact: the
        knot set is the union of ``self``'s kinks and the preimages of
        ``other``'s kinks, so no kink of either map can fall strictly inside a
        segment of the result. Insertions in ``other``, which have no preimage
        width, are re-attached afterwards as explicit zero-width segments.

        Args:
            other: Map from this map's output timeline to the final timeline.
            op: Operation name for the result; defaults to ``"<a>+<b>"``.

        Returns:
            The composed :class:`TimeMap`.

        Raises:
            ValueError: The two maps do not meet on a common timeline.
        """
        if abs(other.old_duration - self.new_duration) > 1e-6:
            raise ValueError(
                f"cannot compose: intermediate duration {self.new_duration} "
                f"!= {other.old_duration}"
            )
        knots: list[float] = [0.0, self.old_duration]
        for seg in self.segments:
            knots.extend((seg.old_start, seg.old_end))
        for seg in other.segments:
            # `other` is indexed by *intermediate* time, which is this map's
            # output, so its kinks are pulled back with `unmap_time`.
            knots.extend((self.unmap_time(seg.old_start), self.unmap_time(seg.old_end)))
        knots = _dedupe(knots, lo=0.0, hi=self.old_duration)

        segments: list[Segment] = []
        for a, b in pairwise(knots):
            new_a = other.map_time(self.map_time(a))
            new_b = other.map_time(self.map_time(b))
            same = abs((new_b - new_a) - (b - a)) <= _EPS and abs(new_a - a) <= _EPS
            segments.append(Segment(a, b, new_a, new_b, kind="identity" if same else "scale"))
        segments = _reattach_insertions(segments, self, other)
        return TimeMap(
            segments=tuple(segments),
            op=op if op is not None else f"{self.op}+{other.op}",
        )

    # ----------------------------------------------------------------- #
    # Serialisation
    # ----------------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "old_duration": round_time(self.old_duration),
            "new_duration": round_time(self.new_duration),
            "segments": [seg.to_dict() for seg in self.segments],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TimeMap:
        return cls(
            segments=tuple(Segment.from_dict(s) for s in data["segments"]),
            op=str(data.get("op", "")),
        )

    # ----------------------------------------------------------------- #
    # Internals
    # ----------------------------------------------------------------- #

    def _segment_at(self, t: float) -> Segment:
        """Return the *earliest* segment whose old interval contains ``t``.

        "Earliest" is what makes :meth:`map_time` return the low side of an
        insertion step: the identity segment ending at the insertion point is
        found before the insertion segment that starts there.
        """
        ends = [seg.old_end for seg in self.segments]
        i = min(bisect.bisect_left(ends, t - _EPS), len(self.segments) - 1)
        return self.segments[i]


def _dedupe(values: Sequence[float], *, lo: float, hi: float) -> list[float]:
    """Sort, clamp to ``[lo, hi]`` and drop near-duplicate knots."""
    out: list[float] = []
    for v in sorted(min(max(float(v), lo), hi) for v in values):
        if not out or v - out[-1] > _EPS:
            out.append(v)
    if len(out) < 2:
        out = [lo, hi]
    return out


def _reattach_insertions(segments: list[Segment], first: TimeMap, second: TimeMap) -> list[Segment]:
    """Re-insert the zero-old-width segments that sampling at knots cannot express.

    :meth:`TimeMap.then` evaluates the composed function at knots, and a step
    has no knot representation: both maps' insertions would silently vanish,
    leaving the composed map shorter than the audio. Each one is therefore
    spliced back in explicitly, located at its clean-time preimage and sized by
    where its new material ends on the *final* timeline.

    The knot-sampled segment that follows an insertion starts one insertion
    width too early, because :meth:`TimeMap.map_time` deliberately returns the
    earliest new time at an insertion point; its ``new_start`` is corrected
    here. Its ``new_end``, computed from a time strictly after the insertion,
    was already right.

    Args:
        segments: Composed segments as sampled at the knots.
        first: The clean -> intermediate map.
        second: The intermediate -> final map.

    Returns:
        Segments with every insertion restored and contiguity re-established.
    """
    # (clean-time point, end of the inserted material on the final timeline)
    inserts: list[tuple[float, float]] = [
        (seg.old_start, second.map_time(seg.new_end))
        for seg in first.segments
        if seg.old_duration <= _EPS
    ]
    inserts += [
        (first.unmap_time(seg.old_start), seg.new_end)
        for seg in second.segments
        if seg.old_duration <= _EPS
    ]
    for at_old, new_end in sorted(inserts):
        cut = next(
            (
                i
                for i, seg in enumerate(segments)
                if abs(seg.old_end - at_old) <= _EPS and seg.new_end < new_end - _EPS
            ),
            None,
        )
        if cut is None:
            continue
        before = segments[: cut + 1]
        gap = Segment(at_old, at_old, segments[cut].new_end, new_end, kind="insert")
        after = list(segments[cut + 1 :])
        if after:
            head = after[0]
            after[0] = Segment(head.old_start, head.old_end, new_end, head.new_end, kind=head.kind)
        segments = [*before, gap, *after]
    return segments
