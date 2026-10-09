"""Envelopes of skilled readings: what the acceptable range actually is (spec 10.2).

A single "correct" reading does not exist, so the reference for this project is
not one recording but the *spread* across several skilled readings of the same
text. For each unit - word *w*, gap *w*, phrase *w* - and each feature *k*:

.. math::

    \\mu_{wk} = \\operatorname{median}_r x^{(r)}_{wk}, \\qquad
    \\sigma_{wk} = 1.4826 \\cdot \\operatorname{MAD}_r x^{(r)}_{wk}, \\qquad
    \\hat\\sigma_{wk} = \\max(\\sigma_{wk}, \\sigma^{floor}_k)

Median and MAD rather than mean and standard deviation because R is small (4 to
20 readers): one reader who sneezed mid-sentence would move a mean by a third of
a standard deviation and inflate the spread enough to hide a real flaw, while
the median is unmoved and the MAD barely so. The 1.4826 factor makes
``1.4826 * MAD`` an estimate of :math:`\\sigma` for normally distributed data, so
the z-scores downstream stay interpretable.

**The sigma floor is the leakage-sensitive part.** Some units are genuinely
unanimous: every skilled reader pauses after "...shall not perish from the
earth." If :math:`\\sigma_{wk}` for that gap is 8 ms, a participant who pauses
40 ms shorter scores z = -4 and gets flagged for a difference nobody can hear.
The floor is the 25th percentile of :math:`\\sigma_{\\cdot k}` over **all units of
all training texts** (:func:`fit_sigma_floors`), which is why building an
envelope is a three-step process rather than one call:

1. :func:`build_envelope` per text, with no floor yet;
2. :func:`fit_sigma_floors` over the fold's *training* texts only;
3. :func:`apply_sigma_floors` to every text in the fold, held-out included -
   the held-out text gets the floor *fitted elsewhere*, never its own.

:func:`build_fold_envelopes` does all three and is what callers normally want.
Fitting the floor on the held-out text would leak exactly the information LOTO
exists to withhold (CLAUDE.md invariants 10-11).

**Per-unit reference values are kept**, not just the summary. They cost little
(R floats per unit per feature), they make :func:`leave_one_out` an exact
recomputation rather than an approximation, and they are what lets an
explanation card say "17 of 19 readers paused here" instead of "the median gap
was 0.42 s" (:meth:`UnitEnvelope.count_where`).

**Word matching** is by canonical unit index: identical transcripts mean word
*i* of one reading corresponds to word *i* of another, with no DTW (spec 10.1).
A recording whose alignment dropped or merged a word would silently shift every
later index, so :class:`WordIndexMap` carries a per-recording override from
local index to canonical index, and :func:`build_envelope` applies it before
anything is compared.

**Frame curves** (:func:`word_curves`, :func:`build_curve_envelope`) exist only
for the dashboard overlay. Each reader's F0, energy and rate curve inside a word
is resampled to a fixed number of points on the word's own normalised time axis,
so readers who took 0.3 s and 0.5 s over the same word can be averaged
point-by-point. The curve envelope never feeds detection.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.features import names as N
from rostrum.types import Envelope

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations

    from rostrum.config import Config
    from rostrum.types import FeatureTable

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CURVE_POINTS",
    "MAD_SCALE",
    "CurveEnvelope",
    "EnvelopeError",
    "Reference",
    "TextEnvelope",
    "UnitEnvelope",
    "WordIndexMap",
    "apply_sigma_floors",
    "build_curve_envelope",
    "build_envelope",
    "build_fold_envelopes",
    "fit_sigma_floors",
    "leave_one_out",
    "mad",
    "robust_centre_spread",
    "summarise_counts",
    "warp_to_envelope",
    "word_curves",
]


MAD_SCALE: float = Envelope.MAD_SCALE
"""1.4826, the constant making ``1.4826 * MAD`` a sigma estimate for normals."""

DEFAULT_CURVE_POINTS: int = 20
"""Points per word on the normalised time axis, when no config is supplied."""

DEFAULT_CURVE_SIGNALS: tuple[str, ...] = ("f0_st", "energy_z", "rate_local_syll_per_s")
"""Frame/word signals plotted as an envelope band (spec 15.2)."""


class EnvelopeError(ValueError):
    """An envelope cannot be built from the references supplied."""


@contextmanager
def _quiet_all_nan() -> Iterator[None]:
    """Suppress numpy's all-NaN-slice warning, which is an expected state here.

    A unit no reader measured must produce ``NaN`` with ``n = 0``; that is the
    designed behaviour, so the warning carries no information and would fire on
    every recording with an unvoiced word.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        yield


# --------------------------------------------------------------------------- #
# Robust statistics
# --------------------------------------------------------------------------- #


def mad(values: Sequence[float] | np.ndarray, *, axis: int | None = None) -> Any:
    """Median absolute deviation about the median, ignoring ``NaN``.

    Args:
        values: Sample values. ``NaN`` entries are excluded, so a reader whose
            word had no voiced frames does not drag the spread to ``NaN``.
        axis: Axis to reduce, or ``None`` to reduce everything.

    Returns:
        The MAD, **not** scaled by 1.4826; multiply by :data:`MAD_SCALE` for a
        sigma estimate. ``NaN`` when every value is ``NaN``.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.float64("nan") if axis is None else np.full(0, np.nan)
    # A unit no reader measured is a legitimate state (NaN centre, NaN spread,
    # n = 0); numpy's all-NaN-slice warning would be noise on every run.
    with _quiet_all_nan():
        centre = np.nanmedian(arr, axis=axis, keepdims=axis is not None)
        out = np.nanmedian(np.abs(arr - centre), axis=axis)
    return out


def robust_centre_spread(
    values: np.ndarray, *, axis: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(median, 1.4826*MAD, n_finite)`` along ``axis``, ignoring ``NaN``.

    Args:
        values: Array with references along ``axis``.
        axis: The reference axis.

    Returns:
        Three arrays with ``axis`` removed: the median, the scaled MAD, and how
        many finite values each statistic was computed from. A unit no reader
        measured gets ``NaN`` centre, ``NaN`` spread and ``n = 0`` - never a
        silent zero, which would read as "every reader agreed exactly".
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    n = finite.sum(axis=axis)
    with _quiet_all_nan():
        centre = np.where(n > 0, np.nanmedian(np.where(finite, arr, np.nan), axis=axis), np.nan)
        spread = np.where(n > 0, MAD_SCALE * mad(np.where(finite, arr, np.nan), axis=axis), np.nan)
    return centre, spread, n.astype(np.int64)


# --------------------------------------------------------------------------- #
# Canonical word indices
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WordIndexMap:
    """Per-recording map from local unit index to canonical unit index.

    Word *i* of one reading means word *i* of another only while both
    alignments contain every word of the text. When MFA drops a word - an
    ``<unk>``, a reader's elision, a repeated word collapsed into one interval -
    every later local index is off by one against the canonical transcript, and
    comparing them would compare different words.

    The override is therefore stored with the recording, not inferred:

    * ``to_canonical`` holds only the entries that *differ* from identity, so an
      ordinary recording carries an empty dict and :attr:`is_identity` is
      ``True``;
    * ``dropped`` lists canonical indices this recording has no unit for. They
      simply contribute nothing to that unit's statistics.

    Attributes:
        file_id: The recording this map belongs to.
        unit: ``"word"``, ``"gap"`` or ``"phrase"``.
        to_canonical: ``{local_idx: canonical_idx}``, overrides only.
        dropped: Canonical indices missing from this recording, sorted.
    """

    file_id: str = ""
    unit: str = "word"
    to_canonical: Mapping[int, int] = field(default_factory=dict)
    dropped: tuple[int, ...] = ()

    @property
    def is_identity(self) -> bool:
        """Whether this map leaves every index unchanged."""
        return not self.to_canonical

    def canonical_index(self, local_idx: int) -> int:
        """Canonical index of ``local_idx`` (identity unless overridden)."""
        return int(self.to_canonical.get(int(local_idx), int(local_idx)))

    def apply(self, table: FeatureTable) -> FeatureTable:
        """Return ``table`` re-indexed on canonical unit indices.

        Args:
            table: A per-unit table following ``FEATURE_TABLE_CONTRACT``.

        Returns:
            The same table with its index remapped and sorted; ``attrs`` are
            preserved, plus ``word_index_map`` recording that a remap happened.

        Raises:
            EnvelopeError: The map sends two local indices to one canonical
                index, which would make the comparison key ambiguous.
        """
        import pandas as pd

        if self.is_identity:
            return table
        new_index = [self.canonical_index(i) for i in table.index]
        if len(set(new_index)) != len(new_index):
            dupes = sorted({i for i in new_index if new_index.count(i) > 1})
            raise EnvelopeError(
                f"{self.file_id}: word index map is not injective; canonical "
                f"index/indices {dupes} claimed by more than one local unit"
            )
        out = table.copy()
        out.index = pd.Index(new_index, name="unit_idx", dtype="int64")
        out = out.sort_index()
        out.attrs.update(dict(table.attrs))
        out.attrs["word_index_map"] = dict(self.to_canonical)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WordIndexMap:
        """Rebuild from the JSON form written next to an alignment."""
        return cls(
            file_id=str(data.get("file_id", "")),
            unit=str(data.get("unit", "word")),
            to_canonical={int(k): int(v) for k, v in dict(data.get("to_canonical", {})).items()},
            dropped=tuple(int(i) for i in data.get("dropped", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON form: only the overrides, so an ordinary recording is tiny."""
        return {
            "file_id": self.file_id,
            "unit": self.unit,
            "to_canonical": {str(k): int(v) for k, v in dict(self.to_canonical).items()},
            "dropped": [int(i) for i in self.dropped],
        }


# --------------------------------------------------------------------------- #
# References
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Reference:
    """One skilled reading contributing to an envelope.

    Attributes:
        file_id: Recording id; the envelope's reference key.
        channel_id: Recording channel, so the channel rule can be enforced
            (CLAUDE.md invariant 7).
        tables: ``{unit: FeatureTable}`` for this recording.
        index_map: ``{unit: WordIndexMap}``; units absent here use identity.
    """

    file_id: str
    channel_id: str = ""
    tables: Mapping[str, FeatureTable] = field(default_factory=dict)
    index_map: Mapping[str, WordIndexMap] = field(default_factory=dict)

    def table(self, unit: str) -> FeatureTable | None:
        """Return this reading's table for ``unit``, canonically indexed."""
        table = self.tables.get(unit)
        if table is None:
            return None
        imap = self.index_map.get(unit)
        return table if imap is None else imap.apply(table)


def _as_references(
    reference_files: Sequence[str],
    feature_tables: Mapping[str, Mapping[str, FeatureTable]] | Sequence[Reference],
    index_maps: Mapping[str, Mapping[str, WordIndexMap]] | None = None,
) -> list[Reference]:
    """Normalise the two accepted input shapes into a sorted list of references.

    ``feature_tables`` may be either ``{file_id: {unit: table}}`` - the shape a
    feature cache hands over - or a sequence of :class:`Reference` objects.
    """
    if isinstance(feature_tables, Mapping):
        maps = index_maps or {}
        refs = [
            Reference(
                file_id=fid,
                channel_id=(
                    str(next(iter(tbls.values())).attrs.get("channel_id", "")) if tbls else ""
                ),
                tables=tbls,
                index_map=dict(maps.get(fid, {})),
            )
            for fid, tbls in feature_tables.items()
            if fid in set(reference_files)
        ]
    else:
        wanted = set(reference_files)
        refs = [r for r in feature_tables if r.file_id in wanted]

    missing = sorted(set(reference_files) - {r.file_id for r in refs})
    if missing:
        raise EnvelopeError(f"no feature tables for reference file(s): {missing}")
    # Sorted so reference_ids, and therefore the leave-one-out band, are
    # reproducible regardless of dict ordering (CLAUDE.md invariant 13).
    return sorted(refs, key=lambda r: r.file_id)


# --------------------------------------------------------------------------- #
# Per-unit envelope
# --------------------------------------------------------------------------- #


@dataclass
class UnitEnvelope:
    """Envelope statistics for one unit kind, with the raw reference values.

    Attributes:
        envelope: The serialisable summary (:class:`~rostrum.types.Envelope`).
        values: ``{unit_idx: array of shape (n_references, n_features)}`` in
            ``envelope.reference_ids`` / ``envelope.features`` order. ``NaN``
            where a reader has no measurement for that unit.
        gap_labels: ``{unit_idx: label}`` for a gap envelope, so the pause
            rules can read the text expectation without the alignment.
    """

    envelope: Envelope
    values: dict[int, np.ndarray] = field(default_factory=dict)
    gap_labels: dict[int, str] = field(default_factory=dict)

    # -- convenience accessors ---------------------------------------------

    @property
    def unit(self) -> str:
        """``"word"``, ``"gap"`` or ``"phrase"``."""
        return str(self.envelope.unit)

    @property
    def features(self) -> list[str]:
        """Feature columns, in the order of the stored arrays."""
        return list(self.envelope.features)

    @property
    def unit_indices(self) -> list[int]:
        """Canonical unit indices covered, sorted."""
        return sorted(self.envelope.median)

    @property
    def n_references(self) -> int:
        """Number of contributing readers."""
        return len(self.envelope.reference_ids)

    def column(self, feature: str) -> int:
        """Column position of ``feature``.

        Raises:
            KeyError: The feature is not in this envelope.
        """
        try:
            return self.features.index(feature)
        except ValueError:
            raise KeyError(f"{feature!r} is not in this {self.unit} envelope") from None

    def has(self, feature: str) -> bool:
        """Whether ``feature`` is in this envelope."""
        return feature in self.envelope.features

    def reference_values(self, unit_idx: int, feature: str) -> np.ndarray:
        """Every reader's value for one unit and feature, ``NaN`` included.

        Parallel to ``envelope.reference_ids``, so the two can be zipped.
        """
        return np.asarray(self.values[int(unit_idx)][:, self.column(feature)], dtype=np.float64)

    def count_where(
        self, unit_idx: int, feature: str, *, at_least: float | None = None
    ) -> tuple[int, int]:
        """Count readers satisfying a threshold, for the card's context line.

        Args:
            unit_idx: Canonical unit index.
            feature: Feature column.
            at_least: Lower bound. ``None`` counts readers with any measurement.

        Returns:
            ``(matching, measured)`` - e.g. ``(17, 19)`` renders as "17 of 19
            readers paused here".
        """
        vals = self.reference_values(unit_idx, feature)
        measured = vals[np.isfinite(vals)]
        if at_least is None:
            return int(measured.size), int(measured.size)
        return int((measured >= float(at_least)).sum()), int(measured.size)

    def n_at(self, unit_idx: int) -> int:
        """How many readers measured ``unit_idx`` at all."""
        return int(self.envelope.n_references.get(int(unit_idx), 0))


def _unit_envelope(
    text_id: str,
    unit: str,
    refs: Sequence[Reference],
    columns: Sequence[str],
    *,
    fold: str,
) -> UnitEnvelope:
    """Stack the references for one unit kind and reduce them robustly."""
    tables = {r.file_id: r.table(unit) for r in refs}
    present = [r for r in refs if tables[r.file_id] is not None]
    if not present:
        raise EnvelopeError(f"{text_id}: no reference has a {unit} table")

    cols = [c for c in columns if all(c in tables[r.file_id].columns for r in present)]
    dropped = [c for c in columns if c not in cols]
    if dropped:
        logger.debug(
            "%s/%s: %d feature(s) missing from at least one reference, excluded: %s",
            text_id,
            unit,
            len(dropped),
            dropped,
        )

    indices = sorted({int(i) for r in present for i in tables[r.file_id].index})
    n_r, n_u, n_k = len(present), len(indices), len(cols)
    stack = np.full((n_r, n_u, n_k), np.nan, dtype=np.float64)
    row_of = {idx: row for row, idx in enumerate(indices)}

    for r_i, ref in enumerate(present):
        table = tables[ref.file_id]
        sub = table.reindex(index=indices, columns=cols)
        stack[r_i] = sub.to_numpy(dtype=np.float64, na_value=np.nan)

    centre, spread, counts = robust_centre_spread(stack, axis=0)

    envelope = Envelope(
        text_id=text_id,
        fold=fold,
        unit=unit,  # type: ignore[arg-type]
        features=list(cols),
        median={idx: [float(v) for v in centre[row]] for idx, row in row_of.items()},
        sigma={idx: [float(v) for v in spread[row]] for idx, row in row_of.items()},
        sigma_floor={},
        # One count per unit: the readers who measured anything at that unit.
        n_references={idx: int(counts[row].max(initial=0)) for idx, row in row_of.items()},
        reference_ids=[r.file_id for r in present],
        channel_ids=[r.channel_id for r in present],
    )

    gap_labels: dict[int, str] = {}
    if unit == "gap":
        for ref in present:
            table = tables[ref.file_id]
            if "gap_label" not in table.columns:
                continue
            for idx, label in table["gap_label"].items():
                # The label is text-derived, so every rendition agrees; the
                # first reader that has it settles it.
                gap_labels.setdefault(int(idx), str(label))

    return UnitEnvelope(
        envelope=envelope,
        values={idx: stack[:, row, :] for idx, row in row_of.items()},
        gap_labels=gap_labels,
    )


# --------------------------------------------------------------------------- #
# Frame curves
# --------------------------------------------------------------------------- #


@dataclass
class CurveEnvelope:
    """Time-normalised frame curves per word, median and spread per point.

    Used by the dashboard to draw a shaded "skilled readers did this" band
    behind the participant's own curve (spec 15.2). Never used by detection:
    the band is a picture of the same statistics the z-scores already carry.

    Attributes:
        text_id: Text these curves belong to.
        fold: LOTO fold, or ``"all"``.
        signals: Signal names, in the order of the stored arrays' last axis.
        n_points: Points per word on the normalised axis.
        median: ``{word_idx: array (n_points, n_signals)}``.
        sigma: ``{word_idx: array (n_points, n_signals)}``, ``1.4826 * MAD``.
        n_references: ``{word_idx: readers contributing}``.
        reference_ids: Contributing readers, sorted.
    """

    text_id: str
    fold: str = "all"
    signals: tuple[str, ...] = ()
    n_points: int = DEFAULT_CURVE_POINTS
    median: dict[int, np.ndarray] = field(default_factory=dict)
    sigma: dict[int, np.ndarray] = field(default_factory=dict)
    n_references: dict[int, int] = field(default_factory=dict)
    reference_ids: list[str] = field(default_factory=list)

    def band(self, word_idx: int, signal: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(x, lower, upper)`` for one word and signal.

        ``x`` runs 0 to 1 across the word, and the band is median +/- 1 sigma.
        """
        col = self.signals.index(signal)
        x = np.linspace(0.0, 1.0, self.n_points)
        mu = self.median[int(word_idx)][:, col]
        sd = self.sigma[int(word_idx)][:, col]
        return x, mu - sd, mu + sd


def word_curves(
    frames: FeatureTable,
    word_spans: Mapping[int, tuple[float, float]],
    signals: Sequence[str],
    *,
    n_points: int = DEFAULT_CURVE_POINTS,
    word_values: Mapping[str, Mapping[int, float]] | None = None,
) -> dict[int, np.ndarray]:
    """Resample each word's frame curves onto a fixed normalised time axis.

    Two readers take different amounts of time over the same word, so their
    curves cannot be averaged frame by frame. Mapping each word onto
    ``n_points`` evenly spaced positions in its *own* duration makes the point
    index mean "15% of the way through this word" for every reader, which is
    what an overlay needs.

    Args:
        frames: The frame table from ``FeatureBundle.frames``, with ``time_s``.
        word_spans: ``{word_idx: (t_start, t_end)}`` from the alignment.
        signals: Columns of ``frames`` to resample. A name not in ``frames`` is
            taken from ``word_values`` instead and held constant over the word,
            which is how a per-word quantity such as rate joins the overlay.
        n_points: Points per word.
        word_values: ``{signal: {word_idx: value}}`` for the per-word fallback.

    Returns:
        ``{word_idx: array (n_points, len(signals))}``, ``NaN`` where a word had
        no usable frames for that signal.
    """
    times = np.asarray(frames["time_s"], dtype=np.float64)
    axis = np.linspace(0.0, 1.0, int(n_points))
    per_word: dict[int, np.ndarray] = {}
    word_values = word_values or {}

    columns = {s: np.asarray(frames[s], dtype=np.float64) for s in signals if s in frames.columns}

    for idx, (t0, t1) in word_spans.items():
        out = np.full((int(n_points), len(signals)), np.nan, dtype=np.float64)
        inside = np.flatnonzero((times >= float(t0)) & (times < float(t1)))
        duration = float(t1) - float(t0)
        for col, signal in enumerate(signals):
            if signal in columns:
                if inside.size == 0 or duration <= 0:
                    continue
                values = columns[signal][inside]
                ok = np.isfinite(values)
                if ok.sum() == 0:
                    continue
                if ok.sum() == 1:
                    out[:, col] = float(values[ok][0])
                    continue
                pos = (times[inside][ok] - float(t0)) / duration
                out[:, col] = np.interp(axis, pos, values[ok])
            else:
                value = word_values.get(signal, {}).get(int(idx))
                if value is not None and np.isfinite(float(value)):
                    out[:, col] = float(value)
        per_word[int(idx)] = out

    return per_word


def _dtw_path(participant: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    """The DTW path between two curves, or ``None`` when it cannot be run.

    Uses ``librosa.sequence.dtw`` on a single normalised signal. Non-finite
    points are filled by interpolation first: DTW has no notion of a missing
    observation, and an unvoiced frame inside a word is missing, not zero.
    """
    try:
        from librosa.sequence import dtw as librosa_dtw
    except ImportError:  # pragma: no cover - depends on the environment
        logger.warning("librosa is unavailable; curves are not warped")
        return None

    def fill(series: np.ndarray) -> np.ndarray | None:
        values = np.asarray(series, dtype=np.float64)
        ok = np.isfinite(values)
        if ok.sum() < 2:
            return None
        index = np.arange(values.size, dtype=np.float64)
        return np.interp(index, index[ok], values[ok])

    a, b = fill(participant), fill(reference)
    if a is None or b is None:
        return None
    scale = float(np.nanstd(np.concatenate([a, b]))) or 1.0
    _, path = librosa_dtw(
        X=(a / scale)[None, :],
        Y=(b / scale)[None, :],
        metric="euclidean",
        subseq=False,
        backtrack=True,
    )
    return np.asarray(path, dtype=int)


def warp_to_envelope(
    participant: Mapping[int, np.ndarray],
    envelope: CurveEnvelope,
    *,
    signals: Sequence[str] | None = None,
    dtw: bool = True,
) -> dict[int, np.ndarray]:
    """Warp a participant's within-word curves onto the reference timeline.

    Spec 10.1 allows DTW in exactly one place: *inside* a word, to line a
    reading up with the envelope for plotting. Detection never uses it - word
    *i* is word *i* - so this function exists purely so the overlay in spec
    15.2 compares like with like when a reader lingers on the first syllable
    and the participant on the last.

    The warp is computed per word on the first available signal and applied to
    every signal of that word, so the curves stay mutually consistent: warping
    F0 one way and energy another would draw a picture of nothing.

    Args:
        participant: ``{word_idx: array (n_points, n_signals)}`` from
            :func:`word_curves`.
        envelope: The reference :class:`CurveEnvelope` to warp onto.
        signals: Signal order of the participant arrays; the envelope's own
            order when omitted.
        dtw: Run the warp. With ``False`` the curves are returned unchanged,
            which is the ablation the report compares against.

    Returns:
        ``{word_idx: array (n_points, n_signals)}`` on the envelope's axis.
        A word the envelope does not cover is returned unchanged.
    """
    names = list(signals) if signals is not None else list(envelope.signals)
    out: dict[int, np.ndarray] = {}
    warped_words = 0

    for word_idx, curve in participant.items():
        array = np.asarray(curve, dtype=np.float64)
        median = envelope.median.get(int(word_idx))
        if not dtw or median is None or array.ndim != 2:
            out[int(word_idx)] = array
            continue

        path = None
        for col, name in enumerate(names):
            if name not in envelope.signals or col >= array.shape[1]:
                continue
            ref_col = envelope.signals.index(name)
            path = _dtw_path(array[:, col], median[:, ref_col])
            if path is not None:
                break
        if path is None:
            out[int(word_idx)] = array
            continue

        # path is (steps, 2) of (participant point, reference point), from the
        # end backwards. Average every participant point mapped to one
        # reference point, so the output is on the reference axis.
        warped = np.full((median.shape[0], array.shape[1]), np.nan, dtype=np.float64)
        for ref_point in range(median.shape[0]):
            rows = path[path[:, 1] == ref_point, 0]
            if rows.size:
                with _quiet_all_nan():
                    warped[ref_point] = np.nanmean(array[rows], axis=0)
        out[int(word_idx)] = warped
        warped_words += 1

    logger.debug("warped %d of %d word curve(s) onto the envelope", warped_words, len(participant))
    return out


def build_curve_envelope(
    text_id: str,
    per_reference: Mapping[str, Mapping[int, np.ndarray]],
    signals: Sequence[str],
    *,
    fold: str = "all",
    n_points: int = DEFAULT_CURVE_POINTS,
) -> CurveEnvelope:
    """Median and ``1.4826*MAD`` per normalised time point across readers.

    Args:
        text_id: Text being described.
        per_reference: ``{file_id: {word_idx: array (n_points, n_signals)}}``,
            as returned by :func:`word_curves`.
        signals: Signal order of the arrays' last axis.
        fold: LOTO fold tag.
        n_points: Points per word; must match the arrays.

    Returns:
        A :class:`CurveEnvelope`.
    """
    reference_ids = sorted(per_reference)
    indices = sorted({int(i) for curves in per_reference.values() for i in curves})
    median: dict[int, np.ndarray] = {}
    sigma: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}

    for idx in indices:
        stack = np.stack(
            [
                per_reference[fid].get(idx, np.full((n_points, len(signals)), np.nan))
                for fid in reference_ids
            ],
            axis=0,
        )
        centre, spread, n = robust_centre_spread(stack, axis=0)
        median[idx] = centre
        sigma[idx] = spread
        counts[idx] = int(n.max(initial=0))

    return CurveEnvelope(
        text_id=text_id,
        fold=fold,
        signals=tuple(signals),
        n_points=int(n_points),
        median=median,
        sigma=sigma,
        n_references=counts,
        reference_ids=reference_ids,
    )


# --------------------------------------------------------------------------- #
# Text envelope
# --------------------------------------------------------------------------- #


@dataclass
class TextEnvelope:
    """Every unit's envelope for one text, plus the overlay curves.

    :class:`~rostrum.types.Envelope` in ``types.py`` is deliberately the
    *serialisable* summary of one unit kind; this is the working container that
    holds the three unit kinds together with the raw reference values and the
    curves. Only the inner :class:`~rostrum.types.Envelope` objects go to JSON,
    so the determinism hash is unaffected by anything added here.

    Attributes:
        text_id: Text these envelopes describe.
        fold: LOTO fold the sigma floors were fitted in, or ``"all"``.
        units: ``{unit: UnitEnvelope}``.
        curves: The overlay band, or ``None`` when curves were not requested.
        reference_ids: Contributing readers, sorted.
        channel_ids: Their channels, parallel to ``reference_ids``.
        same_channel: ``True`` when every reader shares one channel, which is
            the only case in which channel-sensitive features may fire
            (CLAUDE.md invariant 7).
    """

    text_id: str
    fold: str = "all"
    units: dict[str, UnitEnvelope] = field(default_factory=dict)
    curves: CurveEnvelope | None = None
    reference_ids: list[str] = field(default_factory=list)
    channel_ids: list[str] = field(default_factory=list)

    @property
    def n_references(self) -> int:
        """Number of contributing readers."""
        return len(self.reference_ids)

    @property
    def same_channel(self) -> bool:
        """Whether every contributing reader shares a single channel."""
        channels = {c for c in self.channel_ids if c}
        return len(channels) == 1 and len(self.channel_ids) == len(self.reference_ids)

    def unit(self, unit: str) -> UnitEnvelope:
        """Return the envelope for ``unit``.

        Raises:
            KeyError: This text has no envelope for that unit.
        """
        try:
            return self.units[unit]
        except KeyError:
            raise KeyError(
                f"{self.text_id} has no {unit!r} envelope; built: {sorted(self.units)}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """JSON form: the per-unit :class:`~rostrum.types.Envelope` summaries."""
        return {
            "text_id": self.text_id,
            "fold": self.fold,
            "n_references": self.n_references,
            "reference_ids": list(self.reference_ids),
            "channel_ids": list(self.channel_ids),
            "same_channel": self.same_channel,
            "units": {u: ue.envelope.to_dict() for u, ue in sorted(self.units.items())},
        }

    def summary(self) -> str:
        """One line for a build log: readers and units per kind."""
        parts = ", ".join(
            f"{len(ue.unit_indices)} {u}s x {len(ue.features)} feat"
            for u, ue in sorted(self.units.items())
        )
        return f"{self.text_id} [{self.fold}]: n={self.n_references} readers; {parts}"


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _envelope_conf(config: Config | None) -> dict[str, Any]:
    """Envelope settings from ``configs/detection.yaml``, with code defaults."""
    defaults: dict[str, Any] = {
        "mad_scale": MAD_SCALE,
        "sigma_floor_percentile": 25.0,
        "min_references": 3,
        "features": {},
        "channel_sensitive_features": {},
        "curve_points": DEFAULT_CURVE_POINTS,
        "curve_signals": DEFAULT_CURVE_SIGNALS,
    }
    if config is None:
        return defaults
    env = config.detection["envelope"]
    curves = env.get("curves", {}) if hasattr(env, "get") else {}
    defaults.update(
        mad_scale=float(env["mad_scale"]),
        sigma_floor_percentile=float(env["sigma_floor_percentile"]),
        min_references=int(env["min_references"]),
        features={k: tuple(str(x) for x in v) for k, v in dict(env.get("features", {})).items()},
        channel_sensitive_features={
            k: tuple(str(x) for x in v)
            for k, v in dict(env.get("channel_sensitive_features", {})).items()
        },
        curve_points=int(dict(curves).get("n_points", DEFAULT_CURVE_POINTS)),
        curve_signals=tuple(str(s) for s in dict(curves).get("signals", DEFAULT_CURVE_SIGNALS)),
    )
    if abs(defaults["mad_scale"] - MAD_SCALE) > 1e-9:
        raise EnvelopeError(
            f"detection.envelope.mad_scale is {defaults['mad_scale']}, but spec 10.2 "
            f"fixes it at {MAD_SCALE}"
        )
    return defaults


def _columns_for(
    unit: str,
    table: FeatureTable,
    conf: Mapping[str, Any],
    *,
    include_channel_sensitive: bool,
) -> list[str]:
    """Resolve the configured canonical names to this table's columns."""
    wanted = list(conf["features"].get(unit, ()))
    if include_channel_sensitive:
        wanted += list(conf["channel_sensitive_features"].get(unit, ()))
    columns = list(table.columns)
    if wanted:
        return N.numeric_feature_columns(N.resolve_many(wanted, columns))
    # No configured list for this unit: fall back to every comparable column,
    # which keeps a new unit kind usable before the config catches up.
    logger.debug("no configured envelope features for unit %r; using every column", unit)
    return N.numeric_feature_columns(columns)


def build_envelope(
    text_id: str,
    reference_files: Sequence[str],
    feature_tables: Mapping[str, Mapping[str, FeatureTable]] | Sequence[Reference],
    *,
    fold: str = "all",
    units: Sequence[str] = ("word", "gap", "phrase"),
    index_maps: Mapping[str, Mapping[str, WordIndexMap]] | None = None,
    config: Config | None = None,
    include_channel_sensitive: bool | None = None,
    frames: Mapping[str, FeatureTable] | None = None,
    word_spans: Mapping[str, Mapping[int, tuple[float, float]]] | None = None,
    sigma_floors: Mapping[str, Mapping[str, float]] | None = None,
) -> TextEnvelope:
    """Build the per-unit envelope of one text from its skilled readings.

    The sigma floor is **not** fitted here, because it is a property of the
    fold's training texts rather than of this text (spec 10.2). Either pass one
    in through ``sigma_floors``, or call :func:`build_fold_envelopes`, which
    fits the floor on the training texts and applies it to the whole fold.

    Args:
        text_id: The text being described; the envelope key.
        reference_files: ``file_id`` of each skilled reading to include. Human
            flawed takes and injected files must never appear here.
        feature_tables: ``{file_id: {unit: table}}``, or :class:`Reference`
            objects carrying their own index maps.
        fold: LOTO fold tag, recorded on every unit envelope.
        units: Unit kinds to build.
        index_maps: ``{file_id: {unit: WordIndexMap}}`` for recordings whose
            local unit indices differ from the canonical ones.
        config: Supplies the feature lists, the MAD scale and the curve
            settings. ``None`` uses this module's defaults.
        include_channel_sensitive: Add the channel-sensitive features to the
            envelope. ``None`` (the default) decides by the channel rule: they
            are included only when every reference shares one channel
            (CLAUDE.md invariant 7).
        frames: ``{file_id: frame table}``; supplying these builds the overlay
            curve envelope.
        word_spans: ``{file_id: {word_idx: (t_start, t_end)}}``, required
            alongside ``frames``.
        sigma_floors: ``{unit: {feature: floor}}`` to apply immediately.

    Returns:
        A :class:`TextEnvelope`.

    Raises:
        EnvelopeError: Fewer references than ``min_references``, a reference
            with no feature tables, or a non-injective word index map.
    """
    conf = _envelope_conf(config)
    refs = _as_references(reference_files, feature_tables, index_maps)

    if len(refs) < int(conf["min_references"]):
        raise EnvelopeError(
            f"{text_id}: {len(refs)} reference(s) but detection.envelope."
            f"min_references is {conf['min_references']}; this text cannot form an envelope"
        )

    channels = {r.channel_id for r in refs if r.channel_id}
    same_channel = len(channels) == 1 and all(r.channel_id for r in refs)
    if include_channel_sensitive is None:
        include_channel_sensitive = same_channel
    elif include_channel_sensitive and not same_channel:
        raise EnvelopeError(
            f"{text_id}: channel-sensitive features were requested but the "
            f"references span channels {sorted(channels) or ['<unset>']}; "
            "comparing them across recordings is forbidden (CLAUDE.md invariant 7)"
        )

    built: dict[str, UnitEnvelope] = {}
    for unit in units:
        sample = next((r.tables[unit] for r in refs if unit in r.tables), None)
        if sample is None:
            logger.debug("%s: no reference has a %s table; skipping that unit", text_id, unit)
            continue
        columns = _columns_for(
            unit, sample, conf, include_channel_sensitive=include_channel_sensitive
        )
        if not columns:
            logger.warning("%s/%s: no comparable feature columns; unit skipped", text_id, unit)
            continue
        built[unit] = _unit_envelope(text_id, unit, refs, columns, fold=fold)

    if not built:
        raise EnvelopeError(f"{text_id}: no unit could be built from {len(refs)} reference(s)")

    curves = None
    if frames:
        signals = tuple(conf["curve_signals"])
        n_points = int(conf["curve_points"])
        spans = word_spans or {}
        word_env = built.get("word")
        per_ref: dict[str, dict[int, np.ndarray]] = {}
        for ref in refs:
            frame_table = frames.get(ref.file_id)
            if frame_table is None or ref.file_id not in spans:
                continue
            # Per-word quantities (rate) have no frame column; they are held
            # constant across the word so they can share the overlay axis.
            fallback: dict[str, dict[int, float]] = {}
            table = ref.table("word")
            if table is not None and word_env is not None:
                for signal in signals:
                    if signal in frame_table.columns or signal not in table.columns:
                        continue
                    fallback[signal] = {
                        int(i): float(v) for i, v in table[signal].items() if np.isfinite(float(v))
                    }
            imap = ref.index_map.get("word")
            local_spans = spans[ref.file_id]
            canonical_spans = (
                local_spans
                if imap is None or imap.is_identity
                else {imap.canonical_index(i): v for i, v in local_spans.items()}
            )
            per_ref[ref.file_id] = word_curves(
                frame_table,
                canonical_spans,
                signals,
                n_points=n_points,
                word_values=fallback,
            )
        if per_ref:
            curves = build_curve_envelope(text_id, per_ref, signals, fold=fold, n_points=n_points)

    envelope = TextEnvelope(
        text_id=text_id,
        fold=fold,
        units=built,
        curves=curves,
        reference_ids=[r.file_id for r in refs],
        channel_ids=[r.channel_id for r in refs],
    )
    if sigma_floors:
        envelope = apply_sigma_floors(envelope, sigma_floors)
    return envelope


# --------------------------------------------------------------------------- #
# Sigma floors (the leakage-sensitive step)
# --------------------------------------------------------------------------- #


def fit_sigma_floors(
    envelopes: Iterable[TextEnvelope],
    train_texts: Sequence[str],
    *,
    percentile: float = 25.0,
    config: Config | None = None,
) -> dict[str, dict[str, float]]:
    """Fit ``sigma_floor[unit][feature]`` on the training texts only (spec 10.2).

    The floor for feature *k* is the ``percentile``-th percentile of
    :math:`\\sigma_{\\cdot k}` pooled over **every unit of every training text**.
    Pooling across texts rather than within one text is deliberate: a single
    text has too few units to estimate a stable low quantile, and the floor is
    meant to express "how precisely can this feature be reproduced at all",
    which is a property of the feature, not of the passage.

    Args:
        envelopes: Envelopes for every text in the fold, held-out included.
        train_texts: ``text_id`` of the fold's training texts. Any envelope
            whose ``text_id`` is not listed contributes nothing - that is the
            leakage guard (CLAUDE.md invariant 11).
        percentile: Which percentile to take; spec 10.2 fixes it at 25.
        config: When given, ``detection.envelope.sigma_floor_percentile``
            overrides ``percentile``.

    Returns:
        ``{unit: {feature: floor}}``. A feature with no finite sigma on the
        training texts gets ``0.0``, which disables the floor for it rather
        than silently flooring every z-score to zero.

    Raises:
        EnvelopeError: ``train_texts`` matched none of the envelopes.
    """
    if config is not None:
        percentile = float(config.detection["envelope"]["sigma_floor_percentile"])

    wanted = set(train_texts)
    pooled: dict[str, dict[str, list[float]]] = {}
    used: set[str] = set()

    for env in envelopes:
        if env.text_id not in wanted:
            continue
        used.add(env.text_id)
        for unit, ue in env.units.items():
            per_feature = pooled.setdefault(unit, {})
            features = ue.features
            for row in ue.envelope.sigma.values():
                for col, feature in enumerate(features):
                    value = float(row[col])
                    if np.isfinite(value):
                        per_feature.setdefault(feature, []).append(value)

    if not used:
        raise EnvelopeError(
            f"none of the envelopes supplied belongs to the training texts {sorted(wanted)}"
        )
    missing = sorted(wanted - used)
    if missing:
        logger.warning("no envelope supplied for training text(s) %s", missing)

    floors: dict[str, dict[str, float]] = {}
    for unit, per_feature in pooled.items():
        floors[unit] = {}
        for feature, values in sorted(per_feature.items()):
            arr = np.asarray(values, dtype=np.float64)
            floors[unit][feature] = float(np.percentile(arr, percentile)) if arr.size else 0.0
    logger.info(
        "sigma floors fitted on %d training text(s) %s",
        len(used),
        sorted(used),
    )
    return floors


def apply_sigma_floors(
    envelope: TextEnvelope, sigma_floors: Mapping[str, Mapping[str, float]]
) -> TextEnvelope:
    """Return ``envelope`` with ``sigma_floor`` set on each unit envelope.

    The reference values and medians are untouched; only
    :attr:`~rostrum.types.Envelope.sigma_floor` is filled, which is what
    :meth:`~rostrum.types.Envelope.effective_sigma` reads.
    """
    for unit, ue in envelope.units.items():
        floors = {
            feature: float(sigma_floors.get(unit, {}).get(feature, 0.0)) for feature in ue.features
        }
        ue.envelope.sigma_floor = floors
    return envelope


def build_fold_envelopes(
    references_by_text: Mapping[str, Sequence[str]],
    feature_tables: Mapping[str, Mapping[str, FeatureTable]] | Sequence[Reference],
    *,
    fold: str,
    train_texts: Sequence[str],
    units: Sequence[str] = ("word", "gap", "phrase"),
    index_maps: Mapping[str, Mapping[str, WordIndexMap]] | None = None,
    config: Config | None = None,
    frames: Mapping[str, FeatureTable] | None = None,
    word_spans: Mapping[str, Mapping[int, tuple[float, float]]] | None = None,
    skip_unbuildable: bool = False,
) -> dict[str, TextEnvelope]:
    """Build every text's envelope for one fold, with floors fitted on training.

    This is the ordering the leakage guard requires: build unfloored, fit the
    floors on ``train_texts`` alone, then apply those floors to every text
    including the held-out one.

    Args:
        references_by_text: ``{text_id: [reference file_id, ...]}``.
        feature_tables: As :func:`build_envelope`.
        fold: Fold tag, e.g. ``"loto_s1_gettysburg"``.
        train_texts: The fold's training texts.
        units: Unit kinds to build.
        index_maps: Per-recording canonical-index overrides.
        config: Configuration.
        frames: ``{file_id: frame table}`` to build overlay curves.
        word_spans: ``{file_id: {word_idx: (t0, t1)}}``, with ``frames``.
        skip_unbuildable: Log and skip a text with too few references instead of
            raising. Useful while the dataset is still being collected.

    Returns:
        ``{text_id: TextEnvelope}``, floors applied.

    Raises:
        EnvelopeError: A text cannot be built and ``skip_unbuildable`` is
            ``False``, or no text could be built at all.
    """
    built: dict[str, TextEnvelope] = {}
    for text_id in sorted(references_by_text):
        try:
            built[text_id] = build_envelope(
                text_id,
                references_by_text[text_id],
                feature_tables,
                fold=fold,
                units=units,
                index_maps=index_maps,
                config=config,
                frames=frames,
                word_spans=word_spans,
            )
        except EnvelopeError as exc:
            if not skip_unbuildable:
                raise
            logger.warning("%s: skipped (%s)", text_id, exc)

    if not built:
        raise EnvelopeError(f"fold {fold}: no text could be built")

    floors = fit_sigma_floors(built.values(), train_texts, config=config)
    for env in built.values():
        apply_sigma_floors(env, floors)

    for env in built.values():
        logger.info("%s", env.summary())
    return built


# --------------------------------------------------------------------------- #
# Leave one reference out
# --------------------------------------------------------------------------- #


def leave_one_out(envelope: TextEnvelope, file_id: str) -> TextEnvelope:
    """Recompute ``envelope`` without reader ``file_id`` (spec 11, confidence band).

    The rubric's confidence interval is the standard deviation of the category
    score over the R envelopes each built from R-1 readers (CLAUDE.md "Key
    formulas"). Because the per-reader values are stored, this is an exact
    recomputation of the median and MAD, not a jackknife approximation.

    The sigma floors are **carried over unchanged**: they were fitted on the
    training texts' spreads, and dropping one reader of one text does not refit
    them. Refitting here would also make the band depend on the floor, which is
    a fold-level quantity.

    Args:
        envelope: The full envelope.
        file_id: The reader to drop.

    Returns:
        A new :class:`TextEnvelope`; the input is not modified. Overlay curves
        are dropped, since the band is not plotted per leave-one-out replicate.

    Raises:
        KeyError: ``file_id`` is not one of this envelope's references.
        EnvelopeError: Dropping it would leave no reader.
    """
    if file_id not in envelope.reference_ids:
        raise KeyError(
            f"{file_id!r} is not a reference of {envelope.text_id}; "
            f"have {envelope.reference_ids}"
        )
    if len(envelope.reference_ids) <= 1:
        raise EnvelopeError(f"{envelope.text_id}: cannot leave out the only reference {file_id!r}")

    keep = [i for i, fid in enumerate(envelope.reference_ids) if fid != file_id]
    units: dict[str, UnitEnvelope] = {}

    for unit, ue in envelope.units.items():
        inner = ue.envelope
        # Each unit envelope may have been built from a subset of the readers,
        # so positions are resolved against its own reference_ids.
        rows = [i for i, fid in enumerate(inner.reference_ids) if fid != file_id]
        if not rows:
            continue
        values = {idx: arr[rows, :] for idx, arr in ue.values.items()}
        median: dict[int, list[float]] = {}
        sigma: dict[int, list[float]] = {}
        counts: dict[int, int] = {}
        for idx, arr in values.items():
            centre, spread, n = robust_centre_spread(arr, axis=0)
            median[idx] = [float(v) for v in centre]
            sigma[idx] = [float(v) for v in spread]
            counts[idx] = int(n.max(initial=0))
        units[unit] = UnitEnvelope(
            envelope=Envelope(
                text_id=inner.text_id,
                fold=inner.fold,
                unit=inner.unit,
                features=list(inner.features),
                median=median,
                sigma=sigma,
                sigma_floor=dict(inner.sigma_floor),
                n_references=counts,
                reference_ids=[inner.reference_ids[i] for i in rows],
                channel_ids=(
                    [inner.channel_ids[i] for i in rows]
                    if len(inner.channel_ids) == len(inner.reference_ids)
                    else list(inner.channel_ids)
                ),
            ),
            values=values,
            gap_labels=dict(ue.gap_labels),
        )

    if not units:
        raise EnvelopeError(
            f"{envelope.text_id}: leaving out {file_id!r} empties every unit envelope"
        )

    return TextEnvelope(
        text_id=envelope.text_id,
        fold=envelope.fold,
        units=units,
        curves=None,
        reference_ids=[envelope.reference_ids[i] for i in keep],
        channel_ids=(
            [envelope.channel_ids[i] for i in keep]
            if len(envelope.channel_ids) == len(envelope.reference_ids)
            else list(envelope.channel_ids)
        ),
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise_counts(envelopes: Mapping[str, TextEnvelope]) -> list[dict[str, Any]]:
    """One row per text: readers and unit counts, for the acceptance print-out.

    Returns:
        Rows sorted by ``text_id``, each with ``text_id``, ``fold``, ``n``,
        ``same_channel`` and a ``units`` sub-dict of unit counts and feature
        counts.
    """
    rows: list[dict[str, Any]] = []
    for text_id in sorted(envelopes):
        env = envelopes[text_id]
        rows.append(
            {
                "text_id": text_id,
                "fold": env.fold,
                "n": env.n_references,
                "same_channel": env.same_channel,
                "reference_ids": list(env.reference_ids),
                "units": {
                    unit: {
                        "n_units": len(ue.unit_indices),
                        "n_features": len(ue.features),
                    }
                    for unit, ue in sorted(env.units.items())
                },
            }
        )
    return rows
