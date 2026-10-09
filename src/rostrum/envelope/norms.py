"""Pooled norms for an unknown text: Norm mode (spec 13).

In Reference mode word *i* of the upload is compared with word *i* of the same
text read by skilled readers, which is the comparison the whole system is
built on. An unknown text has no such anchor, and spec 13 says what to do
instead: compare against **pooled norms** - a rate band, a pitch range and
pause norms by gap label, computed across all skilled readings the library
has - and treat the result as lower confidence.

The pooling is deliberately crude where the text used to carry the
information:

* a word-level feature gets **one** median and spread for all words, because
  without the text there is no reason to expect this word to be slower than
  that one;
* a gap gets one median and spread **per gap label**, because the text still
  says whether a pause was invited there - that is punctuation, not identity;
* a phrase-level feature is pooled like a word-level one.

The result is an ordinary :class:`~rostrum.envelope.build.TextEnvelope` whose
statistics happen not to vary by unit, so z-scoring, region formation,
scoring and the cards work unchanged. Two things differ by construction:

* ``channel_ids`` holds every contributing channel, so
  :attr:`TextEnvelope.same_channel` is ``False`` and every channel-sensitive
  rule is skipped - spec 9.3's "reported but never trigger" (the features are
  still measured and shown, they just cannot fire a flaw);
* the sigma floor is the pooled spread's own 25th percentile across features,
  fitted on the pooled references only, which are training references by
  construction (CLAUDE.md invariant 11).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.envelope.build import (
    Reference,
    TextEnvelope,
    UnitEnvelope,
    robust_centre_spread,
)
from rostrum.features import names as N
from rostrum.types import Envelope

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.types import FeatureTable

logger = logging.getLogger(__name__)

__all__ = [
    "POOLED_TEXT_ID",
    "PooledNorms",
    "build_pooled_norms",
    "pooled_envelope",
]

POOLED_TEXT_ID: str = "__pooled__"
"""``text_id`` of an envelope built from pooled norms rather than one text."""

_GAP_LABEL_COLUMN = "gap_label"


@dataclass(frozen=True)
class PooledNorms:
    """One median and spread per feature, pooled over every reading.

    Attributes:
        units: ``{unit: {feature: (median, sigma, n)}}`` for word and phrase
            features.
        gaps: ``{gap_label: {feature: (median, sigma, n)}}``.
        reference_ids: Readings that contributed, sorted.
        channel_ids: Their channels, parallel to ``reference_ids``.
        sigma_floor: ``{unit: {feature: floor}}``, the 25th percentile of the
            pooled spreads.
    """

    units: Mapping[str, Mapping[str, tuple[float, float, int]]]
    gaps: Mapping[str, Mapping[str, tuple[float, float, int]]]
    reference_ids: tuple[str, ...] = ()
    channel_ids: tuple[str, ...] = ()
    sigma_floor: Mapping[str, Mapping[str, float]] = None  # type: ignore[assignment]

    @property
    def n_references(self) -> int:
        """Readings behind the norms."""
        return len(self.reference_ids)

    def features(self, unit: str) -> list[str]:
        """Pooled features of one unit, sorted."""
        if unit == "gap":
            return sorted({f for block in self.gaps.values() for f in block})
        return sorted(self.units.get(unit, {}))

    def to_dict(self) -> dict[str, Any]:
        """JSON form, for the report and the API's ``/library`` endpoint."""
        return {
            "text_id": POOLED_TEXT_ID,
            "n_references": self.n_references,
            "reference_ids": list(self.reference_ids),
            "units": {
                unit: {
                    feature: {"median": m, "sigma": s, "n": n}
                    for feature, (m, s, n) in sorted(block.items())
                }
                for unit, block in sorted(self.units.items())
            },
            "gaps": {
                label: {
                    feature: {"median": m, "sigma": s, "n": n}
                    for feature, (m, s, n) in sorted(block.items())
                }
                for label, block in sorted(self.gaps.items())
            },
        }


def _columns(table: FeatureTable, wanted: Sequence[str] | None) -> list[str]:
    """Numeric feature columns of a table, optionally restricted."""
    columns = N.numeric_feature_columns(table.columns)
    if wanted is None:
        return columns
    allowed = {c for name in wanted for c in N.resolve(name, table.columns)}
    return [c for c in columns if c in allowed]


def _pool(values: Sequence[np.ndarray]) -> tuple[float, float, int]:
    """Median, ``1.4826 * MAD`` and count over every value of every reading."""
    if not values:
        return float("nan"), float("nan"), 0
    stacked = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in values])
    finite = stacked[np.isfinite(stacked)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    centre, spread, _ = robust_centre_spread(finite[:, None], axis=0)
    return float(centre[0]), float(spread[0]), int(finite.size)


def _wanted_features(config: Config | None, unit: str) -> list[str] | None:
    """Envelope features for a unit, from ``detection.envelope.features``."""
    if config is None:
        return None
    block = dict(config.detection["envelope"].get("features", {}))
    names = block.get(unit)
    return [str(n) for n in names] if names else None


def build_pooled_norms(
    references: Sequence[Reference],
    *,
    config: Config | None = None,
    units: Sequence[str] = ("word", "gap", "phrase"),
    sigma_floor_percentile: float = 25.0,
) -> PooledNorms:
    """Pool every reading's feature values into one set of norms (spec 13).

    Args:
        references: The skilled readings to pool. Under leave-one-text-out
            these must be the fold's **training** references only; the caller
            selects them, as it does for an envelope.
        config: Supplies which features enter the comparison.
        units: Unit kinds to pool.
        sigma_floor_percentile: Percentile of the pooled spreads used as the
            floor, matching ``detection.envelope.sigma_floor_percentile``.

    Returns:
        A :class:`PooledNorms`.
    """
    if config is not None:
        sigma_floor_percentile = float(
            config.detection["envelope"].get("sigma_floor_percentile", sigma_floor_percentile)
        )

    per_unit: dict[str, dict[str, tuple[float, float, int]]] = {}
    per_gap: dict[str, dict[str, tuple[float, float, int]]] = {}

    for unit in units:
        wanted = _wanted_features(config, unit)
        columns: list[str] = []
        tables: list[FeatureTable] = []
        for reference in references:
            table = reference.tables.get(unit)
            if table is None or len(table) == 0:
                continue
            tables.append(table)
            for column in _columns(table, wanted):
                if column not in columns:
                    columns.append(column)
        if not tables:
            continue

        if unit == "gap":
            labels = sorted(
                {
                    str(value)
                    for table in tables
                    if _GAP_LABEL_COLUMN in table.columns
                    for value in table[_GAP_LABEL_COLUMN]
                }
            ) or ["none"]
            for label in labels:
                block: dict[str, tuple[float, float, int]] = {}
                for column in columns:
                    values = []
                    for table in tables:
                        if column not in table.columns:
                            continue
                        if _GAP_LABEL_COLUMN in table.columns:
                            rows = table.loc[table[_GAP_LABEL_COLUMN].astype(str) == label]
                        else:
                            rows = table
                        if len(rows):
                            values.append(rows[column].to_numpy(dtype=np.float64, na_value=np.nan))
                    block[column] = _pool(values)
                per_gap[label] = block
            continue

        block = {}
        for column in columns:
            block[column] = _pool(
                [
                    table[column].to_numpy(dtype=np.float64, na_value=np.nan)
                    for table in tables
                    if column in table.columns
                ]
            )
        per_unit[unit] = block

    floors: dict[str, dict[str, float]] = {}
    for unit, block in per_unit.items():
        spreads = np.asarray([s for _, s, _ in block.values() if np.isfinite(s)], dtype=np.float64)
        floor = float(np.percentile(spreads, sigma_floor_percentile)) if spreads.size else 0.0
        floors[unit] = dict.fromkeys(block, floor)
    if per_gap:
        spreads = np.asarray(
            [s for block in per_gap.values() for _, s, _ in block.values() if np.isfinite(s)],
            dtype=np.float64,
        )
        floor = float(np.percentile(spreads, sigma_floor_percentile)) if spreads.size else 0.0
        floors["gap"] = dict.fromkeys({f for block in per_gap.values() for f in block}, floor)

    norms = PooledNorms(
        units=per_unit,
        gaps=per_gap,
        reference_ids=tuple(sorted(r.file_id for r in references)),
        channel_ids=tuple(sorted({r.channel_id for r in references if r.channel_id})),
        sigma_floor=floors,
    )
    logger.info(
        "pooled norms from %d reading(s): %s",
        norms.n_references,
        ", ".join(f"{u}={len(norms.features(u))} feat" for u in units if norms.features(u)),
    )
    return norms


def pooled_envelope(
    norms: PooledNorms,
    tables: Mapping[str, FeatureTable],
    *,
    fold: str = "all",
) -> TextEnvelope:
    """Project pooled norms onto one upload's units (spec 13).

    Every unit of the upload is given the pooled median and spread for its
    kind - a gap gets the norms for *its own* label - so the comparison
    machinery of Reference mode runs unchanged.

    Args:
        norms: The pooled norms.
        tables: ``{unit: FeatureTable}`` of the recording being analysed; the
            unit indices to cover come from here.
        fold: Fold tag to record.

    Returns:
        A :class:`~rostrum.envelope.build.TextEnvelope` with
        ``text_id = "__pooled__"``.
    """
    units: dict[str, UnitEnvelope] = {}
    n_refs = max(norms.n_references, 1)

    for unit, table in tables.items():
        features = [f for f in norms.features(unit) if f in table.columns]
        if not features or len(table) == 0:
            continue
        median: dict[int, list[float]] = {}
        sigma: dict[int, list[float]] = {}
        counts: dict[int, int] = {}
        values: dict[int, np.ndarray] = {}
        gap_labels: dict[int, str] = {}

        for unit_idx in table.index:
            idx = int(unit_idx)
            if unit == "gap":
                label = (
                    str(table.loc[unit_idx, _GAP_LABEL_COLUMN])
                    if _GAP_LABEL_COLUMN in table.columns
                    else "none"
                )
                gap_labels[idx] = label
                block = norms.gaps.get(label) or norms.gaps.get("none") or {}
            else:
                block = norms.units.get(unit, {})
            median[idx] = [float(block.get(f, (np.nan, np.nan, 0))[0]) for f in features]
            sigma[idx] = [float(block.get(f, (np.nan, np.nan, 0))[1]) for f in features]
            counts[idx] = n_refs
            # The pooled norms keep no per-reader values, so the "k of n
            # readers" context line has nothing to count; the card falls back
            # to the median-and-spread sentence.
            values[idx] = np.full((n_refs, len(features)), np.nan, dtype=np.float64)

        floors = dict(norms.sigma_floor.get(unit, {})) if norms.sigma_floor else {}
        units[unit] = UnitEnvelope(
            envelope=Envelope(
                text_id=POOLED_TEXT_ID,
                fold=fold,
                unit=unit,  # type: ignore[arg-type]
                features=list(features),
                median=median,
                sigma=sigma,
                sigma_floor={f: float(floors.get(f, 0.0)) for f in features},
                n_references=counts,
                reference_ids=list(norms.reference_ids),
                channel_ids=list(norms.channel_ids),
            ),
            values=values,
            gap_labels=gap_labels,
        )

    envelope = TextEnvelope(
        text_id=POOLED_TEXT_ID,
        fold=fold,
        units=units,
        reference_ids=list(norms.reference_ids),
        # Several channels by construction, so same_channel is False and the
        # channel-sensitive rules stay informational (spec 9.3, 13).
        channel_ids=list(norms.channel_ids),
    )
    logger.info("pooled envelope over %s", ", ".join(sorted(units)) or "nothing")
    return envelope
