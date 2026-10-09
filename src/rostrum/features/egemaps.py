"""eGeMAPSv02 functionals per phrase via openSMILE (spec 9.1 EG) [R13].

The extended Geneva Minimalistic Acoustic Parameter Set is the standard
88-dimensional affective-speech descriptor: functionals (means, percentiles,
slopes, rise/fall statistics) over a fixed low-level set. Rostrum computes its
own prosodic features word by word, so eGeMAPS is not used for grounding - it
is a **phrase-level descriptor**, reported for the record and used as input to
the flaw-type classifier, where a conventional feature set is a useful control
against the hand-built ones.

A phrase is the span between two ``boundary`` gaps
(``configs/features.yaml``: ``phrase_definition``). Spans come from the caller;
:func:`rostrum.features.extract_all.phrase_spans` derives them from an
alignment.

**Channel rule (CLAUDE.md invariant 7).** eGeMAPS mixes channel-robust
descriptors (F0, rate-like functionals) with strongly channel-sensitive ones
(spectral slopes, formant amplitudes, HNR). The set as a whole is therefore
treated as channel sensitive and never compared across recordings from
different sources.

**openSMILE availability.** ``opensmile`` ships a compiled extension, and
loading it is the one part of the feature stack that can fail for reasons
outside the code: a Windows application-control policy refused the DLL once on
the development machine (``DLL load failed while importing _fs``) before
allowing it. Rather than let that take the whole build down, these functions log
once and return a table carrying the reason in ``attrs["egemaps_error"]``, with
``attrs["egemaps_available"] = False`` and no feature columns. Callers that
require the values pass ``strict=True`` or check :func:`available` first, and
the cache key records whether the block was computed, so a partial entry is
never served as a complete one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config
    from rostrum.types import FeatureTable

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FEATURE_LEVEL",
    "DEFAULT_FEATURE_SET",
    "EgemapsUnavailable",
    "available",
    "feature_names",
    "phrase_egemaps",
    "unavailable_reason",
]

DEFAULT_FEATURE_SET: str = "eGeMAPSv02"
"""openSMILE feature set name (spec 9.1 EG)."""

DEFAULT_FEATURE_LEVEL: str = "Functionals"
"""Functionals rather than low-level descriptors: one row per phrase."""

MIN_PHRASE_S: float = 0.100
"""Shortest phrase openSMILE is asked to analyse."""


class EgemapsUnavailable(RuntimeError):
    """openSMILE could not be loaded in this environment."""


@lru_cache(maxsize=1)
def _smile(feature_set: str, feature_level: str) -> Any:
    """Build and cache one openSMILE extractor.

    Raises:
        EgemapsUnavailable: The ``opensmile`` package cannot be imported.
    """
    try:
        import opensmile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise EgemapsUnavailable(f"opensmile is not importable: {exc}") from exc
    try:
        return opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel[feature_level],
            num_workers=1,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise EgemapsUnavailable(f"opensmile could not build {feature_set}: {exc}") from exc


def unavailable_reason(
    feature_set: str = DEFAULT_FEATURE_SET, feature_level: str = DEFAULT_FEATURE_LEVEL
) -> str | None:
    """Why openSMILE cannot run here, or ``None`` when it can."""
    try:
        _smile(feature_set, feature_level)
    except EgemapsUnavailable as exc:
        return str(exc)
    return None


def available(
    feature_set: str = DEFAULT_FEATURE_SET, feature_level: str = DEFAULT_FEATURE_LEVEL
) -> bool:
    """Whether openSMILE can be used in this environment."""
    return unavailable_reason(feature_set, feature_level) is None


def feature_names(
    feature_set: str = DEFAULT_FEATURE_SET, feature_level: str = DEFAULT_FEATURE_LEVEL
) -> list[str]:
    """The feature set's column names, prefixed ``eg_``.

    Raises:
        EgemapsUnavailable: openSMILE cannot be loaded.
    """
    smile = _smile(feature_set, feature_level)
    return [f"eg_{name}" for name in smile.feature_names]


def _egemaps_conf(config: Config | None) -> tuple[str, str]:
    """``(feature_set, feature_level)`` from ``configs/features.yaml``."""
    if config is None:
        return DEFAULT_FEATURE_SET, DEFAULT_FEATURE_LEVEL
    section = config.features.get("egemaps", {})
    return (
        str(section.get("feature_set", DEFAULT_FEATURE_SET)),
        str(section.get("feature_level", DEFAULT_FEATURE_LEVEL)),
    )


def phrase_egemaps(
    samples: np.ndarray,
    sr: int,
    spans: Sequence[tuple[float, float]],
    *,
    config: Config | None = None,
    strict: bool = False,
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """eGeMAPS functionals for each phrase span.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        spans: ``(t_start, t_end)`` per phrase, in phrase-index order.
        config: Loaded config; ``configs/features.yaml`` ``egemaps:``.
        strict: Raise when openSMILE is unavailable instead of returning an
            empty table. Set this in the dataset build, where a silently
            missing feature set would be worse than a failure.
        file_id: File tag for the table.
        channel_id: Channel tag for the table.

    Returns:
        A table indexed by phrase index. One column per functional, prefixed
        ``eg_``. When openSMILE is unavailable and ``strict`` is false the table
        has the right index and no feature columns, and ``attrs`` records why.

    Raises:
        EgemapsUnavailable: ``strict`` is set and openSMILE cannot be loaded.
    """
    import pandas as pd

    feature_set, feature_level = _egemaps_conf(config)
    index = pd.Index(range(len(spans)), name="unit_idx", dtype="int64")
    attrs: dict[str, Any] = {
        "unit": "phrase",
        "file_id": file_id,
        "channel_id": channel_id,
        "channel_sensitive": True,
        "egemaps_feature_set": feature_set,
        "egemaps_available": True,
    }

    reason = unavailable_reason(feature_set, feature_level)
    if reason is not None:
        if strict:
            raise EgemapsUnavailable(reason)
        logger.warning("eGeMAPS skipped: %s", reason)
        table = pd.DataFrame(index=index)
        attrs.update(egemaps_available=False, egemaps_error=reason)
        table.attrs.update(attrs)
        return table

    smile = _smile(feature_set, feature_level)
    names = [f"eg_{name}" for name in smile.feature_names]
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    rows = np.full((len(spans), len(names)), np.nan, dtype=np.float64)
    for row, (t0, t1) in enumerate(spans):
        lo = max(0, round(float(t0) * sr))
        hi = min(x.size, round(float(t1) * sr))
        if hi - lo < int(MIN_PHRASE_S * sr):
            continue
        try:
            out = smile.process_signal(x[lo:hi].reshape(1, -1), sr)
        except Exception as exc:  # pragma: no cover - openSMILE rejects odd input
            logger.debug("eGeMAPS failed on phrase %d (%.3f-%.3f s): %s", row, t0, t1, exc)
            continue
        rows[row] = np.asarray(out.to_numpy(), dtype=np.float64).reshape(-1)[: len(names)]

    table = pd.DataFrame(rows, columns=names, index=index)
    table.attrs.update(attrs)
    return table
