"""Canonical feature names and the columns they resolve to (spec 9.1, 9.3).

The configuration files talk about features in short, stable names -
``rate_local``, ``gap_duration``, ``energy_rel_z``, ``cpp`` - because those
names appear in the spec's trigger table (spec 10.3), in the rubric's
``primary_features`` (spec 11) and in the channel rule (spec 9.3). The feature
tables, in contrast, carry the *measured* column with its unit baked into the
name: ``rate_local_syll_per_s``, ``gap_duration_s``, ``energy_mean_z``,
``cpps_db``.

Both naming schemes are worth keeping. A config that said
``spectral_tilt_db_per_khz`` would break the moment the tilt unit changed, and a
table column called ``cpp`` would hide which of the several published cepstral
peak measures was computed. This module is the one place that maps between
them, so no other module has to guess.

Three things are looked up here:

* :func:`resolve` - canonical name to the table column(s) that carry it. Most
  names resolve to exactly one column; ``mfcc`` resolves to the whole cepstral
  block, which is why the return type is a list.
* :func:`canonical` - the inverse, used to label evidence rows and tuning
  reports with the name a reader of the spec will recognise.
* :func:`is_channel_sensitive` - whether a feature may only be compared within
  one channel (CLAUDE.md invariant 7). The answer comes from
  ``configs/features.yaml``'s ``channel_rule``, never from a literal here.

Everything in this module is a pure lookup over module-level tables and a
:class:`~rostrum.config.Config`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "ALIASES",
    "BOOKKEEPING_COLUMNS",
    "DISPLAY_UNITS",
    "IDENTITY_COLUMNS",
    "MFCC_PREFIX",
    "canonical",
    "channel_robust_names",
    "channel_sensitive_names",
    "display_unit",
    "is_channel_sensitive",
    "numeric_feature_columns",
    "resolve",
    "resolve_many",
]


MFCC_PREFIX: str = "mfcc_"
"""Every cepstral column starts with this; ``mfcc_n_frames`` is bookkeeping."""

ALIASES: dict[str, tuple[str, ...]] = {
    # --- timing and rate (channel-robust) ---------------------------------
    "word_duration": ("word_duration_s",),
    "duration_per_phone": ("duration_per_phone_s",),
    "duration_per_syllable": ("duration_per_syllable_s",),
    # spec 10.3 calls the mumbling component "vowel duration"; the measured
    # quantity is the word's duration divided by its vowel-nucleus count,
    # which is per-nucleus duration - see features/duration.py.
    "vowel_duration_norm": ("duration_per_syllable_s",),
    "duration_rel_median": ("duration_rel_median",),
    "rate_local": ("rate_local_syll_per_s",),
    "rate_relative": ("rate_relative",),
    "gap_duration": ("gap_duration_s",),
    # --- pitch (channel-robust: semitones re speaker median) --------------
    "f0_semitones": ("f0_mean_st",),
    "f0_mean_st": ("f0_mean_st",),
    "f0_max_st": ("f0_max_st",),
    "f0_min_st": ("f0_min_st",),
    "f0_range_st": ("f0_range_st",),
    "f0_std_st": ("f0_std_st",),
    "f0_slope_st_s": ("f0_slope_st_per_s",),
    # --- energy (channel-robust: dB re voiced median, then z) -------------
    "energy_rel_z": ("energy_mean_z",),
    "energy_peak_z": ("energy_peak_z",),
    "energy_mean_db_rel": ("energy_mean_db_rel",),
    "energy_slope_db_s": ("energy_slope_db_per_s",),
    # --- prominence (channel-robust) --------------------------------------
    "prominence": ("prominence",),
    "prominence_peak_rel": ("prominence_peak_rel",),
    # --- spectral (channel-sensitive) -------------------------------------
    "spectral_tilt": ("spectral_tilt_db_per_khz",),
    "spectral_centroid": ("spectral_centroid_hz",),
    "spectral_flux": ("spectral_flux",),
    "band_energy_ratio": ("band_energy_ratio_db",),
    "spectral_rolloff": ("spectral_rolloff_hz",),
    # --- voice quality (channel-sensitive) --------------------------------
    "hnr": ("hnr_db",),
    "jitter_local": ("jitter_local",),
    "shimmer_local": ("shimmer_local",),
    "cpp": ("cpps_db",),
    # --- phrase-level -----------------------------------------------------
    "phrase_f0_range_st": ("phrase_f0_range_st",),
    "phrase_f0_max_st": ("phrase_f0_max_st",),
    "phrase_rate": ("phrase_rate_syll_per_s",),
    "phrase_duration": ("phrase_duration_s",),
    # ``mfcc`` is a block, not a column; see _expand_mfcc.
    "mfcc": (MFCC_PREFIX,),
}
"""Canonical name to the feature-table column(s) carrying it.

The ``mfcc`` entry holds a *prefix*, expanded against the table's real columns
by :func:`resolve`, because the number of cepstral columns depends on whether
deltas were requested.
"""

DISPLAY_UNITS: dict[str, str] = {
    "word_duration": "s",
    "duration_per_phone": "s",
    "duration_per_syllable": "s",
    "vowel_duration_norm": "s",
    "duration_rel_median": "x median",
    "rate_local": "syll/s",
    "rate_relative": "x speaker rate",
    "gap_duration": "s",
    "f0_semitones": "st",
    "f0_mean_st": "st",
    "f0_max_st": "st",
    "f0_min_st": "st",
    "f0_range_st": "st",
    "f0_std_st": "st",
    "f0_slope_st_s": "st/s",
    "energy_rel_z": "z",
    "energy_peak_z": "z",
    "energy_mean_db_rel": "dB",
    "energy_slope_db_s": "dB/s",
    "prominence": "",
    "prominence_peak_rel": "fraction of word",
    "spectral_tilt": "dB/kHz",
    "spectral_centroid": "Hz",
    "spectral_flux": "",
    "band_energy_ratio": "dB",
    "spectral_rolloff": "Hz",
    "hnr": "dB",
    "jitter_local": "",
    "shimmer_local": "",
    "cpp": "dB",
    "phrase_f0_range_st": "st",
    "phrase_f0_max_st": "st",
    "phrase_rate": "syll/s",
    "phrase_duration": "s",
    "mfcc": "",
}
"""Human-readable unit per canonical name, for :class:`~rostrum.types.Evidence`."""

IDENTITY_COLUMNS: frozenset[str] = frozenset(
    {
        "text",
        "t_start",
        "t_end",
        "gap_label",
        "gap_is_boundary",
        "gap_is_pause",
        "gap_vad_refined",
        "gap_prev_word_idx",
        "gap_next_word_idx",
        "first_word_idx",
        "last_word_idx",
    }
)
"""Columns that identify a unit rather than measure it; never averaged."""

BOOKKEEPING_COLUMNS: frozenset[str] = frozenset(
    {
        "f0_n_voiced",
        "f0_voiced_frac",
        "energy_n_frames",
        "spectral_n_frames",
        "prominence_n_frames",
        "mfcc_n_frames",
        "vq_n_periods",
        "n_phones",
        "n_syllables",
        "rate_window_syllables",
        "rate_window_span_s",
        "phrase_n_words",
    }
)
"""Counts and coverage fractions: diagnostics, not comparable quantities.

They are deliberately excluded from an envelope. A z-score on "number of
voiced frames" would fire on every word a reader said faster, which is already
measured properly by ``rate_local``.
"""

# Reverse index, built once. ``mfcc`` is omitted because its value is a prefix.
_BY_COLUMN: dict[str, str] = {}
for _name, _cols in ALIASES.items():
    for _col in _cols:
        if _col == MFCC_PREFIX:
            continue
        # First alias wins, so the canonical name a reader of the spec expects
        # (``energy_rel_z``) beats a pass-through synonym declared later.
        _BY_COLUMN.setdefault(_col, _name)
del _name, _cols, _col


def _expand_mfcc(columns: Iterable[str]) -> list[str]:
    """Return the cepstral columns of ``columns``, bookkeeping excluded."""
    return sorted(c for c in columns if c.startswith(MFCC_PREFIX) and c not in BOOKKEEPING_COLUMNS)


def resolve(name: str, columns: Sequence[str] | None = None) -> list[str]:
    """Return the table column(s) carrying the canonical feature ``name``.

    Args:
        name: A canonical name from :data:`ALIASES`, or a column name already
            (so a caller may pass either and get a sensible answer).
        columns: The table's columns. Required to expand ``mfcc``; when given,
            columns the table does not actually have are dropped.

    Returns:
        Column names, in a stable order. Empty when the feature is absent from
        ``columns``.

    Raises:
        KeyError: ``name`` is neither a canonical name nor a column in
            ``columns``.

    Example:
        >>> resolve("rate_local")
        ['rate_local_syll_per_s']
    """
    available = None if columns is None else list(columns)

    if name in ALIASES:
        targets = ALIASES[name]
        if targets == (MFCC_PREFIX,):
            if available is None:
                raise KeyError("resolving 'mfcc' needs the table's columns")
            return _expand_mfcc(available)
        out = list(targets)
    elif available is not None and name in available:
        out = [name]
    elif available is None:
        raise KeyError(f"unknown feature name {name!r}; not in ALIASES")
    else:
        raise KeyError(f"unknown feature name {name!r}; not in ALIASES and not a column")

    if available is None:
        return out
    return [c for c in out if c in available]


def resolve_many(
    names: Iterable[str], columns: Sequence[str] | None = None, *, strict: bool = False
) -> list[str]:
    """Resolve several canonical names, de-duplicated, order preserved.

    Args:
        names: Canonical names or column names.
        columns: The table's columns, as for :func:`resolve`.
        strict: Raise when a name resolves to nothing. When ``False`` (the
            default) a missing feature is logged at debug level and skipped,
            which is what an optional feature block needs.

    Returns:
        Column names present in ``columns``.

    Raises:
        KeyError: ``strict`` and a name resolved to no column.
    """
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        try:
            cols = resolve(name, columns)
        except KeyError:
            if strict:
                raise
            logger.debug("feature %r is not in this table; skipping", name)
            continue
        if not cols and strict:
            raise KeyError(f"feature {name!r} resolved to no column in this table")
        if not cols:
            logger.debug("feature %r is not in this table; skipping", name)
        for c in cols:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def canonical(column: str) -> str:
    """Return the canonical name for a table ``column``.

    Falls back to ``"mfcc"`` for any cepstral column and to ``column`` itself
    for anything unregistered, so this never raises: it is used while rendering
    explanation cards, where a missing alias must not lose a measurement.

    Example:
        >>> canonical("cpps_db")
        'cpp'
    """
    if column in _BY_COLUMN:
        return _BY_COLUMN[column]
    if column.startswith(MFCC_PREFIX) and column not in BOOKKEEPING_COLUMNS:
        return "mfcc"
    return column


def display_unit(column_or_name: str) -> str:
    """Return the human-readable unit for a column or canonical name."""
    return DISPLAY_UNITS.get(canonical(column_or_name), "")


def channel_sensitive_names(config: Config) -> frozenset[str]:
    """Canonical names the channel rule marks as channel-sensitive (spec 9.3)."""
    rule = config.features["channel_rule"]
    return frozenset(str(n) for n in rule["channel_sensitive"])


def channel_robust_names(config: Config) -> tuple[str, ...]:
    """Canonical names the channel rule marks as channel-robust (spec 9.3)."""
    rule = config.features["channel_rule"]
    return tuple(str(n) for n in rule["channel_robust"])


def is_channel_sensitive(column_or_name: str, config: Config) -> bool:
    """Whether this feature may only be compared within one channel.

    Args:
        column_or_name: A canonical name or a table column.
        config: Provides ``features.channel_rule``; nothing is hard-coded here.

    Returns:
        ``True`` when the channel rule lists the feature as channel-sensitive
        (CLAUDE.md invariant 7).
    """
    return canonical(column_or_name) in channel_sensitive_names(config)


def numeric_feature_columns(columns: Iterable[str]) -> list[str]:
    """Filter ``columns`` down to comparable measurements.

    Drops the identity columns a table carries for convenience
    (:data:`IDENTITY_COLUMNS`) and the counts that are diagnostics rather than
    quantities (:data:`BOOKKEEPING_COLUMNS`). Order is preserved so the column
    order of an envelope is the column order of the table it came from.
    """
    return [c for c in columns if c not in IDENTITY_COLUMNS and c not in BOOKKEEPING_COLUMNS]
