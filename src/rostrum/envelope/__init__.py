"""rostrum.envelope -- Acceptable-variance envelopes over skilled readings of one text.

The public surface is :mod:`rostrum.envelope.build`; see spec 10.2.
"""

from rostrum.envelope.build import (
    CurveEnvelope,
    EnvelopeError,
    Reference,
    TextEnvelope,
    UnitEnvelope,
    WordIndexMap,
    apply_sigma_floors,
    build_curve_envelope,
    build_envelope,
    build_fold_envelopes,
    fit_sigma_floors,
    leave_one_out,
    summarise_counts,
    word_curves,
)

__all__ = [
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
    "summarise_counts",
    "word_curves",
]
