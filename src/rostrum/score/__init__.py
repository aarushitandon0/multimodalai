"""rostrum.score -- Rubric scoring and the leave-one-reference-out confidence band.

The public surface is :mod:`rostrum.score.rubric`; see spec 11.
"""

from rostrum.score.rubric import (
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    CalibrationCase,
    build_descriptors,
    calibrate_k,
    category_order,
    category_score,
    confidence_band,
    descriptors,
    deviation_mass,
    k_from_config,
    label_to_category,
    rubric_score,
    suppressed_from_skipped,
)

__all__ = [
    "CATEGORY_LABELS",
    "CATEGORY_ORDER",
    "CalibrationCase",
    "build_descriptors",
    "calibrate_k",
    "category_order",
    "category_score",
    "confidence_band",
    "descriptors",
    "deviation_mass",
    "k_from_config",
    "label_to_category",
    "rubric_score",
    "suppressed_from_skipped",
]
