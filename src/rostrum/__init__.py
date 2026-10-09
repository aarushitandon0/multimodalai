"""Rostrum - contrastive speech analytics and temporal flaw grounding.

Rostrum compares a spoken delivery against an envelope of skilled readings of
the *same text*, grounds each deviation to a start/end timestamp, and explains
it with the numbers that caused it.

Typical use::

    from rostrum import load_config
    from rostrum.pipeline import analyze

    cfg = load_config()
    result = analyze("take.wav", "transcript.txt", text_id="gettysburg")
    print(result.score.overall, [r.label for r in result.regions])

Only :mod:`rostrum.config` and :mod:`rostrum.types` are imported eagerly, so
``import rostrum`` stays cheap and does not pull in torch or librosa.
"""

from __future__ import annotations

from rostrum.config import Config, load_config, set_seeds
from rostrum.types import (
    Alignment,
    AnalysisResult,
    CategoryScore,
    Envelope,
    Evidence,
    Gap,
    Region,
    RubricScore,
    Word,
)

__version__ = "0.1.0"

__all__ = [
    "Alignment",
    "AnalysisResult",
    "CategoryScore",
    "Config",
    "Envelope",
    "Evidence",
    "Gap",
    "Region",
    "RubricScore",
    "Word",
    "__version__",
    "load_config",
    "set_seeds",
]
