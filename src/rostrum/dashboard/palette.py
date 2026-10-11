"""Colours, labels and badges for the dashboard (spec 15.2).

Accessibility is a requirement of prompt 22, not a nicety, and it comes down to
two rules this module exists to make unavoidable:

1. **Never colour alone.** Every colour defined here is paired with a text
   label and, where a shape can carry it, a hatch or a dash pattern. A judge
   with a colour deficiency, or looking at a greyscale print of the PDF, must
   be able to read the same information.
2. **A colour-blind-safe palette.** The six rubric categories use Okabe-Ito,
   which stays distinguishable under deuteranopia, protanopia and tritanopia,
   and is the same palette ``eval.figures`` uses, so the dashboard and the
   report's figures agree on what "pausing" looks like.

The deviation scale is sequential and perceptually ordered on lightness alone
(a single hue ramp from near-white to deep blue), so a word's tint reads as
"more deviant" even in greyscale; the numeric :math:`D_w` is always printed in
the hover as well.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

__all__ = [
    "CATEGORY_COLOURS",
    "CATEGORY_HATCH",
    "CATEGORY_LABELS",
    "DEVIATION_SCALE",
    "ENVELOPE_COLOUR",
    "LEVEL_LABELS",
    "MODE_BADGES",
    "PARTICIPANT_COLOUR",
    "category_colour",
    "deviation_colour",
    "level_label",
    "mode_badge",
    "text_on",
]

# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #

CATEGORY_COLOURS: dict[str, str] = {
    "pace": "#0072B2",
    "pausing": "#E69F00",
    "intonation": "#CC79A7",
    "emphasis": "#009E73",
    "energy": "#D55E00",
    "clarity": "#56B4E9",
}
"""Okabe-Ito, which stays distinguishable for the common colour deficiencies.

Shared with :mod:`eval.figures` so a figure in the report and the same region
on the dashboard are the same colour.
"""

CATEGORY_HATCH: dict[str, str] = {
    "pace": "/",
    "pausing": "\\",
    "intonation": "x",
    "emphasis": "-",
    "energy": "+",
    "clarity": ".",
}
"""A second, non-colour channel for each category, used in the PDF's patches."""

CATEGORY_LABELS: dict[str, str] = {
    "pace": "Pace",
    "pausing": "Pausing",
    "intonation": "Intonation",
    "emphasis": "Emphasis",
    "energy": "Energy",
    "clarity": "Clarity",
}
"""Display names, so a category is never identified by colour alone."""

UNKNOWN_COLOUR = "#555555"
"""For a category this module does not know; it is still labelled in text."""

PARTICIPANT_COLOUR = "#111111"
"""The uploaded reading: near-black, the figure's foreground."""

ENVELOPE_COLOUR = "#0072B2"
"""The skilled readers' median and band."""


def category_colour(category: str) -> str:
    """The colour of one rubric category, or a neutral grey."""
    return CATEGORY_COLOURS.get(str(category), UNKNOWN_COLOUR)


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #

LEVEL_LABELS: dict[str, str] = {
    "L0": "L0 - within the envelope",
    "L1": "L1 - barely perceptible",
    "L2": "L2 - clearly audible",
    "L3": "L3 - disruptive",
    "L4": "L4 - severe",
}
"""The severity ladder in words. The rungs are 0.00 / 0.15 / 0.35 / 0.60 / 0.90."""

LEVEL_OPACITY: dict[str, float] = {
    "L0": 0.10,
    "L1": 0.14,
    "L2": 0.20,
    "L3": 0.26,
    "L4": 0.32,
}
"""Shading weight of a region, so severity is visible without a second hue.

Kept low: the band and the curves underneath must stay readable through it.
"""


def level_label(level: str) -> str:
    """One severity rung in words, e.g. ``L2 - clearly audible``."""
    return LEVEL_LABELS.get(str(level), str(level))


def level_opacity(level: str) -> float:
    """Shading weight for one severity rung."""
    return LEVEL_OPACITY.get(str(level), 0.18)


# --------------------------------------------------------------------------- #
# Operating mode
# --------------------------------------------------------------------------- #

MODE_BADGES: dict[str, dict[str, str]] = {
    "reference": {
        "title": "Reference mode",
        "tone": "success",
        "detail": (
            "Compared word by word against the envelope of skilled readings of this " "exact text."
        ),
    },
    "norm": {
        "title": "Norm mode - lower confidence",
        "tone": "warning",
        "detail": (
            "No library text matched this transcript, so the comparison is against "
            "pooled norms over every skilled reading. Clarity and voice quality are "
            "measured but cannot raise a flaw, because they are only comparable "
            "within one recording channel."
        ),
    },
    "reject": {
        "title": "Rejected - the audio and the transcript do not match",
        "tone": "error",
        "detail": (
            "Nothing is scored. A rubric score computed against the wrong text would "
            "be a number with no meaning, so none is reported."
        ),
    },
}
"""What each operating mode must say on screen, per spec 13 and prompt 22."""


def mode_badge(mode: str, n_references: int = 0, reason: str = "") -> dict[str, str]:
    """The header badge for one operating mode.

    Args:
        mode: ``reference``, ``norm`` or ``reject``.
        n_references: Skilled readings behind the comparison; named in the
            reference badge, since "compared against an envelope" means little
            without knowing how many readers built it.
        reason: ``AnalysisResult.mode_reason``, appended when present.

    Returns:
        ``{"title", "tone", "detail"}`` - ``tone`` is one of ``success``,
        ``warning``, ``error``, mapping onto the Streamlit callout to use.
    """
    badge = dict(MODE_BADGES.get(str(mode), MODE_BADGES["norm"]))
    if mode == "reference":
        badge["title"] = f"Reference mode - {int(n_references)} skilled reader(s)"
    if reason:
        badge["detail"] = f"{badge['detail']} ({reason})"
    return badge


# --------------------------------------------------------------------------- #
# The deviation ramp
# --------------------------------------------------------------------------- #

DEVIATION_SCALE: tuple[tuple[float, str], ...] = (
    (0.00, "#F7FBFF"),
    (0.25, "#D6E6F4"),
    (0.50, "#A6CEE3"),
    (0.75, "#4A98C9"),
    (1.00, "#08519C"),
)
r"""Sequential ramp for :math:`D_w`, light to dark on one hue.

Monotonic in lightness, so the ordering survives greyscale printing and the
common colour deficiencies; the stops are the ColorBrewer *Blues* family.
"""


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """``#rrggbb`` to ``(r, g, b)``."""
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def deviation_colour(
    deviation: float | None,
    *,
    cap: float = 6.12,
    floor: float = 0.0,
) -> str:
    r"""The background tint for one word of the transcript.

    The ramp runs between two **anchors with meanings**, not between 0 and an
    arbitrary number. :math:`D_w = \sqrt{\sum_k \lambda_k z_{wk}^2}` is a sum
    over weighted features, so a word that is perfectly ordinary - every
    feature exactly 1 sigma from the readers' median - already scores
    :math:`\sqrt{\sum_k \lambda_k}`, which with six unit weights is 2.45.
    Starting the ramp at zero would therefore paint an unremarkable reading
    half-dark everywhere and leave no contrast for the words that matter.

    ``floor`` is that ordinary value and ``cap`` is where the ramp saturates;
    :class:`~rostrum.dashboard.view.AnalysisView` derives both from the
    detection configuration.

    Args:
        deviation: :math:`D_w` for that word, or ``None`` when nothing was
            measurable there - a real state, drawn as neutral grey rather than
            as "typical".
        cap: Saturation point. One pathological word would otherwise flatten
            every other word to white.
        floor: Where the ramp starts; below it a word is drawn as ordinary.

    Returns:
        A ``#rrggbb`` colour.
    """
    if deviation is None:
        return "#EFEFEF"
    try:
        value = float(deviation)
    except (TypeError, ValueError):
        return "#EFEFEF"
    if value != value:  # NaN
        return "#EFEFEF"

    span = float(cap) - float(floor)
    fraction = min(max((value - float(floor)) / span, 0.0), 1.0) if span > 0 else 0.0
    lo = DEVIATION_SCALE[0]
    hi = DEVIATION_SCALE[-1]
    for left, right in pairwise(DEVIATION_SCALE):
        if left[0] <= fraction <= right[0]:
            lo, hi = left, right
            break

    span = hi[0] - lo[0]
    t = 0.0 if span <= 0 else (fraction - lo[0]) / span
    a, b = _hex_to_rgb(lo[1]), _hex_to_rgb(hi[1])
    mixed = tuple(round(a[i] + t * (b[i] - a[i])) for i in range(3))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def text_on(background: str) -> str:
    """Black or white text, whichever is readable on ``background``.

    Uses the WCAG relative-luminance threshold, so a dark tint gets white text
    instead of unreadable black.
    """
    r, g, b = (c / 255.0 for c in _hex_to_rgb(background))

    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    luminance = 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)
    return "#000000" if luminance > 0.45 else "#FFFFFF"


def plotly_layout(height: int = 760) -> dict[str, Any]:
    """Shared Plotly layout: readable at 1280 px, legible fonts, light theme.

    Kept in one place so the overlay, the radar and the waveform are visibly
    one figure family rather than three defaults.
    """
    return {
        "height": int(height),
        "margin": {"l": 70, "r": 24, "t": 48, "b": 56},
        "font": {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "size": 13},
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FCFCFD",
        "hovermode": "closest",
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
        },
    }
