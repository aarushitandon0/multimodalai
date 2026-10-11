"""The dashboard's pure layer: palette, figures, transcript and report (spec 15.2).

``app/streamlit_app.py`` is the screens - uploads, buttons, session state - and
is only runnable under Streamlit. Everything it *draws* lives here instead, as
plain functions over an :class:`~rostrum.types.AnalysisResult`, for three
reasons:

* it is unit-testable without a browser, so the figures are covered by
  ``make test`` like any other module;
* the same functions render the PDF export and the figures in ``docs/figures``,
  so the report and the screen cannot disagree;
* a result read back from the cache draws identically to one just computed -
  nothing here measures anything, it only reads what the pipeline recorded.

No function in this package computes a number that a score depends on. The
deviation values, the band, the regions and the scores are all taken from the
result as the pipeline rounded them (CLAUDE.md invariant 12).
"""

from __future__ import annotations

from rostrum.dashboard.figures import overlay_figure, radar_figure, waveform_figure
from rostrum.dashboard.palette import (
    CATEGORY_COLOURS,
    CATEGORY_HATCH,
    LEVEL_LABELS,
    deviation_colour,
    mode_badge,
)
from rostrum.dashboard.report import write_pdf
from rostrum.dashboard.service import Backend, BackendError, make_backend
from rostrum.dashboard.transcript import (
    TranscriptWord,
    deviation_legend_html,
    transcript_html,
    transcript_words,
)
from rostrum.dashboard.view import PANELS, AnalysisView, PanelSpec, overlay_panels

__all__ = [
    "CATEGORY_COLOURS",
    "CATEGORY_HATCH",
    "LEVEL_LABELS",
    "PANELS",
    "AnalysisView",
    "Backend",
    "BackendError",
    "PanelSpec",
    "TranscriptWord",
    "deviation_colour",
    "deviation_legend_html",
    "make_backend",
    "mode_badge",
    "overlay_figure",
    "overlay_panels",
    "radar_figure",
    "transcript_html",
    "transcript_words",
    "waveform_figure",
    "write_pdf",
]
