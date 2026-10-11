"""Plotly figures for the dashboard (spec 15.2).

Three figures, each built from an :class:`~rostrum.dashboard.view.AnalysisView`
and nothing else:

:func:`overlay_figure`
    The main evidence picture - three panels sharing one time axis, the
    participant's curve against the skilled readers' median and +/- 1 sigma
    band, with every flaw region shaded in its category's colour.
:func:`radar_figure`
    The six rubric categories at a glance, with the confidence band drawn as a
    second, lighter outline rather than left implicit.
:func:`waveform_figure`
    The recording itself with the same regions marked, so a judge can see
    where in the speech they are before reading any number.

Design rules applied throughout, from prompt 22:

* **Context, not clips.** The x-axis is the whole recording even when every
  region sits in its first ten seconds (spec 15.2, [R43]).
* **Never colour alone.** Each region carries its category *and* severity as
  text in the hover and in a label; each category present gets a legend entry,
  because a shaded rectangle does not appear in a Plotly legend by itself.
* **Breaks over pauses.** The series arrive with ``NaN`` separators between
  words (see :mod:`rostrum.dashboard.view`), and ``connectgaps`` is left off
  so a silence is drawn as a silence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.dashboard.palette import (
    CATEGORY_LABELS,
    ENVELOPE_COLOUR,
    PARTICIPANT_COLOUR,
    category_colour,
    level_label,
    level_opacity,
    plotly_layout,
)
from rostrum.dashboard.view import AnalysisView, overlay_panels

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    import plotly.graph_objects as go

    from rostrum.types import Region

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_BOUNDARY_TICKS",
    "overlay_figure",
    "radar_figure",
    "region_hover",
    "waveform_figure",
]

MAX_BOUNDARY_TICKS = 160
"""Above this many words the boundary ticks are thinned, or they become a smear."""


def _go() -> Any:
    """Import ``plotly.graph_objects`` lazily, so importing this module is cheap."""
    import plotly.graph_objects as go

    return go


def _rgba(colour: str, alpha: float) -> str:
    """``#rrggbb`` plus an alpha, as Plotly wants it for a fill."""
    text = colour.lstrip("#")
    r, g, b = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.3f})"


# --------------------------------------------------------------------------- #
# Hover text
# --------------------------------------------------------------------------- #


def _band_polygon(series: Any) -> tuple[np.ndarray, np.ndarray]:
    """Closed polygon vertices for a band, **one subpath per word**.

    The series carries a ``NaN`` between words so the line breaks over pauses.
    Naively building ``[t, t[::-1]]`` for a ``toself`` fill then produces a
    single path whose forward and reversed halves have their gaps in mirrored
    - not matching - places, and Plotly closes the mismatched subpaths into
    spurious triangles spanning the silences.

    Emitting each word as its own closed loop, separated by ``NaN``, gives one
    polygon per word and no fill across a pause, which is also what the band
    means: the readers' spread is defined on words, not on the silences
    between them.

    Returns:
        ``(x, y)`` ready for a single ``fill="toself"`` trace.
    """
    t = np.asarray(series.t, dtype=np.float64)
    lower = np.asarray(series.lower, dtype=np.float64)
    upper = np.asarray(series.upper, dtype=np.float64)
    if t.size == 0 or lower.size != t.size or upper.size != t.size:
        return np.empty(0), np.empty(0)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gap = np.array([np.nan])
    start = 0
    breaks = [*np.flatnonzero(~np.isfinite(t)).tolist(), t.size]
    for stop in breaks:
        block = slice(start, stop)
        start = stop + 1
        bt, bl, bu = t[block], lower[block], upper[block]
        keep = np.isfinite(bt) & np.isfinite(bl) & np.isfinite(bu)
        if keep.sum() < 2:
            continue
        bt, bl, bu = bt[keep], bl[keep], bu[keep]
        xs.extend([bt, bt[::-1], gap])
        ys.extend([bu, bl[::-1], gap])

    if not xs:
        return np.empty(0), np.empty(0)
    return np.concatenate(xs), np.concatenate(ys)


def region_hover(region: Region, card: dict[str, Any] | None = None) -> str:
    """The hover text for one flaw region: what, when, how certain, and why.

    Everything in it is already in the result. The top evidence row is included
    because the number is the explanation - a region that says "rushing" without
    "4.9 sigma above the readers' 3.8 syll/s" is an opinion.
    """
    card = card or {}
    lines = [
        f"<b>{CATEGORY_LABELS.get(region.category, region.category)}: {region.label}</b>",
        f"{region.t_start:.3f} - {region.t_end:.3f} s " f"({region.t_end - region.t_start:.2f} s)",
        level_label(region.level),
        f"confidence {region.confidence:.2f} - {int(region.envelope_n)} reader(s)",
    ]
    if region.words:
        words = region.words if len(region.words) <= 70 else region.words[:67] + "..."
        lines.append(f"&ldquo;{words}&rdquo;")
    for row in list(region.evidence)[:2]:
        lines.append(
            f"{row.feature}: {row.value:.3g} {row.unit} "
            f"vs {row.env_median:.3g} (z = {row.z:+.2f})"
        )
    if region.refined:
        lines.append("boundaries refined by ruptures (PELT, RBF)")
    headline = str(card.get("headline", ""))
    if headline:
        lines.append(f"<i>{headline}</i>")
    return "<br>".join(lines)


# --------------------------------------------------------------------------- #
# The overlay
# --------------------------------------------------------------------------- #


def _boundary_ticks(view: AnalysisView) -> tuple[list[float], list[str]]:
    """Word onsets and the words themselves, thinned to stay legible."""
    spans = view.word_spans
    if not spans:
        return [], []
    words = {}
    if view.result.alignment is not None:
        words = {int(w.idx): (w.display or w.text) for w in view.result.alignment.words}
    indices = sorted(spans)
    step = max(1, len(indices) // MAX_BOUNDARY_TICKS)
    kept = indices[::step]
    return [spans[i][0] for i in kept], [str(words.get(i, i)) for i in kept]


def overlay_figure(
    view: AnalysisView,
    *,
    highlight: int | None = None,
    height: int = 760,
) -> go.Figure:
    """The main time-series overlay (spec 15.2).

    Args:
        view: The analysis to draw.
        highlight: ``region_id`` to pick out, from the dashboard's "Jump"
            button. The region keeps its category colour and gains a solid
            outline, so the highlight does not overwrite the category encoding.
        height: Figure height in pixels; the default fits three panels at
            1280 px wide without scrolling.

    Returns:
        A Plotly figure with one shared x-axis.
    """
    go = _go()
    from plotly.subplots import make_subplots

    panels = overlay_panels(view)
    if not panels:
        figure = go.Figure()
        figure.update_layout(
            **plotly_layout(240),
            title="No curves were recorded for this analysis",
            annotations=[
                {
                    "text": (
                        "Curves are produced in Reference and Norm mode. A rejected "
                        "upload has none, because nothing was compared."
                    ),
                    "showarrow": False,
                    "x": 0.5,
                    "y": 0.5,
                    "xref": "paper",
                    "yref": "paper",
                }
            ],
        )
        return figure

    figure = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[p.title for p, _, _ in panels],
    )

    for row, (panel, mine, theirs) in enumerate(panels, start=1):
        first = row == 1

        # The band first, so the participant's line is drawn over it.
        if theirs and theirs.lower is not None and theirs.upper is not None:
            band_x, band_y = _band_polygon(theirs)
            figure.add_trace(
                go.Scatter(
                    x=band_x,
                    y=band_y,
                    fill="toself",
                    fillcolor=_rgba(ENVELOPE_COLOUR, 0.16),
                    line={"width": 0},
                    hoverinfo="skip",
                    name=f"skilled readers, +/- 1 sigma (n = {theirs.n_references})",
                    legendgroup="envelope",
                    showlegend=first,
                ),
                row=row,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=theirs.t,
                    y=theirs.value,
                    mode="lines",
                    line={"color": ENVELOPE_COLOUR, "width": 1.6, "dash": "dash"},
                    name="skilled readers, median",
                    legendgroup="envelope-median",
                    showlegend=first,
                    hovertemplate=(
                        f"{panel.title} median %{{y:.2f}} {panel.unit}<br>"
                        "%{x:.3f} s<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        if mine:
            figure.add_trace(
                go.Scatter(
                    x=mine.t,
                    y=mine.value,
                    mode="lines",
                    line={"color": PARTICIPANT_COLOUR, "width": 1.9},
                    name="this reading",
                    legendgroup="participant",
                    showlegend=first,
                    connectgaps=False,
                    hovertemplate=(
                        f"{panel.title} %{{y:.2f}} {panel.unit}<br>%{{x:.3f}} s<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        figure.update_yaxes(
            title_text=panel.unit,
            row=row,
            col=1,
            gridcolor="#EEEEEE",
            zeroline=False,
        )

    _add_regions(figure, view, rows=len(panels), highlight=highlight)
    _add_word_hover(figure, view)

    ticks, labels = _boundary_ticks(view)
    t0, t1 = view.time_range
    figure.update_xaxes(range=[t0, t1], gridcolor="#F0F0F0", row=len(panels), col=1)
    for row in range(1, len(panels) + 1):
        # Word onsets as ticks on every panel's x-axis; only the bottom panel
        # shows the words themselves, or three rows of labels would collide.
        figure.update_xaxes(
            tickvals=ticks,
            ticktext=labels if row == len(panels) else [""] * len(ticks),
            tickangle=-60 if row == len(panels) else 0,
            tickfont={"size": 9},
            ticks="outside",
            ticklen=4,
            showticklabels=True,
            row=row,
            col=1,
        )
    figure.update_xaxes(title_text="time (s) - ticks are word onsets", row=len(panels), col=1)

    layout = plotly_layout(height)
    # The legend goes under the x-axis: with three stacked panels there is no
    # room above the first one for a title and a legend both, and the word
    # ticks already make the bottom the busy edge to read anyway.
    layout["margin"] = {**layout["margin"], "t": 64, "b": 150}
    layout["legend"] = {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.16,
        "xanchor": "left",
        "x": 0.0,
    }
    figure.update_layout(
        **layout,
        title={
            "text": (
                f"{view.result.file_id} - this reading against "
                f"{view.result.n_references} skilled reading(s)"
                if view.result.mode == "reference"
                else f"{view.result.file_id} - {view.result.mode} mode"
            ),
            "x": 0.0,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "yref": "container",
        },
    )
    return figure


def _add_regions(
    figure: Any,
    view: AnalysisView,
    *,
    rows: int,
    highlight: int | None = None,
) -> None:
    """Shade every flaw region across every panel, and label it in text."""
    go = _go()
    seen: set[str] = set()

    # The hover handles sit just above the first panel's data rather than on
    # top of it, so they read as markers in a margin and not as measurements.
    top_panel = view.panels[0] if view.panels else None
    values = (
        view.participant_series(top_panel.signal).value if top_panel is not None else np.empty(0)
    )
    finite = values[np.isfinite(values)] if values.size else np.empty(0)
    marker_y = float(np.max(finite)) if finite.size else 0.0

    for region in view.regions:
        colour = category_colour(region.category)
        is_highlight = highlight is not None and int(region.region_id) == int(highlight)
        for row in range(1, rows + 1):
            figure.add_vrect(
                x0=float(region.t_start),
                x1=float(region.t_end),
                fillcolor=colour,
                opacity=level_opacity(region.level) * (1.6 if is_highlight else 1.0),
                layer="below",
                line_width=2.0 if is_highlight else 0.0,
                line_color=colour if is_highlight else None,
                row=row,
                col=1,
            )
        # A vrect carries no hover and no legend entry, so both are added as
        # traces: an invisible marker at the region's midpoint for the hover,
        # and one dummy trace per category for the legend.
        figure.add_trace(
            go.Scatter(
                x=[(float(region.t_start) + float(region.t_end)) / 2.0],
                y=[marker_y],
                mode="markers",
                marker={
                    "size": 11,
                    "color": colour,
                    "symbol": "triangle-down",
                    "line": {"width": 1.0, "color": "#FFFFFF"},
                },
                name=f"region {region.region_id}",
                showlegend=False,
                hovertext=[region_hover(region, view.card_for(region.region_id))],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        if region.category not in seen:
            seen.add(region.category)
            label = CATEGORY_LABELS.get(region.category, region.category)
            figure.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"size": 12, "color": colour, "symbol": "square"},
                    name=f"{label} flaw region",
                    legendgroup=f"cat-{region.category}",
                    showlegend=True,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )


def _add_word_hover(figure: Any, view: AnalysisView) -> None:
    """An invisible trace carrying each word, its D_w and its strongest z-scores."""
    go = _go()
    spans = view.word_spans
    if not spans or not view.has_deviation:
        return
    words = (
        {int(w.idx): (w.display or w.text) for w in view.result.alignment.words}
        if view.result.alignment is not None
        else {}
    )

    x: list[float] = []
    text: list[str] = []
    for idx in sorted(spans):
        t0, t1 = spans[idx]
        deviation = view.deviation(idx)
        lines = [f"<b>{words.get(idx, idx)}</b> ({t0:.3f} - {t1:.3f} s)"]
        lines.append("D = nothing measurable here" if deviation is None else f"D = {deviation:.2f}")
        for row in view.top_z(idx):
            lines.append(f"{row.get('feature')}: z = {float(row.get('z', 0.0)):+.2f}")
        x.append((t0 + t1) / 2.0)
        text.append("<br>".join(lines))

    figure.add_trace(
        go.Scatter(
            x=x,
            y=[None] * len(x),
            mode="markers",
            marker={"size": 1, "opacity": 0.0},
            hovertext=text,
            hoverinfo="text",
            name="words",
            showlegend=False,
        ),
        row=1,
        col=1,
    )


# --------------------------------------------------------------------------- #
# The radar
# --------------------------------------------------------------------------- #


def radar_figure(view: AnalysisView, *, height: int = 420) -> go.Figure:
    """The six rubric categories, with the confidence band drawn explicitly.

    A **suppressed** category - one that could not be measured at all, which in
    practice means clarity without a same-channel reference (invariant 7) - is
    deliberately left out of the filled polygon and marked with a grey cross on
    its own axis instead. Spec 11 excludes such a category from the overall
    mean, and a filled vertex at 10 would say the opposite of that: it would
    read as a perfect score for something nobody looked at.
    """
    go = _go()
    rows = view.category_scores()
    if not rows:
        figure = go.Figure()
        figure.update_layout(**_polar_layout(height), title="Nothing was scored")
        return figure

    labels = [
        CATEGORY_LABELS.get(r["category"], r["category"])
        + (" (not measurable)" if r["suppressed"] else "")
        for r in rows
    ]
    scored = [(label, r) for label, r in zip(labels, rows, strict=True) if not r["suppressed"]]
    missing = [(label, r) for label, r in zip(labels, rows, strict=True) if r["suppressed"]]

    figure = go.Figure()

    if scored:
        theta = [label for label, _ in scored]
        values = [r["score"] for _, r in scored]
        bands = [r["band"] for _, r in scored]

        if any(b > 0 for b in bands):
            for sign in (1.0, -1.0):
                edge = [
                    min(10.0, max(0.0, v + sign * b)) for v, b in zip(values, bands, strict=True)
                ]
                figure.add_trace(
                    go.Scatterpolar(
                        r=[*edge, edge[0]],
                        theta=[*theta, theta[0]],
                        mode="lines",
                        line={"color": ENVELOPE_COLOUR, "width": 1, "dash": "dot"},
                        name="+/- leave-one-reader-out band",
                        showlegend=sign > 0,
                        hoverinfo="skip",
                    )
                )

        figure.add_trace(
            go.Scatterpolar(
                r=[*values, values[0]],
                theta=[*theta, theta[0]],
                fill="toself",
                fillcolor=_rgba(ENVELOPE_COLOUR, 0.22),
                line={"color": ENVELOPE_COLOUR, "width": 2.2},
                name="this reading",
                customdata=[
                    [r["band"], r["n_regions"], r["deviation_mass"], r["k"]] for _, r in scored
                ]
                + [
                    [
                        scored[0][1]["band"],
                        scored[0][1]["n_regions"],
                        scored[0][1]["deviation_mass"],
                        scored[0][1]["k"],
                    ]
                ],
                hovertemplate=(
                    "<b>%{theta}</b><br>%{r:.2f} / 10 +/- %{customdata[0]:.2f}<br>"
                    "%{customdata[1]} region(s), deviation mass %{customdata[2]:.3f}, "
                    "k = %{customdata[3]:.2f}<extra></extra>"
                ),
            )
        )

    if missing:
        figure.add_trace(
            go.Scatterpolar(
                r=[10.0] * len(missing),
                theta=[label for label, _ in missing],
                mode="markers",
                marker={"symbol": "x", "size": 11, "color": "#8A8A8A"},
                name="not measurable - excluded from the mean",
                hovertext=[r["suppressed_reason"] or "not measurable" for _, r in missing],
                hoverinfo="text",
            )
        )

    figure.update_layout(
        **_polar_layout(height),
        title={
            "text": "Rubric:  S_c = 10 exp(-k_c M_c)",
            "x": 0.0,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
            "yref": "container",
        },
    )
    return figure


def _polar_layout(height: int) -> dict[str, Any]:
    """Shared layout for the radar: legend underneath, 0-10 radial axis."""
    layout = plotly_layout(height)
    layout.pop("plot_bgcolor", None)
    layout["margin"] = {"l": 70, "r": 70, "t": 56, "b": 76}
    layout["legend"] = {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.04,
        "xanchor": "left",
        "x": 0.0,
        "font": {"size": 11},
    }
    layout["polar"] = {
        "radialaxis": {
            "range": [0, 10],
            "tickvals": [0, 2, 4, 6, 8, 10],
            "gridcolor": "#E6E6E6",
            "angle": 90,
            "tickangle": 0,
        },
        "angularaxis": {"tickfont": {"size": 11}},
    }
    return layout


# --------------------------------------------------------------------------- #
# The waveform
# --------------------------------------------------------------------------- #


def waveform_figure(
    view: AnalysisView,
    samples: np.ndarray,
    sample_rate: int,
    *,
    height: int = 200,
    max_points: int = 4000,
) -> go.Figure:
    """The recording with its flaw regions marked (spec 15.2).

    The waveform is decimated to a min/max envelope rather than subsampled:
    taking every *n*-th sample of a 16 kHz signal aliases the peaks and draws a
    quieter recording than the one that was analysed.

    Args:
        view: The analysis, for the regions and the time range.
        samples: Mono float32 of the processed recording.
        sample_rate: Its sample rate.
        height: Figure height in pixels.
        max_points: Columns in the decimated envelope.

    Returns:
        A Plotly figure.
    """
    go = _go()
    x = np.asarray(samples, dtype=np.float64)
    figure = go.Figure()

    if x.size:
        bucket = max(1, x.size // max(1, int(max_points)))
        usable = (x.size // bucket) * bucket
        blocks = x[:usable].reshape(-1, bucket)
        times = (np.arange(blocks.shape[0]) * bucket + bucket / 2.0) / float(sample_rate)
        lo, hi = blocks.min(axis=1), blocks.max(axis=1)
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([times, times[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=_rgba(PARTICIPANT_COLOUR, 0.72),
                line={"width": 0},
                hoverinfo="skip",
                name="waveform",
                showlegend=False,
            )
        )

    for region in view.regions:
        figure.add_vrect(
            x0=float(region.t_start),
            x1=float(region.t_end),
            fillcolor=category_colour(region.category),
            opacity=max(0.22, level_opacity(region.level)),
            layer="above",
            line_width=0,
            annotation_text=f"{region.label} {region.level}",
            annotation_position="top left",
            annotation_font_size=9,
        )

    t0, t1 = view.time_range
    figure.update_layout(
        **plotly_layout(height),
        showlegend=False,
        title={"text": "Where in the speech", "x": 0.0, "xanchor": "left"},
    )
    figure.update_xaxes(range=[t0, t1], title_text="time (s)", gridcolor="#F0F0F0")
    figure.update_yaxes(
        range=[-1.0, 1.0], showticklabels=False, zeroline=True, zerolinecolor="#DDDDDD"
    )
    return figure
