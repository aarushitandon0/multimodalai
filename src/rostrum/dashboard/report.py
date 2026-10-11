"""The PDF export (spec 15.2 step 4).

A judge's feedback has to leave the browser: the dashboard is where a flaw is
explored, and the PDF is what gets handed to the speaker. So the report carries
the same four things the screen does, in the same order and the same colours -
the verdict, the rubric, the overlay, and one page of cards - plus the
provenance line that makes it checkable: analysis id, config hash, code
version, and how many skilled readers stood behind the comparison.

Rendered with matplotlib's PDF backend rather than ``reportlab``: the project
already depends on matplotlib for ``eval.figures``, the overlay is a plot
anyway, and one fewer dependency is one fewer thing to pin for the
reproducibility requirement. The cost is that layout is done in figure
coordinates by hand, which is why the text wrapping below is explicit.

Greyscale-safe: every category patch carries a hatch as well as a colour
(see :mod:`rostrum.dashboard.palette`), and every severity is printed as text.
"""

from __future__ import annotations

import logging
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.dashboard.palette import (
    CATEGORY_HATCH,
    CATEGORY_LABELS,
    ENVELOPE_COLOUR,
    PARTICIPANT_COLOUR,
    category_colour,
    level_label,
)
from rostrum.dashboard.view import AnalysisView, overlay_panels

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

__all__ = ["write_pdf"]

_PAGE = (8.27, 11.69)  # A4 portrait, inches
_BODY = 8.5
_SMALL = 7.0


def _pyplot() -> Any:
    """Import pyplot with the non-interactive backend (no display needed)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt


def _wrap(text: str, width: int) -> str:
    """Wrap one paragraph to ``width`` characters."""
    return "\n".join(textwrap.wrap(str(text), width=width)) if text else ""


# --------------------------------------------------------------------------- #
# Page 1: verdict, rubric, overlay
# --------------------------------------------------------------------------- #


def _header(figure: Figure, view: AnalysisView) -> None:
    """The verdict block: mode, score, and the provenance that makes it checkable."""
    result = view.result
    figure.text(0.07, 0.962, "Rostrum delivery report", fontsize=17, fontweight="bold")
    figure.text(
        0.07,
        0.941,
        f"{result.file_id}"
        + (f"  -  text {result.text_id}" if result.text_id else "")
        + f"  -  {result.duration:.1f} s",
        fontsize=_BODY + 1,
        color="#333333",
    )

    if result.mode == "reference":
        verdict = (
            f"Reference mode: compared word by word against {result.n_references} "
            "skilled reading(s) of this exact text."
        )
    elif result.mode == "norm":
        verdict = (
            "Norm mode (lower confidence): no library text matched this transcript, so "
            "the comparison is against pooled norms. Clarity and voice quality are "
            "measured but cannot raise a flaw."
        )
    else:
        verdict = "Rejected: the audio and the transcript do not match, so nothing is scored. " + (
            result.mode_reason or ""
        )
    figure.text(
        0.07,
        0.917,
        _wrap(verdict, 108),
        fontsize=_BODY,
        color="#111111",
        va="top",
    )

    if view.overall is not None:
        figure.text(
            0.07,
            0.866,
            f"{view.overall:.2f}",
            fontsize=27,
            fontweight="bold",
            color="#111111",
        )
        figure.text(0.192, 0.872, "/ 10", fontsize=13, color="#444444")
        figure.text(
            0.192,
            0.855,
            f"+/- {view.overall_band:.2f}  (leave-one-reference-out)",
            fontsize=_SMALL,
            color="#666666",
        )

    stamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    figure.text(
        0.07,
        0.028,
        "  |  ".join(
            [
                f"analysis {result.analysis_id[:16]}",
                f"config {result.config_hash[:12]}",
                f"code {result.code_version or 'n/a'}",
                f"features v{result.features_version}",
                stamp,
            ]
        ),
        fontsize=6.4,
        color="#666666",
    )
    figure.text(
        0.07,
        0.014,
        "Every number and timestamp is computed by DSP; no language model is in the "
        "scoring loop. Times are seconds from the start of the processed 16 kHz WAV.",
        fontsize=6.4,
        color="#666666",
    )


def _rubric_axes(figure: Figure, view: AnalysisView) -> None:
    """Horizontal bars for the six categories, with the band as an error bar."""
    rows = view.category_scores()
    axes = figure.add_axes((0.135, 0.648, 0.335, 0.172))
    if not rows:
        axes.axis("off")
        axes.text(0.0, 0.5, "Nothing was scored.", fontsize=_BODY, color="#444444")
        return

    y = np.arange(len(rows))[::-1]
    axes.barh(
        y,
        [r["score"] for r in rows],
        xerr=[r["band"] for r in rows],
        color=[category_colour(r["category"]) for r in rows],
        edgecolor="white",
        height=0.66,
        error_kw={"ecolor": "#333333", "elinewidth": 0.9, "capsize": 2.5},
    )
    for i, row in enumerate(rows):
        patch = axes.patches[i]
        patch.set_hatch(CATEGORY_HATCH.get(row["category"], ""))
        axes.text(
            row["score"] + row["band"] + 0.22,
            y[i],
            f"{row['score']:.1f}" + ("  (not measurable)" if row["suppressed"] else ""),
            va="center",
            fontsize=_SMALL,
            color="#222222",
        )
    axes.set_yticks(y)
    axes.set_yticklabels(
        [CATEGORY_LABELS.get(r["category"], r["category"]) for r in rows], fontsize=_SMALL
    )
    axes.set_xlim(0, 11.6)
    axes.set_xticks([0, 2, 4, 6, 8, 10])
    axes.tick_params(labelsize=_SMALL)
    axes.set_xlabel("score out of 10", fontsize=_SMALL)
    axes.set_title("Rubric:  S_c = 10 exp(-k_c M_c)", fontsize=_BODY, loc="left")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def _descriptor_axes(figure: Figure, view: AnalysisView) -> None:
    """The raw descriptors with the skilled readers' range beside them."""
    axes = figure.add_axes((0.55, 0.648, 0.40, 0.172))
    axes.axis("off")
    rows = view.descriptor_rows()
    axes.set_title("Descriptors vs skilled readers", fontsize=_BODY, loc="left")
    if not rows:
        axes.text(0.0, 0.9, "No descriptors were computed.", fontsize=_SMALL, color="#444444")
        return

    lines: list[str] = []
    for row in rows[:8]:
        value = row["value"]
        shown = "n/a" if value is None else f"{float(value):.2f}"
        low, high = row.get("reference_low"), row.get("reference_high")
        if low is None or high is None:
            lines.append(f"{row['descriptor']}: {shown}")
        else:
            lines.append(
                f"{row['descriptor']}: {shown}   "
                f"(readers {float(low):.2f}-{float(high):.2f}, n={row.get('n_references', 0)})"
            )
    axes.text(
        0.0,
        0.92,
        "\n".join(_wrap(line, 62) for line in lines),
        fontsize=_SMALL,
        va="top",
        color="#222222",
        linespacing=1.6,
    )


def _overlay_axes(figure: Figure, view: AnalysisView) -> None:
    """The same three-panel overlay the screen shows, in print colours."""
    panels = overlay_panels(view)
    if not panels:
        axes = figure.add_axes((0.07, 0.10, 0.86, 0.46))
        axes.axis("off")
        axes.text(
            0.0,
            0.9,
            "No curves were recorded for this analysis, so there is no overlay. "
            "Curves exist in Reference and Norm mode.",
            fontsize=_BODY,
            color="#444444",
        )
        return

    top, height = 0.570, 0.475 / len(panels)
    t0, t1 = view.time_range
    for i, (panel, mine, theirs) in enumerate(panels):
        bottom = top - (i + 1) * height
        axes = figure.add_axes((0.07, bottom + 0.035, 0.86, height - 0.045))

        if theirs and theirs.lower is not None and theirs.upper is not None:
            axes.fill_between(
                theirs.t,
                theirs.lower,
                theirs.upper,
                color=ENVELOPE_COLOUR,
                alpha=0.18,
                linewidth=0,
                label=f"skilled readers +/- 1 sigma (n={theirs.n_references})",
            )
            axes.plot(
                theirs.t,
                theirs.value,
                color=ENVELOPE_COLOUR,
                linewidth=1.0,
                linestyle="--",
                label="readers' median",
            )
        if mine:
            axes.plot(
                mine.t,
                mine.value,
                color=PARTICIPANT_COLOUR,
                linewidth=1.1,
                label="this reading",
            )

        for region in view.regions:
            axes.axvspan(
                float(region.t_start),
                float(region.t_end),
                color=category_colour(region.category),
                alpha=0.20,
                linewidth=0,
                hatch=CATEGORY_HATCH.get(region.category, ""),
            )

        axes.set_xlim(t0, t1)
        axes.set_ylabel(f"{panel.title}\n{panel.unit}", fontsize=_SMALL)
        axes.tick_params(labelsize=_SMALL)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        if i == 0:
            # The caption sits above the whole block as figure text, so the
            # legend can have the axes' own top row to itself.
            figure.text(
                0.07,
                top + 0.010,
                "This reading against the envelope of skilled readings; "
                "shaded spans are flagged regions",
                fontsize=_BODY,
            )
            axes.legend(
                fontsize=6.0,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.005),
                frameon=False,
                ncol=3,
            )
        if i < len(panels) - 1:
            axes.set_xticklabels([])
        else:
            axes.set_xlabel("time (s)", fontsize=_SMALL)


# --------------------------------------------------------------------------- #
# Page 2+: the cards
# --------------------------------------------------------------------------- #


def _card_pages(figure_factory: Any, view: AnalysisView, pdf: Any) -> None:
    """One page of region cards, continued onto further pages as needed."""
    plt = _pyplot()
    cards = view.cards
    if not cards:
        return

    per_page = 3
    for start in range(0, len(cards), per_page):
        figure = plt.figure(figsize=_PAGE)
        figure.text(
            0.07,
            0.962,
            f"Flaw regions {start + 1}-{min(start + per_page, len(cards))} of {len(cards)}",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.07,
            0.943,
            "Each region is a time span, a measured deviation, and what to do about it.",
            fontsize=_SMALL,
            color="#555555",
        )

        block = cards[start : start + per_page]
        for i, card in enumerate(block):
            _draw_card(figure, card, top=0.900 - i * 0.295)
        pdf.savefig(figure)
        plt.close(figure)


def _draw_card(figure: Figure, card: dict[str, Any], *, top: float) -> None:
    """One region card: headline, evidence table, rationale, fix."""
    category = str(card.get("category", ""))
    colour = category_colour(category)
    label = CATEGORY_LABELS.get(category, category or "region")

    axes = figure.add_axes((0.07, top - 0.270, 0.86, 0.262))
    axes.axis("off")
    _colour_bar(axes, colour)

    t_start = float(card.get("t_start", 0.0))
    t_end = float(card.get("t_end", 0.0))
    axes.text(
        0.012,
        0.93,
        f"{t_start:.3f} - {t_end:.3f} s   |   {label}: {card.get('label', '')}   |   "
        f"{level_label(str(card.get('level', 'L0')))}   |   "
        f"confidence {float(card.get('confidence', 0.0)):.2f}",
        fontsize=_BODY,
        fontweight="bold",
        va="top",
        transform=axes.transAxes,
    )
    words = str(card.get("words", ""))
    if words:
        axes.text(
            0.012,
            0.845,
            _wrap(f'"{words}"', 112),
            fontsize=_SMALL,
            style="italic",
            color="#333333",
            va="top",
            transform=axes.transAxes,
        )

    rows = list(card.get("evidence", []))[:4]
    y = 0.70
    if rows:
        axes.text(
            0.012,
            y,
            f"{'feature':<26}{'you':>12}{'readers':>12}{'sigma':>10}{'z':>9}",
            fontsize=6.4,
            family="monospace",
            color="#444444",
            va="top",
            transform=axes.transAxes,
        )
        y -= 0.075
        for row in rows:
            axes.text(
                0.012,
                y,
                f"{str(row.get('feature', ''))[:25]:<26}"
                f"{_num(row.get('value')):>12}"
                f"{_num(row.get('env_median')):>12}"
                f"{_num(row.get('env_sigma')):>10}"
                f"{_num(row.get('z'), signed=True):>9}",
                fontsize=6.4,
                family="monospace",
                color="#111111",
                va="top",
                transform=axes.transAxes,
            )
            y -= 0.070

    for key, prefix in (("rationale", "Why: "), ("fix", "Fix: ")):
        text = str(card.get(key, ""))
        if not text:
            continue
        wrapped = _wrap(prefix + text, 116)
        axes.text(
            0.012,
            y - 0.02,
            wrapped,
            fontsize=_SMALL,
            color="#111111",
            va="top",
            transform=axes.transAxes,
        )
        # One line of 7 pt text is about 0.052 of this card's height; the
        # constant is the gap before the next block.
        y -= 0.052 * len(wrapped.split("\n")) + 0.030


def _num(value: Any, *, signed: bool = False) -> str:
    """One number for the evidence table, or ``-``."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:+.2f}" if signed else f"{number:.3g}"


def _colour_bar(axes: Any, colour: str) -> None:
    """Draw a card's left-edge category bar, in axes coordinates.

    The bar is the second, non-textual cue for the category - the same hatch
    and colour the overlay uses - so a card can be matched to its region in the
    figure above without reading either caption.
    """
    from matplotlib.patches import Rectangle

    axes.add_patch(
        Rectangle(
            (-0.004, 0.0),
            0.005,
            1.0,
            transform=axes.transAxes,
            facecolor=colour,
            edgecolor="none",
            clip_on=False,
        )
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def write_pdf(view: AnalysisView, path: Path | str) -> Path:
    """Render the whole report to a PDF.

    Args:
        view: The analysis to report on.
        path: Where to write the file; the parent directory is created.

    Returns:
        The path written.
    """
    plt = _pyplot()
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        figure = plt.figure(figsize=_PAGE)
        _header(figure, view)
        _rubric_axes(figure, view)
        _descriptor_axes(figure, view)
        _overlay_axes(figure, view)
        pdf.savefig(figure)
        plt.close(figure)

        _card_pages(plt.figure, view, pdf)

        info = pdf.infodict()
        info["Title"] = f"Rostrum delivery report - {view.result.file_id}"
        info["Subject"] = (
            f"mode={view.result.mode}, config={view.result.config_hash[:12]}, "
            f"analysis={view.result.analysis_id[:16]}"
        )
        info["Creator"] = "Rostrum"

    logger.info("wrote %s", out)
    return out
