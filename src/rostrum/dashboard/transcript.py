r"""The transcript coloured by per-word deviation (spec 15.2, [R41]).

This is the view that makes the analysis legible to someone who does not read
time-series plots: the speech as text, with each word tinted by how far it sat
from the skilled readers,

.. math:: D_w = \sqrt{\sum_k \lambda_k z_{wk}^2}

and the words inside a flagged region outlined in their category's colour. A
judge can see at a glance *where in the sentence* the delivery went wrong, and
hover for the numbers behind the tint.

Three things this module is careful about:

* **Colour is never the only channel.** A word in a region is underlined and
  outlined as well as tinted, its title attribute names the flaw in words, and
  the tint has a printed legend with numbers on it.
* **"Not measurable" is not "typical".** A word with no finite deviation - an
  unvoiced monosyllable, say - is drawn in neutral grey, never in the ramp's
  pale "zero" colour, which would claim it was measured and found ordinary.
* **Punctuation survives.** The tint is applied to the aligner's token, but the
  text shown is the transcript's own display form, so the reader sees commas
  and capitals and can match the view against the speech they know.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any

from rostrum.dashboard.palette import (
    CATEGORY_LABELS,
    DEVIATION_SCALE,
    category_colour,
    deviation_colour,
    level_label,
    text_on,
)

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.dashboard.view import AnalysisView

__all__ = [
    "TranscriptWord",
    "deviation_legend_html",
    "transcript_html",
    "transcript_words",
]


@dataclass(frozen=True)
class TranscriptWord:
    """One word of the transcript as the dashboard draws it.

    Attributes:
        idx: Word index, shared with the alignment and the curves.
        text: What to print - the display form, with punctuation.
        t_start: Onset in seconds.
        t_end: Offset in seconds.
        deviation: :math:`D_w`, or ``None`` when nothing was measurable.
        colour: Background tint.
        region_id: The flaw region covering this word, if any.
        category: That region's rubric category.
        label: That region's flaw label.
        level: That region's severity rung.
        top_z: The strongest z-scores here, for the tooltip.
    """

    idx: int
    text: str
    t_start: float
    t_end: float
    deviation: float | None
    colour: str
    region_id: int | None = None
    category: str = ""
    label: str = ""
    level: str = ""
    top_z: tuple[dict[str, Any], ...] = ()

    @property
    def in_region(self) -> bool:
        """Whether this word sits inside a flagged region."""
        return self.region_id is not None

    def tooltip(self) -> str:
        """The hover text: the numbers, then the flaw if there is one."""
        lines = [f"{self.text}  {self.t_start:.3f}-{self.t_end:.3f} s"]
        lines.append(
            "D = not measurable here" if self.deviation is None else f"D = {self.deviation:.2f}"
        )
        for row in self.top_z:
            feature = row.get("feature", "?")
            z = float(row.get("z", 0.0))
            value, median = row.get("value"), row.get("env_median")
            if value is None or median is None:
                lines.append(f"{feature}: z = {z:+.2f}")
            else:
                lines.append(f"{feature}: {float(value):.3g} vs {float(median):.3g} (z {z:+.2f})")
        if self.in_region:
            category = CATEGORY_LABELS.get(self.category, self.category)
            lines.append(f"{category}: {self.label}, {level_label(self.level)}")
        return "\n".join(lines)


def transcript_words(view: AnalysisView) -> list[TranscriptWord]:
    """Every aligned word with its tint and its region, in transcript order.

    Args:
        view: The analysis to read.

    Returns:
        One :class:`TranscriptWord` per aligned word. Empty when the analysis
        has no alignment, which is the Reject case.
    """
    if view.result.alignment is None:
        return []

    # Region membership by word index. Regions store (first, last), so the
    # span is expanded rather than treated as a list of two indices.
    by_word: dict[int, Any] = {}
    for region in view.regions:
        indices = [int(i) for i in region.word_idx]
        if not indices:
            continue
        for idx in range(min(indices), max(indices) + 1):
            # First region wins, so a word in two regions keeps the earlier
            # one and the later one is still visible in its own card.
            by_word.setdefault(idx, region)

    cap, floor = view.deviation_cap, view.deviation_floor
    out: list[TranscriptWord] = []
    for word in sorted(view.result.alignment.words, key=lambda w: int(w.idx)):
        idx = int(word.idx)
        deviation = view.deviation(idx)
        region = by_word.get(idx)
        out.append(
            TranscriptWord(
                idx=idx,
                text=(word.display or word.text),
                t_start=float(word.t_start),
                t_end=float(word.t_end),
                deviation=deviation,
                colour=deviation_colour(deviation, cap=cap, floor=floor),
                region_id=None if region is None else int(region.region_id),
                category="" if region is None else region.category,
                label="" if region is None else region.label,
                level="" if region is None else region.level,
                top_z=tuple(view.top_z(idx)),
            )
        )
    return out


def transcript_html(view: AnalysisView, *, max_words: int | None = None) -> str:
    """The coloured transcript as a block of HTML.

    Args:
        view: The analysis to render.
        max_words: Truncate after this many words, for a long speech in a
            narrow column. ``None`` renders all of them.

    Returns:
        A ``<div>`` of tinted word spans. Safe to hand to Streamlit with
        ``unsafe_allow_html=True``: every piece of text that came from the
        transcript or the dataset is escaped.
    """
    words = transcript_words(view)
    if not words:
        return (
            '<div style="padding:12px;border:1px solid #DDD;border-radius:8px;'
            'color:#555">No alignment, so no per-word view. A rejected upload is '
            "not aligned, because the words in the audio are not the words in the "
            "transcript.</div>"
        )

    shown = words if max_words is None else words[: int(max_words)]
    pieces: list[str] = []
    for word in shown:
        outline = (
            f"border-bottom:3px solid {category_colour(word.category)};"
            if word.in_region
            else "border-bottom:3px solid transparent;"
        )
        pieces.append(
            f'<span title="{escape(word.tooltip(), quote=True)}" style="background:{word.colour};color:{text_on(word.colour)};{outline}'
            "padding:2px 4px;margin:1px 1px;border-radius:3px;display:inline-block;"
            f'font-size:0.95rem;line-height:1.5;">{escape(word.text)}</span>'
        )

    if max_words is not None and len(words) > len(shown):
        pieces.append(
            f'<span style="color:#666;font-size:0.9rem;padding:2px 6px;">'
            f"... {len(words) - len(shown)} more word(s)</span>"
        )

    return (
        '<div style="padding:14px;border:1px solid #E4E4E7;border-radius:10px;'
        'background:#FFFFFF;line-height:2.0;">' + "".join(pieces) + "</div>"
    )


def deviation_legend_html(cap: float = 6.12, floor: float = 0.0) -> str:
    r"""A legend for the tint, with the numbers it stands for.

    A colour ramp with no scale is decoration. This prints the ramp with the
    :math:`D_w` values under it, plus the "not measurable" swatch, so the
    transcript can be read without guessing - and the two ends are labelled
    with what they mean, not just with numbers.
    """
    swatches: list[str] = []
    for stop, _ in DEVIATION_SCALE:
        value = float(floor) + stop * (float(cap) - float(floor))
        colour = deviation_colour(value, cap=cap, floor=floor)
        label = f"{value:.1f}" + ("+" if stop >= 1.0 else "")
        swatches.append(
            '<span style="display:inline-block;text-align:center;margin-right:2px;">'
            f'<span style="display:block;width:46px;height:14px;background:{colour};'
            'border:1px solid #CCC;"></span>'
            f'<span style="font-size:0.72rem;color:#444;">{label}</span></span>'
        )
    swatches.append(
        '<span style="display:inline-block;text-align:center;margin-left:14px;">'
        '<span style="display:block;width:46px;height:14px;background:#EFEFEF;'
        'border:1px solid #CCC;"></span>'
        '<span style="font-size:0.72rem;color:#444;">n/a</span></span>'
    )
    return (
        '<div style="margin:6px 0 10px 0;">'
        '<span style="font-size:0.8rem;color:#333;margin-right:10px;">'
        "word deviation D:</span>" + "".join(swatches) + "</div>"
        '<div style="font-size:0.72rem;color:#666;margin:-4px 0 8px 0;">'
        f"{float(floor):.1f} is an ordinary word (every feature 1 sigma from the "
        f"readers' median); {float(cap):.1f} is every feature at the flag threshold "
        "at once."
        "</div>"
    )
