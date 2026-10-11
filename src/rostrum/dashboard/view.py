"""Reading an :class:`~rostrum.types.AnalysisResult` into plottable series (spec 15.2).

This module is the single adapter between what the pipeline *recorded* and what
the dashboard *draws*. It computes nothing: every value it returns is already
in the result, rounded as the determinism hash saw it. That is the point - a
figure cannot drift from the numbers in the cards, and a result loaded from the
cache draws exactly like one just computed.

The one piece of real work here is rebuilding a time axis. The result stores
curves per *word* on a normalised axis of ``n_points`` - that is what makes two
readers of different speeds averageable (spec 10.1) - so a word's curve is laid
back onto the clock by stretching its points across that word's own span:

.. code-block:: text

    word i spans [t_start, t_end]      values: n_points samples
    x = linspace(t_start, t_end, n_points)

and a ``NaN`` is inserted between consecutive words so the line **breaks over
pauses** instead of drawing a straight segment through a silence that the
speaker did not phonate. A bridged pause would be a picture of something that
did not happen, and pauses are half of what this system judges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.config import Config, load_config

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.types import AnalysisResult, Region

__all__ = [
    "PANELS",
    "AnalysisView",
    "PanelSpec",
    "Series",
    "overlay_panels",
]


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PanelSpec:
    """One stacked panel of the overlay.

    Attributes:
        signal: Column name in the curves payload.
        title: Panel heading.
        unit: Axis label; states the normalisation, because an un-normalised
            pitch or energy comparison across speakers would be meaningless
            (CLAUDE.md invariants 5 and 6).
        why: One line on what this panel is evidence for, shown as help text.
    """

    signal: str
    title: str
    unit: str
    why: str = ""


PANELS: tuple[PanelSpec, ...] = (
    PanelSpec(
        signal="f0_st",
        title="Pitch",
        unit="semitones re speaker median",
        why=(
            "Semitones relative to this speaker's own median, so a bass and a soprano "
            "reading the same line sit on the same axis. Flat here is monotone; a "
            "misplaced peak is misplaced stress."
        ),
    ),
    PanelSpec(
        signal="energy_z",
        title="Energy",
        unit="z re voiced median",
        why=(
            "Loudness in dB relative to this speaker's voiced median, then z-scored, "
            "so microphone gain cannot masquerade as projection. Sagging here is low "
            "energy or fading."
        ),
    ),
    PanelSpec(
        signal="rate_local_syll_per_s",
        title="Local rate",
        unit="syllables / s",
        why=(
            "Syllables per second over a short window of words, held constant across "
            "each word. Above the band is rushing, below it is dragging."
        ),
    ),
)
"""The three panels spec 15.2 asks for, in order."""


@dataclass(frozen=True)
class Series:
    """One line, or one band, ready to hand to a plotting library.

    Attributes:
        t: Times in seconds, with ``NaN`` separators between words.
        value: The participant's curve, or the envelope median.
        lower: Band floor (median - sigma), when this is an envelope series.
        upper: Band ceiling (median + sigma).
        n_references: Readers behind the band, 0 for a participant series.
    """

    t: np.ndarray
    value: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    n_references: int = 0

    def __bool__(self) -> bool:
        """Whether there is anything finite to draw."""
        return bool(self.t.size) and bool(np.isfinite(self.value).any())


def _concat(blocks: list[np.ndarray]) -> np.ndarray:
    """Join per-word blocks with a ``NaN`` between them, so the line breaks."""
    if not blocks:
        return np.empty(0, dtype=np.float64)
    separator = np.array([np.nan], dtype=np.float64)
    out: list[np.ndarray] = []
    for i, block in enumerate(blocks):
        if i:
            out.append(separator)
        out.append(np.asarray(block, dtype=np.float64))
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# The view
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisView:
    """Everything the dashboard reads off one analysis.

    Attributes:
        result: The analysis.
        config: Configuration, for the deviation cap and the category order.
    """

    result: AnalysisResult
    config: Config = field(default_factory=load_config)

    # -- curves ------------------------------------------------------------

    @property
    def _curves(self) -> dict[str, Any]:
        return dict(self.result.curves or {})

    @property
    def _band(self) -> dict[str, Any]:
        return dict(self.result.envelope_curves or {})

    @property
    def participant_signals(self) -> list[str]:
        """Signals the participant's curves carry."""
        return [str(s) for s in self._curves.get("signals", [])]

    @property
    def envelope_signals(self) -> list[str]:
        """Signals the envelope band carries; empty in Norm and Reject mode."""
        return [str(s) for s in self._band.get("signals", [])]

    @property
    def panels(self) -> list[PanelSpec]:
        """The panels that have data in this result, in spec order.

        A panel with neither a participant line nor a band is dropped rather
        than drawn empty: an axis with nothing on it reads as "the speaker did
        nothing here", which is not what it means.
        """
        available = set(self.participant_signals) | set(self.envelope_signals)
        return [p for p in PANELS if p.signal in available]

    @property
    def word_spans(self) -> dict[int, tuple[float, float]]:
        """``{word_idx: (t_start, t_end)}`` from the curves, else the alignment."""
        words = dict(self._curves.get("words", {}))
        if words:
            return {
                int(idx): (float(block["t_start"]), float(block["t_end"]))
                for idx, block in words.items()
            }
        if self.result.alignment is None:
            return {}
        return {int(w.idx): (float(w.t_start), float(w.t_end)) for w in self.result.alignment.words}

    def participant_series(self, signal: str) -> Series:
        """The participant's own curve for one signal, broken at every pause."""
        signals = self.participant_signals
        if signal not in signals:
            return Series(np.empty(0), np.empty(0))
        column = signals.index(signal)
        words = dict(self._curves.get("words", {}))

        times: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for idx in sorted(words, key=lambda k: int(k)):
            block = words[idx]
            curve = np.asarray(block.get("values", []), dtype=np.float64)
            if curve.ndim != 2 or column >= curve.shape[1]:
                continue
            t0, t1 = float(block["t_start"]), float(block["t_end"])
            times.append(np.linspace(t0, t1, curve.shape[0]))
            values.append(curve[:, column])
        return Series(_concat(times), _concat(values))

    def envelope_series(self, signal: str) -> Series:
        """The skilled readers' median and +/- 1 sigma band for one signal.

        Drawn on the participant's word spans: the within-word warp (spec 10.1)
        already put the two curves on the same normalised axis, so point *j* of
        word *i* means the same moment of the same word on both sides, and the
        participant's clock is the one a judge is listening to.
        """
        signals = self.envelope_signals
        if signal not in signals:
            return Series(np.empty(0), np.empty(0))
        column = signals.index(signal)
        words = dict(self._band.get("words", {}))
        spans = self.word_spans

        times: list[np.ndarray] = []
        medians: list[np.ndarray] = []
        lowers: list[np.ndarray] = []
        uppers: list[np.ndarray] = []
        for idx in sorted(words, key=lambda k: int(k)):
            if int(idx) not in spans:
                continue
            block = words[idx]
            median = np.asarray(block.get("median", []), dtype=np.float64)
            sigma = np.asarray(block.get("sigma", []), dtype=np.float64)
            if median.ndim != 2 or column >= median.shape[1]:
                continue
            mu = median[:, column]
            sd = sigma[:, column] if sigma.shape == median.shape else np.zeros_like(mu)
            t0, t1 = spans[int(idx)]
            times.append(np.linspace(t0, t1, mu.shape[0]))
            medians.append(mu)
            lowers.append(mu - sd)
            uppers.append(mu + sd)

        return Series(
            _concat(times),
            _concat(medians),
            _concat(lowers),
            _concat(uppers),
            n_references=int(self._band.get("n_references", self.result.n_references)),
        )

    # -- words -------------------------------------------------------------

    @property
    def _lambda_total(self) -> float:
        r"""``sum_k lambda_k``, the weights behind :math:`D_w`."""
        weights = dict(self.config.detection["deviation"]["lambda"])
        return float(sum(float(v) for v in weights.values())) or 1.0

    @property
    def deviation_floor(self) -> float:
        r"""The :math:`D_w` of a perfectly ordinary word.

        Every weighted feature exactly 1 sigma from the readers' median gives
        :math:`D_w = \sqrt{\sum_k \lambda_k}`. That is the ramp's white end:
        below it, a word is not worth a judge's attention.
        """
        configured = self.config.serving["dashboard"].get("deviation_floor")
        if configured is not None:
            return float(configured)
        return float(np.sqrt(self._lambda_total))

    @property
    def deviation_cap(self) -> float:
        r"""Where the ramp saturates: every feature at the region threshold.

        A region is entered at :math:`|z| > \tau_{in}`, so a word with *every*
        weighted feature simultaneously at that threshold scores
        :math:`\tau_{in}\sqrt{\sum_k \lambda_k}`. Nothing darker is a
        meaningful distinction - it is already as deviant as the detector can
        call a word on every axis at once.
        """
        configured = self.config.serving["dashboard"].get("deviation_cap")
        if configured is not None:
            return float(configured)
        tau_in = float(self.config.detection["regions"]["tau_in"])
        return float(tau_in * np.sqrt(self._lambda_total))

    def deviation(self, word_idx: int) -> float | None:
        r""":math:`D_w` for one word, or ``None`` when nothing was measurable."""
        block = dict(self._curves.get("deviation", {}))
        value = block.get(str(int(word_idx)))
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def top_z(self, word_idx: int) -> list[dict[str, Any]]:
        """The strongest z-scores at one word, for the transcript hover."""
        block = dict(self._curves.get("top_z", {}))
        return [dict(row) for row in block.get(str(int(word_idx)), [])]

    @property
    def has_deviation(self) -> bool:
        """Whether this result carries per-word deviations at all.

        An analysis cached before the deviation was recorded, and any Reject
        mode result, has none; the transcript is then shown uncoloured with a
        note saying so rather than tinted a misleading uniform white.
        """
        return bool(self._curves.get("deviation"))

    # -- regions -----------------------------------------------------------

    @property
    def regions(self) -> list[Region]:
        """Regions in time order."""
        return sorted(self.result.regions, key=lambda r: (r.t_start, r.t_end, r.label))

    def card_for(self, region_id: int) -> dict[str, Any]:
        """The explanation card of one region, or ``{}``."""
        for card in self.result.cards:
            if int(card.get("region_id", -1)) == int(region_id):
                return dict(card)
        return {}

    @property
    def cards(self) -> list[dict[str, Any]]:
        """Cards in time order, each paired with its region."""
        return sorted(
            (dict(c) for c in self.result.cards),
            key=lambda c: (float(c.get("t_start", 0.0)), float(c.get("t_end", 0.0))),
        )

    # -- score -------------------------------------------------------------

    @property
    def category_order(self) -> list[str]:
        """The six categories in report order."""
        return [str(c) for c in self.config.scoring["category_order"]]

    def category_scores(self) -> list[dict[str, Any]]:
        """One row per rubric category, in report order, including suppressed ones.

        A suppressed category is carried through with ``suppressed=True`` rather
        than dropped or scored 10: "this could not be measured" and "this was
        perfect" are different findings and must look different on screen
        (spec 11).
        """
        if self.result.score is None:
            return []
        by_name = {c.category: c for c in self.result.score.categories}
        rows: list[dict[str, Any]] = []
        for name in self.category_order:
            score = by_name.get(name)
            if score is None:
                continue
            rows.append(
                {
                    "category": name,
                    "score": float(score.score),
                    "band": float(score.band),
                    "deviation_mass": float(score.deviation_mass),
                    "k": float(score.k),
                    "n_regions": int(score.n_regions),
                    "suppressed": bool(score.suppressed),
                    "suppressed_reason": score.suppressed_reason,
                }
            )
        return rows

    def descriptor_rows(self) -> list[dict[str, Any]]:
        """Raw descriptors with the skilled readers' range beside them."""
        if self.result.score is None:
            return []
        rows: list[dict[str, Any]] = []
        for name, block in sorted((self.result.score.descriptors or {}).items()):
            if not isinstance(block, dict):
                continue
            row = {
                "descriptor": name.replace("_", " "),
                "value": block.get("value"),
                "reference_low": block.get("reference_low"),
                "reference_median": block.get("reference_median"),
                "reference_high": block.get("reference_high"),
                "n_references": block.get("n_references", 0),
            }
            rows.append(row)
        return rows

    # -- summary -----------------------------------------------------------

    @property
    def overall(self) -> float | None:
        """Overall score out of 10, or ``None`` when nothing was scored."""
        return None if self.result.score is None else float(self.result.score.overall)

    @property
    def overall_band(self) -> float:
        """The +/- confidence band on the overall score."""
        return 0.0 if self.result.score is None else float(self.result.score.overall_band)

    @property
    def time_range(self) -> tuple[float, float]:
        """The x-axis range: the whole recording, not just the flagged parts.

        Spec 15.2 is explicit that flaws are shown in the context of the whole
        speech, so the axis is the recording's duration even when every region
        sits in the first ten seconds.
        """
        duration = float(self.result.duration or 0.0)
        if duration > 0:
            return 0.0, duration
        spans = self.word_spans
        if spans:
            return 0.0, max(t1 for _, t1 in spans.values())
        return 0.0, 1.0


def overlay_panels(view: AnalysisView) -> list[tuple[PanelSpec, Series, Series]]:
    """``(panel, participant, envelope)`` for every panel with data.

    The envelope series is empty in Norm mode - pooled norms are not per-word
    curves, so there is a comparison but no band to draw - and the caller must
    say so on screen rather than leaving an unexplained bare line.
    """
    out: list[tuple[PanelSpec, Series, Series]] = []
    for panel in view.panels:
        mine = view.participant_series(panel.signal)
        theirs = view.envelope_series(panel.signal)
        if mine or theirs:
            out.append((panel, mine, theirs))
    return out
