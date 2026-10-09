"""The report's figures (prompt 20, spec 14.2), as PNG at 200 dpi and SVG.

Six figures, every one of them rendered from a file in ``results/`` rather than
from a fresh run, so a figure and the number quoted beside it in the report
cannot disagree:

``severity_curves``
    Mean category score per label against the injected rung L0-L4, with the
    standard deviation as an error bar. This is the picture of spec 14.3's
    Spearman row: the score is supposed to fall monotonically with severity.
``event_f1_by_label``
    Event-F1 at the 200 ms onset collar per flaw label, injected beside human
    where the corpus has human takes.
``onset_error_hist``
    Signed onset error over the IoU-matched pairs, which is the distribution
    behind the single MAE number.
``utmos_by_level``
    UTMOS of the edited file per label and rung, from ``dataset/qc/utmos.csv``:
    the realism gate of invariant 9. It needs the dataset, and is skipped with
    a stated reason when the dataset is not built.
``example_overlay``
    One participant against the envelope: F0 in semitones, relative energy and
    local rate, with the median +/- 1 sigma band, the detected regions and the
    ground-truth span. The one figure that needs the corpus rather than the
    metrics, so it re-runs a single fold.
``confusion_matrix``
    Flaw typing, truth against the emitted (rule) label.

Each figure is written twice - ``<name>.png`` at 200 dpi and ``<name>.svg`` -
and a figure that cannot be built writes nothing and says why, both in the log
and in the returned manifest. ``docs/figures/figures.json`` records what was
written, from which inputs, under which config hash.

Usage::

    python -m eval.figures                       # everything it can build
    python -m eval.figures --only severity_curves confusion_matrix
    python -m eval.figures --metrics results/metrics.json --out docs/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rostrum.config import Config, code_version, load_config, set_seeds

logger = logging.getLogger("eval.figures")

__all__ = [
    "DPI",
    "FIGURES",
    "FigureResult",
    "main",
    "render_all",
]

DPI: int = 200
"""Raster resolution; the spec asks for 200 dpi PNG beside the SVG."""

FORMATS: tuple[str, ...] = ("png", "svg")
"""Written for every figure, so the report can use either."""

LEVEL_ORDER: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")
"""The severity ladder, in order, for every x axis that shows rungs."""

CATEGORY_COLOURS: dict[str, str] = {
    "pace": "#0072B2",
    "pausing": "#009E73",
    "intonation": "#CC79A7",
    "emphasis": "#E69F00",
    "energy": "#56B4E9",
    "clarity": "#D55E00",
}
"""Okabe-Ito, which stays distinguishable for the common colour deficiencies.

Nothing in these figures is distinguished by colour *alone*: every series also
carries a label, a marker or a hatch.
"""

FIGURES: tuple[str, ...] = (
    "severity_curves",
    "event_f1_by_label",
    "onset_error_hist",
    "utmos_by_level",
    "example_overlay",
    "confusion_matrix",
)
"""Figure names, in the order :func:`render_all` builds them."""


@dataclass
class FigureResult:
    """What happened to one figure.

    Attributes:
        name: Figure name.
        paths: Files written, PNG first.
        built: Whether anything was written.
        reason: Why not, when nothing was.
        n: How many observations went into it, for the caption.
        caption: One line the report can use verbatim.
        source: Where the data came from.
    """

    name: str
    paths: list[str] = field(default_factory=list)
    built: bool = False
    reason: str = ""
    n: int = 0
    caption: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "name": self.name,
            "built": self.built,
            "paths": self.paths,
            "reason": self.reason,
            "n": self.n,
            "caption": self.caption,
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #


def _pyplot() -> Any:
    """``matplotlib.pyplot`` with the Agg backend and the project's defaults."""
    import matplotlib

    matplotlib.use("Agg")  # no display in CI or over SSH
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )
    return plt


def _save(figure: Any, out_dir: Path, name: str) -> list[str]:
    """Write ``name`` in every format and return the paths, PNG first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for suffix in FORMATS:
        path = out_dir / f"{name}.{suffix}"
        figure.savefig(path, format=suffix)
        written.append(str(path))
    import matplotlib.pyplot as plt

    plt.close(figure)
    logger.info("wrote %s", ", ".join(written))
    return written


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    """Mean and population-free standard deviation; ``sd`` is 0 for one value."""
    array = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def _folds(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every fold payload of an aggregate, in fold order."""
    per_fold = payload.get("per_fold") or {}
    return [per_fold[key] for key in sorted(per_fold)]


def _corpus_note(payload: Mapping[str, Any]) -> str:
    """The caveat line for a figure's title, when the corpus is generated."""
    if str(payload.get("corpus")) == "synthetic":
        return " (synthetic corpus - generated feature tables)"
    return " (dataset corpus)"


# --------------------------------------------------------------------------- #
# 1. Severity curves
# --------------------------------------------------------------------------- #


def severity_curves(payload: Mapping[str, Any], out_dir: Path) -> FigureResult:
    """Mean category score against rung, per label, with the spread.

    The score plotted for a file is the score of the category its *own* flaw
    belongs to, which is the quantity spec 14.3's Spearman row correlates with
    severity. A clean file (L0) contributes to every label it has a sibling
    for, since L0 is the same recording for all of them.
    """
    result = FigureResult(name="severity_curves", source="results/metrics.json per_fold")
    # {label: {level: [score, ...]}}
    collected: dict[str, dict[str, list[float]]] = {}
    clean: list[float] = []
    for fold in _folds(payload):
        cases = fold.get("cases") or {}
        categories = fold.get("scores", {}).get("categories", {})
        for file_id, case in cases.items():
            scores = categories.get(file_id) or {}
            label, level, category = case.get("label"), case.get("level"), case.get("category")
            if not case.get("n_truth"):
                overall = fold.get("scores", {}).get("overall", {}).get(file_id)
                if overall is not None:
                    clean.append(float(overall))
                continue
            if not label or not category or category not in scores:
                continue
            collected.setdefault(str(label), {}).setdefault(str(level), []).append(
                float(scores[category])
            )

    if not collected:
        result.reason = "no held-out file carries both a label and a category score"
        return result

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    n = 0
    for i, label in enumerate(sorted(collected)):
        rungs = [lv for lv in LEVEL_ORDER if lv in collected[label]]
        means, sds = [], []
        for level in rungs:
            mean, sd = _mean_sd(collected[label][level])
            means.append(mean)
            sds.append(0.0 if math.isnan(sd) else sd)
            n += len(collected[label][level])
        axes.errorbar(
            [LEVEL_ORDER.index(lv) for lv in rungs],
            means,
            yerr=sds,
            marker=markers[i % len(markers)],
            capsize=3,
            linewidth=1.4,
            label=label,
        )
    if clean:
        mean, _ = _mean_sd(clean)
        axes.axhline(
            mean,
            linestyle=":",
            color="0.35",
            linewidth=1.2,
            label=f"clean files, overall ({mean:.2f})",
        )
    axes.set_xticks(range(len(LEVEL_ORDER)), LEVEL_ORDER)
    axes.set_xlabel("injected severity rung")
    axes.set_ylabel("category score (0-10)")
    axes.set_ylim(0, 10.5)
    axes.set_title("Category score falls with injected severity" + _corpus_note(payload))
    axes.legend(fontsize=7, ncol=2, loc="lower left")
    result.paths = _save(figure, out_dir, result.name)
    result.built = True
    result.n = n
    result.caption = (
        f"Mean category score per label against the injected rung, error bars 1 sd, "
        f"{n} held-out file(s){_corpus_note(payload)}."
    )
    return result


# --------------------------------------------------------------------------- #
# 2. Event-F1 per label
# --------------------------------------------------------------------------- #


def event_f1_by_label(payload: Mapping[str, Any], out_dir: Path) -> FigureResult:
    """Per-label event-F1 at the 200 ms collar, injected beside human."""
    result = FigureResult(name="event_f1_by_label", source="results/metrics.json per_fold")
    per_split: dict[str, dict[str, list[float]]] = {"injected": {}, "human": {}}
    for fold in _folds(payload):
        for split in ("injected", "human"):
            block = fold.get("splits", {}).get(split, {})
            if not block.get("available"):
                continue
            for label, counts in (
                block.get("events", {}).get("rostrum", {}).get("per_label", {}).items()
            ):
                if int(counts.get("tp", 0)) + int(counts.get("fn", 0)) == 0:
                    continue  # never asked about on this fold
                per_split[split].setdefault(str(label), []).append(float(counts.get("f1", 0.0)))

    labels = sorted(set(per_split["injected"]) | set(per_split["human"]))
    if not labels:
        result.reason = "no fold reported a per-label event score"
        return result

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.4, 4.0))
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    for offset, (split, hatch, colour) in zip(
        (-width / 2, width / 2),
        (("injected", "", "#0072B2"), ("human", "//", "#D55E00")),
        strict=True,
    ):
        means = [_mean_sd(per_split[split].get(label, []))[0] for label in labels]
        sds = [_mean_sd(per_split[split].get(label, []))[1] for label in labels]
        present = [0.0 if math.isnan(m) else m for m in means]
        errors = [0.0 if math.isnan(s) else s for s in sds]
        axes.bar(
            x + offset,
            present,
            width,
            yerr=errors,
            capsize=2,
            label=f"{split} ({'test-only' if split == 'human' else 'synthetic edits'})",
            color=colour,
            hatch=hatch,
            edgecolor="white",
        )
        if all(math.isnan(m) for m in means):
            axes.text(
                0.5,
                0.9,
                f"no {split} takes in this corpus",
                transform=axes.transAxes,
                ha="center",
                fontsize=8,
                color=colour,
            )
    for key, value in payload.get("targets", {}).items():
        for row in value:
            if row["key"] == "event_f1_200ms" and row["target"] is not None:
                axes.axhline(
                    float(row["target"]),
                    linestyle="--",
                    linewidth=1.0,
                    color="0.4" if key == "synthetic" else "0.65",
                )
                axes.text(
                    len(labels) - 0.4,
                    float(row["target"]) + 0.01,
                    f"{key} target",
                    fontsize=7,
                    ha="right",
                    color="0.35",
                )
    axes.set_xticks(x, labels, rotation=20, ha="right")
    axes.set_ylabel("event-F1 (200 ms onset collar)")
    axes.set_ylim(0, 1.05)
    axes.set_title("Grounding per flaw label" + _corpus_note(payload))
    axes.legend(fontsize=8)
    result.paths = _save(figure, out_dir, result.name)
    result.built = True
    result.n = len(labels)
    result.caption = (
        f"Event-F1 per label, mean over folds, 200 ms onset collar and "
        f"max(200 ms, 20%) offset collar{_corpus_note(payload)}."
    )
    return result


# --------------------------------------------------------------------------- #
# 3. Onset error histogram
# --------------------------------------------------------------------------- #


def onset_error_hist(payload: Mapping[str, Any], out_dir: Path) -> FigureResult:
    """Signed onset error over the IoU-matched pairs of every fold."""
    result = FigureResult(name="onset_error_hist", source="results/metrics.json diagnostics")
    errors: dict[str, list[float]] = {"injected": [], "human": []}
    for fold in _folds(payload):
        for split in ("injected", "human"):
            block = fold.get("splits", {}).get(split, {})
            if not block.get("available"):
                continue
            for pair in block.get("diagnostics", {}).get("pairs", []):
                errors[split].append(float(pair["onset_error_ms"]))

    pooled = errors["injected"] + errors["human"]
    if not pooled:
        result.reason = "no matched pair in any fold, so there is no boundary error to plot"
        return result

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(6.4, 3.8))
    # Scale to the bulk, not to the worst case: one 400 ms outlier among 70
    # regions otherwise compresses the whole distribution into a single bar.
    # The tail is not hidden - it is clipped into the end bins and counted in
    # the axis label, so the figure cannot understate how far an edge can move.
    span = max(60.0, float(np.ceil(np.percentile(np.abs(pooled), 95) / 20.0) * 20.0))
    beyond = int(np.sum(np.abs(pooled) > span))
    bins = np.linspace(-span, span, 25)
    axes.hist(
        [np.clip(errors["injected"], -span, span), np.clip(errors["human"], -span, span)],
        bins=bins,
        stacked=True,
        color=["#0072B2", "#D55E00"],
        label=[f"injected (n={len(errors['injected'])})", f"human (n={len(errors['human'])})"],
        edgecolor="white",
    )
    mae = float(np.mean(np.abs(pooled)))
    axes.axvline(0.0, color="0.3", linewidth=1.0)
    axes.axvline(
        float(np.mean(pooled)),
        color="#009E73",
        linestyle="--",
        linewidth=1.3,
        label=f"bias {np.mean(pooled):+.0f} ms, MAE {mae:.0f} ms",
    )
    axes.set_xlabel(
        "onset error, prediction minus truth (ms); positive is late"
        + (
            f"\n{beyond} region(s) beyond +/-{span:.0f} ms, clipped into the end bins"
            if beyond
            else ""
        )
    )
    axes.set_ylabel("matched regions")
    axes.set_title("Where the region edges land" + _corpus_note(payload))
    axes.legend(fontsize=8)
    result.paths = _save(figure, out_dir, result.name)
    result.built = True
    result.n = len(pooled)
    result.caption = (
        f"Signed onset error over {len(pooled)} IoU-matched region(s); MAE {mae:.0f} ms, "
        f"bias {np.mean(pooled):+.0f} ms, {beyond} beyond the +/-{span:.0f} ms axis"
        f"{_corpus_note(payload)}."
    )
    return result


# --------------------------------------------------------------------------- #
# 4. UTMOS against level
# --------------------------------------------------------------------------- #


def utmos_by_level(payload: Mapping[str, Any], out_dir: Path, *, config: Config) -> FigureResult:
    """UTMOS of each edited file against its rung, per label (invariant 9).

    Reads ``dataset/qc/utmos.csv`` and ``dataset/MANIFEST.csv``: the gate is a
    property of the dataset build, not of an evaluation run, so this figure is
    the one that does not come from ``results/``.
    """
    result = FigureResult(name="utmos_by_level", source="dataset/qc/utmos.csv + MANIFEST.csv")
    utmos_path = config.path("data.qc_utmos")
    manifest_path = config.path("data.manifest")
    if not utmos_path.is_file() or not manifest_path.is_file():
        result.reason = (
            f"needs {utmos_path} and {manifest_path}; run `make data` (the UTMOS gate is "
            "part of `make qc`)"
        )
        return result

    with utmos_path.open(encoding="utf-8", newline="") as handle:
        scores = {
            str(row["file_id"]): float(row["utmos"])
            for row in csv.DictReader(handle)
            if row.get("utmos")
        }
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    collected: dict[str, dict[str, list[float]]] = {}
    clean: list[float] = []
    for row in rows:
        file_id = str(row.get("file_id") or "")
        if file_id not in scores:
            continue
        if str(row.get("origin")) == "clean":
            clean.append(scores[file_id])
            continue
        label, level = str(row.get("label") or ""), str(row.get("level") or "")
        if not label or level not in LEVEL_ORDER:
            continue
        collected.setdefault(label, {}).setdefault(level, []).append(scores[file_id])

    if not collected:
        result.reason = "no scored file in the UTMOS cache has a label and a rung"
        return result

    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    n = 0
    for i, label in enumerate(sorted(collected)):
        rungs = [lv for lv in LEVEL_ORDER if lv in collected[label]]
        means = [_mean_sd(collected[label][lv])[0] for lv in rungs]
        sds = [_mean_sd(collected[label][lv])[1] or 0.0 for lv in rungs]
        n += sum(len(collected[label][lv]) for lv in rungs)
        axes.errorbar(
            [LEVEL_ORDER.index(lv) for lv in rungs],
            means,
            yerr=sds,
            marker=markers[i % len(markers)],
            capsize=3,
            linewidth=1.3,
            label=label,
        )
    if clean:
        mean, _ = _mean_sd(clean)
        axes.axhline(mean, linestyle=":", color="0.3", label=f"clean sources ({mean:.2f})")
        drop = float(config.scoring["targets"]["synthetic"].get("utmos_drop_at_l1", 0.3))
        axes.axhline(
            mean - drop,
            linestyle="--",
            color="#D55E00",
            linewidth=1.1,
            label=f"L1 gate: clean - {drop:g}",
        )
    axes.set_xticks(range(len(LEVEL_ORDER)), LEVEL_ORDER)
    axes.set_xlabel("severity rung")
    axes.set_ylabel("UTMOS (predicted MOS)")
    axes.set_title("Edit realism against severity (dataset QC)")
    axes.legend(fontsize=7, ncol=2)
    result.paths = _save(figure, out_dir, result.name)
    result.built = True
    result.n = n
    result.caption = (
        f"UTMOS of {n} edited file(s) against the rung, per label, with the clean baseline "
        "and the L1 gate of invariant 9."
    )
    return result


# --------------------------------------------------------------------------- #
# 5. Example overlay
# --------------------------------------------------------------------------- #


def example_overlay(
    out_dir: Path,
    *,
    config: Config,
    corpus: str = "synthetic",
    file_id: str = "",
    texts: Sequence[str] = (),
    held_out: str = "",
) -> FigureResult:
    """One participant against the envelope band, with the regions marked.

    This is the only figure that needs the corpus rather than a results file,
    so it re-runs one fold. It picks a flawed held-out file - the one named, or
    else the first whose flaw the detector found, so the figure shows the
    system working rather than a case chosen for being pretty.

    Args:
        out_dir: Where to write the figure.
        config: Loaded config.
        corpus: ``"synthetic"`` or ``"dataset"``.
        file_id: The recording to plot; the first flawed one otherwise.
        texts: The corpus's texts, so the fold rebuilt here is the fold the
            metrics file reported rather than a differently sized one.
        held_out: Which text to hold out; the first of ``texts`` otherwise.

    Returns:
        A :class:`FigureResult`.
    """
    result = FigureResult(name="example_overlay", source=f"{corpus} corpus, one fold")
    from eval.cases import build_fold_cases
    from rostrum.score.rubric import label_to_category

    if corpus == "dataset":
        from eval.dataset_corpus import CorpusUnavailable, DatasetCorpus

        try:
            data = DatasetCorpus.load(config)
            bundle = data.fold(config=config, held_out=held_out or data.texts[0])
        except (CorpusUnavailable, IndexError) as error:
            result.reason = str(error)
            return result
    else:
        from eval.run_loto import DEFAULT_TEXTS

        fold_texts = tuple(texts) or DEFAULT_TEXTS
        bundle = build_fold_cases(
            config=config, held_out=held_out or fold_texts[0], texts=fold_texts
        )

    flawed = [case for case in bundle.test() if case.truth]
    if not flawed:
        result.reason = "the held-out text has no flawed file to plot"
        return result
    # The rule, so the choice is not a judgement about which plot looks best:
    # the named file, else the most severe flaw the detector actually found,
    # with the file id breaking ties. A figure of an L1 edit would show a
    # participant sitting inside the band, which is what L1 means and not what
    # this figure is for.
    detected = [case for case in flawed if case.regions] or flawed
    chosen = next(
        (case for case in flawed if case.file_id == file_id),
        max(
            detected,
            key=lambda case: (
                float(case.truth[0].s) if case.truth else 0.0,
                case.file_id,
            ),
        ),
    )
    frames = chosen.frames
    if frames is None or "time_s" not in getattr(frames, "columns", []):
        result.reason = f"{chosen.file_id} has no frame curves to plot"
        return result

    envelope = bundle.envelopes[chosen.text_id]
    curves = envelope.curves
    words = chosen.tables["word"]
    spans = {int(idx): (float(r["t_start"]), float(r["t_end"])) for idx, r in words.iterrows()}

    # Panels: the frame signals that exist, then rate from the word table.
    panels: list[tuple[str, str, str]] = []
    if "f0_st" in frames.columns:
        panels.append(("f0_st", "F0 (semitones re median)", "frames"))
    for column in ("energy_z", "energy_db_rel"):
        if column in frames.columns:
            panels.append((column, "energy (z re voiced median)", "frames"))
            break
    rate_column = next(
        (c for c in ("rate_local_syll_per_s", "rate_local") if c in words.columns), ""
    )
    if rate_column:
        panels.append((rate_column, "local rate (syll/s)", "words"))
    if not panels:
        result.reason = "no plottable signal in this recording"
        return result

    plt = _pyplot()
    figure, axes = plt.subplots(
        len(panels), 1, figsize=(9.0, 2.3 * len(panels)), sharex=True, squeeze=False
    )
    times = np.asarray(frames["time_s"], dtype=float)
    categories = label_to_category(config)

    for row, (column, ylabel, source) in enumerate(panels):
        axis = axes[row][0]
        if source == "frames":
            values = np.asarray(frames[column], dtype=float)
            axis.plot(times, values, color="#111111", linewidth=1.1, label="participant")
            if curves is not None and column in curves.signals:
                first = True
                for idx, (t0, t1) in sorted(spans.items()):
                    if idx not in curves.median:
                        continue
                    x, low, high = curves.band(idx, column)
                    axis.fill_between(
                        t0 + x * (t1 - t0),
                        low,
                        high,
                        color="#0072B2",
                        alpha=0.18,
                        linewidth=0,
                        label="skilled readers, median +/- 1 sigma" if first else None,
                    )
                    mid = curves.median[idx][:, curves.signals.index(column)]
                    axis.plot(t0 + x * (t1 - t0), mid, color="#0072B2", linewidth=0.9)
                    first = False
        else:
            unit = envelope.unit("word")
            xs, ys, los, his = [], [], [], []
            for idx, (t0, t1) in sorted(spans.items()):
                value = words[column].get(idx)
                if value is None or not math.isfinite(float(value)):
                    continue
                xs += [t0, t1]
                ys += [float(value), float(value)]
                feature = next(
                    (f for f in unit.features if f.startswith("rate_local")),
                    "",
                )
                if feature and idx in unit.envelope.median:
                    col = unit.column(feature)
                    mu = float(unit.envelope.median[idx][col])
                    sigma = max(
                        float(unit.envelope.sigma[idx][col]),
                        float(unit.envelope.sigma_floor.get(feature, 0.0)),
                    )
                    los += [mu - sigma, mu - sigma]
                    his += [mu + sigma, mu + sigma]
                else:
                    los += [np.nan, np.nan]
                    his += [np.nan, np.nan]
            axis.plot(xs, ys, color="#111111", linewidth=1.1, label="participant")
            if any(math.isfinite(v) for v in los):
                axis.fill_between(
                    xs,
                    los,
                    his,
                    color="#0072B2",
                    alpha=0.18,
                    linewidth=0,
                    label="skilled readers, median +/- 1 sigma",
                )

        for flaw in chosen.truth:
            axis.axvspan(
                float(flaw.t_start),
                float(flaw.t_end),
                color="0.6",
                alpha=0.22,
                linewidth=0,
                label="ground truth" if row == 0 and flaw is chosen.truth[0] else None,
            )
        for region in chosen.regions:
            colour = CATEGORY_COLOURS.get(categories.get(region.label, ""), "#555555")
            axis.axvspan(
                float(region.t_start),
                float(region.t_end),
                color=colour,
                alpha=0.16,
                linewidth=0,
            )
            if row == 0:
                axis.annotate(
                    f"{region.label}",
                    xy=(float(region.t_start), 0.98),
                    xycoords=("data", "axes fraction"),
                    fontsize=7,
                    rotation=90,
                    va="top",
                    color=colour,
                )
        axis.set_ylabel(ylabel, fontsize=8)
        if row == 0:
            axis.legend(fontsize=7, loc="lower left", ncol=3)

    bottom = axes[-1][0]
    bottom.set_xlabel("time (s)")
    bottom.set_xticks(
        [round(t0, 2) for t0, _ in sorted(spans.values())[:: max(1, len(spans) // 12)]],
        minor=True,
    )
    truth = ", ".join(f"{f.label} {f.level}" for f in chosen.truth) or "none"
    axes[0][0].set_title(
        f"{chosen.file_id}: truth {truth}; {len(chosen.regions)} region(s) found"
        + (" (synthetic corpus)" if corpus == "synthetic" else " (dataset corpus)")
    )
    result.paths = _save(figure, out_dir, result.name)
    result.built = True
    result.n = len(chosen.regions)
    result.caption = (
        f"{chosen.file_id} against the envelope of {envelope.n_references} skilled reading(s): "
        f"truth {truth}, {len(chosen.regions)} detected region(s)."
    )
    return result


# --------------------------------------------------------------------------- #
# 6. Confusion matrix
# --------------------------------------------------------------------------- #


def confusion_matrix(payload: Mapping[str, Any], out_dir: Path) -> FigureResult:
    """Truth against the emitted label, summed over folds.

    The emitted label is the **rule's** (spec 10.6's label policy), which is
    what a card shows, so that is what this matrix scores.
    """
    result = FigureResult(name="confusion_matrix", source="results/metrics.json typing.rule")
    labels: list[str] = []
    total: np.ndarray | None = None
    for fold in _folds(payload):
        block = fold.get("splits", {}).get("injected", {})
        if not block.get("available"):
            continue
        typing = block.get("typing", {}).get("rule", {})
        matrix, names = typing.get("confusion"), typing.get("labels")
        if not matrix or not names:
            continue
        if not labels:
            labels = [str(name) for name in names]
            total = np.zeros((len(labels), len(labels)), dtype=float)
        if [str(name) for name in names] != labels or total is None:
            logger.warning("fold %s has a different label set; skipped", fold.get("fold"))
            continue
        total += np.asarray(matrix, dtype=float)

    if total is None or not total.any():
        result.reason = "no fold reported a typing confusion matrix"
        return result

    plt = _pyplot()
    size = max(4.2, 0.62 * len(labels) + 2.0)
    figure, axes = plt.subplots(figsize=(size, size * 0.92))
    rows = total.sum(axis=1, keepdims=True)
    shown = np.divide(total, rows, out=np.zeros_like(total), where=rows > 0)
    image = axes.imshow(shown, cmap="Blues", vmin=0.0, vmax=1.0)
    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if total[i, j] == 0:
                continue
            axes.text(
                j,
                i,
                f"{int(total[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if shown[i, j] > 0.55 else "#222222",
            )
    axes.set_xlabel("emitted label (rule)")
    axes.set_ylabel("ground truth")
    axes.set_title("Flaw typing over matched regions" + _corpus_note(payload))
    figure.colorbar(image, ax=axes, shrink=0.8, label="row share")
    result.paths = _save(figure, out_dir, result.name)
    result.built = True
    result.n = int(total.sum())
    result.caption = (
        f"Typing confusion over {int(total.sum())} matched region(s); cells are counts, "
        f"shading is the row share{_corpus_note(payload)}."
    )
    return result


# --------------------------------------------------------------------------- #
# All of them
# --------------------------------------------------------------------------- #


def render_all(
    *,
    config: Config,
    metrics: Path | None = None,
    out_dir: Path | None = None,
    only: Sequence[str] | None = None,
    corpus: str | None = None,
    overlay_file_id: str = "",
) -> dict[str, Any]:
    """Render every figure that can be built and write ``figures.json``.

    Args:
        config: Loaded config.
        metrics: ``results/metrics.json``; the configured path by default.
        out_dir: Output directory; ``docs/figures`` by default.
        only: Build just these figure names.
        corpus: Corpus for the overlay figure; the metrics file's by default.
        overlay_file_id: Which recording to put in the overlay figure.

    Returns:
        The manifest written to ``<out_dir>/figures.json``.

    Raises:
        FileNotFoundError: There is no metrics file to read.
    """
    metrics_path = metrics or config.path("results.metrics")
    out = out_dir or config.path("docs.figures")
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"no metrics at {metrics_path}; run `make eval` first - every figure except the "
            "overlay is rendered from it"
        )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    wanted = set(only) if only else set(FIGURES)
    overlay_corpus = corpus or str(payload.get("corpus") or "synthetic")
    # Rebuild the same fold the metrics file reported, not a differently sized
    # one: a fold over two texts has different sigma floors from one over three.
    fold_texts = tuple(str(fold).removeprefix("loto_") for fold in (payload.get("folds") or ()))

    results: list[FigureResult] = []
    for name in FIGURES:
        if name not in wanted:
            continue
        try:
            if name == "severity_curves":
                results.append(severity_curves(payload, out))
            elif name == "event_f1_by_label":
                results.append(event_f1_by_label(payload, out))
            elif name == "onset_error_hist":
                results.append(onset_error_hist(payload, out))
            elif name == "utmos_by_level":
                results.append(utmos_by_level(payload, out, config=config))
            elif name == "example_overlay":
                results.append(
                    example_overlay(
                        out,
                        config=config,
                        corpus=overlay_corpus,
                        file_id=overlay_file_id,
                        texts=fold_texts,
                    )
                )
            elif name == "confusion_matrix":
                results.append(confusion_matrix(payload, out))
        except Exception as error:  # pragma: no cover - a plotting failure
            logger.exception("%s: failed", name)
            results.append(FigureResult(name=name, reason=f"{type(error).__name__}: {error}"))

    manifest = {
        "schema": "rostrum.eval.figures.v1",
        "corpus": payload.get("corpus"),
        "overlay_corpus": overlay_corpus,
        "metrics": str(metrics_path),
        "config_hash": config.hash,
        "code_version": code_version(config.root),
        "dpi": DPI,
        "formats": list(FORMATS),
        "figures": [item.to_dict() for item in results],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    built = sum(1 for item in results if item.built)
    logger.info("%d of %d figure(s) built into %s", built, len(results), out)
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints one line per figure."""
    parser = argparse.ArgumentParser(description="Report figures (prompt 20, spec 14.2)")
    parser.add_argument("--metrics", type=Path, default=None, help="results/metrics.json")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--only", nargs="*", default=None, choices=list(FIGURES))
    parser.add_argument(
        "--corpus",
        choices=("synthetic", "dataset"),
        default=None,
        help="corpus for the overlay figure (default: the metrics file's)",
    )
    parser.add_argument("--file-id", default="", help="recording for the overlay figure")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    set_seeds(config.seed)
    try:
        manifest = render_all(
            config=config,
            metrics=args.metrics,
            out_dir=args.out,
            only=args.only,
            corpus=args.corpus,
            overlay_file_id=args.file_id,
        )
    except FileNotFoundError as error:
        print(f"cannot render figures: {error}", file=sys.stderr)
        return 2

    for figure in manifest["figures"]:
        if figure["built"]:
            print(f"  {figure['name']:<20} n={figure['n']:<5} {figure['paths'][0]}")
        else:
            print(f"  {figure['name']:<20} NOT BUILT: {figure['reason']}")
    built = sum(1 for f in manifest["figures"] if f["built"])
    print(f"{built} of {len(manifest['figures'])} figure(s) built")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
