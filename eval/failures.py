"""Where it fails: the three largest misses, with a suspected cause (spec 14.3).

Spec 14.3 asks the report to carry a "Where it fails" section with three
concrete failure cases, and prompt 20 fixes which three:

* the **false negative** with the highest severity - the flaw the system should
  have found most easily and did not;
* the **false positive** with the highest region score - the thing it was most
  confident about and that is not there;
* the **worst boundary error** among the matched regions - found, but in the
  wrong place.

Everything comes from the per-event records :mod:`eval.run_loto` writes into
each fold's ``diagnostics`` block, so a case named here is the same event the
aggregate numbers counted; nothing is re-derived and nothing is hand-picked.

**The cause is a hypothesis.** The evidence is measured - what overlapped what,
which label was emitted, how far the edge moved - and the sentence that follows
from it is an inference, labelled as one in the output. Reading a cause here as
a finding would be exactly the kind of unearned claim the rest of this project
is built to avoid.

Usage::

    python -m eval.failures                      # results/metrics.json -> failures.md
    python -m eval.failures --metrics results/metrics.json --out results/failures.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rostrum.config import Config, code_version, load_config
from rostrum.detect.tune import (
    DEFAULT_OFFSET_COLLAR_FRAC,
    DEFAULT_OFFSET_COLLAR_S,
    DEFAULT_ONSET_COLLAR_S,
)

logger = logging.getLogger("eval.failures")

__all__ = [
    "KINDS",
    "Failure",
    "collect_failures",
    "failures_markdown",
    "main",
    "write_failures",
]

KINDS: tuple[tuple[str, str], ...] = (
    ("false_negative", "Missed flaw with the highest severity"),
    ("false_positive", "False alarm with the highest region score"),
    ("boundary", "Worst boundary error among the regions that matched"),
)
"""The three failure kinds, in report order."""

PAUSE_LABELS: frozenset[str] = frozenset({"missing_pause", "hesitation_pause"})
"""Labels whose evidence is a gap rather than a stretch of speech."""


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Seconds of overlap between two spans given as ``t_start``/``t_end``."""
    start = max(float(a["t_start"]), float(b["t_start"]))
    end = min(float(a["t_end"]), float(b["t_end"]))
    return max(0.0, end - start)


@dataclass
class Failure:
    """One failure case, with the measurements behind it.

    Attributes:
        kind: One of :data:`KINDS`' keys.
        title: Row title for the report.
        found: Whether a case of this kind exists at all.
        reason: Why not, when none does.
        file_id: The recording - an audio file id, so it can be listened to.
        fold: The fold it was held out in.
        split: ``injected`` or ``human``.
        label: The label of the event.
        what: What happened, in one measured sentence.
        cause: The suspected cause, stated as a hypothesis.
        evidence: The numbers the two sentences are built from.
    """

    kind: str
    title: str
    found: bool = False
    reason: str = ""
    file_id: str = ""
    fold: str = ""
    split: str = ""
    label: str = ""
    what: str = ""
    cause: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "kind": self.kind,
            "title": self.title,
            "found": self.found,
            "reason": self.reason,
            "file_id": self.file_id,
            "fold": self.fold,
            "split": self.split,
            "label": self.label,
            "what": self.what,
            "cause": self.cause,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------------------- #
# Gathering
# --------------------------------------------------------------------------- #


def _rows(
    payload: Mapping[str, Any],
    *,
    merge_gap_s: float = 0.200,
    refinement_window_s: float = 0.300,
) -> list[dict[str, Any]]:
    """Every diagnostic record of every fold and split, flattened.

    Each row carries the fold, the split, the kind, the case register entry for
    its file and the two region-formation constants the cause heuristics need -
    which keeps those heuristics from hard-coding a number the config owns.

    Args:
        payload: The aggregate payload.
        merge_gap_s: ``detection.regions.merge_gap_s``.
        refinement_window_s: ``detection.refinement.window_s``.

    Returns:
        One row per diagnostic record.
    """
    out: list[dict[str, Any]] = []
    for fold_id, fold in sorted((payload.get("per_fold") or {}).items()):
        cases = fold.get("cases") or {}
        for split in ("injected", "human"):
            block = fold.get("splits", {}).get(split, {})
            if not block.get("available"):
                continue
            diagnostics = block.get("diagnostics") or {}
            for kind, key in (
                ("false_negative", "false_negatives"),
                ("false_positive", "false_positives"),
                ("boundary", "pairs"),
            ):
                for record in diagnostics.get(key, []):
                    out.append(
                        {
                            "kind": kind,
                            "fold": fold_id,
                            "split": split,
                            "record": dict(record),
                            "case": dict(cases.get(str(record.get("file_id")), {})),
                            "siblings": diagnostics,
                            "variant": fold.get("variant", {}),
                            "thresholds": fold.get("fit", {}).get("thresholds", {}),
                            "merge_gap_s": merge_gap_s,
                            "refinement_window_s": refinement_window_s,
                        }
                    )
    return out


def _predictions_in(diagnostics: Mapping[str, Any], file_id: str) -> list[dict[str, Any]]:
    """Every predicted region in one file, with how each matching treated it.

    The two matchings are not the same question and a region can answer them
    differently: ``pairs`` is IoU-matched, which is what the boundary error is
    measured over, and ``false_positives`` is what the **collar** matching
    rejected, which is what the event-F1 counts. A region that overlaps a flaw
    well but starts 300 ms late is in both lists, so the records are merged on
    the span and carry both flags rather than appearing twice.
    """
    merged: dict[tuple[str, float, float], dict[str, Any]] = {}

    def add(span: Mapping[str, Any], **flags: Any) -> None:
        key = (str(span["label"]), float(span["t_start"]), float(span["t_end"]))
        record = merged.setdefault(
            key,
            {
                "t_start": span["t_start"],
                "t_end": span["t_end"],
                "label": span["label"],
                "score": span.get("score"),
                "iou_matched": False,
                "rejected_by_collar": False,
            },
        )
        record.update(flags)

    for pair in diagnostics.get("pairs", []):
        if str(pair.get("file_id")) == file_id:
            add(
                {
                    "t_start": pair["pred_start"],
                    "t_end": pair["pred_end"],
                    "label": pair["label"],
                    "score": pair.get("score"),
                },
                iou_matched=True,
            )
    for row in diagnostics.get("false_positives", []):
        if str(row.get("file_id")) == file_id:
            add(row, rejected_by_collar=True)
    return [merged[key] for key in sorted(merged)]


def _truth_in(diagnostics: Mapping[str, Any], file_id: str) -> list[dict[str, Any]]:
    """Every ground-truth event in one file: the matched ones and the missed ones.

    One truth event can appear in several pairs - two regions of its label can
    both overlap it, while only one is matched - so the list is de-duplicated on
    the span and the label, or the evidence block shows the same flaw twice.
    """
    seen: set[tuple[str, float, float]] = set()
    out: list[dict[str, Any]] = []
    candidates = [
        {"t_start": pair["ref_start"], "t_end": pair["ref_end"], "label": pair["label"]}
        for pair in diagnostics.get("pairs", [])
        if str(pair.get("file_id")) == file_id
    ] + [
        {"t_start": row["t_start"], "t_end": row["t_end"], "label": row["label"]}
        for row in diagnostics.get("false_negatives", [])
        if str(row.get("file_id")) == file_id
    ]
    for span in candidates:
        key = (str(span["label"]), float(span["t_start"]), float(span["t_end"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def _false_negative(row: Mapping[str, Any]) -> Failure:
    """Describe a missed flaw and infer why it was missed."""
    record, case, diagnostics = row["record"], row["case"], row["siblings"]
    file_id = str(record["file_id"])
    predictions = _predictions_in(diagnostics, file_id)
    overlapping = sorted(
        (p for p in predictions if _overlap(record, p) > 0.0),
        key=lambda p: -_overlap(record, p),
    )
    duration = float(record["t_end"]) - float(record["t_start"])
    failure = Failure(
        kind="false_negative",
        title=dict(KINDS)["false_negative"],
        found=True,
        file_id=file_id,
        fold=str(row["fold"]),
        split=str(row["split"]),
        label=str(record["label"]),
        evidence={
            "span_s": [record["t_start"], record["t_end"]],
            "duration_s": round(duration, 3),
            "severity_s": case.get("severity_s"),
            "level": case.get("level"),
            "n_regions_in_file": case.get("n_regions"),
            "overlapping_predictions": overlapping[:3],
            "tau_in": row["thresholds"].get("tau_in"),
            "refined": row["variant"].get("refine"),
        },
    )
    failure.what = (
        f"`{record['label']}` at {record['t_start']:.3f}-{record['t_end']:.3f} s "
        f"({duration:.2f} s, {case.get('level', '?')}, s={case.get('severity_s')}) was not "
        f"reported: no region of that label matched it within the collars."
    )
    if not predictions:
        failure.cause = (
            "Suspected cause: nothing in this file exceeded the enter threshold "
            f"tau_in={failure.evidence['tau_in']}, so the deviation this edit produced is "
            "smaller than the envelope's spread at those words - either the edit is weak "
            "relative to how differently the skilled readers read them, or the feature that "
            "should carry it is not in the trigger set for this comparison."
        )
    elif overlapping and str(overlapping[0]["label"]) != str(record["label"]):
        failure.cause = (
            f"Suspected cause: a region *was* found over this span, but labelled "
            f"`{overlapping[0]['label']}`. The event is therefore a typing failure rather "
            "than a grounding one, and it is counted twice - once here and once as a false "
            "positive for the other label."
        )
    elif overlapping:
        best = overlapping[0]
        share = _overlap(record, best) / max(duration, 1e-9)
        onset_err = (float(best["t_start"]) - float(record["t_start"])) * 1000.0
        offset_err = (float(best["t_end"]) - float(record["t_end"])) * 1000.0
        onset_collar = DEFAULT_ONSET_COLLAR_S * 1000.0
        offset_collar = max(DEFAULT_OFFSET_COLLAR_S, DEFAULT_OFFSET_COLLAR_FRAC * duration) * 1000.0
        exceeded = []
        if abs(onset_err) > onset_collar:
            exceeded.append(
                f"the onset is {onset_err:+.0f} ms, past the {onset_collar:.0f} ms collar"
            )
        if abs(offset_err) > offset_collar:
            exceeded.append(
                f"the offset is {offset_err:+.0f} ms, past the {offset_collar:.0f} ms collar "
                f"(max(200 ms, 20% of {duration:.2f} s))"
            )
        failure.evidence["best_overlap_share"] = round(share, 3)
        failure.evidence["best_overlap_onset_error_ms"] = round(onset_err, 1)
        failure.evidence["best_overlap_offset_error_ms"] = round(offset_err, 1)
        failure.evidence["collars_exceeded"] = exceeded
        failure.cause = (
            f"Suspected cause: a region of the right label covers {share * 100:.0f}% of this "
            "flaw, so it was *found* and mis-placed rather than missed: "
            + ("; ".join(exceeded) if exceeded else "both edges are inside the collars")
            + "."
            + (
                " It is close enough to be IoU-matched - the boundary error is measured over "
                "it - and too far to be counted as a detection, which is the gap between the "
                "two notions of 'matched' this report carries both of."
                if best.get("iou_matched")
                else ""
            )
        )
    elif str(record["label"]) in PAUSE_LABELS:
        failure.cause = (
            "Suspected cause: the pause rules read the gap unit, and this gap's deviation "
            "did not reach the enter threshold - a cut or inserted pause of this size is "
            "within the readers' own spread at that boundary."
        )
    else:
        failure.cause = (
            "Suspected cause: regions were found elsewhere in the file but none over this "
            "span, which points at the region's own words rather than at the thresholds."
        )
    return failure


def _false_positive(row: Mapping[str, Any]) -> Failure:
    """Describe a spurious region and infer where it came from."""
    record, case, diagnostics = row["record"], row["case"], row["siblings"]
    file_id = str(record["file_id"])
    truth = _truth_in(diagnostics, file_id)
    overlapping = sorted(
        (t for t in truth if _overlap(record, t) > 0.0), key=lambda t: -_overlap(record, t)
    )
    nearest = min(
        (
            (abs(float(record["t_start"]) - float(t["t_end"])), t)
            for t in truth
            if float(t["t_end"]) <= float(record["t_start"])
        ),
        default=(float("inf"), {}),
    )
    failure = Failure(
        kind="false_positive",
        title=dict(KINDS)["false_positive"],
        found=True,
        file_id=file_id,
        fold=str(row["fold"]),
        split=str(row["split"]),
        label=str(record["label"]),
        evidence={
            "span_s": [record["t_start"], record["t_end"]],
            "score_peak_abs_z": record.get("score"),
            "file_origin": case.get("origin"),
            "file_has_truth": bool(case.get("n_truth")),
            "overlapping_truth": overlapping[:3],
            "gap_to_previous_truth_s": (
                None if nearest[0] == float("inf") else round(nearest[0], 3)
            ),
            "tau_in": row["thresholds"].get("tau_in"),
            "merge_gap_s": row.get("merge_gap_s"),
        },
    )
    score = record.get("score")
    failure.what = (
        f"`{record['label']}` reported at {record['t_start']:.3f}-{record['t_end']:.3f} s "
        f"in a file whose truth has no such event there"
        + (f"; region score |z|={float(score):.2f}" if score is not None else "")
        + "."
    )
    if overlapping and str(overlapping[0]["label"]) == str(record["label"]):
        failure.cause = (
            "Suspected cause: a second region of the same label over a flaw that another "
            "region already matched. One truth event produced two regions; the matcher is "
            "one-to-one, so it takes the closer one and counts this one as a false alarm. "
            "The region fragmented instead of merging - regions separated by less than "
            f"{row.get('merge_gap_s')} s are joined, and this pair is further apart than "
            "that."
        )
    elif overlapping:
        failure.cause = (
            f"Suspected cause: a real `{overlapping[0]['label']}` flaw sits under this span. "
            "One perturbation moves several features at once - a word read fast is also "
            "quieter and flatter - so a second label fires on the same evidence. This is "
            "the co-occurrence the taxonomy's label policy has to arbitrate."
        )
    elif not case.get("n_truth"):
        failure.cause = (
            "Suspected cause: a false alarm on a file with no flaw at all. At this word the "
            "envelope's spread is narrower than the variation a skilled reader actually "
            "shows, so the sigma floor did not protect the comparison - this is the quantity "
            "the FP/min row of spec 14.3 budgets."
        )
    elif failure.evidence["gap_to_previous_truth_s"] is not None and (
        float(failure.evidence["gap_to_previous_truth_s"]) < 0.5
    ):
        failure.cause = (
            "Suspected cause: spill from the flaw just before it. The hysteresis exit "
            "threshold keeps a region open while the score stays above tau_out, and the "
            "deviation decays over the following words rather than stopping at the edit."
        )
    else:
        failure.cause = (
            "Suspected cause: a genuine prosodic deviation in a part of the recording the "
            "labels say nothing about. The labels cover the injected edit only, so a reader "
            "who also did something unusual elsewhere is scored as a false alarm."
        )
    return failure


def _boundary(row: Mapping[str, Any]) -> Failure:
    """Describe the worst-placed matched region and infer why the edge moved."""
    record, case = row["record"], row["case"]
    onset = float(record["onset_error_ms"])
    offset = float(record["offset_error_ms"])
    failure = Failure(
        kind="boundary",
        title=dict(KINDS)["boundary"],
        found=True,
        file_id=str(record["file_id"]),
        fold=str(row["fold"]),
        split=str(row["split"]),
        label=str(record["label"]),
        evidence={
            "truth_span_s": [record["ref_start"], record["ref_end"]],
            "region_span_s": [record["pred_start"], record["pred_end"]],
            "onset_error_ms": onset,
            "offset_error_ms": offset,
            "iou": record.get("iou"),
            "level": case.get("level"),
            "refined": row["variant"].get("refine"),
            "penalty": row["thresholds"].get("penalty"),
        },
    )
    failure.what = (
        f"`{record['label']}` was found, but its edges are off by {onset:+.0f} ms at the "
        f"onset and {offset:+.0f} ms at the offset (IoU {record.get('iou')}): truth "
        f"{record['ref_start']:.3f}-{record['ref_end']:.3f} s, region "
        f"{record['pred_start']:.3f}-{record['pred_end']:.3f} s."
    )
    window_ms = float(row.get("refinement_window_s") or 0.0) * 1000.0
    worst_ms = max(abs(onset), abs(offset))
    # Refinement searches +/- window_s and discards a larger shift, so an error
    # bigger than that window cannot be its doing; blaming it there would be a
    # story the numbers do not support.
    refinement_could_explain = bool(row["variant"].get("refine")) and worst_ms <= window_ms * 1.1
    failure.evidence["refinement_window_ms"] = round(window_ms, 1)
    failure.evidence["refinement_could_explain"] = refinement_could_explain
    too_wide = onset < -50.0 and offset > 50.0
    too_narrow = onset > 50.0 and offset < -50.0
    # Order matters: a region wrong at *both* ends in the same direction is an
    # extent problem whatever refinement did, so the shape is read first and
    # refinement is only blamed for a one-sided error inside its own window.
    if too_wide:
        failure.cause = (
            "Suspected cause: the region is wider than the flaw on both sides. Regions are "
            "built on whole units, so a word only partly inside the edit is still pulled in, "
            f"and the {row.get('merge_gap_s')} s merge rule can join it to the next "
            "deviating unit."
        )
    elif too_narrow:
        failure.cause = (
            "Suspected cause: the region is narrower than the flaw at both ends - the "
            "deviation at the edit's outer words stayed below the exit threshold, so "
            "hysteresis closed the region early."
        )
    elif refinement_could_explain:
        failure.cause = (
            "Suspected cause: boundary refinement moved this edge. Ruptures searches "
            f"+/-{window_ms:.0f} ms around the initial boundary and the error is inside that "
            "window; where the strongest change in the signal is not the edit's edge - a "
            "syllable onset nearby, say - it moves the edge to the wrong one. The "
            "`no_refinement` ablation is the direct test of this."
        )
    else:
        failure.cause = (
            "Suspected cause: region extent rather than edge placement. The error is larger "
            f"than the +/-{window_ms:.0f} ms refinement window, so refinement cannot account "
            "for it: the region covers a different set of units from the flaw, because "
            "hysteresis held it open past the edit (or closed it early) and the merge rule "
            "then joined or split it on whole units."
        )
    return failure


def collect_failures(payload: Mapping[str, Any], *, config: Config | None = None) -> list[Failure]:
    """Pick the three failure cases of prompt 20 out of an aggregate payload.

    Args:
        payload: ``results/metrics.json``, which must carry ``per_fold`` with
            the ``diagnostics`` and ``cases`` blocks.
        config: Supplies the merge gap and the refinement window the cause
            heuristics quote; their spec 10.4-10.5 defaults are used without it.

    Returns:
        One :class:`Failure` per kind in :data:`KINDS`, in that order; a kind
        with no case carries ``found=False`` and the reason, because "the
        system produced no false alarm" is itself a result.
    """
    regions = (config.detection["regions"] if config is not None else {}) or {}
    refinement = (config.detection["refinement"] if config is not None else {}) or {}
    rows = _rows(
        payload,
        merge_gap_s=float(regions.get("merge_gap_s", 0.200)),
        refinement_window_s=float(refinement.get("window_s", 0.300)),
    )
    out: list[Failure] = []

    negatives = [r for r in rows if r["kind"] == "false_negative"]
    if negatives:
        worst = max(
            negatives,
            key=lambda r: (
                float(r["case"].get("severity_s") or 0.0),
                float(r["record"]["t_end"]) - float(r["record"]["t_start"]),
            ),
        )
        out.append(_false_negative(worst))
    else:
        out.append(
            Failure(
                kind="false_negative",
                title=dict(KINDS)["false_negative"],
                reason="no reference event was missed on any fold",
            )
        )

    positives = [r for r in rows if r["kind"] == "false_positive"]
    if positives:
        worst = max(
            positives,
            key=lambda r: (
                float(r["record"].get("score") or 0.0),
                float(r["record"]["t_end"]) - float(r["record"]["t_start"]),
            ),
        )
        out.append(_false_positive(worst))
    else:
        out.append(
            Failure(
                kind="false_positive",
                title=dict(KINDS)["false_positive"],
                reason="no region was reported that the truth does not contain",
            )
        )

    pairs = [r for r in rows if r["kind"] == "boundary"]
    if pairs:
        worst = max(
            pairs,
            key=lambda r: abs(float(r["record"]["onset_error_ms"]))
            + abs(float(r["record"]["offset_error_ms"])),
        )
        out.append(_boundary(worst))
    else:
        out.append(
            Failure(
                kind="boundary",
                title=dict(KINDS)["boundary"],
                reason="no region matched a reference event, so no boundary error exists",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def failures_markdown(payload: Mapping[str, Any], failures: Sequence[Failure]) -> str:
    """Render ``results/failures.md`` (prompt 20, spec 14.3's "Where it fails")."""
    corpus = str(payload.get("corpus", "?"))
    lines = [
        "# Where it fails",
        "",
        f"* corpus: **{corpus}**"
        + (
            "  - generated feature tables (eval/synthetic.py), so these are failures of the "
            "protocol on known perturbations, not of speech analysis on real readings"
            if corpus == "synthetic"
            else ""
        ),
        f"* folds: {', '.join(payload.get('folds') or ['-'])}",
        f"* variant: `{payload.get('variant', {}).get('name', '?')}`",
        "",
        "The three cases below are selected by rule, not by hand: the missed flaw with the "
        "highest severity, the false alarm with the highest region score, and the matched "
        "region with the largest total edge error. Each is the same event the aggregate "
        "numbers counted.",
        "",
        "**The 'what' is measured; the 'suspected cause' is an inference** from the evidence "
        "listed under it, and is written to be falsifiable rather than reassuring.",
        "",
    ]
    for failure in failures:
        lines.append(f"## {failure.title}")
        lines.append("")
        if not failure.found:
            lines += [f"None on this corpus: {failure.reason}.", ""]
            continue
        lines += [
            f"* **file:** `{failure.file_id}` (fold `{failure.fold}`, {failure.split} split)",
            f"* **label:** `{failure.label}`",
            f"* **what happened:** {failure.what}",
            f"* **{failure.cause}**",
            "",
            "Evidence:",
            "",
            "```json",
            json.dumps(failure.evidence, indent=2, sort_keys=True),
            "```",
            "",
        ]
    return "\n".join(lines) + "\n"


def write_failures(
    *,
    config: Config,
    metrics: Path | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """Read the metrics, pick the three cases and write ``failures.{md,json}``.

    Args:
        config: Loaded config.
        metrics: ``results/metrics.json``; the configured path by default.
        out: Markdown destination; ``results/failures.md`` by default.

    Returns:
        ``{"failures": [...], ...}``, as written to the JSON beside the markdown.

    Raises:
        FileNotFoundError: There is no metrics file to read.
    """
    metrics_path = metrics or config.path("results.metrics")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no metrics at {metrics_path}; run `make eval` first")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    failures = collect_failures(payload, config=config)

    destination = out or (config.path("results.dir") / "failures.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(failures_markdown(payload, failures), encoding="utf-8")
    record = {
        "schema": "rostrum.eval.failures.v1",
        "corpus": payload.get("corpus"),
        "metrics": str(metrics_path),
        "config_hash": config.hash,
        "code_version": code_version(config.root),
        "failures": [failure.to_dict() for failure in failures],
    }
    destination.with_suffix(".json").write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s and %s", destination, destination.with_suffix(".json"))
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints ``results/failures.md``."""
    parser = argparse.ArgumentParser(description="Failure analysis (prompt 20, spec 14.3)")
    parser.add_argument("--metrics", type=Path, default=None, help="results/metrics.json")
    parser.add_argument("--out", type=Path, default=None, help="results/failures.md")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    try:
        record = write_failures(config=config, metrics=args.metrics, out=args.out)
    except FileNotFoundError as error:
        print(f"cannot analyse failures: {error}", file=sys.stderr)
        return 2

    destination = args.out or (config.path("results.dir") / "failures.md")
    print(destination.read_text(encoding="utf-8"))
    print(f"wrote {destination} and {destination.with_suffix('.json')}")
    return 0 if record["failures"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
