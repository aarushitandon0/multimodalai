#!/usr/bin/env python
"""Summarise the injection realism gates into ``dataset/qc/summary.md``.

Reads what ``scripts/qc_dataset.py`` produced - the label JSONs plus
``dataset/qc/{utmos,realign,rejections}.csv`` - and writes four tables
(spec 5.1, 5.4):

1. **UTMOS mean ± sd per label x level**, with the clean sources as the L0
   column, so the naturalness cost of each severity step is visible.
2. **Monotonicity check.** UTMOS should fall as severity rises, for every
   label. Where it does not, the label is named and the offending step is
   printed: a non-monotonic column means the edit at that level is *less*
   audible as an artifact than the one below it, which is worth knowing before
   anyone reads a detection number off that label.
3. **Re-alignment boundary error**: mean, p95, max and the share of words
   within the 20 ms agreement target, overall and per label.
4. **Rejection counts** by gate, label and level, plus the rejection rate.

This script only *reports*; it never re-aligns, never loads UTMOS and never
edits a label, so it runs offline on a machine without the ``mfa`` conda env.

Usage::

    python scripts/qc_report.py                       # dataset/qc/summary.md
    python scripts/qc_report.py --out - | less        # to stdout
    python scripts/qc_report.py --labels-dir tmp/labels --qc-dir tmp/qc

Exit codes:
    0  summary written
    2  no label JSONs found - run scripts/build_injected.py and qc_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

# Allow `python scripts/qc_report.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.config import load_config
from rostrum.inject import qc

logger = logging.getLogger("qc_report")

LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4")
"""Injected severity levels; ``L0`` is the clean source, shown separately."""

NA = "—"
"""Placeholder for a cell with no file."""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Record:
    """One generated file as the report sees it.

    Attributes:
        file_id: The edited file.
        source_file: Its clean source.
        label: Flaw label, or ``"composite"``.
        level: Severity ``L1``-``L4``.
        utmos_clean: MOS of the clean source, or ``None`` if unscored.
        utmos_edited: MOS of the edited file, or ``None`` if unscored.
        passed: Whether the file passed both gates.
    """

    file_id: str
    source_file: str
    label: str
    level: str
    utmos_clean: float | None
    utmos_edited: float | None
    passed: bool

    @property
    def drop(self) -> float | None:
        """``utmos_clean - utmos_edited``, or ``None`` when either is missing."""
        if self.utmos_clean is None or self.utmos_edited is None:
            return None
        return self.utmos_clean - self.utmos_edited


def collect(labels_dir: Path | str, utmos_csv: Path | str) -> list[Record]:
    """Read every label JSON and pair it with the cached UTMOS scores.

    Scores come from the cache first and from the label's ``qc`` block second,
    so a report can be produced from either half of the pipeline.

    Args:
        labels_dir: ``dataset/labels``.
        utmos_csv: ``dataset/qc/utmos.csv``.

    Returns:
        One :class:`Record` per label JSON, sorted by ``file_id``. Clean files
        (``origin == "clean"``) are skipped: they are the baseline, not a
        generated file.
    """
    scores = qc.read_utmos_cache(utmos_csv)
    records: list[Record] = []
    for path in sorted(Path(labels_dir).glob("*.json")):
        try:
            label = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping unreadable label %s", path.name)
            continue
        if label.get("origin") == "clean":
            continue
        block = label.get("qc") if isinstance(label.get("qc"), dict) else {}
        file_id = str(label.get("file_id", path.stem))
        source = str(label.get("source_file", ""))
        records.append(
            Record(
                file_id=file_id,
                source_file=source,
                label=qc.label_of(label),
                level=qc.level_of(label),
                utmos_clean=_first_float(scores.get(source), block.get("utmos_clean")),
                utmos_edited=_first_float(scores.get(file_id), block.get("utmos_edited")),
                passed=qc.label_passed(label),
            )
        )
    return records


def _first_float(*values: object) -> float | None:
    """Return the first value that is a usable float."""
    for value in values:
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    return None


def read_realign(path: Path | str) -> list[dict[str, str]]:
    """Read ``dataset/qc/realign.csv``; ``[]`` when it does not exist."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_rejections(path: Path | str) -> list[dict[str, str]]:
    """Read ``dataset/qc/rejections.csv``; ``[]`` when it does not exist."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# UTMOS table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cell:
    """Mean ± sd of one label x level cell.

    Attributes:
        n: Files in the cell.
        mean: Mean UTMOS.
        sd: Sample standard deviation; ``0.0`` for a single file.
    """

    n: int
    mean: float
    sd: float

    def __str__(self) -> str:
        return f"{self.mean:.2f} ± {self.sd:.2f}"


def _cell(values: list[float]) -> Cell:
    """Aggregate scores into a :class:`Cell` (sample sd, 0 for one value)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return Cell(n=n, mean=mean, sd=0.0)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return Cell(n=n, mean=mean, sd=math.sqrt(var))


def utmos_stats(records: list[Record]) -> dict[tuple[str, str], Cell]:
    """Aggregate UTMOS per ``(label, level)``.

    The clean sources appear as ``(label, "L0")``: each label's L0 cell holds
    the sources its own files were made from, which is the baseline the drop is
    measured against.

    Args:
        records: Records from :func:`collect`.

    Returns:
        ``{(label, level): Cell}`` for every populated cell.
    """
    edited: dict[tuple[str, str], list[float]] = {}
    clean: dict[tuple[str, str], dict[str, float]] = {}
    for rec in records:
        if rec.utmos_edited is not None:
            edited.setdefault((rec.label, rec.level), []).append(rec.utmos_edited)
        if rec.utmos_clean is not None:
            # One clean file per source, however many edits were made from it.
            clean.setdefault((rec.label, "L0"), {})[rec.source_file] = rec.utmos_clean

    stats = {key: _cell(values) for key, values in edited.items() if values}
    stats.update({key: _cell(list(by_source.values())) for key, by_source in clean.items()})
    return stats


def _labels_in(records: list[Record]) -> list[str]:
    """Flaw labels present, alphabetically, with ``composite`` last."""
    names = {r.label for r in records if r.label}
    return sorted(names - {"composite"}) + (["composite"] if "composite" in names else [])


def format_utmos_table(records: list[Record]) -> str:
    """Markdown table of UTMOS mean ± sd per label x level, L0 included."""
    stats = utmos_stats(records)
    header = "| label | " + " | ".join(("L0", *LEVELS)) + " | n |"
    rule = "|" + "---|" * (len(LEVELS) + 3)
    lines = [header, rule]
    for label in _labels_in(records):
        cells = [str(stats[(label, lv)]) if (label, lv) in stats else NA for lv in ("L0", *LEVELS)]
        n = sum(stats[(label, lv)].n for lv in LEVELS if (label, lv) in stats)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {n} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Monotonicity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Violation:
    """A severity step at which mean UTMOS went *up* instead of down.

    Attributes:
        label: The flaw label.
        previous_level: The lower level of the pair.
        previous_mean: Its mean UTMOS.
        level: The higher level, whose mean should not exceed it.
        mean: Its mean UTMOS.
    """

    label: str
    previous_level: str
    previous_mean: float
    level: str
    mean: float

    def __str__(self) -> str:
        return (
            f"{self.label}: {self.previous_level} {self.previous_mean:.2f} -> "
            f"{self.level} {self.mean:.2f} (+{self.mean - self.previous_mean:.2f})"
        )


def monotonicity_violations(records: list[Record]) -> list[Violation]:
    """Severity steps where mean UTMOS rose, per label (acceptance criterion).

    Walks ``L0 -> L1 -> ... -> L4`` through the populated cells of each label
    and reports every step whose mean is higher than the step before it.
    Missing levels are skipped, not interpolated.

    Args:
        records: Records from :func:`collect`.

    Returns:
        Violations, label order then level order. Empty means UTMOS decreases
        monotonically on average with level for every label.
    """
    stats = utmos_stats(records)
    out: list[Violation] = []
    for label in _labels_in(records):
        chain = [(lv, stats[(label, lv)].mean) for lv in ("L0", *LEVELS) if (label, lv) in stats]
        for (prev_lv, prev_mean), (lv, mean) in pairwise(chain):
            if mean > prev_mean:
                out.append(Violation(label, prev_lv, prev_mean, lv, mean))
    return out


# --------------------------------------------------------------------------- #
# Realign distribution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Distribution:
    """Boundary-error distribution over a set of files.

    Attributes:
        n: Files counted.
        mean_err_ms: Mean of the per-file worst boundary error, or ``None``.
        p95_err_ms: 95th percentile of it, or ``None``.
        max_err_ms: Worst error seen, or ``None``.
        frac_within_20ms: Share of files whose worst error is within the spec
            5.1 agreement target, or ``None``.
    """

    n: int
    mean_err_ms: float | None
    p95_err_ms: float | None
    max_err_ms: float | None
    frac_within_20ms: float | None


def realign_distribution(rows: list[dict[str, str]], *, target_ms: float = 20.0) -> Distribution:
    """Summarise ``realign.csv`` rows.

    The statistic per file is its ``max_err_ms`` - the gated quantity - so
    ``frac_within_20ms`` is the share of *files* in which **every** word met
    the target, which is the honest reading of the spec 5.1 target.

    Args:
        rows: Rows from :func:`read_realign`.
        target_ms: Agreement target; 20 ms per spec 5.1.

    Returns:
        A :class:`Distribution`; all-``None`` fields when ``rows`` is empty.
    """
    values = sorted(float(r["max_err_ms"]) for r in rows if r.get("max_err_ms"))
    if not values:
        return Distribution(0, None, None, None, None)
    return Distribution(
        n=len(values),
        mean_err_ms=sum(values) / len(values),
        p95_err_ms=_percentile(values, 95.0),
        max_err_ms=values[-1],
        frac_within_20ms=sum(v <= target_ms for v in values) / len(values),
    )


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching ``numpy.percentile``."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def format_realign_table(rows: list[dict[str, str]]) -> str:
    """Markdown table of the boundary-error distribution, overall and per label."""
    header = "| files | n | mean ms | p95 ms | max ms | ≤ 20 ms |"
    lines = [header, "|---|---|---|---|---|---|"]

    def row(name: str, subset: list[dict[str, str]]) -> str:
        dist = realign_distribution(subset)
        if dist.n == 0:
            return f"| {name} | 0 | {NA} | {NA} | {NA} | {NA} |"
        return (
            f"| {name} | {dist.n} | {dist.mean_err_ms:.1f} | {dist.p95_err_ms:.1f} | "
            f"{dist.max_err_ms:.1f} | {100.0 * dist.frac_within_20ms:.0f}% |"
        )

    lines.append(row("**all**", rows))
    for label in sorted({r.get("label", "") for r in rows} - {""}):
        lines.append(row(label, [r for r in rows if r.get("label") == label]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RejectionCounts:
    """Rejections grouped three ways.

    Attributes:
        total: Rows in the rejection log.
        by_gate: Counts keyed by ``realign`` / ``utmos`` / ``other``.
        by_label: Counts per flaw label.
        by_level: Counts per severity level.
    """

    total: int
    by_gate: Counter[str]
    by_label: Counter[str]
    by_level: Counter[str]


def rejection_counts(rows: list[dict[str, str]]) -> RejectionCounts:
    """Group rejection rows by gate, label and level.

    A row can name both gates, in which case it counts once under each: the
    question the table answers is "how often did this gate fire", not "how
    were the files partitioned".
    """
    by_gate: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    for row in rows:
        reason = row.get("reason", "").lower()
        gates = [g for g in ("realign", "utmos") if g in reason]
        by_gate.update(gates or ["other"])
        by_label[row.get("label", "") or "?"] += 1
        by_level[row.get("level", "") or "?"] += 1
    return RejectionCounts(len(rows), by_gate, by_label, by_level)


def rejection_rate(records: list[Record], rejections: list[dict[str, str]]) -> float | None:
    """Rejected files as a share of every generated file.

    The denominator is the label count: a file that was generated and then
    rejected still has a label on disk, so ``len(records)`` is the number of
    files the engine produced.
    """
    if not records:
        return None
    return len(rejections) / len(records)


def format_rejection_tables(records: list[Record], rows: list[dict[str, str]]) -> str:
    """Markdown for the rejection counts and the rejection rate."""
    counts = rejection_counts(rows)
    rate = rejection_rate(records, rows)
    rate_text = NA if rate is None else f"{100.0 * rate:.1f}%"
    lines = [
        f"Generated files: **{len(records)}** · rejected: **{counts.total}** · "
        f"rejection rate: **{rate_text}**",
        "",
        "| gate | rejections |",
        "|---|---|",
    ]
    for gate in ("realign", "utmos", "other"):
        if counts.by_gate.get(gate):
            lines.append(f"| {gate} | {counts.by_gate[gate]} |")
    if not counts.total:
        lines.append(f"| {NA} | 0 |")

    lines += ["", "| label | level | rejections |", "|---|---|---|"]
    pairs = Counter((r.get("label", "?"), r.get("level", "?")) for r in rows)
    for (label, level), n in sorted(pairs.items()):
        lines.append(f"| {label} | {level} | {n} |")
    if not pairs:
        lines.append(f"| {NA} | {NA} | 0 |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# summary.md
# --------------------------------------------------------------------------- #


def build_summary(
    *,
    labels_dir: Path | str,
    utmos_csv: Path | str,
    realign_csv: Path | str,
    rejections_csv: Path | str,
) -> str:
    """Assemble ``dataset/qc/summary.md``.

    Args:
        labels_dir: ``dataset/labels``.
        utmos_csv: ``dataset/qc/utmos.csv``.
        realign_csv: ``dataset/qc/realign.csv``.
        rejections_csv: ``dataset/qc/rejections.csv``.

    Returns:
        The markdown document.
    """
    records = collect(labels_dir, utmos_csv)
    realign_rows = read_realign(realign_csv)
    rejection_rows = read_rejections(rejections_csv)
    violations = monotonicity_violations(records)
    scored = sum(1 for r in records if r.utmos_edited is not None)

    parts = [
        "# Injection QC summary",
        "",
        "Realism gates of the injected half of the dataset (spec 5.1, 5.4): a file is "
        "rejected when any word boundary moves more than 50 ms under re-alignment, or "
        "when its UTMOS drop exceeds the budget for its severity level. Generated by "
        "`scripts/qc_report.py`; UTMOS model "
        f"`{qc.UTMOS_HUB_REPO}` / `{qc.UTMOS_MODEL}`.",
        "",
        "## UTMOS by label and level",
        "",
        f"Predicted naturalness MOS, mean ± sd. L0 is the clean source. "
        f"{scored} of {len(records)} generated files scored.",
        "",
        format_utmos_table(records),
        "",
        "### Monotonicity with severity",
        "",
    ]
    if not records:
        parts.append("No generated files, so nothing to check.")
    elif violations:
        parts.append(
            f"UTMOS is **NOT monotonic** in severity for {len(violations)} step(s). "
            "An edit that scores *better* than a milder one at the same label means the "
            "severity ladder is not monotone in audible artifact, and detection numbers "
            "for that label should be read with that in mind:"
        )
        parts.append("")
        parts += [f"- {v}" for v in violations]
    else:
        parts.append("UTMOS decreases monotonically with severity for every label. ✅")

    parts += [
        "",
        "## Re-alignment boundary error",
        "",
        "Per file, the worst word-boundary displacement between the expected "
        "(time-map transformed) alignment and MFA's re-alignment of the edited WAV. "
        "The 20 ms column is the share of files in which *every* word met the spec 5.1 "
        "agreement target; 50 ms is the spec 5.4 rejection threshold.",
        "",
        format_realign_table(realign_rows),
        "",
        "## Rejections",
        "",
        format_rejection_tables(records, rejection_rows),
        "",
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels-dir", type=Path, default=None, help="dataset/labels")
    p.add_argument("--qc-dir", type=Path, default=None, help="dataset/qc")
    p.add_argument("--out", type=Path, default=None, help="summary.md, or - for stdout")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write ``dataset/qc/summary.md`` and print the rejection rate."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    cfg = load_config(apply_seeds=False)
    labels_dir = args.labels_dir or cfg.path("data.labels")
    qc_dir = args.qc_dir or cfg.path("data.qc_dir")
    out = args.out if args.out is not None else qc_dir / "summary.md"

    if not any(Path(labels_dir).glob("*.json")):
        print(
            f"no label JSONs in {labels_dir}: run scripts/build_injected.py "
            "then scripts/qc_dataset.py first",
            file=sys.stderr,
        )
        return 2

    text = build_summary(
        labels_dir=labels_dir,
        utmos_csv=qc_dir / "utmos.csv",
        realign_csv=qc_dir / "realign.csv",
        rejections_csv=qc_dir / "rejections.csv",
    )

    if str(out) == "-":
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("wrote %s", out)

    records = collect(labels_dir, qc_dir / "utmos.csv")
    rejections = read_rejections(qc_dir / "rejections.csv")
    rate = rejection_rate(records, rejections)
    print(
        f"QC rejection rate: {0.0 if rate is None else 100.0 * rate:.1f}% "
        f"({len(rejections)} of {len(records)} generated files)"
    )
    for violation in monotonicity_violations(records):
        print(f"WARNING: UTMOS not monotonic - {violation}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
