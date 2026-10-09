"""Pure helpers for packaging the dataset (spec 7.4).

Three jobs, kept out of ``scripts/package_dataset.py`` so they can be tested
without a dataset on disk:

* turning a label JSON into its ``MANIFEST.csv`` row;
* rendering the datasheet's generated tables (composition, label x level);
* choosing the ``dataset_sample/`` subset and reporting what it covers.

Everything is deterministic: rows are sorted by ``file_id`` before any choice is
made, so the same manifest always yields the same sample (invariant 13).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rostrum.data.manifest import ManifestRow

__all__ = [
    "ALL_LABELS",
    "AUTOBLOCK_END",
    "AUTOBLOCK_START",
    "INJECTABLE_LABELS",
    "LEVELS",
    "SampleCoverage",
    "composition_table",
    "label_table",
    "manifest_row_from_label",
    "render_autoblocks",
    "sample_coverage",
    "select_sample",
]

logger = logging.getLogger(__name__)

INJECTABLE_LABELS: tuple[str, ...] = (
    "rushing",
    "dragging",
    "missing_pause",
    "hesitation_pause",
    "monotone",
    "misplaced_stress",
    "low_energy",
)
"""The seven synthesisable labels, in dataset order.

Duplicated from ``rostrum.inject.engine`` on purpose: packaging must not import
the injection engine, which pulls in numpy and parselmouth.
"""

ALL_LABELS: tuple[str, ...] = (*INJECTABLE_LABELS, "mumbling")
"""All eight flaw labels; ``mumbling`` is human-recorded only (spec 5.2)."""

LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4")
"""Severity levels that produce a file. ``L0`` is the clean reading itself."""

ORIGIN_DIRS: tuple[tuple[str, str, str], ...] = (
    ("clean", "audio/clean/", "Skilled readings; build the per-text envelope"),
    ("injected", "audio/injected/", "Programmatic single-flaw and composite files, L1-L4"),
    ("human", "audio/human_flawed/", "Team recordings with deliberate flaws - test only"),
)
"""``(origin, directory, role)`` in datasheet order."""


# --------------------------------------------------------------------------- #
# Label JSON -> manifest row
# --------------------------------------------------------------------------- #


def manifest_row_from_label(label: Mapping[str, Any], duration_s: float) -> ManifestRow:
    """Build the ``MANIFEST.csv`` row described by one label JSON (spec 7.1).

    A file with several flaws is recorded as ``composite``; the flaws themselves
    stay in the label JSON and in ``events.tsv``.

    Args:
        label: Parsed ``labels/<file_id>.json``.
        duration_s: Duration of the processed WAV, seconds.

    Returns:
        A row with ``split_fold`` left empty for the split builder to fill.
    """
    flaws = list(label.get("flaws") or ())
    if not flaws:
        flaw_label, level = "", ""
    elif len(flaws) == 1:
        flaw_label, level = str(flaws[0]["label"]), str(flaws[0]["level"])
    else:
        flaw_label, level = "composite", str(flaws[0]["level"])
    return ManifestRow(
        file_id=str(label["file_id"]),
        text_id=str(label.get("text_id") or ""),
        tier=str(label.get("tier") or ""),
        origin=str(label.get("origin") or ""),
        label=flaw_label,
        level=level,
        duration_s=round(float(duration_s), 3),
    )


# --------------------------------------------------------------------------- #
# Datasheet tables
# --------------------------------------------------------------------------- #


def composition_table(rows: Iterable[ManifestRow]) -> str:
    """Render the composition table: files, duration and role per origin."""
    items = list(rows)
    if not items:
        return "_The dataset holds no files yet; run `make data` first._"

    lines = [
        "| Split | Files | Duration (min) | Speakers | Role |",
        "|---|---|---|---|---|",
    ]
    for origin, directory, role in ORIGIN_DIRS:
        group = [r for r in items if r.origin == origin]
        lines.append(
            f"| `{directory}` | {len(group)} | {_minutes(group)} | "
            f"{_n_speakers(group)} | {role} |"
        )
    lines.append(f"| **Total** | {len(items)} | {_minutes(items)} | {_n_speakers(items)} | |")
    return "\n".join(lines)


def label_table(rows: Iterable[ManifestRow]) -> str:
    """Render the label x severity grid, with human takes in their own column."""
    items = list(rows)
    header = [
        "| Label | " + " | ".join((*LEVELS, "human")) + " |",
        "|---" * (len(LEVELS) + 2) + "|",
    ]
    body = []
    for label in (*ALL_LABELS, "composite"):
        counts = [
            sum(1 for r in items if r.origin == "injected" and r.label == label and r.level == lvl)
            for lvl in LEVELS
        ]
        human = sum(1 for r in items if r.origin == "human" and r.label == label)
        if label == "composite" and not any(counts) and not human:
            continue
        body.append(f"| {label} | " + " | ".join(str(c) for c in (*counts, human)) + " |")
    return "\n".join((*header, *body))


def _minutes(rows: Sequence[ManifestRow]) -> str:
    total = 0.0
    for r in rows:
        try:
            total += float(r.duration_s)
        except (TypeError, ValueError):
            continue
    return f"{total / 60.0:.1f}"


def _n_speakers(rows: Sequence[ManifestRow]) -> int:
    """Distinct source recordings, which stand in for distinct speakers.

    An injected file inherits its speaker from the clean file whose ``file_id``
    prefixes its own (spec 7.1 naming), so the prefix is the speaker key.
    """
    return len(
        {r.file_id.split("_")[0] + "_" + r.file_id.split("_")[1] for r in rows if "_" in r.file_id}
    )


# --------------------------------------------------------------------------- #
# dataset_sample/ selection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleCoverage:
    """What ``dataset_sample/`` covers, and what it cannot.

    Attributes:
        n_files: Files selected.
        n_clean: Clean readings selected.
        n_human: Human flawed takes selected.
        texts: Texts represented, sorted.
        pairs: ``(label, level)`` pairs present in the sample, sorted.
        missing_pairs: Pairs that exist in the dataset but missed the cap.
        absent_pairs: Pairs the dataset never built at all.
        complete: Nothing missing: every label x level pair is in the sample
            and at least two human takes are included.
    """

    n_files: int
    n_clean: int
    n_human: int
    texts: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    missing_pairs: tuple[tuple[str, str], ...]
    absent_pairs: tuple[tuple[str, str], ...]
    complete: bool


def select_sample(
    rows: Iterable[ManifestRow],
    *,
    max_files: int = 20,
    n_human: int = 2,
    min_clean_per_text: int = 2,
    sizes: Mapping[str, int] | None = None,
    budget_bytes: int | None = None,
) -> list[str]:
    """Choose the ``dataset_sample/`` file ids (spec 7.4).

    Priority, so that a cap never costs label coverage:

    1. one injected file per ``(label, level)`` pair that exists;
    2. ``n_human`` human flawed takes, spread over labels then texts;
    3. ``min_clean_per_text`` clean references for each text represented, taken
       round-robin so no single text soaks up the budget;
    4. any remaining budget on further clean references.

    Args:
        rows: Manifest rows, any order.
        max_files: Hard cap on the sample size.
        n_human: Human takes to include, if the dataset has them.
        min_clean_per_text: Clean references per represented text - two is the
            minimum for an envelope with a leave-one-reference-out band.
        sizes: File id to size in bytes, used only with ``budget_bytes``. A file
            absent from the mapping is treated as free.
        budget_bytes: Total size the sample must fit in. Clean references are
            dropped from the end of the priority order until it does; flaw
            coverage and the human takes are never dropped, so an impossible
            budget yields an over-budget sample rather than a useless one, and
            the caller reports it.

    Returns:
        Sorted file ids, at most ``max_files`` of them.
    """
    items = sorted((r for r in rows), key=lambda r: r.file_id)
    chosen: list[str] = []

    def take(file_id: str) -> None:
        if file_id not in chosen:
            chosen.append(file_id)

    protected: set[str] = set()

    for _label, _level, file_id in _pair_representatives(items):
        take(file_id)
        protected.add(file_id)

    humans = [r for r in items if r.origin == "human"]
    for r in sorted(humans, key=lambda r: (r.label, r.text_id, r.file_id))[:n_human]:
        take(r.file_id)
        protected.add(r.file_id)

    flawed_texts = {r.text_id for r in items if r.origin in ("injected", "human")}
    texts = sorted(flawed_texts) or sorted({r.text_id for r in items})
    clean_by_text = {
        t: [r.file_id for r in items if r.origin == "clean" and r.text_id == t] for t in texts
    }
    for rank in range(min_clean_per_text):
        for t in texts:
            pool = clean_by_text[t]
            if rank < len(pool):
                take(pool[rank])

    remaining = [r.file_id for r in items if r.origin == "clean" and r.file_id not in chosen]
    for file_id in remaining:
        if len(chosen) >= max_files:
            break
        take(file_id)

    kept = chosen[:max_files]
    if budget_bytes is not None:
        kept = _fit_budget(kept, protected, sizes or {}, budget_bytes)
    return sorted(kept)


def _fit_budget(
    ordered: Sequence[str],
    protected: set[str],
    sizes: Mapping[str, int],
    budget_bytes: int,
) -> list[str]:
    """Drop unprotected files from the end of ``ordered`` until it fits."""
    kept = [f for f in ordered if f in protected]
    total = sum(sizes.get(f, 0) for f in kept)
    for file_id in ordered:
        if file_id in protected:
            continue
        size = sizes.get(file_id, 0)
        if total + size > budget_bytes:
            continue
        kept.append(file_id)
        total += size
    if total > budget_bytes:
        logger.warning(
            "sample cannot fit %d byte(s): flaw coverage alone needs %d", budget_bytes, total
        )
    return kept


def sample_coverage(rows: Iterable[ManifestRow], chosen: Sequence[str]) -> SampleCoverage:
    """Describe what ``chosen`` covers of ``rows`` (spec 7.4 acceptance)."""
    items = list(rows)
    picked = set(chosen)
    in_sample = [r for r in items if r.file_id in picked]

    wanted = {(lab, lvl) for lab in INJECTABLE_LABELS for lvl in LEVELS}
    built = {(lab, lvl) for lab, lvl, _ in _pair_representatives(items)}
    sampled = {
        (r.label, r.level) for r in in_sample if r.origin == "injected" and r.label and r.level
    }

    n_human = sum(1 for r in in_sample if r.origin == "human")
    missing = tuple(sorted(built - sampled))
    absent = tuple(sorted(wanted - built))
    return SampleCoverage(
        n_files=len(in_sample),
        n_clean=sum(1 for r in in_sample if r.origin == "clean"),
        n_human=n_human,
        texts=tuple(sorted({r.text_id for r in in_sample})),
        pairs=tuple(sorted(sampled)),
        missing_pairs=missing,
        absent_pairs=absent,
        complete=not missing and not absent and n_human >= 2,
    )


def _pair_representatives(items: Sequence[ManifestRow]) -> list[tuple[str, str, str]]:
    """One ``(label, level, file_id)`` per injected pair, lowest file id wins."""
    best: dict[tuple[str, str], str] = {}
    for r in sorted(items, key=lambda r: r.file_id):
        if r.origin != "injected" or not r.label or not r.level:
            continue
        best.setdefault((r.label, r.level), r.file_id)
    return [(lab, lvl, fid) for (lab, lvl), fid in sorted(best.items())]


# --------------------------------------------------------------------------- #
# Generated blocks inside the datasheet
# --------------------------------------------------------------------------- #

_MARKER = re.compile(r"<!-- (BEGIN|END):([a-z0-9_]+) \(generated[^>]*-->")


def AUTOBLOCK_START(name: str) -> str:
    """Opening marker of the generated block ``name``."""
    return f"<!-- BEGIN:{name} (generated by scripts/package_dataset.py; do not edit) -->"


def AUTOBLOCK_END(name: str) -> str:
    """Closing marker of the generated block ``name``."""
    return f"<!-- END:{name} (generated by scripts/package_dataset.py; do not edit) -->"


def render_autoblocks(document: str, blocks: Mapping[str, str]) -> str:
    """Replace each generated block in ``document`` with its new content.

    Args:
        document: The datasheet, with ``BEGIN``/``END`` marker pairs.
        blocks: Block name to replacement markdown.

    Returns:
        The document with every named block's body replaced. Markers are kept,
        so rendering twice gives the same result.

    Raises:
        ValueError: ``document`` holds no generated markers at all.
        KeyError: A name in ``blocks`` has no marker pair in ``document``.
    """
    present = {m.group(2) for m in _MARKER.finditer(document)}
    if not present:
        raise ValueError(
            "document has no generated markers; expected "
            + ", ".join(sorted(blocks))
            + " wrapped in BEGIN/END comments"
        )
    unknown = sorted(set(blocks) - present)
    if unknown:
        raise KeyError(f"no marker pair in the document for block(s): {', '.join(unknown)}")

    out = document
    for name, content in blocks.items():
        start, end = AUTOBLOCK_START(name), AUTOBLOCK_END(name)
        head, _, rest = out.partition(start)
        _, _, tail = rest.partition(end)
        out = f"{head}{start}\n{content.strip()}\n{end}{tail}"
    return out
