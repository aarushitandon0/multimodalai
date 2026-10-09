#!/usr/bin/env python
"""Import completed annotation TextGrids as dataset labels (spec 6.2, 7.1, 7.2).

Reads every finished grid under ``dataset/annotation/done/`` - flat, or under
``done/<annotator>/`` when a file was annotated by more than one person - reads
its ``flaws`` tier, validates each interval against the taxonomy, and writes:

* ``dataset/labels/<file_id>.json`` - the spec 7.1 label JSON, ``origin``
  ``"human"``, ``split`` ``"test"`` (``CLAUDE.md`` invariant 11);
* ``dataset/labels/events.tsv`` - rebuilt from every label JSON on disk, so the
  sed_eval event list covers injected and human files alike;
* ``dataset/alignments/<file_id>.TextGrid`` - words, phones and the imported
  ``flaws`` tier (spec 7.2);
* a ``MANIFEST.csv`` row per imported file.

When a file carries two annotations the **primary** annotator's is imported and
the other is left for ``scripts/annotator_agreement.py``; disagreements are
reported, never averaged. The primary is taken from
``dataset/annotation/assignments.csv`` if ``annotate_prepare.py`` wrote one,
and can be forced with ``--primary``.

Nothing is coerced: an interval whose text is not ``<label>:<severity 1-3>``
with a label from the taxonomy is reported and the file is skipped, because a
label quietly repaired into something plausible becomes wrong ground truth that
every metric downstream is then measured against.

Usage::

    python scripts/annotate_import.py
    python scripts/annotate_import.py --primary ann1
    python scripts/annotate_import.py --only spkA_s1_rush --dry-run
    python scripts/annotate_import.py --events-only   # rebuild events.tsv only

Exit codes:
    0  every finished grid imported
    1  at least one grid was rejected
    2  nothing to import
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

# Allow `python scripts/annotate_import.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.align.mfa import parse_textgrid, read_textgrid, write_textgrid
from rostrum.config import Config, load_config
from rostrum.data.annotation import (
    AnnotationError,
    AnnotationSettings,
    FlawAnnotation,
    build_label_json,
    flaw_tier,
    read_annotations,
    settings_from_config,
)
from rostrum.data.manifest import ManifestRow, append_manifest, read_manifest
from rostrum.inject import engine

logger = logging.getLogger("annotate_import")

LEVEL_ORDER: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4")
"""Severity ladder in order, for picking a file's worst level for MANIFEST.csv."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Finished:
    """One completed annotation TextGrid found under ``done/``.

    Attributes:
        file_id: The recording's id (the grid's stem).
        annotator: Whose annotation this is; ``""`` for a grid sitting directly
            in ``done/`` rather than in ``done/<annotator>/``.
        path: The TextGrid.
    """

    file_id: str
    annotator: str
    path: Path


def find_finished(done_dir: Path, only: set[str] | None = None) -> list[Finished]:
    """Return every completed TextGrid under ``done/``, sorted.

    Both layouts are accepted: ``done/<file_id>.TextGrid`` for a file one
    person annotated, and ``done/<annotator>/<file_id>.TextGrid`` for the
    double-annotated subset. Sorting is by ``(file_id, annotator)`` so the
    import order - and therefore the rebuilt ``events.tsv`` - does not depend
    on directory iteration order (``CLAUDE.md`` invariant 13).
    """
    if not done_dir.is_dir():
        return []
    found: list[Finished] = []
    for grid in sorted(done_dir.glob("*.TextGrid")):
        found.append(Finished(file_id=grid.stem, annotator="", path=grid))
    for sub in sorted(p for p in done_dir.iterdir() if p.is_dir()):
        for grid in sorted(sub.glob("*.TextGrid")):
            found.append(Finished(file_id=grid.stem, annotator=sub.name, path=grid))
    if only is not None:
        found = [f for f in found if f.file_id in only]
    return sorted(found, key=lambda f: (f.file_id, f.annotator))


def read_primary(cfg: Config) -> str:
    """Return the primary annotator from ``assignments.csv``, or ``""``.

    Written by ``annotate_prepare.py``; absent when the grids were prepared by
    hand, in which case the caller falls back to ``--primary`` or to the
    alphabetically first annotator.
    """
    path = cfg.path("data.annotation") / "assignments.csv"
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("role") == "primary":
                return str(row.get("annotator", ""))
    return ""


def choose(
    grids: Sequence[Finished],
    primary: str,
) -> tuple[Finished, list[str]]:
    """Pick which annotation of one file to import, and name the others.

    The primary annotator wins. Failing that, a grid sitting directly in
    ``done/`` wins (there is only one annotation of it), then the
    alphabetically first annotator - an arbitrary but stable choice, which is
    better than an order-dependent one.

    Returns:
        ``(chosen, other_annotators)``.
    """
    by_annotator = {g.annotator: g for g in grids}
    if primary and primary in by_annotator:
        chosen = by_annotator[primary]
    elif "" in by_annotator:
        chosen = by_annotator[""]
    else:
        chosen = by_annotator[min(by_annotator)]
    others = sorted(g.annotator for g in grids if g is not chosen)
    return chosen, others


# --------------------------------------------------------------------------- #
# Import of one file
# --------------------------------------------------------------------------- #


def worst_level(flaws: Sequence[Mapping[str, object]]) -> str:
    """Return the highest severity level among a file's flaws, or ``"L0"``."""
    levels = [str(f["level"]) for f in flaws]
    return max(levels, key=LEVEL_ORDER.index) if levels else "L0"


def manifest_row(
    label: Mapping[str, object],
    existing: ManifestRow | None,
) -> ManifestRow:
    """Build the ``MANIFEST.csv`` row of one imported human file.

    Follows the convention ``build_injected.py`` set: a file with more than one
    flaw has ``label`` ``"composite"`` and the labels themselves live in the
    label JSON and ``events.tsv``. ``split_fold`` is left as the manifest has
    it - human takes are test-only whatever fold the texts fall into, and that
    is recorded in the label JSON's ``split``.
    """
    flaws = list(label.get("flaws", []))  # type: ignore[arg-type]
    name = str(flaws[0]["label"]) if len(flaws) == 1 else ("composite" if flaws else "")
    return ManifestRow(
        file_id=str(label["file_id"]),
        text_id=str(label["text_id"] or ""),
        tier=str(label["tier"] or (existing.tier if existing else "bronze")),
        origin="human",
        label=name,
        level=worst_level(flaws),
        duration_s=f"{float(label['duration']):.3f}",  # type: ignore[arg-type]
        split_fold=existing.split_fold if existing else "",
    )


def write_alignment_grid(
    cfg: Config,
    file_id: str,
    source: Path,
    annotations: Sequence[FlawAnnotation],
    settings: AnnotationSettings,
) -> Path:
    """Write ``dataset/alignments/<file_id>.TextGrid`` with the flaws tier.

    The annotator's own grid is the source of truth for the tier contents, but
    it is re-emitted through :func:`~rostrum.align.mfa.write_textgrid` rather
    than copied, so the published file has canonical formatting and 3-decimal
    times whatever Praat wrote (invariants 2 and 13).

    Co-occurring labels cannot share one interval tier, so a file whose
    annotations overlap gets one ``flaws_<label>`` tier per label involved.
    """
    prefix = settings.tier_name.lower()
    tiers = [t for t in parse_textgrid(source) if not t.name.strip().lower().startswith(prefix)]
    if not tiers:
        raise ValueError(f"{source}: nothing but flaw tiers; the words tier is missing")
    xmin = min(t.xmin for t in tiers)
    xmax = max(t.xmax for t in tiers)

    ordered = sorted(annotations, key=lambda a: a.t_start)
    overlapping = any(b.t_start < a.t_end - 1e-9 for a, b in pairwise(ordered))
    if not overlapping:
        tiers.append(flaw_tier(ordered, xmin, xmax, settings))
    else:
        by_label: dict[str, list[FlawAnnotation]] = {}
        for ann in ordered:
            by_label.setdefault(ann.label, []).append(ann)
        for name in sorted(by_label):
            tier = flaw_tier(by_label[name], xmin, xmax, settings)
            tiers.append(replace(tier, name=f"{settings.tier_name}_{name}"))

    return write_textgrid(cfg.ensure_dir("data.alignments") / f"{file_id}.TextGrid", tiers)


def _relative(path: Path, root: Path) -> str:
    """Render ``path`` relative to the repo root, or absolute if it is outside it.

    A label records where its annotation came from; a path that cannot be made
    relative (a grid read from elsewhere on disk) is recorded as it is rather
    than failing the import over a cosmetic field.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def import_one(
    cfg: Config,
    chosen: Finished,
    others: Sequence[str],
    settings: AnnotationSettings,
    text_id: str | None,
    *,
    dry_run: bool,
) -> tuple[dict[str, object], list[FlawAnnotation]]:
    """Read one finished grid and write its label JSON and alignment grid.

    Returns:
        ``(label_json, annotations)``.

    Raises:
        AnnotationError: The grid has no ``flaws`` tier, or an interval the
            guide does not allow.
        ValueError: The grid's word tier does not parse as an alignment.
    """
    annotations, _ = read_annotations(chosen.path, settings, strict=True)
    alignment = read_textgrid(chosen.path, file_id=chosen.file_id, text_id=text_id)
    label = build_label_json(
        chosen.file_id,
        annotations,
        alignment,
        settings,
        annotator=chosen.annotator,
        annotators=[chosen.annotator, *others],
        source_textgrid=_relative(chosen.path, cfg.root),
        config_hash=cfg.hash,
    )
    if not dry_run:
        out = cfg.ensure_dir("data.labels") / f"{chosen.file_id}.json"
        out.write_text(
            json.dumps(label, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        write_alignment_grid(cfg, chosen.file_id, chosen.path, annotations, settings)
    return label, annotations


# --------------------------------------------------------------------------- #
# events.tsv
# --------------------------------------------------------------------------- #


def rebuild_events(cfg: Config) -> int:
    """Rewrite ``dataset/labels/events.tsv`` from every label JSON on disk.

    The same routine ``build_injected.py`` uses, so an injected and a human
    label produce the same row format and one file holds both.

    Returns:
        How many event rows were written.
    """
    labels_dir = cfg.path("data.labels")
    labels = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(labels_dir.glob("*.json"))
        if p.name != "events.tsv"
    ]
    text = engine.format_events_tsv(labels)
    out = cfg.path("data.events_tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    n_rows = sum(1 for line in text.splitlines() if line.strip())
    logger.info("events.tsv: %d event(s) from %d label file(s)", n_rows, len(labels))
    return n_rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="import only these file ids")
    p.add_argument(
        "--primary",
        default=None,
        metavar="NAME",
        help="annotator whose labels are imported for double-annotated files "
        "(default: the primary in annotation/assignments.csv)",
    )
    p.add_argument(
        "--events-only", action="store_true", help="only rebuild labels/events.tsv and exit"
    )
    p.add_argument("--dry-run", action="store_true", help="validate and report, but write nothing")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Import the finished annotations. See the docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(apply_seeds=False)
    settings = settings_from_config(cfg)

    if args.events_only:
        rebuild_events(cfg)
        return 0

    done_dir = cfg.path("data.annotation_done")
    finished = find_finished(done_dir, set(args.only) if args.only else None)
    if not finished:
        logger.error(
            "nothing to import: no .TextGrid under %s (or %s/<annotator>/). Run "
            "scripts/annotate_prepare.py, annotate the grids in Praat, then copy "
            "them here.",
            done_dir,
            done_dir,
        )
        return 2

    primary = args.primary if args.primary is not None else read_primary(cfg)
    manifest = {row.file_id: row for row in read_manifest(cfg.path("data.manifest"))}

    by_file: dict[str, list[Finished]] = {}
    for item in finished:
        by_file.setdefault(item.file_id, []).append(item)

    imported: list[str] = []
    problems: list[str] = []
    counts: Counter[str] = Counter()
    rows: list[ManifestRow] = []

    for file_id in sorted(by_file):
        chosen, others = choose(by_file[file_id], primary)
        row = manifest.get(file_id)
        if row is None:
            problems.append(
                f"{file_id}: no MANIFEST.csv row, so its text_id is unknown - "
                f"the label JSON would have nothing to key the envelope on"
            )
            continue
        try:
            label, annotations = import_one(
                cfg, chosen, others, settings, row.text_id, dry_run=args.dry_run
            )
        except (AnnotationError, ValueError) as exc:
            problems.append(f"{file_id}: {exc}")
            continue
        rows.append(manifest_row(label, row))
        imported.append(file_id)
        counts.update(a.label for a in annotations)
        logger.info(
            "%s: %d flaw(s) from %s%s",
            file_id,
            len(annotations),
            chosen.annotator or chosen.path.parent.name,
            f" (also annotated by {', '.join(others)})" if others else "",
        )

    if not args.dry_run and rows:
        append_manifest(cfg.path("data.manifest"), rows)
        rebuild_events(cfg)

    for problem in problems:
        logger.error("%s", problem)

    print(f"\nimported {len(imported)} file(s), {sum(counts.values())} flaw(s)")
    for name in sorted(counts):
        print(f"  {name:<20} {counts[name]:>4}")
    if problems:
        print(f"\nREJECTED {len(problems)} file(s); see the errors above", file=sys.stderr)
    if args.dry_run:
        print("\n(dry run: no files were written)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
