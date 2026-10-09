#!/usr/bin/env python
"""Prepare human flawed takes for hand annotation in Praat (spec 6.2).

Force-aligns every WAV under ``dataset/audio/human_flawed/`` with MFA, then
writes each TextGrid twice:

* ``dataset/alignments/<file_id>.TextGrid`` - the plain MFA output, tiers
  ``words`` and ``phones``, which the rest of the pipeline reads;
* ``dataset/annotation/todo/<annotator>/<file_id>.TextGrid`` - the same two
  tiers plus an **empty** interval tier named ``flaws``, which is what the
  annotator opens in Praat.

The annotator splits the ``flaws`` tier and types ``<label>:<severity 1-3>``
into every interval that holds a flaw (see ``dataset/ANNOTATION_GUIDE.md``),
then copies the finished file into ``dataset/annotation/done/<annotator>/`` for
``scripts/annotate_import.py``.

Spec 6.2 wants at least 30% of the human files labelled independently by two
people. Pass ``--annotators`` and this script does the bookkeeping: the first
name listed is the primary and receives every file, each further name receives
a deterministic subset (ordered by SHA-256 of the file id, so the assignment
does not depend on directory order or on the master seed). Who got what is
recorded in ``dataset/annotation/assignments.csv``.

Usage::

    python scripts/annotate_prepare.py
    python scripts/annotate_prepare.py --annotators ann1 ann2
    python scripts/annotate_prepare.py --only spkA_s1_rush --allow-oov
    python scripts/annotate_prepare.py --dry-run

Exit codes:
    0  every requested file aligned and a todo TextGrid was written
    1  alignment failed, a transcript is missing, or OOV words were found
    2  nothing to prepare, or MFA/conda is not installed
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

# Allow `python scripts/annotate_prepare.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_clean import MissingTranscript, align_batch, check_models, transcript_for
from rostrum.align.mfa import (
    MFAError,
    MFANotAvailableError,
    mfa_version,
    parse_textgrid,
    read_textgrid,
    write_textgrid,
)
from rostrum.config import Config, load_config
from rostrum.data.annotation import AnnotationSettings, blank_flaw_tier, settings_from_config
from rostrum.data.manifest import read_manifest
from rostrum.io.hashing import sha256_bytes, sha256_file

logger = logging.getLogger("annotate_prepare")

ASSIGNMENT_FIELDS: tuple[str, ...] = ("annotator", "file_id", "role")
"""Columns of ``dataset/annotation/assignments.csv``."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def human_items(
    cfg: Config, only: set[str] | None
) -> tuple[list[tuple[Path, str, str]], dict[str, str], list[str]]:
    """Collect ``(wav, transcript, file_id)`` triples for the human corpus.

    Mirrors ``align_clean.clean_items`` but over ``audio/human_flawed/`` and
    without requiring ``origin == "human"`` in the manifest - a take that has
    been recorded but not yet classified still needs aligning. The ``text_id``
    does have to be known: without it there is no envelope to compare against
    and no transcript to align with.

    Returns:
        ``(items, text_ids, problems)``.
    """
    human_dir = cfg.path("data.audio_human_flawed")
    texts_dir = cfg.path("data.texts")
    by_file = {row.file_id: row for row in read_manifest(cfg.path("data.manifest"))}

    items: list[tuple[Path, str, str]] = []
    text_ids: dict[str, str] = {}
    problems: list[str] = []

    if not human_dir.is_dir():
        return items, text_ids, [f"{human_dir} does not exist; no human takes to prepare"]

    for wav in sorted(human_dir.glob("*.wav")):
        file_id = wav.stem
        if only is not None and file_id not in only:
            continue
        row = by_file.get(file_id)
        if row is None:
            problems.append(
                f"{file_id}: no MANIFEST.csv row, so its text_id is unknown. "
                f"Add the row before aligning - the text_id is what the "
                f"envelope is keyed on."
            )
            continue
        try:
            transcript, source = transcript_for(file_id, row.text_id, texts_dir)
        except MissingTranscript as exc:
            problems.append(str(exc))
            continue
        logger.debug("%s: transcript from %s", file_id, source)
        items.append((wav, transcript, file_id))
        text_ids[file_id] = row.text_id

    return items, text_ids, problems


# --------------------------------------------------------------------------- #
# Annotator assignment (spec 6.2)
# --------------------------------------------------------------------------- #


def assign(
    file_ids: Sequence[str],
    annotators: Sequence[str],
    double_frac: float,
) -> dict[str, list[str]]:
    """Return ``{annotator: file_ids}`` honouring the double-annotation quota.

    The first annotator is the primary and gets every file. Each further
    annotator gets ``ceil(double_frac * n)`` files, chosen by sorting the ids
    on their SHA-256 - deterministic, independent of directory order, and
    independent of the master seed, so re-running this script never reshuffles
    an assignment an annotator has already started working through.

    Args:
        file_ids: Files to hand out.
        annotators: Names, primary first. A single name means no second pass.
        double_frac: Share of files that must be annotated twice (spec 6.2
            asks for at least 0.30).

    Returns:
        ``{annotator: sorted file ids}``.

    Raises:
        ValueError: ``annotators`` is empty, holds a duplicate, or
            ``double_frac`` is outside ``[0, 1]``.
    """
    if not annotators:
        raise ValueError("at least one annotator is needed")
    if len(set(annotators)) != len(annotators):
        raise ValueError(f"duplicate annotator in {list(annotators)}")
    if not 0.0 <= double_frac <= 1.0:
        raise ValueError(f"double_frac {double_frac} is not a fraction")

    ordered = sorted(file_ids)
    n_second = min(len(ordered), math.ceil(double_frac * len(ordered)))
    by_hash = sorted(ordered, key=lambda f: sha256_bytes(f.encode("utf-8")))
    second = sorted(by_hash[:n_second])

    out: dict[str, list[str]] = {annotators[0]: ordered}
    for name in annotators[1:]:
        out[name] = list(second)
    return out


def write_assignments(cfg: Config, assignment: Mapping[str, Sequence[str]], primary: str) -> Path:
    """Write ``dataset/annotation/assignments.csv``.

    Rows are sorted by annotator then file id so the file is byte-stable, and
    it is rewritten rather than appended to: it is a statement of the current
    assignment, not a log.
    """
    out = cfg.ensure_dir("data.annotation") / "assignments.csv"
    rows = [
        {"annotator": name, "file_id": file_id, "role": "primary" if name == primary else "second"}
        for name in sorted(assignment)
        for file_id in sorted(assignment[name])
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ASSIGNMENT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s (%d row(s))", out, len(rows))
    return out


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #


def write_todo(
    grid: Path,
    dest: Path,
    settings: AnnotationSettings,
) -> Path:
    """Copy an MFA TextGrid to ``dest`` with an empty ``flaws`` tier appended.

    The ``words`` and ``phones`` tiers are passed through unchanged, so the
    annotator sees exactly the forced alignment the labels will be grounded
    against, and the ``flaws`` tier is a single empty interval over the whole
    file - the blank tier Praat lets them split.

    Raises:
        ValueError: ``grid`` has no interval tiers, or already has a ``flaws``
            tier (which would mean overwriting somebody's work).
    """
    tiers = parse_textgrid(grid)
    existing = [t.name for t in tiers if t.name.strip().lower() == settings.tier_name.lower()]
    if existing:
        raise ValueError(
            f"{grid} already has a {settings.tier_name!r} tier; refusing to "
            f"replace it with an empty one"
        )
    xmin = min(t.xmin for t in tiers)
    xmax = max(t.xmax for t in tiers)
    return write_textgrid(dest, [*tiers, blank_flaw_tier(xmin, xmax, settings)])


def publish(
    cfg: Config,
    grids: Sequence[Path],
    staged: Mapping[str, object],
    text_ids: Mapping[str, str],
    assignment: Mapping[str, Sequence[str]],
    settings: AnnotationSettings,
    version: str,
    *,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    """Publish alignments and per-annotator todo TextGrids.

    Each grid is parsed with :func:`~rostrum.align.mfa.read_textgrid` before
    anything is written: that is where a word-count mismatch against the
    transcript surfaces, and a mismatched human take must not reach an
    annotator, because its word indices would not line up with the references.

    Returns:
        ``(published_file_ids, problems)``.
    """
    align_dir = cfg.ensure_dir("data.alignments")
    todo_dir = cfg.path("data.annotation_todo")
    human_dir = cfg.path("data.audio_human_flawed")

    published: list[str] = []
    problems: list[str] = []
    for grid in grids:
        file_id = grid.stem
        item = staged.get(file_id)
        if item is None:
            problems.append(f"{grid.name}: MFA wrote a TextGrid for an unknown file_id")
            continue
        wav = human_dir / f"{file_id}.wav"
        try:
            alignment = read_textgrid(
                grid,
                file_id=file_id,
                text=item.text,  # type: ignore[attr-defined]
                text_id=text_ids.get(file_id),
                aligner=f"mfa {version}",
                acoustic_model=str(cfg.paths["mfa"]["acoustic_model"]),
                dictionary=str(cfg.paths["mfa"]["dictionary"]),
                audio_sha256=sha256_file(wav) if wav.is_file() else "",
            )
        except ValueError as exc:
            problems.append(f"{file_id}: {exc}")
            continue

        targets = [name for name, files in assignment.items() if file_id in files]
        if not targets:
            problems.append(f"{file_id}: assigned to no annotator")
            continue

        if not dry_run:
            shutil.copyfile(grid, align_dir / f"{file_id}.TextGrid")
            for name in sorted(targets):
                try:
                    write_todo(grid, todo_dir / name / f"{file_id}.TextGrid", settings)
                except ValueError as exc:
                    problems.append(f"{file_id}: {exc}")
        published.append(file_id)
        logger.info(
            "%s: %d words, %.3f s -> %s",
            file_id,
            alignment.n_words,
            alignment.duration,
            ", ".join(sorted(targets)),
        )
    return published, problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="prepare only these file ids")
    p.add_argument(
        "--annotators",
        nargs="+",
        default=["annotator1"],
        metavar="NAME",
        help="annotator names, primary first (default: annotator1)",
    )
    p.add_argument(
        "--double-frac",
        type=float,
        default=None,
        metavar="F",
        help="share of files given to each non-primary annotator "
        "(default: injection.annotation.agreement.min_double_annotated_frac)",
    )
    p.add_argument(
        "--allow-oov",
        action="store_true",
        help="align even though the dictionary is missing words (they become <unk>)",
    )
    p.add_argument(
        "--keep-corpus", action="store_true", help="keep the staged corpus under .cache/mfa"
    )
    p.add_argument("--dry-run", action="store_true", help="align and report, but write nothing")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Prepare the human set for annotation. See the docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(apply_seeds=False)
    settings = settings_from_config(cfg)
    double_frac = (
        settings.min_double_annotated_frac if args.double_frac is None else args.double_frac
    )

    items, text_ids, problems = human_items(cfg, set(args.only) if args.only else None)
    if problems:
        print("\nCANNOT PREPARE these files:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("", file=sys.stderr)
    if not items:
        logger.error(
            "nothing to prepare: no usable (wav, transcript) pair under %s",
            cfg.path("data.audio_human_flawed"),
        )
        return 2

    try:
        assignment = assign([file_id for _, _, file_id in items], args.annotators, double_frac)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if len(args.annotators) < 2:
        logger.warning(
            "only one annotator: spec 6.2 wants >= %.0f%% of human files labelled "
            "independently by two people. Pass --annotators NAME1 NAME2.",
            settings.min_double_annotated_frac * 100,
        )

    try:
        version = mfa_version(env_name=str(cfg.paths["mfa"]["env_name"]))
        model_problems = check_models(cfg)
        if model_problems:
            for problem in model_problems:
                logger.error("%s", problem)
            return 2
        staged, grids = align_batch(
            cfg, items, allow_oov=args.allow_oov, keep_corpus=args.keep_corpus
        )
    except MFANotAvailableError as exc:
        logger.error("MFA is not available: %s", exc)
        return 2
    except MFAError as exc:
        logger.error("alignment failed: %s", exc)
        return 1

    published, publish_problems = publish(
        cfg,
        grids,
        staged,
        text_ids,
        assignment,
        settings,
        version,
        dry_run=args.dry_run,
    )
    if not args.dry_run and published:
        write_assignments(cfg, assignment, args.annotators[0])

    for problem in publish_problems:
        logger.error("%s", problem)

    print(f"\nprepared {len(published)} file(s) for annotation")
    for name in sorted(assignment):
        n = len([f for f in assignment[name] if f in published])
        role = "primary" if name == args.annotators[0] else "second"
        print(f"  {name:<16} {n:>3} file(s)  ({role})")
    if len(args.annotators) > 1:
        covered = len([f for f in assignment[args.annotators[1]] if f in published])
        share = covered / len(published) if published else 0.0
        print(
            f"\ndouble-annotated: {covered}/{len(published)} ({share:.0%}); "
            f"spec 6.2 target is {settings.min_double_annotated_frac:.0%}"
        )
    print(f"\nnext: annotators fill the {settings.tier_name!r} tier in Praat, following")
    print(f"      {cfg.path('data.annotation_guide')}")
    print(f"      then copy finished grids into {cfg.path('data.annotation_done')}/<annotator>/")
    if args.dry_run:
        print("\n(dry run: no files were written)")
    return 1 if (problems or publish_problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
