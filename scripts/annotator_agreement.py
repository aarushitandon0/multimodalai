#!/usr/bin/env python
"""Inter-annotator agreement over the double-annotated human files (spec 6.2).

Finds every file that two people annotated independently - a
``done/<annotator>/<file_id>.TextGrid`` under each of their names - and reports

* **event-F1** with a 200 ms onset collar and an offset collar of
  max(200 ms, 20% of the reference event's length), matching one-to-one and
  label-agnostically: did the two annotators find the same *moments*;
* **Cohen's kappa** on the labels of the matched events: did they call them the
  same *thing*;
* onset/offset MAE, mean temporal IoU, a label confusion matrix, and the
  double-annotated share against the 30% target of spec 6.2.

Collars come from ``injection.annotation.agreement``; nothing is hard-coded
here. Output goes to ``dataset/qc/agreement.md``, with the same numbers as JSON
beside it (``agreement.json``) for anything that needs to parse them.

With more than two annotators every pair that shares a file is reported, and
the pair with the most shared files leads the Markdown report.

Usage::

    python scripts/annotator_agreement.py
    python scripts/annotator_agreement.py --done-dir tests/fixtures/annotation/done
    python scripts/annotator_agreement.py --annotators ann1 ann2 --stdout

Exit codes:
    0  a report was written
    1  some grid could not be read, or coverage is below the spec 6.2 target
    2  no file was annotated by two people, so there is nothing to compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

# Allow `python scripts/annotator_agreement.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.config import Config, load_config
from rostrum.data.annotation import (
    AnnotationError,
    AnnotationSettings,
    FlawAnnotation,
    agreement,
    double_annotated,
    format_agreement_md,
    read_annotations,
    settings_from_config,
)

logger = logging.getLogger("annotator_agreement")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def collect(
    done_dir: Path,
    settings: AnnotationSettings,
    *,
    keep: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, list[FlawAnnotation]]], list[str]]:
    """Read every ``done/<annotator>/<file_id>.TextGrid`` under ``done_dir``.

    Grids sitting directly in ``done/`` are ignored: they carry no annotator
    name, so they cannot be attributed to one side of a comparison.

    Problems are collected rather than raised, so one unreadable grid does not
    hide the agreement of all the others - but it is reported, and it makes the
    script exit non-zero.

    Args:
        done_dir: ``dataset/annotation/done``.
        settings: Taxonomy and severity scale.
        keep: Only these annotators, if given.

    Returns:
        ``({annotator: {file_id: annotations}}, problems)``.
    """
    wanted = set(keep) if keep else None
    by_annotator: dict[str, dict[str, list[FlawAnnotation]]] = {}
    problems: list[str] = []
    if not done_dir.is_dir():
        return by_annotator, [f"{done_dir} does not exist"]

    for sub in sorted(p for p in done_dir.iterdir() if p.is_dir()):
        if wanted is not None and sub.name not in wanted:
            continue
        for grid in sorted(sub.glob("*.TextGrid")):
            try:
                annotations, _ = read_annotations(grid, settings, strict=True)
            except (AnnotationError, ValueError) as exc:
                problems.append(f"{sub.name}/{grid.name}: {exc}")
                continue
            by_annotator.setdefault(sub.name, {})[grid.stem] = annotations
    return by_annotator, problems


def count_human_files(cfg: Config) -> int | None:
    """Return how many human takes exist, for the coverage check.

    ``None`` when ``dataset/audio/human_flawed`` is absent - the coverage line
    is then left out rather than guessed at.
    """
    human_dir = cfg.path("data.audio_human_flawed")
    if not human_dir.is_dir():
        return None
    return len(list(human_dir.glob("*.wav"))) or None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--done-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="directory of done/<annotator>/*.TextGrid " "(default: paths.data.annotation_done)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="where to write the report (default: paths.data.qc_agreement)",
    )
    p.add_argument(
        "--annotators",
        nargs="+",
        default=None,
        metavar="NAME",
        help="restrict the comparison to these annotators",
    )
    p.add_argument("--stdout", action="store_true", help="also print the report")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Report inter-annotator agreement. See the docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(apply_seeds=False)
    settings = settings_from_config(cfg)
    done_dir = args.done_dir or cfg.path("data.annotation_done")

    by_annotator, problems = collect(done_dir, settings, keep=args.annotators)
    for problem in problems:
        logger.error("%s", problem)

    pairs = double_annotated(by_annotator)
    if not pairs:
        logger.error(
            "no file under %s was annotated by two people. Spec 6.2 wants >= %.0f%% "
            "of the human files double-annotated; prepare them with "
            "scripts/annotate_prepare.py --annotators NAME1 NAME2.",
            done_dir,
            settings.min_double_annotated_frac * 100,
        )
        return 2

    n_human = count_human_files(cfg)
    # The pair sharing the most files leads the report; ties break on the names
    # so the choice does not depend on dict order.
    ordered_pairs = sorted(pairs, key=lambda pair: (-len(pairs[pair]), pair))
    reports = []
    for a, b in ordered_pairs:
        by_file = {
            file_id: (by_annotator[a][file_id], by_annotator[b][file_id])
            for file_id in pairs[(a, b)]
        }
        reports.append(
            agreement(
                by_file,
                settings,
                annotator_a=a,
                annotator_b=b,
                n_human_files=n_human,
            )
        )

    out = args.out or cfg.path("data.qc_agreement")
    out.parent.mkdir(parents=True, exist_ok=True)
    markdown = "\n\n---\n\n".join(format_agreement_md(r, settings) for r in reports)
    out.write_text(markdown, encoding="utf-8", newline="\n")
    json_out = out.with_suffix(".json")
    json_out.write_text(
        json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("wrote %s and %s", out, json_out)

    if args.stdout:
        print()
        print(markdown)

    lead = reports[0]
    print(
        f"\n{lead.annotator_a} vs {lead.annotator_b} over {len(lead.file_ids)} file(s): "
        f"event-F1 {lead.f1:.3f} @ {round(lead.onset_collar_s * 1000)} ms, "
        f"kappa {lead.kappa:.3f}"
    )
    below = lead.coverage is not None and lead.coverage < settings.min_double_annotated_frac
    if below:
        logger.warning(
            "double-annotated coverage is %.0f%%, below the %.0f%% target of spec 6.2",
            (lead.coverage or 0.0) * 100,
            settings.min_double_annotated_frac * 100,
        )
    return 1 if (problems or below) else 0


if __name__ == "__main__":
    raise SystemExit(main())
