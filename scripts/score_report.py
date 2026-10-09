"""Acceptance run for the rubric (spec 11).

Prints what the build prompt asks for, per LOTO fold:

* the calibrated :math:`k_c` per category, fitted on the fold's **training**
  texts from single-flaw L2 files only;
* the six category scores for one **clean** file of the held-out text and for its
  **L1-L4 rushing** versions, which must decrease down the ladder;
* the raw descriptors beside the skilled readers' range;
* the leave-one-reference-out confidence band on the L3 file, so the band's size
  is visible next to the score it qualifies.

The corpus is the graded synthetic one of :mod:`eval.cases`: generated feature
tables, so this exercises spec 11 end to end and is not a dataset result.

Usage::

    python -m scripts.score_report
    python -m scripts.score_report --out results/rubric_demo.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.cases import LEVELS, build_fold_cases

from rostrum.config import load_config, set_seeds
from rostrum.detect.regions import build_regions
from rostrum.detect.zscore import envelope_z, trigger_signals
from rostrum.envelope.build import leave_one_out
from rostrum.score.rubric import (
    build_descriptors,
    calibrate_k,
    category_order,
    confidence_band,
    rubric_score,
)
from rostrum.types import RubricScore

logger = logging.getLogger("scripts.score_report")

__all__ = ["main", "run"]


def _print_k(k_by_category: Mapping[str, float], order: Sequence[str]) -> None:
    """The calibrated k_c table."""
    print("\ncalibrated k_c (training texts, single-flaw L2 files -> median S_c = 6.0)")
    print("  " + "  ".join(f"{c}={k_by_category.get(c, float('nan')):.3f}" for c in order))


def _print_scores(rows: Sequence[tuple[str, RubricScore]], order: Sequence[str]) -> None:
    """One row per file, one column per category, overall last."""
    header = f"  {'file':<22}" + "".join(f"{c[:9]:>10}" for c in order) + f"{'overall':>10}"
    print("\ncategory scores (0-10, higher is better)")
    print(header)
    for name, score in rows:
        cells = ""
        for category in order:
            block = score.get(category)  # type: ignore[arg-type]
            if block is None:
                cells += f"{'-':>10}"
            elif block.suppressed:
                cells += f"{'(supp)':>10}"
            else:
                cells += f"{block.score:>10.2f}"
        print(f"  {name:<22}{cells}{score.overall:>10.2f}")


def _print_descriptors(block: Mapping[str, Any]) -> None:
    """The four raw descriptors with the readers' range."""
    print("\nraw descriptors (measurements, never scored)")
    for name, entry in block.items():
        value = entry.get("value")
        shown = "n/a" if value is None else f"{float(value):.2f}"
        band = ""
        if "reference_low" in entry:
            band = (
                f"   readers {entry['reference_low']:.2f}-{entry['reference_high']:.2f}"
                f" (median {entry['reference_median']:.2f}, n={entry['n_references']})"
            )
        extra = ""
        if name == "pause_rate_per_min":
            boundary = entry.get("boundary")
            mid = entry.get("mid_clause")
            if boundary is not None and mid is not None:
                extra = f"   boundary {float(boundary):.2f} / mid-clause {float(mid):.2f}"
        print(f"  {name:<28}{shown:>8}{extra}{band}")


def run(
    *,
    config: Any,
    texts: Sequence[str] = ("syn_a", "syn_b", "syn_c"),
    n_readers: int = 6,
    n_words: int = 24,
    refine: bool = True,
    demo_label: str = "rushing",
) -> dict[str, Any]:
    """Calibrate and score every fold; print the ladder for one held-out text."""
    order = category_order(config)
    payload: dict[str, Any] = {"source": "synthetic", "folds": {}}

    for held_out in texts:
        bundle = build_fold_cases(
            config=config,
            held_out=held_out,
            texts=texts,
            n_readers=n_readers,
            n_words=n_words,
            refine=refine,
        )
        print(f"\n{'=' * 78}\nfold {bundle.fold}  (train: {', '.join(bundle.train_texts)})")
        print("=" * 78)

        k_by_category = calibrate_k(
            [c.as_calibration_case(config=config) for c in bundle.train()],
            train_texts=bundle.train_texts,
            config=config,
        )
        _print_k(k_by_category, order)

        reference_tables = bundle.reference_tables.get(held_out, [])
        wanted = {"clean": "L0", **{f"{demo_label}_{level}": level for level in LEVELS}}
        rows: list[tuple[str, RubricScore]] = []
        fold_payload: dict[str, Any] = {
            "train_texts": list(bundle.train_texts),
            "held_out": held_out,
            "k_c": dict(k_by_category),
            "files": {},
        }

        for case in bundle.test():
            suffix = case.file_id.split("_", 3)[-1]
            if suffix not in wanted:
                continue
            descriptor_block = build_descriptors(case.tables, reference_tables=reference_tables)
            score = rubric_score(
                case.regions,
                case.speech_duration,
                k_by_category=k_by_category,
                config=config,
                descriptor_block=descriptor_block,
            )
            rows.append((suffix, score))
            fold_payload["files"][suffix] = {
                "level": case.level,
                "n_regions": len(case.regions),
                "score": score.to_dict(),
            }

        rows.sort(key=lambda row: ("" if row[0] == "clean" else row[0]))
        _print_scores(rows, order)

        # The confidence band on the L3 file: each envelope member dropped once
        # and the whole comparison rerun, which is the only honest way to show
        # how much of the score is the references' own disagreement.
        target = next(
            (c for c in bundle.test() if c.file_id.endswith(f"{demo_label}_L3")),
            None,
        )
        if target is not None:
            base = rubric_score(
                target.regions,
                target.speech_duration,
                k_by_category=k_by_category,
                config=config,
            )
            variants: list[RubricScore] = []
            envelope = bundle.envelopes[held_out]
            for dropped in envelope.reference_ids:
                reduced = leave_one_out(envelope, dropped)
                z_by_unit = envelope_z(
                    target.tables,
                    reduced,
                    config=config,
                    channel_matched=True,
                    file_id=target.file_id,
                )
                signals, _ = trigger_signals(z_by_unit, config=config, same_channel=True)
                variants.append(
                    rubric_score(
                        build_regions(
                            signals,
                            z_by_unit,
                            config=config,
                            thresholds=bundle.thresholds,
                        ),
                        target.speech_duration,
                        k_by_category=k_by_category,
                        config=config,
                    )
                )
            banded = confidence_band(base, variants)
            pace = banded.get("pace")
            print(
                f"\nleave-one-reference-out band on {target.file_id} "
                f"({len(variants)} recomputations)"
            )
            if pace is not None:
                print(f"  pace    {pace.score:.2f} +/- {pace.band:.2f}")
            print(f"  overall {banded.overall:.2f} +/- {banded.overall_band:.2f}")
            fold_payload["band"] = {
                "file_id": target.file_id,
                "n_recomputations": len(variants),
                "score": banded.to_dict(),
            }

        demo = next((c for c in bundle.test() if c.file_id.endswith(f"{demo_label}_L3")), None)
        if demo is not None:
            _print_descriptors(build_descriptors(demo.tables, reference_tables=reference_tables))

        payload["folds"][bundle.fold] = fold_payload

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--readers", type=int, default=6, help="synthetic readers per text")
    parser.add_argument("--words", type=int, default=24, help="synthetic words per text")
    parser.add_argument("--no-refine", action="store_true", help="skip boundary refinement")
    parser.add_argument(
        "--label", default="rushing", help="label whose L1-L4 ladder is printed (default rushing)"
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the payload as JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    set_seeds(config.seed)

    payload = run(
        config=config,
        n_readers=args.readers,
        n_words=args.words,
        refine=not args.no_refine,
        demo_label=args.label,
    )
    print(
        "\nNOTE: these numbers come from generated feature tables (eval.cases). "
        "They exercise spec 11 end to end; they are not dataset results."
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
