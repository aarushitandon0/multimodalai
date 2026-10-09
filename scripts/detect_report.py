"""Acceptance run for the envelope and the detector (spec 10.2-10.5).

Prints the two things the build prompts ask for:

* ``n`` per text, once the envelopes are built for a fold (spec 10.2);
* per-label precision and recall on the **training** fold, after region
  formation and boundary refinement (spec 10.3-10.5).

Two sources are supported, and the script says which one it used:

``--source dataset``
    The real corpus. Needs cached features, which need alignments, which need
    Montreal Forced Aligner. When the alignments or the feature cache are
    missing the script says so and exits non-zero rather than inventing
    numbers.

``--source synthetic`` (the default while the dataset is being built)
    :mod:`eval.synthetic` - feature tables with a known perturbation over a
    known span. This exercises every line of spec 10.2-10.5 deterministically,
    and it is honest about what it does *not* test: it says nothing about
    whether the feature extractors recover a flaw a human put into audio. Any
    figure taken from this run must be labelled synthetic in the report.

Usage::

    python -m scripts.detect_report                      # synthetic, all folds
    python -m scripts.detect_report --write-tuning       # also fit thresholds
    python -m scripts.detect_report --source dataset     # the real corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rostrum.config import load_config, set_seeds
from rostrum.detect.refine import RefinementSettings, refine_regions
from rostrum.detect.regions import RegionThresholds, build_regions, label_counts
from rostrum.detect.tune import (
    Event,
    TuningCase,
    event_f1,
    events_from_regions,
    grid_search,
    write_tuning,
)
from rostrum.detect.zscore import envelope_z, trigger_signals
from rostrum.envelope.build import Reference, build_fold_envelopes, summarise_counts

logger = logging.getLogger("scripts.detect_report")

__all__ = ["main", "run_synthetic"]


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def _print_envelope_counts(rows: Sequence[Mapping[str, Any]]) -> None:
    """Print the ``n`` per text table (spec 10.2 acceptance)."""
    print("\nEnvelopes built (n = contributing skilled readers)")
    print(f"  {'text_id':<16}{'fold':<22}{'n':>4}  {'same ch':>8}  units")
    for row in rows:
        units = ", ".join(
            f"{block['n_units']} {unit}s x {block['n_features']}f"
            for unit, block in row["units"].items()
        )
        print(
            f"  {row['text_id']:<16}{row['fold']:<22}{row['n']:>4}  "
            f"{row['same_channel']!s:>8}  {units}"
        )


def _print_per_label(metrics: Mapping[str, Any], title: str) -> None:
    """Print the per-label precision/recall table (spec 10.3-10.5 acceptance)."""
    print(f"\n{title}")
    print(f"  {'label':<20}{'TP':>4}{'FP':>4}{'FN':>4}{'P':>8}{'R':>8}{'F1':>8}")
    for label, counts in metrics["per_label"].items():
        print(
            f"  {label:<20}{counts['tp']:>4}{counts['fp']:>4}{counts['fn']:>4}"
            f"{counts['precision']:>8.3f}{counts['recall']:>8.3f}{counts['f1']:>8.3f}"
        )
    micro = metrics["micro"]
    print(
        f"  {'MICRO':<20}{micro['tp']:>4}{micro['fp']:>4}{micro['fn']:>4}"
        f"{micro['precision']:>8.3f}{micro['recall']:>8.3f}{micro['f1']:>8.3f}"
        f"   macro-F1 {metrics['macro_f1']:.3f}"
    )


# --------------------------------------------------------------------------- #
# Synthetic run
# --------------------------------------------------------------------------- #


def run_synthetic(
    *,
    config: Any,
    n_readers: int = 6,
    n_words: int = 24,
    refine: bool = True,
    tune: bool = False,
    extend_on: str | None = None,
) -> dict[str, Any]:
    """Build envelopes and detect on the synthetic corpus for every fold.

    Returns:
        A payload with the envelope counts, the per-label metrics on the
        training fold, and the tuning result when ``tune`` is set.
    """
    from eval.synthetic import build_corpus, truth_events

    references, participants = build_corpus(n_readers=n_readers, n_words=n_words)
    texts = sorted(references)
    payload: dict[str, Any] = {"source": "synthetic", "folds": {}}

    for held_out in texts:
        fold = f"loto_{held_out}"
        train_texts = [t for t in texts if t != held_out]
        print(f"\n{'=' * 72}\nfold {fold}  (train: {', '.join(train_texts)})\n{'=' * 72}")

        refs = [
            Reference(
                file_id=r.file_id,
                channel_id=r.channel_id,
                tables=r.tables,
            )
            for text in texts
            for r in references[text]
        ]
        envelopes = build_fold_envelopes(
            {text: [r.file_id for r in references[text]] for text in texts},
            refs,
            fold=fold,
            train_texts=train_texts,
            config=config,
            frames={r.file_id: r.frames for text in texts for r in references[text]},
            word_spans={r.file_id: r.word_spans for text in texts for r in references[text]},
        )
        counts = summarise_counts(envelopes)
        _print_envelope_counts(counts)

        thresholds = RegionThresholds.from_config(config, extend_on=extend_on)
        settings = RefinementSettings.from_config(config) if refine else None

        cases: list[TuningCase] = []
        for text in texts:
            envelope = envelopes[text]
            for recording in participants[text]:
                z_by_unit = envelope_z(
                    recording.tables,
                    envelope,
                    config=config,
                    # The synthetic corpus is one channel throughout, which is
                    # what lets `mumbling` be evaluated at all.
                    channel_matched=True,
                    file_id=recording.file_id,
                )
                signals, skipped = trigger_signals(z_by_unit, config=config, same_channel=True)
                cases.append(
                    TuningCase(
                        file_id=recording.file_id,
                        text_id=text,
                        signals=signals,
                        z_by_unit=z_by_unit,
                        truth=truth_events(recording),
                        frames=recording.frames,
                    )
                )
                if skipped:
                    logger.info(
                        "%s: %s", recording.file_id, [f"{s.label}:{s.reason}" for s in skipped]
                    )

        train_cases = [c for c in cases if c.text_id in set(train_texts)]
        test_cases = [c for c in cases if c.text_id == held_out]

        fold_payload: dict[str, Any] = {
            "train_texts": train_texts,
            "envelopes": counts,
            "extend_on": thresholds.extend_on,
        }

        if tune:
            tuned = grid_search(
                train_cases,
                fold=fold,
                train_texts=train_texts,
                config=config,
                refine=refine,
                base_thresholds=thresholds,
            )
            thresholds = tuned.thresholds(base=thresholds)
            if settings is not None:
                settings = RefinementSettings.from_config(config, penalty=tuned.penalty)
            print(
                f"\nTuned on training texts only: tau_in={tuned.tau_in} "
                f"tau_out={tuned.tau_out} penalty={tuned.penalty} "
                f"-> training event-F1 {tuned.event_f1:.3f}"
            )
            fold_payload["tuning"] = tuned.to_dict()

        for name, subset in (("TRAINING fold", train_cases), ("HELD-OUT text", test_cases)):
            truth: list[Event] = []
            predicted: list[Event] = []
            region_labels: dict[str, int] = {}
            for case in subset:
                regions = build_regions(
                    case.signals, case.z_by_unit, config=config, thresholds=thresholds
                )
                if settings is not None:
                    regions = refine_regions(regions, case.frames, settings=settings)
                predicted += events_from_regions(regions, file_id=case.file_id)
                truth += list(case.truth)
                for key, value in label_counts(regions).items():
                    region_labels[key] = region_labels.get(key, 0) + value
            metrics = event_f1(truth, predicted)
            _print_per_label(metrics, f"{name}: per-label event scores, 200 ms onset collar")
            fold_payload["train" if name.startswith("TRAINING") else "test"] = {
                "metrics": metrics,
                "regions_by_label": region_labels,
            }

        payload["folds"][fold] = fold_payload

    return payload


# --------------------------------------------------------------------------- #
# Dataset run
# --------------------------------------------------------------------------- #


def run_dataset(*, config: Any) -> int:
    """Report what the real corpus still needs, or run it when it is ready.

    The chain is alignments -> cached features -> envelopes -> detection. Rather
    than half-run it, this reports the first missing link and exits non-zero.
    """
    alignments = config.path("data.alignments")
    cache = config.path("cache.features")
    labels = config.path("data.labels")

    missing: list[str] = []
    if not alignments.is_dir() or not any(alignments.glob("*")):
        missing.append(f"alignments ({alignments}) - run `make data` (needs the mfa conda env)")
    if not cache.is_dir() or not any(cache.glob("*")):
        missing.append(f"feature cache ({cache}) - run `make features`")
    if not labels.is_dir() or not any(labels.glob("*.json")):
        missing.append(f"label JSONs ({labels}) - produced by the injection engine")

    if missing:
        print("Cannot run the dataset detector yet. Missing:")
        for item in missing:
            print(f"  - {item}")
        print(
            "\nUse `--source synthetic` to exercise spec 10.2-10.5 on generated "
            "feature tables in the meantime; those numbers are not dataset results."
        )
        return 1

    print(
        "Alignments, features and labels are all present. The dataset path is not "
        "wired up in this script yet: it belongs in eval/run_loto.py, which reads "
        "the manifest and the feature cache."
    )
    return 2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        choices=("synthetic", "dataset"),
        default="synthetic",
        help="where the feature tables come from (default: synthetic)",
    )
    parser.add_argument("--readers", type=int, default=6, help="synthetic readers per text")
    parser.add_argument("--words", type=int, default=24, help="synthetic words per text")
    parser.add_argument("--no-refine", action="store_true", help="skip boundary refinement")
    parser.add_argument(
        "--tune", action="store_true", help="grid-search the thresholds on each training fold"
    )
    parser.add_argument(
        "--extend-on",
        choices=("smoothed", "raw"),
        default=None,
        help=(
            "which score the hysteresis stay condition reads; default from "
            "detection.regions.extend_on. 'smoothed' is the spec 10.4 literal "
            "reading, 'raw' avoids the one-unit dilation at a region edge"
        ),
    )
    parser.add_argument(
        "--write-tuning",
        action="store_true",
        help="fit thresholds and write results/tuning.json (implies --tune)",
    )
    parser.add_argument("--out", type=Path, default=None, help="also write the payload as JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    set_seeds(config.seed)

    if args.source == "dataset":
        return run_dataset(config=config)

    tune = bool(args.tune or args.write_tuning)
    payload = run_synthetic(
        config=config,
        n_readers=args.readers,
        n_words=args.words,
        refine=not args.no_refine,
        tune=tune,
        extend_on=args.extend_on,
    )

    if args.write_tuning:
        from rostrum.detect.tune import TuningResult

        results = [
            TuningResult.from_dict(fold["tuning"])
            for fold in payload["folds"].values()
            if "tuning" in fold
        ]
        path = write_tuning(
            results, config.path("results.tuning"), config=config, source="synthetic"
        )
        print(f"\nwrote {len(results)} fold(s) of tuning to {path}")
        print(
            "NOTE: these thresholds were fitted on the SYNTHETIC corpus. Re-run on "
            "the dataset before any reported metric depends on them."
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    print(
        "\nSOURCE: synthetic feature tables (eval/synthetic.py). These numbers "
        "validate spec 10.2-10.5 region formation, not the feature extractors.\n"
        "The dataset acceptance run (`--source dataset`) needs MFA alignments."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
