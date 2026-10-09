"""Acceptance run for the flaw-type classifier (spec 10.6).

Prints the two things the build prompt asks for:

* **macro-F1 on the held-out fold** of every LOTO fold, with the rule's own
  macro-F1 beside it so "the classifier types flaws better than the rule" is a
  claim backed by both numbers;
* a **confusion matrix PNG**, pooled over folds, written to ``docs/figures``.

Per fold it fits on the training texts only (:func:`rostrum.detect.classifier.
fit_fold` applies the filter itself), saves the model with ``joblib`` and the
config hash under ``.cache/models``, and evaluates on the held-out text.

The corpus is the graded synthetic one of :mod:`eval.cases`: generated feature
tables with known spans and known rungs. That exercises every line of spec 10.6
deterministically and says nothing about whether the feature extractors recover
a flaw a speaker put into audio - any figure taken from this run must be
labelled synthetic in the report. The dataset path belongs in ``eval/run_loto.py``
once ``make data`` can run.

Usage::

    python -m scripts.clf_report
    python -m scripts.clf_report --out results/classifier.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from eval.cases import build_fold_cases

from rostrum.config import load_config, set_seeds
from rostrum.detect.classifier import (
    ClassifierError,
    RegionFeatureSpec,
    build_examples,
    evaluate,
    fit_fold,
    model_path,
    save_model,
)

logger = logging.getLogger("scripts.clf_report")

__all__ = ["main", "run", "save_confusion_png"]


def _print_fold(fold: str, metrics: Mapping[str, Any], n_train: int) -> None:
    """Per-label table plus the headline macro-F1 for one fold."""
    print(f"\n{'label':<20}{'P':>8}{'R':>8}{'F1':>8}{'n':>6}")
    for label, row in metrics["per_label"].items():
        if row["support"] == 0 and row["f1"] == 0.0:
            continue
        print(
            f"{label:<20}{row['precision']:>8.3f}{row['recall']:>8.3f}"
            f"{row['f1']:>8.3f}{row['support']:>6}"
        )
    print(
        f"\nfold {fold}: macro-F1 {metrics['macro_f1']:.3f} "
        f"(rule {metrics['rule_macro_f1']:.3f})  accuracy {metrics['accuracy']:.3f}  "
        f"{metrics['n']} held-out region(s), {n_train} training example(s)"
    )


def _pool(matrices: Sequence[Mapping[str, Any]]) -> tuple[list[str], np.ndarray]:
    """Sum per-fold confusion matrices onto one shared label order."""
    labels = sorted({label for block in matrices for label in block["labels"]})
    pooled = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for block in matrices:
        index = {label: i for i, label in enumerate(block["labels"])}
        for truth in block["labels"]:
            for predicted in block["labels"]:
                pooled[labels.index(truth), labels.index(predicted)] += int(
                    block["matrix"][index[truth]][index[predicted]]
                )
    return labels, pooled


def save_confusion_png(
    labels: Sequence[str],
    matrix: np.ndarray,
    path: Path,
    *,
    title: str = "Flaw-type confusion (held-out folds, synthetic)",
) -> Path:
    """Render a confusion matrix as a PNG (spec 10.6 acceptance).

    Rows are the ground truth and columns the prediction, each cell annotated
    with its count. Row-normalised shading, because an unnormalised image is
    dominated by whichever label happens to be most frequent.
    """
    import matplotlib

    matplotlib.use("Agg")  # no display on CI
    import matplotlib.pyplot as plt

    counts = np.asarray(matrix, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    shaded = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    size = max(4.5, 0.75 * len(labels) + 2.0)
    figure, axes = plt.subplots(figsize=(size, size * 0.92), dpi=150)
    image = axes.imshow(shaded, cmap="Blues", vmin=0.0, vmax=1.0)
    axes.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axes.set_yticks(range(len(labels)), labels, fontsize=8)
    axes.set_xlabel("classifier label")
    axes.set_ylabel("ground truth")
    axes.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j] == 0:
                continue
            axes.text(
                j,
                i,
                f"{int(counts[i, j])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if shaded[i, j] > 0.5 else "black",
            )
    figure.colorbar(image, ax=axes, fraction=0.046, label="row fraction")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    logger.info("wrote %s", path)
    return path


def run(
    *,
    config: Any,
    texts: Sequence[str] = ("syn_a", "syn_b", "syn_c"),
    n_readers: int = 6,
    n_words: int = 24,
    refine: bool = True,
    figure: Path | None = None,
) -> dict[str, Any]:
    """Fit, save and evaluate one classifier per LOTO fold.

    Returns:
        A payload with each fold's metrics and the pooled confusion matrix.
    """
    payload: dict[str, Any] = {"source": "synthetic", "folds": {}}
    spec = RegionFeatureSpec.from_config(config)
    print(f"region feature vector: {spec.n_columns} columns over {len(spec.features)} features")

    confusions: list[Mapping[str, Any]] = []
    macro: list[float] = []
    rule_macro: list[float] = []

    for held_out in texts:
        bundle = build_fold_cases(
            config=config,
            held_out=held_out,
            texts=texts,
            n_readers=n_readers,
            n_words=n_words,
            refine=refine,
        )
        print(f"\n{'=' * 72}\nfold {bundle.fold}  (train: {', '.join(bundle.train_texts)})")
        print("=" * 72)

        cases = [c.as_classifier_case() for c in bundle.cases]
        try:
            classifier = fit_fold(
                cases,
                fold=bundle.fold,
                train_texts=bundle.train_texts,
                config=config,
                spec=spec,
            )
        except ClassifierError as error:
            print(f"fold {bundle.fold}: cannot fit - {error}")
            payload["folds"][bundle.fold] = {"error": str(error)}
            continue

        destination = save_model(classifier, model_path(bundle.fold, config))
        held_examples = build_examples([c.as_classifier_case() for c in bundle.test()], spec)
        metrics = evaluate(classifier, held_examples)
        _print_fold(bundle.fold, metrics, classifier.n_train)
        print(f"model: {destination}  (config {classifier.config_hash[:12]})")

        confusions.append(metrics["confusion"])
        macro.append(float(metrics["macro_f1"]))
        rule_macro.append(float(metrics["rule_macro_f1"]))
        payload["folds"][bundle.fold] = {
            "train_texts": list(bundle.train_texts),
            "held_out": bundle.held_out,
            "n_train": classifier.n_train,
            "classes": list(classifier.classes),
            "severity": classifier.severity.to_meta(),
            "metrics": metrics,
            "model": str(destination),
        }

    if macro:
        print(f"\n{'=' * 72}")
        print(
            f"held-out macro-F1 over {len(macro)} fold(s): "
            f"mean {np.mean(macro):.3f} +/- {np.std(macro, ddof=0):.3f}   "
            f"(rule {np.mean(rule_macro):.3f})"
        )
        payload["macro_f1_mean"] = float(np.mean(macro))
        payload["macro_f1_std"] = float(np.std(macro, ddof=0))
        payload["rule_macro_f1_mean"] = float(np.mean(rule_macro))

    if confusions:
        labels, pooled = _pool(confusions)
        destination = figure or (config.path("docs.figures") / "confusion_matrix.png")
        save_confusion_png(labels, pooled, destination)
        print(f"confusion matrix: {destination}")
        payload["confusion"] = {
            "labels": labels,
            "matrix": [[int(v) for v in row] for row in pooled],
            "figure": str(destination),
        }
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
        "--figure", type=Path, default=None, help="confusion matrix PNG (default docs/figures)"
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

    try:
        payload = run(
            config=config,
            n_readers=args.readers,
            n_words=args.words,
            refine=not args.no_refine,
            figure=args.figure,
        )
    except ImportError as error:  # pragma: no cover - environment, not logic
        print(
            f"scikit-learn could not be imported: {error}\n"
            "The classifier needs sklearn.ensemble; the rest of the pipeline does not.",
            file=sys.stderr,
        )
        return 3

    print(
        "\nNOTE: these numbers come from generated feature tables (eval.cases). "
        "They exercise spec 10.6 end to end; they are not dataset results."
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.out}")
    return 0 if payload.get("folds") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
