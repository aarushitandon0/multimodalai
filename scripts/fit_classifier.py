"""Fit the live dashboard's flaw-type classifier on the whole real corpus.

``eval.run_loto`` fits a classifier per leave-one-text-out fold and keeps it in
memory, because a fold's model must never see its held-out text. The dashboard
has no held-out text: it analyses a file the user just uploaded, so the model it
loads - ``.cache/models/clf_all.joblib``, the ``fold="all"`` model
:func:`rostrum.pipeline.load_library` looks for - is fitted on every text in the
corpus. That is the right thing here and the wrong thing in an evaluation, so
this script exists separately from the eval harness and writes a separate file.

Human-recorded takes stay out of the fit regardless: ``fit_fold`` applies the
``detection.classifier.train_on`` origin filter itself (CLAUDE.md invariant 11).

Usage::

    python -m scripts.fit_classifier                     # -> .cache/models/clf_all.joblib
    python -m scripts.fit_classifier --out results/classifier_all.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from eval.dataset_corpus import CorpusUnavailable
from eval.run_loto import _classifier_case, dataset_corpus
from rostrum.config import Config, load_config
from rostrum.detect.classifier import ClassifierError, fit_fold, model_path, save_model

logger = logging.getLogger("fit_classifier")

FOLD = "all"


def fit_all(config: Config, *, refine: bool = True) -> tuple[Path, dict[str, object]]:
    """Fit one classifier on every text and save it as the ``all`` fold.

    A LOTO bundle carries both its training and its held-out recordings in
    ``cases``, already detected, so one bundle is enough to reach the whole
    corpus; which text was nominally held out makes no difference once every
    text is named in ``train_texts``.

    Args:
        config: Loaded config.
        refine: Run ruptures boundary refinement during detection.

    Returns:
        ``(path_written, summary)``.

    Raises:
        CorpusUnavailable: The corpus is not built; the message names what is
            missing.
        ClassifierError: Too few labelled regions to fit.
    """
    corpus = dataset_corpus(config)
    texts = sorted(corpus.texts)
    if not texts:
        raise CorpusUnavailable("no texts in the corpus; run `make data` and `make features`")

    bundle = None
    errors: list[str] = []
    for held_out in texts:
        try:
            bundle = corpus.fold(config=config, held_out=held_out, refine=refine)
            break
        except CorpusUnavailable as error:  # a one-reading text has no envelope
            errors.append(f"{held_out}: {error}")
    if bundle is None:
        raise CorpusUnavailable("no text could form a fold: " + "; ".join(errors))

    train_texts = sorted({case.text_id for case in bundle.cases})
    cases = [_classifier_case(case, case.regions) for case in bundle.cases]
    logger.info(
        "fitting fold %s on %d recordings across %d texts", FOLD, len(cases), len(train_texts)
    )

    classifier = fit_fold(cases, fold=FOLD, train_texts=train_texts, config=config)
    destination = save_model(classifier, model_path(FOLD, config))

    summary = {
        "fold": FOLD,
        "config_hash": str(config.hash),
        "texts": train_texts,
        "n_cases_offered": len(cases),
        "classes": list(classifier.classes),
        "model_path": str(destination),
        "meta": classifier.to_meta(),
    }
    return destination, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make classifier-all``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-refine", action="store_true", help="skip boundary refinement")
    parser.add_argument("--out", type=Path, default=None, help="also write the summary as JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()

    try:
        destination, summary = fit_all(config, refine=not args.no_refine)
    except (CorpusUnavailable, ClassifierError) as error:
        logger.error("%s", error)
        return 2

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("summary written to %s", args.out)

    print(f"\nfold {FOLD} classifier -> {destination}")
    print(f"classes: {', '.join(summary['classes'])}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
