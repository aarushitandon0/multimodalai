"""Acceptance run for the explanation cards (spec 12).

Prints rendered cards for five different labels, each with its JSON form, so
the acceptance criterion of the phase - "print 5 rendered cards from different
labels" - can be read off the terminal.

The regions come from the graded synthetic corpus of :mod:`eval.cases`: real
detection (envelope, z-scores, hysteresis, ruptures refinement) over
*generated* feature tables. The numbers therefore exercise spec 12 end to end
and say nothing about dataset performance, which needs MFA.

The text context is real. :data:`DEMO_TEXT` is a 24-word passage whose
punctuation reproduces exactly the gap labels
:func:`eval.synthetic._gap_label` generated the features with - a boundary
every six words, a comma halfway - so the card's "repeated phrase (anaphora)"
and the detector's "gap 5 is a boundary" describe the same text.

Usage::

    python -m scripts.cards_demo
    python -m scripts.cards_demo --all --out results/cards_demo.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.cases import build_fold_cases

from rostrum.config import Config, load_config
from rostrum.detect.classifier import ClassifierError, fit_fold
from rostrum.explain.cards import Card, build_cards, render_markdown
from rostrum.text.rhetoric import TextAnnotation, annotate

logger = logging.getLogger("scripts.cards_demo")

__all__ = ["DEMO_LABELS", "DEMO_TEXT", "main", "run"]

DEMO_TEXT: str = (
    "We can not, and will not. "
    "We can not, we shall not. "
    "We can not, we must not. "
    "Here we stand, we hold fast."
)
"""24 words; boundary after words 5, 11, 17, comma after words 2, 8, 14, 20."""

DEMO_LABELS: tuple[str, ...] = (
    "rushing",
    "missing_pause",
    "hesitation_pause",
    "monotone",
    "low_energy",
)
"""The five labels printed by default; ``--all`` prints all eight."""


def _annotation(config: Config) -> TextAnnotation:
    """The demo text, annotated for context."""
    return annotate(DEMO_TEXT, text_id="demo_24w", config=config)


def _pick(cards: Sequence[Card], labels: Sequence[str]) -> list[Card]:
    """One card per label, the highest-confidence one, in ``labels`` order."""
    out: list[Card] = []
    for label in labels:
        matches = [c for c in cards if c.label == label]
        if not matches:
            logger.warning("no region was detected for %s in the demo corpus", label)
            continue
        out.append(max(matches, key=lambda c: (c.confidence, abs(c.evidence[0].z))))
    return out


def run(
    *,
    config: Config,
    held_out: str = "syn_a",
    labels: Sequence[str] = DEMO_LABELS,
    n_readers: int = 6,
) -> dict[str, Any]:
    """Detect one fold, explain every region and print the selected cards."""
    bundle = build_fold_cases(
        config=config,
        held_out=held_out,
        n_readers=n_readers,
        n_words=24,
        refine=True,
    )
    annotation = _annotation(config)

    # The severity rung and the classifier probability on a card come from the
    # fold's classifier, fitted on the TRAINING texts only (spec 10.6). Without
    # it every card would read L0, which would say more about this script than
    # about the regions.
    classifier = None
    try:
        classifier = fit_fold(
            [case.as_classifier_case() for case in bundle.cases],
            fold=bundle.fold,
            train_texts=bundle.train_texts,
            config=config,
        )
    except (ClassifierError, ImportError) as exc:
        # scikit-learn is unavailable in some environments (an application
        # control policy blocks its compiled extensions on the author's
        # machine), and a fold can be too small to fit. Either way the cards
        # still render; they just carry no severity rung.
        print(f"\n[no classifier: {type(exc).__name__}: {exc}]")
        print("[cards below therefore report level L0 and no classifier probability]")

    cards: list[Card] = []
    for case in bundle.test():
        if not case.regions:
            continue
        if classifier is not None:
            classifier.apply(case.regions, case.z_by_unit)
        cards += build_cards(
            case.regions,
            annotation=annotation,
            envelope=bundle.envelopes[case.text_id],
            z_by_unit=case.z_by_unit,
            mode="reference",
            config=config,
        )

    chosen = _pick(cards, labels)
    print(f"\nfold {bundle.fold}: {len(cards)} card(s) from {len(bundle.test())} recordings")
    print(
        f"text context from DEMO_TEXT ({annotation.n_words} words, "
        f"{len(annotation.sentences)} sentences, {len(annotation.repeats)} repeated spans)"
    )
    for card in chosen:
        print("\n" + "=" * 78)
        print(render_markdown(card))
        print("-" * 78)
        print(json.dumps(card.to_dict(), indent=2, sort_keys=False))

    return {
        "fold": bundle.fold,
        "source": "synthetic",
        "n_cards": len(cards),
        "cards": [card.to_dict() for card in chosen],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-out", default="syn_a", help="LOTO fold to detect")
    parser.add_argument("--readers", type=int, default=6, help="synthetic readers per text")
    parser.add_argument("--all", action="store_true", help="print all eight labels")
    parser.add_argument("--out", type=Path, default=None, help="write the cards as JSON")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    labels = (
        (
            "rushing",
            "dragging",
            "missing_pause",
            "hesitation_pause",
            "monotone",
            "misplaced_stress",
            "low_energy",
            "mumbling",
        )
        if args.all
        else DEMO_LABELS
    )
    payload = run(config=config, held_out=args.held_out, labels=labels, n_readers=args.readers)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
