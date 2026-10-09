"""The five ablations of spec 14.2, each a full leave-one-text-out run.

Every row of the table is produced by :func:`eval.run_loto.run_loto` with one
:class:`~eval.run_loto.Variant` changed, so an ablation can never differ from
the headline run by anything except the thing being ablated - the folds, the
fitting, the leakage guard and the metrics are the same code.

The rows, in the spec's order:

1. **Single reference vs envelope.** ``envelope_size=1``: a gold-tier reading
   where the corpus has one, else one reading drawn with a seeded RNG
   (:meth:`eval.dataset_corpus.DatasetCorpus.pick_references`). This needs
   ``detection.envelope.min_references`` overridden to 1, since the envelope
   builder refuses a single reader by default - that refusal is the guard
   working, so the override is explicit and recorded in the row.
2. **Rules vs rules + classifier.** The headline macro-F1 is the *rule's*
   label, because that is the label a card shows (spec 10.6's label policy
   leaves it in place); this row is where the classifier's own label is
   compared against it.
3. **± boundary refinement.** Ruptures PELT on or off (spec 10.5).
4. **± speaker normalisation.** On the synthetic corpus each speaker is given
   back a habitual pitch, loudness and tempo
   (:func:`eval.cases.denormalise_speaker`), which is a *simulation* of the
   variance invariants 5 and 6 remove and is labelled as one. On the dataset
   corpus it cannot be run at all from cached features, because normalisation
   happens inside extraction; the row says so rather than reporting a number
   from somewhere else.
5. **Synthetic vs human test.** Not a variant: it is the full run's two splits
   side by side, which is why this row carries no separate LOTO run.

The table reports each row's delta against the full system on the metrics spec
14.3 pre-registers, and the output says which corpus produced it. A synthetic
delta is a statement about the protocol, not about speech.

Usage::

    python -m eval.ablations                     # synthetic corpus
    python -m eval.ablations --corpus dataset    # the real one
    python -m eval.ablations --only single_reference rules_only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.dataset_corpus import CorpusUnavailable, DatasetCorpus
from eval.run_loto import LeakageError, Variant, run_loto
from rostrum.config import Config, code_version, load_config, set_seeds

logger = logging.getLogger("eval.ablations")

__all__ = [
    "ABLATIONS",
    "SCHEMA",
    "Ablation",
    "AblationRow",
    "ablations_markdown",
    "main",
    "run_ablations",
]

SCHEMA: str = "rostrum.eval.ablations.v1"
"""Written into ``results/ablations.json`` so a consumer can check what it has."""

HEADLINE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("event_f1_200ms", "Event-F1 @ 200 ms", "higher"),
    ("onset_offset_mae_ms", "Onset/offset MAE (ms)", "lower"),
    ("flaw_type_macro_f1", "Macro-F1 (type)", "higher"),
    ("severity_spearman_rho", "Spearman rho", "lower"),
    ("false_positives_per_min_on_clean", "FP on clean (per min)", "lower"),
)
"""The spec 14.3 quantities compared per row, with the direction that is better."""


@dataclass(frozen=True)
class Ablation:
    """One ablation: a variant, and what has to be true to run it.

    Attributes:
        name: Row id, also the variant name.
        title: Row name in the table.
        variant: The pipeline configuration to run.
        overrides: Config overrides the variant needs. Only ``detection`` keys
            are allowed: an override of another section would change the
            feature cache key, so the row would no longer be comparable with
            the full run.
        corpora: Corpora this row can be run on.
        note: One line for the report.
        runs: Whether the row is its own LOTO run at all. Ablation 5 is not.
    """

    name: str
    title: str
    variant: Variant | None = None
    overrides: Mapping[str, Any] = field(default_factory=dict)
    corpora: tuple[str, ...] = ("synthetic", "dataset")
    note: str = ""
    runs: bool = True


ABLATIONS: tuple[Ablation, ...] = (
    Ablation(
        name="full",
        title="Full system",
        variant=Variant(name="full", description="the system as specified"),
        note="the baseline every delta is measured against",
    ),
    Ablation(
        name="single_reference",
        title="1. Single reference (not an envelope)",
        variant=Variant(
            name="single_reference",
            single_reference=True,
            description="one reference reading instead of the envelope",
        ),
        overrides={"envelope": {"min_references": 1}},
        note=(
            "gold reading where the corpus has one, else one drawn with a seeded RNG; "
            "`min_references` is overridden to 1 so the envelope builder allows it"
        ),
    ),
    Ablation(
        name="rules_only",
        title="2. Rules only (no flaw-type classifier)",
        variant=Variant(
            name="rules_only",
            classifier=False,
            description="region rules without the GBM flaw-type classifier",
        ),
        note=(
            "the emitted label is the rule's either way (spec 10.6), so this row shows "
            "what the classifier adds as a second opinion, not a changed headline label"
        ),
    ),
    Ablation(
        name="no_refinement",
        title="3. No boundary refinement",
        variant=Variant(
            name="no_refinement",
            refine=False,
            description="hysteresis regions without ruptures refinement",
        ),
        note="spec 10.5; expect the onset/offset MAE to move and the F1 to move less",
    ),
    Ablation(
        name="no_speaker_norm",
        title="4. No speaker normalisation",
        variant=Variant(
            name="no_speaker_norm",
            speaker_normalisation=False,
            description="between-speaker offsets left in the comparison",
        ),
        corpora=("synthetic",),
        note=(
            "SIMULATED on the synthetic corpus: each speaker is given back a habitual "
            "pitch, loudness and tempo. On the dataset it would need a second feature "
            "pass with normalisation off, which `make features` does not produce"
        ),
    ),
    Ablation(
        name="human_test",
        title="5. Human test takes (vs synthetic)",
        variant=None,
        runs=False,
        note=(
            "the full run's two splits side by side; no separate fit, because human "
            "takes are test-only (spec 7.3)"
        ),
    ),
)
"""The five ablations of spec 14.2, plus the full system they are measured against."""


# --------------------------------------------------------------------------- #
# One row
# --------------------------------------------------------------------------- #


@dataclass
class AblationRow:
    """One ablation's result.

    Attributes:
        name: Row id.
        title: Row name.
        available: Whether the row produced numbers.
        reason: Why not, when it did not.
        note: The ablation's note.
        split: Which split the metrics come from.
        variant: The variant that produced them.
        overrides: Config overrides used.
        metrics: ``{key: mean}`` for :data:`HEADLINE_METRICS`.
        delta: ``{key: mean - full's mean}``; empty for the full row.
        extra: Anything specific to the row, e.g. the classifier comparison.
    """

    name: str
    title: str
    available: bool = False
    reason: str = ""
    note: str = ""
    split: str = "injected"
    variant: dict[str, Any] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    delta: dict[str, float | None] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "name": self.name,
            "title": self.title,
            "available": self.available,
            "reason": self.reason,
            "note": self.note,
            "split": self.split,
            "variant": self.variant,
            "overrides": self.overrides,
            "metrics": self.metrics,
            "delta": self.delta,
            "extra": self.extra,
        }


def _means(payload: Mapping[str, Any], split: str) -> dict[str, float | None]:
    """The headline means of one aggregate payload, or ``None`` per missing key."""
    block = payload.get("splits", {}).get(split, {})
    metrics = block.get("metrics", {}) if block.get("available") else {}
    return {key: (metrics.get(key, {}) or {}).get("mean") for key, _, _ in HEADLINE_METRICS}


def _n_predicted(payload: Mapping[str, Any], split: str) -> int | None:
    """Regions emitted over all folds on one split, or ``None`` if not recorded.

    A variant that emits nothing cannot be read off the headline means alone -
    an F1 of 0 looks the same whether the regions were wrong or absent.
    """
    total: int | None = None
    for fold in payload.get("per_fold", {}).values():
        block = fold.get("splits", {}).get(split, {})
        if not block.get("available") or "n_predicted_events" not in block:
            continue
        total = int(block["n_predicted_events"]) + (total or 0)
    return total


def _delta(
    row: Mapping[str, float | None], base: Mapping[str, float | None]
) -> dict[str, float | None]:
    """``row - base`` per key, ``None`` where either side is missing."""
    out: dict[str, float | None] = {}
    for key, _, _ in HEADLINE_METRICS:
        here, there = row.get(key), base.get(key)
        out[key] = None if here is None or there is None else round(float(here - there), 4)
    return out


def _typing_comparison(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rule macro-F1 against the classifier's, per fold and averaged.

    Spec 10.6 keeps the rule's label and records the classifier's beside it, so
    the two can only be compared inside one run; this is ablation 2's substance.
    """
    rule: list[float] = []
    clf: list[float] = []
    for fold in payload.get("per_fold", {}).values():
        block = fold.get("splits", {}).get("injected", {})
        if not block.get("available"):
            continue
        typing = block.get("typing", {})
        if typing.get("rule_macro_f1") is not None:
            rule.append(float(typing["rule_macro_f1"]))
        if typing.get("classifier_macro_f1") is not None:
            clf.append(float(typing["classifier_macro_f1"]))
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {
        "rule_macro_f1": mean(rule),
        "classifier_macro_f1": mean(clf),
        "n_folds_with_classifier": len(clf),
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run_ablations(
    *,
    config: Config,
    corpus: str = "synthetic",
    texts: Sequence[str] | None = None,
    only: Sequence[str] | None = None,
    out: Path | None = None,
) -> dict[str, Any]:
    """Run every applicable ablation and write ``results/ablations.{json,md}``.

    Args:
        config: Loaded base config. Each row that needs an override reloads it
            with that override applied; the feature tables stay the base
            config's, so only detection behaviour changes.
        corpus: ``"synthetic"`` or ``"dataset"``.
        texts: Texts to fold over; the corpus's own when ``None``.
        only: Run just these row names; everything applicable when ``None``.
            ``full`` is always run, because every delta is measured against it.
        out: Where to write the JSON; ``results/ablations.json`` by default.

    Returns:
        The payload written to the JSON.

    Raises:
        CorpusUnavailable: ``corpus="dataset"`` without a built dataset.
    """
    wanted = None if only is None else {"full", *only}
    corpus_data: DatasetCorpus | None = None
    if corpus == "dataset":
        # Loaded once, under the BASE config: the feature cache key includes
        # the config hash, so a row that overrides `min_references` would
        # otherwise re-extract the whole corpus under a different key.
        corpus_data = DatasetCorpus.load(config)
        if texts is None:
            texts = corpus_data.texts

    rows: list[AblationRow] = []
    payloads: dict[str, dict[str, Any]] = {}
    base_means: dict[str, float | None] = {}

    for ablation in ABLATIONS:
        if wanted is not None and ablation.name not in wanted:
            continue
        row = AblationRow(name=ablation.name, title=ablation.title, note=ablation.note)
        if corpus not in ablation.corpora:
            row.reason = f"not available on the {corpus} corpus"
            rows.append(row)
            logger.warning("%s: %s", ablation.name, row.reason)
            continue
        if not ablation.runs:
            rows.append(row)  # filled in below from the full run
            continue

        assert ablation.variant is not None  # every runnable row has one
        row.variant = ablation.variant.to_dict()
        row.overrides = dict(ablation.overrides)
        row_config = (
            config
            if not ablation.overrides
            else load_config(overrides={"detection": dict(ablation.overrides)})
        )
        try:
            payload = run_loto(
                config=row_config,
                texts=texts,
                variant=ablation.variant,
                corpus=corpus,
                out=None,
                write_folds=False,
                corpus_data=corpus_data,
            )
        except (CorpusUnavailable, LeakageError, NotImplementedError) as error:
            row.reason = str(error)
            rows.append(row)
            logger.warning("%s: did not run (%s)", ablation.name, error)
            continue

        payloads[ablation.name] = payload
        row.available = True
        row.metrics = _means(payload, "injected")
        row.extra["n_predicted_events"] = _n_predicted(payload, "injected")
        if row.extra["n_predicted_events"] == 0:
            # A row that detected nothing is not a row that scored zero on a
            # working system: its F1 is a real 0, its MAE and false-alarm rate
            # are undefined. Say so, or the table reads as if the ablation
            # improved the edges it never placed.
            row.reason = (
                "this variant emitted no region at all, so only the F1 is a measurement; "
                "the edge error has no matched pair to average and the false-alarm rate "
                "no alarm to count"
            )
        if ablation.name == "full":
            base_means = dict(row.metrics)
        else:
            row.delta = _delta(row.metrics, base_means)
        if ablation.name in ("full", "rules_only"):
            row.extra["typing"] = _typing_comparison(payload)
        rows.append(row)

    # Ablation 5 is the full run's human split beside its injected split.
    full = payloads.get("full")
    for row in rows:
        if row.name != "human_test":
            continue
        if full is None:
            row.reason = "the full run produced no result to compare against"
            continue
        row.split = "human"
        human_available = full.get("splits", {}).get("human", {}).get("available")
        row.variant = dict(full.get("variant", {}))
        if not human_available:
            row.reason = f"the {corpus} corpus has no human takes"
            continue
        row.available = True
        row.metrics = _means(full, "human")
        row.delta = _delta(row.metrics, base_means)

    payload = {
        "schema": SCHEMA,
        "corpus": corpus,
        "config_hash": config.hash,
        "code_version": code_version(config.root),
        # On the synthetic corpus `texts` is left None so run_loto uses its own
        # default, so the folds of whichever run happened are the honest record
        # of what was folded over.
        "texts": list(texts or next(iter(payloads.values()), {}).get("texts") or ()),
        "metrics": [
            {"key": key, "label": label, "better": better}
            for key, label, better in HEADLINE_METRICS
        ],
        "rows": [row.to_dict() for row in rows],
    }

    destination = out or (config.path("results.ablations"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = destination.with_suffix(".md")
    markdown.write_text(ablations_markdown(payload), encoding="utf-8")
    logger.info("wrote %s and %s", destination, markdown)
    return payload


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def _cell(value: float | None, delta: float | None, *, better: str) -> str:
    """One table cell: the value, and the signed delta with a better/worse mark."""
    if value is None:
        return "-"
    text = f"{value:g}"
    if delta is None:
        return text
    if abs(delta) < 1e-9:
        return f"{text} (=)"
    improved = delta > 0 if better == "higher" else delta < 0
    return f"{text} ({delta:+g} {'better' if improved else 'worse'})"


def ablations_markdown(payload: Mapping[str, Any]) -> str:
    """Render ``results/ablations.md`` (prompt 20)."""
    corpus = str(payload.get("corpus", "?"))
    columns = list(payload["metrics"])
    lines = [
        "# Ablations (spec 14.2)",
        "",
        f"* corpus: **{corpus}**"
        + (
            "  - generated feature tables (eval/synthetic.py), not a dataset result"
            if corpus == "synthetic"
            else ""
        ),
        f"* texts: {', '.join(payload.get('texts') or ['-'])}",
        f"* config hash: `{str(payload['config_hash'])[:12]}`, code `{payload['code_version']}`",
        "",
        "Each row is a full leave-one-text-out run with one thing changed; values are the "
        "mean over folds on the injected split, and the bracket is the change against the "
        "full system.",
        "",
        "| Ablation | " + " | ".join(str(c["label"]) for c in columns) + " |",
        "|---" * (len(columns) + 1) + "|",
    ]
    for row in payload["rows"]:
        if not row["available"]:
            lines.append(f"| {row['title']} | " + " | ".join(["not run"] * len(columns)) + " |")
            continue
        # A variant that emitted no region has nothing to be better or worse at:
        # its zeros are the absence of a detection, not a good one, so the
        # delta's verdict is withheld rather than flattering the ablation.
        empty = row.get("extra", {}).get("n_predicted_events") == 0
        cells = [
            _cell(
                row["metrics"].get(c["key"]),
                None if empty else row["delta"].get(c["key"]),
                better=str(c["better"]),
            )
            for c in columns
        ]
        if empty:
            cells = [f"{cell} (no regions)" if cell != "-" else cell for cell in cells]
        lines.append(f"| {row['title']} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Row notes", ""]
    for row in payload["rows"]:
        if not row["available"]:
            status = f" **Not run:** {row['reason'] or 'unavailable'}."
        elif row["reason"]:
            # A row can run and still need a caveat - the single-reference run
            # completes and detects nothing.
            status = f" **Caveat:** {row['reason']}."
        else:
            status = ""
        split = f" (split: {row['split']})" if row["split"] != "injected" else ""
        note = f" {row['note']}." if row["note"] else ""
        lines.append(f"* **{row['title']}**{split}:{note}{status}")
        typing = (row.get("extra") or {}).get("typing")
        if typing:
            lines.append(
                f"  * macro-F1 of the rule's label {typing['rule_macro_f1']}, of the "
                f"classifier's {typing['classifier_macro_f1']} "
                f"({typing['n_folds_with_classifier']} fold(s) had a classifier)."
            )
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints ``results/ablations.md`` on success."""
    parser = argparse.ArgumentParser(description="Ablation table (spec 14.2)")
    parser.add_argument("--out", type=Path, default=None, help="ablations JSON (default results/)")
    parser.add_argument(
        "--corpus",
        choices=("synthetic", "dataset"),
        default="synthetic",
        help="synthetic graded corpus (default) or the real dataset",
    )
    parser.add_argument("--texts", nargs="*", default=None, help="texts to fold over")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=[a.name for a in ABLATIONS],
        help="run only these rows (the full system is always run)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    set_seeds(config.seed)
    try:
        payload = run_ablations(
            config=config,
            corpus=args.corpus,
            texts=tuple(args.texts) if args.texts else None,
            only=args.only,
            out=args.out,
        )
    except CorpusUnavailable as error:
        print(f"cannot run ablations: {error}", file=sys.stderr)
        return 2

    destination = args.out or config.path("results.ablations")
    print(ablations_markdown(payload))
    print(f"wrote {destination} and {destination.with_suffix('.md')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
