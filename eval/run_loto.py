"""Leave-one-text-out evaluation (spec 14, prompt 19).

One fold is one text held out. On the fold's **training** texts the harness
fits everything that can be fitted - the region thresholds
(:func:`rostrum.detect.tune.grid_search`), the rubric's :math:`k_c`
(:func:`rostrum.score.rubric.calibrate_k`) and the flaw-type classifier
(:func:`rostrum.detect.classifier.fit_fold`) - then runs the held-out text
once and never looks back. Spec 7.3's two rules are enforced rather than
documented:

* **nothing fitted on test data.** Every file id that reaches a fitting call is
  collected and :func:`assert_no_leakage` fails the run if one of them is also
  a test file. The check is on ids, not on texts, so it catches a leak that
  slipped in through a hand-assembled list as well as one that came from the
  wrong ``train_texts``.
* **human takes are test-only**, which ``fit_fold`` enforces itself through
  ``detection.classifier.train_on``; the harness asserts the outcome anyway.

The metrics are :mod:`eval.metrics`'s and the targets are
``configs/scoring.yaml``'s ``targets`` block, pre-registered per spec 14.3.
``results/metrics.md`` prints target, actual and PASS/MISS per row, including
the rows that could not be measured - a row that quietly disappears is a row
nobody has to explain.

**Two corpora.** ``--corpus synthetic`` (the default) runs on the graded
synthetic corpus of :mod:`eval.cases`: generated feature tables with known
spans and known rungs. It exercises every line of the protocol deterministically
and says **nothing** about whether the feature extractors recover a flaw a
speaker put into audio - any number from it must be labelled synthetic in the
report. ``--corpus dataset`` is the real thing and needs ``make data``, which
needs Montreal Forced Aligner; where the dataset is absent the harness refuses
with the list of what is missing rather than reporting a number it cannot
stand behind.

Usage::

    python -m eval.run_loto --out results/metrics.json
    python -m eval.run_loto --no-classifier --no-refine   # ablation variants
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from eval.cases import Case, FoldCases, build_fold_cases
from eval.dataset_corpus import CorpusUnavailable, DatasetCorpus
from eval.metrics import (
    ScoredEvent,
    boundary_errors,
    check_targets,
    crosscheck_event_f1,
    event_diagnostics,
    false_positives_per_minute,
    mean_sd,
    pr_curve,
    sed_eval_available,
    severity_spearman,
    sex_score_gap,
    targets_markdown,
    typing_metrics,
)
from rostrum.config import (
    Config,
    check_pythonhashseed,
    code_version,
    load_config,
    set_seeds,
)
from rostrum.data.splits import TEST_ONLY_ORIGINS as _SPLIT_TEST_ONLY_ORIGINS
from rostrum.detect.classifier import (
    ClassifierError,
    FlawClassifier,
    RegionFeatureSpec,
    build_examples,
    evaluate,
    fit_fold,
)
from rostrum.detect.refine import RefinementSettings, refine_regions
from rostrum.detect.regions import RegionThresholds, build_regions
from rostrum.detect.tune import (
    Event,
    TuningCase,
    TuningResult,
    events_from_regions,
    grid_search,
)
from rostrum.detect.zscore import trigger_signals
from rostrum.score.rubric import calibrate_k, category_order, label_to_category, rubric_score
from rostrum.types import Region

logger = logging.getLogger("eval.run_loto")

__all__ = [
    "DEFAULT_TEXTS",
    "TEST_ONLY_ORIGINS",
    "CorpusUnavailable",
    "DatasetCorpus",
    "FoldReport",
    "LeakageError",
    "Variant",
    "aggregate",
    "assert_no_leakage",
    "main",
    "metrics_markdown",
    "run_fold",
    "run_loto",
]


DEFAULT_TEXTS: tuple[str, ...] = ("syn_a", "syn_b", "syn_c")
"""The synthetic corpus's texts; the dataset corpus reads its own from the splits."""

SCHEMA: str = "rostrum.eval.metrics.v1"
"""Written into ``results/metrics.json`` so a consumer can check what it has."""


TEST_ONLY_ORIGINS: tuple[str, ...] = tuple(_SPLIT_TEST_ONLY_ORIGINS)
"""Origins that may never be fitted on; ``rostrum.data.splits`` owns the list."""


class LeakageError(AssertionError):
    """A test file reached a fitting call (spec 7.3, CLAUDE.md invariant 11)."""


@dataclass(frozen=True)
class Variant:
    """One configuration of the pipeline, so the ablations are the same code path.

    Attributes:
        name: Short id, used as the row name in the ablation table.
        refine: Run ruptures boundary refinement (ablation 3).
        classifier: Fit and apply the flaw-type classifier (ablation 2).
        single_reference: Build the envelope from one reference (ablation 1).
        speaker_normalisation: Keep speaker normalisation (ablation 4).
        tune: Search the threshold grid on the training texts. Off means the
            config's defaults, which is how a fold with too few training events
            still produces a number.
        description: One line for the report.
    """

    name: str = "full"
    refine: bool = True
    classifier: bool = True
    single_reference: bool = False
    speaker_normalisation: bool = True
    tune: bool = True
    description: str = "the system as specified"

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return asdict(self)


@dataclass
class FoldReport:
    """One fold's result, ready to be written to ``results/fold_<text>.json``.

    Attributes:
        fold: Fold id.
        held_out: The held-out text.
        train_texts: Texts everything was fitted on.
        payload: The full metric payload.
    """

    fold: str
    held_out: str
    train_texts: tuple[str, ...]
    payload: dict[str, Any] = field(default_factory=dict)

    def write(self, directory: Path) -> Path:
        """Write ``fold_<held_out>.json`` into ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"fold_{self.held_out}.json"
        path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #


def assert_no_leakage(
    fitted_on: Sequence[str],
    test_ids: Sequence[str],
    *,
    fold: str = "",
    test_only_origins: Sequence[str] = ("human",),
    origins: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail the run if a test file, or a test-only origin, was fitted on.

    Args:
        fitted_on: Every file id that reached a fitting call.
        test_ids: Every file id of the held-out text.
        fold: Fold id, for the message.
        test_only_origins: Origins that may never be fitted on (spec 7.3).
        origins: ``{file_id: origin}``; without it the origin check is skipped
            and the returned payload says so.

    Returns:
        An audit record for the fold JSON.

    Raises:
        LeakageError: An overlap exists.
    """
    fitted = sorted(set(fitted_on))
    held_out = set(test_ids)
    overlap = sorted(f for f in fitted if f in held_out)
    if overlap:
        raise LeakageError(
            f"fold {fold or '?'}: {len(overlap)} held-out file(s) reached a fitting call: "
            f"{overlap[:5]}{' ...' if len(overlap) > 5 else ''}"
        )

    forbidden: list[str] = []
    if origins is not None:
        forbidden = sorted(f for f in fitted if str(origins.get(f, "")) in set(test_only_origins))
        if forbidden:
            raise LeakageError(
                f"fold {fold or '?'}: test-only origin(s) {sorted(test_only_origins)} reached a "
                f"fitting call: {forbidden[:5]}"
            )

    return {
        "n_fitted_on": len(fitted),
        "n_test": len(held_out),
        "overlap": [],
        "test_only_origins": list(test_only_origins),
        "origins_checked": origins is not None,
    }


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def _synthetic_fold(
    *, config: Config, held_out: str, texts: Sequence[str], variant: Variant
) -> FoldCases:
    """Build one fold of the graded synthetic corpus under ``variant``."""
    return build_fold_cases(
        config=config,
        held_out=held_out,
        texts=texts,
        refine=variant.refine,
        envelope_size=1 if variant.single_reference else None,
        speaker_offsets=not variant.speaker_normalisation,
    )


_DATASET_CACHE: dict[tuple[str, bool], DatasetCorpus] = {}
"""One loaded corpus per (config hash, lean) so N folds read the tables once."""


def dataset_corpus(config: Config, *, lean: bool = False) -> DatasetCorpus:
    """The real corpus, loaded once per process.

    Args:
        config: Loaded config.
        lean: Load lean feature entries; tests only.

    Returns:
        The cached :class:`~eval.dataset_corpus.DatasetCorpus`.

    Raises:
        CorpusUnavailable: ``make data`` or ``make features`` has not run; the
            message lists what is missing and what produces each piece.
    """
    key = (str(config.hash), bool(lean))
    corpus = _DATASET_CACHE.get(key)
    if corpus is None:
        corpus = DatasetCorpus.load(config, lean=lean)
        _DATASET_CACHE[key] = corpus
    return corpus


def _dataset_fold(
    *,
    config: Config,
    held_out: str,
    variant: Variant,
    lean: bool = False,
    corpus_data: DatasetCorpus | None = None,
) -> FoldCases:
    """Build one fold of the real corpus under ``variant``.

    Raises:
        CorpusUnavailable: The corpus is not built, or the held-out text has no
            envelope.
        NotImplementedError: ``variant.speaker_normalisation`` is off. On the
            dataset that ablation cannot be run from cached tables: speaker
            normalisation happens *inside* feature extraction (F0 in semitones
            re the speaker's median, energy in dB re their voiced median -
            CLAUDE.md invariants 5 and 6), so removing it means re-extracting,
            which ``make features`` does not produce. :mod:`eval.ablations`
            reports the row as unavailable on this corpus and quotes the
            synthetic simulation instead.
    """
    if not variant.speaker_normalisation:
        raise NotImplementedError(
            "ablation 4 (no speaker normalisation) cannot be run on the dataset corpus: "
            "normalisation is applied inside feature extraction, so it would need a second "
            "`make features` pass with it switched off"
        )
    corpus = corpus_data if corpus_data is not None else dataset_corpus(config, lean=lean)
    return corpus.fold(
        config=config,
        held_out=held_out,
        refine=variant.refine,
        envelope_size=1 if variant.single_reference else None,
    )


# --------------------------------------------------------------------------- #
# One fold
# --------------------------------------------------------------------------- #


def _truth_events(case: Case) -> list[Event]:
    """The case's ground-truth flaws as events."""
    return [
        Event(label=flaw.label, t_start=flaw.t_start, t_end=flaw.t_end, file_id=case.file_id)
        for flaw in case.truth
    ]


def _redetect(
    case: Case,
    *,
    config: Config,
    thresholds: RegionThresholds,
    refinement: RefinementSettings | None,
) -> list[Region]:
    """Re-run region formation for one case at the fold's tuned thresholds.

    ``same_channel`` comes from the case, not from a constant: on the dataset a
    participant compared against a multi-source envelope is not channel-matched,
    and the spectral and voice-quality triggers must stay off for it (CLAUDE.md
    invariant 7).
    """
    signals, _ = trigger_signals(case.z_by_unit, config=config, same_channel=case.same_channel)
    regions = build_regions(signals, case.z_by_unit, config=config, thresholds=thresholds)
    if refinement is not None and refinement.enabled:
        regions = refine_regions(regions, case.frames, settings=refinement)
    return regions


def _tuning_cases(cases: Sequence[Case], *, config: Config) -> list[TuningCase]:
    """Training cases in the form the threshold search wants."""
    out: list[TuningCase] = []
    for case in cases:
        signals, _ = trigger_signals(case.z_by_unit, config=config, same_channel=case.same_channel)
        out.append(
            TuningCase(
                file_id=case.file_id,
                text_id=case.text_id,
                signals=signals,
                z_by_unit=case.z_by_unit,
                truth=_truth_events(case),
                frames=case.frames,
            )
        )
    return out


def _apply_classifier(
    regions: Sequence[Region], classifier: FlawClassifier, case: Case
) -> list[Region]:
    """Attach the classifier's type and severity to each region (spec 10.6).

    :meth:`~rostrum.detect.classifier.FlawClassifier.annotate` applies the
    label policy, which leaves the rule's label in place and records the
    classifier's beside it - the card still explains what the rule measured.
    """
    return classifier.annotate(list(regions), case.z_by_unit)


def _split_metrics(
    cases: Sequence[Case],
    predictions: Mapping[str, list[Region]],
    *,
    config: Config,
    classifier: FlawClassifier | None,
    spec: RegionFeatureSpec,
    label: str,
) -> dict[str, Any]:
    """Every grounding metric for one split of the held-out text.

    Args:
        cases: The split's flawed cases; clean files belong in the false-alarm
            measurement instead.
        predictions: ``{file_id: regions}`` for those cases.
        config: Loaded config.
        classifier: The fold's classifier, or ``None`` for a rules-only run.
        spec: Region feature spec, for the typing examples.
        label: Split name, for the payload.

    Returns:
        The split's payload, or a ``{"available": False}`` record when the split
        is empty - a corpus with no human takes must say so, not report zero.
    """
    if not cases:
        return {"available": False, "reason": f"no {label} files in this corpus", "n_files": 0}

    reference: list[Event] = []
    predicted: list[Event] = []
    scored: list[ScoredEvent] = []
    for case in cases:
        reference.extend(_truth_events(case))
        regions = predictions.get(case.file_id, [])
        events = events_from_regions(regions, file_id=case.file_id)
        predicted.extend(events)
        by_span = {(round(r.t_start, 3), round(r.t_end, 3)): r for r in regions}
        for event in events:
            region = by_span.get((round(event.t_start, 3), round(event.t_end, 3)))
            scored.append(
                ScoredEvent(event=event, score=abs(float(region.peak_z)) if region else 0.0)
            )

    events_block = crosscheck_event_f1(reference, predicted)
    boundaries = boundary_errors(reference, predicted)
    curve = pr_curve(reference, scored)
    diagnostics = event_diagnostics(
        reference,
        predicted,
        scores={
            (item.event.file_id, item.event.t_start, item.event.t_end): item.score
            for item in scored
        },
    )

    # Typing: the rule's label and the classifier's, over the regions that
    # matched a truth flaw. build_examples does the matching, which keeps this
    # identical to what the classifier was trained against.
    typed_cases = [_classifier_case(case, predictions.get(case.file_id, [])) for case in cases]
    examples = build_examples(typed_cases, spec)
    rule_typing = typing_metrics(
        [example.label for example in examples],
        [str(example.rule_label) for example in examples],
    )
    clf_typing: dict[str, Any] | None = None
    if classifier is not None and examples:
        try:
            clf_typing = evaluate(classifier, examples)
        except Exception as error:  # pragma: no cover - defensive
            logger.warning("%s: classifier evaluation failed: %s", label, error)

    # The headline is the **rule's** macro-F1, because the rule's label is what
    # the system emits: spec 10.6's label policy records the classifier's type
    # beside it and never overrules it (see
    # :func:`rostrum.detect.classifier.apply_label_policy`). Quoting the
    # classifier here would score a label no card ever shows. Ablation 2 is the
    # place the two are compared.
    headline_macro = float(rule_typing["macro_f1"])

    return {
        "available": True,
        "n_files": len(cases),
        "n_reference_events": len(reference),
        "n_predicted_events": len(predicted),
        "events": events_block,
        "event_f1": float(events_block["rostrum"]["f1"]),
        "boundaries": boundaries,
        # Undefined, not zero, when nothing matched: BoundaryErrors defaults its
        # means to 0.0 with n=0, which would otherwise read as a perfect 0 ms
        # edge placement for a run that detected nothing at all (the single-
        # reference ablation does exactly that).
        "onset_offset_mae_ms": (
            float(boundaries["iou_matched"]["mae_ms"])
            if int(boundaries["iou_matched"]["n"]) > 0
            else None
        ),
        # Per-event records, for the onset histogram and the failure analysis
        # of prompt 20; the aggregates above are computed from the same pairs.
        "diagnostics": diagnostics,
        "pr_curve": curve,
        "typing": {
            "rule": rule_typing,
            "classifier": clf_typing,
            "n_examples": len(examples),
            "macro_f1": round(headline_macro, 4),
            "rule_macro_f1": round(float(rule_typing["macro_f1"]), 4),
            "classifier_macro_f1": (
                round(float(clf_typing["macro_f1"]), 4) if clf_typing is not None else None
            ),
        },
    }


def _classifier_case(case: Case, regions: Sequence[Region]) -> Any:
    """The case as a :class:`~rostrum.detect.classifier.ClassifierCase` with ``regions``.

    The case's own ``origin`` is carried through, which is what makes
    :func:`~rostrum.detect.classifier.fit_fold`'s ``train_on`` filter able to
    hold human takes out of the fit (spec 7.3).
    """
    from rostrum.detect.classifier import ClassifierCase

    return ClassifierCase(
        file_id=case.file_id,
        text_id=case.text_id,
        regions=list(regions),
        z_by_unit=case.z_by_unit,
        truth=case.truth,
        origin=case.origin,
    )


def _severity_block(
    cases: Sequence[Case],
    scores: Mapping[str, Any],
    *,
    config: Config,
) -> dict[str, Any]:
    """Spearman rho of injected severity against the flaw's own category score.

    Pooled over labels and per category, because a pooled rho can hide one
    category whose score does not move at all.
    """
    mapping = label_to_category(config)
    pooled_levels: list[float] = []
    pooled_scores: list[float] = []
    by_category: dict[str, tuple[list[float], list[float]]] = {}
    for case in cases:
        if not case.truth or not case.label:
            continue
        category = case.category or mapping.get(case.label, "")
        score = scores.get(case.file_id)
        if score is None or not category:
            continue
        value = score.get(category)
        if value is None:
            continue
        severity = float(case.truth[0].s)
        pooled_levels.append(severity)
        pooled_scores.append(float(value))
        levels, values = by_category.setdefault(category, ([], []))
        levels.append(severity)
        values.append(float(value))

    return {
        "pooled": severity_spearman(pooled_levels, pooled_scores),
        "per_category": {
            category: severity_spearman(levels, values)
            for category, (levels, values) in sorted(by_category.items())
        },
        "n": len(pooled_levels),
    }


def run_fold(
    *,
    config: Config,
    held_out: str,
    texts: Sequence[str] = DEFAULT_TEXTS,
    variant: Variant | None = None,
    corpus: str = "synthetic",
    lean: bool = False,
    corpus_data: DatasetCorpus | None = None,
) -> FoldReport:
    """Fit on the fold's training texts, then evaluate the held-out text once.

    Args:
        config: Loaded config.
        held_out: The text to hold out.
        texts: Every text in the corpus.
        variant: Which pipeline configuration to run; the full system by
            default.
        corpus: ``"synthetic"`` or ``"dataset"``.
        lean: Load lean feature-cache entries for the dataset corpus. Tests
            only: a lean entry has a different cache key from a full one, so a
            reported number must never come from one.
        corpus_data: An already-loaded dataset corpus to fold, instead of the
            one this ``config`` would open. :mod:`eval.ablations` passes the
            corpus it loaded under the **base** config, because the feature
            cache key includes the config hash and a variant that overrides a
            detection threshold would otherwise re-extract every file.

    Returns:
        A :class:`FoldReport`.

    Raises:
        CorpusUnavailable: ``corpus="dataset"`` without a built dataset.
        LeakageError: A held-out file reached a fitting call.
    """
    variant = variant or Variant()
    bundle = (
        _synthetic_fold(config=config, held_out=held_out, texts=texts, variant=variant)
        if corpus == "synthetic"
        else _dataset_fold(
            config=config,
            held_out=held_out,
            variant=variant,
            lean=lean,
            corpus_data=corpus_data,
        )
    )
    spec = RegionFeatureSpec.from_config(config)
    test = bundle.test()
    # Fitting sees the training texts minus the test-only origins. Dropping
    # human takes here rather than relying on each fitter's own filter is what
    # lets the leakage guard below be an assertion about *inputs*: anything in
    # `fitted_on` was genuinely offered to a fit.
    train = [case for case in bundle.train() if case.origin not in TEST_ONLY_ORIGINS]
    withheld = [case.file_id for case in bundle.train() if case.origin in TEST_ONLY_ORIGINS]
    if withheld:
        logger.info(
            "fold %s: %d training-text file(s) withheld from every fit, origin in %s: %s",
            bundle.fold,
            len(withheld),
            list(TEST_ONLY_ORIGINS),
            withheld[:5],
        )
    fitted_on: list[str] = []

    # -- fit: thresholds ----------------------------------------------------
    base = RegionThresholds.from_config(config)
    tuning: TuningResult | None = None
    if variant.tune:
        cases = _tuning_cases(train, config=config)
        fitted_on.extend(case.file_id for case in cases)
        tuning = grid_search(
            cases,
            fold=bundle.fold,
            train_texts=bundle.train_texts,
            config=config,
            refine=variant.refine,
            base_thresholds=base,
        )
        thresholds = tuning.thresholds(config=config, base=base)
        penalty = tuning.penalty
    else:
        thresholds = base
        penalty = float(config.detection["refinement"].get("penalty", 3.0))

    refinement: RefinementSettings | None = None
    if variant.refine:
        from dataclasses import replace as _replace

        refinement = _replace(RefinementSettings.from_config(config), penalty=penalty)

    # -- re-detect every case at the fold's operating point -----------------
    detected: dict[str, list[Region]] = {
        case.file_id: _redetect(case, config=config, thresholds=thresholds, refinement=refinement)
        for case in bundle.cases
    }

    # -- fit: classifier ----------------------------------------------------
    classifier: FlawClassifier | None = None
    classifier_error = ""
    if variant.classifier:
        train_cases = [_classifier_case(c, detected[c.file_id]) for c in train]
        fitted_on.extend(case.file_id for case in train_cases)
        try:
            classifier = fit_fold(
                train_cases,
                fold=bundle.fold,
                train_texts=bundle.train_texts,
                config=config,
                spec=spec,
            )
        except ClassifierError as error:
            classifier_error = str(error)
            logger.warning("fold %s: no classifier - %s", bundle.fold, error)

    if classifier is not None:
        for case in test:
            detected[case.file_id] = _apply_classifier(detected[case.file_id], classifier, case)

    # -- fit: k_c -----------------------------------------------------------
    calibration = []
    for case in train:
        fitted_on.append(case.file_id)
        calibration.append(_calibration_case(case, detected[case.file_id], config=config))
    k_by_category = calibrate_k(calibration, train_texts=bundle.train_texts, config=config)

    # -- leakage guard ------------------------------------------------------
    audit = assert_no_leakage(
        fitted_on,
        [case.file_id for case in test],
        fold=bundle.fold,
        test_only_origins=TEST_ONLY_ORIGINS,
        origins={case.file_id: case.origin for case in bundle.cases},
    )

    # -- score the held-out text -------------------------------------------
    scores: dict[str, Any] = {}
    overall: dict[str, float] = {}
    for case in test:
        score = rubric_score(
            detected[case.file_id],
            case.speech_duration,
            k_by_category=k_by_category,
            config=config,
        )
        scores[case.file_id] = {
            str(category.category): float(category.score)
            for category in score.categories
            if not category.suppressed
        }
        overall[case.file_id] = float(score.overall)

    flawed = [case for case in test if case.truth]
    clean = [case for case in test if not case.truth]
    human_ids = {case.file_id for case in flawed if case.origin == "human"}
    human = [case for case in flawed if case.file_id in human_ids]
    injected = [case for case in flawed if case.file_id not in human_ids]

    payload: dict[str, Any] = {
        "schema": "rostrum.eval.fold.v1",
        "corpus": corpus,
        "fold": bundle.fold,
        "held_out": held_out,
        "train_texts": list(bundle.train_texts),
        "variant": variant.to_dict(),
        "config_hash": config.hash,
        "code_version": code_version(config.root),
        "sed_eval_available": sed_eval_available(),
        "fit": {
            "thresholds": {
                "tau_in": float(thresholds.tau_in),
                "tau_out": float(thresholds.tau_out),
                "penalty": float(penalty),
                "tuned": bool(variant.tune),
            },
            "tuning": tuning.to_dict() if tuning is not None else None,
            "k_by_category": {k: round(float(v), 4) for k, v in sorted(k_by_category.items())},
            "classifier": (
                {
                    "fitted": True,
                    "n_train": int(classifier.n_train),
                    "classes": list(classifier.classes),
                }
                if classifier is not None
                else {"fitted": False, "reason": classifier_error or "disabled by variant"}
            ),
            "n_train_files": len(train),
        },
        "leakage_check": audit,
        "splits": {
            "injected": _split_metrics(
                injected,
                detected,
                config=config,
                classifier=classifier,
                spec=spec,
                label="injected",
            ),
            "human": _split_metrics(
                human, detected, config=config, classifier=classifier, spec=spec, label="human"
            ),
        },
        "clean": false_positives_per_minute(
            [
                event
                for case in clean
                for event in events_from_regions(detected[case.file_id], file_id=case.file_id)
            ],
            {case.file_id: case.duration for case in clean},
        ),
        "severity": _severity_block(flawed, scores, config=config),
        # What each held-out file is, so a figure can group by label and rung
        # and a failure can be named without re-reading the corpus (prompt 20).
        "cases": {
            case.file_id: {
                "text_id": case.text_id,
                "origin": case.origin,
                "tier": case.tier,
                "label": case.label,
                "level": case.level,
                "category": case.category,
                "n_flaws": case.n_flaws,
                "same_channel": bool(case.same_channel),
                "duration_s": round(float(case.duration), 3),
                "speech_duration_s": round(float(case.speech_duration), 3),
                "n_truth": len(case.truth),
                "n_regions": len(detected[case.file_id]),
                "severity_s": (round(float(case.truth[0].s), 4) if case.truth else 0.0),
            }
            for case in sorted(test, key=lambda c: c.file_id)
        },
        "scores": {
            "overall": {file_id: round(value, 4) for file_id, value in sorted(overall.items())},
            "categories": {
                file_id: {k: round(v, 4) for k, v in sorted(block.items())}
                for file_id, block in sorted(scores.items())
            },
            "category_order": list(category_order(config)),
        },
        "fairness": sex_score_gap(overall, _sex_register(config)),
    }
    return FoldReport(
        fold=bundle.fold,
        held_out=held_out,
        train_texts=bundle.train_texts,
        payload=payload,
    )


def _calibration_case(case: Case, regions: Sequence[Region], *, config: Config) -> Any:
    """The case as a rubric :class:`~rostrum.score.rubric.CalibrationCase`."""
    from rostrum.score.rubric import CalibrationCase, deviation_mass

    return CalibrationCase(
        file_id=case.file_id,
        text_id=case.text_id,
        category=case.category,
        level=case.level,
        mass=(
            deviation_mass(regions, case.speech_duration, category=case.category, config=config)
            if case.category
            else 0.0
        ),
        n_flaws=case.n_flaws,
        origin=case.origin,
    )


def _sex_register(config: Config) -> dict[str, str]:
    """``{file_id: sex}`` from ``dataset/speakers.csv``, when the dataset has one.

    Spec 7.4 fixes ``MANIFEST.csv``'s columns and none of them is the speaker's
    sex, so the male-female gap of spec 14.3 reads an optional side register
    instead: ``dataset/speakers.csv`` with columns ``file_id,sex``. Without it
    the gap is reported as unmeasurable, which is the honest answer for a
    corpus that does not record sex.
    """
    import csv

    try:
        path = config.path("data.dir") / "speakers.csv"
    except Exception:  # pragma: no cover - a config without that key
        return {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        str(row.get("file_id", "")): str(row.get("sex", "")) for row in rows if row.get("file_id")
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _fold_target_values(payload: Mapping[str, Any], split: str) -> dict[str, float | None]:
    """The spec 14.3 quantities for one fold and one split."""
    block = payload["splits"].get(split, {})
    if not block.get("available"):
        return dict.fromkeys(
            (
                "event_f1_200ms",
                "onset_offset_mae_ms",
                "flaw_type_macro_f1",
                "severity_spearman_rho",
                "false_positives_per_min_on_clean",
                "male_female_score_gap_points",
            ),
            None,
        )
    severity = payload.get("severity", {}).get("pooled", {})
    return {
        "event_f1_200ms": float(block["event_f1"]),
        "onset_offset_mae_ms": (
            None
            if block["onset_offset_mae_ms"] is None
            else float(block["onset_offset_mae_ms"])
        ),
        "flaw_type_macro_f1": float(block["typing"]["macro_f1"]),
        "severity_spearman_rho": severity.get("rho"),
        "false_positives_per_min_on_clean": payload.get("clean", {}).get("fp_per_min"),
        "male_female_score_gap_points": payload.get("fairness", {}).get("gap_points"),
    }


def aggregate(
    reports: Sequence[FoldReport], *, config: Config, corpus: str, variant: Variant
) -> dict[str, Any]:
    """Mean ± sd across folds, with the pre-registered targets attached.

    Args:
        reports: One per fold.
        config: Loaded config, for ``scoring.targets``.
        corpus: Which corpus produced these folds.
        variant: Which pipeline configuration produced them.

    Returns:
        The payload of ``results/metrics.json``.
    """
    targets = dict(config.scoring.get("targets", {}))
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "corpus": corpus,
        "variant": variant.to_dict(),
        "config_hash": config.hash,
        "code_version": code_version(config.root),
        "sed_eval_available": sed_eval_available(),
        "n_folds": len(reports),
        "folds": [report.fold for report in reports],
        # The held-out texts, in fold order: the caller may not have passed a
        # text list (the corpora supply their own default), so this is the only
        # record of what was actually folded over.
        "texts": [report.held_out for report in reports],
        "splits": {},
        "targets": {},
        "per_fold": {report.fold: report.payload for report in reports},
    }

    for split in ("injected", "human"):
        per_fold = [_fold_target_values(report.payload, split) for report in reports]
        keys = sorted({key for row in per_fold for key in row})
        summary = {key: mean_sd([row.get(key) for row in per_fold]) for key in keys}
        out["splits"][split] = {
            "available": any(
                report.payload["splits"].get(split, {}).get("available") for report in reports
            ),
            "metrics": summary,
        }
        target_split = "synthetic" if split == "injected" else "human"
        rows = check_targets(
            {key: summary[key]["mean"] for key in keys},
            targets.get(target_split, {}),
        )
        out["targets"][target_split] = [row.to_dict() for row in rows]

    out["clean_minutes"] = round(
        float(sum(float(report.payload["clean"].get("minutes") or 0.0) for report in reports)), 4
    )
    deltas = [
        report.payload["splits"]["injected"]["events"].get("delta_f1")
        for report in reports
        if report.payload["splits"].get("injected", {}).get("available")
    ]
    out["sed_eval_crosscheck"] = {
        "available": sed_eval_available(),
        "delta_f1": mean_sd(deltas),
    }
    return out


def metrics_markdown(payload: Mapping[str, Any]) -> str:
    """Render ``results/metrics.md``: targets, actuals and PASS/MISS (prompt 19)."""
    corpus = str(payload.get("corpus", "?"))
    lines = [
        "# LOTO evaluation",
        "",
        f"* corpus: **{corpus}**"
        + (
            "  - generated feature tables, not a dataset result (spec 14.1 caveat)"
            if corpus == "synthetic"
            else ""
        ),
        f"* variant: `{payload['variant']['name']}` - {payload['variant']['description']}",
        f"* folds: {payload['n_folds']} ({', '.join(payload['folds'])})",
        f"* config hash: `{str(payload['config_hash'])[:12]}`, "
        f"code `{payload['code_version']}`",
        f"* event-F1 cross-checked against sed_eval: "
        f"{'yes' if payload.get('sed_eval_available') else 'no (not importable here)'}",
        "",
        "Values are mean over folds; the per-fold spread is in `results/fold_<text>.json`.",
        "",
    ]
    for split, title in (("synthetic", "Synthetic (held-out text)"), ("human", "Human takes")):
        rows_key = split
        rows = payload["targets"].get(rows_key, [])
        source = "injected" if split == "synthetic" else "human"
        available = payload["splits"].get(source, {}).get("available")
        lines.append(f"## {title}")
        lines.append("")
        if not available:
            lines += [
                f"Not evaluated: this corpus has no {source} files.",
                "",
            ]
            continue
        from eval.metrics import TargetRow

        lines.append(targets_markdown([TargetRow(**row) for row in rows]))
        summary = payload["splits"][source]["metrics"]
        clean_minutes = payload.get("clean_minutes")
        if clean_minutes is not None and clean_minutes < 2.0:
            lines += [
                f"The false-alarm rate is measured over **{clean_minutes:g} minute(s)** of clean "
                "audio in total, so one stray region moves it by several per minute. Read it "
                "with the per-fold counts in the fold JSON, not on its own.",
                "",
            ]
        lines += ["", "| Metric | Mean | SD | Folds |", "|---|---|---|---|"]
        for key, block in sorted(summary.items()):
            mean = "-" if block["mean"] is None else f"{block['mean']:g}"
            sd = "-" if block["sd"] is None else f"{block['sd']:g}"
            lines.append(f"| {key} | {mean} | {sd} | {block['n']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def corpus_texts(
    config: Config, *, corpus: str, texts: Sequence[str] | None = None, lean: bool = False
) -> tuple[str, ...]:
    """The texts to fold over: the caller's, or the corpus's own.

    Args:
        config: Loaded config.
        corpus: ``"synthetic"`` or ``"dataset"``.
        texts: An explicit list, which wins when given.
        lean: Passed through when the dataset has to be opened to answer.

    Returns:
        The text ids, in fold order.

    Raises:
        CorpusUnavailable: The dataset corpus is not built.
    """
    if texts:
        return tuple(texts)
    if corpus == "synthetic":
        return DEFAULT_TEXTS
    return dataset_corpus(config, lean=lean).texts


def run_loto(
    *,
    config: Config,
    texts: Sequence[str] | None = None,
    variant: Variant | None = None,
    corpus: str = "synthetic",
    out: Path | None = None,
    write_folds: bool = True,
    lean: bool = False,
    corpus_data: DatasetCorpus | None = None,
) -> dict[str, Any]:
    """Run every fold, aggregate, and write ``metrics.json`` / ``metrics.md``.

    Args:
        config: Loaded config.
        texts: Texts to fold over. ``None`` means the corpus's own texts -
            :data:`DEFAULT_TEXTS` for the synthetic corpus, every text with
            reference readings for the dataset.
        variant: Pipeline configuration; the full system by default.
        corpus: ``"synthetic"`` or ``"dataset"``.
        out: Where to write the JSON, with the markdown beside it. ``None``
            writes nothing and only returns the payload, which is what an
            ablation row wants - it must not overwrite the headline metrics.
        write_folds: Also write one ``fold_<text>.json`` per fold.
        lean: Lean feature-cache entries for the dataset corpus; tests only.
        corpus_data: An already-loaded dataset corpus; see :func:`run_fold`.

    Returns:
        The aggregate payload.
    """
    variant = variant or Variant()
    if texts is None and corpus_data is not None:
        texts = corpus_data.texts
    fold_texts = corpus_texts(config, corpus=corpus, texts=texts, lean=lean)
    if len(fold_texts) < 2:
        logger.warning(
            "only %d text(s) in the %s corpus: a leave-one-text-out fold then has %d training "
            "text(s), so nothing can be fitted across texts",
            len(fold_texts),
            corpus,
            len(fold_texts) - 1,
        )
    reports: list[FoldReport] = []
    for held_out in fold_texts:
        logger.info("fold: holding out %s", held_out)
        report = run_fold(
            config=config,
            held_out=held_out,
            texts=fold_texts,
            variant=variant,
            corpus=corpus,
            lean=lean,
            corpus_data=corpus_data,
        )
        reports.append(report)

    payload = aggregate(reports, config=config, corpus=corpus, variant=variant)
    if out is None:
        # A library call writes nothing. Defaulting to results/metrics.json here
        # made every ablation row overwrite the headline run's metrics with its
        # own ablated numbers, so `make ablations` left results/ describing the
        # last variant that happened to run. The CLI resolves its own default.
        return payload
    destination = out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = destination.with_suffix(".md")
    markdown.write_text(metrics_markdown(payload), encoding="utf-8")
    if write_folds:
        for report in reports:
            report.write(destination.parent)
    logger.info("wrote %s and %s", destination, markdown)
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; prints ``results/metrics.md`` on success."""
    parser = argparse.ArgumentParser(description="Leave-one-text-out evaluation (spec 14)")
    parser.add_argument("--out", type=Path, default=None, help="metrics JSON (default results/)")
    parser.add_argument(
        "--corpus",
        choices=("synthetic", "dataset"),
        default="synthetic",
        help="synthetic graded corpus (default) or the real dataset",
    )
    parser.add_argument(
        "--texts",
        nargs="*",
        default=None,
        help="texts to fold over; default: the corpus's own",
    )
    parser.add_argument("--no-refine", action="store_true", help="skip boundary refinement")
    parser.add_argument("--no-classifier", action="store_true", help="rules only")
    parser.add_argument("--no-tune", action="store_true", help="use the config's thresholds")
    parser.add_argument("--single-reference", action="store_true", help="envelope of one reader")
    parser.add_argument(
        "--no-speaker-norm", action="store_true", help="re-inject between-speaker offsets"
    )
    parser.add_argument("--name", default="full", help="variant name for the report")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    overrides = (
        {"detection": {"envelope": {"min_references": 1}}} if args.single_reference else None
    )
    config = load_config(overrides=overrides)
    # Invariant 13: anything that reaches hash() of a str is only
    # reproducible with the seed pinned, so say so before producing numbers
    # someone will quote.
    check_pythonhashseed()
    set_seeds(config.seed)

    variant = Variant(
        name=args.name,
        refine=not args.no_refine,
        classifier=not args.no_classifier,
        single_reference=args.single_reference,
        speaker_normalisation=not args.no_speaker_norm,
        tune=not args.no_tune,
        description="the system as specified" if args.name == "full" else args.name,
    )
    # Resolved here, not inside run_loto: a library call with out=None writes
    # nothing, so the CLI owns the default path.
    destination = args.out or (config.path("results.dir") / "metrics.json")
    try:
        payload = run_loto(
            config=config,
            texts=tuple(args.texts) if args.texts else None,
            variant=variant,
            corpus=args.corpus,
            out=destination,
        )
    except CorpusUnavailable as error:
        print(f"cannot evaluate: {error}", file=sys.stderr)
        return 2
    except NotImplementedError as error:
        print(f"cannot evaluate: {error}", file=sys.stderr)
        return 4
    except LeakageError as error:
        print(f"LEAKAGE: {error}", file=sys.stderr)
        return 3

    print(metrics_markdown(payload))
    print(f"wrote {destination} and {destination.with_suffix('.md')}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
