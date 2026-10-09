"""Graded synthetic cases for the classifier and the rubric (spec 10.6, 11).

:mod:`eval.synthetic` generates feature tables with a known perturbation over a
known span, at one severity. Both spec 10.6 and spec 11 need the *ladder*: the
classifier's isotonic calibrator has nothing to fit if every training flaw has
the same ``s``, and spec 11 calibrates :math:`k_c` from single-flaw **L2** files
specifically. This module builds that graded corpus and runs the detector over
it, so an acceptance script stays a report rather than half a pipeline.

The mapping from a rung to :data:`~eval.synthetic.Flaw.severity` is one number:
``Flaw.severity == 1.0`` is documented there as roughly an L3, so a rung's
multiplier is ``s(level) / s(L3)`` with both values read from
``configs/injection.yaml``. Nothing else about the ladder is assumed.

The usual caveat applies and must be repeated wherever these numbers are: the
tables are generated, so they exercise spec 10.6 and 11 end to end but say
nothing about whether the feature extractors recover a flaw a *speaker* put into
audio. Only the real S1 injected files answer that, and they need MFA.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from eval.synthetic import Flaw, Recording, build_corpus, make_recording
from rostrum.detect.classifier import ClassifierCase, TruthFlaw
from rostrum.detect.refine import RefinementSettings, refine_regions
from rostrum.detect.regions import RegionThresholds, build_regions
from rostrum.detect.zscore import envelope_z, trigger_signals
from rostrum.envelope.build import Reference, TextEnvelope, build_fold_envelopes
from rostrum.score.rubric import CalibrationCase, deviation_mass, label_to_category

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.types import Region

logger = logging.getLogger(__name__)

__all__ = [
    "GRADED_LABELS",
    "LEVELS",
    "REFERENCE_LEVEL",
    "SPEAKER_OFFSET_COLUMNS",
    "Case",
    "FoldCases",
    "build_fold_cases",
    "denormalise_speaker",
    "graded_participants",
    "severity_multiplier",
]


LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4")
"""The injectable rungs; ``L0`` is the clean file, which has no flaw to grade."""

REFERENCE_LEVEL: str = "L3"
"""``eval.synthetic`` documents ``Flaw.severity == 1.0`` as roughly this rung."""

GRADED_LABELS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("rushing", {"first_word": 7, "last_word": 11}),
    ("dragging", {"first_word": 13, "last_word": 16}),
    ("monotone", {"first_word": 6, "last_word": 11}),
    ("low_energy", {"first_word": 12, "last_word": 17}),
    ("misplaced_stress", {"first_word": 4, "last_word": 5}),
    ("missing_pause", {"gap_idx": 5}),
    ("hesitation_pause", {"gap_idx": 13}),
    ("mumbling", {"first_word": 8, "last_word": 13}),
)
"""One span per label, held fixed across rungs so only the severity varies."""


def severity_multiplier(level: str, config: Config | None = None) -> float:
    """The :data:`~eval.synthetic.Flaw.severity` standing for one ladder rung.

    Args:
        level: ``"L1"`` to ``"L4"``.
        config: Supplies ``injection.severity``; without one the spec 5.3
            values are used.

    Returns:
        ``s(level) / s(L3)``, so ``L3`` is 1.0 by construction.
    """
    ladder: Mapping[str, float] = (
        config.injection["severity"]
        if config is not None
        else {"L0": 0.00, "L1": 0.15, "L2": 0.35, "L3": 0.60, "L4": 0.90}
    )
    return float(ladder[level]) / float(ladder[REFERENCE_LEVEL])


def _s_for(level: str, config: Config | None = None) -> float:
    """The ladder's ``s`` for one rung."""
    if config is None:
        return {"L0": 0.00, "L1": 0.15, "L2": 0.35, "L3": 0.60, "L4": 0.90}[level]
    return float(config.injection["severity"][level])


def graded_participants(
    *,
    labels: Sequence[tuple[str, dict[str, Any]]] = GRADED_LABELS,
    levels: Sequence[str] = LEVELS,
    config: Config | None = None,
) -> list[tuple[str, str, list[Flaw]]]:
    """One single-flaw participant per (label, rung), plus one clean file.

    Returns:
        ``[(suffix, level, [Flaw, ...]), ...]``; the clean file comes first and
        carries level ``"L0"`` and no flaw.
    """
    out: list[tuple[str, str, list[Flaw]]] = [("clean", "L0", [])]
    for label, span in labels:
        for level in levels:
            out.append(
                (
                    f"{label}_{level}",
                    level,
                    [Flaw(label=label, severity=severity_multiplier(level, config), **span)],
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Ablation 4: undoing speaker normalisation
# --------------------------------------------------------------------------- #


SPEAKER_OFFSET_COLUMNS: dict[str, tuple[str, ...]] = {
    "shift_st": ("f0_mean_st", "f0_max_st", "f0_min_st", "f0_st"),
    "shift_level": (
        "energy_mean_z",
        "energy_peak_z",
        "energy_mean_db_rel",
        "energy_z",
        "energy_db_rel",
    ),
    "scale_rate": ("rate_local_syll_per_s", "rate_relative"),
    "scale_duration": ("word_duration_s", "duration_per_syllable_s", "duration_rel_median"),
}
"""Which columns a speaker's habitual pitch, loudness and tempo move.

A *shift* applies to the pitch and level columns and an inverse *scale* to the
duration columns, which is what those quantities do between two speakers.
Spreads - ``f0_range_st``, ``f0_std_st``, every ``*_slope_*`` - are differences
within one recording and a habitual offset cancels in them, so they are absent
here on purpose.
"""


def denormalise_speaker(
    tables: Mapping[str, Any],
    frames: Any | None,
    *,
    speaker_id: str,
    seed: int,
) -> tuple[dict[str, Any], Any | None]:
    """Re-inject the between-speaker offsets speaker normalisation removes.

    Ablation 4 of spec 14.2 asks what the system scores **without** speaker
    normalisation. On the real corpus that means comparing raw Hz and raw dB;
    on this generated corpus the tables are already expressed re the speaker's
    own median (:mod:`eval.synthetic` emits ``*_st`` and ``*_z`` columns), so
    the ablation is run the other way round - each speaker is given back a
    habitual pitch, loudness and tempo of their own, which is exactly the
    variance CLAUDE.md invariants 5 and 6 exist to remove. Any number from this
    ablation must be labelled as the simulation it is.

    The offsets are drawn from ``seed`` and the speaker id, so one speaker gets
    the same voice in every fold and every run (CLAUDE.md invariant 13).

    Args:
        tables: ``{unit: FeatureTable}``; copied, never mutated.
        frames: The recording's frame curves, or ``None``.
        speaker_id: Whose habitual values these are.
        seed: Master seed.

    Returns:
        ``(tables, frames)`` with the offsets applied.
    """
    rng = np.random.default_rng(abs(hash((int(seed), str(speaker_id)))) % (2**32))
    shift_st = float(rng.uniform(-3.0, 3.0))
    shift_level = float(rng.uniform(-2.5, 2.5))
    rate_scale = float(rng.uniform(0.85, 1.15))

    def apply(table: Any) -> Any:
        out = table.copy()
        for column in SPEAKER_OFFSET_COLUMNS["shift_st"]:
            if column in out.columns:
                out[column] = out[column] + shift_st
        for column in SPEAKER_OFFSET_COLUMNS["shift_level"]:
            if column in out.columns:
                out[column] = out[column] + shift_level
        for column in SPEAKER_OFFSET_COLUMNS["scale_rate"]:
            if column in out.columns:
                out[column] = out[column] * rate_scale
        for column in SPEAKER_OFFSET_COLUMNS["scale_duration"]:
            if column in out.columns:
                out[column] = out[column] / rate_scale
        return out

    return (
        {unit: apply(table) for unit, table in tables.items()},
        None if frames is None else apply(frames),
    )


# --------------------------------------------------------------------------- #
# One detected recording
# --------------------------------------------------------------------------- #


@dataclass
class Case:
    """One detected recording with everything both acceptance reports need.

    Attributes:
        file_id: Recording id.
        text_id: Text id.
        level: Ladder rung of its flaw, ``"L0"`` for the clean file.
        label: The injected label, ``""`` for the clean file.
        category: Rubric category of that label.
        n_flaws: Flaws injected; the rubric calibrates on single-flaw files only.
        regions: What the detector found.
        z_by_unit: The comparisons behind them.
        truth: Ground-truth flaws with their severity.
        tables: The recording's own feature tables, for the descriptors.
        speech_duration: :math:`T_{speech}`: first word onset to last word
            offset, which is ``scoring.score.t_speech`` ("excluding leading and
            trailing silence").
        frames: The recording's frame curves. Boundary refinement needs them, so
            a consumer that re-detects this case at other thresholds - as
            :mod:`eval.run_loto` does once a fold is tuned - cannot work without
            them being carried here.
        duration: Wall duration of the recording in seconds, which is the
            denominator of the false-alarm rate of spec 14.3 (and differs from
            :attr:`speech_duration`, which excludes the outer silence).
        origin: ``clean`` | ``injected`` | ``human``. Human takes are test-only
            (spec 7.3), which the harness enforces on this field.
        same_channel: Whether the comparison behind :attr:`z_by_unit` was within
            one channel, which is the only case in which the channel-sensitive
            triggers may fire (CLAUDE.md invariant 7). Carried here because a
            consumer that re-detects the case - :mod:`eval.run_loto` does, once
            the fold is tuned - has to make the same decision again.
        tier: ``gold`` | ``silver`` | ``bronze``, when the corpus records one.
    """

    file_id: str
    text_id: str
    level: str
    label: str
    category: str
    n_flaws: int
    regions: list[Region]
    z_by_unit: Mapping[str, Any]
    truth: list[TruthFlaw]
    tables: Mapping[str, Any]
    speech_duration: float
    frames: Any | None = None
    duration: float = 0.0
    origin: str = "injected"
    same_channel: bool = True
    tier: str = ""

    def as_classifier_case(self) -> ClassifierCase:
        """The same recording as the classifier's input (spec 10.6)."""
        return ClassifierCase(
            file_id=self.file_id,
            text_id=self.text_id,
            regions=self.regions,
            z_by_unit=self.z_by_unit,
            truth=self.truth,
            origin=self.origin,
        )

    def as_calibration_case(self, *, config: Config | None = None) -> CalibrationCase:
        """The same recording as :func:`~rostrum.score.rubric.calibrate_k`'s input."""
        return CalibrationCase(
            file_id=self.file_id,
            text_id=self.text_id,
            category=self.category,
            level=self.level,
            mass=(
                deviation_mass(
                    self.regions, self.speech_duration, category=self.category, config=config
                )
                if self.category
                else 0.0
            ),
            n_flaws=self.n_flaws,
            origin=self.origin,
        )


def _truth_for(recording: Recording, level: str, config: Config | None) -> list[TruthFlaw]:
    """Ground truth with the severity attached, from the recording's own timeline."""
    words = recording.tables["word"]
    gaps = recording.tables["gap"]
    out: list[TruthFlaw] = []
    for flaw in recording.flaws:
        if flaw.gap_idx is not None:
            row = gaps.loc[int(flaw.gap_idx)]
            t_start, t_end = float(row["t_start"]), float(row["t_end"])
        else:
            t_start = float(words.loc[int(flaw.first_word), "t_start"])
            t_end = float(words.loc[int(flaw.last_word), "t_end"])
        out.append(
            TruthFlaw(
                label=flaw.event_label,
                t_start=t_start,
                t_end=t_end,
                s=_s_for(level, config),
                level=level,
                file_id=recording.file_id,
            )
        )
    return out


@dataclass
class FoldCases:
    """Every recording of one LOTO fold, detected against that fold's envelopes.

    Attributes:
        fold: Fold id.
        held_out: The held-out text.
        train_texts: The fold's training texts.
        envelopes: ``{text_id: TextEnvelope}``.
        cases: Detected recordings, training and held-out alike; the consumers
            select by ``text_id``.
        thresholds: The thresholds detection ran at.
        reference_tables: ``{text_id: [tables per envelope member]}``, for the
            descriptor range.
    """

    fold: str
    held_out: str
    train_texts: tuple[str, ...]
    envelopes: Mapping[str, TextEnvelope]
    cases: list[Case] = field(default_factory=list)
    thresholds: RegionThresholds | None = None
    reference_tables: Mapping[str, list[Mapping[str, Any]]] = field(default_factory=dict)

    def train(self) -> list[Case]:
        """Cases from the fold's training texts."""
        return [c for c in self.cases if c.text_id in set(self.train_texts)]

    def test(self) -> list[Case]:
        """Cases from the held-out text."""
        return [c for c in self.cases if c.text_id == self.held_out]


def build_fold_cases(
    *,
    config: Config,
    held_out: str,
    texts: Sequence[str] = ("syn_a", "syn_b", "syn_c"),
    n_readers: int = 6,
    n_words: int = 24,
    refine: bool = True,
    thresholds: RegionThresholds | None = None,
    envelope_size: int | None = None,
    speaker_offsets: bool = False,
) -> FoldCases:
    """Build, detect and label one fold's graded corpus.

    The envelopes come from :func:`rostrum.envelope.build.build_fold_envelopes`,
    whose ``sigma_floor`` is fitted on ``train_texts`` only, so the leakage guard
    of CLAUDE.md invariant 10 holds here as it does on the real corpus.

    Args:
        config: Loaded config.
        held_out: The fold's test text.
        texts: Every text in the corpus.
        n_readers: Clean readers per text; they form the envelope.
        n_words: Words per text.
        refine: Run ruptures boundary refinement.
        thresholds: Detection thresholds; the config's defaults when ``None``.
        envelope_size: Keep only this many readers per text in the envelope.
            ``1`` is ablation 1 of spec 14.2, "single reference vs envelope";
            it needs ``detection.envelope.min_references`` overridden to 1 or
            the envelope refuses to build, which is the guard working.
        speaker_offsets: Give every speaker a habitual pitch, loudness and tempo
            of their own before comparing, which is ablation 4 - see
            :func:`denormalise_speaker`.

    Returns:
        A :class:`FoldCases`.

    Raises:
        rostrum.envelope.build.EnvelopeError: ``envelope_size`` is below
            ``detection.envelope.min_references``.
    """
    references, _ = build_corpus(
        texts=texts, n_readers=n_readers, n_words=n_words, seed=config.seed
    )
    train_texts = tuple(t for t in texts if t != held_out)
    fold = f"loto_{held_out}"

    if envelope_size is not None:
        references = {
            text: list(members)[: int(envelope_size)] for text, members in references.items()
        }

    reference_frames: dict[str, Any] = {}
    reference_data: dict[str, Mapping[str, Any]] = {}
    for text in texts:
        for r in references[text]:
            tables, frames = (
                denormalise_speaker(r.tables, r.frames, speaker_id=r.file_id, seed=config.seed)
                if speaker_offsets
                else (dict(r.tables), r.frames)
            )
            reference_data[r.file_id] = tables
            reference_frames[r.file_id] = frames

    refs = [
        Reference(file_id=r.file_id, channel_id=r.channel_id, tables=reference_data[r.file_id])
        for text in texts
        for r in references[text]
    ]
    envelopes = build_fold_envelopes(
        {text: [r.file_id for r in references[text]] for text in texts},
        refs,
        fold=fold,
        train_texts=train_texts,
        config=config,
        frames=reference_frames,
        word_spans={r.file_id: r.word_spans for text in texts for r in references[text]},
    )

    if thresholds is None:
        thresholds = RegionThresholds.from_config(config)
    settings = RefinementSettings.from_config(config) if refine else None
    mapping = label_to_category(config)
    plan = graded_participants(config=config)

    out = FoldCases(
        fold=fold,
        held_out=held_out,
        train_texts=train_texts,
        envelopes=envelopes,
        thresholds=thresholds,
        reference_tables={
            text: [reference_data[r.file_id] for r in references[text]] for text in texts
        },
    )

    for text in texts:
        for index, (suffix, level, flaws) in enumerate(plan):
            recording = make_recording(
                f"{text}_p{index:02d}_{suffix}",
                text,
                n_words=n_words,
                seed=config.seed + 1000,
                # One reader per text, shared by every participant of it. Spec
                # 5.4 builds the single-flaw files by *editing one clean file*,
                # so the L1-L4 ladder must be the same underlying reading with a
                # graded edit; a new reader per rung would mix reader variation
                # into what is meant to be a severity comparison.
                reader_id=f"{text}_participant",
                flaws=flaws,
                with_channel_features=True,
            )
            participant_tables, participant_frames = (
                denormalise_speaker(
                    recording.tables,
                    recording.frames,
                    speaker_id=f"{text}_participant",
                    seed=config.seed,
                )
                if speaker_offsets
                else (dict(recording.tables), recording.frames)
            )
            z_by_unit = envelope_z(
                participant_tables,
                envelopes[text],
                config=config,
                channel_matched=True,
                file_id=recording.file_id,
            )
            signals, skipped = trigger_signals(z_by_unit, config=config, same_channel=True)
            regions = build_regions(signals, z_by_unit, config=config, thresholds=thresholds)
            if settings is not None:
                regions = refine_regions(regions, participant_frames, settings=settings)
            if skipped:
                logger.debug(
                    "%s: skipped %s",
                    recording.file_id,
                    [f"{s.label}:{s.reason}" for s in skipped],
                )
            words = participant_tables["word"]
            label = flaws[0].event_label if flaws else ""
            out.cases.append(
                Case(
                    file_id=recording.file_id,
                    text_id=text,
                    level=level,
                    label=label,
                    category=mapping.get(label, ""),
                    n_flaws=len(flaws),
                    regions=regions,
                    z_by_unit=z_by_unit,
                    truth=_truth_for(recording, level, config),
                    tables=participant_tables,
                    speech_duration=float(words["t_end"].max()) - float(words["t_start"].min()),
                    frames=participant_frames,
                    duration=float(words["t_end"].max()),
                )
            )

    logger.info(
        "fold %s: %d case(s) over %d text(s), held out %s",
        fold,
        len(out.cases),
        len(texts),
        held_out,
    )
    return out
