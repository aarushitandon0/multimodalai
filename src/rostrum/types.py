"""Core data model for Rostrum.

Every dataclass here round-trips losslessly through :meth:`to_dict` /
:meth:`from_dict` so that an analysis can be written to JSON, re-read, and
hashed for the determinism test.

Two project invariants are enforced in this module rather than left to callers
(see ``CLAUDE.md``):

* **Time** is in seconds, float, measured from the start of the *processed*
  WAV, and rounded to :data:`TIME_DECIMALS` places when serialised.
* **Values** are rounded to :data:`VALUE_SIGFIGS` significant figures when
  serialised, so the SHA-256 of a report does not depend on float noise in the
  last bits, or on how large the quantity happens to be (spec 16).

Rounding happens on the way *out* (``to_dict``) and never in the computation
path, so no module loses precision by using these types internally.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias

__all__ = [
    "FEATURE_TABLE_CONTRACT",
    "TIME_DECIMALS",
    "VALUE_SIGFIGS",
    "Alignment",
    "AnalysisResult",
    "CategoryScore",
    "Envelope",
    "Evidence",
    "FeatureTable",
    "FlawLabel",
    "Gap",
    "GapLabel",
    "OperatingMode",
    "Origin",
    "Region",
    "RubricCategory",
    "RubricScore",
    "SeverityLevel",
    "Tier",
    "Word",
    "round_sigfig",
    "round_time",
    "round_value",
]

# --------------------------------------------------------------------------- #
# Serialisation constants (CLAUDE.md invariants 2 and 13)
# --------------------------------------------------------------------------- #

TIME_DECIMALS: int = 3
"""Timestamps are serialised to millisecond resolution."""

VALUE_SIGFIGS: int = 4
"""Feature values, z-scores and scores are serialised to 4 significant figures.

Significant figures rather than decimal places because the quantities differ by
orders of magnitude - a gap of 0.06 s and a rate of 6.4 syll/s need the same
*relative* precision for the determinism hash to be stable across machines
(spec 16).
"""


# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #

FlawLabel = Literal[
    "rushing",
    "dragging",
    "missing_pause",
    "hesitation_pause",
    "monotone",
    "misplaced_stress",
    "low_energy",
    "mumbling",
]
"""The eight flaw labels. ``mumbling`` is human-recorded only."""

RubricCategory = Literal["pace", "pausing", "intonation", "emphasis", "energy", "clarity"]
"""The six rubric categories scored 0-10."""

SeverityLevel = Literal["L0", "L1", "L2", "L3", "L4"]
"""Severity ladder; see ``configs/injection.yaml`` for the s value of each."""

GapLabel = Literal["boundary", "minor", "none"]
"""Text-derived expectation for an inter-word gap (plan 7.3)."""

OperatingMode = Literal["reference", "norm", "reject"]
"""How the analysis was produced (plan 8.5)."""

Origin = Literal["clean", "injected", "human", "participant"]
"""Provenance of an audio file (spec 7.1: ``injected | human | clean``).

``participant`` is the runtime addition: audio uploaded for analysis, which has
no dataset provenance at all.
"""

Tier = Literal["gold", "silver", "bronze"]
"""Sourcing tier: original orator recording, skilled public-domain reading, or
team recording (spec 4)."""


# --------------------------------------------------------------------------- #
# Rounding helpers
# --------------------------------------------------------------------------- #


def round_time(t: float | None) -> float | None:
    """Round a timestamp to :data:`TIME_DECIMALS`, passing ``None`` through."""
    if t is None:
        return None
    return round(float(t), TIME_DECIMALS)


def round_sigfig(x: float, sigfigs: int = VALUE_SIGFIGS) -> float | None:
    """Round ``x`` to ``sigfigs`` significant figures.

    Zero has no significant figures to speak of and is returned unchanged.
    Non-finite values become ``None`` - JSON has no ``NaN`` literal.
    """
    if not math.isfinite(x):
        return None
    if x == 0.0:
        return 0.0
    # Scale so the first significant digit sits just left of the point, round,
    # then scale back. Decimal would be exact but is far slower, and the result
    # is re-rounded identically on every machine either way.
    exponent = math.floor(math.log10(abs(x)))
    return round(x, -(exponent - (sigfigs - 1)))


def round_value(x: Any, sigfigs: int = VALUE_SIGFIGS) -> Any:
    """Round floats (recursively inside lists/dicts) for stable serialisation.

    Floats are rounded to ``sigfigs`` significant figures. Non-finite floats
    become ``None``: JSON has no ``NaN``, and silently emitting the
    non-standard literal would break round-tripping.
    """
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, float):
        return round_sigfig(x, sigfigs)
    if isinstance(x, int):
        return x
    if isinstance(x, Mapping):
        return {k: round_value(v, sigfigs) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [round_value(v, sigfigs) for v in x]
    return x


# --------------------------------------------------------------------------- #
# Feature tables
# --------------------------------------------------------------------------- #

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotation
    import pandas as pd

    FeatureTable: TypeAlias = pd.DataFrame
else:  # pandas is imported lazily so `import rostrum` stays cheap.
    FeatureTable = "pandas.DataFrame"

FEATURE_TABLE_CONTRACT = """A per-unit feature table (spec 17).

A ``pandas.DataFrame`` indexed by unit, with one column per feature. It is an
alias rather than a dataclass because every consumer wants vectorised column
maths. Conventions each producer must honour:

* the index is the word, gap or phrase index, named ``unit_idx``;
* ``df.attrs["unit"]`` records which of the three it is;
* ``df.attrs["file_id"]`` and ``df.attrs["channel_id"]`` travel with the table,
  so the channel rule can be enforced at comparison time;
* column names match the feature names in ``configs/features.yaml``.
"""


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@dataclass
class Word:
    """One word interval from the forced alignment.

    Attributes:
        idx: Position in the utterance, zero-based. Because participant and
            reference share a transcript, ``idx`` is the comparison key.
        text: Orthographic form as aligned (lowercased, verbatim).
        t_start: Onset in seconds from the start of the processed WAV.
        t_end: Offset in seconds.
        phones: MFA phone labels inside this word, in order.
        n_syllables: Vowel-nucleus count, used for rate and duration norms.
        display: Form from the display transcript, with original case and any
            trailing punctuation; drives the gap labels and the dashboard.
    """

    idx: int
    text: str
    t_start: float
    t_end: float
    phones: list[str] = field(default_factory=list)
    n_syllables: int = 0
    display: str = ""

    @property
    def duration(self) -> float:
        """Word duration in seconds."""
        return self.t_end - self.t_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": int(self.idx),
            "text": self.text,
            "t_start": round_time(self.t_start),
            "t_end": round_time(self.t_end),
            "phones": list(self.phones),
            "n_syllables": int(self.n_syllables),
            "display": self.display,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Word:
        return cls(
            idx=int(data["idx"]),
            text=str(data["text"]),
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            phones=list(data.get("phones", [])),
            n_syllables=int(data.get("n_syllables", 0)),
            display=str(data.get("display", "")),
        )


@dataclass
class Gap:
    """A silence between two consecutive words.

    ``label`` is the *expectation* derived from the display transcript, not a
    judgement: a long ``boundary`` gap is good rhetoric, while the same gap
    with label ``none`` is a hesitation (plan 7.3).

    Attributes:
        idx: Position in the gap sequence; gap ``i`` follows word ``i``.
        prev_word_idx: Index of the word before the gap.
        next_word_idx: Index of the word after the gap.
        t_start: Onset in seconds (offset of the previous word).
        t_end: Offset in seconds (onset of the next word).
        label: ``boundary``, ``minor`` or ``none``.
        punctuation: The punctuation that produced the label, if any.
        vad_refined: Whether Silero VAD adjusted these boundaries.
    """

    idx: int
    prev_word_idx: int
    next_word_idx: int
    t_start: float
    t_end: float
    label: GapLabel = "none"
    punctuation: str = ""
    vad_refined: bool = False

    @property
    def duration(self) -> float:
        """Gap duration in seconds."""
        return self.t_end - self.t_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "idx": int(self.idx),
            "prev_word_idx": int(self.prev_word_idx),
            "next_word_idx": int(self.next_word_idx),
            "t_start": round_time(self.t_start),
            "t_end": round_time(self.t_end),
            "label": self.label,
            "punctuation": self.punctuation,
            "vad_refined": bool(self.vad_refined),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Gap:
        return cls(
            idx=int(data["idx"]),
            prev_word_idx=int(data["prev_word_idx"]),
            next_word_idx=int(data["next_word_idx"]),
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            label=data.get("label", "none"),
            punctuation=str(data.get("punctuation", "")),
            vad_refined=bool(data.get("vad_refined", False)),
        )


@dataclass
class Alignment:
    """Forced alignment of one recording: words, gaps and provenance.

    Attributes:
        file_id: Stable identifier, also the audio basename.
        text_id: Which library text this is a rendition of; ``None`` for an
            unknown text (norm mode).
        speaker_id: Speaker identifier, used for speaker normalisation.
        channel_id: Recording channel. Channel-sensitive features may only be
            compared within one ``channel_id`` (CLAUDE.md invariant 7).
        tier: Sourcing tier of this recording, or ``None`` for uploaded audio.
        words: Word intervals, ordered by ``idx``.
        gaps: Inter-word gaps, ordered by ``idx``.
        duration: Length of the processed WAV in seconds.
        sample_rate: Sample rate of the processed WAV; always 16000.
        aligner: Tool and version string, e.g. ``"mfa 3.2.1"``.
        acoustic_model: MFA acoustic model name.
        dictionary: MFA dictionary name.
        oov_words: Words the dictionary could not pronounce.
        audio_sha256: Hash of the processed WAV these timings refer to.
    """

    file_id: str
    text_id: str | None = None
    speaker_id: str = ""
    channel_id: str = ""
    tier: Tier | None = None
    words: list[Word] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    duration: float = 0.0
    sample_rate: int = 16000
    aligner: str = ""
    acoustic_model: str = ""
    dictionary: str = ""
    oov_words: list[str] = field(default_factory=list)
    audio_sha256: str = ""

    @property
    def n_words(self) -> int:
        """Number of aligned words."""
        return len(self.words)

    @property
    def speech_duration(self) -> float:
        """Span from the first word onset to the last word offset, in seconds.

        This is the ``T_speech`` of the scoring formula (plan 8.2): it excludes
        leading and trailing silence, so a long run-in cannot dilute a score.
        """
        if not self.words:
            return 0.0
        return self.words[-1].t_end - self.words[0].t_start

    def word(self, idx: int) -> Word:
        """Return the word with index ``idx``."""
        return self.words[idx]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "text_id": self.text_id,
            "speaker_id": self.speaker_id,
            "channel_id": self.channel_id,
            "tier": self.tier,
            "words": [w.to_dict() for w in self.words],
            "gaps": [g.to_dict() for g in self.gaps],
            "duration": round_time(self.duration),
            "sample_rate": int(self.sample_rate),
            "aligner": self.aligner,
            "acoustic_model": self.acoustic_model,
            "dictionary": self.dictionary,
            "oov_words": list(self.oov_words),
            "audio_sha256": self.audio_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Alignment:
        return cls(
            file_id=str(data["file_id"]),
            text_id=data.get("text_id"),
            speaker_id=str(data.get("speaker_id", "")),
            channel_id=str(data.get("channel_id", "")),
            tier=data.get("tier"),
            words=[Word.from_dict(w) for w in data.get("words", [])],
            gaps=[Gap.from_dict(g) for g in data.get("gaps", [])],
            duration=float(data.get("duration", 0.0)),
            sample_rate=int(data.get("sample_rate", 16000)),
            aligner=str(data.get("aligner", "")),
            acoustic_model=str(data.get("acoustic_model", "")),
            dictionary=str(data.get("dictionary", "")),
            oov_words=list(data.get("oov_words", [])),
            audio_sha256=str(data.get("audio_sha256", "")),
        )


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


@dataclass
class Envelope:
    """Acceptable-variance statistics over skilled readings of one text.

    For each unit (word index or gap index) and feature ``k`` we store a robust
    centre and spread across references ``r``:

    .. math::

        \\mu_{wk} = \\mathrm{median}_r\\, x^{(r)}_{wk}, \\qquad
        \\sigma_{wk} = 1.4826 \\cdot \\mathrm{MAD}_r\\, x^{(r)}_{wk}

    and compare a participant with
    :math:`z_{wk} = (x^{(p)}_{wk} - \\mu_{wk}) / \\max(\\sigma_{wk},
    \\sigma^{\\mathrm{floor}}_k)`.

    ``sigma_floor`` is fitted on **training texts only** so no information from
    a held-out text reaches the thresholds (CLAUDE.md invariants 10-11).

    Attributes:
        text_id: The text these statistics describe.
        fold: Leave-one-text-out fold name, or ``"all"`` outside of CV.
        unit: ``"word"``, ``"gap"`` or ``"phrase"``.
        features: Feature names, in the column order used by ``median``/``sigma``.
        median: ``{unit_idx: [median per feature]}``.
        sigma: ``{unit_idx: [1.4826*MAD per feature]}``, before flooring.
        sigma_floor: ``{feature: floor}``, the 25th percentile of sigma on the
            training texts.
        n_references: Number of references contributing to each unit.
        reference_ids: ``file_id`` of every contributing reference, sorted, so a
            leave-one-reference-out confidence band is reproducible.
        channel_ids: Channel of each reference, parallel to ``reference_ids``.
    """

    text_id: str
    fold: str = "all"
    unit: Literal["word", "gap", "phrase"] = "word"
    features: list[str] = field(default_factory=list)
    median: dict[int, list[float]] = field(default_factory=dict)
    sigma: dict[int, list[float]] = field(default_factory=dict)
    sigma_floor: dict[str, float] = field(default_factory=dict)
    n_references: dict[int, int] = field(default_factory=dict)
    reference_ids: list[str] = field(default_factory=list)
    channel_ids: list[str] = field(default_factory=list)

    MAD_SCALE: ClassVar[float] = 1.4826
    """Consistency constant making 1.4826*MAD a sigma estimate for normal data."""

    def effective_sigma(self, unit_idx: int, feature: str) -> float:
        """Return ``max(sigma, sigma_floor)`` for one unit and feature."""
        col = self.features.index(feature)
        return max(self.sigma[unit_idx][col], self.sigma_floor.get(feature, 0.0))

    def z(self, unit_idx: int, feature: str, value: float) -> float:
        """Standardise ``value`` against this envelope."""
        col = self.features.index(feature)
        return (value - self.median[unit_idx][col]) / self.effective_sigma(unit_idx, feature)

    def to_dict(self) -> dict[str, Any]:
        # JSON object keys must be strings; from_dict casts them back to int.
        return {
            "text_id": self.text_id,
            "fold": self.fold,
            "unit": self.unit,
            "features": list(self.features),
            "median": {str(k): round_value(v) for k, v in self.median.items()},
            "sigma": {str(k): round_value(v) for k, v in self.sigma.items()},
            "sigma_floor": round_value(dict(self.sigma_floor)),
            "n_references": {str(k): int(v) for k, v in self.n_references.items()},
            "reference_ids": list(self.reference_ids),
            "channel_ids": list(self.channel_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Envelope:
        return cls(
            text_id=str(data["text_id"]),
            fold=str(data.get("fold", "all")),
            unit=data.get("unit", "word"),
            features=list(data.get("features", [])),
            median={int(k): [float(x) for x in v] for k, v in data.get("median", {}).items()},
            sigma={int(k): [float(x) for x in v] for k, v in data.get("sigma", {}).items()},
            sigma_floor={str(k): float(v) for k, v in data.get("sigma_floor", {}).items()},
            n_references={int(k): int(v) for k, v in data.get("n_references", {}).items()},
            reference_ids=list(data.get("reference_ids", [])),
            channel_ids=list(data.get("channel_ids", [])),
        )


# --------------------------------------------------------------------------- #
# Regions and explanations
# --------------------------------------------------------------------------- #


@dataclass
class Evidence:
    """One measured feature behind a flagged region.

        This is the row that makes an explanation causal rather than assertive: the
        participant value, what skilled readers did, and the standardised distance
        between them.

    Field names follow the explanation schema of spec 12 so a card can be
        serialised without a translation layer.

        Attributes:
            feature: Feature name, e.g. ``"rate_syll_s"``.
            value: The participant's value, in the feature's own units.
            env_median: Robust centre across references.
            env_sigma: Effective spread (already floored).
            z: Signed standardised deviation.
            unit: Human-readable unit, e.g. ``"syll/s"``, ``"s"``, ``"st"``, ``"dB"``.
            gap_label: For gap features, the text expectation at that gap.
            n_references: How many references the statistics came from.
            channel_matched: True when this comparison was within one channel;
                channel-sensitive features are only reported when this holds.
    """

    feature: str
    value: float
    env_median: float
    env_sigma: float
    z: float
    unit: str = ""
    gap_label: GapLabel | None = None
    n_references: int = 0
    channel_matched: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "value": round_value(self.value),
            "env_median": round_value(self.env_median),
            "env_sigma": round_value(self.env_sigma),
            "z": round_value(self.z),
            "unit": self.unit,
            "gap_label": self.gap_label,
            "n_references": int(self.n_references),
            "channel_matched": bool(self.channel_matched),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Evidence:
        return cls(
            feature=str(data["feature"]),
            value=float(data["value"]),
            env_median=float(data["env_median"]),
            env_sigma=float(data["env_sigma"]),
            z=float(data["z"]),
            unit=str(data.get("unit", "")),
            gap_label=data.get("gap_label"),
            n_references=int(data.get("n_references", 0)),
            channel_matched=bool(data.get("channel_matched", True)),
        )


@dataclass
class Region:
    """A temporally grounded flaw: where it is, what it is, why we say so.

    Attributes:
        region_id: Index within the analysis, stable across runs.
        t_start: Flaw onset in seconds.
        t_end: Flaw offset in seconds.
        label: Flaw label.
        category: Rubric category ``label`` belongs to.
        level: Severity level predicted for this region (spec 12 names this
            field ``level``, matching the dataset label JSON of spec 7.1).
        s: Continuous severity in [0, 1], from the isotonic calibrator.
        confidence: Confidence in this region overall.
        rule_label: The label the deterministic rule produced.
        clf_prob: The classifier's probability for its own label. When rule and
            classifier disagree the card shows ``rule_label`` and reports
            ``clf_prob`` beside it (spec 10.6) - the rule always wins, because
            it is the part that is explainable.
        clf_label: The classifier's label, kept so a disagreement is visible
            rather than silently discarded.
        envelope_n: How many references the comparison used.
        word_idx: ``[first, last]`` word indices covered, inclusive.
        gap_idx: Gap index, for the two pause labels.
        words: The spoken text of the region, for the explanation card.
        evidence: Measured rows justifying the region.
        context: Text-derived context, e.g. ``"anaphora"``.
        rationale: Rendered sentence stating the mathematical delta.
        fix: Rendered actionable advice.
        peak_z: Largest ``|z|`` inside the region, on the primary feature.
        mean_excess_z: Mean of ``(|z| - tau_out)_+``; feeds the deviation mass.
        refined: Whether ruptures moved the boundaries.
        t_start_unrefined: Pre-refinement onset, kept for the ablation.
        t_end_unrefined: Pre-refinement offset, kept for the ablation.
    """

    region_id: int
    t_start: float
    t_end: float
    label: FlawLabel
    category: RubricCategory
    level: SeverityLevel = "L0"
    s: float = 0.0
    confidence: float = 0.0
    rule_label: FlawLabel | None = None
    clf_label: FlawLabel | None = None
    clf_prob: float | None = None
    envelope_n: int = 0
    word_idx: list[int] = field(default_factory=list)
    gap_idx: int | None = None
    words: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    context: str = ""
    rationale: str = ""
    fix: str = ""
    peak_z: float = 0.0
    mean_excess_z: float = 0.0
    refined: bool = False
    t_start_unrefined: float | None = None
    t_end_unrefined: float | None = None

    @property
    def duration(self) -> float:
        """Region duration in seconds."""
        return self.t_end - self.t_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": int(self.region_id),
            "t_start": round_time(self.t_start),
            "t_end": round_time(self.t_end),
            "label": self.label,
            "category": self.category,
            "level": self.level,
            "s": round_value(self.s),
            "confidence": round_value(self.confidence),
            "rule_label": self.rule_label,
            "clf_label": self.clf_label,
            "clf_prob": round_value(self.clf_prob),
            "envelope_n": int(self.envelope_n),
            "word_idx": [int(i) for i in self.word_idx],
            "gap_idx": None if self.gap_idx is None else int(self.gap_idx),
            "words": self.words,
            "evidence": [e.to_dict() for e in self.evidence],
            "context": self.context,
            "rationale": self.rationale,
            "fix": self.fix,
            "peak_z": round_value(self.peak_z),
            "mean_excess_z": round_value(self.mean_excess_z),
            "refined": bool(self.refined),
            "t_start_unrefined": round_time(self.t_start_unrefined),
            "t_end_unrefined": round_time(self.t_end_unrefined),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Region:
        return cls(
            region_id=int(data["region_id"]),
            t_start=float(data["t_start"]),
            t_end=float(data["t_end"]),
            label=data["label"],
            category=data["category"],
            level=data.get("level", "L0"),
            s=float(data.get("s", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            rule_label=data.get("rule_label"),
            clf_label=data.get("clf_label"),
            clf_prob=None if data.get("clf_prob") is None else float(data["clf_prob"]),
            envelope_n=int(data.get("envelope_n", 0)),
            word_idx=[int(i) for i in data.get("word_idx", [])],
            gap_idx=None if data.get("gap_idx") is None else int(data["gap_idx"]),
            words=str(data.get("words", "")),
            evidence=[Evidence.from_dict(e) for e in data.get("evidence", [])],
            context=str(data.get("context", "")),
            rationale=str(data.get("rationale", "")),
            fix=str(data.get("fix", "")),
            peak_z=float(data.get("peak_z", 0.0)),
            mean_excess_z=float(data.get("mean_excess_z", 0.0)),
            refined=bool(data.get("refined", False)),
            t_start_unrefined=(
                None if data.get("t_start_unrefined") is None else float(data["t_start_unrefined"])
            ),
            t_end_unrefined=(
                None if data.get("t_end_unrefined") is None else float(data["t_end_unrefined"])
            ),
        )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass
class CategoryScore:
    """Score for one rubric category.

    ``score = 10 * exp(-k * deviation_mass)`` (plan 8.2). ``band`` is the
    standard deviation across leave-one-reference-out recomputations, so the
    dashboard can show ``7.2 +/- 0.4`` without pretending to more precision
    than the envelope supports.

    Attributes:
        category: Which of the six categories.
        score: 0-10, higher is better.
        band: Confidence half-width (std across leave-one-reference-out runs).
        deviation_mass: ``M_c`` from the scoring formula.
        k: The ``k_c`` used, as calibrated on training texts.
        n_regions: Regions in this category.
        suppressed: True when the category could not be scored (e.g. clarity
            with no same-channel reference). Suppressed categories are excluded
            from the overall mean rather than scored 10.
        suppressed_reason: Why, for the UI badge.
    """

    category: RubricCategory
    score: float
    band: float = 0.0
    deviation_mass: float = 0.0
    k: float = 1.0
    n_regions: int = 0
    suppressed: bool = False
    suppressed_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "score": round_value(self.score),
            "band": round_value(self.band),
            "deviation_mass": round_value(self.deviation_mass),
            "k": round_value(self.k),
            "n_regions": int(self.n_regions),
            "suppressed": bool(self.suppressed),
            "suppressed_reason": self.suppressed_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CategoryScore:
        return cls(
            category=data["category"],
            score=float(data["score"]),
            band=float(data.get("band", 0.0)),
            deviation_mass=float(data.get("deviation_mass", 0.0)),
            k=float(data.get("k", 1.0)),
            n_regions=int(data.get("n_regions", 0)),
            suppressed=bool(data.get("suppressed", False)),
            suppressed_reason=str(data.get("suppressed_reason", "")),
        )


@dataclass
class RubricScore:
    """The six category scores, the overall mean, and raw descriptors.

    Attributes:
        categories: One :class:`CategoryScore` per rubric category.
        overall: Unweighted mean of the non-suppressed category scores.
        overall_band: Confidence half-width on ``overall``.
        descriptors: Raw numbers shown beside the score (articulation rate,
            pitch variation, mean length of runs, pause rate), each optionally
            with its envelope range (plan 8.3).
    """

    categories: list[CategoryScore] = field(default_factory=list)
    overall: float = 0.0
    overall_band: float = 0.0
    descriptors: dict[str, Any] = field(default_factory=dict)

    def get(self, category: RubricCategory) -> CategoryScore | None:
        """Return the score for ``category``, or ``None`` if absent."""
        for c in self.categories:
            if c.category == category:
                return c
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": [c.to_dict() for c in self.categories],
            "overall": round_value(self.overall),
            "overall_band": round_value(self.overall_band),
            "descriptors": round_value(self.descriptors),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RubricScore:
        return cls(
            categories=[CategoryScore.from_dict(c) for c in data.get("categories", [])],
            overall=float(data.get("overall", 0.0)),
            overall_band=float(data.get("overall_band", 0.0)),
            descriptors=dict(data.get("descriptors", {})),
        )


# --------------------------------------------------------------------------- #
# Top-level result
# --------------------------------------------------------------------------- #


@dataclass
class AnalysisResult:
    """Everything ``rostrum.pipeline.analyze`` returns for one recording.

    This object is the contract between the pipeline, the API, the dashboard
    and the evaluation harness. Its serialised form is what the determinism
    test hashes, so nothing unstable (timestamps of the run, absolute paths,
    dict iteration order) may be added to it.

    Attributes:
        file_id: Identifier for the analysed recording.
        text_id: Library text matched, or ``None`` in norm/reject mode.
        speaker_id: Speaker identifier.
        channel_id: Channel identifier.
        origin: Provenance of the audio.
        tier: Sourcing tier, or ``None`` for uploaded audio.
        mode: ``reference``, ``norm`` or ``reject`` (plan 8.5).
        mode_reason: Why that mode was chosen, e.g. ``"wer=0.41 > 0.25"``.
        alignment: The forced alignment used.
        regions: Grounded flaw regions, ordered by ``t_start``.
        score: The rubric score, or ``None`` in reject mode.
        duration: Length of the processed audio in seconds.
        speech_duration: ``T_speech`` used in the scoring formula.
        wer: ASR-vs-transcript word error rate from the mismatch check.
        n_references: Envelope size used for the comparison.
        reference_ids: The envelope members, sorted.
        features_version: ``configs/features.yaml`` cache version.
        config_hash: SHA-256 over the resolved configs, so a report states
            exactly which parameters produced it.
        code_version: Git describe / commit of the code that ran.
        warnings: Non-fatal issues (OOV words, pitch cross-check failures,
            suppressed categories) surfaced in the UI.
        analysis_id: ``sha256(audio bytes + transcript + config hash)``; the
            cache key of this analysis and the handle ``GET /clip`` takes.
        cards: Serialised explanation cards, one per region, in region order
            (:mod:`rostrum.explain.cards`). Kept beside ``regions`` because
            the card carries prose the region does not: its context sentence.
        curves: Participant frame curves for the overlay (spec 15.2), already
            warped onto the reference timeline where there is one.
        envelope_curves: The skilled readers' median and sigma per word, the
            band drawn behind ``curves``.
    """

    file_id: str
    text_id: str | None = None
    speaker_id: str = ""
    channel_id: str = ""
    origin: Origin = "participant"
    tier: Tier | None = None
    mode: OperatingMode = "reference"
    mode_reason: str = ""
    alignment: Alignment | None = None
    regions: list[Region] = field(default_factory=list)
    score: RubricScore | None = None
    duration: float = 0.0
    speech_duration: float = 0.0
    wer: float | None = None
    n_references: int = 0
    reference_ids: list[str] = field(default_factory=list)
    features_version: int = 1
    config_hash: str = ""
    code_version: str = ""
    warnings: list[str] = field(default_factory=list)
    analysis_id: str = ""
    cards: list[dict[str, Any]] = field(default_factory=list)
    curves: dict[str, Any] = field(default_factory=dict)
    envelope_curves: dict[str, Any] = field(default_factory=dict)

    SCHEMA_VERSION: ClassVar[int] = 1
    """Bumped when the serialised shape changes incompatibly."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "file_id": self.file_id,
            "text_id": self.text_id,
            "speaker_id": self.speaker_id,
            "channel_id": self.channel_id,
            "origin": self.origin,
            "tier": self.tier,
            "mode": self.mode,
            "mode_reason": self.mode_reason,
            "alignment": None if self.alignment is None else self.alignment.to_dict(),
            # Sorted so two runs that discover regions in a different order
            # still serialise - and therefore hash - identically.
            "regions": [r.to_dict() for r in _sorted_regions(self.regions)],
            "score": None if self.score is None else self.score.to_dict(),
            "duration": round_time(self.duration),
            "speech_duration": round_time(self.speech_duration),
            "wer": round_value(self.wer),
            "n_references": int(self.n_references),
            "reference_ids": list(self.reference_ids),
            "features_version": int(self.features_version),
            "config_hash": self.config_hash,
            "code_version": self.code_version,
            "warnings": list(self.warnings),
            "analysis_id": self.analysis_id,
            "cards": [dict(c) for c in self.cards],
            # Curves are rounded like every other value, so the determinism
            # hash does not depend on float noise in a plotted point.
            "curves": round_value(self.curves),
            "envelope_curves": round_value(self.envelope_curves),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnalysisResult:
        return cls(
            file_id=str(data["file_id"]),
            text_id=data.get("text_id"),
            speaker_id=str(data.get("speaker_id", "")),
            channel_id=str(data.get("channel_id", "")),
            origin=data.get("origin", "participant"),
            tier=data.get("tier"),
            mode=data.get("mode", "reference"),
            mode_reason=str(data.get("mode_reason", "")),
            alignment=(
                None if data.get("alignment") is None else Alignment.from_dict(data["alignment"])
            ),
            regions=[Region.from_dict(r) for r in data.get("regions", [])],
            score=None if data.get("score") is None else RubricScore.from_dict(data["score"]),
            duration=float(data.get("duration", 0.0)),
            speech_duration=float(data.get("speech_duration", 0.0)),
            wer=None if data.get("wer") is None else float(data["wer"]),
            n_references=int(data.get("n_references", 0)),
            reference_ids=list(data.get("reference_ids", [])),
            features_version=int(data.get("features_version", 1)),
            config_hash=str(data.get("config_hash", "")),
            code_version=str(data.get("code_version", "")),
            warnings=list(data.get("warnings", [])),
            analysis_id=str(data.get("analysis_id", "")),
            cards=[dict(c) for c in data.get("cards", [])],
            curves=dict(data.get("curves", {})),
            envelope_curves=dict(data.get("envelope_curves", {})),
        )

    def to_events(self) -> list[tuple[str, float, float, str]]:
        """Return ``(file_id, onset, offset, label)`` rows for ``sed_eval``.

        The evaluation harness writes these to ``events.tsv``; keeping the
        conversion next to the data stops the harness and the dataset builder
        from drifting apart.
        """
        return [
            (self.file_id, round_time(r.t_start), round_time(r.t_end), r.label)
            for r in self.regions
        ]


def _sorted_regions(regions: Sequence[Region]) -> list[Region]:
    """Order regions by onset, then offset, then label - a total order.

    Used before serialising so two runs that discover regions in a different
    order still hash identically.
    """
    return sorted(regions, key=lambda r: (r.t_start, r.t_end, r.label))
