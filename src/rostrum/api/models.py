"""Pydantic mirrors of :mod:`rostrum.types`, for the API schema (spec 15.1).

Why mirror rather than serve the dataclasses directly: the dataclasses in
:mod:`rostrum.types` are the analysis's data model, and their ``to_dict`` is
what the determinism hash is taken over (CLAUDE.md invariant 13). Deriving an
OpenAPI schema from them would couple the wire format to the hashed format, so
a purely presentational change to the API could change a hash. The mirrors here
are built **from** ``to_dict`` output and never feed back into it: they
describe the response, they do not define it.

Every model therefore follows two rules:

* ``model_config = ConfigDict(extra="allow")`` - a field added to a dataclass
  still reaches the client, and an older client is not broken by it. The schema
  documents what is guaranteed, not what is possible.
* Construction is only ever ``Model.model_validate(result.to_dict())``, so the
  values a client sees are the rounded values the hash was taken over, to the
  digit.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AlignmentModel",
    "AnalysisResultModel",
    "CategoryScoreModel",
    "ClipInfoModel",
    "ErrorModel",
    "EvidenceModel",
    "GapModel",
    "HealthModel",
    "LibraryModel",
    "LibrarySampleModel",
    "LibraryTextModel",
    "RegionModel",
    "RubricScoreModel",
    "StageModel",
    "WordModel",
]


class _Base(BaseModel):
    """Permissive base: unknown keys pass through rather than 500 the request."""

    model_config = ConfigDict(extra="allow")


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


class WordModel(_Base):
    """One aligned word; mirrors :class:`~rostrum.types.Word`."""

    idx: int = Field(description="Position in the transcript, 0-based.")
    text: str = Field(description="The aligner's token.")
    t_start: float = Field(description="Onset in seconds from the processed WAV start.")
    t_end: float = Field(description="Offset in seconds.")
    phones: list[str] = Field(default_factory=list, description="Aligned phone labels.")
    n_syllables: int = Field(default=0, description="Syllable count, for rate.")
    display: str = Field(default="", description="The word as written, with punctuation.")


class GapModel(_Base):
    """One silence between words; mirrors :class:`~rostrum.types.Gap`."""

    idx: int = Field(description="Position in the gap sequence.")
    prev_word_idx: int = Field(description="Word before the gap.")
    next_word_idx: int = Field(description="Word after the gap.")
    t_start: float
    t_end: float
    label: str = Field(
        default="none",
        description="What the text expects here: boundary, minor or none.",
    )
    punctuation: str = Field(default="", description="Punctuation that set the label.")
    vad_refined: bool = Field(default=False, description="Edges refined by VAD.")


class AlignmentModel(_Base):
    """A forced alignment; mirrors :class:`~rostrum.types.Alignment`."""

    file_id: str
    text_id: str | None = None
    speaker_id: str = ""
    channel_id: str = ""
    tier: str | None = None
    words: list[WordModel] = Field(default_factory=list)
    gaps: list[GapModel] = Field(default_factory=list)
    duration: float = 0.0
    sample_rate: int = 16000
    aligner: str = Field(default="", description="Always MFA for timing (invariant 4).")
    acoustic_model: str = ""
    dictionary: str = ""
    oov_words: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Grounding
# --------------------------------------------------------------------------- #


class EvidenceModel(_Base):
    """One measurement behind a region; mirrors :class:`~rostrum.types.Evidence`."""

    feature: str
    value: float = Field(description="What the participant did.")
    env_median: float = Field(description="The skilled readers' median here.")
    env_sigma: float = Field(description="1.4826 * MAD, floored (invariant 10).")
    z: float = Field(description="(value - median) / sigma.")
    unit: str = ""
    gap_label: str | None = None
    n_references: int = 0
    channel_matched: bool = Field(
        default=True,
        description="Whether the comparison was within one channel (invariant 7).",
    )


class RegionModel(_Base):
    """One timestamped flaw region; mirrors :class:`~rostrum.types.Region`."""

    region_id: int
    t_start: float = Field(description="Flaw onset, seconds, 3 dp (invariant 2).")
    t_end: float = Field(description="Flaw offset, seconds.")
    label: str = Field(description="One of the eight flaw labels.")
    category: str = Field(description="One of the six rubric categories.")
    level: str = Field(default="L0", description="Severity rung, L0 to L4.")
    s: float = Field(default=0.0, description="Continuous severity behind the rung.")
    confidence: float = 0.0
    rule_label: str | None = None
    clf_label: str | None = None
    clf_prob: float | None = None
    envelope_n: int = Field(default=0, description="Skilled readers behind the comparison.")
    word_idx: list[int] = Field(default_factory=list, description="First and last word index.")
    gap_idx: int | None = None
    words: str = ""
    evidence: list[EvidenceModel] = Field(default_factory=list)
    context: str = ""
    rationale: str = Field(default="", description="The deviation in words and numbers.")
    fix: str = ""
    peak_z: float = 0.0
    mean_excess_z: float = 0.0
    refined: bool = Field(default=False, description="Boundary refined by ruptures.")
    t_start_unrefined: float | None = None
    t_end_unrefined: float | None = None


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #


class CategoryScoreModel(_Base):
    """One rubric category; mirrors :class:`~rostrum.types.CategoryScore`."""

    category: str
    score: float = Field(description="S_c = 10 * exp(-k_c * M_c), 0 to 10.")
    band: float = Field(default=0.0, description="Leave-one-reference-out std.")
    deviation_mass: float = 0.0
    k: float = 1.0
    n_regions: int = 0
    suppressed: bool = Field(
        default=False,
        description="Excluded from the mean because it could not be measured.",
    )
    suppressed_reason: str = ""


class RubricScoreModel(_Base):
    """The six-category score; mirrors :class:`~rostrum.types.RubricScore`."""

    categories: list[CategoryScoreModel] = Field(default_factory=list)
    overall: float = Field(default=0.0, description="Mean of the unsuppressed categories.")
    overall_band: float = Field(default=0.0, description="+/- confidence band.")
    descriptors: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw descriptors with the readers' range beside them.",
    )


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #


class AnalysisResultModel(_Base):
    """One analysis; mirrors :class:`~rostrum.types.AnalysisResult`."""

    schema_version: int = 1
    file_id: str
    text_id: str | None = None
    speaker_id: str = ""
    channel_id: str = ""
    origin: str = "participant"
    tier: str | None = None
    mode: str = Field(description="reference, norm or reject (spec 13).")
    mode_reason: str = Field(default="", description="Why that mode was chosen.")
    alignment: AlignmentModel | None = None
    regions: list[RegionModel] = Field(default_factory=list)
    score: RubricScoreModel | None = Field(
        default=None, description="Absent in reject mode: nothing is scored."
    )
    duration: float = 0.0
    speech_duration: float = 0.0
    wer: float | None = Field(
        default=None, description="ASR-vs-transcript WER from the mismatch check."
    )
    n_references: int = 0
    reference_ids: list[str] = Field(default_factory=list)
    features_version: int = 1
    config_hash: str = ""
    code_version: str = ""
    warnings: list[str] = Field(default_factory=list)
    analysis_id: str = Field(
        default="", description="sha256(audio bytes + transcript + config hash)."
    )
    cards: list[dict[str, Any]] = Field(
        default_factory=list, description="Explanation cards, in time order (spec 12)."
    )
    curves: dict[str, Any] = Field(
        default_factory=dict,
        description="Participant overlay curves, per-word D_w and top-z (spec 15.2).",
    )
    envelope_curves: dict[str, Any] = Field(
        default_factory=dict,
        description="Envelope median and sigma per word; empty in norm mode.",
    )


# --------------------------------------------------------------------------- #
# The other endpoints
# --------------------------------------------------------------------------- #


class StageModel(_Base):
    """One pipeline stage and what it cost, for the processing panel."""

    name: str
    seconds: float
    note: str = ""


class LibraryTextModel(_Base):
    """One text the library can analyse in Reference mode."""

    text_id: str
    title: str = ""
    n_references: int = Field(description="Skilled readings behind this text.")
    reference_ids: list[str] = Field(default_factory=list)
    n_words: int = Field(default=0, description="Words in the transcript.")
    has_envelope: bool = Field(default=False, description="An envelope is built and loaded.")
    mode: str = Field(
        default="norm",
        description="The mode an upload of this text would be analysed in.",
    )
    transcript: str = Field(default="", description="The text as written.")


class LibrarySampleModel(_Base):
    """One committed sample recording a client can analyse without uploading."""

    file_id: str
    text_id: str | None = None
    path: str = Field(description="Path relative to the repository root.")
    origin: str = Field(default="", description="clean, injected or human_flawed.")
    tier: str | None = None
    label: str = Field(default="", description="The injected flaw, when there is one.")
    level: str = Field(default="", description="Severity rung of that flaw.")
    duration_s: float | None = None


class LibraryModel(_Base):
    """What this deployment can compare an upload against (spec 15.1)."""

    fold: str = "all"
    texts: list[LibraryTextModel] = Field(default_factory=list)
    samples: list[LibrarySampleModel] = Field(default_factory=list)
    min_references: int = Field(
        default=0, description="Readers a text needs before Reference mode is offered."
    )
    pooled_references: int = Field(
        default=0, description="Readings behind the pooled norms used in Norm mode."
    )
    has_classifier: bool = False
    warnings: list[str] = Field(default_factory=list)


class HealthModel(_Base):
    """Versions, hashes and what is installed (spec 15.1)."""

    status: str = Field(description="ok, or degraded when something is missing.")
    code_version: str = Field(description="git describe, or the package version.")
    config_hash: str = Field(description="SHA-256 over every config section.")
    library_fingerprint: str = Field(description="Digest of the loaded references.")
    schema_version: int = 1
    python: str = ""
    packages: dict[str, str] = Field(default_factory=dict)
    aligner: dict[str, Any] = Field(
        default_factory=dict, description="MFA environment and model names."
    )
    limits: dict[str, Any] = Field(default_factory=dict, description="Upload limits.")
    texts: int = 0
    pooled_references: int = 0
    warnings: list[str] = Field(default_factory=list)


class ClipInfoModel(_Base):
    """Provenance of an excerpt, returned in the response headers of ``/clip``."""

    which: str
    file_id: str
    t_start: float
    t_end: float
    duration: float
    sample_rate: int = 16000
    words: str = ""
    word_idx: list[int] | None = None
    note: str = ""


class ErrorModel(_Base):
    """The error body every failing endpoint returns."""

    detail: str = Field(description="What went wrong, in one sentence.")
