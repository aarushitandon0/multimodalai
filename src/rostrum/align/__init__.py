"""rostrum.align -- Forced alignment (MFA) and ASR-based transcript mismatch checks.

Montreal Forced Aligner is the only aligner used, for dataset labels and at
runtime (``CLAUDE.md`` invariant 4). ``faster-whisper``/WhisperX appear only in
the ASR-vs-transcript WER check that feeds Reject mode, never for timestamps.
"""

from rostrum.align.mfa import (
    CorpusItem,
    LabText,
    MFAError,
    MFANotAvailableError,
    ValidationReport,
    expand_numerals,
    list_models,
    mfa_version,
    normalise_for_alignment,
    parse_textgrid,
    prepare_corpus,
    read_textgrid,
    run_mfa,
    validate,
)

__all__ = [
    "CorpusItem",
    "LabText",
    "MFAError",
    "MFANotAvailableError",
    "ValidationReport",
    "expand_numerals",
    "list_models",
    "mfa_version",
    "normalise_for_alignment",
    "parse_textgrid",
    "prepare_corpus",
    "read_textgrid",
    "run_mfa",
    "validate",
]
