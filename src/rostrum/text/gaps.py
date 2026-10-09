"""Gap labels and punctuation from the display transcript (spec 7.1, 10.3).

A silence is only a flaw once the text says what belonged there. The same
700 ms is skilled rhetoric after a full stop and a hesitation in the middle of
a noun phrase, so every pause rule in :mod:`rostrum.detect.zscore` is gated on
:attr:`~rostrum.types.Gap.label`, and that label comes from here - from the
*text*, never from the audio.

The mapping is done in two steps because the aligner and the reader see
different token streams:

1. :func:`rostrum.align.mfa.normalise_for_alignment` splits the transcript into
   ``lab_tokens`` (what MFA aligns, so what :attr:`~rostrum.types.Word.idx`
   indexes) and ``display_tokens`` (as written), plus the index map between
   them. ``Four-score`` is one display token and two lab tokens; ``--`` is a
   display token and no lab token at all.
2. Gap *i* follows lab word *i*. Its punctuation is what stands between the
   display token that lab word *i* came from and the one lab word *i+1* came
   from: that token's trailing punctuation, plus any punctuation-only tokens in
   between, plus a paragraph break if the source has a blank line there.

Two lab tokens from one display token (``Four-score`` to ``four score``) have
no text between them, so their gap is ``none`` whatever the token's
punctuation.

Everything here is a pure function of the transcript string and the
configuration; the audio is never consulted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rostrum.align.mfa import normalise_for_alignment

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config
    from rostrum.types import GapLabel

logger = logging.getLogger(__name__)

__all__ = [
    "BOUNDARY_PUNCTUATION",
    "MINOR_PUNCTUATION",
    "DisplayToken",
    "GapLabelRules",
    "counts",
    "gap_label_map",
    "gap_labels",
    "tokenise_display",
    "trailing_punctuation",
]

BOUNDARY_PUNCTUATION: tuple[str, ...] = (".", ";", ":", "!", "?", "--", "—")
"""Code default for ``detection.gap_labels.boundary_punctuation``."""

MINOR_PUNCTUATION: tuple[str, ...] = (",",)
"""Code default for ``detection.gap_labels.minor_punctuation``."""

_TOKEN = re.compile(r"\S+")
_PARAGRAPH = re.compile(r"\n[ \t]*\n")
# Punctuation that may trail a word. Quotes and brackets are included so
# `equal."` still reads as a sentence end; the apostrophe is not, because
# `readers'` is a word.
_TRAILING = '.,;:!?\u2014-\u2026)]}"\u201d\u2019'
_PUNCT_ONLY = re.compile(r"^[^\w]+$", re.UNICODE)


@dataclass(frozen=True)
class DisplayToken:
    """One whitespace-separated token of the display transcript.

    Attributes:
        idx: Position in the display token sequence.
        text: The token as written, punctuation included.
        word: The token with trailing punctuation stripped; empty for a
            punctuation-only token such as a standalone ``--``.
        punctuation: The trailing punctuation that was stripped.
        paragraph_break_after: Whether a blank line follows this token.
        start: Character offset of the token in the source string.
        end: Character offset one past the token.
    """

    idx: int
    text: str
    word: str
    punctuation: str
    paragraph_break_after: bool
    start: int
    end: int

    @property
    def is_punctuation_only(self) -> bool:
        """Whether the token carries no letters or digits at all."""
        return not self.word


@dataclass(frozen=True)
class GapLabelRules:
    """Which punctuation marks make a gap ``boundary`` or ``minor``.

    Attributes:
        boundary: Marks that make the following gap a rhetorical boundary.
        minor: Marks that make it a minor break.
        default: Label for a gap with no punctuation.
        paragraph_is_boundary: Whether a blank line counts as a boundary even
            without punctuation.
    """

    boundary: tuple[str, ...] = BOUNDARY_PUNCTUATION
    minor: tuple[str, ...] = MINOR_PUNCTUATION
    default: str = "none"
    paragraph_is_boundary: bool = True

    @classmethod
    def from_config(cls, config: Config | None) -> GapLabelRules:
        """Read ``detection.gap_labels``; code defaults without a config."""
        if config is None:
            return cls()
        block = config.detection["gap_labels"]
        return cls(
            boundary=tuple(str(p) for p in block["boundary_punctuation"]),
            minor=tuple(str(p) for p in block["minor_punctuation"]),
            default=str(block.get("default", "none")),
            paragraph_is_boundary=bool(block.get("paragraph_is_boundary", True)),
        )

    def label_for(self, punctuation: str, *, paragraph_break: bool = False) -> GapLabel:
        """Classify one gap from the text standing between its words.

        Args:
            punctuation: Everything punctuational between the two words, in
                source order.
            paragraph_break: Whether a blank line stands there too.

        Returns:
            ``"boundary"``, ``"minor"`` or ``"none"``. A gap carrying both -
            ``",--"`` - is a boundary: the stronger break wins.
        """
        if paragraph_break and self.paragraph_is_boundary:
            return "boundary"
        if any(mark in punctuation for mark in self.boundary):
            return "boundary"
        if any(mark in punctuation for mark in self.minor):
            return "minor"
        return self.default  # type: ignore[return-value]


def trailing_punctuation(token: str) -> tuple[str, str]:
    """Split a display token into ``(word, trailing punctuation)``.

    Examples:
        >>> trailing_punctuation("equal.")
        ('equal', '.')
        >>> trailing_punctuation("ground.--")
        ('ground', '.--')
    """
    stripped = token.rstrip(_TRAILING)
    return stripped, token[len(stripped) :]


def tokenise_display(display: str) -> list[DisplayToken]:
    """Tokenise the display transcript, keeping punctuation and paragraphs.

    The split is on runs of non-whitespace, the same tokenisation
    :func:`rostrum.align.mfa.normalise_for_alignment` performs, so the indices
    of the returned tokens are its ``display_tokens`` indices.

    Args:
        display: The transcript as written, newlines included.

    Returns:
        One :class:`DisplayToken` per token, in order.
    """
    spans = [(m.start(), m.end()) for m in _TOKEN.finditer(display)]
    out: list[DisplayToken] = []
    for idx, (start, end) in enumerate(spans):
        text = display[start:end]
        word, punctuation = trailing_punctuation(text)
        if _PUNCT_ONLY.match(text):
            word, punctuation = "", text
        next_start = spans[idx + 1][0] if idx + 1 < len(spans) else len(display)
        between = display[end:next_start]
        out.append(
            DisplayToken(
                idx=idx,
                text=text,
                word=word,
                punctuation=punctuation,
                paragraph_break_after=bool(_PARAGRAPH.search(between)),
                start=start,
                end=end,
            )
        )
    return out


def gap_labels(
    display: str,
    *,
    config: Config | None = None,
    rules: GapLabelRules | None = None,
    prefer_year: bool = True,
) -> list[tuple[GapLabel, str]]:
    """Label every inter-word gap of a transcript (spec 7.1).

    Args:
        display: The transcript as written. Punctuation and blank lines carry
            the whole signal, so a stripped transcript yields all ``none``.
        config: Supplies ``detection.gap_labels``.
        rules: Explicit rules, overriding the config.
        prefer_year: Passed to the numeral expansion, which decides whether
            ``1863`` becomes "eighteen sixty three" or "one thousand ...".

    Returns:
        ``[(label, punctuation), ...]``, one entry per gap, so entry *i*
        describes the gap between aligned words *i* and *i+1*. The list is one
        shorter than the lab token count, which is exactly what
        :func:`rostrum.features.pauses.gaps_from_words` produces.
    """
    rules = rules or GapLabelRules.from_config(config)
    tokens = tokenise_display(display)
    lab = normalise_for_alignment(display, prefer_year=prefer_year)
    mapping = lab.lab_to_display

    out: list[tuple[GapLabel, str]] = []
    for i in range(len(mapping) - 1):
        first, second = mapping[i], mapping[i + 1]
        if first == second:
            # Two lab tokens from one written word: no text stands between them.
            out.append((rules.default, ""))  # type: ignore[arg-type]
            continue
        marks = tokens[first].punctuation
        paragraph = tokens[first].paragraph_break_after
        for between in tokens[first + 1 : second]:
            marks += between.text if between.is_punctuation_only else between.punctuation
            paragraph = paragraph or between.paragraph_break_after
        out.append((rules.label_for(marks, paragraph_break=paragraph), marks))
    return out


def gap_label_map(
    display: str,
    *,
    n_gaps: int | None = None,
    config: Config | None = None,
    rules: GapLabelRules | None = None,
) -> dict[int, tuple[str, str]]:
    """Gap labels in the form :func:`rostrum.features.pauses.apply_gap_labels` takes.

    Args:
        display: The transcript as written.
        n_gaps: Gaps in the alignment. When it disagrees with the transcript the
            surplus entries are dropped and a warning is logged: a reader who
            elided a word shifts every later gap, and a wrong label is worse
            than no label.
        config: Supplies ``detection.gap_labels``.
        rules: Explicit rules, overriding the config.

    Returns:
        ``{gap_idx: (label, punctuation)}``.
    """
    labels = gap_labels(display, config=config, rules=rules)
    if n_gaps is not None and n_gaps != len(labels):
        logger.warning(
            "transcript implies %d gaps but the alignment has %d; labelling the first %d",
            len(labels),
            n_gaps,
            min(len(labels), n_gaps),
        )
        labels = labels[:n_gaps]
    return {i: (str(label), punctuation) for i, (label, punctuation) in enumerate(labels)}


def counts(labels: Sequence[tuple[str, str]] | Mapping[int, tuple[str, str]]) -> dict[str, int]:
    """Label histogram, for a build log."""
    values = list(labels.values()) if isinstance(labels, Mapping) else list(labels)
    out: dict[str, int] = {}
    for label, _ in values:
        out[str(label)] = out.get(str(label), 0) + 1
    return out
