"""Repetition, anaphora and sentence position in the transcript (spec 12).

The causal explanation of a flaw is more useful when it says *where in the text*
the flaw sits [R39]: a cut pause after a full stop is a different mistake from
the same cut inside a noun phrase, and a rushed phrase that the author repeated
three times is rushed through a rhetorical figure.

This module annotates the text alone. It never sees the audio, the features or
the envelope, and it produces no judgement - only the facts
:mod:`rostrum.explain.context` turns into a context clause:

* **sentences**, from terminal punctuation, so a word is sentence-initial,
  sentence-final or mid-clause;
* **gap labels**, delegated to :mod:`rostrum.text.gaps`;
* **repeated word sequences**, classified as ``anaphora`` (every occurrence
  begins a clause, as in "we can not dedicate - we can not consecrate"),
  ``parallel_structure`` (a repeated multi-word sequence elsewhere in the
  clause) or ``repetition`` (a repeated single word);
* **list items**, a run of three or more clauses separated by commas.

Indices are *aligned word* indices - positions in
:attr:`rostrum.align.mfa.LabText.lab_tokens`, which is what
:attr:`rostrum.types.Word.idx` counts - so a span here can be compared directly
with a :class:`~rostrum.types.Region`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rostrum.align.mfa import normalise_for_alignment
from rostrum.text import gaps as G

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_FUNCTION_WORDS",
    "DEFAULT_MAX_NGRAM",
    "DEFAULT_MIN_NGRAM",
    "RepeatKind",
    "RepeatedSpan",
    "RhetoricRules",
    "Sentence",
    "TextAnnotation",
    "WordPosition",
    "annotate",
    "repeated_spans",
]

RepeatKind = Literal["anaphora", "parallel_structure", "repetition"]
"""How a repeated sequence is used; see :class:`RepeatedSpan`."""

WordPosition = Literal["sentence_initial", "sentence_final", "mid_clause", "clause_boundary"]
"""Where a word sits in its sentence; see :meth:`TextAnnotation.position`."""

DEFAULT_MIN_NGRAM: int = 2
"""Shortest repeated sequence reported. A single repeated ``the`` is noise."""

DEFAULT_MAX_NGRAM: int = 8
"""Longest repeated sequence searched for."""

DEFAULT_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
    }
)
"""Code default for ``detection.rhetoric.function_words``.

A sequence made only of these - "of that", "in a" - repeats in any English
text and is an accident of the grammar, not a figure the reader is performing.
"""

_TERMINATORS = ".!?"


@dataclass(frozen=True)
class RhetoricRules:
    """Thresholds for the text annotation (``detection.rhetoric``).

    Attributes:
        min_ngram: Shortest repeated sequence reported.
        max_ngram: Longest repeated sequence searched for.
        min_occurrences: Occurrences required to count as a repetition.
        min_list_items: Comma-separated clauses that make a sentence a list.
        function_words: Sequences made only of these are not reported.
    """

    min_ngram: int = DEFAULT_MIN_NGRAM
    max_ngram: int = DEFAULT_MAX_NGRAM
    min_occurrences: int = 2
    min_list_items: int = 3
    function_words: frozenset[str] = DEFAULT_FUNCTION_WORDS

    @classmethod
    def from_config(cls, config: Config | None) -> RhetoricRules:
        """Read ``detection.rhetoric``; code defaults without a config."""
        if config is None:
            return cls()
        block = config.detection.get("rhetoric", {}) if hasattr(config.detection, "get") else {}
        if not block:
            return cls()
        return cls(
            min_ngram=int(block.get("min_ngram", DEFAULT_MIN_NGRAM)),
            max_ngram=int(block.get("max_ngram", DEFAULT_MAX_NGRAM)),
            min_occurrences=int(block.get("min_occurrences", 2)),
            min_list_items=int(block.get("min_list_items", 3)),
            function_words=frozenset(
                str(w).lower() for w in block.get("function_words", DEFAULT_FUNCTION_WORDS)
            ),
        )


@dataclass(frozen=True)
class Sentence:
    """One sentence of the transcript, in aligned-word indices.

    Attributes:
        idx: Sentence number, from zero.
        first_word: Index of its first aligned word.
        last_word: Index of its last aligned word, inclusive.
        terminator: The punctuation that ended it; empty at end of text.
        n_commas: Comma-separated clauses inside it, minus one.
    """

    idx: int
    first_word: int
    last_word: int
    terminator: str = ""
    n_commas: int = 0

    def covers(self, word_idx: int) -> bool:
        """Whether ``word_idx`` falls inside this sentence."""
        return self.first_word <= int(word_idx) <= self.last_word


@dataclass(frozen=True)
class RepeatedSpan:
    """One occurrence of a repeated word sequence.

    Attributes:
        group_id: Shared by every occurrence of the same sequence.
        kind: ``anaphora``, ``parallel_structure`` or ``repetition``.
        first_word: Index of the occurrence's first aligned word.
        last_word: Index of its last aligned word, inclusive.
        text: The sequence as written, lower-cased.
        n_occurrences: How many times the sequence occurs in the text.
        occurrence_starts: ``first_word`` of every occurrence, sorted.
    """

    group_id: int
    kind: RepeatKind
    first_word: int
    last_word: int
    text: str
    n_occurrences: int
    occurrence_starts: tuple[int, ...]

    @property
    def n_words(self) -> int:
        """Words in the sequence."""
        return self.last_word - self.first_word + 1

    def overlaps(self, first_word: int, last_word: int) -> bool:
        """Whether this occurrence intersects the word range given."""
        return not (last_word < self.first_word or first_word > self.last_word)


@dataclass(frozen=True)
class TextAnnotation:
    """Everything the explanation needs from the text alone.

    Attributes:
        text_id: Text identifier, or empty for an unknown text.
        words: Display form of every aligned word, parallel to
            :attr:`~rostrum.types.Word.idx`.
        lab_words: The lower-case aligned tokens themselves.
        gap_labels: ``[(label, punctuation), ...]``, one per inter-word gap.
        sentences: The sentences, in order.
        repeats: Every occurrence of every repeated sequence, by onset.
        list_items: First-word index of each item of a three-or-more-item list.
    """

    text_id: str = ""
    words: tuple[str, ...] = ()
    lab_words: tuple[str, ...] = ()
    gap_labels: tuple[tuple[str, str], ...] = ()
    sentences: tuple[Sentence, ...] = ()
    repeats: tuple[RepeatedSpan, ...] = ()
    list_items: tuple[int, ...] = field(default=())

    # -- lookups ------------------------------------------------------------

    @property
    def n_words(self) -> int:
        """Aligned words in the text."""
        return len(self.lab_words)

    def word_text(self, first_word: int, last_word: int | None = None) -> str:
        """The written words of a span, space-joined.

        Args:
            first_word: First aligned word index.
            last_word: Last aligned word index, inclusive; ``first_word`` alone
                when omitted.

        Returns:
            The display forms, or an empty string when the span lies outside
            the text.
        """
        last = first_word if last_word is None else last_word
        lo, hi = max(0, int(first_word)), min(len(self.words) - 1, int(last))
        if lo > hi:
            return ""
        return " ".join(self.words[lo : hi + 1])

    def gap(self, gap_idx: int) -> tuple[str, str]:
        """``(label, punctuation)`` of one gap; ``("none", "")`` if unknown."""
        if 0 <= int(gap_idx) < len(self.gap_labels):
            return self.gap_labels[int(gap_idx)]
        return ("none", "")

    def sentence_of(self, word_idx: int) -> Sentence | None:
        """The sentence containing ``word_idx``, or ``None``."""
        for sentence in self.sentences:
            if sentence.covers(word_idx):
                return sentence
        return None

    def position(self, word_idx: int) -> WordPosition:
        """Where ``word_idx`` sits in its sentence (spec 12).

        ``clause_boundary`` means the word directly follows a comma or a
        stronger mark without starting a new sentence - the position where a
        skilled reader breathes.
        """
        sentence = self.sentence_of(word_idx)
        if sentence is not None:
            if int(word_idx) == sentence.first_word:
                return "sentence_initial"
            if int(word_idx) == sentence.last_word:
                return "sentence_final"
        label, _ = self.gap(int(word_idx) - 1)
        if label in {"boundary", "minor"}:
            return "clause_boundary"
        return "mid_clause"

    def repeats_at(self, first_word: int, last_word: int | None = None) -> tuple[RepeatedSpan, ...]:
        """Repeated sequences overlapping a word span, longest first."""
        last = first_word if last_word is None else last_word
        hits = [r for r in self.repeats if r.overlaps(int(first_word), int(last))]
        hits.sort(key=lambda r: (-r.n_words, r.first_word))
        return tuple(hits)

    def in_list(self, first_word: int, last_word: int | None = None) -> bool:
        """Whether a span starts an item of a three-or-more-item list."""
        last = first_word if last_word is None else last_word
        return any(int(first_word) <= i <= int(last) for i in self.list_items)


# --------------------------------------------------------------------------- #
# Repeated sequences
# --------------------------------------------------------------------------- #


def _starts_clause(start: int, gap_labels: Sequence[tuple[str, str]]) -> bool:
    """Whether a word begins the text or follows a labelled break."""
    if start == 0:
        return True
    if start - 1 >= len(gap_labels):
        return False
    return gap_labels[start - 1][0] in {"boundary", "minor"}


def _crosses_boundary(start: int, n: int, gap_labels: Sequence[tuple[str, str]]) -> bool:
    """Whether a sequence spans a full stop, and so is not one figure."""
    return any(
        gap_idx < len(gap_labels) and gap_labels[gap_idx][0] == "boundary"
        for gap_idx in range(start, start + n - 1)
    )


def repeated_spans(
    lab_words: Sequence[str],
    *,
    gap_labels: Sequence[tuple[str, str]] = (),
    rules: RhetoricRules | None = None,
) -> list[RepeatedSpan]:
    """Find repeated word sequences and say how they are used (spec 12).

    Longest match first: once "we can not" is accepted, its sub-sequence "we
    can" is not reported again for the same words, so a region is explained by
    the longest figure it sits in rather than by every fragment of it.

    Args:
        lab_words: The aligned tokens, lower-case.
        gap_labels: ``[(label, punctuation), ...]`` from
            :func:`rostrum.text.gaps.gap_labels`; used to decide whether every
            occurrence begins a clause, which is what makes a repetition an
            *anaphora* rather than an incidental one.
        rules: Lengths, occurrence count and function-word list.

    Returns:
        Every occurrence of every accepted sequence, sorted by onset. Two
        occurrences of one sequence share a ``group_id``.
    """
    rules = rules or RhetoricRules()
    words = [str(w).lower() for w in lab_words]
    n_words = len(words)

    # Pass 1: every sequence that repeats, with its non-overlapping occurrences.
    candidates: list[tuple[int, tuple[str, ...], list[int]]] = []
    for n in range(max(1, int(rules.min_ngram)), min(int(rules.max_ngram), n_words) + 1):
        positions: dict[tuple[str, ...], list[int]] = {}
        for start in range(n_words - n + 1):
            # A figure sits inside one clause. A sequence that runs over a full
            # stop - "...not. We can..." - is an artefact of the window, not
            # something the author repeated.
            if _crosses_boundary(start, n, gap_labels):
                continue
            positions.setdefault(tuple(words[start : start + n]), []).append(start)
        for sequence, starts in positions.items():
            if all(word in rules.function_words for word in sequence):
                continue
            # Non-overlapping occurrences only: in "very very very" the middle
            # start is part of the first occurrence, not a third one.
            kept: list[int] = []
            for start in starts:
                if kept and start < kept[-1] + n:
                    continue
                kept.append(start)
            if len(kept) >= int(rules.min_occurrences):
                candidates.append((n, sequence, kept))

    # Pass 2: accept by how much text the figure explains - length times
    # occurrences - so "we can not" three times beats "we can not we" twice,
    # and a sub-sequence of an accepted figure is not reported again.
    claimed: set[int] = set()
    out: list[RepeatedSpan] = []
    group_id = 0
    candidates.sort(key=lambda c: (-c[0] * len(c[2]), -c[0], c[2][0]))
    for n, sequence, starts in candidates:
        kept = [s for s in starts if not any(i in claimed for i in range(s, s + n))]
        if len(kept) < int(rules.min_occurrences):
            continue
        all_clause_initial = all(_starts_clause(s, gap_labels) for s in kept)
        if n < 2:
            kind: RepeatKind = "repetition"
        elif all_clause_initial:
            kind = "anaphora"
        else:
            kind = "parallel_structure"
        for start in kept:
            claimed.update(range(start, start + n))
            out.append(
                RepeatedSpan(
                    group_id=group_id,
                    kind=kind,
                    first_word=start,
                    last_word=start + n - 1,
                    text=" ".join(sequence),
                    n_occurrences=len(kept),
                    occurrence_starts=tuple(kept),
                )
            )
        group_id += 1

    out.sort(key=lambda r: (r.first_word, -r.n_words))
    return out


# --------------------------------------------------------------------------- #
# Sentences and lists
# --------------------------------------------------------------------------- #


def _sentences(
    lab_to_display: Sequence[int],
    tokens: Sequence[G.DisplayToken],
) -> list[Sentence]:
    """Split the aligned words into sentences on terminal punctuation."""
    out: list[Sentence] = []
    first = 0
    commas = 0
    for i, display_idx in enumerate(lab_to_display):
        punctuation = tokens[display_idx].punctuation if display_idx < len(tokens) else ""
        last_of_token = i + 1 >= len(lab_to_display) or lab_to_display[i + 1] != display_idx
        if not last_of_token:
            continue
        if "," in punctuation:
            commas += 1
        terminator = next((c for c in _TERMINATORS if c in punctuation), "")
        end_of_text = i + 1 >= len(lab_to_display)
        if terminator or end_of_text:
            out.append(
                Sentence(
                    idx=len(out),
                    first_word=first,
                    last_word=i,
                    terminator=terminator,
                    n_commas=commas,
                )
            )
            first, commas = i + 1, 0
    return out


def _list_items(
    sentences: Sequence[Sentence],
    gap_labels: Sequence[tuple[str, str]],
    *,
    min_items: int = 3,
) -> tuple[int, ...]:
    """First word of each item of every three-or-more-item comma list."""
    out: list[int] = []
    for sentence in sentences:
        starts = [sentence.first_word]
        for gap_idx in range(sentence.first_word, sentence.last_word):
            if gap_idx < len(gap_labels) and gap_labels[gap_idx][0] == "minor":
                starts.append(gap_idx + 1)
        if len(starts) >= int(min_items):
            out.extend(starts)
    return tuple(sorted(set(out)))


def annotate(
    display: str,
    *,
    text_id: str = "",
    config: Config | None = None,
    rules: RhetoricRules | None = None,
) -> TextAnnotation:
    """Annotate a transcript with everything spec 12 needs from the text.

    Args:
        display: The transcript as written - punctuation and blank lines are
            the whole signal, so a stripped transcript yields no sentences and
            all-``none`` gaps.
        text_id: Text identifier to record.
        config: Supplies ``detection.gap_labels`` and ``detection.rhetoric``.
        rules: Explicit rhetoric rules, overriding the config.

    Returns:
        A :class:`TextAnnotation` indexed on aligned word indices.
    """
    rules = rules or RhetoricRules.from_config(config)
    lab = normalise_for_alignment(display)
    tokens = G.tokenise_display(display)
    labels = [
        (str(label), punctuation) for label, punctuation in G.gap_labels(display, config=config)
    ]
    sentences = _sentences(lab.lab_to_display, tokens)
    annotation = TextAnnotation(
        text_id=text_id,
        words=tuple(lab.display_for(i) for i in range(len(lab.lab_tokens))),
        lab_words=lab.lab_tokens,
        gap_labels=tuple(labels),
        sentences=tuple(sentences),
        repeats=tuple(repeated_spans(lab.lab_tokens, gap_labels=labels, rules=rules)),
        list_items=_list_items(sentences, labels, min_items=rules.min_list_items),
    )
    logger.debug(
        "%s: %d words, %d sentences, %d repeated spans, gaps %s",
        text_id or "text",
        annotation.n_words,
        len(annotation.sentences),
        len(annotation.repeats),
        G.counts(labels),
    )
    return annotation
