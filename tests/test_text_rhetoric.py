"""Tests for the text annotation behind a card's context clause (spec 12)."""

from __future__ import annotations

import json

import pytest

from rostrum.config import load_config
from rostrum.text.rhetoric import RhetoricRules, annotate, repeated_spans

ANAPHORA = (
    "We can not dedicate, we can not consecrate. "
    "We can not hallow this ground, we can not forget."
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def gettysburg(cfg):
    # dataset/texts/ is tracked in git, so this works in a fresh clone and does
    # not depend on which files dataset_sample/ happens to carry.
    path = cfg.path("data.texts") / "s1_gettysburg.json"
    display = json.loads(path.read_text(encoding="utf-8"))["display"]
    return annotate(display, text_id="s1_gettysburg", config=cfg)


# --------------------------------------------------------------------------- #
# Repeated sequences
# --------------------------------------------------------------------------- #


def test_a_repeated_clause_opening_is_anaphora(cfg):
    annotation = annotate(ANAPHORA, text_id="t", config=cfg)
    hits = [r for r in annotation.repeats if r.text == "we can not"]
    assert hits, [r.text for r in annotation.repeats]
    assert {r.kind for r in hits} == {"anaphora"}
    assert hits[0].n_occurrences == 4
    assert hits[0].occurrence_starts == (0, 4, 8, 14)


def test_the_longest_most_frequent_figure_wins(cfg):
    """ "we can not" three times explains more text than a 4-gram twice."""
    annotation = annotate(ANAPHORA, text_id="t", config=cfg)
    assert all("we can not" in r.text or r.n_words < 3 for r in annotation.repeats)


def test_a_repeat_does_not_cross_a_full_stop(cfg):
    annotation = annotate("alpha bravo. alpha bravo. charlie", text_id="t", config=cfg)
    for repeat in annotation.repeats:
        assert repeat.last_word - repeat.first_word < 2 or repeat.text == "alpha bravo"


def test_function_word_sequences_are_not_figures(cfg):
    annotation = annotate("of the man, of the hour", text_id="t", config=cfg)
    assert [r.text for r in annotation.repeats] == []


def test_overlapping_occurrences_are_not_counted_twice():
    spans = repeated_spans(["very", "very", "very", "very"], rules=RhetoricRules(min_ngram=2))
    group = [s for s in spans if s.text == "very very"]
    assert group and group[0].n_occurrences == 2


def test_repeats_at_finds_the_span_covering_a_region(cfg):
    annotation = annotate(ANAPHORA, text_id="t", config=cfg)
    hits = annotation.repeats_at(4, 6)
    assert hits and hits[0].text == "we can not"
    assert annotation.repeats_at(3, 3) == () or hits[0].kind == "anaphora"


# --------------------------------------------------------------------------- #
# Sentences and positions
# --------------------------------------------------------------------------- #


def test_sentences_split_on_terminal_punctuation(cfg):
    annotation = annotate("alpha bravo. charlie delta echo!", text_id="t", config=cfg)
    assert [(s.first_word, s.last_word, s.terminator) for s in annotation.sentences] == [
        (0, 1, "."),
        (2, 4, "!"),
    ]


def test_the_tail_of_an_unterminated_text_is_still_a_sentence(cfg):
    annotation = annotate("alpha bravo. charlie delta", text_id="t", config=cfg)
    assert len(annotation.sentences) == 2
    assert annotation.sentences[-1].terminator == ""


def test_position_names_the_sentence_edges(cfg):
    annotation = annotate("alpha bravo charlie. delta echo", text_id="t", config=cfg)
    assert annotation.position(0) == "sentence_initial"
    assert annotation.position(1) == "mid_clause"
    assert annotation.position(2) == "sentence_final"
    assert annotation.position(3) == "sentence_initial"


def test_position_marks_a_clause_boundary_inside_a_sentence(cfg):
    annotation = annotate("alpha bravo, charlie delta echo.", text_id="t", config=cfg)
    assert annotation.position(2) == "clause_boundary"


def test_a_three_item_list_is_detected(cfg):
    annotation = annotate("of the people, by the people, for the people.", text_id="t", config=cfg)
    assert annotation.in_list(0, 2)
    assert annotation.in_list(3, 5)


def test_a_two_clause_sentence_is_not_a_list(cfg):
    annotation = annotate("alpha bravo, charlie delta.", text_id="t", config=cfg)
    assert not annotation.in_list(0, 3)


# --------------------------------------------------------------------------- #
# Word spans
# --------------------------------------------------------------------------- #


def test_word_text_quotes_the_display_form(cfg):
    annotation = annotate("Four score and seven years ago", text_id="t", config=cfg)
    assert annotation.word_text(0, 2) == "Four score and"
    assert annotation.word_text(5) == "ago"


def test_word_text_is_clipped_to_the_text(cfg):
    annotation = annotate("alpha bravo", text_id="t", config=cfg)
    assert annotation.word_text(0, 99) == "alpha bravo"
    assert annotation.word_text(5, 9) == ""


def test_gap_outside_the_text_is_unlabelled(cfg):
    annotation = annotate("alpha bravo.", text_id="t", config=cfg)
    assert annotation.gap(99) == ("none", "")


# --------------------------------------------------------------------------- #
# The real text
# --------------------------------------------------------------------------- #


def test_gettysburg_finds_the_known_figures(gettysburg):
    """The three "we can not" clauses are the text's defining anaphora."""
    we_can_not = [r for r in gettysburg.repeats if r.text == "we can not"]
    assert len(we_can_not) == 3
    assert {r.kind for r in we_can_not} == {"anaphora"}
    assert any(r.text == "the people" and r.n_occurrences == 3 for r in gettysburg.repeats)


def test_gettysburg_gap_labels_are_mostly_none(gettysburg):
    """A label per gap, and the boundaries are the minority they should be."""
    labels = [label for label, _ in gettysburg.gap_labels]
    assert len(labels) == gettysburg.n_words - 1
    assert labels.count("boundary") < labels.count("none")
    assert labels.count("boundary") >= len(gettysburg.sentences) - 1
