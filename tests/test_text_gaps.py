"""Tests for gap labels derived from the transcript (spec 7.1)."""

from __future__ import annotations

import pytest

from rostrum.config import load_config
from rostrum.text.gaps import (
    GapLabelRules,
    counts,
    gap_label_map,
    gap_labels,
    tokenise_display,
    trailing_punctuation,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# Tokenising
# --------------------------------------------------------------------------- #


def test_trailing_punctuation_splits_the_mark_off():
    assert trailing_punctuation("equal.") == ("equal", ".")
    assert trailing_punctuation("nation,") == ("nation", ",")
    assert trailing_punctuation("ground") == ("ground", "")


def test_trailing_punctuation_keeps_a_word_internal_apostrophe():
    assert trailing_punctuation("nation's") == ("nation's", "")


def test_tokenise_display_marks_a_punctuation_only_token():
    tokens = tokenise_display("dedicate -- we")
    assert [t.word for t in tokens] == ["dedicate", "", "we"]
    assert tokens[1].is_punctuation_only
    assert tokens[1].punctuation == "--"


def test_tokenise_display_finds_the_paragraph_break():
    tokens = tokenise_display("one two\n\nthree")
    assert tokens[1].paragraph_break_after
    assert not tokens[0].paragraph_break_after


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #


def test_gap_labels_one_entry_per_gap(cfg):
    """n words give n-1 gaps, which is what gaps_from_words produces."""
    labels = gap_labels("one two three four", config=cfg)
    assert len(labels) == 3


def test_gap_labels_read_the_punctuation(cfg):
    labels = gap_labels("alpha. bravo, charlie delta", config=cfg)
    assert [label for label, _ in labels] == ["boundary", "minor", "none"]
    assert [mark for _, mark in labels] == [".", ",", ""]


@pytest.mark.parametrize("mark", [".", ";", ":", "!", "?", "--"])
def test_every_configured_boundary_mark_is_a_boundary(cfg, mark):
    labels = gap_labels(f"alpha{mark} bravo", config=cfg)
    assert labels[0][0] == "boundary"


def test_a_standalone_dash_between_words_is_a_boundary(cfg):
    """The Bliss copy writes "we can not dedicate -- we can not consecrate"."""
    labels = gap_labels("dedicate -- we", config=cfg)
    assert len(labels) == 1  # the dash is no aligned word, so there is one gap
    assert labels[0] == ("boundary", "--")


def test_a_hyphenated_word_has_no_internal_gap(cfg):
    """`Four-score` aligns as two words with no text between them."""
    labels = gap_labels("Four-score and seven", config=cfg)
    assert [label for label, _ in labels] == ["none", "none", "none"]


def test_a_paragraph_break_is_a_boundary_without_a_mark(cfg):
    labels = gap_labels("alpha\n\nbravo", config=cfg)
    assert labels[0][0] == "boundary"


def test_the_stronger_mark_wins(cfg):
    labels = gap_labels("alpha,-- bravo", config=cfg)
    assert labels[0][0] == "boundary"


def test_an_expanded_numeral_keeps_its_following_mark(cfg):
    """1863, aligns as three words; the comma belongs after the last of them."""
    labels = gap_labels("in 1863, we", config=cfg)
    assert [label for label, _ in labels] == ["none", "none", "none", "minor"]


def test_rules_can_be_overridden_without_a_config():
    rules = GapLabelRules(boundary=("!",), minor=())
    labels = gap_labels("alpha. bravo! charlie", rules=rules)
    assert [label for label, _ in labels] == ["none", "boundary"]


def test_an_unpunctuated_transcript_yields_no_expectation(cfg):
    labels = gap_labels("alpha bravo charlie", config=cfg)
    assert counts(labels) == {"none": 2}


# --------------------------------------------------------------------------- #
# The map handed to apply_gap_labels
# --------------------------------------------------------------------------- #


def test_gap_label_map_is_keyed_by_gap_index(cfg):
    mapping = gap_label_map("alpha. bravo, charlie", config=cfg)
    assert mapping == {0: ("boundary", "."), 1: ("minor", ",")}


def test_gap_label_map_truncates_when_the_alignment_is_shorter(cfg, caplog):
    with caplog.at_level("WARNING"):
        mapping = gap_label_map("alpha. bravo, charlie", n_gaps=1, config=cfg)
    assert set(mapping) == {0}
    assert "alignment has" in caplog.text


def test_gap_label_map_feeds_apply_gap_labels(cfg):
    """The two halves of the contract fit: labels in, labelled gaps out."""
    from rostrum.features.pauses import apply_gap_labels
    from rostrum.types import Word

    words = [
        Word(idx=i, text=t, t_start=float(i), t_end=float(i) + 0.5)
        for i, t in enumerate(["alpha", "bravo", "charlie"])
    ]
    from rostrum.features.pauses import gaps_from_words

    gaps = gaps_from_words(words)
    labelled = apply_gap_labels(gaps, gap_label_map("alpha. bravo, charlie", config=cfg))
    assert [g.label for g in labelled] == ["boundary", "minor"]
    assert [g.punctuation for g in labelled] == [".", ","]
