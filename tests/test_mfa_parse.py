"""Tests for the pure half of :mod:`rostrum.align.mfa`.

Nothing here needs MFA, conda or audio: the TextGrid parser, the numeral
expander and the transcript normaliser are pure functions, and the fixtures in
``tests/fixtures`` are hand-written. ``prepare_corpus`` is exercised against a
stub WAV, since it only copies bytes.

The fixture ``tiny_long.TextGrid`` is the opening of the Gettysburg Address
("four score and seven years ago") with one 150 ms gap after "and", so the
expected times are known by construction rather than read back off a tool.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from rostrum.align.mfa import (
    ARPA_VOWELS,
    LabText,
    MFAError,
    expand_numerals,
    normalise_for_alignment,
    parse_textgrid,
    prepare_corpus,
    read_textgrid,
)

FIXTURES = Path(__file__).parent / "fixtures"
LONG_GRID = FIXTURES / "tiny_long.TextGrid"
SHORT_GRID = FIXTURES / "tiny_short.TextGrid"

#: Word intervals hand-written into ``tiny_long.TextGrid``.
EXPECTED_WORDS: list[tuple[str, float, float]] = [
    ("four", 0.20, 0.48),
    ("score", 0.48, 0.83),
    ("and", 0.83, 0.99),
    ("seven", 1.14, 1.52),
    ("years", 1.52, 1.87),
    ("ago", 1.87, 2.19),
]


# --------------------------------------------------------------------------- #
# parse_textgrid
# --------------------------------------------------------------------------- #


def test_parse_long_form_returns_both_tiers() -> None:
    tiers = parse_textgrid(LONG_GRID)
    assert [t.name for t in tiers] == ["words", "phones"]
    assert len(tiers[0].intervals) == 9  # 6 words + leading, medial, trailing silence
    assert len(tiers[1].intervals) == 25
    assert tiers[0].xmin == 0.0
    assert tiers[0].xmax == pytest.approx(2.4)


def test_parse_short_form_skips_point_tiers() -> None:
    tiers = parse_textgrid(SHORT_GRID)
    assert [t.name for t in tiers] == ["words"]  # the TextTier is dropped
    assert [label for _, _, label in tiers[0].intervals] == ["", "we", "", "hold", "these"]


def test_parse_textgrid_rejects_non_textgrid(tmp_path: Path) -> None:
    bogus = tmp_path / "not.TextGrid"
    bogus.write_text("xmin = 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a text-format TextGrid"):
        parse_textgrid(bogus)


def test_parse_textgrid_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_textgrid(tmp_path / "absent.TextGrid")


def test_utf16_textgrid_is_decoded(tmp_path: Path) -> None:
    """Praat writes UTF-16 with a BOM for non-ASCII texts; we must read it."""
    target = tmp_path / "utf16.TextGrid"
    target.write_bytes(LONG_GRID.read_text(encoding="utf-8").encode("utf-16"))
    assert [t.name for t in parse_textgrid(target)] == ["words", "phones"]


# --------------------------------------------------------------------------- #
# read_textgrid -> Alignment
# --------------------------------------------------------------------------- #


def test_read_textgrid_words_and_times() -> None:
    alignment = read_textgrid(LONG_GRID)
    assert alignment.file_id == "tiny_long"
    assert alignment.n_words == len(EXPECTED_WORDS)
    for word, (text, t_start, t_end) in zip(alignment.words, EXPECTED_WORDS, strict=True):
        assert word.text == text
        assert word.t_start == pytest.approx(t_start)
        assert word.t_end == pytest.approx(t_end)
    assert [w.idx for w in alignment.words] == list(range(len(EXPECTED_WORDS)))


def test_read_textgrid_times_are_monotonic() -> None:
    """The acceptance criterion for the smoke run, asserted on the fixture."""
    words = read_textgrid(LONG_GRID).words
    for word in words:
        assert word.t_end > word.t_start
    for earlier, later in pairwise(words):
        assert later.t_start >= earlier.t_end


def test_read_textgrid_excludes_leading_and_trailing_silence() -> None:
    alignment = read_textgrid(LONG_GRID)
    # Leading silence runs 0.0-0.2 and trailing 2.19-2.4; neither becomes a gap.
    assert alignment.words[0].t_start == pytest.approx(0.20)
    assert alignment.words[-1].t_end == pytest.approx(2.19)
    assert alignment.duration == pytest.approx(2.4)
    assert alignment.speech_duration == pytest.approx(1.99)


def test_read_textgrid_gaps_are_between_consecutive_words() -> None:
    alignment = read_textgrid(LONG_GRID)
    assert len(alignment.gaps) == alignment.n_words - 1
    for gap in alignment.gaps:
        assert gap.next_word_idx == gap.prev_word_idx + 1
        assert gap.t_start == pytest.approx(alignment.words[gap.prev_word_idx].t_end)
        assert gap.t_end == pytest.approx(alignment.words[gap.next_word_idx].t_start)
        assert gap.duration >= 0.0
        assert gap.label == "none"  # punctuation labels are rostrum.text's job
    # Only the gap after "and" is non-zero, and it is exactly 150 ms.
    durations = [round(g.duration, 3) for g in alignment.gaps]
    assert durations == [0.0, 0.0, 0.15, 0.0, 0.0]


def test_read_textgrid_assigns_phones_and_syllables() -> None:
    alignment = read_textgrid(LONG_GRID)
    by_text = {w.text: w for w in alignment.words}
    assert by_text["four"].phones == ["F", "AO1", "R"]
    assert by_text["score"].phones == ["S", "K", "AO1", "R"]
    assert by_text["seven"].phones == ["S", "EH1", "V", "AH0", "N"]
    # Vowel nuclei, so "seven" and "ago" are disyllabic and the rest are not.
    assert [w.n_syllables for w in alignment.words] == [1, 1, 1, 2, 1, 2]
    # The 'sil' phone between "and" and "seven" is attached to no word.
    assert all("sil" not in w.phones for w in alignment.words)


def test_read_textgrid_phone_stress_digits_do_not_break_vowel_lookup() -> None:
    assert "AO" in ARPA_VOWELS and "AO1" not in ARPA_VOWELS


def test_read_textgrid_fills_display_tokens_from_labtext() -> None:
    text = normalise_for_alignment("Four-score and 7 years ago,")
    alignment = read_textgrid(LONG_GRID, text=text, text_id="gettysburg", speaker_id="s01")
    assert [w.display for w in alignment.words] == [
        "Four-score",
        "Four-score",
        "and",
        "7",
        "years",
        "ago,",
    ]
    assert alignment.text_id == "gettysburg"
    assert alignment.speaker_id == "s01"


def test_read_textgrid_rejects_token_count_mismatch() -> None:
    """A mismatch means word indices no longer align across renditions."""
    text = normalise_for_alignment("four score and seven")
    with pytest.raises(ValueError, match="word indices would not match"):
        read_textgrid(LONG_GRID, text=text)


def test_read_textgrid_records_oov_as_unk(tmp_path: Path) -> None:
    grid = tmp_path / "oov.TextGrid"
    grid.write_text(
        LONG_GRID.read_text(encoding="utf-8").replace('text = "score"', 'text = "<unk>"'),
        encoding="utf-8",
    )
    alignment = read_textgrid(grid)
    assert alignment.oov_words == ["<unk>"]
    # The unknown word keeps its slot, so later indices are unaffected.
    assert alignment.n_words == len(EXPECTED_WORDS)
    assert alignment.words[2].text == "and"


def _words_only_grid(path: Path, intervals: list[tuple[float, float, str]]) -> Path:
    """Write a minimal long-format TextGrid holding just a words tier."""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0 ",
        f"xmax = {intervals[-1][1]} ",
        "tiers? <exists> ",
        "size = 1 ",
        "item []: ",
        "    item [1]:",
        '        class = "IntervalTier" ',
        '        name = "words" ',
        "        xmin = 0 ",
        f"        xmax = {intervals[-1][1]} ",
        f"        intervals: size = {len(intervals)} ",
    ]
    for n, (start, end, label) in enumerate(intervals, start=1):
        lines += [
            f"        intervals [{n}]:",
            f"            xmin = {start} ",
            f"            xmax = {end} ",
            f'            text = "{label}" ',
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_textgrid_rejects_overlapping_words(tmp_path: Path) -> None:
    grid = _words_only_grid(
        tmp_path / "overlap.TextGrid",
        [(0.0, 0.5, "four"), (0.3, 0.9, "score")],  # starts before "four" ends
    )
    with pytest.raises(ValueError, match="before word"):
        read_textgrid(grid)


def test_read_textgrid_rejects_inverted_interval(tmp_path: Path) -> None:
    grid = _words_only_grid(
        tmp_path / "inverted.TextGrid", [(0.0, 0.5, "four"), (0.9, 0.7, "score")]
    )
    with pytest.raises(ValueError, match="ends before it starts"):
        read_textgrid(grid)


def test_read_textgrid_rejects_silence_only_grid(tmp_path: Path) -> None:
    grid = tmp_path / "silent.TextGrid"
    source = LONG_GRID.read_text(encoding="utf-8")
    for label in ("four", "score", "and", "seven", "years", "ago"):
        source = source.replace(f'text = "{label}"', 'text = ""')
    grid.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="no word intervals"):
        read_textgrid(grid)


def test_alignment_round_trips_through_dict() -> None:
    from rostrum.types import Alignment

    alignment = read_textgrid(LONG_GRID)
    restored = Alignment.from_dict(alignment.to_dict())
    assert restored.to_dict() == alignment.to_dict()


# --------------------------------------------------------------------------- #
# expand_numerals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1863", ["eighteen", "sixty", "three"]),
        ("1,863", ["eighteen", "sixty", "three"]),
        ("1863,", ["eighteen", "sixty", "three"]),
        ("1900", ["nineteen", "hundred"]),
        ("1805", ["eighteen", "oh", "five"]),
        ("2020", ["twenty", "twenty"]),
        ("2005", ["two", "thousand", "five"]),
        ("2000", ["two", "thousand"]),
        ("87", ["eighty", "seven"]),
        ("7", ["seven"]),
        ("0", ["zero"]),
        ("13", ["thirteen"]),
        ("100", ["one", "hundred"]),
        ("101", ["one", "hundred", "one"]),
        ("3000", ["three", "thousand"]),
        ("1000000", ["one", "million"]),
        ("1st", ["first"]),
        ("2nd", ["second"]),
        ("3rd", ["third"]),
        ("4th", ["fourth"]),
        ("5th", ["fifth"]),
        ("12th", ["twelfth"]),
        ("20th", ["twentieth"]),
        ("21st", ["twenty", "first"]),
        ("3.5", ["three", "point", "five"]),
        ("50%", ["fifty", "percent"]),
        ("$5", ["five", "dollars"]),
        ("word", ["word"]),
        ("nation's", ["nation's"]),
        ("", []),
    ],
)
def test_expand_numerals(token: str, expected: list[str]) -> None:
    assert expand_numerals(token) == expected


def test_expand_numerals_prefer_year_off() -> None:
    assert expand_numerals("1863", prefer_year=False) == [
        "one",
        "thousand",
        "eight",
        "hundred",
        "sixty",
        "three",
    ]


def test_expand_numerals_never_emits_digits_or_hyphens() -> None:
    for token in ("1863", "21st", "3.5", "1,000,000", "B29", "50%"):
        for out in expand_numerals(token):
            assert not any(ch.isdigit() for ch in out), token
            assert "-" not in out, token


# --------------------------------------------------------------------------- #
# normalise_for_alignment
# --------------------------------------------------------------------------- #


def test_normalise_strips_punctuation_and_lowercases() -> None:
    text = normalise_for_alignment("Four score and seven years ago, our fathers...")
    assert text.lab_text == "four score and seven years ago our fathers"
    assert text.display_tokens[-1] == "fathers..."


def test_normalise_keeps_internal_apostrophes() -> None:
    text = normalise_for_alignment("A nation's people don't yield.")
    assert text.lab_tokens == ("a", "nation's", "people", "don't", "yield")


def test_normalise_folds_curly_quotes() -> None:
    """A curly apostrophe would otherwise make "nation's" out-of-vocabulary."""
    text = normalise_for_alignment("the \u201cnation\u2019s\u201d fate\u2014now")
    assert text.lab_tokens == ("the", "nation's", "fate", "now")


def test_normalise_splits_hyphens_and_maps_to_display() -> None:
    text = normalise_for_alignment("Four-score and 7 years ago,")
    assert text.lab_text == "four score and seven years ago"
    assert text.display_tokens == ("Four-score", "and", "7", "years", "ago,")
    assert text.lab_to_display == (0, 0, 1, 2, 3, 4)
    assert text.display_for(0) == "Four-score"
    assert text.display_for(1) == "Four-score"
    assert text.display_for(3) == "7"


def test_normalise_map_is_non_decreasing_and_in_range() -> None:
    text = normalise_for_alignment(
        "In 1863, four-score men -- and 2 women -- met; they spoke of 50% of a nation's hope."
    )
    assert len(text.lab_to_display) == len(text.lab_tokens)
    assert list(text.lab_to_display) == sorted(text.lab_to_display)
    assert max(text.lab_to_display) < len(text.display_tokens)
    assert all(tok for tok in text.lab_tokens)  # no empty lab tokens


def test_normalise_drops_standalone_punctuation() -> None:
    text = normalise_for_alignment("we met -- then parted")
    assert text.lab_tokens == ("we", "met", "then", "parted")
    # "--" survives as a display token but contributes no lab token.
    assert "--" in text.display_tokens


def test_normalise_newlines_are_whitespace() -> None:
    assert normalise_for_alignment("one\ntwo\n\nthree").lab_tokens == ("one", "two", "three")


def test_labtext_rejects_inconsistent_map() -> None:
    with pytest.raises(ValueError, match="lab_to_display"):
        LabText(lab_tokens=("a", "b"), display_tokens=("a",), lab_to_display=(0,))


# --------------------------------------------------------------------------- #
# prepare_corpus
# --------------------------------------------------------------------------- #


def _stub_wav(path: Path) -> Path:
    """Write a byte-valid 16 kHz mono WAV header with no samples.

    ``prepare_corpus`` only copies the file, so no real audio is needed here.
    """
    import struct

    header = b"RIFF" + struct.pack("<I", 36) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
    header += b"data" + struct.pack("<I", 0)
    path.write_bytes(header)
    return path


def test_prepare_corpus_writes_wav_and_lab(tmp_path: Path) -> None:
    wav = _stub_wav(tmp_path / "source.wav")
    corpus = tmp_path / "corpus"
    staged = prepare_corpus([(wav, "Four-score and 7 years ago,", "g01")], corpus)

    assert set(staged) == {"g01"}
    item = staged["g01"]
    assert item.wav_path == corpus / "g01.wav"
    assert item.lab_path == corpus / "g01.lab"
    assert item.wav_path.is_file()
    assert item.source_wav == wav
    # Lowercase, punctuation-free, numerals expanded, LF-terminated.
    assert item.lab_path.read_bytes() == b"four score and seven years ago\n"
    assert item.text.lab_to_display == (0, 0, 1, 2, 3, 4)


def test_prepare_corpus_clean_removes_stale_files(tmp_path: Path) -> None:
    wav = _stub_wav(tmp_path / "source.wav")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    stale = corpus / "old.wav"
    stale.write_bytes(b"stale")

    prepare_corpus([(wav, "we hold these truths", "n01")], corpus, clean=True)
    assert not stale.exists()
    assert (corpus / "n01.wav").is_file()

    prepare_corpus([(wav, "we hold these truths", "n02")], corpus, clean=False)
    assert (corpus / "n01.wav").is_file()  # kept when clean=False
    assert (corpus / "n02.wav").is_file()


def test_prepare_corpus_missing_audio(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="audio for 'x01' not found"):
        prepare_corpus([(tmp_path / "nope.wav", "hello", "x01")], tmp_path / "corpus")


def test_prepare_corpus_rejects_unsafe_id(tmp_path: Path) -> None:
    wav = _stub_wav(tmp_path / "source.wav")
    with pytest.raises(ValueError, match="unsafe file_id"):
        prepare_corpus([(wav, "hello", "../escape")], tmp_path / "corpus")


def test_prepare_corpus_rejects_duplicate_id(tmp_path: Path) -> None:
    wav = _stub_wav(tmp_path / "source.wav")
    with pytest.raises(ValueError, match="duplicate file_id"):
        prepare_corpus([(wav, "hello", "d01"), (wav, "goodbye", "d01")], tmp_path / "corpus")


def test_prepare_corpus_rejects_empty_transcript(tmp_path: Path) -> None:
    wav = _stub_wav(tmp_path / "source.wav")
    with pytest.raises(ValueError, match="normalised to zero tokens"):
        prepare_corpus([(wav, " -- ... ", "e01")], tmp_path / "corpus")


# --------------------------------------------------------------------------- #
# Subprocess guards (no MFA needed: these are the pre-flight checks)
# --------------------------------------------------------------------------- #


def test_run_mfa_rejects_missing_corpus(tmp_path: Path) -> None:
    from rostrum.align.mfa import run_mfa

    with pytest.raises(FileNotFoundError, match="corpus dir does not exist"):
        run_mfa(tmp_path / "absent", tmp_path / "out")


def test_run_mfa_rejects_corpus_without_wav(tmp_path: Path) -> None:
    from rostrum.align.mfa import run_mfa

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "only.lab").write_text("hello\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"no \.wav files"):
        run_mfa(corpus, tmp_path / "out")


def test_validate_rejects_missing_corpus(tmp_path: Path) -> None:
    from rostrum.align.mfa import validate as mfa_validate

    with pytest.raises(FileNotFoundError, match="corpus dir does not exist"):
        mfa_validate(tmp_path / "absent")


def test_oov_parser_pulls_words_from_validate_output() -> None:
    from rostrum.align.mfa import _parse_oov_words

    out = (
        "INFO - Found 3 OOV words: Gettysburg, consecrate, hallow\n"
        "INFO - Out of vocabulary words: Gettysburg\n"
    )
    assert _parse_oov_words(out) == ["consecrate", "gettysburg", "hallow"]


def test_oov_parser_returns_empty_for_clean_output() -> None:
    from rostrum.align.mfa import _parse_oov_words

    assert _parse_oov_words("INFO - Corpus validation complete.\n") == []


def test_mfa_error_is_runtime_error() -> None:
    from rostrum.align.mfa import MFANotAvailableError

    assert issubclass(MFAError, RuntimeError)
    assert issubclass(MFANotAvailableError, MFAError)
