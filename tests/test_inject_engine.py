"""Unit tests for the injection engine (``rostrum.inject.engine``).

As in :mod:`tests.test_inject_ops`, everything runs on a **synthetic** utterance
whose ground truth is known by construction, so no dataset audio and no MFA run
is needed. The fixture is built to satisfy the config's own span rules rather
than to dodge them: three six-word phrases separated by 550 ms ``boundary`` gaps
(long enough to shorten, and long enough to donate room tone), 100 ms ``none``
gaps inside each phrase (short enough to host a hesitation), real function words
in the word tier, and 8 semitones of F0 excursion so the ``monotone`` range gate
passes.

The three assertions the prompt asks for have a test each:

* the same seed produces a byte-identical WAV and byte-identical JSON;
* composite flaws never overlap;
* every flaw time lies inside the edited audio's duration.
"""

from __future__ import annotations

import json
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from rostrum.config import load_config
from rostrum.inject import engine
from rostrum.inject.engine import CleanFile, NoValidSpanError, Span
from rostrum.io.audio import write_wav16
from rostrum.types import Alignment
from tests.test_inject_ops import make_utterance

SR = 16_000

SENTENCE = (  # noqa: SIM905 - a sentence, so the phrase comments line up
    "the people of this land are free "  # phrase 0, words 0-5
    "and the cause of freedom lives "  # phrase 1, words 6-11
    "but our fathers shall not perish "  # phrase 2, words 12-17
    "and the nation shall not vanish"  # phrase 3, words 18-23
).split()
"""Word tier of the fixture: every phrase holds function words (``the``, ``of``,
``and``, ``but``, ``our``) and content words, which is what
``misplaced_stress`` needs."""

BOUNDARY_AFTER = (5, 11, 17)
"""Phrase-final word indices; the gaps after these are the ``boundary`` gaps."""

FUNCTION_WORDS = frozenset({"the", "of", "this", "are", "and", "but", "our", "shall", "not"})
"""Which words of :data:`SENTENCE` are closed-class. Asserted against the config
list in :func:`test_the_fixture_uses_real_function_words`, so the fixture cannot
drift away from the rule it is meant to exercise."""


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cfg():
    """The real ``configs/*.yaml``: the span rules under test are the shipped ones."""
    return load_config(apply_seeds=False)


def make_clean(
    *,
    n_words: int = 24,
    content_word_s: float = 0.40,
    function_word_s: float = 0.20,
    short_gap_s: float = 0.10,
    boundary_gap_s: float = 0.55,
    depth_st: float = 8.0,
    seed: int = 11,
) -> CleanFile:
    """Build the synthetic clean recording described in the module docstring.

    Content words are made twice as long as function words, which is what gives
    the ``misplaced_stress`` rule a content word to find: on a fixture whose
    words are interchangeable, the most prominent word of a phrase is decided by
    noise, and half the time it is an article.

    Args:
        n_words: Word count; the default 24 gives four six-word phrases, which
            is enough room for a composite's 3-5 flaws a second apart.
        content_word_s: Duration of each content word.
        function_word_s: Duration of each function word.
        short_gap_s: Within-phrase gap, labelled ``none``.
        boundary_gap_s: Phrase-final gap, labelled ``boundary``.
        depth_st: F0 excursion, which drives the ``monotone`` range gate.
        seed: Noise seed.

    Returns:
        A :class:`CleanFile` whose alignment carries real word texts and the
        gap labels ``rostrum.text`` would have produced for this punctuation.
    """
    gaps = [boundary_gap_s if i in BOUNDARY_AFTER else short_gap_s for i in range(n_words - 1)]
    durations = [
        function_word_s if word in FUNCTION_WORDS else content_word_s for word in SENTENCE[:n_words]
    ]
    audio, alignment = make_utterance(
        word_durations=durations,
        gap_durations=gaps,
        depth_st=depth_st,
        seed=seed,
    )
    words = [
        replace(w, text=SENTENCE[i], display=SENTENCE[i], n_syllables=1)
        for i, w in enumerate(alignment.words)
    ]
    labelled = [
        replace(g, label="boundary" if g.idx in BOUNDARY_AFTER else "none") for g in alignment.gaps
    ]
    alignment = replace(
        alignment,
        file_id="s0_synth",
        text_id="s0_synth_text",
        speaker_id="synth",
        tier="silver",
        words=words,
        gaps=labelled,
    )
    return CleanFile(audio=audio, sr=SR, alignment=alignment)


@pytest.fixture(scope="module")
def clean() -> CleanFile:
    """The default synthetic clean recording."""
    return make_clean()


# --------------------------------------------------------------------------- #
# Severity and parameters
# --------------------------------------------------------------------------- #


def test_severity_ladder_matches_the_spec(cfg):
    assert [engine.severity_s(level, cfg) for level in ("L0", "L1", "L2", "L3", "L4")] == [
        0.0,
        0.15,
        0.35,
        0.60,
        0.90,
    ]
    with pytest.raises(KeyError):
        engine.severity_s("L9", cfg)


@pytest.mark.parametrize(
    ("label", "level", "name", "expected"),
    [
        ("rushing", "L4", "factor", 1 - 0.45 * 0.90),
        ("dragging", "L4", "factor", 1 + 0.8 * 0.90),
        ("missing_pause", "L4", "keep_fraction", 1 - 0.95 * 0.90),
        ("hesitation_pause", "L4", "duration_s", 0.3 + 1.7 * 0.90),
        ("monotone", "L4", "exponent", 1 - 0.9 * 0.90),
        ("misplaced_stress", "L4", "f0_shift_st", 4.0 * 0.90),
        ("low_energy", "L4", "end_gain_db", -18.0 * 0.90),
    ],
)
def test_resolve_params_evaluates_the_config_formula(cfg, label, level, name, expected):
    params, formulas, _ = engine.resolve_params(label, level, cfg)
    assert params[name] == pytest.approx(expected)
    assert "s" in formulas[name] or formulas[name].replace(".", "").isdigit()


def test_resolve_params_clamps_to_the_configured_bounds(cfg):
    # hesitation duration is 0.3 + 1.7*s = 1.83 s at L4, inside [0.30, 2.00];
    # pushing the formula past the bound must clamp rather than pass through.
    harsh = load_config(
        apply_seeds=False,
        overrides={
            "injection": {"labels": {"hesitation_pause": {"params": {"duration_s": "5*s"}}}}
        },
    )
    params, _, op_params = engine.resolve_params("hesitation_pause", "L4", harsh)
    assert params["duration_s"] == pytest.approx(2.00)
    assert op_params["dur_s"] == pytest.approx(2.00)


def test_resolve_params_renames_to_the_op_keywords(cfg):
    _, _, op_params = engine.resolve_params("low_energy", "L2", cfg)
    assert op_params["db_end"] == pytest.approx(-18.0 * 0.35)
    assert op_params["ramp"] == "linear_db"
    assert op_params["crossfade_ms"] == pytest.approx(10.0)


def test_resolve_params_gives_the_content_word_the_inverse_duration(cfg):
    params, _, op_params = engine.resolve_params("misplaced_stress", "L3", cfg)
    assert op_params["content_duration_factor"] == pytest.approx(1.0 / params["duration_factor"])


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def test_file_seed_is_stable_and_id_specific():
    assert engine.file_seed("s1_lv07_rushing_L3", 1337) == engine.file_seed(
        "s1_lv07_rushing_L3", 1337
    )
    assert engine.file_seed("s1_lv07_rushing_L3", 1337) != engine.file_seed(
        "s1_lv07_rushing_L4", 1337
    )
    assert engine.file_seed("x", 1337) != engine.file_seed("x", 7)
    assert 0 <= engine.file_seed("x", 1337) < 2**63


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #


def test_the_fixture_uses_real_function_words(cfg):
    """The fixture's closed-class words must be the config's, or it tests nothing."""
    configured = set(cfg.injection["labels"]["misplaced_stress"]["valid_span"]["function_words"])
    assert configured >= FUNCTION_WORDS
    assert {w for w in SENTENCE if w in configured} == set(SENTENCE) & FUNCTION_WORDS


def test_phrases_split_at_boundary_gaps_only(clean):
    assert engine.phrases(clean.alignment) == ((0, 5), (6, 11), (12, 17), (18, 23))


def test_phrases_ignore_minor_gaps(clean):
    with_comma = replace(
        clean.alignment,
        gaps=[replace(g, label="minor" if g.idx == 2 else g.label) for g in clean.alignment.gaps],
    )
    assert engine.phrases(with_comma) == ((0, 5), (6, 11), (12, 17), (18, 23))


def test_phrases_of_an_empty_alignment_are_empty():
    assert engine.phrases(Alignment(file_id="empty")) == ()


# --------------------------------------------------------------------------- #
# Span selection
# --------------------------------------------------------------------------- #


def test_pace_spans_respect_length_phrase_and_margins(clean, cfg):
    spans = engine.candidate_spans("rushing", clean, cfg)
    assert spans
    phrase_of = {i: p for p in engine.phrases(clean.alignment) for i in range(p[0], p[1] + 1)}
    for span in spans:
        lo, hi = span.word_range
        assert 3 <= hi - lo + 1 <= 10
        assert phrase_of[lo] == phrase_of[hi], "a pace span may not cross a boundary gap"
        assert lo >= 2 and hi <= clean.alignment.n_words - 3
    assert engine.candidate_spans("dragging", clean, cfg) == spans


def test_missing_pause_takes_only_long_boundary_gaps(clean, cfg):
    spans = engine.candidate_spans("missing_pause", clean, cfg)
    assert [s.gap_idx for s in spans] == list(BOUNDARY_AFTER)
    for span in spans:
        gap = next(g for g in clean.alignment.gaps if g.idx == span.gap_idx)
        assert gap.label == "boundary"
        assert gap.t_end - gap.t_start >= 0.30


def test_hesitation_pause_takes_only_short_unmarked_gaps(clean, cfg):
    spans = engine.candidate_spans("hesitation_pause", clean, cfg)
    assert spans
    for span in spans:
        gap = next(g for g in clean.alignment.gaps if g.idx == span.gap_idx)
        assert gap.label == "none"
        assert gap.t_end - gap.t_start < 0.15


def test_monotone_spans_are_whole_phrases_that_pass_the_voicing_gate(clean, cfg):
    spans = engine.candidate_spans("monotone", clean, cfg)
    assert [s.word_range for s in spans] == [(0, 5), (6, 11), (12, 17), (18, 23)]


def test_monotone_rejects_an_already_flat_phrase(cfg):
    flat = make_clean(depth_st=0.2)
    assert engine.candidate_spans("monotone", flat, cfg) == ()


def test_misplaced_stress_pairs_a_function_word_with_the_prominent_word(clean, cfg):
    function_words = set(
        cfg.injection["labels"]["misplaced_stress"]["valid_span"]["function_words"]
    )
    spans = engine.candidate_spans("misplaced_stress", clean, cfg)
    assert spans
    phrase_of = {i: p for p in engine.phrases(clean.alignment) for i in range(p[0], p[1] + 1)}
    for span in spans:
        assert span.kind == "word_pair"
        fn_idx, ct_idx = span.words
        assert clean.alignment.words[fn_idx].text in function_words
        assert clean.alignment.words[ct_idx].text not in function_words
        assert phrase_of[fn_idx] == phrase_of[ct_idx]


def test_low_energy_spans_end_at_a_boundary_gap(clean, cfg):
    spans = engine.candidate_spans("low_energy", clean, cfg)
    assert spans
    for span in spans:
        lo, hi = span.word_range
        assert hi in BOUNDARY_AFTER
        assert 4 <= hi - lo + 1 <= 12


def test_no_boundary_gap_means_no_span_for_the_labels_that_need_one(clean, cfg):
    unlabelled = CleanFile(
        audio=clean.audio,
        sr=clean.sr,
        alignment=replace(
            clean.alignment, gaps=[replace(g, label="none") for g in clean.alignment.gaps]
        ),
    )
    assert engine.candidate_spans("missing_pause", unlabelled, cfg) == ()
    assert engine.candidate_spans("low_energy", unlabelled, cfg) == ()
    with pytest.raises(NoValidSpanError):
        engine.select_span("missing_pause", unlabelled, np.random.default_rng(0), cfg)


def test_select_span_is_a_seeded_draw_from_the_candidates(clean, cfg):
    pool = engine.candidate_spans("rushing", clean, cfg)
    first = engine.select_span("rushing", clean, np.random.default_rng(1337), cfg)
    again = engine.select_span("rushing", clean, np.random.default_rng(1337), cfg)
    other = engine.select_span("rushing", clean, np.random.default_rng(999), cfg)
    assert first == again
    assert first in pool and other in pool
    drawn = {engine.select_span("rushing", clean, np.random.default_rng(s), cfg) for s in range(40)}
    assert len(drawn) > 1, "the draw must actually explore the candidate list"


def test_word_prominence_is_one_score_per_word(clean, cfg):
    scores = engine.word_prominence(clean, (0, 5), cfg)
    assert scores.shape == (6,)
    assert 0 <= engine.most_prominent_word(clean, (0, 5), cfg) <= 5


# --------------------------------------------------------------------------- #
# generate_single
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def singles(clean, cfg) -> dict[str, engine.InjectionResult]:
    """One L3 file per injectable label, generated once for the whole module."""
    return {
        label: engine.generate_single(clean, label, "L3", cfg=cfg)
        for label in engine.INJECTABLE_LABELS
    }


def test_every_injectable_label_generates_a_file(singles, clean):
    assert set(singles) == set(engine.INJECTABLE_LABELS)
    for label, result in singles.items():
        assert result.file_id == f"{clean.file_id}_{label}_L3"
        assert result.audio.size > 0
        assert result.label["flaws"][0]["label"] == label
        assert result.label["level"] if False else result.label["flaws"][0]["level"] == "L3"


def test_flaw_times_lie_inside_the_edited_audio(singles):
    """The prompt's third assertion, for every label."""
    for label, result in singles.items():
        duration = result.audio.size / result.sr
        # The label rounds to milliseconds (invariant 2), so half a millisecond
        # of disagreement with the sample count is the exact expectation.
        assert result.label["duration"] == pytest.approx(duration, abs=1e-3)
        for flaw in result.label["flaws"]:
            assert 0.0 <= flaw["t_start"] < flaw["t_end"] <= duration + 1e-3, label


def test_the_time_map_describes_the_audio_it_produced(singles):
    for result in singles.values():
        assert result.time_map.new_duration == pytest.approx(
            result.audio.size / result.sr, abs=1e-3
        )
        assert result.alignment.duration == pytest.approx(result.audio.size / result.sr, abs=1e-3)


def test_label_json_carries_the_spec_7_1_fields(singles, clean, cfg):
    result = singles["rushing"]
    label = result.label
    assert label["source_file"] == clean.file_id
    assert label["text_id"] == clean.alignment.text_id
    assert label["origin"] == "injected"
    assert label["tier"] == "silver"
    assert label["seed"] == engine.file_seed(result.file_id, cfg.seed)
    assert label["config_hash"] == cfg.hash
    assert label["qc"] is None, "qc is filled by the QC pass, not claimed here"
    flaw = label["flaws"][0]
    assert flaw["s"] == pytest.approx(0.60)
    assert flaw["method"] == "psola_lengthen"
    assert flaw["params"]["factor"] == pytest.approx(0.73)
    assert flaw["formulas"]["factor"] == "1 - 0.45 * s"
    assert flaw["word_idx"] == list(Span(**label["flaws"][0]["span"]).word_range)


def test_flaw_times_are_rounded_to_milliseconds(singles):
    for result in singles.values():
        for flaw in result.label["flaws"]:
            assert flaw["t_start"] == round(flaw["t_start"], 3)
            assert flaw["t_end"] == round(flaw["t_end"], 3)


def test_pace_flaw_span_is_scaled_by_the_factor(clean, cfg):
    span = Span(kind="words", words=(3, 7))
    result = engine.generate_single(clean, "rushing", "L4", cfg=cfg, span=span)
    a, b = span.clean_interval(clean.alignment)
    flaw = result.label["flaws"][0]
    factor = flaw["params"]["factor"]
    assert flaw["t_end"] - flaw["t_start"] == pytest.approx(factor * (b - a), abs=0.01)
    assert result.time_map.delta == pytest.approx((factor - 1.0) * (b - a), abs=0.01)


def test_remapped_word_times_shift_by_exactly_the_delta(clean, cfg):
    """``CLAUDE.md`` testing expectation: later timestamps shift by the computed delta."""
    span = Span(kind="words", words=(3, 7))
    result = engine.generate_single(clean, "dragging", "L2", cfg=cfg, span=span)
    delta = result.time_map.delta
    for before, after in zip(clean.alignment.words, result.alignment.words, strict=True):
        if before.idx < 3:
            assert after.t_start == pytest.approx(before.t_start, abs=1e-3)
        elif before.idx > 7:
            assert after.t_start == pytest.approx(before.t_start + delta, abs=1e-3)
            assert after.t_end == pytest.approx(before.t_end + delta, abs=1e-3)
    assert result.alignment.n_words == clean.alignment.n_words, "no op adds or drops a word"


def test_remapped_gaps_agree_with_the_remapped_words(singles):
    for result in singles.values():
        by_idx = {w.idx: w for w in result.alignment.words}
        for gap in result.alignment.gaps:
            assert gap.t_start == pytest.approx(by_idx[gap.prev_word_idx].t_end)
            assert gap.t_end == pytest.approx(by_idx[gap.next_word_idx].t_start)
            assert gap.t_end >= gap.t_start - 1e-9


def test_hesitation_grows_the_file_by_the_requested_duration(clean, cfg):
    result = engine.generate_single(clean, "hesitation_pause", "L2", cfg=cfg)
    flaw = result.label["flaws"][0]
    added = (result.audio.size / result.sr) - clean.duration
    assert added == pytest.approx(flaw["params"]["duration_s"], abs=1e-3)
    # The label covers the inserted silence, not just the instant it went in.
    assert flaw["t_end"] - flaw["t_start"] >= flaw["params"]["duration_s"] - 1e-3


def test_gap_flaws_record_the_gap_and_its_neighbours(singles):
    flaw = singles["missing_pause"].label["flaws"][0]
    assert flaw["gap_label"] == "boundary"
    assert flaw["gap_idx"] in BOUNDARY_AFTER
    assert flaw["word_idx"] == [flaw["gap_idx"], flaw["gap_idx"] + 1]


def test_stress_flaw_names_both_words(singles):
    flaw = singles["misplaced_stress"].label["flaws"][0]
    assert flaw["function_word_idx"] != flaw["content_word_idx"]
    lo, hi = flaw["word_idx"]
    assert {lo, hi} == {flaw["function_word_idx"], flaw["content_word_idx"]}


def test_generate_single_rejects_a_non_injectable_label_or_l0(clean, cfg):
    with pytest.raises(ValueError, match="not injectable"):
        engine.generate_single(clean, "mumbling", "L2", cfg=cfg)
    with pytest.raises(ValueError, match="L0"):
        engine.generate_single(clean, "rushing", "L0", cfg=cfg)


# --------------------------------------------------------------------------- #
# Determinism (the prompt's first assertion)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", list(engine.INJECTABLE_LABELS))
def test_the_same_seed_gives_a_byte_identical_wav_and_json(clean, cfg, tmp_path, label):
    first = engine.generate_single(clean, label, "L3", cfg=cfg)
    second = engine.generate_single(clean, label, "L3", cfg=cfg)

    wav_a = write_wav16(tmp_path / f"{label}_a.wav", first.audio, first.sr)
    wav_b = write_wav16(tmp_path / f"{label}_b.wav", second.audio, second.sr)
    assert wav_a.read_bytes() == wav_b.read_bytes()
    assert json.dumps(first.label, sort_keys=True) == json.dumps(second.label, sort_keys=True)
    assert first.alignment.to_dict() == second.alignment.to_dict()


def test_a_different_seed_changes_the_file(clean, cfg):
    default = engine.generate_single(clean, "rushing", "L3", cfg=cfg)
    forced = engine.generate_single(clean, "rushing", "L3", seed=424242, cfg=cfg)
    assert forced.label["seed"] == 424242
    assert forced.label["flaws"][0]["span"] != default.label["flaws"][0][
        "span"
    ] or not np.array_equal(forced.audio, default.audio)


def test_composite_is_byte_identical_across_runs(clean, cfg, tmp_path):
    first = engine.generate_composite(clean, "L2", 1, cfg=cfg)
    second = engine.generate_composite(clean, "L2", 1, cfg=cfg)
    wav_a = write_wav16(tmp_path / "c_a.wav", first.audio, first.sr)
    wav_b = write_wav16(tmp_path / "c_b.wav", second.audio, second.sr)
    assert wav_a.read_bytes() == wav_b.read_bytes()
    assert json.dumps(first.label, sort_keys=True) == json.dumps(second.label, sort_keys=True)


# --------------------------------------------------------------------------- #
# generate_composite (the prompt's second assertion)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def composite(clean, cfg) -> engine.InjectionResult:
    """One L4 composite, generated once for the whole module."""
    return engine.generate_composite(clean, "L4", 1, cfg=cfg)


def test_composite_holds_three_to_five_distinct_flaws(composite, cfg):
    flaws = composite.label["flaws"]
    lo, hi = cfg.injection["composite"]["n_flaws"]
    assert lo <= len(flaws) <= hi
    labels = [f["label"] for f in flaws]
    assert len(set(labels)) == len(labels), "composite.distinct_labels"
    assert composite.label["n_flaws"] == len(flaws)
    assert all(f["level"] == "L4" for f in flaws)


def test_composite_flaws_never_overlap(composite):
    """The prompt's second assertion, in the edited file's own timeline."""
    flaws = sorted(composite.label["flaws"], key=lambda f: f["t_start"])
    for earlier, later in pairwise(flaws):
        assert earlier["t_end"] <= later["t_start"] + 1e-9, (earlier, later)


def test_composite_spans_are_a_second_apart_on_the_clean_file(composite, clean, cfg):
    min_sep = float(cfg.injection["composite"]["min_separation_s"])
    spans = sorted((f["clean_t_start"], f["clean_t_end"]) for f in composite.label["flaws"])
    for (_, end), (start, _) in pairwise(spans):
        assert start - end >= min_sep - 1e-6


def test_composite_flaw_times_lie_inside_the_audio(composite):
    duration = composite.audio.size / composite.sr
    for flaw in composite.label["flaws"]:
        assert 0.0 <= flaw["t_start"] < flaw["t_end"] <= duration + 1e-3


def test_composite_time_map_covers_the_whole_recording(composite, clean):
    tmap = composite.time_map
    assert tmap.old_duration == pytest.approx(clean.duration, abs=1e-6)
    assert tmap.new_duration == pytest.approx(composite.audio.size / composite.sr, abs=1e-3)
    assert tmap.segments[0].old_start == pytest.approx(0.0)


def test_composite_file_id_is_deterministic(clean, cfg):
    assert (
        engine.generate_composite(clean, "L2", 3, cfg=cfg).file_id
        == f"{clean.file_id}_composite3_L2"
    )


def test_composite_fails_loudly_when_the_file_is_too_short(cfg):
    tiny = make_clean(n_words=6)
    with pytest.raises(NoValidSpanError, match="placed only"):
        engine.generate_composite(tiny, "L4", 1, cfg=cfg)


def test_forced_flaw_count_is_honoured(clean, cfg):
    result = engine.generate_composite(clean, "L2", 1, n_flaws=3, cfg=cfg)
    assert len(result.label["flaws"]) == 3


# --------------------------------------------------------------------------- #
# events.tsv
# --------------------------------------------------------------------------- #


def test_events_rows_are_one_per_flaw(composite):
    rows = engine.events_rows(composite.label)
    assert len(rows) == len(composite.label["flaws"])
    for (filename, onset, offset, event), flaw in zip(rows, composite.label["flaws"], strict=True):
        assert filename == f"{composite.file_id}.wav"
        assert (onset, offset, event) == (flaw["t_start"], flaw["t_end"], flaw["label"])


def test_events_tsv_is_sorted_and_tab_separated(singles, composite):
    text = engine.format_events_tsv([composite.label, *(r.label for r in singles.values())])
    lines = text.rstrip("\n").split("\n")
    assert all(len(line.split("\t")) == 4 for line in lines)
    assert lines == sorted(lines)
    assert text.endswith("\n")
    expected = sum(len(r.label["flaws"]) for r in singles.values()) + len(composite.label["flaws"])
    assert len(lines) == expected


def test_events_tsv_header_is_optional(composite):
    with_header = engine.format_events_tsv([composite.label], header=True)
    assert with_header.split("\n")[0] == "filename\tonset\toffset\tevent_label"
    assert "filename" not in engine.format_events_tsv([composite.label]).split("\n")[0]


# --------------------------------------------------------------------------- #
# Spans as values
# --------------------------------------------------------------------------- #


def test_span_validates_its_own_shape():
    with pytest.raises(ValueError, match="gap_idx"):
        Span(kind="gap")
    with pytest.raises(ValueError, match="word indices"):
        Span(kind="words")
    assert Span(kind="word_pair", words=(9, 4)).word_range == (4, 9)
    assert Span(kind="gap", gap_idx=2).op_span == 2
    assert Span(kind="words", words=(1, 3)).op_span == (1, 3)


def test_span_clean_interval_matches_the_alignment(clean):
    words = Span(kind="words", words=(2, 4)).clean_interval(clean.alignment)
    assert words == (clean.alignment.words[2].t_start, clean.alignment.words[4].t_end)
    gap = next(g for g in clean.alignment.gaps if g.idx == 5)
    assert Span(kind="gap", gap_idx=5).clean_interval(clean.alignment) == (gap.t_start, gap.t_end)
