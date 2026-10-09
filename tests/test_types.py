"""Round-trip serialisation tests for :mod:`rostrum.types`.

The whole project depends on an :class:`~rostrum.types.AnalysisResult`
surviving a trip through JSON unchanged: the API returns it, the dashboard
reads it, the evaluation harness scores it, and the determinism test hashes it.
A field that silently fails to round-trip would make two identical runs look
different, or - worse - make a wrong run look identical.
"""

from __future__ import annotations

import json
import math

import pytest

from rostrum.types import (
    TIME_DECIMALS,
    VALUE_SIGFIGS,
    Alignment,
    AnalysisResult,
    CategoryScore,
    Envelope,
    Evidence,
    Gap,
    Region,
    RubricScore,
    Word,
    round_sigfig,
    round_time,
    round_value,
)

# --------------------------------------------------------------------------- #
# Fixtures: one fully populated object of every type
# --------------------------------------------------------------------------- #


@pytest.fixture
def word() -> Word:
    return Word(
        idx=52,
        text="dedicate",
        t_start=41.2117,
        t_end=41.8842,
        phones=["D", "EH1", "D", "AH0", "K", "EY2", "T"],
        n_syllables=3,
        display="dedicate,",
    )


@pytest.fixture
def gap() -> Gap:
    return Gap(
        idx=51,
        prev_word_idx=51,
        next_word_idx=52,
        t_start=40.4091,
        t_end=41.2117,
        label="boundary",
        punctuation=".",
        vad_refined=True,
    )


@pytest.fixture
def alignment(word: Word, gap: Gap) -> Alignment:
    second = Word(idx=53, text="we", t_start=41.9, t_end=42.05, phones=["W", "IY1"], n_syllables=1)
    return Alignment(
        file_id="gettysburg_lv07_rushing_L3",
        text_id="gettysburg",
        speaker_id="librivox_LAW",
        channel_id="librivox_LAW_usb",
        tier="silver",
        words=[word, second],
        gaps=[gap],
        duration=172.4453,
        sample_rate=16000,
        aligner="mfa 3.2.1",
        acoustic_model="english_us_arpa",
        dictionary="english_us_arpa",
        oov_words=["fourscore"],
        audio_sha256="a" * 64,
    )


@pytest.fixture
def envelope() -> Envelope:
    return Envelope(
        text_id="gettysburg",
        fold="loto_jfk",
        unit="word",
        features=["rate_local", "f0_range_st"],
        median={52: [3.9123456789, 7.25], 53: [4.1, 6.0]},
        # Unit 53's f0_range_st spread (0.3) sits below the 0.8 floor, so the
        # floor must take over; every other spread here is above its floor.
        sigma={52: [0.6123456789, 1.5], 53: [0.55, 0.3]},
        sigma_floor={"rate_local": 0.25, "f0_range_st": 0.8},
        n_references={52: 15, 53: 15},
        reference_ids=["gettysburg_lv01", "gettysburg_lv07"],
        channel_ids=["librivox_a", "librivox_LAW_usb"],
    )


@pytest.fixture
def region() -> Region:
    return Region(
        region_id=3,
        t_start=41.2117,
        t_end=45.8842,
        label="rushing",
        category="pace",
        level="L3",
        s=0.6,
        confidence=0.8612345,
        rule_label="rushing",
        clf_label="rushing",
        clf_prob=0.9123,
        envelope_n=15,
        word_idx=[52, 61],
        gap_idx=None,
        words="we can not dedicate we can not consecrate",
        evidence=[
            Evidence(
                feature="rate_syll_s",
                value=6.4123,
                env_median=3.9,
                env_sigma=0.6,
                z=4.1098,
                unit="syll/s",
                n_references=15,
            ),
            Evidence(
                feature="gap_before_s",
                value=0.06,
                env_median=0.82,
                env_sigma=0.21,
                z=-3.6,
                unit="s",
                gap_label="boundary",
                n_references=15,
            ),
        ],
        context="anaphora",
        rationale="Rate is 4.1 sigma above 15 skilled readings.",
        fix="Slow to ~4 syll/s and hold a ~0.8 s pause.",
        peak_z=4.1098,
        mean_excess_z=2.3456,
        refined=True,
        t_start_unrefined=41.0917,
        t_end_unrefined=45.9842,
    )


@pytest.fixture
def rubric() -> RubricScore:
    return RubricScore(
        categories=[
            CategoryScore(
                category="pace",
                score=6.0123,
                band=0.4123,
                deviation_mass=0.5098,
                k=1.0,
                n_regions=1,
            ),
            CategoryScore(
                category="clarity",
                score=0.0,
                suppressed=True,
                suppressed_reason="no same-channel reference",
            ),
        ],
        overall=7.2345,
        overall_band=0.4,
        descriptors={"articulation_rate_syll_s": 6.4123, "pause_rate_per_min": 8.0},
    )


@pytest.fixture
def result(alignment: Alignment, region: Region, rubric: RubricScore) -> AnalysisResult:
    return AnalysisResult(
        file_id="gettysburg_lv07_rushing_L3",
        text_id="gettysburg",
        speaker_id="librivox_LAW",
        channel_id="librivox_LAW_usb",
        origin="injected",
        tier="silver",
        mode="reference",
        mode_reason="transcript matched library text gettysburg",
        alignment=alignment,
        regions=[region],
        score=rubric,
        duration=172.4453,
        speech_duration=170.1234,
        wer=0.0123,
        n_references=15,
        reference_ids=["gettysburg_lv01", "gettysburg_lv07"],
        features_version=1,
        config_hash="b" * 64,
        code_version="v0.1.0-3-gabc1234",
        warnings=["pitch cross-check excluded 3.2% of voiced frames"],
    )


# --------------------------------------------------------------------------- #
# Rounding helpers
# --------------------------------------------------------------------------- #


def test_round_time_uses_millisecond_resolution():
    assert round_time(41.21169) == 41.212
    assert round_time(None) is None


def test_round_value_recurses_and_preserves_types():
    out = round_value({"a": [1.23456789012, {"b": 2.0000000001}], "c": 3, "d": True, "e": "x"})
    assert out == {"a": [1.235, {"b": 2.0}], "c": 3, "d": True, "e": "x"}
    assert isinstance(out["c"], int)
    assert out["d"] is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (6.4123, 6.412),
        (0.0612345, 0.06123),  # same relative precision as the value above
        (3.9, 3.9),
        (1234567.0, 1235000.0),
        (-4.10987, -4.11),
        (0.0, 0.0),
    ],
)
def test_round_sigfig_keeps_four_significant_figures(value: float, expected: float):
    # Significant figures, not decimal places: a 0.06 s gap and a 6.4 syll/s
    # rate must survive hashing with the same relative precision (spec 16).
    assert round_sigfig(value) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_round_sigfig_maps_non_finite_to_none(bad: float):
    assert round_sigfig(bad) is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_round_value_maps_non_finite_to_none(bad: float):
    # JSON has no NaN/Infinity literal; emitting one would break round-tripping.
    assert round_value(bad) is None


def test_rounding_constants_match_invariants():
    # spec 16: times to 3 decimal places, feature values to 4 significant figures.
    assert TIME_DECIMALS == 3
    assert VALUE_SIGFIGS == 4


# --------------------------------------------------------------------------- #
# Per-type round trips
# --------------------------------------------------------------------------- #


def test_word_round_trip(word: Word):
    back = Word.from_dict(word.to_dict())
    assert back.to_dict() == word.to_dict()
    assert back.idx == word.idx
    assert back.phones == word.phones
    assert back.t_start == round_time(word.t_start)


def test_word_duration(word: Word):
    assert word.duration == pytest.approx(0.6725)


def test_gap_round_trip(gap: Gap):
    back = Gap.from_dict(gap.to_dict())
    assert back.to_dict() == gap.to_dict()
    assert back.label == "boundary"
    assert back.vad_refined is True


def test_gap_duration(gap: Gap):
    assert gap.duration == pytest.approx(0.8026)


def test_alignment_round_trip(alignment: Alignment):
    back = Alignment.from_dict(alignment.to_dict())
    assert back.to_dict() == alignment.to_dict()
    assert back.n_words == 2
    assert len(back.gaps) == 1
    assert back.oov_words == ["fourscore"]


def test_alignment_speech_duration_excludes_outer_silence(alignment: Alignment):
    # First word onset to last word offset, not the file duration.
    assert alignment.speech_duration == pytest.approx(42.05 - 41.2117)
    assert alignment.speech_duration < alignment.duration


def test_alignment_handles_empty_word_list():
    empty = Alignment(file_id="silent")
    assert empty.speech_duration == 0.0
    assert Alignment.from_dict(empty.to_dict()).to_dict() == empty.to_dict()


def test_envelope_round_trip_restores_int_keys(envelope: Envelope):
    # JSON object keys are strings; from_dict must cast them back or every
    # lookup by word index would miss.
    payload = json.loads(json.dumps(envelope.to_dict()))
    assert set(payload["median"]) == {"52", "53"}

    back = Envelope.from_dict(payload)
    assert set(back.median) == {52, 53}
    assert set(back.n_references) == {52, 53}
    assert back.to_dict() == envelope.to_dict()


def test_envelope_effective_sigma_applies_floor(envelope: Envelope):
    # sigma 0.612 is above the 0.25 floor, so it is used as is...
    assert envelope.effective_sigma(52, "rate_local") == pytest.approx(0.6123456789)
    # ...while a sigma below the floor is lifted to it.
    assert envelope.effective_sigma(53, "f0_range_st") == pytest.approx(0.8)


def test_envelope_z_matches_the_formula(envelope: Envelope):
    # z = (x - median) / max(1.4826*MAD, sigma_floor)
    z = envelope.z(52, "rate_local", 6.4)
    assert z == pytest.approx((6.4 - 3.9123456789) / 0.6123456789)


def test_evidence_round_trip(region: Region):
    ev = region.evidence[1]
    back = Evidence.from_dict(ev.to_dict())
    assert back.to_dict() == ev.to_dict()
    assert back.gap_label == "boundary"
    assert back.z == pytest.approx(-3.6)


def test_evidence_gap_label_may_be_absent(region: Region):
    ev = region.evidence[0]
    assert ev.to_dict()["gap_label"] is None
    assert Evidence.from_dict(ev.to_dict()).gap_label is None


def test_region_round_trip(region: Region):
    back = Region.from_dict(region.to_dict())
    assert back.to_dict() == region.to_dict()
    assert len(back.evidence) == 2
    assert back.word_idx == [52, 61]
    assert back.duration == pytest.approx(45.884 - 41.212, abs=1e-3)


def test_region_keeps_unrefined_boundaries_for_the_ablation(region: Region):
    d = region.to_dict()
    assert d["t_start_unrefined"] == 41.092
    assert d["t_end_unrefined"] == 45.984
    assert Region.from_dict(d).refined is True


def test_region_round_trip_without_refinement(region: Region):
    region.refined = False
    region.t_start_unrefined = None
    region.t_end_unrefined = None
    back = Region.from_dict(region.to_dict())
    assert back.t_start_unrefined is None
    assert back.t_end_unrefined is None


def test_category_score_round_trip(rubric: RubricScore):
    cs = rubric.categories[1]
    back = CategoryScore.from_dict(cs.to_dict())
    assert back.to_dict() == cs.to_dict()
    assert back.suppressed is True
    assert back.suppressed_reason == "no same-channel reference"


def test_rubric_round_trip_and_lookup(rubric: RubricScore):
    back = RubricScore.from_dict(rubric.to_dict())
    assert back.to_dict() == rubric.to_dict()
    pace = back.get("pace")
    assert pace is not None
    # 6.0123 serialises to 4 significant figures, so it returns as 6.012.
    assert pace.score == pytest.approx(6.012)
    assert back.get("intonation") is None


# --------------------------------------------------------------------------- #
# AnalysisResult: the contract that matters
# --------------------------------------------------------------------------- #


def test_analysis_result_round_trip(result: AnalysisResult):
    back = AnalysisResult.from_dict(result.to_dict())
    assert back.to_dict() == result.to_dict()


def test_analysis_result_survives_real_json(result: AnalysisResult):
    # allow_nan=False is the point: a NaN anywhere must already have become None.
    payload = json.dumps(result.to_dict(), sort_keys=True, allow_nan=False)
    back = AnalysisResult.from_dict(json.loads(payload))
    assert json.dumps(back.to_dict(), sort_keys=True, allow_nan=False) == payload


def test_analysis_result_nested_objects_are_rebuilt_as_dataclasses(result: AnalysisResult):
    back = AnalysisResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert isinstance(back.alignment, Alignment)
    assert isinstance(back.alignment.words[0], Word)
    assert isinstance(back.alignment.gaps[0], Gap)
    assert isinstance(back.regions[0], Region)
    assert isinstance(back.regions[0].evidence[0], Evidence)
    assert isinstance(back.score, RubricScore)
    assert isinstance(back.score.categories[0], CategoryScore)


def test_analysis_result_handles_reject_mode():
    rejected = AnalysisResult(
        file_id="mismatch",
        mode="reject",
        mode_reason="wer=0.41 > 0.25",
        wer=0.41,
        alignment=None,
        score=None,
    )
    back = AnalysisResult.from_dict(json.loads(json.dumps(rejected.to_dict())))
    assert back.mode == "reject"
    assert back.alignment is None
    assert back.score is None
    assert back.to_dict() == rejected.to_dict()


def test_analysis_result_carries_the_schema_version(result: AnalysisResult):
    assert result.to_dict()["schema_version"] == AnalysisResult.SCHEMA_VERSION


def test_regions_are_serialised_in_temporal_order(region: Region):
    # Two runs may discover regions in a different order; the serialised form
    # must still be identical, or the determinism hash is meaningless.
    late = Region(region_id=9, t_start=90.0, t_end=91.0, label="monotone", category="intonation")
    early = Region(region_id=1, t_start=2.0, t_end=3.0, label="low_energy", category="energy")

    a = AnalysisResult(file_id="f", regions=[late, region, early]).to_dict()
    b = AnalysisResult(file_id="f", regions=[early, late, region]).to_dict()

    assert [r["t_start"] for r in a["regions"]] == [2.0, 41.212, 90.0]
    assert a == b


def test_to_events_emits_sed_eval_rows(result: AnalysisResult):
    events = result.to_events()
    assert events == [("gettysburg_lv07_rushing_L3", 41.212, 45.884, "rushing")]


def test_from_dict_ignores_unknown_keys(result: AnalysisResult):
    # Forward compatibility: a newer report must not crash an older reader.
    payload = result.to_dict()
    payload["a_field_from_the_future"] = 42
    assert AnalysisResult.from_dict(payload).to_dict() == result.to_dict()


def test_to_dict_is_idempotent(result: AnalysisResult):
    once = result.to_dict()
    twice = AnalysisResult.from_dict(once).to_dict()
    thrice = AnalysisResult.from_dict(twice).to_dict()
    assert once == twice == thrice
