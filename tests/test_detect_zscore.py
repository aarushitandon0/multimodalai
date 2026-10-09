"""Tests for the z-scores and the spec 10.3 trigger table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.detect.zscore import (
    Signal,
    SkippedSignal,
    combined_deviation,
    envelope_z,
    trigger_signals,
    zscores,
)
from rostrum.envelope.build import apply_sigma_floors, build_envelope
from tests.test_envelope_build import RATE, gap_table, tables_for, word_table

N_WORDS = 12


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def reference_tables(n: int = 5, *, rate: float = 5.0, with_gaps: bool = True) -> dict:
    """Five readers who differ slightly, so every sigma is positive."""
    rng = np.random.default_rng(0)
    out = {}
    for r in range(n):
        gaps = None
        if with_gaps:
            base = np.array([0.5 if i % 3 == 0 else 0.06 for i in range(N_WORDS - 1)], dtype=float)
            gaps = base + rng.normal(0.0, 0.02, size=base.size)
        out[f"r{r}"] = tables_for(f"r{r}", rate + rng.normal(0.0, 0.2, size=N_WORDS), gaps=gaps)
    return out


def built(cfg, tables, units=("word", "gap")):
    envelope = build_envelope("T", sorted(tables), tables, units=units, config=cfg)
    floors = {unit: dict.fromkeys(envelope.unit(unit).features, 0.05) for unit in units}
    return apply_sigma_floors(envelope, floors)


# --------------------------------------------------------------------------- #
# Layer 1: arithmetic
# --------------------------------------------------------------------------- #


def test_z_is_value_minus_median_over_effective_sigma(cfg):
    tables = reference_tables(with_gaps=False)
    envelope = built(cfg, tables, units=("word",))
    unit = envelope.unit("word")

    participant = word_table("p", np.full(N_WORDS, 5.1))
    # clip_abs_z off, so the identity is tested rather than the clip.
    result = zscores(participant, unit, config=cfg, clip_abs_z=1e9)

    col = unit.column(RATE)
    for idx in result.z.index:
        median = unit.envelope.median[idx][col]
        sigma = unit.envelope.effective_sigma(idx, RATE)
        assert result.z.loc[idx, RATE] == pytest.approx((5.1 - median) / sigma)
    assert result.n_clipped == 0


def test_a_zero_scale_gives_nan_not_infinity(cfg):
    """Unanimous readers with no floor mean "no information", not "infinitely wrong"."""
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0)) for r in range(5)}
    envelope = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    apply_sigma_floors(envelope, {"word": {RATE: 0.0}})

    result = zscores(word_table("p", np.full(N_WORDS, 9.0)), envelope.unit("word"), config=cfg)
    assert result.z[RATE].isna().all()
    assert np.isfinite(result.z[RATE].to_numpy(dtype=float)).sum() == 0
    assert result.no_scale > 0


def test_z_is_clipped_and_the_clipping_is_reported(cfg):
    tables = reference_tables(with_gaps=False)
    envelope = built(cfg, tables, units=("word",))
    result = zscores(word_table("p", np.full(N_WORDS, 500.0)), envelope.unit("word"), config=cfg)
    limit = float(cfg.detection["deviation"]["clip_abs_z"])
    assert result.z[RATE].max() == pytest.approx(limit)
    assert result.n_clipped == N_WORDS


def test_unit_mismatch_is_rejected(cfg):
    tables = reference_tables()
    envelope = built(cfg, tables)
    with pytest.raises(ValueError, match="but the envelope is for"):
        zscores(gap_table("p", np.full(N_WORDS - 1, 0.3)), envelope.unit("word"), config=cfg)


def test_combined_deviation_weights_by_lambda(cfg):
    tables = reference_tables(with_gaps=False)
    envelope = built(cfg, tables, units=("word",))
    result = zscores(word_table("p", np.full(N_WORDS, 7.0)), envelope.unit("word"), config=cfg)
    deviation = combined_deviation(result, cfg)
    assert (deviation.dropna() >= 0).all()
    # Only the lambda-weighted features enter D, and rate is one of them, so a
    # 10 syll/s deviation must dominate.
    assert float(deviation.max()) > 1.0


def test_combined_deviation_is_nan_where_nothing_is_measurable(cfg):
    tables = reference_tables(with_gaps=False)
    envelope = built(cfg, tables, units=("word",))
    participant = word_table("p", np.full(N_WORDS, 7.0))
    for column in participant.columns:
        if column not in {"text", "t_start", "t_end"}:
            participant[column] = np.nan
    result = zscores(participant, envelope.unit("word"), config=cfg)
    assert combined_deviation(result, cfg).isna().all()


# --------------------------------------------------------------------------- #
# Layer 2: the trigger table
# --------------------------------------------------------------------------- #


def signals_for(cfg, participant_tables, *, same_channel=False, labels=None):
    tables = reference_tables()
    envelope = built(cfg, tables)
    z = envelope_z(participant_tables, envelope, config=cfg, channel_matched=same_channel)
    return trigger_signals(z, config=cfg, same_channel=same_channel, labels=labels), z


def test_rushing_and_dragging_are_the_two_signs_of_one_feature(cfg):
    fast = {
        "word": word_table("p", np.full(N_WORDS, 9.0)),
        "gap": gap_table("p", np.full(N_WORDS - 1, 0.2)),
    }
    (signals, _), _ = signals_for(cfg, fast, labels=["rushing", "dragging"])
    by_label = {s.label: s for s in signals}
    assert set(by_label) == {"rushing", "dragging"}
    # Oriented so positive means "towards this flaw": a fast reading scores
    # positive on rushing and negative on dragging, with the same magnitude.
    assert float(by_label["rushing"].score.iloc[0]) > 0
    assert float(by_label["dragging"].score.iloc[0]) < 0
    assert by_label["rushing"].score.iloc[0] == pytest.approx(-by_label["dragging"].score.iloc[0])


def test_missing_pause_only_fires_where_the_text_expects_a_pause(cfg):
    short = np.full(N_WORDS - 1, 0.03)
    participant = {"word": word_table("p", np.full(N_WORDS, 5.0)), "gap": gap_table("p", short)}
    (signals, _), z = signals_for(cfg, participant, labels=["missing_pause"])
    signal = signals[0]
    labels = z["gap"].table["gap_label"]
    for idx in signal.gate.index:
        assert bool(signal.gate.loc[idx]) == (labels.loc[idx] == "boundary")
    # Every boundary gap is far too short, so the gated score is strongly positive.
    gated = signal.score[signal.gate]
    assert (gated > 2.5).all()


def test_hesitation_pause_needs_both_the_text_gate_and_the_absolute_threshold(cfg):
    # A 0.1 s mid-clause gap is long relative to the envelope but below the
    # 0.25 s fluency threshold, so it must not be called a hesitation.
    durations = np.array([0.5 if i % 3 == 0 else 0.10 for i in range(N_WORDS - 1)])
    participant = {"word": word_table("p", np.full(N_WORDS, 5.0)), "gap": gap_table("p", durations)}
    (signals, _), z = signals_for(cfg, participant, labels=["hesitation_pause"])
    assert not signals[0].gate.any()

    # Make the same gaps 0.6 s and the rule fires on exactly the `none` gaps.
    durations = np.array([0.5 if i % 3 == 0 else 0.60 for i in range(N_WORDS - 1)])
    participant = {"word": word_table("p", np.full(N_WORDS, 5.0)), "gap": gap_table("p", durations)}
    (signals, _), z = signals_for(cfg, participant, labels=["hesitation_pause"])
    gate = signals[0].gate
    labels = z["gap"].table["gap_label"]
    assert gate.any()
    for idx in gate.index:
        assert bool(gate.loc[idx]) == (labels.loc[idx] == "none")


def test_mumbling_is_skipped_without_a_same_channel_reference(cfg):
    participant = {
        "word": word_table("p", np.full(N_WORDS, 5.0)),
        "gap": gap_table("p", np.full(N_WORDS - 1, 0.2)),
    }
    (signals, skipped), _ = signals_for(cfg, participant, same_channel=False, labels=["mumbling"])
    assert signals == []
    assert [s.reason for s in skipped] == ["no_same_channel_reference"]
    assert "channel" in skipped[0].detail


def test_mumbling_fires_on_a_same_channel_composite(cfg):
    def with_clarity(fid, *, clear=True):
        table = word_table(fid, np.full(N_WORDS, 5.0 if clear else 4.0))
        table["cpps_db"] = np.full(N_WORDS, 15.0 if clear else 9.0)
        table["spectral_tilt_db_per_khz"] = np.full(N_WORDS, -8.0 if clear else -13.0)
        table["duration_per_syllable_s"] = np.full(N_WORDS, 0.175 if clear else 0.130)
        table["energy_mean_z"] = np.full(N_WORDS, 0.0 if clear else -1.5)
        return table

    rng = np.random.default_rng(1)
    tables = {}
    for r in range(5):
        table = with_clarity(f"r{r}")
        for column in ("cpps_db", "spectral_tilt_db_per_khz", "energy_mean_z"):
            table[column] = table[column] + rng.normal(0.0, 0.4, size=N_WORDS)
        tables[f"r{r}"] = {"word": table}

    envelope = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    assert envelope.same_channel
    apply_sigma_floors(envelope, {"word": dict.fromkeys(envelope.unit("word").features, 0.05)})

    z = envelope_z(
        {"word": with_clarity("p", clear=False)}, envelope, config=cfg, channel_matched=True
    )
    signals, skipped = trigger_signals(z, config=cfg, same_channel=True, labels=["mumbling"])
    assert skipped == []
    signal = signals[0]
    assert signal.primary_feature == "clarity_composite"
    assert signal.channel_matched
    assert (signal.score > 2.5).all(), "every component points the same way"


def test_monotone_needs_both_pitch_range_and_f0_variation(cfg):
    """min() of the two oriented z keeps the spec's `and` at any threshold."""
    rng = np.random.default_rng(2)
    tables = {}
    for r in range(5):
        word = word_table(f"r{r}", np.full(N_WORDS, 5.0))
        word["f0_range_st"] = 6.0 + rng.normal(0.0, 0.5, size=N_WORDS)
        word["f0_std_st"] = 2.0 + rng.normal(0.0, 0.2, size=N_WORDS)
        phrase = pd.DataFrame(
            {
                "t_start": [0.0, 3.0],
                "t_end": [2.8, 5.8],
                "first_word_idx": [0, 6],
                "last_word_idx": [5, N_WORDS - 1],
                "phrase_f0_range_st": [6.0 + rng.normal(0, 0.4), 6.0 + rng.normal(0, 0.4)],
                "phrase_duration_s": [2.8, 2.8],
                "phrase_rate_syll_per_s": [5.0, 5.0],
            },
            index=pd.Index([0, 1], name="unit_idx", dtype="int64"),
        )
        phrase.attrs.update(unit="phrase", file_id=f"r{r}", channel_id="ch_a")
        tables[f"r{r}"] = {"word": word, "phrase": phrase}

    envelope = build_envelope("T", sorted(tables), tables, units=("word", "phrase"), config=cfg)
    apply_sigma_floors(
        envelope,
        {u: dict.fromkeys(envelope.unit(u).features, 0.05) for u in ("word", "phrase")},
    )

    def participant(range_st: float, std_st: float):
        word = word_table("p", np.full(N_WORDS, 5.0))
        word["f0_range_st"] = np.full(N_WORDS, range_st)
        word["f0_std_st"] = np.full(N_WORDS, std_st)
        phrase = tables["r0"]["phrase"].copy()
        phrase["phrase_f0_range_st"] = [range_st, range_st]
        phrase.attrs.update(unit="phrase", file_id="p", channel_id="ch_a")
        return {"word": word, "phrase": phrase}

    def score(range_st, std_st):
        z = envelope_z(participant(range_st, std_st), envelope, config=cfg)
        signals, _ = trigger_signals(z, config=cfg, labels=["monotone"])
        return float(signals[0].score.iloc[0])

    # Both low: fires.
    assert score(1.0, 0.4) > 2.5
    # Range low but variation normal: the `and` must veto it.
    assert score(1.0, 2.0) < 2.5
    # Variation low but range normal: likewise.
    assert score(6.0, 0.4) < 2.5


def test_misplaced_stress_ignores_a_uniform_prominence_boost(cfg):
    """A uniformly louder-sounding reading has not MISPLACED anything."""
    contour = np.array([0.4, 1.1, 0.5, 0.9, 0.6, 2.2, 0.5, 1.0, 0.6, 0.8, 0.4, 2.1])
    rng = np.random.default_rng(3)

    def make(fid, prominence):
        word = word_table(fid, np.full(N_WORDS, 5.0))
        word["prominence"] = prominence
        phrase = pd.DataFrame(
            {
                "t_start": [0.0, 3.0],
                "t_end": [2.8, 5.8],
                "first_word_idx": [0, 6],
                "last_word_idx": [5, N_WORDS - 1],
                "phrase_f0_range_st": [6.0, 6.0],
                "phrase_duration_s": [2.8, 2.8],
                "phrase_rate_syll_per_s": [5.0, 5.0],
            },
            index=pd.Index([0, 1], name="unit_idx", dtype="int64"),
        )
        phrase.attrs.update(unit="phrase", file_id=fid, channel_id="ch_a")
        return {"word": word, "phrase": phrase}

    tables = {
        f"r{r}": make(f"r{r}", contour + rng.normal(0.0, 0.08, size=N_WORDS)) for r in range(5)
    }
    envelope = build_envelope("T", sorted(tables), tables, units=("word", "phrase"), config=cfg)
    apply_sigma_floors(
        envelope,
        {u: dict.fromkeys(envelope.unit(u).features, 0.02) for u in ("word", "phrase")},
    )

    def gate_for(prominence):
        z = envelope_z(make("p", prominence), envelope, config=cfg)
        signals, _ = trigger_signals(z, config=cfg, labels=["misplaced_stress"])
        return signals[0]

    uniform = gate_for(contour + 1.5)
    assert not uniform.gate.any(), "a uniform boost leaves the ranking intact"

    # Move the accent off word 5 and onto word 4: the ranking now disagrees.
    moved = contour.copy()
    moved[4] += 2.0
    moved[5] -= 2.0
    shifted = gate_for(moved)
    assert shifted.gate.any()
    assert set(shifted.gate[shifted.gate].index) <= {4, 5}


def test_low_energy_has_a_fading_variant_on_the_slope(cfg):
    tables = reference_tables()
    envelope = built(cfg, tables)
    participant = {
        "word": word_table("p", np.full(N_WORDS, 5.0)),
        "gap": gap_table("p", np.full(N_WORDS - 1, 0.2)),
    }
    participant["word"]["energy_slope_db_per_s"] = np.full(N_WORDS, -20.0)
    z = envelope_z(participant, envelope, config=cfg)
    signals, _ = trigger_signals(z, config=cfg, labels=["low_energy"])
    keys = {s.key for s in signals}
    assert "low_energy" in keys
    # The variant is a variant of the same label, not a ninth label.
    assert all(s.label == "low_energy" for s in signals)


def test_pause_rules_are_skipped_with_a_reason_when_there_is_no_gap_envelope(cfg):
    tables = reference_tables(with_gaps=False)
    envelope = built(cfg, tables, units=("word",))
    z = envelope_z({"word": word_table("p", np.full(N_WORDS, 5.0))}, envelope, config=cfg)
    signals, skipped = trigger_signals(z, config=cfg, labels=["missing_pause", "hesitation_pause"])
    assert signals == []
    assert {s.reason for s in skipped} == {"no_gap_envelope"}


def test_trigger_signals_needs_the_word_comparison(cfg):
    with pytest.raises(KeyError, match="at least the word-level"):
        trigger_signals({}, config=cfg)


def test_every_signal_is_oriented_so_positive_means_worse(cfg):
    """The sign convention the whole of region formation depends on."""
    tables = reference_tables()
    envelope = built(cfg, tables)
    participant = {
        "word": word_table("p", np.full(N_WORDS, 5.0)),
        "gap": gap_table("p", np.full(N_WORDS - 1, 0.2)),
    }
    z = envelope_z(participant, envelope, config=cfg)
    signals, skipped = trigger_signals(z, config=cfg)
    assert signals, "a word+gap comparison must produce several signals"
    for signal in signals:
        assert isinstance(signal, Signal)
        assert signal.unit in {"word", "gap", "phrase"}
        assert list(signal.score.index) == list(signal.gate.index)
    for item in skipped:
        assert isinstance(item, SkippedSignal)
        assert item.detail, "a skipped label must say why, for the dashboard"
