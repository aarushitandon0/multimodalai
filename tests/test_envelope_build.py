"""Tests for the envelope builder (spec 10.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.envelope.build import (
    MAD_SCALE,
    EnvelopeError,
    Reference,
    WordIndexMap,
    apply_sigma_floors,
    build_curve_envelope,
    build_envelope,
    build_fold_envelopes,
    fit_sigma_floors,
    leave_one_out,
    mad,
    robust_centre_spread,
    summarise_counts,
    word_curves,
)

RATE = "rate_local_syll_per_s"
GAP = "gap_duration_s"
N_WORDS = 8


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def word_table(file_id: str, rate: np.ndarray, *, channel_id: str = "ch_a") -> pd.DataFrame:
    """A minimal word table carrying the columns the envelope config asks for."""
    n = len(rate)
    table = pd.DataFrame(
        {
            "text": [f"w{i}" for i in range(n)],
            "t_start": [i * 0.5 for i in range(n)],
            "t_end": [i * 0.5 + 0.4 for i in range(n)],
            RATE: np.asarray(rate, dtype=float),
            "f0_range_st": np.full(n, 6.0),
            "f0_std_st": np.full(n, 2.0),
            "energy_mean_z": np.zeros(n),
            "prominence": np.linspace(0.0, 1.0, n),
            "word_duration_s": np.full(n, 0.4),
        },
        index=pd.Index(range(n), name="unit_idx", dtype="int64"),
    )
    table.attrs.update(unit="word", file_id=file_id, channel_id=channel_id)
    return table


def gap_table(file_id: str, durations: np.ndarray, *, channel_id: str = "ch_a") -> pd.DataFrame:
    n = len(durations)
    labels = ["boundary" if i % 3 == 0 else "none" for i in range(n)]
    table = pd.DataFrame(
        {
            "t_start": [i * 0.5 + 0.4 for i in range(n)],
            "t_end": [i * 0.5 + 0.4 + d for i, d in enumerate(durations)],
            GAP: np.asarray(durations, dtype=float),
            "gap_label": labels,
            "gap_prev_word_idx": list(range(n)),
            "gap_next_word_idx": list(range(1, n + 1)),
        },
        index=pd.Index(range(n), name="unit_idx", dtype="int64"),
    )
    table.attrs.update(unit="gap", file_id=file_id, channel_id=channel_id)
    return table


def tables_for(file_id: str, rate: np.ndarray, gaps: np.ndarray | None = None) -> dict:
    out = {"word": word_table(file_id, rate)}
    if gaps is not None:
        out["gap"] = gap_table(file_id, gaps)
    return out


# --------------------------------------------------------------------------- #
# Robust statistics
# --------------------------------------------------------------------------- #


def test_mad_is_unscaled_and_nan_tolerant():
    assert mad([1.0, 1.0, 1.0, 5.0]) == pytest.approx(0.0)
    assert mad([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)
    assert mad([1.0, np.nan, 1.0]) == pytest.approx(0.0)
    assert np.isnan(mad([np.nan, np.nan]))


def test_robust_centre_spread_reports_zero_n_for_an_all_nan_unit():
    stack = np.array([[1.0, np.nan], [3.0, np.nan], [2.0, np.nan]])
    centre, spread, n = robust_centre_spread(stack, axis=0)
    assert centre[0] == pytest.approx(2.0)
    assert spread[0] == pytest.approx(MAD_SCALE * 1.0)
    assert n[0] == 3
    # The second column was never measured: NaN, and n = 0, never a silent zero.
    assert np.isnan(centre[1]) and np.isnan(spread[1]) and n[1] == 0


# --------------------------------------------------------------------------- #
# The two tests the build prompt names
# --------------------------------------------------------------------------- #


def test_five_identical_references_give_sigma_equal_to_the_floor(cfg):
    """With no reader-to-reader variation, the effective sigma IS the floor.

    The floor has to come from somewhere, and spec 10.2 says it comes from the
    training texts. So the fold here has a unanimous text and a varied one, and
    the unanimous text's effective sigma must equal the floor fitted on the
    varied one - exactly, not approximately.
    """
    identical = {f"id{r}": tables_for(f"id{r}", np.full(N_WORDS, 5.0)) for r in range(5)}
    varied = {f"var{r}": tables_for(f"var{r}", 5.0 + np.full(N_WORDS, 0.4 * r)) for r in range(5)}
    tables = {**identical, **varied}

    envelopes = build_fold_envelopes(
        {"T_identical": sorted(identical), "T_varied": sorted(varied)},
        tables,
        fold="loto_T_identical",
        train_texts=["T_varied"],  # the held-out text must not set its own floor
        units=("word",),
        config=cfg,
    )

    floor = envelopes["T_varied"].unit("word").envelope.sigma_floor[RATE]
    assert floor > 0.0

    unanimous = envelopes["T_identical"].unit("word")
    for idx in unanimous.unit_indices:
        assert unanimous.envelope.sigma[idx][unanimous.column(RATE)] == pytest.approx(0.0)
        assert unanimous.envelope.effective_sigma(idx, RATE) == pytest.approx(floor)


def test_median_is_robust_to_one_outlier_reader(cfg):
    """One wild reader moves the median by nothing and the MAD by little."""
    good = {
        f"g{r}": tables_for(f"g{r}", np.array([4.8, 5.0, 5.2, 5.0, 4.9, 5.1, 5.0, 5.0]))
        for r in range(4)
    }
    good["g0"]["word"].loc[:, RATE] = np.array([4.9, 5.0, 5.1, 5.0, 5.0, 5.0, 5.0, 5.0])

    without = build_envelope("T", sorted(good), good, units=("word",), config=cfg).unit("word")

    outlier = dict(good)
    outlier["bad"] = tables_for("bad", np.full(N_WORDS, 40.0))
    with_outlier = build_envelope("T", sorted(outlier), outlier, units=("word",), config=cfg).unit(
        "word"
    )

    col = without.column(RATE)
    for idx in without.unit_indices:
        before = without.envelope.median[idx][col]
        after = with_outlier.envelope.median[idx][col]
        # A 40 syll/s reader is 8x the population value; the median must barely move.
        assert abs(after - before) < 0.2, f"unit {idx}: median moved {after - before}"

    # The arithmetic mean, by contrast, would have moved by ~7 syll/s - stated
    # here because it is the reason spec 10.2 specifies a median at all.
    mean_shift = (40.0 - 5.0) / 5
    assert mean_shift > 6.0


# --------------------------------------------------------------------------- #
# Sigma floors and the leakage guard
# --------------------------------------------------------------------------- #


def test_sigma_floor_is_the_25th_percentile_of_training_sigmas(cfg):
    tables = {
        f"{text}_r{r}": tables_for(f"{text}_r{r}", 5.0 + np.arange(N_WORDS) * 0.1 * (r + 1))
        for text in ("A", "B")
        for r in range(4)
    }
    envelopes = {
        text: build_envelope(
            text,
            sorted(f for f in tables if f.startswith(text)),
            tables,
            units=("word",),
            config=cfg,
        )
        for text in ("A", "B")
    }
    floors = fit_sigma_floors(envelopes.values(), ["A"], config=cfg)

    sigmas = [
        row[envelopes["A"].unit("word").column(RATE)]
        for row in envelopes["A"].unit("word").envelope.sigma.values()
    ]
    assert floors["word"][RATE] == pytest.approx(float(np.percentile(sigmas, 25)))


def test_fit_sigma_floors_ignores_non_training_texts(cfg):
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0 + r)) for r in range(4)}
    held = build_envelope("HELD", sorted(tables), tables, units=("word",), config=cfg)
    with pytest.raises(EnvelopeError, match="training texts"):
        fit_sigma_floors([held], ["SOME_OTHER_TEXT"], config=cfg)


def test_build_fold_envelopes_applies_one_floor_to_every_text(cfg):
    tables = {
        f"{text}_r{r}": tables_for(f"{text}_r{r}", 5.0 + np.arange(N_WORDS) * 0.05 * (r + 1))
        for text in ("A", "B", "C")
        for r in range(4)
    }
    envelopes = build_fold_envelopes(
        {t: sorted(f for f in tables if f.startswith(t)) for t in ("A", "B", "C")},
        tables,
        fold="loto_A",
        train_texts=["B", "C"],
        units=("word",),
        config=cfg,
    )
    floors = {t: e.unit("word").envelope.sigma_floor[RATE] for t, e in envelopes.items()}
    assert len(set(floors.values())) == 1, "every text in a fold shares one floor"


# --------------------------------------------------------------------------- #
# Reference count and minimum
# --------------------------------------------------------------------------- #


def test_too_few_references_is_an_error_not_a_weak_envelope(cfg):
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0)) for r in range(2)}
    with pytest.raises(EnvelopeError, match="min_references"):
        build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)


def test_n_per_text_is_reported(cfg):
    tables = {
        f"{t}_r{r}": tables_for(f"{t}_r{r}", np.full(N_WORDS, 5.0 + r))
        for t, n in (("A", 4), ("B", 6))
        for r in range(n)
    }
    envelopes = build_fold_envelopes(
        {
            "A": sorted(f for f in tables if f.startswith("A")),
            "B": sorted(f for f in tables if f.startswith("B")),
        },
        tables,
        fold="loto_A",
        train_texts=["B"],
        units=("word",),
        config=cfg,
    )
    rows = {row["text_id"]: row for row in summarise_counts(envelopes)}
    assert rows["A"]["n"] == 4
    assert rows["B"]["n"] == 6
    assert rows["A"]["units"]["word"]["n_units"] == N_WORDS


# --------------------------------------------------------------------------- #
# Channel rule
# --------------------------------------------------------------------------- #


def test_channel_sensitive_features_need_one_channel(cfg):
    tables = {}
    for r in range(4):
        fid = f"r{r}"
        table = word_table(fid, np.full(N_WORDS, 5.0), channel_id=f"ch_{r}")
        table["cpps_db"] = np.full(N_WORDS, 15.0)
        tables[fid] = {"word": table}

    envelope = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    assert not envelope.same_channel
    assert "cpps_db" not in envelope.unit("word").features

    with pytest.raises(EnvelopeError, match=r"CLAUDE\.md invariant 7"):
        build_envelope(
            "T",
            sorted(tables),
            tables,
            units=("word",),
            config=cfg,
            include_channel_sensitive=True,
        )


def test_one_channel_admits_the_channel_sensitive_block(cfg):
    tables = {}
    for r in range(4):
        fid = f"r{r}"
        table = word_table(fid, np.full(N_WORDS, 5.0), channel_id="ch_a")
        table["cpps_db"] = np.full(N_WORDS, 15.0 + r * 0.1)
        tables[fid] = {"word": table}
    envelope = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    assert envelope.same_channel
    assert "cpps_db" in envelope.unit("word").features


# --------------------------------------------------------------------------- #
# Canonical word indices
# --------------------------------------------------------------------------- #


def test_word_index_map_remaps_and_identity_is_free():
    table = word_table("r0", np.arange(N_WORDS, dtype=float))
    assert WordIndexMap().apply(table) is table

    shifted = WordIndexMap(file_id="r0", to_canonical={i: i + 1 for i in range(N_WORDS)})
    out = shifted.apply(table)
    assert list(out.index) == list(range(1, N_WORDS + 1))
    assert out.attrs["word_index_map"] == {i: i + 1 for i in range(N_WORDS)}


def test_a_non_injective_word_index_map_is_rejected():
    table = word_table("r0", np.arange(N_WORDS, dtype=float))
    bad = WordIndexMap(file_id="r0", to_canonical={0: 1, 1: 1})
    with pytest.raises(EnvelopeError, match="not injective"):
        bad.apply(table)


def test_a_dropped_word_shifts_nothing_for_the_other_readers(cfg):
    """A reader missing word 3 must not shift words 4+ against the others."""
    tables = {f"r{r}": tables_for(f"r{r}", np.arange(N_WORDS, dtype=float)) for r in range(4)}
    # r3's aligner dropped word 3, so its local indices 3.. mean canonical 4..
    local = word_table("r3", np.delete(np.arange(N_WORDS, dtype=float), 3))
    local.index = pd.Index(range(N_WORDS - 1), name="unit_idx", dtype="int64")
    tables["r3"] = {"word": local}
    index_maps = {
        "r3": {
            "word": WordIndexMap(
                file_id="r3",
                to_canonical={i: (i if i < 3 else i + 1) for i in range(N_WORDS - 1)},
                dropped=(3,),
            )
        }
    }
    envelope = build_envelope(
        "T", sorted(tables), tables, units=("word",), config=cfg, index_maps=index_maps
    )
    unit = envelope.unit("word")
    col = unit.column(RATE)
    # Every canonical word still has the right value, and word 3 has one reader
    # fewer rather than a wrong value.
    for idx in range(N_WORDS):
        assert unit.envelope.median[idx][col] == pytest.approx(float(idx))
    assert unit.n_at(3) == 3
    assert unit.n_at(4) == 4


# --------------------------------------------------------------------------- #
# Reference values and the card's context line
# --------------------------------------------------------------------------- #


def test_reference_values_support_the_n_of_m_context_line(cfg):
    durations = [
        np.array([0.5, 0.05, 0.5, 0.05, 0.5, 0.05, 0.5]),
        np.array([0.6, 0.04, 0.5, 0.06, 0.4, 0.05, 0.5]),
        np.array([0.4, 0.06, 0.6, 0.04, 0.5, 0.04, 0.6]),
        np.array([0.5, 0.05, 0.5, 0.05, 0.5, 0.06, 0.5]),
        np.array([0.05, 0.05, 0.5, 0.05, 0.5, 0.05, 0.5]),  # one reader ran through
    ]
    tables = {
        f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0), gaps=d) for r, d in enumerate(durations)
    }
    envelope = build_envelope("T", sorted(tables), tables, units=("word", "gap"), config=cfg)
    gaps = envelope.unit("gap")

    matching, measured = gaps.count_where(0, GAP, at_least=0.25)
    assert (matching, measured) == (4, 5), "4 of 5 readers paused at gap 0"
    assert gaps.gap_labels[0] == "boundary"
    assert np.allclose(sorted(gaps.reference_values(0, GAP)), [0.05, 0.4, 0.5, 0.5, 0.6])


# --------------------------------------------------------------------------- #
# Leave one reference out
# --------------------------------------------------------------------------- #


def test_leave_one_out_is_an_exact_recomputation(cfg):
    values = [4.0, 4.5, 5.0, 5.5, 9.0]
    tables = {f"r{i}": tables_for(f"r{i}", np.full(N_WORDS, v)) for i, v in enumerate(values)}
    full = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    apply_sigma_floors(full, {"word": {RATE: 0.125}})

    reduced = leave_one_out(full, "r4")
    assert reduced.reference_ids == ["r0", "r1", "r2", "r3"]
    unit = reduced.unit("word")
    col = unit.column(RATE)
    # median of [4.0, 4.5, 5.0, 5.5] is 4.75
    assert unit.envelope.median[0][col] == pytest.approx(4.75)
    # The floor is a fold-level quantity and must survive untouched.
    assert unit.envelope.sigma_floor[RATE] == pytest.approx(0.125)
    # The full envelope is not mutated.
    assert full.reference_ids == ["r0", "r1", "r2", "r3", "r4"]


def test_leave_one_out_rejects_an_unknown_reader(cfg):
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0 + r)) for r in range(4)}
    envelope = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    with pytest.raises(KeyError, match="not a reference"):
        leave_one_out(envelope, "nobody")


def test_leave_one_out_over_every_reader_gives_a_band(cfg):
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0 + 0.5 * r)) for r in range(5)}
    envelope = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    medians = []
    for fid in envelope.reference_ids:
        unit = leave_one_out(envelope, fid).unit("word")
        medians.append(unit.envelope.median[0][unit.column(RATE)])
    assert len(medians) == 5
    assert float(np.std(medians)) > 0.0, "the band must have non-zero width"


# --------------------------------------------------------------------------- #
# Frame curves
# --------------------------------------------------------------------------- #


def frames_for(level: float, n: int = 120) -> pd.DataFrame:
    time = np.arange(n) * 0.010
    return pd.DataFrame({"time_s": time, "f0_st": np.full(n, level), "energy_z": np.zeros(n)})


def test_word_curves_time_normalise_to_a_fixed_point_count():
    frames = frames_for(3.0)
    curves = word_curves(frames, {0: (0.1, 0.4), 1: (0.5, 1.0)}, ["f0_st", "energy_z"], n_points=20)
    assert set(curves) == {0, 1}
    # Both words give 20 points although they are 0.3 s and 0.5 s long: that is
    # the whole purpose of the normalised axis.
    for arr in curves.values():
        assert arr.shape == (20, 2)
        assert np.allclose(arr[:, 0], 3.0)


def test_word_curves_hold_a_per_word_quantity_constant():
    frames = frames_for(3.0)
    curves = word_curves(
        frames,
        {0: (0.1, 0.4)},
        ["f0_st", "rate_local_syll_per_s"],
        n_points=5,
        word_values={"rate_local_syll_per_s": {0: 4.2}},
    )
    assert np.allclose(curves[0][:, 1], 4.2)


def test_curve_envelope_is_median_and_scaled_mad_per_point():
    per_reference = {
        f"r{i}": {0: np.full((5, 1), value)} for i, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0])
    }
    envelope = build_curve_envelope("T", per_reference, ["f0_st"], n_points=5)
    assert envelope.n_references[0] == 5
    assert np.allclose(envelope.median[0][:, 0], 3.0)
    assert np.allclose(envelope.sigma[0][:, 0], MAD_SCALE * 1.0)
    x, lower, upper = envelope.band(0, "f0_st")
    assert x[0] == 0.0 and x[-1] == 1.0
    assert np.allclose(upper - lower, 2 * MAD_SCALE)


def test_build_envelope_builds_curves_when_frames_are_supplied(cfg):
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0 + r)) for r in range(4)}
    spans = {f"r{r}": {0: (0.1, 0.4), 1: (0.5, 0.9)} for r in range(4)}
    frames = {f"r{r}": frames_for(2.0 + r) for r in range(4)}
    envelope = build_envelope(
        "T",
        sorted(tables),
        tables,
        units=("word",),
        config=cfg,
        frames=frames,
        word_spans=spans,
    )
    assert envelope.curves is not None
    assert envelope.curves.n_points == 20
    assert set(envelope.curves.median) == {0, 1}


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_reference_order_does_not_depend_on_dict_order(cfg):
    rate = {f"r{r}": np.full(N_WORDS, 5.0 + r) for r in range(5)}
    forward = {fid: tables_for(fid, v) for fid, v in rate.items()}
    backward = {fid: tables_for(fid, rate[fid]) for fid in reversed(list(rate))}
    one = build_envelope("T", list(forward), forward, units=("word",), config=cfg)
    two = build_envelope("T", list(backward), backward, units=("word",), config=cfg)
    assert one.reference_ids == two.reference_ids
    assert one.to_dict() == two.to_dict()


def test_reference_objects_and_table_dicts_agree(cfg):
    tables = {f"r{r}": tables_for(f"r{r}", np.full(N_WORDS, 5.0 + r)) for r in range(4)}
    refs = [Reference(file_id=f, channel_id="ch_a", tables=t) for f, t in tables.items()]
    from_dict = build_envelope("T", sorted(tables), tables, units=("word",), config=cfg)
    from_refs = build_envelope("T", sorted(tables), refs, units=("word",), config=cfg)
    assert from_dict.to_dict() == from_refs.to_dict()


# --------------------------------------------------------------------------- #
# Warping a participant's curves onto the envelope (spec 10.1, 15.2)
# --------------------------------------------------------------------------- #


def _curve_envelope_of(shape):
    """A one-signal curve envelope holding ``shape`` as its median."""
    from rostrum.envelope.build import CurveEnvelope

    median = np.asarray(shape, dtype=float)[:, None]
    return CurveEnvelope(
        text_id="t",
        signals=("f0_st",),
        n_points=median.shape[0],
        median={0: median},
        sigma={0: np.full_like(median, 0.1)},
        n_references={0: 5},
        reference_ids=["r0"],
    )


def test_warping_lines_a_shifted_curve_up_with_the_envelope():
    """DTW inside a word, which is the only place spec 10.1 allows it."""
    from rostrum.envelope.build import warp_to_envelope

    axis = np.linspace(0.0, 1.0, 20)
    reference = np.sin(np.pi * axis)
    # The same gesture, finished in the first half of the word.
    participant = np.sin(np.pi * np.clip(axis * 2.0, 0.0, 1.0))
    envelope = _curve_envelope_of(reference)

    before = np.corrcoef(participant, reference)[0, 1]
    warped = warp_to_envelope({0: participant[:, None]}, envelope, signals=["f0_st"])
    after = np.corrcoef(warped[0][:, 0], reference)[0, 1]
    assert after > before
    assert after > 0.9


def test_warping_can_be_switched_off_for_the_ablation():
    from rostrum.envelope.build import warp_to_envelope

    participant = np.linspace(0.0, 1.0, 20)[:, None]
    envelope = _curve_envelope_of(np.zeros(20))
    out = warp_to_envelope({0: participant}, envelope, signals=["f0_st"], dtw=False)
    assert np.allclose(out[0], participant)


def test_a_word_the_envelope_does_not_cover_is_returned_unchanged():
    from rostrum.envelope.build import warp_to_envelope

    participant = np.linspace(0.0, 1.0, 20)[:, None]
    envelope = _curve_envelope_of(np.zeros(20))
    out = warp_to_envelope({7: participant}, envelope, signals=["f0_st"])
    assert np.allclose(out[7], participant)


def test_an_unvoiced_curve_is_returned_unchanged():
    """DTW has no notion of a missing observation; it must not invent one."""
    from rostrum.envelope.build import warp_to_envelope

    participant = np.full((20, 1), np.nan)
    envelope = _curve_envelope_of(np.linspace(0.0, 1.0, 20))
    out = warp_to_envelope({0: participant}, envelope, signals=["f0_st"])
    assert np.isnan(out[0]).all()
