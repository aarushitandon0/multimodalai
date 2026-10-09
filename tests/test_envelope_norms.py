"""Tests for the pooled norms behind Norm mode (spec 13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rostrum.config import load_config
from rostrum.envelope.build import Reference
from rostrum.envelope.norms import POOLED_TEXT_ID, build_pooled_norms, pooled_envelope


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def word_table(rates, file_id="r0", channel="ch0") -> pd.DataFrame:
    index = pd.Index(range(len(rates)), name="unit_idx", dtype="int64")
    table = pd.DataFrame(
        {
            "t_start": np.arange(len(rates), dtype=float),
            "t_end": np.arange(len(rates), dtype=float) + 0.5,
            "text": [f"w{i}" for i in range(len(rates))],
            "rate_local_syll_per_s": np.asarray(rates, dtype=float),
        },
        index=index,
    )
    table.attrs.update({"unit": "word", "file_id": file_id, "channel_id": channel})
    return table


def gap_table(durations, labels, file_id="r0", channel="ch0") -> pd.DataFrame:
    index = pd.Index(range(len(durations)), name="unit_idx", dtype="int64")
    table = pd.DataFrame(
        {
            "t_start": np.arange(len(durations), dtype=float),
            "t_end": np.arange(len(durations), dtype=float) + 0.1,
            "gap_duration_s": np.asarray(durations, dtype=float),
            "gap_label": list(labels),
            "gap_prev_word_idx": list(range(len(durations))),
            "gap_next_word_idx": [i + 1 for i in range(len(durations))],
        },
        index=index,
    )
    table.attrs.update({"unit": "gap", "file_id": file_id, "channel_id": channel})
    return table


def references(n=4) -> list[Reference]:
    out = []
    for i in range(n):
        offset = 0.1 * i
        out.append(
            Reference(
                file_id=f"r{i}",
                channel_id=f"ch{i}",
                tables={
                    "word": word_table([4.0 + offset, 3.8 + offset, 4.2 + offset]),
                    "gap": gap_table([0.8 + offset, 0.05 + 0.01 * i], ["boundary", "none"]),
                },
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #


def test_a_word_feature_gets_one_median_for_every_word(cfg):
    norms = build_pooled_norms(references(), config=cfg)
    median, sigma, n = norms.units["word"]["rate_local_syll_per_s"]
    assert 3.8 <= median <= 4.4
    assert sigma > 0
    assert n == 12  # three words times four readings


def test_a_gap_keeps_one_norm_per_label(cfg):
    """The text still says whether a pause was invited; that is punctuation."""
    norms = build_pooled_norms(references(), config=cfg)
    boundary = norms.gaps["boundary"]["gap_duration_s"][0]
    none = norms.gaps["none"]["gap_duration_s"][0]
    assert boundary > none * 5


def test_the_norms_record_their_readings(cfg):
    norms = build_pooled_norms(references(3), config=cfg)
    assert norms.n_references == 3
    assert norms.reference_ids == ("r0", "r1", "r2")


def test_the_norms_serialise(cfg):
    payload = build_pooled_norms(references(), config=cfg).to_dict()
    assert payload["text_id"] == POOLED_TEXT_ID
    assert payload["units"]["word"]["rate_local_syll_per_s"]["n"] == 12


def test_pooling_nothing_is_not_an_error(cfg):
    norms = build_pooled_norms([], config=cfg)
    assert norms.n_references == 0
    assert norms.features("word") == []


# --------------------------------------------------------------------------- #
# Projecting onto one recording
# --------------------------------------------------------------------------- #


def test_the_pooled_envelope_covers_every_unit_of_the_upload(cfg):
    norms = build_pooled_norms(references(), config=cfg)
    tables = {
        "word": word_table([4.0, 9.0, 4.1, 4.0, 3.9], file_id="p", channel="mine"),
        "gap": gap_table(
            [0.05, 0.9, 0.5, 0.04], ["boundary", "none", "minor", "none"], "p", "mine"
        ),
    }
    envelope = pooled_envelope(norms, tables)
    assert envelope.text_id == POOLED_TEXT_ID
    assert sorted(envelope.units) == ["gap", "word"]
    assert envelope.unit("word").unit_indices == [0, 1, 2, 3, 4]
    assert envelope.unit("gap").unit_indices == [0, 1, 2, 3]


def test_each_gap_is_given_the_norm_for_its_own_label(cfg):
    norms = build_pooled_norms(references(), config=cfg)
    tables = {
        "word": word_table([4.0], file_id="p", channel="mine"),
        "gap": gap_table([0.05, 0.9], ["boundary", "none"], "p", "mine"),
    }
    envelope = pooled_envelope(norms, tables)
    gaps = envelope.unit("gap")
    boundary_median = gaps.envelope.median[0][gaps.column("gap_duration_s")]
    none_median = gaps.envelope.median[1][gaps.column("gap_duration_s")]
    assert boundary_median > none_median


def test_a_pooled_envelope_is_never_single_channel(cfg):
    """Spec 9.3/13: the channel-sensitive rules must not fire in Norm mode."""
    norms = build_pooled_norms(references(), config=cfg)
    tables = {"word": word_table([4.0, 4.1], file_id="p", channel="mine")}
    envelope = pooled_envelope(norms, tables)
    assert not envelope.same_channel


def test_a_deviation_is_standardised_against_the_pooled_spread(cfg):
    from rostrum.detect.zscore import zscores

    norms = build_pooled_norms(references(), config=cfg)
    tables = {"word": word_table([4.0, 9.0, 4.1], file_id="p", channel="mine")}
    envelope = pooled_envelope(norms, tables)
    result = zscores(tables["word"], envelope.unit("word"), text_id=POOLED_TEXT_ID, config=cfg)
    z = result.z["rate_local_syll_per_s"]
    assert abs(z.iloc[0]) < 2.0
    assert z.iloc[1] > 2.5  # the one fast word stands out
