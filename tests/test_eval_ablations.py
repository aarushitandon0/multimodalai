"""Tests for ``eval/ablations.py`` (spec 14.2, prompt 20).

The table-building, the delta arithmetic and the "not run" paths are tested on
hand-built payloads so they run in milliseconds. One end-to-end run over two
synthetic texts with two rows is exercised too, because a table whose parts all
pass separately can still be wired to the wrong variant.
"""

from __future__ import annotations

import json

import pytest
from eval.ablations import (
    ABLATIONS,
    HEADLINE_METRICS,
    AblationRow,
    ablations_markdown,
    run_ablations,
)
from eval.run_loto import Variant

from rostrum.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


# --------------------------------------------------------------------------- #
# The ablation set
# --------------------------------------------------------------------------- #


def test_the_five_ablations_of_the_spec_are_all_present():
    names = [a.name for a in ABLATIONS]
    assert names == [
        "full",
        "single_reference",
        "rules_only",
        "no_refinement",
        "no_speaker_norm",
        "human_test",
    ]


def test_each_row_changes_exactly_one_thing_against_the_full_system():
    full = next(a for a in ABLATIONS if a.name == "full")
    assert full.variant is not None
    baseline = full.variant.to_dict()
    for ablation in ABLATIONS:
        if ablation.variant is None or ablation.name == "full":
            continue
        differing = {
            key
            for key, value in ablation.variant.to_dict().items()
            if value != baseline[key] and key not in ("name", "description")
        }
        assert len(differing) == 1, f"{ablation.name} changes {differing}"


def test_only_detection_is_overridden_so_the_feature_cache_key_is_unchanged():
    # The feature cache key includes the config hash; an override outside
    # `detection` would make a row's features incomparable with the full run's.
    for ablation in ABLATIONS:
        for key in ablation.overrides:
            assert key in {"envelope", "regions", "refinement", "classifier", "deviation"}


def test_speaker_normalisation_is_only_ablatable_on_the_synthetic_corpus():
    row = next(a for a in ABLATIONS if a.name == "no_speaker_norm")
    assert row.corpora == ("synthetic",)
    assert "SIMULATED" in row.note


def test_the_human_row_is_not_its_own_run():
    row = next(a for a in ABLATIONS if a.name == "human_test")
    assert row.runs is False and row.variant is None


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def _rows() -> list[dict]:
    full = AblationRow(
        name="full",
        title="Full system",
        available=True,
        metrics={"event_f1_200ms": 0.80, "onset_offset_mae_ms": 100.0},
    )
    worse = AblationRow(
        name="no_refinement",
        title="3. No boundary refinement",
        available=True,
        metrics={"event_f1_200ms": 0.70, "onset_offset_mae_ms": 160.0},
        delta={"event_f1_200ms": -0.10, "onset_offset_mae_ms": 60.0},
    )
    absent = AblationRow(
        name="no_speaker_norm",
        title="4. No speaker normalisation",
        reason="not available on the dataset corpus",
    )
    return [row.to_dict() for row in (full, worse, absent)]


def _payload() -> dict:
    return {
        "corpus": "synthetic",
        "config_hash": "abc123def456",
        "code_version": "testing",
        "texts": ["syn_a", "syn_b"],
        "metrics": [
            {"key": key, "label": label, "better": better}
            for key, label, better in HEADLINE_METRICS
        ],
        "rows": _rows(),
    }


def test_the_markdown_marks_a_worse_number_as_worse_in_both_directions():
    text = ablations_markdown(_payload())
    # F1 fell, which is worse for a higher-is-better metric...
    assert "0.7 (-0.1 worse)" in text
    # ...and the MAE rose, which is worse for a lower-is-better one.
    assert "160 (+60 worse)" in text


def test_the_markdown_states_the_corpus_caveat_and_the_unrun_rows():
    text = ablations_markdown(_payload())
    assert "corpus: **synthetic**" in text
    assert "not a dataset result" in text
    assert "not run" in text
    assert "Not run:** not available on the dataset corpus" in text


def test_every_headline_metric_gets_a_column():
    text = ablations_markdown(_payload())
    for _, label, _ in HEADLINE_METRICS:
        assert label in text


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_two_rows_run_end_to_end_and_the_deltas_are_against_the_full_row(config, tmp_path):
    out = tmp_path / "ablations.json"
    payload = run_ablations(
        config=config,
        corpus="synthetic",
        texts=("syn_a", "syn_b"),
        only=["no_refinement"],
        out=out,
    )
    rows = {row["name"]: row for row in payload["rows"]}
    assert set(rows) == {"full", "no_refinement"}
    assert rows["full"]["available"] is True
    assert rows["full"]["delta"] == {}
    assert rows["no_refinement"]["variant"]["refine"] is False
    for key, _, _ in HEADLINE_METRICS:
        actual = rows["no_refinement"]["metrics"][key]
        base = rows["full"]["metrics"][key]
        if actual is None or base is None:
            continue
        assert rows["no_refinement"]["delta"][key] == pytest.approx(actual - base, abs=1e-4)
    assert out.is_file()
    assert (tmp_path / "ablations.md").is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["corpus"] == "synthetic"


@pytest.mark.slow
def test_the_rules_only_row_reports_both_macro_f1_values(config, tmp_path):
    payload = run_ablations(
        config=config,
        corpus="synthetic",
        texts=("syn_a", "syn_b"),
        only=["rules_only"],
        out=tmp_path / "ablations.json",
    )
    rows = {row["name"]: row for row in payload["rows"]}
    typing = rows["full"]["extra"]["typing"]
    assert typing["rule_macro_f1"] is not None
    # The full run has a classifier; the rules-only run must not.
    assert rows["rules_only"]["extra"]["typing"]["classifier_macro_f1"] is None
    assert rows["rules_only"]["variant"]["classifier"] is False


def test_a_dataset_ablation_without_a_dataset_refuses(config, tmp_path):
    from eval.dataset_corpus import CorpusUnavailable, is_available

    if is_available(config):  # pragma: no cover - only on a built checkout
        pytest.skip("this checkout has a built dataset")
    with pytest.raises(CorpusUnavailable):
        run_ablations(
            config=config, corpus="dataset", only=["no_refinement"], out=tmp_path / "a.json"
        )


def test_the_variants_are_the_ones_the_harness_knows():
    # A Variant field renamed in run_loto must break here, not silently turn an
    # ablation into a copy of the full system.
    fields = set(Variant().to_dict())
    for ablation in ABLATIONS:
        if ablation.variant is None:
            continue
        assert set(ablation.variant.to_dict()) == fields
