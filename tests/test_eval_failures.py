"""Tests for ``eval/failures.py``: the three failure cases of spec 14.3.

The selection rules and the cause heuristics are tested on hand-built payloads
whose answer is known by inspection, which is the only way to assert that the
*right* event was picked - on a real run every candidate is plausible.
"""

from __future__ import annotations

import json

import pytest
from eval.failures import KINDS, collect_failures, failures_markdown, write_failures

from rostrum.config import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


def _payload(
    *,
    false_negatives=(),
    false_positives=(),
    pairs=(),
    cases=None,
    refine: bool = True,
) -> dict:
    """An aggregate payload with one fold and only the blocks failures.py reads."""
    return {
        "corpus": "synthetic",
        "folds": ["loto_a"],
        "variant": {"name": "full"},
        "per_fold": {
            "loto_a": {
                "fold": "loto_a",
                "variant": {"name": "full", "refine": refine},
                "fit": {"thresholds": {"tau_in": 2.5, "tau_out": 1.5, "penalty": 3.0}},
                "cases": cases
                or {
                    "f1": {
                        "origin": "injected",
                        "label": "rushing",
                        "level": "L4",
                        "severity_s": 0.9,
                        "n_truth": 1,
                        "n_regions": 2,
                    }
                },
                "splits": {
                    "injected": {
                        "available": True,
                        "diagnostics": {
                            "false_negatives": list(false_negatives),
                            "false_positives": list(false_positives),
                            "pairs": list(pairs),
                        },
                    },
                    "human": {"available": False},
                },
            }
        },
    }


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def test_the_missed_flaw_with_the_highest_severity_is_chosen():
    payload = _payload(
        false_negatives=[
            {"file_id": "mild", "label": "rushing", "t_start": 1.0, "t_end": 2.0},
            {"file_id": "severe", "label": "dragging", "t_start": 3.0, "t_end": 4.0},
        ],
        cases={
            "mild": {"severity_s": 0.15, "level": "L1", "n_truth": 1, "n_regions": 0},
            "severe": {"severity_s": 0.90, "level": "L4", "n_truth": 1, "n_regions": 0},
        },
    )
    failure = collect_failures(payload)[0]
    assert failure.kind == "false_negative"
    assert failure.file_id == "severe"
    assert failure.evidence["severity_s"] == 0.90


def test_the_false_alarm_with_the_highest_score_is_chosen():
    payload = _payload(
        false_positives=[
            {"file_id": "quiet", "label": "monotone", "t_start": 1.0, "t_end": 2.0, "score": 2.9},
            {"file_id": "loud", "label": "low_energy", "t_start": 5.0, "t_end": 6.0, "score": 7.4},
        ],
        cases={
            "quiet": {"origin": "injected", "n_truth": 1, "n_regions": 2},
            "loud": {"origin": "clean", "n_truth": 0, "n_regions": 1},
        },
    )
    failure = collect_failures(payload)[1]
    assert failure.file_id == "loud"
    assert failure.evidence["score_peak_abs_z"] == 7.4


def test_the_worst_total_edge_error_is_chosen():
    payload = _payload(
        pairs=[
            {
                "file_id": "close",
                "label": "rushing",
                "ref_start": 1.0,
                "ref_end": 2.0,
                "pred_start": 1.02,
                "pred_end": 2.01,
                "onset_error_ms": 20.0,
                "offset_error_ms": 10.0,
                "iou": 0.97,
            },
            {
                "file_id": "far",
                "label": "dragging",
                "ref_start": 4.0,
                "ref_end": 5.0,
                "pred_start": 3.6,
                "pred_end": 5.5,
                "onset_error_ms": -400.0,
                "offset_error_ms": 500.0,
                "iou": 0.52,
            },
        ]
    )
    failure = collect_failures(payload)[2]
    assert failure.file_id == "far"
    assert failure.evidence["onset_error_ms"] == -400.0


def test_every_kind_is_reported_even_when_there_is_nothing_to_report():
    failures = collect_failures(_payload())
    assert [f.kind for f in failures] == [kind for kind, _ in KINDS]
    assert all(f.found is False for f in failures)
    assert all(f.reason for f in failures)


# --------------------------------------------------------------------------- #
# Cause heuristics
# --------------------------------------------------------------------------- #


def test_a_miss_with_no_region_anywhere_blames_the_enter_threshold():
    payload = _payload(
        false_negatives=[{"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0}]
    )
    cause = collect_failures(payload)[0].cause
    assert "tau_in" in cause and "2.5" in cause


def test_a_miss_covered_by_another_label_is_called_a_typing_failure():
    payload = _payload(
        false_negatives=[{"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0}],
        false_positives=[
            {"file_id": "f1", "label": "low_energy", "t_start": 1.1, "t_end": 1.9, "score": 3.0}
        ],
    )
    failure = collect_failures(payload)[0]
    assert "typing failure" in failure.cause
    assert failure.evidence["overlapping_predictions"][0]["label"] == "low_energy"


def test_a_miss_of_the_right_label_names_the_collar_that_rejected_it():
    # Onset 500 ms late and offset 400 ms late against a 1.00 s flaw: both the
    # 200 ms onset collar and the max(200 ms, 20%) offset collar are exceeded.
    payload = _payload(
        false_negatives=[{"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0}],
        false_positives=[
            {"file_id": "f1", "label": "rushing", "t_start": 1.5, "t_end": 2.4, "score": 3.0}
        ],
    )
    failure = collect_failures(payload)[0]
    assert "mis-placed rather than missed" in failure.cause
    assert failure.evidence["best_overlap_onset_error_ms"] == 500.0
    assert failure.evidence["best_overlap_offset_error_ms"] == 400.0
    assert len(failure.evidence["collars_exceeded"]) == 2
    assert failure.evidence["best_overlap_share"] == pytest.approx(0.5)


def test_a_miss_whose_overlap_is_inside_both_collars_says_that_instead():
    # 150 ms late at both ends: inside the collars, so the region was rejected
    # by the one-to-one assignment rather than by the collars themselves.
    payload = _payload(
        false_negatives=[{"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0}],
        false_positives=[
            {"file_id": "f1", "label": "rushing", "t_start": 1.15, "t_end": 2.15, "score": 3.0}
        ],
    )
    failure = collect_failures(payload)[0]
    assert failure.evidence["collars_exceeded"] == []
    assert "inside the collars" in failure.cause


def test_a_false_alarm_on_a_clean_file_blames_the_envelope_spread():
    payload = _payload(
        false_positives=[
            {"file_id": "c1", "label": "monotone", "t_start": 1.0, "t_end": 2.0, "score": 3.1}
        ],
        cases={"c1": {"origin": "clean", "n_truth": 0, "n_regions": 1}},
    )
    failure = collect_failures(payload)[1]
    assert "sigma floor" in failure.cause
    assert failure.evidence["file_has_truth"] is False


def test_a_duplicate_region_over_a_matched_flaw_is_called_fragmentation():
    payload = _payload(
        false_positives=[
            {"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0, "score": 9.0}
        ],
        pairs=[
            {
                "file_id": "f1",
                "label": "rushing",
                "ref_start": 1.0,
                "ref_end": 2.0,
                "pred_start": 1.0,
                "pred_end": 2.0,
                "onset_error_ms": 0.0,
                "offset_error_ms": 0.0,
                "iou": 1.0,
            }
        ],
    )
    failure = collect_failures(payload)[1]
    assert "same label" in failure.cause and "fragmented" in failure.cause
    # The same truth event must not be listed twice in the evidence.
    assert len(failure.evidence["overlapping_truth"]) == 1


def test_a_false_alarm_over_a_real_flaw_of_another_label_is_called_co_occurrence():
    payload = _payload(
        false_positives=[
            {"file_id": "f1", "label": "low_energy", "t_start": 1.0, "t_end": 2.0, "score": 4.0}
        ],
        pairs=[
            {
                "file_id": "f1",
                "label": "rushing",
                "ref_start": 1.0,
                "ref_end": 2.0,
                "pred_start": 1.0,
                "pred_end": 2.0,
                "onset_error_ms": 0.0,
                "offset_error_ms": 0.0,
                "iou": 1.0,
            }
        ],
    )
    assert "co-occurrence" in collect_failures(payload)[1].cause


def test_a_false_alarm_just_after_a_true_flaw_is_called_spill():
    payload = _payload(
        false_positives=[
            {"file_id": "f1", "label": "rushing", "t_start": 2.2, "t_end": 2.8, "score": 3.0}
        ],
        pairs=[
            {
                "file_id": "f1",
                "label": "dragging",
                "ref_start": 1.0,
                "ref_end": 2.0,
                "pred_start": 1.0,
                "pred_end": 2.0,
                "onset_error_ms": 0.0,
                "offset_error_ms": 0.0,
                "iou": 1.0,
            }
        ],
    )
    failure = collect_failures(payload)[1]
    assert "Spill" in failure.cause or "spill" in failure.cause
    assert failure.evidence["gap_to_previous_truth_s"] == pytest.approx(0.2)


def test_a_region_wider_than_the_flaw_on_both_sides_blames_the_unit_grid():
    # Wrong at both ends in opposite directions is an extent problem whatever
    # refinement did, so the shape is read before the refinement window.
    payload = _payload(pairs=_shifted(-300.0, 300.0))
    assert "wider than the flaw" in collect_failures(payload)[2].cause


def test_a_region_narrower_than_the_flaw_blames_the_exit_threshold():
    payload = _payload(pairs=_shifted(300.0, -300.0))
    assert "narrower than the flaw" in collect_failures(payload)[2].cause


def _shifted(onset: float, offset: float) -> list[dict]:
    """One matched pair whose edges are off by the given amounts."""
    return [
        {
            "file_id": "f1",
            "label": "rushing",
            "ref_start": 1.0,
            "ref_end": 2.0,
            "pred_start": 1.0 + onset / 1000.0,
            "pred_end": 2.0 + offset / 1000.0,
            "onset_error_ms": onset,
            "offset_error_ms": offset,
            "iou": 0.5,
        }
    ]


def test_an_error_larger_than_the_refinement_window_is_not_blamed_on_refinement():
    # Refinement searches +/-300 ms and discards a larger shift, so a 400 ms
    # offset error cannot be its doing however tempting the story is.
    failure = collect_failures(_payload(pairs=_shifted(200.0, 400.0), refine=True))[2]
    assert "region extent" in failure.cause
    assert "Ruptures" not in failure.cause
    assert failure.evidence["refinement_could_explain"] is False
    assert failure.evidence["refinement_window_ms"] == 300.0


def test_a_one_sided_error_inside_the_window_points_at_ruptures():
    failure = collect_failures(_payload(pairs=_shifted(40.0, 240.0), refine=True))[2]
    assert "Ruptures" in failure.cause
    assert failure.evidence["refinement_could_explain"] is True


def test_without_refinement_the_same_error_is_blamed_on_region_extent():
    failure = collect_failures(_payload(pairs=_shifted(40.0, 240.0), refine=False))[2]
    assert "region extent" in failure.cause
    assert failure.evidence["refined"] is False


def test_the_cause_quotes_the_configs_merge_gap_rather_than_a_literal(config):
    payload = _payload(pairs=_shifted(-300.0, 300.0))
    failure = collect_failures(payload, config=config)[2]
    assert str(config.detection["regions"]["merge_gap_s"]) in failure.cause


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #


def test_the_markdown_names_the_file_the_cause_and_the_corpus_caveat():
    payload = _payload(
        false_negatives=[{"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0}]
    )
    text = failures_markdown(payload, collect_failures(payload))
    assert "# Where it fails" in text
    assert "`f1`" in text
    assert "suspected cause" in text.lower()
    assert "not of speech analysis" in text  # the synthetic caveat
    for _, title in KINDS:
        assert title in text


def test_writing_produces_a_markdown_and_a_json_beside_it(config, tmp_path):
    metrics = tmp_path / "metrics.json"
    payload = _payload(
        false_negatives=[{"file_id": "f1", "label": "rushing", "t_start": 1.0, "t_end": 2.0}]
    )
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "failures.md"
    record = write_failures(config=config, metrics=metrics, out=out)
    assert out.is_file() and out.with_suffix(".json").is_file()
    assert record["failures"][0]["file_id"] == "f1"
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["corpus"] == (
        "synthetic"
    )


def test_a_missing_metrics_file_is_an_error_not_an_empty_report(config, tmp_path):
    with pytest.raises(FileNotFoundError, match="make eval"):
        write_failures(config=config, metrics=tmp_path / "nope.json", out=tmp_path / "f.md")
