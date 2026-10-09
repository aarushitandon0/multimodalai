"""End-to-end envelope -> z -> signals -> regions -> refine on the synthetic corpus.

The unit tests in ``test_envelope_build``, ``test_detect_zscore``,
``test_detect_regions`` and ``test_detect_refine`` each pin one stage. This file
checks that the stages compose: that a flaw injected at a known word span comes
back out as a region at the right seconds with the right label, and that two
runs produce byte-identical output (CLAUDE.md invariant 13).

These are *synthetic* feature tables (:mod:`eval.synthetic`), so a pass here
says region formation works, not that the feature extractors recover a flaw a
speaker put into audio. Only the real injected files answer that, and they need
Montreal Forced Aligner.
"""

from __future__ import annotations

import json

import pytest
from eval.synthetic import Flaw, build_corpus, make_recording, truth_events

from rostrum.config import load_config
from rostrum.detect.refine import RefinementSettings, refine_regions
from rostrum.detect.regions import RegionThresholds, build_regions
from rostrum.detect.tune import TuningCase, event_f1, events_from_regions, grid_search
from rostrum.detect.zscore import envelope_z, trigger_signals
from rostrum.envelope.build import Reference, build_fold_envelopes, leave_one_out

TEXTS = ("syn_a", "syn_b", "syn_c")
N_READERS = 6


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(texts=TEXTS, n_readers=N_READERS)


@pytest.fixture(scope="module")
def envelopes(cfg, corpus):
    """Envelopes for the fold that holds out the first text."""
    references, _ = corpus
    refs = [
        Reference(file_id=r.file_id, channel_id=r.channel_id, tables=r.tables)
        for text in TEXTS
        for r in references[text]
    ]
    return build_fold_envelopes(
        {text: [r.file_id for r in references[text]] for text in TEXTS},
        refs,
        fold=f"loto_{TEXTS[0]}",
        train_texts=list(TEXTS[1:]),
        config=cfg,
        frames={r.file_id: r.frames for text in TEXTS for r in references[text]},
        word_spans={r.file_id: r.word_spans for text in TEXTS for r in references[text]},
    )


def detect(cfg, envelope, recording, *, thresholds=None, refine=False):
    """Run the whole detector on one recording."""
    z = envelope_z(recording.tables, envelope, config=cfg, channel_matched=True)
    signals, skipped = trigger_signals(z, config=cfg, same_channel=True)
    regions = build_regions(
        signals,
        z,
        config=cfg,
        thresholds=thresholds or RegionThresholds.from_config(cfg),
    )
    if refine:
        regions = refine_regions(
            regions, recording.frames, settings=RefinementSettings.from_config(cfg)
        )
    return regions, signals, skipped


# --------------------------------------------------------------------------- #
# Envelopes
# --------------------------------------------------------------------------- #


def test_every_text_gets_an_envelope_with_n_readers(cfg, envelopes):
    assert set(envelopes) == set(TEXTS)
    for text, envelope in envelopes.items():
        assert envelope.n_references == N_READERS, text
        assert envelope.same_channel
        assert set(envelope.units) == {"word", "gap", "phrase"}
        assert envelope.curves is not None


def test_the_fold_shares_one_sigma_floor_per_feature(cfg, envelopes):
    floors = [
        tuple(sorted(e.unit("word").envelope.sigma_floor.items())) for e in envelopes.values()
    ]
    assert len(set(floors)) == 1


def test_leave_one_out_works_on_every_reader(cfg, envelopes):
    envelope = envelopes[TEXTS[1]]
    for file_id in envelope.reference_ids:
        reduced = leave_one_out(envelope, file_id)
        assert reduced.n_references == N_READERS - 1
        assert file_id not in reduced.reference_ids


# --------------------------------------------------------------------------- #
# A clean reader against the envelope of their own text
# --------------------------------------------------------------------------- #


def test_a_clean_reader_produces_few_regions(cfg, corpus, envelopes):
    """Spec 14.3 budgets <= 1 false positive per minute on clean speech."""
    references, _ = corpus
    text = TEXTS[1]
    total_regions = 0
    total_seconds = 0.0
    for recording in references[text]:
        regions, _, _ = detect(cfg, envelopes[text], recording)
        total_regions += len(regions)
        total_seconds += float(recording.tables["word"]["t_end"].max())

    per_minute = total_regions / (total_seconds / 60.0)
    # A reader who IS in the envelope should essentially never be flagged; the
    # bound is loose because each reader contributes to their own median.
    assert per_minute < 6.0, f"{per_minute:.1f} regions/min on clean readings"


# --------------------------------------------------------------------------- #
# Each injected flaw comes back
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("flaw", "expected_label"),
    [
        (Flaw("rushing", 7, 11), "rushing"),
        (Flaw("dragging", 13, 16), "dragging"),
        (Flaw("monotone", 6, 11), "monotone"),
        (Flaw("low_energy", 12, 17), "low_energy"),
        (Flaw("fading", 12, 17), "low_energy"),
        (Flaw("missing_pause", gap_idx=5), "missing_pause"),
        (Flaw("hesitation_pause", gap_idx=13), "hesitation_pause"),
        (Flaw("misplaced_stress", 4, 5), "misplaced_stress"),
        (Flaw("mumbling", 8, 13), "mumbling"),
    ],
)
def test_an_injected_flaw_is_detected_with_the_right_label(cfg, envelopes, flaw, expected_label):
    text = TEXTS[1]
    recording = make_recording(
        f"p_{flaw.label}", text, seed=99, flaws=[flaw], with_channel_features=True
    )
    regions, _, _ = detect(cfg, envelopes[text], recording)

    truth = truth_events(recording)[0]
    hits = [
        r
        for r in regions
        if r.label == expected_label
        # Overlapping the truth span, which is a far weaker claim than the
        # 200 ms collar - the collar is measured by the acceptance run, this
        # test only asks whether the right label fires in the right place.
        and r.t_end > truth.t_start and r.t_start < truth.t_end
    ]
    assert hits, (
        f"{expected_label} not found over {truth.t_start:.2f}-{truth.t_end:.2f} s; "
        f"got {[(r.label, round(r.t_start, 2), round(r.t_end, 2)) for r in regions]}"
    )
    assert hits[0].evidence, "a detected region must carry its numeric evidence"
    assert hits[0].envelope_n == N_READERS


def test_mumbling_is_withheld_without_a_same_channel_reference(cfg, envelopes):
    text = TEXTS[1]
    recording = make_recording(
        "p_mumble", text, seed=99, flaws=[Flaw("mumbling", 8, 13)], with_channel_features=True
    )
    z = envelope_z(recording.tables, envelopes[text], config=cfg, channel_matched=False)
    signals, skipped = trigger_signals(z, config=cfg, same_channel=False)
    assert "mumbling" not in {s.label for s in signals}
    assert "mumbling" in {s.label for s in skipped}


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #


def test_refinement_never_exceeds_the_shift_limit_end_to_end(cfg, corpus, envelopes):
    _, participants = corpus
    settings = RefinementSettings.from_config(cfg)
    checked = 0
    for text in TEXTS:
        for recording in participants[text]:
            regions, _, _ = detect(cfg, envelopes[text], recording, refine=True)
            for region in regions:
                assert abs(region.t_start - region.t_start_unrefined) < settings.max_shift_s
                assert abs(region.t_end - region.t_end_unrefined) < settings.max_shift_s
                assert region.t_end > region.t_start
                checked += 1
    assert checked > 0, "the corpus must produce regions to check"


def test_refinement_does_not_change_which_regions_exist(cfg, corpus, envelopes):
    """Spec 10.5 moves boundaries; it must not add or remove a region."""
    _, participants = corpus
    text = TEXTS[1]
    for recording in participants[text]:
        raw, _, _ = detect(cfg, envelopes[text], recording, refine=False)
        refined, _, _ = detect(cfg, envelopes[text], recording, refine=True)
        assert [r.label for r in raw] == [r.label for r in refined]
        assert [r.word_idx for r in raw] == [r.word_idx for r in refined]


# --------------------------------------------------------------------------- #
# Tuning on the fold
# --------------------------------------------------------------------------- #


def test_tuning_on_the_training_fold_does_not_see_the_held_out_text(cfg, corpus, envelopes):
    _, participants = corpus
    cases = []
    for text in TEXTS:
        z_cache = {}
        for recording in participants[text]:
            z = envelope_z(recording.tables, envelopes[text], config=cfg, channel_matched=True)
            signals, _ = trigger_signals(z, config=cfg, same_channel=True)
            z_cache[recording.file_id] = (signals, z)
            cases.append(
                TuningCase(
                    file_id=recording.file_id,
                    text_id=text,
                    signals=signals,
                    z_by_unit=z,
                    truth=truth_events(recording),
                    frames=recording.frames,
                )
            )

    tuned = grid_search(
        cases,
        fold=f"loto_{TEXTS[0]}",
        train_texts=list(TEXTS[1:]),
        config=cfg,
        refine=False,
    )
    held_out_files = {r.file_id for r in participants[TEXTS[0]]}
    assert tuned.n_cases == len(cases) - len(held_out_files)
    assert tuned.train_texts == tuple(sorted(TEXTS[1:]))
    assert tuned.event_f1 > 0.0


def test_tuning_improves_the_training_objective(cfg, corpus, envelopes):
    _, participants = corpus
    cases = []
    for text in TEXTS[1:]:
        for recording in participants[text]:
            z = envelope_z(recording.tables, envelopes[text], config=cfg, channel_matched=True)
            signals, _ = trigger_signals(z, config=cfg, same_channel=True)
            cases.append(
                TuningCase(
                    file_id=recording.file_id,
                    text_id=text,
                    signals=signals,
                    z_by_unit=z,
                    truth=truth_events(recording),
                )
            )
    tuned = grid_search(
        cases, fold=f"loto_{TEXTS[0]}", train_texts=list(TEXTS[1:]), config=cfg, refine=False
    )
    default = RegionThresholds.from_config(cfg)
    baseline = next(
        c for c in tuned.grid if c["tau_in"] == default.tau_in and c["tau_out"] == default.tau_out
    )
    assert tuned.event_f1 >= baseline["event_f1"]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_the_synthetic_corpus_is_reproducible():
    one = make_recording("x", "t", seed=7, flaws=[Flaw("rushing", 3, 5)])
    two = make_recording("x", "t", seed=7, flaws=[Flaw("rushing", 3, 5)])
    assert one.tables["word"].equals(two.tables["word"])
    assert one.tables["gap"].equals(two.tables["gap"])


def test_two_runs_give_an_identical_rounded_json(cfg, corpus, envelopes):
    """CLAUDE.md invariant 13: identical SHA-256 of the rounded output."""
    _, participants = corpus
    text = TEXTS[0]

    def run() -> str:
        payload = []
        for recording in participants[text]:
            regions, _, skipped = detect(cfg, envelopes[text], recording, refine=True)
            payload.append(
                {
                    "file_id": recording.file_id,
                    "regions": [r.to_dict() for r in regions],
                    "skipped": sorted((s.label, s.reason) for s in skipped),
                }
            )
        return json.dumps(payload, indent=2, sort_keys=True)

    first, second = run(), run()
    assert first == second

    import hashlib

    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()


def test_region_order_is_stable_and_ids_follow_onsets(cfg, corpus, envelopes):
    _, participants = corpus
    for text in TEXTS:
        for recording in participants[text]:
            regions, _, _ = detect(cfg, envelopes[text], recording, refine=True)
            assert [r.region_id for r in regions] == list(range(len(regions)))


def test_an_event_f1_can_be_computed_over_the_whole_fold(cfg, corpus, envelopes):
    """The acceptance number, computed inside the test suite as a smoke check."""
    _, participants = corpus
    truth, predicted = [], []
    for text in TEXTS[1:]:
        for recording in participants[text]:
            regions, _, _ = detect(cfg, envelopes[text], recording, refine=True)
            predicted += events_from_regions(regions, file_id=recording.file_id)
            truth += truth_events(recording)

    metrics = event_f1(truth, predicted)
    assert metrics["micro"]["tp"] > 0
    assert 0.0 <= metrics["f1"] <= 1.0
    # Recall is the claim this file can honestly make: the flaws are found.
    # Precision and the collar are reported by scripts/detect_report.py.
    assert metrics["micro"]["recall"] > 0.5
