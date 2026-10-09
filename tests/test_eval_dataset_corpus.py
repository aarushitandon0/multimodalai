"""Tests for ``eval/dataset_corpus.py``: the real-corpus path of the LOTO harness.

Montreal Forced Aligner is not installed in this checkout, so these tests build
a dataset tree of their own - generated WAVs from :mod:`eval.synthetic_audio`,
the TextGrid that describes each one exactly, label JSONs in the spec 7.1 form
and a MANIFEST.csv - and point the config at it. Every stage the real corpus
goes through after alignment is then exercised for real: manifest and label
reading, the feature cache, the envelope, the leave-one-reference-out
comparison for clean files, the channel rule, and one complete fold through
:func:`eval.run_loto.run_fold`.

What they do **not** test is forced alignment or whether the extractors recover
a flaw a speaker produced: the flaw spans here are asserted, not performed. The
numbers from this tree are fixtures, never results.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from eval import dataset_corpus as dc
from eval.run_loto import TEST_ONLY_ORIGINS, run_fold

from rostrum.align.mfa import TextGridTier, write_textgrid
from rostrum.config import load_config
from rostrum.types import Alignment

TEXTS: dict[str, str] = {
    "t_alpha": "We can not dedicate, we can not consecrate. "
    "We can not hallow this ground, we can not forget.",
    "t_beta": "The world will little note what we say here, "
    "but it can never forget what they did.",
}
"""Two texts, so a leave-one-text-out fold has a training text to fit on."""

N_READERS = 3
"""``detection.envelope.min_references`` is 3; one fewer and no envelope builds."""


# --------------------------------------------------------------------------- #
# Building the tree
# --------------------------------------------------------------------------- #


def _tiers(alignment: Alignment) -> list[TextGridTier]:
    """The ``words`` and ``phones`` tiers describing one generated utterance.

    Both tiers tile the whole file: silence is an empty interval, which is what
    MFA writes and what :func:`rostrum.align.mfa.read_textgrid` reads gaps from.
    Phones alternate a consonant and an ARPA vowel per syllable, so the
    syllable count the rate features need survives the round trip.
    """
    # Interval edges are rounded to 3 decimals (invariant 2), so the tier end
    # has to be the rounded end too, or the last interval runs past it.
    duration = max(round(float(alignment.duration), 3), round(float(alignment.words[-1].t_end), 3))
    words: list[tuple[float, float, str]] = []
    phones: list[tuple[float, float, str]] = []
    cursor = 0.0
    for word in alignment.words:
        start, end = round(float(word.t_start), 3), round(float(word.t_end), 3)
        if start > cursor:
            words.append((round(cursor, 3), start, ""))
            phones.append((round(cursor, 3), start, ""))
        words.append((start, end, word.text))
        n = max(1, 2 * int(word.n_syllables))
        edges = [round(start + (end - start) * i / n, 3) for i in range(n + 1)]
        for i in range(n):
            if edges[i + 1] > edges[i]:
                phones.append((edges[i], edges[i + 1], "T" if i % 2 == 0 else "AH1"))
        cursor = end
    if duration > cursor:
        words.append((round(cursor, 3), duration, ""))
        phones.append((round(cursor, 3), duration, ""))
    return [
        TextGridTier("words", 0.0, duration, tuple(words)),
        TextGridTier("phones", 0.0, duration, tuple(phones)),
    ]


def _label_json(
    file_id: str,
    *,
    text_id: str,
    origin: str,
    flaws: list[dict],
    passed: bool = True,
) -> dict:
    """One label JSON in the spec 7.1 shape."""
    return {
        "file_id": file_id,
        "source_file": file_id.split("_")[0],
        "text_id": text_id,
        "speaker_id": file_id,
        "origin": origin,
        "tier": "silver",
        "seed": 1337,
        "flaws": flaws,
        "qc": {"utmos_clean": 3.9, "utmos_edited": 3.7, "realign_max_err_ms": 12, "passed": passed},
    }


def _build_tree(root: Path) -> dict[str, dict]:
    """Write audio, alignments, labels, texts and the manifest under ``root``.

    Returns:
        ``{file_id: {"text_id", "origin", "label", "level", "duration"}}``.
    """
    from eval.synthetic_audio import make_utterance, parse_text

    clean_dir = root / "audio" / "clean"
    injected_dir = root / "audio" / "injected"
    human_dir = root / "audio" / "human_flawed"
    alignments = root / "alignments"
    labels = root / "labels"
    texts_dir = root / "texts"
    for directory in (clean_dir, injected_dir, human_dir, alignments, labels, texts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    roster: dict[str, dict] = {}

    def emit(file_id: str, text_id: str, directory: Path, **kwargs) -> None:
        utterance = make_utterance(file_id, directory, text=parse_text(TEXTS[text_id]), **kwargs)
        tiers = _tiers(utterance.alignment)
        write_textgrid(alignments / f"{file_id}.TextGrid", tiers, xmin=0.0, xmax=tiers[0].xmax)
        roster[file_id] = {
            "text_id": text_id,
            "duration": float(utterance.alignment.duration),
            "words": utterance.alignment.words,
        }

    for text_id in TEXTS:
        (texts_dir / f"{text_id}.json").write_text(
            json.dumps({"text_id": text_id, "display": TEXTS[text_id]}), encoding="utf-8"
        )
        for i in range(N_READERS):
            emit(f"{text_id}_r{i:02d}", text_id, clean_dir, seed=10 + i, f0_hz=110.0 + 6.0 * i)
            roster[f"{text_id}_r{i:02d}"].update(origin="clean", label="", level="")

    # One injected take per text: words 7-10 read twice as fast is `rushing`.
    for text_id in TEXTS:
        parent = f"{text_id}_r00"
        file_id = f"{parent}_rushing_L3"
        scale = [1.0] * 18
        for i in (7, 8, 9, 10):
            scale[i] = 2.0
        emit(file_id, text_id, injected_dir, seed=10, f0_hz=110.0, rate_scale=scale)
        words = roster[file_id]["words"]
        (labels / f"{file_id}.json").write_text(
            json.dumps(
                _label_json(
                    file_id,
                    text_id=text_id,
                    origin="injected",
                    flaws=[
                        {
                            "label": "rushing",
                            "t_start": round(float(words[7].t_start), 3),
                            "t_end": round(float(words[10].t_end), 3),
                            "word_idx": [7, 10],
                            "level": "L3",
                            "s": 0.60,
                        }
                    ],
                )
            ),
            encoding="utf-8",
        )
        roster[file_id].update(origin="injected", label="rushing", level="L3")

    # One human take, on the TRAINING text on purpose: it must never be fitted
    # on even though its text is a training text (spec 7.3).
    human_id = "t_beta_human_take1"
    emit(human_id, "t_beta", human_dir, seed=40, f0_hz=118.0, monotone=True)
    words = roster[human_id]["words"]
    (labels / f"{human_id}.json").write_text(
        json.dumps(
            _label_json(
                human_id,
                text_id="t_beta",
                origin="human",
                flaws=[
                    {
                        "label": "monotone",
                        "t_start": round(float(words[2].t_start), 3),
                        "t_end": round(float(words[9].t_end), 3),
                        "word_idx": [2, 9],
                        "level": "L2",
                        "s": 0.35,
                    }
                ],
            )
        ),
        encoding="utf-8",
    )
    roster[human_id].update(origin="human", label="monotone", level="L2")

    # Two rows the loader must drop: a flawed file with no label JSON, and one
    # the QC gates rejected.
    emit("t_alpha_r01_dragging_L2", "t_alpha", injected_dir, seed=11, rate_scale=[0.6] * 18)
    roster["t_alpha_r01_dragging_L2"].update(origin="injected", label="dragging", level="L2")

    rejected = "t_alpha_r02_monotone_L4"
    emit(rejected, "t_alpha", injected_dir, seed=12, monotone=True)
    (labels / f"{rejected}.json").write_text(
        json.dumps(
            _label_json(
                rejected,
                text_id="t_alpha",
                origin="injected",
                flaws=[
                    {"label": "monotone", "t_start": 1.0, "t_end": 2.0, "level": "L4", "s": 0.9}
                ],
                passed=False,
            )
        ),
        encoding="utf-8",
    )
    roster[rejected].update(origin="injected", label="monotone", level="L4")

    lines = ["file_id,text_id,tier,origin,label,level,duration_s,split_fold"]
    for file_id, row in sorted(roster.items()):
        tier = "gold" if file_id.endswith("_r00") else "silver"
        lines.append(
            f"{file_id},{row['text_id']},{tier},{row['origin']},{row['label']},"
            f"{row['level']},{row['duration']:.3f},loto_{row['text_id']}"
        )
    (root / "MANIFEST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return roster


def _config(root: Path, cache: Path):
    """The real config, pointed at the generated tree."""
    return load_config(
        overrides={
            "paths": {
                "data": {
                    "dir": str(root),
                    "audio_clean": str(root / "audio" / "clean"),
                    "audio_injected": str(root / "audio" / "injected"),
                    "audio_human_flawed": str(root / "audio" / "human_flawed"),
                    "alignments": str(root / "alignments"),
                    "labels": str(root / "labels"),
                    "manifest": str(root / "MANIFEST.csv"),
                    "texts": str(root / "texts"),
                },
                "cache": {"features": str(cache)},
            }
        }
    )


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """The generated dataset, its config, and the roster that describes it."""
    base = tmp_path_factory.mktemp("dataset_corpus")
    root = base / "dataset"
    roster = _build_tree(root)
    return _config(root, base / "cache"), root, roster


@pytest.fixture(scope="module")
def corpus(tree):
    """The loaded corpus; lean entries, because this is a fixture not a result."""
    config, _, _ = tree
    return dc.DatasetCorpus.load(config, lean=True)


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


def test_an_empty_checkout_names_every_missing_piece_and_its_make_target(tmp_path):
    config = _config(tmp_path / "nothing", tmp_path / "cache")
    keys = {item.key for item in dc.missing_requirements(config)}
    assert keys == {
        "data.manifest",
        "data.audio_clean",
        "data.alignments",
        "data.labels",
        "cache.features",
    }
    message = dc.describe_missing(config)
    assert "make align" in message and "make features" in message and "make inject" in message
    assert str(tmp_path / "nothing") in message
    assert dc.is_available(config) is False


def test_a_built_tree_is_available(tree, corpus):
    config, _, _ = tree
    assert dc.missing_requirements(config) == []
    assert dc.is_available(config) is True


def test_loading_an_unbuilt_corpus_raises_with_the_same_message(tmp_path):
    config = _config(tmp_path / "nothing", tmp_path / "cache")
    with pytest.raises(dc.CorpusUnavailable, match="not built in this checkout"):
        dc.DatasetCorpus.load(config)


# --------------------------------------------------------------------------- #
# The roster
# --------------------------------------------------------------------------- #


def test_a_flawed_file_without_a_label_json_is_dropped(tree):
    config, _, _ = tree
    assert "t_alpha_r01_dragging_L2" not in {item.file_id for item in dc.read_files(config)}


def test_a_file_the_qc_gates_rejected_is_dropped(tree):
    config, _, _ = tree
    assert "t_alpha_r02_monotone_L4" not in {item.file_id for item in dc.read_files(config)}


def test_a_clean_reading_needs_no_label_json(tree):
    config, _, _ = tree
    clean = {item.file_id: item for item in dc.read_files(config) if item.origin == "clean"}
    assert len(clean) == 2 * N_READERS
    assert all(item.truth == () for item in clean.values())
    assert all(item.level == "L0" for item in clean.values())


def test_the_ground_truth_carries_the_span_the_rung_and_s(tree):
    config, _, roster = tree
    item = {i.file_id: i for i in dc.read_files(config)}["t_alpha_r00_rushing_L3"]
    assert item.origin == "injected"
    assert len(item.truth) == 1
    flaw = item.truth[0]
    words = roster["t_alpha_r00_rushing_L3"]["words"]
    assert flaw.label == "rushing"
    assert flaw.level == "L3"
    assert flaw.s == pytest.approx(0.60)
    assert flaw.t_start == pytest.approx(float(words[7].t_start), abs=1e-3)
    assert flaw.t_end == pytest.approx(float(words[10].t_end), abs=1e-3)


# --------------------------------------------------------------------------- #
# Features and the corpus
# --------------------------------------------------------------------------- #


def test_the_corpus_loads_tables_for_every_usable_file(corpus):
    assert sorted(corpus.texts) == ["t_alpha", "t_beta"]
    assert len(corpus.files) == 2 * N_READERS + 3  # clean + 2 injected + 1 human
    for item in corpus.files:
        assert set(corpus.tables[item.file_id]) == set(dc.UNITS)
        assert corpus.durations[item.file_id] > 1.0


def test_a_second_load_reads_the_feature_cache(tree):
    config, _, _ = tree
    again = dc.DatasetCorpus.load(config, lean=True)
    assert again.n_extracted == 0


def test_only_clean_readings_are_envelope_references(corpus):
    by_text = corpus.references_by_text()
    assert set(by_text) == {"t_alpha", "t_beta"}
    for ids in by_text.values():
        assert len(ids) == N_READERS
        assert all("rushing" not in fid and "human" not in fid for fid in ids)


# --------------------------------------------------------------------------- #
# Ablation 1's reference choice (spec 14.2)
# --------------------------------------------------------------------------- #


def test_a_single_reference_prefers_the_gold_reading(corpus):
    ids = corpus.references_by_text()["t_alpha"]
    assert corpus.pick_references("t_alpha", ids, size=1, seed=1337) == ["t_alpha_r00"]


def test_without_a_gold_reading_the_draw_is_seeded_and_repeatable(corpus):
    ids = ["t_beta_r01", "t_beta_r02"]  # both silver
    first = corpus.pick_references("t_beta", ids, size=1, seed=1337)
    assert first == corpus.pick_references("t_beta", ids, size=1, seed=1337)
    assert len(first) == 1 and first[0] in ids


def test_asking_for_more_references_than_exist_keeps_them_all(corpus):
    ids = corpus.references_by_text()["t_beta"]
    assert corpus.pick_references("t_beta", ids, size=99, seed=1337) == sorted(ids)


# --------------------------------------------------------------------------- #
# One fold of the dataset
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def fold(tree, corpus):
    config, _, _ = tree
    return corpus.fold(config=config, held_out="t_alpha", refine=False)


def test_the_fold_holds_out_one_text_and_keeps_the_other_for_fitting(fold):
    assert fold.held_out == "t_alpha"
    assert fold.train_texts == ("t_beta",)
    assert {case.text_id for case in fold.test()} == {"t_alpha"}
    assert {case.text_id for case in fold.train()} == {"t_beta"}


def test_every_origin_reaches_its_case_unchanged(fold):
    origins = {case.file_id: case.origin for case in fold.cases}
    assert origins["t_alpha_r00"] == "clean"
    assert origins["t_alpha_r00_rushing_L3"] == "injected"
    assert origins["t_beta_human_take1"] == "human"


def test_a_multi_source_envelope_is_not_channel_matched(fold):
    # Three readers, three channels: the spectral and voice-quality features
    # must stay out of the comparison (CLAUDE.md invariant 7).
    assert fold.envelopes["t_alpha"].same_channel is False
    assert all(case.same_channel is False for case in fold.cases)


def test_a_clean_reading_is_scored_against_the_envelope_without_itself(tree, corpus):
    config, _, _ = tree
    with_loo = corpus.fold(config=config, held_out="t_alpha", refine=False, leave_clean_out=True)
    without = corpus.fold(config=config, held_out="t_alpha", refine=False, leave_clean_out=False)

    def pick(bundle):
        return next(c for c in bundle.cases if c.file_id == "t_alpha_r01")

    a, b = pick(with_loo), pick(without)
    # Dropping a reader from the three-reader envelope moves its median and MAD,
    # so the same reading cannot score identically against both.
    left = np.asarray(a.z_by_unit["word"].z, dtype=float)
    right = np.asarray(b.z_by_unit["word"].z, dtype=float)
    assert left.shape == right.shape
    assert not np.array_equal(left, right, equal_nan=True)


def test_the_injected_case_carries_its_rung_and_category(fold):
    case = next(c for c in fold.cases if c.file_id == "t_alpha_r00_rushing_L3")
    assert (case.label, case.level, case.category, case.n_flaws) == ("rushing", "L3", "pace", 1)
    assert case.duration > case.speech_duration > 0.0


def test_a_text_with_too_few_references_has_no_fold(tree, corpus):
    config, _, _ = tree
    with pytest.raises(dc.CorpusUnavailable, match="no text"):
        corpus.fold(config=config, held_out="not_a_text")


# --------------------------------------------------------------------------- #
# Through the harness (prompt 19)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def report(tree):
    config, _, _ = tree
    from eval.run_loto import Variant

    return run_fold(
        config=config,
        held_out="t_alpha",
        corpus="dataset",
        lean=True,
        variant=Variant(name="smoke", tune=False, classifier=False),
    )


def test_a_dataset_fold_runs_end_to_end(report):
    payload = report.payload
    assert payload["corpus"] == "dataset"
    assert payload["held_out"] == "t_alpha"
    assert payload["train_texts"] == ["t_beta"]
    assert payload["leakage_check"]["overlap"] == []
    assert payload["leakage_check"]["origins_checked"] is True
    assert payload["splits"]["injected"]["available"] is True
    assert payload["splits"]["injected"]["n_reference_events"] == 1


def test_the_human_take_of_a_training_text_is_never_fitted_on(report):
    # It is in the fold's training text, so only the origin filter keeps it out.
    assert "human" in TEST_ONLY_ORIGINS
    fitted = report.payload["fit"]["n_train_files"]
    assert fitted == N_READERS + 1  # t_beta's three clean readings and its injected take
    assert "t_beta_human_take1" not in report.payload["cases"]


def test_the_case_register_describes_every_held_out_file(report):
    cases = report.payload["cases"]
    assert set(cases) == {
        "t_alpha_r00",
        "t_alpha_r01",
        "t_alpha_r02",
        "t_alpha_r00_rushing_L3",
    }
    assert cases["t_alpha_r00_rushing_L3"]["label"] == "rushing"
    assert cases["t_alpha_r00_rushing_L3"]["level"] == "L3"
    assert cases["t_alpha_r00"]["origin"] == "clean"
    assert cases["t_alpha_r00"]["n_truth"] == 0


def test_the_clean_denominator_is_the_held_out_clean_audio(report):
    clean = report.payload["clean"]
    assert clean["minutes"] > 0.0
    assert clean["fp_per_min"] >= 0.0


def test_ablation_4_is_refused_on_the_dataset_rather_than_faked(tree):
    config, _, _ = tree
    from eval.run_loto import Variant

    with pytest.raises(NotImplementedError, match="speaker normalisation"):
        run_fold(
            config=config,
            held_out="t_alpha",
            corpus="dataset",
            lean=True,
            variant=Variant(name="no_norm", speaker_normalisation=False),
        )
