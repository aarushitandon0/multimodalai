"""One-call extraction, phrase segmentation and the Parquet cache (spec 9).

The caching acceptance test from the build prompt is
``test_cache_round_trip_returns_identical_tables``: saving a bundle and loading
it back must give tables that compare equal, column for column and dtype for
dtype.

eGeMAPS is exercised only through its availability contract: openSMILE ships a
compiled extension that can be blocked by the host, so the tests assert the
behaviour in both cases rather than requiring one.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rostrum.config import load_config
from rostrum.features import egemaps as EG
from rostrum.features import extract_all as X
from rostrum.features.frames import FrameGrid
from rostrum.features.pauses import gaps_from_words
from rostrum.types import Alignment, Gap, Word

SR = 16000
GRID = FrameGrid()
WORDS = ["now", "is", "the", "time", "for", "all", "good", "folk"]
BOUNDARY_AFTER = 3
"""A sentence break after the fourth word, so the utterance is two phrases."""


def syllable(f_hz: float, seconds: float, amplitude: float, sr: int = SR) -> np.ndarray:
    """A harmonic buzz with a short raised onset and offset."""
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    out = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        out += weight * np.sin(2.0 * np.pi * f_hz * harmonic * t)
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.03)
    return amplitude * out * envelope / 1.9


def synthetic() -> tuple[np.ndarray, Alignment]:
    """A two-phrase synthetic utterance with one emphasised word and its alignment."""
    parts = [np.zeros(int(0.2 * SR))]
    words: list[Word] = []
    t = 0.2
    for i, text in enumerate(WORDS):
        emphatic = i == BOUNDARY_AFTER
        seconds = 0.45 if emphatic else 0.28
        parts.append(syllable(170.0 if emphatic else 140.0, seconds, 0.6 if emphatic else 0.3))
        words.append(
            Word(
                idx=i,
                text=text,
                t_start=round(t, 3),
                t_end=round(t + seconds, 3),
                phones=["N", "AW1"],
                n_syllables=1,
                display=text,
            )
        )
        t += seconds
        gap = 0.35 if i == BOUNDARY_AFTER else 0.07
        parts.append(np.zeros(int(gap * SR)))
        t += gap

    samples = np.concatenate(parts).astype(np.float32)
    gaps = gaps_from_words(words)
    for gap in gaps:
        if gap.prev_word_idx == BOUNDARY_AFTER:
            gap.label = "boundary"
            gap.punctuation = "."
    return samples, Alignment(
        file_id="synth01",
        text_id="t1",
        speaker_id="sp1",
        channel_id="ch1",
        tier="silver",
        words=words,
        gaps=gaps,
        duration=samples.size / SR,
        sample_rate=SR,
        audio_sha256="a" * 64,
    )


@pytest.fixture(scope="module")
def config():
    """The project configuration."""
    return load_config()


@pytest.fixture(scope="module")
def bundle(config):
    """One extraction, shared across the assertions - it is not cheap."""
    samples, alignment = synthetic()
    return X.extract_all(samples, SR, alignment, config=config, crosscheck=False)


# --------------------------------------------------------------------------- #
# Phrase segmentation
# --------------------------------------------------------------------------- #


def test_phrases_are_cut_at_boundary_gaps() -> None:
    _, alignment = synthetic()
    spans = X.phrase_spans(alignment.words, alignment.gaps)
    assert len(spans) == 2
    assert spans[0].first_word_idx == 0
    assert spans[0].last_word_idx == BOUNDARY_AFTER
    assert spans[1].first_word_idx == BOUNDARY_AFTER + 1
    assert spans[1].last_word_idx == len(WORDS) - 1
    assert spans[0].n_words == 4
    assert spans[0].duration == pytest.approx(spans[0].t_end - spans[0].t_start)


def test_an_alignment_with_no_boundary_gap_is_one_phrase() -> None:
    _, alignment = synthetic()
    for gap in alignment.gaps:
        gap.label = "none"
    spans = X.phrase_spans(alignment.words, alignment.gaps)
    assert len(spans) == 1
    assert spans[0].n_words == len(WORDS)


def test_no_words_means_no_phrases() -> None:
    assert X.phrase_spans([], []) == []


def test_a_trailing_boundary_gap_does_not_create_an_empty_phrase() -> None:
    words = [
        Word(idx=0, text="a", t_start=0.0, t_end=0.3, phones=["AH1"], n_syllables=1),
        Word(idx=1, text="b", t_start=0.4, t_end=0.7, phones=["AH1"], n_syllables=1),
    ]
    gaps = [Gap(idx=0, prev_word_idx=0, next_word_idx=1, t_start=0.3, t_end=0.4, label="boundary")]
    spans = X.phrase_spans(words, gaps)
    assert [s.n_words for s in spans] == [1, 1]


# --------------------------------------------------------------------------- #
# The bundle
# --------------------------------------------------------------------------- #


def test_bundle_has_one_row_per_unit(bundle) -> None:
    assert len(bundle.words) == len(WORDS)
    assert len(bundle.gaps) == len(WORDS) - 1
    assert len(bundle.phrases) == 2
    assert len(bundle.frames) == bundle.meta["n_frames"]


def test_every_table_follows_the_feature_table_contract(bundle) -> None:
    for unit in ("word", "gap", "phrase"):
        table = bundle.table(unit)
        assert table.index.name == "unit_idx"
        assert table.attrs["unit"] == unit
        assert table.attrs["file_id"] == "synth01"
        assert table.attrs["channel_id"] == "ch1"


def test_unknown_unit_is_rejected(bundle) -> None:
    with pytest.raises(KeyError, match="unknown unit"):
        bundle.table("syllable")


def test_word_table_carries_every_feature_block(bundle) -> None:
    columns = set(bundle.words.columns)
    for expected in (
        "text",
        "t_start",
        "f0_mean_st",  # F0
        "energy_mean_z",  # EN
        "word_duration_s",  # DUR
        "rate_local_syll_per_s",  # RATE
        "prominence_z",  # PROM
        "spectral_tilt_db_per_khz",  # SPEC
        "mfcc_00",  # MFCC
        "hnr_db",  # VQ
    ):
        assert expected in columns, expected


def test_frame_curves_share_one_length_and_start_at_the_window_centre(bundle) -> None:
    n = bundle.meta["n_frames"]
    assert (bundle.frames.notna().any() | bundle.frames.isna().any()).all()
    assert len(bundle.frames) == n
    assert bundle.frames["time_s"].iloc[0] == pytest.approx(GRID.window_s / 2)
    assert "mfcc_00" not in bundle.frames.columns  # documented: not cached per frame


def test_the_emphasised_word_is_the_most_prominent(bundle) -> None:
    assert int(bundle.words["prominence"].idxmax()) == BOUNDARY_AFTER


def test_the_boundary_gap_is_the_only_pause(bundle) -> None:
    pauses = bundle.gaps[bundle.gaps["gap_is_pause"]]
    assert list(pauses.index) == [BOUNDARY_AFTER]
    assert bool(pauses.iloc[0]["gap_is_boundary"])


def test_phrase_pitch_range_is_measured_against_the_final_low(bundle) -> None:
    final_low = bundle.meta["final_low_st"]
    assert np.isfinite(final_low)
    row = bundle.phrases.iloc[0]
    assert row["phrase_f0_range_st"] == pytest.approx(row["phrase_f0_max_st"] - final_low)


def test_metadata_records_the_pitch_range_and_the_cost(bundle) -> None:
    meta = bundle.meta
    assert meta["pitch_floor_hz"] > 0
    assert meta["pitch_ceiling_hz"] > meta["pitch_floor_hz"]
    assert meta["audio_duration_s"] > 0
    assert meta["elapsed_s"] > 0
    assert set(meta["stage_timings_s"]) >= {"pitch", "energy", "spectral", "mfcc"}
    assert meta["feature_cache_version"] == X.FEATURE_CACHE_VERSION
    assert "per minute of audio" in bundle.summary()


def test_a_sample_rate_mismatch_is_refused(config) -> None:
    samples, alignment = synthetic()
    with pytest.raises(ValueError, match="frame grid"):
        X.extract_all(samples, 22050, alignment, config=config, crosscheck=False)


def test_optional_blocks_can_be_switched_off(config) -> None:
    samples, alignment = synthetic()
    lean = X.extract_all(
        samples,
        SR,
        alignment,
        config=config,
        crosscheck=False,
        with_mfcc=False,
        with_voice_quality=False,
        with_egemaps=False,
    )
    assert "mfcc_00" not in lean.words.columns
    assert "hnr_db" not in lean.words.columns
    assert "spectral_tilt_db_per_khz" in lean.words.columns


def test_duplicate_feature_columns_are_refused() -> None:
    import pandas as pd

    _, alignment = synthetic()
    part = pd.DataFrame(
        {"f0_mean_st": [0.0]},
        index=pd.Index([0], name="unit_idx"),
    )
    with pytest.raises(ValueError, match="duplicate feature columns"):
        X.word_table(part, part, words=alignment.words[:1])


# --------------------------------------------------------------------------- #
# Cache keys
# --------------------------------------------------------------------------- #


def test_alignment_fingerprint_changes_with_a_moved_boundary() -> None:
    _, first = synthetic()
    _, second = synthetic()
    assert X.alignment_fingerprint(first) == X.alignment_fingerprint(second)
    second.words[2].t_end += 0.05
    assert X.alignment_fingerprint(first) != X.alignment_fingerprint(second)


def test_alignment_fingerprint_ignores_the_aligner_version() -> None:
    _, first = synthetic()
    _, second = synthetic()
    second.aligner = "mfa 9.9.9"
    second.oov_words = ["folk"]
    assert X.alignment_fingerprint(first) == X.alignment_fingerprint(second)


@pytest.mark.parametrize(
    "changed",
    [
        {"audio_sha256": "b" * 64},
        {"config_hash": "other"},
        {"alignment_hash": "other"},
        {"options": ["with_mfcc"]},
    ],
)
def test_every_part_of_the_key_changes_the_key(changed: dict) -> None:
    base = {
        "audio_sha256": "a" * 64,
        "config_hash": "cfg",
        "alignment_hash": "al",
        "options": ["with_mfcc", "with_spectral"],
    }
    assert X.cache_key(**base) != X.cache_key(**{**base, **changed})


def test_the_key_is_stable_across_option_order() -> None:
    first = X.cache_key(audio_sha256="a", config_hash="c", options=["b", "a"])
    second = X.cache_key(audio_sha256="a", config_hash="c", options=["a", "b"])
    assert first == second


# --------------------------------------------------------------------------- #
# The cache: the acceptance test
# --------------------------------------------------------------------------- #


def test_cache_round_trip_returns_identical_tables(bundle, tmp_path) -> None:
    key = "k" * 16
    X.save_bundle(bundle, tmp_path, key)
    loaded = X.load_bundle(tmp_path, key)
    assert loaded is not None
    for name in ("words", "gaps", "phrases", "frames"):
        original = getattr(bundle, name)
        cached = getattr(loaded, name)
        assert cached.equals(original), name
        assert list(cached.columns) == list(original.columns), name
        assert cached.attrs["unit"] == original.attrs["unit"], name


def test_a_missing_entry_loads_as_none(tmp_path) -> None:
    assert X.load_bundle(tmp_path, "nothing-here") is None


def test_an_incomplete_entry_loads_as_none(bundle, tmp_path) -> None:
    key = "k" * 16
    X.save_bundle(bundle, tmp_path, key)
    (tmp_path / key / X.CACHE_FILES["frames"]).unlink()
    assert X.load_bundle(tmp_path, key) is None


def test_the_entry_records_its_key_and_provenance(bundle, tmp_path) -> None:
    key = "k" * 16
    target = X.save_bundle(bundle, tmp_path, key)
    meta = json.loads((target / X.CACHE_FILES["meta"]).read_text(encoding="utf-8"))
    assert meta["cache_key"] == key
    assert meta["audio_sha256"] == "a" * 64
    assert set(meta["attrs"]) == {"words", "gaps", "phrases", "frames"}


def test_extract_all_cached_computes_then_reads_back(config, tmp_path) -> None:
    samples, alignment = synthetic()
    first, from_cache = X.extract_all_cached(
        samples, SR, alignment, config=config, cache_dir=tmp_path, crosscheck=False
    )
    assert from_cache is False
    second, from_cache = X.extract_all_cached(
        samples, SR, alignment, config=config, cache_dir=tmp_path, crosscheck=False
    )
    assert from_cache is True
    assert second.words.equals(first.words)
    assert second.frames.equals(first.frames)


def test_refresh_recomputes_an_existing_entry(config, tmp_path) -> None:
    samples, alignment = synthetic()
    X.extract_all_cached(
        samples, SR, alignment, config=config, cache_dir=tmp_path, crosscheck=False
    )
    _, from_cache = X.extract_all_cached(
        samples,
        SR,
        alignment,
        config=config,
        cache_dir=tmp_path,
        crosscheck=False,
        refresh=True,
    )
    assert from_cache is False


def test_a_changed_alignment_does_not_hit_the_cache(config, tmp_path) -> None:
    samples, alignment = synthetic()
    X.extract_all_cached(
        samples, SR, alignment, config=config, cache_dir=tmp_path, crosscheck=False
    )
    alignment.words[2].t_end += 0.05
    _, from_cache = X.extract_all_cached(
        samples, SR, alignment, config=config, cache_dir=tmp_path, crosscheck=False
    )
    assert from_cache is False


def test_without_an_audio_hash_nothing_is_cached(config, tmp_path) -> None:
    samples, alignment = synthetic()
    alignment.audio_sha256 = ""
    _, from_cache = X.extract_all_cached(
        samples, SR, alignment, config=config, cache_dir=tmp_path, crosscheck=False
    )
    assert from_cache is False
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# eGeMAPS availability contract
# --------------------------------------------------------------------------- #


def test_egemaps_is_either_present_with_columns_or_absent_with_a_reason(bundle) -> None:
    if bundle.meta["egemaps_available"]:
        assert any(c.startswith("eg_") for c in bundle.phrases.columns)
        assert bundle.meta["egemaps_error"] is None
    else:
        assert not any(c.startswith("eg_") for c in bundle.phrases.columns)
        assert bundle.meta["egemaps_error"]


def test_strict_egemaps_raises_when_opensmile_is_unavailable() -> None:
    if EG.available():
        pytest.skip("openSMILE loads in this environment")
    samples, alignment = synthetic()
    spans = [(s.t_start, s.t_end) for s in X.phrase_spans(alignment.words, alignment.gaps)]
    with pytest.raises(EG.EgemapsUnavailable):
        EG.phrase_egemaps(samples, SR, spans, strict=True)


def test_egemaps_table_is_indexed_by_phrase_either_way() -> None:
    samples, alignment = synthetic()
    spans = [(s.t_start, s.t_end) for s in X.phrase_spans(alignment.words, alignment.gaps)]
    table = EG.phrase_egemaps(samples, SR, spans, file_id="synth01", channel_id="ch1")
    assert list(table.index) == [0, 1]
    assert table.index.name == "unit_idx"
    assert table.attrs["unit"] == "phrase"
    assert table.attrs["channel_sensitive"] is True
    assert table.attrs["egemaps_feature_set"] == EG.DEFAULT_FEATURE_SET
