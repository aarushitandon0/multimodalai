"""Tests for the ASR-vs-transcript mismatch check (spec 8, 13)."""

from __future__ import annotations

import json

import pytest

from rostrum.align.asr import (
    ASRError,
    ASRNotAvailableError,
    ASRSettings,
    max_wer,
    normalise_tokens,
    transcribe,
    wer,
)
from rostrum.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_the_comparison_ignores_case_and_punctuation():
    assert normalise_tokens("Four score, and seven!") == ["four", "score", "and", "seven"]


def test_the_comparison_spells_numerals_out():
    """ "1863" against "eighteen sixty three" is not three errors."""
    assert normalise_tokens("in 1863") == normalise_tokens("in eighteen sixty three")


# --------------------------------------------------------------------------- #
# Word error rate
# --------------------------------------------------------------------------- #


def test_an_exact_match_scores_zero():
    assert wer("four score and seven", "Four score and 7.").wer == 0.0


def test_one_substitution_in_four_words():
    errors = wer("the quick brown fox", "the quick green fox")
    assert errors.wer == pytest.approx(0.25)
    assert (errors.substitutions, errors.deletions, errors.insertions) == (1, 0, 0)


def test_a_deletion_and_an_insertion_are_counted_separately():
    assert wer("a b c", "a c").deletions == 1
    assert wer("a b c", "a b x c").insertions == 1


def test_the_rate_is_unbounded_above():
    """A transcript far shorter than the speech gives a WER above one."""
    errors = wer("a b", "a b c d e f")
    assert errors.wer > 1.0
    assert errors.insertions == 4
    assert errors.n_reference == 2


def test_an_empty_transcript_does_not_divide_by_zero():
    assert wer("", "").wer == 0.0
    assert wer("", "something").wer == 1.0


def test_the_counts_serialise(cfg):
    payload = wer("the quick brown fox", "the quick fox").to_dict()
    assert payload["n_reference"] == 4
    assert json.dumps(payload)


def test_the_reject_threshold_comes_from_the_config(cfg):
    assert max_wer(cfg) == pytest.approx(0.25)
    assert max_wer(None) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Settings and the model
# --------------------------------------------------------------------------- #


def test_the_settings_come_from_the_config(cfg):
    settings = ASRSettings.from_config(cfg)
    assert settings.enabled
    assert settings.device == "cpu"
    assert settings.local_files_only, "no network at inference (CLAUDE.md invariant 13)"
    assert settings.beam_size > 0


def test_the_cache_key_changes_with_the_model():
    assert ASRSettings(model="tiny.en").key != ASRSettings(model="base.en").key


def test_a_missing_file_is_refused(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        transcribe(tmp_path / "nope.wav", config=cfg)


def test_an_unusable_model_is_an_error_rather_than_a_download(cfg, tmp_path):
    """``local_files_only`` must fail loudly, never fetch weights mid-analysis."""
    audio = cfg.path("sample.audio") / "s1_lv01.wav"
    if not audio.is_file():  # pragma: no cover - the sample is committed
        pytest.skip("dataset_sample audio is not present")
    settings = ASRSettings(model="no-such-model-xyz", local_files_only=True)
    with pytest.raises(ASRError):
        transcribe(audio, settings=settings, cache_dir=tmp_path)


def test_a_transcription_is_cached_by_audio_hash(cfg, tmp_path):
    audio = cfg.path("sample.audio") / "s1_lv01.wav"
    if not audio.is_file():  # pragma: no cover - the sample is committed
        pytest.skip("dataset_sample audio is not present")
    try:
        first = transcribe(audio, config=cfg, cache_dir=tmp_path)
    except ASRNotAvailableError as exc:  # pragma: no cover - no weights here
        pytest.skip(f"the recogniser cannot run here: {exc}")
    assert not first.from_cache
    assert first.text
    second = transcribe(audio, config=cfg, cache_dir=tmp_path)
    assert second.from_cache
    assert second.text == first.text


def test_the_recogniser_hears_the_text_it_was_given(cfg, tmp_path):
    """A sanity check on the real sample: the WER against its own transcript is low."""
    audio = cfg.path("sample.audio") / "s1_lv01.wav"
    transcript_path = cfg.path("sample.transcripts") / "s1_gettysburg.json"
    if not (audio.is_file() and transcript_path.is_file()):  # pragma: no cover
        pytest.skip("dataset_sample is not present")
    try:
        heard = transcribe(audio, config=cfg, cache_dir=tmp_path)
    except ASRNotAvailableError as exc:  # pragma: no cover - no weights here
        pytest.skip(f"the recogniser cannot run here: {exc}")
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))["display"]
    errors = wer(transcript, heard.text)
    # The LibriVox file carries a spoken intro and outro that are not in the
    # text (see the transcript's own notes), so this is not a tight bound - it
    # is a check that the recogniser heard the Gettysburg Address at all.
    assert errors.wer < 0.5
