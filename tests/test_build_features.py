"""Tests for ``scripts/build_features.py`` (spec 9 acceptance, `make features`).

MFA is not available in this checkout, so the end-to-end test builds its own
tiny dataset tree - a synthetic WAV, a hand-written TextGrid and a manifest -
and runs the script over it. That exercises the whole path the real build takes:
manifest lookup, speaker and channel derivation, alignment reading, extraction,
Parquet caching and the timing report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_features as bf
from rostrum.align.mfa import TextGridTier, write_textgrid
from rostrum.config import load_config
from rostrum.io.audio import write_wav16

SR = 16000
WORDS = ["now", "is", "the", "time"]
WORD_S = 0.30
GAP_S = 0.08
LEAD_S = 0.20


def _syllable(f_hz: float, seconds: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    out = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
        out += weight * np.sin(2.0 * np.pi * f_hz * harmonic * t)
    envelope = np.minimum(1.0, np.minimum(t, seconds - t) / 0.03)
    return amplitude * out * envelope / 1.9


def _utterance() -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Synthetic speech plus the word intervals that describe it exactly."""
    parts = [np.zeros(int(LEAD_S * SR))]
    intervals: list[tuple[float, float, str]] = []
    t = LEAD_S
    for text in WORDS:
        parts.append(_syllable(140.0, WORD_S, 0.3))
        intervals.append((round(t, 3), round(t + WORD_S, 3), text))
        t += WORD_S
        parts.append(np.zeros(int(GAP_S * SR)))
        t += GAP_S
    return np.concatenate(parts).astype(np.float32), intervals


def _write_dataset(root: Path, file_ids: list[str]) -> float:
    """Write audio, alignments and a manifest under ``root``; return duration."""
    audio_dir = root / "audio" / "clean"
    alignment_dir = root / "alignments"
    samples, intervals = _utterance()
    duration = len(samples) / SR

    # Two phones per word, splitting each word interval in half.
    phone_intervals: list[tuple[float, float, str]] = []
    for t0, t1, _ in intervals:
        mid = round((t0 + t1) / 2.0, 3)
        phone_intervals += [(t0, mid, "N"), (mid, t1, "AW1")]

    for file_id in file_ids:
        write_wav16(audio_dir / f"{file_id}.wav", samples, SR)
        write_textgrid(
            alignment_dir / f"{file_id}.TextGrid",
            [
                TextGridTier("words", 0.0, duration, tuple(intervals)),
                TextGridTier("phones", 0.0, duration, tuple(phone_intervals)),
            ],
            xmin=0.0,
            xmax=duration,
        )

    lines = ["file_id,text_id,tier,origin,label,level,duration_s,split_fold"]
    for file_id in file_ids:
        if file_id.endswith("_rushing_L2"):
            lines.append(f"{file_id},t1,silver,injected,rushing,L2,{duration:.3f},loto_t1")
        else:
            lines.append(f"{file_id},t1,silver,clean,,,{duration:.3f},loto_t1")
    (root / "MANIFEST.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return duration


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A tiny dataset tree with the config pointed at it."""
    root = tmp_path / "dataset"
    duration = _write_dataset(root, ["lv01"])
    cfg = load_config(
        overrides={
            "paths": {
                "data": {
                    "dir": str(root),
                    "audio_clean": str(root / "audio" / "clean"),
                    "audio_injected": str(root / "audio" / "injected"),
                    "audio_human_flawed": str(root / "audio" / "human_flawed"),
                    "alignments": str(root / "alignments"),
                    "manifest": str(root / "MANIFEST.csv"),
                    "texts": str(root / "texts"),
                },
                "cache": {"features": str(tmp_path / "cache")},
            }
        }
    )
    monkeypatch.setattr(bf, "load_config", lambda *a, **k: cfg)
    return cfg, root, duration


# --------------------------------------------------------------------------- #
# Speaker and channel derivation
# --------------------------------------------------------------------------- #


def test_the_longest_matching_clean_prefix_wins() -> None:
    clean = ["s1_lv01", "s1_lv0", "s1_lv02"]
    assert bf.parent_file_id("s1_lv01_rushing_L3", clean) == "s1_lv01"
    assert bf.parent_file_id("s1_lv02_monotone_L1", clean) == "s1_lv02"


def test_an_unrelated_file_has_no_parent() -> None:
    assert bf.parent_file_id("team_a_take3", ["s1_lv01"]) == ""


def test_a_clean_recording_is_its_own_speaker_and_channel(dataset) -> None:
    cfg, _, _ = dataset
    targets = bf.collect_targets(cfg, sample=False)
    assert [t.file_id for t in targets] == ["lv01"]
    target = targets[0]
    assert target.origin == "clean"
    assert target.speaker_id == "lv01"
    assert target.channel_id == "lv01"
    assert target.text_id == "t1"
    assert target.tier == "silver"


def test_an_injected_file_inherits_its_parents_channel(tmp_path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, ["lv01", "lv01_rushing_L2"])
    cfg = load_config(
        overrides={
            "paths": {
                "data": {
                    "audio_clean": str(root / "audio" / "clean"),
                    "audio_injected": str(root / "audio" / "injected"),
                    "audio_human_flawed": str(root / "audio" / "human_flawed"),
                    "alignments": str(root / "alignments"),
                    "manifest": str(root / "MANIFEST.csv"),
                }
            }
        }
    )
    derived = {t.file_id: t for t in bf.collect_targets(cfg, sample=False)}["lv01_rushing_L2"]
    assert derived.origin == "injected"
    assert derived.channel_id == "lv01"
    assert derived.speaker_id == "lv01"


def test_only_filters_the_target_list(dataset) -> None:
    cfg, _, _ = dataset
    assert bf.collect_targets(cfg, sample=False, only=["nothing"]) == []


# --------------------------------------------------------------------------- #
# The build
# --------------------------------------------------------------------------- #


def test_build_one_extracts_and_caches(dataset) -> None:
    cfg, _, duration = dataset
    target = bf.collect_targets(cfg, sample=False)[0]
    bundle, from_cache = bf.build_one(target, cfg, crosscheck=False, lean=True, refresh=False)
    assert from_cache is False
    assert len(bundle.words) == len(WORDS)
    assert bundle.meta["channel_id"] == "lv01"
    assert bundle.meta["audio_duration_s"] == pytest.approx(duration, abs=0.01)

    again, from_cache = bf.build_one(target, cfg, crosscheck=False, lean=True, refresh=False)
    assert from_cache is True
    assert again.words.equals(bundle.words)


def test_main_reports_timing_per_minute_of_audio(dataset, capsys) -> None:
    cfg, _, _ = dataset
    assert bf.main(["--lean"]) == 0
    out = capsys.readouterr().out
    assert "s per minute of audio" in out
    assert "built 1 entries" in out

    index = json.loads((cfg.path("cache.features") / bf.INDEX_FILE).read_text(encoding="utf-8"))
    assert index["n_built"] == 1
    assert index["n_failed"] == 0
    assert index["s_per_minute_of_audio"] > 0
    assert "pitch" in index["stage_s_per_minute_of_audio"]
    assert index["entries"]["lv01"]["n_words"] == len(WORDS)
    assert index["entries"]["lv01"]["cache_key"]


def test_a_second_run_reads_the_cache(dataset, capsys) -> None:
    assert bf.main(["--lean"]) == 0
    capsys.readouterr()
    assert bf.main(["--lean"]) == 0
    assert "already cached" in capsys.readouterr().out


def test_a_recording_without_an_alignment_is_skipped_not_guessed(dataset, capsys) -> None:
    _, root, _ = dataset
    (root / "alignments" / "lv01.TextGrid").unlink()
    assert bf.main(["--lean"]) == 2
    assert "no alignment" in capsys.readouterr().out


def test_no_audio_at_all_exits_two(tmp_path, monkeypatch) -> None:
    cfg = load_config(
        overrides={
            "paths": {
                "data": {
                    "audio_clean": str(tmp_path / "none"),
                    "audio_injected": str(tmp_path / "none"),
                    "audio_human_flawed": str(tmp_path / "none"),
                    "alignments": str(tmp_path / "none"),
                    "manifest": str(tmp_path / "none" / "MANIFEST.csv"),
                },
                "cache": {"features": str(tmp_path / "cache")},
            }
        }
    )
    monkeypatch.setattr(bf, "load_config", lambda *a, **k: cfg)
    assert bf.main([]) == 2


def test_gap_labels_from_the_text_json_create_phrases(dataset) -> None:
    cfg, root, _ = dataset
    texts = root / "texts"
    texts.mkdir(parents=True, exist_ok=True)
    (texts / "t1.json").write_text(
        json.dumps({"text_id": "t1", "gap_labels": ["none", "boundary", "none"]}),
        encoding="utf-8",
    )
    target = bf.collect_targets(cfg, sample=False)[0]
    bundle, _ = bf.build_one(target, cfg, crosscheck=False, lean=True, refresh=True)
    assert len(bundle.phrases) == 2


def test_missing_gap_labels_leave_one_phrase(dataset) -> None:
    cfg, _, _ = dataset
    assert bf._gap_labels(cfg, "t1") is None
    target = bf.collect_targets(cfg, sample=False)[0]
    bundle, _ = bf.build_one(target, cfg, crosscheck=False, lean=True, refresh=True)
    assert len(bundle.phrases) == 1
