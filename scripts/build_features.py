#!/usr/bin/env python
"""Build the feature cache for the dataset (spec 9 acceptance, `make features`).

Walks the manifest, extracts every feature of spec 9.1 for each recording that
has a forced alignment, writes one Parquet cache entry per recording, and prints
the cost **per minute of audio** - the acceptance criterion from the build
prompt - both overall and broken down by stage.

A recording without an alignment is skipped and counted, not guessed at: every
word-level feature is defined on MFA intervals, so there is nothing to compute
until `make align` has run (MFA lives in its own conda environment).

Speaker and channel identity are not columns of ``MANIFEST.csv``, so they are
derived here and recorded in each cache entry:

* a **clean** recording is its own speaker and its own channel;
* an **injected** or **human** recording inherits both from the clean recording
  whose ``file_id`` prefixes its own - which is exactly what the channel rule
  needs, since an injected file shares a microphone with its parent and nothing
  else (CLAUDE.md invariant 7).

Usage::

    python scripts/build_features.py                  # whole dataset
    python scripts/build_features.py --sample         # dataset_sample/ only
    python scripts/build_features.py --file-id s1_lv01 --refresh
    python scripts/build_features.py --crosscheck     # with the CREPE check
    python scripts/build_features.py --lean           # skip MFCC/VQ/eGeMAPS

Exit codes:
    0  every eligible recording has a cache entry
    1  at least one recording failed
    2  nothing to do (no audio, or no alignments at all)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.align.mfa import read_textgrid
from rostrum.config import Config, load_config
from rostrum.data.manifest import clean_file_ids, parent_file_id, read_manifest
from rostrum.features import extract_all as X
from rostrum.features import pauses as PA
from rostrum.io.audio import load_any
from rostrum.io.hashing import sha256_file

logger = logging.getLogger("build_features")

INDEX_FILE = "index.json"
"""Written inside the feature cache directory: what was built, and how fast."""


@dataclass
class Target:
    """One recording to extract features for.

    Attributes:
        file_id: Recording identifier.
        audio: Path to the processed WAV.
        alignment: Path to the MFA TextGrid, which may not exist yet.
        text_id: Library text this is a rendition of.
        tier: ``gold`` | ``silver`` | ``bronze``.
        origin: ``clean`` | ``injected`` | ``human``.
        speaker_id: Derived speaker identity.
        channel_id: Derived channel identity (invariant 7).
    """

    file_id: str
    audio: Path
    alignment: Path
    text_id: str = ""
    tier: str = ""
    origin: str = ""
    speaker_id: str = ""
    channel_id: str = ""


# --------------------------------------------------------------------------- #
# Finding the work
# --------------------------------------------------------------------------- #


def _audio_dirs(cfg: Config, *, sample: bool) -> list[Path]:
    """Directories holding processed WAVs, in build order."""
    if sample:
        return [cfg.path("sample.audio")]
    return [
        cfg.path("data.audio_clean"),
        cfg.path("data.audio_injected"),
        cfg.path("data.audio_human_flawed"),
    ]


def collect_targets(cfg: Config, *, sample: bool, only: list[str] | None = None) -> list[Target]:
    """List the recordings to process, newest manifest information attached."""
    manifest_path = cfg.path("sample.dir") / "MANIFEST.csv" if sample else cfg.path("data.manifest")
    rows = {r.file_id: r for r in read_manifest(manifest_path)} if manifest_path.is_file() else {}
    clean_ids = clean_file_ids(rows.values())
    alignment_dir = cfg.path("sample.alignments") if sample else cfg.path("data.alignments")

    targets: list[Target] = []
    seen: set[str] = set()
    for directory in _audio_dirs(cfg, sample=sample):
        for wav in sorted(directory.glob("*.wav")):
            file_id = wav.stem
            if file_id in seen or (only and file_id not in only):
                continue
            seen.add(file_id)
            row = rows.get(file_id)
            origin = row.origin if row else ("clean" if directory.name == "clean" else "")
            parent = (
                file_id if origin == "clean" else (parent_file_id(file_id, clean_ids) or file_id)
            )
            targets.append(
                Target(
                    file_id=file_id,
                    audio=wav,
                    alignment=alignment_dir / f"{file_id}.TextGrid",
                    text_id=row.text_id if row else "",
                    tier=row.tier if row else "",
                    origin=origin,
                    speaker_id=parent,
                    channel_id=parent,
                )
            )
    return targets


def _gap_labels(cfg: Config, text_id: str) -> list[str] | None:
    """Per-gap labels from ``dataset/texts/<text_id>.json``, if they exist yet.

    The expectation that a gap *should* be there comes from the transcript's
    punctuation, never from the audio. Until ``rostrum.text.gaps`` has filled
    ``gap_labels`` for a text, every gap stays ``none`` and the recording is one
    phrase - which only costs the phrase-level features their resolution.
    """
    path = cfg.path("data.texts") / f"{text_id}.json"
    if not text_id or not path.is_file():
        return None
    try:
        labels = json.loads(path.read_text(encoding="utf-8")).get("gap_labels")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read gap labels for %s: %s", text_id, exc)
        return None
    return list(labels) if labels else None


# --------------------------------------------------------------------------- #
# Doing the work
# --------------------------------------------------------------------------- #


def build_one(
    target: Target,
    cfg: Config,
    *,
    crosscheck: bool,
    lean: bool,
    refresh: bool,
) -> tuple[X.FeatureBundle, bool]:
    """Extract and cache the features of one recording.

    Returns:
        ``(bundle, from_cache)``.
    """
    samples, sr = load_any(target.audio)
    digest = sha256_file(target.audio)
    alignment = read_textgrid(
        target.alignment,
        file_id=target.file_id,
        text_id=target.text_id or None,
        speaker_id=target.speaker_id,
        channel_id=target.channel_id,
        audio_sha256=digest,
        duration=len(samples) / float(sr),
    )
    alignment.tier = target.tier or None

    labels = _gap_labels(cfg, target.text_id)
    if labels:
        mapping = {
            i: (label, "")
            for i, label in enumerate(labels)
            if i < len(alignment.gaps) and label in PA.GAP_LABELS
        }
        alignment.gaps = PA.apply_gap_labels(alignment.gaps, mapping)

    options = (
        {}
        if not lean
        else {
            "with_mfcc": False,
            "with_voice_quality": False,
            "with_egemaps": False,
        }
    )
    return X.extract_all_cached(
        samples,
        sr,
        alignment,
        config=cfg,
        audio_sha256=digest,
        refresh=refresh,
        crosscheck=crosscheck,
        **options,
    )


def _report(
    built: list[X.FeatureBundle],
    *,
    cached: int,
    skipped: list[tuple[str, str]],
    failed: list[tuple[str, str]],
    wall_s: float,
) -> dict[str, object]:
    """Print the timing table and return it for ``index.json``."""
    audio_s = sum(float(b.meta.get("audio_duration_s", 0.0)) for b in built)
    compute_s = sum(float(b.meta.get("elapsed_s", 0.0)) for b in built)
    minutes = audio_s / 60.0

    stages: dict[str, float] = defaultdict(float)
    for bundle in built:
        for stage, seconds in (bundle.meta.get("stage_timings_s") or {}).items():
            stages[stage] += float(seconds)

    print()
    print(
        f"built {len(built)} entries, {cached} already cached, {len(skipped)} skipped, "
        f"{len(failed)} failed"
    )
    if built:
        print(f"audio processed:  {audio_s:8.1f} s ({minutes:.2f} min)")
        print(f"extraction time:  {compute_s:8.1f} s")
        print(f"wall clock:       {wall_s:8.1f} s")
        print(
            f"cost:             {compute_s / minutes:8.1f} s per minute of audio"
            if minutes > 0
            else "cost:                   n/a"
        )
        print()
        print(f"{'stage':<22}{'total s':>10}{'s / min audio':>16}{'share':>8}")
        for stage, seconds in sorted(stages.items(), key=lambda kv: -kv[1]):
            per_minute = seconds / minutes if minutes > 0 else float("nan")
            share = seconds / compute_s * 100.0 if compute_s > 0 else float("nan")
            print(f"{stage:<22}{seconds:>10.2f}{per_minute:>16.2f}{share:>7.1f}%")
    for file_id, reason in skipped:
        print(f"  skipped {file_id}: {reason}")
    for file_id, reason in failed:
        print(f"  FAILED  {file_id}: {reason}")

    return {
        "n_built": len(built),
        "n_cached": cached,
        "n_skipped": len(skipped),
        "n_failed": len(failed),
        "audio_s": round(audio_s, 3),
        "extraction_s": round(compute_s, 3),
        "wall_s": round(wall_s, 3),
        "s_per_minute_of_audio": round(compute_s / minutes, 3) if minutes > 0 else None,
        "stage_s_per_minute_of_audio": (
            {stage: round(seconds / minutes, 3) for stage, seconds in sorted(stages.items())}
            if minutes > 0
            else {}
        ),
        "entries": {
            b.file_id: {
                "cache_key": b.meta.get("cache_key"),
                "audio_sha256": b.meta.get("audio_sha256"),
                "n_words": b.meta.get("n_words"),
                "n_phrases": b.meta.get("n_phrases"),
                "audio_duration_s": b.meta.get("audio_duration_s"),
                "elapsed_s": b.meta.get("elapsed_s"),
                "egemaps_available": b.meta.get("egemaps_available"),
                "pitch_agreement_rate": b.meta.get("pitch_agreement_rate"),
            }
            for b in built
        },
        "skipped": dict(skipped),
        "failed": dict(failed),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``make features``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="use dataset_sample/ instead")
    parser.add_argument("--file-id", action="append", dest="file_ids", help="restrict to these ids")
    parser.add_argument("--refresh", action="store_true", help="recompute existing entries")
    parser.add_argument(
        "--crosscheck",
        action="store_true",
        help="run the torchcrepe pitch cross-check (slow on CPU, needed for the report)",
    )
    parser.add_argument("--lean", action="store_true", help="skip MFCC, voice quality and eGeMAPS")
    parser.add_argument("--cache-dir", type=Path, help="override the configured cache directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    cache_dir = args.cache_dir or cfg.path("cache.features")
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets = collect_targets(cfg, sample=args.sample, only=args.file_ids)
    if not targets:
        logger.error("no audio found; run `make data` first")
        return 2

    built: list[X.FeatureBundle] = []
    cached = 0
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    started = time.perf_counter()

    for target in targets:
        if not target.alignment.is_file():
            skipped.append((target.file_id, f"no alignment at {target.alignment}"))
            continue
        try:
            bundle, from_cache = build_one(
                target,
                cfg,
                crosscheck=args.crosscheck,
                lean=args.lean,
                refresh=args.refresh,
            )
        except Exception as exc:  # one bad file must not stop the build
            logger.exception("feature extraction failed for %s", target.file_id)
            failed.append((target.file_id, f"{type(exc).__name__}: {exc}"))
            continue
        if from_cache:
            cached += 1
            logger.info("%s: already cached", target.file_id)
        else:
            built.append(bundle)
            logger.info("%s", bundle.summary())

    index = _report(
        built,
        cached=cached,
        skipped=skipped,
        failed=failed,
        wall_s=time.perf_counter() - started,
    )
    (cache_dir / INDEX_FILE).write_text(
        json.dumps(index, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(f"\nindex written to {cache_dir / INDEX_FILE}")

    if failed:
        return 1
    if not built and not cached:
        logger.error(
            "no recording had an alignment; run `make align` (MFA lives in the `mfa` conda env)"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
