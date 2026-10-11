#!/usr/bin/env python
"""Generate the injected half of the dataset (spec 5.4, 7.1-7.2).

For every clean recording, across all tiers::

    7 labels x L1-L4   = 28 single-flaw files
    2 composites at L2 + 2 at L4 = 4 composite files

and writes, per generated file::

    dataset/audio/injected/<file_id>.wav   16-bit PCM, 16 kHz mono
    dataset/labels/<file_id>.json          spec 7.1 label
    dataset/alignments/<file_id>.json      the clean alignment pushed through
                                           the time map (derived, not MFA)

plus, once per run::

    dataset/labels/events.tsv              sed_eval events for every label
    dataset/MANIFEST.csv                   one upserted row per generated file
    dataset/qc/rejections.csv              every file that could not be made

``file_id`` is ``"<clean_id>_<label>_<level>"`` or
``"<clean_id>_composite<k>_<level>"``, and each file's seed is derived from that
id, so a single file can be regenerated on its own and still come out
byte-identical (``CLAUDE.md`` invariant 13).

Inputs it needs, and what happens when they are missing:

* ``dataset/audio/clean/<file_id>.wav`` and
  ``dataset/alignments/<file_id>.TextGrid`` - run ``scripts/align_clean.py``
  first; without the TextGrid a file is skipped, not guessed at.
* **gap labels** (``boundary`` / ``minor`` / ``none``) come from the display
  transcript's punctuation via ``rostrum.text``, which is not written yet. MFA
  leaves every gap at ``"none"``, and four of the seven labels need a
  ``boundary`` gap to exist, so until ``rostrum.text`` lands this script reports
  those as rejections with the reason ``needs gap labels``. It does **not**
  invent sentence structure from silence durations.

Usage::

    python scripts/build_injected.py --text-id s1_gettysburg      # S1 first
    python scripts/build_injected.py                              # every text
    python scripts/build_injected.py --only s1_lv07 -v
    python scripts/build_injected.py --labels rushing --levels L3
    python scripts/build_injected.py --dry-run     # plan and count, write nothing
    python scripts/build_injected.py --events-only # rebuild events.tsv

Exit codes:
    0  every planned file was generated (or the plan was empty by request)
    1  at least one file was rejected
    2  nothing to generate: no clean WAV with an alignment
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow `python scripts/build_injected.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.align.mfa import read_textgrid
from rostrum.config import Config, load_config
from rostrum.data.manifest import ManifestRow, append_manifest, read_licenses, read_manifest
from rostrum.inject import engine, qc
from rostrum.inject.engine import CleanFile, InjectionResult, NoValidSpanError
from rostrum.inject.ops import InjectionError
from rostrum.io.audio import SAMPLE_RATE, load_any, write_wav16
from rostrum.io.hashing import sha256_file
from rostrum.types import Alignment

logger = logging.getLogger("build_injected")

LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4")
"""The four injected severity levels; ``L0`` is the clean file itself."""

REJECTION_COLUMNS = qc.REJECTION_COLUMNS
"""Columns of ``dataset/qc/rejections.csv``; owned by :mod:`rostrum.inject.qc`,
which appends to the same log when a generated file fails a realism gate."""


@dataclass
class Rejection:
    """One file that could not be generated, with why."""

    file_id: str
    source_file: str
    label: str
    level: str
    stage: str
    reason: str


# --------------------------------------------------------------------------- #
# Loading clean files
# --------------------------------------------------------------------------- #


def _speaker_ids(cfg: Config) -> dict[str, str]:
    """Map ``file_id`` to a speaker id from ``LICENSES.csv``.

    MFA does not know who is reading, and ``align_clean.py`` leaves
    ``speaker_id`` empty; the licence table is where the reader's name lives, so
    it is the only place the id can come from. Names are slugged, not copied, so
    the label JSON carries an identifier rather than personal data.
    """
    out: dict[str, str] = {}
    path = cfg.path("data.licenses")
    if not path.is_file():
        return out
    for row in read_licenses(path):
        speaker = (row.speaker or "").strip()
        if speaker:
            out[row.file_id] = "_".join(speaker.lower().split())
    return out


def load_clean_file(cfg: Config, row: ManifestRow, speaker_id: str = "") -> CleanFile:
    """Load one clean recording and its alignment into a :class:`CleanFile`.

    Args:
        cfg: Resolved config.
        row: The clean file's ``MANIFEST.csv`` row, which supplies ``text_id``
            and ``tier``.
        speaker_id: Speaker identifier from :func:`_speaker_ids`.

    Returns:
        The recording, ready for the engine.

    Raises:
        FileNotFoundError: The WAV or the TextGrid is missing.
        ValueError: The WAV is not at the working sample rate, so its timestamps
            would not line up with the alignment.
    """
    wav = cfg.path("data.audio_clean") / f"{row.file_id}.wav"
    grid = cfg.path("data.alignments") / f"{row.file_id}.TextGrid"
    if not wav.is_file():
        raise FileNotFoundError(f"no clean WAV: {wav}")
    if not grid.is_file():
        raise FileNotFoundError(f"no alignment: {grid} (run scripts/align_clean.py)")

    audio, sr = load_any(wav, allow_ffmpeg=False)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{wav.name} is {sr} Hz; the pipeline works at {SAMPLE_RATE} Hz")

    alignment = read_textgrid(
        grid,
        file_id=row.file_id,
        text_id=row.text_id or None,
        speaker_id=speaker_id,
        channel_id=row.file_id.rsplit("_", 1)[0] if row.tier == "bronze" else row.file_id,
        audio_sha256=sha256_file(wav),
        duration=audio.size / float(sr),
    )
    alignment.tier = row.tier or None  # type: ignore[assignment]
    return CleanFile(audio=np.asarray(audio, dtype=np.float64), sr=sr, alignment=alignment)


def cap_parents_per_text(rows: list[ManifestRow], max_per_text: int | None) -> list[ManifestRow]:
    """Keep at most ``max_per_text`` clean parents per text, spread across readers.

    The full 7-label x L1-L4 grid is generated for every parent that survives, so
    capping parents (rather than labels or levels) keeps the severity ladder and
    the label taxonomy complete while bounding the QC cost: every variant is
    re-aligned with MFA, which is roughly real-time, so an uncapped 62-parent
    build would need ~20 h of alignment.

    Selection is deterministic and reader-spread, not "first N": rows are sorted
    by ``file_id`` and sampled at an even stride, so a cap of 2 over readers
    01..16 takes 01 and 09 rather than 01 and 02.

    Args:
        rows: clean manifest rows, any order.
        max_per_text: cap per ``text_id``; ``None`` or <= 0 means no cap.

    Returns:
        The kept rows, sorted by ``(text_id, file_id)``.
    """
    if not max_per_text or max_per_text <= 0:
        return sorted(rows, key=lambda r: (r.text_id, r.file_id))
    by_text: dict[str, list[ManifestRow]] = {}
    for row in sorted(rows, key=lambda r: (r.text_id, r.file_id)):
        by_text.setdefault(row.text_id, []).append(row)
    kept: list[ManifestRow] = []
    for text_id, group in sorted(by_text.items()):
        n = min(max_per_text, len(group))
        stride = len(group) / n
        picked = [group[int(i * stride)] for i in range(n)]
        if len(group) > n:
            logger.info(
                "%s: capped %d clean parents to %d (%s)",
                text_id,
                len(group),
                n,
                ", ".join(r.file_id for r in picked),
            )
        kept.extend(picked)
    return sorted(kept, key=lambda r: (r.text_id, r.file_id))


def clean_rows(
    cfg: Config,
    *,
    text_id: str | None,
    only: set[str] | None,
    max_per_text: int | None = None,
) -> list[ManifestRow]:
    """Return the clean ``MANIFEST.csv`` rows to inject into, in file order."""
    rows = [r for r in read_manifest(cfg.path("data.manifest")) if r.origin == "clean"]
    if text_id:
        rows = [r for r in rows if r.text_id == text_id]
    if only:
        rows = [r for r in rows if r.file_id in only]
    if only:  # an explicit file list is never capped
        return sorted(rows, key=lambda r: (r.text_id, r.file_id))
    return cap_parents_per_text(rows, max_per_text)


# --------------------------------------------------------------------------- #
# Writing one result
# --------------------------------------------------------------------------- #


def write_result(cfg: Config, result: InjectionResult) -> Path:
    """Write one generated file's WAV, label JSON and derived alignment.

    The WAV is written first and hashed, and the hash goes into the alignment, so
    an alignment on disk always names the audio it describes.

    Returns:
        Path of the written WAV.
    """
    wav_path = cfg.ensure_dir("data.audio_injected") / f"{result.file_id}.wav"
    write_wav16(wav_path, result.audio, result.sr)

    alignment: Alignment = result.alignment
    alignment.audio_sha256 = sha256_file(wav_path)
    label = dict(result.label)
    label["audio_sha256"] = alignment.audio_sha256

    labels_dir = cfg.ensure_dir("data.labels")
    (labels_dir / f"{result.file_id}.json").write_text(
        json.dumps(label, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (cfg.ensure_dir("data.alignments") / f"{result.file_id}.json").write_text(
        json.dumps(alignment.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return wav_path


def manifest_row(result: InjectionResult, row: ManifestRow) -> ManifestRow:
    """Build the ``MANIFEST.csv`` row of a generated file.

    A composite has several labels, so its ``label`` column is ``"composite"``
    and the labels themselves live in the label JSON and in ``events.tsv``.
    """
    flaws = result.label["flaws"]
    label = flaws[0]["label"] if len(flaws) == 1 else "composite"
    return ManifestRow(
        file_id=result.file_id,
        text_id=row.text_id,
        tier=row.tier,
        origin="injected",
        label=label,
        level=str(flaws[0]["level"]),
        duration_s=f"{result.audio.size / float(result.sr):.3f}",
        split_fold=row.split_fold,
    )


def rebuild_events(cfg: Config) -> int:
    """Rewrite ``dataset/labels/events.tsv`` from every label JSON on disk.

    Rebuilt from the directory rather than from this run's results, so a partial
    run cannot leave the events file describing only part of the dataset.

    Returns:
        Number of event rows written.
    """
    labels_dir = cfg.path("data.labels")
    labels = []
    for path in sorted(labels_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("flaws"):
            labels.append(data)
    text = engine.format_events_tsv(labels)
    out = cfg.path("data.events_tsv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n_rows = sum(len(engine.events_rows(label)) for label in labels)
    logger.info("events.tsv: %d event(s) from %d label file(s)", n_rows, len(labels))
    return n_rows


def write_rejections(cfg: Config, rejections: list[Rejection]) -> None:
    """Append to ``dataset/qc/rejections.csv`` (spec 5.4 logs rejections)."""
    if not rejections:
        return
    path = cfg.path("data.qc_rejections")
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REJECTION_COLUMNS)
        if not exists:
            writer.writeheader()
        for rej in rejections:
            writer.writerow(
                {
                    "file_id": rej.file_id,
                    "source_file": rej.source_file,
                    "label": rej.label,
                    "level": rej.level,
                    "stage": rej.stage,
                    "reason": rej.reason,
                }
            )
    logger.warning("logged %d rejection(s) to %s", len(rejections), path)


# --------------------------------------------------------------------------- #
# Generating one clean file's worth
# --------------------------------------------------------------------------- #


def inject_one_file(
    cfg: Config,
    clean: CleanFile,
    row: ManifestRow,
    *,
    labels: tuple[str, ...],
    levels: tuple[str, ...],
    composites: bool,
    dry_run: bool,
) -> tuple[list[InjectionResult], list[Rejection], list[ManifestRow]]:
    """Generate every injected file derived from one clean recording.

    A failure on one (label, level) pair is recorded and the loop continues: a
    recording with no 300 ms boundary pause still yields its 24 other files.
    """
    results: list[InjectionResult] = []
    rejections: list[Rejection] = []
    manifest_rows: list[ManifestRow] = []

    def attempt(file_id: str, label: str, level: str, make) -> None:
        try:
            result = make()
        except (NoValidSpanError, InjectionError, ValueError) as exc:
            stage = "span" if isinstance(exc, NoValidSpanError) else "op"
            rejections.append(
                Rejection(
                    file_id, clean.file_id, label, level, stage, f"{type(exc).__name__}: {exc}"
                )
            )
            logger.warning("rejected %s: %s", file_id, exc)
            return
        results.append(result)
        manifest_rows.append(manifest_row(result, row))
        if not dry_run:
            write_result(cfg, result)

    for label in labels:
        for level in levels:
            attempt(
                f"{clean.file_id}_{label}_{level}",
                label,
                level,
                lambda label=label, level=level: engine.generate_single(
                    clean, label, level, cfg=cfg
                ),
            )

    if composites:
        comp = cfg.injection["composite"]
        for level in comp["levels"]:
            for k in range(1, int(comp["files_per_level"]) + 1):
                attempt(
                    f"{clean.file_id}_composite{k}_{level}",
                    "composite",
                    str(level),
                    lambda level=level, k=k: engine.generate_composite(
                        clean, str(level), k, cfg=cfg
                    ),
                )
    return results, rejections, manifest_rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_counts(
    made: Counter[tuple[str, str]],
    rejected: Counter[tuple[str, str]],
    levels: tuple[str, ...],
) -> None:
    """Print the generated-file count per label and level, rejections in brackets."""
    labels = [*engine.INJECTABLE_LABELS, "composite"]
    width = max(len(label) for label in labels) + 1
    header = "label".ljust(width) + "".join(level.rjust(10) for level in levels) + "     total"
    print("\n" + header)
    print("-" * len(header))
    for label in labels:
        cells = []
        for level in levels:
            ok = made[(label, level)]
            bad = rejected[(label, level)]
            cells.append((f"{ok}" + (f" ({bad})" if bad else "")).rjust(10))
        total = sum(made[(label, level)] for level in levels)
        print(label.ljust(width) + "".join(cells) + f"{total:>10}")
    print("-" * len(header))
    totals = [str(sum(made[(label, level)] for label in labels)).rjust(10) for level in levels]
    print("total".ljust(width) + "".join(totals) + f"{sum(made.values()):>10}")
    if sum(rejected.values()):
        print(f"\n({sum(rejected.values())} rejected; see dataset/qc/rejections.csv)")


def gap_label_warning(clean: CleanFile) -> str | None:
    """Return a warning when an alignment has no ``boundary`` gap.

    Four labels and every phrase-scoped label need one, so a whole-file skip is
    worth saying out loud rather than letting 28 identical rejections explain it.
    """
    if any(gap.label == "boundary" for gap in clean.alignment.gaps):
        return None
    return (
        f"{clean.file_id}: no gap is labelled 'boundary', so the phrase structure is "
        "a single phrase - missing_pause, low_energy, monotone and misplaced_stress "
        "have no valid span. Gap labels come from rostrum.text (spec 7.1); run it "
        "over the alignments first."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--text-id", help="only clean files of this text (e.g. s1_gettysburg)")
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="only these clean file ids")
    p.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        default=list(engine.INJECTABLE_LABELS),
        choices=list(engine.INJECTABLE_LABELS),
        help="labels to inject (default: all seven)",
    )
    p.add_argument(
        "--levels",
        nargs="+",
        metavar="LEVEL",
        default=list(LEVELS),
        choices=list(LEVELS),
        help="severity levels (default: L1 L2 L3 L4)",
    )
    p.add_argument("--no-composites", action="store_true", help="skip the composite files")
    p.add_argument("--dry-run", action="store_true", help="plan and count, writing no files at all")
    p.add_argument(
        "--events-only", action="store_true", help="only rebuild labels/events.tsv and exit"
    )
    p.add_argument(
        "--max-parents-per-text",
        type=int,
        metavar="N",
        help="cap clean parents per text (default: injection.plan.max_parents_per_text; "
        "0 means no cap). Ignored when --only is given.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the injected dataset. See the module docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(apply_seeds=False)

    if args.events_only:
        rebuild_events(cfg)
        return 0

    plan_cfg = cfg.injection.get("plan") or {}
    max_per_text = args.max_parents_per_text
    if max_per_text is None:
        max_per_text = plan_cfg.get("max_parents_per_text")
    rows = clean_rows(
        cfg,
        text_id=args.text_id,
        only=set(args.only) if args.only else None,
        max_per_text=int(max_per_text) if max_per_text else None,
    )
    if not rows:
        logger.error(
            "nothing to generate: no clean rows in %s%s",
            cfg.path("data.manifest"),
            f" for text_id={args.text_id}" if args.text_id else "",
        )
        return 2

    speakers = _speaker_ids(cfg)
    labels = tuple(args.labels)
    levels = tuple(args.levels)
    made: Counter[tuple[str, str]] = Counter()
    rejected: Counter[tuple[str, str]] = Counter()
    all_rejections: list[Rejection] = []
    all_manifest: list[ManifestRow] = []
    loaded = 0

    for row in rows:
        try:
            clean = load_clean_file(cfg, row, speakers.get(row.file_id, ""))
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("skipping %s: %s", row.file_id, exc)
            all_rejections.append(Rejection(row.file_id, row.file_id, "", "", "load", str(exc)))
            continue
        loaded += 1
        warning = gap_label_warning(clean)
        if warning:
            logger.warning("%s", warning)

        results, rejections, manifest_rows = inject_one_file(
            cfg,
            clean,
            row,
            labels=labels,
            levels=levels,
            composites=not args.no_composites,
            dry_run=args.dry_run,
        )
        for result in results:
            flaws = result.label["flaws"]
            key = (flaws[0]["label"] if len(flaws) == 1 else "composite", str(flaws[0]["level"]))
            made[key] += 1
        for rej in rejections:
            rejected[(rej.label, rej.level)] += 1
        all_rejections.extend(rejections)
        all_manifest.extend(manifest_rows)
        logger.info("%s: %d generated, %d rejected", row.file_id, len(results), len(rejections))

    if not loaded:
        logger.error(
            "no clean file could be loaded; run scripts/align_clean.py so that "
            "dataset/alignments/<file_id>.TextGrid exists"
        )
        write_rejections(cfg, all_rejections)
        return 2

    if not args.dry_run:
        if all_manifest:
            append_manifest(cfg.path("data.manifest"), all_manifest)
        rebuild_events(cfg)
    write_rejections(cfg, all_rejections)

    print_counts(made, rejected, levels)
    if args.dry_run:
        print("\n(dry run: no files were written)")
    return 1 if all_rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())
