#!/usr/bin/env python
"""Preprocess every declared source recording into a clean dataset WAV (spec 4.6).

For each file in ``dataset/SOURCES.yaml`` that has already been downloaded into
``dataset/audio/raw/<source_id>/``, this script runs the canonical chain -
decode, mono 16 kHz, excerpt trim, 300 ms lead/tail silence, -23 LUFS with a
-1 dBFS peak ceiling - and writes::

    dataset/audio/clean/<file_id>.wav      16-bit PCM, the file the dataset ships
    dataset/MANIFEST.csv                   tier, origin=clean, duration_s
    dataset/LICENSES.csv                   licence row, sha256 of that WAV
    dataset/qc/clean_build.csv             per-file QC: LUFS, peak, both digests
    dataset/checksums.sha256               raw + clean digests

The licence gate from ``rostrum.data.sources`` runs *before* anything is read,
so a source whose licence is still ``TODO(human)`` or names NoDerivatives never
reaches the dataset (spec 4.1).

Team recordings need no download: drop them under
``dataset/audio/raw/team/<file_id>.<ext>``, declare them in ``SOURCES.yaml``,
and they are picked up like any other source.

Usage::

    python scripts/build_clean.py
    python scripts/build_clean.py --only S1 S1b          # selected sources
    python scripts/build_clean.py --file s3_fdr_gold     # selected files
    python scripts/build_clean.py --dry-run              # report, write nothing
    python scripts/build_clean.py --force                # rebuild existing WAVs

Exit codes:
    0  every ready file is built (or was already built)
    1  at least one file failed, or a non-optional source was refused
    2  nothing was built: no raw audio yet, or every field is still TODO(human)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/build_clean.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.config import load_config
from rostrum.data.manifest import (
    LicenseRow,
    ManifestRow,
    append_licenses,
    append_manifest,
    write_checksums,
)
from rostrum.data.sources import (
    TODO_SENTINEL,
    Source,
    SourceFile,
    SourcesError,
    licence_verdict,
    load_sources,
)
from rostrum.io.audio import AudioError, AudioStats, describe, preprocess
from rostrum.io.hashing import sha256_file

logger = logging.getLogger("build_clean")

QC_FIELDS: tuple[str, ...] = (
    "file_id",
    "source_id",
    "text_id",
    "tier",
    "raw_file",
    "raw_sha256",
    "wav_sha256",
    "sample_rate",
    "duration_s",
    "lufs",
    "peak_dbfs",
    "excerpt_start_s",
    "excerpt_end_s",
    "status",
)
"""Column order of ``dataset/qc/clean_build.csv``."""

LUFS_TOLERANCE_DB = 1.0
"""How far the measured loudness may sit from the target before we warn."""

PEAK_TOLERANCE_DB = 0.05
"""Slack on the peak ceiling, covering 16-bit quantisation of the last sample."""

IGNORED_RAW_SUFFIXES = frozenset({".part", ".tmp", ".md", ".gitkeep", ".txt"})
"""Sidecars that may sit beside a raw download without being audio."""


@dataclass
class Outcome:
    """What happened to one declared file."""

    file_id: str
    source_id: str
    status: str  # built | cached | missing_raw | todo | refused | failed
    detail: str = ""
    raw_path: Path | None = None
    wav_path: Path | None = None
    raw_sha256: str = ""
    wav_sha256: str = ""
    stats: AudioStats | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def find_raw(raw_root: Path, source_dirname: str, file_id: str) -> Path | None:
    """Return the downloaded (or hand-placed) audio for ``file_id``, if present.

    The extension is whatever the source served, so the lookup is a glob. A
    sidecar (``.part``, ``.md``, ...) beside the audio is ignored.

    Args:
        raw_root: ``dataset/audio/raw``.
        source_dirname: Subdirectory name, e.g. ``S1`` or ``team`` (a source's
            :attr:`~rostrum.data.sources.Source.raw_dirname`).
        file_id: The file id declared in ``SOURCES.yaml``.

    Returns:
        The audio path, or ``None`` if nothing is there yet.
    """
    candidates = [
        p
        for p in sorted((raw_root / source_dirname).glob(f"{file_id}.*"))
        if p.is_file() and p.suffix.lower() not in IGNORED_RAW_SUFFIXES
    ]
    return candidates[0] if candidates else None


def excerpt_bounds(sf: SourceFile) -> tuple[float | None, float | None, str]:
    """Return ``(start, end, problem)`` for one file's excerpt window.

    ``problem`` is non-empty when a bound is still ``TODO(human)``: trimming to
    a window nobody has chosen would silently put the wrong audio in the
    dataset, so the file is skipped instead.
    """
    pending = [
        name
        for name, value in (
            ("excerpt_start_s", sf.excerpt_start_s),
            ("excerpt_end_s", sf.excerpt_end_s),
        )
        if value == TODO_SENTINEL
    ]
    if pending:
        return None, None, f"{' and '.join(pending)} is still {TODO_SENTINEL}"
    start = sf.excerpt_start_s if isinstance(sf.excerpt_start_s, (int, float)) else None
    end = sf.excerpt_end_s if isinstance(sf.excerpt_end_s, (int, float)) else None
    return start, end, ""


def check_invariants(stats: AudioStats, *, target_lufs: float, peak_ceiling: float) -> list[str]:
    """Return the ``CLAUDE.md`` invariant-1 violations of a processed WAV."""
    problems: list[str] = []
    if stats.sample_rate != 16_000:
        problems.append(f"sample rate is {stats.sample_rate}, expected 16000")
    if stats.peak_dbfs > peak_ceiling + PEAK_TOLERANCE_DB:
        problems.append(f"peak is {stats.peak_dbfs:.2f} dBFS, above the {peak_ceiling} ceiling")
    if stats.lufs is None:
        problems.append("integrated loudness could not be measured (clip under 400 ms)")
    return problems


def build_one(
    source: Source,
    sf: SourceFile,
    raw_root: Path,
    clean_root: Path,
    *,
    audio_cfg: dict,
    dry_run: bool,
    force: bool,
) -> Outcome:
    """Preprocess one declared file, or explain why it was skipped."""
    raw = find_raw(raw_root, source.raw_dirname, sf.file_id)
    if raw is None:
        hint = (
            "not recorded yet"
            if sf.download_url == TODO_SENTINEL
            else "not downloaded yet (run scripts/download_sources.py)"
        )
        return Outcome(sf.file_id, source.id, "missing_raw", detail=hint)

    start, end, problem = excerpt_bounds(sf)
    if problem:
        return Outcome(sf.file_id, source.id, "todo", detail=problem, raw_path=raw)

    dest = clean_root / f"{sf.file_id}.wav"
    if dest.is_file() and not force:
        stats = describe(*_read(dest))
        return Outcome(
            sf.file_id,
            source.id,
            "cached",
            raw_path=raw,
            wav_path=dest,
            raw_sha256=sha256_file(raw),
            wav_sha256=sha256_file(dest),
            stats=stats,
        )

    if dry_run:
        return Outcome(
            sf.file_id,
            source.id,
            "todo",
            detail=f"dry run: would build {dest.name} from {raw.name}",
            raw_path=raw,
        )

    try:
        wav, stats = preprocess(
            raw,
            dest,
            excerpt_start_s=start,
            excerpt_end_s=end,
            target_sr=int(audio_cfg["sample_rate"]),
            target_lufs=float(audio_cfg["target_lufs"]),
            peak_ceiling_dbfs=float(audio_cfg["peak_ceiling_dbfs"]),
            keep_s=float(audio_cfg["trim_silence_keep_s"]),
        )
    except (AudioError, OSError) as exc:
        return Outcome(sf.file_id, source.id, "failed", detail=str(exc), raw_path=raw)

    violations = check_invariants(
        stats,
        target_lufs=float(audio_cfg["target_lufs"]),
        peak_ceiling=float(audio_cfg["peak_ceiling_dbfs"]),
    )
    if violations:
        return Outcome(
            sf.file_id,
            source.id,
            "failed",
            detail="; ".join(violations),
            raw_path=raw,
            wav_path=wav,
            stats=stats,
        )

    if stats.lufs is not None and abs(stats.lufs - float(audio_cfg["target_lufs"])) > (
        LUFS_TOLERANCE_DB
    ):
        # Legitimate when the peak ceiling bit first (see loudness_normalize),
        # so it is reported, not fatal.
        logger.warning(
            "%s: %.2f LUFS, %.2f dB from the %.1f target (peak ceiling held it back)",
            sf.file_id,
            stats.lufs,
            stats.lufs - float(audio_cfg["target_lufs"]),
            float(audio_cfg["target_lufs"]),
        )

    logger.info(
        "%s: %.3f s, %s LUFS, peak %.2f dBFS -> %s",
        sf.file_id,
        stats.duration_s,
        "n/a" if stats.lufs is None else f"{stats.lufs:.2f}",
        stats.peak_dbfs,
        wav.name,
    )
    return Outcome(
        sf.file_id,
        source.id,
        "built",
        raw_path=raw,
        wav_path=wav,
        raw_sha256=sha256_file(raw),
        wav_sha256=sha256_file(wav),
        stats=stats,
    )


def _read(path: Path):
    """Read a WAV as ``(samples, sample_rate)`` for re-measurement."""
    import soundfile

    return soundfile.read(path, dtype="float32", always_2d=False)


def rows_for(
    source: Source, sf: SourceFile, outcome: Outcome, retrieved_on: str
) -> tuple[LicenseRow, ManifestRow]:
    """Build the ``LICENSES.csv`` and ``MANIFEST.csv`` rows for a built file.

    The licence row's ``sha256`` is the digest of the **processed** WAV, because
    that is the artefact the published dataset contains; the raw download's
    digest stays in ``checksums.sha256`` and in the QC table.
    """
    assert outcome.stats is not None
    return (
        LicenseRow(
            file_id=sf.file_id,
            title=source.title,
            speaker=sf.reader if sf.reader and sf.reader != TODO_SENTINEL else "unknown",
            source_url=source.page_url,
            licence=" ".join(source.licence.split()),
            retrieved_on=retrieved_on,
            sha256=outcome.wav_sha256,
        ),
        ManifestRow(
            file_id=sf.file_id,
            text_id=source.text_id,
            tier=source.tier,
            origin="clean",
            duration_s=outcome.stats.duration_s,
        ),
    )


def qc_row(source: Source, sf: SourceFile, outcome: Outcome) -> dict[str, str]:
    """Render one row of ``dataset/qc/clean_build.csv``."""
    stats = outcome.stats
    return {
        "file_id": sf.file_id,
        "source_id": source.id,
        "text_id": source.text_id,
        "tier": source.tier,
        "raw_file": outcome.raw_path.name if outcome.raw_path else "",
        "raw_sha256": outcome.raw_sha256,
        "wav_sha256": outcome.wav_sha256,
        "sample_rate": "" if stats is None else str(stats.sample_rate),
        "duration_s": "" if stats is None else f"{stats.duration_s:.3f}",
        "lufs": "" if stats is None or stats.lufs is None else f"{stats.lufs:.2f}",
        "peak_dbfs": "" if stats is None else f"{stats.peak_dbfs:.2f}",
        "excerpt_start_s": "" if sf.excerpt_start_s is None else str(sf.excerpt_start_s),
        "excerpt_end_s": "" if sf.excerpt_end_s is None else str(sf.excerpt_end_s),
        "status": outcome.status,
    }


def write_qc(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the QC table, sorted by ``file_id`` so diffs stay readable."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=QC_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["file_id"]):
            writer.writerow(row)
    import os

    os.replace(tmp, path)
    logger.info("%s: %d rows", path.name, len(rows))


def refresh_checksums(dataset_dir: Path, roots: list[Path], checksums_path: Path) -> int:
    """Rewrite ``checksums.sha256`` over ``roots``, keeping unrelated entries.

    Entries outside ``roots`` are preserved unless their file has disappeared,
    so running this after the downloader does not drop the raw digests (and vice
    versa).
    """
    digests: dict[str, str] = {}
    if checksums_path.is_file():
        for line in checksums_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("  ", 1)
            if len(parts) == 2 and (dataset_dir / parts[1]).is_file():
                digests[parts[1]] = parts[0]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() not in IGNORED_RAW_SUFFIXES:
                digests[path.relative_to(dataset_dir).as_posix()] = sha256_file(path)
    if digests:
        write_checksums(checksums_path, digests)
    return len(digests)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sources", type=Path, default=None, help="path to SOURCES.yaml")
    p.add_argument("--only", nargs="+", metavar="ID", help="build only these source ids")
    p.add_argument("--file", nargs="+", metavar="FILE_ID", help="build only these file ids")
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    p.add_argument("--force", action="store_true", help="rebuild WAVs that already exist")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the clean build. See the module docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # No random draws here, so skip torch seeding (see download_sources.py).
    cfg = load_config(apply_seeds=False)
    sources_path = args.sources or (cfg.path("data.dir") / "SOURCES.yaml")

    try:
        sources = list(load_sources(sources_path))
    except SourcesError as exc:
        logger.error("%s", exc)
        return 1

    if args.only:
        wanted = {s.lower() for s in args.only}
        sources = [s for s in sources if s.id.lower() in wanted]
        if not sources:
            logger.error("no source matched %s", args.only)
            return 1

    audio_cfg = dict(cfg.features["audio"])
    raw_root = cfg.path("data.audio_raw")
    clean_root = (
        cfg.ensure_dir("data.audio_clean") if not args.dry_run else (cfg.path("data.audio_clean"))
    )
    only_files = {f.lower() for f in args.file} if args.file else None

    outcomes: list[Outcome] = []
    qc_rows: list[dict[str, str]] = []
    licence_rows: list[LicenseRow] = []
    manifest_rows: list[ManifestRow] = []
    refused_required = False

    for source in sources:
        allowed, reason = licence_verdict(source.licence)
        if not allowed:
            unfilled = TODO_SENTINEL in source.licence
            status = "todo" if unfilled else "refused"
            refused_required = refused_required or (not unfilled and not source.optional)
            (logger.warning if (unfilled or source.optional) else logger.error)(
                "%s %s (%s): %s",
                "SKIPPED" if unfilled else "REFUSED",
                source.id,
                source.page_url,
                reason,
            )
            for sf in source.files:
                if only_files is None or sf.file_id.lower() in only_files:
                    outcomes.append(Outcome(sf.file_id, source.id, status, detail=reason))
            continue

        for sf in source.files:
            if only_files is not None and sf.file_id.lower() not in only_files:
                continue
            outcome = build_one(
                source,
                sf,
                raw_root,
                clean_root,
                audio_cfg=audio_cfg,
                dry_run=args.dry_run,
                force=args.force,
            )
            outcomes.append(outcome)
            qc_rows.append(qc_row(source, sf, outcome))
            if outcome.status in {"built", "cached"}:
                lic, man = rows_for(source, sf, outcome, _retrieved_on(sources_path))
                licence_rows.append(lic)
                manifest_rows.append(man)
            elif outcome.status == "failed":
                logger.error("%s: %s", outcome.file_id, outcome.detail)
            else:
                logger.info("%s: %s (%s)", outcome.file_id, outcome.status, outcome.detail)

    if not args.dry_run and licence_rows:
        append_licenses(cfg.path("data.licenses"), licence_rows)
        append_manifest(cfg.path("data.manifest"), manifest_rows)
        logger.info("MANIFEST.csv / LICENSES.csv: %d clean rows upserted", len(licence_rows))
        write_qc(cfg.path("data.qc_dir") / "clean_build.csv", qc_rows)
        n = refresh_checksums(
            cfg.path("data.dir"), [raw_root, clean_root], cfg.path("data.checksums")
        )
        logger.info("checksums.sha256: %d files", n)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    logger.info(
        "done: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing declared"
    )

    pending = [o for o in outcomes if o.status in {"missing_raw", "todo"}]
    if pending:
        print("\nNOT BUILT - one line per file, with what it is waiting for:", file=sys.stderr)
        for o in pending:
            print(f"  - {o.file_id} ({o.source_id}): {o.detail}", file=sys.stderr)
        print("", file=sys.stderr)

    if counts.get("failed") or refused_required:
        return 1
    if counts.get("built", 0) + counts.get("cached", 0) == 0:
        return 2
    return 0


def _retrieved_on(sources_path: Path) -> str:
    """Read ``retrieved_on`` from the YAML, falling back to today's date."""
    import yaml

    with Path(sources_path).open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
    value = str(document.get("retrieved_on") or "").strip()
    if value and not value.startswith(TODO_SENTINEL):
        return value
    today = dt.date.today().isoformat()
    logger.warning("retrieved_on is unset in %s; using today (%s)", sources_path, today)
    return today


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
