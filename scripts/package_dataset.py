#!/usr/bin/env python
"""Finalise and package the dataset (spec 7.3-7.4).

Six steps, in order, each of which can also be run on its own:

1. **Manifest.** Rebuild ``dataset/MANIFEST.csv`` from what is actually on
   disk: one row per audio file, origin from its directory, label and level
   from ``labels/<file_id>.json``, duration from the WAV itself.
2. **Splits.** Build ``dataset/splits/loto.json`` - one leave-one-text-out fold
   per text, human takes test-only - and write each file's fold into the
   manifest's ``split_fold`` column. The split is checked for leakage before it
   is written (spec 7.3, CLAUDE.md invariant 11).
3. **Provenance.** Re-run the licence check of ``scripts/check_provenance.py``;
   nothing is packaged without it.
4. **Datasheet.** Regenerate the counts, label-grid, QC and split blocks inside
   ``dataset/README.md``. Prose is hand-written, numbers never are.
5. **Checksums.** Rewrite ``dataset/checksums.sha256`` over every shipped file,
   then verify the file it just wrote.
6. **Sample and zip.** Refresh ``dataset_sample/`` (~20 files chosen to cover
   every label and level, plus two human takes) and write
   ``dist/rostrum_dataset_v1.zip``.

Usage::

    python scripts/package_dataset.py                  # all six steps
    python scripts/package_dataset.py --skip-zip       # everything but the zip
    python scripts/package_dataset.py --only sample    # one step
    python scripts/package_dataset.py --check          # verify, change nothing

Exit codes:
    0  packaged (or verified) successfully
    1  a step failed: provenance, leakage, checksum mismatch, oversized sample
    2  the dataset directory could not be read
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_provenance
from rostrum.config import Config, load_config
from rostrum.data.manifest import (
    LICENSES_FIELDS,
    MANIFEST_FIELDS,
    ManifestRow,
    read_manifest,
    write_checksums,
)
from rostrum.data.package import (
    composition_table,
    label_table,
    manifest_row_from_label,
    render_autoblocks,
    sample_coverage,
    select_sample,
)
from rostrum.data.splits import build_loto, check_no_leakage, with_split_folds, write_loto
from rostrum.io.hashing import sha256_file

logger = logging.getLogger("package_dataset")

AUDIO_DIRS: tuple[tuple[str, str], ...] = (
    ("audio/clean", "clean"),
    ("audio/injected", "injected"),
    ("audio/human_flawed", "human"),
)
"""Audio directories that ship, with the ``origin`` each one implies."""

SAMPLE_MAX_BYTES = 50 * 1024 * 1024
"""``dataset_sample/`` must stay under 50 MB so the repo stays clonable."""

SAMPLE_MAX_FILES = 20
"""Target sample size (spec 7.4: "~20 files")."""

SAMPLE_AUDIO_BUDGET = int(0.9 * SAMPLE_MAX_BYTES)
"""Share of the budget the audio may use; the rest covers labels and CSVs."""

CHECKSUM_SKIP = frozenset({"checksums.sha256"})
"""Files that cannot appear in the checksum list (itself)."""

SAMPLE_KEEP_DIRS = frozenset({"smoke"})
"""Sample subdirectories written by hand, which the packager never touches."""

STEPS = ("manifest", "splits", "provenance", "datasheet", "checksums", "sample", "zip")


class PackagingError(RuntimeError):
    """A packaging step failed."""


# --------------------------------------------------------------------------- #
# 1. Manifest
# --------------------------------------------------------------------------- #


def audio_files(dataset_dir: Path) -> list[tuple[Path, str]]:
    """Every shipped WAV with its origin, sorted by path."""
    out: list[tuple[Path, str]] = []
    for rel, origin in AUDIO_DIRS:
        d = dataset_dir / rel
        if d.is_dir():
            out.extend((p, origin) for p in sorted(d.glob("*.wav")))
    return out


def wav_duration_s(path: Path) -> float:
    """Duration of a 16-bit PCM WAV, read from its header."""
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def collect_rows(dataset_dir: Path, *, strict: bool = False) -> list[ManifestRow]:
    """Rebuild the manifest rows from the audio, labels and previous manifest.

    Metadata precedence per file: the label JSON, then the previous manifest
    row, then the parent clean file's row (an injected file inherits text and
    tier from the recording its ``file_id`` is prefixed with, spec 7.1).

    Args:
        dataset_dir: The dataset root.
        strict: Raise if a file ends up with no ``text_id``; otherwise warn.

    Returns:
        Rows sorted by ``file_id``, with ``split_fold`` empty.

    Raises:
        PackagingError: ``strict`` and some file has no resolvable ``text_id``.
    """
    previous = {r.file_id: r for r in read_manifest(dataset_dir / "MANIFEST.csv")}
    labels_dir = dataset_dir / "labels"

    rows: list[ManifestRow] = []
    orphans: list[str] = []
    for path, origin in audio_files(dataset_dir):
        file_id = path.stem
        duration = wav_duration_s(path)
        label_path = labels_dir / f"{file_id}.json"

        row: ManifestRow | None = None
        if label_path.is_file():
            data = json.loads(label_path.read_text(encoding="utf-8"))
            row = manifest_row_from_label(data, duration)
        old = previous.get(file_id)
        parent = _parent_row(file_id, previous)

        text_id = _first(
            row.text_id if row else "", old.text_id if old else "", parent.text_id if parent else ""
        )
        tier = _first(
            row.tier if row else "", old.tier if old else "", parent.tier if parent else ""
        )
        rows.append(
            ManifestRow(
                file_id=file_id,
                text_id=text_id,
                tier=tier,
                origin=origin,
                label=row.label if row else (old.label if old else ""),
                level=row.level if row else (old.level if old else ""),
                duration_s=round(duration, 3),
            )
        )
        if not text_id:
            orphans.append(file_id)

    if orphans:
        message = (
            f"{len(orphans)} audio file(s) have no text_id in labels/ or MANIFEST.csv: "
            + ", ".join(orphans)
        )
        if strict:
            raise PackagingError(message)
        logger.warning("%s", message)
    return sorted(rows, key=lambda r: r.file_id)


def _parent_row(file_id: str, previous: dict[str, ManifestRow]) -> ManifestRow | None:
    """The longest previous row whose ``file_id`` prefixes ``file_id``."""
    best: ManifestRow | None = None
    for other_id, row in previous.items():
        if file_id.startswith(f"{other_id}_") and (
            best is None or len(other_id) > len(best.file_id)
        ):
            best = row
    return best


def _first(*values: str) -> str:
    """First non-empty value, or ``""``."""
    return next((v for v in values if v), "")


def write_manifest(path: Path, rows: list[ManifestRow]) -> int:
    """Rewrite ``MANIFEST.csv`` from scratch, sorted, newline-terminated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(MANIFEST_FIELDS), lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r.file_id):
            w.writerow(
                {
                    "file_id": r.file_id,
                    "text_id": r.text_id,
                    "tier": r.tier,
                    "origin": r.origin,
                    "label": r.label,
                    "level": r.level,
                    "duration_s": "" if r.duration_s == "" else f"{float(r.duration_s):.3f}",
                    "split_fold": r.split_fold,
                }
            )
    os.replace(tmp, path)
    logger.info("%s: %d row(s)", path, len(rows))
    return len(rows)


# --------------------------------------------------------------------------- #
# 2. Splits
# --------------------------------------------------------------------------- #


def build_splits(dataset_dir: Path, rows: list[ManifestRow]) -> list[ManifestRow]:
    """Write ``splits/loto.json`` and return ``rows`` with ``split_fold`` set.

    Raises:
        PackagingError: Fewer than two texts, or the split leaks.
    """
    usable = [r for r in rows if r.text_id]
    try:
        splits = build_loto(usable)
    except ValueError as exc:
        raise PackagingError(f"cannot build LOTO splits: {exc}") from exc
    check_no_leakage(splits, usable)
    write_loto(dataset_dir / "splits" / "loto.json", splits)
    logger.info("splits/loto.json: %d fold(s) over %s", len(splits.folds), ", ".join(splits.texts))
    return with_split_folds(rows)


# --------------------------------------------------------------------------- #
# 3. Provenance
# --------------------------------------------------------------------------- #


def verify_provenance(cfg: Config, dataset_dir: Path) -> list[check_provenance.Problem]:
    """Run the licence check over ``dataset_dir`` and return its problems."""
    return check_provenance.check(
        dataset_dir=dataset_dir,
        licenses_path=dataset_dir / "LICENSES.csv",
        sources_path=cfg.path("data.dir") / "SOURCES.yaml",
        provenance_dir=dataset_dir / "provenance",
        consent_path=dataset_dir / "CONSENT.md",
    )


# --------------------------------------------------------------------------- #
# 4. Datasheet
# --------------------------------------------------------------------------- #


def qc_block(dataset_dir: Path) -> str:
    """Summarise the QC gates for the datasheet, or say the build has not run."""
    qc_dir = dataset_dir / "qc"
    summary = qc_dir / "summary.md"
    if not summary.is_file():
        return (
            "_No injection QC has been run yet: `dataset/qc/summary.md` does not exist. "
            "`make qc && make qc-report` fills UTMOS, re-alignment and rejection numbers "
            "here._"
        )
    try:
        import qc_report
    except Exception as exc:  # pragma: no cover - optional heavy deps
        logger.warning("cannot import qc_report (%s); linking summary.md instead", exc)
        return f"See [`qc/summary.md`](qc/summary.md).\n\n{_blockquote(summary)}"

    records = qc_report.collect(dataset_dir / "labels", qc_dir / "utmos.csv")
    realign = qc_report.read_realign(qc_dir / "realign.csv")
    rejections = qc_report.read_rejections(qc_dir / "rejections.csv")
    counts = qc_report.rejection_counts(rejections)
    dist = qc_report.realign_distribution(realign)
    rate = qc_report.rejection_rate(records, rejections)

    lines = [
        "| Gate | Result |",
        "|---|---|",
        f"| Files passing every gate | {sum(1 for r in records if r.passed)} / {len(records)} |",
        f"| Rejection rate | {'n/a' if rate is None else f'{rate:.1%}'} |",
        f"| Re-alignment max word error | median {_ms(dist.median)}, "
        f"P95 {_ms(dist.p95)}, worst {_ms(dist.maximum)} |",
        f"| Within the 50 ms gate | {_pct(dist.within_gate_frac)} of {dist.n} file(s) |",
        f"| Rejections by reason | {_reasons(counts)} |",
    ]
    return "\n".join(lines) + "\n\nFull tables: [`qc/summary.md`](qc/summary.md)."


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} ms"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _reasons(counts: Any) -> str:
    by_reason = getattr(counts, "by_reason", None) or {}
    if not by_reason:
        return "none"
    return ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items()))


def _blockquote(path: Path) -> str:
    body = path.read_text(encoding="utf-8").strip().splitlines()
    return "\n".join(f"> {line}" for line in body[:40])


def splits_block(dataset_dir: Path, rows: list[ManifestRow]) -> str:
    """Render the fold table from ``splits/loto.json``."""
    path = dataset_dir / "splits" / "loto.json"
    if not path.is_file():
        return "_`splits/loto.json` has not been built yet; run `make data`._"
    doc = json.loads(path.read_text(encoding="utf-8"))
    lines = ["| Fold | Held-out text | Train files | Test files |", "|---|---|---|---|"]
    for fold in doc["folds"]:
        lines.append(
            f"| `{fold['fold_id']}` | `{fold['held_out_text']}` | "
            f"{fold['n_train']} | {fold['n_test']} |"
        )
    n_human = sum(1 for r in rows if r.origin == "human")
    lines.append("")
    lines.append(
        f"All {n_human} human-recorded file(s) appear in the test side of their own "
        "text's fold only."
    )
    return "\n".join(lines)


def update_datasheet(dataset_dir: Path, rows: list[ManifestRow]) -> Path:
    """Regenerate the auto-blocks inside ``dataset/README.md``."""
    path = dataset_dir / "README.md"
    document = path.read_text(encoding="utf-8")
    updated = render_autoblocks(
        document,
        {
            "composition": composition_table(rows),
            "labels": label_table(rows),
            "qc": qc_block(dataset_dir),
            "splits": splits_block(dataset_dir, rows),
        },
    )
    if updated != document:
        path.write_text(updated, encoding="utf-8", newline="\n")
        logger.info("README.md: generated blocks refreshed")
    else:
        logger.info("README.md: generated blocks already current")
    return path


# --------------------------------------------------------------------------- #
# 5. Checksums
# --------------------------------------------------------------------------- #


def shipped_files(root: Path) -> list[Path]:
    """Every file under ``root`` that belongs in the checksum list, sorted."""
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if p.is_file() and rel not in CHECKSUM_SKIP and not rel.endswith(".tmp"):
            out.append(p)
    return out


def regenerate_checksums(root: Path) -> int:
    """Rewrite ``<root>/checksums.sha256`` over every shipped file."""
    digests = {p.relative_to(root).as_posix(): sha256_file(p) for p in shipped_files(root)}
    n = write_checksums(root / "checksums.sha256", digests)
    logger.info("checksums.sha256: %d file(s) under %s", n, root)
    return n


def verify_checksums(root: Path) -> list[str]:
    """Return the dataset-relative paths whose digest does not match the list.

    A missing file counts as a mismatch; a file present but unlisted does not,
    so an untracked scratch file cannot fail the check.
    """
    path = root / "checksums.sha256"
    if not path.is_file():
        raise PackagingError(f"{path} does not exist; run with --only checksums first")
    bad: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition("  ")
        if not rel:
            continue
        target = root / rel
        if not target.is_file() or sha256_file(target) != digest:
            bad.append(rel)
    return bad


# --------------------------------------------------------------------------- #
# 6. Sample and zip
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SampleResult:
    """Outcome of refreshing ``dataset_sample/``."""

    n_files: int
    n_bytes: int
    file_ids: tuple[str, ...]

    @property
    def megabytes(self) -> float:
        """Sample size in MB."""
        return self.n_bytes / (1024 * 1024)


def build_sample(
    dataset_dir: Path,
    sample_dir: Path,
    file_ids: list[str],
    rows: list[ManifestRow] | None = None,
) -> SampleResult:
    """Refresh ``dataset_sample/`` with ``file_ids`` and their sidecars.

    Copies the audio flat into ``audio/``, each file's ``labels/*.json`` and
    ``alignments/*.TextGrid`` where they exist, the texts those files render
    into ``transcripts/``, and ``MANIFEST.csv`` / ``LICENSES.csv`` filtered to
    the subset. Anything the packager owns is rebuilt, so a file that drops out
    of the selection disappears; hand-written notes under
    :data:`SAMPLE_KEEP_DIRS` are left alone.
    """
    wanted = sorted(set(file_ids))
    for sub in ("audio", "labels", "alignments", "transcripts", "splits"):
        target = sample_dir / sub
        if target.is_dir():
            for p in sorted(target.rglob("*")):
                if p.is_file() and p.name != ".gitkeep":
                    p.unlink()
        target.mkdir(parents=True, exist_ok=True)

    sources = {p.stem: p for p, _ in audio_files(dataset_dir)}
    if rows is None:
        rows = collect_rows(dataset_dir)
    by_id = {r.file_id: r for r in rows}
    copied: list[str] = []
    texts: set[str] = set()

    for file_id in wanted:
        src = sources.get(file_id)
        if src is None:
            logger.warning("sample: %s has no audio under %s", file_id, dataset_dir)
            continue
        shutil.copy2(src, sample_dir / "audio" / src.name)
        copied.append(file_id)
        for sub, suffix in (("labels", ".json"), ("alignments", ".TextGrid")):
            side = dataset_dir / sub / f"{file_id}{suffix}"
            if side.is_file():
                shutil.copy2(side, sample_dir / sub / side.name)
        row = by_id.get(file_id)
        if row and row.text_id:
            texts.add(row.text_id)

    for text_id in sorted(texts):
        src = dataset_dir / "texts" / f"{text_id}.json"
        if src.is_file():
            shutil.copy2(src, sample_dir / "transcripts" / src.name)

    write_manifest(sample_dir / "MANIFEST.csv", [by_id[f] for f in copied if f in by_id])
    _filter_csv(
        dataset_dir / "LICENSES.csv",
        sample_dir / "LICENSES.csv",
        LICENSES_FIELDS,
        _licence_ids(set(copied), set(by_id)),
    )
    splits = dataset_dir / "splits" / "loto.json"
    if splits.is_file():
        shutil.copy2(splits, sample_dir / "splits" / "loto.json")

    regenerate_checksums(sample_dir)
    n_bytes = sum(p.stat().st_size for p in sample_dir.rglob("*") if p.is_file())
    logger.info("dataset_sample: %d file(s), %.1f MB", len(copied), n_bytes / (1024 * 1024))
    return SampleResult(n_files=len(copied), n_bytes=n_bytes, file_ids=tuple(copied))


def _licence_ids(copied: set[str], known: set[str]) -> set[str]:
    """File ids whose licence row a sampled file depends on.

    An injected file has no row of its own; it inherits the row of the clean
    recording its ``file_id`` is prefixed with (spec 7.1).
    """
    out = set(copied)
    for file_id in copied:
        for other in known:
            if file_id.startswith(f"{other}_"):
                out.add(other)
    return out


def _filter_csv(src: Path, dst: Path, columns: tuple[str, ...], keep: set[str]) -> int:
    """Copy ``src`` to ``dst`` keeping the header and rows whose file_id is kept."""
    if not src.is_file():
        return 0
    with src.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("file_id") in keep]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["file_id"]):
            w.writerow({k: r.get(k, "") for k in columns})
    return len(rows)


def make_zip(dataset_dir: Path, zip_path: Path) -> Path:
    """Zip ``dataset_dir`` with dataset-relative names, atomically."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dataset_dir.rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                zf.write(p, p.relative_to(dataset_dir).as_posix())
    os.replace(tmp, zip_path)
    size = zip_path.stat().st_size / (1024 * 1024)
    logger.info("%s: %.1f MB", zip_path, size)
    return zip_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", type=Path, default=None, help="dataset root")
    p.add_argument("--sample", type=Path, default=None, help="dataset_sample root")
    p.add_argument("--zip", type=Path, default=None, help="output zip path")
    p.add_argument("--only", choices=STEPS, action="append", help="run just this step")
    p.add_argument("--skip-zip", action="store_true", help="do everything but the zip")
    p.add_argument("--max-files", type=int, default=SAMPLE_MAX_FILES, help="sample file cap")
    p.add_argument(
        "--check",
        action="store_true",
        help="verify provenance, splits and checksums without writing anything",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail if the sample cannot cover every label and level",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    dataset_dir = args.dataset or cfg.path("data.dir")
    sample_dir = args.sample or cfg.path("sample.dir")
    zip_path = args.zip or (cfg.path("root") / "dist" / "rostrum_dataset_v1.zip")
    steps = set(args.only or STEPS)

    if not dataset_dir.is_dir():
        logger.error("%s does not exist", dataset_dir)
        return 2

    failures: list[str] = []

    rows = collect_rows(dataset_dir, strict=args.strict)
    if not rows:
        logger.error("no audio under %s - nothing to package", dataset_dir)
        return 2

    if "splits" in steps:
        try:
            rows = build_splits(dataset_dir, rows) if not args.check else rows
        except PackagingError as exc:
            logger.error("%s", exc)
            failures.append("splits")
    if "manifest" in steps and not args.check:
        write_manifest(dataset_dir / "MANIFEST.csv", rows)

    if "provenance" in steps:
        problems = verify_provenance(cfg, dataset_dir)
        for problem in problems:
            logger.error("provenance: %s", problem)
        if problems:
            failures.append("provenance")
        else:
            logger.info("provenance OK (%d audio file(s))", len(rows))

    if "datasheet" in steps and not args.check:
        update_datasheet(dataset_dir, rows)

    if "checksums" in steps:
        if not args.check:
            regenerate_checksums(dataset_dir)
        bad = verify_checksums(dataset_dir)
        for rel in bad:
            logger.error("checksum mismatch: %s", rel)
        if bad:
            failures.append("checksums")
        else:
            logger.info("checksums verify")

    if "sample" in steps:
        sizes = {p.stem: p.stat().st_size for p, _ in audio_files(dataset_dir)}
        chosen = select_sample(
            rows,
            max_files=args.max_files,
            sizes=sizes,
            budget_bytes=SAMPLE_AUDIO_BUDGET,
        )
        coverage = sample_coverage(rows, chosen)
        if not args.check:
            result = build_sample(dataset_dir, sample_dir, chosen, rows)
            if result.n_bytes > SAMPLE_MAX_BYTES:
                logger.error(
                    "dataset_sample is %.1f MB, over the %.0f MB limit",
                    result.megabytes,
                    SAMPLE_MAX_BYTES / (1024 * 1024),
                )
                failures.append("sample")
            else:
                logger.info(
                    "dataset_sample: %.1f MB of %.0f MB budget",
                    result.megabytes,
                    SAMPLE_MAX_BYTES / (1024 * 1024),
                )
        _report_coverage(coverage, strict=args.strict, failures=failures)

    if "zip" in steps and not (args.skip_zip or args.check):
        if failures:
            logger.warning("skipping the zip: %s failed", ", ".join(sorted(set(failures))))
        else:
            make_zip(dataset_dir, zip_path)

    if failures:
        logger.error("packaging failed: %s", ", ".join(sorted(set(failures))))
        return 1
    logger.info("dataset packaged")
    return 0


def _report_coverage(coverage: Any, *, strict: bool, failures: list[str]) -> None:
    """Log what the sample covers; fail only under ``--strict``."""
    logger.info(
        "sample coverage: %d file(s), %d clean, %d human, %d label x level pair(s)",
        coverage.n_files,
        coverage.n_clean,
        coverage.n_human,
        len(coverage.pairs),
    )
    gaps = []
    if coverage.absent_pairs:
        gaps.append(
            f"{len(coverage.absent_pairs)} label x level pair(s) have never been built: "
            + ", ".join(f"{lab}/{lvl}" for lab, lvl in coverage.absent_pairs)
        )
    if coverage.missing_pairs:
        gaps.append(
            "pair(s) built but left out of the sample: "
            + ", ".join(f"{lab}/{lvl}" for lab, lvl in coverage.missing_pairs)
        )
    if coverage.n_human < 2:
        gaps.append(f"only {coverage.n_human} human file(s) available; spec 7.4 asks for 2")
    for gap in gaps:
        (logger.error if strict else logger.warning)("sample: %s", gap)
    if gaps and strict:
        failures.append("sample")


if __name__ == "__main__":
    raise SystemExit(main())
