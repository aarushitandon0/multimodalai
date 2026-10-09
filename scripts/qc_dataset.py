#!/usr/bin/env python
"""Run the realism gates over the injected dataset (spec 5.1, 5.4).

For every label JSON in ``dataset/labels`` whose ``origin`` is ``injected``:

1. **Re-align** the edited WAV with MFA and compare every word boundary against
   the expected alignment - ``dataset/alignments/<file_id>.json``, which
   ``build_injected.py`` wrote by pushing the clean alignment through the op's
   time map.
2. **Score UTMOS** for the edited file and for its clean source, caching both
   in ``dataset/qc/utmos.csv`` so a re-run scores nothing twice.
3. **Gate** (``rostrum.inject.qc.gate``): reject on a boundary error over
   50 ms, or on a UTMOS drop over the budget for the file's level.

and then writes::

    dataset/labels/<file_id>.json     qc block filled in, qc.passed set
    dataset/qc/realign.csv            one row per file checked
    dataset/qc/utmos.csv              score cache
    dataset/qc/rejections.csv         appended, with the failing gate's reason
    dataset/MANIFEST.csv              split_fold = "excluded" for failures

A rejected file is **kept on disk** and marked, not deleted: the datasheet
should be able to say what was generated and why it was dropped.

Needs the ``mfa`` conda env (invariant 4) and, the first time, a network
download of the UTMOS checkpoint. ``--no-realign`` / ``--no-utmos`` run the
other gate alone, which is useful on a machine that has only one of the two.

Usage::

    python scripts/qc_dataset.py                    # the whole injected set
    python scripts/qc_dataset.py --only f1_rushing_L3 -v
    python scripts/qc_dataset.py --no-realign       # UTMOS gate only
    python scripts/qc_dataset.py --dry-run          # plan only, write nothing

Then ``python scripts/qc_report.py`` turns the CSVs into
``dataset/qc/summary.md``.

Exit codes:
    0  every file was checked (rejections are an outcome, not a failure)
    2  nothing to check: no injected label JSON with its inputs present
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/qc_dataset.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.config import Config, load_config
from rostrum.inject import qc
from rostrum.types import Alignment

logger = logging.getLogger("qc_dataset")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Layout:
    """The dataset directories the pass reads and writes.

    Passed explicitly rather than read from ``Config`` inside every function,
    so the tests can point the pass at a miniature dataset.

    Attributes:
        labels: ``dataset/labels``.
        alignments: ``dataset/alignments``.
        texts: ``dataset/texts``.
        clean_audio: ``dataset/audio/clean``.
        injected_audio: ``dataset/audio/injected``.
        qc_dir: ``dataset/qc``.
        manifest: ``dataset/MANIFEST.csv``.
    """

    labels: Path
    alignments: Path
    texts: Path
    clean_audio: Path
    injected_audio: Path
    qc_dir: Path
    manifest: Path

    @classmethod
    def from_config(cls, cfg: Config) -> Layout:
        """Build the layout from ``configs/paths.yaml``."""
        return cls(
            labels=cfg.path("data.labels"),
            alignments=cfg.path("data.alignments"),
            texts=cfg.path("data.texts"),
            clean_audio=cfg.path("data.audio_clean"),
            injected_audio=cfg.path("data.audio_injected"),
            qc_dir=cfg.path("data.qc_dir"),
            manifest=cfg.path("data.manifest"),
        )

    @property
    def utmos_csv(self) -> Path:
        """``dataset/qc/utmos.csv``."""
        return self.qc_dir / "utmos.csv"

    @property
    def realign_csv(self) -> Path:
        """``dataset/qc/realign.csv``."""
        return self.qc_dir / "realign.csv"

    @property
    def rejections_csv(self) -> Path:
        """``dataset/qc/rejections.csv``."""
        return self.qc_dir / "rejections.csv"


@dataclass(frozen=True)
class Job:
    """One injected file to check, with everything needed to check it.

    Attributes:
        file_id: The edited file.
        source_file: Its clean source ``file_id``.
        label: Flaw label, or ``"composite"``.
        level: Severity ``L1``-``L4``.
        label_path: The label JSON to update.
        label_json: The label as read.
        edited_wav: The generated WAV.
        clean_wav: Its clean source WAV.
        expected: Where the words should be - the time-map transformed
            alignment written by ``build_injected.py``.
        transcript: Verbatim transcript, for the aligner.
    """

    file_id: str
    source_file: str
    label: str
    level: str
    label_path: Path
    label_json: dict
    edited_wav: Path
    clean_wav: Path
    expected: Alignment
    transcript: str


@dataclass(frozen=True)
class Summary:
    """What one pass did.

    Attributes:
        checked: Files that went through the gates.
        rejected: Of those, how many failed.
        skipped: Files whose inputs were incomplete.
    """

    checked: int
    rejected: int
    skipped: int


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _transcript_for(text_id: str, file_id: str, texts_dir: Path) -> str:
    """Return the verbatim transcript for a text, or ``""`` when absent.

    Reuses ``align_clean.transcript_for`` so the clean alignment and the
    re-alignment are driven by exactly the same text - a different transcript
    would move boundaries for reasons that have nothing to do with the edit.
    """
    from align_clean import MissingTranscript, transcript_for

    try:
        text, _ = transcript_for(file_id, text_id, texts_dir)
    except MissingTranscript:
        return ""
    return text


def plan(
    layout: Layout, *, only: list[str] | None = None
) -> tuple[list[Job], list[tuple[str, str]]]:
    """Collect the injected files that can be checked.

    Args:
        layout: Dataset directories.
        only: Restrict to these ``file_id``s.

    Returns:
        ``(jobs, skipped)`` where ``skipped`` is ``(file_id, reason)`` for every
        label whose audio, expected alignment or transcript is missing. A
        skipped file is **not** a rejection: it was never judged, and saying so
        is more useful than logging a gate failure that did not happen.
    """
    wanted = set(only) if only else None
    jobs: list[Job] = []
    skipped: list[tuple[str, str]] = []

    for path in sorted(layout.labels.glob("*.json")):
        try:
            label = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            skipped.append((path.stem, f"unreadable label JSON: {exc}"))
            continue
        file_id = str(label.get("file_id", path.stem))
        if wanted is not None and file_id not in wanted:
            continue
        if label.get("origin") != "injected":
            continue

        source_file = str(label.get("source_file", ""))
        edited = layout.injected_audio / f"{file_id}.wav"
        clean = layout.clean_audio / f"{source_file}.wav"
        expected_path = layout.alignments / f"{file_id}.json"

        if not edited.is_file():
            skipped.append((file_id, f"edited WAV missing: {edited}"))
            continue
        if not expected_path.is_file():
            skipped.append((file_id, f"expected alignment missing: {expected_path}"))
            continue

        transcript = _transcript_for(str(label.get("text_id", "")), source_file, layout.texts)
        if not transcript:
            skipped.append((file_id, "no transcript for its text"))
            continue

        jobs.append(
            Job(
                file_id=file_id,
                source_file=source_file,
                label=qc.label_of(label),
                level=qc.level_of(label),
                label_path=path,
                label_json=label,
                edited_wav=edited,
                clean_wav=clean,
                expected=Alignment.from_dict(json.loads(expected_path.read_text(encoding="utf-8"))),
                transcript=transcript,
            )
        )

    if wanted:
        for file_id in sorted(wanted - {j.file_id for j in jobs} - {s[0] for s in skipped}):
            skipped.append((file_id, "no label JSON"))
    return jobs, skipped


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def score_utmos(
    jobs: list[Job], layout: Layout, *, scorer: qc.Scorer | None = None
) -> dict[str, float]:
    """Score every edited file and every clean source, through the cache.

    A clean source appears once however many edits were made from it, so it is
    scored once per build rather than once per edit.

    Args:
        jobs: Files to score.
        layout: For the cache path.
        scorer: Injected scorer; the UTMOS model is loaded lazily when omitted.

    Returns:
        ``{file_id: mos}`` covering edited files and clean sources.
    """
    paths: dict[str, Path] = {}
    for job in jobs:
        paths[job.file_id] = job.edited_wav
        if job.clean_wav.is_file():
            paths[job.source_file] = job.clean_wav
        else:
            logger.warning("clean source WAV missing, no UTMOS baseline: %s", job.clean_wav)
    return qc.utmos_for_files(
        [paths[k] for k in sorted(paths)], cache_path=layout.utmos_csv, scorer=scorer
    )


def check_one(
    job: Job,
    *,
    scores: dict[str, float],
    cfg: Config | None = None,
    align_fn: qc.Aligner | None = None,
    realign: bool = True,
    work_root: Path | None = None,
) -> tuple[qc.QCVerdict, qc.RealignReport | None]:
    """Gate one file, turning an incomparable re-alignment into a rejection.

    Args:
        job: The file to check.
        scores: UTMOS scores by ``file_id``; missing entries disable that gate.
        cfg: Config holding ``injection.qc``; loaded when omitted.
        align_fn: Aligner; MFA when omitted.
        realign: Run the re-alignment gate at all.
        work_root: Scratch root for the aligner.

    Returns:
        ``(verdict, report)``; ``report`` is ``None`` when re-alignment was
        skipped or could not be compared.
    """
    report: qc.RealignReport | None = None
    if realign:
        try:
            report = qc.realign_check(
                job.edited_wav,
                job.transcript,
                job.expected,
                source_file=job.source_file,
                label=job.label,
                level=job.level,
                align_fn=align_fn,
                work_dir=(work_root / job.file_id) if work_root is not None else None,
                cfg=cfg,
            )
        except ValueError as exc:
            # MFA re-segmented the edited audio: a QC failure, not a crash.
            logger.warning("%s: re-alignment not comparable (%s)", job.file_id, exc)
            return (
                qc.unalignable_verdict(file_id=job.file_id, level=job.level, reason=str(exc)),
                None,
            )

    verdict = qc.gate(
        realign=report,
        utmos_clean=scores.get(job.source_file),
        utmos_edited=scores.get(job.file_id),
        level=job.level,
        file_id=job.file_id,
        cfg=cfg,
    )
    return verdict, report


def run(
    layout: Layout,
    *,
    only: list[str] | None = None,
    cfg: Config | None = None,
    align_fn: qc.Aligner | None = None,
    scorer: qc.Scorer | None = None,
    realign: bool = True,
    utmos: bool = True,
    work_root: Path | None = None,
) -> Summary:
    """Run the gates over the dataset and write every QC artefact.

    Args:
        layout: Dataset directories.
        only: Restrict to these ``file_id``s.
        cfg: Config; loaded when omitted.
        align_fn: Aligner; MFA when omitted.
        scorer: UTMOS scorer; the pinned model when omitted.
        realign: Run the re-alignment gate.
        utmos: Run the UTMOS gate.
        work_root: Scratch root for the aligner.

    Returns:
        A :class:`Summary` of the pass.
    """
    jobs, skipped = plan(layout, only=only)
    for file_id, reason in skipped:
        logger.warning("skipping %s: %s", file_id, reason)
    if not jobs:
        return Summary(checked=0, rejected=0, skipped=len(skipped))

    scores = score_utmos(jobs, layout, scorer=scorer) if utmos else {}

    verdicts: list[qc.QCVerdict] = []
    reports: list[qc.RealignReport] = []
    rejections: list[dict[str, str]] = []
    marked: list[dict] = []

    for job in jobs:
        verdict, report = check_one(
            job,
            scores=scores,
            cfg=cfg,
            align_fn=align_fn,
            realign=realign,
            work_root=work_root,
        )
        verdicts.append(verdict)
        if report is not None:
            reports.append(report)

        label = qc.apply_verdict(job.label_json, verdict)
        job.label_path.write_text(
            json.dumps(label, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        marked.append(label)
        if not verdict.passed:
            rejections.append(qc.rejection_row(label, verdict))
            logger.warning("%s rejected: %s", job.file_id, "; ".join(verdict.reasons))

    layout.qc_dir.mkdir(parents=True, exist_ok=True)
    if reports:
        qc.append_realign(layout.realign_csv, reports)
    if rejections:
        qc.append_rejections(layout.rejections_csv, rejections)
    if layout.manifest.is_file():
        qc.exclude_rejected(layout.manifest, marked)

    return Summary(
        checked=len(jobs),
        rejected=sum(1 for v in verdicts if not v.passed),
        skipped=len(skipped),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="check only these files")
    p.add_argument("--no-realign", action="store_true", help="skip the MFA re-alignment gate")
    p.add_argument("--no-utmos", action="store_true", help="skip the UTMOS gate")
    p.add_argument("--dry-run", action="store_true", help="list what would be checked")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the QC pass over ``dataset/``."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    cfg = load_config(apply_seeds=False)
    layout = Layout.from_config(cfg)

    if args.dry_run:
        jobs, skipped = plan(layout, only=args.only)
        for job in jobs:
            print(f"{job.file_id}\t{job.label}\t{job.level}")
        for file_id, reason in skipped:
            print(f"SKIP {file_id}: {reason}", file=sys.stderr)
        print(f"{len(jobs)} file(s) to check, {len(skipped)} skipped")
        return 0 if jobs else 2

    summary = run(
        layout,
        only=args.only,
        cfg=cfg,
        realign=not args.no_realign,
        utmos=not args.no_utmos,
        work_root=cfg.path("cache.dir") / "mfa" / "realign",
    )
    if summary.checked == 0:
        print(
            f"nothing to check in {layout.labels}: run scripts/build_injected.py first",
            file=sys.stderr,
        )
        return 2

    rate = 100.0 * summary.rejected / summary.checked
    print(
        f"checked {summary.checked} file(s): {summary.rejected} rejected ({rate:.1f}%), "
        f"{summary.skipped} skipped"
    )
    print(f"next: python scripts/qc_report.py  ->  {layout.qc_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
