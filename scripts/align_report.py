#!/usr/bin/env python
"""Report alignment quality per file and flag the ones a human must fix (spec 8).

Reads every ``dataset/alignments/<file_id>.TextGrid`` written by
``scripts/align_clean.py``, pairs it with the transcript that produced it, and
reports per file:

====================  ===============================================================
``n_words``           aligned word intervals (silences excluded)
``n_unaligned``       ``<unk>`` intervals + zero-length words + transcript tokens
                      that never made it into the grid
``mean_word_dur_s``   mean aligned word duration
``longest_gap_s``     longest inter-word silence, and the word index it follows
====================  ===============================================================

A file is **flagged** when ``n_unaligned / n_words`` exceeds
``paths.mfa.qc.max_unaligned_frac`` (2% by default): that is the point at which
the word-index correspondence between renditions of a text stops being
trustworthy, and the fix is a transcript override or a project dictionary entry,
not a tolerance change.

Output: ``dataset/qc/align_report.csv`` plus a table on stdout.

Usage::

    python scripts/align_report.py
    python scripts/align_report.py --only s1_lv01 s1_lv02
    python scripts/align_report.py --max-unaligned-frac 0.05
    python scripts/align_report.py --alignments dataset_sample/alignments

Exit codes:
    0  every file is within the unaligned-word budget
    1  at least one file is flagged (each is listed with what to do)
    2  no TextGrids to report on
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow `python scripts/align_report.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.align.mfa import UNKNOWN_TOKEN, normalise_for_alignment, read_textgrid
from rostrum.config import load_config
from rostrum.data.manifest import read_manifest
from rostrum.types import Alignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_clean import MissingTranscript, transcript_for

logger = logging.getLogger("align_report")

REPORT_FIELDS: tuple[str, ...] = (
    "file_id",
    "text_id",
    "n_words",
    "n_expected_tokens",
    "n_unk",
    "n_zero_length",
    "n_missing",
    "n_unaligned",
    "unaligned_frac",
    "mean_word_dur_s",
    "median_word_dur_s",
    "longest_gap_s",
    "longest_gap_after_word",
    "speech_duration_s",
    "duration_s",
    "flagged",
    "reason",
)
"""Column order of ``dataset/qc/align_report.csv``."""

ZERO_LENGTH_S = 0.001
"""A word shorter than this never carried any audio; MFA emits these on failure."""


@dataclass(frozen=True)
class FileReport:
    """Alignment quality for one file. Field order matches :data:`REPORT_FIELDS`."""

    file_id: str
    text_id: str
    n_words: int
    n_expected_tokens: int
    n_unk: int
    n_zero_length: int
    n_missing: int
    n_unaligned: int
    unaligned_frac: float
    mean_word_dur_s: float
    median_word_dur_s: float
    longest_gap_s: float
    longest_gap_after_word: int
    speech_duration_s: float
    duration_s: float
    flagged: bool
    reason: str


# --------------------------------------------------------------------------- #
# Pure summarisation
# --------------------------------------------------------------------------- #


def summarise(
    alignment: Alignment,
    *,
    n_expected_tokens: int,
    max_unaligned_frac: float,
    text_id: str = "",
) -> FileReport:
    """Summarise one parsed alignment.

    ``n_unaligned`` counts three distinct failures, all of which break the
    word-index correspondence the envelope depends on:

    * ``<unk>`` - the dictionary had no pronunciation, so MFA aligned a
      placeholder;
    * zero-length - MFA emitted an interval with no audio in it;
    * missing - the grid has fewer words than the transcript has tokens, so some
      token was dropped entirely.

    Args:
        alignment: Parsed TextGrid.
        n_expected_tokens: Token count of the transcript that was aligned.
        max_unaligned_frac: Flag threshold, as a fraction of ``n_words``.
        text_id: Library text id, for the report row.

    Returns:
        A :class:`FileReport`.
    """
    words = alignment.words
    n_words = len(words)
    durations = sorted(w.duration for w in words)

    n_unk = sum(1 for w in words if w.text.lower() == UNKNOWN_TOKEN)
    n_zero = sum(1 for w in words if w.duration < ZERO_LENGTH_S)
    n_missing = max(0, n_expected_tokens - n_words)
    n_unaligned = n_unk + n_zero + n_missing
    frac = n_unaligned / n_words if n_words else 1.0

    longest_gap = max(alignment.gaps, key=lambda g: g.duration, default=None)

    reasons: list[str] = []
    if not n_words:
        reasons.append("no aligned words at all")
    if n_unk:
        reasons.append(f"{n_unk} <unk> word(s): add pronunciations to a project dictionary")
    if n_zero:
        reasons.append(f"{n_zero} zero-length word(s): alignment failed inside the file")
    if n_missing:
        reasons.append(
            f"{n_missing} transcript token(s) never aligned: the reading does not match the "
            "text (try dataset/texts/overrides/<file_id>.txt)"
        )
    flagged = frac > max_unaligned_frac or not n_words

    return FileReport(
        file_id=alignment.file_id,
        text_id=text_id or (alignment.text_id or ""),
        n_words=n_words,
        n_expected_tokens=n_expected_tokens,
        n_unk=n_unk,
        n_zero_length=n_zero,
        n_missing=n_missing,
        n_unaligned=n_unaligned,
        unaligned_frac=round(frac, 4),
        mean_word_dur_s=round(sum(durations) / n_words, 3) if n_words else 0.0,
        median_word_dur_s=round(durations[n_words // 2], 3) if n_words else 0.0,
        longest_gap_s=round(longest_gap.duration, 3) if longest_gap else 0.0,
        longest_gap_after_word=longest_gap.prev_word_idx if longest_gap else -1,
        speech_duration_s=round(alignment.speech_duration, 3),
        duration_s=round(alignment.duration, 3),
        flagged=flagged,
        reason="; ".join(reasons) if flagged else "",
    )


def format_table(reports: list[FileReport]) -> str:
    """Render the reports as a fixed-width table for the terminal."""
    header = (
        f"{'file_id':<16}{'words':>7}{'unaligned':>11}{'frac':>8}"
        f"{'mean_dur':>10}{'max_gap':>9}{'after':>7}  flag"
    )
    lines = [header, "-" * len(header)]
    for r in reports:
        lines.append(
            f"{r.file_id:<16}{r.n_words:>7}{r.n_unaligned:>11}{r.unaligned_frac:>8.3f}"
            f"{r.mean_word_dur_s:>10.3f}{r.longest_gap_s:>9.3f}"
            f"{r.longest_gap_after_word:>7}  {'FLAG' if r.flagged else 'ok'}"
        )
    return "\n".join(lines)


def write_report(path: Path, reports: list[FileReport]) -> None:
    """Write ``align_report.csv``, sorted by ``file_id``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for report in sorted(reports, key=lambda r: r.file_id):
            writer.writerow(asdict(report))
    os.replace(tmp, path)
    logger.info("%s: %d rows", path.name, len(reports))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="report only these file ids")
    p.add_argument(
        "--alignments", type=Path, default=None, help="TextGrid directory (default: from config)"
    )
    p.add_argument(
        "--max-unaligned-frac",
        type=float,
        default=None,
        help="flag threshold (default: paths.mfa.qc.max_unaligned_frac)",
    )
    p.add_argument("--out", type=Path, default=None, help="where to write align_report.csv")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Report on the stored alignments. See the module docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(apply_seeds=False)

    grid_dir = args.alignments or cfg.path("data.alignments")
    texts_dir = cfg.path("data.texts")
    threshold = (
        args.max_unaligned_frac
        if args.max_unaligned_frac is not None
        else float(cfg.paths["mfa"]["qc"]["max_unaligned_frac"])
    )
    out_path = args.out or (cfg.path("data.qc_dir") / "align_report.csv")

    grids = sorted(grid_dir.glob("*.TextGrid")) if grid_dir.is_dir() else []
    if args.only:
        wanted = set(args.only)
        grids = [g for g in grids if g.stem in wanted]
    if not grids:
        logger.error("no TextGrids under %s - run scripts/align_clean.py first", grid_dir)
        return 2

    text_ids = {row.file_id: row.text_id for row in read_manifest(cfg.path("data.manifest"))}

    reports: list[FileReport] = []
    errors: list[str] = []
    for grid in grids:
        file_id = grid.stem
        text_id = text_ids.get(file_id, "")
        n_expected = 0
        if text_id:
            try:
                transcript, _ = transcript_for(file_id, text_id, texts_dir)
                n_expected = len(normalise_for_alignment(transcript).lab_tokens)
            except MissingTranscript:
                logger.warning("%s: no transcript found; missing-token count unavailable", file_id)
        try:
            alignment = read_textgrid(grid, file_id=file_id, text_id=text_id or None)
        except (ValueError, OSError) as exc:
            errors.append(f"{file_id}: unreadable TextGrid: {exc}")
            continue
        reports.append(
            summarise(
                alignment,
                n_expected_tokens=n_expected,
                max_unaligned_frac=threshold,
                text_id=text_id,
            )
        )

    write_report(out_path, reports)
    print(format_table(sorted(reports, key=lambda r: r.file_id)))

    flagged = [r for r in reports if r.flagged]
    print(
        f"\n{len(reports)} file(s) reported, {len(flagged)} flagged "
        f"(threshold: >{threshold:.1%} unaligned words)"
    )
    if flagged:
        print("\nFLAGGED - fix each with a transcript override or a dictionary entry:\n")
        for r in flagged:
            print(f"  - {r.file_id}: {r.reason}")
        print("")
    for error in errors:
        print(f"  - {error}", file=sys.stderr)

    return 1 if (flagged or errors) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
