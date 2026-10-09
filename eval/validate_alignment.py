"""MFA word-boundary accuracy against hand labels (spec 8, 14.1 "Components").

Every timestamp Rostrum reports is anchored to an MFA word boundary, so the
grounding metrics can never be better than the aligner underneath them. This
check reads boundaries labelled by hand in Praat and reports, for onsets and
offsets separately and pooled:

* **MAE** in milliseconds, and the **P95** - the tail matters more than the
  mean, because one 300 ms error puts a flaw region on the wrong word;
* **bias**, the signed mean, which says whether MFA systematically places
  boundaries early or late;
* the percentage of boundaries within **20 ms** and within **50 ms**. 50 ms is
  the threshold the injection QC already uses to reject a re-aligned file
  (CLAUDE.md invariant 9), so the two numbers are directly comparable.

Input
    ``dataset/qc/hand_boundaries.csv`` with columns
    ``file_id,word_idx,true_onset,true_offset`` and an optional ``word`` column.
    Times are in seconds from the start of the **processed** WAV - the same
    16 kHz, loudness-normalised file the aligner saw, not the original download.
    An optional ``word`` column is checked against the alignment and a mismatch
    is reported rather than silently averaged in: it means the hand labels and
    the alignment disagree about which word is which, and no error statistic
    computed across that offset would mean anything.

    ``--write-template`` writes the header so the file can be filled in Praat.
    The spec asks for 50 boundaries across 5 files.

Outputs
    ``results/components.json`` section ``alignment``;
    ``docs/figures/components_alignment_error_hist.png``;
    ``docs/figures/components_alignment_error_cdf.png``.

Exit codes:
    0  the section was written with measured numbers
    2  the hand labels or the alignments do not exist yet; the section is
       written as pending with the reason
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import components as C
from rostrum.align.mfa import read_textgrid
from rostrum.config import Config, load_config
from rostrum.types import Alignment

logger = logging.getLogger("validate_alignment")

HAND_FIELDS: tuple[str, ...] = ("file_id", "word_idx", "true_onset", "true_offset", "word")
"""Columns of ``hand_boundaries.csv``; ``word`` is optional but recommended."""

TARGET_BOUNDARIES = 50
"""Spec 8: hand-label 50 word boundaries across 5 files."""

TARGET_FILES = 5
"""Spec 8: across five files."""


@dataclass
class HandLabel:
    """One hand-labelled word interval."""

    file_id: str
    word_idx: int
    true_onset: float
    true_offset: float
    word: str = ""


@dataclass
class BoundaryError:
    """The aligner's error on one hand-labelled word, in seconds (signed)."""

    file_id: str
    word_idx: int
    word: str
    onset_error: float
    offset_error: float
    text_matches: bool


# --------------------------------------------------------------------------- #
# Reading the hand labels
# --------------------------------------------------------------------------- #


def write_template(path: Path) -> Path:
    """Write an empty ``hand_boundaries.csv`` with the expected header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(HAND_FIELDS)
    logger.info(
        "wrote the header to %s - label %d boundaries across %d files in Praat, "
        "times in seconds from the start of the processed WAV",
        path,
        TARGET_BOUNDARIES,
        TARGET_FILES,
    )
    return path


def read_hand_labels(path: Path) -> list[HandLabel]:
    """Read the hand-labelled boundaries, skipping blank and malformed rows.

    Raises:
        ValueError: A required column is missing from the header.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"file_id", "word_idx", "true_onset", "true_offset"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"{path} is missing column(s) {sorted(missing)}")

        out: list[HandLabel] = []
        for line, row in enumerate(reader, start=2):
            if not (row.get("file_id") or "").strip():
                continue
            try:
                out.append(
                    HandLabel(
                        file_id=row["file_id"].strip(),
                        word_idx=int(row["word_idx"]),
                        true_onset=float(row["true_onset"]),
                        true_offset=float(row["true_offset"]),
                        word=(row.get("word") or "").strip(),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning("%s line %d is malformed (%s); skipped", path, line, exc)
    return out


# --------------------------------------------------------------------------- #
# Comparing
# --------------------------------------------------------------------------- #


def load_alignments(cfg: Config, file_ids: set[str], *, sample: bool) -> dict[str, Alignment]:
    """Read the TextGrid of each requested recording that has one."""
    directory = cfg.path("sample.alignments") if sample else cfg.path("data.alignments")
    out: dict[str, Alignment] = {}
    for file_id in sorted(file_ids):
        path = directory / f"{file_id}.TextGrid"
        if not path.is_file():
            logger.warning("no alignment for %s at %s", file_id, path)
            continue
        out[file_id] = read_textgrid(path, file_id=file_id)
    return out


def compare(labels: list[HandLabel], alignments: dict[str, Alignment]) -> list[BoundaryError]:
    """Error of each hand-labelled boundary, skipping words with no alignment."""
    errors: list[BoundaryError] = []
    for label in labels:
        alignment = alignments.get(label.file_id)
        if alignment is None:
            continue
        by_idx = {w.idx: w for w in alignment.words}
        word = by_idx.get(label.word_idx)
        if word is None:
            logger.warning(
                "%s has no word %d (alignment has %d words); skipped",
                label.file_id,
                label.word_idx,
                len(alignment.words),
            )
            continue
        matches = (not label.word) or word.text.lower() == label.word.lower()
        if not matches:
            logger.warning(
                "%s word %d is %r in the alignment but %r in the hand labels",
                label.file_id,
                label.word_idx,
                word.text,
                label.word,
            )
        errors.append(
            BoundaryError(
                file_id=label.file_id,
                word_idx=label.word_idx,
                word=word.text,
                onset_error=float(word.t_start) - label.true_onset,
                offset_error=float(word.t_end) - label.true_offset,
                text_matches=matches,
            )
        )
    return errors


def summarise(errors: list[BoundaryError]) -> dict[str, object]:
    """Onset, offset and pooled error statistics, plus per-file breakdown."""
    onsets = np.array([e.onset_error for e in errors], dtype=np.float64)
    offsets = np.array([e.offset_error for e in errors], dtype=np.float64)
    per_file: dict[str, dict[str, float | int]] = {}
    grouped: dict[str, list[float]] = defaultdict(list)
    for error in errors:
        grouped[error.file_id] += [error.onset_error, error.offset_error]
    for file_id, values in sorted(grouped.items()):
        per_file[file_id] = C.summarise_errors_ms(values)
    return {
        "onset": C.summarise_errors_ms(onsets),
        "offset": C.summarise_errors_ms(offsets),
        "pooled": C.summarise_errors_ms(np.concatenate([onsets, offsets])),
        "per_file": per_file,
        "n_boundaries": len(errors),
        "n_files": len(grouped),
        "n_text_mismatches": sum(1 for e in errors if not e.text_matches),
        "meets_spec_sample_size": len(errors) >= TARGET_BOUNDARIES and len(grouped) >= TARGET_FILES,
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_histogram(cfg: Config, errors: list[BoundaryError]) -> Path:
    """Signed onset and offset errors in milliseconds."""
    figure, axes = C.new_figure(size=(7.0, 4.0))
    onsets = np.array([e.onset_error for e in errors]) * 1000.0
    offsets = np.array([e.offset_error for e in errors]) * 1000.0
    span = max(50.0, float(np.abs(np.concatenate([onsets, offsets])).max()))
    bins = np.linspace(-span, span, 41)
    axes.hist(onsets, bins=bins, alpha=0.65, label="onset", color="#4878a8")
    axes.hist(offsets, bins=bins, alpha=0.65, label="offset", color="#d08c38")
    axes.axvline(0.0, color="#444444", linewidth=0.8)
    axes.set_xlabel("MFA minus hand label (ms); negative = MFA is early")
    axes.set_ylabel("boundaries")
    axes.set_title("MFA word-boundary error against hand labels")
    axes.legend(frameon=False)
    return C.save_figure(figure, C.figure_path(cfg, "components_alignment_error_hist.png"))


def plot_cdf(cfg: Config, errors: list[BoundaryError]) -> Path:
    """Cumulative distribution of absolute error, with the 20 and 50 ms marks."""
    figure, axes = C.new_figure(size=(6.0, 4.0))
    absolute = np.sort(
        np.abs(np.concatenate([[e.onset_error for e in errors], [e.offset_error for e in errors]]))
        * 1000.0
    )
    fraction = np.arange(1, absolute.size + 1) / absolute.size * 100.0
    axes.step(absolute, fraction, where="post", color="#4878a8")
    for mark in (20.0, 50.0):
        axes.axvline(mark, color="#b4422a", linestyle="--", linewidth=1.0)
        reached = float((absolute <= mark + C.TOLERANCE_EPSILON_MS).mean() * 100.0)
        axes.text(mark, 5.0, f" {reached:.0f}% <= {mark:.0f} ms", fontsize=9, color="#b4422a")
    axes.set_xlabel("absolute boundary error (ms)")
    axes.set_ylabel("boundaries within (%)")
    axes.set_ylim(0, 102)
    axes.set_title("MFA boundary error, cumulative")
    return C.save_figure(figure, C.figure_path(cfg, "components_alignment_error_cdf.png"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Measure MFA boundary accuracy and write the ``alignment`` section."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="read alignments from the sample")
    parser.add_argument("--hand-boundaries", type=Path, help="override the CSV path")
    parser.add_argument(
        "--write-template",
        action="store_true",
        help="write the CSV header for hand labelling and exit",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    path = args.hand_boundaries or cfg.path("data.qc_hand_boundaries")

    if args.write_template:
        write_template(path)
        return 0

    def pending(reason: str) -> int:
        C.write_section(
            cfg,
            "alignment",
            {
                "status": C.STATUS_PENDING,
                "reason": reason,
                "expects": {
                    "path": str(path),
                    "columns": list(HAND_FIELDS),
                    "n_boundaries": TARGET_BOUNDARIES,
                    "n_files": TARGET_FILES,
                },
            },
        )
        logger.error("%s", reason)
        return 2

    if not path.is_file():
        return pending(
            f"no hand-labelled boundaries at {path}; label {TARGET_BOUNDARIES} boundaries "
            f"across {TARGET_FILES} files in Praat "
            "(`python -m eval.validate_alignment --write-template` writes the header)"
        )

    labels = read_hand_labels(path)
    if not labels:
        return pending(f"{path} has a header but no rows yet")

    alignments = load_alignments(cfg, {label.file_id for label in labels}, sample=args.sample)
    if not alignments:
        return pending(
            "no alignments found for the hand-labelled files; run `make align` "
            "(MFA lives in the `mfa` conda env)"
        )

    errors = compare(labels, alignments)
    if not errors:
        return pending(
            "no hand-labelled word could be matched to an aligned word; check that "
            "word_idx is zero-based and refers to the same transcript"
        )

    summary = summarise(errors)
    figures: list[str] = []
    if not args.no_figures:
        figures.append(plot_histogram(cfg, errors).name)
        figures.append(plot_cdf(cfg, errors).name)

    C.write_section(
        cfg,
        "alignment",
        {
            "status": C.STATUS_OK,
            "metric": "MFA word-boundary error against hand labels: MAE, P95 and "
            "percentage within 20 and 50 ms",
            "hand_boundaries": str(path),
            **summary,
            "per_boundary": [vars(e) for e in errors],
            "figures": figures,
        },
    )

    pooled = summary["pooled"]
    onset = summary["onset"]
    offset = summary["offset"]
    print()
    print(f"{'':<10}{'n':>6}{'MAE ms':>9}{'P95 ms':>9}{'bias ms':>9}{'<=20ms':>9}{'<=50ms':>9}")
    for name, stats in (("onset", onset), ("offset", offset), ("pooled", pooled)):
        print(
            f"{name:<10}{stats['n']:>6}{stats['mae_ms']:>9.1f}{stats['p95_ms']:>9.1f}"
            f"{stats['bias_ms']:>9.1f}{stats['within_20ms_pct']:>8.1f}%{stats['within_50ms_pct']:>8.1f}%"
        )
    print(
        f"\n{summary['n_boundaries']} boundaries across {summary['n_files']} files"
        f" (spec asks for {TARGET_BOUNDARIES} across {TARGET_FILES}:"
        f" {'met' if summary['meets_spec_sample_size'] else 'NOT met'})"
    )
    if summary["n_text_mismatches"]:
        print(f"WARNING: {summary['n_text_mismatches']} hand label(s) name a different word")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
