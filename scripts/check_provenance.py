#!/usr/bin/env python
"""Fail if any audio in ``dataset/`` is missing licence provenance (spec 4.1 rule 4).

Run by CI and by ``make data``. It answers one question: could a reviewer, for
every audio file we ship, find the page that grants us the right to publish an
edited version of it?

Three checks per audio file:

1. **A LICENSES.csv row.** Clean, raw and human-recorded files need a row of
   their own. An injected file inherits from its source: its ``file_id`` starts
   with the source ``file_id`` (spec 7.1 naming, e.g. ``s1_lv07_rushing_L3``
   from ``s1_lv07``), so the parent row is what is required.
2. **A permissive licence** in that row - re-checked here, not just at download
   time, so hand-edited rows cannot smuggle in an ND licence.
3. **A provenance artefact** under ``dataset/provenance/``: the screenshot or
   PDF named by the owning source in ``SOURCES.yaml``. Team recordings (tier
   ``bronze``, which are covered by ``dataset/CONSENT.md``) are satisfied by
   that file instead.

An empty dataset passes - there is nothing unlicensed in it.

Usage::

    python scripts/check_provenance.py
    python scripts/check_provenance.py --dataset dataset_sample --strict-sample

Exit codes:
    0  every audio file has a licence row, a permissive licence and provenance
    1  at least one file does not
    2  the dataset directory or SOURCES.yaml could not be read
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/check_provenance.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.config import load_config
from rostrum.data.manifest import LicenseRow, read_licenses
from rostrum.data.sources import Source, SourcesError, licence_verdict, load_sources

logger = logging.getLogger("check_provenance")

AUDIO_SUFFIXES = frozenset(
    {".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".mp4", ".aac"}
)
"""Extensions treated as audio. Everything else under the dataset is ignored."""

IGNORED_PARTS = frozenset({"provenance", "qc", "texts", "alignments", "labels", "splits"})
"""Dataset subdirectories that never hold audio needing its own licence row."""


@dataclass(frozen=True)
class Problem:
    """One provenance failure."""

    path: str
    reason: str

    def __str__(self) -> str:
        return f"{self.path}: {self.reason}"


def find_audio(dataset_dir: Path) -> list[Path]:
    """Return every audio file under ``dataset_dir``, sorted, ignoring sidecars."""
    found: list[Path] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        rel_parts = set(path.relative_to(dataset_dir).parts[:-1])
        if rel_parts & IGNORED_PARTS:
            continue
        found.append(path)
    return found


def resolve_row(file_id: str, rows: dict[str, LicenseRow]) -> LicenseRow | None:
    """Find the licence row covering ``file_id``.

    An exact match wins. Otherwise the longest row ``file_id`` that ``file_id``
    extends with ``_`` is used, which is how a derivative (``s1_lv07_rushing_L3``)
    inherits from its clean source (``s1_lv07``). Longest-prefix rather than any
    prefix, so ``s1_lv1`` cannot accidentally claim ``s1_lv16``'s derivatives.
    """
    if file_id in rows:
        return rows[file_id]
    candidates = [k for k in rows if file_id.startswith(f"{k}_")]
    if not candidates:
        return None
    return rows[max(candidates, key=len)]


def owning_source(file_id: str, sources: list[Source]) -> Source | None:
    """Return the :class:`Source` that declares ``file_id`` or its parent."""
    best: tuple[int, Source] | None = None
    for source in sources:
        for sf in source.files:
            if file_id == sf.file_id or file_id.startswith(f"{sf.file_id}_"):
                length = len(sf.file_id)
                if best is None or length > best[0]:
                    best = (length, source)
    return best[1] if best else None


def check(
    dataset_dir: Path,
    licenses_path: Path,
    sources_path: Path,
    provenance_dir: Path,
    consent_path: Path,
) -> list[Problem]:
    """Return every provenance problem found; empty means the dataset is clean."""
    problems: list[Problem] = []

    audio = find_audio(dataset_dir)
    if not audio:
        logger.info("no audio under %s - nothing to verify", dataset_dir)
        return problems

    rows = {r.file_id: r for r in read_licenses(licenses_path)}
    sources: list[Source] = []
    if sources_path.is_file():
        sources = list(load_sources(sources_path))
    else:
        logger.warning("%s not found; provenance filenames cannot be checked", sources_path)

    for path in audio:
        rel = path.relative_to(dataset_dir).as_posix()
        file_id = path.stem

        row = resolve_row(file_id, rows)
        if row is None:
            problems.append(
                Problem(rel, f"no LICENSES.csv row for file_id {file_id!r} or any parent of it")
            )
            continue

        allowed, reason = licence_verdict(row.licence)
        if not allowed:
            problems.append(Problem(rel, f"LICENSES.csv licence rejected: {reason}"))
        if not row.source_url.strip():
            problems.append(Problem(rel, "LICENSES.csv row has no source_url"))
        if not row.sha256.strip():
            problems.append(Problem(rel, "LICENSES.csv row has no sha256"))

        source = owning_source(file_id, sources)
        if source is None:
            # Not declared in SOURCES.yaml: a team recording. CONSENT.md is its
            # provenance (spec 4.4).
            if not consent_path.is_file():
                problems.append(
                    Problem(
                        rel,
                        f"not declared in SOURCES.yaml and {consent_path.name} is missing, "
                        "so nothing records the right to publish it",
                    )
                )
            continue
        artefact = provenance_dir / source.provenance
        if not artefact.is_file():
            problems.append(
                Problem(
                    rel,
                    f"missing provenance artefact {artefact.relative_to(dataset_dir).as_posix()} "
                    f"(save the licence statement from {source.page_url})",
                )
            )

    return problems


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="dataset directory to check (default: paths.data.dir)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the provenance check. See the module docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # No random draws happen here, so skip torch seeding (see download_sources.py).
    cfg = load_config(apply_seeds=False)
    dataset_dir = args.dataset or cfg.path("data.dir")
    if not dataset_dir.is_dir():
        logger.error("not a directory: %s", dataset_dir)
        return 2

    licenses_path = (
        cfg.path("data.licenses")
        if args.dataset is None
        else dataset_dir / cfg.path("data.licenses").name
    )
    provenance_dir = (
        cfg.path("data.provenance")
        if args.dataset is None
        else dataset_dir / cfg.path("data.provenance").name
    )
    consent_path = (
        cfg.path("data.consent")
        if args.dataset is None
        else dataset_dir / cfg.path("data.consent").name
    )
    sources_path = dataset_dir / "SOURCES.yaml"

    try:
        problems = check(dataset_dir, licenses_path, sources_path, provenance_dir, consent_path)
    except (SourcesError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    if problems:
        print(f"\nPROVENANCE CHECK FAILED - {len(problems)} problem(s):\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nSpec 4.1 rule 4: every audio file needs a LICENSES.csv row and a saved\n"
            "licence statement under dataset/provenance/.\n",
            file=sys.stderr,
        )
        return 1

    logger.info("provenance OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
