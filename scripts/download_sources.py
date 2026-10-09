#!/usr/bin/env python
"""Download the dataset sources listed in ``dataset/SOURCES.yaml`` (spec 4.2).

This is step 1 of ``make data``. It does three things and nothing else:

1. **Vets licences before touching the network.** Any entry whose licence is
   empty, still a ``TODO(human)``, or names ND / NoDerivatives / TED is refused
   (spec 4.1). Rostrum publishes edited versions of every clean reading, so a
   No-Derivatives licence is not a licence we can work with.
2. **Downloads** each ready file into ``<audio_raw>/<source_id>/`` with retries
   and resumable-safe atomic writes, and SHA-256s the result.
3. **Registers** every downloaded file in ``dataset/LICENSES.csv`` and
   ``dataset/MANIFEST.csv``, and refreshes ``dataset/checksums.sha256``.

Decoding, trimming and loudness normalisation are *not* done here - that is
preprocessing (spec 4.6), which reads the excerpt bounds back out of this file.

Typical use, with URLs not yet filled in::

    $ python scripts/download_sources.py
    ... prints exactly which page to open and which field to paste where ...
    exit 2

Once at least one URL is filled::

    python scripts/download_sources.py                # every ready file
    python scripts/download_sources.py --source S2    # one source
    python scripts/download_sources.py --dry-run      # report, download nothing
    python scripts/download_sources.py --force        # re-download existing files

Exit codes:
    0  every declared file is present (or was skipped as already downloaded)
    1  a download failed, or a licence was refused
    2  nothing to do: fields still hold TODO(human) placeholders
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Allow `python scripts/download_sources.py` without installing the package.
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
from rostrum.io.hashing import sha256_file

logger = logging.getLogger("download_sources")

USER_AGENT = "rostrum-dataset-builder/0.1 (research; contact via repository README)"
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".mp4", ".aac")
DEFAULT_SUFFIX = ".bin"
RETRY_WAITS = (1.0, 3.0, 8.0)
"""Backoff between attempts; three retries after the first try."""


class DownloadError(RuntimeError):
    """A file could not be fetched after every retry."""


@dataclass
class Outcome:
    """What happened to one declared file."""

    file_id: str
    source_id: str
    status: str  # downloaded | cached | todo | refused | failed
    path: Path | None = None
    sha256: str = ""
    detail: str = ""


# --------------------------------------------------------------------------- #
# Downloading
# --------------------------------------------------------------------------- #


def guess_suffix(url: str) -> str:
    """Return the audio suffix implied by a URL, or :data:`DEFAULT_SUFFIX`.

    The raw file keeps the server's container; preprocessing is what produces a
    16 kHz WAV, so guessing wrong here costs nothing but a filename.

    Example:
        >>> guess_suffix("https://x/Roosevelt_Pearl_Harbor.ogg?download=1")
        '.ogg'
        >>> guess_suffix("https://x/stream")
        '.bin'
    """
    name = unquote(urlsplit(url).path).lower()
    for suffix in AUDIO_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return DEFAULT_SUFFIX


def fetch(url: str, dest: Path, *, timeout: float = 60.0, retries: int = len(RETRY_WAITS)) -> int:
    """Download ``url`` to ``dest`` atomically, with retries. Return byte count.

    Written to ``dest.part`` and renamed on success, so an interrupted run never
    leaves a truncated file that a later run would hash and trust.

    Args:
        url: Direct link to the asset.
        dest: Destination path; parent directories are created.
        timeout: Per-attempt socket timeout in seconds.
        retries: Attempts after the first.

    Raises:
        DownloadError: Every attempt failed.
    """
    import time

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = 0
                with part.open("wb") as fh:
                    while chunk := response.read(1 << 16):
                        fh.write(chunk)
                        total += len(chunk)
            if total == 0:
                raise DownloadError(f"{url} returned an empty body")
            os.replace(part, dest)
            return total
        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last = exc
            part.unlink(missing_ok=True)
            if attempt < retries:
                wait = RETRY_WAITS[attempt]
                logger.warning(
                    "attempt %d/%d failed for %s (%s); retrying in %.0fs",
                    attempt + 1,
                    retries + 1,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)
    raise DownloadError(f"could not download {url}: {last}")


# --------------------------------------------------------------------------- #
# Per-source handling
# --------------------------------------------------------------------------- #


def handle_file(
    source: Source,
    sf: SourceFile,
    raw_root: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Outcome:
    """Download (or skip) one declared file and report what happened."""
    if not sf.ready:
        return Outcome(
            sf.file_id,
            source.id,
            "todo",
            detail=f"download_url is {TODO_SENTINEL}",
        )

    source_dir = raw_root / source.raw_dirname
    dest = source_dir / f"{sf.file_id}{guess_suffix(sf.download_url)}"
    existing = next(
        (p for p in sorted(source_dir.glob(f"{sf.file_id}.*")) if p.is_file()),
        None,
    )
    if existing is not None and not force:
        return Outcome(sf.file_id, source.id, "cached", path=existing, sha256=sha256_file(existing))

    if dry_run:
        return Outcome(sf.file_id, source.id, "todo", detail="dry run: would download")

    try:
        size = fetch(sf.download_url, dest)
    except DownloadError as exc:
        return Outcome(sf.file_id, source.id, "failed", detail=str(exc))
    digest = sha256_file(dest)
    logger.info("%s: %s (%.1f MB) sha256=%s", sf.file_id, dest.name, size / 1e6, digest[:12])
    return Outcome(sf.file_id, source.id, "downloaded", path=dest, sha256=digest)


def rows_for(
    source: Source, sf: SourceFile, outcome: Outcome, retrieved_on: str
) -> tuple[LicenseRow, ManifestRow]:
    """Build the ``LICENSES.csv`` and ``MANIFEST.csv`` rows for one file.

    ``duration_s`` and ``split_fold`` stay empty: the duration is only known
    after trimming and normalisation, and the fold after the LOTO split is
    built (spec 7.3). Both are filled by later stages, which upsert the row.
    """
    licence_row = LicenseRow(
        file_id=sf.file_id,
        title=source.title,
        speaker=sf.reader if sf.reader and sf.reader != TODO_SENTINEL else "unknown",
        source_url=source.page_url,
        licence=" ".join(source.licence.split()),
        retrieved_on=retrieved_on,
        sha256=outcome.sha256,
    )
    manifest_row = ManifestRow(
        file_id=sf.file_id,
        text_id=source.text_id,
        tier=source.tier,
        origin="clean",
    )
    return licence_row, manifest_row


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report_todos(sources: list[Source], outcomes: list[Outcome]) -> list[str]:
    """Return the human-readable to-do list: what to open and what to paste.

    Driven by the YAML rather than by the outcomes, so a source stopped at the
    licence gate still gets its missing audio links listed - otherwise a human
    would have to fill the licence, re-run, and only then learn that the URLs
    are also missing.
    """
    lines: list[str] = []
    done = {o.file_id for o in outcomes if o.status in {"downloaded", "cached"}}
    for source in sources:
        source_todo = source.todo
        pending = [f for f in source.files if f.todo and f.file_id not in done]
        if not source_todo and not pending:
            continue
        lines.append(f"\n  {source.id} - {source.title}")
        lines.append(f"    page: {source.page_url}")
        if source_todo:
            lines.append(
                f"    [ ] fill source field(s): {', '.join(source_todo)} "
                "(copy the licence statement verbatim)"
            )
        lines.append(
            f"    [ ] save the licence statement to dataset/provenance/{source.provenance}"
        )
        if pending:
            lines.append(f"    [ ] fill {len(pending)} file entr(ies):")
            for sf in pending:
                lines.append(f"          {sf.file_id}: {', '.join(sf.todo)}")
    return lines


def summarise(outcomes: list[Outcome]) -> dict[str, int]:
    """Count outcomes by status."""
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sources", type=Path, default=None, help="path to SOURCES.yaml")
    p.add_argument(
        "--source",
        action="append",
        dest="only",
        metavar="ID",
        help="restrict to this source id (repeatable)",
    )
    p.add_argument("--dry-run", action="store_true", help="report only; download nothing")
    p.add_argument("--force", action="store_true", help="re-download files already present")
    p.add_argument(
        "--retrieved-on",
        default=None,
        help="ISO date for LICENSES.csv; default: SOURCES.yaml retrieved_on, else today",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the downloader. See the module docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # No random draws happen here, so skip torch seeding: importing torch just to
    # read a YAML file fails on machines where its DLLs are blocked.
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

    retrieved_on = args.retrieved_on or _retrieved_on(sources_path)

    # ---- licence gate, before any network access (spec 4.1) ---------------
    allowed: list[Source] = []
    gated: list[Outcome] = []
    for source in sources:
        ok, reason = licence_verdict(source.licence)
        if ok:
            logger.debug("%s: %s", source.id, reason)
            allowed.append(source)
            continue
        # A licence nobody has filled in yet is a to-do, not a policy violation:
        # it must block the download, but it must not fail the build the way a
        # genuine ND licence does.
        unfilled = TODO_SENTINEL in source.licence
        status = "todo" if unfilled else "refused"
        level = logger.warning if (unfilled or source.optional) else logger.error
        level(
            "%s %s (%s): %s",
            "SKIPPED" if unfilled else "REFUSED",
            source.id,
            source.page_url,
            reason,
        )
        gated.extend(Outcome(f.file_id, source.id, status, detail=reason) for f in source.files)

    raw_root = cfg.path("data.audio_raw")
    outcomes: list[Outcome] = list(gated)
    licence_rows: list[LicenseRow] = []
    manifest_rows: list[ManifestRow] = []

    for source in allowed:
        for sf in source.files:
            outcome = handle_file(source, sf, raw_root, dry_run=args.dry_run, force=args.force)
            outcomes.append(outcome)
            if outcome.status in {"downloaded", "cached"}:
                lic, man = rows_for(source, sf, outcome, retrieved_on)
                licence_rows.append(lic)
                manifest_rows.append(man)

    # ---- registers --------------------------------------------------------
    if licence_rows and not args.dry_run:
        n = append_licenses(cfg.path("data.licenses"), licence_rows)
        logger.info("LICENSES.csv: %d rows", n)
        append_manifest(cfg.path("data.manifest"), manifest_rows)
        _refresh_checksums(cfg.path("data.dir"), raw_root, cfg.path("data.checksums"))

    # ---- report -----------------------------------------------------------
    counts = summarise(outcomes)
    logger.info(
        "done: %s",
        ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing declared",
    )

    todo_lines = report_todos(sources, outcomes)
    if todo_lines:
        print("\nHUMAN STEPS REQUIRED before these files can be downloaded:", file=sys.stderr)
        print("\n".join(todo_lines), file=sys.stderr)
        print(
            f"\n  Edit {sources_path} - replace every {TODO_SENTINEL} with the real value,\n"
            "  then re-run: python scripts/download_sources.py\n",
            file=sys.stderr,
        )

    if counts.get("failed") or any(
        o.status == "refused" and not _is_optional(sources, o.source_id) for o in outcomes
    ):
        return 1
    if counts.get("downloaded", 0) + counts.get("cached", 0) == 0:
        return 2
    return 0


def _is_optional(sources: list[Source], source_id: str) -> bool:
    """Whether ``source_id`` is marked optional in ``SOURCES.yaml``."""
    return any(s.id == source_id and s.optional for s in sources)


def _retrieved_on(sources_path: Path) -> str:
    """Read ``retrieved_on`` from the YAML, falling back to today's date."""
    import datetime

    import yaml

    with Path(sources_path).open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
    value = str(document.get("retrieved_on") or "").strip()
    if value and not value.startswith(TODO_SENTINEL):
        return value
    today = datetime.date.today().isoformat()
    logger.warning("retrieved_on is unset in %s; using today (%s)", sources_path, today)
    return today


def _refresh_checksums(dataset_dir: Path, raw_root: Path, checksums_path: Path) -> None:
    """Rewrite ``checksums.sha256`` for everything currently under ``raw_root``."""
    digests: dict[str, str] = {}
    for path in sorted(raw_root.rglob("*")):
        if path.is_file() and not path.name.endswith((".part", ".tmp")):
            digests[path.relative_to(dataset_dir).as_posix()] = sha256_file(path)
    if digests:
        write_checksums(checksums_path, digests)
        logger.info("checksums.sha256: %d files", len(digests))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
