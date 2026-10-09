"""``LICENSES.csv`` and ``MANIFEST.csv`` read/write (spec 7.4).

Both files are append-mostly registers keyed by ``file_id``. The writers here
*upsert*: re-running the downloader after fixing one URL updates that row
instead of duplicating it, and rows are written back sorted by ``file_id`` so
the CSVs have a canonical form that diffs cleanly and hashes stably (spec 16).

Column orders are fixed by the spec and are asserted by the tests:

* ``LICENSES.csv``: ``file_id,title,speaker,source_url,licence,retrieved_on,sha256``
* ``MANIFEST.csv``: ``file_id,text_id,tier,origin,label,level,duration_s,split_fold``
"""

from __future__ import annotations

import csv
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

__all__ = [
    "LICENSES_FIELDS",
    "MANIFEST_FIELDS",
    "LicenseRow",
    "ManifestRow",
    "append_licenses",
    "append_manifest",
    "clean_file_ids",
    "parent_file_id",
    "read_licenses",
    "read_manifest",
    "write_checksums",
]

logger = logging.getLogger(__name__)

LICENSES_FIELDS: tuple[str, ...] = (
    "file_id",
    "title",
    "speaker",
    "source_url",
    "licence",
    "retrieved_on",
    "sha256",
)
"""Column order of ``dataset/LICENSES.csv`` (spec 7.4)."""

MANIFEST_FIELDS: tuple[str, ...] = (
    "file_id",
    "text_id",
    "tier",
    "origin",
    "label",
    "level",
    "duration_s",
    "split_fold",
)
"""Column order of ``dataset/MANIFEST.csv`` (spec 7.4)."""


@dataclass(frozen=True)
class LicenseRow:
    """One row of ``LICENSES.csv``.

    Attributes:
        file_id: Matches the audio file stem.
        title: Title of the work or collection as given on the source page.
        speaker: Reader or orator credited on the page.
        source_url: The page the licence is stated on - not the asset URL, so a
            reviewer lands on the licence text.
        licence: Licence exactly as stated there.
        retrieved_on: ISO date of download.
        sha256: Hex digest of the file this row describes.
    """

    file_id: str
    title: str
    speaker: str
    source_url: str
    licence: str
    retrieved_on: str
    sha256: str


@dataclass(frozen=True)
class ManifestRow:
    """One row of ``MANIFEST.csv``.

    ``label`` and ``level`` are empty for clean readings; ``split_fold`` is
    filled later by the LOTO split builder (spec 7.3), so it is optional here.

    Attributes:
        file_id: Matches the audio file stem.
        text_id: The shared text, used as the envelope key.
        tier: ``gold`` | ``silver`` | ``bronze``.
        origin: ``clean`` | ``injected`` | ``human``.
        label: Flaw label, or ``""`` for a clean file.
        level: Severity ``L0``-``L4``, or ``""``.
        duration_s: Duration of the processed WAV in seconds, 3 decimals.
        split_fold: LOTO fold name, or ``""`` until the split is built.
    """

    file_id: str
    text_id: str
    tier: str
    origin: str
    label: str = ""
    level: str = ""
    duration_s: float | str = ""
    split_fold: str = ""


# --------------------------------------------------------------------------- #
# Generic CSV upsert
# --------------------------------------------------------------------------- #


def _row_to_strings(row: Mapping[str, object], columns: Sequence[str]) -> dict[str, str]:
    """Render a row as strings in ``columns`` order, dropping unknown keys."""
    unknown = set(row) - set(columns)
    if unknown:
        raise KeyError(f"unknown column(s) {sorted(unknown)}; expected {list(columns)}")
    return {c: "" if row.get(c) is None else str(row[c]) for c in columns}


def _read_csv(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    """Read a CSV, returning ``[]`` when it does not exist yet.

    Raises:
        ValueError: The existing header does not match ``columns``.
    """
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        if header != tuple(columns):
            raise ValueError(
                f"{path} has header {list(header)}, expected {list(columns)}. "
                "Fix or delete the file; the column order is fixed by spec 7.4."
            )
        return [{c: (r.get(c) or "") for c in columns} for r in reader]


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    """Write a CSV atomically with ``\\n`` line endings.

    The temp-file-then-replace dance means an interrupted run cannot leave a
    half-written register behind, and the explicit newline keeps the bytes (and
    therefore the checksum) identical on Windows and POSIX.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _upsert(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, object]],
    *,
    key: str = "file_id",
) -> int:
    """Merge ``rows`` into the CSV at ``path`` by ``key``; return rows written."""
    existing = {r[key]: r for r in _read_csv(path, columns)}
    for row in rows:
        rendered = _row_to_strings(row, columns)
        if not rendered[key]:
            raise ValueError(f"row is missing {key}: {row!r}")
        if rendered[key] in existing and existing[rendered[key]] != rendered:
            logger.info("%s: updating row %s", path.name, rendered[key])
        existing[rendered[key]] = rendered
    ordered = [existing[k] for k in sorted(existing)]
    _write_csv(path, columns, ordered)
    return len(ordered)


# --------------------------------------------------------------------------- #
# LICENSES.csv
# --------------------------------------------------------------------------- #


def read_licenses(path: Path | str) -> list[LicenseRow]:
    """Read ``LICENSES.csv``, returning ``[]`` when the file does not exist."""
    return [LicenseRow(**r) for r in _read_csv(Path(path), LICENSES_FIELDS)]  # type: ignore[arg-type]


def append_licenses(path: Path | str, rows: Iterable[LicenseRow | Mapping[str, object]]) -> int:
    """Upsert licence rows by ``file_id``; return the total row count.

    Args:
        path: Destination ``LICENSES.csv``; created with a header if absent.
        rows: :class:`LicenseRow` instances or plain mappings.

    Raises:
        ValueError: A row has no ``file_id``, or the existing header is wrong.
    """
    return _upsert(Path(path), LICENSES_FIELDS, [_as_mapping(r) for r in rows])


# --------------------------------------------------------------------------- #
# MANIFEST.csv
# --------------------------------------------------------------------------- #


def read_manifest(path: Path | str) -> list[ManifestRow]:
    """Read ``MANIFEST.csv``, returning ``[]`` when the file does not exist."""
    return [ManifestRow(**r) for r in _read_csv(Path(path), MANIFEST_FIELDS)]  # type: ignore[arg-type]


def append_manifest(path: Path | str, rows: Iterable[ManifestRow | Mapping[str, object]]) -> int:
    """Upsert manifest rows by ``file_id``; return the total row count."""
    return _upsert(Path(path), MANIFEST_FIELDS, [_as_mapping(r) for r in rows])


def clean_file_ids(rows: Iterable[ManifestRow]) -> list[str]:
    """The ``file_id`` of every clean recording, shortest first."""
    return sorted((r.file_id for r in rows if r.origin == "clean"), key=len)


def parent_file_id(file_id: str, clean_ids: Sequence[str]) -> str:
    """The clean recording a derived file came from, or ``""``.

    Injected files are named ``<clean id>_<label>_L<level>`` and human flawed
    takes ``<speaker id>_<label>_takeN``, so the parent is recovered by prefix.
    The *longest* match wins, so ``s1_lv01_rushing_L3`` resolves to ``s1_lv01``
    rather than to a shorter identifier that happens to match as well.

    This is how speaker and channel identity reach the feature tables: a derived
    file shares a microphone and a voice with its parent and with nothing else,
    which is exactly the scope the channel rule allows comparisons in
    (CLAUDE.md invariant 7).

    Args:
        file_id: The derived recording's identifier.
        clean_ids: Candidate clean identifiers, e.g. from :func:`clean_file_ids`.

    Returns:
        The parent identifier, or ``""`` when nothing matches.
    """
    matches = [c for c in clean_ids if file_id.startswith(c)]
    return max(matches, key=len) if matches else ""


def _as_mapping(row: LicenseRow | ManifestRow | Mapping[str, object]) -> Mapping[str, object]:
    """Normalise a dataclass row or mapping to a mapping."""
    if isinstance(row, Mapping):
        return row
    if any(isinstance(row, t) for t in (LicenseRow, ManifestRow)):
        return asdict(row)
    raise TypeError(f"expected a row dataclass or mapping, got {type(row).__name__}")


# --------------------------------------------------------------------------- #
# checksums.sha256
# --------------------------------------------------------------------------- #


def write_checksums(path: Path | str, digests: Mapping[str, str]) -> int:
    """Write a ``sha256sum``-format file, sorted by path; return the line count.

    Args:
        path: Destination (``dataset/checksums.sha256``).
        digests: Maps a dataset-relative path (POSIX separators) to its digest.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    lines = [f"{digests[k]}  {k}\n" for k in sorted(digests)]
    tmp.write_text("".join(lines), encoding="utf-8", newline="")
    os.replace(tmp, p)
    return len(lines)


def _field_names(cls: type) -> tuple[str, ...]:
    """Dataclass field names, used by the tests to assert column order."""
    return tuple(f.name for f in fields(cls))
