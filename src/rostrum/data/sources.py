"""Parse and vet ``dataset/SOURCES.yaml`` (spec 4.1-4.2).

This module is the only place that decides whether a source may be downloaded.
It is pure: no network, no filesystem writes, so the licence policy can be
unit-tested directly instead of through the downloader.

Two kinds of problem are reported separately, because they need different
actions from a human:

* :class:`SourceFile.todo` - a field still holds the ``TODO(human)`` sentinel.
  Someone must open the page and paste the direct audio link.
* :func:`licence_verdict` returning a refusal - the licence as stated forbids
  derivative works, so the entry must be dropped from the dataset entirely.
  Rostrum publishes *edited* versions of every clean reading, so a
  No-Derivatives licence is not a technicality we can work around (spec 4.1
  rule 2; TED is CC BY-NC-ND and explicitly forbids edits, [R8]).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "TODO_SENTINEL",
    "LicenceRefusal",
    "Source",
    "SourceFile",
    "SourcesError",
    "check_licence",
    "iter_files",
    "licence_verdict",
    "load_sources",
    "parse_sources",
    "retrieved_on",
]

TODO_SENTINEL = "TODO(human)"
"""Placeholder a human must replace before a download can run."""

# Matched against the licence text with its original capitalisation. "ND" has to
# be a standalone token: "Public Domain" and "Unrestricted" must not trip it,
# while "CC BY-NC-ND 4.0" must. Same for "TED", which would otherwise fire on
# "stated".
_FORBIDDEN_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?<![A-Za-z])ND(?![A-Za-z])", "names ND (NoDerivatives)"),
    # Covers "NoDerivatives", "NoDerivs" and "No Derivative Works".
    (r"(?i)no[\s_-]?deriv", "names NoDerivatives"),
    (r"(?<![A-Za-z])TED(?![A-Za-z])", "comes from TED, whose usage policy forbids edits"),
)

_ND_FREE_LICENCES = ("public domain", "cc0", "cc by")
"""Substrings (lower-cased) that a licence should contain to be clearly allowed."""


class SourcesError(Exception):
    """``SOURCES.yaml`` is missing, malformed, or internally inconsistent."""


class LicenceRefusal(Exception):
    """A source's licence forbids the derivative works this project produces."""


@dataclass(frozen=True)
class SourceFile:
    """One audio file to download.

    Attributes:
        file_id: Dataset-wide unique id; becomes the processed WAV stem.
        source_id: The :class:`Source` this file belongs to.
        reader: Speaker/reader name as credited on the source page.
        download_url: Direct link to the audio asset.
        excerpt_start_s: Trim start in the downloaded file, or ``None`` for its
            beginning. Applied by preprocessing, not by the downloader.
        excerpt_end_s: Trim end, or ``None`` for the end of the file.
        notes: Free text from the YAML.
    """

    file_id: str
    source_id: str
    reader: str
    download_url: str
    excerpt_start_s: float | None = None
    excerpt_end_s: float | None = None
    notes: str = ""

    @property
    def todo(self) -> tuple[str, ...]:
        """Names of fields still holding :data:`TODO_SENTINEL`."""
        pending = [
            name
            for name in ("reader", "download_url", "excerpt_start_s", "excerpt_end_s")
            if getattr(self, name) == TODO_SENTINEL
        ]
        return tuple(pending)

    @property
    def ready(self) -> bool:
        """Whether this file can be downloaded (a real URL is present)."""
        return bool(self.download_url) and self.download_url != TODO_SENTINEL


@dataclass(frozen=True)
class Source:
    """One source page: a licence statement plus the files it covers.

    Attributes:
        id: Short id (``S1``, ``S2``, ...); also the raw subdirectory name.
        text_id: The shared text, used as the envelope key.
        title: Human-readable title of the collection or item.
        tier: ``gold`` | ``silver`` | ``bronze`` (spec 4.2).
        page_url: The page the licence is stated on.
        licence: Licence text as stated on that page.
        licence_evidence_url: Optional second URL backing the licence claim.
        provenance: Filename under ``dataset/provenance/`` holding the
            screenshot or PDF of the licence statement.
        expected_files: How many files the page is expected to yield; a
            mismatch with ``len(files)`` is reported as a warning, not an error,
            because a collection can gain or lose readings.
        optional: Entry may be skipped without failing the build.
        raw_subdir: Subdirectory of ``dataset/audio/raw`` holding this source's
            files; defaults to :attr:`id`. Set it when several sources share one
            folder - the team recordings are three sources (one per text) whose
            audio all lives in ``dataset/audio/raw/team``.
        speech_date: ISO date of the original speech, for the datasheet.
        notes: Free text from the YAML.
        files: The files to download.
    """

    id: str
    text_id: str
    title: str
    tier: str
    page_url: str
    licence: str
    provenance: str
    licence_evidence_url: str | None = None
    expected_files: int | None = None
    optional: bool = False
    raw_subdir: str | None = None
    speech_date: str | None = None
    notes: str = ""
    files: tuple[SourceFile, ...] = field(default_factory=tuple)

    @property
    def raw_dirname(self) -> str:
        """Folder under ``dataset/audio/raw`` that holds this source's audio."""
        return self.raw_subdir or self.id

    @property
    def todo(self) -> tuple[str, ...]:
        """Names of source-level fields still holding :data:`TODO_SENTINEL`."""
        return tuple(
            name
            for name in ("page_url", "licence", "provenance")
            if getattr(self, name) == TODO_SENTINEL
            or str(getattr(self, name)).startswith(TODO_SENTINEL)
        )


# --------------------------------------------------------------------------- #
# Licence policy
# --------------------------------------------------------------------------- #


def licence_verdict(licence: str | None) -> tuple[bool, str]:
    """Decide whether audio under ``licence`` may be downloaded and edited.

    The check is deliberately conservative: an unrecognised licence string is
    refused rather than assumed permissive, since the cost of guessing wrong is
    republishing someone's recording without the right.

    Args:
        licence: The licence text exactly as stated on the source page.

    Returns:
        ``(allowed, reason)``. ``reason`` explains a refusal, or names the
        matched permissive licence when allowed.

    Example:
        >>> licence_verdict("Public Domain")[0]
        True
        >>> licence_verdict("CC BY-NC-ND 4.0")
        (False, 'licence names ND (NoDerivatives)')
        >>> licence_verdict("")[0]
        False
    """
    text = (licence or "").strip()
    if not text:
        return False, "licence field is empty"
    if text == TODO_SENTINEL or text.startswith(TODO_SENTINEL):
        return False, f"licence is still {TODO_SENTINEL}"

    for pattern, why in _FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            return False, f"licence {why}"

    lowered = text.lower()
    for allowed in _ND_FREE_LICENCES:
        if allowed in lowered:
            return True, f"licence states {allowed}"

    return False, (
        "licence does not state public domain, CC0 or CC BY; spec 4.1 allows no "
        f"other licence (got: {text[:80]!r})"
    )


def check_licence(source: Source) -> None:
    """Raise :class:`LicenceRefusal` if ``source`` may not be used.

    Raises:
        LicenceRefusal: The licence as stated forbids derivatives, is empty, or
            is not one of the permitted licences.
    """
    allowed, reason = licence_verdict(source.licence)
    if not allowed:
        raise LicenceRefusal(f"{source.id} ({source.page_url}): {reason}")


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _opt_float(value: Any, where: str) -> float | None:
    """Coerce an excerpt bound to float, keeping ``None`` and the TODO sentinel."""
    if value is None or value == TODO_SENTINEL:
        return value
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SourcesError(f"{where}: expected a number, null or {TODO_SENTINEL}") from exc


def _parse_file(raw: Mapping[str, Any], source_id: str, index: int) -> SourceFile:
    """Build one :class:`SourceFile` from a YAML mapping."""
    where = f"sources[{source_id}].files[{index}]"
    if not isinstance(raw, Mapping):
        raise SourcesError(f"{where}: expected a mapping, got {type(raw).__name__}")
    file_id = str(raw.get("file_id") or "").strip()
    if not file_id:
        raise SourcesError(f"{where}: file_id is required")
    url = str(raw.get("download_url") or "").strip()
    if not url:
        raise SourcesError(f"{where}: download_url is required (use {TODO_SENTINEL} if unknown)")
    return SourceFile(
        file_id=file_id,
        source_id=source_id,
        reader=str(raw.get("reader") or "").strip(),
        download_url=url,
        excerpt_start_s=_opt_float(raw.get("excerpt_start_s"), f"{where}.excerpt_start_s"),
        excerpt_end_s=_opt_float(raw.get("excerpt_end_s"), f"{where}.excerpt_end_s"),
        notes=str(raw.get("notes") or ""),
    )


def _parse_source(raw: Mapping[str, Any], index: int) -> Source:
    """Build one :class:`Source` from a YAML mapping."""
    if not isinstance(raw, Mapping):
        raise SourcesError(f"sources[{index}]: expected a mapping, got {type(raw).__name__}")
    source_id = str(raw.get("id") or "").strip()
    if not source_id:
        raise SourcesError(f"sources[{index}]: id is required")
    for required in ("text_id", "page_url", "licence", "provenance"):
        if not str(raw.get(required) or "").strip():
            raise SourcesError(f"sources[{source_id}]: {required} is required")
    files = tuple(_parse_file(f, source_id, i) for i, f in enumerate(raw.get("files") or ()))
    seen: set[str] = set()
    for f in files:
        if f.file_id in seen:
            raise SourcesError(f"sources[{source_id}]: duplicate file_id {f.file_id!r}")
        seen.add(f.file_id)
    return Source(
        id=source_id,
        text_id=str(raw["text_id"]).strip(),
        title=str(raw.get("title") or source_id),
        tier=str(raw.get("tier") or "").strip().lower(),
        page_url=str(raw["page_url"]).strip(),
        licence=str(raw["licence"]).strip(),
        provenance=str(raw["provenance"]).strip(),
        licence_evidence_url=(
            str(raw["licence_evidence_url"]).strip() if raw.get("licence_evidence_url") else None
        ),
        expected_files=(
            int(raw["expected_files"]) if raw.get("expected_files") is not None else None
        ),
        optional=bool(raw.get("optional", False)),
        raw_subdir=(str(raw["raw_subdir"]).strip() or None) if raw.get("raw_subdir") else None,
        speech_date=str(raw["speech_date"]) if raw.get("speech_date") else None,
        notes=str(raw.get("notes") or ""),
        files=files,
    )


def parse_sources(document: Mapping[str, Any]) -> tuple[Source, ...]:
    """Parse an already-loaded ``SOURCES.yaml`` document.

    Args:
        document: The top-level mapping, with a ``sources`` list.

    Returns:
        The sources in file order.

    Raises:
        SourcesError: The document is malformed, or two sources or two files
            share an id.
    """
    if not isinstance(document, Mapping):
        raise SourcesError("SOURCES.yaml must contain a mapping at the top level")
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise SourcesError("SOURCES.yaml needs a top-level `sources:` list")

    sources = tuple(_parse_source(raw, i) for i, raw in enumerate(raw_sources))

    seen_sources: set[str] = set()
    seen_files: dict[str, str] = {}
    for source in sources:
        if source.id in seen_sources:
            raise SourcesError(f"duplicate source id {source.id!r}")
        seen_sources.add(source.id)
        for f in source.files:
            if f.file_id in seen_files:
                raise SourcesError(
                    f"file_id {f.file_id!r} is used by both {seen_files[f.file_id]} "
                    f"and {source.id}"
                )
            seen_files[f.file_id] = source.id
    return sources


def load_sources(path: Path | str) -> tuple[Source, ...]:
    """Load and parse ``SOURCES.yaml``.

    Args:
        path: Path to the YAML file.

    Raises:
        SourcesError: The file is missing or malformed.
    """
    p = Path(path)
    if not p.is_file():
        raise SourcesError(f"missing sources file: {p}")
    with p.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    if document is None:
        raise SourcesError(f"{p} is empty")
    return parse_sources(document)


def iter_files(sources: Sequence[Source]) -> Iterator[tuple[Source, SourceFile]]:
    """Yield every ``(source, file)`` pair in declaration order."""
    for source in sources:
        for f in source.files:
            yield source, f


def retrieved_on(document: Mapping[str, Any]) -> str | None:
    """Return the document's ``retrieved_on`` date, or ``None`` if still a TODO."""
    value = str(document.get("retrieved_on") or "").strip()
    if not value or value.startswith(TODO_SENTINEL):
        return None
    return value
