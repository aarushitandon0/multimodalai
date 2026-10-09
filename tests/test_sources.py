"""Tests for the licence filter and SOURCES.yaml parsing (spec 4.1-4.2)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from rostrum.data.sources import (
    TODO_SENTINEL,
    SourcesError,
    iter_files,
    licence_verdict,
    load_sources,
    parse_sources,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_YAML = REPO_ROOT / "dataset" / "SOURCES.yaml"


# --------------------------------------------------------------------------- #
# Licence filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "licence",
    [
        "Public Domain",
        "public domain",
        "Copyright Status: Public Domain",
        "CC0 1.0 Universal",
        "CC BY 4.0",
        "CC BY-SA 4.0",
        "Public domain. LibriVox recordings may be reproduced, distributed or modified.",
    ],
)
def test_permissive_licences_allowed(licence: str) -> None:
    allowed, reason = licence_verdict(licence)
    assert allowed, reason


@pytest.mark.parametrize(
    "licence",
    [
        "CC BY-NC-ND 4.0",
        "CC BY-ND 4.0",
        "Creative Commons Attribution-NoDerivatives 4.0",
        "Attribution-NonCommercial-NoDerivs 3.0",
        "TED Talks Usage Policy",
        "Downloaded from TED",
    ],
)
def test_no_derivatives_and_ted_refused(licence: str) -> None:
    allowed, reason = licence_verdict(licence)
    assert not allowed
    assert "ND" in reason or "NoDerivatives" in reason or "TED" in reason


@pytest.mark.parametrize("licence", ["", "   ", None, TODO_SENTINEL, f"{TODO_SENTINEL} - ask"])
def test_empty_or_todo_licence_refused(licence: str | None) -> None:
    allowed, _ = licence_verdict(licence)
    assert not allowed


@pytest.mark.parametrize(
    "licence",
    ["All rights reserved", "Copyright 2024 Example News", "Unknown", "Fair use"],
)
def test_unrecognised_licence_refused(licence: str) -> None:
    """An unrecognised licence is refused, not assumed permissive."""
    allowed, reason = licence_verdict(licence)
    assert not allowed
    assert "public domain" in reason


def test_nd_match_does_not_fire_on_ordinary_words() -> None:
    """'ND' must be a standalone token: 'Domain' and 'Unrestricted' contain 'nd'."""
    for licence in ("Public Domain", "Rights Status: Unrestricted, public domain", "CC0, no ands"):
        assert licence_verdict(licence)[0], licence


def test_ted_match_does_not_fire_on_stated() -> None:
    assert licence_verdict("Public domain as stated on the item page")[0]


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _doc(body: str) -> dict:
    return yaml.safe_load(textwrap.dedent(body))


def test_parse_minimal_source() -> None:
    sources = parse_sources(_doc("""
            sources:
              - id: S9
                text_id: t9
                title: A Title
                tier: silver
                page_url: https://example.org/page
                licence: Public Domain
                provenance: S9.pdf
                files:
                  - file_id: s9_a
                    reader: A Reader
                    download_url: https://example.org/a.mp3
                    excerpt_start_s: 1.5
                    excerpt_end_s: null
            """))
    assert len(sources) == 1
    (source,) = sources
    assert source.id == "S9"
    assert source.tier == "silver"
    assert source.todo == ()
    (sf,) = source.files
    assert sf.ready
    assert sf.todo == ()
    assert sf.excerpt_start_s == 1.5
    assert sf.excerpt_end_s is None


def test_todo_url_is_not_ready() -> None:
    sources = parse_sources(_doc(f"""
            sources:
              - id: S9
                text_id: t9
                page_url: https://example.org/page
                licence: Public Domain
                provenance: S9.pdf
                files:
                  - file_id: s9_a
                    download_url: "{TODO_SENTINEL}"
            """))
    (sf,) = sources[0].files
    assert not sf.ready
    assert "download_url" in sf.todo


def test_todo_page_url_reported_at_source_level() -> None:
    sources = parse_sources(_doc(f"""
            sources:
              - id: S9
                text_id: t9
                page_url: "{TODO_SENTINEL}"
                licence: "{TODO_SENTINEL} - verify"
                provenance: S9.pdf
                files: []
            """))
    assert set(sources[0].todo) == {"page_url", "licence"}


@pytest.mark.parametrize("missing", ["text_id", "page_url", "licence", "provenance"])
def test_missing_required_source_field_raises(missing: str) -> None:
    doc = {
        "sources": [
            {
                "id": "S9",
                "text_id": "t9",
                "page_url": "https://example.org",
                "licence": "Public Domain",
                "provenance": "S9.pdf",
                "files": [],
            }
        ]
    }
    del doc["sources"][0][missing]
    with pytest.raises(SourcesError, match=missing):
        parse_sources(doc)


def test_duplicate_file_id_across_sources_raises() -> None:
    doc = {
        "sources": [
            {
                "id": s,
                "text_id": "t",
                "page_url": "https://example.org",
                "licence": "Public Domain",
                "provenance": f"{s}.pdf",
                "files": [{"file_id": "dupe", "download_url": TODO_SENTINEL}],
            }
            for s in ("S1", "S2")
        ]
    }
    with pytest.raises(SourcesError, match="dupe"):
        parse_sources(doc)


def test_missing_sources_list_raises() -> None:
    with pytest.raises(SourcesError, match="sources"):
        parse_sources({"version": 1})


def test_bad_excerpt_value_raises() -> None:
    doc = {
        "sources": [
            {
                "id": "S9",
                "text_id": "t9",
                "page_url": "https://example.org",
                "licence": "Public Domain",
                "provenance": "S9.pdf",
                "files": [
                    {"file_id": "a", "download_url": "https://x/a.mp3", "excerpt_start_s": "soon"}
                ],
            }
        ]
    }
    with pytest.raises(SourcesError, match="excerpt_start_s"):
        parse_sources(doc)


# --------------------------------------------------------------------------- #
# The committed SOURCES.yaml
# --------------------------------------------------------------------------- #


def test_repo_sources_yaml_parses() -> None:
    sources = load_sources(SOURCES_YAML)
    ids = [s.id for s in sources]
    assert ids == ["S1", "S1b", "S2", "S3", "S4", "team_s1", "team_s2", "team_s3"]


def test_repo_sources_declare_expected_file_counts() -> None:
    for source in load_sources(SOURCES_YAML):
        if source.expected_files is not None:
            assert len(source.files) == source.expected_files, source.id


def test_repo_sources_have_no_forbidden_licence() -> None:
    """Every non-TODO licence in the committed file must pass the filter.

    A licence still marked TODO(human) is expected to fail - that is the signal
    telling a human to open the page.
    """
    for source in load_sources(SOURCES_YAML):
        allowed, reason = licence_verdict(source.licence)
        if not allowed:
            assert TODO_SENTINEL in source.licence, f"{source.id}: {reason}"


def test_repo_sources_urls_are_http() -> None:
    for source in load_sources(SOURCES_YAML):
        if source.page_url != TODO_SENTINEL:
            assert source.page_url.startswith("http"), source.id


def test_repo_sources_have_unique_file_ids_and_provenance_names() -> None:
    sources = load_sources(SOURCES_YAML)
    file_ids = [sf.file_id for _, sf in iter_files(sources)]
    assert len(file_ids) == len(set(file_ids))
    # One provenance artefact per source *page*. The three team_* sources share
    # both a page and dataset/provenance/team_consent.md, which is correct: the
    # evidence for all of them is the one signed CONSENT.md.
    by_artefact: dict[str, set[str]] = {}
    for source in sources:
        by_artefact.setdefault(source.provenance, set()).add(source.page_url)
    for artefact, pages in by_artefact.items():
        assert len(pages) == 1, f"{artefact} is claimed by several pages: {sorted(pages)}"
