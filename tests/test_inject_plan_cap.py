"""Tests for the clean-parent cap that bounds how many variants are generated.

The cap exists because QC re-aligns every generated variant with MFA, so the
number of parents is the only knob that bounds the build without punching holes
in the 7-label x L1-L4 grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_injected import cap_parents_per_text
from rostrum.data.manifest import ManifestRow


def _rows(text_id: str, n: int) -> list[ManifestRow]:
    """n clean rows for one text, ids ``<text>_lv01``..``<text>_lvNN``."""
    return [
        ManifestRow(
            file_id=f"{text_id}_lv{i:02d}",
            text_id=text_id,
            tier="silver",
            origin="clean",
        )
        for i in range(1, n + 1)
    ]


def test_no_cap_keeps_every_row_sorted() -> None:
    rows = _rows("s5", 3) + _rows("s1", 2)
    for cap in (None, 0):
        kept = cap_parents_per_text(rows, cap)
        assert [r.file_id for r in kept] == [
            "s1_lv01",
            "s1_lv02",
            "s5_lv01",
            "s5_lv02",
            "s5_lv03",
        ]


def test_cap_applies_per_text_not_globally() -> None:
    kept = cap_parents_per_text(_rows("s1", 16) + _rows("s5", 16), 2)
    assert len(kept) == 4
    assert sum(r.text_id == "s1" for r in kept) == 2
    assert sum(r.text_id == "s5" for r in kept) == 2


def test_selection_is_reader_spread_not_first_n() -> None:
    kept = cap_parents_per_text(_rows("s1", 16), 2)
    assert [r.file_id for r in kept] == ["s1_lv01", "s1_lv09"]


def test_cap_above_available_keeps_all_and_never_repeats() -> None:
    kept = cap_parents_per_text(_rows("s3", 1), 4)
    assert [r.file_id for r in kept] == ["s3_lv01"]


@pytest.mark.parametrize("cap", [1, 2, 3, 5, 7])
def test_selection_is_deterministic_and_distinct(cap: int) -> None:
    rows = _rows("s1", 14)
    first = [r.file_id for r in cap_parents_per_text(rows, cap)]
    assert first == [r.file_id for r in cap_parents_per_text(list(reversed(rows)), cap)]
    assert len(set(first)) == len(first) == min(cap, 14)
