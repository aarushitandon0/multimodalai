"""Tests for the QC summary report (spec 5.4 acceptance).

The tables are built from hand-written label JSONs and QC CSVs, so the suite
covers the report without MFA, without UTMOS and without a built dataset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qc_report as qr
from rostrum.inject import qc

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _label(
    file_id: str,
    source: str,
    label: str,
    level: str,
    *,
    passed: bool = True,
    max_err_ms: float = 10.0,
) -> dict:
    return {
        "file_id": file_id,
        "source_file": source,
        "text_id": "s1_gettysburg",
        "origin": "injected",
        "flaws": [{"label": label, "level": level, "t_start": 1.0, "t_end": 2.0}],
        "qc": {
            "utmos_clean": 4.0,
            "utmos_edited": 3.5,
            "realign_max_err_ms": max_err_ms,
            "passed": passed,
            "reasons": [] if passed else ["realign boundary error 90.0 ms > 50 ms (spec 5.4)"],
        },
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """A miniature ``dataset/`` with labels and the three QC CSVs."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    qc_dir = tmp_path / "qc"

    utmos = {"f1": 4.20, "f2": 4.00}
    rows = []
    # rushing: monotonically worse with level. dragging: L2 better than L1.
    means = {
        ("rushing", "L1"): (4.10, 4.05),
        ("rushing", "L2"): (3.80, 3.90),
        ("rushing", "L3"): (3.40, 3.50),
        ("rushing", "L4"): (2.90, 3.00),
        ("dragging", "L1"): (3.50, 3.60),
        ("dragging", "L2"): (3.90, 3.95),
    }
    for (label, level), (a, b) in means.items():
        for source, score in (("f1", a), ("f2", b)):
            file_id = f"{source}_{label}_{level}"
            utmos[file_id] = score
            lab = _label(file_id, source, label, level)
            lab["qc"]["utmos_clean"] = utmos[source]
            lab["qc"]["utmos_edited"] = score
            (labels_dir / f"{file_id}.json").write_text(json.dumps(lab), encoding="utf-8")
            rows.append(
                qc.RealignReport(
                    file_id=file_id,
                    source_file=source,
                    label=label,
                    level=level,
                    n_words=10,
                    max_err_ms=12.0 if level in ("L1", "L2") else 31.0,
                    mean_err_ms=6.0,
                    p95_err_ms=18.0,
                    frac_within_target=0.9,
                    target_ms=20.0,
                    max_allowed_ms=50.0,
                )
            )

    # One rejected file, logged and marked.
    rejected = _label("f1_monotone_L1", "f1", "monotone", "L1", passed=False, max_err_ms=90.0)
    (labels_dir / "f1_monotone_L1.json").write_text(json.dumps(rejected), encoding="utf-8")

    qc.append_utmos(qc_dir / "utmos.csv", utmos)
    qc.append_realign(qc_dir / "realign.csv", rows)
    qc.append_rejections(
        qc_dir / "rejections.csv",
        [
            qc.rejection_row(
                rejected,
                qc.QCVerdict(
                    file_id="f1_monotone_L1",
                    realign=None,
                    utmos_clean=4.2,
                    utmos_edited=3.0,
                    level="L1",
                    max_utmos_drop=0.3,
                    reasons=("realign boundary error 90.0 ms > 50 ms (spec 5.4)",),
                ),
            )
        ],
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


def test_records_pair_each_edited_file_with_its_clean_source(dataset: Path) -> None:
    records = qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")

    rushing_l4 = [r for r in records if r.label == "rushing" and r.level == "L4"]
    assert {r.file_id for r in rushing_l4} == {"f1_rushing_L4", "f2_rushing_L4"}
    rec = next(r for r in rushing_l4 if r.file_id == "f1_rushing_L4")
    assert rec.utmos_clean == pytest.approx(4.20)
    assert rec.utmos_edited == pytest.approx(2.90)
    assert rec.drop == pytest.approx(1.30)
    assert rec.passed is True


def test_a_rejected_file_is_still_a_record(dataset: Path) -> None:
    records = qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")
    rejected = next(r for r in records if r.file_id == "f1_monotone_L1")
    assert rejected.passed is False


# --------------------------------------------------------------------------- #
# UTMOS table
# --------------------------------------------------------------------------- #


def test_utmos_is_aggregated_per_label_and_level(dataset: Path) -> None:
    records = qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")

    stats = qr.utmos_stats(records)

    cell = stats[("rushing", "L2")]
    assert cell.n == 2
    assert cell.mean == pytest.approx(3.85)
    assert cell.sd == pytest.approx(0.0707, abs=1e-3)  # sample sd of {3.80, 3.90}


def test_a_single_file_has_no_spread() -> None:
    stats = qr.utmos_stats(
        [qr.Record("a", "s", "rushing", "L1", 4.0, 3.5, True)],
    )
    assert stats[("rushing", "L1")].sd == pytest.approx(0.0)


def test_the_utmos_table_is_markdown_with_a_mean_and_sd_per_cell(dataset: Path) -> None:
    records = qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")

    table = qr.format_utmos_table(records)

    assert "| label | L0 | L1 | L2 | L3 | L4 | n |" in table
    assert "3.85 ± 0.07" in table
    assert table.count("rushing") == 1
    assert "—" in table  # dragging has no L3/L4 file


# --------------------------------------------------------------------------- #
# Monotonicity (the stated acceptance criterion)
# --------------------------------------------------------------------------- #


def test_monotonicity_is_reported_per_label(dataset: Path) -> None:
    records = qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")

    violations = qr.monotonicity_violations(records)

    assert [v.label for v in violations] == ["dragging"]
    assert violations[0].level == "L2"
    assert violations[0].previous_level == "L1"
    assert violations[0].mean > violations[0].previous_mean


def test_a_monotone_label_reports_no_violation(dataset: Path) -> None:
    records = [
        r
        for r in qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")
        if r.label == "rushing"
    ]
    assert qr.monotonicity_violations(records) == []


def test_the_clean_source_is_the_l0_baseline(dataset: Path) -> None:
    """A label whose L1 scores *above* its clean source breaks monotonicity."""
    records = [
        qr.Record("a_x_L1", "a", "x", "L1", utmos_clean=3.0, utmos_edited=4.0, passed=True),
    ]
    violations = qr.monotonicity_violations(records)
    assert [(v.previous_level, v.level) for v in violations] == [("L0", "L1")]


# --------------------------------------------------------------------------- #
# Realign distribution
# --------------------------------------------------------------------------- #


def test_realign_distribution_reports_mean_p95_and_the_20_ms_share(dataset: Path) -> None:
    rows = qr.read_realign(dataset / "qc" / "realign.csv")

    dist = qr.realign_distribution(rows)

    # 8 files at 12 ms (rushing L1-L2, dragging L1-L2), 4 at 31 ms (rushing L3-L4)
    assert dist.n == 12
    assert dist.mean_err_ms == pytest.approx((12.0 * 8 + 31.0 * 4) / 12)
    assert dist.p95_err_ms == pytest.approx(31.0)
    assert dist.frac_within_20ms == pytest.approx(8 / 12)
    assert dist.max_err_ms == pytest.approx(31.0)


def test_realign_distribution_of_nothing_is_empty() -> None:
    dist = qr.realign_distribution([])
    assert dist.n == 0
    assert dist.mean_err_ms is None


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #


def test_rejection_counts_are_grouped_by_gate_and_level(dataset: Path) -> None:
    rows = qr.read_rejections(dataset / "qc" / "rejections.csv")

    counts = qr.rejection_counts(rows)

    assert counts.total == 1
    assert counts.by_gate["realign"] == 1
    assert counts.by_level["L1"] == 1
    assert counts.by_label["monotone"] == 1


def test_the_rejection_rate_is_over_every_generated_file(dataset: Path) -> None:
    records = qr.collect(dataset / "labels", dataset / "qc" / "utmos.csv")
    rows = qr.read_rejections(dataset / "qc" / "rejections.csv")

    rate = qr.rejection_rate(records, rows)

    assert rate == pytest.approx(1 / 13)


# --------------------------------------------------------------------------- #
# summary.md
# --------------------------------------------------------------------------- #


def test_summary_markdown_has_every_required_section(dataset: Path) -> None:
    text = qr.build_summary(
        labels_dir=dataset / "labels",
        utmos_csv=dataset / "qc" / "utmos.csv",
        realign_csv=dataset / "qc" / "realign.csv",
        rejections_csv=dataset / "qc" / "rejections.csv",
    )

    assert "# Injection QC summary" in text
    assert "## UTMOS by label and level" in text
    assert "## Re-alignment boundary error" in text
    assert "## Rejections" in text
    assert "dragging" in text  # the monotonicity warning names the label
    assert "NOT monotonic" in text


def test_main_writes_summary_md_and_prints_the_rejection_rate(dataset, capsys) -> None:
    out = dataset / "qc" / "summary.md"

    code = qr.main(
        [
            "--labels-dir",
            str(dataset / "labels"),
            "--qc-dir",
            str(dataset / "qc"),
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert out.is_file()
    assert "# Injection QC summary" in out.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    assert "rejection rate" in printed.lower()
    assert "7.7%" in printed  # 1 of 13


def test_main_reports_an_empty_dataset_without_crashing(tmp_path, capsys) -> None:
    (tmp_path / "labels").mkdir()
    (tmp_path / "qc").mkdir()

    code = qr.main(
        [
            "--labels-dir",
            str(tmp_path / "labels"),
            "--qc-dir",
            str(tmp_path / "qc"),
            "--out",
            str(tmp_path / "qc" / "summary.md"),
        ]
    )

    assert code == 2
    assert "no label" in capsys.readouterr().err.lower()
