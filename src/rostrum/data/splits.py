"""Leave-one-text-out (LOTO) cross-validation splits (spec 7.3).

One fold per text: every *other* text calibrates, the held-out text tests.
Two rules make the protocol honest and are enforced here rather than left to
callers (CLAUDE.md invariant 11):

* **Nothing from the held-out text reaches the training side.** Thresholds,
  weights, ``k_c`` and the classifier see ``N-1`` texts only.
* **Human flawed takes are test-only.** They never calibrate anything, in any
  fold, so a synthetic artefact cannot be learned and then credited on real
  speech.

The output is deterministic: texts and file ids are sorted, so the JSON is
byte-identical whatever order the manifest happens to be in (invariant 13).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rostrum.data.manifest import ManifestRow

__all__ = [
    "FOLD_PREFIX",
    "TEST_ONLY_ORIGINS",
    "Fold",
    "LeakageError",
    "LotoSplits",
    "build_loto",
    "check_no_leakage",
    "fold_id",
    "load_loto",
    "with_split_folds",
    "write_loto",
]

logger = logging.getLogger(__name__)

FOLD_PREFIX = "loto_"
"""Fold ids are ``loto_<held_out_text>``, so a fold name states its own test set."""

TEST_ONLY_ORIGINS: tuple[str, ...] = ("human",)
"""Origins that may never appear on the training side of any fold (spec 7.3)."""

PROTOCOL = "leave_one_text_out"
SCHEMA = "rostrum.splits.loto.v1"


class LeakageError(AssertionError):
    """A fold violates the LOTO contract."""


def fold_id(text_id: str) -> str:
    """Return the fold id that holds out ``text_id``."""
    return f"{FOLD_PREFIX}{text_id}"


@dataclass(frozen=True)
class Fold:
    """One LOTO fold.

    Attributes:
        fold_id: ``loto_<held_out_text>``.
        held_out_text: The text under test in this fold.
        train_texts: Every other text, sorted.
        test_texts: Always a single-element tuple, ``(held_out_text,)``.
        train: File ids usable for calibration, sorted.
        test: File ids of the held-out text, sorted.
    """

    fold_id: str
    held_out_text: str
    train_texts: tuple[str, ...]
    test_texts: tuple[str, ...]
    train: tuple[str, ...]
    test: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "held_out_text": self.held_out_text,
            "train_texts": list(self.train_texts),
            "test_texts": list(self.test_texts),
            "n_train": len(self.train),
            "n_test": len(self.test),
            "train": list(self.train),
            "test": list(self.test),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Fold:
        return cls(
            fold_id=str(data["fold_id"]),
            held_out_text=str(data["held_out_text"]),
            train_texts=tuple(data["train_texts"]),
            test_texts=tuple(data["test_texts"]),
            train=tuple(data["train"]),
            test=tuple(data["test"]),
        )


@dataclass(frozen=True)
class LotoSplits:
    """The complete set of folds over the dataset's texts."""

    folds: tuple[Fold, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        """Every text covered, in fold order."""
        return tuple(f.held_out_text for f in self.folds)

    def fold(self, text_id: str) -> Fold:
        """Return the fold that holds out ``text_id``."""
        for f in self.folds:
            if f.held_out_text == text_id:
                return f
        raise KeyError(f"no fold holds out text {text_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "test_only_origins": list(TEST_ONLY_ORIGINS),
            "n_folds": len(self.folds),
            "texts": list(self.texts),
            "folds": [f.to_dict() for f in self.folds],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LotoSplits:
        return cls(folds=tuple(Fold.from_dict(f) for f in data["folds"]))


def build_loto(rows: Iterable[ManifestRow | Mapping[str, Any]]) -> LotoSplits:
    """Build one fold per text from manifest rows.

    Args:
        rows: Manifest rows, in any order.

    Returns:
        Folds ordered by held-out text id.

    Raises:
        ValueError: A row has no ``text_id``, or fewer than two texts are
            present (one text cannot be held out *and* calibrated on).
    """
    items = [_as_row(r) for r in rows]
    missing = sorted(r.file_id for r in items if not r.text_id)
    if missing:
        raise ValueError(f"manifest rows without a text_id cannot be split: {missing}")

    texts = sorted({r.text_id for r in items})
    if len(texts) < 2:
        raise ValueError(f"leave-one-text-out needs at least two texts, got {len(texts)}")

    folds = []
    for held_out in texts:
        train_texts = tuple(t for t in texts if t != held_out)
        train = tuple(
            sorted(
                r.file_id
                for r in items
                if r.text_id != held_out and r.origin not in TEST_ONLY_ORIGINS
            )
        )
        test = tuple(sorted(r.file_id for r in items if r.text_id == held_out))
        folds.append(
            Fold(
                fold_id=fold_id(held_out),
                held_out_text=held_out,
                train_texts=train_texts,
                test_texts=(held_out,),
                train=train,
                test=test,
            )
        )
    logger.info("built %d LOTO folds over texts %s", len(folds), texts)
    return LotoSplits(folds=tuple(folds))


def with_split_folds(rows: Iterable[ManifestRow]) -> list[ManifestRow]:
    """Return ``rows`` with ``split_fold`` set to the fold each file is tested in.

    Under LOTO a file is tested in exactly one fold - the one holding out its
    own text - which is what ``MANIFEST.csv``'s ``split_fold`` column records.
    """
    return [replace(r, split_fold=fold_id(r.text_id) if r.text_id else "") for r in rows]


def check_no_leakage(splits: LotoSplits, rows: Iterable[ManifestRow | Mapping[str, Any]]) -> None:
    """Assert every fold honours the LOTO contract.

    Raises:
        LeakageError: A file is on both sides of a fold, a test-only origin
            appears in training data, or a fold's training files belong to the
            held-out text.
    """
    origin = {r.file_id: r.origin for r in (_as_row(x) for x in rows)}
    text_of = {r.file_id: r.text_id for r in (_as_row(x) for x in rows)}

    for f in splits.folds:
        both = sorted(set(f.train) & set(f.test))
        if both:
            raise LeakageError(f"{f.fold_id}: files on both sides of the split: {both}")

        humans = sorted(fid for fid in f.train if origin.get(fid) in TEST_ONLY_ORIGINS)
        if humans:
            raise LeakageError(
                f"{f.fold_id}: human (test-only) files in the training set: {humans}"
            )

        wrong_text = sorted(fid for fid in f.train if text_of.get(fid) == f.held_out_text)
        if wrong_text:
            raise LeakageError(f"{f.fold_id}: training files from the held-out text: {wrong_text}")


def write_loto(path: Path | str, splits: LotoSplits) -> Path:
    """Write ``splits`` as JSON, sorted and newline-terminated.

    No timestamp is written: the file must hash identically across runs.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(splits.to_dict(), indent=2, ensure_ascii=False) + "\n"
    out.write_text(text, encoding="utf-8", newline="\n")
    logger.info("wrote %s (%d folds)", out, len(splits.folds))
    return out


def load_loto(path: Path | str) -> LotoSplits:
    """Read a ``loto.json`` written by :func:`write_loto`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return LotoSplits.from_dict(data)


def _as_row(row: ManifestRow | Mapping[str, Any]) -> ManifestRow:
    """Accept either a dataclass row or the dict a CSV reader produces."""
    if isinstance(row, ManifestRow):
        return row
    return ManifestRow(
        file_id=str(row["file_id"]),
        text_id=str(row.get("text_id", "")),
        tier=str(row.get("tier", "")),
        origin=str(row.get("origin", "")),
        label=str(row.get("label", "")),
        level=str(row.get("level", "")),
        duration_s=row.get("duration_s", ""),  # type: ignore[arg-type]
        split_fold=str(row.get("split_fold", "")),
    )
