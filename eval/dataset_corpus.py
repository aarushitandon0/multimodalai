"""The real corpus, in the shape the LOTO harness evaluates (prompt 19, spec 7).

:mod:`eval.cases` generates graded feature tables; this module reads the dataset
that ``make data`` + ``make features`` produce and hands :mod:`eval.run_loto`
the same :class:`~eval.cases.FoldCases` object, so one protocol implementation
serves both corpora and the only difference between a synthetic number and a
dataset number is where the tables came from.

What it reads, and what each piece is for:

``dataset/MANIFEST.csv``
    The roster: ``file_id, text_id, tier, origin, label, level, duration_s``
    (spec 7.4). ``origin`` decides a file's role - ``clean`` readings form the
    envelope, ``injected`` and ``human`` files are participants - and ``human``
    is test-only (spec 7.3).
``dataset/labels/*.json``
    Ground truth per file (spec 7.1): one record per flaw with its span, rung
    and ``s``. A clean reading needs no label JSON; a flawed file without one
    is **skipped**, because a missing label is not the same as no flaw.
``dataset/alignments/*.TextGrid`` and the feature cache
    Features are not re-derived here on purpose. The cache key is the audio
    hash plus the alignment fingerprint plus the config hash, so the one safe
    way to read the entry ``make features`` wrote is to build the key with the
    same code: :func:`load_tables` calls ``scripts/build_features.py``'s own
    ``collect_targets`` and ``build_one``, which read the cache when it is warm
    and extract when it is not. The number of files that had to be extracted is
    reported, so a run that silently paid for ``make features`` is visible.

**The envelope and the clean files.** A clean reading contributes to its text's
envelope, so comparing it against that same envelope would measure it partly
against itself and make the false-alarm rate of spec 14.3 look better than it
is. Every clean file is therefore scored against the
:func:`~rostrum.envelope.build.leave_one_out` envelope that drops it, which is
the construction spec 11 already uses for the confidence band; the sigma floors
carry over, as that function documents.

**The channel rule** (CLAUDE.md invariant 7) is enforced by the data rather
than by a flag: the envelope of a text read by many sources is not
single-channel, so :func:`~rostrum.detect.zscore.envelope_z` and the trigger
set drop the spectral, MFCC and voice-quality features for every participant
compared against it. ``mumbling`` consequently cannot fire in this comparison.
That is the rule working, and the report says so rather than working around it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from eval.cases import Case, FoldCases
from rostrum.detect.classifier import TruthFlaw
from rostrum.detect.refine import RefinementSettings, refine_regions
from rostrum.detect.regions import RegionThresholds, build_regions
from rostrum.detect.zscore import envelope_z, trigger_signals
from rostrum.envelope.build import (
    EnvelopeError,
    Reference,
    TextEnvelope,
    build_fold_envelopes,
    leave_one_out,
)
from rostrum.score.rubric import label_to_category

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "REFERENCE_ORIGINS",
    "UNITS",
    "CorpusUnavailable",
    "DatasetCorpus",
    "DatasetFile",
    "Requirement",
    "blocking_requirements",
    "describe_missing",
    "is_available",
    "load_tables",
    "missing_requirements",
    "read_files",
    "read_labels",
    "requirements",
]

UNITS: tuple[str, ...] = ("word", "gap", "phrase")
"""Unit kinds carried into the envelope comparison."""

REFERENCE_ORIGINS: tuple[str, ...] = ("clean",)
"""Origins that contribute to a text's envelope (spec 10.2)."""


class CorpusUnavailable(RuntimeError):
    """The requested corpus is not present, or not usable, in this checkout."""


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Requirement:
    """One thing the dataset corpus needs, and whether it is there.

    Attributes:
        key: The ``configs/paths.yaml`` key, so a message names a real path.
        path: Where it was looked for.
        ok: Whether it is present and non-empty.
        hint: What to run to produce it.
        blocking: Whether its absence stops the corpus from loading. The
            feature cache is the one piece that does not: features can be
            extracted on demand from the audio and the alignment, which is slow
            and is reported, but not fatal.
    """

    key: str
    path: Path
    ok: bool
    hint: str
    blocking: bool = True

    def describe(self) -> str:
        """One line for the refusal message."""
        return f"{self.key} ({self.path}): {self.hint}"


def _present(path: Path, pattern: str | None) -> bool:
    """Whether ``path`` exists and, for a directory, holds a match for ``pattern``."""
    if pattern is None:
        return path.is_file() and path.stat().st_size > 0
    return path.is_dir() and any(path.glob(pattern))


def requirements(config: Config) -> list[Requirement]:
    """Every prerequisite of the dataset corpus, present or not.

    Args:
        config: Loaded config.

    Returns:
        One :class:`Requirement` per prerequisite, in build order.
    """
    wanted: tuple[tuple[str, str | None, str, bool], ...] = (
        (
            "data.manifest",
            None,
            "run `make data` (the manifest is written by `make package`)",
            True,
        ),
        ("data.audio_clean", "*.wav", "run `make data` (download + build-clean)", True),
        (
            "data.alignments",
            "*.TextGrid",
            "run `make align` in the `mfa` conda env (`make setup-mfa` first)",
            True,
        ),
        (
            "data.labels",
            "*.json",
            "run `make inject` (and `make annotate-import` for human takes)",
            True,
        ),
        ("cache.features", "*", "run `make features`", False),
    )
    out: list[Requirement] = []
    for key, pattern, hint, blocking in wanted:
        try:
            path = config.path(key)
        except (KeyError, TypeError):  # pragma: no cover - the keys exist
            out.append(
                Requirement(
                    key=key, path=Path(key), ok=False, hint="no such config key", blocking=blocking
                )
            )
            continue
        out.append(
            Requirement(
                key=key, path=path, ok=_present(path, pattern), hint=hint, blocking=blocking
            )
        )
    return out


def missing_requirements(config: Config) -> list[Requirement]:
    """The prerequisites that are absent or empty."""
    return [item for item in requirements(config) if not item.ok]


def blocking_requirements(config: Config) -> list[Requirement]:
    """The missing prerequisites that stop the corpus from loading at all."""
    return [item for item in missing_requirements(config) if item.blocking]


def is_available(config: Config) -> bool:
    """Whether the dataset corpus can be loaded (the feature cache may be cold)."""
    return not blocking_requirements(config)


def describe_missing(config: Config) -> str:
    """The refusal message: what is missing, and what to run for each piece."""
    missing = missing_requirements(config)
    if not missing:
        return ""
    return "\n".join(
        [
            "the dataset corpus is not built in this checkout; missing:",
            *(
                f"  - {item.describe()}"
                + ("" if item.blocking else "  [not fatal: features are extracted on demand]")
                for item in missing
            ),
            "Until it is built, use --corpus synthetic, whose numbers are generated by",
            "eval/synthetic.py and must be labelled as such wherever they appear.",
        ]
    )


# --------------------------------------------------------------------------- #
# The roster
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetFile:
    """One recording of the dataset, with its ground truth attached.

    Attributes:
        file_id: Recording id, the audio file's stem.
        text_id: The text it is a rendition of; the envelope key.
        tier: ``gold`` | ``silver`` | ``bronze``.
        origin: ``clean`` | ``injected`` | ``human``.
        label: The manifest's flaw label, ``""`` for a clean reading.
        level: Severity rung, ``"L0"`` for a clean reading.
        duration_s: Duration from the manifest, used only as a fallback.
        truth: Ground-truth flaws read from ``labels/<file_id>.json``.
        qc_passed: ``qc.passed`` from the label JSON; a file the QC gates
            rejected (spec 5.4, invariant 9) is not evaluated.
    """

    file_id: str
    text_id: str
    tier: str = ""
    origin: str = "clean"
    label: str = ""
    level: str = "L0"
    duration_s: float = 0.0
    truth: tuple[TruthFlaw, ...] = ()
    qc_passed: bool = True

    @property
    def is_reference(self) -> bool:
        """Whether this reading contributes to its text's envelope."""
        return self.origin in REFERENCE_ORIGINS


def _flaws_from_label(payload: Mapping[str, Any], file_id: str) -> list[TruthFlaw]:
    """Ground-truth flaws from one label JSON (spec 7.1)."""
    out: list[TruthFlaw] = []
    for raw in payload.get("flaws") or ():
        try:
            out.append(
                TruthFlaw(
                    label=str(raw["label"]),
                    t_start=float(raw["t_start"]),
                    t_end=float(raw["t_end"]),
                    s=float(raw.get("s", 0.0)),
                    level=str(raw.get("level", "")),
                    file_id=file_id,
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            logger.warning("%s: unusable flaw record (%s): %r", file_id, error, raw)
    return sorted(out, key=lambda flaw: (flaw.t_start, flaw.label))


def read_labels(config: Config) -> dict[str, dict[str, Any]]:
    """``{file_id: label JSON}`` from ``dataset/labels/`` (spec 7.1)."""
    try:
        directory = config.path("data.labels")
    except (KeyError, TypeError):  # pragma: no cover - the key exists
        return {}
    out: dict[str, dict[str, Any]] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("could not read label %s: %s", path, error)
            continue
        out[str(payload.get("file_id") or path.stem)] = payload
    return out


def read_files(config: Config) -> list[DatasetFile]:
    """The manifest joined with the label JSONs, in text then file order.

    A flawed file (``origin`` other than ``clean``) with no label JSON is
    dropped with a warning: counted as clean, every region found in it would be
    scored as a false alarm. Files whose ``qc.passed`` is false are dropped as
    well (invariant 9).

    Args:
        config: Loaded config.

    Returns:
        One :class:`DatasetFile` per usable recording.

    Raises:
        CorpusUnavailable: There is no manifest to read.
    """
    manifest = config.path("data.manifest")
    if not manifest.is_file():
        raise CorpusUnavailable(describe_missing(config) or f"no manifest at {manifest}")
    labels = read_labels(config)

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    out: list[DatasetFile] = []
    for row in rows:
        file_id = str(row.get("file_id") or "").strip()
        text_id = str(row.get("text_id") or "").strip()
        if not file_id or not text_id:
            continue
        origin = str(row.get("origin") or "clean").strip()
        payload = labels.get(file_id)
        if payload is None and origin not in REFERENCE_ORIGINS:
            logger.warning("%s: origin %s but no label JSON; skipped", file_id, origin)
            continue
        qc = dict((payload or {}).get("qc") or {})
        if "passed" in qc and not bool(qc["passed"]):
            logger.warning("%s: label JSON says qc.passed is false; skipped", file_id)
            continue
        try:
            duration = float(row.get("duration_s") or 0.0)
        except ValueError:
            duration = 0.0
        out.append(
            DatasetFile(
                file_id=file_id,
                text_id=text_id,
                tier=str(row.get("tier") or ""),
                origin=origin,
                label=str(row.get("label") or "").strip(),
                level=str(row.get("level") or "").strip() or "L0",
                duration_s=duration,
                truth=tuple(_flaws_from_label(payload, file_id)) if payload else (),
                qc_passed=bool(qc.get("passed", True)),
            )
        )
    return sorted(out, key=lambda item: (item.text_id, item.file_id))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def load_tables(
    config: Config,
    file_ids: Sequence[str],
    *,
    lean: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, float], int]:
    """Load every wanted recording's cached feature tables.

    Args:
        config: Loaded config.
        file_ids: Recordings wanted; anything else on disk is ignored.
        lean: Skip MFCC, voice quality and eGeMAPS. For tests only - a lean
            entry has a different cache key from a full one.

    Returns:
        ``(tables, frames, durations, n_extracted)``: ``tables`` is
        ``{file_id: {unit: table}}``, ``frames`` is ``{file_id: frame table}``,
        ``durations`` is the audio duration per file in seconds, and
        ``n_extracted`` counts the files whose features were **not** in the
        cache and had to be computed here.
    """
    from scripts.build_features import build_one, collect_targets

    wanted = set(file_ids)
    targets = [t for t in collect_targets(config, sample=False) if t.file_id in wanted]
    for file_id in sorted(wanted - {t.file_id for t in targets}):
        logger.warning("%s: in the manifest but has no audio file; skipped", file_id)

    tables: dict[str, dict[str, Any]] = {}
    frames: dict[str, Any] = {}
    durations: dict[str, float] = {}
    extracted = 0
    for target in targets:
        if not target.alignment.is_file():
            logger.warning("%s: no alignment at %s; skipped", target.file_id, target.alignment)
            continue
        try:
            bundle, from_cache = build_one(
                target, config, crosscheck=False, lean=lean, refresh=False
            )
        except Exception as error:  # pragma: no cover - depends on the dataset
            logger.warning("%s: features unavailable (%s); skipped", target.file_id, error)
            continue
        if not from_cache:
            extracted += 1
        tables[target.file_id] = {unit: bundle.table(unit) for unit in UNITS}
        frames[target.file_id] = bundle.frames
        durations[target.file_id] = float(bundle.meta.get("audio_duration_s") or 0.0)

    if extracted:
        logger.warning(
            "%d file(s) were not in the feature cache and were extracted now; run "
            "`make features` so that an evaluation never pays for extraction",
            extracted,
        )
    return tables, frames, durations, extracted


def _word_spans(table: Any) -> dict[int, tuple[float, float]]:
    """``{word_idx: (t_start, t_end)}`` from a word table, for the overlay curves."""
    return {int(idx): (float(row["t_start"]), float(row["t_end"])) for idx, row in table.iterrows()}


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #


@dataclass
class DatasetCorpus:
    """Every usable dataset recording with its features and ground truth.

    Loaded once and folded many times: what depends on the fold is the sigma
    floors, and therefore the envelopes, not the feature tables.

    Attributes:
        files: The roster, in text then file order.
        tables: ``{file_id: {unit: table}}``.
        frames: ``{file_id: frame table}``.
        durations: ``{file_id: audio duration in seconds}``.
        n_extracted: Files whose features were missing from the cache.
    """

    files: list[DatasetFile] = field(default_factory=list)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    frames: dict[str, Any] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    n_extracted: int = 0

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, config: Config, *, lean: bool = False) -> DatasetCorpus:
        """Read the manifest, the labels and the cached features.

        Args:
            config: Loaded config.
            lean: Passed to :func:`load_tables`; tests only.

        Returns:
            A :class:`DatasetCorpus` holding the files that have a label where
            one is required and usable features.

        Raises:
            CorpusUnavailable: A prerequisite is missing, or nothing survived.
        """
        if blocking_requirements(config):
            raise CorpusUnavailable(describe_missing(config))
        for item in missing_requirements(config):
            logger.warning("%s is missing: %s", item.key, item.hint)
        roster = read_files(config)
        tables, frames, durations, extracted = load_tables(
            config, [item.file_id for item in roster], lean=lean
        )
        kept = [item for item in roster if item.file_id in tables]
        if not kept:
            raise CorpusUnavailable(
                "no dataset file has both a label and cached features; "
                "run `make data`, then `make features`"
            )
        if len(kept) != len(roster):
            logger.warning(
                "%d of %d manifest row(s) had no usable features",
                len(roster) - len(kept),
                len(roster),
            )
        return cls(
            files=kept, tables=tables, frames=frames, durations=durations, n_extracted=extracted
        )

    # -- views --------------------------------------------------------------

    @property
    def texts(self) -> tuple[str, ...]:
        """Every text with at least one reference reading, sorted."""
        return tuple(sorted({item.text_id for item in self.files if item.is_reference}))

    def references_by_text(self) -> dict[str, list[str]]:
        """``{text_id: [reference file_id, ...]}`` for the envelope builder."""
        out: dict[str, list[str]] = {}
        for item in self.files:
            if item.is_reference:
                out.setdefault(item.text_id, []).append(item.file_id)
        return {text: sorted(ids) for text, ids in out.items()}

    def pick_references(
        self, text_id: str, ids: Sequence[str], *, size: int, seed: int
    ) -> list[str]:
        """Choose ``size`` of a text's references, deterministically.

        Spec 14.2's first ablation is "single reference (Gold if available else
        one random Silver, fixed seed) vs envelope", so with ``size == 1`` a
        gold-tier reading wins and otherwise one reading is drawn with an RNG
        seeded from ``seed`` and the text id. The draw uses a SHA-256 digest of
        the text id rather than :func:`hash`, so it does not depend on
        ``PYTHONHASHSEED``.

        Args:
            text_id: The text being reduced.
            ids: Its reference file ids.
            size: How many to keep.
            seed: Master seed.

        Returns:
            The kept ids, sorted.
        """
        import numpy as np

        ordered = sorted(ids)
        if size >= len(ordered):
            return ordered
        if size != 1:
            return ordered[:size]
        tiers = {item.file_id: item.tier for item in self.files}
        gold = [fid for fid in ordered if tiers.get(fid) == "gold"]
        if gold:
            return gold[:1]
        digest = hashlib.sha256(f"{seed}:{text_id}".encode()).hexdigest()[:8]
        rng = np.random.default_rng(int(digest, 16))
        return [ordered[int(rng.integers(len(ordered)))]]

    def channel_of(self, file_id: str) -> str:
        """The recording's channel id, as the feature tables recorded it."""
        tables = self.tables.get(file_id, {})
        word = tables.get("word")
        if word is None:
            return file_id
        return str(word.attrs.get("channel_id", "")) or file_id

    def _references(self, members: Sequence[str]) -> list[Reference]:
        """``members`` as :class:`~rostrum.envelope.build.Reference` objects."""
        wanted = set(members)
        return [
            Reference(
                file_id=item.file_id,
                channel_id=self.channel_of(item.file_id),
                tables=self.tables[item.file_id],
            )
            for item in self.files
            if item.is_reference and item.file_id in wanted
        ]

    # -- one fold -----------------------------------------------------------

    def fold(
        self,
        *,
        config: Config,
        held_out: str,
        refine: bool = True,
        thresholds: RegionThresholds | None = None,
        envelope_size: int | None = None,
        leave_clean_out: bool = True,
    ) -> FoldCases:
        """Build, detect and label one LOTO fold of the dataset.

        Args:
            config: Loaded config.
            held_out: The fold's test text.
            refine: Run ruptures boundary refinement.
            thresholds: Detection thresholds; the config's defaults when
                ``None``. :mod:`eval.run_loto` re-detects at its tuned values
                afterwards, so this only has to be a usable starting point.
            envelope_size: Keep only this many references per text, which is
                ablation 1 of spec 14.2. The members kept are the first in
                file-id order, so the choice is deterministic.
            leave_clean_out: Score each clean reading against the envelope that
                drops it. Switching it off measures clean files partly against
                themselves; it exists to show what that is worth.

        Returns:
            A :class:`~eval.cases.FoldCases`.

        Raises:
            CorpusUnavailable: The held-out text has no envelope, so the fold
                cannot be run at all.
        """
        texts = self.texts
        if held_out not in texts:
            raise CorpusUnavailable(
                f"no text {held_out!r} with reference readings in the dataset; have {list(texts)}"
            )
        train_texts = tuple(text for text in texts if text != held_out)
        fold = f"loto_{held_out}"

        references_by_text = self.references_by_text()
        if envelope_size is not None:
            references_by_text = {
                text: self.pick_references(
                    text, ids, size=int(envelope_size), seed=int(config.seed)
                )
                for text, ids in references_by_text.items()
            }
        members = [fid for ids in references_by_text.values() for fid in ids]

        envelopes = build_fold_envelopes(
            references_by_text,
            self._references(members),
            fold=fold,
            train_texts=train_texts,
            config=config,
            frames={fid: self.frames[fid] for fid in members if fid in self.frames},
            word_spans={
                fid: _word_spans(self.tables[fid]["word"]) for fid in members if fid in self.tables
            },
            skip_unbuildable=True,
        )
        if held_out not in envelopes:
            raise CorpusUnavailable(
                f"{held_out}: fewer than {config.detection['envelope']['min_references']} "
                "reference reading(s) with usable features, so this fold has no envelope"
            )

        if thresholds is None:
            thresholds = RegionThresholds.from_config(config)
        settings = RefinementSettings.from_config(config) if refine else None

        out = FoldCases(
            fold=fold,
            held_out=held_out,
            train_texts=train_texts,
            envelopes=envelopes,
            thresholds=thresholds,
            reference_tables={
                text: [self.tables[fid] for fid in ids if fid in self.tables]
                for text, ids in references_by_text.items()
            },
        )

        for item in self.files:
            envelope = envelopes.get(item.text_id)
            if envelope is None:
                logger.debug("%s: text %s has no envelope; skipped", item.file_id, item.text_id)
                continue
            if leave_clean_out and item.is_reference and item.file_id in envelope.reference_ids:
                try:
                    envelope = leave_one_out(envelope, item.file_id)
                except (EnvelopeError, KeyError) as error:
                    logger.warning(
                        "%s: cannot leave its own reading out (%s); skipped", item.file_id, error
                    )
                    continue
            out.cases.append(
                self._case(item, envelope, config=config, thresholds=thresholds, settings=settings)
            )

        logger.info(
            "fold %s: %d dataset case(s) over %d text(s) (%d clean, %d injected, %d human), "
            "held out %s",
            fold,
            len(out.cases),
            len(texts),
            sum(1 for c in out.cases if c.origin == "clean"),
            sum(1 for c in out.cases if c.origin == "injected"),
            sum(1 for c in out.cases if c.origin == "human"),
            held_out,
        )
        return out

    def _case(
        self,
        item: DatasetFile,
        envelope: TextEnvelope,
        *,
        config: Config,
        thresholds: RegionThresholds,
        settings: RefinementSettings | None,
    ) -> Case:
        """Compare one recording against ``envelope`` and wrap it as a :class:`Case`."""
        tables = self.tables[item.file_id]
        channel = self.channel_of(item.file_id)
        same_channel = bool(envelope.same_channel and channel in envelope.channel_ids)

        z_by_unit = envelope_z(
            tables, envelope, config=config, channel_matched=same_channel, file_id=item.file_id
        )
        signals, skipped = trigger_signals(z_by_unit, config=config, same_channel=same_channel)
        regions = build_regions(signals, z_by_unit, config=config, thresholds=thresholds)
        if settings is not None:
            regions = refine_regions(regions, self.frames.get(item.file_id), settings=settings)
        if skipped:
            logger.debug("%s: skipped %s", item.file_id, [f"{s.label}:{s.reason}" for s in skipped])

        words = tables["word"]
        speech = float(words["t_end"].max()) - float(words["t_start"].min())
        label = item.label or (item.truth[0].label if item.truth else "")
        categories = label_to_category(config)
        return Case(
            file_id=item.file_id,
            text_id=item.text_id,
            level=(item.level or item.truth[0].level) if item.truth else "L0",
            label=label if item.truth else "",
            # The rubric calibrates on single-flaw files only (spec 11), so a
            # multi-flaw take carries no single category.
            category=categories.get(label, "") if len(item.truth) == 1 else "",
            n_flaws=len(item.truth),
            regions=regions,
            z_by_unit=z_by_unit,
            truth=list(item.truth),
            tables=tables,
            speech_duration=speech,
            frames=self.frames.get(item.file_id),
            duration=float(self.durations.get(item.file_id) or item.duration_s or speech),
            origin=item.origin,
            same_channel=same_channel,
            tier=item.tier,
        )
