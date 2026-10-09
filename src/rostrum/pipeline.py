"""One recording in, one :class:`~rostrum.types.AnalysisResult` out (spec 13).

This module is the spine of the system: it is the only place that knows the
order of the stages, and the only place that decides which **operating mode**
an upload is analysed in.

.. code-block:: text

    preprocess -> ASR mismatch check -> text match -> MFA align -> features
        -> z vs envelope -> regions -> refine -> classify -> score -> cards
        -> curves -> round and cache

The three modes of spec 13 differ in exactly one thing: what the recording is
compared against.

=========  ==========================================  ===============================
Mode       Trigger                                     Comparison
=========  ==========================================  ===============================
Reference  transcript matches a library text at        that text's envelope of skilled
           >= ``pipeline.text_match_threshold`` and    readings, word by word
           the text has >= ``modes.reference.
           min_references`` readers
Norm       alignment succeeded, no such text           pooled norms over every skilled
                                                       reading (spec 13); the
                                                       channel-sensitive features are
                                                       measured but cannot fire
Reject     WER > ``modes.reject.max_wer``, or the      nothing; no region and no score
           alignment failed                            are reported
=========  ==========================================  ===============================

Rules this module exists to enforce:

* **MFA only** for timing (CLAUDE.md invariant 4). ``faster-whisper`` appears
  once, to compare texts, and its output never reaches a timestamp.
* **No leakage** (invariant 11). ``analyze`` fits nothing. Thresholds,
  ``k_c``, the classifier and the sigma floors arrive already fitted, from the
  :class:`ReferenceLibrary` or from the caller's fold; there is no code path
  here that could learn from the file being analysed.
* **No LLM** (invariant 12) and **no network** (invariant 13).
* **Determinism**: the same bytes, transcript and config give a
  byte-identical JSON, which is what ``results/determinism.json`` checks. The
  analysis is cached under
  ``sha256(audio bytes + transcript + config hash)``.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rostrum.align import asr as A
from rostrum.align import mfa as M
from rostrum.config import Config, code_version, load_config
from rostrum.detect import refine as RF
from rostrum.detect.regions import RegionThresholds, build_regions
from rostrum.detect.zscore import envelope_z, trigger_signals
from rostrum.envelope import norms as NO
from rostrum.envelope.build import (
    CurveEnvelope,
    Reference,
    TextEnvelope,
    build_fold_envelopes,
    leave_one_out,
    warp_to_envelope,
    word_curves,
)
from rostrum.explain.cards import build_cards
from rostrum.features import extract_all as X
from rostrum.features import pauses as PA
from rostrum.io.audio import preprocess
from rostrum.io.hashing import sha256_bytes, sha256_file
from rostrum.score.rubric import (
    build_descriptors,
    confidence_band,
    rubric_score,
    suppressed_from_skipped,
)
from rostrum.text import gaps as G
from rostrum.text import rhetoric as R
from rostrum.types import Alignment, AnalysisResult, OperatingMode, round_time

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.detect.classifier import FlawClassifier
    from rostrum.types import FeatureTable

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisError",
    "LibraryText",
    "ReferenceLibrary",
    "Stage",
    "analyse_id",
    "analyze",
    "load_library",
    "text_similarity",
]

_UNITS: tuple[str, ...] = ("word", "gap", "phrase")


class AnalysisError(RuntimeError):
    """The analysis could not be run at all (bad audio, unusable transcript)."""


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def analyse_id(audio_bytes: bytes, transcript: str, config_hash: str) -> str:
    """``sha256(audio bytes + transcript + config hash)``.

    The cache key and the handle the API hands back, so two uploads of the same
    file with the same transcript under the same configuration are one
    analysis. Everything that can change a single number is in it: the audio,
    the text, and every parameter of every stage.
    """
    return sha256_bytes(
        bytes(audio_bytes) + transcript.encode("utf-8") + str(config_hash).encode("utf-8")
    )


def text_similarity(a: str, b: str) -> float:
    """Normalised similarity of two transcripts, in ``[0, 1]`` (spec 13).

    Both sides are reduced to the aligner's own tokens - lower case, no
    punctuation, numerals spelt out - so a different edition's comma or a
    "1863" against "eighteen sixty three" does not count against the match.
    The ratio is :class:`difflib.SequenceMatcher`'s, computed over tokens
    rather than characters so one misheard word costs one token.
    """
    left = A.normalise_tokens(a)
    right = A.normalise_tokens(b)
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(a=left, b=right, autojunk=False).ratio())


# --------------------------------------------------------------------------- #
# The library an analysis is compared against
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LibraryText:
    """One text the library can analyse in Reference mode.

    Attributes:
        text_id: Identifier, e.g. ``s1_gettysburg``.
        display: The transcript as written, with punctuation and paragraphs;
            the source of every gap label and every context clause.
        verbatim: The aligner's input form, when the text file carries one.
        title: Human-readable title for the dashboard's library screen.
        reference_ids: Skilled readings available for it, sorted.
    """

    text_id: str
    display: str = ""
    verbatim: str = ""
    title: str = ""
    reference_ids: tuple[str, ...] = ()

    @property
    def n_references(self) -> int:
        """Skilled readings behind this text."""
        return len(self.reference_ids)

    @property
    def transcript(self) -> str:
        """The form to match an upload against."""
        return self.display or self.verbatim


@dataclass
class ReferenceLibrary:
    """Everything the pipeline compares an upload against.

    Nothing in here is fitted at analysis time: an envelope, a pooled-norm set,
    a classifier and a ``k_c`` table are all products of a *training* fold, and
    this object is how they reach :func:`analyze` without it being able to
    recompute them (CLAUDE.md invariant 11).

    Attributes:
        fold: The fold these statistics were fitted in, or ``"all"``.
        texts: ``{text_id: LibraryText}``.
        envelopes: ``{text_id: TextEnvelope}``, floors applied.
        reference_tables: ``{text_id: [per-reference {unit: table}]}``, for the
            descriptor ranges of spec 11.
        pooled: Pooled norms for Norm mode, or ``None``.
        classifier: The fold's flaw-type model, or ``None``.
        k_by_category: The fold's calibrated ``k_c``.
        thresholds: The fold's tuned hysteresis thresholds, or ``None`` for the
            configuration's defaults.
        warnings: What could not be loaded, surfaced rather than hidden.
    """

    fold: str = "all"
    texts: dict[str, LibraryText] = field(default_factory=dict)
    envelopes: dict[str, TextEnvelope] = field(default_factory=dict)
    reference_tables: dict[str, list[Mapping[str, Any]]] = field(default_factory=dict)
    pooled: NO.PooledNorms | None = None
    classifier: FlawClassifier | None = None
    k_by_category: dict[str, float] = field(default_factory=dict)
    thresholds: RegionThresholds | None = None
    warnings: list[str] = field(default_factory=list)

    def match(self, transcript: str, *, threshold: float = 0.95) -> tuple[str | None, float]:
        """Best-matching library text for a transcript (spec 13).

        Args:
            transcript: The uploaded transcript.
            threshold: Minimum normalised similarity.

        Returns:
            ``(text_id, score)``; ``(None, best_score)`` when nothing reaches
            the threshold.
        """
        best_id, best = None, 0.0
        for text_id, text in sorted(self.texts.items()):
            score = text_similarity(transcript, text.transcript)
            if score > best:
                best_id, best = text_id, score
        if best_id is not None and best >= float(threshold):
            return best_id, best
        return None, best

    def envelope_for(self, text_id: str | None) -> TextEnvelope | None:
        """The envelope of one text, or ``None``."""
        return self.envelopes.get(str(text_id)) if text_id else None

    def fingerprint(self) -> str:
        """A short digest of what the library holds, for the result cache.

        The analysis id covers the audio, the transcript and the configuration
        but not the dataset, so a cached result must also know which references
        produced it.
        """
        payload = {
            "fold": self.fold,
            "texts": {t: list(self.texts[t].reference_ids) for t in sorted(self.texts)},
            "pooled": list(self.pooled.reference_ids) if self.pooled else [],
            "classifier": self.classifier.fold if self.classifier else "",
            "k": {k: round(float(v), 6) for k, v in sorted(self.k_by_category.items())},
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256_bytes(blob.encode("utf-8"))[:16]


def _read_texts(config: Config) -> dict[str, LibraryText]:
    """Read ``<text_id>.json`` from the dataset, falling back to the sample."""
    out: dict[str, LibraryText] = {}
    for key in ("data.texts", "sample.transcripts"):
        try:
            directory = config.path(key)
        except (KeyError, TypeError):  # pragma: no cover - key always present
            continue
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("could not read text %s: %s", path, exc)
                continue
            text_id = str(payload.get("text_id") or path.stem)
            if text_id in out:
                continue
            out[text_id] = LibraryText(
                text_id=text_id,
                display=str(payload.get("display") or payload.get("verbatim") or ""),
                verbatim=str(payload.get("verbatim") or ""),
                title=str(payload.get("title") or ""),
            )
    return out


def _manifest_references(
    config: Config, *, origins: Sequence[str] = ("clean",)
) -> dict[str, list[str]]:
    """``{text_id: [file_id, ...]}`` of the manifest's clean recordings."""
    path = config.path("data.manifest")
    if not path.is_file():
        return {}
    out: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("origin") not in origins or not row.get("text_id"):
                continue
            out.setdefault(str(row["text_id"]), []).append(str(row["file_id"]))
    return {text: sorted(ids) for text, ids in out.items()}


def _load_reference(
    file_id: str,
    text_id: str,
    config: Config,
) -> Reference | None:
    """Load one reference's cached features, or ``None`` when unavailable.

    The library never *computes* features: an analysis must not pay for the
    dataset, and a missing cache entry means the dataset build has not been run
    for that file, which is a condition to report rather than to repair here.
    """
    audio = config.path("data.audio_clean") / f"{file_id}.wav"
    grid = config.path("data.alignments") / f"{file_id}.TextGrid"
    if not audio.is_file() or not grid.is_file():
        return None
    try:
        digest = sha256_file(audio)
        alignment = M.read_textgrid(
            grid,
            file_id=file_id,
            text_id=text_id,
            speaker_id=file_id,
            channel_id=file_id,
            audio_sha256=digest,
        )
    except (OSError, ValueError) as exc:
        logger.warning("%s: unusable alignment (%s)", file_id, exc)
        return None

    bundle = None
    for options in (
        [*X.OPTIONAL_BLOCKS, "crosscheck"],
        list(X.OPTIONAL_BLOCKS),
    ):
        key = X.cache_key(
            audio_sha256=digest,
            config_hash=config.hash,
            alignment_hash=X.alignment_fingerprint(alignment),
            options=options,
        )
        bundle = X.load_bundle(config.path("cache.features"), key)
        if bundle is not None:
            break
    if bundle is None:
        return None
    return Reference(
        file_id=file_id,
        channel_id=file_id,
        tables={unit: bundle.table(unit) for unit in _UNITS},
    )


def load_library(
    config: Config | None = None,
    *,
    fold: str = "all",
    train_texts: Sequence[str] | None = None,
    with_classifier: bool = True,
    k_by_category: Mapping[str, float] | None = None,
    thresholds: RegionThresholds | None = None,
) -> ReferenceLibrary:
    """Assemble the library from the dataset on disk (spec 13).

    Reads the texts, finds each text's clean recordings in ``MANIFEST.csv``,
    loads their **cached** features, builds the per-text envelopes with the
    sigma floors fitted on ``train_texts`` only, and pools the same references
    into the Norm-mode norms.

    Args:
        config: Configuration; loaded from ``configs/`` when omitted.
        fold: Fold tag recorded on the envelopes.
        train_texts: Texts whose statistics the floors may be fitted on.
            ``None`` means every text present, which is right for a demo and
            wrong inside leave-one-text-out - there the harness passes the
            fold's training texts (CLAUDE.md invariant 11).
        with_classifier: Load ``.cache/models/clf_<fold>.joblib`` if present.
        k_by_category: The fold's calibrated ``k_c``.
        thresholds: The fold's tuned thresholds.

    Returns:
        A :class:`ReferenceLibrary`. Missing pieces are recorded in its
        ``warnings`` rather than raised: a library with no envelope still
        serves Norm mode, and one with neither still serves Reject mode.
    """
    config = config or load_config()
    texts = _read_texts(config)
    by_text = _manifest_references(config)
    warnings: list[str] = []

    references: dict[str, list[Reference]] = {}
    for text_id, file_ids in sorted(by_text.items()):
        loaded = [
            reference
            for reference in (_load_reference(fid, text_id, config) for fid in file_ids)
            if reference is not None
        ]
        if loaded:
            references[text_id] = loaded
        missing = len(file_ids) - len(loaded)
        if missing:
            warnings.append(
                f"{text_id}: {missing} of {len(file_ids)} reference(s) have no cached "
                "features or alignment; run `make data` and `make features`"
            )

    for text_id, loaded in references.items():
        if text_id in texts:
            texts[text_id] = LibraryText(
                **{
                    **vars(texts[text_id]),
                    "reference_ids": tuple(sorted(r.file_id for r in loaded)),
                }
            )

    envelopes: dict[str, TextEnvelope] = {}
    if references:
        flat = [r for group in references.values() for r in group]
        train = list(train_texts) if train_texts is not None else sorted(references)
        try:
            envelopes = build_fold_envelopes(
                {text: [r.file_id for r in group] for text, group in references.items()},
                flat,
                fold=fold,
                train_texts=train,
                config=config,
                skip_unbuildable=True,
            )
        except Exception as exc:  # pragma: no cover - depends on the dataset
            warnings.append(f"no envelope could be built: {exc}")
            logger.warning("no envelope could be built: %s", exc)

    pooled = None
    if references:
        pool_from = [
            r
            for text, group in references.items()
            for r in group
            if train_texts is None or text in set(train_texts)
        ]
        if pool_from:
            pooled = NO.build_pooled_norms(pool_from, config=config)

    classifier = None
    if with_classifier:
        from rostrum.detect.classifier import load_model, model_path

        path = model_path(fold, config)
        if path.is_file():
            try:
                classifier = load_model(path, config=config)
            except Exception as exc:  # pragma: no cover - depends on the model file
                warnings.append(f"classifier {path.name} could not be loaded: {exc}")
        else:
            warnings.append(
                f"no classifier at {path.name}; regions carry the rule label and no severity"
            )

    library = ReferenceLibrary(
        fold=fold,
        texts=texts,
        envelopes=envelopes,
        reference_tables={
            text: [dict(r.tables) for r in group] for text, group in references.items()
        },
        pooled=pooled,
        classifier=classifier,
        k_by_category=dict(k_by_category or {}),
        thresholds=thresholds,
        warnings=warnings,
    )
    logger.info(
        "library [%s]: %d text(s), %d envelope(s), pooled n=%d, classifier=%s",
        fold,
        len(texts),
        len(envelopes),
        pooled.n_references if pooled else 0,
        bool(classifier),
    )
    return library


# --------------------------------------------------------------------------- #
# Stage timings
# --------------------------------------------------------------------------- #


@dataclass
class Stage:
    """One stage of the pipeline and what it cost, for the dashboard (spec 15.2).

    Attributes:
        name: Stage name, in pipeline order.
        seconds: Wall-clock seconds the stage took.
        note: One line about what it did or why it was skipped.
    """

    name: str
    seconds: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {"name": self.name, "seconds": round(float(self.seconds), 3), "note": self.note}


class _Timer:
    """Collects :class:`Stage` records in pipeline order."""

    def __init__(self) -> None:
        self.stages: list[Stage] = []
        self._t0 = time.perf_counter()

    def mark(self, name: str, note: str = "") -> None:
        """Close the current stage and start the next."""
        now = time.perf_counter()
        self.stages.append(Stage(name=name, seconds=now - self._t0, note=note))
        self._t0 = now
        logger.info("%s: %.2f s %s", name, self.stages[-1].seconds, f"({note})" if note else "")

    def to_list(self) -> list[dict[str, Any]]:
        """JSON form of every stage."""
        return [s.to_dict() for s in self.stages]


# --------------------------------------------------------------------------- #
# Curves for the overlay
# --------------------------------------------------------------------------- #


def _curve_settings(config: Config) -> tuple[bool, bool, list[str], int]:
    """``(enabled, dtw, signals, n_points)`` from the configuration."""
    block = dict(config.detection["pipeline"].get("curves", {}))
    envelope_block = dict(config.detection["envelope"].get("curves", {}))
    signals = [str(s) for s in block.get("signals", envelope_block.get("signals", []))]
    return (
        bool(block.get("enabled", True)),
        bool(block.get("dtw", True)),
        signals,
        int(envelope_block.get("n_points", 20)),
    )


def _word_spans(alignment: Alignment) -> dict[int, tuple[float, float]]:
    """``{word_idx: (t_start, t_end)}`` from the alignment."""
    return {int(w.idx): (float(w.t_start), float(w.t_end)) for w in alignment.words}


def _curve_payload(
    bundle: X.FeatureBundle,
    alignment: Alignment,
    curve_envelope: CurveEnvelope | None,
    *,
    signals: Sequence[str],
    n_points: int,
    dtw: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Participant and envelope curves, ready to serialise (spec 15.2).

    The participant's within-word curves are resampled onto the same
    normalised axis as the envelope and then DTW-warped onto it (spec 10.1),
    so the overlay compares the same point of the same word.
    """
    spans = _word_spans(alignment)
    if not signals or bundle.frames is None or len(bundle.frames) == 0 or not spans:
        return {}, {}

    mine = word_curves(bundle.frames, spans, list(signals), n_points=n_points)
    warped = mine
    if curve_envelope is not None:
        warped = warp_to_envelope(mine, curve_envelope, signals=list(signals), dtw=dtw)

    participant = {
        "signals": list(signals),
        "n_points": int(n_points),
        "dtw": bool(dtw and curve_envelope is not None),
        "words": {
            str(idx): {
                "t_start": round_time(spans[idx][0]),
                "t_end": round_time(spans[idx][1]),
                "values": np.asarray(curve, dtype=float).tolist(),
            }
            for idx, curve in sorted(warped.items())
            if idx in spans
        },
    }

    band: dict[str, Any] = {}
    if curve_envelope is not None:
        band = {
            "signals": list(curve_envelope.signals),
            "n_points": int(curve_envelope.n_points),
            "n_references": len(curve_envelope.reference_ids),
            "words": {
                str(idx): {
                    "median": np.asarray(curve_envelope.median[idx], dtype=float).tolist(),
                    "sigma": np.asarray(curve_envelope.sigma[idx], dtype=float).tolist(),
                    "n_references": int(curve_envelope.n_references.get(idx, 0)),
                }
                for idx in sorted(curve_envelope.median)
                if idx in spans
            },
        }
    return participant, band


# --------------------------------------------------------------------------- #
# Detection, scoring and explanation of one comparison
# --------------------------------------------------------------------------- #


@dataclass
class _Detection:
    """The result of comparing one recording with one envelope."""

    regions: list[Any] = field(default_factory=list)
    z_by_unit: Mapping[str, Any] = field(default_factory=dict)
    skipped: list[Any] = field(default_factory=list)
    same_channel: bool = False


def _detect(
    tables: Mapping[str, FeatureTable],
    envelope: TextEnvelope,
    *,
    config: Config,
    file_id: str,
    thresholds: RegionThresholds | None,
    frames: FeatureTable | None,
    refine: bool,
) -> _Detection:
    """z -> signals -> regions -> refinement, for one envelope."""
    z_by_unit = envelope_z(dict(tables), envelope, config=config, file_id=file_id)
    if "word" not in z_by_unit:
        raise AnalysisError(
            f"no word-level comparison could be formed against {envelope.text_id!r}; "
            "the envelope and the recording share no features"
        )
    same_channel = bool(z_by_unit["word"].channel_matched)
    signals, skipped = trigger_signals(z_by_unit, config=config, same_channel=same_channel)
    regions = build_regions(signals, z_by_unit, config=config, thresholds=thresholds)
    if refine:
        RF.refine_regions(regions, frames, config=config)
    return _Detection(
        regions=regions, z_by_unit=z_by_unit, skipped=list(skipped), same_channel=same_channel
    )


def _speech_duration(alignment: Alignment) -> float:
    """``T_speech``: first word onset to last word offset (spec 11)."""
    return float(alignment.speech_duration)


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #


def analyze(
    audio_path: Path | str,
    transcript: str,
    text_id: str | None = None,
    *,
    config: Config | None = None,
    library: ReferenceLibrary | None = None,
    alignment: Alignment | None = None,
    file_id: str = "",
    speaker_id: str = "",
    channel_id: str = "",
    origin: str = "participant",
    check_text_match: bool = True,
    refine: bool = True,
    with_curves: bool = True,
    with_band: bool = True,
    thresholds: RegionThresholds | None = None,
    k_by_category: Mapping[str, float] | None = None,
    cache: bool = True,
    refresh: bool = False,
    work_dir: Path | str | None = None,
) -> AnalysisResult:
    """Analyse one recording against its transcript (spec 13).

    Args:
        audio_path: Any decodable audio file. It is preprocessed to the project
            invariants (16 kHz mono, -23 LUFS, peak <= -1 dBFS) before anything
            measures it, so an upload and a dataset file are treated alike.
        transcript: What the speaker was reading. Punctuation matters: every
            gap label and every context clause comes from it.
        text_id: Force a library text instead of matching by similarity. A
            text with too few readers still falls back to Norm mode.
        config: Configuration; loaded from ``configs/`` when omitted.
        library: What to compare against. Loaded from the dataset when
            omitted; pass an explicit one in a LOTO fold so only that fold's
            training statistics are visible (CLAUDE.md invariant 11).
        alignment: A ready-made alignment, which skips the MFA call. This is
            how the evaluation harness re-analyses dataset files whose
            TextGrids already exist, and the only way to run the pipeline on a
            machine with no conda environment for MFA.
        file_id: Identifier for the recording; the audio's stem by default.
        speaker_id: Speaker identifier, for the channel rule.
        channel_id: Channel identifier; defaults to ``speaker_id``.
        origin: Provenance recorded in the result.
        check_text_match: Run the ASR mismatch check. Switching it off skips
            the only stage that can produce Reject mode, so it is for offline
            runs and tests, never for a user upload.
        refine: Run the ruptures boundary refinement (spec 10.5).
        with_curves: Compute the overlay curves (spec 15.2).
        with_band: Recompute the score with each reference left out, for the
            confidence band (spec 11). Costs one detection pass per reader.
        thresholds: Tuned hysteresis thresholds; the library's, then the
            configuration's, when omitted.
        k_by_category: Calibrated ``k_c``; the library's when omitted.
        cache: Read and write ``.cache/analyses/<analysis_id>.json``.
        refresh: Recompute even when a cache entry exists.
        work_dir: Directory for the processed WAV and the MFA corpus. A
            temporary directory is used and removed when omitted.

    Returns:
        An :class:`~rostrum.types.AnalysisResult`, with ``mode`` set to
        ``reference``, ``norm`` or ``reject``.

    Raises:
        FileNotFoundError: The audio does not exist.
        AnalysisError: The transcript is empty, or no comparison could be
            formed at all.
    """
    config = config or load_config()
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"audio does not exist: {source}")
    if not transcript.strip():
        raise AnalysisError("the transcript is empty; audio alone cannot be analysed")

    timer = _Timer()
    recording_id = file_id or source.stem
    speaker_id = speaker_id or recording_id
    channel_id = channel_id or speaker_id
    library = library if library is not None else load_library(config)
    thresholds = thresholds if thresholds is not None else library.thresholds
    k_values = dict(k_by_category if k_by_category is not None else library.k_by_category)

    analysis = analyse_id(source.read_bytes(), transcript, config.hash)
    cache_path = config.path("cache.analyses") / f"{analysis}.json"
    if cache and not refresh and cache_path.is_file():
        cached = _read_cache(cache_path, library.fingerprint())
        if cached is not None:
            logger.info("%s: served from cache", recording_id)
            return cached

    warnings: list[str] = list(library.warnings)

    with tempfile.TemporaryDirectory(prefix="rostrum_") as tmp:
        work = Path(work_dir) if work_dir is not None else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)

        # -- 1. preprocess -------------------------------------------------
        wav = work / f"{recording_id}.wav"
        processed, stats = preprocess(source, wav)
        digest = sha256_file(processed)
        timer.mark(
            "preprocess",
            f"{stats.duration_s:.1f} s, {stats.lufs:.1f} LUFS" if stats.lufs else "",
        )

        result = AnalysisResult(
            file_id=recording_id,
            text_id=None,
            speaker_id=speaker_id,
            channel_id=channel_id,
            origin=origin,  # type: ignore[arg-type]
            mode="norm",
            duration=float(stats.duration_s),
            features_version=int(config.features.get("version", 1)),
            config_hash=config.hash,
            code_version=code_version(config.root),
            analysis_id=analysis,
        )

        # -- 2. ASR mismatch check (spec 8, 13) ----------------------------
        if check_text_match:
            try:
                errors, heard = A.check_transcript(
                    processed, transcript, config=config, audio_sha256=digest
                )
                result.wer = float(errors.wer)
                limit = A.max_wer(config)
                timer.mark("asr", f"wer={errors.wer:.3f} (limit {limit:.2f})")
                if errors.wer > limit:
                    result.mode = "reject"
                    result.mode_reason = f"wer={errors.wer:.3f} > {limit:.2f}"
                    warnings.append(
                        # The breakdown matters: a WER above 1 is not a bug but
                        # a transcript far shorter than what was spoken, and the
                        # insertion count is what says so.
                        f"The transcript does not match the audio (WER {errors.wer:.2f} = "
                        f"{errors.substitutions} substitution(s), {errors.deletions} "
                        f"deletion(s) and {errors.insertions} insertion(s) over "
                        f"{errors.n_reference} transcript word(s)). The recogniser heard: "
                        + (heard.text[:200] + ("..." if len(heard.text) > 200 else ""))
                    )
                    result.warnings = warnings
                    return _finish(result, timer, cache_path, library, cache=cache)
            except A.ASRError as exc:
                warnings.append(f"transcript mismatch check skipped: {exc}")
                timer.mark("asr", "skipped")
        else:
            timer.mark("asr", "not requested")

        # -- 3. text matching (spec 13) ------------------------------------
        threshold = float(config.detection["pipeline"].get("text_match_threshold", 0.95))
        min_refs = int(config.detection["modes"]["reference"]["min_references"])
        matched, score = (
            (text_id, 1.0) if text_id else library.match(transcript, threshold=threshold)
        )
        if text_id and text_id in library.texts:
            score = text_similarity(transcript, library.texts[text_id].transcript)
        envelope = library.envelope_for(matched)
        mode: OperatingMode = "norm"
        if envelope is not None and envelope.n_references >= min_refs and score >= threshold:
            mode, result.text_id = "reference", matched
            result.mode_reason = (
                f"transcript matches {matched} at {score:.3f} "
                f"with {envelope.n_references} readers"
            )
        else:
            envelope = None
            why = (
                f"no library text matched (best {score:.3f} < {threshold:.2f})"
                if matched is None
                else f"{matched} has too few readings for an envelope (< {min_refs})"
            )
            result.mode_reason = f"{why}; compared against pooled norms"
        result.mode = mode
        timer.mark("text_match", result.mode_reason)

        # The text that gap labels and context clauses come from: the library's
        # edition when this is a rendition of one, the upload's own otherwise.
        library_text = library.texts.get(str(matched)) if matched else None
        display = library_text.transcript if (mode == "reference" and library_text) else transcript
        annotation = R.annotate(display, text_id=str(result.text_id or ""), config=config)

        # -- 4. alignment (MFA only; invariant 4) --------------------------
        if alignment is None:
            try:
                alignment = _align(
                    processed,
                    transcript,
                    config=config,
                    work=work,
                    file_id=recording_id,
                    text_id=result.text_id,
                    speaker_id=speaker_id,
                    channel_id=channel_id,
                    audio_sha256=digest,
                    duration=float(stats.duration_s),
                )
            except (M.MFAError, FileNotFoundError, ValueError) as exc:
                result.mode = "reject"
                result.mode_reason = f"alignment failed: {exc}"
                warnings.append(f"forced alignment failed: {exc}")
                result.warnings = warnings
                timer.mark("align", "failed")
                return _finish(result, timer, cache_path, library, cache=cache)
            timer.mark("align", f"{alignment.n_words} words")
        else:
            alignment = _relabel(alignment, file_id=recording_id, text_id=result.text_id)
            timer.mark("align", "supplied")

        # Gap labels come from the text's punctuation, never from the audio.
        alignment.gaps = PA.apply_gap_labels(
            alignment.gaps,
            G.gap_label_map(display, n_gaps=len(alignment.gaps), config=config),
        )
        if alignment.oov_words:
            warnings.append(
                f"{len(alignment.oov_words)} out-of-vocabulary word(s): "
                + ", ".join(alignment.oov_words[:10])
            )
        result.alignment = alignment
        result.speech_duration = _speech_duration(alignment)

        # -- 5. features ---------------------------------------------------
        import librosa

        samples, sr = librosa.load(str(processed), sr=config.sample_rate, mono=True)
        bundle, from_cache = X.extract_all_cached(
            np.asarray(samples, dtype=np.float32),
            int(sr),
            alignment,
            config=config,
            audio_sha256=digest,
            crosscheck=False,
        )
        tables = {unit: bundle.table(unit) for unit in _UNITS}
        timer.mark("features", "from cache" if from_cache else bundle.summary())

        # -- 5b. what to compare against -----------------------------------
        if mode == "reference" and envelope is not None:
            comparison = envelope
            reference_tables = library.reference_tables.get(str(result.text_id), [])
        else:
            if library.pooled is None:
                result.mode = "reject"
                result.mode_reason = (
                    "no envelope for this text and no pooled norms in the library; "
                    "nothing to compare against"
                )
                warnings.append(result.mode_reason)
                result.warnings = warnings
                timer.mark("compare", "impossible")
                return _finish(result, timer, cache_path, library, cache=cache)
            comparison = NO.pooled_envelope(library.pooled, tables, fold=library.fold)
            reference_tables = []
            warnings.append(
                "Norm mode: the comparison is against pooled norms over all skilled "
                "readings, not this text's own readers. Clarity and voice quality are "
                "measured but cannot raise a flaw (spec 9.3)."
            )

        result.n_references = comparison.n_references
        result.reference_ids = list(comparison.reference_ids)

        # -- 6. detection, scoring, explanation ----------------------------
        detection = _detect(
            tables,
            comparison,
            config=config,
            file_id=recording_id,
            thresholds=thresholds,
            frames=bundle.frames,
            refine=refine,
        )
        timer.mark("detect", f"{len(detection.regions)} region(s)")

        if library.classifier is not None:
            try:
                library.classifier.apply(detection.regions, detection.z_by_unit)
                timer.mark("classify", f"{len(detection.regions)} region(s)")
            except Exception as exc:  # pragma: no cover - depends on sklearn
                warnings.append(f"classifier skipped: {exc}")
                timer.mark("classify", "skipped")
        else:
            warnings.append(
                "No flaw-type classifier was available, so every region carries the rule "
                "label and no calibrated severity."
            )
            timer.mark("classify", "no model")

        suppressed = suppressed_from_skipped(detection.skipped, config=config)
        for item in detection.skipped:
            warnings.append(f"{item.label} not evaluated: {item.detail or item.reason}")
        descriptor_block = build_descriptors(tables, reference_tables=reference_tables)
        score = rubric_score(
            detection.regions,
            result.speech_duration,
            k_by_category=k_values,
            config=config,
            suppressed=suppressed,
            descriptor_block=descriptor_block,
        )
        if with_band and mode == "reference" and comparison.n_references >= 2:
            score = confidence_band(
                score,
                _band_variants(
                    tables,
                    comparison,
                    config=config,
                    file_id=recording_id,
                    thresholds=thresholds,
                    frames=bundle.frames,
                    refine=refine,
                    speech_duration=result.speech_duration,
                    k_values=k_values,
                    suppressed=suppressed,
                ),
            )
        result.score = score
        timer.mark("score", f"overall {score.overall:.2f} +/- {score.overall_band:.2f}")

        cards = build_cards(
            detection.regions,
            annotation=annotation,
            envelope=comparison,
            z_by_unit=detection.z_by_unit,
            mode=result.mode,
            config=config,
        )
        result.regions = list(detection.regions)
        result.cards = [card.to_dict() for card in cards]
        timer.mark("explain", f"{len(cards)} card(s)")

        # -- 7. curves ------------------------------------------------------
        if with_curves:
            enabled, dtw, signals, n_points = _curve_settings(config)
            if enabled:
                participant, band = _curve_payload(
                    bundle,
                    alignment,
                    comparison.curves if mode == "reference" else None,
                    signals=signals,
                    n_points=n_points,
                    dtw=dtw,
                )
                result.curves = participant
                result.envelope_curves = band
                timer.mark("curves", f"{len(participant.get('words', {}))} word(s)")

    # -- 8. round, cache, return -------------------------------------------
    result.warnings = warnings
    return _finish(result, timer, cache_path, library, cache=cache)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _align(
    wav: Path,
    transcript: str,
    *,
    config: Config,
    work: Path,
    file_id: str,
    text_id: str | None,
    speaker_id: str,
    channel_id: str,
    audio_sha256: str,
    duration: float,
) -> Alignment:
    """Align one file with MFA in its own corpus directory (spec 8)."""
    corpus = work / "corpus"
    out = work / "aligned"
    staged = M.prepare_corpus([(wav, transcript, file_id)], corpus)
    grids = M.run_mfa(
        corpus,
        out,
        acoustic=str(config.paths["mfa"]["acoustic_model"]),
        dictionary=str(config.paths["mfa"]["dictionary"]),
        extra_dict=config.paths["mfa"].get("extra_dict"),
        env_name=str(config.paths["mfa"]["env_name"]),
        num_jobs=int(config.paths["mfa"]["num_jobs"]),
    )
    return M.read_textgrid(
        grids[0],
        file_id=file_id,
        text=staged[file_id].text,
        text_id=text_id,
        speaker_id=speaker_id,
        channel_id=channel_id,
        aligner="mfa",
        acoustic_model=str(config.paths["mfa"]["acoustic_model"]),
        dictionary=str(config.paths["mfa"]["dictionary"]),
        audio_sha256=audio_sha256,
        duration=duration,
    )


def _relabel(alignment: Alignment, *, file_id: str, text_id: str | None) -> Alignment:
    """Stamp a supplied alignment with this analysis's identifiers."""
    alignment.file_id = alignment.file_id or file_id
    if text_id:
        alignment.text_id = text_id
    return alignment


def _band_variants(
    tables: Mapping[str, FeatureTable],
    envelope: TextEnvelope,
    *,
    config: Config,
    file_id: str,
    thresholds: RegionThresholds | None,
    frames: FeatureTable | None,
    refine: bool,
    speech_duration: float,
    k_values: Mapping[str, float],
    suppressed: Mapping[str, str],
) -> list[Any]:
    """One score per reference left out, for the confidence band (spec 11)."""
    variants = []
    for reference_id in envelope.reference_ids:
        reduced = leave_one_out(envelope, reference_id)
        try:
            detection = _detect(
                tables,
                reduced,
                config=config,
                file_id=file_id,
                thresholds=thresholds,
                frames=frames,
                refine=refine,
            )
        except AnalysisError:  # pragma: no cover - a thin envelope
            continue
        variants.append(
            rubric_score(
                detection.regions,
                speech_duration,
                k_by_category=k_values,
                config=config,
                suppressed=dict(suppressed),
            )
        )
    logger.info("confidence band from %d leave-one-reference-out run(s)", len(variants))
    return variants


def _read_cache(path: Path, fingerprint: str) -> AnalysisResult | None:
    """Read a cached analysis, or ``None`` when it is stale or unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - corrupt cache
        logger.warning("ignoring unreadable analysis cache %s: %s", path, exc)
        return None
    if payload.get("library_fingerprint") != fingerprint:
        logger.info("cache %s was built against a different library; recomputing", path.name)
        return None
    try:
        return AnalysisResult.from_dict(payload["result"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - corrupt cache
        logger.warning("ignoring unusable analysis cache %s: %s", path, exc)
        return None


def _finish(
    result: AnalysisResult,
    timer: _Timer,
    cache_path: Path,
    library: ReferenceLibrary,
    *,
    cache: bool,
) -> AnalysisResult:
    """Record the stage timings, write the cache entry and return the result.

    The timings are *not* part of the serialised result - they are wall-clock
    numbers and would break the determinism hash - so they are written beside
    it in the cache entry and logged.
    """
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "library_fingerprint": library.fingerprint(),
                    "stages": timer.to_list(),
                    "result": result.to_dict(),
                },
                indent=2,
                sort_keys=False,
            )
            + "\n",
            encoding="utf-8",
        )
    logger.info(
        "%s: mode=%s, %d region(s), overall %s",
        result.file_id,
        result.mode,
        len(result.regions),
        f"{result.score.overall:.2f}" if result.score else "n/a",
    )
    return result


def stages_of(analysis_id: str, config: Config | None = None) -> list[dict[str, Any]]:
    """Stage timings of a cached analysis, for the dashboard's processing screen."""
    config = config or load_config()
    path = config.path("cache.analyses") / f"{analysis_id}.json"
    if not path.is_file():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("stages", []))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt cache
        return []


def analyze_many(
    items: Iterable[tuple[Path | str, str, str | None]],
    **kwargs: Any,
) -> list[AnalysisResult]:
    """Analyse several ``(audio, transcript, text_id)`` triples with one library."""
    config = kwargs.pop("config", None) or load_config()
    library = kwargs.pop("library", None) or load_library(config)
    return [
        analyze(audio, transcript, text_id, config=config, library=library, **kwargs)
        for audio, transcript, text_id in items
    ]
