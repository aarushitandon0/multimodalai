"""Realism gates for injected files: re-alignment and UTMOS (spec 5.1, 5.4).

An edit that moves a word boundary, or that a listener hears as a *recording*
artifact rather than as a *delivery* flaw, would teach the detector the wrong
thing. Two gates stand between the injection engine and the dataset
(``CLAUDE.md`` invariant 9):

1. **Re-alignment.** Run MFA on the edited WAV and compare every word boundary
   against the expectation - the clean alignment pushed through the op's
   :class:`~rostrum.inject.timemap.TimeMap`. Spec 5.1 sets a **20 ms agreement
   target** (reported, not enforced); spec 5.4 **rejects** any file whose worst
   word boundary moved by more than **50 ms**. A file that fails this gate has
   a label whose timestamps nobody can trust, so it cannot be ground truth.
2. **UTMOS.** Predicted naturalness MOS [R19] of the edited file against its own
   clean source. At L1 the edit is meant to be barely perceptible, so a large
   MOS drop means the *artifact* became the flaw. Permitted drops per level
   live in ``injection.qc.utmos.max_drop_by_level``.

Both thresholds, and the per-level UTMOS budget, come from
``configs/injection.yaml`` - never from this module.

Division of labour: the comparison arithmetic and the gate decisions are pure
functions over :class:`~rostrum.types.Alignment` objects and floats, so the
test suite covers them without the ``mfa`` conda env and without downloading a
checkpoint. The two outside-world dependencies are injected:

* ``align_fn`` - defaults to :func:`mfa_realign`, which shells out to MFA;
* ``scorer`` - defaults to :func:`utmos_scorer`, which loads the pinned
  ``torch.hub`` model.

UTMOS model provenance (verified against the package README on 2026-10-06):
``torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)``
returns a predictor called as ``predictor(wave, sr=16000)`` with ``wave`` of
shape ``(batch, time)``, returning one MOS per item. The tag is pinned so a
re-run scores the same files the same way; note that PyPI's ``speechmos``
package is Microsoft's AECMOS/DNSMOS/PLCMOS and does **not** contain UTMOS.

Scores are cached in ``dataset/qc/utmos.csv``, so the model is loaded only for
files that have never been scored - and never at inference time (invariant 13).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rostrum.align.mfa import prepare_corpus, read_textgrid, run_mfa
from rostrum.config import Config, load_config
from rostrum.data.manifest import append_manifest, read_manifest
from rostrum.types import Alignment

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

UTMOS_HUB_REPO = "tarepan/SpeechMOS:v1.2.0"
"""Pinned ``torch.hub`` source of the UTMOS predictor [R19]."""

UTMOS_MODEL = "utmos22_strong"
"""Entry point inside :data:`UTMOS_HUB_REPO`; the strong UTMOS22 system."""

UTMOS_SAMPLE_RATE = 16_000
"""Sample rate UTMOS is called with; the dataset is 16 kHz (invariant 1)."""

REALIGN_COLUMNS: tuple[str, ...] = (
    "file_id",
    "source_file",
    "label",
    "level",
    "n_words",
    "max_err_ms",
    "mean_err_ms",
    "p95_err_ms",
    "frac_within_target",
    "target_ms",
    "max_allowed_ms",
    "passed",
)
"""Columns of ``dataset/qc/realign.csv`` (spec 7.4)."""

UTMOS_COLUMNS: tuple[str, ...] = ("file_id", "utmos", "model", "sr")
"""Columns of ``dataset/qc/utmos.csv`` (spec 7.4)."""

REJECTION_COLUMNS: tuple[str, ...] = (
    "file_id",
    "source_file",
    "label",
    "level",
    "stage",
    "reason",
)
"""Columns of ``dataset/qc/rejections.csv``; shared with ``build_injected.py``."""

EXCLUDED_FOLD = "excluded"
"""``MANIFEST.csv`` ``split_fold`` value for a file that failed QC (spec 7.3)."""

_MS = 1000.0


class Aligner(Protocol):
    """Callable that re-aligns one edited WAV (see :func:`mfa_realign`)."""

    def __call__(
        self, wav_path: Path, transcript: str, file_id: str, work_dir: Path
    ) -> Alignment: ...


Scorer = Callable[[Path], float]
"""Callable mapping a WAV path to a predicted MOS."""


def _cfg(cfg: Config | None) -> Config:
    """Return ``cfg``, loading ``configs/*.yaml`` when none was passed."""
    return cfg if cfg is not None else load_config(apply_seeds=False)


def _qc_cfg(cfg: Config | None) -> Mapping[str, Any]:
    """Return the ``injection.qc`` section."""
    return _cfg(cfg).injection["qc"]


# --------------------------------------------------------------------------- #
# Re-alignment comparison (pure)
# --------------------------------------------------------------------------- #


def boundary_errors_ms(expected: Alignment, realigned: Alignment) -> np.ndarray:
    """Per-word boundary error in milliseconds between two alignments.

    A word contributes **one** number: the worse of its onset and offset
    displacement, which is the quantity spec 5.4 thresholds ("boundary error >
    50 ms on any word").

    Both alignments must describe the same words in the same order. No op in
    :mod:`rostrum.inject.ops` deletes or inserts a word, so a word-count or
    word-text mismatch means MFA re-segmented the edited audio - the file is
    unusable rather than merely inaccurate, and that is an error, not a large
    number.

    Args:
        expected: Where the words should be - the clean alignment pushed
            through the op's time map (see
            :func:`rostrum.inject.engine.remap_alignment`).
        realigned: MFA's alignment of the edited WAV.

    Returns:
        Array of shape ``(n_words,)`` of non-negative errors in ms.

    Raises:
        ValueError: The alignments disagree on the number of words or on a
            word's text.
    """
    exp, got = expected.words, realigned.words
    if len(exp) != len(got):
        raise ValueError(
            f"word count differs: expected {len(exp)} words, re-alignment has {len(got)}"
        )
    for i, (a, b) in enumerate(zip(exp, got, strict=True)):
        if a.text.casefold() != b.text.casefold():
            raise ValueError(f"word {i} differs: expected {a.text!r}, re-aligned {b.text!r}")

    onset = np.array([abs(a.t_start - b.t_start) for a, b in zip(exp, got, strict=True)])
    offset = np.array([abs(a.t_end - b.t_end) for a, b in zip(exp, got, strict=True)])
    return np.maximum(onset, offset) * _MS


@dataclass(frozen=True)
class RealignReport:
    """Boundary agreement between the expected and the re-aligned times.

    Attributes:
        file_id: The edited file this describes.
        source_file: Its clean source ``file_id``.
        label: Flaw label, or ``"composite"``; ``""`` when unknown.
        level: Severity ``L1``-``L4``; ``""`` when unknown.
        n_words: Words compared.
        max_err_ms: Worst per-word boundary error; the gated quantity.
        mean_err_ms: Mean per-word boundary error.
        p95_err_ms: 95th percentile (numpy linear interpolation).
        frac_within_target: Fraction of words inside :attr:`target_ms`, the
            spec 5.1 agreement target.
        target_ms: Reported-only agreement target, 20 ms.
        max_allowed_ms: Rejection threshold, 50 ms (spec 5.4).
    """

    file_id: str
    source_file: str
    label: str
    level: str
    n_words: int
    max_err_ms: float
    mean_err_ms: float
    p95_err_ms: float
    frac_within_target: float
    target_ms: float
    max_allowed_ms: float

    @property
    def passed(self) -> bool:
        """True when no word moved further than :attr:`max_allowed_ms`."""
        return self.max_err_ms <= self.max_allowed_ms

    def to_row(self) -> dict[str, str]:
        """Render as a ``dataset/qc/realign.csv`` row."""
        return {
            "file_id": self.file_id,
            "source_file": self.source_file,
            "label": self.label,
            "level": self.level,
            "n_words": str(self.n_words),
            "max_err_ms": f"{self.max_err_ms:.2f}",
            "mean_err_ms": f"{self.mean_err_ms:.2f}",
            "p95_err_ms": f"{self.p95_err_ms:.2f}",
            "frac_within_target": f"{self.frac_within_target:.4f}",
            "target_ms": f"{self.target_ms:.1f}",
            "max_allowed_ms": f"{self.max_allowed_ms:.1f}",
            "passed": "true" if self.passed else "false",
        }


def realign_report(
    expected: Alignment,
    realigned: Alignment,
    *,
    file_id: str | None = None,
    source_file: str = "",
    label: str = "",
    level: str = "",
    cfg: Config | None = None,
) -> RealignReport:
    """Summarise :func:`boundary_errors_ms` into the gated statistics.

    Args:
        expected: Expected (time-map transformed) alignment.
        realigned: MFA's alignment of the edited WAV.
        file_id: Identifier for the report; defaults to ``realigned.file_id``.
        source_file: Clean source ``file_id``, for the CSV.
        label: Flaw label, for the CSV.
        level: Severity level, for the CSV.
        cfg: Config holding ``injection.qc.realign``; loaded when omitted.

    Returns:
        A :class:`RealignReport`.

    Raises:
        ValueError: The alignments are not comparable (see
            :func:`boundary_errors_ms`).
    """
    rc = _qc_cfg(cfg)["realign"]
    errors = boundary_errors_ms(expected, realigned)
    target = float(rc["target_boundary_err_ms"])
    return RealignReport(
        file_id=file_id if file_id is not None else realigned.file_id,
        source_file=source_file,
        label=label,
        level=level,
        n_words=int(errors.size),
        max_err_ms=float(errors.max()),
        mean_err_ms=float(errors.mean()),
        p95_err_ms=float(np.percentile(errors, 95)),
        frac_within_target=float(np.mean(errors <= target)),
        target_ms=target,
        max_allowed_ms=float(rc["max_boundary_err_ms"]),
    )


# --------------------------------------------------------------------------- #
# Re-alignment via MFA
# --------------------------------------------------------------------------- #


def mfa_realign(wav_path: Path, transcript: str, file_id: str, work_dir: Path) -> Alignment:
    """Align one WAV with MFA and read the TextGrid back (invariant 4).

    A corpus of one file is staged under ``work_dir`` and aligned on its own:
    the files QC checks are generated one at a time, and a per-file corpus
    keeps a failure attributable to the file that caused it.

    Args:
        wav_path: The edited WAV.
        transcript: Display transcript of the text.
        file_id: Identifier; becomes the staged basename.
        work_dir: Scratch directory for the corpus and the TextGrid.

    Returns:
        The :class:`~rostrum.types.Alignment` MFA produced.

    Raises:
        rostrum.align.mfa.MFANotAvailableError: conda or the env is missing.
        rostrum.align.mfa.MFAError: Alignment failed.
    """
    cfg = load_config(apply_seeds=False)
    mfa_cfg = cfg.paths["mfa"]
    corpus_dir = Path(work_dir) / "corpus"
    out_dir = Path(work_dir) / "aligned"

    staged = prepare_corpus([(wav_path, transcript, file_id)], corpus_dir)
    grids = run_mfa(
        corpus_dir,
        out_dir,
        acoustic=str(mfa_cfg["acoustic_model"]),
        dictionary=str(mfa_cfg["dictionary"]),
        extra_dict=mfa_cfg["extra_dict"],
        env_name=str(mfa_cfg["env_name"]),
        num_jobs=int(mfa_cfg["num_jobs"]),
    )
    return read_textgrid(
        grids[0],
        file_id=file_id,
        text=staged[file_id].text,
        aligner="mfa",
        acoustic_model=str(mfa_cfg["acoustic_model"]),
        dictionary=str(mfa_cfg["dictionary"]),
    )


def realign_check(
    edited_wav: Path | str,
    transcript: str,
    expected_alignment: Alignment,
    *,
    source_file: str = "",
    label: str = "",
    level: str = "",
    work_dir: Path | str | None = None,
    align_fn: Aligner | None = None,
    cfg: Config | None = None,
) -> RealignReport:
    """Re-align an edited file and report its boundary agreement (spec 5.1).

    Args:
        edited_wav: The generated WAV. Its stem is the ``file_id``.
        transcript: Display transcript, as handed to the clean alignment.
        expected_alignment: Where the words should be - the clean alignment
            pushed through the op's time map.
        source_file: Clean source ``file_id``, recorded in the report.
        label: Flaw label, recorded in the report.
        level: Severity level, recorded in the report.
        work_dir: Scratch directory for the aligner; defaults to
            ``.cache/mfa/realign/<file_id>``.
        align_fn: Aligner to use; defaults to :func:`mfa_realign`.
        cfg: Config; loaded when omitted.

    Returns:
        A :class:`RealignReport` for the file.

    Raises:
        FileNotFoundError: ``edited_wav`` does not exist.
        ValueError: The re-alignment is not comparable word-for-word.
    """
    wav = Path(edited_wav)
    if not wav.is_file():
        raise FileNotFoundError(f"edited WAV not found: {wav}")
    conf = _cfg(cfg)
    file_id = wav.stem
    scratch = (
        Path(work_dir)
        if work_dir is not None
        else conf.path("cache.dir") / "mfa" / "realign" / file_id
    )
    scratch.mkdir(parents=True, exist_ok=True)

    aligner: Aligner = align_fn if align_fn is not None else mfa_realign
    realigned = aligner(wav, transcript, file_id, scratch)
    report = realign_report(
        expected_alignment,
        realigned,
        file_id=file_id,
        source_file=source_file,
        label=label,
        level=level,
        cfg=conf,
    )
    logger.debug(
        "%s: realign max %.1f ms, p95 %.1f ms, %.0f%% within %.0f ms",
        file_id,
        report.max_err_ms,
        report.p95_err_ms,
        100.0 * report.frac_within_target,
        report.target_ms,
    )
    return report


# --------------------------------------------------------------------------- #
# UTMOS
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def load_utmos(repo: str = UTMOS_HUB_REPO, model: str = UTMOS_MODEL) -> Any:
    """Load the pinned UTMOS predictor on CPU, once per process.

    The tag in :data:`UTMOS_HUB_REPO` is pinned so the same audio scores the
    same on a re-run. The first call downloads the checkpoint into
    ``TORCH_HOME``; every later call - and every cached score - needs no
    network (invariant 13).

    Args:
        repo: ``torch.hub`` source, ``owner/repo:tag``.
        model: Entry point inside that repo.

    Returns:
        A predictor in ``eval`` mode, callable as ``predictor(wave, sr=16000)``
        with ``wave`` of shape ``(batch, time)``.

    Raises:
        ImportError: ``torch`` is not installed.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        raise ImportError("UTMOS scoring needs torch; `pip install torch`") from exc

    logger.info("loading UTMOS predictor %s from %s (CPU)", model, repo)
    predictor = torch.hub.load(repo, model, trust_repo=True)
    predictor = predictor.to("cpu")
    predictor.eval()
    return predictor


def utmos_score(samples: np.ndarray, sr: int, *, predictor: Any | None = None) -> float:
    """Predicted MOS of one mono signal.

    Args:
        samples: Mono float waveform.
        sr: Sample rate; must be :data:`UTMOS_SAMPLE_RATE`.
        predictor: A loaded predictor; :func:`load_utmos` is used when omitted.

    Returns:
        The predicted MOS.

    Raises:
        ValueError: ``sr`` is not 16 kHz, or ``samples`` is not mono/non-empty.
    """
    import torch

    if sr != UTMOS_SAMPLE_RATE:
        raise ValueError(f"UTMOS expects {UTMOS_SAMPLE_RATE} Hz audio, got {sr}")
    wave = np.asarray(samples, dtype=np.float32)
    if wave.ndim != 1 or wave.size == 0:
        raise ValueError(f"expected a non-empty mono signal, got shape {wave.shape}")

    model = predictor if predictor is not None else load_utmos()
    with torch.no_grad():
        out = model(torch.from_numpy(wave)[None, :], sr=UTMOS_SAMPLE_RATE)
    return float(np.asarray(out.detach().cpu().numpy()).reshape(-1)[0])


def utmos_scorer(predictor: Any | None = None) -> Scorer:
    """Return a :data:`Scorer` that reads a WAV off disk and scores it.

    ``soundfile`` reads the WAV directly - the dataset is already 16 kHz mono
    PCM (invariant 1), so no resampling, and therefore no ``librosa``, is
    involved.

    Args:
        predictor: A loaded predictor; :func:`load_utmos` is called lazily on
            the first file when omitted.

    Returns:
        A callable mapping a WAV path to its predicted MOS.
    """
    import soundfile as sf

    state: dict[str, Any] = {"predictor": predictor}

    def score(path: Path) -> float:
        samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if samples.ndim > 1:
            raise ValueError(f"{path} is not mono (shape {samples.shape})")
        if state["predictor"] is None:
            state["predictor"] = load_utmos()
        return utmos_score(samples, int(sr), predictor=state["predictor"])

    return score


def read_utmos_cache(path: Path | str) -> dict[str, float]:
    """Read ``dataset/qc/utmos.csv``; ``{}`` when the file does not exist."""
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8", newline="") as fh:
        return {row["file_id"]: float(row["utmos"]) for row in csv.DictReader(fh)}


def append_utmos(path: Path | str, scores: Mapping[str, float]) -> int:
    """Merge ``{file_id: mos}`` into the UTMOS cache; return the row count.

    Rows are rewritten sorted by ``file_id`` so the cache is byte-identical
    whatever order the files were scored in (invariant 13).

    Args:
        path: Destination CSV; created with a header if absent.
        scores: Scores to add; an existing ``file_id`` is overwritten.

    Returns:
        Total rows in the cache after the merge.
    """
    p = Path(path)
    merged = read_utmos_cache(p)
    merged.update({k: float(v) for k, v in scores.items()})
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=UTMOS_COLUMNS)
        writer.writeheader()
        for file_id in sorted(merged):
            writer.writerow(
                {
                    "file_id": file_id,
                    "utmos": f"{merged[file_id]:.4f}",
                    "model": UTMOS_MODEL,
                    "sr": str(UTMOS_SAMPLE_RATE),
                }
            )
    return len(merged)


def utmos_for_files(
    wav_paths: Iterable[Path | str],
    *,
    cache_path: Path | str,
    scorer: Scorer | None = None,
) -> dict[str, float]:
    """Return ``{file_id: mos}`` for ``wav_paths``, scoring only what is new.

    The cache is the reason this function exists: UTMOS is the slowest step in
    the dataset build, and re-running ``make data`` must not re-score files
    that have not changed.

    Args:
        wav_paths: WAVs to score; the ``file_id`` is each path's stem.
        cache_path: ``dataset/qc/utmos.csv``.
        scorer: Callable scoring one WAV; :func:`utmos_scorer` is built lazily,
            and only if at least one file is missing from the cache.

    Returns:
        ``{file_id: mos}`` for every requested file, in cache-then-scored order.
    """
    paths = [Path(p) for p in wav_paths]
    cached = read_utmos_cache(cache_path)
    missing = [p for p in paths if p.stem not in cached]

    fresh: dict[str, float] = {}
    if missing:
        score = scorer if scorer is not None else utmos_scorer()
        for path in missing:
            fresh[path.stem] = float(score(path))
            logger.debug("utmos %s = %.3f", path.stem, fresh[path.stem])
        append_utmos(cache_path, fresh)

    return {p.stem: cached.get(p.stem, fresh.get(p.stem, float("nan"))) for p in paths}


# --------------------------------------------------------------------------- #
# Gates (spec 5.4)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QCVerdict:
    """The outcome of both gates for one generated file.

    Attributes:
        file_id: The file judged.
        realign: Its boundary-agreement report, or ``None`` when re-alignment
            is disabled in the config.
        utmos_clean: MOS of the clean source, or ``None`` when not scored.
        utmos_edited: MOS of the edited file, or ``None`` when not scored.
        level: Severity level, which picks the permitted UTMOS drop.
        max_utmos_drop: The permitted drop at :attr:`level`, or ``None`` when
            the UTMOS gate does not apply.
        reasons: One string per failed gate; empty when the file passed.
    """

    file_id: str
    realign: RealignReport | None
    utmos_clean: float | None
    utmos_edited: float | None
    level: str
    max_utmos_drop: float | None
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True when no gate reported a reason to reject the file."""
        return not self.reasons

    @property
    def utmos_drop(self) -> float | None:
        """``utmos_clean - utmos_edited``, or ``None`` when either is missing."""
        if self.utmos_clean is None or self.utmos_edited is None:
            return None
        return self.utmos_clean - self.utmos_edited

    def to_dict(self) -> dict[str, Any]:
        """Render the ``qc`` block of the label JSON (spec 7.1)."""
        drop = self.utmos_drop
        return {
            "utmos_clean": _round(self.utmos_clean, 3),
            "utmos_edited": _round(self.utmos_edited, 3),
            "utmos_drop": _round(drop, 3),
            "max_utmos_drop": _round(self.max_utmos_drop, 3),
            "realign_max_err_ms": _round(self.realign.max_err_ms, 1) if self.realign else None,
            "realign_mean_err_ms": _round(self.realign.mean_err_ms, 1) if self.realign else None,
            "realign_p95_err_ms": _round(self.realign.p95_err_ms, 1) if self.realign else None,
            "realign_frac_within_target": (
                _round(self.realign.frac_within_target, 4) if self.realign else None
            ),
            "utmos_model": UTMOS_MODEL if self.utmos_edited is not None else None,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def _round(x: float | None, ndigits: int) -> float | None:
    """Round for output, keeping ``None`` as ``None``."""
    return None if x is None else round(float(x), ndigits)


def gate(
    *,
    realign: RealignReport | None,
    utmos_clean: float | None,
    utmos_edited: float | None,
    level: str,
    file_id: str = "",
    cfg: Config | None = None,
) -> QCVerdict:
    """Apply the spec 5.4 realism gates to one generated file.

    Reject when either holds:

    * any word boundary moved by more than ``qc.realign.max_boundary_err_ms``
      (50 ms), or the re-alignment was not comparable word-for-word;
    * the UTMOS drop exceeds ``qc.utmos.max_drop_by_level[level]``. Spec 5.4
      names the L1 budget (0.3); the config carries one per level, and a level
      absent from it is not gated on naturalness.

    A missing UTMOS score is never a rejection - an unscored file is reported
    as unscored rather than silently dropped.

    Args:
        realign: Boundary-agreement report, or ``None`` if the gate is off.
        utmos_clean: MOS of the clean source, or ``None``.
        utmos_edited: MOS of the edited file, or ``None``.
        level: Severity level ``L1``-``L4``.
        file_id: Identifier for the verdict; taken from ``realign`` when empty.
        cfg: Config holding ``injection.qc``; loaded when omitted.

    Returns:
        A :class:`QCVerdict`.
    """
    qc_cfg = _qc_cfg(cfg)
    reasons: list[str] = []

    if realign is not None and bool(qc_cfg["realign"]["enabled"]) and not realign.passed:
        reasons.append(
            f"realign boundary error {realign.max_err_ms:.1f} ms "
            f"> {realign.max_allowed_ms:.0f} ms (spec 5.4)"
        )

    utmos_cfg = qc_cfg["utmos"]
    budget = utmos_cfg["max_drop_by_level"].get(level) if bool(utmos_cfg["enabled"]) else None
    max_drop = None if budget is None else float(budget)
    if (
        max_drop is not None
        and utmos_clean is not None
        and utmos_edited is not None
        and (utmos_clean - utmos_edited) > max_drop
    ):
        reasons.append(
            f"utmos drop {utmos_clean - utmos_edited:.2f} > {max_drop:.2f} at {level} (spec 5.4)"
        )

    return QCVerdict(
        file_id=file_id or (realign.file_id if realign is not None else ""),
        realign=realign,
        utmos_clean=utmos_clean,
        utmos_edited=utmos_edited,
        level=level,
        max_utmos_drop=max_drop,
        reasons=tuple(reasons),
    )


def unalignable_verdict(
    *, file_id: str, level: str, reason: str, cfg: Config | None = None
) -> QCVerdict:
    """A rejection for a file whose re-alignment could not even be compared.

    :func:`boundary_errors_ms` raises when MFA re-segmented the edited audio
    into a different word sequence. That is a QC failure, not a crash, so the
    caller turns the exception into this verdict and the file is logged like
    any other rejection.

    Args:
        file_id: The file that failed.
        level: Its severity level.
        reason: The exception message.
        cfg: Unused today; accepted so callers can pass their config.

    Returns:
        A failed :class:`QCVerdict` carrying ``reason``.
    """
    return QCVerdict(
        file_id=file_id,
        realign=None,
        utmos_clean=None,
        utmos_edited=None,
        level=level,
        max_utmos_drop=None,
        reasons=(f"realign not comparable: {reason}",),
    )


# --------------------------------------------------------------------------- #
# Label JSON and MANIFEST
# --------------------------------------------------------------------------- #


def label_of(label: Mapping[str, Any]) -> str:
    """Flaw label of a label JSON: the single label, or ``"composite"``."""
    flaws = label.get("flaws") or []
    names = {str(f.get("label", "")) for f in flaws}
    if len(names) == 1:
        return names.pop()
    return "composite" if names else ""


def level_of(label: Mapping[str, Any]) -> str:
    """Severity level of a label JSON: the shared level of its flaws."""
    flaws = label.get("flaws") or []
    levels = {str(f.get("level", "")) for f in flaws}
    if len(levels) == 1:
        return levels.pop()
    return ""


def apply_verdict(label: Mapping[str, Any], verdict: QCVerdict) -> dict[str, Any]:
    """Return a copy of ``label`` with its ``qc`` block filled in (spec 7.1).

    The input is not mutated: the caller still holds the unverified label, and
    writing the verified one is a separate, explicit step.
    """
    out = dict(label)
    out["qc"] = verdict.to_dict()
    return out


def label_passed(label: Mapping[str, Any]) -> bool:
    """True only when ``label`` carries a ``qc`` block that passed.

    An unverified label (``qc`` still ``None``) has **not** passed: invariant 9
    says labels are verified before use, so absence of evidence is not a pass.
    """
    block = label.get("qc")
    return bool(isinstance(block, Mapping) and block.get("passed") is True)


def rejection_row(label: Mapping[str, Any], verdict: QCVerdict) -> dict[str, str]:
    """Render one ``dataset/qc/rejections.csv`` row for a failed verdict."""
    return {
        "file_id": str(label.get("file_id", verdict.file_id)),
        "source_file": str(label.get("source_file", "")),
        "label": label_of(label),
        "level": level_of(label) or verdict.level,
        "stage": "qc",
        "reason": "; ".join(verdict.reasons),
    }


def append_rejections(path: Path | str, rows: Sequence[Mapping[str, str]]) -> int:
    """Append rejection rows, writing the header only on a new file.

    Appending rather than rewriting is deliberate: spec 5.4 says rejections are
    *logged*, and a later run must not erase why an earlier file was dropped.

    Args:
        path: ``dataset/qc/rejections.csv``.
        rows: Rows from :func:`rejection_row`.

    Returns:
        Number of rows appended.
    """
    if not rows:
        return 0
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.is_file()
    with p.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REJECTION_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in REJECTION_COLUMNS})
    logger.warning("logged %d QC rejection(s) to %s", len(rows), p)
    return len(rows)


def append_realign(path: Path | str, reports: Iterable[RealignReport]) -> int:
    """Upsert realign reports into ``dataset/qc/realign.csv`` by ``file_id``.

    Unlike the rejection log this is a *state* table - one current row per file
    - so it is rewritten sorted by ``file_id`` for determinism.

    Returns:
        Total rows in the file after the merge.
    """
    p = Path(path)
    merged: dict[str, dict[str, str]] = {}
    if p.is_file():
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                merged[row["file_id"]] = {k: row.get(k, "") for k in REALIGN_COLUMNS}
    for report in reports:
        row = report.to_row()
        merged[row["file_id"]] = row

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REALIGN_COLUMNS)
        writer.writeheader()
        for file_id in sorted(merged):
            writer.writerow(merged[file_id])
    return len(merged)


def exclude_rejected(manifest_path: Path | str, labels: Iterable[Mapping[str, Any]]) -> int:
    """Mark QC-failed files as ``excluded`` in ``MANIFEST.csv`` (spec 7.3).

    The row is kept, not deleted: the dataset should say what was generated and
    dropped, and the LOTO split builder skips the
    :data:`EXCLUDED_FOLD` fold rather than filtering by a rule of its own.

    Args:
        manifest_path: ``dataset/MANIFEST.csv``.
        labels: Label JSONs carrying a ``qc`` block.

    Returns:
        Number of rows whose ``split_fold`` was changed.
    """
    path = Path(manifest_path)
    rows = {row.file_id: row for row in read_manifest(path)}
    failed = {
        str(lab["file_id"])
        for lab in labels
        if isinstance(lab.get("qc"), Mapping) and not label_passed(lab)
    }

    changed = [
        replace(rows[fid], split_fold=EXCLUDED_FOLD)
        for fid in sorted(failed)
        if fid in rows and rows[fid].split_fold != EXCLUDED_FOLD
    ]
    if changed:
        append_manifest(path, changed)
        logger.warning("excluded %d QC-failed file(s) from the splits", len(changed))
    return len(changed)
