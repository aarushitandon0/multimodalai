"""Human flaw annotation: TextGrid tier in, label JSON and agreement out (spec 6.2).

Human-recorded flawed takes have no time map to derive labels from, so the
labels are hand-drawn in Praat: the MFA TextGrid gains an interval tier named
``flaws`` whose intervals are labelled ``<label>:<severity 1-3>``. This module
is the whole of the logic behind that workflow -

* :func:`blank_flaw_tier` / :func:`flaw_tier` build the tier an annotator fills
  in and the tier that records what they filled in;
* :func:`parse_interval_label` and :func:`read_annotations` read it back,
  rejecting anything outside the taxonomy rather than guessing;
* :func:`build_label_json` converts it to the spec 7.1 label JSON, with
  ``origin`` ``"human"``;
* :func:`match_events`, :func:`event_f1` and :func:`cohens_kappa` compute the
  inter-annotator agreement that spec 6.2 and 14.1 ask us to report.

Everything here is a pure function of its arguments plus the config: reading
and writing files is left to ``scripts/annotate_*.py``, apart from
:func:`read_annotations`, which parses one TextGrid because the TextGrid *is*
the annotation format.

Two project invariants show up repeatedly:

* human takes are **test-only** (``CLAUDE.md`` invariant 11), so every label
  JSON this module emits carries ``"split": "test"``;
* times are seconds, float, 3 decimals, relative to the processed WAV
  (invariant 2), which :func:`~rostrum.types.round_time` enforces on output.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from rostrum.align.mfa import TextGridTier, parse_textgrid
from rostrum.config import Config
from rostrum.types import Alignment, FlawLabel, round_time, round_value

__all__ = [
    "AgreementReport",
    "AnnotationError",
    "AnnotationSettings",
    "EventMatch",
    "FlawAnnotation",
    "agreement",
    "blank_flaw_tier",
    "build_label_json",
    "cohens_kappa",
    "double_annotated",
    "event_f1",
    "flaw_tier",
    "format_agreement_md",
    "label_categories",
    "match_events",
    "parse_interval_label",
    "read_annotations",
    "settings_from_config",
]

logger = logging.getLogger(__name__)


class AnnotationError(Exception):
    """An annotation file does not follow the guide.

    Raised rather than repaired: a mislabelled interval silently coerced into
    something plausible would become a wrong ground-truth label, and every
    metric downstream would be measured against it.
    """


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AnnotationSettings:
    """Resolved ``injection.annotation`` config (plus the severity ladder).

    Attributes:
        tier_name: Name of the interval tier annotators fill in.
        separator: Character between label and severity in an interval text.
        severity_levels: ``{1: "L2", 2: "L3", 3: "L4"}`` - the ladder rung each
            hand-assigned severity maps to.
        severity_s: ``{"L2": 0.35, ...}`` from the ``severity`` ladder.
        categories: ``{label: rubric category}`` for all eight labels.
        min_duration_s: Shortest interval accepted as a real annotation.
        split: Split every imported file belongs to; ``"test"`` by invariant 11.
        guide_version: Version of ``ANNOTATION_GUIDE.md`` these labels follow.
        onset_collar_s: Onset collar for event matching, in seconds.
        offset_collar_s: Floor of the offset collar, in seconds.
        offset_collar_frac: Offset collar as a fraction of event length; the
            collar used is ``max(offset_collar_s, frac * length)``.
        min_double_annotated_frac: Fraction of human files that must be
            annotated twice (spec 6.2).
    """

    tier_name: str
    separator: str
    severity_levels: Mapping[int, str]
    severity_s: Mapping[str, float]
    categories: Mapping[str, str]
    min_duration_s: float
    split: str
    guide_version: str
    onset_collar_s: float
    offset_collar_s: float
    offset_collar_frac: float
    min_double_annotated_frac: float

    @property
    def labels(self) -> tuple[str, ...]:
        """Every valid flaw label, sorted, as the guide lists them."""
        return tuple(sorted(self.categories))

    def level_for(self, severity: int) -> str:
        """Return the severity level ``L*`` for a hand-assigned 1-3 severity."""
        try:
            return self.severity_levels[severity]
        except KeyError:
            raise AnnotationError(
                f"severity {severity} is not on the annotation scale "
                f"{sorted(self.severity_levels)}"
            ) from None

    def s_for(self, severity: int) -> float:
        """Return the continuous severity ``s`` for a 1-3 severity."""
        return float(self.severity_s[self.level_for(severity)])


def label_categories(cfg: Config) -> dict[str, str]:
    """Return ``{flaw label: rubric category}`` from ``configs/injection.yaml``.

    Includes ``mumbling``, which has a category but no injection op - the
    taxonomy is one list whether a label can be synthesised or not.
    """
    return {str(name): str(spec["category"]) for name, spec in cfg.injection["labels"].items()}


def settings_from_config(cfg: Config) -> AnnotationSettings:
    """Resolve :class:`AnnotationSettings` from a loaded :class:`Config`."""
    ann = cfg.injection["annotation"]
    agree = ann["agreement"]
    return AnnotationSettings(
        tier_name=str(ann["tier_name"]),
        separator=str(ann["separator"]),
        severity_levels={int(k): str(v) for k, v in ann["severity_levels"].items()},
        severity_s={str(k): float(v) for k, v in cfg.injection["severity"].items()},
        categories=label_categories(cfg),
        min_duration_s=float(ann["min_duration_s"]),
        split=str(ann["split"]),
        guide_version=str(ann["guide_version"]),
        onset_collar_s=float(agree["onset_collar_s"]),
        offset_collar_s=float(agree["offset_collar_s"]),
        offset_collar_frac=float(agree["offset_collar_frac"]),
        min_double_annotated_frac=float(agree["min_double_annotated_frac"]),
    )


# --------------------------------------------------------------------------- #
# One annotation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FlawAnnotation:
    """One interval of the ``flaws`` tier.

    Attributes:
        label: Flaw label from the taxonomy.
        severity: Hand-assigned severity, 1-3 (see ``ANNOTATION_GUIDE.md``).
        t_start: Onset in seconds from the start of the processed WAV: the
            onset of the first affected word.
        t_end: Offset in seconds: the offset of the last affected word.
    """

    label: FlawLabel
    severity: int
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        """Annotation length in seconds."""
        return self.t_end - self.t_start

    @property
    def text(self) -> str:
        """The interval text this annotation was written as."""
        return f"{self.label}:{self.severity}"


def parse_interval_label(
    text: str,
    settings: AnnotationSettings,
) -> tuple[FlawLabel, int]:
    """Parse one ``<label>:<severity>`` interval text.

    Args:
        text: Interval text, as typed in Praat. Surrounding whitespace and an
            optional trailing separator are tolerated; nothing else is.
        settings: Taxonomy and severity scale to validate against.

    Returns:
        ``(label, severity)``.

    Raises:
        AnnotationError: The text is empty, has no separator, names a label
            outside the taxonomy, or carries a severity off the scale. The
            message lists the valid options, since choosing one is the fix.
    """
    raw = text.strip().rstrip(settings.separator).strip()
    if not raw:
        raise AnnotationError("empty interval text")
    if settings.separator not in raw:
        raise AnnotationError(
            f"{raw!r} is missing the severity: write "
            f"<label>{settings.separator}<severity 1-3>, e.g. "
            f"rushing{settings.separator}2"
        )
    label_part, _, severity_part = raw.rpartition(settings.separator)
    label = label_part.strip().lower()
    if label not in settings.categories:
        raise AnnotationError(
            f"{label!r} is not a flaw label; the taxonomy is " f"{', '.join(settings.labels)}"
        )
    try:
        severity = int(severity_part.strip())
    except ValueError:
        raise AnnotationError(
            f"severity {severity_part.strip()!r} in {raw!r} is not an integer "
            f"(expected one of {sorted(settings.severity_levels)})"
        ) from None
    if severity not in settings.severity_levels:
        raise AnnotationError(
            f"severity {severity} in {raw!r} is off the scale "
            f"{sorted(settings.severity_levels)}"
        )
    return label, severity  # type: ignore[return-value]


def _check_annotations(
    annotations: Sequence[FlawAnnotation],
    settings: AnnotationSettings,
    *,
    where: str,
) -> list[str]:
    """Return one message per structural problem in a file's annotations.

    Checks that no interval is shorter than the mis-click threshold and that
    two intervals with the *same* label do not overlap. Different labels may
    overlap: mumbling through a hesitation pause is one moment with two flaws,
    and splitting it would lose that.
    """
    problems: list[str] = []
    for ann in annotations:
        if ann.duration < settings.min_duration_s - 1e-9:
            problems.append(
                f"{where}: {ann.text} at {ann.t_start:.3f}s is only "
                f"{ann.duration * 1000:.0f} ms long; the guide's floor is "
                f"{settings.min_duration_s * 1000:.0f} ms"
            )
    by_label: dict[str, list[FlawAnnotation]] = {}
    for ann in annotations:
        by_label.setdefault(ann.label, []).append(ann)
    for label, group in by_label.items():
        ordered = sorted(group, key=lambda a: a.t_start)
        for first, second in pairwise(ordered):
            if second.t_start < first.t_end - 1e-9:
                problems.append(
                    f"{where}: two {label!r} intervals overlap "
                    f"([{first.t_start:.3f}, {first.t_end:.3f}] and "
                    f"[{second.t_start:.3f}, {second.t_end:.3f}]); merge them "
                    f"or mark the stronger stretch only"
                )
    return problems


def read_annotations(
    path: str | Path,
    settings: AnnotationSettings,
    *,
    strict: bool = True,
) -> tuple[list[FlawAnnotation], list[str]]:
    """Read the flaw tiers of a completed TextGrid.

    The tier named ``flaws`` is read, and so is any tier whose name starts
    ``flaws_`` - an interval tier cannot hold two overlapping intervals, so an
    annotator marking a flaw *inside* another one (mumbling through a
    hesitation pause) has to put it on a second tier, and the guide tells them
    to call it ``flaws_2``. All of them are one annotation set.

    Empty intervals are a tier's background and are skipped; every non-empty
    one must parse. Annotations come back sorted by onset then label, so the
    label JSON of a file does not depend on the order the annotator drew them
    in, or on the order the tiers happen to sit in (``CLAUDE.md`` invariant 13).

    Args:
        path: The completed ``.TextGrid``.
        settings: Taxonomy and severity scale.
        strict: Raise on the first problem. With ``strict=False`` the problems
            are returned instead, so a batch import can report every bad file
            in one pass rather than one per run.

    Returns:
        ``(annotations, problems)``. ``problems`` is empty when ``strict`` is
        true, because any problem was raised.

    Raises:
        FileNotFoundError: The file does not exist.
        AnnotationError: No flaw tier at all, or (when ``strict``) an interval
            the guide does not allow.
    """
    grid = Path(path)
    tiers = parse_textgrid(grid)
    base = settings.tier_name.lower()
    flaw_tiers = [
        t
        for t in tiers
        if t.name.strip().lower() == base or t.name.strip().lower().startswith(f"{base}_")
    ]
    if not flaw_tiers:
        raise AnnotationError(
            f"{grid}: no {settings.tier_name!r} tier; found "
            f"{', '.join(repr(t.name) for t in tiers)}. Run "
            f"scripts/annotate_prepare.py to get a grid with the tier in place."
        )

    annotations: list[FlawAnnotation] = []
    problems: list[str] = []
    for tier in flaw_tiers:
        for start, end, text in tier.intervals:
            if not text.strip():
                continue
            try:
                label, severity = parse_interval_label(text, settings)
            except AnnotationError as exc:
                message = f"{grid.name} [{start:.3f}, {end:.3f}] on {tier.name!r}: {exc}"
                if strict:
                    raise AnnotationError(message) from None
                problems.append(message)
                continue
            annotations.append(
                FlawAnnotation(label=label, severity=severity, t_start=start, t_end=end)
            )

    annotations.sort(key=lambda a: (a.t_start, a.t_end, a.label))
    structural = _check_annotations(annotations, settings, where=grid.name)
    if structural and strict:
        raise AnnotationError("; ".join(structural))
    problems.extend(structural)
    return annotations, problems


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #


def blank_flaw_tier(xmin: float, xmax: float, settings: AnnotationSettings) -> TextGridTier:
    """Return an empty ``flaws`` tier spanning ``[xmin, xmax]``.

    One empty interval over the whole file: that is what Praat shows as a blank
    tier ready to be split, and it is what the annotator starts from.
    """
    return TextGridTier(
        name=settings.tier_name,
        xmin=float(xmin),
        xmax=float(xmax),
        intervals=((float(xmin), float(xmax), ""),),
    )


def flaw_tier(
    annotations: Sequence[FlawAnnotation],
    xmin: float,
    xmax: float,
    settings: AnnotationSettings,
) -> TextGridTier:
    """Return a ``flaws`` tier holding ``annotations``.

    Used to write the imported labels back into
    ``dataset/alignments/<file_id>.TextGrid`` (spec 7.2: tiers ``words``,
    ``phones``, ``flaws``). Annotations must not overlap, which only holds for
    a single label; for a file with co-occurring labels, write one tier per
    label instead.

    Raises:
        ValueError: Two annotations overlap, so they cannot share one tier.
    """
    ordered = sorted(annotations, key=lambda a: a.t_start)
    for first, second in pairwise(ordered):
        if second.t_start < first.t_end - 1e-9:
            raise ValueError(
                f"{first.text} and {second.text} overlap; one interval tier " f"cannot hold both"
            )
    return TextGridTier(
        name=settings.tier_name,
        xmin=float(xmin),
        xmax=float(xmax),
        intervals=tuple((a.t_start, a.t_end, a.text) for a in ordered),
    )


# --------------------------------------------------------------------------- #
# Label JSON (spec 7.1)
# --------------------------------------------------------------------------- #


def _word_span(alignment: Alignment, t_start: float, t_end: float) -> list[int]:
    """Return ``[first, last]`` indices of the words overlapping a region.

    An annotation drawn on word boundaries covers whole words, so plain
    interval overlap is enough; the tolerance only absorbs the rounding of a
    boundary the annotator snapped to.
    """
    hit = [w.idx for w in alignment.words if w.t_end > t_start + 1e-3 and w.t_start < t_end - 1e-3]
    if not hit:
        # A pause annotation can sit entirely inside a gap. Anchor it to the
        # word before it so the region still has a position in the text.
        before = [w.idx for w in alignment.words if w.t_end <= t_start + 1e-3]
        return [before[-1], before[-1]] if before else []
    return [hit[0], hit[-1]]


def _gap_index(alignment: Alignment, t_start: float, t_end: float) -> int | None:
    """Return the gap index a pause annotation sits in, or ``None``.

    The gap with the largest overlap wins; a tie cannot occur, because gaps do
    not overlap each other.
    """
    best: tuple[float, int] | None = None
    for gap in alignment.gaps:
        overlap = min(t_end, gap.t_end) - max(t_start, gap.t_start)
        if overlap > 0 and (best is None or overlap > best[0]):
            best = (overlap, gap.idx)
    return None if best is None else best[1]


_PAUSE_LABELS = frozenset({"missing_pause", "hesitation_pause"})


def _flaw_dict(
    ann: FlawAnnotation,
    alignment: Alignment,
    settings: AnnotationSettings,
) -> dict[str, Any]:
    """Render one annotation as a label-JSON flaw entry (spec 7.1)."""
    out: dict[str, Any] = {
        "label": ann.label,
        "category": settings.categories[ann.label],
        "t_start": round_time(ann.t_start),
        "t_end": round_time(ann.t_end),
        "word_idx": _word_span(alignment, ann.t_start, ann.t_end),
        "level": settings.level_for(ann.severity),
        "s": round_value(settings.s_for(ann.severity)),
        # `severity` is the annotator's own 1-3 rating, kept beside the derived
        # level so the mapping stays auditable rather than baked in.
        "severity": int(ann.severity),
        "params": {},
    }
    if ann.label in _PAUSE_LABELS:
        out["gap_idx"] = _gap_index(alignment, ann.t_start, ann.t_end)
    return out


def build_label_json(
    file_id: str,
    annotations: Sequence[FlawAnnotation],
    alignment: Alignment,
    settings: AnnotationSettings,
    *,
    annotator: str = "",
    annotators: Sequence[str] = (),
    source_textgrid: str = "",
    duration: float | None = None,
    tier: str | None = None,
    config_hash: str = "",
) -> dict[str, Any]:
    """Build the spec 7.1 label JSON for one human-recorded flawed take.

    Differences from an injected file's label, all of them forced by the data:

    * ``origin`` is ``"human"`` and ``seed`` is ``None`` - nothing was
      generated, so there is no seed to regenerate it from;
    * ``split`` is pinned to ``"test"`` (``CLAUDE.md`` invariant 11);
    * ``qc`` records that there is no edit to verify instead of carrying UTMOS
      and re-alignment numbers that only mean something for an edited file;
    * an ``annotation`` block names who labelled it, from which TextGrid, and
      under which version of the guide.

    Args:
        file_id: The recording's id, matching the audio stem.
        annotations: The ``flaws`` tier contents, from :func:`read_annotations`.
        alignment: The file's forced alignment, for ``word_idx``/``gap_idx``.
        settings: Taxonomy, severity scale and guide version.
        annotator: Whose annotation this label is built from.
        annotators: Everyone who annotated this file, for the agreement report.
        source_textgrid: Path of the TextGrid, relative to the repo root.
        duration: WAV duration in seconds; defaults to the alignment's.
        tier: Sourcing tier; defaults to the alignment's (``bronze`` for team
            recordings).
        config_hash: ``Config.hash``, so a label names its parameters.

    Returns:
        A JSON-serialisable dict, times already rounded to 3 decimals.

    Raises:
        AnnotationError: An annotation falls outside the aligned speech, which
            means the TextGrid and the WAV are not the same recording.
        ValueError: An annotation carries a label outside the taxonomy.
    """
    total = float(alignment.duration if duration is None else duration)
    for ann in annotations:
        if ann.t_start < -1e-3 or ann.t_end > total + 1e-3:
            raise AnnotationError(
                f"{file_id}: {ann.text} spans [{ann.t_start:.3f}, {ann.t_end:.3f}] "
                f"but the recording is {total:.3f} s long - the TextGrid and the "
                f"WAV are out of sync"
            )
    return {
        "file_id": file_id,
        "source_file": file_id,
        "text_id": alignment.text_id,
        "speaker_id": alignment.speaker_id,
        "channel_id": alignment.channel_id,
        "origin": "human",
        "tier": alignment.tier if tier is None else tier,
        "seed": None,
        "duration": round_time(total),
        "split": settings.split,
        "config_hash": config_hash,
        "annotation": {
            "annotator": annotator,
            "annotators": sorted(
                {a for a in annotators if a} | ({annotator} if annotator else set())
            ),
            "guide_version": settings.guide_version,
            "source_textgrid": source_textgrid,
            "severity_levels": {str(k): v for k, v in sorted(settings.severity_levels.items())},
        },
        "flaws": [_flaw_dict(a, alignment, settings) for a in annotations],
        "qc": {
            "utmos_clean": None,
            "utmos_edited": None,
            "realign_max_err_ms": None,
            "passed": True,
            "note": "human recording: no edit to verify, labels are hand-drawn",
        },
    }


# --------------------------------------------------------------------------- #
# Inter-annotator agreement (spec 6.2, 14.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EventMatch:
    """One matched pair of annotations from two annotators.

    Attributes:
        a_idx: Index into the first annotator's list.
        b_idx: Index into the second annotator's list.
        onset_err: ``b.t_start - a.t_start`` in seconds, signed.
        offset_err: ``b.t_end - a.t_end`` in seconds, signed.
        iou: Temporal intersection over union of the two intervals.
        same_label: Whether the two annotators chose the same label.
    """

    a_idx: int
    b_idx: int
    onset_err: float
    offset_err: float
    iou: float
    same_label: bool


def _iou(a: FlawAnnotation, b: FlawAnnotation) -> float:
    """Temporal intersection over union of two annotations."""
    inter = min(a.t_end, b.t_end) - max(a.t_start, b.t_start)
    if inter <= 0:
        return 0.0
    union = max(a.t_end, b.t_end) - min(a.t_start, b.t_start)
    return inter / union if union > 0 else 0.0


def match_events(
    a: Sequence[FlawAnnotation],
    b: Sequence[FlawAnnotation],
    settings: AnnotationSettings,
    *,
    require_same_label: bool = False,
) -> tuple[list[EventMatch], list[int], list[int]]:
    """Match two annotators' events one-to-one within the collars.

    Two events may be matched when the onset difference is within
    ``onset_collar_s`` and the offset difference within
    ``max(offset_collar_s, offset_collar_frac * length_a)`` - the sed_eval
    convention of spec 14.1, with ``a`` as the reference. Among the admissible
    pairs the matching is greedy on onset error, then offset error, then IoU,
    so it does not depend on the input order (invariant 13).

    Matching is label-*agnostic* by default: it measures whether the two
    annotators found the same moments, and :func:`cohens_kappa` then measures
    whether they called them the same thing. Spec 6.2 asks for both numbers
    separately, and a label-aware matching would fold one into the other.

    Args:
        a: First annotator's events (the reference, for the offset collar).
        b: Second annotator's events.
        settings: Supplies the collars.
        require_same_label: Only match events the annotators labelled the same.

    Returns:
        ``(matches, unmatched_a, unmatched_b)``, where the two lists hold
        indices into ``a`` and ``b``. ``matches`` is sorted by ``a_idx``.
    """
    candidates: list[tuple[float, float, float, int, int]] = []
    for i, ev_a in enumerate(a):
        offset_collar = max(settings.offset_collar_s, settings.offset_collar_frac * ev_a.duration)
        for j, ev_b in enumerate(b):
            if require_same_label and ev_a.label != ev_b.label:
                continue
            onset_err = abs(ev_b.t_start - ev_a.t_start)
            offset_err = abs(ev_b.t_end - ev_a.t_end)
            if onset_err > settings.onset_collar_s + 1e-9:
                continue
            if offset_err > offset_collar + 1e-9:
                continue
            candidates.append((onset_err, offset_err, -_iou(ev_a, ev_b), i, j))

    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[EventMatch] = []
    for _, _, _, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append(
            EventMatch(
                a_idx=i,
                b_idx=j,
                onset_err=b[j].t_start - a[i].t_start,
                offset_err=b[j].t_end - a[i].t_end,
                iou=_iou(a[i], b[j]),
                same_label=a[i].label == b[j].label,
            )
        )
    matches.sort(key=lambda m: m.a_idx)
    return (
        matches,
        [i for i in range(len(a)) if i not in used_a],
        [j for j in range(len(b)) if j not in used_b],
    )


def event_f1(n_matched: int, n_a: int, n_b: int) -> tuple[float, float, float]:
    """Return ``(precision, recall, f1)`` for a matched event count.

    With annotator ``a`` as the reference, precision is over ``b``'s events and
    recall over ``a``'s. F1 is symmetric under swapping the annotators, so
    which one is called the reference does not change the headline number -
    which is what makes it reportable as *inter-annotator* agreement rather
    than as one annotator scored against the other.
    """
    precision = n_matched / n_b if n_b else (1.0 if n_matched == 0 else 0.0)
    recall = n_matched / n_a if n_a else (1.0 if n_matched == 0 else 0.0)
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def cohens_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> float:
    """Cohen's kappa between two annotators' labels for the same events.

    Computed here rather than imported so the agreement report has no
    dependency beyond the standard library: the quantity is a four-line
    formula and the implementation is checked against ``sklearn`` in the tests.

    Args:
        labels_a: First annotator's label per matched event.
        labels_b: Second annotator's label for the same events, same order.
        labels: Label universe, for the marginals. Defaults to the labels that
            actually occur.

    Returns:
        Kappa in ``[-1, 1]``. ``nan`` when there are no events; ``1.0`` when
        both annotators used a single label and agreed on all of it - chance
        agreement is then 1 and kappa is undefined, but reporting perfect
        agreement as perfect is less misleading than reporting ``nan``.

    Raises:
        ValueError: The two sequences have different lengths.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(f"{len(labels_a)} labels vs {len(labels_b)}: not the same events")
    n = len(labels_a)
    if n == 0:
        return math.nan
    universe = sorted(set(labels) if labels is not None else set(labels_a) | set(labels_b))
    index = {label: i for i, label in enumerate(universe)}
    counts = [[0] * len(universe) for _ in universe]
    for la, lb in zip(labels_a, labels_b, strict=True):
        counts[index[la]][index[lb]] += 1

    observed = sum(counts[i][i] for i in range(len(universe))) / n
    expected = sum(
        (sum(counts[i]) / n) * (sum(row[i] for row in counts) / n) for i in range(len(universe))
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


@dataclass(frozen=True)
class AgreementReport:
    """Inter-annotator agreement over the double-annotated files (spec 6.2).

    Attributes:
        annotator_a: Name of the first annotator.
        annotator_b: Name of the second.
        file_ids: The double-annotated files, sorted.
        n_events_a: Events drawn by ``annotator_a`` across those files.
        n_events_b: Events drawn by ``annotator_b``.
        n_matched: Events matched within the collars.
        precision: Matched / ``n_events_b``.
        recall: Matched / ``n_events_a``.
        f1: Event-F1 at the configured collars.
        kappa: Cohen's kappa on the labels of the matched events.
        label_agreement: Fraction of matched events given the same label.
        onset_mae: Mean absolute onset difference over matched events, seconds.
        offset_mae: Mean absolute offset difference, seconds.
        mean_iou: Mean temporal IoU over matched events.
        per_file: One row per file with its own counts and F1.
        confusion: ``{(label_a, label_b): count}`` over matched events.
        onset_collar_s: Collar the matching used.
        offset_collar_s: Floor of the offset collar used.
        offset_collar_frac: Length fraction of the offset collar used.
        coverage: Double-annotated share of the human files, if known.
        n_human_files: How many human files exist, if known.
    """

    annotator_a: str
    annotator_b: str
    file_ids: tuple[str, ...]
    n_events_a: int
    n_events_b: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    kappa: float
    label_agreement: float
    onset_mae: float
    offset_mae: float
    mean_iou: float
    per_file: tuple[dict[str, Any], ...]
    confusion: Mapping[tuple[str, str], int]
    onset_collar_s: float
    offset_collar_s: float
    offset_collar_frac: float
    coverage: float | None = None
    n_human_files: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy, values rounded for stable hashing."""
        return round_value(
            {
                "annotator_a": self.annotator_a,
                "annotator_b": self.annotator_b,
                "file_ids": list(self.file_ids),
                "n_events_a": self.n_events_a,
                "n_events_b": self.n_events_b,
                "n_matched": self.n_matched,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "kappa": None if math.isnan(self.kappa) else self.kappa,
                "label_agreement": self.label_agreement,
                "onset_mae": self.onset_mae,
                "offset_mae": self.offset_mae,
                "mean_iou": self.mean_iou,
                "per_file": [dict(row) for row in self.per_file],
                "confusion": {f"{a}->{b}": n for (a, b), n in sorted(self.confusion.items())},
                "collars": {
                    "onset_s": self.onset_collar_s,
                    "offset_s": self.offset_collar_s,
                    "offset_frac": self.offset_collar_frac,
                },
                "coverage": self.coverage,
                "n_human_files": self.n_human_files,
            }
        )


def agreement(
    by_file: Mapping[str, tuple[Sequence[FlawAnnotation], Sequence[FlawAnnotation]]],
    settings: AnnotationSettings,
    *,
    annotator_a: str = "a",
    annotator_b: str = "b",
    n_human_files: int | None = None,
) -> AgreementReport:
    """Compute event-F1 and kappa over every double-annotated file.

    Events are pooled across files before F1 and kappa are computed, so a file
    with many flaws weighs more than one with a single flaw - the quantity
    being reported is agreement per *event*, not per file. Per-file numbers are
    kept alongside so an outlier file is visible rather than averaged away.

    Args:
        by_file: ``{file_id: (events_a, events_b)}``.
        settings: Supplies the collars and the label universe.
        annotator_a: Name of the first annotator.
        annotator_b: Name of the second.
        n_human_files: Total human files, for the spec 6.2 coverage check.

    Returns:
        An :class:`AgreementReport`.
    """
    total_a = total_b = total_matched = 0
    onset_errs: list[float] = []
    offset_errs: list[float] = []
    ious: list[float] = []
    pairs_a: list[str] = []
    pairs_b: list[str] = []
    confusion: dict[tuple[str, str], int] = {}
    per_file: list[dict[str, Any]] = []

    for file_id in sorted(by_file):
        events_a, events_b = by_file[file_id]
        matches, unmatched_a, unmatched_b = match_events(events_a, events_b, settings)
        total_a += len(events_a)
        total_b += len(events_b)
        total_matched += len(matches)
        for m in matches:
            onset_errs.append(abs(m.onset_err))
            offset_errs.append(abs(m.offset_err))
            ious.append(m.iou)
            la, lb = events_a[m.a_idx].label, events_b[m.b_idx].label
            pairs_a.append(la)
            pairs_b.append(lb)
            confusion[(la, lb)] = confusion.get((la, lb), 0) + 1
        precision, recall, f1 = event_f1(len(matches), len(events_a), len(events_b))
        per_file.append(
            {
                "file_id": file_id,
                "n_events_a": len(events_a),
                "n_events_b": len(events_b),
                "n_matched": len(matches),
                "n_unmatched_a": len(unmatched_a),
                "n_unmatched_b": len(unmatched_b),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "label_agreement": (
                    sum(m.same_label for m in matches) / len(matches) if matches else math.nan
                ),
            }
        )

    precision, recall, f1 = event_f1(total_matched, total_a, total_b)
    n_same = sum(1 for la, lb in zip(pairs_a, pairs_b, strict=True) if la == lb)
    return AgreementReport(
        annotator_a=annotator_a,
        annotator_b=annotator_b,
        file_ids=tuple(sorted(by_file)),
        n_events_a=total_a,
        n_events_b=total_b,
        n_matched=total_matched,
        precision=precision,
        recall=recall,
        f1=f1,
        kappa=cohens_kappa(pairs_a, pairs_b, labels=settings.labels),
        label_agreement=n_same / total_matched if total_matched else math.nan,
        onset_mae=sum(onset_errs) / len(onset_errs) if onset_errs else math.nan,
        offset_mae=sum(offset_errs) / len(offset_errs) if offset_errs else math.nan,
        mean_iou=sum(ious) / len(ious) if ious else math.nan,
        per_file=tuple(per_file),
        confusion=confusion,
        onset_collar_s=settings.onset_collar_s,
        offset_collar_s=settings.offset_collar_s,
        offset_collar_frac=settings.offset_collar_frac,
        coverage=(len(by_file) / n_human_files if n_human_files else None),
        n_human_files=n_human_files,
    )


def _fmt(x: float, places: int = 3) -> str:
    """Format a float for the report, writing ``n/a`` for ``nan``."""
    return "n/a" if math.isnan(x) else f"{x:.{places}f}"


def format_agreement_md(report: AgreementReport, settings: AnnotationSettings) -> str:
    """Render :func:`agreement` as ``dataset/qc/agreement.md`` (spec 6.2).

    Markdown rather than CSV because this file is read by a human and quoted in
    the technical report. The numbers are also written as JSON beside it by
    ``scripts/annotator_agreement.py``, for anything that needs to parse them.
    """
    r = report
    collar_ms = round(r.onset_collar_s * 1000)
    lines = [
        "# Inter-annotator agreement",
        "",
        f"Annotators: **{r.annotator_a}** (A) and **{r.annotator_b}** (B).",
        f"Guide version: {settings.guide_version}.",
        "",
        f"Matching: onset collar {collar_ms} ms, offset collar "
        f"max({round(r.offset_collar_s * 1000)} ms, "
        f"{r.offset_collar_frac:.0%} of the A event's length), one-to-one, "
        "label-agnostic (spec 6.2, 14.1).",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Files annotated twice | {len(r.file_ids)}"
        + (f" of {r.n_human_files}" if r.n_human_files else "")
        + " |",
    ]
    if r.coverage is not None:
        ok = "meets" if r.coverage >= settings.min_double_annotated_frac else "**below**"
        lines.append(
            f"| Double-annotated share | {r.coverage:.0%} ({ok} the "
            f"{settings.min_double_annotated_frac:.0%} target) |"
        )
    lines += [
        f"| Events (A / B) | {r.n_events_a} / {r.n_events_b} |",
        f"| Matched events | {r.n_matched} |",
        f"| Event-F1 @ {collar_ms} ms | **{_fmt(r.f1)}** |",
        f"| Precision / recall | {_fmt(r.precision)} / {_fmt(r.recall)} |",
        f"| Cohen's kappa (labels of matched events) | **{_fmt(r.kappa)}** |",
        f"| Same-label rate on matched events | {_fmt(r.label_agreement)} |",
        f"| Onset MAE | {_fmt(r.onset_mae * 1000, 1)} ms |",
        f"| Offset MAE | {_fmt(r.offset_mae * 1000, 1)} ms |",
        f"| Mean temporal IoU | {_fmt(r.mean_iou)} |",
        "",
        "## Per file",
        "",
        "| file_id | A | B | matched | F1 | same-label |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in r.per_file:
        lines.append(
            f"| {row['file_id']} | {row['n_events_a']} | {row['n_events_b']} | "
            f"{row['n_matched']} | {_fmt(float(row['f1']))} | "
            f"{_fmt(float(row['label_agreement']))} |"
        )

    lines += ["", "## Label confusion on matched events", ""]
    if not r.confusion:
        lines.append("No matched events, so there is nothing to confuse.")
    else:
        used = sorted({label for pair in r.confusion for label in pair})
        lines.append("| A \\ B | " + " | ".join(used) + " |")
        lines.append("| --- |" + " --- |" * len(used))
        for la in used:
            cells = [str(r.confusion.get((la, lb), 0)) for lb in used]
            lines.append(f"| {la} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## How to read this",
        "",
        "Event-F1 says whether the two annotators found the same *moments*; "
        "kappa says whether they called them the same *thing*. A high F1 with "
        "a low kappa means the guide's label definitions need sharpening, not "
        "its boundary rules. Disagreements are not resolved by averaging: the "
        "primary annotator's file is the one imported, and this table is what "
        "tells the reader how much to trust it.",
        "",
    ]
    return "\n".join(lines)


def double_annotated(
    by_annotator: Mapping[str, Iterable[str]],
) -> dict[tuple[str, str], list[str]]:
    """Group file ids by the annotator pair that both labelled them.

    Args:
        by_annotator: ``{annotator: file_ids}``.

    Returns:
        ``{(annotator_a, annotator_b): shared file ids}`` for every pair with
        at least one file in common, with each pair in sorted order.
    """
    names = sorted(by_annotator)
    shared: dict[tuple[str, str], list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sorted(set(by_annotator[a]) & set(by_annotator[b]))
            if common:
                shared[(a, b)] = common
    return shared
