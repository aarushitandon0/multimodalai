"""Span selection and the generation plan for injected files (spec 5.3-5.4, 7.1).

:mod:`rostrum.inject.ops` knows *how* to make one edit; this module decides
**where** the edit goes, **how strong** it is, and **what the label says** about
it. The split matters for rigour: the ops are the DSP and are tested against
synthetic signals, while everything here is bookkeeping that must be exactly
reproducible from a seed and a config hash.

Three rules shape the module.

**Span selection is seeded and enumerable.** For every label a deterministic
list of *valid* spans is built from the alignment (and, for the two gates that
need it, from the clean audio), then one is drawn with a
:class:`numpy.random.Generator` seeded from the output ``file_id``. The list is
built in alignment order, never from a set or a dict iteration, so the same
config and the same clean file always offer the same candidates in the same
order - without which the seed would not pin the choice (``CLAUDE.md``
invariant 13).

**Severity is a single knob.** ``injection.yaml`` maps ``L1..L4`` to
``s in {0.15, 0.35, 0.60, 0.90}`` and every parameter is a formula in ``s``
alone. The formulas stay strings in the config and are evaluated here, so each
label records both the formula and the number it produced and a reader can
check one against the other.

**Timestamps come from the time map, never from a re-measurement.** A flaw's
``t_start``/``t_end`` are the clean-file span pushed through the composed
:class:`~rostrum.inject.timemap.TimeMap` (invariant 9). The returned
:class:`~rostrum.types.Alignment` is the clean alignment pushed through the
same map, which is what makes word-anchored comparison possible on an edited
file whose words have all moved.

Two dependencies are worth stating plainly:

* ``Gap.label`` (``boundary`` / ``minor`` / ``none``) is **text**-derived and is
  filled by ``rostrum.text``. ``missing_pause``, ``hesitation_pause``,
  phrase-final ``low_energy`` and every phrase-scoped label need it; on an
  alignment whose gaps are all at the default ``"none"`` the engine raises
  :class:`NoValidSpanError` rather than guessing a sentence structure from
  silence durations.
* "Most prominent word in the phrase" is read off the clean audio with the
  weighted proxy in ``labels.misplaced_stress.valid_span.prominence_proxy``.
  TODO(features): replace with ``features/prominence.py`` (CWT, spec 9.1) once
  it exists, and regenerate - the proxy is a documented design choice, not a
  measurement taken from the literature.

I/O lives in ``scripts/build_injected.py``: the functions here take audio and
an alignment in memory and return audio, an alignment and a label *dict*. The
one concession is that ``cfg=None`` loads ``configs/*.yaml``, so the signatures
in the prompt (``generate_single(clean_file, label, level, seed)``) work as
written.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeAlias

import numpy as np

from rostrum.config import Config, load_config
from rostrum.inject import ops
from rostrum.inject.ops import InjectionError
from rostrum.inject.timemap import TimeMap
from rostrum.io.hashing import sha256_bytes
from rostrum.types import (
    Alignment,
    FlawLabel,
    Gap,
    round_time,
    round_value,
)

__all__ = [
    "INJECTABLE_LABELS",
    "CleanFile",
    "InjectionResult",
    "NoValidSpanError",
    "PlannedFlaw",
    "Span",
    "apply_flaw",
    "candidate_spans",
    "events_rows",
    "file_seed",
    "format_events_tsv",
    "generate_composite",
    "generate_single",
    "most_prominent_word",
    "phrases",
    "remap_alignment",
    "resolve_params",
    "select_span",
    "severity_s",
    "word_prominence",
]

logger = logging.getLogger(__name__)

INJECTABLE_LABELS: tuple[FlawLabel, ...] = (
    "rushing",
    "dragging",
    "missing_pause",
    "hesitation_pause",
    "monotone",
    "misplaced_stress",
    "low_energy",
)
"""The seven synthesisable labels, in dataset order.

``mumbling`` is absent on purpose: a low-pass filter models a bad microphone,
not bad articulation, so clarity is human-recorded only (spec 5.2).
"""

SpanKind: TypeAlias = Literal["words", "gap", "word_pair"]

_OPS = {
    "rushing": ops.psola_lengthen,
    "dragging": ops.psola_lengthen,
    "missing_pause": ops.shorten_gap,
    "hesitation_pause": ops.insert_roomtone,
    "monotone": ops.psola_pitch_compress,
    "misplaced_stress": ops.stress_swap,
    "low_energy": ops.gain_ramp,
}

_OP_PARAM_NAMES: dict[str, dict[str, str]] = {
    # config name -> op keyword. Only the names differ; the values are the
    # config's, so one YAML edit moves the whole dataset.
    "rushing": {"factor": "factor"},
    "dragging": {"factor": "factor"},
    "missing_pause": {"keep_fraction": "keep_frac"},
    "hesitation_pause": {"duration_s": "dur_s"},
    "monotone": {"exponent": "exponent"},
    "misplaced_stress": {
        "f0_shift_st": "f0_shift_st",
        "gain_db": "gain_db",
        "duration_factor": "duration_factor",
    },
    "low_energy": {"end_gain_db": "db_end", "ramp": "ramp"},
}

_PRAAT_SEED_MAX = 2**31 - 1
"""Praat's RNG is seeded with a positive 32-bit integer."""

_SEED_MASK = 2**63 - 1
"""Seeds are kept inside a signed 64-bit range so they survive JSON and CSV."""


class NoValidSpanError(InjectionError):
    """No span in this recording satisfies a label's validity rules.

    A normal outcome, not a bug: a 20-second reading has no 15-word phrase, and
    a recording whose gaps carry no ``boundary`` label cannot host a
    ``missing_pause``. The generation plan skips the file for that label and
    logs it (spec 5.4 logs rejections rather than dropping them silently).
    """


# --------------------------------------------------------------------------- #
# Spans
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Span:
    """Where one edit goes, in alignment terms only - never in samples.

    Attributes:
        kind: ``"words"`` (an inclusive word-index range), ``"gap"`` (one
            :class:`~rostrum.types.Gap` by its ``idx``) or ``"word_pair"``
            (the function/content pair of ``misplaced_stress``).
        words: ``(first, last)`` inclusive for ``"words"``; ``(function_word,
            content_word)`` for ``"word_pair"``, in that order and **not**
            necessarily in textual order.
        gap_idx: ``Gap.idx`` for ``"gap"``.
    """

    kind: SpanKind
    words: tuple[int, int] | None = None
    gap_idx: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "gap":
            if self.gap_idx is None:
                raise ValueError("a gap span needs gap_idx")
        elif self.words is None:
            raise ValueError(f"a {self.kind} span needs word indices")

    @property
    def word_range(self) -> tuple[int, int]:
        """The span's word indices in textual order, low first."""
        if self.words is None:
            raise ValueError("this span has no words")
        lo, hi = self.words
        return (lo, hi) if lo <= hi else (hi, lo)

    @property
    def op_span(self) -> tuple[int, int] | int:
        """The ``span`` argument the matching op in :mod:`rostrum.inject.ops` wants."""
        if self.kind == "gap":
            assert self.gap_idx is not None
            return int(self.gap_idx)
        assert self.words is not None
        return self.words

    def clean_interval(self, alignment: Alignment) -> tuple[float, float]:
        """The span's extent on the **clean** recording, in seconds.

        For a word span that is the first word's onset to the last word's
        offset; for a gap it is the silence itself. Flaw timestamps are this
        interval pushed through the time map, so a label can never claim a
        region the edit did not touch.
        """
        if self.kind == "gap":
            gap = _gap_by_idx(alignment, self.gap_idx)
            return gap.t_start, gap.t_end
        lo, hi = self.word_range
        return alignment.words[lo].t_start, alignment.words[hi].t_end

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "words": list(self.words) if self.words is not None else None,
            "gap_idx": self.gap_idx,
        }


@dataclass(frozen=True)
class PlannedFlaw:
    """One resolved edit: label, severity, span and the numbers it will use.

    Attributes:
        label: Flaw label.
        level: Severity level ``L1``-``L4``.
        s: The continuous severity the level maps to.
        span: Where the edit goes.
        method: ``injection.yaml``'s ``method`` for this label, recorded so the
            label names the operation rather than implying it.
        params: Resolved parameters under their **config** names, for the label.
        formulas: The formula strings those values came from.
        op_params: The same values under the op's keyword names, plus the splice
            and room-tone settings and the Praat seed.
    """

    label: FlawLabel
    level: str
    s: float
    span: Span
    method: str
    params: dict[str, Any]
    formulas: dict[str, str]
    op_params: dict[str, Any]


@dataclass(frozen=True)
class CleanFile:
    """A clean recording in memory: audio, its sample rate and its alignment.

    The engine never opens a file, so this is what a caller hands it
    (``scripts/build_injected.py`` builds one per clean WAV).

    Attributes:
        audio: Mono float signal of the processed WAV.
        sr: Sample rate; 16 kHz throughout (invariant 1).
        alignment: MFA alignment of **this** audio, with gap labels applied.
    """

    audio: np.ndarray
    sr: int
    alignment: Alignment

    @property
    def file_id(self) -> str:
        """Identifier of the clean recording."""
        return self.alignment.file_id

    @property
    def duration(self) -> float:
        """Duration of the audio in seconds, measured from the samples."""
        return self.audio.size / float(self.sr)


@dataclass(frozen=True)
class InjectionResult:
    """Everything one generated file consists of.

    Attributes:
        file_id: Deterministic id of the generated file.
        audio: Edited mono signal.
        sr: Sample rate.
        alignment: The clean alignment pushed through :attr:`time_map`, so word
            times refer to the *edited* WAV.
        time_map: Exact clean -> edited map; for a composite, the composition of
            every op's map in time order.
        label: The label JSON (spec 7.1) as a plain dict, ready to dump.
        flaws: The planned flaws, in time order.
    """

    file_id: str
    audio: np.ndarray
    sr: int
    alignment: Alignment
    time_map: TimeMap
    label: dict[str, Any]
    flaws: tuple[PlannedFlaw, ...]


# --------------------------------------------------------------------------- #
# Config access
# --------------------------------------------------------------------------- #


def _cfg(cfg: Config | None) -> Config:
    """Return ``cfg``, loading ``configs/*.yaml`` when none was passed."""
    return cfg if cfg is not None else load_config(apply_seeds=False)


def _label_cfg(cfg: Config, label: str) -> Mapping[str, Any]:
    """Return ``injection.labels.<label>``.

    Raises:
        KeyError: The label is not in the config.
    """
    labels = cfg.injection["labels"]
    if label not in labels:
        raise KeyError(f"no injection config for label {label!r}")
    return labels[label]


def severity_s(level: str, cfg: Config | None = None) -> float:
    """Return the continuous severity ``s`` of a level (spec 5.3).

    Args:
        level: ``"L0"`` .. ``"L4"``.
        cfg: Config; loaded if omitted.

    Returns:
        ``s`` from ``injection.severity``.

    Raises:
        KeyError: Unknown level.
    """
    ladder = _cfg(cfg).injection["severity"]
    if level not in ladder:
        raise KeyError(f"unknown severity level {level!r}; expected one of {sorted(ladder)}")
    return float(ladder[level])


def _eval_formula(expr: Any, s: float) -> Any:
    """Evaluate one ``injection.yaml`` parameter at severity ``s``.

    Numbers and plain mode names (``"linear_db"``) pass through; a formula string
    is evaluated with ``s`` as the only name in scope and no builtins, so a
    config cannot smuggle code into the dataset build.

    Raises:
        ValueError: The expression looks like a formula but does not evaluate to
            a number.
    """
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    text = str(expr)
    if not any(ch.isdigit() for ch in text) and "s" not in text.split():
        return text  # a mode name such as "linear_db"
    try:
        value = eval(text, {"__builtins__": {}}, {"s": float(s)})
    except Exception as exc:  # pragma: no cover - a malformed config
        raise ValueError(f"cannot evaluate parameter formula {text!r}: {exc}") from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"parameter formula {text!r} did not produce a number")
    return float(value)


def resolve_params(
    label: str,
    level: str,
    cfg: Config | None = None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Resolve one label's parameters at one severity level.

    Each ``labels.<label>.params`` entry is evaluated at ``s`` and clamped to
    ``labels.<label>.bounds``, which is what keeps an out-of-range config edit
    from producing a file whose label claims a severity the audio does not have.

    Args:
        label: Flaw label.
        level: ``L1``-``L4``.
        cfg: Config; loaded if omitted.

    Returns:
        ``(params, formulas, op_params)``: the resolved values under their
        config names, the formula strings they came from, and the same values
        under the op keyword names plus the shared splice / room-tone settings.

    Raises:
        KeyError: Unknown label or level.
    """
    conf = _cfg(cfg)
    s = severity_s(level, conf)
    label_cfg = _label_cfg(conf, label)
    bounds = label_cfg.get("bounds", {})

    params: dict[str, Any] = {}
    formulas: dict[str, str] = {}
    for name, expr in label_cfg.get("params", {}).items():
        value = _eval_formula(expr, s)
        if isinstance(value, float) and name in bounds:
            lo, hi = (float(v) for v in bounds[name])
            clamped = min(max(value, lo), hi)
            if abs(clamped - value) > 1e-12:
                logger.debug("%s.%s clamped %.4f -> %.4f", label, name, value, clamped)
            value = clamped
        params[name] = value
        formulas[name] = str(expr)

    names = _OP_PARAM_NAMES[label]
    op_params: dict[str, Any] = {names[k]: v for k, v in params.items() if k in names}
    op_params.update(_shared_op_params(conf, label))

    if label == "misplaced_stress":
        # `inverse_scale: 1.0` means "same magnitude, opposite sign" on the
        # content word. For a duration *factor* the opposite sign is the
        # reciprocal, so that is what the content word gets; ops' own default
        # (1 - 0.2*s) is overridden here to keep the config authoritative.
        scale = float(params.get("inverse_scale", 1.0))
        factor = float(params["duration_factor"])
        content_factor = factor ** (-scale)
        op_params["content_duration_factor"] = content_factor
        op_params["s"] = s
        params["content_duration_factor"] = content_factor
        formulas["content_duration_factor"] = f"duration_factor ** -{scale:g}"

    return params, formulas, op_params


def _shared_op_params(cfg: Config, label: str) -> dict[str, Any]:
    """Splice and room-tone settings the ops accept, from ``injection.yaml``."""
    splice = cfg.injection["splice"]
    shared: dict[str, Any] = {
        "crossfade_ms": float(splice["crossfade_ms"]),
        "snap_ms": float(splice["zero_crossing_search_ms"]),
    }
    if label == "hesitation_pause":
        tone = cfg.injection["room_tone"]
        shared.update(
            min_donor_gap_s=float(tone["min_donor_gap_s"]),
            donor_margin_ms=float(tone["donor_margin_ms"]),
            loop_crossfade_ms=float(tone["loop_crossfade_ms"]),
        )
    return shared


# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #


def file_seed(file_id: str, master_seed: int) -> int:
    """Derive one file's seed from its id and the master seed.

    ``injection.yaml``: *per-file seed = stable_hash(file_id) ^ seed*. The hash
    is SHA-256 of the id, not :func:`hash`, so the seed does not depend on
    ``PYTHONHASHSEED`` or on the interpreter build - a regenerated dataset has
    to match its published checksums (invariant 13).

    Args:
        file_id: The **output** file id, e.g. ``"s1_lv07_rushing_L3"``.
        master_seed: ``injection.seed``.

    Returns:
        A non-negative 63-bit seed.
    """
    digest = sha256_bytes(file_id.encode("utf-8"))
    return (int(digest[:16], 16) ^ int(master_seed)) & _SEED_MASK


def _praat_seed(rng: np.random.Generator) -> int:
    """Draw the Praat RNG seed for one op (see :func:`rostrum.inject.ops.seed_praat`)."""
    return int(rng.integers(1, _PRAAT_SEED_MAX))


# --------------------------------------------------------------------------- #
# Structure: phrases, gaps
# --------------------------------------------------------------------------- #


def _gap_by_idx(alignment: Alignment, gap_idx: int | None) -> Gap:
    """Return the gap with ``idx == gap_idx``.

    Raises:
        InjectionError: No gap carries that index.
    """
    if gap_idx is None:
        raise InjectionError("gap lookup needs a gap index")
    for gap in alignment.gaps:
        if gap.idx == int(gap_idx):
            return gap
    raise InjectionError(f"no gap with idx {gap_idx} in {alignment.file_id!r}")


def phrases(alignment: Alignment) -> tuple[tuple[int, int], ...]:
    """Split the utterance into phrases at every ``boundary`` gap.

    A phrase is a maximal run of consecutive words containing no ``boundary``
    gap, which is the unit spec 5.2 scopes ``monotone`` and ``misplaced_stress``
    to, and the "one sentence" the pace labels may not reach outside of. Minor
    gaps (commas) do **not** split a phrase: a comma is a breath inside a
    clause, not the end of one.

    Args:
        alignment: Alignment of the clean recording, with gap labels applied.

    Returns:
        Inclusive ``(first_word_idx, last_word_idx)`` pairs, in order. Empty
        when the alignment has no words.
    """
    if not alignment.words:
        return ()
    breaks = {g.prev_word_idx for g in alignment.gaps if g.label == "boundary"}
    out: list[tuple[int, int]] = []
    start = alignment.words[0].idx
    for word in alignment.words:
        if word.idx in breaks:
            out.append((start, word.idx))
            start = word.idx + 1
    last = alignment.words[-1].idx
    if start <= last:
        out.append((start, last))
    return tuple(out)


def _internal_gaps(alignment: Alignment, lo: int, hi: int) -> list[Gap]:
    """Gaps strictly inside the word range ``[lo, hi]``."""
    return [g for g in alignment.gaps if lo <= g.prev_word_idx < hi]


# --------------------------------------------------------------------------- #
# Audio-dependent gates
# --------------------------------------------------------------------------- #


def _voiced_stats(audio: np.ndarray, sr: int, t0: float, t1: float) -> tuple[float, int, float]:
    """Voiced fraction, voiced frame count and F0 range of ``[t0, t1]``.

    The range is the 10th-to-90th percentile spread in **semitones** re the
    span's own median, never in Hz: a Hz-domain spread is sex-biased
    (invariant 5), so a gate stated in Hz would admit different phrases for a
    low and a high voice.

    Returns:
        ``(voiced_fraction, n_voiced, range_st)``; all zero when the span is
        shorter than 50 ms, which is too short to track.
    """
    n0 = max(round(t0 * sr), 0)
    n1 = min(round(t1 * sr), np.asarray(audio).size)
    chunk = np.asarray(audio, dtype=np.float64)[n0:n1]
    if chunk.size < sr // 20:
        return 0.0, 0, 0.0
    floor_hz, ceiling_hz = ops.two_pass_pitch_range(chunk, sr)
    _, freqs = ops.pitch_track(chunk, sr, floor_hz, ceiling_hz)
    if freqs.size == 0:
        return 0.0, 0, 0.0
    voiced = freqs[freqs > 0.0]
    fraction = float(voiced.size) / float(freqs.size)
    if voiced.size < 2:
        return fraction, int(voiced.size), 0.0
    lo, hi = (float(v) for v in np.percentile(voiced, [10.0, 90.0]))
    range_st = 12.0 * float(np.log2(hi / lo)) if lo > 0.0 else 0.0
    return fraction, int(voiced.size), range_st


def word_prominence(
    clean: CleanFile,
    phrase: tuple[int, int],
    cfg: Config | None = None,
) -> np.ndarray:
    """Score every word of a phrase for prominence, on the clean audio.

    A weighted sum of three cues, each z-scored **within the phrase** so the
    comparison is local and needs no speaker-level normalisation:

    * level, as the word's RMS in dB;
    * pitch, as the word's maximum voiced F0 in semitones re the phrase median;
    * duration *per syllable*, which is what keeps a long function word from
      automatically beating a short stressed noun.

    TODO(features): swap for ``features/prominence.py`` (CWT) when it lands; the
    weights live in ``labels.misplaced_stress.valid_span.prominence_proxy`` and
    are an engineering choice, documented as such.

    Args:
        clean: The clean recording.
        phrase: Inclusive word-index range.
        cfg: Config; loaded if omitted.

    Returns:
        One score per word of the phrase, in word order.
    """
    conf = _cfg(cfg)
    proxy = _label_cfg(conf, "misplaced_stress")["valid_span"]["prominence_proxy"]
    weights = proxy["weights"]
    lo, hi = phrase
    words = [w for w in clean.alignment.words if lo <= w.idx <= hi]
    if not words:
        return np.zeros(0, dtype=np.float64)
    audio = np.asarray(clean.audio, dtype=np.float64)
    sr = clean.sr

    t0, t1 = words[0].t_start, words[-1].t_end
    n0, n1 = max(round(t0 * sr), 0), min(round(t1 * sr), audio.size)
    floor_hz, ceiling_hz = ops.two_pass_pitch_range(audio[n0:n1], sr)
    times, freqs = ops.pitch_track(audio[n0:n1], sr, floor_hz, ceiling_hz)
    times = times + t0
    voiced = freqs[freqs > 0.0]
    median_hz = float(np.median(voiced)) if voiced.size else 0.0

    energy_db: list[float] = []
    f0_max_st: list[float] = []
    dur_per_syll: list[float] = []
    for word in words:
        a = max(round(word.t_start * sr), 0)
        b = min(round(word.t_end * sr), audio.size)
        rms = float(np.sqrt(np.mean(np.square(audio[a:b])))) if b > a else 0.0
        energy_db.append(20.0 * float(np.log10(max(rms, 1e-12))))
        inside = (times >= word.t_start) & (times <= word.t_end) & (freqs > 0.0)
        if inside.any() and median_hz > 0.0:
            f0_max_st.append(12.0 * float(np.log2(float(freqs[inside].max()) / median_hz)))
        else:
            f0_max_st.append(0.0)
        dur_per_syll.append((word.t_end - word.t_start) / max(int(word.n_syllables), 1))

    cues = {
        "energy_db": np.asarray(energy_db, dtype=np.float64),
        "f0_max_st": np.asarray(f0_max_st, dtype=np.float64),
        "duration_per_syllable": np.asarray(dur_per_syll, dtype=np.float64),
    }
    score = np.zeros(len(words), dtype=np.float64)
    for name, values in cues.items():
        weight = float(weights.get(name, 0.0))
        if weight == 0.0:
            continue
        sd = float(values.std())
        if sd <= 1e-12:
            continue
        score += weight * (values - float(values.mean())) / sd
    return score


def most_prominent_word(
    clean: CleanFile,
    phrase: tuple[int, int],
    cfg: Config | None = None,
) -> int:
    """Return the word index of the phrase's most prominent word.

    Ties go to the earlier word (``argmax``), so the choice does not depend on
    how NumPy happens to order equal maxima.
    """
    scores = word_prominence(clean, phrase, cfg)
    lo, hi = phrase
    words = [w for w in clean.alignment.words if lo <= w.idx <= hi]
    return int(words[int(np.argmax(scores))].idx)


# --------------------------------------------------------------------------- #
# Span selection
# --------------------------------------------------------------------------- #


def candidate_spans(
    label: str,
    clean: CleanFile,
    cfg: Config | None = None,
) -> tuple[Span, ...]:
    """Every span of ``clean`` that is valid for ``label`` (spec 5.2, "Span").

    The rules, all from ``labels.<label>.valid_span``:

    ================  ==========================================================
    Label             Valid span
    ================  ==========================================================
    rushing/dragging  3-10 consecutive words inside one phrase, no internal
                      pause over 0.5 s, clear of the first and last 2 words
    missing_pause     a ``boundary`` gap of at least 300 ms
    hesitation_pause  a ``none`` gap shorter than 150 ms
    monotone          a phrase of 5-15 words, at least 60% voiced frames and at
                      least 4 st of F0 range
    misplaced_stress  a phrase holding a function word and whose most prominent
                      word is a content word
    low_energy        4-12 words ending at a ``boundary`` gap
    ================  ==========================================================

    Args:
        label: Flaw label.
        clean: The clean recording.
        cfg: Config; loaded if omitted.

    Returns:
        Candidates in alignment order - the order the seeded draw indexes into,
        so it must not depend on set or dict iteration.

    Raises:
        KeyError: Unknown label.
    """
    conf = _cfg(cfg)
    rules = _label_cfg(conf, label).get("valid_span", {})
    alignment = clean.alignment
    if label in ("rushing", "dragging"):
        return _pace_spans(alignment, rules)
    if label == "missing_pause":
        return _gap_spans(alignment, rules, shrink=True)
    if label == "hesitation_pause":
        return _gap_spans(alignment, rules, shrink=False)
    if label == "monotone":
        return _monotone_spans(clean, rules)
    if label == "misplaced_stress":
        return _stress_spans(clean, rules, conf)
    if label == "low_energy":
        return _energy_spans(alignment, rules)
    raise KeyError(f"no span rule for label {label!r}")


def _pace_spans(alignment: Alignment, rules: Mapping[str, Any]) -> tuple[Span, ...]:
    """Word runs for ``rushing`` / ``dragging``."""
    n_min, n_max = (int(v) for v in rules.get("span_words", (3, 10)))
    skip_head = int(rules.get("exclude_first_n_words", 0))
    skip_tail = int(rules.get("exclude_last_n_words", 0))
    max_pause = float(rules.get("require_no_internal_pause_over_s", float("inf")))
    if not alignment.words:
        return ()
    first = alignment.words[0].idx + skip_head
    last = alignment.words[-1].idx - skip_tail

    units = (
        phrases(alignment)
        if rules.get("within_one_sentence", True)
        else ((alignment.words[0].idx, alignment.words[-1].idx),)
    )
    out: list[Span] = []
    for p_lo, p_hi in units:
        for i0 in range(max(p_lo, first), p_hi + 1):
            for length in range(n_min, n_max + 1):
                i1 = i0 + length - 1
                if i1 > min(p_hi, last):
                    break
                if any(g.t_end - g.t_start > max_pause for g in _internal_gaps(alignment, i0, i1)):
                    continue
                out.append(Span(kind="words", words=(i0, i1)))
    return tuple(out)


def _gap_spans(
    alignment: Alignment,
    rules: Mapping[str, Any],
    *,
    shrink: bool,
) -> tuple[Span, ...]:
    """Gaps for ``missing_pause`` (``shrink``) or ``hesitation_pause``."""
    wanted = tuple(rules.get("gap_label", ()))
    out: list[Span] = []
    for gap in alignment.gaps:
        if wanted and gap.label not in wanted:
            continue
        duration = gap.t_end - gap.t_start
        if shrink and duration < float(rules.get("min_gap_s", 0.0)):
            continue
        if not shrink and duration >= float(rules.get("max_existing_gap_s", float("inf"))):
            continue
        out.append(Span(kind="gap", gap_idx=gap.idx))
    return tuple(out)


def _monotone_spans(clean: CleanFile, rules: Mapping[str, Any]) -> tuple[Span, ...]:
    """Phrases with enough voicing and enough F0 range to be worth flattening."""
    n_min = int(rules.get("min_phrase_words", 5))
    n_max = int(rules.get("max_phrase_words", 15))
    min_fraction = float(rules.get("min_voiced_fraction", 0.0))
    min_frames = int(rules.get("min_voiced_frames", 0))
    min_range = float(rules.get("min_phrase_f0_range_st", 0.0))

    out: list[Span] = []
    for lo, hi in phrases(clean.alignment):
        if not n_min <= hi - lo + 1 <= n_max:
            continue
        t0 = clean.alignment.words[lo].t_start
        t1 = clean.alignment.words[hi].t_end
        fraction, n_voiced, range_st = _voiced_stats(clean.audio, clean.sr, t0, t1)
        if fraction < min_fraction or n_voiced < min_frames or range_st < min_range:
            logger.debug(
                "monotone span (%d, %d) rejected: voiced=%.2f n=%d range=%.2fst",
                lo,
                hi,
                fraction,
                n_voiced,
                range_st,
            )
            continue
        out.append(Span(kind="words", words=(lo, hi)))
    return tuple(out)


def _stress_spans(
    clean: CleanFile,
    rules: Mapping[str, Any],
    cfg: Config,
) -> tuple[Span, ...]:
    """Function/content word pairs inside one phrase.

    The content word is the phrase's most prominent word *in the clean file*,
    which is what makes the edit a genuine *swap*: prominence is taken off the
    word that had it and given to one that should not have it. When the most
    prominent word is itself a function word the phrase is skipped - there is no
    prominence to move.
    """
    function_words = {str(w).lower() for w in rules.get("function_words", ())}
    min_words = int(rules.get("prominence_proxy", {}).get("min_phrase_words", 3))
    require_content = bool(rules.get("require_content_word_prominent", True))

    out: list[Span] = []
    for lo, hi in phrases(clean.alignment):
        if hi - lo + 1 < min_words:
            continue
        function_idx = [
            w.idx
            for w in clean.alignment.words
            if lo <= w.idx <= hi and w.text.strip().lower() in function_words
        ]
        if not function_idx:
            continue
        content_idx = most_prominent_word(clean, (lo, hi), cfg)
        if require_content and clean.alignment.words[content_idx].text.lower() in function_words:
            logger.debug("stress span (%d, %d) rejected: prominent word is a function word", lo, hi)
            continue
        out.extend(
            Span(kind="word_pair", words=(fn_idx, content_idx))
            for fn_idx in function_idx
            if fn_idx != content_idx
        )
    return tuple(out)


def _energy_spans(alignment: Alignment, rules: Mapping[str, Any]) -> tuple[Span, ...]:
    """Word runs for ``low_energy``, ending at a ``boundary`` gap when required."""
    n_min, n_max = (int(v) for v in rules.get("span_words", (4, 12)))
    phrase_final = bool(rules.get("require_phrase_final", True))
    # "Phrase-final" means the span ends where the speaker then paused, so the
    # last phrase of the file - which ends at the last word, with no gap after
    # it - does not qualify.
    boundary_ends = {g.prev_word_idx for g in alignment.gaps if g.label == "boundary"}

    out: list[Span] = []
    for p_lo, p_hi in phrases(alignment):
        if phrase_final and p_hi not in boundary_ends:
            continue
        tails = [p_hi] if phrase_final else list(range(p_lo + n_min - 1, p_hi + 1))
        for i1 in tails:
            for length in range(n_min, n_max + 1):
                i0 = i1 - length + 1
                if i0 < p_lo:
                    break
                out.append(Span(kind="words", words=(i0, i1)))
    return tuple(out)


def select_span(
    label: str,
    clean: CleanFile,
    rng: np.random.Generator,
    cfg: Config | None = None,
    *,
    candidates: Sequence[Span] | None = None,
) -> Span:
    """Draw one valid span with the seeded generator.

    Args:
        label: Flaw label.
        clean: The clean recording.
        rng: Seeded generator; one ``integers`` draw picks the span.
        cfg: Config; loaded if omitted.
        candidates: Pre-filtered candidate list, used by
            :func:`generate_composite` to exclude spans too close to a flaw it
            has already placed.

    Returns:
        The chosen :class:`Span`.

    Raises:
        NoValidSpanError: The candidate list is empty.
    """
    pool = tuple(candidates) if candidates is not None else candidate_spans(label, clean, cfg)
    if not pool:
        n_boundary = sum(g.label == "boundary" for g in clean.alignment.gaps)
        raise NoValidSpanError(
            f"no valid {label} span in {clean.file_id!r} "
            f"({clean.alignment.n_words} words, {n_boundary} boundary gaps)"
        )
    return pool[int(rng.integers(len(pool)))]


# --------------------------------------------------------------------------- #
# Applying edits
# --------------------------------------------------------------------------- #


def apply_flaw(
    audio: np.ndarray,
    sr: int,
    alignment: Alignment,
    flaw: PlannedFlaw,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, TimeMap]:
    """Apply one planned flaw with the matching op.

    Args:
        audio: Signal the flaw is applied to. For a composite this is the audio
            as left by the previous edit, and ``alignment`` must be the
            alignment already pushed through that edit's map.
        sr: Sample rate.
        alignment: Alignment of ``audio``.
        flaw: The planned edit.
        rng: Seeded generator. Only the room-tone loop phase consumes it; the
            Praat seed comes from ``flaw.op_params["praat_seed"]``, which
            :func:`generate_single` always sets.

    Returns:
        ``(new_audio, time_map)`` straight from the op.
    """
    op = _OPS[flaw.label]
    return op(audio, sr, alignment, flaw.span.op_span, flaw.op_params, rng)


def remap_alignment(
    alignment: Alignment,
    tmap: TimeMap,
    *,
    file_id: str | None = None,
    duration: float | None = None,
) -> Alignment:
    """Push an alignment through a time map (invariant 9).

    Word onsets and offsets are mapped with :meth:`TimeMap.map_time`, so a word
    that ended exactly where room tone was inserted keeps its offset and the
    inserted silence lands in the gap rather than inside the word. Gap times are
    then taken from the neighbouring words, which keeps the two tiers exactly
    consistent instead of merely close.

    Word *indices*, texts and phones are untouched: no op deletes or inserts a
    word, which is why word-anchored comparison still works on an edited file.

    Args:
        alignment: Alignment of the clean recording.
        tmap: Clean -> edited map.
        file_id: ``file_id`` for the result; defaults to the input's.
        duration: Duration of the edited WAV; defaults to ``tmap.new_duration``.

    Returns:
        A new :class:`~rostrum.types.Alignment` whose times refer to the edited
        WAV. ``audio_sha256`` is cleared: the edited WAV is a different file and
        whoever writes it fills the hash in.
    """
    words = [
        replace(w, t_start=tmap.map_time(w.t_start), t_end=tmap.map_time(w.t_end))
        for w in alignment.words
    ]
    by_idx = {w.idx: w for w in words}
    gaps: list[Gap] = []
    for gap in alignment.gaps:
        prev_word = by_idx.get(gap.prev_word_idx)
        next_word = by_idx.get(gap.next_word_idx)
        t_start = prev_word.t_end if prev_word is not None else tmap.map_time(gap.t_start)
        t_end = next_word.t_start if next_word is not None else tmap.map_time(gap.t_end)
        gaps.append(replace(gap, t_start=t_start, t_end=t_end))
    return replace(
        alignment,
        file_id=file_id if file_id is not None else alignment.file_id,
        words=words,
        gaps=gaps,
        duration=float(duration if duration is not None else tmap.new_duration),
        audio_sha256="",
    )


# --------------------------------------------------------------------------- #
# Label JSON (spec 7.1)
# --------------------------------------------------------------------------- #


def _flaw_dict(flaw: PlannedFlaw, alignment: Alignment, tmap: TimeMap) -> dict[str, Any]:
    """Render one flaw for the label JSON, timed through ``tmap``.

    ``t_start``/``t_end`` are the clean span mapped with
    :meth:`TimeMap.map_interval`, which grows an interval to contain inserted
    material - a hesitation's label covers the silence that was added, not just
    the instant it was added at. The clean-file times are kept alongside so a
    file can be regenerated and checked without re-deriving the map.
    """
    a, b = flaw.span.clean_interval(alignment)
    t_start, t_end = tmap.map_interval(a, b)
    out: dict[str, Any] = {
        "label": flaw.label,
        "t_start": round_time(t_start),
        "t_end": round_time(t_end),
        "level": flaw.level,
        "s": round_value(flaw.s),
        "method": flaw.method,
        "span": flaw.span.to_dict(),
        "params": {k: round_value(v) for k, v in flaw.params.items()},
        "formulas": dict(flaw.formulas),
        "clean_t_start": round_time(a),
        "clean_t_end": round_time(b),
    }
    if flaw.span.kind == "gap":
        gap = _gap_by_idx(alignment, flaw.span.gap_idx)
        out["gap_idx"] = flaw.span.gap_idx
        out["gap_label"] = gap.label
        out["word_idx"] = [gap.prev_word_idx, gap.next_word_idx]
    else:
        lo, hi = flaw.span.word_range
        out["word_idx"] = [lo, hi]
        if flaw.span.kind == "word_pair":
            assert flaw.span.words is not None
            out["function_word_idx"] = int(flaw.span.words[0])
            out["content_word_idx"] = int(flaw.span.words[1])
    return out


def _build_label(
    *,
    file_id: str,
    clean: CleanFile,
    flaws: Sequence[PlannedFlaw],
    tmap: TimeMap,
    seed: int,
    duration: float,
    cfg: Config,
) -> dict[str, Any]:
    """Assemble the label JSON of one generated file (spec 7.1).

    ``qc`` is ``None`` until ``inject/qc.py`` re-aligns the file and scores it
    with UTMOS: an unverified file must not carry a ``passed`` flag.
    """
    alignment = clean.alignment
    return {
        "file_id": file_id,
        "source_file": clean.file_id,
        "text_id": alignment.text_id,
        "speaker_id": alignment.speaker_id,
        "channel_id": alignment.channel_id,
        "origin": "injected",
        "tier": alignment.tier,
        "seed": int(seed),
        "duration": round_time(duration),
        "clean_duration": round_time(clean.duration),
        "config_hash": cfg.hash,
        "flaws": [_flaw_dict(f, alignment, tmap) for f in flaws],
        "qc": None,
    }


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _plan(
    label: str,
    level: str,
    span: Span,
    cfg: Config,
    rng: np.random.Generator,
) -> PlannedFlaw:
    """Resolve parameters and draw the Praat seed for one chosen span."""
    params, formulas, op_params = resolve_params(label, level, cfg)
    praat_seed = _praat_seed(rng)
    op_params["praat_seed"] = praat_seed
    params["praat_seed"] = praat_seed
    return PlannedFlaw(
        label=label,  # type: ignore[arg-type]
        level=level,
        s=severity_s(level, cfg),
        span=span,
        method=str(_label_cfg(cfg, label)["method"]),
        params=params,
        formulas=formulas,
        op_params=op_params,
    )


def generate_single(
    clean: CleanFile,
    label: str,
    level: str,
    seed: int | None = None,
    *,
    cfg: Config | None = None,
    span: Span | None = None,
) -> InjectionResult:
    """Generate one single-flaw file (spec 5.4).

    Args:
        clean: The clean recording, with gap labels applied to its alignment.
        label: One of :data:`INJECTABLE_LABELS`.
        level: ``L1``-``L4``; ``L0`` is the clean file and is not generated here.
        seed: Overrides the derived seed; normally left out so the seed is
            :func:`file_seed` of the output id (invariant 13).
        cfg: Config; loaded if omitted.
        span: Force a span instead of drawing one - for tests, and for
            regenerating a file from its label.

    Returns:
        An :class:`InjectionResult` whose ``file_id`` is
        ``"<clean_id>_<label>_<level>"``.

    Raises:
        ValueError: ``label`` is not injectable, or ``level`` is ``L0``.
        NoValidSpanError: The recording offers no valid span for ``label``.
        InjectionError: The op refused the span - for instance no pause in the
            file is long enough to donate room tone.
    """
    conf = _cfg(cfg)
    if label not in INJECTABLE_LABELS:
        raise ValueError(f"{label!r} is not injectable; expected one of {INJECTABLE_LABELS}")
    if level == "L0":
        raise ValueError("L0 is the clean file; there is nothing to inject")

    file_id = f"{clean.file_id}_{label}_{level}"
    resolved_seed = file_seed(file_id, conf.seed) if seed is None else int(seed)
    rng = np.random.default_rng(resolved_seed)

    chosen = span if span is not None else select_span(label, clean, rng, conf)
    flaw = _plan(label, level, chosen, conf, rng)
    audio, tmap = apply_flaw(clean.audio, clean.sr, clean.alignment, flaw, rng)

    duration = audio.size / float(clean.sr)
    alignment = remap_alignment(clean.alignment, tmap, file_id=file_id, duration=duration)
    label_json = _build_label(
        file_id=file_id,
        clean=clean,
        flaws=[flaw],
        tmap=tmap,
        seed=resolved_seed,
        duration=duration,
        cfg=conf,
    )
    logger.info(
        "%s: %s %s at [%.3f, %.3f]s (delta %+.3fs)",
        file_id,
        label,
        level,
        label_json["flaws"][0]["t_start"],
        label_json["flaws"][0]["t_end"],
        tmap.delta,
    )
    return InjectionResult(
        file_id=file_id,
        audio=audio,
        sr=clean.sr,
        alignment=alignment,
        time_map=tmap,
        label=label_json,
        flaws=(flaw,),
    )


def _separated(
    interval: tuple[float, float],
    taken: Sequence[tuple[float, float]],
    min_sep: float,
) -> bool:
    """True when ``interval`` clears every taken interval by ``min_sep`` seconds."""
    a, b = interval
    return all(a >= t1 + min_sep or b <= t0 - min_sep for t0, t1 in taken)


def _place_flaws(
    clean: CleanFile,
    level: str,
    pools: Mapping[str, Sequence[Span]],
    rng: np.random.Generator,
    cfg: Config,
    *,
    wanted: int,
    minimum: int,
    min_sep: float,
    attempts: int,
    file_id: str,
) -> list[PlannedFlaw]:
    """Place up to ``wanted`` distinct-label flaws at least ``min_sep`` apart.

    One greedy pass can block itself - a 10-word ``dragging`` span taken first
    can leave no room for the rest - so the pass is retried up to ``attempts``
    times, each attempt re-drawing the label order and the spans from the same
    seeded stream. The best attempt wins, and the first attempt that reaches
    ``wanted`` stops the loop, so a file that fits easily costs exactly one pass.

    Returns:
        The planned flaws of the best attempt, in draw order. Shorter than
        ``minimum`` when the recording simply has no room; the caller raises.
    """
    best: list[PlannedFlaw] = []
    for attempt in range(max(attempts, 1)):
        order = [INJECTABLE_LABELS[i] for i in rng.permutation(len(INJECTABLE_LABELS))]
        taken: list[tuple[float, float]] = []
        planned: list[PlannedFlaw] = []
        for label in order:
            if len(planned) >= wanted:
                break
            pool = [
                sp
                for sp in pools[label]
                if _separated(sp.clean_interval(clean.alignment), taken, min_sep)
            ]
            if not pool:
                continue
            span = pool[int(rng.integers(len(pool)))]
            planned.append(_plan(label, level, span, cfg, rng))
            taken.append(span.clean_interval(clean.alignment))
        if len(planned) > len(best):
            best = planned
        if len(best) >= wanted:
            break
        logger.debug(
            "%s: attempt %d placed %d/%d flaws", file_id, attempt + 1, len(planned), wanted
        )
    if len(best) < wanted and len(best) >= minimum:
        logger.info(
            "%s: placed %d of %d flaws after %d attempts", file_id, len(best), wanted, attempts
        )
    return best


def generate_composite(
    clean: CleanFile,
    level: str,
    k: int = 1,
    *,
    n_flaws: int | None = None,
    seed: int | None = None,
    cfg: Config | None = None,
) -> InjectionResult:
    """Generate one composite file: 3-5 non-overlapping flaws (spec 5.4).

    Labels are drawn without replacement (``composite.distinct_labels``) in a
    seeded order, and each one takes a span that clears every already-placed
    flaw by ``composite.min_separation_s``; see :func:`_place_flaws` for the
    retry that keeps one unlucky wide span from blocking the rest. Edits are
    then applied in **time
    order**, each on the audio and the alignment the previous edit left behind,
    and their maps are composed - so the returned map is still clean -> final
    and every flaw timestamp is in the final file's timeline.

    Args:
        clean: The clean recording.
        level: ``L2`` or ``L4`` in the dataset plan; any level works.
        k: Which composite of this level this is, 1-based; it names the file.
        n_flaws: Force the flaw count instead of drawing from
            ``composite.n_flaws``.
        seed: Overrides the derived seed.
        cfg: Config; loaded if omitted.

    Returns:
        An :class:`InjectionResult` whose ``file_id`` is
        ``"<clean_id>_composite<k>_<level>"``.

    Raises:
        NoValidSpanError: Fewer than ``composite.n_flaws[0]`` flaws could be
            placed with the required separation.
    """
    conf = _cfg(cfg)
    comp = conf.injection["composite"]
    lo_n, hi_n = (int(v) for v in comp["n_flaws"])
    min_sep = float(comp["min_separation_s"])

    file_id = f"{clean.file_id}_composite{int(k)}_{level}"
    resolved_seed = file_seed(file_id, conf.seed) if seed is None else int(seed)
    rng = np.random.default_rng(resolved_seed)
    wanted = int(n_flaws) if n_flaws is not None else int(rng.integers(lo_n, hi_n + 1))

    # Candidates are enumerated once: the span rules do not depend on what has
    # already been placed, only the separation filter does, and for `monotone`
    # and `misplaced_stress` the enumeration costs a pitch track per phrase.
    pools = {label: candidate_spans(label, clean, conf) for label in INJECTABLE_LABELS}
    planned = _place_flaws(
        clean,
        level,
        pools,
        rng,
        conf,
        wanted=wanted,
        minimum=lo_n,
        min_sep=min_sep,
        attempts=int(comp.get("placement_attempts", 1)),
        file_id=file_id,
    )

    if len(planned) < lo_n:
        raise NoValidSpanError(
            f"{file_id}: placed only {len(planned)} of {wanted} flaws "
            f"(minimum {lo_n}) with {min_sep:.2f}s separation"
        )

    planned.sort(key=lambda f: f.span.clean_interval(clean.alignment))
    audio = np.asarray(clean.audio, dtype=np.float64)
    alignment = clean.alignment
    total: TimeMap | None = None
    for flaw in planned:
        audio, step = apply_flaw(audio, clean.sr, alignment, flaw, rng)
        # The next edit works on this audio, so it needs these word times.
        alignment = remap_alignment(alignment, step, duration=audio.size / float(clean.sr))
        total = step if total is None else total.then(step)
    assert total is not None  # len(planned) >= lo_n >= 1

    duration = audio.size / float(clean.sr)
    label_json = _build_label(
        file_id=file_id,
        clean=clean,
        flaws=planned,
        tmap=total,
        seed=resolved_seed,
        duration=duration,
        cfg=conf,
    )
    label_json["n_flaws"] = len(planned)
    logger.info(
        "%s: %d flaws (%s), delta %+.3fs",
        file_id,
        len(planned),
        ", ".join(f.label for f in planned),
        total.delta,
    )
    return InjectionResult(
        file_id=file_id,
        audio=audio,
        sr=clean.sr,
        alignment=remap_alignment(clean.alignment, total, file_id=file_id, duration=duration),
        time_map=TimeMap.from_segments(total.segments, op="composite"),
        label=label_json,
        flaws=tuple(planned),
    )


# --------------------------------------------------------------------------- #
# events.tsv (spec 7.2)
# --------------------------------------------------------------------------- #


def events_rows(label: Mapping[str, Any]) -> list[tuple[str, float, float, str]]:
    """Return one ``events.tsv`` row per flaw of one label JSON.

    The format is ``sed_eval``'s: ``filename``, ``onset``, ``offset``,
    ``event_label``, with the filename as written on disk so the evaluation
    tooling can find the audio.
    """
    filename = f"{label['file_id']}.wav"
    return [
        (filename, float(f["t_start"]), float(f["t_end"]), str(f["label"]))
        for f in label.get("flaws", ())
    ]


def format_events_tsv(
    labels: Iterable[Mapping[str, Any]],
    *,
    header: bool = False,
) -> str:
    """Render ``labels/events.tsv`` (spec 7.2).

    Rows are sorted by filename then onset, so the file is byte-stable however
    the labels were enumerated - it is covered by ``checksums.sha256``.

    Args:
        labels: Label JSONs, as found in :attr:`InjectionResult.label`.
        header: Write the ``filename/onset/offset/event_label`` header line.
            ``sed_eval`` expects none, so the default is off.

    Returns:
        The file contents, ending in a newline.
    """
    rows = sorted(
        (row for label in labels for row in events_rows(label)),
        key=lambda r: (r[0], r[1], r[2], r[3]),
    )
    lines = ["filename\tonset\toffset\tevent_label"] if header else []
    lines += [f"{name}\t{onset:.3f}\t{offset:.3f}\t{event}" for name, onset, offset, event in rows]
    return "\n".join(lines) + "\n"
