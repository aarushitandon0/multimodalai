"""One call from audio to every feature, with a content-addressed cache (spec 9).

:func:`extract_all` runs the whole of spec 9.1 on one recording and returns a
:class:`FeatureBundle`: three per-unit tables (``word``, ``gap``, ``phrase``)
following ``FEATURE_TABLE_CONTRACT``, the frame curves the dashboard plots, and
a metadata block recording how the numbers were produced - the pitch range the
two-pass extractor settled on, the Praat/CREPE agreement rate, the intensity-RMS
correlation, and the wall-clock cost per stage.

**Caching.** The cache key is the SHA-256 of three things: the processed WAV's
own hash, the configuration hash, and a fingerprint of the alignment. Any of
them changing changes the key, so a stale cache cannot be read by accident, and
a re-run with the same inputs reads the tables back byte for byte. Tables go to
Parquet, metadata to JSON.

Values are **not** rounded on the way into the cache. Rounding happens when a
result is serialised (:mod:`rostrum.types`), never in the computation path, so a
cached run and a fresh run produce the same downstream z-scores (spec 16).

**Order matters.** Pitch runs first because the voicing mask and the speaker's
pitch range are inputs to almost everything else: energy normalises over voiced
frames, prominence needs ``f0_st``, and period detection for jitter and shimmer
needs the two-pass floor and ceiling. Every track is forced to the pitch track's
frame count so that one frame index means one instant across all of them.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from rostrum.features import duration as D
from rostrum.features import egemaps as EG
from rostrum.features import energy as E
from rostrum.features import mfcc as MF
from rostrum.features import pauses as PA
from rostrum.features import pitch as P
from rostrum.features import prominence as PR
from rostrum.features import rate as R
from rostrum.features import spectral as SP
from rostrum.features import voice_quality as VQ
from rostrum.features.frames import FrameGrid
from rostrum.io.hashing import sha256_bytes

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from rostrum.config import Config
    from rostrum.types import Alignment, FeatureTable, Gap, Word

logger = logging.getLogger(__name__)

__all__ = [
    "CACHE_FILES",
    "FEATURE_CACHE_VERSION",
    "OPTIONAL_BLOCKS",
    "FeatureBundle",
    "PhraseSpan",
    "alignment_fingerprint",
    "cache_key",
    "extract_all",
    "extract_all_cached",
    "load_bundle",
    "phrase_spans",
    "save_bundle",
    "word_table",
]

FEATURE_CACHE_VERSION: int = 1
"""Bump to invalidate every cached feature file after a change in this module."""

CACHE_FILES: dict[str, str] = {
    "words": "words.parquet",
    "gaps": "gaps.parquet",
    "phrases": "phrases.parquet",
    "frames": "frames.parquet",
    "meta": "meta.json",
}
"""File names inside one cache entry's directory."""

OPTIONAL_BLOCKS: tuple[str, ...] = (
    "with_spectral",
    "with_mfcc",
    "with_voice_quality",
    "with_prominence",
    "with_egemaps",
)
"""Feature blocks a caller can switch off; part of the cache key."""

FINAL_LOW_WINDOW_S: float = 0.300
"""Window at the end of the utterance that defines the final low F0 (spec 9.2)."""

FINAL_LOW_PERCENTILE: float = 10.0
"""Percentile of ``f0_st`` in that window taken as the final low."""


# --------------------------------------------------------------------------- #
# Phrases
# --------------------------------------------------------------------------- #


class PhraseSpan(NamedTuple):
    """One phrase: the span between two ``boundary`` gaps (spec 9).

    Attributes:
        idx: Phrase index, zero-based.
        t_start: Onset in seconds, the first word's onset.
        t_end: Offset in seconds, the last word's offset.
        first_word_idx: Index of the first word in the phrase.
        last_word_idx: Index of the last word in the phrase.
    """

    idx: int
    t_start: float
    t_end: float
    first_word_idx: int
    last_word_idx: int

    @property
    def duration(self) -> float:
        """Phrase duration in seconds."""
        return self.t_end - self.t_start

    @property
    def n_words(self) -> int:
        """Number of words in the phrase."""
        return self.last_word_idx - self.first_word_idx + 1


def phrase_spans(words: Sequence[Word], gaps: Sequence[Gap]) -> list[PhraseSpan]:
    """Split an alignment into phrases at ``boundary`` gaps.

    The gap label comes from the display transcript's punctuation, not from the
    audio (``rostrum.text.gaps``), so phrase boundaries are identical across
    every rendition of a text. That is what makes a phrase-level feature
    comparable between a reference and a participant.

    Args:
        words: Aligned words, ordered by index.
        gaps: Inter-word gaps, ordered by index; gap ``i`` follows word ``i``.

    Returns:
        Phrases in order. An alignment with no ``boundary`` gap is one phrase.
    """
    if not words:
        return []
    cuts = {int(g.prev_word_idx) for g in gaps if g.label == "boundary"}
    spans: list[PhraseSpan] = []
    first = 0
    for position, word in enumerate(words):
        is_last = position == len(words) - 1
        if word.idx in cuts or is_last:
            spans.append(
                PhraseSpan(
                    idx=len(spans),
                    t_start=float(words[first].t_start),
                    t_end=float(word.t_end),
                    first_word_idx=int(words[first].idx),
                    last_word_idx=int(word.idx),
                )
            )
            first = position + 1
    return spans


def _final_low_st(pitch: P.PitchTrack, words: Sequence[Word]) -> float:
    """The utterance-final low F0 in semitones (spec 9.2).

    Phrase pitch range is measured against the speaker's final low rather than
    against the phrase's own minimum: the final low is the one anchor that is
    stable across readers [R29], while a phrase minimum sits wherever the
    nearest voiceless consonant happens to be.
    """
    if not words or pitch.n_frames == 0:
        return float("nan")
    end = float(words[-1].t_end)
    window = max(end - FINAL_LOW_WINDOW_S, float(words[-1].t_start))
    mask = (pitch.times >= window) & (pitch.times <= end)
    values = pitch.f0_st[mask]
    ok = np.isfinite(values)
    if ok.sum() < 2:
        values = pitch.f0_st
        ok = np.isfinite(values)
        if not ok.any():
            return float("nan")
    return float(np.percentile(values[ok], FINAL_LOW_PERCENTILE))


# --------------------------------------------------------------------------- #
# The bundle
# --------------------------------------------------------------------------- #


@dataclass
class FeatureBundle:
    """Every feature of one recording, in the three units plus frame curves.

    Attributes:
        file_id: Recording identifier.
        words: Per-word table (spec 9.1 F0, EN, DUR, RATE, PROM, SPEC, MFCC, VQ).
        gaps: Per-gap table (PAU).
        phrases: Per-phrase table (phrase pitch range, rate, eGeMAPS).
        frames: Frame curves, one column per signal, with ``time_s`` first.
        meta: Provenance and timings; see :meth:`FeatureBundle.summary`.
    """

    file_id: str
    words: FeatureTable
    gaps: FeatureTable
    phrases: FeatureTable
    frames: FeatureTable
    meta: dict[str, Any] = field(default_factory=dict)

    def table(self, unit: str) -> FeatureTable:
        """Return the table for ``unit`` (``word``, ``gap`` or ``phrase``).

        Raises:
            KeyError: Unknown unit name.
        """
        try:
            return {"word": self.words, "gap": self.gaps, "phrase": self.phrases}[unit]
        except KeyError:
            raise KeyError(f"unknown unit {unit!r}; expected word, gap or phrase") from None

    def summary(self) -> str:
        """One line for a build log: units, duration and cost."""
        duration = float(self.meta.get("audio_duration_s", 0.0) or 0.0)
        elapsed = float(self.meta.get("elapsed_s", 0.0) or 0.0)
        per_minute = (elapsed / duration * 60.0) if duration > 0 else float("nan")
        return (
            f"{self.file_id}: {len(self.words)} words, {len(self.gaps)} gaps, "
            f"{len(self.phrases)} phrases, {duration:.1f} s audio in {elapsed:.1f} s "
            f"({per_minute:.1f} s per minute of audio)"
        )


# --------------------------------------------------------------------------- #
# Assembling the tables
# --------------------------------------------------------------------------- #


def word_table(
    *parts: FeatureTable,
    words: Sequence[Word],
    file_id: str = "",
    channel_id: str = "",
) -> FeatureTable:
    """Join per-word tables side by side, with the word's identity in front.

    Args:
        *parts: Tables indexed by word index, each contributing its own columns.
        words: The aligned words, for the identity columns.
        file_id: File tag for ``attrs``.
        channel_id: Channel tag for ``attrs``.

    Returns:
        One table indexed by ``unit_idx``.

    Raises:
        ValueError: Two parts contribute the same column name.
    """
    import pandas as pd

    identity = pd.DataFrame(
        {
            "text": [w.text for w in words],
            "t_start": [float(w.t_start) for w in words],
            "t_end": [float(w.t_end) for w in words],
        },
        index=pd.Index([w.idx for w in words], name="unit_idx", dtype="int64"),
    )
    seen = set(identity.columns)
    for part in parts:
        clash = seen & set(part.columns)
        if clash:
            raise ValueError(f"duplicate feature columns across tables: {sorted(clash)}")
        seen |= set(part.columns)

    table = pd.concat([identity, *parts], axis=1)
    table.index.name = "unit_idx"
    table.attrs.update(unit="word", file_id=file_id, channel_id=channel_id)
    return table


def _gap_table(
    gaps: Sequence[Gap], *, config: Config | None, file_id: str, channel_id: str
) -> FeatureTable:
    """Per-gap table with the gap's span in front."""
    import pandas as pd

    features = PA.gap_features(gaps, config=config, file_id=file_id, channel_id=channel_id)
    identity = pd.DataFrame(
        {
            "t_start": [float(g.t_start) for g in gaps],
            "t_end": [float(g.t_end) for g in gaps],
        },
        index=pd.Index([g.idx for g in gaps], name="unit_idx", dtype="int64"),
    )
    table = pd.concat([identity, features], axis=1)
    table.index.name = "unit_idx"
    table.attrs.update(features.attrs)
    return table


def _phrase_table(
    spans: Sequence[PhraseSpan],
    words: Sequence[Word],
    pitch: P.PitchTrack,
    *,
    grid: FrameGrid,
    final_low_st: float,
    egemaps: FeatureTable | None,
    file_id: str,
    channel_id: str,
) -> FeatureTable:
    """Per-phrase table: pitch range (spec 9.2), rate, and eGeMAPS functionals."""
    import pandas as pd

    by_idx = {w.idx: w for w in words}
    rows: list[dict[str, float]] = []
    for span in spans:
        members = [
            by_idx[i] for i in range(span.first_word_idx, span.last_word_idx + 1) if i in by_idx
        ]
        idx = grid.frame_indices(span.t_start, span.t_end, pitch.n_frames)
        st = pitch.f0_st[idx] if idx.size else np.empty(0)
        ok = np.isfinite(st)
        f0_max = float(st[ok].max()) if ok.any() else float("nan")
        rows.append(
            {
                "t_start": float(span.t_start),
                "t_end": float(span.t_end),
                "first_word_idx": int(span.first_word_idx),
                "last_word_idx": int(span.last_word_idx),
                "phrase_duration_s": float(span.duration),
                "phrase_n_words": int(span.n_words),
                "phrase_f0_max_st": f0_max,
                "phrase_f0_range_st": f0_max - final_low_st,
                "phrase_rate_syll_per_s": R.articulation_rate(members) if members else float("nan"),
            }
        )

    table = pd.DataFrame(rows)
    table.index = pd.Index([s.idx for s in spans], name="unit_idx", dtype="int64")
    attrs: dict[str, Any] = {
        "unit": "phrase",
        "file_id": file_id,
        "channel_id": channel_id,
        "final_low_st": final_low_st,
    }
    if egemaps is not None:
        attrs.update({k: v for k, v in egemaps.attrs.items() if k.startswith("egemaps")})
        if len(egemaps.columns):
            table = pd.concat([table, egemaps], axis=1)
    table.index.name = "unit_idx"
    table.attrs.update(attrs)
    return table


def _frame_table(
    *,
    grid: FrameGrid,
    pitch: P.PitchTrack,
    energy: E.EnergyTrack,
    spectral: SP.SpectralTrack | None,
    prominence: PR.ProminenceTrack | None,
    harmonicity: VQ.VoiceQualityTrack | None,
    file_id: str,
    channel_id: str,
) -> FeatureTable:
    """The frame curves the dashboard plots, one column per signal.

    MFCCs are deliberately not here: 39 columns per frame would dominate the
    cache, and nothing plots them. Their word means are in the word table.
    """
    import pandas as pd

    n = pitch.n_frames
    columns: dict[str, np.ndarray] = {
        "time_s": grid.centres(n),
        "f0_hz": pitch.f0_hz,
        "f0_st": pitch.f0_st,
        "voiced": pitch.voiced.astype(np.int8),
        "usable": pitch.usable.astype(np.int8),
        "energy_db": energy.db,
        "energy_db_rel": energy.db_rel,
        "energy_z": energy.z,
    }
    if spectral is not None:
        columns.update(
            spectral_centroid_hz=spectral.centroid_hz,
            spectral_flux=spectral.flux,
            spectral_tilt_db_per_khz=spectral.tilt_db_per_khz,
            band_energy_ratio_db=spectral.band_ratio_db,
        )
    if prominence is not None:
        columns.update(
            prominence_composite=prominence.composite,
            prominence_response=prominence.response,
        )
    if harmonicity is not None:
        columns.update(hnr_db=harmonicity.hnr_db)

    table = pd.DataFrame({k: v[:n] for k, v in columns.items()})
    table.index = pd.Index(range(n), name="frame_idx", dtype="int64")
    table.attrs.update(
        unit="frame",
        file_id=file_id,
        channel_id=channel_id,
        hop_s=grid.hop_s,
        window_s=grid.window_s,
    )
    return table


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def extract_all(
    samples: np.ndarray,
    sr: int,
    alignment: Alignment,
    *,
    config: Config | None = None,
    grid: FrameGrid | None = None,
    crosscheck: bool | None = None,
    with_spectral: bool = True,
    with_mfcc: bool = True,
    with_voice_quality: bool = True,
    with_prominence: bool = True,
    with_egemaps: bool = True,
    strict_egemaps: bool = False,
    audio_sha256: str = "",
) -> FeatureBundle:
    """Extract every feature of spec 9.1 for one recording.

    Args:
        samples: Mono audio of the processed WAV, 16 kHz float.
        sr: Sample rate.
        alignment: The recording's forced alignment; supplies words and gaps.
        config: Loaded config; every module reads its own section.
        grid: Frame grid; defaults to the project grid.
        crosscheck: Run the CREPE pitch cross-check. Defaults to the config.
            It costs seconds per second of audio on CPU, so the dataset build
            enables it and interactive analysis does not.
        with_spectral: Compute the SPEC block.
        with_mfcc: Compute the MFCC block.
        with_voice_quality: Compute the VQ block.
        with_prominence: Compute the PROM block.
        with_egemaps: Compute the EG block (phrase level).
        strict_egemaps: Fail rather than skip when openSMILE is unavailable.
        audio_sha256: Hash of the WAV these features describe; stored in the
            metadata and used as part of the cache key.

    Returns:
        A :class:`FeatureBundle`.

    Raises:
        ValueError: ``sr`` does not match the grid's sample rate.
    """
    grid = grid or (FrameGrid.from_config(config) if config is not None else FrameGrid())
    if sr != grid.sample_rate:
        raise ValueError(
            f"audio is {sr} Hz but the frame grid is {grid.sample_rate} Hz; "
            "resample with rostrum.io.audio.preprocess first (invariant 1)"
        )

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    file_id = alignment.file_id
    channel_id = alignment.channel_id
    words = list(alignment.words)
    gaps = list(alignment.gaps) or PA.gaps_from_words(words)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def stage(name: str, fn: Any) -> Any:
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t0, 4)
        return out

    pitch = stage(
        "pitch", lambda: P.pitch_track(x, sr, grid=grid, config=config, crosscheck=crosscheck)
    )
    n_frames = pitch.n_frames
    energy = stage(
        "energy",
        lambda: E.energy_track(
            x, sr, grid=grid, config=config, voiced=pitch.voiced, n_frames=n_frames
        ),
    )

    parts: list[FeatureTable] = [
        stage(
            "pitch_words",
            lambda: P.word_pitch(pitch, words, grid=grid, file_id=file_id, channel_id=channel_id),
        ),
        stage(
            "energy_words",
            lambda: E.word_energy(energy, words, grid=grid, file_id=file_id, channel_id=channel_id),
        ),
        stage(
            "duration_words",
            lambda: D.word_duration_features(words, file_id=file_id, channel_id=channel_id),
        ),
        stage(
            "rate_words",
            lambda: R.word_rate(words, config=config, file_id=file_id, channel_id=channel_id),
        ),
    ]

    prominence: PR.ProminenceTrack | None = None
    if with_prominence:
        prominence = stage(
            "prominence",
            lambda: PR.prominence_track(pitch, energy, words, grid=grid, config=config),
        )
        parts.append(
            stage(
                "prominence_words",
                lambda: PR.word_prominence(
                    prominence, words, grid=grid, file_id=file_id, channel_id=channel_id
                ),
            )
        )

    spectral: SP.SpectralTrack | None = None
    if with_spectral:
        spectral = stage(
            "spectral",
            lambda: SP.spectral_track(
                x, sr, grid=grid, config=config, n_frames=n_frames, channel_id=channel_id
            ),
        )
        parts.append(
            stage(
                "spectral_words",
                lambda: SP.word_spectral(spectral, words, grid=grid, file_id=file_id),
            )
        )

    if with_mfcc:
        cepstra = stage(
            "mfcc",
            lambda: MF.mfcc_track(
                x, sr, grid=grid, config=config, n_frames=n_frames, channel_id=channel_id
            ),
        )
        parts.append(
            stage("mfcc_words", lambda: MF.word_mfcc(cepstra, words, grid=grid, file_id=file_id))
        )

    harmonicity: VQ.VoiceQualityTrack | None = None
    if with_voice_quality:
        harmonicity = stage(
            "harmonicity",
            lambda: VQ.harmonicity_track(
                x,
                sr,
                grid=grid,
                floor_hz=pitch.floor_hz,
                ceiling_hz=pitch.ceiling_hz,
                n_frames=n_frames,
                channel_id=channel_id,
            ),
        )
        parts.append(
            stage(
                "voice_quality_words",
                lambda: VQ.word_voice_quality(
                    x,
                    sr,
                    words,
                    grid=grid,
                    config=config,
                    floor_hz=pitch.floor_hz,
                    ceiling_hz=pitch.ceiling_hz,
                    track=harmonicity,
                    file_id=file_id,
                    channel_id=channel_id,
                ),
            )
        )

    spans = phrase_spans(words, gaps)
    egemaps: FeatureTable | None = None
    if with_egemaps:
        egemaps = stage(
            "egemaps",
            lambda: EG.phrase_egemaps(
                x,
                sr,
                [(s.t_start, s.t_end) for s in spans],
                config=config,
                strict=strict_egemaps,
                file_id=file_id,
                channel_id=channel_id,
            ),
        )

    final_low = _final_low_st(pitch, words)
    bundle = FeatureBundle(
        file_id=file_id,
        words=word_table(*parts, words=words, file_id=file_id, channel_id=channel_id),
        gaps=_gap_table(gaps, config=config, file_id=file_id, channel_id=channel_id),
        phrases=_phrase_table(
            spans,
            words,
            pitch,
            grid=grid,
            final_low_st=final_low,
            egemaps=egemaps,
            file_id=file_id,
            channel_id=channel_id,
        ),
        frames=_frame_table(
            grid=grid,
            pitch=pitch,
            energy=energy,
            spectral=spectral,
            prominence=prominence,
            harmonicity=harmonicity,
            file_id=file_id,
            channel_id=channel_id,
        ),
    )
    bundle.meta = {
        "file_id": file_id,
        "text_id": alignment.text_id,
        "speaker_id": alignment.speaker_id,
        "channel_id": channel_id,
        "tier": alignment.tier,
        "audio_sha256": audio_sha256 or alignment.audio_sha256,
        "audio_duration_s": float(x.size) / float(sr),
        "sample_rate": int(sr),
        "n_frames": int(n_frames),
        "hop_s": grid.hop_s,
        "window_s": grid.window_s,
        "n_words": len(words),
        "n_gaps": len(gaps),
        "n_phrases": len(spans),
        "pitch_floor_hz": pitch.floor_hz,
        "pitch_ceiling_hz": pitch.ceiling_hz,
        "pitch_two_pass": bool(pitch.two_pass),
        "pitch_median_hz": pitch.median_hz,
        "pitch_voiced_frac": pitch.voiced_frac,
        "pitch_crosscheck": pitch.crosscheck,
        "pitch_agreement_rate": pitch.agreement_rate,
        "energy_voiced_median_db": energy.voiced_median_db,
        "energy_rms_correlation": energy.rms_correlation,
        "final_low_st": final_low,
        "prominence_scale_range_s": (
            list(prominence.scale_range_s) if prominence is not None else None
        ),
        "egemaps_available": bool(egemaps is not None and egemaps.attrs.get("egemaps_available")),
        "egemaps_error": (egemaps.attrs.get("egemaps_error") if egemaps is not None else None),
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "config_hash": config.hash if config is not None else "",
        "stage_timings_s": timings,
        "elapsed_s": round(time.perf_counter() - started, 4),
    }
    return bundle


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def alignment_fingerprint(alignment: Alignment) -> str:
    """A hash of the word and gap timings features depend on.

    Not the whole alignment: the aligner version string and the OOV list do not
    change a single feature value, while a shifted word boundary changes every
    aggregate in the word's row.
    """
    payload = {
        "words": [
            [w.idx, w.text, round(float(w.t_start), 3), round(float(w.t_end), 3), list(w.phones)]
            for w in alignment.words
        ],
        "gaps": [
            [g.idx, round(float(g.t_start), 3), round(float(g.t_end), 3), g.label]
            for g in alignment.gaps
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(blob.encode("utf-8"))


def cache_key(
    *,
    audio_sha256: str,
    config_hash: str,
    alignment_hash: str = "",
    options: Sequence[str] = (),
) -> str:
    """The cache key: audio hash + config hash + alignment + option set.

    Args:
        audio_sha256: Hash of the processed WAV.
        config_hash: :attr:`Config.hash`.
        alignment_hash: :func:`alignment_fingerprint` of the alignment used.
        options: Names of the optional blocks that were computed, so a cache
            entry built without, say, eGeMAPS is never mistaken for a full one.

    Returns:
        A 64-character hex digest.
    """
    blob = json.dumps(
        {
            "audio": audio_sha256,
            "config": config_hash,
            "alignment": alignment_hash,
            "options": sorted(options),
            "version": FEATURE_CACHE_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(blob.encode("utf-8"))


def _entry_dir(cache_dir: Path | str, key: str) -> Path:
    """Directory holding one cache entry."""
    return Path(cache_dir) / key


def save_bundle(bundle: FeatureBundle, cache_dir: Path | str, key: str) -> Path:
    """Write a bundle to its cache entry and return the directory.

    Tables go to Parquet, which round-trips float64, the integer index and the
    column order exactly; ``attrs`` do not survive Parquet, so they are rebuilt
    from the metadata on load.
    """
    target = _entry_dir(cache_dir, key)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("words", "gaps", "phrases", "frames"):
        table = getattr(bundle, name)
        table.to_parquet(target / CACHE_FILES[name], index=True)
    bundle.meta["cache_key"] = key
    meta = dict(bundle.meta)
    meta["attrs"] = {
        name: dict(getattr(bundle, name).attrs) for name in ("words", "gaps", "phrases", "frames")
    }
    (target / CACHE_FILES["meta"]).write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    logger.debug("cached features for %s at %s", bundle.file_id, target)
    return target


def load_bundle(cache_dir: Path | str, key: str) -> FeatureBundle | None:
    """Read a cache entry, or ``None`` when it is missing or incomplete."""
    import pandas as pd

    target = _entry_dir(cache_dir, key)
    if not target.is_dir():
        return None
    paths = {name: target / filename for name, filename in CACHE_FILES.items()}
    if not all(p.is_file() for p in paths.values()):
        logger.debug("cache entry %s is incomplete; recomputing", target)
        return None

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    tables: dict[str, pd.DataFrame] = {}
    for name in ("words", "gaps", "phrases", "frames"):
        table = pd.read_parquet(paths[name])
        table.attrs.update(meta.get("attrs", {}).get(name, {}))
        tables[name] = table
    return FeatureBundle(
        file_id=str(meta.get("file_id", "")),
        words=tables["words"],
        gaps=tables["gaps"],
        phrases=tables["phrases"],
        frames=tables["frames"],
        meta=meta,
    )


def extract_all_cached(
    samples: np.ndarray,
    sr: int,
    alignment: Alignment,
    *,
    config: Config,
    cache_dir: Path | str | None = None,
    audio_sha256: str = "",
    refresh: bool = False,
    **kwargs: Any,
) -> tuple[FeatureBundle, bool]:
    """Extract features, reading from and writing to the Parquet cache.

    Args:
        samples: Mono audio of the processed WAV.
        sr: Sample rate.
        alignment: The recording's alignment.
        config: Loaded config; supplies the cache directory and the hash.
        cache_dir: Override the configured cache directory.
        audio_sha256: Hash of the WAV. Falls back to
            :attr:`Alignment.audio_sha256`; without either, caching is skipped
            because the key would not identify the audio.
        refresh: Recompute and overwrite an existing entry.
        **kwargs: Passed to :func:`extract_all`.

    Returns:
        ``(bundle, from_cache)``.
    """
    directory = Path(cache_dir) if cache_dir is not None else config.path("cache.features")
    digest = audio_sha256 or alignment.audio_sha256
    # The key records the options actually in force, defaults included: an entry
    # built without eGeMAPS must not be served to a caller that asked for it.
    effective = dict.fromkeys(OPTIONAL_BLOCKS, True)
    effective.update({k: bool(v) for k, v in kwargs.items() if k in OPTIONAL_BLOCKS})
    effective["crosscheck"] = bool(
        kwargs.get("crosscheck", config.features["pitch"]["crosscheck"]["enabled"])
    )
    options = [name for name, on in sorted(effective.items()) if on]
    if not digest:
        logger.warning(
            "no audio hash for %s; computing features without caching", alignment.file_id
        )
        return (
            extract_all(samples, sr, alignment, config=config, audio_sha256="", **kwargs),
            False,
        )

    key = cache_key(
        audio_sha256=digest,
        config_hash=config.hash,
        alignment_hash=alignment_fingerprint(alignment),
        options=options,
    )
    if not refresh:
        cached = load_bundle(directory, key)
        if cached is not None:
            return cached, True

    bundle = extract_all(samples, sr, alignment, config=config, audio_sha256=digest, **kwargs)
    save_bundle(bundle, directory, key)
    return bundle, False
