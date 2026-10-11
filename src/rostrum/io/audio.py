"""Audio loading and the canonical preprocessing chain (spec 4.6).

Every recording that enters Rostrum goes through :func:`preprocess` exactly
once, in this order:

1. decode to float32 (:func:`load_any`),
2. mono + 16 kHz (:func:`to_mono_16k`),
3. trim to the chosen excerpt (:func:`trim_excerpt`),
4. trim leading/trailing silence down to 300 ms (:func:`keep_padding`),
5. loudness-normalise to -23 LUFS, peak-limit to -1 dBFS
   (:func:`loudness_normalize`),
6. write 16-bit PCM WAV (:func:`write_wav16`) and hash it
   (:func:`sha256_file`, re-exported here).

The numbers above are defaults mirroring ``configs/features.yaml``
(``audio.target_lufs``, ``audio.peak_ceiling_dbfs``,
``audio.trim_silence_keep_s``); callers that have a :class:`~rostrum.config.Config`
should pass them in rather than rely on the defaults, so one config edit moves
the whole dataset (``CLAUDE.md`` coding conventions).

Everything but :func:`load_any`/:func:`write_wav16` is a pure array function, so
the invariants - 16 kHz mono float32 internally, 16-bit PCM on disk, -23 LUFS,
peak <= -1 dBFS (``CLAUDE.md`` invariant 1) - are unit-testable on synthetic
signals.

Resampling uses ``soxr`` directly and the lead/tail trim is plain NumPy, so
this module does not import ``librosa`` (and therefore ``numba``) at all on the
happy path - only the decode *fallback* reaches for it, lazily. That keeps the
chain fast to start, byte-reproducible, and usable on a machine whose numba
install is broken.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from rostrum.io.hashing import sha256_bytes, sha256_file

__all__ = [
    "DEFAULT_KEEP_PAD_S",
    "DEFAULT_PEAK_DBFS",
    "DEFAULT_TARGET_LUFS",
    "DEFAULT_TOP_DB",
    "SAMPLE_RATE",
    "AudioError",
    "AudioStats",
    "describe",
    "duration_s",
    "integrated_lufs",
    "keep_padding",
    "load_any",
    "loudness_normalize",
    "peak_dbfs",
    "preprocess",
    "read_mono",
    "sha256_bytes",
    "sha256_file",
    "to_mono_16k",
    "trim_excerpt",
    "write_wav16",
]

logger = logging.getLogger(__name__)

SAMPLE_RATE: int = 16_000
"""Working sample rate (``CLAUDE.md`` invariant 1)."""

DEFAULT_TARGET_LUFS: float = -23.0
"""Integrated loudness target, EBU R128 (spec 4.6 step 4)."""

DEFAULT_PEAK_DBFS: float = -1.0
"""Sample-peak ceiling after normalisation (spec 4.6 step 4)."""

DEFAULT_KEEP_PAD_S: float = 0.300
"""Lead/tail silence kept around the speech (spec 4.6 step 3)."""

DEFAULT_TOP_DB: float = 45.0
"""Silence threshold for :func:`keep_padding`, in dB below the clip's peak."""

_RESAMPLE_QUALITY = "VHQ"
"""Fixed ``soxr`` quality: deterministic output for the same input (invariant 13)."""

_TRIM_FRAME_S = 0.025
_TRIM_HOP_S = 0.010
"""Frame grid for the lead/tail silence search (``CLAUDE.md`` invariant 3)."""

_MIN_LUFS_BLOCK_S = 0.400
"""``pyloudnorm`` needs one 400 ms block to measure integrated loudness."""

_SILENT_DBFS = -120.0
"""Reported peak for an all-zero signal, instead of ``-inf``."""


class AudioError(RuntimeError):
    """A recording could not be decoded, measured or written."""


@dataclass(frozen=True)
class AudioStats:
    """Measurements of one clip, as recorded in the build QC table.

    Attributes:
        sample_rate: Samples per second.
        n_samples: Length in samples.
        duration_s: Length in seconds, rounded to 3 decimals.
        lufs: Integrated loudness (EBU R128), or ``None`` if the clip is shorter
            than one 400 ms measurement block.
        peak_dbfs: Sample peak in dBFS.
    """

    sample_rate: int
    n_samples: int
    duration_s: float
    lufs: float | None
    peak_dbfs: float


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def load_any(path: Path | str, *, allow_ffmpeg: bool = True) -> tuple[np.ndarray, int]:
    """Decode any audio file to float32 samples at its native rate.

    Three decoders are tried in order, because the dataset mixes MP3 (LibriVox),
    OGG Vorbis (Wikimedia) and whatever a team phone produced:

    1. ``libsndfile`` via :mod:`soundfile` - handles WAV/FLAC/OGG, and MP3 from
       libsndfile 1.1;
    2. :mod:`librosa` (``audioread``/``av`` backends);
    3. ``ffmpeg`` as a subprocess, decoding to a temporary 32-bit float WAV.

    Args:
        path: File to decode.
        allow_ffmpeg: Whether the ``ffmpeg`` fallback may be used.

    Returns:
        ``(samples, sample_rate)``. ``samples`` is float32, shape ``(n,)`` for
        mono or ``(n, channels)`` for multichannel, as stored on disk.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        AudioError: Every decoder failed; the message lists what each said.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"audio file not found: {src}")

    failures: list[str] = []
    try:
        data, sr = sf.read(src, dtype="float32", always_2d=False)
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
    except Exception as exc:
        failures.append(f"soundfile: {exc}")

    try:
        import librosa

        data, sr = librosa.load(str(src), sr=None, mono=False, dtype=np.float32)
        # librosa returns (channels, n); this module works in (n, channels).
        if data.ndim == 2:
            data = data.T
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
    except Exception as exc:
        failures.append(f"librosa: {exc}")

    if allow_ffmpeg:
        try:
            return _load_via_ffmpeg(src)
        except Exception as exc:
            failures.append(f"ffmpeg: {exc}")

    raise AudioError(f"could not decode {src}:\n  " + "\n  ".join(failures))


def read_mono(
    path: Path | str,
    *,
    sr: int = SAMPLE_RATE,
    allow_ffmpeg: bool = True,
) -> tuple[np.ndarray, int]:
    """Decode a file to mono float32 at ``sr`` (spec 4.6 step 1).

    This is what every stage downstream of :func:`preprocess` wants: the
    processed WAV is already 16 kHz mono 16-bit PCM, so the read is a plain
    ``libsndfile`` call and the downmix and resample are no-ops. Going through
    :func:`load_any` and :func:`to_mono_16k` rather than ``librosa.load``
    keeps the happy path free of ``numba``, which is both faster to start and
    the difference between running and not running on a machine whose
    ``llvmlite`` shared object will not load.

    Args:
        path: File to decode.
        sr: Target sample rate.
        allow_ffmpeg: Whether :func:`load_any` may fall back to ``ffmpeg``.

    Returns:
        ``(samples, sr)`` - contiguous mono float32 at ``sr``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        AudioError: Every decoder failed, or the signal was empty.
    """
    samples, native = load_any(path, allow_ffmpeg=allow_ffmpeg)
    return to_mono_16k(samples, native, target_sr=int(sr)), int(sr)


def _load_via_ffmpeg(src: Path) -> tuple[np.ndarray, int]:
    """Decode ``src`` with ``ffmpeg`` into a temporary float WAV, then read it.

    Raises:
        AudioError: ``ffmpeg`` is not on ``PATH`` or exited non-zero.
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioError("ffmpeg is not on PATH")
    with tempfile.TemporaryDirectory(prefix="rostrum-decode-") as tmp:
        out = Path(tmp) / "decoded.wav"
        cmd = [exe, "-nostdin", "-v", "error", "-y", "-i", str(src), "-c:a", "pcm_f32le", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.is_file():
            raise AudioError(f"exit {proc.returncode}: {(proc.stderr or '').strip()}")
        data, sr = sf.read(out, dtype="float32", always_2d=False)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


# --------------------------------------------------------------------------- #
# Channel / rate
# --------------------------------------------------------------------------- #


def to_mono_16k(
    samples: np.ndarray,
    sr: int,
    *,
    target_sr: int = SAMPLE_RATE,
    quality: str = _RESAMPLE_QUALITY,
) -> np.ndarray:
    """Downmix to mono and resample to ``target_sr`` (spec 4.6 step 1).

    Downmix is the mean of the channels, which is what ``ffmpeg -ac 1`` does.
    The resampler is fixed (``soxr`` at ``VHQ``, the same engine ``librosa``
    defaults to) so the same input always gives the same samples.

    Args:
        samples: ``(n,)`` or ``(n, channels)`` float array.
        sr: Sample rate of ``samples``.
        target_sr: Desired rate; 16 kHz unless a test says otherwise.
        quality: ``soxr`` quality setting.

    Returns:
        Contiguous float32, shape ``(n_out,)``.

    Raises:
        AudioError: ``samples`` is empty or has more than two dimensions.
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim > 2:
        raise AudioError(f"expected mono or (n, channels) audio, got shape {x.shape}")
    if x.size == 0:
        raise AudioError("cannot process an empty signal")
    if x.ndim == 2:
        x = x.mean(axis=1, dtype=np.float32)

    if int(sr) != int(target_sr):
        x = soxr.resample(x, int(sr), int(target_sr), quality=quality)
    return np.ascontiguousarray(x, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Trimming
# --------------------------------------------------------------------------- #


def trim_excerpt(
    samples: np.ndarray, sr: int, start: float | None = None, end: float | None = None
) -> np.ndarray:
    """Cut ``samples`` to ``[start, end)`` seconds (spec 4.6 step 2).

    ``None`` means "from the beginning" / "to the end". Bounds are clamped to
    the signal, so an ``end`` past the file is not an error - a recording that
    was re-encoded a little shorter should not fail the build.

    Args:
        samples: Mono float array.
        sr: Sample rate.
        start: Excerpt start in seconds, or ``None``.
        end: Excerpt end in seconds, or ``None``.

    Returns:
        The excerpt, as a copy.

    Raises:
        AudioError: ``start``/``end`` are negative, ``end <= start``, or the
            excerpt is empty.
    """
    x = np.asarray(samples, dtype=np.float32)
    n = x.shape[0]
    i0 = 0 if start is None else round(float(start) * sr)
    i1 = n if end is None else round(float(end) * sr)
    if i0 < 0 or i1 < 0:
        raise AudioError(f"excerpt bounds must be non-negative, got start={start}, end={end}")
    if i1 <= i0:
        raise AudioError(f"excerpt end ({end}) must be after start ({start})")
    i0, i1 = min(i0, n), min(i1, n)
    if i1 <= i0:
        raise AudioError(f"excerpt [{start}, {end}) lies outside the {n / sr:.3f} s recording")
    return np.ascontiguousarray(x[i0:i1], dtype=np.float32)


def _speech_span(samples: np.ndarray, sr: int, *, top_db: float) -> tuple[int, int]:
    """Return ``(first, last)`` sample indices of non-silent audio.

    Frames are ``25 ms`` long on a ``10 ms`` hop (invariant 3) and a frame is
    silent when its RMS is more than ``top_db`` below the loudest frame's.
    ``(0, 0)`` means every frame is silent.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.shape[0]
    frame = max(2, round(_TRIM_FRAME_S * sr))
    hop = max(1, round(_TRIM_HOP_S * sr))
    if n < frame:
        return (0, n) if np.any(x) else (0, 0)

    n_frames = 1 + (n - frame) // hop
    # One strided view instead of a copy: a 10-minute file is 60k frames.
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame), strides=(x.strides[0] * hop, x.strides[0])
    )
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    if not np.any(rms > 0.0):
        return 0, 0
    threshold = rms.max() * (10.0 ** (-abs(top_db) / 20.0))
    loud = np.flatnonzero(rms > threshold)
    if loud.size == 0:
        return 0, 0
    first = int(loud[0]) * hop
    last = min(n, int(loud[-1]) * hop + frame)
    return first, last


def keep_padding(
    samples: np.ndarray,
    sr: int,
    *,
    keep_s: float = DEFAULT_KEEP_PAD_S,
    top_db: float = DEFAULT_TOP_DB,
) -> np.ndarray:
    """Trim leading/trailing silence down to ``keep_s`` seconds (spec 4.6 step 3).

    Speech onset and offset are the first and last frames on the 25 ms/10 ms
    grid whose RMS is within ``top_db`` of the loudest frame; then ``keep_s`` of
    the recording's *own* silence is kept on each side. Nothing is synthesised:
    if a side has less than ``keep_s`` of silence, all of it is kept and a
    warning is logged, because inserting digital zeros would create a
    noise-floor discontinuity the pause features would then measure
    (``CLAUDE.md`` invariant 8 forbids zero-fill).

    Args:
        samples: Mono float array.
        sr: Sample rate.
        keep_s: Silence to keep on each side.
        top_db: Frames more than this many dB below the loudest frame count as
            silence.

    Returns:
        The trimmed signal, as a copy. An all-silent clip is returned unchanged.

    Raises:
        AudioError: ``keep_s`` is negative.
    """
    if keep_s < 0:
        raise AudioError(f"keep_s must be >= 0, got {keep_s}")

    x = np.asarray(samples, dtype=np.float32)
    speech_start, speech_end = _speech_span(x, sr, top_db=top_db)
    if speech_end <= speech_start:
        logger.warning("keep_padding: no frame above the silence threshold; keeping the clip as is")
        return np.ascontiguousarray(x, dtype=np.float32)

    pad = round(keep_s * sr)
    i0 = max(0, speech_start - pad)
    i1 = min(x.shape[0], speech_end + pad)
    if speech_start - i0 < pad or i1 - speech_end < pad:
        logger.warning(
            "keep_padding: only %.3f s lead / %.3f s tail silence available, wanted %.3f s each",
            (speech_start - i0) / sr,
            (i1 - speech_end) / sr,
            keep_s,
        )
    return np.ascontiguousarray(x[i0:i1], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def duration_s(samples: np.ndarray, sr: int) -> float:
    """Return the duration in seconds, rounded to 3 decimals (invariant 2)."""
    return round(np.asarray(samples).shape[0] / float(sr), 3)


def peak_dbfs(samples: np.ndarray) -> float:
    """Return the sample peak in dBFS; ``-120.0`` for an all-zero signal."""
    peak = float(np.max(np.abs(np.asarray(samples, dtype=np.float64)), initial=0.0))
    if peak <= 0.0:
        return _SILENT_DBFS
    return 20.0 * math.log10(peak)


def integrated_lufs(samples: np.ndarray, sr: int) -> float | None:
    """Return EBU R128 integrated loudness, or ``None`` for a too-short clip.

    Args:
        samples: Mono float array.
        sr: Sample rate.

    Returns:
        LUFS, or ``None`` when the clip is shorter than one 400 ms block or is
        digital silence (both of which make the measurement undefined).
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.shape[0] < int(_MIN_LUFS_BLOCK_S * sr):
        return None
    import pyloudnorm

    meter = pyloudnorm.Meter(int(sr))
    value = float(meter.integrated_loudness(x))
    return None if not math.isfinite(value) else value


def describe(samples: np.ndarray, sr: int) -> AudioStats:
    """Measure a clip for the QC table."""
    x = np.asarray(samples, dtype=np.float32)
    return AudioStats(
        sample_rate=int(sr),
        n_samples=int(x.shape[0]),
        duration_s=duration_s(x, sr),
        lufs=integrated_lufs(x, sr),
        peak_dbfs=round(peak_dbfs(x), 3),
    )


# --------------------------------------------------------------------------- #
# Loudness
# --------------------------------------------------------------------------- #


def loudness_normalize(
    samples: np.ndarray,
    sr: int,
    *,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    peak_ceiling_dbfs: float = DEFAULT_PEAK_DBFS,
) -> np.ndarray:
    """Normalise to ``target_lufs``, then hold the peak at ``peak_ceiling_dbfs``.

    Both stages are a single broadband gain, never a limiter or clipper: a
    dynamics processor would change the energy contour, which is one of the
    features being measured (spec 9.1).

    Because the peak stage can only attenuate, a clip with an unusually high
    crest factor ends up quieter than ``target_lufs``; that is the invariant
    priority (peak <= -1 dBFS is hard, invariant 1) and it is logged.

    Args:
        samples: Mono float array.
        sr: Sample rate.
        target_lufs: Integrated loudness target.
        peak_ceiling_dbfs: Sample-peak ceiling.

    Returns:
        Float32 samples meeting the peak ceiling.

    Raises:
        AudioError: The clip is digital silence, or too short to measure (no
            400 ms block), so no gain can be derived.
    """
    x = np.asarray(samples, dtype=np.float64)
    if not np.any(x):
        raise AudioError("cannot loudness-normalise digital silence")

    measured = integrated_lufs(x, sr)
    if measured is None:
        raise AudioError(
            f"clip is {x.shape[0] / sr:.3f} s, shorter than the {_MIN_LUFS_BLOCK_S:.1f} s "
            "pyloudnorm needs for integrated loudness"
        )

    y = x * (10.0 ** ((target_lufs - measured) / 20.0))

    ceiling = 10.0 ** (peak_ceiling_dbfs / 20.0)
    peak = float(np.max(np.abs(y)))
    if peak > ceiling:
        y *= ceiling / peak
        logger.debug(
            "peak-limited by %.2f dB; integrated loudness is now %.2f LUFS, below the %.1f target",
            20.0 * math.log10(peak / ceiling),
            target_lufs - 20.0 * math.log10(peak / ceiling),
            target_lufs,
        )
    return np.ascontiguousarray(y, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def write_wav16(path: Path | str, samples: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write mono 16-bit PCM WAV atomically (invariant 1).

    Written to a sibling ``.tmp`` then :func:`os.replace`-d into place, so an
    interrupted build never leaves a half-written WAV that a later stage would
    happily hash.

    Args:
        path: Destination ``.wav``; parent directories are created.
        samples: Mono float array, expected within ``[-1, 1]``.
        sr: Sample rate to record in the header.

    Returns:
        The written path.

    Raises:
        AudioError: ``samples`` is not mono, is empty, or exceeds full scale
            (which would wrap on conversion to PCM).
    """
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1:
        raise AudioError(f"write_wav16 expects mono audio, got shape {x.shape}")
    if x.size == 0:
        raise AudioError("refusing to write an empty WAV")
    peak = float(np.max(np.abs(x)))
    if peak > 1.0:
        raise AudioError(f"samples exceed full scale (peak {peak:.4f}); normalise before writing")

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    sf.write(tmp, x, int(sr), subtype="PCM_16", format="WAV")
    os.replace(tmp, dest)
    logger.debug("wrote %s (%.3f s, %d Hz)", dest, duration_s(x, sr), sr)
    return dest


# --------------------------------------------------------------------------- #
# The whole chain
# --------------------------------------------------------------------------- #


def preprocess(
    src: Path | str,
    dest: Path | str,
    *,
    excerpt_start_s: float | None = None,
    excerpt_end_s: float | None = None,
    target_sr: int = SAMPLE_RATE,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    peak_ceiling_dbfs: float = DEFAULT_PEAK_DBFS,
    keep_s: float = DEFAULT_KEEP_PAD_S,
    top_db: float = DEFAULT_TOP_DB,
) -> tuple[Path, AudioStats]:
    """Run spec 4.6 end to end on one file and write the processed WAV.

    Args:
        src: Any decodable audio file (the raw download or a team recording).
        dest: Destination ``.wav``.
        excerpt_start_s: Excerpt start in the *source* file, or ``None``.
        excerpt_end_s: Excerpt end in the source file, or ``None``.
        target_sr: Output sample rate.
        target_lufs: Integrated loudness target.
        peak_ceiling_dbfs: Sample-peak ceiling.
        keep_s: Lead/tail silence to keep.
        top_db: Silence threshold for the lead/tail trim.

    Returns:
        ``(written_path, stats)``, where ``stats`` describes the file on disk -
        re-measured after writing, so the numbers in ``MANIFEST.csv`` are the
        numbers a reader of the WAV gets, quantisation included.

    Raises:
        FileNotFoundError: ``src`` does not exist.
        AudioError: Any stage of the chain failed.
    """
    raw, sr = load_any(src)
    mono = to_mono_16k(raw, sr, target_sr=target_sr)
    excerpt = trim_excerpt(mono, target_sr, excerpt_start_s, excerpt_end_s)
    padded = keep_padding(excerpt, target_sr, keep_s=keep_s, top_db=top_db)
    normalised = loudness_normalize(
        padded, target_sr, target_lufs=target_lufs, peak_ceiling_dbfs=peak_ceiling_dbfs
    )
    written = write_wav16(dest, normalised, target_sr)

    on_disk, disk_sr = sf.read(written, dtype="float32", always_2d=False)
    return written, describe(on_disk, disk_sr)
