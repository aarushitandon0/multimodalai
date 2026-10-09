#!/usr/bin/env python
"""Plot F0 (semitones), energy and speech rate for one reading (spec 9 acceptance).

Three stacked panels over a common time axis:

1. **F0 in semitones** re the speaker's own median, with the pass-2 Praat range
   and, with ``--crosscheck``, the frames where ``torchcrepe`` disagreed by more
   than 50 cents marked as excluded.
2. **Energy**, in dB re the speaker's voiced median (left axis) - the quantity
   the envelope compares, not raw dB.
3. **Speech rate**: the local syllable rate over a +/- 3-word window from the
   alignment, with the speaker's global articulation rate as a reference line.

Word boundaries are drawn from the MFA TextGrid. Without one (MFA is a separate
conda environment and may not have been run), the script still plots panels 1
and 2 and falls back to the de Jong & Wempe syllable-nuclei rate for panel 3,
marking the figure clearly as having no word boundaries.

Usage::

    python scripts/feature_demo.py                      # first LibriVox reader
    python scripts/feature_demo.py --file-id s1_lv07    # a specific reading
    python scripts/feature_demo.py --audio take.wav     # any WAV, dataset or not
    python scripts/feature_demo.py --crosscheck --seconds 30

Exit codes:
    0  a PNG was written
    2  no audio to plot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.config import Config, load_config
from rostrum.data.manifest import read_manifest
from rostrum.features import energy as E
from rostrum.features import pauses as PA
from rostrum.features import pitch as P
from rostrum.features import rate as R
from rostrum.features.frames import FrameGrid
from rostrum.types import Alignment, Word

logger = logging.getLogger("feature_demo")

NUCLEI_WINDOW_S = 2.0
"""Sliding window for the alignment-free rate fallback."""


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def pick_file(cfg: Config, file_id: str | None) -> tuple[str, Path]:
    """Choose the recording to plot: ``file_id`` if given, else the first clean one."""
    clean_dir = cfg.path("data.audio_clean")
    if file_id:
        path = clean_dir / f"{file_id}.wav"
        if not path.is_file():
            raise SystemExit(f"no such clean recording: {path}")
        return file_id, path

    rows = [r for r in read_manifest(cfg.path("data.manifest")) if r.origin == "clean"]
    for row in sorted(rows, key=lambda r: r.file_id):
        path = clean_dir / f"{row.file_id}.wav"
        if path.is_file():
            return row.file_id, path
    candidates = sorted(clean_dir.glob("*.wav"))
    if not candidates:
        raise SystemExit(f"no clean audio under {clean_dir}")
    return candidates[0].stem, candidates[0]


def load_alignment(
    cfg: Config, file_id: str, duration: float, explicit: Path | None = None
) -> Alignment | None:
    """Read the MFA TextGrid of ``file_id``, or ``None`` if it has not been built."""
    path = explicit if explicit is not None else cfg.path("data.alignments") / f"{file_id}.TextGrid"
    if not path.is_file():
        logger.warning(
            "no alignment at %s - plotting without word boundaries and using the "
            "syllable-nuclei rate instead (run `make align` for the full figure)",
            path,
        )
        return None
    from rostrum.align.mfa import read_textgrid

    return read_textgrid(path, file_id=file_id, duration=duration)


# --------------------------------------------------------------------------- #
# The rate curve
# --------------------------------------------------------------------------- #


def rate_curve_from_words(words: list[Word], times: np.ndarray, window_words: int) -> np.ndarray:
    """Step curve of the local word-window rate, sampled on ``times``."""
    curve = np.full(times.shape, np.nan, dtype=np.float64)
    for i, word in enumerate(words):
        inside = (times >= word.t_start) & (times < word.t_end)
        curve[inside] = R.local_rate(words, i, window_words=window_words)
    return curve


def rate_curve_from_nuclei(
    nuclei: list[float], times: np.ndarray, window_s: float = NUCLEI_WINDOW_S
) -> np.ndarray:
    """Sliding-window nuclei rate, for use when no alignment exists."""
    t = np.asarray(sorted(nuclei), dtype=np.float64)
    curve = np.full(times.shape, np.nan, dtype=np.float64)
    if t.size == 0:
        return curve
    half = window_s / 2.0
    for i, centre in enumerate(times):
        lo, hi = centre - half, centre + half
        count = int(np.sum((t >= lo) & (t < hi)))
        curve[i] = count / window_s
    return curve


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #


def plot(
    *,
    file_id: str,
    out_path: Path,
    grid: FrameGrid,
    pitch: P.PitchTrack,
    energy: E.EnergyTrack,
    rate_curve: np.ndarray,
    rate_label: str,
    global_rate: float | None,
    alignment: Alignment | None,
    t_end: float,
) -> Path:
    """Draw and save the three-panel figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = pitch.times
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Rostrum feature demo - {file_id}"
        + ("" if alignment else "  (no alignment: word boundaries unavailable)"),
        fontsize=13,
    )

    # -- F0 ---------------------------------------------------------------- #
    ax = axes[0]
    ax.plot(times, pitch.f0_st, ".", markersize=2.5, color="#1f77b4", label="F0 (usable frames)")
    excluded = pitch.voiced & ~pitch.usable
    if excluded.any():
        ax.plot(
            times[excluded],
            np.zeros(int(excluded.sum())),
            "x",
            markersize=4,
            color="#d62728",
            label=f"excluded by cross-check ({int(excluded.sum())} frames)",
        )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    median = pitch.median_hz if pitch.median_hz is not None else float("nan")
    ax.set_ylabel("F0 (st re median)")
    ax.set_title(
        f"median {median:.1f} Hz · Praat pass-2 range {pitch.floor_hz:.0f}-"
        f"{pitch.ceiling_hz:.0f} Hz"
        + (
            f" · cross-check agreement {pitch.agreement_rate:.1%}"
            if pitch.agreement_rate is not None
            else " · no cross-check"
        ),
        fontsize=9,
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=8)

    # -- Energy ------------------------------------------------------------ #
    ax = axes[1]
    ax.plot(energy.times, energy.db_rel, linewidth=0.9, color="#2ca02c")
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylabel("energy (dB re voiced median)")
    reference = (
        f"{energy.voiced_median_db:.1f} dB" if energy.voiced_median_db is not None else "n/a"
    )
    correlation = (
        f" · RMS cross-check r={energy.rms_correlation:.3f}"
        if energy.rms_correlation is not None
        else ""
    )
    ax.set_title(f"voiced median {reference}{correlation}", fontsize=9, loc="left")

    # -- Rate -------------------------------------------------------------- #
    ax = axes[2]
    ax.plot(times, rate_curve, linewidth=1.1, color="#9467bd", label=rate_label)
    if global_rate is not None and np.isfinite(global_rate):
        ax.axhline(
            global_rate,
            color="grey",
            linewidth=0.9,
            linestyle="--",
            label=f"global articulation rate {global_rate:.2f} syll/s",
        )
    ax.set_ylabel("rate (syll/s)")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", fontsize=8)

    # -- Word boundaries --------------------------------------------------- #
    if alignment is not None:
        for word in alignment.words:
            if word.t_start > t_end:
                break
            for a in axes:
                a.axvline(word.t_start, color="black", linewidth=0.3, alpha=0.25)
        label_every = max(1, len(alignment.words) // 40)
        for i, word in enumerate(alignment.words):
            if word.t_start > t_end:
                break
            if i % label_every == 0:
                axes[0].text(
                    word.t_start,
                    axes[0].get_ylim()[1],
                    word.text,
                    fontsize=6,
                    rotation=90,
                    va="top",
                )
        for gap in alignment.gaps:
            if gap.duration >= PA.DEFAULT_MIN_PAUSE_S and gap.t_start <= t_end:
                for a in axes:
                    a.axvspan(gap.t_start, gap.t_end, color="#ff7f0e", alpha=0.15)

    for a in axes:
        a.set_xlim(0.0, t_end)
        a.grid(alpha=0.15)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--file-id", default=None, help="clean recording in the dataset to plot")
    p.add_argument(
        "--audio", type=Path, default=None, help="any WAV to plot, inside the dataset or not"
    )
    p.add_argument(
        "--alignment", type=Path, default=None, help="TextGrid for --audio, if there is one"
    )
    p.add_argument("--seconds", type=float, default=30.0, help="how much audio to plot")
    p.add_argument("--out", type=Path, default=None, help="output PNG")
    p.add_argument(
        "--crosscheck",
        action="store_true",
        help="run the torchcrepe cross-check (slow on CPU: seconds per second of audio)",
    )
    p.add_argument("--crosscheck-model", default="tiny", choices=("tiny", "full"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    import soundfile as sf

    cfg = load_config()
    grid = FrameGrid.from_config(cfg)
    if args.audio is not None:
        if not args.audio.is_file():
            logger.error("no such audio file: %s", args.audio)
            return 2
        file_id, path = args.audio.stem, args.audio
    else:
        file_id, path = pick_file(cfg, args.file_id)
    logger.info("plotting %s", path)

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if args.seconds > 0:
        audio = audio[: int(args.seconds * sr)]
    if audio.size < grid.window_samples:
        logger.error("%s is shorter than one analysis window", path)
        return 2
    duration = audio.size / float(sr)

    pitch = P.pitch_track(
        audio,
        sr,
        grid=grid,
        config=cfg,
        crosscheck=args.crosscheck,
        crosscheck_model=args.crosscheck_model,
    )
    energy = E.energy_track(
        audio,
        sr,
        grid=grid,
        config=cfg,
        voiced=pitch.voiced,
        rms_crosscheck=True,
        n_frames=pitch.n_frames,
    )

    alignment = load_alignment(cfg, file_id, duration, args.alignment)
    if alignment is not None and alignment.words:
        window = int(cfg.section("features")["rate"]["local_window_words"])
        rate_curve = rate_curve_from_words(alignment.words, pitch.times, window)
        rate_label = f"local rate (+/- {window} words)"
        global_rate = R.articulation_rate(alignment.words)
    else:
        nuclei = R.syllable_nuclei(audio, sr, grid=grid)
        rate_curve = rate_curve_from_nuclei(nuclei, pitch.times)
        rate_label = f"syllable-nuclei rate ({NUCLEI_WINDOW_S:.0f} s window)"
        global_rate = R.nuclei_rate(nuclei, speech_duration_s=duration)
        logger.info("syllable nuclei: %d over %.1f s", len(nuclei), duration)

    out = args.out or (cfg.path("docs.figures") / f"feature_demo_{file_id}.png")
    plot(
        file_id=file_id,
        out_path=out,
        grid=grid,
        pitch=pitch,
        energy=energy,
        rate_curve=rate_curve,
        rate_label=rate_label,
        global_rate=global_rate,
        alignment=alignment,
        t_end=duration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
