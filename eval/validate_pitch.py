"""Praat vs CREPE pitch agreement (spec 14.1 "Components").

The pitch feature is the backbone of three of the six rubric categories, and it
comes from one algorithm - Praat's autocorrelation tracker, run twice to narrow
the speaker's range. This check asks a neural tracker with completely different
failure modes, ``torchcrepe``, the same question and reports **how often the two
agree within 50 cents**, per recording, per tier and per speaker.

50 cents is half a semitone: the threshold the feature extractor itself uses to
decide whether a frame may enter a statistic
(``configs/features.yaml``: ``pitch.crosscheck.agreement_cents``). The number
reported here is therefore not decoration - it is the fraction of frames the
pipeline is willing to measure on, and a tier whose agreement is low is a tier
whose pitch deviations should be read with suspicion.

Only commonly voiced frames are compared. Praat marks unvoiced frames ``NaN``
and CREPE's periodicity gate does the same, and a frame neither tracker voiced
is not a disagreement about pitch.

Outputs
    ``results/components.json`` section ``pitch``;
    ``docs/figures/components_pitch_cents_hist.png`` (pooled error histogram);
    ``docs/figures/components_pitch_agreement_by_tier.png``.

Usage::

    python -m eval.validate_pitch                      # every clean recording
    python -m eval.validate_pitch --seconds 30         # first 30 s of each
    python -m eval.validate_pitch --model tiny         # faster, less accurate
    python -m eval.validate_pitch --origin clean --origin injected

Exit codes:
    0  the section was written with measured numbers
    2  no audio to measure; the section is written as pending
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import components as C
from rostrum.config import Config, load_config
from rostrum.data.manifest import clean_file_ids, parent_file_id, read_manifest
from rostrum.features import pitch as P
from rostrum.features.frames import FrameGrid
from rostrum.io.audio import load_any

logger = logging.getLogger("validate_pitch")

HIST_CLIP_CENTS = 300.0
"""Histogram range; beyond half an octave the error is an octave slip, not noise."""

OCTAVE_CENTS = 1200.0
"""Errors within 50 cents of an octave are counted separately as octave slips."""


@dataclass
class FileAgreement:
    """Praat-CREPE agreement for one recording.

    Attributes:
        file_id: Recording identifier.
        text_id: Library text.
        tier: ``gold`` | ``silver`` | ``bronze``.
        origin: ``clean`` | ``injected`` | ``human``.
        speaker_id: Speaker identity (a derived file inherits its parent's).
        duration_s: Audio actually analysed, after any ``--seconds`` limit.
        n_frames: Frames on the project grid.
        n_praat_voiced: Frames Praat voiced.
        n_crepe_voiced: Frames CREPE kept after its periodicity gate.
        n_common_voiced: Frames both voiced - the comparison population.
        agreement_pct: Percentage of those within 50 cents.
        median_abs_cents: Median absolute difference over those frames.
        p95_abs_cents: 95th percentile of the same.
        bias_cents: Signed mean (Praat minus CREPE); a systematic offset would
            show here rather than in the absolute statistics.
        octave_slip_pct: Percentage within 50 cents of exactly one octave.
        praat_floor_hz: Pass-2 floor Praat settled on.
        praat_ceiling_hz: Pass-2 ceiling Praat settled on.
    """

    file_id: str
    text_id: str
    tier: str
    origin: str
    speaker_id: str
    duration_s: float
    n_frames: int
    n_praat_voiced: int
    n_crepe_voiced: int
    n_common_voiced: int
    agreement_pct: float
    median_abs_cents: float
    p95_abs_cents: float
    bias_cents: float
    octave_slip_pct: float
    praat_floor_hz: float
    praat_ceiling_hz: float


# --------------------------------------------------------------------------- #
# Measuring one file
# --------------------------------------------------------------------------- #


def compare_one(
    samples: np.ndarray,
    sr: int,
    *,
    cfg: Config,
    grid: FrameGrid,
    model: str,
) -> tuple[np.ndarray, P.PitchTrack, int]:
    """Run both trackers on one recording.

    Returns:
        ``(cents, praat_track, n_crepe_voiced)`` where ``cents`` holds the
        signed Praat-minus-CREPE difference for commonly voiced frames only.
    """
    conf = cfg.features["pitch"]
    praat = P.pitch_track(samples, sr, grid=grid, config=cfg, crosscheck=False)
    crepe_hz, _ = P.crepe_pitch_hz(
        samples,
        sr,
        grid=grid,
        floor_hz=float(conf["pass1"]["floor_hz"]),
        ceiling_hz=float(conf["pass1"]["ceiling_hz"]),
        model=model,
        periodicity_threshold=float(conf["crosscheck"].get("periodicity_threshold", 0.5)),
        n_frames=praat.n_frames,
    )
    cents = P.cents_difference(praat.f0_hz, crepe_hz)
    return cents[np.isfinite(cents)], praat, int(np.isfinite(crepe_hz).sum())


def summarise(cents: np.ndarray, *, agreement_cents: float) -> dict[str, float]:
    """Agreement statistics for one population of cent differences."""
    absolute = np.abs(cents)
    if absolute.size == 0:
        return {
            "agreement_pct": float("nan"),
            "median_abs_cents": float("nan"),
            "p95_abs_cents": float("nan"),
            "bias_cents": float("nan"),
            "octave_slip_pct": float("nan"),
        }
    near_octave = np.abs(absolute - OCTAVE_CENTS) <= agreement_cents
    return {
        "agreement_pct": float((absolute <= agreement_cents).mean() * 100.0),
        "median_abs_cents": float(np.median(absolute)),
        "p95_abs_cents": float(np.percentile(absolute, 95)),
        "bias_cents": float(cents.mean()),
        "octave_slip_pct": float(near_octave.mean() * 100.0),
    }


# --------------------------------------------------------------------------- #
# Walking the dataset
# --------------------------------------------------------------------------- #


@dataclass
class Recording:
    """One audio file to check, with the manifest information attached."""

    file_id: str
    path: Path
    text_id: str = ""
    tier: str = ""
    origin: str = ""
    speaker_id: str = ""


def collect(
    cfg: Config, *, sample: bool, origins: list[str], only: list[str] | None
) -> list[Recording]:
    """List the recordings to check, in a stable order."""
    if sample:
        directories = [(cfg.path("sample.audio"), "clean")]
        manifest = cfg.path("sample.dir") / "MANIFEST.csv"
    else:
        directories = [
            (cfg.path("data.audio_clean"), "clean"),
            (cfg.path("data.audio_injected"), "injected"),
            (cfg.path("data.audio_human_flawed"), "human"),
        ]
        manifest = cfg.path("data.manifest")

    rows = {r.file_id: r for r in read_manifest(manifest)} if manifest.is_file() else {}
    clean_ids = clean_file_ids(rows.values())

    out: list[Recording] = []
    seen: set[str] = set()
    for directory, default_origin in directories:
        for wav in sorted(directory.glob("*.wav")):
            file_id = wav.stem
            if file_id in seen or (only and file_id not in only):
                continue
            row = rows.get(file_id)
            origin = (row.origin if row else default_origin) or default_origin
            if origin not in origins:
                continue
            seen.add(file_id)
            speaker = (
                file_id if origin == "clean" else (parent_file_id(file_id, clean_ids) or file_id)
            )
            out.append(
                Recording(
                    file_id=file_id,
                    path=wav,
                    text_id=row.text_id if row else "",
                    tier=row.tier if row else "",
                    origin=origin,
                    speaker_id=speaker,
                )
            )
    return out


def _group(
    rows: list[FileAgreement], pooled: dict[str, np.ndarray], key: str, *, agreement_cents: float
) -> dict[str, dict[str, float]]:
    """Pool the frame-level differences by one row attribute and summarise."""
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        name = getattr(row, key) or "unknown"
        buckets[name].append(pooled[row.file_id])
    return {
        name: {
            "n_files": sum(1 for r in rows if (getattr(r, key) or "unknown") == name),
            **summarise(
                np.concatenate(values) if values else np.empty(0), agreement_cents=agreement_cents
            ),
        }
        for name, values in sorted(buckets.items())
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_histogram(cfg: Config, cents: np.ndarray, *, agreement_cents: float) -> Path:
    """Pooled histogram of Praat-minus-CREPE differences, with the threshold marked."""
    figure, axes = C.new_figure(size=(7.0, 4.0))
    clipped = np.clip(cents, -HIST_CLIP_CENTS, HIST_CLIP_CENTS)
    axes.hist(clipped, bins=121, color="#4878a8", edgecolor="none")
    for sign in (-1.0, 1.0):
        axes.axvline(
            sign * agreement_cents,
            color="#b4422a",
            linestyle="--",
            linewidth=1.0,
            label=f"+/-{agreement_cents:.0f} cents" if sign > 0 else None,
        )
    inside = (
        float((np.abs(cents) <= agreement_cents).mean() * 100.0) if cents.size else float("nan")
    )
    axes.set_xlabel("Praat - CREPE (cents, clipped at +/-300)")
    axes.set_ylabel("commonly voiced frames")
    axes.set_title(
        f"Pitch cross-check: {inside:.1f}% of frames agree within {agreement_cents:.0f} cents"
    )
    axes.legend(loc="upper right", frameon=False)
    return C.save_figure(figure, C.figure_path(cfg, "components_pitch_cents_hist.png"))


def plot_by_tier(cfg: Config, by_tier: dict[str, dict[str, float]]) -> Path:
    """Agreement percentage per tier."""
    figure, axes = C.new_figure(size=(5.5, 3.5))
    names = list(by_tier)
    values = [by_tier[n]["agreement_pct"] for n in names]
    axes.bar(names, values, color="#4878a8")
    for x, value in enumerate(values):
        if np.isfinite(value):
            axes.text(x, value + 1.0, f"{value:.1f}%", ha="center", fontsize=9)
    axes.set_ylim(0, 105)
    axes.set_ylabel("frames agreeing within 50 cents (%)")
    axes.set_title("Praat-CREPE agreement by tier")
    return C.save_figure(figure, C.figure_path(cfg, "components_pitch_agreement_by_tier.png"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Measure pitch agreement and write the ``pitch`` section."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="use dataset_sample/ instead")
    parser.add_argument("--file-id", action="append", dest="file_ids")
    parser.add_argument(
        "--origin",
        action="append",
        dest="origins",
        choices=["clean", "injected", "human"],
        help="which origins to include (default: clean only)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="analyse only the first N seconds of each file (CREPE is slow on CPU)",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--model", default=None, choices=["full", "tiny"])
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    grid = FrameGrid.from_config(cfg)
    conf = cfg.features["pitch"]
    agreement_cents = float(conf["crosscheck"]["agreement_cents"])
    model = args.model or str(conf["crosscheck"].get("model", "full"))

    recordings = collect(
        cfg,
        sample=args.sample,
        origins=args.origins or ["clean"],
        only=args.file_ids,
    )
    if args.max_files:
        recordings = recordings[: args.max_files]

    if not recordings:
        C.write_section(
            cfg,
            "pitch",
            {
                "status": C.STATUS_PENDING,
                "reason": "no audio found; run `make data` first",
                "agreement_cents": agreement_cents,
            },
        )
        logger.error("no audio to check")
        return 2

    rows: list[FileAgreement] = []
    pooled: dict[str, np.ndarray] = {}
    for recording in recordings:
        samples, sr = load_any(recording.path)
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        if args.seconds:
            samples = samples[: int(args.seconds * sr)]
        cents, praat, n_crepe = compare_one(samples, sr, cfg=cfg, grid=grid, model=model)
        pooled[recording.file_id] = cents
        stats = summarise(cents, agreement_cents=agreement_cents)
        rows.append(
            FileAgreement(
                file_id=recording.file_id,
                text_id=recording.text_id,
                tier=recording.tier,
                origin=recording.origin,
                speaker_id=recording.speaker_id,
                duration_s=samples.size / float(sr),
                n_frames=praat.n_frames,
                n_praat_voiced=int(praat.voiced.sum()),
                n_crepe_voiced=n_crepe,
                n_common_voiced=int(cents.size),
                praat_floor_hz=praat.floor_hz,
                praat_ceiling_hz=praat.ceiling_hz,
                **stats,
            )
        )
        logger.info(
            "%s: %d common voiced frames, %.1f%% within %.0f cents (median %.1f)",
            recording.file_id,
            cents.size,
            stats["agreement_pct"],
            agreement_cents,
            stats["median_abs_cents"],
        )

    everything = np.concatenate([v for v in pooled.values() if v.size]) if pooled else np.empty(0)
    overall = summarise(everything, agreement_cents=agreement_cents)
    by_tier = _group(rows, pooled, "tier", agreement_cents=agreement_cents)
    by_speaker = _group(rows, pooled, "speaker_id", agreement_cents=agreement_cents)

    figures: list[str] = []
    if not args.no_figures and everything.size:
        figures.append(plot_histogram(cfg, everything, agreement_cents=agreement_cents).name)
        figures.append(plot_by_tier(cfg, by_tier).name)

    C.write_section(
        cfg,
        "pitch",
        {
            "status": C.STATUS_OK,
            "metric": "percentage of commonly voiced frames where Praat and CREPE "
            "agree within 50 cents",
            "agreement_cents": agreement_cents,
            "crepe_model": model,
            "seconds_per_file": args.seconds,
            "n_files": len(rows),
            "n_frames_compared": int(everything.size),
            "overall": overall,
            "by_tier": by_tier,
            "by_speaker": by_speaker,
            "per_file": [asdict(r) for r in rows],
            "figures": figures,
        },
    )

    print()
    print(f"{'file_id':<18}{'tier':<8}{'frames':>9}{'agree %':>10}{'median c':>10}{'P95 c':>9}")
    for row in rows:
        print(
            f"{row.file_id:<18}{row.tier:<8}{row.n_common_voiced:>9}"
            f"{row.agreement_pct:>10.1f}{row.median_abs_cents:>10.1f}{row.p95_abs_cents:>9.1f}"
        )
    print(
        f"{'ALL':<18}{'':<8}{everything.size:>9}{overall['agreement_pct']:>10.1f}"
        f"{overall['median_abs_cents']:>10.1f}{overall['p95_abs_cents']:>9.1f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
