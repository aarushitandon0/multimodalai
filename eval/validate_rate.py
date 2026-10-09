"""Speech-rate cross-check: MFA syllables vs de Jong & Wempe (spec 14.1).

The rate feature counts **vowel phones from the forced alignment** per second of
articulation. That is only as good as the aligner's phone tier, so it is checked
against a completely independent estimate: the de Jong & Wempe syllable-nuclei
detector [R14], which finds intensity peaks that are separated by a dip of at
least 2 dB and that fall at a voiced moment, and never looks at the transcript
at all.

Three correlations are reported, from the most local to the most global:

* **per word** - the MFA vowel count in each word against the number of nuclei
  that fall inside it;
* **local window** - the ``+/-3``-word local rate the feature actually uses
  (``configs/features.yaml``: ``rate.local_window_words``) against the nuclei
  rate over the same span. This is the number the report should quote, because
  it is the quantity the detector compares;
* **per file** - each recording's global articulation rate against its global
  nuclei rate.

Agreement is expected to be good but not perfect, and the ratio matters as much
as the correlation: the detector misses a syllable in a cluster of unstressed
ones and finds one in a long diphthong, so a ratio near 1.0 with a high
correlation is the result that supports using the MFA count.

Outputs
    ``results/components.json`` section ``rate``;
    ``docs/figures/components_rate_local_scatter.png``;
    ``docs/figures/components_rate_per_file.png``.

Exit codes:
    0  the section was written with measured numbers
    2  no alignment exists yet; the section is written as pending
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import components as C
from rostrum.align.mfa import read_textgrid
from rostrum.config import Config, load_config
from rostrum.data.manifest import read_manifest
from rostrum.features import rate as R
from rostrum.features.frames import FrameGrid
from rostrum.io.audio import load_any
from rostrum.types import Word

logger = logging.getLogger("validate_rate")

MIN_POINTS = 3
"""Fewest paired points a correlation is computed on."""


@dataclass
class FileRates:
    """Global rate agreement for one recording.

    Attributes:
        file_id: Recording identifier.
        tier: Sourcing tier.
        duration_s: Audio duration.
        speech_duration_s: First word onset to last word offset.
        n_mfa_syllables: Vowel phones in the alignment.
        n_nuclei: Nuclei the detector accepted.
        ratio: ``n_nuclei / n_mfa_syllables``.
        mfa_rate: MFA articulation rate, syllables per second.
        nuclei_rate: Nuclei per second of speech.
        per_word_correlation: Pearson r of the per-word counts, or ``None``.
    """

    file_id: str
    tier: str
    duration_s: float
    speech_duration_s: float
    n_mfa_syllables: int
    n_nuclei: int
    ratio: float | None
    mfa_rate: float
    nuclei_rate: float
    per_word_correlation: float | None


def correlate(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Pearson and Spearman correlation of the pairs where both are finite."""
    from scipy import stats

    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < MIN_POINTS or np.std(a) == 0 or np.std(b) == 0:
        return {"n": int(a.size), "pearson_r": float("nan"), "spearman_rho": float("nan")}
    pearson = stats.pearsonr(a, b)
    spearman = stats.spearmanr(a, b)
    return {
        "n": int(a.size),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "mean_absolute_difference": float(np.abs(a - b).mean()),
    }


def local_nuclei_rate(
    words: list[Word], idx: int, nuclei: np.ndarray, *, window: int, exclude_s: float
) -> float:
    """Nuclei per second over the same ``+/-window``-word span the feature uses."""
    lo = max(0, idx - window)
    hi = min(len(words), idx + window + 1)
    span = words[lo:hi]
    if not span:
        return float("nan")
    t0, t1 = float(span[0].t_start), float(span[-1].t_end)
    seconds = R.articulation_time(span, exclude_pauses_over_s=exclude_s)
    if seconds <= 0:
        return float("nan")
    count = int(np.sum((nuclei >= t0) & (nuclei < t1)))
    return count / seconds


# --------------------------------------------------------------------------- #
# Walking the dataset
# --------------------------------------------------------------------------- #


def collect(
    cfg: Config, *, sample: bool, only: list[str] | None
) -> list[tuple[str, Path, Path, str]]:
    """``(file_id, audio, alignment, tier)`` for every clean recording."""
    audio_dir = cfg.path("sample.audio") if sample else cfg.path("data.audio_clean")
    alignment_dir = cfg.path("sample.alignments") if sample else cfg.path("data.alignments")
    manifest = cfg.path("sample.dir") / "MANIFEST.csv" if sample else cfg.path("data.manifest")
    rows = {r.file_id: r for r in read_manifest(manifest)} if manifest.is_file() else {}
    out = []
    for wav in sorted(audio_dir.glob("*.wav")):
        if only and wav.stem not in only:
            continue
        row = rows.get(wav.stem)
        out.append((wav.stem, wav, alignment_dir / f"{wav.stem}.TextGrid", row.tier if row else ""))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def plot_local_scatter(cfg: Config, mfa: np.ndarray, nuclei: np.ndarray, stats: dict) -> Path:
    """Local MFA rate against local nuclei rate, with the identity line."""
    figure, axes = C.new_figure(size=(5.5, 5.0))
    axes.scatter(mfa, nuclei, s=8, alpha=0.35, color="#4878a8", edgecolors="none")
    finite = np.concatenate([mfa[np.isfinite(mfa)], nuclei[np.isfinite(nuclei)]])
    top = float(finite.max()) * 1.05 if finite.size else 1.0
    axes.plot([0, top], [0, top], color="#444444", linewidth=0.8, linestyle="--", label="y = x")
    axes.set_xlim(0, top)
    axes.set_ylim(0, top)
    axes.set_xlabel("MFA vowel-phone rate (syll/s, +/-3-word window)")
    axes.set_ylabel("de Jong & Wempe nuclei rate (syll/s)")
    axes.set_title(
        f"Local rate cross-check: r = {stats.get('pearson_r', float('nan')):.3f} "
        f"(n = {stats.get('n', 0)})"
    )
    axes.legend(frameon=False, loc="lower right")
    return C.save_figure(figure, C.figure_path(cfg, "components_rate_local_scatter.png"))


def plot_per_file(cfg: Config, rows: list[FileRates], stats: dict) -> Path:
    """Global MFA rate against global nuclei rate, one point per recording."""
    figure, axes = C.new_figure(size=(5.5, 5.0))
    x = np.array([r.mfa_rate for r in rows], dtype=np.float64)
    y = np.array([r.nuclei_rate for r in rows], dtype=np.float64)
    axes.scatter(x, y, s=28, color="#d08c38", edgecolors="none")
    for row in rows:
        axes.annotate(
            row.file_id,
            (row.mfa_rate, row.nuclei_rate),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    top = float(finite.max()) * 1.1 if finite.size else 1.0
    axes.plot([0, top], [0, top], color="#444444", linewidth=0.8, linestyle="--", label="y = x")
    axes.set_xlim(0, top)
    axes.set_ylim(0, top)
    axes.set_xlabel("MFA articulation rate (syll/s)")
    axes.set_ylabel("nuclei rate (syll/s)")
    axes.set_title(
        f"Per-file rate cross-check: r = {stats.get('pearson_r', float('nan')):.3f} "
        f"(n = {stats.get('n', 0)})"
    )
    axes.legend(frameon=False, loc="lower right")
    return C.save_figure(figure, C.figure_path(cfg, "components_rate_per_file.png"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Measure rate agreement and write the ``rate`` section."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", action="store_true", help="use dataset_sample/ instead")
    parser.add_argument("--file-id", action="append", dest="file_ids")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config()
    grid = FrameGrid.from_config(cfg)
    window = int(cfg.features["rate"]["local_window_words"])
    exclude_s = float(cfg.features["rate"]["exclude_pauses_over_s"])

    targets = collect(cfg, sample=args.sample, only=args.file_ids)
    aligned = [t for t in targets if t[2].is_file()]
    if args.max_files:
        aligned = aligned[: args.max_files]

    if not aligned:
        reason = (
            "no audio found; run `make data` first"
            if not targets
            else "no alignment found for any recording; run `make align` "
            "(MFA lives in the `mfa` conda env)"
        )
        C.write_section(cfg, "rate", {"status": C.STATUS_PENDING, "reason": reason})
        logger.error("%s", reason)
        return 2

    rows: list[FileRates] = []
    word_mfa: list[float] = []
    word_nuclei: list[float] = []
    local_mfa: list[float] = []
    local_nuclei: list[float] = []

    for file_id, audio_path, alignment_path, tier in aligned:
        samples, sr = load_any(audio_path)
        samples = np.asarray(samples, dtype=np.float64).reshape(-1)
        duration = samples.size / float(sr)
        alignment = read_textgrid(alignment_path, file_id=file_id, duration=duration)
        words = list(alignment.words)
        if not words:
            logger.warning("%s has no aligned words; skipped", file_id)
            continue

        nuclei = R.syllable_nuclei(samples, sr, grid=grid)
        report = R.nuclei_crosscheck(words, nuclei)
        times = np.asarray(sorted(nuclei), dtype=np.float64)

        for idx, _word in enumerate(words):
            local = R.local_rate(words, idx, window_words=window, exclude_pauses_over_s=exclude_s)
            local_mfa.append(local)
            local_nuclei.append(
                local_nuclei_rate(words, idx, times, window=window, exclude_s=exclude_s)
            )
        word_mfa += [float(m) for m, _ in report.per_word]
        word_nuclei += [float(n) for _, n in report.per_word]

        rows.append(
            FileRates(
                file_id=file_id,
                tier=tier,
                duration_s=duration,
                speech_duration_s=alignment.speech_duration,
                n_mfa_syllables=report.n_mfa_syllables,
                n_nuclei=report.n_nuclei,
                ratio=report.ratio,
                mfa_rate=R.articulation_rate(words, exclude_pauses_over_s=exclude_s),
                nuclei_rate=R.nuclei_rate(nuclei, speech_duration_s=alignment.speech_duration),
                per_word_correlation=report.correlation,
            )
        )
        logger.info(
            "%s: %d MFA syllables vs %d nuclei (ratio %.3f), per-word r = %s",
            file_id,
            report.n_mfa_syllables,
            report.n_nuclei,
            report.ratio if report.ratio is not None else float("nan"),
            f"{report.correlation:.3f}" if report.correlation is not None else "n/a",
        )

    per_word_stats = correlate(np.array(word_mfa), np.array(word_nuclei))
    local_stats = correlate(np.array(local_mfa), np.array(local_nuclei))
    per_file_stats = correlate(
        np.array([r.mfa_rate for r in rows]), np.array([r.nuclei_rate for r in rows])
    )
    ratios = [r.ratio for r in rows if r.ratio is not None]

    figures: list[str] = []
    if not args.no_figures:
        figures.append(
            plot_local_scatter(cfg, np.array(local_mfa), np.array(local_nuclei), local_stats).name
        )
        if len(rows) >= 2:
            figures.append(plot_per_file(cfg, rows, per_file_stats).name)

    C.write_section(
        cfg,
        "rate",
        {
            "status": C.STATUS_OK,
            "metric": "correlation between the MFA vowel-phone syllable rate and the "
            "de Jong & Wempe nuclei rate",
            "local_window_words": window,
            "exclude_pauses_over_s": exclude_s,
            "n_files": len(rows),
            "local_window": local_stats,
            "per_word": per_word_stats,
            "per_file": per_file_stats,
            "nuclei_to_mfa_ratio": C.describe(ratios),
            "files": [asdict(r) for r in rows],
            "figures": figures,
        },
    )

    print()
    print(f"{'file_id':<18}{'MFA syll':>10}{'nuclei':>8}{'ratio':>8}{'MFA r/s':>9}{'nuc r/s':>9}")
    for row in rows:
        print(
            f"{row.file_id:<18}{row.n_mfa_syllables:>10}{row.n_nuclei:>8}"
            f"{(row.ratio if row.ratio is not None else float('nan')):>8.3f}"
            f"{row.mfa_rate:>9.2f}{row.nuclei_rate:>9.2f}"
        )
    print()
    for name, stats in (
        ("local +/-3-word window", local_stats),
        ("per word", per_word_stats),
        ("per file", per_file_stats),
    ):
        print(
            f"{name:<24} n = {stats['n']:>6}  "
            f"pearson r = {stats['pearson_r']:.3f}  "
            f"spearman rho = {stats['spearman_rho']:.3f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
