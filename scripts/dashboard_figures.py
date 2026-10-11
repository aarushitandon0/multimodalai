"""Render the dashboard's own figures to ``docs/figures`` (prompt 22).

These are not browser screenshots. They are the *same* figure objects the
Streamlit app renders, exported straight from Plotly and matplotlib, which is
the more useful artefact for a report: it cannot drift from the dashboard, it
is reproducible, and it has no browser chrome in it. The distinction is stated
here rather than left for a reader to discover.

What is written:

======================================  ===========================================
``dashboard_overlay.png``               The three-panel overlay with flaw regions.
``dashboard_radar.png``                 The six-category rubric with its band.
``dashboard_waveform.png``              The recording with its regions marked.
``dashboard_transcript.png``            The transcript tinted by D_w.
``dashboard_report.pdf``                The PDF export, in full.
======================================  ===========================================

Usage::

    python -m scripts.dashboard_figures --out docs/figures
    python -m scripts.dashboard_figures --from results/examples/reference.json

With no ``--from``, a reference-mode analysis is generated the way the tests do
it: five synthetic skilled readers and one participant who rushes a clause.
That is honest demo material, and the caption written beside it says so - it
exercises every stage of the pipeline but no number in it is a dataset result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("scripts.dashboard_figures")

__all__ = ["main", "render"]

OVERLAY_SIZE = (1280, 780)
RADAR_SIZE = (620, 440)
WAVEFORM_SIZE = (1280, 230)


def _demo_result(work: Path) -> Any:
    """Generate a reference-mode analysis with one rushed clause."""
    from eval.synthetic_audio import DEMO_TEXT, demo_library, make_utterance, parse_text

    from rostrum.config import load_config
    from rostrum.pipeline import analyze

    config = load_config()
    library, _ = demo_library(config, work / "library", n_readers=5)
    spoken = parse_text(DEMO_TEXT)
    rate = [1.0] * len(spoken.words)
    rate[4:8] = [1.9] * 4
    pause = [1.0] * len(spoken.words)
    pause[7] = 0.05
    utterance = make_utterance(
        "demo_rushing", work, text=spoken, seed=321, rate_scale=rate, pause_scale=pause
    )
    result = analyze(
        utterance.path,
        DEMO_TEXT,
        "demo_anaphora",
        config=config,
        library=library,
        alignment=utterance.alignment,
        check_text_match=False,
        cache=False,
        work_dir=work / "analysis",
    )
    return result, utterance.path


def _transcript_png(view: Any, path: Path) -> Path | None:
    """Draw the tinted transcript with matplotlib, since it is HTML on screen."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from rostrum.dashboard.palette import category_colour, text_on
    from rostrum.dashboard.transcript import transcript_words

    words = transcript_words(view)
    if not words:
        return None

    per_row, pad = 9, 0.012
    rows = (len(words) + per_row - 1) // per_row
    figure, axes = plt.subplots(figsize=(12.8, 0.55 * rows + 0.8))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, rows + 0.4)
    axes.axis("off")

    for i, word in enumerate(words):
        row, col = divmod(i, per_row)
        x = pad + col * (1.0 - 2 * pad) / per_row
        y = rows - row - 0.6
        width = (1.0 - 2 * pad) / per_row - 0.006
        axes.add_patch(
            Rectangle(
                (x, y),
                width,
                0.42,
                facecolor=word.colour,
                edgecolor=category_colour(word.category) if word.in_region else "#DDDDDD",
                linewidth=2.2 if word.in_region else 0.6,
            )
        )
        label = word.text if len(word.text) <= 13 else word.text[:12] + "."
        axes.text(
            x + width / 2,
            y + 0.21,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=text_on(word.colour),
        )
        axes.text(
            x + width / 2,
            y - 0.10,
            "-" if word.deviation is None else f"{word.deviation:.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="#666666",
        )

    axes.set_title(
        "Transcript tinted by per-word deviation D (number under each word); "
        "a coloured outline marks a flagged region",
        fontsize=11,
        loc="left",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def render(out_dir: Path, source: Path | None = None) -> dict[str, Path]:
    """Render every dashboard figure and return ``{name: path}``."""
    import tempfile

    from rostrum.config import load_config
    from rostrum.dashboard.figures import overlay_figure, radar_figure, waveform_figure
    from rostrum.dashboard.report import write_pdf
    from rostrum.dashboard.view import AnalysisView
    from rostrum.io.audio import read_mono
    from rostrum.types import AnalysisResult

    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    written: dict[str, Path] = {}

    with tempfile.TemporaryDirectory(prefix="rostrum_figs_") as tmp:
        if source is not None:
            result = AnalysisResult.from_dict(json.loads(source.read_text(encoding="utf-8")))
            audio = None
            logger.info("rendering from %s", source)
        else:
            result, audio = _demo_result(Path(tmp))
            logger.info("rendering from a generated reference-mode analysis")

        view = AnalysisView(result=result, config=config)

        overlay = out_dir / "dashboard_overlay.png"
        overlay_figure(
            view, highlight=view.regions[0].region_id if view.regions else None
        ).write_image(overlay, width=OVERLAY_SIZE[0], height=OVERLAY_SIZE[1])
        written["overlay"] = overlay

        radar = out_dir / "dashboard_radar.png"
        radar_figure(view).write_image(radar, width=RADAR_SIZE[0], height=RADAR_SIZE[1])
        written["radar"] = radar

        if audio is not None:
            samples, rate = read_mono(audio, sr=config.sample_rate)
            wave = out_dir / "dashboard_waveform.png"
            waveform_figure(view, samples, rate).write_image(
                wave, width=WAVEFORM_SIZE[0], height=WAVEFORM_SIZE[1]
            )
            written["waveform"] = wave

        transcript = _transcript_png(view, out_dir / "dashboard_transcript.png")
        if transcript is not None:
            written["transcript"] = transcript

        written["report"] = write_pdf(view, out_dir / "dashboard_report.pdf")

        (out_dir / "dashboard_figures.json").write_text(
            json.dumps(
                {
                    "source": str(source) if source else "generated reference-mode analysis",
                    "note": (
                        "Rendered from the dashboard's own figure builders, not captured "
                        "from a browser. With no --from, the readings are generated "
                        "signals that exercise every stage of the pipeline; no number in "
                        "them is a dataset result."
                    ),
                    "mode": result.mode,
                    "analysis_id": result.analysis_id,
                    "config_hash": result.config_hash,
                    "n_references": result.n_references,
                    "regions": len(result.regions),
                    "files": {k: v.name for k, v in written.items()},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for name, path in written.items():
        print(f"{name:<12} {path}")
    return written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--from",
        dest="source",
        type=Path,
        default=None,
        help="An AnalysisResult JSON to render instead of generating one.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    render(args.out, args.source)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
