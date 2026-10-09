"""Acceptance run for the pipeline: one analysis per operating mode (spec 13).

Writes one :class:`~rostrum.types.AnalysisResult` JSON per mode to
``results/examples/`` and prints what each one decided.

What is real and what is not, stated plainly because the whole project is
judged on that distinction:

* **Reject** is a real recording with a real recogniser: a LibriVox reading of
  the Gettysburg Address from ``dataset_sample/`` is offered with the wrong
  transcript, faster-whisper hears a different text, and the pipeline refuses
  to analyse it. Nothing is simulated.
* **Reference** and **Norm** use *generated* audio from
  :mod:`eval.synthetic_audio`: a tone-burst signal per word with a known
  alignment. They exercise every stage of spec 13 - preprocessing, gap labels,
  Praat features, the envelope, hysteresis, ruptures refinement, the rubric,
  the confidence band, the cards and the DTW-warped overlay curves - but they
  are not speech, and no number in them is a dataset result.

Running these two modes on the dataset instead needs a forced alignment, and
forced alignment is **MFA only** (CLAUDE.md invariant 4), which needs the
``mfa`` conda environment of ``environment.mfa.yml``. ``--dataset`` attempts
exactly that and reports what is missing when it cannot.

Usage::

    python -m scripts.analyze_examples
    python -m scripts.analyze_examples --dataset        # needs conda + MFA
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.synthetic_audio import DEMO_TEXT, OTHER_TEXT, demo_library, make_utterance, parse_text

from rostrum.align import mfa as M
from rostrum.config import Config, load_config
from rostrum.pipeline import AnalysisError, ReferenceLibrary, analyze, load_library
from rostrum.types import AnalysisResult

logger = logging.getLogger("scripts.analyze_examples")

__all__ = ["main", "run"]

_SAMPLE_AUDIO = "s1_lv01.wav"
_SAMPLE_TEXT_ID = "s1_gettysburg"


def _summary(result: AnalysisResult) -> str:
    """One line per analysis, for the terminal."""
    score = (
        f"{result.score.overall:.2f} +/- {result.score.overall_band:.2f}" if result.score else "n/a"
    )
    return (
        f"{result.file_id:<22} mode={result.mode:<9} "
        f"regions={len(result.regions):<3} score={score:<16} "
        f"n_ref={result.n_references:<3} wer={'n/a' if result.wer is None else f'{result.wer:.3f}'}"
    )


def _write(result: AnalysisResult, out_dir: Path, name: str) -> Path:
    """Write one result as JSON and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return path


def _print_result(result: AnalysisResult, path: Path) -> None:
    """The decision, the regions and the first card."""
    print(f"\n{'=' * 78}")
    print(_summary(result))
    print(f"  why: {result.mode_reason}")
    print(f"  wrote {path}")
    for region in result.regions[:4]:
        print(
            f"  [{region.t_start:6.2f}-{region.t_end:6.2f}] {region.category:<10} "
            f"{region.label:<18} {region.level} peak z={region.peak_z:+.1f}"
        )
    if result.cards:
        card = result.cards[0]
        print(f"  card: {card['rationale']}")
        print(f"  fix:  {card['fix']}")
    if result.curves:
        print(
            f"  curves: {len(result.curves.get('words', {}))} word(s), "
            f"dtw={result.curves.get('dtw')}, "
            f"band over {len(result.envelope_curves.get('words', {}))} word(s)"
        )
    for warning in result.warnings:
        print(f"  ! {warning}")


def run(
    *,
    config: Config,
    work: Path,
    out_dir: Path,
    n_readers: int = 5,
    with_dataset: bool = False,
) -> dict[str, Any]:
    """Analyse one example per mode and write the JSON."""
    written: dict[str, str] = {}
    results: list[AnalysisResult] = []

    print("building the generated reference library ...")
    library, _ = demo_library(config, work / "library", n_readers=n_readers)
    envelope = next(iter(library.envelopes.values()))
    print(
        f"  envelope n={envelope.n_references}, units {sorted(envelope.units)}, "
        f"pooled n={library.pooled.n_references if library.pooled else 0}"
    )

    # -- Reference mode ----------------------------------------------------
    spoken = parse_text(DEMO_TEXT)
    n = len(spoken.words)
    rate_scale = [1.0] * n
    rate_scale[4:8] = [1.9] * 4  # a rushed clause
    pause_scale = [1.0] * n
    pause_scale[7] = 0.05  # with its boundary pause cut
    rushed = make_utterance(
        "example_reference",
        work,
        text=spoken,
        seed=77,
        rate_scale=rate_scale,
        pause_scale=pause_scale,
    )
    reference = analyze(
        rushed.path,
        DEMO_TEXT,
        config=config,
        library=library,
        alignment=rushed.alignment,
        check_text_match=False,
        cache=False,
    )
    results.append(reference)
    written["reference"] = str(_write(reference, out_dir, "reference"))
    _print_result(reference, Path(written["reference"]))

    # -- Norm mode ---------------------------------------------------------
    other = parse_text(OTHER_TEXT)
    m = len(other.words)
    slow = [1.0] * m
    slow[5:10] = [0.45] * 5  # a dragging clause
    fading = [0.0] * m
    fading[12:] = [-9.0] * (m - 12)  # and a fade at the end
    unknown = make_utterance(
        "example_norm", work, text=other, seed=42, rate_scale=slow, level_scale=fading
    )
    norm = analyze(
        unknown.path,
        OTHER_TEXT,
        config=config,
        library=library,
        alignment=unknown.alignment,
        check_text_match=False,
        cache=False,
    )
    results.append(norm)
    written["norm"] = str(_write(norm, out_dir, "norm"))
    _print_result(norm, Path(written["norm"]))

    # -- Reject mode: a real recording with the wrong transcript -----------
    sample = config.path("sample.audio") / _SAMPLE_AUDIO
    if sample.is_file():
        reject = analyze(
            sample,
            DEMO_TEXT,
            config=config,
            # An empty library: in Reject mode there is nothing to compare
            # against, and the demo library's caveats do not apply.
            library=ReferenceLibrary(fold="reject"),
            check_text_match=True,
            cache=False,
        )
        results.append(reject)
        written["reject"] = str(_write(reject, out_dir, "reject"))
        _print_result(reject, Path(written["reject"]))
    else:  # pragma: no cover - the sample is committed
        print(f"\n! {sample} is missing; the Reject example needs it")

    # -- The same two modes on the dataset, which needs MFA ----------------
    dataset_note = ""
    if with_dataset:
        dataset_note = _try_dataset(config, out_dir, written, results)
    else:
        dataset_note = (
            "Reference and Norm mode on dataset_sample/ need a forced alignment; "
            "run with --dataset once the mfa conda environment exists."
        )
        print(f"\n[{dataset_note}]")

    print(f"\n{'=' * 78}\nsummary")
    for result in results:
        print("  " + _summary(result))

    index = {
        "source": "generated audio for reference/norm, dataset_sample for reject",
        "config_hash": config.hash,
        "dataset_note": dataset_note,
        "examples": {
            name: {
                "path": path,
                "mode": result.mode,
                "analysis_id": result.analysis_id,
            }
            for (name, path), result in zip(written.items(), results, strict=False)
        },
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'index.json'}")
    return index


def _try_dataset(
    config: Config,
    out_dir: Path,
    written: dict[str, str],
    results: list[AnalysisResult],
) -> str:
    """Attempt the dataset run, reporting the blocker when there is one."""
    audio = config.path("sample.audio") / _SAMPLE_AUDIO
    texts = load_library(config).texts
    text = texts.get(_SAMPLE_TEXT_ID)
    if text is None or not text.transcript or text.transcript.startswith("TODO"):
        return f"{_SAMPLE_TEXT_ID} has no transcript on disk yet"
    try:
        M.mfa_version(env_name=str(config.paths["mfa"]["env_name"]))
    except M.MFAError as exc:
        return f"MFA is not callable here ({exc}); see environment.mfa.yml"
    try:
        result = analyze(
            audio,
            text.transcript,
            _SAMPLE_TEXT_ID,
            config=config,
            check_text_match=True,
            cache=True,
        )
    except (AnalysisError, M.MFAError) as exc:
        return f"the dataset run failed: {exc}"
    results.append(result)
    written["dataset"] = str(_write(result, out_dir, "dataset_s1_lv01"))
    _print_result(result, Path(written["dataset"]))
    return "ran on dataset_sample/audio/s1_lv01.wav with MFA"


def main(argv: Sequence[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readers", type=int, default=5, help="generated readers in the envelope")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--work", type=Path, default=None, help="where to keep the generated WAVs")
    parser.add_argument(
        "--dataset", action="store_true", help="also analyse a dataset_sample file with MFA"
    )
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    config = load_config()
    out_dir = args.out or config.path("results.examples")
    work = args.work or (config.path("cache.dir") / "examples")
    work.mkdir(parents=True, exist_ok=True)
    run(
        config=config,
        work=work,
        out_dir=out_dir,
        n_readers=args.readers,
        with_dataset=args.dataset,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
