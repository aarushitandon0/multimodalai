#!/usr/bin/env python
"""Prove forced alignment works end to end on one real recording.

This is the day-1 gate of the project (spec section 8): before any feature,
envelope or score is written, MFA must align a file we supply and return
monotonic word timestamps.

The script does **not** synthesise audio. Drop one recording and its verbatim
transcript into ``dataset_sample/smoke/`` as a matching pair::

    dataset_sample/smoke/<id>.wav
    dataset_sample/smoke/<id>.txt

Then run::

    python scripts/smoke_align.py                 # auto-discovers the pair
    python scripts/smoke_align.py --wav a.wav --txt a.txt
    python scripts/smoke_align.py --keep          # keep the corpus + TextGrid

It also performs the verification the spec asks for on day 1 - it prints the
installed MFA version and the locally available acoustic and dictionary model
names, and fails loudly if the names configured in ``configs/paths.yaml``
(``english_us_arpa``) are not among them.

Exit codes:
    0  alignment succeeded and the timestamps are monotonic
    1  alignment failed, or the timestamps are not monotonic
    2  nothing to align (no smoke file), or MFA/conda is not installed
"""

from __future__ import annotations

import argparse
import logging
import sys
from itertools import pairwise
from pathlib import Path

# Allow `python scripts/smoke_align.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.align.mfa import (
    MFAError,
    MFANotAvailableError,
    list_models,
    mfa_version,
    prepare_corpus,
    read_textgrid,
    run_mfa,
    validate,
)
from rostrum.config import load_config

logger = logging.getLogger("smoke_align")

N_WORDS_TO_PRINT = 10


def find_smoke_pair(smoke_dir: Path) -> tuple[Path, Path]:
    """Find one ``<id>.wav`` / ``<id>.txt`` pair in ``smoke_dir``.

    Args:
        smoke_dir: Directory holding the smoke-test recording.

    Returns:
        ``(wav_path, txt_path)``.

    Raises:
        FileNotFoundError: No WAV, or no transcript beside the WAV.
    """
    wavs = sorted(p for p in smoke_dir.glob("*.wav") if p.is_file())
    if not wavs:
        raise FileNotFoundError(
            f"no .wav in {smoke_dir}.\n"
            "Put one recording and its verbatim transcript there, with matching\n"
            "basenames, e.g. smoke01.wav + smoke01.txt, then re-run."
        )
    wav = wavs[0]
    txt = wav.with_suffix(".txt")
    if not txt.is_file():
        raise FileNotFoundError(f"transcript not found for {wav.name}: expected {txt}")
    if len(wavs) > 1:
        logger.warning("%d WAVs in %s; aligning %s", len(wavs), smoke_dir, wav.name)
    return wav, txt


def describe_audio(wav: Path) -> str:
    """Return a one-line description of a WAV, or a note if it cannot be read.

    Reports against invariant 1 (16 kHz mono) without enforcing it: MFA
    resamples internally, so a mismatch is a warning here, not a failure. The
    real enforcement lives in ``rostrum.io``.
    """
    try:
        import soundfile as sf
    except ImportError:  # pragma: no cover - soundfile is a hard dependency
        return "soundfile not installed; cannot inspect audio"
    try:
        info = sf.info(str(wav))
    except Exception as exc:
        return f"could not read audio: {exc}"
    note = ""
    if info.samplerate != 16000 or info.channels != 1:
        note = "  <- not yet 16 kHz mono (invariant 1); MFA will resample"
    return (
        f"{info.duration:.2f} s, {info.samplerate} Hz, " f"{info.channels} ch, {info.subtype}{note}"
    )


def check_models(env_name: str, acoustic: str, dictionary: str) -> list[str]:
    """Print the installed MFA version and model names; return any problems.

    Args:
        env_name: Conda environment holding MFA.
        acoustic: Acoustic model name from the config.
        dictionary: Dictionary name from the config.

    Returns:
        Human-readable problem strings; empty when everything checks out.
    """
    problems: list[str] = []

    version = mfa_version(env_name=env_name)
    print(f"MFA version              : {version}")

    for kind, wanted in (("acoustic", acoustic), ("dictionary", dictionary)):
        try:
            available = list_models(kind, env_name=env_name)
        except MFAError as exc:
            print(f"{kind + ' models':25}: could not list ({exc.__class__.__name__})")
            problems.append(f"could not list {kind} models: {exc}")
            continue
        shown = ", ".join(available) if available else "(none downloaded)"
        print(f"{kind + ' models':25}: {shown}")
        print(f"{'  configured':25}: {wanted}")
        if wanted not in available:
            problems.append(
                f"configured {kind} model {wanted!r} is not installed. Run:\n"
                f"    conda run -n {env_name} mfa model download {kind} {wanted}\n"
                f"  If that name no longer exists, pick one from the list above and\n"
                f"  update configs/paths.yaml -> mfa.{kind}"
                + ("_model" if kind == "acoustic" else "")
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Align the smoke file and print its first words. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", type=Path, help="recording to align")
    parser.add_argument("--txt", type=Path, help="verbatim transcript of --wav")
    parser.add_argument("--file-id", default=None, help="corpus id (default: WAV stem)")
    parser.add_argument(
        "--n", type=int, default=N_WORDS_TO_PRINT, help="words to print (default 10)"
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the staged corpus and the TextGrid"
    )
    parser.add_argument(
        "--skip-validate", action="store_true", help="skip the OOV check before aligning"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config()
    mfa_cfg = cfg.paths["mfa"]
    env_name = str(mfa_cfg["env_name"])
    acoustic = str(mfa_cfg["acoustic_model"])
    dictionary = str(mfa_cfg["dictionary"])
    num_jobs = int(mfa_cfg["num_jobs"])
    extra_dict_cfg = mfa_cfg.get("extra_dict")
    extra_dict = (cfg.root / str(extra_dict_cfg)) if extra_dict_cfg else None

    # -- locate the input ---------------------------------------------------
    if args.wav and args.txt:
        wav, txt = args.wav, args.txt
    elif args.wav or args.txt:
        parser.error("--wav and --txt must be given together")
    else:
        smoke_dir = cfg.path("sample.dir") / "smoke"
        try:
            wav, txt = find_smoke_pair(smoke_dir)
        except FileNotFoundError as exc:
            print(f"\nNothing to align.\n\n{exc}", file=sys.stderr)
            return 2
    if not wav.is_file():
        print(f"audio not found: {wav}", file=sys.stderr)
        return 2
    if not txt.is_file():
        print(f"transcript not found: {txt}", file=sys.stderr)
        return 2

    file_id = args.file_id or wav.stem
    transcript = txt.read_text(encoding="utf-8")

    print("=" * 72)
    print("Rostrum forced-alignment smoke test")
    print("=" * 72)
    print(f"audio                    : {wav}")
    print(f"  {describe_audio(wav)}")
    print(f"transcript               : {txt}")
    print(f"conda env                : {env_name}")
    if extra_dict is not None:
        print(f"project dictionary       : {extra_dict}")

    # -- verify MFA and its models -----------------------------------------
    try:
        problems = check_models(env_name, acoustic, dictionary)
    except MFANotAvailableError as exc:
        print(f"\nMFA is not reachable.\n\n{exc}", file=sys.stderr)
        return 2
    except MFAError as exc:
        print(f"\n`mfa version` failed:\n{exc}", file=sys.stderr)
        return 2
    if problems:
        print("\nModel check failed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    # -- stage, validate, align --------------------------------------------
    corpus_dir = cfg.root / str(mfa_cfg["corpus_dir"]) / "smoke"
    out_dir = cfg.root / str(mfa_cfg["output_dir"]) / "smoke"

    staged = prepare_corpus([(wav, transcript, file_id)], corpus_dir)
    item = staged[file_id]
    print(f"\nlab tokens ({len(item.text.lab_tokens):d}) : {item.text.lab_text[:140]}")
    print(f"display tokens ({len(item.text.display_tokens):d})")

    if not args.skip_validate:
        report = validate(
            corpus_dir,
            acoustic=acoustic,
            dictionary=dictionary,
            extra_dict=extra_dict,
            env_name=env_name,
            num_jobs=num_jobs,
        )
        if report.oov_words:
            print(f"\nOOV words ({len(report.oov_words)}): {', '.join(report.oov_words)}")
            print(
                "  These align as <unk> and break word-index correspondence.\n"
                "  Add pronunciations to a project dictionary, or expand the text."
            )
        else:
            print("\nOOV words                : none")

    try:
        grids = run_mfa(
            corpus_dir,
            out_dir,
            acoustic=acoustic,
            dictionary=dictionary,
            extra_dict=extra_dict,
            env_name=env_name,
            num_jobs=num_jobs,
        )
    except MFAError as exc:
        print(f"\nAlignment FAILED.\n{exc}", file=sys.stderr)
        return 1

    grid = next((g for g in grids if g.stem == file_id), grids[0])
    print(f"TextGrid                 : {grid}")

    try:
        alignment = read_textgrid(
            grid,
            file_id=file_id,
            text=item.text,
            aligner=f"mfa {mfa_version(env_name=env_name)}",
            acoustic_model=acoustic,
            dictionary=dictionary,
        )
    except ValueError as exc:
        print(f"\nTextGrid parsed but is inconsistent with the transcript:\n{exc}", file=sys.stderr)
        return 1

    # -- report -------------------------------------------------------------
    print(f"\naligned words            : {alignment.n_words}")
    print(f"speech span              : {alignment.speech_duration:.3f} s")
    print(f"\nfirst {min(args.n, alignment.n_words)} words:")
    print(f"  {'idx':>3}  {'start':>8}  {'end':>8}  {'dur':>6}  {'syl':>3}  word")
    for word in alignment.words[: args.n]:
        print(
            f"  {word.idx:>3}  {word.t_start:>8.3f}  {word.t_end:>8.3f}  "
            f"{word.duration:>6.3f}  {word.n_syllables:>3}  {word.text}"
            f"{'' if word.display == word.text else f'  ({word.display})'}"
        )

    # -- the acceptance check ----------------------------------------------
    violations: list[str] = []
    for word in alignment.words:
        if word.t_end <= word.t_start:
            violations.append(f"word {word.idx} ({word.text!r}) has non-positive duration")
    for earlier, later in pairwise(alignment.words):
        if later.t_start < earlier.t_end:
            violations.append(
                f"word {later.idx} ({later.text!r}) starts at {later.t_start:.3f} "
                f"before word {earlier.idx} ends at {earlier.t_end:.3f}"
            )

    if violations:
        print("\nTIMESTAMPS NOT MONOTONIC:", file=sys.stderr)
        for violation in violations[:20]:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    longest = max(alignment.gaps, key=lambda g: g.duration, default=None)
    if longest is not None:
        print(
            f"\nlongest gap              : {longest.duration:.3f} s at "
            f"{longest.t_start:.3f} s (after {alignment.words[longest.prev_word_idx].text!r})"
        )

    if not args.keep:
        import shutil

        shutil.rmtree(corpus_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        print("(staged corpus and TextGrid removed; pass --keep to retain them)")

    print("\nSMOKE TEST PASSED: alignment succeeded, timestamps are monotonic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
