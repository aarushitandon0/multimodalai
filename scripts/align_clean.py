#!/usr/bin/env python
"""Force-align every clean WAV with MFA and store the TextGrids (spec 8).

Reads ``dataset/audio/clean/*.wav``, pairs each file with the verbatim
transcript of its text (``dataset/MANIFEST.csv`` gives the ``text_id``), stages
one MFA corpus, runs ``mfa validate`` then ``mfa align`` in the separate
``mfa`` conda environment, and copies the result to::

    dataset/alignments/<file_id>.TextGrid

Transcripts are looked up in this order, first hit wins:

1. ``dataset/texts/overrides/<file_id>.txt`` - for a reading that deviates from
   the library text (a LibriVox intro, a dropped sentence, a different edition);
2. ``dataset/texts/<text_id>.json`` - its ``verbatim`` field;
3. ``dataset/texts/<text_id>.txt``.

Alignment is one batch, not one run per file: MFA's speaker-adapted training
sees every rendition of the text at once, and a single ``--num_jobs 1`` run
keeps the output deterministic (``CLAUDE.md`` invariant 13).

Usage::

    python scripts/align_clean.py
    python scripts/align_clean.py --only s1_lv01 s1_lv02
    python scripts/align_clean.py --allow-oov        # align despite OOV words
    python scripts/align_clean.py --keep-corpus      # leave .cache/mfa in place

Exit codes:
    0  every requested file aligned and parsed
    1  alignment failed, a transcript is missing, or OOV words were found
    2  nothing to align, or MFA/conda is not installed
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Allow `python scripts/align_clean.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rostrum.align.mfa import (
    CorpusItem,
    MFAError,
    MFANotAvailableError,
    list_models,
    mfa_version,
    prepare_corpus,
    read_textgrid,
    run_mfa,
    validate,
)
from rostrum.config import Config, load_config
from rostrum.data.manifest import read_manifest
from rostrum.io.hashing import sha256_file

logger = logging.getLogger("align_clean")


class MissingTranscript(Exception):
    """No transcript could be found for a clean WAV."""


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


def transcript_for(file_id: str, text_id: str, texts_dir: Path) -> tuple[str, Path]:
    """Return ``(transcript, source_path)`` for one recording.

    Args:
        file_id: The recording's id.
        text_id: The library text it is a rendition of.
        texts_dir: ``dataset/texts``.

    Raises:
        MissingTranscript: Nothing in the lookup order exists, or the file that
            does exist holds no usable text. The message names every path that
            was tried, because filling one of them in is the fix.
    """
    override = texts_dir / "overrides" / f"{file_id}.txt"
    as_json = texts_dir / f"{text_id}.json"
    as_txt = texts_dir / f"{text_id}.txt"

    if override.is_file():
        text = override.read_text(encoding="utf-8").strip()
        if text:
            return text, override
    if as_json.is_file():
        data = json.loads(as_json.read_text(encoding="utf-8"))
        text = str(data.get("verbatim") or "").strip()
        if text and not text.startswith("TODO"):
            return text, as_json
    if as_txt.is_file():
        text = as_txt.read_text(encoding="utf-8").strip()
        if text:
            return text, as_txt

    raise MissingTranscript(
        f"{file_id}: no verbatim transcript for text_id {text_id!r}. Create one of:\n"
        f'    {as_json}   (field "verbatim")\n'
        f"    {as_txt}\n"
        f"    {override}  (only if this reading differs from the library text)"
    )


def clean_items(
    cfg: Config, only: set[str] | None
) -> tuple[list[tuple[Path, str, str]], dict[str, str], list[str]]:
    """Collect ``(wav, transcript, file_id)`` triples for the clean corpus.

    Returns:
        ``(items, text_ids, problems)`` - the corpus triples, a
        ``{file_id: text_id}`` map for the TextGrid metadata, and one message
        per file that could not be paired with a transcript.
    """
    clean_dir = cfg.path("data.audio_clean")
    texts_dir = cfg.path("data.texts")
    by_file = {row.file_id: row for row in read_manifest(cfg.path("data.manifest"))}

    items: list[tuple[Path, str, str]] = []
    text_ids: dict[str, str] = {}
    problems: list[str] = []

    for wav in sorted(clean_dir.glob("*.wav")):
        file_id = wav.stem
        if only is not None and file_id not in only:
            continue
        row = by_file.get(file_id)
        if row is None:
            problems.append(f"{file_id}: no MANIFEST.csv row, so its text_id is unknown")
            continue
        try:
            transcript, source = transcript_for(file_id, row.text_id, texts_dir)
        except MissingTranscript as exc:
            problems.append(str(exc))
            continue
        logger.debug("%s: transcript from %s", file_id, source)
        items.append((wav, transcript, file_id))
        text_ids[file_id] = row.text_id

    return items, text_ids, problems


# --------------------------------------------------------------------------- #
# MFA
# --------------------------------------------------------------------------- #


def check_models(cfg: Config) -> list[str]:
    """Verify the configured MFA model names exist locally (spec 8, day-1 note).

    Returns:
        One message per missing model; empty when both are present.
    """
    mfa_cfg = cfg.paths["mfa"]
    env = str(mfa_cfg["env_name"])
    problems: list[str] = []
    for kind, key in (("acoustic", "acoustic_model"), ("dictionary", "dictionary")):
        wanted = str(mfa_cfg[key])
        available = list_models(kind, env_name=env)
        logger.info("mfa %s models available: %s", kind, ", ".join(available) or "(none)")
        if wanted not in available:
            problems.append(
                f"{kind} model {wanted!r} is not downloaded; run "
                f"`conda run -n {env} mfa model download {kind} {wanted}`"
            )
    return problems


def align_batch(
    cfg: Config, items: list[tuple[Path, str, str]], *, allow_oov: bool, keep_corpus: bool
) -> tuple[dict[str, CorpusItem], list[Path]]:
    """Stage, validate and align the whole clean corpus in one MFA run.

    Raises:
        MFAError: Validation found OOV words and ``allow_oov`` is false, or the
            alignment itself failed.
    """
    mfa_cfg = cfg.paths["mfa"]
    env = str(mfa_cfg["env_name"])
    corpus_dir = cfg.path("mfa.corpus_dir")
    out_dir = cfg.path("mfa.output_dir")
    extra_dict = mfa_cfg["extra_dict"]

    staged = prepare_corpus(items, corpus_dir)
    report = validate(
        corpus_dir,
        acoustic=str(mfa_cfg["acoustic_model"]),
        dictionary=str(mfa_cfg["dictionary"]),
        extra_dict=extra_dict,
        env_name=env,
        num_jobs=int(mfa_cfg["num_jobs"]),
    )
    if report.oov_words and not allow_oov:
        raise MFAError(
            f"mfa validate found {len(report.oov_words)} out-of-vocabulary word(s): "
            f"{', '.join(report.oov_words[:20])}\n"
            "Each becomes <unk> in the TextGrid and breaks the word-index correspondence.\n"
            "Fix the transcript, or add pronunciations to a project dictionary and set\n"
            "paths.mfa.extra_dict, or re-run with --allow-oov to accept the damage."
        )

    grids = run_mfa(
        corpus_dir,
        out_dir,
        acoustic=str(mfa_cfg["acoustic_model"]),
        dictionary=str(mfa_cfg["dictionary"]),
        extra_dict=extra_dict,
        env_name=env,
        num_jobs=int(mfa_cfg["num_jobs"]),
    )
    if not keep_corpus:
        shutil.rmtree(corpus_dir, ignore_errors=True)
    return staged, grids


def publish(
    cfg: Config,
    grids: list[Path],
    staged: dict[str, CorpusItem],
    text_ids: dict[str, str],
    version: str,
) -> tuple[list[str], list[str]]:
    """Copy TextGrids into ``dataset/alignments`` after parsing each one.

    Parsing before publishing is the point: :func:`read_textgrid` raises when the
    aligned word count does not match the transcript, which is exactly the
    failure that would silently misalign one rendition against the others.

    Returns:
        ``(published_file_ids, problems)``.
    """
    dest_dir = cfg.ensure_dir("data.alignments")
    clean_dir = cfg.path("data.audio_clean")
    mfa_cfg = cfg.paths["mfa"]

    published: list[str] = []
    problems: list[str] = []
    for grid in grids:
        file_id = grid.stem
        item = staged.get(file_id)
        if item is None:
            problems.append(f"{grid.name}: MFA wrote a TextGrid for an unknown file_id")
            continue
        wav = clean_dir / f"{file_id}.wav"
        try:
            alignment = read_textgrid(
                grid,
                file_id=file_id,
                text=item.text,
                text_id=text_ids.get(file_id),
                aligner=f"mfa {version}",
                acoustic_model=str(mfa_cfg["acoustic_model"]),
                dictionary=str(mfa_cfg["dictionary"]),
                audio_sha256=sha256_file(wav) if wav.is_file() else "",
            )
        except ValueError as exc:
            problems.append(f"{file_id}: {exc}")
            continue
        shutil.copyfile(grid, dest_dir / f"{file_id}.TextGrid")
        published.append(file_id)
        logger.info(
            "%s: %d words, %.3f s, %d OOV",
            file_id,
            alignment.n_words,
            alignment.duration,
            len(alignment.oov_words),
        )
    return published, problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--only", nargs="+", metavar="FILE_ID", help="align only these file ids")
    p.add_argument(
        "--allow-oov",
        action="store_true",
        help="align even though the dictionary is missing words (they become <unk>)",
    )
    p.add_argument(
        "--keep-corpus", action="store_true", help="keep the staged corpus under .cache/mfa"
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Align the clean set. See the module docstring for exit codes."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(apply_seeds=False)

    items, text_ids, problems = clean_items(cfg, set(args.only) if args.only else None)
    if problems:
        print("\nCANNOT ALIGN these files:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("", file=sys.stderr)
    if not items:
        logger.error(
            "nothing to align: no usable (wav, transcript) pair under %s",
            cfg.path("data.audio_clean"),
        )
        return 2

    try:
        version = mfa_version(env_name=str(cfg.paths["mfa"]["env_name"]))
        logger.info("MFA %s", version)
        model_problems = check_models(cfg)
    except MFANotAvailableError as exc:
        logger.error("%s", exc)
        return 2
    except MFAError as exc:
        logger.error("%s", exc)
        return 1
    if model_problems:
        for problem in model_problems:
            logger.error("%s", problem)
        return 1

    try:
        staged, grids = align_batch(
            cfg, items, allow_oov=args.allow_oov, keep_corpus=args.keep_corpus
        )
    except MFANotAvailableError as exc:
        logger.error("%s", exc)
        return 2
    except MFAError as exc:
        logger.error("%s", exc)
        return 1

    published, publish_problems = publish(cfg, grids, staged, text_ids, version)
    logger.info(
        "aligned %d/%d file(s) into %s", len(published), len(items), cfg.path("data.alignments")
    )
    if publish_problems:
        print("\nALIGNMENT REJECTED for these files:\n", file=sys.stderr)
        for problem in publish_problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nA word-count mismatch means the transcript does not match what was read.\n"
            "Put the reading's own words in dataset/texts/overrides/<file_id>.txt and re-run.\n",
            file=sys.stderr,
        )

    if problems or publish_problems or len(published) != len(items):
        return 1
    print("\nNext: python scripts/align_report.py", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
