"""Committed paths that scripts and `make` targets name literally.

``dataset_sample/`` is rebuilt by ``scripts/package_dataset.py``, and its
contents change whenever the selection rules change. Three places hard-code a
file from it — the ``docker-verify`` smoke pair in ``Makefile`` and
``scripts/docker_verify.sh``, and the dataset example in
``scripts/analyze_examples.py``. A sample rebuild once left all three pointing
at files that no longer existed, and nothing failed until Docker ran. These
tests fail in the repo instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "dataset_sample"


def _assign(text: str, name: str, pattern: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"{name} is no longer assigned the way this test expects"
    return match.group(1)


@pytest.mark.parametrize("name", ["DOCKER_SMOKE_WAV", "DOCKER_SMOKE_TEXT"])
def test_makefile_smoke_path_exists(name: str) -> None:
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    rel = _assign(text, name, rf"^{name} \?= (\S+)$")
    assert (ROOT / rel).is_file(), rel


@pytest.mark.parametrize("name", ["WAV", "TEXT"])
def test_docker_verify_default_path_exists(name: str) -> None:
    text = (ROOT / "scripts" / "docker_verify.sh").read_text(encoding="utf-8")
    rel = _assign(text, name, rf"^{name}=\$\{{DOCKER_SMOKE_\w+:-(\S+?)\}}$")
    assert (ROOT / rel).is_file(), rel


def test_makefile_and_docker_verify_agree() -> None:
    """The shell default and the make variable must name the same pair."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "docker_verify.sh").read_text(encoding="utf-8")
    assert _assign(makefile, "DOCKER_SMOKE_WAV", r"^DOCKER_SMOKE_WAV \?= (\S+)$") == _assign(
        shell, "WAV", r"^WAV=\$\{DOCKER_SMOKE_WAV:-(\S+?)\}$"
    )
    assert _assign(makefile, "DOCKER_SMOKE_TEXT", r"^DOCKER_SMOKE_TEXT \?= (\S+)$") == _assign(
        shell, "TEXT", r"^TEXT=\$\{DOCKER_SMOKE_TEXT:-(\S+?)\}$"
    )


def test_analyze_examples_sample_audio_exists() -> None:
    text = (ROOT / "scripts" / "analyze_examples.py").read_text(encoding="utf-8")
    name = _assign(text, "_SAMPLE_AUDIO", r'^_SAMPLE_AUDIO = "(\S+)"$')
    assert (SAMPLE / "audio" / name).is_file(), name


def test_analyze_examples_sample_transcript_exists() -> None:
    text = (ROOT / "scripts" / "analyze_examples.py").read_text(encoding="utf-8")
    text_id = _assign(text, "_SAMPLE_TEXT_ID", r'^_SAMPLE_TEXT_ID = "(\S+)"$')
    assert (SAMPLE / "transcripts" / f"{text_id}.json").is_file(), text_id


def test_the_smoke_audio_is_a_clean_reading() -> None:
    """A derived file would make the smoke test align an edited recording."""
    from rostrum.data.manifest import read_manifest

    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    rel = _assign(text, "DOCKER_SMOKE_WAV", r"^DOCKER_SMOKE_WAV \?= (\S+)$")
    stem = Path(rel).stem
    rows = {r.file_id: r for r in read_manifest(SAMPLE / "MANIFEST.csv")}
    assert stem in rows, stem
    assert rows[stem].origin == "clean", stem
