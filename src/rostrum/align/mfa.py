"""Montreal Forced Aligner wrapper: corpus prep, invocation, TextGrid parsing.

MFA is the *only* aligner Rostrum uses, both for dataset label verification and
at inference (``CLAUDE.md`` invariant 4; spec section 8).
``torchaudio.functional.forced_align`` is never used - it was deprecated in
torchaudio 2.8 and removed in 2.9 [R12].

MFA pins Kaldi, OpenFst and pynini through conda-forge and cannot share an
environment with librosa/torch/streamlit, so it lives in its own conda env and
is reached by shelling out::

    conda run -n mfa mfa align <corpus> english_us_arpa english_us_arpa <out>

Three things in this module are deliberately *pure* and therefore unit-testable
without MFA installed:

* :func:`expand_numerals` - ``"1863"`` -> ``["eighteen", "sixty", "three"]``
* :func:`normalise_for_alignment` - transcript -> ``.lab`` tokens **plus** the
  index map back to the display tokens
* :func:`read_textgrid` - Praat TextGrid -> :class:`~rostrum.types.Alignment`

The TextGrid parser is hand-written rather than delegated to ``praatio``/``tgt``
so that the one function every downstream timestamp depends on carries no
third-party version risk, and so it can be tested against a fixture.

Token mapping
-------------
Alignment text and display text do not have the same token count: ``"1863,"`` is
one display token but three ``.lab`` tokens. Every function here therefore
carries a :class:`LabText`, whose ``lab_to_display`` gives, for each aligned
token, the index of the display token it came from. Gap labels come from the
*display* side (punctuation), so that map has to survive as far as
:mod:`rostrum.text`.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from rostrum.types import Alignment, Gap, Word

__all__ = [
    "ARPA_VOWELS",
    "SILENCE_TOKENS",
    "UNKNOWN_TOKEN",
    "CorpusItem",
    "LabText",
    "MFAError",
    "MFANotAvailableError",
    "TextGridTier",
    "ValidationReport",
    "expand_numerals",
    "format_textgrid",
    "list_models",
    "mfa_version",
    "normalise_for_alignment",
    "parse_textgrid",
    "prepare_corpus",
    "read_textgrid",
    "run_mfa",
    "validate",
    "write_textgrid",
]

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SILENCE_TOKENS: frozenset[str] = frozenset({"", "sil", "sp", "spn", "silence", "<eps>"})
"""Words-tier labels that denote silence rather than a spoken word.

MFA writes empty intervals for inter-word silence; ``sil``/``sp``/``spn`` appear
in the phones tier and, with some acoustic models, in the words tier too.
"""

UNKNOWN_TOKEN: str = "<unk>"
"""Words-tier label MFA writes where the dictionary had no pronunciation.

These *are* real words - they consume time and must keep their index - so they
are aligned as words and additionally recorded in ``Alignment.oov_words``.
"""

ARPA_VOWELS: frozenset[str] = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)
"""ARPAbet vowel nuclei; one per syllable, so counting them counts syllables."""

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))

_ORDINAL_ONES = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

# Characters folded to ASCII before tokenising, so a curly apostrophe in a
# Project Gutenberg text does not turn "nation's" into an out-of-vocabulary word.
_UNICODE_FOLD = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201b": "'",
    "\u02bc": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\u2026": " ",
    "\u00a0": " ",
    "\u2009": " ",
    "\u202f": " ",
}

# Punctuation stripped from the edges of a token on the way into a .lab file.
_STRIP_CHARS = "\"#$%&()*+,./:;<=>?@[\\]^_`{|}~!'\u00b7-"

# The same set minus '$' and '%', which carry meaning ("dollars", "percent")
# and so must be read off a numeral before its punctuation is stripped.
_STRIP_KEEP_UNITS = _STRIP_CHARS.replace("$", "").replace("%", "")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class MFAError(RuntimeError):
    """An MFA subprocess failed. The message carries the captured stderr."""


class MFANotAvailableError(MFAError):
    """``conda`` or the ``mfa`` environment could not be found on this machine."""


# --------------------------------------------------------------------------- #
# Numeral expansion (pure)
# --------------------------------------------------------------------------- #


def _int_to_words(n: int) -> list[str]:
    """Spell a non-negative integer in English as whitespace-separated tokens.

    Args:
        n: Integer in ``[0, 10**12)``.

    Returns:
        Tokens, e.g. ``123`` -> ``["one", "hundred", "twenty", "three"]``.
        Hyphens are never emitted - ``twenty three`` is two ``.lab`` tokens,
        because the ARPA dictionary keys single words.

    Raises:
        ValueError: ``n`` is negative or too large to spell.
    """
    if n < 0:
        raise ValueError(f"cannot spell a negative integer: {n}")
    if n >= 10**12:
        raise ValueError(f"integer too large to spell: {n}")
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        tens, ones = divmod(n, 10)
        return [_TENS[tens]] + ([_ONES[ones]] if ones else [])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return [_ONES[hundreds], "hundred"] + (_int_to_words(rest) if rest else [])
    for value, name in _SCALES:
        if n >= value:
            count, rest = divmod(n, value)
            return _int_to_words(count) + [name] + (_int_to_words(rest) if rest else [])
    raise AssertionError("unreachable")  # pragma: no cover


def _year_to_words(n: int) -> list[str]:
    """Spell a year the way a speaker reads it aloud.

    ``1863`` -> ``eighteen sixty three`` (not "one thousand eight hundred and
    sixty three"), ``1900`` -> ``nineteen hundred``, ``1805`` -> ``eighteen oh
    five``, ``2005`` -> ``two thousand five``, ``2020`` -> ``twenty twenty``.

    Args:
        n: Year in ``[1100, 2099]``.

    Returns:
        Spelled tokens.
    """
    high, low = divmod(n, 100)
    if 2000 <= n <= 2009:
        return ["two", "thousand"] + (_int_to_words(low) if low else [])
    if low == 0:
        return [*_int_to_words(high), "hundred"]
    if low < 10:
        return [*_int_to_words(high), "oh", _ONES[low]]
    return _int_to_words(high) + _int_to_words(low)


def _ordinal(tokens: list[str]) -> list[str]:
    """Turn cardinal tokens into their ordinal form (``twenty three`` -> ...``third``)."""
    if not tokens:
        return tokens
    last = tokens[-1]
    if last in _ORDINAL_ONES:
        suffix = _ORDINAL_ONES[last]
    elif last.endswith("y"):
        suffix = last[:-1] + "ieth"  # twenty -> twentieth
    else:
        suffix = last + "th"
    return [*tokens[:-1], suffix]


def expand_numerals(token: str, *, prefer_year: bool = True) -> list[str]:
    """Expand one numeric display token into spoken ``.lab`` tokens.

    Covers the forms that actually occur in the public-domain speeches this
    project uses: plain cardinals, thousands separators, years, ordinals,
    decimals, percentages and dollar amounts. A token with no digits is
    returned unchanged, so this is safe to call on every token.

    Args:
        token: One whitespace-free display token, e.g. ``"1863,"``. Surrounding
            punctuation is tolerated.
        prefer_year: Read a bare four-digit number in ``[1100, 2099]`` as a
            year. Set ``False`` for texts where such numbers are quantities.

    Returns:
        Lower-case tokens containing no digits and no hyphens.

    Examples:
        >>> expand_numerals("1863")
        ['eighteen', 'sixty', 'three']
        >>> expand_numerals("87")
        ['eighty', 'seven']
        >>> expand_numerals("4th")
        ['fourth']
        >>> expand_numerals("word")
        ['word']
    """
    raw = token.strip()
    if not raw or not any(ch.isdigit() for ch in raw):
        return [raw] if raw else []

    core = raw.strip(_STRIP_KEEP_UNITS)
    if not core:
        return []

    leading_dollar = core.startswith("$")
    if leading_dollar:
        core = core[1:]
    trailing_percent = core.endswith("%")
    if trailing_percent:
        core = core[:-1]
    core = core.strip(_STRIP_CHARS)
    if not core:
        return []

    suffix: list[str] = []
    if leading_dollar:
        suffix.append("dollars")
    if trailing_percent:
        suffix.append("percent")

    # Ordinals: 1st, 2nd, 3rd, 4th, 21st.
    m = re.fullmatch(r"(\d+)(st|nd|rd|th)", core, flags=re.IGNORECASE)
    if m:
        return _ordinal(_int_to_words(int(m.group(1)))) + suffix

    digits = core.replace(",", "")

    # Decimals: the fractional digits are read one by one.
    m = re.fullmatch(r"(\d*)\.(\d+)", digits)
    if m:
        whole = _int_to_words(int(m.group(1))) if m.group(1) else ["zero"]
        return whole + ["point"] + [_ONES[int(d)] for d in m.group(2)] + suffix

    if re.fullmatch(r"\d+", digits):
        value = int(digits)
        if prefer_year and len(digits) == 4 and 1100 <= value <= 2099 and not suffix:
            return _year_to_words(value)
        return _int_to_words(value) + suffix

    # Mixed alphanumerics ("B29"): split into runs and expand the digit runs.
    out: list[str] = []
    for part in re.findall(r"\d+|[^\W\d_]+", core, flags=re.UNICODE):
        out.extend(_int_to_words(int(part)) if part.isdigit() else [part.lower()])
    return out + suffix


# --------------------------------------------------------------------------- #
# Transcript normalisation (pure)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LabText:
    """A transcript in both of the forms the pipeline needs.

    Attributes:
        lab_tokens: Lower-case, punctuation-free, digit-free tokens. These are
            what MFA aligns, so ``Word.idx`` indexes this tuple.
        display_tokens: Tokens as written, with case and punctuation. Gap
            labels (boundary/minor/none) are derived from these.
        lab_to_display: For each entry of ``lab_tokens``, the index into
            ``display_tokens`` it came from. Non-decreasing.
    """

    lab_tokens: tuple[str, ...]
    display_tokens: tuple[str, ...]
    lab_to_display: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.lab_tokens) != len(self.lab_to_display):
            raise ValueError(
                f"lab_to_display has {len(self.lab_to_display)} entries for "
                f"{len(self.lab_tokens)} lab tokens"
            )

    @property
    def lab_text(self) -> str:
        """Contents of the ``.lab`` file: tokens joined by single spaces."""
        return " ".join(self.lab_tokens)

    def display_for(self, lab_idx: int) -> str:
        """Return the display token behind aligned word ``lab_idx``."""
        return self.display_tokens[self.lab_to_display[lab_idx]]


def _fold_unicode(text: str) -> str:
    """Map typographic punctuation to ASCII and drop combining marks."""
    folded = "".join(_UNICODE_FOLD.get(ch, ch) for ch in text)
    decomposed = unicodedata.normalize("NFKD", folded)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _clean_lab_token(token: str) -> str:
    """Lower-case a token and strip punctuation, keeping word-internal apostrophes."""
    lowered = token.lower().strip(_STRIP_CHARS)
    if not lowered:
        return ""
    kept = "".join(ch for ch in lowered if ch.isalpha() or ch == "'")
    return kept.strip("'")


def normalise_for_alignment(transcript: str, *, prefer_year: bool = True) -> LabText:
    """Split a transcript into aligned ``.lab`` tokens and display tokens.

    Rules, in order:

    1. Typographic punctuation is folded to ASCII (a curly apostrophe would
       otherwise make ``nation's`` out-of-vocabulary).
    2. Display tokens are whitespace-separated, keeping case and punctuation.
    3. Each display token is split on hyphens - ``Four-score`` yields two lab
       tokens, because the ARPA dictionary keys single words.
    4. Numeric pieces are expanded by :func:`expand_numerals`.
    5. The remainder is lower-cased and stripped of punctuation, keeping
       word-internal apostrophes (``don't``, ``nation's``).
    6. Tokens that reduce to nothing (a bare dash) are dropped and produce no
       entry in ``lab_to_display``.

    Args:
        transcript: Verbatim transcript text; newlines count as spaces.
        prefer_year: Passed through to :func:`expand_numerals`.

    Returns:
        A :class:`LabText`.

    Examples:
        >>> lt = normalise_for_alignment("Four-score and 7 years ago,")
        >>> lt.lab_text
        'four score and seven years ago'
        >>> lt.display_tokens
        ('Four-score', 'and', '7', 'years', 'ago,')
        >>> lt.lab_to_display
        (0, 0, 1, 2, 3, 4)
    """
    folded = _fold_unicode(transcript)
    display_tokens = tuple(folded.split())

    lab_tokens: list[str] = []
    lab_to_display: list[int] = []

    for d_idx, display in enumerate(display_tokens):
        for piece in display.split("-"):
            for expanded in expand_numerals(piece, prefer_year=prefer_year):
                cleaned = _clean_lab_token(expanded)
                if cleaned:
                    lab_tokens.append(cleaned)
                    lab_to_display.append(d_idx)

    return LabText(
        lab_tokens=tuple(lab_tokens),
        display_tokens=display_tokens,
        lab_to_display=tuple(lab_to_display),
    )


# --------------------------------------------------------------------------- #
# Corpus preparation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorpusItem:
    """One recording staged into an MFA corpus directory.

    Attributes:
        file_id: Basename shared by the staged ``.wav`` and ``.lab``.
        wav_path: The staged WAV inside the corpus directory.
        lab_path: The staged ``.lab`` transcript.
        text: The normalised transcript and its display-token map.
        source_wav: Where the audio was copied from.
    """

    file_id: str
    wav_path: Path
    lab_path: Path
    text: LabText
    source_wav: Path


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def prepare_corpus(
    items: Iterable[tuple[str | Path, str, str]],
    corpus_dir: str | Path,
    *,
    clean: bool = True,
    prefer_year: bool = True,
) -> dict[str, CorpusItem]:
    """Stage ``(wav, transcript, file_id)`` triples as an MFA corpus directory.

    MFA wants a flat directory of ``<id>.wav`` plus ``<id>.lab`` with matching
    basenames (spec section 8). The WAV is *copied*, not linked: symlinks need
    elevation on Windows, and a copy means a later re-run cannot be invalidated
    by the source moving.

    The returned :class:`CorpusItem` map is the only place the display-token
    mapping exists, so callers must keep it to pass to :func:`read_textgrid`.

    Args:
        items: Triples of ``(wav_path, transcript, file_id)``.
        corpus_dir: Directory to create and fill.
        clean: Delete an existing ``corpus_dir`` first. Leftover files from a
            previous run would otherwise be aligned too.
        prefer_year: Passed to :func:`normalise_for_alignment`.

    Returns:
        ``{file_id: CorpusItem}``, in input order.

    Raises:
        FileNotFoundError: A source WAV does not exist.
        ValueError: A ``file_id`` is unsafe as a filename, is duplicated, or a
            transcript normalises to no tokens at all.
    """
    corpus = Path(corpus_dir)
    if clean and corpus.exists():
        logger.debug("clearing corpus dir %s", corpus)
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True, exist_ok=True)

    staged: dict[str, CorpusItem] = {}
    for wav_path, transcript, file_id in items:
        if not _SAFE_ID.fullmatch(file_id):
            raise ValueError(
                f"unsafe file_id {file_id!r}: use letters, digits, dot, dash, underscore"
            )
        if file_id in staged:
            raise ValueError(f"duplicate file_id {file_id!r} in corpus items")

        source = Path(wav_path)
        if not source.is_file():
            raise FileNotFoundError(f"audio for {file_id!r} not found: {source}")

        text = normalise_for_alignment(transcript, prefer_year=prefer_year)
        if not text.lab_tokens:
            raise ValueError(f"transcript for {file_id!r} normalised to zero tokens")

        wav_dest = corpus / f"{file_id}.wav"
        lab_dest = corpus / f"{file_id}.lab"
        shutil.copyfile(source, wav_dest)
        # newline="\n" keeps the .lab byte-identical across platforms, which the
        # determinism test needs (CLAUDE.md invariant 13).
        with lab_dest.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.lab_text + "\n")

        staged[file_id] = CorpusItem(
            file_id=file_id,
            wav_path=wav_dest,
            lab_path=lab_dest,
            text=text,
            source_wav=source,
        )
        logger.debug("staged %s (%d lab tokens)", file_id, len(text.lab_tokens))

    logger.info("prepared MFA corpus at %s with %d file(s)", corpus, len(staged))
    return staged


# --------------------------------------------------------------------------- #
# Subprocess plumbing
# --------------------------------------------------------------------------- #


def _conda_executable() -> str:
    """Return the ``conda`` launcher to shell out to.

    Raises:
        MFANotAvailableError: ``conda`` is not on ``PATH``. The message names
            the setup target, because this is the first failure a fresh
            machine hits.
    """
    for name in ("conda", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            return found
    raise MFANotAvailableError(
        "conda is not on PATH, so the 'mfa' environment cannot be reached. "
        "Install Miniforge, then run `make setup-mfa` "
        "(or `conda env create -f environment.mfa.yml`)."
    )


def _run_mfa_command(
    args: Sequence[str],
    *,
    env_name: str = "mfa",
    timeout: float | None = 3600.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``conda run -n <env_name> mfa <args...>`` and capture its output.

    Args:
        args: Arguments after the ``mfa`` executable, e.g. ``["version"]``.
        env_name: Conda environment holding MFA.
        timeout: Seconds before the subprocess is killed; ``None`` to wait.
        check: Raise :class:`MFAError` on a non-zero exit.

    Returns:
        The completed process, with text ``stdout`` and ``stderr``.

    Raises:
        MFANotAvailableError: ``conda`` is missing.
        MFAError: The command failed, timed out, or the environment is absent.
    """
    conda = _conda_executable()
    # --no-capture-output is deliberately NOT passed: MFA's stderr is the only
    # diagnostic a failed alignment leaves behind, and it belongs in the error.
    cmd = [conda, "run", "-n", env_name, "mfa", *args]
    logger.debug("running %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise MFAError(f"`mfa {' '.join(args)}` timed out after {timeout}s") from exc
    except OSError as exc:  # pragma: no cover - conda vanished between calls
        raise MFANotAvailableError(f"could not launch {conda}: {exc}") from exc

    if check and proc.returncode != 0:
        raise MFAError(
            f"`mfa {' '.join(args)}` failed with exit code {proc.returncode}.\n"
            f"--- stderr ---\n{(proc.stderr or '').strip()}\n"
            f"--- stdout ---\n{(proc.stdout or '').strip()}"
        )
    return proc


def mfa_version(*, env_name: str = "mfa") -> str:
    """Return the MFA version string from ``mfa version``.

    Recorded in ``Alignment.aligner`` so a report names the aligner that
    produced its timestamps.

    Args:
        env_name: Conda environment holding MFA.

    Returns:
        The version, e.g. ``"3.2.1"``.

    Raises:
        MFANotAvailableError: conda or the environment is missing.
        MFAError: ``mfa version`` failed.
    """
    out = _run_mfa_command(["version"], env_name=env_name, timeout=300.0)
    return (out.stdout or out.stderr).strip().splitlines()[-1].strip()


def list_models(kind: str = "acoustic", *, env_name: str = "mfa") -> list[str]:
    """Return the names of locally downloaded MFA models of one kind.

    Used to check that the names in ``configs/paths.yaml`` (``english_us_arpa``)
    still exist before an alignment run, per the spec's day-1 note.

    Two output formats are accepted, because they differ by MFA version and the
    name check is the gate in front of every alignment run:

    * MFA 3.x prints one Python list literal - ``['english_us_arpa']`` -
      verified against ``mfa model list acoustic`` under MFA 3.2.1, which
      offers no plain-text or JSON output flag;
    * older builds print one name per line, optionally bulleted.

    Args:
        kind: ``acoustic``, ``dictionary``, ``g2p``, ``language_model`` or
            ``ivector`` - whatever ``mfa model list`` accepts.
        env_name: Conda environment holding MFA.

    Returns:
        Model names, in the order MFA listed them.

    Raises:
        MFANotAvailableError: conda or the environment is missing.
        MFAError: ``mfa model list`` failed.
    """
    out = _run_mfa_command(["model", "list", kind], env_name=env_name, timeout=300.0)
    names: list[str] = []
    for line in (out.stdout or "").splitlines():
        stripped = line.strip()
        # MFA 3.x: a list literal on one line. Split it into candidate tokens so
        # the same name check below applies to both formats.
        if stripped.startswith("[") and stripped.endswith("]"):
            candidates = stripped[1:-1].split(",")
        else:
            candidates = [stripped]
        for candidate in candidates:
            token = candidate.strip().lstrip("-*• ").strip().strip("\"'").strip()
            if token and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", token):
                names.append(token)
    return names


# --------------------------------------------------------------------------- #
# validate / align
# --------------------------------------------------------------------------- #

_OOV_LINE = re.compile(
    r"(?:oov|out[- ]of[- ]vocabulary)[^:\n]*:\s*(?P<words>.+)", flags=re.IGNORECASE
)


@dataclass(frozen=True)
class ValidationReport:
    """Result of ``mfa validate`` on a staged corpus.

    Attributes:
        oov_words: Words the dictionary has no pronunciation for. These must be
            fixed before aligning - by adding pronunciations to ``extra_dict``
            or by expanding the text (spec section 8, OOV handling).
        oov_file: The OOV listing MFA wrote, when one was found.
        returncode: Exit code of ``mfa validate``.
        stdout: Captured stdout, kept for the run log.
        stderr: Captured stderr.
    """

    oov_words: list[str] = field(default_factory=list)
    oov_file: Path | None = None
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        """Whether validation exited cleanly with no OOV words."""
        return self.returncode == 0 and not self.oov_words


def _parse_oov_words(text: str) -> list[str]:
    """Pull OOV words out of ``mfa validate`` output, de-duplicated and sorted."""
    found: set[str] = set()
    for match in _OOV_LINE.finditer(text):
        for token in re.split(r"[,\s]+", match.group("words")):
            cleaned = token.strip().strip("'\"[](){}.,")
            if cleaned and re.fullmatch(r"[A-Za-z'][A-Za-z']*", cleaned):
                found.add(cleaned.lower())
    return sorted(found)


def _find_oov_file(search_dirs: Iterable[Path]) -> Path | None:
    """Locate an ``oovs_found*`` listing written by ``mfa validate``.

    TODO(verify): MFA 3.x writes this under its own temporary root, whose
    location depends on ``MFA_ROOT_DIR``. Confirm the exact path against
    `mfa validate --help` once conda is available, and pass that directory in.
    """
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.rglob("oovs_found*")):
            if candidate.is_file():
                return candidate
    return None


def validate(
    corpus_dir: str | Path,
    *,
    acoustic: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
    extra_dict: str | Path | None = None,
    env_name: str = "mfa",
    num_jobs: int = 1,
    timeout: float | None = 14400.0,
    search_dirs: Sequence[str | Path] = (),
) -> ValidationReport:
    """Run ``mfa validate`` on a staged corpus and collect its OOV words.

    Validation is run before every alignment: an unpronounceable word silently
    becomes ``<unk>`` in the TextGrid, which would corrupt the word-index
    correspondence the whole comparison rests on.

    A non-zero exit does **not** raise - a corpus with OOV words is a result to
    act on, not a crash. Only an unreachable MFA raises.

    Args:
        corpus_dir: Directory from :func:`prepare_corpus`.
        acoustic: Acoustic model name.
        dictionary: Dictionary name.
        extra_dict: Optional project dictionary of extra pronunciations. When
            given it is passed instead of ``dictionary``, so it must already
            include the base entries that are needed.
        env_name: Conda environment holding MFA.
        num_jobs: MFA worker count; 1 keeps output ordering deterministic.
        timeout: Seconds before the subprocess is killed.
        search_dirs: Extra directories to search for an ``oovs_found*`` file.

    Returns:
        A :class:`ValidationReport`.

    Raises:
        FileNotFoundError: ``corpus_dir`` does not exist.
        MFANotAvailableError: conda or the environment is missing.
    """
    corpus = Path(corpus_dir)
    if not corpus.is_dir():
        raise FileNotFoundError(f"corpus dir does not exist: {corpus}")

    dict_arg = str(extra_dict) if extra_dict else dictionary
    proc = _run_mfa_command(
        [
            "validate",
            str(corpus),
            dict_arg,
            acoustic,
            "--num_jobs",
            str(num_jobs),
            "--clean",
            # OOV detection is a dictionary/text check. Without this flag MFA
            # trains a monophone model over the whole corpus first, which costs
            # hours and nothing here reads its output.
            "--ignore_acoustics",
        ],
        env_name=env_name,
        timeout=timeout,
        check=False,
    )
    combined = f"{proc.stdout or ''}\n{proc.stderr or ''}"
    oov_words = _parse_oov_words(combined)

    oov_file = _find_oov_file([corpus, corpus.parent, *(Path(d) for d in search_dirs)])
    if oov_file is not None:
        file_words = _parse_oov_words("oov: " + oov_file.read_text(encoding="utf-8"))
        oov_words = sorted(set(oov_words) | set(file_words))

    if oov_words:
        logger.warning("mfa validate found %d OOV word(s): %s", len(oov_words), oov_words)
    elif proc.returncode != 0:
        logger.warning("mfa validate exited %d with no parsable OOV list", proc.returncode)

    return ValidationReport(
        oov_words=oov_words,
        oov_file=oov_file,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def run_mfa(
    corpus_dir: str | Path,
    out_dir: str | Path,
    *,
    acoustic: str = "english_us_arpa",
    dictionary: str = "english_us_arpa",
    extra_dict: str | Path | None = None,
    env_name: str = "mfa",
    num_jobs: int = 1,
    beam: int | None = None,
    retry_beam: int | None = None,
    extra_args: Sequence[str] = (),
    timeout: float | None = 21600.0,
) -> list[Path]:
    """Align a staged corpus, returning the TextGrids MFA wrote.

    Invokes ``mfa align <corpus> <dictionary> <acoustic> <out>`` with
    ``--clean`` so no state from a previous run can leak in, and a fixed job
    count (1 by default) because MFA's multi-job output ordering is not
    deterministic (``CLAUDE.md`` invariant 13).

    Args:
        corpus_dir: Directory from :func:`prepare_corpus`.
        out_dir: Directory for the output TextGrids; created if absent.
        acoustic: Acoustic model name, e.g. ``english_us_arpa``.
        dictionary: Dictionary name, e.g. ``english_us_arpa``.
        extra_dict: Project dictionary of extra pronunciations, used instead of
            ``dictionary`` when given.
        env_name: Conda environment holding MFA.
        num_jobs: Worker count. Leave at 1 for reproducible output.
        beam: Decoding beam; ``None`` leaves the MFA default.
        retry_beam: Retry beam for utterances that fail at ``beam``.
        extra_args: Further flags appended verbatim, for anything this
            signature does not model.
        timeout: Seconds before the subprocess is killed.

    Returns:
        The ``.TextGrid`` files under ``out_dir``, sorted by path.

    Raises:
        FileNotFoundError: ``corpus_dir`` does not exist or holds no ``.wav``.
        MFANotAvailableError: conda or the environment is missing.
        MFAError: Alignment failed (stderr is included), or it exited cleanly
            but wrote no TextGrid.
    """
    corpus = Path(corpus_dir)
    if not corpus.is_dir():
        raise FileNotFoundError(f"corpus dir does not exist: {corpus}")
    if not any(corpus.glob("*.wav")):
        raise FileNotFoundError(f"no .wav files in corpus dir: {corpus}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dict_arg = str(extra_dict) if extra_dict else dictionary
    args: list[str] = [
        "align",
        str(corpus),
        dict_arg,
        acoustic,
        str(out),
        "--clean",
        "--num_jobs",
        str(num_jobs),
    ]
    if beam is not None:
        args += ["--beam", str(beam)]
    if retry_beam is not None:
        args += ["--retry_beam", str(retry_beam)]
    args += list(extra_args)

    _run_mfa_command(args, env_name=env_name, timeout=timeout)

    grids = sorted(out.rglob("*.TextGrid"))
    if not grids:
        raise MFAError(
            f"mfa align exited 0 but wrote no TextGrid to {out}. "
            "Run validate() first - an all-OOV corpus can align to nothing."
        )
    logger.info("aligned %d file(s) into %s", len(grids), out)
    return grids


# --------------------------------------------------------------------------- #
# TextGrid parsing (pure)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TextGridTier:
    """One interval tier of a parsed TextGrid.

    Attributes:
        name: Tier name, e.g. ``words`` or ``phones``. MFA 3 names word tiers
            after the speaker, so lookup is by position as well as by name.
        xmin: Tier start in seconds.
        xmax: Tier end in seconds.
        intervals: ``(start, end, label)`` triples in file order.
    """

    name: str
    xmin: float
    xmax: float
    intervals: tuple[tuple[float, float, str], ...]


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _read_text(path: Path) -> str:
    """Read a TextGrid, honouring the encodings Praat and MFA emit.

    Praat writes UTF-16 with a BOM for texts with non-ASCII characters and
    UTF-8 otherwise; ``utf-8-sig`` covers a UTF-8 BOM.
    """
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"could not decode TextGrid: {path}")  # pragma: no cover


def _tokenise_short_form(body: str) -> list[str]:
    """Split a short-format TextGrid body into quoted strings and bare numbers."""
    tokens: list[str] = []
    for match in re.finditer(r'"((?:[^"]|"")*)"|(\S+)', body):
        if match.group(1) is not None:
            tokens.append(match.group(1).replace('""', '"'))
        else:
            tokens.append(match.group(2))
    return tokens


def _parse_long_form(text: str) -> list[TextGridTier]:
    """Parse the long (``item [1]: ... intervals [1]:``) TextGrid format."""
    tiers: list[TextGridTier] = []
    # Each tier starts at an `item [n]:` header; split on them and parse each.
    chunks = re.split(r"\n\s*item\s*\[\d+\]\s*:", text)[1:]
    for chunk in chunks:
        cls = re.search(r'class\s*=\s*"([^"]*)"', chunk)
        if cls is None or cls.group(1) != "IntervalTier":
            continue  # TextTier/point tiers carry no word intervals
        name_m = re.search(r'name\s*=\s*"((?:[^"]|"")*)"', chunk)
        name = name_m.group(1).replace('""', '"') if name_m else ""
        bounds = re.search(
            r"xmin\s*=\s*(" + _NUMBER.pattern + r")\s*\n\s*xmax\s*=\s*(" + _NUMBER.pattern + r")",
            chunk,
        )
        t_min = float(bounds.group(1)) if bounds else 0.0
        t_max = float(bounds.group(2)) if bounds else 0.0

        intervals: list[tuple[float, float, str]] = []
        pattern = re.compile(
            r"intervals\s*\[\d+\]\s*:\s*"
            r"xmin\s*=\s*(" + _NUMBER.pattern + r")\s*"
            r"xmax\s*=\s*(" + _NUMBER.pattern + r")\s*"
            r'text\s*=\s*"((?:[^"]|"")*)"',
            flags=re.DOTALL,
        )
        for m in pattern.finditer(chunk):
            intervals.append(
                (float(m.group(1)), float(m.group(2)), m.group(3).replace('""', '"').strip())
            )
        tiers.append(TextGridTier(name=name, xmin=t_min, xmax=t_max, intervals=tuple(intervals)))
    return tiers


def _parse_short_form(text: str) -> list[TextGridTier]:
    """Parse the short (bare values, one per line) TextGrid format."""
    header_end = text.find("<exists>")
    if header_end < 0:
        return []
    tokens = _tokenise_short_form(text[header_end + len("<exists>") :])
    if not tokens:
        return []

    pos = 0

    def take() -> str:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("truncated short-format TextGrid")
        value = tokens[pos]
        pos += 1
        return value

    n_tiers = int(float(take()))
    tiers: list[TextGridTier] = []
    for _ in range(n_tiers):
        tier_class = take()
        name = take()
        t_min = float(take())
        t_max = float(take())
        count = int(float(take()))
        intervals: list[tuple[float, float, str]] = []
        for _ in range(count):
            if tier_class == "IntervalTier":
                start = float(take())
                end = float(take())
                label = take().strip()
                intervals.append((start, end, label))
            else:  # TextTier: time + label, no interval end
                take()
                take()
        if tier_class == "IntervalTier":
            tiers.append(
                TextGridTier(name=name, xmin=t_min, xmax=t_max, intervals=tuple(intervals))
            )
    return tiers


def parse_textgrid(path: str | Path) -> list[TextGridTier]:
    """Parse a Praat TextGrid into its interval tiers.

    Handles both the long and the short text formats, UTF-8 (with or without
    BOM) and UTF-16. Point tiers are skipped. This is the low-level half of
    :func:`read_textgrid`, exposed so a fixture can be checked tier by tier.

    Args:
        path: Path to the ``.TextGrid`` file.

    Returns:
        Interval tiers in file order.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: The file is not a text-format TextGrid, or is truncated.
    """
    grid = Path(path)
    if not grid.is_file():
        raise FileNotFoundError(f"TextGrid not found: {grid}")
    text = _read_text(grid)
    if "ooTextFile" not in text:
        raise ValueError(
            f"{grid} is not a text-format TextGrid (binary TextGrids are not supported)"
        )

    tiers = _parse_long_form(text) if re.search(r"\n\s*item\s*\[", text) else []
    if not tiers:
        tiers = _parse_short_form(text)
    if not tiers:
        raise ValueError(f"no interval tiers found in {grid}")
    return tiers


# --------------------------------------------------------------------------- #
# Writing TextGrids
# --------------------------------------------------------------------------- #


def _quote(label: str) -> str:
    """Escape a label for a TextGrid string literal (``"`` doubles)."""
    return label.replace('"', '""')


def _fill_gaps(
    intervals: Sequence[tuple[float, float, str]],
    xmin: float,
    xmax: float,
) -> list[tuple[float, float, str]]:
    """Return ``intervals`` padded with empty ones so they tile ``[xmin, xmax]``.

    Praat rejects a tier with holes in it, so every stretch not covered by an
    interval becomes an explicitly empty one. Intervals must already be sorted
    and non-overlapping.
    """
    filled: list[tuple[float, float, str]] = []
    cursor = xmin
    for start, end, label in intervals:
        if end <= start:
            raise ValueError(f"interval [{start}, {end}] has non-positive length")
        if start < cursor - 1e-9:
            raise ValueError(
                f"interval [{start}, {end}] overlaps the previous one, which ends at {cursor}"
            )
        if start > cursor + 1e-9:
            filled.append((cursor, start, ""))
        filled.append((start, end, label))
        cursor = end
    if cursor > xmax + 1e-9:
        raise ValueError(f"intervals run to {cursor}, past the tier end {xmax}")
    if cursor < xmax - 1e-9 or not filled:
        filled.append((cursor, xmax, ""))
    return filled


def format_textgrid(
    tiers: Sequence[TextGridTier],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    decimals: int = 3,
) -> str:
    """Render interval tiers as a long-format Praat TextGrid.

    The output is the dialect Praat and MFA both write, so Praat opens it
    without complaint and :func:`parse_textgrid` round-trips it. Each tier is
    tiled with empty intervals where it has none, and times are written with a
    fixed number of decimals so the file is byte-stable (``CLAUDE.md``
    invariants 2 and 13).

    Args:
        tiers: The tiers to write, in the order they should appear. Every tier
            is treated as an ``IntervalTier``.
        xmin: Overall start; defaults to the smallest tier ``xmin``.
        xmax: Overall end; defaults to the largest tier ``xmax``.
        decimals: Decimal places for every timestamp. The project default of 3
            is millisecond resolution.

    Returns:
        The file contents, ending in a newline.

    Raises:
        ValueError: ``tiers`` is empty, or a tier's intervals overlap, are out
            of order, or run past its ``xmax``.
    """
    if not tiers:
        raise ValueError("a TextGrid needs at least one tier")
    t_min = min(t.xmin for t in tiers) if xmin is None else float(xmin)
    t_max = max(t.xmax for t in tiers) if xmax is None else float(xmax)
    if t_max <= t_min:
        raise ValueError(f"TextGrid xmax {t_max} is not after xmin {t_min}")

    def num(x: float) -> str:
        return f"{x:.{decimals}f}"

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {num(t_min)}",
        f"xmax = {num(t_max)}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for i, tier in enumerate(tiers, start=1):
        intervals = _fill_gaps(tier.intervals, tier.xmin, tier.xmax)
        lines += [
            f"    item [{i}]:",
            '        class = "IntervalTier"',
            f'        name = "{_quote(tier.name)}"',
            f"        xmin = {num(tier.xmin)}",
            f"        xmax = {num(tier.xmax)}",
            f"        intervals: size = {len(intervals)}",
        ]
        for j, (start, end, label) in enumerate(intervals, start=1):
            lines += [
                f"        intervals [{j}]:",
                f"            xmin = {num(start)}",
                f"            xmax = {num(end)}",
                f'            text = "{_quote(label)}"',
            ]
    return "\n".join(lines) + "\n"


def write_textgrid(
    path: str | Path,
    tiers: Sequence[TextGridTier],
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    decimals: int = 3,
) -> Path:
    """Write interval tiers to ``path`` as a long-format TextGrid.

        UTF-8 without a BOM and with ``
    `` line endings, which both Praat and
        :func:`parse_textgrid` read, and which keeps the file hash the same on
        Windows and POSIX.

        Args:
            path: Destination ``.TextGrid``; parent directories are created.
            tiers: Tiers to write; see :func:`format_textgrid`.
            xmin: Overall start, defaulting to the smallest tier ``xmin``.
            xmax: Overall end, defaulting to the largest tier ``xmax``.
            decimals: Decimal places for timestamps.

        Returns:
            The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = format_textgrid(tiers, xmin=xmin, xmax=xmax, decimals=decimals)
    out.write_text(text, encoding="utf-8", newline="\n")
    logger.debug("wrote %s (%d tier(s))", out, len(tiers))
    return out


def _pick_tier(tiers: Sequence[TextGridTier], *names: str) -> TextGridTier | None:
    """Return the first tier whose name matches, case-insensitively.

    MFA 3 names the word tier after the speaker rather than ``words``, so the
    caller falls back to tier order when this returns ``None``.
    """
    wanted = {n.lower() for n in names}
    for tier in tiers:
        if tier.name.strip().lower() in wanted:
            return tier
    return None


def _is_speech(label: str) -> bool:
    """Whether a words-tier label denotes a spoken word rather than silence."""
    return label.strip().lower() not in SILENCE_TOKENS


def _count_syllables(phones: Sequence[str]) -> int:
    """Count ARPAbet vowel nuclei, which equals the syllable count."""
    total = 0
    for phone in phones:
        base = re.sub(r"\d+$", "", phone.strip().upper())
        if base in ARPA_VOWELS:
            total += 1
    return total


def read_textgrid(
    path: str | Path,
    *,
    file_id: str | None = None,
    text: LabText | None = None,
    text_id: str | None = None,
    speaker_id: str = "",
    channel_id: str = "",
    aligner: str = "",
    acoustic_model: str = "",
    dictionary: str = "",
    audio_sha256: str = "",
    duration: float | None = None,
) -> Alignment:
    """Read an MFA TextGrid into an :class:`~rostrum.types.Alignment`.

    Words come from the ``words`` tier (first interval tier if MFA named it
    after the speaker); phones from the ``phones`` tier, assigned to the word
    whose interval contains their midpoint. Gaps are the silences *between*
    consecutive words - leading and trailing silence is excluded, since neither
    is a pause the speaker chose to make inside the text.

    Gap ``label`` is left at its default ``"none"``: the boundary/minor/none
    expectation comes from the display transcript's punctuation and is applied
    by :mod:`rostrum.text`, not here.

    Args:
        path: The ``.TextGrid`` to read.
        file_id: Identifier for the recording; defaults to the file stem.
        text: The :class:`LabText` used to build the ``.lab``. When given, each
            ``Word.display`` is filled from it and a token-count mismatch is an
            error - that mismatch means word indices no longer line up across
            renditions, which would invalidate every comparison.
        text_id: Library text this is a rendition of.
        speaker_id: Speaker identifier, for speaker normalisation.
        channel_id: Recording channel, for the channel rule (invariant 7).
        aligner: Tool and version, e.g. ``"mfa 3.2.1"``; see :func:`mfa_version`.
        acoustic_model: Acoustic model name used.
        dictionary: Dictionary name used.
        audio_sha256: Hash of the processed WAV these timings refer to.
        duration: WAV duration in seconds; defaults to the tier ``xmax``.

    Returns:
        An :class:`~rostrum.types.Alignment` with monotonic word times and the
        gaps between them.

    Raises:
        FileNotFoundError: The file does not exist.
        ValueError: No word intervals, non-monotonic or overlapping words, or a
            token-count mismatch against ``text``.
    """
    grid = Path(path)
    tiers = parse_textgrid(grid)

    word_tier = _pick_tier(tiers, "words", "word")
    phone_tier = _pick_tier(tiers, "phones", "phone")
    if word_tier is None:
        # MFA 3 writes "<speaker> - words"; otherwise take the first tier.
        named = [t for t in tiers if "word" in t.name.lower()]
        word_tier = named[0] if named else tiers[0]
    if phone_tier is None:
        named = [t for t in tiers if "phone" in t.name.lower()]
        phone_tier = named[0] if named else None

    spoken = [iv for iv in word_tier.intervals if _is_speech(iv[2])]
    if not spoken:
        raise ValueError(f"no word intervals in {grid}")

    phones_by_word: list[list[str]] = [[] for _ in spoken]
    if phone_tier is not None:
        cursor = 0
        for p_start, p_end, label in phone_tier.intervals:
            if not _is_speech(label):
                continue
            midpoint = 0.5 * (p_start + p_end)
            # Both tiers are time-ordered, so a single forward cursor suffices.
            while cursor < len(spoken) and spoken[cursor][1] <= midpoint:
                cursor += 1
            if cursor < len(spoken) and spoken[cursor][0] <= midpoint < spoken[cursor][1]:
                phones_by_word[cursor].append(label.strip())

    oov_words: list[str] = []
    words: list[Word] = []
    for idx, ((t_start, t_end, label), phones) in enumerate(
        zip(spoken, phones_by_word, strict=True)
    ):
        if t_end < t_start:
            raise ValueError(
                f"{grid}: word {idx} ({label!r}) ends before it starts: {t_start} > {t_end}"
            )
        if idx and t_start < words[-1].t_end - 1e-6:
            raise ValueError(
                f"{grid}: word {idx} ({label!r}) starts at {t_start} before word "
                f"{idx - 1} ends at {words[-1].t_end}"
            )
        token = label.strip()
        if token.lower() == UNKNOWN_TOKEN:
            oov_words.append(token)
        words.append(
            Word(
                idx=idx,
                text=token.lower(),
                t_start=t_start,
                t_end=t_end,
                phones=phones,
                n_syllables=_count_syllables(phones),
            )
        )

    if text is not None:
        if len(words) != len(text.lab_tokens):
            raise ValueError(
                f"{grid}: aligned {len(words)} words but the transcript has "
                f"{len(text.lab_tokens)} tokens; word indices would not match "
                "across renditions of this text"
            )
        for word in words:
            word.display = text.display_for(word.idx)

    gaps: list[Gap] = []
    for idx in range(len(words) - 1):
        gaps.append(
            Gap(
                idx=idx,
                prev_word_idx=idx,
                next_word_idx=idx + 1,
                t_start=words[idx].t_end,
                t_end=words[idx + 1].t_start,
            )
        )

    total = duration if duration is not None else word_tier.xmax
    return Alignment(
        file_id=file_id or grid.stem,
        text_id=text_id,
        speaker_id=speaker_id,
        channel_id=channel_id,
        words=words,
        gaps=gaps,
        duration=float(total),
        aligner=aligner,
        acoustic_model=acoustic_model,
        dictionary=dictionary,
        oov_words=oov_words,
        audio_sha256=audio_sha256,
    )
