"""Shared plumbing for the component-validity checks (spec 14.1 "Components").

The three validators - :mod:`eval.validate_pitch`, :mod:`eval.validate_alignment`
and :mod:`eval.validate_rate` - each answer one question about a *component* of
the pipeline rather than about its output:

======================  ==================================================
Component               Question
======================  ==================================================
Pitch (F0, F0x)         How often do Praat and CREPE agree within 50 cents?
Alignment (MFA)         How far are MFA word boundaries from hand labels?
Rate (RATE)             Does the MFA syllable rate track de Jong & Wempe's?
======================  ==================================================

Each writes one section of ``results/components.json`` and its figures into
``docs/figures/``. Sections are merged rather than overwritten, so running one
validator does not discard another's numbers, and ``python -m eval.components``
runs all three.

**Pending is a result.** Two of the three depend on inputs that do not exist
until a human or MFA has produced them (hand-labelled boundaries, forced
alignments). Those sections are written with ``"status": "pending"`` and the
precise reason, so the technical report can state what is missing instead of
quietly omitting a row - and so nothing has to be remembered by hand later.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure

    from rostrum.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "SECTIONS",
    "STATUS_OK",
    "STATUS_PENDING",
    "TOLERANCE_EPSILON_MS",
    "describe",
    "figure_path",
    "new_figure",
    "percentile",
    "read_components",
    "save_figure",
    "summarise_errors_ms",
    "write_section",
]

SECTIONS: tuple[str, ...] = ("pitch", "alignment", "rate")
"""Section names in ``results/components.json``, in report order."""

STATUS_OK = "ok"
"""The section holds measured numbers."""

STATUS_PENDING = "pending"
"""The inputs the section needs do not exist yet; ``reason`` says which."""

TOLERANCE_EPSILON_MS = 1e-6
"""Slack when testing a tolerance band.

A boundary labelled exactly 20 ms early computes as 20.000000000000004 ms once
seconds have been subtracted and scaled, which would fall outside a strict
``<= 20`` test. Float noise should not decide whether a boundary passes, so the
band is widened by a microsecond - six orders of magnitude below anything a hand
label can resolve.
"""


# --------------------------------------------------------------------------- #
# results/components.json
# --------------------------------------------------------------------------- #


def read_components(cfg: Config) -> dict[str, Any]:
    """Read ``results/components.json``, or ``{}`` when it does not exist."""
    path = cfg.path("results.components")
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("%s is not valid JSON (%s); starting a fresh file", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def write_section(cfg: Config, name: str, payload: Mapping[str, Any]) -> Path:
    """Merge one section into ``results/components.json`` and return its path.

    The file records the code version and config hash alongside the sections, so
    a number in the report can be traced to the commit and the parameters that
    produced it. There is deliberately no timestamp: it would make every run
    produce a different file and break diffing.

    Absolute paths inside the payload are rewritten relative to the repository
    root for the same reason - a results file that names someone's home
    directory is not reproducible and leaks the build machine.

    Args:
        cfg: Loaded config.
        name: Section name; one of :data:`SECTIONS`.
        payload: The section's contents. Floats are rounded to four significant
            figures and non-finite values become ``null`` (spec 16).

    Returns:
        The path written.

    Raises:
        KeyError: ``name`` is not a known section.
    """
    from rostrum.config import code_version
    from rostrum.types import round_value

    if name not in SECTIONS:
        raise KeyError(f"unknown component section {name!r}; expected one of {SECTIONS}")

    data = read_components(cfg)
    plain = json.loads(json.dumps(payload, default=_jsonable))
    data[name] = round_value(_relativise(plain, cfg.root))
    data["meta"] = {
        "code_version": code_version(cfg.root),
        "config_hash": cfg.hash,
        "sections": [s for s in SECTIONS if s in data],
    }
    path = cfg.path("results.components")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("wrote %s section of %s", name, path)
    return path


def _relativise(value: Any, root: Path) -> Any:
    """Rewrite absolute paths under ``root`` as repository-relative, recursively.

    Both separators are normalised to ``/`` so the same run on Windows and on
    Linux writes the same string.
    """
    if isinstance(value, dict):
        return {k: _relativise(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_relativise(v, root) for v in value]
    if isinstance(value, str):
        prefix = str(root)
        out = value.replace(prefix + "\\", "").replace(prefix + "/", "")
        return out.replace("\\", "/") if out != value else value
    return value


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars and arrays on the way into JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[float] | np.ndarray, q: float) -> float:
    """The ``q``-th percentile of the finite values, or ``NaN`` if there are none."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


def describe(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """Count, mean, median, P95 and extremes of the finite values."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarise_errors_ms(errors_s: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """Boundary-error summary in milliseconds (spec 8, validation row).

    Args:
        errors_s: Signed errors in seconds (estimate minus truth).

    Returns:
        ``n``, ``mae_ms``, ``p95_ms``, ``bias_ms`` (the signed mean, which says
        whether the aligner runs early or late), and the percentage of
        boundaries within 20 ms and within 50 ms.
    """
    arr = np.asarray(errors_s, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mae_ms": float("nan"),
            "p95_ms": float("nan"),
            "bias_ms": float("nan"),
            "within_20ms_pct": float("nan"),
            "within_50ms_pct": float("nan"),
        }
    absolute_ms = np.abs(arr) * 1000.0
    return {
        "n": int(arr.size),
        "mae_ms": float(absolute_ms.mean()),
        "p95_ms": float(np.percentile(absolute_ms, 95)),
        "bias_ms": float(arr.mean() * 1000.0),
        "within_20ms_pct": float((absolute_ms <= 20.0 + TOLERANCE_EPSILON_MS).mean() * 100.0),
        "within_50ms_pct": float((absolute_ms <= 50.0 + TOLERANCE_EPSILON_MS).mean() * 100.0),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def figure_path(cfg: Config, name: str) -> Path:
    """Path of a figure under ``docs/figures``, creating the directory."""
    directory = cfg.path("docs.figures")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save_figure(figure: Figure, path: Path, *, dpi: int = 150) -> Path:
    """Save and close a figure, returning its path."""
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    figure.clf()
    import matplotlib.pyplot as plt

    plt.close(figure)
    logger.info("wrote %s", path)
    return path


def new_figure(*, size: tuple[float, float] = (7.0, 4.0)) -> tuple[Figure, Any]:
    """A single-axes figure on the non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=size)


# --------------------------------------------------------------------------- #
# Run them all
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """Run every component validator; returns the worst exit code."""
    import argparse

    parser = argparse.ArgumentParser(description="Run every component-validity check")
    parser.add_argument("-v", "--verbose", action="store_true")
    args, rest = parser.parse_known_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from eval import validate_alignment, validate_pitch, validate_rate

    codes = [
        validate_pitch.main(rest),
        validate_alignment.main(rest),
        validate_rate.main(rest),
    ]
    return max(codes)


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
