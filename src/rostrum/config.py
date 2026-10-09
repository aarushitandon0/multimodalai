"""Configuration loading and determinism control.

Every tunable number in Rostrum lives in ``configs/*.yaml``; modules read them
through the frozen :class:`Config` returned by :func:`load_config`. Nothing in
``src/rostrum`` may hard-code a threshold (``CLAUDE.md``, coding conventions).

The loaded object is deep-frozen: attempting to mutate a nested mapping raises
``TypeError``. That matters because the same :class:`Config` is shared by the
pipeline, the evaluation harness and the dashboard inside one process, and a
stray write would silently change results between folds.

:func:`set_seeds` pins every RNG we touch. ``PYTHONHASHSEED`` cannot be set
from inside a running interpreter, so it is exported by the ``Makefile`` and
only *verified* here - see :func:`check_pythonhashseed`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

__all__ = [
    "CONFIG_FILES",
    "Config",
    "FrozenDict",
    "check_pythonhashseed",
    "code_version",
    "config_hash",
    "load_config",
    "repo_root",
    "set_seeds",
]

logger = logging.getLogger(__name__)

CONFIG_FILES: tuple[str, ...] = (
    "paths",
    "injection",
    "features",
    "detection",
    "scoring",
    "templates",
)
"""Config sections, each loaded from ``configs/<name>.yaml``."""

_DEFAULT_SEED = 1337


# --------------------------------------------------------------------------- #
# Immutable mapping
# --------------------------------------------------------------------------- #


class FrozenDict(Mapping[str, Any]):
    """A read-only, dotted-access mapping.

    Supports both ``cfg.features["frame"]["hop_s"]`` and
    ``cfg.features.frame.hop_s``. Nested dicts and lists are frozen on
    construction, so a :class:`Config` cannot be edited after loading.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(
            self, "_data", MappingProxyType({k: _freeze(v) for k, v in data.items()})
        )

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - attribute errors are rare
            raise AttributeError(f"no config key {key!r}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        raise TypeError("Config is frozen; change configs/*.yaml instead")

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, mutable deep copy."""
        return _thaw(self)


def _freeze(value: Any) -> Any:
    """Recursively convert mappings to :class:`FrozenDict` and lists to tuples."""
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`, for JSON dumping and hashing."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Repository location
# --------------------------------------------------------------------------- #


def repo_root(start: Path | None = None) -> Path:
    """Return the repository root: the nearest ancestor holding ``configs/``.

    Walking up from this file keeps the lookup working whether Rostrum is run
    from the root, from ``app/`` via Streamlit, or from an installed package in
    a Docker image.
    """
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "configs").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    # Installed without the source tree alongside: fall back to the CWD.
    return Path.cwd()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Config:
    """The resolved, frozen configuration for one run.

    Attributes:
        root: Repository root; every relative path in ``paths`` resolves here.
        paths: ``configs/paths.yaml``.
        injection: ``configs/injection.yaml``.
        features: ``configs/features.yaml``.
        detection: ``configs/detection.yaml``.
        scoring: ``configs/scoring.yaml``.
        templates: ``configs/templates.yaml``.
        seed: Master seed, taken from ``injection.yaml``.
        hash: SHA-256 over all config sections; recorded in every
            :class:`~rostrum.types.AnalysisResult` so a report names the exact
            parameters that produced it.
    """

    root: Path
    paths: FrozenDict
    injection: FrozenDict
    features: FrozenDict
    detection: FrozenDict
    scoring: FrozenDict
    templates: FrozenDict
    seed: int = _DEFAULT_SEED
    hash: str = ""
    _overrides: FrozenDict = field(default_factory=lambda: FrozenDict({}), repr=False)

    # -- path helpers -------------------------------------------------------

    def path(self, dotted: str) -> Path:
        """Resolve a dotted key from ``paths.yaml`` to an absolute path.

        Example:
            >>> cfg.path("results.metrics")  # doctest: +SKIP
            PosixPath('/repo/results/metrics.json')
        """
        node: Any = self.paths
        for part in dotted.split("."):
            node = node[part]
        if not isinstance(node, str):
            raise TypeError(f"paths.{dotted} is not a path string: {node!r}")
        p = Path(node)
        return p if p.is_absolute() else (self.root / p)

    def ensure_dir(self, dotted: str) -> Path:
        """Resolve a dotted path key and create the directory if needed."""
        p = self.path(dotted)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- frequently used scalars -------------------------------------------

    @property
    def hop_s(self) -> float:
        """Frame hop in seconds (10 ms; CLAUDE.md invariant 3)."""
        return float(self.features["frame"]["hop_s"])

    @property
    def window_s(self) -> float:
        """Analysis window in seconds (25 ms; CLAUDE.md invariant 3)."""
        return float(self.features["frame"]["window_s"])

    @property
    def sample_rate(self) -> int:
        """Working sample rate (16 kHz; CLAUDE.md invariant 1)."""
        return int(self.features["frame"]["sample_rate"])

    def section(self, name: str) -> FrozenDict:
        """Return one config section by name."""
        if name not in CONFIG_FILES:
            raise KeyError(f"unknown config section {name!r}; expected one of {CONFIG_FILES}")
        return getattr(self, name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of every section."""
        return {name: _thaw(getattr(self, name)) for name in CONFIG_FILES}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file, returning ``{}`` for an empty document."""
    if not path.is_file():
        raise FileNotFoundError(f"missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"{path} must contain a mapping at the top level, got {type(data).__name__}"
        )
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = _thaw(value)
    return out


def config_hash(sections: Mapping[str, Any]) -> str:
    """Return a SHA-256 over config sections, stable across runs.

    Keys are sorted and the JSON is compact, so the digest depends on the
    values alone - not on YAML formatting or dict insertion order.
    """
    blob = json.dumps(_thaw(sections), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def code_version(root: Path | None = None) -> str:
    """Return ``git describe --always --dirty``, or ``"unknown"``.

    Recorded in every result so a metric can be traced to the commit that
    produced it. A missing git checkout is not an error - Docker images and
    source archives have none.
    """
    cwd = root or repo_root()
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--tags"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def load_config(
    config_dir: Path | str | None = None,
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    seed: int | None = None,
    apply_seeds: bool = True,
) -> Config:
    """Load ``configs/*.yaml`` into a frozen :class:`Config`.

    Args:
        config_dir: Directory holding the YAML files. Defaults to
            ``<repo_root>/configs``, or ``$ROSTRUM_CONFIG_DIR`` when set.
        overrides: Per-section overrides, deep-merged over the files, e.g.
            ``{"detection": {"regions": {"tau_in": 3.0}}}``. Used by the
            threshold search and the ablation runner so that neither has to
            write to the YAML files.
        seed: Master seed, overriding ``injection.yaml``.
        apply_seeds: Call :func:`set_seeds` after loading. Leave on except in
            tests that assert on seeding behaviour themselves.

    Returns:
        A frozen :class:`Config`.

    Raises:
        FileNotFoundError: A config section is missing.
        TypeError: A section is not a top-level mapping.
    """
    root = repo_root()
    if config_dir is None:
        env_dir = os.environ.get("ROSTRUM_CONFIG_DIR")
        directory = Path(env_dir) if env_dir else root / "configs"
    else:
        directory = Path(config_dir)
    directory = directory.resolve()

    overrides = overrides or {}
    unknown = set(overrides) - set(CONFIG_FILES)
    if unknown:
        raise KeyError(f"override for unknown config section(s): {sorted(unknown)}")

    sections: dict[str, dict[str, Any]] = {}
    for name in CONFIG_FILES:
        data = _read_yaml(directory / f"{name}.yaml")
        if name in overrides:
            data = _deep_merge(data, overrides[name])
        sections[name] = data

    resolved_seed = int(
        seed if seed is not None else sections["injection"].get("seed", _DEFAULT_SEED)
    )
    # Keep the recorded seed and the hashed configs consistent with the
    # override, so two runs with different seeds get different config hashes.
    sections["injection"]["seed"] = resolved_seed

    cfg = Config(
        root=root,
        paths=FrozenDict(sections["paths"]),
        injection=FrozenDict(sections["injection"]),
        features=FrozenDict(sections["features"]),
        detection=FrozenDict(sections["detection"]),
        scoring=FrozenDict(sections["scoring"]),
        templates=FrozenDict(sections["templates"]),
        seed=resolved_seed,
        hash=config_hash(sections),
        _overrides=FrozenDict(dict(overrides)),
    )

    if apply_seeds:
        set_seeds(cfg.seed)
    logger.debug("loaded config from %s (hash=%s)", directory, cfg.hash[:12])
    return cfg


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def check_pythonhashseed(strict: bool = False) -> bool:
    """Return whether ``PYTHONHASHSEED=0`` is in effect.

    Python fixes its hash seed before any user code runs, so this cannot be set
    from here - the ``Makefile`` and ``Dockerfile`` export it. Any code path
    whose output depends on ``hash()`` of a string (set or dict ordering that
    leaks into a result) is only reproducible when this returns ``True``.

    Args:
        strict: Raise instead of warning when the seed is not pinned.

    Raises:
        RuntimeError: ``strict`` is set and ``PYTHONHASHSEED`` is not ``"0"``.
    """
    value = os.environ.get("PYTHONHASHSEED")
    ok = value == "0"
    if not ok:
        msg = (
            f"PYTHONHASHSEED={value!r}, expected '0'. Run through the Makefile "
            "targets, or export PYTHONHASHSEED=0 before starting Python."
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
    return ok


def set_seeds(seed: int = _DEFAULT_SEED, *, deterministic_torch: bool = True) -> None:
    """Seed every RNG Rostrum can reach.

    Covers :mod:`random`, NumPy's legacy global RNG and, when installed, torch
    (CPU and CUDA) plus its deterministic algorithm and cuDNN flags. Torch is
    imported lazily so config loading stays fast for scripts that never touch a
    model.

    Modules that need their own stream should derive it from this seed
    (``numpy.random.default_rng(cfg.seed + offset)``) rather than relying on
    the global state, so execution order cannot change their draws.

    Args:
        seed: The seed to apply everywhere.
        deterministic_torch: Call ``torch.use_deterministic_algorithms`` (spec
            16). Left on for every pipeline run; turn it off only to measure
            what the determinism constraint costs.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        logger.warning("numpy not installed; its RNG was not seeded")

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")
    else:
        torch.manual_seed(seed)
        if deterministic_torch:
            # spec 16. warn_only keeps this from killing a run when an op has no
            # deterministic kernel; the warning tells us which op to replace.
            torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():  # pragma: no cover - CPU-only in CI
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.debug("seeded random/numpy/torch with %d", seed)
