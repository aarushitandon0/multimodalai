"""Template loading, strict rendering and number formatting (spec 12).

``configs/templates.yaml`` holds every sentence a card can print. This module
is the only place that turns one of those strings into text, and it does so
under two rules:

* **Strict.** A placeholder with no value raises :class:`TemplateError` instead
  of rendering ``{median}`` to the user. A card that cannot state its number is
  a bug, not a cosmetic problem, so it fails loudly in the tests rather than
  quietly in the dashboard.
* **No LLM.** Rendering is :mod:`string` formatting over values DSP computed
  (CLAUDE.md invariant 12). The optional ``--polish`` rephrasing of spec 12 is
  deliberately not implemented here.

Numbers are formatted by what they *are*, not by how large they happen to be:
seconds to 2 decimals, rates and semitones and z-scores to 1, taken from
``templates.formatting`` so the choice lives in the configuration. The decimals
per feature come from its display unit in
:data:`rostrum.features.names.DISPLAY_UNITS`, so a new feature inherits the
right precision from its unit without touching this module.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from string import Formatter
from typing import TYPE_CHECKING, Any

from rostrum.features import names as N

if TYPE_CHECKING:  # pragma: no cover - imported only for the annotations
    from rostrum.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "MISSING",
    "Formatting",
    "TemplateError",
    "Templates",
    "collapse",
]

MISSING: str = "n/a"
"""Printed for a quantity that genuinely does not exist (an unmeasured sigma)."""

_WHITESPACE = re.compile(r"\s+")
_ORPHAN = re.compile(r"\s+([,.;:)])")
_FORMATTER = Formatter()


class TemplateError(KeyError):
    """A template asked for a field the caller did not supply."""


def collapse(text: str) -> str:
    """Tidy a rendered sentence.

    Collapses the newlines a folded YAML block leaves, and closes up the space
    a placeholder leaves in front of punctuation when it renders empty - a
    feature with no unit would otherwise print "0.23 , n = 6".
    """
    return _ORPHAN.sub(r"\1", _WHITESPACE.sub(" ", str(text))).strip()


@dataclass(frozen=True)
class Formatting:
    """How many decimals each kind of quantity is printed to.

    Attributes:
        time_decimals: Seconds in a headline or a timestamp.
        z_decimals: Standard deviations.
        probability_decimals: Probabilities and confidences.
        unit_decimals: Decimals per display unit, e.g. ``{"syll/s": 1}``.
        default_decimals: Fallback for a unit not listed.
    """

    time_decimals: int = 2
    z_decimals: int = 1
    probability_decimals: int = 2
    unit_decimals: Mapping[str, int] = field(default_factory=dict)
    default_decimals: int = 2

    @classmethod
    def from_config(cls, config: Config | None) -> Formatting:
        """Read ``templates.formatting``; code defaults without a config."""
        if config is None:
            return cls()
        block = dict(config.templates.get("formatting", {}))
        return cls(
            time_decimals=int(block.get("time_decimals", 2)),
            z_decimals=int(block.get("z_decimals", 1)),
            probability_decimals=int(block.get("probability_decimals", 2)),
            unit_decimals={str(k): int(v) for k, v in dict(block.get("unit_decimals", {})).items()},
            default_decimals=int(block.get("default_decimals", 2)),
        )

    # -- scalar formatters --------------------------------------------------

    def decimals_for(self, unit: str) -> int:
        """Decimals for a display unit."""
        return int(self.unit_decimals.get(str(unit), self.default_decimals))

    def number(self, value: Any, decimals: int) -> str:
        """A finite number to ``decimals`` places, or :data:`MISSING`."""
        try:
            x = float(value)
        except (TypeError, ValueError):
            return MISSING
        if not math.isfinite(x):
            return MISSING
        return f"{x:.{int(decimals)}f}"

    def time(self, value: Any) -> str:
        """A timestamp or duration in seconds."""
        return self.number(value, self.time_decimals)

    def z(self, value: Any) -> str:
        """A standard-deviation count, signed."""
        return self.number(value, self.z_decimals)

    def probability(self, value: Any) -> str:
        """A probability or a confidence in ``[0, 1]``."""
        return self.number(value, self.probability_decimals)

    def value(self, value: Any, unit_or_feature: str = "") -> str:
        """A measured value, to the precision its unit deserves.

        Args:
            value: The number.
            unit_or_feature: A display unit (``"syll/s"``) or a feature name
                (``"rate_local"``), whose unit is then looked up.
        """
        unit = str(unit_or_feature)
        if unit not in self.unit_decimals:
            # Not a unit we know: treat it as a feature name and look its unit
            # up, which yields "" (and so the default decimals) for neither.
            unit = N.display_unit(unit)
        return self.number(value, self.decimals_for(unit))


@dataclass(frozen=True)
class Templates:
    """The sentences of ``configs/templates.yaml``, rendered strictly.

    Attributes:
        data: The parsed template section.
        formatting: Number formatting rules.
    """

    data: Mapping[str, Any] = field(default_factory=dict)
    formatting: Formatting = field(default_factory=Formatting)

    @classmethod
    def from_config(cls, config: Config | None) -> Templates:
        """Build from a loaded configuration."""
        if config is None:
            return cls()
        return cls(data=dict(config.templates), formatting=Formatting.from_config(config))

    # -- lookup -------------------------------------------------------------

    def raw(self, dotted: str, default: str | None = None) -> str | None:
        """The unrendered template at a dotted key, or ``default``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        if node is None:
            return default
        return str(node)

    def has(self, dotted: str) -> bool:
        """Whether a non-empty template exists at a dotted key."""
        return bool(collapse(self.raw(dotted, "") or ""))

    # -- rendering ----------------------------------------------------------

    def render(self, template: str, fields: Mapping[str, Any], *, where: str = "") -> str:
        """Fill ``template`` from ``fields``.

        Args:
            template: The template string, as written in the YAML.
            fields: Values per placeholder. Numbers must already be formatted;
                :func:`region_fields` in :mod:`rostrum.explain.cards` does that.
            where: Dotted key of the template, for the error message.

        Returns:
            The rendered sentence, whitespace collapsed.

        Raises:
            TemplateError: A placeholder has no value in ``fields``.
        """
        text = collapse(template)
        if not text:
            return ""
        wanted = {name for _, name, _, _ in _FORMATTER.parse(text) if name}
        missing = sorted(name for name in wanted if name not in fields)
        if missing:
            raise TemplateError(
                f"template {where or text[:40]!r} needs field(s) {missing}; "
                f"supplied {sorted(fields)}"
            )
        return collapse(text.format(**dict(fields)))

    def render_key(
        self,
        dotted: str,
        fields: Mapping[str, Any],
        *,
        required: bool = True,
    ) -> str:
        """Render the template at a dotted key.

        Args:
            dotted: Key into ``templates.yaml``, e.g. ``"rationale.rushing"``.
            fields: Values per placeholder.
            required: Raise when the key is missing or empty. With ``False`` an
                absent template renders as an empty string, which is how the
                optional clauses (``context.none``) are switched off.

        Returns:
            The rendered sentence.

        Raises:
            TemplateError: The key is missing and ``required`` is set, or a
                placeholder has no value.
        """
        template = self.raw(dotted)
        if template is None or not collapse(template):
            if required:
                raise TemplateError(f"configs/templates.yaml has no template at {dotted!r}")
            return ""
        return self.render(template, fields, where=dotted)
