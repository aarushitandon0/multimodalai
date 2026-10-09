"""The global frame grid every feature is computed on (CLAUDE.md invariant 3).

Hop 10 ms, window 25 ms, everywhere, for every signal. Two conventions matter
downstream and are fixed here rather than per feature:

* frame ``i`` covers ``[i*hop, i*hop + window)``, so slicing a word out of a
  frame array needs no re-indexing (``configs/features.yaml``: ``center: false``);
* a frame belongs to the unit containing its **centre**, ``i*hop + window/2``.
  The 25 ms window is symmetric about that centre, so this is the only choice
  that does not systematically bias a word's features towards its onset.

External extractors (Praat, CREPE, Silero) each use their own internal grid.
They are mapped onto this one with :meth:`FrameGrid.resample_nearest`, which
takes the nearest source frame within half a hop and leaves ``NaN`` elsewhere -
never interpolating a value the extractor did not produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from rostrum.config import Config

__all__ = ["FrameGrid"]


@dataclass(frozen=True)
class FrameGrid:
    """The analysis grid: hop, window and sample rate.

    Attributes:
        hop_s: Frame step in seconds (0.010).
        window_s: Analysis window in seconds (0.025).
        sample_rate: Sample rate of the processed audio (16000).
    """

    hop_s: float = 0.010
    window_s: float = 0.025
    sample_rate: int = 16000

    def __post_init__(self) -> None:
        if self.hop_s <= 0:
            raise ValueError(f"hop must be positive, got {self.hop_s}")
        if self.window_s < self.hop_s:
            raise ValueError(f"window ({self.window_s}) must be at least the hop ({self.hop_s})")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @classmethod
    def from_config(cls, cfg: Config) -> FrameGrid:
        """Build the grid from ``configs/features.yaml``."""
        return cls(hop_s=cfg.hop_s, window_s=cfg.window_s, sample_rate=cfg.sample_rate)

    # -- geometry ---------------------------------------------------------- #

    @property
    def hop_samples(self) -> int:
        """Hop in samples."""
        return round(self.hop_s * self.sample_rate)

    @property
    def window_samples(self) -> int:
        """Window length in samples."""
        return round(self.window_s * self.sample_rate)

    def n_frames(self, duration_s: float) -> int:
        """Number of whole windows that fit in ``duration_s`` seconds."""
        if duration_s < self.window_s:
            return 0
        return int(np.floor((duration_s - self.window_s) / self.hop_s + 1e-9)) + 1

    def n_frames_for(self, n_samples: int) -> int:
        """Number of whole windows that fit in ``n_samples`` samples."""
        return self.n_frames(n_samples / float(self.sample_rate))

    def starts(self, n_frames: int) -> np.ndarray:
        """Start time of each frame, seconds."""
        return np.arange(int(n_frames), dtype=np.float64) * self.hop_s

    def centres(self, n_frames: int) -> np.ndarray:
        """Centre time of each frame, seconds."""
        return self.starts(n_frames) + self.window_s / 2.0

    def frame_indices(self, t_start: float, t_end: float, n_frames: int) -> np.ndarray:
        """Indices of the frames whose centre lies in ``[t_start, t_end)``.

        Args:
            t_start: Span onset in seconds.
            t_end: Span offset in seconds.
            n_frames: Length of the frame arrays being indexed.

        Returns:
            Ascending integer indices; empty when the span holds no centre.
        """
        if n_frames <= 0 or t_end <= t_start:
            return np.empty(0, dtype=np.intp)
        centres = self.centres(n_frames)
        return np.flatnonzero((centres >= t_start) & (centres < t_end))

    # -- mapping other extractors onto the grid ---------------------------- #

    def resample_nearest(
        self,
        src_times: np.ndarray,
        values: np.ndarray,
        n_frames: int,
        *,
        tolerance_s: float | None = None,
    ) -> np.ndarray:
        """Map ``values`` sampled at ``src_times`` onto this grid's centres.

        Nearest neighbour, never interpolation: a frame with no source estimate
        within ``tolerance_s`` stays ``NaN`` so statistics cannot be built on
        invented values.

        Args:
            src_times: Source frame times, ascending, seconds.
            values: Source values, same length as ``src_times``.
            n_frames: Number of output frames.
            tolerance_s: Maximum distance to accept; defaults to half a hop.

        Returns:
            ``float64`` array of length ``n_frames``.

        Raises:
            ValueError: ``src_times`` and ``values`` differ in length.
        """
        src_times = np.asarray(src_times, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if src_times.shape != values.shape:
            raise ValueError(
                f"src_times and values must be the same length, got "
                f"{src_times.shape} and {values.shape}"
            )
        out = np.full(int(n_frames), np.nan, dtype=np.float64)
        if out.size == 0 or src_times.size == 0:
            return out

        tol = self.hop_s / 2.0 if tolerance_s is None else float(tolerance_s)
        centres = self.centres(out.size)
        nearest = np.searchsorted(src_times, centres)
        nearest = np.clip(nearest, 1, src_times.size - 1) if src_times.size > 1 else nearest * 0
        if src_times.size > 1:
            left = src_times[nearest - 1]
            right = src_times[nearest]
            pick = np.where(np.abs(centres - left) <= np.abs(centres - right), nearest - 1, nearest)
        else:
            pick = np.zeros(out.size, dtype=np.intp)

        close = np.abs(centres - src_times[pick]) <= tol + 1e-12
        out[close] = values[pick[close]]
        return out
