"""rostrum.features -- Acoustic feature extraction and speaker normalisation.

Every module here is pure: audio, an alignment and parameters in, frame arrays
and per-unit tables out. Nothing reads or writes files, and nothing compares
across recordings - that is the envelope's job, under the channel rule
(``configs/features.yaml``: ``channel_rule``, CLAUDE.md invariant 7).

All features share one frame grid, 10 ms hop and 25 ms window
(:class:`~rostrum.features.frames.FrameGrid`), and per-unit tables follow
``FEATURE_TABLE_CONTRACT`` in :mod:`rostrum.types`: indexed by ``unit_idx``,
with ``unit``, ``file_id`` and ``channel_id`` in ``attrs``.

The whole of spec 9.1 is implemented: pitch (F0, F0x), energy (EN), duration
(DUR), rate (RATE), pauses (PAU), prominence (PROM), spectral (SPEC), cepstral
(MFCC), voice quality (VQ) and eGeMAPS (EG).
:func:`rostrum.features.extract_all.extract_all` runs all of them on one
recording and returns the three per-unit tables plus the frame curves;
:func:`rostrum.features.extract_all.extract_all_cached` adds the Parquet cache.

The channel-sensitive blocks are SPEC, MFCC and VQ, and eGeMAPS as a whole.

Only :class:`~rostrum.features.frames.FrameGrid` is re-exported here. Importing
``extract_all`` from the package root would make the parent package depend on
every submodule, which turns any future submodule import into a cycle; callers
take it from :mod:`rostrum.features.extract_all` directly.
"""

from rostrum.features.frames import FrameGrid

__all__ = ["FrameGrid"]
