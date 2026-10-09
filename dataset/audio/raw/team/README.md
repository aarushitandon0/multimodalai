# Team recordings (bronze tier)

Raw, unprocessed team takes live here, flat, one file per take. Nothing is
downloaded into this folder: files arrive by hand from whoever recorded them,
and `scripts/build_clean.py` turns the clean ones into
`dataset/audio/clean/<file_id>.wav`.

**Before recording anyone:** their row in [`../../../CONSENT.md`](../../../CONSENT.md)
must be filled in and signed. No consent line, no file in this folder.

## Recording setup (spec 4.4)

| Item | Requirement |
|---|---|
| Room | Quiet, soft furnishings, no fan/AC, door closed. Same room for all of one speaker's takes. |
| Mic | Same phone or mic for **all** of one speaker's takes, 30 cm from the mouth, slightly off-axis. |
| Format | WAV if the recorder offers it, otherwise the highest-quality m4a. 44.1 kHz or better; do not resample by hand. |
| Room tone | Start every file with **3 s of silence** before speaking, and leave ~1 s after the last word. The injection engine loops that silence as room tone (`CLAUDE.md` invariant 8). |
| Level | Speak so the loudest words peak around −12 dBFS; never let the recorder clip. Loudness is fixed later (−23 LUFS), clipping is not. |
| Don't | No noise suppression, no "voice enhancement", no auto-gain, no compression preset, no headphones monitoring with latency. These change the energy contour the system measures. |

Read the text from `dataset/texts/<text_id>.json` (`display`). For S2 and S3,
read only the excerpt named by `excerpt_start_s`/`excerpt_end_s` in
`../../../SOURCES.yaml`.

## Takes per speaker per text

| Take | `origin` | `file_id` |
|---|---|---|
| Clean reading, best effort | `clean` | `tm01_s1_clean` |
| Nervous rush through the middle third | `human` | `tm01_s1_rushing` |
| Monotone reading of one paragraph | `human` | `tm01_s1_monotone` |
| Mumbled + low energy, with 2–3 hesitation pauses mid-sentence | `human` | `tm01_s1_mumbling` |

The clean take is a bronze envelope member; the three flawed takes are the
**test-only** human set (spec 6.1, `CLAUDE.md` invariant 11). Record the clean
take first, in one sitting with the flawed ones, so the channel matches.

`file_id` convention: `<speaker>_<text>_<take>`, with `<text>` one of `s1`, `s2`,
`s3` and `<speaker>` the `speaker_id` from `CONSENT.md` (`tm01`, `tm02`, …).

## After recording

1. Copy the files here as `dataset/audio/raw/team/<file_id>.<ext>`.
2. Add a `files:` entry per clean take under `team_s1` / `team_s2` / `team_s3` in
   `../../../SOURCES.yaml` (the template is in the comment above those blocks).
3. Build and register them:

   ```bash
   python scripts/build_clean.py --only team_s1 team_s2 team_s3
   python scripts/check_provenance.py
   ```

4. Note the device and room in the take's `notes:` field — the channel rule
   needs to be auditable from the manifest alone.

The flawed takes are **not** declared in `SOURCES.yaml`: they are registered with
`origin=human` by the annotation import step (spec 6.2), which also carries their
Praat `flaws` tier. They are still covered by `CONSENT.md`.
