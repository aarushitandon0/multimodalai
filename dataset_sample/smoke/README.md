# Forced-alignment smoke file

`scripts/smoke_align.py` aligns one real recording from this directory. It
synthesises nothing, so you have to supply the pair:

```
dataset_sample/smoke/<id>.wav    # the recording
dataset_sample/smoke/<id>.txt    # its verbatim transcript, UTF-8
```

Matching basenames, one pair. Then:

```bash
make smoke-align
# or
python scripts/smoke_align.py --keep
```

Requirements:

- **Licence.** Anything committed here needs a `LICENSES.csv` row:
  public-domain, CC0/CC BY, or a consented team recording (`CLAUDE.md`
  invariant 14). A quick self-recording of a public-domain text is the easiest
  option for a smoke test.
- **Transcript.** Verbatim - what was actually said, not the printed text. A
  mismatch shows up as out-of-vocabulary words or a word-count error, which is
  exactly what the script is meant to catch.
- **Audio.** 16 kHz mono WAV is preferred (invariant 1). Anything readable
  works for the smoke test - MFA resamples internally and the script warns
  rather than failing.
- **Length.** 10-30 s is plenty, and keeps the repository small.

The script prints the installed MFA version, the locally available acoustic and
dictionary model names, any OOV words, and the first ten aligned words with
their timestamps. It exits non-zero if alignment fails or the timestamps are not
monotonic.

`.wav` files in this directory are not git-ignored - they are meant to be
committed, so CI and anyone cloning the repository can run the same check.
