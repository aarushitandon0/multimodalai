# Handoff — night of 2026-10-13/14

Written for: you, tomorrow morning, picking this up cold.

## The headline

The blocker is gone. MFA runs in Docker, the clean corpus is force-aligned, and
the full feature cache is built. **Reference mode is now reachable** on four of
five texts. What is *not* done is the real `results/metrics.json` — the eval was
~12 min into a ~60 min run when the laptop had to close.

Nothing is committed. Review the diff before making it permanent.

## Resume in one command

Docker Desktop must be running first (it was stopped; that was the original blocker).

```bash
docker run --rm \
  -v "$PWD/src:/app/src" -v "$PWD/configs:/app/configs" \
  -v "$PWD/scripts:/app/scripts" -v "$PWD/dataset:/app/dataset" \
  -v "$PWD/eval:/app/eval" -v "$PWD/results:/app/results" \
  -v "$PWD/.cache:/app/.cache" \
  rostrum:latest bash -lc '.venv/bin/python -u -m eval.run_loto --corpus dataset --out results/metrics.json'
```

Budget **~60 minutes**. It will NOT recompute features — the cache is warm, which
is the whole reason this is now a one-command job. Measured: ~12 min per fold,
5 folds. Add `-v` for live logging; without `-u`/`-v` it prints nothing until the end,
which is how 28 minutes got burned invisibly last night.

Faster if you are short on time: `--no-tune` skips the per-fold grid search and
uses the config thresholds. Still real numbers on the real corpus, but the
thresholds are no longer fit per fold, so say so if you report it.

## What is preserved on disk (survives reboot)

| Artifact | State |
|---|---|
| `dataset/alignments/*.TextGrid` | **60 files**, MFA 3.2.1, 0 OOV on every one |
| `dataset/qc/align_report.csv` | 60 reported, **0 flagged**, unaligned fraction 0.000 |
| `.cache/features/` | **785 entries** — full feature set, all 326 files |
| `rostrum:latest` Docker image | built, MFA + models baked in, verified |

## What changed in the tree

* `README.md` — new §13.3 (alignment results), new Docker subsection in §14.3,
  feature-cost and sharding notes in §15.3, corrected `make envelope`
  description, §13.x subsections renumbered.
* `scripts/build_features.py` — **bug fix**, see below.
* `dataset/alignments/*.TextGrid` — 51 new, 9 rewritten byte-identically.

## The bug worth knowing about

`build_features.py` hardcoded `alignments/<file_id>.TextGrid`. Only clean files go
through MFA; the 266 injected files carry alignments derived from injection time
maps, serialised as `.json` — which `dataset/README` already documented. So **every
injected file was silently skipped**: the run processed 72 of 328 and exited 0.
No error, no warning, just an empty test set.

Fixed with a `_alignment_path()` helper (prefer TextGrid, fall back to JSON) plus a
suffix dispatch at the load site using the existing `Alignment.from_dict`. Clean
files are untouched.

This is also why the repo only ever had synthetic metrics: two blockers were
stacked — no alignments, and no injected features — and the first hid the second.

## Still open

1. **Real metrics** — the eval above. Highest value remaining.
2. **`make classifier`** — `clf_all.joblib` is still missing, so dashboard regions
   carry no severity. `run_loto` fits its own per-fold classifier and does not
   need it; this is for the live dashboard only.
3. **`s7_lv09` / `s7_lv10`** — unaligned. Each read 115 words against a 114-token
   transcript. Fix: `dataset/texts/overrides/<file_id>.txt`. s7 keeps 12
   references without them, so this is cosmetic.
4. **`s3_fdr_infamy`** — one clean reading, so it can never form an envelope. It is
   structurally a Norm-mode text. Say this out loud rather than letting a judge find it.
5. **No human-recorded flawed takes** — 0 in the corpus. `mumbling` has no test
   data and the human-vs-synthetic gap stays unmeasurable. Already N/A in metrics.md.
6. **Test suite** — not verified end to end. The README claims 1,366 tests; a run
   was 13% in and clean when it was stopped. Earlier, 9 failures traced to the
   *stale image*, not the code (host files are self-consistent).
7. **Image is stale** — baked `/app` is from Oct 11 and lacks the current
   `dataset_sample`, so `make docker-verify` / `docker-reproduce` fail. Everything
   above used bind-mounts over the stale copies. A rebuild fixes it; the build
   cache was pruned, so budget ~20-40 min, and C: is at 96% full.

## Reference counts, which decide the operating mode

| Text | Aligned references | Mode |
|---|---|---|
| `s1_gettysburg` | 16 | Reference |
| `s5_im_nobody` | 17 | Reference |
| `s6_sunset_lampman` | 14 | Reference |
| `s7_sonnet18` | 12 | Reference |
| `s3_fdr_infamy` | 1 | Norm (structural) |

`min_references` is 4 (mode gate) and 3 (envelope floor). The 2-reference fallback
you were prepared to defend on camera is **not needed**.

## For the video

Lead with the dataset — it is your strongest card, it is 30% of the weight, and it
works today. Then injection and label provenance (time map → timestamps → re-align
QC → UTMOS rejections; `failures.json` reads as rigour). Then the dashboard on a
known-flawed file.

On metrics, be straight: if the eval has not been rerun, they are from generated
feature tables and `results/metrics.md` says so in its first line. Say the word
"synthetic". The alignment numbers in §13.3 *are* real and measured, so lead with
those for the grounding claim.
