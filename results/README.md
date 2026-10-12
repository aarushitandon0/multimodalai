# results/

Generated, not hand-written. Everything here is rewritten by a `make` target
and nothing in it is committed (`results/*` is in `.gitignore`), so treat a
file's content as evidence of the last run, not of the current code.

## What is in here now (2026-10-13)

| File | Written by | Corpus |
|---|---|---|
| `metrics.json`, `metrics.md` | `make eval` | **synthetic** |
| `metrics_synthetic.json`, `metrics_synthetic.md` | `make eval EVAL_ARGS="--corpus synthetic"` | synthetic |
| `ablations.json`, `ablations.md` | `make ablations` | synthetic |
| `fold_syn_a/b/c.json` | `make eval` | synthetic |
| `failures.json`, `failures.md` | `eval.failures` | synthetic |
| `detect_synthetic.json`, `tuning.json`, `components.json` | component reports | synthetic |
| `cards_demo.json`, `rubric_demo.json` | `scripts/cards_demo.py`, `scripts/score_report.py` | demo files |

`metrics.json` and `metrics_synthetic.json` are **byte-identical**: the current
`metrics.json` is a copy of a `--corpus synthetic` run, not the dataset result
its name suggests. Both files say so internally — `corpus: synthetic`, and
`metrics.md` carries the spec 14.1 caveat — but read the `corpus` field before
quoting any number from here.

**No number in this directory was measured on the 328-file contrastive
dataset.** The targets in `README.md` section 13 are not met or missed yet;
they are untested.

## Why the dataset evaluation has not run

`make eval` already defaults to `--corpus dataset`, so this is not a missing
command. It is blocked upstream:

1. **Only 9 of 62 clean readings are aligned** (the injection parents). The
   per-text envelope is a median and MAD *per word* over the readings of that
   text, and the confidence band is leave-one-reference-out, so a text needs at
   least two aligned readings to produce a band at all. `s3_fdr_infamy` has
   exactly one. See `dataset/texts/README.md`.
2. **Invariant-9 QC has not run**, so no injected label is verified. Scoring
   against unverified ground truth would report a number that cannot be
   defended. See `dataset/qc/README.md`.

Both need the MFA conda environment, so both run in the Docker image rather
than on a Windows host. The order is: align the remaining 53 clean readings →
`make qc` → `make features` → `make envelope` → `make eval`.
