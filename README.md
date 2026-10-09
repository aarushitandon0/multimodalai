# Rostrum

**Contrastive speech analytics and temporal flaw grounding**
Multimodal AI Hackathon 2026 · Track C

Rostrum evaluates *how* a speech is delivered, not what it says. Given a
recording and its transcript, it forced-aligns every word, compares the
delivery against an **envelope of skilled readings of the same text**, grounds
each deviation to a **start/end timestamp**, and explains it with the numbers
that caused it:

> **[41.21–45.88 s] Pace — rushing (L3).** On *"we can not dedicate…"* you spoke at
> **6.4 syll/s** vs **3.9 ± 0.6** for 15 skilled readers (**z = +4.1**), and the
> boundary pause before it dropped from **0.82 s → 0.06 s** (z = −3.6).
> **Fix:** slow to ~4 syll/s and hold the pause.

No LLM sits in the judging loop. DSP computes every number and every timestamp.

---

## Status

Bootstrapped — repository skeleton, configuration, data model and CI are in
place. The pipeline phases are stubbed in the `Makefile` and marked `TODO`.

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo, configs, types, CI | ✅ done |
| 1–2 | Contrastive dataset (sourcing → injection → QC → labels) | ⬜ `make data` |
| 3 | Forced alignment (MFA) | ⬜ |
| 4 | Features + envelopes | ⬜ `make features` `make envelope` |
| 5–6 | Grounding, scoring, explanations | ⬜ |
| 7 | Evaluation + ablations | ⬜ `make eval` `make ablations` |
| 8 | Dashboard | ⬜ `make app` |

---

## Quickstart

Requires **Python 3.11**, **conda** (for the aligner) and **ffmpeg** on `PATH`.

```bash
make setup        # pip venv in .venv, pinned deps, editable install
make setup-mfa    # separate `mfa` conda env + acoustic model & dictionary
make test         # pytest
make lint         # ruff + black --check
```

`make help` lists every target. `make check-env` verifies the install and warns
about a missing `ffmpeg` or `mfa` environment.

### Two environments, on purpose

| Environment | Holds | Created by |
|---|---|---|
| `.venv` (pip) | everything except alignment | `make setup` |
| `mfa` (conda) | Montreal Forced Aligner only | `make setup-mfa` |

MFA pins Kaldi, OpenFst, pynini and an older numeric stack through
conda-forge; installing it beside `librosa`/`torch`/`streamlit` breaks one side
or the other. Rostrum shells out with `conda run -n mfa mfa align …`.

> `torchaudio.functional.forced_align` is **not** an option — it was removed in
> torchaudio 2.9. MFA is the only aligner, for both dataset labels and runtime.
> `faster-whisper` is used solely for the ASR-vs-transcript WER check.

---

## How it works

```
audio + transcript
   │
   ├─ io/        16 kHz mono · −23 LUFS · peak ≤ −1 dBFS · sha256
   ├─ data/      MANIFEST / LICENSES handling, provenance checks
   ├─ align/     MFA word & phone intervals   (+ WER check → reject mode)
   ├─ text/      gap labels: boundary / minor / none; anaphora detection
   ├─ features/  F0 (two-pass Praat, semitones) · energy · rate · pauses
   │             prominence (CWT) · spectral · MFCC · voice quality · eGeMAPS
   ├─ envelope/  median and 1.4826·MAD per word and gap, across references
   ├─ detect/    z-scores → hysteresis regions → ruptures refinement → GBM
   ├─ score/     6 categories, S_c = 10·exp(−k_c·M_c), ±leave-one-reference-out
   └─ explain/   evidence rows → templates → explanation cards
```

**Key formulas**

| | |
|---|---|
| Deviation | `z = (x − median_r) / max(1.4826·MAD_r, σ_floor)` |
| Combined | `D_w = sqrt(Σ_k λ_k · z_wk²)` |
| Regions | enter `|z| > 2.5`, stay `> 1.5`, merge gaps `< 200 ms`, drop `< 250 ms` |
| Refinement | `ruptures` PELT, RBF cost, ±300 ms window |
| Score | `S_c = 10·exp(−k_c·M_c)`, overall = mean of 6 categories |

Every tunable lives in [`configs/`](configs/) — nothing is hard-coded in a
module.

| Config | Holds |
|---|---|
| [`paths.yaml`](configs/paths.yaml) | filesystem layout, MFA env and model names |
| [`injection.yaml`](configs/injection.yaml) | severity ladder L0–L4, per-label parameter formulas, QC gates |
| [`features.yaml`](configs/features.yaml) | 10 ms hop / 25 ms window, two-pass pitch, normalisation, channel rule |
| [`detection.yaml`](configs/detection.yaml) | τ_in 2.5 / τ_out 1.5, merging, refinement, classifier, modes |
| [`scoring.yaml`](configs/scoring.yaml) | category → label map, k_c calibration, pre-registered targets |
| [`templates.yaml`](configs/templates.yaml) | explanation strings (placeholders until Phase 6) |

---

## Flaw taxonomy

Eight labels across six rubric categories, anchored to published NSDA and
Toastmasters judging criteria:

| Category | Labels |
|---|---|
| Pace | `rushing`, `dragging` |
| Pausing | `missing_pause`, `hesitation_pause` |
| Intonation | `monotone` |
| Emphasis | `misplaced_stress` |
| Energy | `low_energy` |
| Clarity | `mumbling` *(human-recorded only)* |

Severity ladder: **L0** 0.00 · **L1** 0.15 · **L2** 0.35 · **L3** 0.60 · **L4** 0.90.

`mumbling` is never synthesised: a low-pass filter models a bad microphone, not
bad articulation. Clear speech differs from conversational speech in vowel
duration, intensity and vowel-space contrast, so a filtered proxy would be a
different phenomenon wearing the same label.

---

## Design commitments

- **Speaker agnostic.** F0 in semitones re speaker median; energy in dB re
  speaker voiced median, then z-scored; MFCC/spectral per-utterance CMVN.
- **The channel rule.** Spectral, MFCC and voice-quality features are compared
  *only within one channel* — a clean file against its own edit, or a speaker's
  clean take against their own flawed take. A 1961 tape and a USB microphone
  differ for reasons that are not delivery.
- **An envelope, not a single reference.** Median and MAD across many skilled
  readings, so a valid stylistic choice is not punished as a flaw.
- **No data leakage.** Thresholds, λ weights, k_c and the classifier are fit on
  training folds only under leave-one-text-out. Human-recorded flawed takes are
  test-only.
- **Three honest modes.** *Reference* (transcript matches a library text at
  ≥ 0.95 and at least 4 readers exist), *Norm* (unknown text, badged
  lower-confidence, channel-sensitive features informational only), *Reject*
  (WER > 25% — audio and transcript do not match). The system never shows
  confident nonsense.
- **Determinism.** Seeds from config, `PYTHONHASHSEED=0`,
  `torch.use_deterministic_algorithms`, outputs rounded before hashing (times to
  3 decimals, values to 4 significant figures), no network at inference.

---

## Licensing

Only public-domain / CC0 / CC BY sources, or team recordings released under
CC BY 4.0 with recorded consent. Every audio file carries a row in
`dataset/LICENSES.csv` with its source URL and retrieval date.

**Not used:** TED talks (CC BY-NC-ND forbids the derivatives this project
requires), YouTube rips, and voice cloning of real public figures.

---

## Repository layout

```
configs/            all tunable parameters (six YAML files)
src/rostrum/        io · data · align · text · inject · features · envelope
                    detect · score · explain · api
app/                Streamlit dashboard
eval/               metrics, leave-one-text-out, ablations, figures
scripts/            dataset build pipeline
dataset_sample/     tiny committed subset for tests and the determinism check
docs/               technical report (≤ 6 pages) and figures
results/            metrics.json, ablations.json (generated)
tests/              unit, synthetic-signal and determinism tests
```

---

## Deliverables

| Deliverable | Where |
|---|---|
| Code + documentation | this repository |
| Custom dataset | `dataset/README.md` (datasheet) + public mirror link |
| Interactive dashboard | `make app` · public demo link *(TBD)* |
| Technical documentation | `docs/TECHNICAL_REPORT.pdf` *(TBD)* |
| Demo video (3–10 min) | *(TBD)* |

Planning documents: [`MASTER_SPEC_PLAN.md`](MASTER_SPEC_PLAN.md) (the source
of truth), [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (the longer narrative plan),
[`REFERENCES.md`](REFERENCES.md) (research backing), and
[`CLAUDE.md`](CLAUDE.md) (working agreement and invariants).

Where the plan and the spec disagree, **the spec wins**.

## Licence

MIT for the code. Dataset licensing is per-file; see `dataset/LICENSES.csv`.
