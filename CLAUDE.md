# CLAUDE.md — Rostrum (Track C: Contrastive Speech Analytics & Temporal Flaw Grounding)

You are working on a hackathon project that will be judged on rigour. Read `MASTER_SPEC_PLAN.md` before any non-trivial task; it is the source of truth. If this file and the spec disagree, the spec wins — tell the user about the conflict.

## What we are building
A system that takes **speech audio + its transcript**, compares the delivery against an **envelope of skilled readings of the same text**, and outputs **timestamped flaw regions** with **numeric causal explanations** and a **6-category rubric score**. Plus a **public contrastive dataset** (good readings + graded flawed versions with exact labels) and a **Streamlit dashboard**.

## Judging weights (build in this order)
Data 30% → Grounding/Explainability 25% → Features 20% → Dashboard 15% → Reproducibility 10%.
Never polish UI before `results/metrics.json` exists.

## Commands
```bash
make setup          # create main env (py3.11) + mfa conda env
make data           # download sources → preprocess → align → inject → QC → labels
make features       # cache features for all files
make envelope       # build per-text envelopes (training folds)
make eval           # leave-one-text-out evaluation → results/metrics.json
make ablations      # ablation table → results/ablations.json
make figures        # docs/figures/*
make app            # streamlit on :8501 (+ FastAPI on :8000)
make test           # pytest -q
make lint           # ruff + black --check
make reproduce      # everything above, deterministic
```

## Architecture map
- `src/rostrum/io/` audio load/resample/loudness/hash
- `src/rostrum/align/` MFA wrapper (`conda run -n mfa ...`), ASR WER check (faster-whisper)
- `src/rostrum/text/` gap labels (boundary/minor/none), repetition/anaphora detection
- `src/rostrum/inject/` PSOLA ops, engine (severity, spans, time maps), QC (re-align, UTMOS)
- `src/rostrum/features/` pitch, energy, rate, pauses, prominence (CWT), spectral, mfcc, voice quality, egemaps, normalize
- `src/rostrum/envelope/` robust median/MAD per word & gap per text
- `src/rostrum/detect/` z-scores, hysteresis regions, ruptures refinement, GBM classifier
- `src/rostrum/score/` rubric + leave-one-reference-out confidence band
- `src/rostrum/explain/` evidence → templates → cards
- `src/rostrum/pipeline.py` `analyze(audio_path, transcript, text_id=None) -> AnalysisResult`
- `src/rostrum/api/` FastAPI · `app/` Streamlit · `eval/` metrics & LOTO · `scripts/` dataset build

## Invariants (never violate)
1. **Audio**: 16 kHz mono, float32 internally, 16-bit PCM WAV on disk, −23 LUFS integrated, peak ≤ −1 dBFS.
2. **Time**: seconds, float, rounded to 3 decimals in outputs; always relative to the processed WAV start.
3. **Frames**: 10 ms hop, 25 ms window everywhere.
4. **Alignment**: Montreal Forced Aligner only. **Never use `torchaudio.functional.forced_align`** (removed in torchaudio 2.9). WhisperX/faster-whisper only for ASR-vs-transcript WER checks.
5. **Pitch**: Praat via parselmouth, two-pass range (pass 1: 50–700 Hz; pass 2: floor 0.75·Q1, ceiling 1.5·Q3). F0 in **semitones re speaker median** for all comparisons.
6. **Normalisation**: energy in dB re speaker voiced median, then z-score; MFCC/spectral per-utterance CMVN.
7. **Channel rule**: spectral/MFCC/voice-quality features are compared **only within the same channel** (clean vs its own edit; a speaker's clean vs flawed take). Never across recordings from different sources.
8. **Injection**: edits only at word boundaries; 10 ms raised-cosine crossfades; inserted silence = looped room tone, never zeros; TD-PSOLA via Praat (`Lengthen (overlap-add)`, Manipulation pitch tier). **No WORLD vocoder.** Every op returns an exact old→new time map.
9. **Labels**: generated from time maps, then verified by re-alignment (reject if any word boundary error > 50 ms) and UTMOS (reject L1 if drop > 0.3).
10. **Envelope**: median and 1.4826·MAD; σ floor = 25th percentile of σ per feature on **training texts only**.
11. **No data leakage**: thresholds, weights, k_c, classifier are fit on training folds only (leave-one-text-out). Human-recorded flawed takes are test-only.
12. **No LLM in the judging loop**. DSP computes every number and timestamp. An LLM may only rephrase template text, temperature 0, cached.
13. **Determinism**: seeds from `configs/*.yaml`; `PYTHONHASHSEED=0`; outputs rounded before hashing; no network at inference.
14. **Licences**: only public-domain/CC BY/CC0 sources or consented team recordings. **No TED, no YouTube rips, no voice cloning.** Every audio file must have a `LICENSES.csv` row.

## Flaw taxonomy (8 labels → 6 categories)
Pace: `rushing`, `dragging` · Pausing: `missing_pause`, `hesitation_pause` · Intonation: `monotone` · Emphasis: `misplaced_stress` · Energy: `low_energy`/`fading` · Clarity: `mumbling` (human-recorded only).
Severity levels: L0 0.00, L1 0.15, L2 0.35, L3 0.60, L4 0.90.

## Key formulas
- z = (x − median_r) / max(1.4826·MAD_r, σ_floor)
- Regions: enter |z| > 2.5, stay > 1.5, merge gaps < 200 ms, drop < 250 ms (pause labels exempt)
- Refinement: ruptures PELT, RBF cost, ±300 ms window
- Score: S_c = 10·exp(−k_c·M_c); overall = mean of 6 categories; CI = leave-one-reference-out std

## Coding conventions
- Python 3.11, type hints, docstrings, `ruff` + `black` (line length 100).
- Pure functions in `features/` and `detect/`; I/O only in `io/`, `scripts/`, `pipeline.py`.
- Config values live in `configs/*.yaml`, never hard-coded in modules.
- Every public function in `inject/`, `features/`, `detect/` has a unit test.
- Use `pathlib`, `logging` (not print), dataclasses from `src/rostrum/types.py`.

## Testing expectations
- Synthetic-signal tests: a 200 Hz sine must yield F0 ≈ 200 Hz; a known silence must yield a gap of the right length.
- Time-map tests: after `lengthen(span, 0.7)`, every later word timestamp shifts by exactly the computed delta (±1 ms).
- Determinism test: run pipeline twice on `dataset_sample/`, assert identical SHA-256 of rounded JSON.

## How to work with me (Claude)
- Before writing code for a phase, restate the acceptance criteria from the prompt and the relevant spec section.
- Prefer small, verifiable steps; run tests after each module.
- If an external API (MFA model names, UTMOS loader, library signatures) might have changed, check the installed version's docs/help output instead of guessing, and note what you verified.
- Never invent dataset URLs or licence statuses; if unsure, leave a `TODO(verify)` and tell the user.
- When a target metric is missed, report it honestly; do not tune on test folds.