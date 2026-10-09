# MASTER SPEC PLAN — Rostrum
### Contrastive Speech Analytics & Temporal Flaw Grounding
**Multimodal AI Hackathon 2026 (KamandPrompt, IIT Mandi) · Track C · Deadline 15 Oct 2026 · Internal target: submit 14 Oct**

> This is the single source of truth. `CLAUDE.md` summarises it for Claude Code; `CLAUDE_PROMPTS.md` turns it into build steps. If anything conflicts, **this file wins**.

---

## Table of contents
1. Submission requirements (official) → our compliance map
2. Judging criteria → design priorities
3. System overview (architecture diagram)
4. Dataset: where to get it, licences, tiers, exact sources
5. Bad-mirror spectrum: injection engine spec
6. Human recordings & annotation spec
7. Labels, file formats, splits
8. Forced alignment spec
9. Feature extraction spec
10. Envelope, detection & temporal grounding spec
11. Scoring rubric spec
12. Causal explanation spec
13. Operating modes
14. Evaluation protocol & pre-registered targets
15. Dashboard & API spec
16. Reproducibility & engineering spec
17. Repository layout & module contracts
18. Deliverables: README, technical report, video, Devpost page
19. Timeline (day by day)
20. Risk register
21. Definition of done (master checklist)
22. References

---

## 1. Submission requirements (official) → compliance map

Devpost submission must include **all four** components before the deadline:

| # | Official requirement | How we comply | Owner check |
|---|---|---|---|
| 1 | **Project description** on Devpost: what we built, problem, how it works | §18.4 Devpost text (problem → dataset → pipeline → results → links) | ☐ |
| 2 | **Public GitHub repo** with README containing **setup instructions, prerequisites/dependencies, and run instructions** | §18.1 README template; `make setup`, `make demo`, Docker one-liner | ☐ |
| 3 | **YouTube demo video, 3–10 min**, unlisted OK, **not private**, shows project running + explains approach, **English audio or English subtitles** | §18.3 script, target ~7 min, English narration + uploaded English captions | ☐ |
| 4 | **Team info**: real full names; every teammate has a Devpost account and is **added to the submission** | Add all members on Devpost before final submit | ☐ |

Track C expected deliverables:

| Deliverable | Where | Spec |
|---|---|---|
| GitHub link with code + docs | GitHub | §17 |
| **Custom dataset**: paired good/bad spectrum audio, transcripts, alignment labels — in GitHub **or a public Google Drive link in the README** | Google Drive (public, "Anyone with the link → Viewer") + small sample in repo | §4–§7 |
| **Interactive dashboard**: audio upload, feature-extraction processing, visual + textual representation | Hugging Face Space (Docker) + local `make app` | §15 |
| **Technical documentation (max 6 pages)**: dataset construction, evaluation rubrics, model architecture, scoring methodology | `docs/TECHNICAL_REPORT.pdf` | §18.2 |
| **YouTube video (3–10 min)**: dataset collection process, stress testing across qualities, dashboard catching deviations | YouTube (unlisted) | §18.3 |

**Drive hygiene:** share the folder as "Anyone with the link — Viewer"; test the link in an incognito window; include a `MANIFEST.csv` + `checksums.sha256` so judges can verify integrity.

---

## 2. Judging criteria → design priorities

| Criterion | Weight | What earns full marks | Our differentiator |
|---|---|---|---|
| Data Engineering & Stress Testing | **30%** | Clever sourcing, clean temporal labels, robust bad-mirror gradient | Multi-reader envelope (15 LibriVox readers of one speech); exact machine labels; L0–L4 gradient; realism verified with UTMOS + re-alignment; human-recorded test set |
| Causal Explainability & Temporal Grounding | **25%** | Accurate timestamps; maths linked to context | Word-anchored z-scores vs skilled-reader distribution; hysteresis + change-point refinement; evidence table + text context per region |
| Feature Extraction | **20%** | Rigour; accurate stress points & energy contours | Two-pass Praat pitch (validated), CREPE cross-check, CWT prominence, eGeMAPS, channel rule |
| Visualization & Dashboard | **15%** | Usability, clear overlays & rationale | Envelope band ±σ overlay, coloured regions, coloured transcript, A/B playback, radar rubric |
| Reproducibility & Code Quality | **10%** | Clean execution, deploy docs, architecture | Docker, pinned env, `make reproduce`, determinism CI test |

**Rule:** build in weight order. No dashboard polish until the evaluation table exists.

---

## 3. System overview

```
                 ┌──────────────────────── OFFLINE (dataset build) ───────────────────────┐
 Public-domain   │  download → verify licence → preprocess (16k mono, −23 LUFS)            │
 sources ───────►│  → MFA align → INJECTION ENGINE (8 flaw ops × L0–L4, seeded)           │
 + team takes    │  → re-align check + UTMOS realism → labels (JSON/TSV/TextGrid)          │
                 │  → features cache → ENVELOPE per text (median/MAD per word & gap)       │
                 └─────────────────────────────────────────────────────────────────────────┘
                 ┌──────────────────────── ONLINE (analysis) ──────────────────────────────┐
 audio +         │  preprocess → mode select (Reference / Norm / Reject via ASR-WER)       │
 transcript ────►│  → MFA align → features (F0 st, energy, rate, pauses, prominence,       │
                 │    spectral, MFCC, voice quality) → z vs envelope per word/gap          │
                 │  → hysteresis regions → ruptures boundary refinement → GBM flaw typing  │
                 │  → rubric scores + leave-one-reference-out CI → explanation objects     │
                 │  → FastAPI JSON → Streamlit dashboard (overlay, regions, cards, A/B)    │
                 └─────────────────────────────────────────────────────────────────────────┘
```

**Multimodality** (hackathon theme): audio signal + transcript text. Text supplies alignment targets, expected pause locations (punctuation/clause boundaries) and rhetorical context (repetition/anaphora) used in explanations.

---

## 4. Dataset: where to get it

### 4.1 Licence policy (hard rules)

1. Only **public domain** or **CC BY / CC0** audio, or our own recordings released under CC BY 4.0 with written consent.
2. Flawed versions are derivatives → **No-Derivatives licences are forbidden** (TED is CC BY-NC-ND 4.0 and explicitly forbids edits) [R8].
3. **No voice cloning** of real people.
4. Every file gets a row in `LICENSES.csv` with the source page URL and a screenshot/PDF of the licence statement saved under `dataset/provenance/`.

### 4.2 Verified sources (checked during research)

| ID | Text | Tier | Source | Licence evidence |
|---|---|---|---|---|
| S1 | **Gettysburg Address** | Silver ×15 | LibriVox *The Gettysburg Address 150th Anniversary* — 15 different volunteer readers [R5] | LibriVox recordings are public domain and may be modified [R6] |
| S1b | Gettysburg Address | Silver | LibriVox *Gettysburg Address (version 2)*, read by John Greenman [R5b] | Public domain (LibriVox) |
| S2 | **JFK Inaugural Address** (20 Jan 1961) | Gold | JFK Library, item **JFKWHA-001** ("Download Audio File" on the page) [R7] | Page states "Copyright Status: Public Domain" |
| S3 | **FDR "Day of Infamy"** (8 Dec 1941) | Gold | Wikimedia Commons `File:Roosevelt_Pearl_Harbor.ogg` [R7b]; FDR Library copy listed "Rights Status: Unrestricted" [R7c] | Commons marks public domain; contains the national anthem after the speech → trim |
| S4 | Lincoln Second Inaugural | Silver (candidate) | LibriVox *American Civil War Collection, Vol. 1*, ch. 1 (read by John Stamper) [R7d] | LibriVox public domain — verify page before use |

**Trimming notes:** S2 includes the swearing-in ceremony → keep only the address segment (select a 90–150 s excerpt). S3 has the anthem appended → cut it.

### 4.3 How to find more (optional, if time allows)

- LibriVox catalogue → search the keyword **"multi-version"** or a speech title; prefer projects with several readers of the same text.
- Presidential library audio pages (JFK Library, FDR Library) → open the item page and confirm the rights field says Public Domain / Unrestricted.
- Wikimedia Commons → file page licence box must say public domain or CC BY/CC0.
- **Never** download from YouTube rips, TED, or news sites.

### 4.4 Team recordings (Bronze tier + human flaws)

- Each team member (or friends with consent) reads **S1, S2-excerpt, S3-excerpt** (texts are public domain: US federal works / 1863 speech).
- Recording setup: quiet room, same phone/mic for all of one speaker's takes, 30 cm distance, WAV if possible (else highest-quality m4a), 3 s of room tone at start.
- Consent: one line per person in `dataset/CONSENT.md`: *"I, <full name>, release my recordings in this dataset under CC BY 4.0."* (date + signature/initials).

### 4.5 Final dataset composition (target)

| Text | Gold | Silver | Bronze (team clean) | Envelope size R |
|---|---|---|---|---|
| Gettysburg (S1) | — | 15–16 | 3–4 | ~19 |
| JFK excerpt (S2) | 1 | — | 3–4 | 4–5 |
| FDR excerpt (S3) | 1 | — | 3–4 | 4–5 |
| (optional) Lincoln 2nd (S4) | — | 1+ | 3–4 | 4–5 |

Minimum R for Reference mode = **4**. Below that, σ uses the pooled floor (§10.2).

### 4.6 Preprocessing (identical for all files)

1. Decode → 16 kHz mono float32 (`ffmpeg -ac 1 -ar 16000`).
2. Trim to the chosen excerpt (start/end recorded in `MANIFEST.csv`).
3. Keep 300 ms lead/tail silence.
4. Loudness-normalise to **−23 LUFS** integrated (`pyloudnorm`); peak-limit to −1 dBFS.
5. Write 16-bit PCM WAV; SHA-256 into `checksums.sha256`.

---

## 5. Bad-mirror spectrum: injection engine spec

### 5.1 Principles

- Edit only at **word boundaries** from MFA alignment of the clean file.
- Splice with **10 ms raised-cosine crossfades**.
- Inserted silence = **room tone** (noise floor looped from the file's own longest pause), never digital zeros.
- Time and pitch modification via **Praat TD-PSOLA** (`parselmouth`). In listening tests TD-PSOLA time-stretching was rated on par with a neural editor and far above the WORLD vocoder [R18].
- Every operation returns an **exact time map** old→new; all downstream timestamps derive from it.
- Every output: re-align with MFA → boundary agreement check (≤ 20 ms target) → UTMOS naturalness score [R19].
- Fully **seeded**; seed + params stored in the label.

### 5.2 Operations (8 labels → 6 rubric categories)

| Category | Label | Operation | Severity mapping (s ∈ [0,1]) | Span |
|---|---|---|---|---|
| Pace | `rushing` | `Lengthen (overlap-add)` factor < 1 on the span | factor = 1 − 0.45·s | 3–10 words |
| Pace | `dragging` | `Lengthen (overlap-add)` factor > 1 | factor = 1 + 0.8·s | 3–10 words |
| Pausing | `missing_pause` | Shorten an existing **boundary** pause (≥ 300 ms in clean) | keep (1 − 0.95·s) of it | 1 gap |
| Pausing | `hesitation_pause` | Insert room tone at a **non-boundary** gap | 0.3 + 1.7·s seconds | 1 gap |
| Intonation | `monotone` | PSOLA pitch tier: f′ = m·(f/m)^(1−0.9·s), m = span median F0 | 0 → 90% compression | 1 phrase (5–15 words) |
| Emphasis | `misplaced_stress` | On a function word: +4s st F0, +4s dB, duration ×(1+0.3s); on the phrase's most prominent content word: inverse | as stated | 2 words in 1 phrase |
| Energy | `low_energy` / `fading` | Smooth gain ramp from 0 dB to −18·s dB across span | as stated | 4–12 words, phrase-final preferred |
| Clarity | `mumbling` | **Not injected** — human-recorded only [R22, R23] | annotator 1–3 | — |

### 5.3 Severity ladder

| Level | s | Meaning |
|---|---|---|
| L0 | 0.00 | clean (no edit) |
| L1 | 0.15 | almost perfect |
| L2 | 0.35 | noticeable |
| L3 | 0.60 | clearly flawed |
| L4 | 0.90 | egregious |

> These parameter values are **engineering design choices**, not taken from literature. Document them as such; adjust only on training texts and only if UTMOS/realism checks demand it.

### 5.4 Generation plan

- **Single-flaw files:** for each clean file × each of 7 injectable labels × L1–L4 → 1 file (span chosen by seeded RNG among valid spans).
- **Composite files:** for each clean file, 2 files at L2 and 2 at L4 with 3–5 non-overlapping flaws (≥ 1 s apart).
- **Realism gates:** reject if re-alignment boundary error > 50 ms on any word, or if UTMOS(L1) < UTMOS(clean) − 0.3. Log rejections.

---

## 6. Human recordings & annotation spec

### 6.1 Flawed takes per team speaker per text

1. **Nervous rush** through the middle third.
2. **Monotone** reading of one paragraph.
3. **Mumbled, low-energy** segment + 2–3 hesitation pauses mid-sentence.

Same mic/room as their clean takes (channel matching).

### 6.2 Annotation

- Tool: Praat. Import the MFA TextGrid, add tier `flaws`; intervals labelled `<label>:<severity 1-3>`.
- Two annotators label ≥ 30% of human files independently.
- Report: inter-annotator **event-F1 at 200 ms collar** and **Cohen's κ** on labels for overlapping events.
- `dataset/ANNOTATION_GUIDE.md`: definition + one audio example + borderline rules for each label.

---

## 7. Labels, file formats, splits

### 7.1 Label JSON (one per audio file)

```json
{
  "file_id": "s1_lv07_rushing_L3",
  "source_file": "s1_lv07",
  "text_id": "s1_gettysburg",
  "speaker_id": "lv_LAW",
  "origin": "injected",            // injected | human | clean
  "tier": "silver",
  "seed": 1337,
  "flaws": [
    {"label": "rushing", "t_start": 41.212, "t_end": 45.884,
     "word_idx": [52, 61], "level": "L3", "s": 0.60,
     "params": {"psola_factor": 0.73}}
  ],
  "qc": {"utmos_clean": 3.91, "utmos_edited": 3.62,
         "realign_max_err_ms": 14, "passed": true}
}
```

All times in **seconds, float, 3 decimals**, relative to the start of the processed WAV.

### 7.2 Other formats

- `labels/events.tsv` — `filename<TAB>onset<TAB>offset<TAB>event_label` (sed_eval format).
- `alignments/*.TextGrid` — tiers `words`, `phones` (MFA) + `flaws`.
- `texts/<text_id>.json` — `{"verbatim": "...", "display": "...", "gap_labels": [...]}`.

### 7.3 Splits

**Leave-one-text-out (LOTO)** cross-validation over texts S1–S3(/S4). All thresholds, weights and the classifier are fit on N−1 texts; test on the held-out text. Human flawed takes are **test-only**. Report mean ± sd across folds.

### 7.4 Dataset folder (Drive) + sample (repo)

```
dataset/
  README.md            # datasheet (motivation, composition, collection, preprocessing,
                       #  labelling, QC, splits, licences, intended use, limitations)
  LICENSES.csv         # file_id,title,speaker,source_url,licence,retrieved_on,sha256
  CONSENT.md
  ANNOTATION_GUIDE.md
  MANIFEST.csv         # file_id,text_id,tier,origin,label,level,duration_s,split_fold
  provenance/          # licence screenshots/PDFs
  texts/
  audio/{clean,injected,human_flawed}/
  alignments/
  labels/{*.json,events.tsv}
  qc/{utmos.csv,realign.csv,rejections.csv}
  splits/loto.json
  checksums.sha256
```

Repo keeps `dataset_sample/` (≈ 20 files) so the demo runs without the Drive download.

---

## 8. Forced alignment spec

| Item | Spec |
|---|---|
| Aligner | **Montreal Forced Aligner** (MFA), pinned version, `english_us_arpa` acoustic model + dictionary. MFA outperformed WhisperX and MMS on hand-aligned TIMIT/Buckeye [R11] |
| Isolation | MFA lives in its own conda env (`mfa`) and is called via `conda run -n mfa mfa align ...` to avoid dependency clashes with the main env |
| Corpus format | `<id>.wav` + `<id>.lab` (verbatim lowercase transcript) |
| OOV handling | `mfa validate` first; add missing pronunciations to a project dictionary or expand text |
| Text-mismatch check | `faster-whisper` ASR → WER vs transcript; WER > 0.25 or alignment failure → Reject mode |
| Banned | `torchaudio.functional.forced_align` — deprecated in 2.8, removed in 2.9 [R12] |
| Validation | Hand-label 50 word boundaries (5 files) in Praat → report MAE, P95, % within 20/50 ms |

> Verify current MFA model names with `mfa model download acoustic --help` on day 1.

---

## 9. Feature extraction spec

Frame hop **10 ms**; window 25 ms. Aggregated to **word**, **gap** (inter-word silence) and **phrase** (between boundary gaps).

### 9.1 Feature table

| ID | Feature | Method | Word/gap/phrase aggregates |
|---|---|---|---|
| F0 | Pitch | Praat autocorrelation via parselmouth, **two-pass range**: pass 1 floor 50 / ceiling 700 Hz → Q1, Q3 → pass 2 floor 0.75·Q1, ceiling 1.5·Q3 | mean st, max st, slope st/s, range st |
| F0x | Pitch cross-check | `torchcrepe` (or `penn`) | % frames agreeing within 50 cents with Praat; disagreeing frames excluded |
| EN | Intensity | Praat intensity (dB) + `librosa` RMS | mean, peak, slope (dB/s) |
| DUR | Duration | MFA intervals | word duration; duration / phone count |
| RATE | Speech rate | vowel phones per second from MFA; cross-check with de Jong & Wempe syllable nuclei [R14] | local rate over ±3 words |
| PAU | Pauses | MFA gaps refined by **Silero VAD** | gap duration + gap label (boundary/minor/none) |
| PROM | Prominence | CWT over composite of F0 + energy + duration (weights 1.0/0.5/1.0) [R25, R26] | per-word prominence |
| SPEC | Spectral (FFT) | STFT 512-pt Hann: centroid, flux, tilt (log-mag slope), band-energy ratio 0–1 k/1–4 k Hz | per word |
| MFCC | Cepstral | 13 MFCC + Δ + ΔΔ, per-utterance CMVN | per word mean |
| VQ | Voice quality | parselmouth HNR, jitter (local), shimmer (local), CPP | per voiced word |
| EG | eGeMAPS | `opensmile` eGeMAPSv02 functionals [R13] | per phrase |

### 9.2 Speaker normalisation

| Signal | Transform |
|---|---|
| F0 | $\mathrm{st}(t)=12\log_2\!\big(f_0(t)/\tilde f_{0}^{\,spk}\big)$, speaker median over the recording |
| Energy | $e(t)-\tilde e^{\,spk}_{voiced}$ (dB), then z-score per recording |
| Rate | absolute syll/s **and** relative to speaker global articulation rate |
| MFCC/SPEC | per-utterance mean/variance normalisation |
| Pitch variation | std of F0 **in semitones** (Hz-ratio PVQ needs sex-specific thresholds [R20]) |
| Phrase pitch range | phrase F0 max − utterance-final low F0 (st) — more reliable than peak–valley [R29] |

Justification: z-score normalisation best matched perceptual normalisation [R24]; normalisation choice affects sensitivity [R30].

### 9.3 Channel rule (state in the report)

- **Channel-robust** (timing, pauses, rate, st-F0, relative energy, prominence) → compared to the multi-speaker envelope.
- **Channel-sensitive** (SPEC, MFCC, VQ) → compared **only within the same channel** (clean vs its own injected edit; a speaker's clean take vs their flawed take). In Norm mode they are reported but never trigger flaws.

---

## 10. Envelope, detection & temporal grounding spec

### 10.1 Word-anchored comparison
Identical transcripts → word *i* ↔ word *i*. No blind DTW for detection. DTW (`librosa.sequence.dtw`) only **within words** to warp frame curves onto the reference timeline for plotting.

### 10.2 Envelope (per text *T*, references r = 1..R, feature k, unit w)

$$\mu_{wk}=\operatorname{median}_r x^{(r)}_{wk},\qquad \sigma_{wk}=1.4826\cdot\operatorname{MAD}_r\,x^{(r)}_{wk},\qquad \hat\sigma_{wk}=\max(\sigma_{wk},\sigma^{floor}_k)$$

$\sigma^{floor}_k$ = 25th percentile of $\sigma_{\cdot k}$ across all units of all **training** texts.

### 10.3 Deviation

$$z_{wk}=\frac{x^{(p)}_{wk}-\mu_{wk}}{\hat\sigma_{wk}},\qquad D_w=\sqrt{\textstyle\sum_k\lambda_k z_{wk}^2}$$

Label-specific primary signals:

| Label | Trigger signal |
|---|---|
| rushing / dragging | $z_{\text{RATE}}$ > +τ / < −τ |
| missing_pause | gap label = boundary and $z_{\text{PAU}}$ < −τ |
| hesitation_pause | gap label = none, gap > 0.25 s and $z_{\text{PAU}}$ > +τ (threshold per [R31]; mid-clause pauses most damaging [R32]) |
| monotone | phrase pitch-range z < −τ and phrase F0-std z < −τ |
| misplaced_stress | prominence-rank correlation within phrase vs envelope median ranking < ρ_min, localised to the words with largest |Δprominence| |
| low_energy / fading | relative-energy z < −τ (fading: negative energy slope z < −τ) |
| mumbling | same-channel clarity composite (rate↓, intensity↓, vowel duration↓, CPP↓, spectral tilt steeper) z beyond τ |

### 10.4 Region formation
1. Per label, signed score per unit; 3-unit median smoothing.
2. Hysteresis: enter |z| > **τ_in = 2.5**, stay while |z| > **τ_out = 1.5**.
3. Merge regions < 200 ms apart; drop regions < 250 ms (pause labels exempt).
4. Initial boundaries = first word onset / last word offset (gap labels: gap edges).
5. τ values tuned per LOTO fold on training texts only (maximise event-F1).

### 10.5 Boundary refinement
Within ±300 ms of each boundary run `ruptures` **PELT, RBF cost, fixed penalty** on the frame-level deviation of the primary feature [R33]. Accept the refined point only if it moves < 300 ms. Report metrics with and without refinement.

### 10.6 Flaw-type classifier (bonus)
- Features per region: mean/max z per feature, duration, gap label one-hot, phrase position.
- `HistGradientBoostingClassifier(random_state=0)`; isotonic calibration for severity.
- Trained on injected flaws of training texts; evaluated on held-out text + human takes.
- If classifier and rule disagree, the card shows the rule label and the classifier probability.

### 10.7 No LLM in the judging loop
Audio LLMs over-predict low-quality speech and lack precision in error detection [R35]. Trustworthy approaches extract interpretable acoustic measurements first [R36, R37]. DSP computes all numbers and timestamps; an LLM is optional and may only rephrase template text (temperature 0, outputs cached in repo).

---

## 11. Scoring rubric spec

| Category | Judging anchor [R1–R3] | Labels |
|---|---|---|
| Pace | speed | rushing, dragging |
| Pausing | effective use of pauses | missing_pause, hesitation_pause |
| Intonation | pitch / tone variety | monotone |
| Emphasis | vocal variety to emphasise | misplaced_stress |
| Energy | volume | low_energy, fading |
| Clarity | clarity | mumbling |

$$M_c=\frac{1}{T_{speech}}\sum_{j\in c}\mathrm{dur}_j\cdot\overline{(|z|-1.5)_+}_{\,j},\qquad S_c=10\,e^{-k_c M_c},\qquad S=\tfrac16\sum_c S_c$$

- $k_c$ calibrated per fold so single-flaw **L2** files score ≈ 6/10 in their category.
- **Confidence band:** leave-one-reference-out recomputation → report mean ± std.
- **Raw descriptors** displayed: articulation rate, pitch variation (st), mean length of runs (syllables between pauses > 250 ms) and pause rate — liveliness correlates with normalised F0 variation and run length [R21].

---

## 12. Causal explanation spec

Feedback that states the error, location, magnitude and fix is more useful than a bare flag [R39].

```json
{
  "region_id": 3, "t_start": 41.27, "t_end": 45.83,
  "words": "we can not dedicate we can not consecrate",
  "category": "Pace", "label": "rushing", "level": "L3",
  "confidence": 0.86, "rule_label": "rushing", "clf_prob": 0.91,
  "evidence": [
    {"feature": "rate_syll_s", "value": 6.4, "env_median": 3.9, "env_sigma": 0.6, "z": 4.1},
    {"feature": "gap_before_s", "value": 0.06, "env_median": 0.82, "env_sigma": 0.21,
     "z": -3.6, "gap_label": "boundary"}
  ],
  "context": "Repeated phrase (anaphora) after a sentence boundary.",
  "rationale": "Rate is 4.1σ above the skilled-reader envelope; the boundary pause was cut from 0.82 s to 0.06 s.",
  "fix": "Slow to ~4 syll/s and hold a ~0.8 s pause before 'we can not dedicate'.",
  "envelope_n": 19, "mode": "reference"
}
```

**Context generator (text modality):** punctuation/gap labels; repeated n-gram detection (anaphora/parallelism); sentence-initial/final position; the reference readers' behaviour at that spot (e.g., "17 of 19 skilled readers paused here").

**Template per label** lives in `configs/templates.yaml`; numbers formatted to 2 significant decimals.

---

## 13. Operating modes

| Mode | Trigger | Behaviour | UI badge |
|---|---|---|---|
| Reference | transcript matches a library text (normalised string match ≥ 0.95) and R ≥ 4 | full envelope | "Reference · n readers" |
| Norm | unknown text, alignment OK | pooled norms (rate band, pitch range, pause norms by gap label) + within-speaker change points; channel-sensitive features informational only | "Norm-based · lower confidence" |
| Reject | WER > 0.25 or alignment failure | message: audio and transcript do not match | red banner |

---

## 14. Evaluation protocol & pre-registered targets

### 14.1 Metrics

| Aspect | Metric | Tool |
|---|---|---|
| Grounding | Event-F1, onset collar 200 ms, offset collar max(200 ms, 20% length) [R9, R34] | `sed_eval` |
| Threshold-free | PR curve / PSDS [R10] | `sed_scores_eval` |
| Boundaries | onset/offset MAE (ms), temporal IoU | custom |
| Typing | macro-F1 + confusion matrix | sklearn |
| Gradient | Spearman ρ (level vs category score), per text and pooled | scipy |
| Speaker-agnostic | FP regions/min on clean files vs envelope; male–female score gap for identical injected flaws | custom |
| Transfer | all of the above on human takes | — |
| Realism | UTMOS drop per level; feature-distribution distance synthetic vs human | — |
| Components | MFA boundary MAE; Praat–CREPE agreement; annotator κ | — |
| Reproducibility | identical SHA-256 of rounded outputs over 2 runs + Docker | `make reproduce` |

### 14.2 Ablations
1. Single reference (Gold or one Silver) vs envelope.
2. Rules vs rules + classifier.
3. ± boundary refinement.
4. ± speaker normalisation.
5. Synthetic test vs human test.

### 14.3 Targets (write them before testing; report actuals honestly)

| Metric | Synthetic (held-out text) | Human |
|---|---|---|
| Event-F1 @ 200 ms | ≥ 0.75 | ≥ 0.50 |
| Onset/offset MAE | ≤ 150 ms | ≤ 250 ms |
| Macro-F1 (type) | ≥ 0.70 | ≥ 0.50 |
| Spearman ρ | ≤ −0.85 | ≤ −0.70 |
| FP on clean | ≤ 1 / min | ≤ 1 / min |
| M–F gap | ≤ 0.5 pt | — |
| UTMOS drop at L1 | ≤ 0.3 | — |
| Output hash | identical | identical |

Include a **"Where it fails"** section with 3 concrete failure cases.

---

## 15. Dashboard & API spec

### 15.1 API (FastAPI)

| Endpoint | Input | Output |
|---|---|---|
| `POST /analyze` | multipart: `audio` (wav/mp3/m4a/ogg ≤ 20 MB, ≤ 5 min), `transcript` (text), optional `text_id` | `AnalysisResult` JSON (mode, words[], gaps[], curves{}, envelope_curves{}, regions[], scores{}, descriptors{}) |
| `GET /clip` | `analysis_id`, `t_start`, `t_end`, `which=participant|reference` | WAV bytes |
| `GET /library` | — | available texts + sample flawed files |
| `GET /health` | — | versions + model hashes |

### 15.2 Streamlit screens
1. **Upload**: audio + transcript (both required) **or** pick a library text + sample file.
2. **Processing**: step status (normalise → align → features → compare → explain) with timings.
3. **Analysis**:
   - Waveform with coloured flaw regions (colour = category).
   - Plotly overlay (shared x): participant vs **envelope median + ±1σ band** for F0 (st), relative energy, local rate; flaw regions as shaded `vrect`s. Envelope curves are per-word time-normalised then averaged (ProsodyPro-style [R40]).
   - Transcript coloured by $D_w$ (prosody mapped onto text [R41]).
   - Explanation cards with evidence table + **A/B playback** (participant vs closest reference reader).
   - Radar chart of 6 categories + overall score ± CI + raw descriptors.
4. **Export**: JSON + PDF.

Show flaws in the context of the whole speech, not isolated clips [R43]; overlaying learner vs model pitch is an established feedback design [R42].

### 15.3 Deployment
Hugging Face Space (Docker SDK). Pre-download models at build time. Analysis of a 2-min file target ≤ 60 s on CPU.

---

## 16. Reproducibility & engineering spec

- Python **3.11**; main env pinned in `requirements.lock`; MFA env pinned in `environment.mfa.yml`.
- Seeds in `configs/*.yaml`; set `PYTHONHASHSEED=0`, `numpy`, `random`, `torch.manual_seed`, `torch.use_deterministic_algorithms(True)` where used; CPU inference for scoring.
- **Hash rounding:** outputs rounded (times 3 dp, features 4 sig. fig.) before hashing so tiny float differences across machines don't break the determinism test.
- No network at inference; models cached in image.
- `make reproduce` = data QC → features → envelope → LOTO eval → figures → `results/metrics.json`.
- CI (GitHub Actions): lint, unit tests, determinism test on `dataset_sample/`.
- Code style: `ruff`, `black`, type hints, docstrings; pure functions for feature/detection code.

---

## 17. Repository layout & module contracts

```
rostrum/
├── README.md
├── CLAUDE.md
├── MASTER_SPEC_PLAN.md
├── Makefile
├── Dockerfile
├── environment.mfa.yml
├── requirements.txt / requirements.lock
├── configs/{paths,injection,features,detection,scoring,templates}.yaml
├── src/rostrum/
│   ├── io/audio.py            # load, resample, loudness, write, hash
│   ├── data/manifest.py       # MANIFEST/LICENSES handling, provenance checks
│   ├── align/mfa.py           # corpus prep, conda-run wrapper, TextGrid → Alignment
│   ├── align/asr_check.py     # faster-whisper WER check
│   ├── text/gaps.py           # gap labels, anaphora/repetition detection
│   ├── inject/ops.py          # psola_lengthen, psola_pitch, gain_ramp, insert_roomtone, shorten_gap
│   ├── inject/engine.py       # span selection, severity mapping, time maps, labels
│   ├── inject/qc.py           # re-align check, UTMOS
│   ├── features/{pitch,energy,rate,pauses,prominence,spectral,mfcc,voice_quality,egemaps}.py
│   ├── features/normalize.py
│   ├── envelope/build.py      # robust μ/σ per text
│   ├── detect/{zscore,regions,refine,classifier}.py
│   ├── score/rubric.py
│   ├── explain/cards.py
│   ├── pipeline.py            # analyze(audio, transcript) -> AnalysisResult
│   └── api/server.py
├── app/streamlit_app.py
├── eval/{metrics,run_loto,ablations,figures}.py
├── scripts/{download_sources,build_dataset,annotate_import}.py
├── tests/
├── dataset_sample/
└── docs/{TECHNICAL_REPORT.md,TECHNICAL_REPORT.pdf,figures/}
```

**Core dataclasses** (`src/rostrum/types.py`): `Word(idx, text, t0, t1, phones)`, `Gap(idx, t0, t1, label)`, `Alignment(words, gaps)`, `FeatureTable` (pandas DataFrame indexed by unit), `Envelope(text_id, mu, sigma, n)`, `Region(label, t0, t1, word_idx, z_evidence, level, confidence)`, `AnalysisResult`.

---

## 18. Deliverables

### 18.1 README (required sections — the official rules demand setup, prerequisites, run instructions)
1. Title, one-line pitch, demo GIF.
2. Links: **Live demo**, **Dataset (Google Drive)**, **YouTube video**, **Technical report**.
3. **Prerequisites**: OS, Docker (recommended) or conda + Python 3.11, ffmpeg, disk/RAM.
4. **Setup**: Docker one-liner; conda path (`make setup`).
5. **Run**: `make demo` (sample data), `make app`, `make reproduce`, API example with `curl`.
6. Dataset summary table + how to download into `dataset/`.
7. Architecture diagram; results table; ablations.
8. Repository structure; licences (code MIT, dataset mixed PD + CC BY 4.0); citation; team.

### 18.2 Technical report (≤ 6 pages)
1. Introduction & related work (0.75 p)
2. Dataset construction (1.5 p)
3. Features & normalisation (1 p)
4. Grounding, rubric & scoring methodology, explanation (1.25 p)
5. Results, ablations, failure cases (1 p)
6. Reproducibility & limitations (0.5 p)

### 18.3 Video (~7 min, English narration + English captions, unlisted)
| Time | Content |
|---|---|
| 0:00–0:30 | Hook: dashboard pinpoints a rushed line; A/B playback |
| 0:30–2:00 | **Dataset collection**: sources & licences, 15 readers envelope, L0→L4 audio ladder, human takes, label files |
| 2:00–3:30 | Approach: alignment → features → envelope z → regions → explanation |
| 3:30–5:30 | **Project running**: upload a human-recorded flawed take + transcript; walk through overlay, regions, cards, scores |
| 5:30–6:30 | **Stress testing**: severity curves, event-F1, ablations, M/F test |
| 6:30–7:00 | Reproducibility + limitations |

### 18.4 Devpost description skeleton
**Inspiration** → subjective judging, vague feedback. **What it does** → upload speech + transcript, get timestamped, explained flaws and a rubric score. **How we built it** → dataset (sources, envelope, injection, labels), pipeline, dashboard. **Challenges** → channel confounds, realistic injection, alignment on flawed audio. **Accomplishments** → metrics table. **What we learned** / **What's next**. **Built with**: Python, Praat/parselmouth, MFA, librosa, openSMILE, ruptures, scikit-learn, FastAPI, Streamlit, Plotly, Docker, Hugging Face Spaces.

---

## 19. Timeline

| Date | Work | Exit criterion |
|---|---|---|
| Oct 2 (Fri) | Repo bootstrap; MFA env installed; download S1–S3 | MFA aligns one file |
| Oct 3 (Sat) | Licences/provenance; transcripts; preprocessing; align all clean | All clean files aligned + LICENSES.csv |
| Oct 4 (Sun) | Injection ops + engine + time maps + unit tests; record team takes | L0–L4 for S1 |
| Oct 5 (Mon) | QC (re-align, UTMOS), composites, all texts; annotate human takes | Dataset v1 frozen |
| Oct 6 (Tue) | Features + normalisation + pitch cross-check | Feature cache |
| Oct 7 (Wed) | Envelope + z + regions + refinement | First grounding predictions |
| Oct 8 (Thu) | Eval harness (sed_eval, Spearman, M/F, FP); LOTO | First metrics table |
| Oct 9 (Fri) | Classifier, rubric, CI band, explanations, modes | Full AnalysisResult |
| Oct 10 (Sat) | FastAPI + Streamlit dashboard | End-to-end demo |
| Oct 11 (Sun) | Ablations, figures, failure analysis; tune on train folds only | Numbers frozen |
| Oct 12 (Mon) | Docker, CI, HF Space deploy, Drive upload + link test | Public demo + dataset link |
| Oct 13 (Tue) | Report (≤ 6 p), README, video record + captions | Video unlisted on YouTube |
| Oct 14 (Wed) | Devpost page, add teammates, **submit** | Submitted |
| Oct 15 (Thu) | Buffer only | — |

---

## 20. Risk register

| Risk | Mitigation |
|---|---|
| MFA install/OOV problems | Day-1 install; separate conda env; project dictionary |
| Injection artifacts → inflated scores | UTMOS gate; human test set; report separately |
| Few texts → unstable stats | LOTO; mean ± sd; envelope from 15+ readers on S1 |
| Channel confounds | Channel rule §9.3 |
| Old recordings (1941/1961) noisy | Use them as Gold anchors; their channel-sensitive features never compared cross-channel |
| Alignment drift on flawed audio | Re-align check; report error distribution |
| Licence doubt | Provenance screenshots; drop any unclear file |
| Drive link not public | Incognito test before submitting |
| Video private by mistake | Set **Unlisted**, test logged-out |
| Teammates missing on Devpost | Add all members 24 h before deadline |
| Dashboard time sink | Streamlit first; polish only after metrics |

---

## 21. Definition of done

- [ ] Dataset on public Drive (link tested incognito) + sample in repo
- [ ] LICENSES.csv + provenance + CONSENT.md complete
- [ ] Labels verified (re-align report, UTMOS table, annotator agreement)
- [ ] LOTO metrics + ablations + failure cases in `results/`
- [ ] Dashboard: upload audio + transcript → overlay + highlighted regions + explanations
- [ ] README has prerequisites, setup, run instructions, all links
- [ ] Docker build + `make reproduce` pass from a fresh clone
- [ ] Technical report ≤ 6 pages
- [ ] Video 3–10 min, unlisted, English audio/captions, shows the project running
- [ ] Devpost: description, repo link, video link, all teammates with real full names added
- [ ] Submitted before deadline

---

## 22. References

**Judging criteria:** [R1] NSDA, How to Judge Declamation — https://www.speechanddebate.org/?p=1070 · [R2] NSDA Declamation ballot — https://newmexicoforensicsclub.readthedocs.io/en/latest/_downloads/8016c624c36316fb5f0f70dd9ef71298/nsda-declamation-ballot.pdf · [R3] Toastmasters evaluation resources — https://content.toastmasters.org/image/upload/8053-generic-evaluation-resource.pdf ; https://ccdn.toastmasters.org/medias/files/department-documents/education-documents/evaluation-resources/english/8104e2-speech-profile-ff.pdf

**Data sources:** [R5] LibriVox, The Gettysburg Address 150th Anniversary — https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/ · [R5b] LibriVox, Gettysburg Address (version 2) — http://librivox.org/the-gettysburg-address-by-abraham-lincoln-version-2/ · [R6] LibriVox public-domain statement — https://www.archive.org/download/LibrivoxCdCoverArt12/Gettysburg_Address_V3_1109.pdf · [R7] JFK Library JFKWHA-001 — https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001 · [R7b] Wikimedia Commons, Roosevelt_Pearl_Harbor.ogg — https://commons.wikimedia.org/wiki/File:Roosevelt_Pearl_Harbor.ogg · [R7c] FDR Library audio record (Rights: Unrestricted) — https://www.presidentialtimeline.org/html/record_id-80.html · [R7d] LibriVox American Civil War Collection Vol. 1, ch. 1 — search the LibriVox catalogue · [R8] TED Talks Usage Policy — https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy

**Evaluation:** [R9] Mesaros, Heittola, Virtanen (2016), Metrics for Polyphonic SED, Applied Sciences 6(6):162 · [R10] Ebbers et al. (2022), Threshold-Independent Evaluation of SED Scores, arXiv:2201.13148 · [R34] DCASE 2020 Task 4 benchmark, arXiv:2011.00801

**Alignment:** [R11] Rousso et al. (2024), Tradition or Innovation, Interspeech, arXiv:2406.19363 · [R12] torchaudio forced_align deprecation — https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.forced_align.html · McAuliffe et al. (2017), Montreal Forced Aligner, Interspeech · Bain et al. (2023), WhisperX, arXiv:2303.00747

**Features:** [R13] Eyben et al. (2016), GeMAPS, IEEE TAC 7(2) · [R14] de Jong & Wempe (2009), Behavior Research Methods 41(2) · [R15] Strömbergsson, Praat vs getF0 vs YIN on PTDB-TUG — https://cdn.dal.ca/content/dam/dalhousie/pdf/sites/icpla/postersjune17/61724.PDF.lt_9ddeaa1253afb7974ae5ffe9e8637664.res/61724.PDF · [R25] Suni, Aalto, Vainio (2015), arXiv:1510.01949 · [R26] Wavelet Prosody Toolkit — https://github.com/asuni/wavelet_prosody_toolkit · [R27] Kim et al. (2018), CREPE, arXiv:1802.06182 · [R28] Morrison et al. (2023), PENN, arXiv:2301.12258 · [R29] den Ouden & Terken (2001), Measuring pitch range, Eurospeech · [R33] Truong, Oudre, Vayatis (2020), Selective review of offline change point detection, arXiv:1801.00718 · [R40] Xu (2013), ProsodyPro, TRASP · Jadoul et al. (2018), Parselmouth, J. Phonetics 71 · de Seyssel et al. (2024), EmphAssess, EMNLP

**Normalisation:** [R20] US Patent 11,205,418 (PVQ limitations) · [R24] Adank, van Hout, Smits (2001), Eurospeech · [R30] Genette, Verhoeven, Gillis (2023), ICPhS

**Fluency & liveliness:** [R21] Hincks (2005), Measuring liveliness in presentation speech, Interspeech · [R31] de Jong & Bosker (2013), DiSS · [R32] Kahng (2018), Applied Psycholinguistics · Bosker et al. (2013), Language Testing 30(2)

**Clarity:** [R22] Ferguson & Kewley-Port (2007), JSLHR — https://pubs.asha.org/doi/10.1044/1092-4388(2007/087) · [R23] Ménard et al. (2016), PLoS ONE 11(9)

**Synthetic data & realism:** [R16] Kourkounakis et al. (2021), FluentNet/LibriStutter, arXiv:2009.11394 · [R17] Mujtaba et al. (2024), NAACL · [R18] Morrison et al. (2024), Fine-Grained and Interpretable Neural Speech Editing, arXiv:2407.05471 · [R19] Saeki et al. (2022), UTMOS, arXiv:2204.02152 · Zero-Shot TTS as Golden Speech Generator, SLaTE 2025

**Explanation:** [R35] Parikh et al. (2025), SLaTE · [R36] Chen, Ma, Hirschberg (2025), Read to Hear, arXiv:2509.14187 · [R37] GSRM, arXiv:2602.13891 · [R39] AI in CALL overview, arXiv:2505.02032

**Closest work & motivation:** [R4] Wang et al. (2020), VoiceCoach, CHI, arXiv:2001.07876 · [R44] Comparative Pronunciation Assessment and Feedback with Interpretable Speech Features, SLaTE 2025 — https://www.repository.cam.ac.uk/items/092494b9-905d-4941-951b-cbd992460ef1 · [R45] Azaïs et al. (2015), Interspeech · [R46] Niebuhr, Voße, Brem (2016), Computers in Human Behavior 64

**Visualisation:** [R41] Sitaram et al. (2011), SLaTE · [R42] Mikhailava et al. (2022), StudyIntonation — https://library.iated.org/view/MIKHAILAVA2022TAI · [R43] Hardison (2005), CALICO Journal 22(2)