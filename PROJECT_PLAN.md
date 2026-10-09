# Rostrum — Contrastive Speech Analytics & Temporal Flaw Grounding

**Multimodal AI Hackathon 2026 · Track C · Submission deadline: 15 Oct 2026 (plan to submit 14 Oct)**

> *Rostrum* is a working name. Rename freely.

---

## 0. One-paragraph summary (use this on Devpost)

Rostrum evaluates **how** a speech is delivered, not what it says. We built a public, license-clean **contrastive dataset** in which the *same famous text* is delivered by many skilled speakers (the "good envelope") and mirrored by a **graded spectrum of flawed deliveries** (from near-perfect to botched), with **exact, machine-generated flaw timestamps**. Given a new recording and its transcript, Rostrum forced-aligns every word, extracts speaker-normalised acoustic features (F0 in semitones, energy, speech rate, pauses, prominence, spectral clarity), compares each word and pause against the distribution of skilled deliveries, **grounds each flaw to a start/end timestamp**, and explains it with the **numbers that caused it** ("4.1σ faster than 15 skilled readers on 'we can not dedicate'; the 0.8 s pause before it was cut to 0.06 s"). Scores follow a published-criteria rubric (NSDA + Toastmasters) and are evaluated with sound-event-detection metrics on held-out speeches and on human-recorded flaws.

---

## 1. Problem framing — what the judges are actually scoring

| Criterion | Weight | What wins it | Where in this plan |
|---|---|---|---|
| Data Engineering & Stress Testing | **30%** | Clever, license-clean sourcing; exact labels; a smooth, realistic severity gradient | §3, §4 |
| Causal Explainability & Temporal Grounding | **25%** | Accurate onsets/offsets; explanations tied to numbers *and* text context | §7, §8 |
| Feature Extraction | **20%** | Mathematical rigour, validated pitch/alignment, stress points, energy contours | §5, §6 |
| Visualization & Dashboard | **15%** | Upload → overlay → highlighted regions → explanation cards | §10 |
| Reproducibility & Code Quality | **10%** | One-command run, Docker, pinned deps, identical outputs across runs | §11 |

**Design priority follows the weights.** 55% of the score is data + grounding. Do not over-invest in the frontend before the evaluation numbers exist.

---

## 2. Core design decisions (and why)

| # | Decision | Choice | Justification |
|---|---|---|---|
| 1 | Speech format | **Declamation** (primary), Interpretive Reading (secondary) | Everyone performs the same text → perfect for contrastive pairs. NSDA states declamation is *not* about mimicking the original speaker, so multiple valid deliveries exist [R1]. LibriVox readers are interpretive readers. |
| 2 | Baselines | **Several** good renditions per text → an *acceptable-variance envelope* | A single reference punishes valid stylistic choices; the envelope models legitimate variation. |
| 3 | Flaws | 8 labels in 6 rubric categories (§4.3) | Matches what NSDA/Toastmasters judges evaluate: voice, pace, volume, pitch, pausing [R1–R3]. |
| 4 | Detection unit | **Words + inter-word gaps** (from forced alignment), phrases for context, frames for boundary refinement | Explanations need text anchors; pause and rate flaws are defined on words/gaps. |
| 5 | Rules vs model | **Statistical rules (z-scores vs envelope) at the core**, small gradient-boosted classifier on top | Explainable + reproducible; classifier adds flaw typing and calibrated severity. |
| 6 | Speaker normalisation | F0 → semitones re speaker median; energy → dB re speaker voiced median, then z-score; MFCC → per-utterance CMVN | Required by brief; z-scores resemble perceptual normalisation [R24]; ratio-based pitch-variation measures fail for female voices with fixed thresholds [R20]. |
| 7 | Score | 6 categories × 0–10, overall = mean; plus raw descriptors and a confidence band | Interpretable, tied to published judging criteria. |
| 8 | "Good enough" | Pre-registered targets (§9.3) | Prevents post-hoc cherry-picking; judges trust it. |
| 9 | No baseline | Three explicit modes: Reference / Norm / Reject, badged in UI | Never shows confident nonsense. |

---

## 3. Phase 1 — Baseline sourcing (the "good" side)

### 3.1 Licensing rules (non-negotiable)

The dataset must be public, and flawed versions are *modified* audio. Therefore every source must permit modification and redistribution.

| Source | Status | Use |
|---|---|---|
| **LibriVox** recordings (e.g. *The Gettysburg Address 150th Anniversary*: 15 recordings by different volunteers) | Public domain; recordings state they may be reproduced, distributed or modified without permission [R5, R6] | Silver tier → builds the envelope |
| **JFK Library** White House audio, e.g. JFKWHA-001 Inaugural Address (20 Jan 1961) | Marked "Public Domain" on the archive page [R7] | Gold tier → anchor reference |
| Other US federal-government speech recordings (e.g. presidential libraries) | Works of US federal employees in official duties are generally public domain — **verify each item page** | Candidate Gold tier |
| Your own / friends' recordings | Release under **CC BY 4.0** with a signed one-line consent from each speaker | Bronze tier + human-recorded flaws |
| **TED talks** | ❌ CC BY-NC-ND 4.0 — no derivatives allowed [R8] | Do not use |
| MLK "I Have a Dream", recent political speeches, debate-championship YouTube | ❌ Copyright unclear or enforced | Do not use |
| Voice-cloning real public figures | ❌ Ethically unacceptable | Do not use |

Every audio file gets one row in `data/LICENSES.csv`:
`file_id, title, speaker, source_url, license, retrieved_on, sha256`.

### 3.2 The tier system

| Tier | What | Example | Role |
|---|---|---|---|
| **Gold** | Original orator recording | JFK inaugural (JFK Library) | Anchor shown in dashboard overlays |
| **Silver** | Multiple skilled public-domain readings of the same text | 15 LibriVox Gettysburg readers | Envelope statistics (μ, σ per word) |
| **Bronze** | Team members' careful readings | 3–5 speakers, mixed gender | Extra envelope members + matched "bad" takes |

> Gettysburg has no original recording (1863), so it is envelope-only. That is fine and should be stated.

### 3.3 Text selection

Target **4–6 texts**, each a **60–150 s segment**. Recommended core:

1. **Gettysburg Address** — LibriVox ×15 (+ team readings).
2. **JFK Inaugural Address (excerpt)** — Gold original + team readings (text is a US government work).
3. 2–4 additional public-domain speeches with LibriVox multi-version recordings (browse the LibriVox "multi-version" keyword) — **verify the license on each page before downloading**.

Mix of speakers: aim for at least 40% female voices across the envelope so the speaker-agnostic claim can be tested.

### 3.4 Transcript preparation

- Start from an authoritative text (archive transcript / LibriVox source text).
- **Edit to verbatim**: the transcript must match what is actually spoken in *that* recording (readers sometimes vary words). One transcript per recording if necessary.
- Normalise: lowercase for alignment, expand numerals ("1863" → "eighteen sixty-three"), keep punctuation in a parallel "display" version (punctuation is used later for expected-pause locations).

### 3.5 Audio preprocessing (identical for every file)

1. Convert to **16 kHz, mono, 16-bit WAV** (`ffmpeg -ac 1 -ar 16000`).
2. Trim leading/trailing silence (keep 300 ms).
3. **Loudness-normalise to −23 LUFS** (`pyloudnorm`), integrated loudness per file.
4. Store original + normalised; hash both.

---

## 4. Phase 2 — The "bad mirror" spectrum (the core of the 30%)

### 4.1 Three sources of flawed audio

| Source | Labels | Strength | Weakness |
|---|---|---|---|
| **A. Programmatic injection** into Gold/Silver/Bronze recordings | **Exact** (we create them) | Precise ground truth; controllable severity | Processing artifacts possible |
| **B. Self-recorded flawed takes** (same text, deliberate flaws) | Hand-labelled in Praat TextGrids, double-annotated subset | Realistic; covers mumbling | Labels less precise |
| **C. Composite takes** (3–5 injected flaws in one file) | Exact | Tests multi-flaw grounding | — |

Precedent: LibriStutter created a large labelled disfluency dataset by inserting synthetic disfluencies into LibriSpeech and validated realism against real stutters via spectrogram similarity [R16]. We follow the same logic for delivery flaws.

### 4.2 Injection engine — principles

- **Edit only at word boundaries** taken from the forced alignment of the clean file.
- **Splice with 10 ms raised-cosine crossfades** at zero crossings.
- **Inserted silence = room tone**, not digital zeros: sample the file's own noise floor from an existing pause and loop it. Digital silence is an artifact a detector can cheat on.
- **Time-scale and pitch modification via TD-PSOLA** (Praat): in a listening test, TD-PSOLA time-stretching scored 63.3 vs 46.5 for the WORLD vocoder and 64.0 for a neural editor [R18]. Avoid WORLD.
- **Track exact time mapping**: every edit records `(orig_t0, orig_t1) → (new_t0, new_t1)`. Recompute all later timestamps from the cumulative offset.
- **Verify**: re-run forced alignment on every edited file; label boundaries must agree with re-aligned word boundaries within 20 ms (report the distribution).
- **Naturalness check**: score every file with **UTMOS** [R19] (a neural MOS predictor) and report the drop vs the clean source per severity level. Reject injected files that drop more than a fixed threshold at L1 (near-perfect). Naturalness falls as modification magnitude grows [R18b], so this is a necessary control.
- **Deterministic**: each injection takes a seed; the seed and all parameters are stored in the label JSON.

### 4.3 Flaw taxonomy (8 labels, 6 rubric categories)

| Rubric category | Label | Definition | Injection method | Severity knob s ∈ [0,1] |
|---|---|---|---|---|
| Pace | `rushing` | Local rate far above envelope | Praat `Lengthen (overlap-add)` with factor < 1 on a 3–10-word span | factor = 1 − 0.45·s (→ 0.55) |
| Pace | `dragging` | Local rate far below envelope | Same, factor > 1 | factor = 1 + 0.8·s (→ 1.8) |
| Pausing | `missing_pause` | Rhetorical pause at a clause boundary removed | Shrink an existing boundary pause | keep (1 − 0.95·s) of the pause |
| Pausing | `hesitation_pause` | Silence inserted mid-clause | Insert room tone between words with no syntactic boundary | 0.3 + 1.7·s seconds |
| Intonation | `monotone` | F0 range compressed over a phrase | PSOLA: f′ = m·(f/m)^(1−0.9·s), m = phrase median | compression 0→90% |
| Emphasis | `misplaced_stress` | Prominence moved from content word to function word | PSOLA +F0, +gain, +duration on function word; inverse on content word | Δ = s·(+4 st, +4 dB, ×1.3) |
| Energy | `low_energy` / `fading` | Energy drops, especially at phrase ends | Smooth gain ramp across span | 0 → −18 dB·s at span end |
| Clarity | `mumbling` | Reduced articulation | **Self-recorded only** (see note) | Rated by annotators (1–3) |

**Why mumbling is not synthesised.** Clear speech differs from conversational speech through longer vowels, greater intensity and sharper vowel contrasts [R22, R23]. A low-pass filter simulates a bad microphone, not bad articulation. If a filtered proxy is added, label it `clarity_proxy` and report it separately.

**Why fillers ("um") are out of scope.** ASR systems show a consistent accuracy bias against disfluent speech [R17], making reliable filler detection a separate project.

### 4.4 Severity ladder

| Level | s | Meaning |
|---|---|---|
| L0 | 0.00 | Clean source (no edit) |
| L1 | 0.15 | "Almost perfect" — barely perceptible |
| L2 | 0.35 | Noticeable to an attentive listener |
| L3 | 0.60 | Clearly flawed |
| L4 | 0.90 | Egregious / botched |

For each clean recording × each flaw type × each level → one file with a **single** injected flaw (location chosen randomly with a fixed seed among valid spans). Plus **composite** files per clean recording at L2 and L4 with 3–5 flaws at non-overlapping locations.

**Size estimate** (6 texts, ~8 clean renditions used for injection each): 6 × 8 × 7 synthesisable labels × 4 levels ≈ 1,350 single-flaw files + ~100 composites. Quality over quantity: if UTMOS or alignment checks fail, drop files rather than keep noisy ones.

### 4.5 Human-recorded flawed takes

Each team member records each text **twice cleanly** (envelope) and **3 times flawed**:
1. Nervous rush (speed up through the middle).
2. Flat/monotone reading of one paragraph.
3. Mumbled, low-energy delivery of a segment + 2–3 hesitation pauses.

Same microphone, same room, same distance as their clean takes (**channel matching**, §6.4).

**Annotation protocol**
- Tool: Praat TextGrid, tier `flaws`, interval labels from the taxonomy + severity 1–3.
- Two annotators label ~30% independently; report inter-annotator agreement (event-F1 between annotators at 200 ms collar and Cohen's κ on labels).
- Written annotation guide with one audio example per label (in `data/ANNOTATION_GUIDE.md`).

### 4.6 Label format

Per file `labels/<file_id>.json`:

```json
{
  "file_id": "gettysburg_lv07_rushing_L3",
  "source_file": "gettysburg_lv07",
  "text_id": "gettysburg",
  "speaker_id": "librivox_LAW",
  "origin": "injected",
  "seed": 1337,
  "flaws": [
    {"label": "rushing", "t_start": 41.212, "t_end": 45.884,
     "word_idx": [52, 61], "severity_level": "L3", "s": 0.60,
     "params": {"psola_factor": 0.73}}
  ],
  "utmos": {"clean": 3.91, "edited": 3.62},
  "realign_max_boundary_err_ms": 14
}
```

Also exported as:
- `labels/events.tsv` → `filename  onset  offset  event_label` (sed_eval / sed_scores_eval format).
- Praat TextGrids with tiers `words`, `phones` (from MFA), `flaws`.

### 4.7 Data splits — leave-one-text-out

With 4–6 texts, a single fixed split is fragile. Use **leave-one-text-out cross-validation**: train/calibrate on all texts but one, test on the held-out text; repeat for every text; report mean ± sd. Human-recorded flaws are **test-only** (never used for calibration).

### 4.8 Dataset deliverables

```
dataset/
  README.md              # datasheet: motivation, composition, collection, licences, splits
  LICENSES.csv
  ANNOTATION_GUIDE.md
  texts/<text_id>.txt    # verbatim + display versions
  audio/clean/*.wav
  audio/injected/*.wav
  audio/human_flawed/*.wav
  alignments/*.TextGrid
  labels/*.json
  labels/events.tsv
  splits/leave_one_text_out.json
  checksums.sha256
```

Host audio on a **public Google Drive / Hugging Face Dataset** if GitHub size limits are hit; link it in the main README (the brief allows a public Drive link).

---

## 5. Phase 3 — Forced alignment

### 5.1 Tool choice

| Tool | Decision | Reason |
|---|---|---|
| **Montreal Forced Aligner (MFA)** | **Primary** (dataset labels + runtime) | On hand-aligned TIMIT and Buckeye, MFA outperformed both WhisperX and MMS at word alignment [R11] |
| WhisperX | Secondary — **only** to detect text deviations (skipped/repeated words) via ASR-vs-transcript edit distance | Good ASR, weaker boundaries |
| torchaudio `forced_align` | ❌ Do not use | Deprecated in 2.8 and **removed in 2.9** [R12] — a reproducibility risk |

### 5.2 MFA commands

```bash
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
# corpus/: <id>.wav + <id>.lab (same basename, transcript text)
mfa validate corpus/ english_us_arpa english_us_arpa
mfa align corpus/ english_us_arpa english_us_arpa aligned/
```

Out-of-vocabulary words reported by `validate` must be fixed (add pronunciations or expand text) before aligning.

### 5.3 Alignment validation (report in docs)

- Hand-label **50 word boundaries** across 5 files in Praat; report MFA mean absolute boundary error and 95th percentile (ms).
- Report % of boundaries within 20 ms / 50 ms.

### 5.4 Text-mismatch detection (feeds "Reject mode")

Run WhisperX ASR on the participant audio; compute WER against the provided transcript. If WER > 25% or MFA fails/returns many unaligned words → **Reject mode** (§8.5).

---

## 6. Phase 4 — Feature extraction (the 20%)

All features computed at **10 ms hop** and then aggregated to **word**, **gap** and **phrase** units.

### 6.1 Feature table

| Feature | Tool | Unit-level aggregates | Used for |
|---|---|---|---|
| **F0** | Praat autocorrelation via `parselmouth` (primary); `torchcrepe`/`penn` cross-check | mean, max, min, slope (st), range (st) | Monotone, emphasis, intonation overlay |
| **Intensity / RMS energy** | Praat intensity; `librosa.feature.rms` | mean dB, peak dB, slope | Energy, fading, emphasis |
| **Duration** | MFA word/phone intervals | word duration, normalised by phone count | Emphasis, pace |
| **Speech rate** | Phones per second from MFA (syllables from vowel phones); cross-check with de Jong & Wempe syllable-nuclei method [R14] | local rate over ±3-word window | Rushing / dragging |
| **Pauses** | Gaps between MFA words, refined with **Silero VAD** | gap duration; boundary vs mid-clause | Missing / hesitation pause |
| **Prominence** | Continuous wavelet transform (CWT) prominence over F0 + energy + duration [R25, R26] | per-word prominence | Stress points |
| **FFT / spectral** | STFT (25 ms Hann, 10 ms hop, 512-pt FFT): spectral centroid, flux, tilt (slope of log-magnitude), band energy ratios | mean per word | Clarity (same-channel only) |
| **MFCC** | `librosa` 13 MFCC + Δ + ΔΔ, CMVN | mean per word | Clarity, classifier input |
| **Voice quality** | parselmouth: HNR, jitter, shimmer, CPP | mean per voiced word | Clarity (same-channel only) |
| **Standard set** | openSMILE **eGeMAPS** [R13] | per phrase | Cited standard; classifier features |

### 6.2 Pitch extraction details

- **Two-pass speaker-adaptive range**: pass 1 with floor 50 Hz, ceiling 700 Hz → compute quartiles Q1, Q3 → pass 2 with floor = 0.75·Q1, ceiling = 1.5·Q3 (common heuristic used in prosody research). Prevents octave errors.
- Why Praat: on the PTDB-TUG benchmark, Praat had 4.8% voicing errors and 1.5% gross pitch errors, versus 33.5% voicing errors for YIN [R15].
- **Cross-check**: run `torchcrepe` and report % of voiced frames where Praat and CREPE agree within 50 cents [R27, R28]. Frames where they disagree are excluded from F0 statistics.
- Unvoiced gaps: linear interpolation in semitones for visualisation and CWT only; never for statistics.

### 6.3 Speaker normalisation (brief: "speaker agnostic")

| Signal | Transform |
|---|---|
| F0 | $\text{st}(t) = 12\log_2\big(f_0(t)/\tilde f_{0,\text{spk}}\big)$, where $\tilde f_{0,\text{spk}}$ = speaker's median F0 over the recording |
| Energy | $e(t) - \tilde e_{\text{voiced}}$ in dB, then z-scored per recording |
| Rate | Absolute syllables/s **and** relative to speaker's global articulation rate |
| MFCC / spectral | Per-utterance cepstral mean and variance normalisation |
| Pitch range (phrase) | Phrase F0 maximum minus utterance-final low F0, in semitones — more reliable than peak–valley distances [R29] |
| Pitch variation | Std of F0 in **semitones** (not the Hz ratio "PVQ", which needs different thresholds for female voices [R20]) |

Justification: z-score normalisation best resembled perceptual normalisation in listener experiments [R24]; F0 normalisation choice affects statistical sensitivity [R30].

### 6.4 The channel rule (a subtle point most teams will miss)

A 1961 tape, a LibriVox USB microphone and a phone differ in spectral tilt, HNR, bandwidth and MFCC levels for reasons that are **not delivery**. Therefore:

- **Channel-robust features** (timing, pauses, rate, semitone F0, relative energy contour, prominence) → compared against the full multi-speaker envelope.
- **Channel-sensitive features** (spectral tilt, centroid, HNR, jitter, shimmer, CPP, raw MFCC) → compared **only within the same channel**: clean vs its own injected edit, or a speaker's clean take vs their own flawed take.

State this explicitly in the documentation.

---

## 7. Phase 5 — Comparison engine & temporal grounding (the 25%)

### 7.1 Word-anchored comparison (no blind DTW needed)

Because transcripts are identical, word *i* in the participant corresponds to word *i* in every reference. Comparison happens per word and per gap.

DTW (`librosa.sequence.dtw`) is used **only inside each word** to warp frame-level curves onto the reference timeline for the overlay plot.

### 7.2 The acceptable-variance envelope

For text *T*, references $r = 1..R$ (Gold + Silver + Bronze clean), feature *k*, word or gap *w*:

$$\mu_{wk} = \operatorname{median}_r(x^{(r)}_{wk}), \qquad \sigma_{wk} = 1.4826 \cdot \operatorname{MAD}_r(x^{(r)}_{wk})$$

Median and MAD are robust to one unusual reader. A floor prevents division by tiny spreads:

$$\hat\sigma_{wk} = \max(\sigma_{wk},\ \sigma^{\text{floor}}_k)$$

where $\sigma^{\text{floor}}_k$ = 25th percentile of $\sigma_{\cdot k}$ across all words of all texts.

**Deviation of participant *p*:**

$$z_{wk} = \frac{x^{(p)}_{wk} - \mu_{wk}}{\hat\sigma_{wk}}$$

**Combined word deviation** (weights $\lambda_k$ fixed per rubric category, documented):

$$D_w = \sqrt{\textstyle\sum_k \lambda_k\, z_{wk}^2}$$

Each flaw label uses the signed z of its primary feature(s), e.g. `rushing` ⇔ $z_{w,\text{rate}} > \tau$; `dragging` ⇔ $z_{w,\text{rate}} < -\tau$; `monotone` ⇔ phrase pitch-range z $< -\tau$.

### 7.3 Expected-pause model (text modality)

Every gap gets a text label from the display transcript:
- `boundary` — after `.`, `;`, `:`, `—`, or a clause boundary (simple rule set; optional spaCy dependency parse).
- `minor` — after `,`.
- `none` — within a phrase.

Rules:
- `missing_pause`: gap label `boundary` and participant gap duration z < −τ vs envelope.
- `hesitation_pause`: gap label `none` and participant gap > 250 ms *and* z > τ. The 250–300 ms threshold follows fluency research [R31]; listeners penalise within-clause pauses most [R32].

### 7.4 Region formation (hysteresis)

1. Per label, compute the per-word signed score $z^{(\ell)}_w$.
2. Smooth with a 3-word median filter.
3. **Enter** a region when $|z| > 2.5$, **stay** while $|z| > 1.5$ (hysteresis), **merge** regions separated by < 200 ms, **drop** regions < 250 ms (except pause flaws, which are single gaps).
4. Initial region boundaries = onset of first word / offset of last word.

Thresholds τ (2.5 / 1.5) are tuned on training texts only, by maximising event-F1 under leave-one-text-out.

### 7.5 Boundary refinement (frame level)

Inside ±300 ms around each boundary, run change-point detection with **`ruptures`** (PELT search, RBF cost, fixed penalty) on the frame-level deviation signal [R33]. Keep the refined boundary only if it moves < 300 ms and lies on a frame where the deviation crosses the exit threshold. Report results **with and without refinement** (ablation).

### 7.6 Flaw-type classifier (bonus layer)

- Input per region: vector of mean z per feature, region duration, gap label, position in phrase.
- Model: **gradient-boosted trees** (`sklearn.ensemble.HistGradientBoostingClassifier`, fixed `random_state`).
- Outputs: flaw label + calibrated severity (isotonic regression on training severities).
- Trained on injected flaws of training texts; tested on held-out text and human-recorded flaws.

### 7.7 Why there is no LLM in the judging loop

A SLaTE 2025 evaluation found an instruction-tuned audio LLM over-predicts low-quality speech and lacks precision in error detection [R35]. Approaches that make LLM feedback trustworthy first extract interpretable acoustic measurements and only then reason over them [R36, R37]. Rostrum follows that: **DSP computes every number and timestamp**. An LLM is optional and may only *rephrase* the template (temperature 0, cached output committed to the repo).

---

## 8. Phase 6 — Scoring & causal explanation

### 8.1 Rubric (mapped to real judging criteria)

| Category | Toastmasters / NSDA anchor [R1–R3] | Primary features |
|---|---|---|
| Pace | "speed" | local rate z |
| Pausing | "effective use of pauses" | gap z × gap label |
| Intonation | "pitch / tone" | phrase pitch range z, F0 std (st) |
| Emphasis | "vocal variety to emphasise" | word prominence profile vs envelope |
| Energy | "volume" | relative energy contour z |
| Clarity | "clarity" | same-channel spectral/voice-quality z |

### 8.2 Category score

Deviation mass for category *c*, over flagged regions *j*:

$$M_c = \frac{1}{T_{\text{speech}}}\sum_{j \in c} \text{dur}_j \cdot \overline{\big(|z| - 1.5\big)_+}_j$$

$$S_c = 10 \cdot e^{-k_c M_c}$$

$k_c$ is calibrated on training texts so that single-flaw L2 files score ≈ 6/10 in that category. **Overall** = mean of the six $S_c$ (equal weights, stated as a design choice).

### 8.3 Raw descriptors shown beside the score

- Articulation rate (syll/s) and its envelope range.
- Pitch variation (st) — liveliness ratings correlated with normalised F0 variation [R21].
- Mean length of runs (syllables between pauses > 250 ms) [R21].
- Pause count per minute (boundary vs mid-clause).

### 8.4 Confidence band

Recompute every score with **leave-one-reference-out** (drop each envelope member once). Show `7.2 ± 0.4` (± = std across the R recomputations). Fully deterministic.

### 8.5 Operating modes

| Mode | Trigger | Behaviour |
|---|---|---|
| **Reference** | Transcript matches a library text | Full envelope comparison |
| **Norm** | Unknown text | Compare against pooled norms from all good renditions (rate band, pitch range, pause norms by gap label) + within-speaker change points; UI badge "norm-based · lower confidence" |
| **Reject** | WER > 25% or alignment failure | Clear message: audio and transcript do not match |

(A zero-shot TTS "golden" reference has been studied for assessment [R38]; it can be an optional, clearly labelled low-confidence add-on to Norm mode.)

### 8.6 Explanation object (structured → human-readable)

Feedback that explains *what* and *how to fix* is more effective than feedback that only flags an error [R39]. Every region produces:

```json
{
  "region_id": 3,
  "t_start": 41.27, "t_end": 45.83,
  "words": "we can not dedicate we can not consecrate",
  "label": "rushing", "severity": "L3", "confidence": 0.86,
  "evidence": [
    {"feature": "articulation_rate_syll_s", "participant": 6.4,
     "envelope_median": 3.9, "envelope_sigma": 0.6, "z": 4.1},
    {"feature": "gap_before_s", "participant": 0.06,
     "envelope_median": 0.82, "envelope_sigma": 0.21, "z": -3.6,
     "gap_label": "boundary"}
  ],
  "context": "Parallel structure (anaphora); skilled readers slow down and pause before it.",
  "rationale": "Rate is 4.1σ above 15 skilled readings and the boundary pause was cut from 0.82 s to 0.06 s.",
  "fix": "Slow to ~4 syllables/s and hold a ~0.8 s pause before 'we can not dedicate'."
}
```

Rendered sentence template:
> **[41.27–45.83 s] Pace — rushing (L3).** On "*we can not dedicate…*", you spoke at **6.4 syll/s** vs **3.9 ± 0.6** for skilled readers (**z = +4.1**), and the pause before it dropped from **0.82 s → 0.06 s** (z = −3.6). **Fix:** slow to ~4 syll/s and hold the pause.

The `context` field comes from text annotations (gap labels, punctuation, simple rhetorical tags such as repetition/anaphora detected by repeated n-grams) — this is the "linking acoustic deviation to context" the 25% criterion asks for.

---

## 9. Phase 7 — Evaluation protocol (your own benchmark)

### 9.1 Metrics

| What | Metric | Tool |
|---|---|---|
| Temporal grounding | **Event-based F1** with 200 ms onset collar; offset collar = max(200 ms, 20% of event length) (DCASE convention) [R9, R34] | `sed_eval` |
| Threshold-independent detection | Precision–recall curve / PSDS across all thresholds [R10] | `sed_scores_eval` |
| Boundary accuracy | Onset/offset MAE (ms), temporal IoU | custom |
| Flaw typing | Macro-F1, confusion matrix | `sklearn` |
| Severity gradient | Spearman ρ between severity level and category score; per text and pooled | `scipy` |
| Speaker-agnostic | (a) false-positive regions/min on clean-vs-envelope; (b) score gap for same flaw on male vs female voices | custom |
| Synthetic → human transfer | All of the above on human-recorded flaws | — |
| Data realism | UTMOS drop per severity level; synthetic vs human feature-distribution similarity | UTMOS |
| Component validity | MFA boundary error (§5.3); Praat–CREPE F0 agreement (§6.2); annotator agreement (§4.5) | — |
| Reproducibility | SHA-256 of output JSON identical across 2 runs + Docker run | `make reproduce` |

### 9.2 Ablations (judges love these)

1. **Single reference vs envelope** → expect fewer false positives with the envelope.
2. Rules only vs rules + classifier.
3. With vs without boundary refinement.
4. With vs without speaker normalisation (shows *why* normalisation matters).
5. Synthetic-only test vs human test.

### 9.3 Pre-registered targets

| Metric | Synthetic (held-out text) | Human-recorded |
|---|---|---|
| Event-F1 @ 200 ms | ≥ 0.75 | ≥ 0.50 |
| Onset/offset MAE | ≤ 150 ms | ≤ 250 ms |
| Flaw-type macro-F1 | ≥ 0.70 | ≥ 0.50 |
| Spearman ρ (severity ↑ → score ↓) | ≤ −0.85 | ≤ −0.70 |
| False positives on clean | ≤ 1 region/min | ≤ 1 region/min |
| Male vs female gap (same flaw) | ≤ 0.5 points | — |
| UTMOS drop at L1 | ≤ 0.3 | — |
| Repeat-run output hash | identical | identical |

Report actual numbers honestly, including misses, plus a **"Where it fails"** section with 3 failure examples and their causes.

---

## 10. Phase 8 — Dashboard

### 10.1 Stack

- **Backend:** FastAPI (`/analyze` → returns analysis JSON; `/clip` → returns an audio snippet).
- **Frontend:** **Streamlit + Plotly** first (fast, reliable). Upgrade the waveform panel to **wavesurfer.js** (regions plugin) only if time remains.
- **Deploy:** Hugging Face Space (Docker) for a public demo link.

### 10.2 Screens

1. **Upload** — audio (wav/mp3/m4a) + transcript (txt or paste) — *both required by the brief*; or "choose a library speech" and "choose a sample flawed file".
2. **Processing** — step indicators: normalise → align → features → compare → explain.
3. **Analysis view**
   - **Waveform** with coloured flaw regions (colour = category).
   - **Time-series overlay** (Plotly, shared x-axis): participant vs envelope **median line + shaded ±1σ band** for (a) F0 in semitones, (b) relative energy, (c) local speech rate. Envelope curves are time-normalised per word so that multiple readers can be averaged, as in ProsodyPro [R40]; the participant curve is DTW-warped onto the reference timeline within each word.
   - **Transcript** with each word coloured by deviation $D_w$ — mapping prosody onto displayed text follows Project LISTEN's reading tutor [R41].
   - **Explanation cards** (one per region): timestamp, label, severity, evidence table, rationale, fix, **A/B playback** (participant snippet vs best-matching Gold/Silver snippet).
   - **Rubric panel**: radar chart of 6 categories + overall score with confidence band.
4. **Export** — JSON report + PDF summary.

Design evidence: plotting a learner's pitch contour against a model contour (DTW-aligned) is an established feedback technique [R42]; discourse-level context transfers better than isolated sentences [R43], so flaws are always shown in the context of the whole speech.

---

## 11. Phase 9 — Reproducibility & code quality

### 11.1 Repository layout

```
rostrum/
  README.md                 # quickstart, results table, links (dataset, demo, video)
  Makefile                  # make setup | data | align | features | train | eval | app | reproduce
  Dockerfile                # condaforge/miniforge3 base; MFA + pinned pip deps
  environment.yml           # conda (MFA) + pinned versions
  requirements.lock         # pip freeze of the final environment
  configs/
    injection.yaml          # all severity parameters + seeds
    detection.yaml          # thresholds, weights λ_k, k_c
  src/rostrum/
    io/                     # loading, resampling, loudness normalisation
    inject/                 # PSOLA ops, splicing, room tone, label writer
    align/                  # MFA wrapper, WhisperX mismatch check
    features/               # pitch, energy, rate, pauses, prominence, spectral, egemaps
    envelope/               # robust μ/σ per word & gap
    detect/                 # z-scores, hysteresis, ruptures refinement, classifier
    score/                  # rubric, confidence band
    explain/                # evidence → templates → cards
    api/                    # FastAPI
  app/                      # Streamlit dashboard
  eval/                     # metrics, ablations, figures
  tests/                    # unit tests + determinism test
  docs/TECHNICAL_REPORT.pdf # ≤ 6 pages
```

### 11.2 Determinism checklist

- Python and every dependency **pinned**; MFA version pinned in `environment.yml`.
- All seeds (`random`, `numpy`, injection, classifier `random_state`) in config.
- No network calls at inference; models downloaded at build time.
- LLM (if used) outputs cached and committed.
- **CI determinism test**: run the pipeline on 3 files twice; assert identical SHA-256 of output JSON.
- `make reproduce` regenerates every number in the report from raw data.

### 11.3 Code quality

Type hints, `ruff` + `black`, docstrings on public functions, unit tests for injection timestamp mapping (most bug-prone part), feature extractors on synthetic signals (e.g., a known sine F0 must be recovered), and region merging logic.

---

## 12. Deliverables checklist (from the brief)

| Deliverable | Content | Done |
|---|---|---|
| GitHub link | Code + README + docs | ☐ |
| Custom dataset | Paired good/bad audio, transcripts, alignment labels, datasheet, licences (GitHub or public Drive link in README) | ☐ |
| Interactive dashboard | Upload → processing → overlay → flaw regions → explanations | ☐ |
| Technical documentation (≤ 6 pages) | Dataset construction, rubrics, architecture, scoring | ☐ |
| YouTube video (3–10 min) | Dataset collection, stress testing across qualities, dashboard catching deviations | ☐ |

### 12.1 Technical report outline (6 pages)

1. **Introduction & related work** (0.75 p) — subjectivity in judging; VoiceCoach and its call for bad examples [R4]; comparative feedback via alignment error [R44]; public-speaking prosody research [R45, R46].
2. **Dataset** (1.5 p) — sourcing tiers & licences, injection engine, severity ladder, human takes, annotation agreement, UTMOS realism, splits.
3. **Features & normalisation** (1 p) — table of features, two-pass pitch, channel rule, validation numbers.
4. **Grounding, scoring & explanation** (1.25 p) — envelope equations, hysteresis, refinement, rubric formula, explanation schema.
5. **Results** (1 p) — targets vs achieved, ablations, severity curves, failure cases.
6. **Reproducibility & limitations** (0.5 p).

### 12.2 Video script (≈ 7 min)

| Time | Segment |
|---|---|
| 0:00–0:30 | Hook: A/B playback of a rushed line, the dashboard pinpointing it |
| 0:30–2:00 | Dataset: 15 Gettysburg readers → envelope; injection L0→L4 audio ladder; human takes; label files |
| 2:00–3:30 | Pipeline: alignment → features → envelope z-scores → regions → explanation |
| 3:30–5:30 | Live dashboard on a **human-recorded** flawed take (not synthetic) |
| 5:30–6:30 | Stress-test results: severity curves, event-F1, ablations, speaker-agnostic test |
| 6:30–7:00 | Limitations & reproducibility (`make reproduce`) |

---

## 13. Timeline (2 → 15 Oct)

| Date | Phase | Exit criterion |
|---|---|---|
| Oct 2–3 | Discord rules check; sourcing + licences; transcripts; **MFA running** | 4–6 texts aligned; LICENSES.csv filled |
| Oct 4–5 | Injection engine + label writer + re-alignment check + UTMOS; record human takes | L0–L4 files for 2 texts; human takes recorded |
| Oct 6 | Annotation of human takes (double-annotate 30%) | TextGrids + agreement score |
| Oct 7–8 | Feature extraction + normalisation + envelope | Features cached for all files; pitch cross-check |
| Oct 9–10 | Detection, refinement, scoring, explanations; eval harness | First full metric table |
| Oct 11 | Dashboard (Streamlit + Plotly) | Upload → analysis works end-to-end |
| Oct 12 | Leave-one-text-out runs, ablations, figures; write report | Final numbers frozen |
| Oct 13 | Docker + HF Space deploy; record video | Public demo link; video uploaded |
| Oct 14 | README polish, Devpost write-up, **submit** | Submitted |
| Oct 15 | Buffer | — |

---

## 14. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| MFA installation issues | Blocks everything | Install on day 1 in conda; fall back to a Docker image with MFA |
| PSOLA artifacts make detection trivial | Inflated synthetic scores | UTMOS check; human-recorded test set; report synthetic vs human separately |
| Too few texts for robust stats | Unstable results | Leave-one-text-out CV; report mean ± sd |
| Channel confounds (old tapes vs phones) | False clarity flaws | Channel rule (§6.4) |
| Alignment drift on flawed audio | Wrong timestamps | Re-alignment check; report boundary errors |
| Licence doubts on a source | Disqualification risk | Only PD / CC BY sources with archive-page proof in LICENSES.csv |
| Dashboard eats time | Lower core scores | Streamlit first; wavesurfer only if ahead of schedule |

---

## 15. References

**Judging criteria**
- [R1] National Speech & Debate Association — How to Judge Declamation. https://www.speechanddebate.org/?p=1070
- [R2] NSDA Declamation ballot (comment sheet). https://newmexicoforensicsclub.readthedocs.io/en/latest/_downloads/8016c624c36316fb5f0f70dd9ef71298/nsda-declamation-ballot.pdf
- [R3] Toastmasters International — Evaluation resources (vocal variety rubric; speech profile incl. pausing). https://content.toastmasters.org/image/upload/8053-generic-evaluation-resource.pdf ; https://ccdn.toastmasters.org/medias/files/department-documents/education-documents/evaluation-resources/english/8104e2-speech-profile-ff.pdf

**Motivation / closest work**
- [R4] Wang, X. et al. (2020). VoiceCoach: Interactive Evidence-based Training for Voice Modulation Skills in Public Speaking. CHI 2020. arXiv:2001.07876
- [R44] Comparative Pronunciation Assessment and Feedback with Interpretable Speech Features. SLaTE 2025 (Univ. of Cambridge). https://www.repository.cam.ac.uk/items/092494b9-905d-4941-951b-cbd992460ef1
- [R45] Azaïs, L. et al. (2015). Does my Speech Rock? Automatic Assessment of Public Speaking Skills. Interspeech 2015. https://www.isca-archive.org/interspeech_2015/azais15_interspeech.pdf
- [R46] Niebuhr, O., Voße, J., Brem, A. (2016). What makes a charismatic speaker? A computer-based acoustic-prosodic analysis of Steve Jobs tone of voice. Computers in Human Behavior 64, 366–382.

**Data sources & licences**
- [R5] LibriVox — The Gettysburg Address 150th Anniversary (15 readings). https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/
- [R6] LibriVox public-domain statement (example cover). https://www.archive.org/download/LibrivoxCdCoverArt12/Gettysburg_Address_V3_1109.pdf
- [R7] JFK Library — Swearing-in Ceremony and Inaugural Address, 20 Jan 1961 (JFKWHA-001). https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001
- [R8] TED Talks Usage Policy (CC BY-NC-ND 4.0). https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy

**Evaluation methodology**
- [R9] Mesaros, A., Heittola, T., Virtanen, T. (2016). Metrics for Polyphonic Sound Event Detection. Applied Sciences 6(6):162. (sed_eval)
- [R10] Ebbers, J. et al. (2022). Threshold-Independent Evaluation of Sound Event Detection Scores. ICASSP 2022. arXiv:2201.13148 (sed_scores_eval)
- [R34] DCASE 2020 Task 4 evaluation (event-F1 with 200 ms collar). arXiv:2011.00801

**Alignment**
- [R11] Rousso, R., Cohen, E., Keshet, J., Chodroff, E. (2024). Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment. Interspeech 2024. arXiv:2406.19363
- [R12] PyTorch Audio — `torchaudio.functional.forced_align` deprecation/removal notice. https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.forced_align.html
- McAuliffe, M. et al. (2017). Montreal Forced Aligner. Interspeech 2017. https://montreal-forced-aligner.readthedocs.io
- Bain, M. et al. (2023). WhisperX. arXiv:2303.00747

**Features**
- [R13] Eyben, F. et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS). IEEE Trans. Affective Computing 7(2).
- [R14] de Jong, N. H., Wempe, T. (2009). Praat script to detect syllable nuclei and measure speech rate automatically. Behavior Research Methods 41(2).
- [R15] Strömbergsson, S. — Comparison of Praat, getF0 and YIN pitch extraction on PTDB-TUG. https://cdn.dal.ca/content/dam/dalhousie/pdf/sites/icpla/postersjune17/61724.PDF.lt_9ddeaa1253afb7974ae5ffe9e8637664.res/61724.PDF
- [R25] Suni, A., Aalto, D., Vainio, M. (2015). Hierarchical Representation of Prosody for Statistical Speech Synthesis. arXiv:1510.01949
- [R26] Wavelet Prosody Toolkit. https://github.com/asuni/wavelet_prosody_toolkit
- [R27] Kim, J. W. et al. (2018). CREPE: A Convolutional Representation for Pitch Estimation. arXiv:1802.06182
- [R28] Morrison, M. et al. (2023). Cross-domain Neural Pitch and Periodicity Estimation (PENN). arXiv:2301.12258
- [R29] den Ouden, H., Terken, J. (2001). Measuring pitch range. Eurospeech 2001.
- [R40] Xu, Y. (2013). ProsodyPro — a tool for large-scale systematic prosody analysis. TRASP 2013.
- Jadoul, Y., Thompson, B., de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics 71.
- Truong, C., Oudre, L., Vayatis, N. (2020). [R33] Selective review of offline change point detection methods. Signal Processing. arXiv:1801.00718 (ruptures)
- de Seyssel, M. et al. (2024). EmphAssess: a Prosodic Benchmark on Assessing Emphasis Transfer in Speech-to-Speech Models. EMNLP 2024. (word-level emphasis classifier; optional cross-check)

**Normalisation**
- [R20] US Patent 11,205,418 — Monotone speech detection (limits of PVQ across sexes). https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11205418
- [R24] Adank, P., van Hout, R., Smits, R. (2001). A comparison between human vowel normalization strategies and acoustic vowel transformation techniques. Eurospeech 2001.
- [R30] Genette, J., Verhoeven, J., Gillis, S. (2023). Fundamental frequency normalization and statistical power: an assessment of 15 normalizing techniques. ICPhS 2023.

**Fluency, pauses, liveliness**
- [R21] Hincks, R. (2005). Measuring liveliness in presentation speech. Interspeech 2005.
- [R31] de Jong, N. H., Bosker, H. R. (2013). Choosing a threshold for silent pauses to measure second language fluency. DiSS 2013.
- [R32] Kahng, J. (2018). The effect of pause location on perceived fluency. Applied Psycholinguistics.
- Bosker, H. R. et al. (2013). What makes speech sound fluent? The contributions of pauses, speed and repairs. Language Testing 30(2).

**Clarity**
- [R22] Ferguson, S. H., Kewley-Port, D. (2007). Talker differences in clear and conversational speech: Acoustic characteristics of vowels. JSLHR. https://pubs.asha.org/doi/10.1044/1092-4388(2007/087)
- [R23] Ménard, L. et al. (2016). Speaking Clearly for the Blind: Acoustic and Articulatory Correlates of Speaking Conditions. PLoS ONE 11(9).

**Synthetic data & realism**
- [R16] Kourkounakis, T. et al. (2021). FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning (LibriStutter). arXiv:2009.11394
- [R17] Mujtaba, D. et al. (2024). Lost in Transcription: Identifying and Quantifying the Accuracy Biases of ASR Systems Against Disfluent Speech. NAACL 2024.
- [R18] Morrison, M. et al. (2024). Fine-Grained and Interpretable Neural Speech Editing. arXiv:2407.05471
- [R18b] Pseudo-Cepstrum: Pitch Modification for Mel-Based Neural Vocoders. arXiv:2512.16519
- [R19] Saeki, T. et al. (2022). UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022. arXiv:2204.02152
- [R38] Zero-Shot Text-to-Speech as Golden Speech Generator: A Systematic Framework and its Applicability in Automatic Pronunciation Assessment. SLaTE 2025.

**Explanation & LLM grounding**
- [R35] Parikh, A. K. et al. (2025). Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech: Challenges and Opportunities. SLaTE 2025.
- [R36] Chen, Y.-W., Ma, M., Hirschberg, J. (2025). Read to Hear: A Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs. arXiv:2509.14187
- [R37] Generative Speech Reward Model (GSRM) with acoustic-feature-grounded reasoning. arXiv:2602.13891
- [R39] An overview of artificial intelligence in computer-assisted language learning (feedback explicitness). arXiv:2505.02032

**Visualisation**
- [R41] Sitaram, S. et al. (2011). What Visual Feedback Should a Reading Tutor Give Children on their Oral Reading Prosody? SLaTE 2011.
- [R42] Mikhailava, V. et al. (2022). StudyIntonation: tailoring multimodal prosody feedback. https://library.iated.org/view/MIKHAILAVA2022TAI
- [R43] Hardison, D. M. (2005). Contextualized Computer-based L2 Prosody Training. CALICO Journal 22(2).

---

## 16. Final pre-submission checklist

- [ ] Every audio file has a licence row with a source URL
- [ ] Consent recorded for every team voice released
- [ ] Label timestamps verified by re-alignment (report included)
- [ ] UTMOS realism table included
- [ ] Leave-one-text-out results + ablations in the report
- [ ] Human-recorded test results reported separately from synthetic
- [ ] Dashboard accepts **audio + transcript upload** and shows overlay + highlighted regions + explanations
- [ ] `make reproduce` works from a fresh clone inside Docker
- [ ] README links: dataset, live demo, video, report
- [ ] Report ≤ 6 pages; video 3–10 min
- [ ] Devpost submitted before the deadline