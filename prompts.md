# CLAUDE_PROMPTS.md — Build Rostrum phase by phase with Claude

## How to use these prompts
1. Put `CLAUDE.md` and `MASTER_SPEC_PLAN.md` in the repo root **before** the first prompt. Claude Code reads `CLAUDE.md` automatically.
2. Run prompts **in order**. Each one ends with acceptance criteria; don't move on until they pass.
3. One prompt per session/turn. If a session gets long, start fresh and say: *"Continue from Prompt NN; read CLAUDE.md and MASTER_SPEC_PLAN.md §X first."*
4. Things only **you** can do are marked **👤 HUMAN**: downloading from websites, recording audio, annotating, uploading to Drive/YouTube/Devpost.
5. After each prompt: `git add -A && git commit -m "phase NN: <title>"`.

---

## PHASE A — Foundation

### Prompt 00 — Repo bootstrap
```
Read CLAUDE.md and MASTER_SPEC_PLAN.md fully. Then bootstrap the repository exactly as in
MASTER_SPEC_PLAN.md §17:

1. Create the directory tree (src/rostrum/... , app/, eval/, scripts/, tests/, configs/,
   dataset_sample/, docs/figures/, results/).
2. pyproject.toml (package "rostrum", Python 3.11), requirements.txt with these deps (pin
   exact versions after resolving): numpy, scipy, pandas, soundfile, librosa, pyloudnorm,
   praat-parselmouth, tgt (TextGrid I/O) or praatio, silero-vad, torch (CPU), torchcrepe,
   opensmile, ruptures, scikit-learn, sed_eval, sed_scores_eval, faster-whisper, pyyaml,
   fastapi, uvicorn, python-multipart, streamlit, plotly, kaleido, pytest, ruff, black.
3. environment.mfa.yml: a SEPARATE conda env named "mfa" with montreal-forced-aligner from
   conda-forge (pin the version you resolve).
4. Makefile with targets: setup, data, features, envelope, eval, ablations, figures, app,
   test, lint, reproduce, demo (as described in CLAUDE.md). Stub targets may echo TODO.
5. configs/: paths.yaml, injection.yaml (severity levels L0–L4 and per-label parameter
   formulas from spec §5.2–5.3), features.yaml (hop 0.01, win 0.025, pitch two-pass
   settings), detection.yaml (tau_in 2.5, tau_out 1.5, merge 0.2 s, min 0.25 s, refine
   window 0.3 s), scoring.yaml (category→labels map), templates.yaml (empty placeholders).
6. src/rostrum/types.py with dataclasses: Word, Gap, Alignment, Envelope, Region,
   AnalysisResult (fields per spec §17), all JSON-serialisable via to_dict()/from_dict().
7. src/rostrum/config.py: load YAMLs, set global seeds (PYTHONHASHSEED note, numpy,
   random, torch), expose a frozen Config object.
8. tests/test_types.py (round-trip serialisation) and a GitHub Actions workflow running
   ruff, black --check, pytest.
9. .gitignore excluding dataset/ (large audio), caches, .venv.

Acceptance: `make test` and `make lint` pass; `python -c "import rostrum"` works;
tree matches the spec. Print the final tree.
```

### Prompt 01 — MFA environment smoke test
```
Goal: prove forced alignment works before anything else (spec §8).

1. Write src/rostrum/align/mfa.py:
   - prepare_corpus(items: list[(wav_path, transcript, file_id)], corpus_dir) → writes
     <id>.wav + <id>.lab (lowercase, punctuation stripped, numerals expanded via a small
     helper you write; keep a mapping to original display tokens).
   - run_mfa(corpus_dir, out_dir, acoustic="english_us_arpa", dictionary="english_us_arpa",
     extra_dict=None) → calls `conda run -n mfa mfa align ...` with --clean and a fixed
     number of jobs; raises a clear error with stderr on failure.
   - validate(corpus_dir) → runs `mfa validate` and parses OOV words into a list.
   - read_textgrid(path) → Alignment(words=[Word...], gaps=[Gap...]) where gaps are the
     intervals between consecutive words (including leading/trailing excluded).
2. FIRST run `conda run -n mfa mfa version` and `conda run -n mfa mfa model download
   acoustic --help` and confirm the current model names; if "english_us_arpa" changed,
   update configs and tell me.
3. scripts/smoke_align.py: synthesises nothing — instead takes one WAV + TXT I provide in
   dataset_sample/smoke/ and aligns it, printing the first 10 words with times.
4. tests/test_mfa_parse.py: parse a tiny hand-written TextGrid fixture.

Acceptance: alignment of the smoke file succeeds; printed timestamps are monotonic; test
passes. Report the MFA version and model names you verified.
```
👤 HUMAN: put any short WAV of yourself reading 2 sentences + its TXT into `dataset_sample/smoke/` before running.

---

## PHASE B — Dataset (30% of the score)

### Prompt 02 — Source manifest & downloader
```
Implement dataset sourcing per MASTER_SPEC_PLAN.md §4. Do NOT invent URLs.

1. Create dataset/SOURCES.yaml with entries for S1 (LibriVox Gettysburg 150th Anniversary,
   15 readers), S1b (LibriVox Gettysburg version 2), S2 (JFK Library JFKWHA-001),
   S3 (Wikimedia Commons Roosevelt_Pearl_Harbor.ogg), S4 (optional Lincoln 2nd Inaugural,
   LibriVox American Civil War Collection Vol 1 ch.1). For each: page_url (from the spec),
   licence text as stated on the page, and an empty `files:` list with fields
   {reader, download_url, excerpt_start_s, excerpt_end_s}. Leave download_url as
   "TODO(human)" — I will paste the direct audio links after opening each page.
2. scripts/download_sources.py: reads SOURCES.yaml, downloads each file with retries,
   stores raw audio in dataset/raw/<source_id>/, computes SHA-256, refuses any entry whose
   licence field is empty or contains "ND", "NoDerivatives" or "TED".
3. src/rostrum/data/manifest.py: create/append LICENSES.csv rows
   (file_id,title,speaker,source_url,licence,retrieved_on,sha256) and MANIFEST.csv.
4. scripts/check_provenance.py: fails if any audio in dataset/ lacks a LICENSES.csv row or
   a provenance file in dataset/provenance/.

Acceptance: running the downloader with TODO urls exits with a clear list of what the human
must fill; provenance checker passes on an empty dataset; unit test for the licence filter.
```
👤 HUMAN: open each page, copy the **direct audio download links** into `SOURCES.yaml`, save a PDF/screenshot of each licence statement into `dataset/provenance/`, then run `python scripts/download_sources.py`.

### Prompt 03 — Transcripts & text processing
```
Implement src/rostrum/text/ per spec §7.2, §10.3, §12.

1. dataset/texts/<text_id>.json with fields verbatim, display. I will paste authoritative
   texts; create the files with placeholders and a loader.
2. text/normalize.py: display → alignment tokens (lowercase, strip punctuation, expand
   numbers like "1863"→"eighteen sixty three", "4"→"four"), returning a token list and a
   mapping token_idx → display span.
3. text/gaps.py: label each inter-word gap as "boundary" (after . ; : ! ? — or end of
   sentence/clause), "minor" (after ,), or "none". Optional: if spaCy is available, upgrade
   "none" to "boundary" at clause boundaries (subordinate/coordinate clause starts); make
   spaCy optional with a clean fallback.
4. text/rhetoric.py: detect repeated n-grams (n=2..5) across the text to tag anaphora /
   parallelism spans; return spans with type and the repeated n-gram.
5. Per-recording verbatim overrides: dataset/texts/overrides/<file_id>.txt — if a reader
   deviates from the canonical text, the override is used for alignment and the word-level
   mapping to the canonical text is computed by edit-distance alignment (so envelope
   statistics only use matched words).

Tests: numeral expansion, gap labelling on a sentence with commas and a full stop,
anaphora detection on "we can not dedicate -- we can not consecrate -- we can not hallow".
Acceptance: tests pass; print gap labels for the Gettysburg text.
```
👤 HUMAN: paste the official texts (Gettysburg — "Bliss copy"; JFK and FDR transcripts from the JFK Library / National Archives pages) into the placeholders. Listen to each LibriVox reading and create an override if the reader changed words.

### Prompt 04 — Preprocessing & clean alignment
```
Implement spec §4.6 and align all clean files.

1. io/audio.py: load_any(path) via soundfile/librosa (ffmpeg fallback), to_mono_16k,
   trim_excerpt(start,end), keep_padding(0.3 s), loudness_normalize(-23 LUFS, peak -1 dBFS),
   write_wav16, sha256.
2. scripts/build_clean.py: for every file in SOURCES.yaml → processed WAV at
   dataset/audio/clean/<file_id>.wav, MANIFEST.csv row (tier gold/silver/bronze,
   origin clean), LICENSES.csv row.
3. Align all clean files with MFA (batch), store TextGrids in dataset/alignments/.
4. scripts/align_report.py: per file → n_words, n_unaligned, mean word duration, longest
   gap; flag files where >2% words are unaligned.

Acceptance: all clean files aligned; report shows no flagged files (or lists them for me to
fix with overrides/dictionary entries).
```
👤 HUMAN: record team clean takes (spec §4.4) into `dataset/raw/team/`, add them to `SOURCES.yaml` with licence "CC BY 4.0 (consent in CONSENT.md)", and fill `CONSENT.md`.

### Prompt 05 — Injection operations (PSOLA etc.)
```
Implement src/rostrum/inject/ops.py per spec §5.1–5.2. Every op signature:
op(audio: np.ndarray, sr: int, alignment: Alignment, span, params, rng) ->
   (new_audio, TimeMap)
where TimeMap maps any original time t to the new time (piecewise linear), with exact
segments recorded.

Ops:
1. psola_lengthen(span_words, factor): cut [t0,t1] at word boundaries, apply Praat
   "Lengthen (overlap-add)" with pitch range from the two-pass estimate, splice back with
   10 ms raised-cosine crossfades. TimeMap: before t0 identity; inside scaled; after
   shifted by (new_len - old_len).
2. shorten_gap(gap_idx, keep_frac): keep the first keep_frac of the pause (min 20 ms),
   crossfade.
3. insert_roomtone(gap_idx, dur_s): estimate room tone from the longest existing pause
   (loop with crossfades, match its RMS), insert at the gap midpoint.
4. psola_pitch_compress(phrase_words, alpha): Praat Manipulation (0.01, floor, ceiling) →
   Extract pitch tier → for every point f: f' = m * (f/m)**(1-alpha) where m = median F0
   of the span → Replace pitch tier → Get resynthesis (overlap-add). Only replace the span
   region (splice), keep the rest untouched.
5. stress_swap(function_word_idx, content_word_idx, s): on the function word +4s st F0
   (pitch tier multiply), +4s dB gain, duration ×(1+0.3s) via Lengthen; inverse on the
   content word (−4s st, −4s dB, ×(1−0.2s)). Return combined TimeMap.
6. gain_ramp(span_words, db_end): smooth (raised-cosine) gain from 0 dB to db_end dB.

Use parselmouth.praat.call for Praat commands. Before using any Praat command, verify its
exact name/arguments in the parselmouth/Praat docs you have available and note them in a
docstring.

Tests (tests/test_inject_ops.py):
- psola_lengthen factor 0.7 on a synthetic 3-word signal: later boundaries shift by the
  exact delta (±1 ms) and duration of span scales by 0.7 (±10 ms).
- insert_roomtone adds exactly dur_s (±1 ms) and inserted RMS within 3 dB of room tone.
- pitch compress alpha=0.9 reduces F0 std (semitones) in span by ≥ 70%.
- gain_ramp end level within 0.5 dB of target.
Acceptance: all tests pass; save 6 example WAVs (one per op, L3) to scratch/ and list them.
```
👤 HUMAN: listen to the 6 examples. If any sounds robotic, tell Claude which and why.

### Prompt 06 — Injection engine, labels & severity ladder
```
Implement src/rostrum/inject/engine.py per spec §5.3–5.4 and §7.

1. Span selection (seeded): valid spans per label:
   rushing/dragging: 3–10 consecutive words inside one sentence;
   missing_pause: a gap labelled "boundary" with clean duration ≥ 0.30 s;
   hesitation_pause: a gap labelled "none" with clean duration < 0.15 s;
   monotone: a phrase (between boundary gaps) of 5–15 words with ≥ 60% voiced frames;
   misplaced_stress: a phrase containing a function word (from a fixed stopword list) and
   a content word that is the phrase's most prominent word in the clean file;
   low_energy/fading: 4–12 words ending at a boundary gap.
2. Severity mapping from configs/injection.yaml (L1 0.15, L2 0.35, L3 0.60, L4 0.90).
3. generate_single(clean_file, label, level, seed) → edited WAV + label JSON (spec §7.1
   schema) with flaw times computed through the TimeMap; also return the new Alignment
   (transform clean word times through the TimeMap).
4. generate_composite(clean_file, level, n_flaws∈[3,5], seed) → non-overlapping spans,
   ≥ 1 s apart; apply ops in time order, composing TimeMaps.
5. scripts/build_injected.py: for every clean file (all tiers) × 7 labels × L1–L4 + 4
   composites, write dataset/audio/injected/*.wav and dataset/labels/*.json; deterministic
   file_id naming "<clean_id>_<label>_<level>" and "<clean_id>_composite<k>_<level>".
6. events.tsv exporter.

Tests: the same seed produces byte-identical WAV and JSON; composite flaws never overlap;
flaw times lie within the audio duration.
Acceptance: dataset generated for S1 first; print counts per label/level; then all texts.
```

### Prompt 07 — Quality control: re-alignment + UTMOS realism
```
Implement src/rostrum/inject/qc.py per spec §5.1 and §5.4 realism gates.

1. realign_check(edited_wav, transcript, expected_alignment): run MFA on the edited file;
   compute per-word |boundary error| between expected (TimeMap-transformed) and re-aligned
   times; return max, mean, p95.
2. UTMOS: load a UTMOS predictor (e.g. the SpeechMOS torch.hub "utmos22_strong" model).
   Verify the current loading API from the package README before coding; pin the version;
   run on CPU; cache scores in dataset/qc/utmos.csv.
3. Gates: reject if realign max error > 50 ms, or (level L1 and UTMOS drop > 0.3). Write
   dataset/qc/rejections.csv with reasons; mark label JSON qc.passed=false and exclude from
   MANIFEST splits.
4. scripts/qc_report.py → dataset/qc/summary.md with tables: UTMOS mean±sd per label×level,
   realign error distribution (mean, p95, % ≤ 20 ms), rejection counts.

Acceptance: summary.md generated; UTMOS decreases monotonically on average with level for
each label (report if not); rejection rate printed.
```

### Prompt 08 — Human recordings import & annotation tooling
```
Support human-recorded flawed takes per spec §6.

1. scripts/annotate_prepare.py: for each human file, align with MFA and write a TextGrid
   with tiers words, phones, and an empty "flaws" interval tier, into
   dataset/annotation/todo/.
2. scripts/annotate_import.py: read completed TextGrids from dataset/annotation/done/
   (interval labels "<label>:<1|2|3>"), validate labels against the taxonomy, convert to
   label JSON (origin "human"), and to events.tsv.
3. scripts/annotator_agreement.py: for files annotated by two people
   (done/<annotator>/<file>.TextGrid), compute event-F1 (200 ms onset collar) between
   annotators and Cohen's kappa on labels of matched events. Write dataset/qc/agreement.md.
4. Write dataset/ANNOTATION_GUIDE.md: definitions, how to mark boundaries (start at the
   onset of the first affected word, end at the offset of the last), severity 1–3 anchors,
   borderline rules, and the label list.

Acceptance: round-trip test with a fixture TextGrid; agreement script runs on fixtures.
```
👤 HUMAN: record the flawed takes (spec §6.1), annotate in Praat, have a second person annotate ≥ 30%.

### Prompt 09 — Splits & dataset packaging
```
Finalise the dataset per spec §7.3–7.4.

1. splits/loto.json: for each text_id a fold with train = other texts, test = this text;
   human files always in test of their own text only.
2. dataset/README.md datasheet: motivation, composition (counts table auto-generated),
   collection process (sources, tiers, recording setup), preprocessing, labelling (injection
   + human), QC results (embed qc/summary.md numbers), splits, licences, consent, intended
   uses, limitations (channel differences, synthetic artifacts, English only, small number
   of texts).
3. scripts/package_dataset.py: verify provenance, regenerate checksums.sha256, build
   MANIFEST.csv, create dataset_sample/ (~20 files covering every label and level + 2
   human files), and zip dataset/ into dist/rostrum_dataset_v1.zip.

Acceptance: check_provenance passes; checksums verify; dataset_sample is < 50 MB.
```
👤 HUMAN: upload `dist/rostrum_dataset_v1.zip` (and/or the unzipped folder) to Google Drive → share "Anyone with the link — Viewer" → test in an incognito window → paste the link into README.

---

## PHASE C — Features (20%)

### Prompt 10 — Pitch, energy, duration, rate, pauses
```
Implement features per spec §9.1–9.2 (hop 10 ms, win 25 ms) as pure functions returning
frame arrays + unit-level aggregates.

1. features/pitch.py: two-pass Praat pitch (pass1 50–700 Hz → Q1,Q3 → pass2 floor 0.75·Q1,
   ceiling 1.5·Q3); semitones re speaker median; torchcrepe cross-check (viterbi decoding,
   periodicity threshold) and agreement mask (|Δ| ≤ 50 cents); frames that disagree are
   NaN for statistics. Word aggregates: mean_st, max_st, slope_st_per_s, range_st.
2. features/energy.py: Praat intensity (dB) + RMS; normalise to dB re speaker voiced median
   then z-score per recording. Aggregates: mean, peak, slope_db_per_s.
3. features/rate.py: syllables = vowel phones from MFA (ARPAbet vowels); local rate over a
   ±3-word window (syll/s, excluding pauses); global articulation rate; relative rate.
   Also implement the de Jong & Wempe syllable-nuclei estimate as a cross-check (port the
   logic: intensity peaks above threshold, dips ≥ 2 dB, voiced) and report correlation.
4. features/pauses.py: gaps from alignment refined with Silero VAD speech probabilities;
   gap duration, gap label (from text/gaps.py), boundary flag.
5. features/duration.py: word duration, duration per phone.

Tests: 200 Hz synthetic tone → F0 within 1 Hz; a 0.5 s inserted silence → gap 0.50±0.02 s;
semitone conversion of 2× frequency = +12 st.
Acceptance: tests pass; scripts/feature_demo.py plots F0 (st), energy and rate for one
LibriVox reader with word boundaries (save PNG).
```

### Prompt 11 — Prominence (stress points), spectral, MFCC, voice quality, eGeMAPS
```
Implement the rest of spec §9.1.

1. features/prominence.py: continuous-wavelet-transform prominence following Suni et al.:
   composite signal = 1.0·F0(z, interpolated over unvoiced) + 0.5·energy(z) + 1.0·duration
   signal (word duration z spread over the word), CWT with a Mexican-hat or Morlet wavelet
   over scales covering ~syllable to ~phrase (document scales), per-word prominence = max
   of the combined scale-summed response within the word. If the wavelet_prosody_toolkit
   can be pip-installed cleanly, cross-check against it; otherwise document our
   implementation.
2. features/spectral.py: STFT (512-pt Hann, 10 ms hop) → spectral centroid, flux, tilt
   (slope of linear fit to log-magnitude 0–4 kHz), band-energy ratio (0–1 kHz vs 1–4 kHz).
3. features/mfcc.py: 13 MFCC + Δ + ΔΔ, per-utterance CMVN, word means.
4. features/voice_quality.py: HNR (cc), jitter local, shimmer local, CPP — per voiced word,
   NaN if < 3 voiced periods.
5. features/egemaps.py: opensmile eGeMAPSv02 functionals per phrase.
6. features/extract_all.py: one call → FeatureTable (words, gaps, phrases) + frame curves;
   cache to parquet keyed by file sha256 + config hash.

Tests: prominence ranks a synthetically louder/longer/higher word first; tilt of white
noise ≈ 0 slope relative to pink noise; caching returns identical tables.
Acceptance: `make features` builds caches for the whole dataset; print timing per minute
of audio.
```

### Prompt 12 — Validation of feature components
```
Produce the component-validity numbers the report needs (spec §14.1 "Components").

1. eval/validate_pitch.py: Praat vs CREPE agreement (% voiced frames within 50 cents) per
   tier/speaker; table + histogram.
2. eval/validate_alignment.py: read dataset/qc/hand_boundaries.csv (file_id, word_idx,
   true_onset, true_offset) that I will create by hand, compute MFA MAE, P95, % within
   20/50 ms.
3. eval/validate_rate.py: correlation between MFA-based syllable rate and de Jong & Wempe
   estimate.
Write results to results/components.json and docs/figures/.
```
👤 HUMAN: hand-label 50 word boundaries in Praat across 5 files → `dataset/qc/hand_boundaries.csv`.

---

## PHASE D — Grounding & explanation (25%)

### Prompt 13 — Envelope
```
Implement src/rostrum/envelope/build.py per spec §10.2.

- build_envelope(text_id, reference_files, feature_tables) → Envelope with per-unit
  (word idx / gap idx / phrase idx) median and 1.4826·MAD for every channel-robust feature,
  n readers, and per-unit reference values (for "17 of 19 readers paused here" context).
- sigma floor per feature = 25th percentile of sigma across all units of TRAINING texts
  only (function takes the fold).
- Word matching uses canonical word indices (respect per-recording overrides).
- Frame-level envelope curves for plotting: per word, time-normalise each reader's F0(st),
  energy and rate curves to 20 points, take median and ±1σ (MAD) per point.
- leave_one_out(envelope, r) helper for the confidence band.

Tests: with 5 identical references sigma → floor; median robust to one outlier reader.
Acceptance: envelopes built for each fold; print n per text.
```

### Prompt 14 — Z-scores, region detection, boundary refinement
```
Implement src/rostrum/detect/ per spec §10.3–10.5.

1. zscore.py: participant FeatureTable vs Envelope → signed z per unit per feature; label
   trigger signals exactly as in the spec table (rushing/dragging, missing_pause,
   hesitation_pause, monotone, misplaced_stress, low_energy/fading, mumbling). Mumbling and
   any channel-sensitive feature are evaluated ONLY when a same-channel reference exists
   (flag in the call); otherwise skipped.
2. regions.py: per label → 3-unit median smoothing → hysteresis (enter tau_in, stay
   tau_out) → merge < 200 ms → drop < 250 ms (pause labels exempt) → Region objects with
   word_idx, t0/t1 from alignment, and z_evidence (top-3 features by |z|).
3. refine.py: ruptures Pelt(model="rbf") on the frame-level deviation of the primary
   feature within ±0.3 s of each boundary; accept only if shift < 0.3 s; keep both raw and
   refined boundaries in the Region.
4. tune.py: grid-search tau_in ∈ {2.0,2.5,3.0}, tau_out ∈ {1.0,1.5,2.0}, penalty values,
   on training folds only, maximising event-F1 (200 ms collar); save chosen values per fold
   to results/tuning.json.

Tests: synthetic z sequences produce expected regions; merge/drop rules; refinement never
moves > 0.3 s.
Acceptance: run on S1 injected files; print precision/recall per label on the training
fold.
```

### Prompt 15 — Flaw-type classifier & severity calibration
```
Implement src/rostrum/detect/classifier.py per spec §10.6.

- Region feature vector: mean & max z per feature, duration, gap label one-hot, phrase
  position (start/mid/end), n words.
- HistGradientBoostingClassifier(random_state=0); isotonic regression for severity (map
  region evidence → s); fit per LOTO fold on training-text injected regions matched to
  ground truth (IoU ≥ 0.3).
- Final label policy: rule label is primary; attach clf_prob; if they disagree and
  clf_prob > 0.8, mark "uncertain_type".
- Save models per fold with joblib + config hash.
Acceptance: macro-F1 on held-out folds printed; confusion matrix PNG saved.
```

### Prompt 16 — Rubric scoring & confidence band
```
Implement src/rostrum/score/rubric.py per spec §11.

- Deviation mass M_c and S_c = 10·exp(−k_c·M_c); overall = mean of 6.
- calibrate_k(train_results): choose k_c so the median S_c of single-flaw L2 files of that
  category = 6.0 (closed form or 1-D search); store per fold.
- Raw descriptors: articulation rate, F0 std (st), mean length of runs (syllables between
  pauses > 0.25 s), pauses/min split boundary vs mid-clause.
- confidence_band(): recompute scores with leave-one-reference-out envelopes → mean ± std.
Tests: no flaws → 10.0; monotonic decrease with M_c.
Acceptance: print category scores for one clean file and its L1–L4 rushing versions
(should decrease).
```

### Prompt 17 — Causal explanation cards
```
Implement src/rostrum/explain/cards.py per spec §12.

- Build the explanation JSON exactly as the schema in the spec.
- Context generator: gap label & punctuation, repetition/anaphora spans from
  text/rhetoric.py, sentence position, and reference behaviour ("k of n skilled readers
  paused ≥ 0.4 s here", "readers' median rate here: x syll/s").
- Templates in configs/templates.yaml per label: rationale sentence + fix sentence with
  numeric placeholders; numbers formatted sensibly (rates 1 dp, seconds 2 dp, z 1 dp,
  semitones 1 dp).
- render_markdown(card) for the dashboard and PDF.
- NO LLM. (Optional flag --polish that calls an LLM is NOT to be implemented now.)
Tests: every label renders without missing placeholders; card JSON validates against a
pydantic model.
Acceptance: print 5 rendered cards from different labels.
```

### Prompt 18 — Pipeline & operating modes
```
Implement src/rostrum/pipeline.py and modes per spec §13.

analyze(audio_path, transcript, text_id=None) -> AnalysisResult:
 1. preprocess (io/audio.py)
 2. ASR WER check with faster-whisper (small/base model, CPU, fixed beam) → Reject mode if
    WER > 0.25
 3. text matching: normalised similarity to library texts ≥ 0.95 and envelope n ≥ 4 →
    Reference mode; else Norm mode
 4. MFA alignment (single-file, temp dir)
 5. features → z (Reference: vs envelope; Norm: vs pooled norms computed from all
    training references, channel-sensitive features informational only)
 6. regions → refinement → classifier → scores → CI → cards
 7. curves for plotting: participant curves DTW-warped within each word onto the reference
    timeline (librosa.sequence.dtw), envelope median ±1σ curves
 8. round outputs (times 3 dp, features 4 sig. fig.) and return.
Also: analysis_id = sha256(audio bytes + transcript + config hash); cache results.
Acceptance: analyze() runs on 3 dataset_sample files (one per mode) and writes JSON to
results/examples/.
```

---

## PHASE E — Evaluation & stress testing

### Prompt 19 — Evaluation harness (LOTO)
```
Implement eval/ per spec §14.

1. eval/metrics.py: sed_eval EventBasedMetrics(t_collar=0.2, percentage_of_length=0.2) →
   event-F1 per label & overall; sed_scores_eval threshold-independent PR/PSDS using region
   scores; onset/offset MAE and temporal IoU for matched events; macro-F1 typing; Spearman
   ρ of level vs category score; FP regions/min on clean files; male–female score gap for
   identical injected flaws (speaker sex from MANIFEST).
2. eval/run_loto.py: for each fold: fit (tuning, k_c, classifier) on train texts, run on the
   test text (injected + human separately) → results/fold_<text>.json; aggregate
   mean ± sd → results/metrics.json and a markdown table results/metrics.md that also
   shows the pre-registered targets and PASS/MISS per row.
3. Never touch test-fold data during fitting — add an assertion that fails if any test
   file_id appears in fitting inputs.
Acceptance: `make eval` produces metrics.md; print it.
```

### Prompt 20 — Ablations, figures, failure analysis
```
Implement eval/ablations.py and eval/figures.py per spec §14.2.

Ablations (each a full LOTO run): (1) single reference (Gold if available else one random
Silver, fixed seed) vs envelope; (2) rules vs rules+classifier; (3) ± boundary refinement;
(4) ± speaker normalisation; (5) synthetic vs human test. Output results/ablations.md.

Figures (docs/figures/, PNG 200 dpi + SVG):
 - severity curves: mean category score vs level L0–L4 per label (with sd)
 - event-F1 per label bar chart (synthetic vs human)
 - onset error histogram
 - UTMOS vs level per label
 - example overlay figure for the report (F0/energy/rate with envelope band and regions)
 - confusion matrix

Failure analysis: pick the 3 largest misses (FN with highest severity, FP with highest z,
worst boundary error); write results/failures.md with audio file ids, what happened, and
the suspected cause.
Acceptance: all files generated; summarise the 5 most important numbers for me.
```

---

## PHASE F — Dashboard (15%)

### Prompt 21 — FastAPI
```
Implement src/rostrum/api/server.py per spec §15.1: /analyze (multipart audio+transcript,
optional text_id; limits 20 MB / 5 min; returns AnalysisResult JSON), /clip (participant or
closest reference reader snippet as WAV), /library, /health (versions + config hash).
Pydantic models mirror types.py. CORS for localhost. Tests with FastAPI TestClient on a
sample file.
```

### Prompt 22 — Streamlit dashboard
```
Build app/streamlit_app.py per spec §15.2. Use Plotly; it must call the pipeline directly
(or the API if ROSTRUM_API_URL is set).

Layout:
 1. Sidebar: upload audio (wav/mp3/m4a/ogg) + transcript (text area or .txt upload) — both
    required — OR pick a library text and a sample file from dataset_sample. "Analyze"
    button.
 2. Processing panel with st.status steps and timings.
 3. Header: mode badge (Reference · n readers / Norm-based · lower confidence / Reject red
    banner), overall score ± CI, radar chart of 6 categories, raw descriptors.
 4. Main overlay: Plotly figure with 3 stacked subplots sharing x (F0 st, relative energy,
    local rate): participant line, envelope median line, ±1σ shaded band; flaw regions as
    vrects coloured by category with hover text; word boundary ticks on the x-axis.
 5. Coloured transcript: each word background coloured by D_w (colour-blind-safe scale);
    hover shows top z features.
 6. Region cards: sorted by time; each shows timestamp, label, level, confidence, evidence
    table, rationale, fix, and two st.audio players (participant clip vs reference clip).
    Clicking "Jump" scrolls/highlights the region in the overlay (use session_state).
 7. Export: download JSON and a PDF report (reportlab or matplotlib → PDF).
Accessibility: colour + text labels (never colour alone), readable fonts, works at
1280 px width. Handle errors with clear messages.
Acceptance: `make app` → full flow works on a dataset_sample human file; take 3
screenshots into docs/figures/.
```

---

## PHASE G — Reproducibility & deployment (10%)

### Prompt 23 — Docker, determinism, CI
```
1. Dockerfile: base condaforge/miniforge3 (pin tag); create the "mfa" env from
   environment.mfa.yml and download MFA models at build time; create main env from
   requirements.lock; pre-download faster-whisper, torchcrepe, silero-vad and UTMOS
   weights; copy repo; default CMD runs Streamlit on 0.0.0.0:7860 (Hugging Face Spaces
   port) with the API in-process.
2. tests/test_determinism.py: run analyze() twice on 3 sample files → identical SHA-256 of
   rounded JSON.
3. `make reproduce`: data QC check → features → envelopes → eval → ablations → figures →
   prints results/metrics.md. Must work offline after build.
4. GitHub Actions: lint, unit tests, determinism test (CPU), docker build (no push).
5. requirements.lock regenerated from the working env; environment.mfa.yml with pinned
   MFA version.
Acceptance: `docker build` succeeds; `docker run` serves the app; determinism test passes.
```

### Prompt 24 — Hugging Face Space deployment
```
Prepare deployment to a Hugging Face Space (Docker SDK):
- README front-matter for the Space (sdk: docker, app_port: 7860).
- Use dataset_sample/ only (no full dataset in the image).
- Keep image size reasonable; document cold-start time.
- Add a "Try a sample" button that pre-fills a human flawed take.
Give me the exact git commands to push to the Space repo.
```
👤 HUMAN: create the Space on Hugging Face, push, test the public URL logged out.

---

## PHASE H — Documentation, video, submission

### Prompt 25 — README (meets official requirements)
```
Write README.md per spec §18.1. It MUST contain: prerequisites/dependencies (OS, Docker or
conda+Python 3.11, ffmpeg, RAM/disk), setup instructions (Docker one-liner and conda path),
instructions for running (make demo, make app, make reproduce, curl example for /analyze),
links section with placeholders: LIVE_DEMO_URL, DATASET_DRIVE_URL, YOUTUBE_URL,
REPORT_PDF; dataset summary table (auto-filled from MANIFEST), architecture diagram
(mermaid), results table (from results/metrics.md), ablations, repository structure,
licences (code MIT; dataset public domain + CC BY 4.0 with per-file LICENSES.csv),
references, team section with placeholders for real full names.
Acceptance: a fresh reader can run the demo from the README alone; check every command
you list actually exists in the Makefile.
```

### Prompt 26 — Technical report (≤ 6 pages)
```
Write docs/TECHNICAL_REPORT.md per spec §18.2, using the real numbers from results/*.md,
dataset/qc/*.md and results/components.json. Sections: 1 Introduction & related work;
2 Dataset construction (sources & licences, tiers, injection ops with severity table,
human takes, annotation agreement, QC realism table, splits); 3 Features & normalisation
(feature table, two-pass pitch, channel rule, component validation); 4 Grounding,
evaluation rubric & scoring methodology (envelope equations, hysteresis, refinement,
classifier, rubric formula, confidence band, explanation schema, modes); 5 Results
(targets vs achieved, ablations, severity curves figure, failure cases); 6 Reproducibility
& limitations. Cite with the [R#] references from MASTER_SPEC_PLAN.md §22 and include a
reference list. Then convert to PDF (pandoc or weasyprint) at 11 pt and verify it is
≤ 6 pages including figures; if longer, cut prose, not results.
```

### Prompt 27 — Video script & captions
```
Write docs/VIDEO_SCRIPT.md for a ~7 minute English video following spec §18.3, with exact
narration lines, what is on screen for each line, and timestamps. Must show: dataset
collection process (sources, licences, envelope, L0→L4 audio ladder, human recordings,
labels), stress testing results across qualities, and the dashboard running live and
catching specific deviations on a human-recorded take. Also produce docs/captions.srt
matching the narration.
```
👤 HUMAN: record (OBS), upload to YouTube as **Unlisted** with the `.srt` captions, check it plays logged out.

### Prompt 28 — Devpost project page
```
Write docs/DEVPOST.md: title, tagline (≤ 200 chars), and sections Inspiration, What it
does, How we built it, Challenges, Accomplishments, What we learned, What's next, Built
with — using real results. Include the four required links (GitHub, video, dataset Drive
link, live demo). Keep it scannable; first paragraph must state the problem, the dataset,
and the headline metric.
```
👤 HUMAN: paste into Devpost, add **every teammate with real full name** (each needs a Devpost account), add links, submit before the deadline.

---

## PHASE I — Final audit

### Prompt 29 — Red-team audit before submission
```
Act as a strict hackathon judge using the Track C criteria (Data 30, Grounding 25,
Features 20, Dashboard 15, Reproducibility 10) and the official submission rules in
MASTER_SPEC_PLAN.md §1. Audit the repository:
1. Run `make test`, `make lint`, the determinism test, and `make demo` from a clean clone.
2. Check every deliverable and requirement in §1 and §21; list anything missing.
3. Check for data leakage (fitting on test folds), licence gaps (files without
   LICENSES.csv rows/provenance), hard-coded paths, missing seeds, README commands that
   don't exist, broken links/placeholders.
4. Score the project per criterion 0–10 with one-line justification, then list the top 5
   fixes ranked by score impact per hour of work.
Do not change code in this prompt; only report.
```

---

## Debugging prompts (use when stuck)

**MFA fails / OOV words**
```
MFA failed with the error below. Read the full stderr, check the MFA version's docs/help
for the command, identify whether it's corpus format, dictionary OOV, model name, or
environment. Propose the minimal fix, apply it, and re-run on the single failing file.
<paste error>
```

**Injected audio sounds robotic**
```
File <id> sounds robotic at level <L>. Diagnose: pitch-range settings for PSOLA,
crossfade length, splice not at a word boundary, room-tone mismatch, or excessive factor.
Generate 3 variants with one change each, report UTMOS for each, and recommend one.
Don't change severity mapping without telling me.
```

**Metric far below target**
```
Metric <name> on fold <text> is <value> vs target <target>. Using TRAINING folds only,
analyse errors by label and level, inspect 5 worst cases (print z evidence and
boundaries), and propose changes ranked by expected impact. Do not tune on the test fold.
```

**Non-deterministic output**
```
The determinism test fails. Find the source: unseeded RNG, float printing, dict ordering,
multithreaded BLAS, model nondeterminism, time-dependent fields. Fix and prove with
3 consecutive identical hashes.
```

---

## Daily checklist mapping (prompts ↔ dates)

| Date | Prompts | Human tasks |
|---|---|---|
| Oct 2 | 00, 01, 02 | Fill SOURCES.yaml links, provenance PDFs |
| Oct 3 | 03, 04 | Paste texts, overrides, record clean team takes, CONSENT.md |
| Oct 4 | 05, 06 | Listen to op examples |
| Oct 5 | 07, 08, 09 | Record flawed takes, annotate, Drive upload |
| Oct 6 | 10, 11 | — |
| Oct 7 | 12, 13, 14 | Hand-label 50 boundaries |
| Oct 8 | 15, 16, 17 | — |
| Oct 9 | 18, 19 | — |
| Oct 10 | 21, 22 | — |
| Oct 11 | 20 | Review failure cases |
| Oct 12 | 23, 24 | HF Space push, Drive link incognito test |
| Oct 13 | 25, 26, 27 | Record + upload video (Unlisted) |
| Oct 14 | 28, 29 | Devpost: add teammates, submit |
| Oct 15 | buffer | — |