# REFERENCES.md — Research backing for Rostrum (Track C)

Every reference below is tied to **a specific design decision** and to **the judging criterion it earns points on**. Use the "Use in Rostrum" column to justify choices in the technical report, README and video.

**Legend**
- ✅ **Seen during our research** — we opened the paper/page and confirmed the claim summarised here.
- 📘 **Standard reference** — well-established citation; confirm volume/page numbers on Google Scholar before the final report.
- `[spec §x]` → section of `MASTER_SPEC_PLAN.md` where it is used.

---

## 0. The argument in one paragraph (for the report introduction)

Public-speaking judging is subjective and feedback is vague. Acoustic prosody predicts perceived speaking quality (Azaïs 2015; Niebuhr 2016), and evidence-based voice-coaching systems exist (VoiceCoach 2020), but their authors explicitly list *negative examples of delivery* as missing. Comparative alignment between a speaker and reference productions can localise errors over continuous stretches of speech (Cambridge, SLaTE 2025). Rostrum combines these ideas: a **license-clean contrastive dataset** with **interventional (injected) flaws of known location and magnitude**, a **speaker-normalised, word-anchored comparison against a distribution of skilled readers**, **sound-event-detection-style temporal evaluation**, and **counterfactual, number-backed explanations**.

---

## 1. Causal Explainability & Temporal Grounding — 25%
*Requirement 3a (timestamps) and 3b (human-readable causal explanation); "sufficient temporal precision".*

### 1.1 How "causal" is justified (framing that wins this criterion)

| Ref | Status | What it gives you | Use in Rostrum |
|---|---|---|---|
| Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. | 📘 | Intervention (*do*-operator) vs observation | **Injected flaws are interventions**: we *set* rate/pitch/pause on a known span of an otherwise identical recording, so the acoustic delta is causally attributable to the flaw on synthetic data. State this explicitly. [spec §5, §12] |
| Wachter, S., Mittelstadt, B., Russell, C. (2018). Counterfactual explanations without opening the black box. *Harvard Journal of Law & Technology* 31(2). arXiv:1711.00399 | 📘 | Explanations of the form "if X had been x′, the outcome would change" | The `fix` field is a **counterfactual**: "at ≈ 4 syll/s (envelope median) this region would not be flagged". [spec §12] |
| Lundberg, S., Lee, S.-I. (2017). A unified approach to interpreting model predictions (SHAP). *NeurIPS*. arXiv:1705.07874 | 📘 | Additive feature attributions | Our deviation $D_w^2=\sum_k \lambda_k z_{wk}^2$ is **additively decomposable**, so each feature's share $\lambda_k z_{wk}^2 / D_w^2$ is an exact attribution — shown as the evidence table. [spec §10.3, §12] |

### 1.2 Temporal grounding & its evaluation

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Mesaros, A., Heittola, T., Virtanen, T. (2016). Metrics for polyphonic sound event detection. *Applied Sciences* 6(6):162. | ✅ | Event-based metrics with onset/offset tolerance collars; `sed_eval` toolbox | Primary grounding metric: event-F1, 200 ms onset collar. [spec §14] |
| Turpault, N. et al. (2020). Sound Event Detection and Separation: a Benchmark on DESED Synthetic Soundscapes (DCASE 2020 Task 4). arXiv:2011.00801 | ✅ | Offset collar = max(200 ms, 20% of event length) | Exact collar convention we adopt. [spec §14.1] |
| Ebbers, J. et al. (2022). Threshold-Independent Evaluation of Sound Event Detection Scores. *ICASSP*. arXiv:2201.13148 | ✅ | Evaluates all decision thresholds jointly; `sed_scores_eval` | PR curves / PSDS so results don't hinge on one threshold. [spec §14.1] |
| Mesaros, A. et al. (2021). Sound Event Detection: A Tutorial. *IEEE Signal Processing Magazine*. arXiv:2107.05463 | ✅ | Why collars exist (subjective boundaries); event- vs segment-based evaluation | Justifies the 200 ms collar in the report. |
| Caba Heilbron, F. et al. (2015). ActivityNet: A large-scale video benchmark for human activity understanding. *CVPR*. | 📘 | Temporal IoU (tIoU) for temporal localisation | Secondary metric: tIoU of matched regions. [spec §14] |

### 1.3 Boundary precision: region formation & refinement

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Truong, C., Oudre, L., Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing* 167. arXiv:1801.00718 | ✅ | Framework (cost + search + penalty); `ruptures` library | Boundary refinement within ±300 ms; report cost (RBF) and search (PELT). [spec §10.5] |
| Killick, R., Fearnhead, P., Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost (PELT). *J. American Statistical Association* 107(500). | 📘 | PELT algorithm | Cite when naming the search method. |
| Canny, J. (1986). A computational approach to edge detection. *IEEE TPAMI* 8(6). | 📘 | Origin of **hysteresis thresholding** (high threshold to start, low to continue) | Our enter 2.5σ / stay 1.5σ region rule. [spec §10.4] |
| Rousseeuw, P. J., Croux, C. (1993). Alternatives to the median absolute deviation. *JASA* 88(424). | 📘 | MAD as robust scale; 1.4826 consistency constant | Envelope σ = 1.4826·MAD. [spec §10.2] |
| Leys, C. et al. (2013). Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median. *J. Experimental Social Psychology* 49(4). | 📘 | Practical case for median/MAD over mean/SD | Why one unusual LibriVox reader doesn't distort the envelope. |
| Sakoe, H., Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE TASSP* 26(1). | 📘 | Dynamic time warping | DTW **within words** to warp curves for the overlay only. [spec §10.1] |

### 1.4 Closest prior work (comparative grounding)

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| *Comparative Pronunciation Assessment and Feedback with Interpretable Speech Features.* SLaTE 2025 (Univ. of Cambridge). https://www.repository.cam.ac.uk/items/092494b9-905d-4941-951b-cbd992460ef1 | ✅ | Aligns learner speech with synthetic native/non-native productions using interpretable features; alignment-error peaks localise errors over continuous stretches | **Closest related work.** Our extensions: multi-reader envelope, per-feature attribution, delivery (not pronunciation). |
| Mikhailava, V. et al. (2022). StudyIntonation: tailoring multimodal feedback in prosody training. https://library.iated.org/view/MIKHAILAVA2022TAI | ✅ | Model vs learner pitch contours plotted together, DTW-based estimate | Precedent for baseline-vs-participant overlay. |

### 1.5 Making explanations useful & trustworthy

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| *An overview of artificial intelligence in computer-assisted language learning.* arXiv:2505.02032 | ✅ | Explicit, direct feedback is more effective than feedback that only flags an error | Every card = what + where + how much + fix. [spec §12] |
| Parikh, A. K. et al. (2025). Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech. *SLaTE 2025*. | ✅ | Audio LLM over-predicts low-quality speech, imprecise error detection | Why **no LLM in the judging loop**. [spec §10.7] |
| Chen, Y.-W., Ma, M., Hirschberg, J. (2025). Read to Hear: Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs (TextPA). arXiv:2509.14187 | ✅ | Convert acoustics → textual descriptors → explainable feedback | Supports "DSP numbers first, language second". |
| *GSRM: Generative Speech Reward Model with acoustic-feature-grounded chain-of-thought.* arXiv:2602.13891 | ✅ | Interpretable acoustic feature extraction, then feature-grounded reasoning | Same design principle. |

---

## 2. Data Engineering & Stress Testing — 30%
*Requirements 1a (good baseline), 1b (bad spectrum), 1c (labels); "Data quality over quantity"; Custom Dataset deliverable.*

### 2.1 Baseline sourcing (cleverness + legality)

| Source | Status | Why it's clever | Use |
|---|---|---|---|
| LibriVox — *The Gettysburg Address 150th Anniversary* (15 readers). https://librivox.org/the-gettysburg-address-150th-anniversary-by-abraham-lincoln/ | ✅ | 15 skilled readings of the **same text** → a statistical envelope, not one "correct" way | Silver tier, envelope. [spec §4.2] |
| LibriVox — *Gettysburg Address (version 2)*, John Greenman. http://librivox.org/the-gettysburg-address-by-abraham-lincoln-version-2/ | ✅ | Extra envelope member | Silver tier. |
| LibriVox public-domain statement (CD cover PDF). https://www.archive.org/download/LibrivoxCdCoverArt12/Gettysburg_Address_V3_1109.pdf | ✅ | States recordings may be reproduced, distributed or **modified** | Licence proof for derivatives. |
| JFK Library — JFKWHA-001, Inaugural Address, 20 Jan 1961. https://www.jfklibrary.org/asset-viewer/archives/jfkwha-001 | ✅ | "Copyright Status: Public Domain"; prominent politician as required by 1a | Gold tier. |
| Wikimedia Commons — `File:Roosevelt_Pearl_Harbor.ogg`. https://commons.wikimedia.org/wiki/File:Roosevelt_Pearl_Harbor.ogg | ✅ | Public domain; anthem follows speech → trim | Gold tier. |
| FDR Library audio record (Rights: Unrestricted). https://www.presidentialtimeline.org/html/record_id-80.html | ✅ | Second provenance for FDR | Licence proof. |
| TED Talks Usage Policy (CC BY-NC-ND 4.0). https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy | ✅ | **No derivatives allowed** | Why we exclude TED despite the brief's example — shows legal awareness. |
| National Speech & Debate Association — How to Judge Declamation. https://www.speechanddebate.org/?p=1070 | ✅ | Declamation isn't meant to mimic the original speaker | Justifies **multiple valid baselines** (envelope). |
| Panayotov, V. et al. (2015). LibriSpeech: an ASR corpus based on public domain audio books. *ICASSP*. | 📘 | Precedent for building research corpora from LibriVox | Cite as precedent for LibriVox-based data. |

### 2.2 The "bad" spectrum: synthesis methods

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Moulines, E., Charpentier, F. (1990). Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones (PSOLA). *Speech Communication* 9(5–6). | 📘 | TD-PSOLA time/pitch modification | Our injection method for pace, pitch, stress. [spec §5.2] |
| Morrison, M. et al. (2024). Fine-Grained and Interpretable Neural Speech Editing. arXiv:2407.05471 | ✅ | Listening test: TD-PSOLA time-stretch ≈ neural editor, WORLD clearly worse | Why PSOLA, not WORLD. |
| *Pseudo-Cepstrum: Pitch Modification for Mel-Based Neural Vocoders.* arXiv:2512.16519 | ✅ | Naturalness declines as pitch modification grows | Why we gate realism per severity level. |
| Verhelst, W., Roelands, M. (1993). An overlap-add technique based on waveform similarity (WSOLA). *ICASSP*. | 📘 | Alternative time-scale method | Mention as considered alternative (rubberband/WSOLA). |
| Boersma, P., Weenink, D. *Praat: doing phonetics by computer.* https://www.praat.org (Manipulation, `Lengthen (overlap-add)`) | 📘 | Reference implementation of PSOLA manipulations | Cite the tool + version used. |
| Kourkounakis, T. et al. (2021). FluentNet / **LibriStutter**. arXiv:2009.11394 | ✅ | Synthetic flaws injected into clean speech with automatic labels; realism checked against real data | Direct precedent for our injection-with-labels approach. [spec §5] |
| Ko, T. et al. (2015). Audio augmentation for speech recognition. *Interspeech*. | 📘 | Speed perturbation as controlled signal modification | Background on controlled audio perturbation. |

### 2.3 Robustness of the gradient (realism + stress testing)

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Saeki, T. et al. (2022). UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022. arXiv:2204.02152 | 📘 | Neural MOS (naturalness) predictor | Realism gate + "UTMOS vs severity" table. [spec §5.4] |
| Mujtaba, D. et al. (2024). Lost in Transcription: ASR biases against disfluent speech. *NAACL*. | ✅ | ASR is unreliable on disfluent speech | Why fillers are out of scope and why we align to the transcript rather than trust ASR. |

### 2.4 Temporal labelling & annotation quality

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement* 20(1). | 📘 | Cohen's κ | Inter-annotator agreement on human-recorded flaw labels. [spec §6.2] |
| Mesaros et al. (2016) — event-F1 (above) | ✅ | Same metric between two annotators | Boundary agreement between annotators at 200 ms collar. |
| Rousso, R. et al. (2024) — see §3.1 | ✅ | MFA boundary accuracy | Justifies using MFA to verify injected-label timestamps (re-alignment check). |

### 2.5 Dataset documentation & "quality over quantity"

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Gebru, T. et al. (2021). Datasheets for Datasets. *Communications of the ACM* 64(12). arXiv:1803.09010 | 📘 | Standard question set for documenting datasets | Structure of `dataset/README.md`. [spec §7.4] |
| Bender, E. M., Friedman, B. (2018). Data Statements for NLP. *TACL* 6. | 📘 | Speaker/recording demographics documentation | Speaker sex, channel, recording setup fields. |
| Sambasivan, N. et al. (2021). "Everyone wants to do the model work, not the data work": Data Cascades in High-Stakes AI. *CHI*. | 📘 | Poor data quality propagates downstream | Motivation for strict QC gates over volume. |

---

## 3. Feature Extraction — 20%
*Requirement 2a (forced alignment) and 2b (FFT, MFCC, pitch/F0, rate, pauses, clarity); "stress-point and energy-contour identification".*

### 3.1 Forced alignment

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| McAuliffe, M. et al. (2017). Montreal Forced Aligner: trainable text-speech alignment using Kaldi. *Interspeech*. https://montreal-forced-aligner.readthedocs.io | 📘 | The aligner | Primary aligner. [spec §8] |
| Rousso, R., Cohen, E., Keshet, J., Chodroff, E. (2024). Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment. *Interspeech*. arXiv:2406.19363 | ✅ | MFA beats WhisperX and MMS on TIMIT/Buckeye word alignment | Why MFA. |
| Bain, M. et al. (2023). WhisperX. arXiv:2303.00747 | 📘 | ASR + word timestamps | ASR-vs-transcript WER check only. |
| Pratap, V. et al. (2023). Scaling Speech Technology to 1,000+ Languages (MMS). arXiv:2305.13516 | 📘 | CTC alignment baseline | Mention as compared alternative. |
| PyTorch Audio — `forced_align` deprecation notice. https://docs.pytorch.org/audio/stable/generated/torchaudio.functional.forced_align.html | ✅ | Deprecated 2.8, removed 2.9 | Reproducibility reason to avoid it. |

### 3.2 Pitch / F0 contours

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Boersma, P. (1993). Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound. *IFA Proceedings* 17. | 📘 | Praat autocorrelation pitch + HNR algorithm | Primary pitch method. [spec §9.1] |
| Jadoul, Y., Thompson, B., de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics* 71. | 📘 | Python access to Praat | Implementation citation. |
| Strömbergsson, S. Comparison of Praat, getF0 and YIN on PTDB-TUG (poster). https://cdn.dal.ca/content/dam/dalhousie/pdf/sites/icpla/postersjune17/61724.PDF.lt_9ddeaa1253afb7974ae5ffe9e8637664.res/61724.PDF | ✅ | Praat had the lowest voicing and gross-pitch error rates of the three | Why Praat. |
| Pirker, G. et al. (2011). A pitch tracking corpus with evaluation on multipitch tracking scenario (PTDB-TUG). *Interspeech*. | 📘 | Pitch benchmark corpus | Optional external validation. |
| Hirst, D. (2011). The analysis by synthesis of speech melody: from data to models. *Journal of Speech Sciences* 1(1). | 📘 | Two-pass speaker-adaptive pitch range (floor ≈ 0.75·Q1, ceiling ≈ 1.5·Q3) | Our two-pass pitch settings. [spec §9.1] — confirm the exact constants in the paper before quoting. |
| Kim, J. W. et al. (2018). CREPE. arXiv:1802.06182 | ✅ | Neural pitch tracker, high raw pitch accuracy | Cross-check tracker. |
| Morrison, M. et al. (2023). Cross-domain Neural Pitch and Periodicity Estimation (PENN). arXiv:2301.12258 | ✅ | Improved neural pitch + periodicity | Alternative cross-check. |
| de Cheveigné, A., Kawahara, H. (2002). YIN. *JASA* 111(4). | 📘 | Classic pitch estimator | Background / compared method. |
| Xu, Y. (2013). ProsodyPro — a tool for large-scale systematic prosody analysis. *TRASP*. | ✅ | Time-normalised F0 contours, excursion size (st), F0 velocity | Envelope band curves; monotone features. [spec §15.2] |
| den Ouden, H., Terken, J. (2001). Measuring pitch range. *Eurospeech*. | ✅ | HiF0-to-final-low pitch-range measure more reliable than peak–valley | Phrase pitch-range definition for `monotone`. |

### 3.3 Stress points (prominence) & energy contours

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Suni, A., Aalto, D., Vainio, M. (2015). Hierarchical Representation of Prosody for Statistical Speech Synthesis. arXiv:1510.01949 | ✅ | Unsupervised CWT prominence & boundary estimation, comparable to supervised methods | Our prominence (stress-point) feature. [spec §9.1] |
| Wavelet Prosody Toolkit. https://github.com/asuni/wavelet_prosody_toolkit | ✅ | Reference implementation | Cross-check of our implementation. |
| Talman, A. et al. (2019). Predicting Prosodic Prominence from Text with Pre-trained Contextualized Word Representations. arXiv:1908.02262 | ✅ | Clear description of the CWT prominence pipeline (F0 + energy + duration → CWT) | Method description to cite. |
| *Prosodic Prominence and Boundaries in Sequence-to-Sequence Speech Synthesis.* arXiv:2006.15967 | ✅ | Composite weights F0 1.0, energy 0.5, duration 1.0 | Our composite weights. |
| Kochanski, G. et al. (2005). Loudness predicts prominence; fundamental frequency lends little. *JASA* 118(2). | 📘 | Loudness/duration are strong prominence cues | Why energy and duration are in the prominence composite, not F0 alone. |
| Sluijter, A., van Heuven, V. (1996). Spectral balance as an acoustic correlate of linguistic stress. *JASA* 100(4). | 📘 | Spectral tilt (balance) marks stress | Spectral tilt feature for emphasis (same-channel). |
| Terken, J. (1991). Fundamental frequency and perceived prominence of accented syllables. *JASA* 89(4). | 📘 | F0 excursion ↔ perceived prominence | Supports F0 term in prominence. |
| de Seyssel, M. et al. (2024). EmphAssess. *EMNLP*. arXiv:2312.14069 | ✅ | Word-level emphasis classifier (EmphaClass) and benchmark | Optional external check of stress detection. |
| ITU-R BS.1770 (loudness algorithms) & EBU R 128 (−23 LUFS target). | 📘 | Integrated loudness standard | Loudness normalisation step. [spec §4.6] |

### 3.4 Speech rate & pause intervals

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| de Jong, N. H., Wempe, T. (2009). Praat script to detect syllable nuclei and measure speech rate automatically. *Behavior Research Methods* 41(2). | 📘 | Automatic syllable-nuclei speech rate | Cross-check of MFA-based rate. [spec §9.1] |
| de Jong, N. H., Bosker, H. R. (2013). Choosing a threshold for silent pauses to measure second language fluency. *DiSS*. | ✅ | Recommends ~250–300 ms silent-pause threshold | Our 250 ms hesitation threshold. |
| Kahng, J. (2018). The effect of pause location on perceived fluency. *Applied Psycholinguistics*. | ✅ | Listeners are sensitive to pause location; within-clause pauses hurt most | Splitting `missing_pause` vs `hesitation_pause`; gap labels. |
| Bosker, H. R. et al. (2013). What makes speech sound fluent? *Language Testing* 30(2). | ✅ | Pause and speed measures best predict fluency ratings | Why pace + pausing are separate rubric categories. |
| Silero VAD. https://github.com/snakers4/silero-vad | 📘 | Neural voice activity detection | Pause refinement. |
| Barrault, L. et al. (2023). Seamless (AutoPCP + local prosody toolkit in `stopes`). arXiv:2312.05187 | ✅ | Pause and speech-rate comparison between utterances; AutoPCP prosody similarity | Optional implementation reference / external validity check. |

### 3.5 FFT, MFCC, vocal clarity

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Davis, S., Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE TASSP* 28(4). | 📘 | MFCC origin | MFCC features. |
| Eyben, F. et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS). *IEEE Trans. Affective Computing* 7(2). | 📘 | Standard, research-accepted voice feature set | eGeMAPS via openSMILE. |
| Hillenbrand, J. et al. (1994). Acoustic correlates of breathy vocal quality. *JSLHR* 37(4). | 📘 | Cepstral peak prominence (CPP) | Clarity/voice-quality feature. |
| Ferguson, S. H., Kewley-Port, D. (2007). Talker differences in clear and conversational speech: acoustic characteristics of vowels. *JSLHR*. https://pubs.asha.org/doi/10.1044/1092-4388(2007/087) | ✅ | Clear speech: vowel-space expansion, longer vowels | Definition of "mumbling" features. |
| Ménard, L. et al. (2016). Speaking Clearly for the Blind. *PLoS ONE* 11(9). | ✅ | Clear speech: longer vowels, greater intensity, stronger vowel contrasts | Why mumbling is human-recorded, not low-pass simulated. |
| Amano-Kusumoto, A., Hosom, J.-P. (2009). Classifying clear and conversational speech based on acoustic features. *Interspeech*. | ✅ | Acoustic features separating clear vs conversational speech | Clarity composite. |

---

## 4. Speaker-agnostic comparison (Technical Consideration)

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Nolan, F. (2003). Intonational equivalence: an experimental evaluation of pitch scales. *Proc. 15th ICPhS*. | 📘 | Psychoacoustic scales (semitones/ERB) best match how listeners equate intonation across male/female voices | **Why F0 in semitones.** [spec §9.2] |
| US Patent 11,205,418 — Monotone speech detection. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11205418 | ✅ | Hz-ratio pitch variation (PVQ) fails to detect monotone in female voices with one threshold | Why not PVQ in Hz. |
| Hincks, R. (2005). Measuring liveliness in presentation speech. *Interspeech*. | ✅ | Liveliness ratings correlate with normalised F0 variation and mean length of runs | Pitch-variation & run-length descriptors. |
| Adank, P., van Hout, R., Smits, R. (2001). A comparison between human vowel normalization strategies and acoustic vowel transformation techniques. *Eurospeech*. | ✅ | z-score transform resembled perceptual data best | z-scoring energy/features. |
| Rose, P. (1987). Considerations in the normalisation of the fundamental frequency of linguistic tone. *Speech Communication* 6(4). | 📘 | z-score F0 normalisation across speakers | Supports speaker-wise z-scores. |
| Genette, J., Verhoeven, J., Gillis, S. (2023). Fundamental frequency normalization and statistical power: 15 techniques. *ICPhS*. | ✅ | Normalisation choice affects statistical sensitivity | Why we report ± normalisation ablation. |

---

## 5. Temporal precision sufficient for causal explanation (Technical Consideration)

How to argue precision is "sufficient":
1. **Measured alignment accuracy** (MFA hand-check: MAE/P95 in ms) — Rousso et al. 2024 for expected ranges.
2. **Label verification** of injected flaws by re-alignment (≤ 20 ms target).
3. **Evaluation collar** 200 ms = DCASE convention (Mesaros 2016; Turpault 2020), smaller than a typical word, so a correctly grounded region always names the right words.
4. **Silent-pause resolution** 250 ms threshold (de Jong & Bosker 2013) — region boundaries are finer than the pause units we explain.

---

## 6. Reproducibility (10% + Technical Consideration)

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| Pineau, J. et al. (2021). Improving reproducibility in machine learning research. *JMLR* 22. arXiv:2003.12206 | 📘 | ML reproducibility checklist | Checklist in README/report. |
| Gundersen, O. E., Kjensmo, S. (2018). State of the art: Reproducibility in artificial intelligence. *AAAI*. | 📘 | Method/data/experiment documentation | Framing for `make reproduce`. |
| Boettiger, C. (2015). An introduction to Docker for reproducible research. *ACM SIGOPS Operating Systems Review* 49(1). | 📘 | Containers for reproducible pipelines | Dockerfile rationale. |

---

## 7. Rubric grounding & public-speaking prosody (motivation)

| Ref | Status | What it gives you | Use |
|---|---|---|---|
| NSDA Declamation ballot. https://newmexicoforensicsclub.readthedocs.io/en/latest/_downloads/8016c624c36316fb5f0f70dd9ef71298/nsda-declamation-ballot.pdf | ✅ | Delivery judged on use of voice to enhance the message | Rubric anchor. |
| Toastmasters evaluation resources. https://content.toastmasters.org/image/upload/8053-generic-evaluation-resource.pdf ; https://ccdn.toastmasters.org/medias/files/department-documents/education-documents/evaluation-resources/english/8104e2-speech-profile-ff.pdf | ✅ | 1–5 vocal variety scale (tone, speed, volume); pausing choices | Rubric categories. |
| Wang, X. et al. (2020). VoiceCoach. *CHI*. arXiv:2001.07876 | ✅ | Evidence-based voice-modulation training from TED talks; future work: negative examples | Motivation: we supply the negative examples. |
| Azaïs, L. et al. (2015). Does my Speech Rock? Automatic Assessment of Public Speaking Skills. *Interspeech*. | ✅ | Prosodic features predict perceived oratory ability | Motivation for acoustic-only delivery scoring. |
| Niebuhr, O., Voße, J., Brem, A. (2016). What makes a charismatic speaker? *Computers in Human Behavior* 64. | ✅ | Pitch, loudness, tempo, fluency profiles of a charismatic speaker | Motivation; feature choices. |
| Rosenberg, A., Hirschberg, J. (2009). Charisma perception from text and speech. *Speech Communication* 51(7). | 📘 | Acoustic and lexical correlates of perceived charisma | Supports text + audio (multimodal) view. |
| Hardison, D. M. (2005). Contextualized Computer-based L2 Prosody Training. *CALICO Journal* 22(2). | ✅ | Discourse-level context improves transfer | Dashboard shows flaws in whole-speech context. |
| Sitaram, S. et al. (2011). What Visual Feedback Should a Reading Tutor Give Children on their Oral Reading Prosody? *SLaTE*. | ✅ | Prosody mapped onto displayed text | Coloured transcript. |
| *Zero-Shot TTS as Golden Speech Generator.* SLaTE 2025. | ✅ (title/venue) | TTS as reference speech | Optional Norm-mode fallback. |

---

## 8. Citation map for the 6-page technical report

| Report section | Cite |
|---|---|
| 1. Introduction & related work | NSDA, Toastmasters, VoiceCoach, Azaïs 2015, Niebuhr 2016, Cambridge SLaTE 2025, StudyIntonation |
| 2. Dataset construction | LibriVox pages, JFK Library, Commons/FDR Library, TED policy (exclusion), LibriSpeech (precedent), Moulines & Charpentier (PSOLA), Morrison 2024, LibriStutter, UTMOS, Cohen 1960, Gebru (datasheet), Bender & Friedman |
| 3. Features & normalisation | McAuliffe (MFA), Rousso 2024, Boersma 1993, Parselmouth, Strömbergsson, Hirst 2011, CREPE/PENN, Suni 2015, Kochanski 2005, Sluijter & van Heuven 1996, de Jong & Wempe 2009, de Jong & Bosker 2013, Kahng 2018, Davis & Mermelstein, Eyben 2016, Hillenbrand 1994, Ferguson & Kewley-Port 2007, Nolan 2003, Adank 2001, Rose 1987, BS.1770 |
| 4. Grounding, rubric & scoring | Rousseeuw & Croux 1993, Canny 1986 (hysteresis), Truong 2020 + Killick 2012, Sakoe & Chiba 1978, Pearl 2009 (intervention), Wachter 2018 (counterfactual), Lundberg & Lee 2017 (additive attribution), Parikh 2025 (no LLM judge), CALL feedback overview |
| 5. Results | Mesaros 2016, Turpault 2020, Ebbers 2022, Heilbron 2015 (tIoU) |
| 6. Reproducibility & limitations | Pineau 2021, Boettiger 2015, torchaudio deprecation notice, Mujtaba 2024 (fillers out of scope) |

---

## 9. Before the final report — verification to-do

- [ ] Confirm volume/page numbers for every 📘 entry on Google Scholar.
- [ ] Confirm Hirst's two-pass constants in the original paper before quoting them as his.
- [ ] Find author names for the Cambridge SLaTE 2025 paper on the repository page and the ISCA archive.
- [ ] Record the exact versions of MFA, Praat/parselmouth, librosa, openSMILE, ruptures and the UTMOS model used (reproducibility section).
- [ ] Re-check each data-source licence page on the day you download and save a PDF to `dataset/provenance/`.