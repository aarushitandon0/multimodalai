# Ablations (spec 14.2)

* corpus: **synthetic**  - generated feature tables (eval/synthetic.py), not a dataset result
* texts: syn_a, syn_b, syn_c
* config hash: `20a3f11bc1a8`, code `d997653-dirty`

Each row is a full leave-one-text-out run with one thing changed; values are the mean over folds on the injected split, and the bracket is the change against the full system.

| Ablation | Event-F1 @ 200 ms | Onset/offset MAE (ms) | Macro-F1 (type) | Spearman rho | FP on clean (per min) |
|---|---|---|---|---|---|
| Full system | 0.5533 | 87.6367 | 0.8905 | -0.6855 | 1.7219 |
| 1. Single reference (not an envelope) | 0 (no regions) | - | 0 (no regions) | - | 0 (no regions) |
| 2. Rules only (no flaw-type classifier) | 0.5533 (=) | 87.6367 (=) | 0.8905 (=) | -0.6855 (=) | 1.7219 (=) |
| 3. No boundary refinement | 0.5533 (=) | 68.0367 (-19.6 better) | 0.8905 (=) | -0.6865 (-0.001 better) | 1.7219 (=) |
| 4. No speaker normalisation | 0.5933 (+0.04 better) | 51.0733 (-36.5634 better) | 0.9655 (+0.075 better) | -0.5524 (+0.1331 worse) | 3.4439 (+1.722 worse) |
| 5. Human test takes (vs synthetic) | not run | not run | not run | not run | not run |

## Row notes

* **Full system**: the baseline every delta is measured against.
  * macro-F1 of the rule's label 0.8905, of the classifier's 0.3531 (3 fold(s) had a classifier).
* **1. Single reference (not an envelope)**: gold reading where the corpus has one, else one drawn with a seeded RNG; `min_references` is overridden to 1 so the envelope builder allows it. **Caveat:** this variant emitted no region at all, so only the F1 is a measurement; the edge error has no matched pair to average and the false-alarm rate no alarm to count.
* **2. Rules only (no flaw-type classifier)**: the emitted label is the rule's either way (spec 10.6), so this row shows what the classifier adds as a second opinion, not a changed headline label.
  * macro-F1 of the rule's label 0.8905, of the classifier's None (0 fold(s) had a classifier).
* **3. No boundary refinement**: spec 10.5; expect the onset/offset MAE to move and the F1 to move less.
* **4. No speaker normalisation**: SIMULATED on the synthetic corpus: each speaker is given back a habitual pitch, loudness and tempo. On the dataset it would need a second feature pass with normalisation off, which `make features` does not produce.
* **5. Human test takes (vs synthetic)** (split: human): the full run's two splits side by side; no separate fit, because human takes are test-only (spec 7.3). **Not run:** the synthetic corpus has no human takes.

