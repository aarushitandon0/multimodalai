# LOTO evaluation

* corpus: **dataset**
* variant: `full` - the system as specified
* folds: 4 (loto_s1_gettysburg, loto_s5_im_nobody, loto_s6_sunset_lampman, loto_s7_sonnet18)
* config hash: `2a4b61dd77f4`, code `unknown`
* event-F1 cross-checked against sed_eval: yes

Values are mean over folds; the per-fold spread is in `results/fold_<text>.json`.

## Injected (held-out text)

| Metric | Target | Actual | Verdict | Note |
|---|---|---|---|---|
| Event-F1 @ 200 ms | >= 0.75 | 0.0083 | MISS |  |
| Onset/offset MAE (ms) | <= 150 | 506.745 | MISS |  |
| Macro-F1 (type) | >= 0.7 | 0.4455 | MISS |  |
| Spearman rho | <= -0.85 | -0.0376 | MISS |  |
| FP on clean (per min) | <= 1 | 5.978 | MISS |  |
| M-F gap (points) | <= 0.5 | - | N/A | not measurable in this run |
| UTMOS drop at L1 | <= 0.3 | - | N/A | not measurable in this run |


| Metric | Mean | SD | Folds |
|---|---|---|---|
| event_f1_200ms | 0.0083 | 0.0058 | 4 |
| false_positives_per_min_on_clean | 5.978 | 0.4115 | 4 |
| flaw_type_macro_f1 | 0.4455 | 0.2148 | 4 |
| male_female_score_gap_points | - | - | 0 |
| onset_offset_mae_ms | 506.745 | 322.407 | 4 |
| severity_spearman_rho | -0.0376 | 0.0677 | 4 |

## Human takes

Not evaluated: this corpus has no human files.

