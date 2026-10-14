# LOTO evaluation

* corpus: **synthetic**  - generated feature tables, not a dataset result (spec 14.1 caveat)
* variant: `full` - the system as specified
* folds: 3 (loto_syn_a, loto_syn_b, loto_syn_c)
* config hash: `20a3f11bc1a8`, code `54188c7-dirty`
* event-F1 cross-checked against sed_eval: yes

Values are mean over folds; the per-fold spread is in `results/fold_<text>.json`.

## Synthetic (held-out text)

| Metric | Target | Actual | Verdict | Note |
|---|---|---|---|---|
| Event-F1 @ 200 ms | >= 0.75 | 0.5533 | MISS |  |
| Onset/offset MAE (ms) | <= 150 | 87.6367 | PASS |  |
| Macro-F1 (type) | >= 0.7 | 0.8905 | PASS |  |
| Spearman rho | <= -0.85 | -0.6855 | MISS |  |
| FP on clean (per min) | <= 1 | 1.7219 | MISS |  |
| M-F gap (points) | <= 0.5 | - | N/A | not measurable in this run |
| UTMOS drop at L1 | <= 0.3 | - | N/A | not measurable in this run |

The false-alarm rate is measured over **0.5734 minute(s)** of clean audio in total, so one stray region moves it by several per minute. Read it with the per-fold counts in the fold JSON, not on its own.


| Metric | Mean | SD | Folds |
|---|---|---|---|
| event_f1_200ms | 0.5533 | 0.0811 | 3 |
| false_positives_per_min_on_clean | 1.7219 | 2.4352 | 3 |
| flaw_type_macro_f1 | 0.8905 | 0.0617 | 3 |
| male_female_score_gap_points | - | - | 0 |
| onset_offset_mae_ms | 87.6367 | 15.4203 | 3 |
| severity_spearman_rho | -0.6855 | 0.0378 | 3 |

## Human takes

Not evaluated: this corpus has no human files.

