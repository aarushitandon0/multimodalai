# QC tables

Four tables, each one stage of the build. Spec 7.4 names `utmos.csv`,
`realign.csv` and `rejections.csv`; `clean_build.csv` covers the stage before
any injection happens.

| File | Stage | Rows |
|---|---|---|
| `clean_build.csv` | download → excerpt → loudness-normalise a source reading | one per clean file built |
| `rejections.csv` | injection attempts that did not produce a file | one per failed attempt |
| `realign.csv` | invariant 9: re-align the edited WAV, reject if any word boundary moved > 50 ms | one per injected file |
| `utmos.csv` | invariant 9: UTMOS before/after, reject L1 if the drop > 0.3 | one per injected file |

## Status

`realign.csv` and `utmos.csv` hold **headers only**. The invariant-9 gates have
not been run over the injected corpus yet, and every label JSON under
`dataset/labels/` still carries `"qc": null`. `rostrum.inject.qc.label_passed`
treats that as *not passed* on purpose — absence of evidence is not a pass — so
nothing downstream may claim these 266 files are verified. Run
`scripts/qc_dataset.py` (needs the MFA conda env, so the Docker image) to fill
them; that writes the `qc` block into each label JSON and appends any newly
failing file to `rejections.csv`.

## Reading `rejections.csv`

One row per *attempt*, not per file, so a `file_id` here may still be in the
dataset: the engine re-draws spans from the same seeded stream and retries
(`placement_attempts` in `configs/injection.yaml`). The `final_status` column
resolves it:

* `retried_ok` — a later attempt succeeded and the file ships.
* `dropped` — no attempt succeeded and there is no such file.

`dropped` rows are why the label grid is not square. 48 `monotone` attempts
failed on unvoiced or too-short spans, which is why `monotone` has 20 files
where the other labels have 32–36.
