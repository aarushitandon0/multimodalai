# Human-recorded flawed takes

Empty. Spec 7.4 reserves this directory for consented team recordings of the
same texts delivered badly on purpose, and CLAUDE.md invariant 11 makes them
**test-only**: nothing in here may ever reach a training fold.

Nothing has been recorded yet, so:

* `mumbling` — the one label that cannot be injected by DSP and must be acted —
  has no files, and the flaw taxonomy ships 7 of 8 labels;
* the Clarity rubric category is scored from spectral features but has no
  ground-truth events to evaluate against;
* `results/metrics.md` reports "Human takes: not evaluated".

The recording protocol is in `../raw/team/README.md`, the consent form in
`../../CONSENT.md`. Audio recorded here stays out of git (`.gitignore`); it
ships in the dataset archive only with a signed consent row in
`../../LICENSES.csv`.
