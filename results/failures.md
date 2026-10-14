# Where it fails

* corpus: **synthetic**  - generated feature tables (eval/synthetic.py), so these are failures of the protocol on known perturbations, not of speech analysis on real readings
* folds: loto_syn_a, loto_syn_b, loto_syn_c
* variant: `full`

The three cases below are selected by rule, not by hand: the missed flaw with the highest severity, the false alarm with the highest region score, and the matched region with the largest total edge error. Each is the same event the aggregate numbers counted.

**The 'what' is measured; the 'suspected cause' is an inference** from the evidence listed under it, and is written to be falsifiable rather than reassuring.

## Missed flaw with the highest severity

* **file:** `syn_a_p16_low_energy_L4` (fold `loto_syn_a`, injected split)
* **label:** `low_energy`
* **what happened:** `low_energy` at 6.321-8.620 s (2.30 s, L4, s=0.9) was not reported: no region of that label matched it within the collars.
* **Suspected cause: a region of the right label covers 100% of this flaw, so it was *found* and mis-placed rather than missed: the offset is +960 ms, past the 460 ms collar (max(200 ms, 20% of 2.30 s)). It is close enough to be IoU-matched - the boundary error is measured over it - and too far to be counted as a detection, which is the gap between the two notions of 'matched' this report carries both of.**

Evidence:

```json
{
  "best_overlap_offset_error_ms": 960.0,
  "best_overlap_onset_error_ms": 0.0,
  "best_overlap_share": 1.0,
  "collars_exceeded": [
    "the offset is +960 ms, past the 460 ms collar (max(200 ms, 20% of 2.30 s))"
  ],
  "duration_s": 2.299,
  "level": "L4",
  "n_regions_in_file": 1,
  "overlapping_predictions": [
    {
      "iou_matched": true,
      "label": "low_energy",
      "rejected_by_collar": true,
      "score": 10.95762084113477,
      "t_end": 9.58,
      "t_start": 6.321
    }
  ],
  "refined": true,
  "severity_s": 0.9,
  "span_s": [
    6.321,
    8.62
  ],
  "tau_in": 3.0
}
```

## False alarm with the highest region score

* **file:** `syn_c_p03_rushing_L3` (fold `loto_syn_c`, injected split)
* **label:** `rushing`
* **what happened:** `rushing` reported at 3.570-6.480 s in a file whose truth has no such event there; region score |z|=12.00.
* **Suspected cause: a second region of the same label over a flaw that another region already matched. One truth event produced two regions; the matcher is one-to-one, so it takes the closer one and counts this one as a false alarm. The region fragmented instead of merging - regions separated by less than 0.2 s are joined, and this pair is further apart than that.**

Evidence:

```json
{
  "file_has_truth": true,
  "file_origin": "injected",
  "gap_to_previous_truth_s": null,
  "merge_gap_s": 0.2,
  "overlapping_truth": [
    {
      "label": "rushing",
      "t_end": 5.581,
      "t_start": 3.563
    }
  ],
  "score_peak_abs_z": 12.0,
  "span_s": [
    3.57,
    6.48
  ],
  "tau_in": 3.5
}
```

## Worst boundary error among the regions that matched

* **file:** `syn_a_p15_low_energy_L3` (fold `loto_syn_a`, injected split)
* **label:** `low_energy`
* **what happened:** `low_energy` was found, but its edges are off by +0 ms at the onset and +960 ms at the offset (IoU 0.7054): truth 6.321-8.620 s, region 6.321-9.580 s.
* **Suspected cause: region extent rather than edge placement. The error is larger than the +/-300 ms refinement window, so refinement cannot account for it: the region covers a different set of units from the flaw, because hysteresis held it open past the edit (or closed it early) and the merge rule then joined or split it on whole units.**

Evidence:

```json
{
  "iou": 0.7054,
  "level": "L3",
  "offset_error_ms": 960.2,
  "onset_error_ms": 0.0,
  "penalty": 1.0,
  "refined": true,
  "refinement_could_explain": false,
  "refinement_window_ms": 300.0,
  "region_span_s": [
    6.321,
    9.58
  ],
  "truth_span_s": [
    6.321,
    8.62
  ]
}
```

