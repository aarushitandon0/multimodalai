"""rostrum.detect -- Deviation z-scores, hysteresis region formation, boundary refinement.

Five modules, in the order the pipeline uses them (spec 10.3-10.6):

* :mod:`rostrum.detect.zscore` - signed z against the envelope, then the
  per-label trigger signals of the spec 10.3 table;
* :mod:`rostrum.detect.regions` - smoothing, hysteresis, merge/drop, evidence;
* :mod:`rostrum.detect.refine` - ruptures PELT boundary refinement;
* :mod:`rostrum.detect.tune` - event-F1 grid search on training folds only;
* :mod:`rostrum.detect.classifier` - the flaw-type model and severity
  calibrator, which annotate the rule's regions but never overrule them.
"""

from rostrum.detect.classifier import (
    ClassifierCase,
    ClassifierError,
    FlawClassifier,
    Prediction,
    RegionFeatureSpec,
    SeverityCalibrator,
    TrainingExample,
    TruthFlaw,
    apply_label_policy,
    build_examples,
    fit_fold,
    fit_severity,
    load_model,
    match_to_truth,
    model_path,
    region_vector,
    save_model,
)
from rostrum.detect.refine import RefinementSettings, refine_boundary, refine_regions
from rostrum.detect.regions import (
    PAUSE_LABELS,
    RegionThresholds,
    Segment,
    build_regions,
    hysteresis_segments,
    label_counts,
    median_smooth,
)
from rostrum.detect.tune import Event, TuningCase, TuningResult, event_f1, grid_search
from rostrum.detect.zscore import (
    Signal,
    SkippedSignal,
    ZResult,
    combined_deviation,
    envelope_z,
    trigger_signals,
    zscores,
)

__all__ = [
    "PAUSE_LABELS",
    "ClassifierCase",
    "ClassifierError",
    "Event",
    "FlawClassifier",
    "Prediction",
    "RefinementSettings",
    "RegionFeatureSpec",
    "RegionThresholds",
    "Segment",
    "SeverityCalibrator",
    "Signal",
    "SkippedSignal",
    "TrainingExample",
    "TruthFlaw",
    "TuningCase",
    "TuningResult",
    "ZResult",
    "apply_label_policy",
    "build_examples",
    "build_regions",
    "combined_deviation",
    "envelope_z",
    "event_f1",
    "fit_fold",
    "fit_severity",
    "grid_search",
    "hysteresis_segments",
    "label_counts",
    "load_model",
    "match_to_truth",
    "median_smooth",
    "model_path",
    "refine_boundary",
    "refine_regions",
    "region_vector",
    "save_model",
    "trigger_signals",
    "zscores",
]
