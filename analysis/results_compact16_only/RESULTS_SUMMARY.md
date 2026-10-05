# Exactly 16 selected inputs — trained fault models

## Result

Fresh dense and causal-TCN models were trained on **all 114,912 training rows**,
using **exactly the 16 requested columns**, with **no PCA, autoencoder, feature
reselection, extra context channels or added availability masks**.

Validation selected the **TCN** (raw macro AP 0.870752 versus dense 0.864407).
The TCN has 16 input channels and trailing 12-reading windows; its best checkpoint
was epoch 4 (eight epochs completed with early stopping). Dense's best checkpoint
was epoch 17 of 18. The model still predicts all 16 multi-label fault targets.

The preprocessing is train-only median imputation plus standardisation. One
validation truck fits sigmoid calibration; the other two select checkpoints,
model kind and decision thresholds. Model/policies were frozen before reading
test feature values. The held-truck benchmark was used in earlier studies, so
this is a comparative retraining study, not a newly untouched experiment.

## Held-truck results — selected TCN

| Metric | Guarded primary | Maximum-F1 alternate |
|---|---:|---:|
| Macro average precision | **0.8496** | **0.8496** |
| Macro F1 | **0.8131** | **0.8272** |
| Macro recall | **78.31%** | **81.59%** |
| Micro F1 | **0.8193** | **0.7947** |
| Normal-reading alarm rate | **5.33%** | **23.27%** |
| Exact multi-label match | **75.62%** | **71.82%** |

Evaluation: **24,624 readings from three held-out virtual trucks**. AP is
average precision, not accuracy. Primary guarded thresholds maximise validation
F1 subject to at most 1% per-label FPR on normal validation readings; this does
not guarantee 1% overall or test/field alarms. Both policies were declared and
fitted before test prediction, not selected by test performance.

Dense guarded comparison: test macro AP **0.8538**, macro F1 **0.8020**, recall
**74.34%**, normal-reading alarms **4.08%**. TCN remains the selected model
because selection used validation evidence, not whichever test metric looked best.

## Confirmatory different-seed check

The existing separate seed-20261006 batch has **15,903 readings** and three
distinct engine-variant seeds, with no overlap with training variants. No model,
calibration or threshold was fitted to it:

| Metric | Selected guarded TCN |
|---|---:|
| Macro AP | **0.8541** |
| Macro F1 | **0.8075** |
| Macro recall | **77.90%** |
| Normal-reading alarm rate | **3.29%** |

This batch was evaluated in another study previously. It is useful cross-seed
confirmation but not newly generated untouched data or field validation. The
CSV's truck-name strings are batch-local; independent engine seeds establish
different virtual-machine variants.

## Weaknesses that must not be hidden

On the original test trucks:

- **Hydraulic normal aging:** AP **0.1143**, guarded recall **3.90%**.
- **Sensor faults:** AP **0.4787**, guarded recall **26.62%**.

Different-seed sensor-fault recall drops to **10.44%** and aging recall is
**8.64%**. Slow wear is weak in short thirty-minute histories, and the chosen
16 inputs lack enough independent sensor-quality evidence. A negative prediction
must not override inspection or safety instrumentation. The model is explicitly
**not deployment-ready**.

GNSS latitude/longitude remain in the input list because they were requested;
they can encode route/journey progress rather than physical failure causes.
Residuals require original sensors and healthy operating-context priors upstream.
Exactly 16 model columns do **not** mean only 16 physical sensors.

No real root cause, repair outcome or physical RUL is learned. This run does not
fit a health/RUL model or modify the older health/RUL calculations.

## Files

- `selected_attributes.csv`, `feature_schema.json`: immutable 16-column order,
  targets and preprocessing/dependency contract.
- **`selected_model.joblib`**: guarded TCN, full preprocessing/calibration/schema
  and embedded checkpoint weights. Primary research bundle.
- `selected_maxf1_model.joblib`: same TCN weights, alternate thresholds.
- `dense_model.joblib`, `tcn_model.joblib`: complete candidate bundles.
- `models/`: native `.pt` checkpoints and epoch histories.
- `model_comparison.csv`, `per_fault_metrics.csv`, `normal_false_alarms.csv`.
- `test_predictions.csv`, `test_predictions_guarded.csv`, `test_predictions_maxf1.csv`.
- `stratified_metrics.csv`, `detection_delay.csv`, calibration metrics/bins.
- `frozen_policy.json`: validation choices saved before test inference.
- `verification_receipt.json`, `RESULTS_VERIFIED.json`, `manifest.json`.
- `EXTERNAL_EVALUATION.json`, `external_per_fault_metrics.csv`, `external_predictions.csv`.

For inference, use `analysis.train_compact16.predict_compact_bundle` with trusted
joblib artifacts and complete chronological episodes/prefixes. TCN call boundaries
reset history. Bundles use relative paths and embedded weights for portability.

## Verification and interrupted-export recovery

- **14 focused tests passed** for fixed input order, no extra inputs, missing
  feature rejection, all-missing-column preservation, causal windows, native
  checkpoints and timestamp validation.
- Default timestamps mix whole-second and fractional ISO8601 precision. This
  parsing issue was corrected before model fitting; regression coverage added.
- Both models completed training once. An export-stage assertion compared a
  full CUDA inference batch with a small single-episode batch and failed a tight
  tolerance. Existing checkpoints/results were inspected and preserved.
- Same-size full-batch reload matched saved TCN probabilities (maximum difference
  about **3e-8**), with **zero decision changes**. Single-episode GPU probabilities
  differed by up to **0.001204**, with **zero threshold decision changes** in the
  checked episode. This numerical batch-shape sensitivity is recorded; portability
  near thresholds needs testing rather than a promise of bit-identical inference.
- `analysis/finalize_compact16.py` finalized the report/manifest from existing
  weights and predictions. **No retraining, weight modification or threshold
  refitting was repeated.**
- All **32 original run artifact digests** checked, both policies' metrics were
  recomputed from all 24,624 saved predictions, and input dimensions verified at
  16 before/after preprocessing. Earlier study artifacts remain preserved.

## Reproduce in a new folder

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/train_compact16.py \
  --output analysis/results_compact16_new --epochs 18 --seed 793
```

Setup and inference details: `analysis/COMPACT16_TRAINING.md`.
