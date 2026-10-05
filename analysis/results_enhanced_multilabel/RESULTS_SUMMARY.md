# Results summary — enhanced synthetic fault classification

## What was trained

Fifteen classifiers were compared: seven tree feature subsets, a full-fit tree
finalist, logistic regression, a dense network on selected inputs, a causal TCN,
three PCA+dense variants (90/95/99% variance), and autoencoder16+dense. A
training-prevalence baseline is also reported. Neural networks had an 18-epoch
budget with validation early stopping; the autoencoder had 15 epochs.

All learned preprocessing/selection used training trucks only. Model selection
used two validation trucks; a third validation truck supplied sigmoid
calibration. Three original test trucks were held out during initial selection.

## Feature results

- **369** allowed inputs initially.
- **256** after train-only constants/all-missing/exact-duplicate removal.
- **161** after train-only correlation pruning.
- A **16-feature original-input subset** was the smallest tree set within the
  declared 0.015 validation-AP tolerance. Its per-fault recall still varies.
- The validation winner was **PCA99 + dense network**, requiring **256 original
  inputs transformed into 109 components**, retaining **99.00698% variance**.

PCA dimensions are **not** a reduction to 109 physical sensors. The 16-feature
subset is not the winning model's input list. It contains GNSS latitude and
longitude, which may encode route/journey progress: it is a synthetic baseline,
not an approved minimum instrumentation recommendation.

Train-only rankings combine per-fault Pearson/point-biserial, Spearman, mutual
information and tree importance. A proxy-ablation comparator, source dependency
notes, PCA loadings and validation permutation explanations are saved. SHAP was
not run; additional rolling min/max/ratio features and waveform-specific models
were not added to this study. Existing residual/rate/window/EGT/lead–lag features
were used rather than fabricating new independent physics.

## Initial held-truck ranking results

| Candidate | Test macro AP | Test macro F1 (raw validation thresholds) |
|---|---:|---:|
| PCA99 + dense | 0.8907 | 0.8462 |
| PCA90 + dense | 0.8791 | 0.8214 |
| PCA95 + dense | 0.8758 | 0.8233 |
| Selected16 + dense | 0.8538 | 0.8121 |
| Selected16 + causal TCN | 0.8505 | 0.8277 |
| Selected16 + full-fit tree | 0.8397 | 0.8060 |
| Autoencoder16 + dense | 0.6677 | 0.6342 |

AP means average precision, not classification accuracy or a causal confidence.
Scores apply only to the synthetic distribution. Different tree fitting budgets
are declared; initial subset trees fit 40,000 sampled training rows, while the
final tree/DL fits use all 114,912 training rows.

## Important threshold failure and correction

The initial runner searched calibrated thresholds at coarse 0.01 increments.
The slow-aging calibrator compressed its scores into a very narrow range. The
coarse grid consequently predicted aging on every normal test row:

- Initial calibrated model macro AP: **0.8906**.
- Initial calibrated normal-row alarm rate: **100%**.
- The original artifact is `selected_model.joblib` and is **not suitable for
  alarm deployment**, despite good aggregate AP.

This failure was preserved, not hidden. `refine_fault_thresholds.py` fixes the
search using actual validation precision–recall thresholds. It also saves an
alternative validation-only normal-alarm constraint (1% per label on normal
validation rows). Weights and calibration are unchanged.

| Post-hoc policy | Macro F1 | Micro F1 | Macro recall | Normal-row alarm rate |
|---|---:|---:|---:|---:|
| Exact validation max-F1 | 0.8464 | 0.8107 | 0.8478 | 26.11% |
| Per-label normal-alarm guard | 0.8393 | 0.8535 | 0.8196 | 6.96% |

These threshold-policy test metrics are **exploratory second-use**, because the
original test had already revealed the failure. No test rows fitted thresholds,
but these are not newly untouched-test results. The guard is not a guarantee
of 1% overall alarms; several labels can collectively exceed their per-label
validation cap.

`alarm_guarded_model.joblib` carries the guarded policy and explicitly records
`deployment_ready=False`. `threshold_refined_model.joblib` carries exact max-F1.
Both require further independent validation before deployment.

## Fresh-seed external evaluation

After freezing `alarm_guarded_model.joblib`, a new batch was generated with seed
**20261006**: **15,903 rows, 93 episodes and three additional virtual engine
variants**, with no engine-variant seed overlap. No weights, calibration or
thresholds were fit on it. All generated rows are external evaluation, regardless
of that batch's local train/validation/test metadata.

| Metric | Frozen guarded model |
|---|---:|
| Macro AP | **0.8897** |
| Macro F1 | **0.8574** |
| Micro F1 | **0.8419** |
| Macro recall | **0.8233** |
| Normal-row alarm rate | **2.96%** |

These are fresh synthetic results, not real-truck accuracy. On this batch,
hydraulic aging AP is only **0.2443** with recall **15.57%**, and sensor-fault
recall is **30.65%**. The model therefore remains **not deployment-ready**.
Evidence is under `independent_seed_validation/` (data, predictions, per-fault
metrics and `RESULTS.json`).

## What remains weak

- **Slow hydraulic aging:** AP **0.1179**, guarded recall **2.34%**, F1 **0.0379**.
  Thirty-minute histories and tiny wear labels are insufficient. Do not suppress
  inspection simply because this classifier predicts no aging.
- **Sensor faults:** AP **0.7456**, guarded recall **57.53%**. Bias/drift/freeze
  can resemble real single-channel changes; missingness is easier.
- **Original missing-channel stress:** macro AP fell to **0.5001**. This masked
  selected feature families, not a full simulator sensor-failure replay. The
  result is a warning about robustness, not a deployment quality guarantee.
- Root cause and mitigation are not independent supervised targets. Classifier
  outputs should support observation-based RCA rules/catalogues and inspection.
- Only three original test trucks; bootstrap intervals are statistically fragile.

## Health and RUL

`health_assessment.csv` preserves sensor-only subsystem/overall health. The
additional `health_weighted_assessment.csv` shows conservative minimum health and
weighted trend health (engine .30, cooling/transmission/hydraulics .15 each,
brakes/structure .10 each, perception .05).

If a required subsystem is missing/invalid, health remains unknown; **830/24,624
original test rows** have unknown aggregate health. DL probabilities are not
multiplied into sensor health, avoiding unvalidated double-counting.

**No physical RUL model was trained.** The dataset has no independent
run-to-failure/censoring histories. RUL remains unavailable and root cause
unconfirmed. The legacy algebraic lifetime formula was not used as truth.

## Saved artifacts

- `feature_selection/`: audit, correlations, MI, ranked importance, redundancy,
  feature subsets, sensor/context dependencies.
- `selected_attributes.csv`: the provisional compact16 baseline.
- `feature_schema.json`: actual model input order and target order.
- `model_comparison.csv`, `per_fault_metrics.csv`: original comparisons.
- `models/`: native dense/TCN/AE checkpoints and per-epoch histories.
- `dense_selected_bundle.joblib`, `tcn_selected_bundle.joblib`: complete compact16
  alternatives with train-only imputer/scaler, original validation thresholds,
  ordered features/targets and native checkpoint references. Reloaded full-test
  AP matched their original comparison rows.
- `dense_pca_*_loadings.csv`: PCA interpretation.
- `calibration_metrics.csv`, `calibration_bins.csv`, `thresholds.csv`.
- `threshold_refinement_comparison.csv`, `threshold_refinement_per_fault.csv`.
- `test_predictions.csv`: original predictions; guarded predictions are separate.
- `RESULTS_VERIFIED.json`, `THRESHOLD_RESULTS_VERIFIED.json`: recalculation checks.
- `per_fault_AP.png`: held-truck AP chart.
- `health_assessment.csv`, `health_weighted_assessment.csv`, `health_policy.json`.
- `manifest.json`: original source hashes, splits, versions and 42 artifact hashes.

Original run source hash refers to the code at execution. After detecting the
threshold issue, the runner's threshold function, undefined-AP handling and
named-input imputer handling were corrected. `THRESHOLD_RESULTS_VERIFIED.json`
records both original/current source hashes. Original models/results remain
unchanged; reproducing with the corrected runner changes threshold results.

## Actual checks

- Original training completed in **261 seconds**, using CUDA for neural models.
- **33 ML/health-focused tests passed** in the isolated training environment.
- Existing default-environment suite: **318 passed, seven skipped**; three added
  ML test modules require the isolated sklearn environment and were separately
  checked. An added health test was also checked separately.
- Saved-model reload matched predictions; 42 original artifact digests checked;
  metrics recomputed from all **24,624 original held-test predictions**.
- Threshold refinement preserved weights and original artifacts; saved corrected
  decisions match their thresholds. All root confirmations are false and no
  numerical RUL is fabricated.

## Recommendation

Use the PCA+dense model as a **synthetic research baseline**, not a finished
maintenance alarm. Keep the compact16/dense/TCN alternatives for interpretability
and Android trade-off studies. Next priority is better long-history aging labels,
sensor-fault diversity, independent routes/seeds and an operationally justified
alarm/recall policy—not interpreting a high macro AP as solved root cause/RUL.
