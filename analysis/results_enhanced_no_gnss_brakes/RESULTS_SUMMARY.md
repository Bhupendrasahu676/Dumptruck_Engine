# Retraining results: no GNSS or brake inputs

## Completed change

Removed **51 GNSS/GPS/brake-related source features**, including raw readings,
masks, residuals, rates, rolling features and the braking-mode indicator.
Dependency-aware checks also reject indirect GNSS/brake feature dependencies.
The old CSV and every previously saved result were preserved.

Preprocessing, rankings, correlation pruning, PCA, neural embeddings and models
were fitted again from the reduced input allowlist. No old model weights or
preprocessors were reused. The same training budgets and truck-disjoint splits
were retained for the original-data benchmark.

## Feature counts

| Stage | Count |
|---|---:|
| Original approved source inputs | 369 |
| Explicitly excluded | 51 |
| Remaining candidates | 318 |
| After training constants/duplicates | 220 |
| After correlation pruning | 138 |
| Provisional compact original-feature subset | **32** |
| Best PCA representation | **94 components** |

The validation winner is **PCA99 + dense neural network**: **220 eligible
inputs → 94 components**, retaining **99.03408%** input variance. The compact32
subset is a separate interpretable baseline, not the winning model's input
list or a validated minimum sensor set.

Exact selected names: `selected_attributes.csv`.
Actual winning input order: `feature_schema.json` → `model_input_columns`.
Every exclusion: `excluded_attributes.csv` and `input_exclusion_policy.json`.

Compact features primarily concern engine oil pressure/condition/boost,
coolant flow/temperature/condition, transmission pressure/condition/vibration,
hydraulic condition, engine/block/exhaust vibration/temperature, lead–lag and
LiDAR confidence. Cumulative fuel volume remains a possible journey-time proxy;
its selection is not proof of mechanical causation.

## Original held-truck benchmark (all 16 labels)

| Metric | New calibrated/guarded model |
|---|---:|
| Macro average precision (AP) | **0.8326** |
| Macro F1 | **0.7812** |
| Macro recall | **0.7621** |
| Micro F1 | **0.8142** |
| Normal-reading alarm rate | **6.02%** |

These original test trucks were already evaluated in the previous study, so
these are ablation benchmark results, not a newly untouched test. Model selection
used two validation trucks; calibration used the third. Exact thresholds were
chosen on validation only, with a predeclared 1% per-label normal-validation
alarm constraint. That constraint does not imply 1% aggregate/field alarms.

### Brake-wear target caveat

All **16 labels were retained for comparable output shape**, but brake-wear
prediction has no direct brake readings and is explicitly unsupported.
On the original benchmark its AP is **0.0621**, recall **0**, F1 **0**.
Do not present this output as usable brake diagnosis, even if incidental
correlations produce a nonzero probability on another batch.

LiDAR/radar/camera inputs remain, so perception degradation can still be
screened without GNSS. This update does not exclude all perception features.

## Comparable 15-label view (excluding unsupported brake target)

Both models use the same original test rows and omit the brake label only for
this reporting view; the models were not refit as 15-head networks.

| Model | Macro AP | Macro F1 | Macro recall |
|---|---:|---:|---:|
| Previous guarded model | 0.8867 | 0.8376 | 0.8146 |
| New no-GNSS/brake guarded model | **0.8839** | **0.8333** | **0.8129** |

Most of the all16 aggregate-score drop comes from removing the very readings
needed for brake-wear classification. Do not interpret the smaller supported15
change as a field-reliability guarantee. `supported_15_label_comparison.csv`
records exact values. Normal in this view means no supported15 labels and can
include source brake-only readings.

## Fresh independently seeded external batch

After freezing weights, calibration and thresholds, seed **20261007** generated
**15,903 readings, 93 episodes and three additional virtual-engine variants**.
Variant seeds do not overlap original training variants. No fitting/tuning was
performed on this batch.

| Metric | All16 comparison targets | Supported15 reporting view |
|---|---:|---:|
| Macro AP | **0.8307** | **0.8772** |
| Macro F1 | **0.7978** | **0.8381** |
| Macro recall | **0.7450** | **0.7842** |
| Normal-reading alarm rate | **4.61%** | **3.92%** |

The previous external run used a different seed; its scores are not a paired
old/new external comparison. All new generated rows are external evaluation,
irrespective of their local generator split metadata. Raw external data retain
the full simulator schema for provenance, but inference reads only the filtered
220-input model allowlist. Brake/GNSS channels never enter the model transforms.

Hydraulic aging remains weak (fresh-batch AP **0.1960**, recall **21.23%**).
Sensor-fault recall is **46.80%**. The model remains **not deployment-ready**.

## Saved models and evidence

- `selected_model.joblib`: complete new PCA+dense bundle, calibration, ordered
  inputs/targets and guarded thresholds; `deployment_ready=False`.
- `dense_selected_bundle.joblib`, `tcn_selected_bundle.joblib`: compact32
  alternatives with train-only preprocessing and native checkpoint references.
- `models/`: native weights and training histories for all DL representations.
- `feature_selection/`, `dense_pca_*_loadings.csv`: ranking/reduction evidence.
- `model_comparison.csv`, `per_fault_metrics.csv`, `test_predictions.csv`.
- `INPUT_EXCLUSION_COMPARISON.md/.json`: old/new comparisons and caveats.
- `RESULTS_VERIFIED.json`, `EXCLUSION_VERIFIED.json`: recalculated metrics,
  probability/threshold checks, all-branch exclusion verification.
- `independent_seed_validation/`: external data, predictions, per-fault metrics,
  `RESULTS.json` and `SUPPORTED15_RESULTS.json`.
- `health_assessment.csv`: existing all-subsystem simulator reference, **not
  health predicted from the reduced model**. It includes instrumentation omitted
  from this training setup and must not be deployed as reduced-sensor health.

No physical RUL or confirmed root-cause model was trained. RUL remains unavailable;
classifier probabilities are synthetic pattern evidence only.

## Checks actually run

- **139 focused ML/policy/comparison tests passed**.
- Default interpreter regression suite: **349 passed, eight skipped**; optional
  ML tests were separately checked in the isolated environment.
- Saved-model reloads match predictions; all **24,624** benchmark predictions
  were recalculated and **44 original new-run artifact hashes** checked.
- No excluded names/dependencies in any input/subset branch or PCA loading index.
- Compact32 dense/TCN preprocessing bundles reload to the original benchmark AP.
- All **68 previously saved result artifacts** retain their original hashes.
- Training completed in **243 seconds** with an 18-epoch neural budget and
  validation early stopping. No global application dependencies were changed.

For exact rerun commands and interpretation, see `analysis/NO_GNSS_BRAKES.md`.
