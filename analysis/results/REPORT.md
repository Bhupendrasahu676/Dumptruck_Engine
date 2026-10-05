# Attribute reduction for the dump-truck simulator

## Scope and recommendation

Analysed **164,160 rows**, 960 configurations, 171 samples each, from the source CSV (unchanged).
The request originally said 171 cycles; this existing dataset contains **171 samples in one 1,800 s journey**
per combination, not 171 independently simulated journeys. This analysis does not expand it.

**202 CSV columns are not 202 independent sensors.** After excluding outputs, configuration shortcuts,
diagnostic flags and constants, there are 95 numeric candidates,
93 after exact deduplication, and 41 after train-only |Pearson r| >= 0.95 pruning.

The validation-selected original-attribute set has **8 inputs**, a reduction of
**91.4%** relative to the 93 eligible nonduplicate inputs.
Use `reduced_attributes_dataset.csv` for an interpretable starting point. It contains metadata and targets
as well as inputs: load **only the input_columns in feature_schema.json as X**.
This is a recommendation for predicting **the simulator's health and RUL**, not a validated minimum
sensor set for physical trucks or for collision, braking, driver-safety or structural-failure detection.

## Recommended original attributes

| attribute | training_importance | health_pearson | rul_pearson |
|---|---|---|---|
| dss.camera_online | 0.097 | 0.588 | 0.543 |
| dss.camera_fps | 0.094 | 0.767 | 0.782 |
| engine.oil_temp_c | 0.092 | -0.635 | -0.624 |
| engine.oil_pressure_kpa | 0.036 | 0.375 | 0.369 |
| suspension.hydraulic_oil_condition_pct | 0.032 | 0.839 | 0.830 |
| engine.coolant_temp_c | 0.028 | -0.373 | -0.353 |
| brake.disc_temp_c | 0.015 | -0.737 | -0.751 |
| transmission.vibration_mm_s | 0.011 | -0.498 | -0.494 |


Training importance is an ExtraTrees multi-output importance with both targets standardised, not
causal importance. Correlated alternatives may substitute for one another. Low single-variable
correlation does not imply a variable is unnecessary; fault interactions can be nonlinear.

![Selected correlations](correlation_selected.png)
![Target correlations](selected_target_correlations.png)

## Correlation outputs

- `correlation_pearson.csv`: full matrix for the 93 eligible nonduplicate inputs.
- `correlation_spearman.csv`: rank correlations for the same inputs (nonlinear monotonic checks).
- `correlation_selected.csv`: smaller matrix for recommended attributes.
- `target_correlations.csv`: individual inputs vs health and RUL, kept separate from feature matrices.
- `high_correlation_pairs.csv`: strongly related pairs; `correlation_pruned.csv`: removed representatives.
- `column_audit.csv`: disposition of every original column; `exact_duplicates.csv`: exact repetitions.

All reported correlations and fitted transforms use **training rows only**. Correlations reflect the
synthetic generator and repeated fault variants; they do not establish real-machine causality.

## PCA dimensionality reduction

StandardScaler (train mean/std) -> PCA (train only) retains **19 components** and
**96.03%** of variance from 41 pruned inputs.
Inputs have different units (kPa, degrees, g, mm/s), so scaling is necessary.
`pca_weights.csv` contains component projection coefficients, not Pearson correlations.
`pca_variance.csv` records individual/cumulative variance.

**PCA reduces model dimensions, not sensor requirements.** Collect all 41 PCA input channels
in schema order to transform a new frame. Prefer original-attribute selection when the aim is fewer
measurements. Retaining 95% of variance does not guarantee retaining 95% of health-prediction information.

![PCA variance](pca_scree.png)

## Leakage-aware validation

No row-wise random train/test split. Group = **scenario + payload class**; both starts, all 32 fault
variants and all temporal samples stay together. Nine groups train, three validate, three test.
This prevents fault-counterpart rows from the same base configuration appearing in different splits.
Training rows: 98,496; validation: 32,832; test: 32,832.
ExtraTrees fits use a reproducible balanced 29,952-row training subsample (52 rows/configuration);
correlations/scaling/PCA use all training rows. Scores use every row in the relevant held-out split.
80 trees, depth <=18, minimum leaf 3, seed 42; both targets standardised for comparable influence.

Validation chooses subset size from 8/12/16/24/all correlation-pruned inputs (capped at available count).
The smallest subset within 10% of the best joint normalised validation MAE is selected before testing.
PCA uses the same downstream learner. Baseline predicts the training-target means.

### Validation comparison

| method | attributes | health_mae | health_r2 | rul_mae_hours | rul_r2 | confidence_macro_f1 |
|---|---|---|---|---|---|---|
| Full telemetry | 93 | 6.994 | 0.682 | 1097.000 | 0.550 | 0.795 |
| Selected 8 | 8 | 6.658 | 0.744 | 1067.672 | 0.612 | 0.784 |
| Selected 12 | 12 | 6.761 | 0.723 | 1078.533 | 0.588 | 0.782 |
| Selected 16 | 16 | 6.948 | 0.712 | 1099.907 | 0.578 | 0.778 |
| Selected 24 | 24 | 7.529 | 0.676 | 1161.504 | 0.548 | 0.765 |
| Selected 41 | 41 | 7.389 | 0.676 | 1135.853 | 0.549 | 0.775 |
| PCA 95% | 19 | 6.890 | 0.744 | 1033.358 | 0.652 | 0.755 |


### Untouched test comparison

Health MAE is in **0–100 score points**; RUL MAE is in **hours**. Higher R²/F1 is better; lower MAE is better.

| method | attributes | health_mae | health_r2 | rul_mae_hours | rul_r2 | confidence_macro_f1 |
|---|---|---|---|---|---|---|
| Full telemetry | 93 | 8.507 | 0.453 | 1228.702 | 0.427 | 0.748 |
| Selected 8 | 8 | 8.256 | 0.488 | 1201.701 | 0.452 | 0.742 |
| PCA 95% | 19 | 8.289 | 0.557 | 1221.850 | 0.540 | 0.710 |
| Training mean baseline | 0 | 14.935 | -0.126 | 2110.358 | -0.066 | 0.191 |


Test groups: ideal|heavy, overhaul|medium, stress|light.
Validation groups: ideal|light, overhaul|heavy, stress|medium.

Confidence macro-F1 is computed by mapping predicted health to the existing High/Medium/Low rules;
confidence is **not an independently trained or calibrated probability**. The source percentage column
uses fixed values 90/75/55; these were assigned by the dataset generator, not estimated uncertainty.

## Limitations that matter for training

1. Health is generated by a hand-written penalty rule; RUL is an algebraic function of health plus
   scenario; confidence/classification are deterministic health bands. Good scores mean rule imitation,
   not verified failure prediction or remaining physical life.
2. Never feed health, RUL, confidence, their ranges/classes, actions, contribution factors, fault
   injection flags, active-fault count or operational-status labels into the prediction model.
3. A single seed and a single synthetic C-WTVC profile are present. Fifteen base groups are too few
   for a strong generalisation claim. The held-out scenarios use the same underlying simulator rules.
   Validate with new seeds, different routes, held-out fault mechanisms and ultimately actual failures.
4. Synthetic channels can be overly correlated. Oil-condition estimates, perception confidence and
   structural proxies require actual sensor/estimator support. Keep safety-critical sensors even if
   this health-target experiment ranks them low. Do not remove IMU/LiDAR/radar solely on these results.
5. The model currently uses coolant as a block-temperature proxy, plus separately synthesised vibration
   velocity in mm/s and acceleration in g. Those are not interchangeable physical measurements.
   Operational-band decisions and health-score decisions are separate rules and may disagree.
6. Selecting against two outputs cannot prove coverage for transmission, hydraulic or brake failure
   labels not independently tested here. This dataset also contains no independently observed failure times.

## Deliverables and reuse

- `reduced_attributes_dataset.csv`: all source rows, selected inputs, labelled targets and split metadata.
- `pca_dataset.csv`: same rows and targets, with fitted component scores.
- `feature_schema.json`: exact input/target/metadata lists.
- `pca_transform.joblib`: train-fitted scaler, PCA and required ordered columns.
- `selected_model.joblib`: demonstration model with target scaler and input list.
- `journey_splits.csv`, `model_comparison.csv`, `manifest.json`: reproducible split and evaluation evidence.

Joblib files are Python serialisations: load only these trusted locally generated files.
Full re-run from Engine:

```bash
uv run --with pandas==3.0.6 --with scikit-learn==1.9.1 --with matplotlib==3.11.2 python analysis/analyse_dataset.py
```
