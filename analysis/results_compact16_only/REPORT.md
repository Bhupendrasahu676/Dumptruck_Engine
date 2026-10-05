# Fixed compact-16 training results

Validation-selected model: **tcn**, exactly **16 model input columns**.
No PCA, feature reselection, added context inputs or autoencoder; both models trained on all 114,912 training rows.

| candidate | split | policy | macro_AP | macro_F1 | macro_recall | normal_false_alarm_rate |
|---|---|---|---|---|---|---|
| dense | validation_tuning | maxf1 | 0.8644 | 0.8512 | 0.8280 | 0.2593 |
| dense | validation_tuning | guarded | 0.8644 | 0.8214 | 0.7753 | 0.0368 |
| tcn | validation_tuning | maxf1 | 0.8694 | 0.8620 | 0.8479 | 0.2123 |
| tcn | validation_tuning | guarded | 0.8694 | 0.8362 | 0.7999 | 0.0370 |
| dense | test | maxf1 | 0.8538 | 0.8094 | 0.7822 | 0.2802 |
| dense | test | guarded | 0.8538 | 0.8020 | 0.7434 | 0.0408 |
| tcn | test | maxf1 | 0.8496 | 0.8272 | 0.8159 | 0.2327 |
| tcn | test | guarded | 0.8496 | 0.8131 | 0.7831 | 0.0533 |

## Interpretation
Average precision is not accuracy. The primary guarded thresholds were frozen on validation before test prediction.
The original held-truck benchmark was used in prior studies; these results are comparative synthetic evidence, not a newly untouched field test.
Exactly sixteen model columns do not imply sixteen physical sensors. Residuals require healthy context priors, commissioning/calibration, source sensors, trailing observation state and episode resets; context is not added to the model input. GNSS longitude/latitude remain route proxies and GNSS accuracy and brake.stroke residual remain explicitly requested inputs. Brake stroke also depends on pedal and pad-wear measurements. No GNSS/brake ablation or reselection is performed.
Slow aging and ambiguous sensor faults may remain weak. All bundles are deployment_ready=False.
No health/RUL training or real root-cause confirmation. Inspect per_fault_metrics.csv and normal_false_alarms.csv, not aggregate scores alone.

## Export verification
Training finished, but a strict single-episode serialization assertion found CUDA batch-size rounding differences.
The finalizer reused existing frozen weights and predictions; full-test reload matched and the prefix differences/decision flips are recorded.
No training or threshold selection was repeated. verification_receipt.json records recovery and numerical evidence.

## Artifacts
selected_model.joblib is the guarded primary; selected_maxf1_model.joblib is the alternate. Candidate bundles and native checkpoints are also saved.
feature_schema.json and selected_attributes.csv give the exact input order. For portable loading use analysis.train_compact16.predict_compact_bundle.

## Reproduce
```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/opencode/fault-training/bin/python analysis/train_compact16.py --output analysis/results_compact16_new --epochs 18 --seed 793
```
