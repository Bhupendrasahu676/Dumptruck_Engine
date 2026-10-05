# Retraining without GNSS or brake measurements

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/train_fault_models.py \
  --output analysis/results_enhanced_no_gnss_brakes \
  --exclude-gnss-brakes --epochs 18
```

The policy removes **51 of 369 source features**, leaving **318 candidates**
before train-only constant/duplicate/correlation filtering. It removes raw
GNSS/GPS/brake measurements, validity masks, residuals, rates, rolling summaries,
lead–lag dependencies where applicable and the braking-mode one-hot. Policy
records and exact removed names are saved in the new results folder.

Selection, imputation, scaling, PCA and representation learning start again
from the reduced allowlist. The old feature ranking/preprocessors/model weights
are not reused. The source CSV and old result files remain unchanged.

## Fair comparison and limitations

- Same source rows and original engine-disjoint splits, random seed and training
  budgets are retained. Prior test results have already been seen, so this is
  an ablation benchmark, not a newly untouched original test.
- All 16 labels remain for comparability. **Brake-wear prediction has no direct
  supporting measurements** and must not be approved based on incidental
  correlations. The model records it as an unsupported target.
- Perception degradation still has LiDAR, radar and camera-related observations;
  removing GNSS alone is not removing the entire perception subsystem.
- Guarded calibrated thresholds are predeclared: maximise validation F1 subject
  to a 1% per-label alarm cap on normal validation rows. This does not guarantee
  1% aggregate/field alarms. No test rows fit thresholds.
- Health CSV is a copied sensor-only simulator reference, not a prediction by
  this reduced model. Its old all-subsystem score should not be deployed when
  the excluded instrumentation is unavailable. Physical RUL stays unavailable.

## Verify and compare

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/verify_fault_training.py \
  --output analysis/results_enhanced_no_gnss_brakes
```

The comparison helper writes only inside the new result folder:

```python
from analysis.compare_fault_runs import compare
compare("analysis/results_enhanced_multilabel",
        "analysis/results_enhanced_no_gnss_brakes")
```

Optional frozen-model external evaluation uses a seed different from the original
training and prior external benchmark:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/evaluate_independent_fault_batch.py \
  --output analysis/results_enhanced_no_gnss_brakes \
  --model-file selected_model.joblib --seed 20261007
```

No weights/calibration/thresholds are changed during external evaluation. New
variant seeds are checked for overlap; batch-local engine-name strings are not
used to claim independent real-machine identities. Synthetic evidence is not
field validation, confirmed root cause or a minimum-sensor proof.
