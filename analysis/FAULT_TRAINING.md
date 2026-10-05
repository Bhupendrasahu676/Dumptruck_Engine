# Enhanced fault-model training

This workflow is separate from the legacy health/RUL regression study. It uses
`datasets/dump_truck_enhanced_all_conditions.csv` and its authoritative schema.
All input features are observation/context-derived; injected labels and
diagnostic outputs are never predictors.

## Environment

The implementation uses an isolated environment, leaving the application/global
dependencies unchanged:

```bash
uv venv --python /usr/bin/python3 --system-site-packages /tmp/opencode/fault-training
uv pip install --python /tmp/opencode/fault-training/bin/python \
  -r analysis/fault_training_requirements.txt
```

PyTorch was already available as `2.13.0+cu130`, with an RTX 3060. For other
machines install the appropriate official CPU/CUDA wheel; do not assume this
particular GPU build is portable.

## Run

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/train_fault_models.py \
  --output analysis/results_enhanced_multilabel --epochs 18
```

The output must be empty/new. The runner never overwrites old studies.

## Design

- Preserve the supplied 14 training / three validation / three test trucks.
- Reserve one validation truck for sigmoid calibration. Model/threshold tuning
  uses the other two validation trucks; test trucks never fit or tune a model.
- Fit imputation, scaling, supervised feature rankings and PCA only on training.
- Remove constants, all-missing features and exact duplicates with NaN-aware
  comparison. Compare correlation-pruned, 8/16/32/64/all and proxy-ablated sets.
- Rank each fault equally using point-biserial/Pearson, Spearman, mutual
  information and independently balanced per-label tree importance. These are
  associations, not causal effects.
- Compare bounded tree classifiers, logistic baseline, dense multilabel neural
  network, causal TCN, PCA90/95/99 + dense networks, and autoencoder16 + dense
  network. Initial tree comparisons use a recorded 40,000-row training sample;
  the selected tree finalist and DL models use the full training partition.
- TCN windows repeat the first observed episode reading for left padding; they
  never cross an episode or look ahead. Each target describes the current row.
- AE learns reconstruction on all training readings, including faults. It is
  an unsupervised embedding, not a separately calibrated anomaly detector.
- Compare by validation macro average precision (AP), not overall accuracy.
  Thresholds maximise validation F1 per fault. F1 tuning is not a safety recall
  guarantee; inspect every class's recall and false alarms.
- Save raw/calibrated AP, F1, recall, precision, TP/FP/FN/TN, Brier/log loss,
  normal false alarms, strata, detection delays and partial missing-sensor stress.
- Explain selected features through per-fault ranks and validation permutation.
  SHAP is explicitly not run. Feature counts do not establish sensor counts.

## Results contract

`analysis/results_enhanced_multilabel/` contains source/split manifests,
`feature_selection/`, subset comparisons, PCA loadings, native DL checkpoints,
serialized model bundles, feature/target order, thresholds, calibration bins,
held-test predictions, sensor-health assessments and a report.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/verify_fault_training.py \
  --output analysis/results_enhanced_multilabel
```

Verification recomputes metrics, checks prediction thresholds, saved artifact
hashes and test memberships, and estimates fragile truck-bootstrap intervals.

## Health and RUL boundary

`health_assessment.csv` carries the simulator's **sensor-based heuristic
subsystem/overall health**, not a model trained on independent clinical or
mechanical condition labels. Model probabilities are kept separate: multiplying
an existing fault-sensitive score by them could double-count the same anomaly.

There is no run-to-failure/censoring target in this dataset. Physical RUL
therefore remains **unavailable / not validated**. The legacy algebraic lifetime
formula is not used as a ground-truth label and no RUL neural network is trained.
Confirmed root cause likewise requires inspection/repair evidence; the classifier
can only reproduce synthetic fault patterns.

## Gaps to retain when interpreting results

- Only three synthetic test trucks, one C-WTVC proxy family and thirty-minute
  independent experiments; no fleet validation or long-term prognostics.
- Weak wear/effect labels may be below sensor noise; mixed faults can share
  signatures. Report weak/developed and single/mixed performance separately.
- Position/fuel/energy channels can encode journey progress. Proxy ablation is
  a check, not proof that every learned correlation is physically causal.
- Full-waveform FFT, authentic maintenance-case matching and real root-cause
  labels are not supplied by this CSV. Do not claim they were learned.
- Fixed-0.5 probabilities, tuned decisions and calibrated probabilities are
  different concepts. Calibration is synthetic-distribution only.
- See `RESULTS_SUMMARY.md` in the results folder before using a bundle: the
  original coarse calibrated-threshold grid produced 100% normal alarms.
  Original results were preserved; exact/guarded post-hoc policies are separate.
  The guarded policy was also evaluated on a fresh independently generated seed
  batch with frozen weights/thresholds, not fitted to those external outcomes.
- A selected feature's baseline dependencies may include frame, context and
  thermal-energy observations; keep them available at Android inference.
- Joblib contains executable Python objects. Load only locally generated,
  trusted bundles; native `.pt` checkpoints use `weights_only=True` loading.
