# Training with exactly the requested sixteen features

`analysis/train_compact16.py` uses the immutable feature names/order from
`analysis/results_enhanced_multilabel/selected_attributes.csv`. It does not rerank,
drop constant columns, run PCA/autoencoders, or add context/availability columns
to the model. Both dense and causal-TCN models have **exactly 16 input channels**
and 16 sigmoid fault outputs.

## Run

Use the isolated environment described in `analysis/FAULT_TRAINING.md`:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/train_compact16.py \
  --output analysis/results_compact16_only --epochs 18 --seed 793
```

Existing nonempty results are refused. Previous studies are preserved.

- All 114,912 original training rows fit the weights.
- Median imputation and scaling are fitted on training trucks only, preserving
  all 16 columns even if a training column is all-missing.
- Two original validation trucks select the checkpoint/model by macro average
  precision and tune exact precision-recall decision thresholds.
- A separate validation truck fits sigmoid calibration.
- Maximum-F1 and a predeclared normal-alarm-guarded policy are both saved. The
  primary model uses the guarded policy selected **before** loading test inputs.
- The guard limits each label to 1% FPR on normal validation readings, not 1%
  overall alarms or guaranteed field/test performance.
- Causal-TCN windows contain at most 12 trailing readings from the same episode;
  short prefixes repeat the first observed reading, never future rows.

The source benchmark has been used in previous studies. Its held-truck scores
are useful comparisons, not newly untouched real-fleet evidence.

## Verify

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /tmp/opencode/fault-training/bin/python analysis/verify_compact16.py \
  --output analysis/results_compact16_only \
  --external-csv analysis/results_enhanced_multilabel/independent_seed_validation/independent_fault_batch.csv
```

This checks input order/dimensions, original artifact digests, partitions and
probability/threshold/metric consistency. Optional evaluation reuses a different
seed batch without fitting models, calibration or thresholds on it. Because that
batch was previously evaluated in another study, this is confirmatory cross-seed
evidence, not a newly untouched experimental holdout.

## Inference

```python
from analysis.train_compact16 import predict_compact_bundle

probabilities = predict_compact_bundle(
    "analysis/results_compact16_only/selected_model.joblib",
    frame, episode_ids=frame["episode_id"].to_numpy())
```

Complete bundles contain imputation/scaling, ordered features/targets,
calibrators, thresholds, and embedded weights. Relative checkpoint paths plus
embedded weights permit relocation. Joblib files must be trusted local artifacts.
For a TCN, pass complete chronological episode blocks or prefixes; a new call
resets history. Model probabilities are separate from calibrated thresholds.

## Limits

Residual features still require their original measurements and context priors
upstream. Sixteen input columns do not mean sixteen independent sensors. GNSS
longitude/latitude are kept because they are explicitly requested; they may
encode route/journey progress rather than physical fault mechanisms.

Calibration is synthetic-only. Missing observations are imputed without adding
new availability inputs, so robustness may degrade. Long-term aging, ambiguous
sensor faults and unseen environments require separate evidence. No physical
RUL or health model is fitted here; diagnosis and root-cause confirmation must
remain separate from fault pattern classification. All bundles are marked
`deployment_ready=False`.
