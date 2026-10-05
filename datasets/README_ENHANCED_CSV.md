# Enhanced single-CSV dataset

## Files

- `datasets/dump_truck_enhanced_all_conditions.csv`
- `datasets/dump_truck_enhanced_all_conditions.schema.json`
- `datasets/dump_truck_enhanced_all_conditions.manifest.json`
- Essential-point audit: `analysis/enhanced_dataset_verification/data_audit.md`
  and `.json`.
- Interpreted verdict: `analysis/enhanced_dataset_verification/ESSENTIAL_POINTS.md`.

Freshly generated: **164,160 rows = 20 virtual trucks × 48 journeys × 171
samples**. Every journey spans 1,800 seconds at approximately 10.588-second
sampling. This preserves the old dataset's row/journey size, not its fault
configuration semantics.

Each truck has 34 primary experiments (normal plus 16 classes, repeated twice),
five sensor-mode experiments, six mixed-pair experiments and three mixed-triple
experiments. Across the batch, every primary class covers all 30 combinations
of three scenarios, five payload categories and two start conditions. There
are all 120 distinct fault pairs and 60 selected triples—not all 65,536 possible
16-fault Boolean combinations.

## Model input/target contract

The schema provides:

- `input_columns`: **369 numeric inputs**, including 113 original sensor/context
  channels, validity masks, operating context and engineered temporal features.
- `target_columns`: **16 binary `fault_label.*` targets**.
- Metadata, truth and diagnostic columns are **not inputs**.

The CSV deliberately contains labels alongside measurements for convenient
loading. Never train on all numeric columns indiscriminately. Exclude IDs,
time/step/split, simulator scenario, injected phases/quality/wear, fault levels,
health, probable faults, possible causes, mitigation descriptions and root-cause
confirmation fields. Use exactly the allowlist in the schema.

Example selection (requires pandas):

```python
import json
import pandas as pd

path = "datasets/dump_truck_enhanced_all_conditions"
with open(path + ".schema.json") as f:
    schema = json.load(f)
inputs, targets = schema["input_columns"], schema["target_columns"]
columns = inputs + targets + ["split", "engine_id", "episode_id", "step"]
data = pd.read_csv(path + ".csv", usecols=columns)
train = data["split"].eq("train")
X_train = data.loc[train, inputs].astype("float32")
y_train = data.loc[train, targets].astype("float32")
```

Blank numeric cells are intentional for missing sensors, unavailable history
or onset associations. Fit imputation, scaling, correlations and feature
selection only on the training trucks. Form temporal windows within each
`engine_id`/`episode_id`, sorted by `step`; never join windows across journeys.

Splits are **14 training, three validation and three test trucks**. The test
rows must stay untouched during preprocessing, selection and model tuning.
Use multi-label outputs because faults can overlap.

## Label and diagnosis meanings

- `fault_label.<kind>` = `effect_level.<kind> > 0.001`.
- `fault_level.<kind>` represents the current schedule envelope.
- `effect_level.<kind>` includes lagged effects and persistent synthetic wear.
- `is_normal` means all effect labels are zero, not simply no active injection.
- `probable_faults_json` is observation-only heuristic inference.
- `possible_causes_json` and `mitigations_json` are **synthetic catalogue lookups
  based on truth labels**, not model predictions or technician confirmations.
- `root_cause_confirmed` remains false; RUL is not numerically fabricated.

Weak effect labels can be below the sensing/noise threshold. Class presence
does not prove detectability, and a rule signature is not a DL accuracy result.
Report performance separately for incipient/developed, single/mixed and
missing-sensor conditions. Measurements such as fuel level or accumulated energy
can act as journey-time proxies despite not being explicit labels; assess their
importance on independent schedules and seeds before retaining them.

This batch contains independent experiments, not run-to-failure longitudinal
histories. It uses the C-WTVC proxy, not every compact-route operating mode.
Scalar sampling cannot provide real vibration FFT evidence. Existing optional
waveform exports belong to their own generated configurations: do not join a
different batch merely because its virtual truck/episode IDs have the same names.

## Reproduce

```bash
python3 generate_training_csv.py \
  --output datasets/dump_truck_enhanced_all_conditions.csv \
  --engines 20 --episodes-per-fault 2 --samples 171 \
  --duration-s 1800 --seed 793
python3 audit_training_csv.py \
  --csv datasets/dump_truck_enhanced_all_conditions.csv \
  --output-prefix analysis/enhanced_dataset_verification/data_audit
```

Reproduction into an existing destination needs explicit `--overwrite`; old
legacy CSVs are not replaced. The manifest stores each seed, configuration,
schedule, split and file/source hash. These are synthetic test-bench data, not
validated physical Cat 793F measurements or confirmed real root causes.
