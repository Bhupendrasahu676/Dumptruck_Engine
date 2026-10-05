# 3D Dump Truck Digital Twin

This folder contains a complete Streamlit dashboard for a synthetic Caterpillar
793F-style dump truck. It replays a deterministic journey and maps telemetry to
a procedural Three.js truck in the browser:

- payload class is categorical and fixed for the entire journey;
- heavy (`94%`) and overloaded (`108%`) payloads push thermal, vibration,
  hydraulic and structural readings above the `85%` stress threshold;
- vibration drives high-frequency truck shaking;
- coolant temperature drives thermal colour on the engine indicators;
- payload drives suspension compression;
- body position and hoist angle drive the dump-body response;
- operating modes cover parked, idle, hauling, upslope, downslope, braking and
  dumping;
- research mode provides a reproducible 1,800-second C-WTVC proxy split into
  urban (0–900 s), highway (900–1368 s) and expressway (1368–1800 s), with
  selectable cold or warm start;
- the research proxy includes thermal inertia, thermostat opening, coolant
  inlet/outlet temperatures and flow, fuel-energy input, coolant/exhaust energy
  shares and cumulative fuel energy. These are calibrated to the attached paper
  qualitatively, not presented as the paper's unavailable raw measurements;
- IMU, engine, fuel/exhaust, transmission, brake, suspension/frame, GNSS/GPS,
  LiDAR, radar and driver-safety telemetry are shown in the dashboard;
- the release gate exposes a 0–100 health score, classification, remaining
  useful life estimate, confidence band and recommended action.
- engine, transmission, hydraulic and coolant readings are also classified
  against the supplied Green / Grey (caution) / Red operational bands; the
  active group summary and per-reading colour are shown in the dashboard and
  included in exports.

## New: engine evidence & lifecycle workflow

### Enhanced training-data simulator

Select **Enhanced Dumper Simulator** in page navigation, or run
`streamlit run enhanced_dashboard.py --server.port 8504`. This additive mode
provides 16 fault/disturbance classes, missing oil-condition channels, temporal
fault schedules/cascades, sensor faults, permanent wear, diverse virtual trucks,
observation-only RCA and leakage-separated training exports.

Generate a new dataset without replacing the legacy CSV:

```bash
python3 generate_enhanced_dataset.py --output-dir datasets/enhanced \
  --engines 12 --episodes-per-fault 2
```

See [ENHANCED_SIMULATION.md](ENHANCED_SIMULATION.md) for schemas, multi-label
targets, independent-engine splits, waveform/FFT exports and history continuation.
The simulator does not automatically load a trained classifier; field-validated
RUL is not supplied. Trained synthetic research models are described below.

### Trained synthetic fault-model study

The enhanced CSV has now been used for feature-selection, tree/logistic/DL,
PCA, autoencoder and causal-TCN comparisons. Results are kept separately at
`analysis/results_enhanced_multilabel/`. Read
`analysis/results_enhanced_multilabel/RESULTS_SUMMARY.md` first: it documents
the original coarse-threshold failure, preserved original results, corrected
alarm policies and frozen-model fresh-seed evaluation. No live dashboard model
integration, field root-cause validation or physical RUL training is implied.

See `analysis/FAULT_TRAINING.md` and `analysis/fault_training_requirements.txt`
for the isolated environment and reproducible commands. The original
`selected_model.joblib` has unsuitable alarm thresholds; corrected variants
and compact dense/TCN bundles are separate and are not deployment-approved.

For a **single CSV** with the earlier 171-sample / 1,800-second journey layout:

```bash
python3 generate_training_csv.py \
  --output datasets/dump_truck_enhanced_all_conditions.csv
python3 audit_training_csv.py \
  --csv datasets/dump_truck_enhanced_all_conditions.csv \
  --output-prefix analysis/enhanced_dataset_verification/data_audit
```

Default: 20 trucks, 960 independent journeys and 164,160 rows. Each primary
class covers all 30 scenario/payload/start contexts; supplements cover all five
sensor modes, all 120 fault pairs and 60 selected triples. This is not an
exhaustive enumeration of the 65,536 Boolean combinations of 16 faults.

The adjacent `.schema.json` names the 369 numeric input columns and 16 target
columns. Load **only `input_columns` as X** and **only `target_columns` as y**;
the unified CSV also contains excluded IDs, truth and diagnostic outputs.
Use the `split` column, not a row-wise random split. See the generated audit
for essential checks that passed and remaining partial/unsupported claims.

Select **Engine Evidence Lab** in either existing dashboard's page navigation, or
run `streamlit run engine_health_lab.py --server.port 8503`.

This additive workflow implements persistent engine/overhaul/test IDs and hours,
component repair history, eight scheduled-fault scenarios, state-conditioned
residuals, data-quality checks, independent hypotheses, separate technician
confirmation, and an explicit RUL evidence gate. Existing generated datasets and
legacy replay calculations are unchanged. No field-validated RUL model is supplied.

See [ENGINE_EVIDENCE_LAB.md](ENGINE_EVIDENCE_LAB.md) for use, storage, evidence
schemas, limitations and tests.

## Run locally

```bash
cd Engine
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_3d.py
```

The browser view includes a Canvas 2D fallback if Three.js/WebGL or the CDN is
not available. The synthetic data is for simulation and UI development only;
it is not validated maintenance or safety evidence.

The research calibration is based on Ma, Zhang, Liang & Chen, *Fuel* 328
(2022) 125307, DOI: `10.1016/j.fuel.2022.125307`. The paper reports a
1,800-second C-WTVC, engine operation mainly around 1,100–1,400 rpm, a cold
start coolant response lag, and late expressway energy shares near 20% coolant
and 32% exhaust. The simulator uses those as qualitative targets and labels
the cycle as a proxy because the authors' raw measurement data is not included.

## Key files

- `app_3d.py` — Streamlit controls, fault injection and full dashboard page.
- `telemetry_model.py` — dependency-light deterministic telemetry generator.
- `telemetry_export.py` — idempotent JSONL/CSV logging and export helpers.
- `operational_ranges.py` — reference operational bands and conservative
  Green/Amber/Red classification.
- `static/digital_twin.js` — procedural Three.js truck with responsive fallback.
- `static/dashboard.css` — minimal dark dashboard styling.
- `test_telemetry_model.py` — behavioral tests for fixed payloads, stress
  escalation and health classification.

To use an actual `.glb` later, replace the procedural scene inside
`static/digital_twin.js` with a `GLTFLoader` asset while keeping the same
`DigitalTwinTruck.init(...).update(...)` interface.

## Fault injection and logging

The five fault toggles are always visible in the Streamlit sidebar:

- coolant failure / overheat;
- bearing failure / vibration;
- low engine oil pressure;
- brake wear / long stroke;
- perception sensor degradation.

Each selected fault is applied to every sample in the current journey and is
shown in the page warning banner and in the exported `manual_faults` metadata.
Perception degradation also changes the GNSS, LiDAR and radar readings so the
fault is visible rather than only changing a flag.

Telemetry is not silently written on every Streamlit rerun. Use **Record
journey locally** to write an idempotent pair of files under `telemetry_logs/`:

- `.jsonl` — full nested telemetry, one self-identifying record per sample;
- `.csv` — compact analysis view with health, IMU, machine, perception and
  fault columns.

The same journey configuration produces the same hash-derived journey ID, so
pressing Record again does not duplicate the log. The dashboard also provides
download buttons for both formats.
