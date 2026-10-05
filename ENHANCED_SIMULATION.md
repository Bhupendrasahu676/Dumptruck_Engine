# Enhanced dumper simulator and training-data generation

This mode combines the existing truck's operating diversity with the attached
fault matrix's condition channels and mechanisms. It is additive: the original
simulator, Engine Evidence Lab and old CSVs are preserved.

**Synthetic only:** this is not calibrated Cat 793F physics, a trained DL model,
a validated diagnostic system, or a field RUL model. Fault causes and mitigation
descriptions are illustrative inspection guidance, not verified repair outcomes.

## Dashboard

Choose **Enhanced Dumper Simulator** in the existing Streamlit page navigation,
or run:

```bash
streamlit run enhanced_dashboard.py --server.port 8504
```

Configure a virtual truck, repeatable engine/episode seeds, scenario, payload,
cold/warm start, ambient temperature, duration, sampling, initial wear, operating
hours and accelerated aging. Add multiple scheduled events, including repeats
and sensor faults. Click **Generate synthetic episode** explicitly.

Replay readings, residuals, rates, subsystem health and observation-only possible
causes. The **Injected truth** tab is separate. Downloads keep sensor readings,
truth labels, diagnostics and configuration separate. No automatic disk logging
occurs. RUL remains **not validated / unavailable**.

## What was added

| Capability | Implementation |
|---|---|
| Engine oil condition | `engine.oil_condition_pct`; not oil quantity |
| Gearbox oil condition | `transmission.oil_condition_pct` |
| Block-temperature channel | `engine.block_temp_c`; still a declared coolant-based synthetic proxy |
| Existing coolant/hydraulic condition | Retained with temporal mechanisms |
| Operating diversity | Existing route/C-WTVC, payload, start states plus ambient, truck variation and noise |
| Fault schedules | Onset, severity, ramp progression, duration, recovery, recurrence and overlap |
| Cascades | Pressure/flow first; illustrative thermal/vibration delays of 25/55 seconds |
| Permanent damage | Exposure-integrated wear persists after schedule recovery |
| Engine history | Explicit wear and operating-hour continuation |
| Sensor faults | Biased, missing, frozen, drifting and noisy; observation-only, no mechanical wear |
| Features | Raw sensors, context, availability masks, residuals, rates, rolling mean/std, residual slopes, EGT spread and onset lead–lag |
| FFT inputs | Separate bounded 4,096 Hz synthetic velocity snippets and spectral features |
| Data quality | Inference derives missing/invalid/frozen-suspected readings without reading injected sensor modes |
| Labels | Separate multi-label mechanism/effect labels, schedule severity and latent wear |
| Validation splits | Whole virtual trucks assigned to train/validation/test |
| RCA/actions | Context/persistence-aware signatures, alternative explanations and a mitigation catalogue |

### Fault catalogue

1. Engine oil-pump wear / lubrication-pressure loss.
2. Piston/ring damage, including cylinder EGT imbalance and time-varying EGT.
3. Turbo/exhaust-side fault, with local boost/EGT effects.
4. Coolant-system failure: flow and whole-engine thermal effects.
5. Coolant degradation: coolant-condition/pump-temperature changes, with block
   and oil temperatures preserved relative to normal context.
6. Gearbox seal loss: low pressure/high vibration with oil condition preserved.
7. Gearbox bearing/gear wear: pressure, vibration and oil-condition changes.
8. Slow hydraulic aging: exposure-driven condition decline and vibration rise.
9. Faster hydraulic contamination: rapid condition decline and actuator vibration.
10. Engine bearing-style failure.
11. Low oil quantity/pressure.
12. Overheating without the specific reduced-flow mechanism.
13. Brake wear/long stroke.
14. Perception degradation across GNSS/LiDAR/radar/camera indicators.
15. Sensor fault, with independently selectable signal and mode.
16. Unrelated multi-sensor disturbance.

These are synthetic mechanism classes; named classes can share observations.
For example, low oil quantity may also resemble a pump-pressure-loss signature.
RCA intentionally permits alternatives rather than claiming unique identification.

## Generate datasets

Standard generation (12 trucks, two primary episodes per class/truck):

```bash
python3 generate_enhanced_dataset.py \
  --output-dir datasets/enhanced \
  --engines 12 --episodes-per-fault 2 \
  --duration-s 1800 --timestep-s 5 --seed 793
```

This yields 492 episodes: per truck, 34 balanced primary episodes (normal plus
16 classes, each repeated twice), five supplemental sensor-mode episodes and
two overlapping-fault episodes. At 5-second sampling, there are 177,612 rows
in each full-rate table. Supplemental episodes mean aggregate fault frequencies
are not perfectly balanced. Repeated primary experiments also exercise recurrence.

Optional waveform/FFT branch:

```bash
python3 generate_enhanced_dataset.py \
  --output-dir datasets/enhanced_with_waveforms \
  --engines 12 --episodes-per-fault 2 \
  --waveforms --waveform-every 180
```

Each exported snippet is 0.25 seconds at 4,096 Hz (1,024 velocity samples).
Do not apply FFT to the 5/10-second scalar CSV readings. Fourier features are
descriptive, with assumed synthetic harmonics; they do not establish bearing
geometry, real fault frequencies or independent high-frequency physics.

Optional sequential history:

```bash
python3 generate_enhanced_dataset.py \
  --output-dir datasets/enhanced_history --engines 12 \
  --episodes-per-fault 2 --longitudinal
```

This carries truck identity, initial wear, accumulated damage, actual operating
hours and sequential timestamps. Earlier damage may remain during later
episodes, so the new injection's class is not the whole truck's condition label.
Pending schedules, sensor state and thermal transient state are not continued.
No real failure-time or censoring outcomes are invented.

Existing generated artifacts require explicit `--overwrite`. The exporter
stages and validates outputs before publication and preserves unrelated files.
Limits bound sample count and waveform storage. For pilot training, increase
independent trucks/repetitions before enlarging the sampling density alone.

## Output contract and leakage prevention

| Artifact | Purpose | Use directly as X? |
|---|---|---|
| `readings.csv` | Original observed sensors, context, availability and join metadata | Only manifest-approved columns |
| `features.csv` | Numeric engineered inputs plus join metadata | Only manifest-approved columns |
| `labels.csv` | Schedule/effect severity, multi-label targets, phases, wear and simulated sensor modes | No; select target columns as y |
| `diagnostics.csv` | Observation-only heuristic hypotheses, health, actions and unknown RUL | No |
| `manifest.json` | Seeds, episode configs, units, schemas, taxonomy, split membership and source hashes | No |
| `waveforms.jsonl` | Optional high-rate raw snippets with metadata | Only waveform values and permitted measured context |
| `spectral_features.csv` | Optional FFT features, at snippet timestamps only | Only listed spectral columns |

Join using `engine_id`, `episode_id`, `elapsed_s`. IDs, timestamp and split are
metadata, never model features. Keep engine IDs disjoint between training,
validation and test. Fit imputation, scaling, correlation pruning and selection
only on the training partition. Do not turn the old health/RUL feature ranking
into a fault feature ranking without evaluating each new target.

The manifest excludes simulator scenario, drive-cycle phase, payload-category
duplicates and episode-hour shortcuts from the default input list. Physical
context (RPM, demand/load, ambient, payload utilization, operating mode and
start state) remains available. Model features never include labels, injected
quality modes, inferred statuses, named hypotheses, latent wear, actions or RUL.
Absolute navigation channels remain in the full sensor table for optional
safety tasks; filter GNSS/LiDAR/radar/DSS/camera inputs for a no-perception model
and omit the perception target when those observations are unavailable.

### Target meanings

- `fault_level.<kind>`: current schedule envelope only; zero after recovery.
- `effect_level.<kind>`: realized lagged disturbance or persistent damage.
- `fault_label.<kind>`: `1` when effect level exceeds 0.001, otherwise `0`.
- Normal: all effect labels zero, not merely no active schedule.

A tiny effect label is not a guarantee that the available readings can detect
the fault. Evaluate incipient versus developed fault performance separately;
some sensor biases and early/slow damage are intentionally ambiguous. Multi-label
classification is required when disturbances overlap or damage carries forward.

## Healthy reference and diagnostic boundary

The simulator retains same-context fault-free `expected` and verbatim
`raw_legacy` values for controlled experiments, but neither enters feature
engineering or inference. The analytic module explicitly allowlists sensor
names and numerical context and ignores `truth`, `expected`, schedules and
categorical injected quality labels—even when callers supply them.

Residuals use transparent synthetic healthy priors conditioned on physical
context. Frame strain is an observed load/stress proxy. Thermal/flow reference
states evolve independently of the possibly faulty temperature/flow readings.
Condition offsets can be commissioned from the first reading, with bounded
offsets and separate low-condition checks. In a real truck these priors need
calibration against trusted healthy measurements. Reference error is not proof
of a physical fault.

Persistence is 15 seconds for oil pressure and 30 seconds for other residuals,
widened in transitions. Historical windows reset per engine/episode. Lead–lag
features reflect residual-onset association, not demonstrated causal direction.
Residual-slope windows span at least 120 seconds and three sampling intervals,
so sparse longitudinal readings still produce a measurable rate. A hydraulic
plateau with no identifiable rate returns an unresolved degradation hypothesis.
Hydraulic fast/slow boundaries are explicitly illustrative and can become
ambiguous under accelerated aging or after the decline reaches a plateau.

Root cause always remains unconfirmed, and the output supplies evidence,
alternatives and targeted inspection. Heuristic health is a prioritisation
index, not a failure probability. RUL is deliberately unavailable.

The reference priors are approximate. A separate stress audit of 24 fault-free
episodes (1,464 samples: ideal/stress, empty/medium/overload, cold/warm and
ambient -5/42 °C) produced no named mechanical-fault hypotheses, but did produce
56 isolated/sensor-ambiguity and 368 unmatched-multiple-deviation hypotheses.
These are not verified sensor failures. In particular, inherited legacy
perception degradation and route-dependent brake wear can disagree with the
reference prior. Do not use these heuristic hypotheses as training labels or
claim a low total false-alarm rate from the mechanical-signature regression.

## Python API

```python
from enhanced_simulation import simulate_enhanced
from enhanced_analytics import analyse_episode, engineer_features

episode = simulate_enhanced(
    {"seed": 793, "engine_seed": 91, "engine_id": "virtual-001",
     "episode_id": "test-001", "payload_class": "heavy",
     "ambient_c": 35, "duration_s": 1800, "timestep_s": 5},
    [{"kind": "gearbox_bearing_wear", "onset_s": 300,
      "duration_s": 900, "severity": .8, "progression": .3,
      "recovery_s": 120}])
features = engineer_features(episode["samples"])
diagnostics = analyse_episode(episode["samples"])
# Explicit continuation: next config can include metadata.final_engine_state.
```

The Android port should reproduce this numerical schema, units, event envelope
and feature windows. Use matched Python reference trajectories for portability
checks. No Android application or DL model is installed by this update.

## What cannot be generated honestly

- Real confirmed repairs, maintenance-case similarity probabilities and field
  failure/censoring outcomes require independent fleet evidence.
- The simulator's fault catalogue is not a real case-history database.
- Legacy fuel/coolant/exhaust energy channels remain available, but fault
  overlays do not recompute an energy-conserving physical engine model.
- Synthetic spectral weights depend on scalar readings; spectra are therefore
  not independent diagnostic evidence. Validate with actual high-rate sensors.
- Distinct mechanisms may be observationally indistinguishable. A classifier
  should support uncertain/unknown outputs, not force a confident root cause.

## Checks

```bash
python3 -m pytest -q
python3 -m pytest -q test_enhanced_simulation.py test_enhanced_analytics.py \
  test_enhanced_dataset.py test_enhanced_dashboard.py
```

Streamlit UI tests require the project's Streamlit dependencies. Tests cover
matrix distinctions, bounds, deterministic trajectories, event phases, wear
continuation, sensor failure/recovery, reference/label isolation, temporal
resets, engine-disjoint splits, overwrite protection, FFT and actual UI replay.
