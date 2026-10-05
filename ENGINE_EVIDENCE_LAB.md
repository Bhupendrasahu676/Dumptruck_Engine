# Engine evidence & lifecycle lab

The new **Engine Evidence Lab** page is available in the sidebar of both
`app_3d.py` and `cabin_dashboard.py`. It also runs independently:

```bash
uv run --with 'streamlit>=1.36,<2' streamlit run engine_health_lab.py \
  --server.headless true --global.developmentMode false \
  --server.address 127.0.0.1 --server.port 8503
```

Open **http://localhost:8503**. For a trusted Android/tablet LAN deployment,
bind to `0.0.0.0` and use the host's LAN IP, not the tablet's `localhost`.
This development app has no authentication: do not expose it to the public internet.

## Start-to-finish workflow

1. **Register engine** in the sidebar. This creates persistent UUID `Engine_ID`
   and baseline `Overhaul_ID`. Initial operating hours are historical hours, not
   inferred wear; new registrations use an explicitly synthetic zero-wear baseline.
2. Choose start temperature condition, fixed payload category, seed, duration,
   sample interval and ambient temperature.
3. Add fault schedules. Each has onset, active duration, severity, repeat interval,
   progression and recovery. Multiple fault types can run together.
4. **Generate preview**. No database write or hour advancement occurs. Observations
   are passed through an explicit allowlist into independent residual analysis;
   injection labels and latent wear are excluded.
5. Inspect samples, expected versus observed readings, residuals, quality,
   anomaly persistence, hypotheses, alternatives and recommended checks.
6. **Save test & advance engine hours / wear**. A new `Test_ID` is recorded,
   with immutable samples, analysis, configuration and reproducibility metadata.
   Saving the same result twice is idempotent. Conflicting IDs, stale engine state
   and outdated overhaul IDs are rejected atomically.
7. Add technician disposition to a saved sample. A confirmed fault requires cause,
   inspection evidence and technician identity. Feedback is append-only and never
   overwrites injected truth or the original unconfirmed diagnostic prediction.
8. In **History & overhaul**, record component repair fractions and work notes.
   A new overhaul ID is created, selected wear is reduced, component before/after
   values are retained, and total operating hours do not reset. Unrepaired wear
   persists into the next test.
9. In **RUL feasibility**, inspect unmet requirements. No number appears simply
   because a health score is available.

Default local database: `telemetry_logs/engine_lab.sqlite3` (SQLite, created on
first use). `ENGINE_LAB_DB` can point to a separate authorised local database.
Back up the database while the app is stopped; it contains engine history and
potentially private technician notes. Do not commit it or use it as training data
without appropriate handling. Tests use temporary databases, never this live file.

## Implemented fault scenarios

| Scenario | Behaviour |
|---|---|
| Lubrication degradation | Progressively lower pressure relative to RPM/thermal baseline, oil heating and accelerated wear |
| Intermittent oil starvation | Larger pressure dips and vibration during load transitions; scheduled recurrence and recovery |
| Bearing degradation | Increasing excess vibration with possible pressure/temperature corroboration |
| Cooling restriction | Reduced coolant flow and elevated temperature; persistent cooling wear |
| Thermostat malfunction | Delayed opening, stuck-open or stuck-closed response |
| Combustion imbalance | Unequal cylinder EGT changes plus vibration; no raw combustion waveform claim |
| Sensor fault | Biased, frozen or missing readings for a chosen channel; latent engine and physical wear unchanged versus the same healthy run |
| Incomplete overhaul | Lubrication abnormality improves partway while bearing abnormality persists; separate recorded partial repairs preserve unrepaired wear |

Repeat interval means **time between event starts**, not a pause after the event.
Overlapping repeats of one schedule use the strongest current envelope. Severity
increases as `severity × (1 + progression × event_fraction)`, capped at one.
Recovery linearly removes the active disturbance; it does **not** undo accumulated
wear. Missing channels remain missing until their event/recovery window finishes.
The backend accepts several schedules of the same type; the basic UI provides
one editor per selected fault type.

## Three different conclusions

- **Anomaly:** usable observations differ from the healthy state-conditioned
  reference. Duration, direction, threshold and trend are recorded.
- **Probable fault:** persistent evidence supports hypotheses with alternatives
  and confirmation steps. A pressure-only deviation remains ambiguous between
  sensor bias and lubrication trouble.
- **Root cause confirmed:** only a separate, explicitly entered human inspection
  record. Technician identity is self-reported in this local prototype, not
  authenticated or independently certified.

Normal cold warm-up, lower idle oil pressure and ordinary load-transition vibration
are part of the healthy reference. Pressure persistence generally requires 15 s,
other channels 30 s, and transitions 45 s. Short residual excursions remain visible
as anomalies without being promoted automatically to persistent mechanical faults.
These are illustrative screening settings, not safety interlocks or OEM limits.

Missing, nonfinite, implausible or frozen channels are masked. Invalid measurement
quality makes overall health **unknown**, rather than inventing a mechanical
failure. Slow isolated residual drift is explicitly ambiguous. Health is a
conservative unvalidated subsystem residual heuristic, not a probability or life
fraction. Evidence strength is qualitative; there are no fixed confidence percentages.

## Files and data boundaries

- `engine_lab/simulation.py`: deterministic physical/measurement model, priors,
  schedules and hidden truth.
- `engine_lab/analytics.py`: observation-only quality, state, residual, persistence,
  hypothesis and RUL-gating functions.
- `engine_lab/storage.py`: atomic SQLite lifecycle, history and feedback.
- `engine_lab/workflow.py`: allowlisted inference boundary, preview and exports.
- `engine_health_lab.py`: Streamlit workflow.
- `pages/1_Engine_Evidence_Lab.py`: shared dashboard page entry.
- `test_engine_lab.py`, `test_engine_lab_ui.py`: regression and UI tests.

Downloads keep **observation CSV/JSONL**, **prediction JSONL**, **ground-truth JSONL**,
**human-feedback JSON** and **RUL gate receipts** separate. IDs are grouping metadata,
not predictor features. The full test record retains schedules and truth for
evaluation, but these never enter `analyse_samples` through the workflow.

Existing `telemetry_model.py`, generated datasets and previous dimensionality
analysis are not rewritten. The original dashboard/replay remains the legacy
model; its health/RUL numbers have not become calibrated CPAI predictions.

## RUL feasibility and empirical uncertainty

There is deliberately **no built-in trained prognostic model** and no numerical
RUL derived from a health-to-life formula. The calibration interface accepts
external model predictions with independent outcome evidence. Its local policy
requires:

- At least five distinct saved tests, at least 100 operating hours of history,
  a sustained five-point-or-greater health decline, and one overhaul segment.
- A current usable endpoint matching the engine's saved hours. Stale or missing
  measurements cannot support an up-to-date forecast.
- Explicit model/domain/failure-definition metadata; a model frozen before
  calibration; prediction produced before failure outcome inspection.
- At least 19 calibration engines and 10 untouched test engines, each with an
  observed failure and one outcome record. Right-censored records cannot be
  substituted for failures.
- Disjoint training, calibration, test and current-engine identities.

These sample counts and health-history screens are **project demonstration policy**,
not scientifically established sufficiency thresholds or CPAI certification criteria.
Healthy baselines come from the simulator itself; matching them is not independent
validation against real machines.

The exact JSON schema is documented in `engine_lab.analytics.assess_rul`:

```text
current_engine_id: selected engine UUID
model_id, domain_id, failure_definition: nonempty strings
source: "simulation" (the lab rejects "field" for its synthetic current engine)
training_engine_ids: unique IDs
model_frozen_before_calibration: true
prediction:
  engine_id, operating_hours, overhaul_id, remaining_hours
  model_id, domain_id, failure_definition, source
calibration_outcomes: list (>=19)
test_outcomes: list (>=10)
```

Every outcome row contains:

```text
engine_id, operating_hours, remaining_hours (pre-outcome model prediction),
failure_hours (actual absolute engine hours at the defined failure),
failure_observed: true, predicted_before_failure: true,
model_id, domain_id, failure_definition, source
```

Calibration uses absolute errors between predicted and observed remaining hours.
The radius is the `ceil((n+1) × 0.90)` order statistic from **calibration only**.
At least 80% of untouched test outcomes must fall within that radius as a coarse
miscalibration check. The nominal 90% split-conformal interval assumes independent,
exchangeable engines and a fixed compatible model. The small test check does not
prove 90% population coverage; it is not an individual failure probability.
Never tune repeatedly against the test cohort and continue to call it untouched.

Input provenance flags are declarations, not independently verified evidence.
Numerical output is labelled **simulation_only**, with limitations, or is withheld
with explicit reasons. Real calibration data, model fitting, censored-survival
modelling and field validation remain future work.

## Verification

```bash
python3 -m pytest -q test_engine_lab.py
uv run --with 'streamlit>=1.36,<2' --with pytest python -m pytest -q
```

Tests cover event timing/progression/recovery, all fault scenarios, sensor/physical
isolation, warm-up/idle/transitions, persistence, ambiguous diagnosis, JSON safety,
fault-label leakage, SQLite restart/idempotency/stale-write protection, partial
overhauls, immutable technician feedback, split exports and RUL rejection/calibration.
The Streamlit AppTest exercises registration → scheduled preview → save → sample
review → technician confirmation → partial overhaul → reopen persisted test.

Verification on 2026-10-03: **89 tests passed** (legacy + new backend + Streamlit
session test). Python compilation passed for all new modules and both existing
dashboards. These software checks do not establish model accuracy on real engines.

## Remaining modelling limits

- This additive lab is a reduced-order six-cylinder demonstration using an
  illustrative repeating 180-second operating profile, not the existing truck's
  entire mechanical/perception schema or a measured test-bench protocol.
- Fault aging is accelerated; numerical wear and thermal parameters are not
  fitted to real failure histories. Independent random seed does not establish
  independent real engines.
- Sampled RMS-style vibration is not a high-rate waveform or bearing spectral
  diagnosis. Events shorter than a sample interval may be missed.
- Recorded component repairs alter latent wear; they do not prove a physical
  maintenance action occurred. Human feedback is traceable local data, not an
  authenticated production approval system.
- Neither the legacy dashboard nor this new workflow is certified CPAI compliance,
  an automatic overhaul-release gate or a replacement for OEM safety procedures.
