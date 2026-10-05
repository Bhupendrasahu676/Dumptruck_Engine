# Dumper Onboard AI — Android Slave and Web Master Specification

**Status:** implementation-ready product/architecture proposal, not an implemented app.  
**Basis:** the user's confirmed choices, current enhanced simulator and compact16 model.  
**Purpose:** simulate truck sensors locally, run fault detection and sensor-health assessment on the truck Android device, and securely present each truck's transmitted data on a fleet web dashboard.

> This is a **synthetic demonstration/pilot**, not a validated vehicle-control,
> safety, physical root-cause or RUL system. AI-negative results must not override
> approved inspection or shutdown procedures. No vehicle actuator control is included.

## 1. Confirmed scope

| Decision | Agreed requirement |
|---|---|
| Master interface | Web application with master login and authorised fleet dashboard |
| Slave interface | Android application with operator login and truck/device assignment |
| Accounts | Created by an administrator; server-assigned roles and fleet permissions |
| Initial hosting | Local network/laptop server first, with a documented later internet migration |
| Pilot scale | 5–20 simulated trucks; Android 10+ phones/tablets |
| Data generation | CSV/reference replay first, then a locally running Android simulator |
| AI placement | Onboard Android inference, not dependent on server inference |
| Fault injection | Slave-side simulation controls only; no master injection/start/stop commands |
| Display timing | Approximately 1-second UI refresh, original model observation cadence preserved |
| Offline | Slave simulation, inference and health continue offline; telemetry queues for upload |
| Version-one features | Fault evidence/possible causes/mitigations, notifications, acknowledgement, technician confirmation |
| Health | Independent sensor residual/persistence assessment; missing essential evidence means UNKNOWN |
| TPM | Six tyre positions with the same fixed demo ideal pressure; display on slave and master |
| RUL | Unavailable until supported by independent failure histories |

**Not included in version one:** fleet map, historical chart exploration, user-facing CSV exports, real sensor/CAN integration, remote vehicle control, remote fault injection, model retraining on the phone, guaranteed offline browser notifications, or real physical RUL. Recent state and durable alert/inspection records are still needed for the selected workflows; this is not a historical analytics product.

### Success scenario

1. Administrator registers a fleet, truck device and users.
2. Slave operator signs in and starts a simulated journey.
3. Operator chooses a fault scenario/schedule on the slave.
4. **Actual simulated readings change**; the fault label is not supplied to AI.
5. The slave calculates healthy-reference residuals and the exact 16 model inputs.
6. Local DL inference produces multi-label fault probabilities; separate rules calculate health and probable-cause evidence.
7. Slave shows the readings, health, alerts and recommended checks, even offline.
8. Backend receives the signed-in device's data; master sees that truck and all transmitted channels.
9. Master acknowledges an alert. A technician later records an inspection/repair separately from the prediction.

## 2. Architecture and boundaries

### Logical topology

```text
Android slave devices, each registered to one truck
  local replay/live simulator
    -> raw observations + physical context
    -> quality checks + healthy-reference/temporal features
    -> onboard compact16 TCN
    -> calibrated probabilities + frozen per-fault thresholds
  raw observations independently
    -> sensor health + probable RCA evidence + local alerts
  both paths
    -> local Room database / durable outbox
    -> MQTT over TLS

Local server
  MQTT broker with per-device ACLs
    -> ingestion/validation/deduplication
    -> PostgreSQL/latest-state cache + alert/inspection records
    -> authorised REST and WebSocket gateway

Web master
  authenticated fleet access
    -> fleet overview / full truck telemetry / alerts / technician workflow
```

The broker/server mediates master–slave communication. This is **not** a direct peer-to-peer phone connection, and the master does not have to stay online for onboard inference to work. The backend validates and routes messages; it does not silently substitute its own classifier outputs for onboard results.

### Recommended stack

| Layer | Proposed technology | Reason |
|---|---|---|
| Android UI | Kotlin, Jetpack Compose, Material 3, ViewModel/StateFlow | Responsive phone/tablet layouts and explicit lifecycle state |
| Local persistence | Room, app-private files, Android Keystore-backed key storage | Durable queued samples and protected device/session keys |
| Local compute | Kotlin simulator/feature/health modules + mobile model runtime | Offline deterministic processing |
| Model runtime | Evaluate **ONNX Runtime Mobile for Android** first | Direct PyTorch export path; conversion/parity remains a release gate |
| Alternative runtime | LiteRT or ExecuTorch only after compatibility tests | Do not change runtimes without verifying the complete preprocessing/output pipeline |
| Telemetry | MQTT 3.1.1 or 5, TLS, QoS 1 | Small messages, reconnect behaviour and per-device topic controls |
| Backend | FastAPI, MQTT ingestion worker, PostgreSQL | Authentication, durable ingestion, scoped APIs and workflow records |
| Master | React + TypeScript web app | Fleet overview and live truck detail |
| Local deployment | Docker Compose, reverse proxy/TLS, private broker | Repeatable laptop/server deployment without public ports |

These are proposed implementation choices, not installed dependencies or completed services. Use a supported MQTT Android client only after foreground/reconnect tests; do not assume every desktop library supports Android lifecycle behaviour.

## 3. Roles, login and device identity

### Roles and permissions

| Role | Allowed | Not allowed |
|---|---|---|
| Administrator | Create/disable users, assign roles/fleets/trucks, enrol/revoke devices, configure approved policies | Alter immutable past predictions or fabricate technician confirmations |
| Master | View assigned fleet/trucks, full transmitted telemetry and alerts; acknowledge and assign inspections | Inject faults, start/stop a slave simulation, view other fleets, confirm an inspection without technician permission |
| Slave/operator | View assigned truck, run replay/local simulation, inject demo faults, view local AI/health, acknowledge local alerts | Choose a master role at login, change fleet assignment, issue remote truck commands |
| Technician | View assigned inspection evidence, record findings/actions/confirmation, request resolution | Rewrite original readings, model output or injected truth |

A person can receive more than one role only through an explicit admin assignment. Technician capability can use a separate account or an additional permission on a master/operator account. Server enforcement is mandatory; hiding a button is not access control.

### Account lifecycle

1. Administrator creates user and assigns fleet/truck permissions.
2. User receives a one-time activation link/code with a short expiry; no shared permanent demo passwords.
3. First login requires setting a password. Production migration should add MFA for admin/master accounts.
4. Backend hashes passwords with a modern password-hashing algorithm, rate-limits login and emits audit records.
5. Short-lived access sessions and rotating/revocable refresh sessions are issued.
6. Android stores session secrets only in protected app storage; browser uses secure HttpOnly session cookies, with CSRF protection for state-changing operations.
7. Logout clears user access locally; device revocation stops accepted telemetry on the server.

There is no public self-registration and no client-side role selector granting permissions. Account recovery, disablement and fleet reassignment are admin flows with an audit trail.

### Device enrolment is separate from operator login

- One Android installation has a unique `device_id`; it is bound server-side to a `truck_id` and `fleet_id`.
- Administrator creates a short-lived, single-use enrolment token, optionally represented by a QR code.
- Device exchanges it for an individual identity/certificate or narrowly scoped device credentials.
- Broker authentication and ingestion derive fleet/truck identity from this registered principal. The server rejects payloads pretending to belong to another truck.
- Operator credentials do not become MQTT device credentials. Operators can change while the registered truck identity stays stable.
- Truck reassignment requires a new server-approved binding, a new run/session and a visible operator confirmation.
- Never embed administrator, shared fleet or unrestricted broker credentials in the APK.

### Offline authorisation

Initial account activation, device enrolment and first sign-in require the server. A previously authorised operator may continue a **local simulation** under a bounded cached offline grant (proposed default: 24 hours); show the offline status and grant expiry. Offline status must not grant new roles, truck assignments or server workflow permissions.

Remote revocation cannot be instantly learned while disconnected. Document this limitation, reject revoked credentials on reconnect, and require reauthentication when the cached grant expires. Pending samples remain local rather than being uploaded under invalid credentials. Acknowledgements created offline are pending until the server accepts them.

## 4. Android slave screens and flows

### Screen list

| Screen | Required contents |
|---|---|
| Login | Server address/profile, username, password, activation/recovery guidance, connection/error state |
| Truck assignment | Operator identity, authorised truck, device enrolment state, fleet, simulation-only banner |
| Truck dashboard | Weighted/minimum health, subsystem health, six-tyre TPM diagram/cards, key sensor cards, predicted faults, quality/freshness, local/server connectivity, model version |
| All sensors | Grouped searchable numerical channels with units, value/null status, operational band, residual if defined, measurement age |
| Simulation setup | Replay/live mode, scenario, fixed payload, warm/cold start, ambient, seed, journey reset/start/pause/stop |
| Fault injection | Fault class, severity, onset, duration, progression, recovery, recurrence; sensor target/mode/bias; multiple schedules |
| Fault detail | Observed evidence, DL pattern, threshold, persistence, alternatives, inspection/mitigation, acknowledge status |
| Technician inspection | Assigned records, findings, actual cause or inconclusive result, action taken, repair/evidence notes |
| Sync/model status | Queue count, last accepted sample, errors, active model/package hash, policy versions; no secret values |

Screen layout should be tablet-friendly, usable in landscape and portrait, with legible numerical cards and colour **plus text/icons**. Do not rely solely on green/red. The one-second UI timer updates sample age, status and animations; it must not invent a new measurement or trigger another inference on the same sample.

### Start/pause/reset behaviour

- A new journey creates a new `run_id` and clears feature/reference state and the TCN buffer.
- Payload **category remains fixed** for a journey; changing payload starts a new run.
- Pause freezes **simulation time**, not wall-clock freshness/connection state. Master shows `PAUSED`, not a communication failure.
- Stop ends the run and flushes queued samples; it does not silently resolve active inspections.
- Resume restarts the scheduler without duplicated sequence numbers or skipped state updates.
- App interruption must not claim that continuous inference occurred. Restore a validated checkpoint of simulator/reference/buffer state or begin a new run with explicit reset/gap status.

## 5. Web master screens

| Screen | Required contents |
|---|---|
| Master login | Credentials, authentication status, fleet authorisation; no slave fault controls |
| Fleet overview | Authorised trucks, online/offline/paused/stale state, minimum/weighted health, active fault patterns, unacknowledged alerts |
| Truck detail | Every transmitted raw observation and context field, six individual tyre pressures/target/quality, onboard inference, health, probable RCA, versions, timestamps and queue status |
| Alert centre | New/acknowledged/inspection/resolved events, severity, truck, acknowledgement/assignment controls |
| Inspection detail | Original prediction/evidence, technician findings and confirmation, action/repair status, unresolved alternatives |
| Admin area | Users, fleets, truck assignments, device enrolment/revocation and policy access; visible only to admins |

“All data” means the complete authorised telemetry snapshot, not only the 16 ML inputs: original 113 numeric channels, the six new demo TPM pressure channels, physical context, health, inferred faults/evidence and versions. Use grouped sensor tables with progressive disclosure rather than placing every sensor card on one screen. Show unknown/unavailable fields explicitly, and distinguish the **last live snapshot** from later-arriving queued history.

Master must never silently display another fleet's telemetry. There is no remote simulator control or fault-injection API in version one.

## 6. Local simulation: replay first, live port second

### Phase-one replay

Ship **small approved reference journeys**, not the entire approximately 795 MiB training CSV. Each fixture contains raw observations, physical context, simulation timestamps and a separate truth manifest. Packages have deterministic identifiers/hashes and a source-simulator version.

Fault controls initially select a prerecorded fault trajectory/scenario with its recorded severity/onset; the UI must clearly indicate fixed/unsupported parameters. For repeatable demos, use matched healthy/faulty trajectories with the same base seed/context. Changing only a displayed flag or target label is **not** fault injection.

Do not jump between unrelated CSV rows mid-journey. A changed replay scenario starts a new run and resets reference/window state. If arbitrary live onset/severity/overlap is needed, wait for the tested Android simulator port; do not pretend a fixed replay supports it.

### Phase-two Android simulator

Port the legacy route plus `enhanced_simulation.py` semantics into an isolated Kotlin module with seeded RNG, bounded numerical values and no network dependency. Preserve:

- C-WTVC proxy: urban `0–900 s`, highway `900–1368 s`, expressway `1368–1800 s`.
- Exactly **171 observations**, including `t=0` and `t=1800`, with `dt=1800/170 ≈ 10.588235 s` for the current classifier contract.
- Scenario, categorical payload, start state, ambient, seed and engine variation.
- Condition channels separate from oil quantity; velocity vibration in mm/s separate from IMU acceleration in g.
- Onset, severity, progression ramp, duration, recovery, recurring/overlapping events.
- Illustrative pressure/flow-to-thermal-to-vibration lags, persistent wear and sensor-only faults.
- Deterministic feature/quality state with correct first-sample initialization and reset.

Current simulator RNG behaviour will not be reproduced just by using the same integer seed with Kotlin's default RNG. Either implement compatible RNG/noise sequencing or define a versioned new generator and validate its trajectories/distributions/model performance. Exact Python reference-fixture replay is the first numerical parity gate.

### Fault catalogue

| Model class | Simulated evidence/mechanism |
|---|---|
| `engine_oil_pump_wear` | Pressure loss and later vibration; primary oil condition preserved |
| `piston_ring_damage` | Oil condition/quantity changes, thermal/vibration effects, unequal/time-varying cylinder EGT |
| `turbo_exhaust_fault` | Local boost/EGT change, without mandatory whole-engine overheating |
| `coolant_system_failure` | Flow restriction followed by multi-channel heating |
| `coolant_degradation` | Coolant condition/pump-temperature changes, block/oil near context baseline |
| `gearbox_seal_wear` | Pressure loss/vibration, primary gearbox oil condition preserved |
| `gearbox_bearing_wear` | Gearbox pressure/vibration/oil-condition changes |
| `hydraulic_normal_aging` | Slow exposure-integrated condition decline and wear-related vibration |
| `hydraulic_contamination` | Faster decline and actuator vibration |
| `bearing_failure` | Engine bearing-style vibration/frictional heating |
| `low_oil` | Oil quantity and lubrication-pressure loss |
| `overheat` | Excess temperature without a uniquely proven cooling cause |
| `brake_wear` | Pad wear/stroke/brake-temperature changes |
| `perception_degraded` | Simulated GNSS/LiDAR/radar/camera-quality changes |
| `sensor_fault` | Missing, biased, frozen, drifting or noisy sensor value |
| `unrelated_multi_blip` | Multiple brief disturbances without proven common mechanical cause |

Sensor-fault controls operate on the selected observation, not mechanical wear. Condition indices remain 0–100, while rated payload utilization may exceed 100. Injected truth stays in a **separate simulation/debug store**, inaccessible to feature extraction, model inference and RCA. A demo truth comparison may be displayed separately; it must never replace the predicted fault.

### Three clocks, not one

1. **Wall clock:** UTC timestamps for connectivity and event audit.
2. **Monotonic device clock:** scheduling and elapsed-real-time freshness, unaffected by wall-clock changes.
3. **Simulation clock:** feature `elapsed_s`, event phase, sample cadence and operating-hour progression.

Normal replay uses 1× simulation speed. Optional accelerated replay must label its speed and preserve sample spacing in **simulation time**; do not process 1-second measurements as though they were the trained 10.588-second sequence. Changing to genuine 1 Hz observations is a separate resampling/retraining/validation task.

## 7. Onboard DL model and exact input contract

### Current research artifact

Use the **compact16 guarded TCN** as the initial conversion reference:

- Bundle: `analysis/results_compact16_only/selected_model.joblib`.
- Native weights: `analysis/results_compact16_only/models/tcn/model.pt`.
- Schema: `analysis/results_compact16_only/feature_schema.json`.
- Frozen choices: `analysis/results_compact16_only/frozen_policy.json`.
- Results/limits: `analysis/results_compact16_only/RESULTS_SUMMARY.md`.

The `.joblib` is a Python research bundle and **cannot be loaded directly into Android**. No Android-compatible model package or native inference app currently exists in this workspace. Converting weights alone is insufficient: preprocessing, reference state, target order, sigmoid calibration and decision thresholds must travel together.

### Exact 16 columns, in this order

| Index | Input |
|---:|---|
| 0 | `engine.oil_pressure_kpa__residual` |
| 1 | `engine.oil_condition_pct__residual` |
| 2 | `engine.boost_kpa__residual` |
| 3 | `engine.coolant_flow_kg_s__residual` |
| 4 | `coolant_condition_pct__residual` |
| 5 | `transmission.oil_pressure_kpa__residual` |
| 6 | `transmission.oil_condition_pct__residual` |
| 7 | `gnss.longitude_deg` |
| 8 | `suspension.hydraulic_oil_condition_pct__residual` |
| 9 | `engine.oil_level_pct` |
| 10 | `engine.coolant_temp_c__residual` |
| 11 | `brake.stroke_pct__residual` |
| 12 | `gnss.accuracy_m__residual` |
| 13 | `gnss.latitude_deg` |
| 14 | `transmission.vibration_mm_s__residual` |
| 15 | `engine.oil_temp_c__residual` |

No automatic column selection, PCA, extra context channels or newly learned scaler is allowed for this model. GNSS inputs are simulated observations, not the Android phone's real location. Changing them to real GPS changes the input distribution and needs validation and user/device location consent.

### Processing sequence

1. Validate original numeric readings and required operating context.
2. Evolve the **observation-only healthy reference**, using the same temporal state as `enhanced_analytics.py`.
3. For residual inputs, compute `observed − expected_healthy(context)`; do **not** consume simulator paired/oracle `expected`, event labels or latent wear.
4. Assemble exactly the ordered 16 columns.
5. Replace missing values with the **saved training medians**, preserving all 16 columns.
6. Apply saved training scaling: `z_i = (x_i − mean_i) / scale_i`.
7. Append `z` to the episode-local 12-reading buffer.
8. Infer logits with tensor shape **`[1, 16, 12]`**, float32, ordered oldest-to-current along time.
9. Apply sigmoid, then saved per-label calibration, then the corresponding guarded threshold.
10. Store predictions with observation IDs, feature/model/calibration/threshold versions and quality status.

TCN probabilities are multi-label; **do not use softmax or require exactly one fault**. “No threshold exceeded” means no supported predicted pattern at this sample—not a safety clearance.

For a short prefix, Python repeats the episode's first observed feature row on the left. Reproduce that padding for parity, while showing **WARMING UP / provisional** until 12 actual observations exist. Do not fill future rows. Reset on journey/truck/sample-contract change; restore only an exact validated state after interruption.

### Calibration and threshold export

The Python helper applies, per fault, a saved logistic transform to clipped probability log-odds. Export the same coefficients/intercepts and clipping conventions, rather than recalibrating on live traffic:

```text
raw_p = sigmoid(logit)
clipped_log_odds = saved_python_clipping_and_log_odds(raw_p)
calibrated_p = sigmoid(a_k * clipped_log_odds + b_k)
prediction_k = calibrated_p >= threshold_k
```

A target without a fitted calibrator passes through unchanged. Retrieve all numbers from the trusted bundle; never invent default `0.5` thresholds. TCN target order is the 16 catalogue keys in the source schema; fault truth and mitigation text are not target-input features.

### Mobile package

| File | Contents |
|---|---|
| `model.onnx` or verified runtime equivalent | Exported compact16 TCN logits graph |
| `model_manifest.json` | Signed/hash-verified package identity, compatible app/runtime versions and research-only status |
| `feature_schema.json` | Exact inputs, targets, units, tensor orientation, cadence and 12-step window |
| `preprocessing.json` | Training medians, means/scales, finite-value handling |
| `calibration.json` | Per-target parameters and passthrough flags |
| `thresholds.json` | Guarded thresholds/policy version |
| `healthy_reference.json` | Versioned constants/rules plus matching Android state-machine implementation |
| `parity_fixtures.jsonl` | Raw context/observations, expected features/logits/calibrated outputs/decisions for validation |

Packages must be immutable, signed, versioned and activated atomically only after validation. Install the first package with the app or explicitly through an authorised update flow; don't silently download/execute arbitrary Python. Keep the last known-good package for rollback and include its identity in every sample. Server/model update management is a proposed deployment mechanism, not permission for master fault controls.

### Conversion release gates

- Export a fixed-shape graph; verify all TCN operators, dilation, padding, causal cropping and logits are supported by the chosen Android runtime.
- Compare raw-feature, preprocessed-feature, logits, calibrated-probability and decision parity against Python, including normal, developing, mixed, missing and cold-start fixtures.
- Test first-prefix padding and reset; no data can cross truck/journey boundaries.
- Use declared numerical tolerances and inspect samples near thresholds. Current CUDA test batches already showed up to approximately **0.001204** probability variation by batch shape; do not promise bit-identical mobile decisions.
- Start float32. Quantisation is a later optimisation requiring full regression and threshold/metric review; unchanged file shape is not proof of equivalent inference.
- Do not substitute the dense model automatically if TCN export fails. A fallback must have its own matching preprocessing/calibration/threshold package and explicit acceptance.

### Evidence quality and model limitations

Current selected guarded TCN: original held-truck macro AP **0.8496**, macro F1 **0.8131**, normal-row alarms **5.33%**. A separate previously evaluated seed batch gives AP **0.8541**, F1 **0.8075**, normal-row alarms **3.29%**. These are **synthetic research results**, not field accuracy.

Slow hydraulic aging and sensor-fault recall are particularly weak. An absent alert does not exclude these causes. Imputation lets the graph run, but cannot restore missing evidence: label inference `DEGRADED/UNRELIABLE` when required inputs/context are missing, frozen or invalid. Show fault hypotheses for inspection where appropriate, but never display “healthy” merely because imputation produces low probabilities. The quality gate is separate from the 16 model-input columns.

## 8. Sensor-derived health score

Compute health independently from the classifier, using the **complete necessary sensor set**, not only the compact16 vector. This includes engine, cooling, transmission, hydraulic, brake, structure and perception channels. The slave transmits these original channels even though DL sees only 16 features.

### Compatibility-first formula

Implement and test the existing `enhanced_analytics.py` residual thresholds,
quality checks, widened transition/warm-up thresholds and persistence semantics.
For persistent signal evidence:

```text
r_i = observed_i - healthy_reference_i(context)
q_i = clip(abs(r_i) / threshold_i / 4, 0, 1)
q_i = 0 if there is no persistent valid anomaly

subsystem_health_s = 100 * (1 - max(q_i for required signals of subsystem s))
```

This is the current conservative heuristic, **not** a trained failure probability. Structure additionally uses the versioned frame-strain rule (current severity proxy `clip((strain_microstrain − 900)/1200, 0, 1)`). Keep explicit pressure/thermal safety rules independent of this aggregate score and of DL thresholds.

Normal warm-up/load can change raw Green/Amber/Red statuses without supporting a fault. Context-conditioned residuals, persistence and supplied operational bands must remain distinct UI concepts. Proposed health-band captions are trend/inspection guidance, not a clearance to operate.

### Aggregate display

When all required subsystems have valid evidence:

```text
minimum_health = min(engine, cooling, transmission, hydraulics,
                     brakes, structure, perception)

weighted_health = 0.30*engine + 0.15*cooling + 0.15*transmission
                  + 0.15*hydraulics + 0.10*brakes + 0.10*structure
                  + 0.05*perception
```

Show both: weighted health for the trend card, minimum/subsystem score for conservative warnings. A good average must not hide one critical subsystem. Do not multiply by DL fault probabilities; this could double-count the same sensor anomaly without validated calibration.

If a subsystem's essential channels/context are missing, invalid, frozen or stale:

- Its score is `null` and its status is **UNKNOWN**, not zero or 100.
- Aggregate scores are **UNKNOWN** until coverage is restored.
- Known subsystem scores remain visible; do not silently renormalise the weights and present incomplete evidence as complete health.
- Display coverage, stale age and affected measurements.

Health values, policies and minimum/weighted calculations are explicitly labelled **synthetic heuristic**. RUL is `null`, with `unavailable_not_validated`; no conversion of a health score into physical lifetime is included.

### TPM — tyre-pressure monitoring with ideal demo values

Add a dedicated **TPM** panel on the Android slave and the master's truck-detail
screen. Use a six-position demo layout: two front tyres and four rear tyres.
Each tyre displays its name, pressure, ideal target, unit, measurement age,
source and availability/status.

**Uniform demo ideal target: 700 kPa = 7 bar ≈ 101.5 psi for every tyre.**
This is an illustrative constant selected for the simulation, **not** a verified
Cat 793F/OEM cold-inflation specification. Before real deployment, a technician
must configure the approved tyre/manufacturer/load/temperature-specific target.
Do not use this document's demo pressure as a real inflation instruction.

| Tyre position | Telemetry channel | Initial and ongoing demo pressure |
|---|---|---:|
| Front left | `tyres.front_left.pressure_kpa` | 700 kPa |
| Front right | `tyres.front_right.pressure_kpa` | 700 kPa |
| Rear left inner | `tyres.rear_left_inner.pressure_kpa` | 700 kPa |
| Rear left outer | `tyres.rear_left_outer.pressure_kpa` | 700 kPa |
| Rear right inner | `tyres.rear_right_inner.pressure_kpa` | 700 kPa |
| Rear right outer | `tyres.rear_right_outer.pressure_kpa` | 700 kPa |

- Add these as **new synthetic pressure readings** in the app simulator/replay
  adapter. The original CSV has no actual tyre-pressure channels; do not relabel
  suspension corner pressure as tyre pressure.
- All six tyres remain at the fixed ideal value in this requested demo mode,
  regardless of other injected faults, payload or ambient temperature. No tyre
  leak, heating or wear mechanism is implied.
- The normal baseline is zero pressure deviation: `reading − ideal_target = 0`.
  Show **AT DEMO IDEAL / SIMULATED**, not a field safety certification.
- If TPM telemetry is missing or stale, show UNKNOWN/STALE for that tyre. Do not
  replace missing transmitted readings with 700 kPa on the master, masking a
  delivery or measurement failure.
- Keep canonical transport/storage units in kPa; convert only presentation
  units using `bar = kPa/100` and `psi = kPa/6.894757`.
- Include target/source/policy metadata such as `tpm_policy_version`,
  `ideal_pressure_kpa: 700.0`, `source: simulation_constant` and
  `oem_validated: false`.
- TPM is a separate monitoring panel. **Do not add these channels to the exact
  16-input DL model**, claim it diagnoses tyre faults, or change the existing
  seven-subsystem health weights. A future tyre-health contribution, abnormal
  pressure thresholds or tyre-fault classifier needs a separately reviewed
  policy and suitable data.

This addition changes the **app specification**; it does not state that the
Python simulator, historical CSV, model or Android application has already been
modified to generate TPM readings.

## 9. Probable RCA, mitigations and technician confirmation

Use two distinguishable sources:

1. **Onboard DL pattern:** multi-label model probabilities and threshold crossings.
2. **Observation-based RCA:** required sensor combinations, residual direction,
   persistence, rates, onset order and context, independent of injected labels.

Agreement strengthens a **probable hypothesis**; disagreement is visible. Do not merge either into a fabricated confirmed cause. A single oil-pressure drop can support a lubrication or sensor hypothesis; it cannot prove oil-pump wear by itself.

Each fault detail should contain:

- Pattern name, calibrated synthetic-model score, threshold and model version.
- Evidence: signal name/unit, reading, reference/residual, age and persistence.
- Probable cause(s), alternative explanations and what evidence is missing.
- Conservative inspection/mitigation catalogue entry with source/policy version.
- Status: **PROBABLE / INSUFFICIENT EVIDENCE / SENSOR QUALITY ISSUE**.
- Confirmation: initially **NOT CONFIRMED**, changed only by an authorised inspection record.

Mitigations are reviewed inspection guidance such as independently verifying oil pressure, checking leaks/filters, or following approved shutdown procedures—not executable commands and not automatic repair instructions. Generic simulation entries must be labelled synthetic and require site/OEM review before use on real vehicles.

Technician workflow:

```text
Alert -> acknowledgement -> inspection assigned -> inspection performed
  -> confirmed cause / inconclusive finding / different cause
  -> action/repair recorded -> verification required -> resolved or reopened
```

Record technician ID, truck/run/sample references, inspected symptoms, findings, confidence category, action taken, verification notes and server timestamp. Keep original predictions and evidence immutable. Use optimistic locking/revisions for conflicting updates.

Acknowledgement means someone saw an alert; it **does not** confirm root cause, fix the fault, clear health warnings or resolve an inspection. Stopping/resetting a demo journey ends that run but must not erase its audit/inspection records.

## 10. Alerts and notifications

### Alert classes

| Class | Trigger source | Behaviour |
|---|---|---|
| Critical sensor condition | Versioned pressure/thermal/vibration quality/safety rule | Immediate local warning and prioritised delivery; no dependency on DL availability |
| Probable fault | Valid model threshold + alert persistence/debounce | Onboard warning and master event with evidence |
| Measurement quality | Missing/invalid/frozen/stale essential evidence | UNKNOWN health or unreliable AI, targeted sensor check |
| Communication stale/offline | Heartbeat/sample timing | Master connectivity warning, not a fabricated mechanical fault |
| Model unavailable/incompatible | Runtime/package validation | AI unavailable; show independent health only if sensor/reference logic is valid |

Use configurable hysteresis, consecutive-sample persistence and a cooldown to avoid notification storms. Proposed demo default for noncritical DL alerts: two consecutive valid sample decisions; direct critical sensor rules bypass that delay. The threshold-validation metrics describe per-reading decisions, **not** guaranteed alert-event performance after this debounce policy—test the event policy separately.

### Event identity and lifecycle

- Alert fingerprint includes fleet, truck, run, fault/rule and policy version.
- Create one event when a condition becomes active; later samples update evidence, rather than issuing a new notification each second.
- `condition_state`: ACTIVE / RECOVERING / CLEARED / UNKNOWN.
- `workflow_state`: NEW / ACKNOWLEDGED / INSPECTION_PENDING / RESOLVED.
- Acknowledging a workflow cannot set condition_state to CLEARED.
- Persistent damage may continue after an injection schedule ends.
- Reappearance after verified resolution creates/reopens a traceable event by explicit policy.

### Platform notification guarantees

- Android uses local notification channels for critical/maintenance/connectivity events. Handle denied notification permissions visibly (Android 13+), including an in-app warning fallback.
- Master receives WebSocket events while its browser is connected. Browser notifications require permission and a secure context; handle unsupported/denied permissions with in-app alerts.
- On a local-only network, do not promise background browser delivery when the browser/server is closed or unreachable. Guaranteed background/mobile push requires an additional approved push architecture and is outside version one.
- A slave offline critical event still alerts locally and enters its durable outbox. Master learns it only after connectivity resumes; show event occurrence time and delivery time separately.

## 11. Telemetry transport and message contract

### MQTT topics

```text
fleets/{fleet_id}/trucks/{truck_id}/devices/{device_id}/telemetry
fleets/{fleet_id}/trucks/{truck_id}/devices/{device_id}/events
fleets/{fleet_id}/trucks/{truck_id}/devices/{device_id}/heartbeat
fleets/{fleet_id}/trucks/{truck_id}/devices/{device_id}/receipts
fleets/{fleet_id}/trucks/{truck_id}/devices/{device_id}/workflow-updates
```

A device may publish only its authorised telemetry/events/heartbeat topics and subscribe only to its ingestion receipts and narrowly scoped workflow updates. A device cannot subscribe to fleet-wide wildcard telemetry. Master browsers do **not** receive broker credentials; they connect to backend REST/WebSocket endpoints with server-enforced fleet permissions.

The server alone publishes receipts/workflow updates. Workflow updates carry accepted acknowledgement/inspection status, never fault-injection or simulator-control commands. Authenticate operator acknowledgement/inspection writes through the scoped HTTPS API, not by treating a device's MQTT identity as a master/technician user.

No master-command or fault-injection topic exists in this version. Do not expose backend mutation endpoints for simulation start/stop/injection.

### Delivery semantics

- QoS 1 is **at-least-once**, not exactly-once. MQTT PUBACK means broker receipt, not durable backend storage.
- Outbox entries become accepted only after a backend application receipt acknowledging durable ingestion.
- Stable IDs survive reconnect: `(device_id, run_id, sequence_number)` or an equivalent `message_id`.
- Backend has a unique constraint and idempotent ingestion; duplicate retransmission does not duplicate samples/events.
- Out-of-order arrivals are stored by original sequence/time but do not regress the live latest-state view.
- Reconnected old data is marked backfill and cannot mark a currently offline truck healthy/online.
- Use exponential reconnect backoff with jitter, bounded batch sizes and separate high-priority event delivery. Do not hammer a failed broker.
- Configure heartbeat/LWT for connectivity, but do not treat broker LWT as a mechanical fault or as guaranteed immediate device-off detection.

### Proposed telemetry payload

The example is abbreviated; production schema includes **all transmitted raw channels**, units and explicit nulls, not only the two shown below.

```json
{
  "schema_version": "dumper-telemetry/1",
  "message_id": "device-A:run-123:42",
  "fleet_id": "fleet-demo",
  "truck_id": "truck-01",
  "device_id": "device-A",
  "run_id": "run-123",
  "sequence_number": 42,
  "source": "simulation_replay",
  "observed_at_utc": "2026-10-05T10:00:00Z",
  "simulation_elapsed_s": 444.705882,
  "simulation_speed": 1.0,
  "sampling_interval_s": 10.588235,
  "operating_context": {
    "start_condition": "warm",
    "payload_class": "medium",
    "ambient_c": 25.0,
    "mode": "hauling"
  },
  "observations": {
    "engine.oil_pressure_kpa": 410.0,
    "engine.oil_temp_c": 98.0,
    "tyres.front_left.pressure_kpa": 700.0,
    "tyres.front_right.pressure_kpa": 700.0,
    "tyres.rear_left_inner.pressure_kpa": 700.0,
    "tyres.rear_left_outer.pressure_kpa": 700.0,
    "tyres.rear_right_inner.pressure_kpa": 700.0,
    "tyres.rear_right_outer.pressure_kpa": 700.0
  },
  "tpm": {
    "policy_version": "tpm-uniform-demo-1",
    "ideal_pressure_kpa": 700.0,
    "source": "simulation_constant",
    "oem_validated": false
  },
  "measurement_quality": {
    "engine.oil_pressure_kpa": "good"
  },
  "ai": {
    "model_version": "compact16-tcn-demo-1",
    "feature_schema_version": "compact16-1",
    "calibration_version": "calibration-1",
    "threshold_policy_version": "guarded-1",
    "status": "valid_simulation_evidence",
    "fault_scores": {"engine_oil_pump_wear": 0.72},
    "predicted_faults": ["engine_oil_pump_wear"]
  },
  "health": {
    "policy_version": "sensor-health-1",
    "subsystems": {"engine": 65.0},
    "weighted": null,
    "minimum": null,
    "coverage": "incomplete_example",
    "rul_hours": null,
    "rul_status": "unavailable_not_validated"
  },
  "root_cause_confirmed": false
}
```

The sample numbers are illustrative, **not** a recomputed inference/health result or a prescribed threshold. Implement schema validation for identities, declared units, finite values/nulls, percentage bounds, model versions, row/sequence shape and maximum size. Server adds `received_at_utc` and `accepted_at_utc`; it does not trust client time for authorisation or chronological latest-state alone.

Telemetry cannot assert a technician-confirmed root cause: reject/ignore client attempts to set confirmation true and derive confirmed findings only from authorised server inspection records. Onboard prediction evidence remains immutable and separate.

Measured/inferred `measurement_quality` must not be copied from the simulator's hidden `sensor_mode` label. Injected truth uses a separate optional simulation-debug envelope/table and clear access controls. It is excluded from model and RCA execution, and from any master “prediction” display.

### Timing and freshness defaults

| Item | Proposed pilot default |
|---|---|
| UI redraw | About 1 second; no new sensor value inferred from redraw |
| Raw sample/inference | 10.588235 simulation seconds at 1× |
| Heartbeat | Every 5 wall-clock seconds while the run is active |
| Stale connectivity | No heartbeat for 20 seconds, configurable |
| Offline connectivity | No heartbeat for 40 seconds or explicit LWT, configurable |
| Stale sensor snapshot | More than three expected real sample intervals, unless explicitly paused |
| Telemetry payload limit | Initial 64 KiB; measure representative full snapshots and enforce limits |

Use server receive-age for master connectivity and record device clock skew. A paused run can have a live heartbeat but no advancing samples. Present these states separately.

## 12. Offline storage, retention and Android lifecycle

### Local records

- Registered identity/session metadata (protected).
- Journey/episode configuration and simulator checkpoints.
- Raw observations with physical context and quality.
- Feature/reference state and 12-reading model buffer.
- Model/health outputs and alert/inspection records.
- Outbox IDs, attempt counts, priority, receipt state and enqueue times.

Use a transactional write: observation/output/event state and corresponding outbox entry either both persist or neither does. Local queue survives app/process death and device reboot. Restoring compute state must account for gaps; do not replay duplicate inferences silently.

### Proposed configurable retention

These are engineering defaults to review with the fleet administrator, not user-selected legal retention requirements:

- Local sample outbox: target 24 hours, capped by a 200 MiB app storage budget.
- Local critical events/inspection updates: preserve until accepted; never silently evict to make room for ordinary telemetry.
- Server raw telemetry: seven days for audit/debug, not user-facing historical charts.
- Server alerts, inspections and authentication/action audit: 90 days initially.
- Administrator changes and deletion/retention processing are audited; confirm any future legal/site requirements before production.

If queue limits are reached, visibly pause/restrict the demo or discard only authorised low-priority acknowledged/expired telemetry with recorded dropped sequence ranges. Never claim the fleet received “all samples” if some were lost. Alerts remain prioritised and storage-health warnings are visible.

Protect keys through Android Keystore; encrypt sensitive local credentials and data at rest as appropriate. Room alone is not encryption. Define backup/export policy explicitly, with no credentials in logs or Android backups.

### Lifecycle promises

For version-one demos, run the simulator with the operator screen active. Use Android foreground execution only when compliant with the declared service type, target SDK restrictions, notification permissions and battery policy. WorkManager handles opportunistic retry/sync, **not** precise 10.588-second continuous inference.

Do not promise unlimited background execution: newer Android versions can limit foreground data-sync services, and OEM battery policies differ. Test screen-off/background/reboot behaviour on actual pilot devices. If continuous operation cannot be sustained, emit `SUSPENDED`/gap status rather than pretending samples continued.

All inference and feature extraction run off the main/UI thread. Bound queue sizes, avoid bundling huge CSVs, profile CPU/thermal/memory and release mobile runtime resources safely.

## 13. Backend records and APIs

### Main entities

| Entity | Required relationship/state |
|---|---|
| Fleet | Authorisation boundary |
| User/role membership | Role + fleet/truck permissions, activation/revocation |
| Truck | Fleet-owned machine identity |
| Device registration | Bound truck/fleet, unique credential/certificate state |
| Journey/run | Truck/device/source/simulator version, configuration, start/end/reset |
| Telemetry sample | Unique device/run/sequence, observations/context/quality, occurrence/receive times |
| Prediction | Linked sample/window, model/calibration/threshold identity, outputs/status |
| Health assessment | Linked sample, subsystem/aggregate scores, coverage and policy version |
| Alert | Condition state + separate workflow state, evidence links, dedupe fingerprint |
| Inspection | Assigned technician, finding/confirmation/action, verification/revision |
| Audit entry | Actor/device, authorised action, server timestamp, referenced entity |
| Model package | Trusted package identity, compatibility, activation/rollback records |

Every query/transition is fleet-scoped; include authorisation and integrity constraints in the database/API design. Model outputs are immutable evidence records, not a mutable field overwritten by a technician finding.

### API surface (proposal)

| Endpoint | Caller/use |
|---|---|
| `POST /auth/login`, `/auth/refresh`, `/auth/logout` | Auth/session flow |
| `POST /auth/activate` | One-time first-account activation |
| `GET /me`, `/me/trucks` | Server-derived identity/permissions/assignment |
| `POST /admin/users`, `/admin/assignments` | Admin-only account/fleet/truck management |
| `POST /admin/devices/enrolments`, `/devices/enrol` | Admin token creation/device exchange |
| `POST /admin/devices/{id}/revoke` | Credential/device revocation |
| `GET /fleets/{id}/trucks`, `/trucks/{id}/latest` | Authorised master full latest telemetry |
| `GET /alerts`, `/alerts/{id}` | Scoped alert/evidence retrieval |
| `POST /alerts/{id}/acknowledgements` | Authorised acknowledgement, idempotent |
| `POST /alerts/{id}/inspections`, `/inspections/{id}/findings` | Assignment and technician records |
| `POST /inspections/{id}/verification` | Verification/resolution workflow |
| `GET /devices/{id}/model-package` | Authenticated trusted package retrieval, if updates enabled |
| `WS /live` | Backend-authorised fleet/truck updates with reconnect snapshot |

Do not implement remote-injection or simulation-start/stop endpoints. Parameterise identifiers, enforce role/scope on every call, reject mass assignment of roles/truck IDs, rate-limit requests and maintain auditable failures.

## 14. Local deployment and security

- Local server hosts web/API, ingestion, broker and database; use a stable LAN hostname/address and firewall rules limited to the pilot subnet.
- Laptop and Android devices must be on reachable Wi-Fi/LAN. Android emulator uses its emulator-specific host addressing; `localhost` on a phone is the phone, not the laptop.
- Serve web/login/API over HTTPS and MQTT over TLS. Use a controlled development CA on managed demo devices or a trusted certificate setup. Do not disable certificate verification or ship permissive cleartext policies.
- Android network security configuration must trust only the intended development CA in a **debug build**; production configuration must not inherit debug trust.
- Database is not exposed to Android or browser clients; keep credentials only server-side.
- No anonymous MQTT, no globally shared device password, no fleet-wide device wildcard ACLs.
- Enforce WebSocket authentication and message scope on reconnect/subscription. Prevent cross-site/session misuse, and do not place long-lived tokens in query strings or logs.
- Use backups/restores, server health monitoring and safe retention jobs; a failed ingestion worker must not cause outbox deletion.
- Later internet deployment requires hardened hosting, certificate lifecycle, MFA/admin policies, capacity planning, monitoring and a reviewed notification strategy. Do not assume a local demo is already internet-safe.

## 15. Implementation sequence and deliverables

### Milestone 1 — identity and live plumbing

Create local deployment configuration, admin/master web login, Android login/enrolment, per-device ACLs and one-truck telemetry reception. Initially use **explicit mock values with AI unavailable**, not fabricated DL results.

**Acceptance:** master sees only its assigned truck; wrong-fleet access is denied; device spoofing is rejected; disconnect produces stale status.

### Milestone 2 — replay and portable model contract

Extract small raw replay fixtures, export the compact16 mobile model/preprocessing/calibration/threshold assets, implement Kotlin reference/feature state, and validate runtime compatibility and parity.

**Acceptance:** a known Python journey produces equivalent feature vectors and checked logits/probabilities/decisions on Android; truth is absent from the inference path. Failure blocks AI activation, not silently falls back to a different model.

### Milestone 3 — onboard dashboard and health

Show full raw sensor groups, six-tyre TPM at the uniform ideal demo pressure, sample freshness, local inference, independent subsystem/minimum/weighted health, quality warnings and provisional warm-up status.

**Acceptance:** missing oil pressure shows UNKNOWN affected health; changing the one-second redraw rate does not change model inputs or timestamps.

### Milestone 4 — alerts, sync and fleet view

Implement local notifications, durable outbox/backend receipts, complete fleet/truck latest-state view, master acknowledgement and technician inspection workflow.

**Acceptance:** offline injection/replay is detected locally; queued data later appears without duplication; acknowledgement does not change sensor evidence or confirm root cause.

### Milestone 5 — live local simulation and schedules

Port/test the simulator's physical-context channels and event mechanisms, then enable arbitrary scheduled fault controls, overlapping/recurrent events and sensor failures.

**Acceptance:** fault settings alter readings; predicted fault is independently produced from those readings; reset/payload change starts a new state; only slave controls injection.

### Milestone 6 — pilot hardening

Test 5–20 devices, security boundaries, queue/backfill scenarios, mobile runtime performance, background interruptions and model-package rollback. Record measurement budgets and unresolved field limitations.

**Acceptance:** approved demo behaviour is reproducible on actual target devices, with honest offline/unknown states and no deployment-ready safety claim.

### Deliverable structure (proposed, not created)

```text
android-slave/        Kotlin/Compose app and device-local compute modules
web-master/           React/TypeScript fleet/admin/inspection UI
backend/              Auth, fleet APIs, ingestion, live updates and records
deployment/           Local Compose/TLS/broker ACL configuration
model-assets/         Trusted versioned mobile package and schemas
reference-fixtures/   Small Python/Android replay/parity fixtures
docs/                 Setup, user/admin guide, model limits, acceptance evidence
```

Suggested Android modules: `auth`, `device-registration`, `simulation`, `feature-engineering`, `model-runtime`, `sensor-health`, `rca`, `alerts`, `telemetry`, `storage`, `sync`, `ui`. Separate domain/state logic from UI and transport for deterministic testing.

## 16. Required tests and demo acceptance checklist

| Test | Pass criterion |
|---|---|
| Role escalation | Slave cannot become master/admin by editing request/UI |
| Fleet isolation | Cross-fleet REST/WebSocket/MQTT access rejected |
| Device identity | Payload truck spoof rejected; revoked device cannot ingest |
| Login/logout | Session rotation/revocation handled; no credentials logged |
| Mobile conversion | Verified TCN operators, tensor order, calibration, thresholds and numerical tolerances |
| Exact compact16 | No unexpected model columns or reordered inputs; no truth/diagnostic output dependency |
| Reference parity | Raw/context -> residual/scaler/window results match controlled Python fixtures |
| TCN causality | No future rows or cross-episode windows; correct prefix padding |
| Sampling | 171 points/1,800 simulation seconds; 1-second redraw changes no observations |
| Fault injection | Actual sensor trajectories change; replay limits shown; no label shortcut |
| Health | Required channels/policy reproduced, critical subsystem not hidden, missing evidence UNKNOWN |
| TPM ideal baseline | Six distinct tyre channels each transmit 700 kPa; slave/master show matching ideal values and units; simulation source clearly labelled |
| TPM isolation/quality | No suspension-pressure substitution or new DL input; missing/stale TPM remains UNKNOWN/STALE; no fabricated tyre-fault diagnosis |
| Alerts | Debounce/cooldown, direct critical rules, acknowledgement separate from condition/confirmation |
| Offline | Local AI/health works, queue survives interruption, auth expiry visible |
| Delivery | Duplicate/reordered samples deduped; backend receipt required; backfill cannot replace live freshness |
| Storage pressure | No silent critical-event loss; dropped ranges/paused demo visible |
| Model versioning | Schema/cadence/runtime mismatch rejected; last known-good package available |
| Technician workflow | Prediction immutable, inconclusive finding allowed, revision conflict handled |
| Device lifecycle | Screen-off/background/reboot gaps handled honestly; no OS execution promises beyond tests |
| Performance | No UI-thread inference stalls; latency/memory/battery measured on selected hardware |

Proposed pilot targets to **measure**, not claim as passed: under one second per onboard inference at trained cadence; under three seconds from accepted message to visible master update on the healthy LAN; zero dropped critical events within configured storage limits; 20 simultaneous truck connections plus multiple master clients. Include p95 latency and a sustained-run/battery report before accepting these budgets.

### Demonstration script

1. Sign in master and two enrolled Android slaves.
2. Start independent normal replay journeys; master sees distinct truck data and versions.
3. On slave A, select a recorded low-oil fault; observe changed pressure/quantity, AI pattern, independent health and alert.
4. Show injected truth only in a labelled debug view, not as prediction.
5. Master acknowledges; no health score or cause-confirmation state changes automatically.
6. Technician records an inconclusive inspection, then a later verified finding as a separate record.
7. Disconnect slave B, run its simulation/AI locally, then reconnect; master shows offline/backfill/live transitions without duplicate events.
8. Demonstrate missing sensor, unknown health and model warm-up/reset.
9. Verify master has no controls for fault injection and cannot see an unauthorised fleet.

## 17. Existing workspace evidence and remaining build prerequisites

| Current evidence | How it is used |
|---|---|
| `telemetry_model.py` | Legacy route/thermal/context source; preserve old path |
| `enhanced_simulation.py` | Extended local fault mechanisms and schedules to port |
| `enhanced_analytics.py` | Observation-only reference, quality, persistence, features and RCA source |
| `operational_ranges.py` | Supplied operational bands; keep separate from model/health |
| `analysis/export_health_summary.py` | Minimum/weighted subsystem display policy reference |
| `analysis/fault_dl.py` | Native TCN architecture and causal-window implementation |
| `analysis/train_compact16.py` | Exact model-input, preprocessing, calibration and threshold contract |
| `analysis/results_compact16_only/` | Trained research model, schemas, thresholds, metrics and limitations |
| `datasets/dump_truck_enhanced_all_conditions.schema.json` | Original observations/units and allowed training roles |
| `analysis/COMPACT16_TRAINING.md` | Training/inference portability and current caveats |

Missing prerequisites are **implementation work**, not completed capabilities: mobile runtime export/sidecars, parity fixtures, Kotlin feature/reference/simulator port, Android UI/session/outbox logic, broker/auth/API services, web master and technician workflow.

### Proposed defaults still needing sign-off during implementation

- Final app name, branding, languages and target devices.
- Supported pilot Android versions beyond Android 10+, minimum hardware and target SDK/background policy.
- Site-reviewed critical alarm/inspection wording, alert persistence and health-band labels.
- Local certificate/server-address management, credential/session lifetime and offline grant duration.
- Storage/retention, full-payload limits, network/performance budgets and technician assignment rules.
- Selected mobile runtime/export toolchain after compatibility testing.

These defaults can be adjusted without changing the confirmed architecture. Any change to the 16 inputs, sample cadence, reference/scaling/calibration/thresholds or causal model is a **versioned model-contract change requiring new validation**, not just a UI setting.

## 18. Final build principle

**The slave owns sensor simulation, feature computation, local fault inference and sensor health. The server authenticates, durably accepts and scopes data. The web master shows authorised data and manages acknowledgement/inspection. Technician evidence confirms a cause; neither an injected label nor a DL score does.**

Build the replay-to-Android parity flow first, then add the live simulator. Keep unknown, stale, offline and not-validated states visible throughout.
