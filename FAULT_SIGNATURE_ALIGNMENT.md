# Fault-signature matrix alignment review

> **Addendum — enhanced path:** this historical audit below describes the legacy
> and Engine Evidence Lab paths. The new `enhanced_simulation.py` and
> `enhanced_analytics.py` add condition channels, temporal matrix mechanisms,
> rate/persistence features and observation-only probable hypotheses. Use
> `ENHANCED_SIMULATION.md` for the enhanced interface and limitations. The old
> paths and old datasets retain the gaps documented below. New simulated
> mechanisms do not establish field-validated root causes or RUL accuracy.

Reviewed source: `fault_signature_matrix.docx` supplied with the project.

This review distinguishes two simulator paths:

1. **Legacy dump-truck path** — `telemetry_model.py`, `app_3d.py`,
   `cabin_dashboard.py`, and `generate_dataset.py`.
2. **CPAI-style evidence lab** — `engine_lab/simulation.py`,
   `engine_lab/analytics.py`, `engine_lab/workflow.py`, and
   `engine_health_lab.py`.

## Executive result

The project now has a meaningful CPAI-style foundation in the **Engine Evidence
Lab**, but it does **not** yet implement the supplied matrix as an exact root-
cause lookup system.

- The evidence lab supports state-conditioned healthy references, residuals,
  anomaly persistence, transient handling, sensor-fault separation, probable
  hypotheses, alternatives and technician confirmation.
- It deliberately does not read hidden injected truth while diagnosing and
  always reports root cause as unconfirmed until human evidence is recorded.
- The legacy dump-truck simulator has the required operational readings and
  manual fault toggles, but its current fault injection is not matrix-based.
- The matrix's engine-condition and gearbox-condition indexes are not fully
  represented. Engine oil level is **not** equivalent to engine oil condition.
- No gearbox seal/bearing signature or hydraulic wear/contamination diagnosis
  exists in the evidence lab.

## Matrix-by-matrix status

| Matrix signature | Legacy dump-truck path | CPAI evidence lab | Verdict |
|---|---|---|---|
| Engine oil pump wear / sudden lubrication loss: low oil pressure + high engine-block vibration + normal oil condition | Manual `low_oil` lowers oil pressure; manual `bearing_failure` raises block vibration; no engine oil-condition signal | State-conditioned oil-pressure residuals + vibration diagnosis; no oil-condition signal | **Partial** |
| Piston / piston-ring damage: vibration + high oil temperature + low oil condition + erratic exhaust | No piston/ring schedule or oil-condition signal | No piston/ring fault kind; EGT imbalance is supported, not piston diagnosis | **Missing** |
| Turbo/exhaust-side fault: isolated abnormal exhaust while block/oil temperatures remain normal | No turbo/exhaust-only fault schedule | No turbo fault kind; combustion imbalance changes EGT and can add vibration | **Missing** |
| Cooling-system failure: high block, oil, exhaust and coolant-pump temperatures together | `overheat` raises coolant, oil, exhaust and coolant-pump temperatures | `cooling_restriction` and thermostat schedules alter coolant/flow/thermostat and can produce EGT effects; diagnosis is broader | **Partial / closest match** |
| Coolant degradation: high coolant-pump temperature + low coolant condition with block near normal | Coolant condition exists, but no coolant-degradation injection or paired signature rule | No coolant-condition observation in evidence-lab signal schema | **Missing** |
| Gearbox seal / sudden transmission-pressure loss: low transmission pressure + high gearbox vibration + normal gearbox condition | Transmission pressure/vibration fields exist, but no matching fault injection or diagnosis | Gearbox subsystem is not in evidence-lab signals | **Missing** |
| Gearbox bearing/gear wear: low pressure + high gearbox vibration + low gearbox condition | Gearbox condition and rate are absent | Not represented | **Missing** |
| Hydraulic pump/actuator aging: slow hydraulic-condition decrease + actuator vibration increase | Hydraulic condition and actuator vibration are generated, but no scheduled hydraulic wear fault or diagnostic rule | Hydraulic signals are not in evidence-lab schema | **Missing** |
| Hydraulic contamination/filter/seal breach: fast condition decrease + actuator vibration | No rate-aware hydraulic fault schedule or classifier | Not represented | **Missing** |
| Isolated/single-sensor deviation | Operational bands classify readings but do not distinguish sensor noise from physical fault | Pressure-only deviations remain unresolved and alternatives include sensor bias; measurement-quality rules exist | **Partial / supported in evidence lab** |
| Multiple unrelated deviations | Health/fault flags are aggregated, not root-cause classified | Generic persistent-deviation hypothesis and manual review path exist | **Partial / conservative** |

## What was actually verified

### Legacy dump-truck simulator

Using a 171-sample C-WTVC journey, the following observations were verified:

| Test | Observed result |
|---|---|
| No manual faults | Baseline has no manually injected fault flags, although automatic DSS/operational flags may occur during the route |
| Low-oil injection | Engine oil pressure becomes red for all 171 frames; oil level also decreases; engine-block vibration does not increase |
| Bearing injection | Engine-block vibration becomes red for all 171 frames; gearbox vibration increases; engine oil pressure remains normal |
| Low-oil + bearing injection | Both low oil pressure and high block vibration are present together |
| Overheat injection | Engine block, engine oil and coolant-pump temperatures become red; exhaust temperature is elevated but not red for every frame |
| Brake injection | Brake readings/fault flag change; it does not participate in the supplied engine/gearbox/hydraulic signature matrix |
| Perception injection | GNSS/LiDAR/radar faults change; these are outside the supplied signature matrix |

The legacy path therefore can **produce some ingredients** for signatures, but
it does not decide “engine oil pump wear”, “bearing wear”, or “coolant-system
failure” from the combinations.

### CPAI-style evidence lab

The evidence lab was exercised with each supported schedule in
`engine_lab.simulation.FAULT_KINDS`:

```text
lubrication_degradation
oil_starvation
bearing_degradation
cooling_restriction
thermostat_malfunction
combustion_imbalance
sensor_fault
incomplete_overhaul
```

Observed capabilities:

- Scheduled onset, active period, progression, recurrence and recovery work.
- Fault truth is kept separate from observations and predictions.
- Sensor faults do not create physical wear.
- Healthy cold/warm operation does not create false anomalies in the tested
  scenarios.
- State-conditioned residuals are calculated before anomaly decisions.
- Persistence gates distinguish transient from persistent deviations.
- Pressure-only deviation remains ambiguous rather than being labelled a root
  cause.
- Diagnoses include evidence, alternatives and required confirmation checks.
- Root cause remains `Not confirmed — technician evidence required`.
- RUL is gated and synthetic-only; it is not inferred from a health score.

The project test suite passed:

```text
61 passed, 1 skipped
```

## Important implementation mismatch

The supplied matrix requires condition indexes and rates:

```text
Engine Oil Condition (Co)
Gearbox Oil Condition (Cg)
Hydraulic Oil Condition (Ch)
Coolant Condition (Cc)
d(condition)/dt
```

The legacy path currently has:

```text
engine.oil_level_pct                 # quantity, not oil condition
suspension.hydraulic_oil_condition_pct
coolant_condition_pct
```

It does not have engine-oil-condition or gearbox-oil-condition observations.
The evidence lab has neither condition indexes nor gearbox/hydraulic
observations in its inference schema.

## Required additions for exact matrix compliance

1. Add observation channels `engine_oil_condition_pct` and
   `gearbox_oil_condition_pct`; retain oil level separately.
2. Add scheduled fault kinds for gearbox seal loss, gearbox bearing/gear wear,
   hydraulic actuator aging and hydraulic contamination.
3. Add fault schedules for piston/ring and turbo/exhaust-side faults.
4. Add condition-index trajectories and explicit rate features, for example:
   `oil_condition_rate_pct_per_h` and `hydraulic_condition_rate_pct_per_h`.
5. Implement a separate signature evaluator that requires **all** specified
   observations, checks persistence and rate, and returns:

   ```text
   anomaly detected
   probable fault hypothesis
   alternative explanations
   evidence
   confidence/evidence strength
   technician confirmation required
   ```

6. Keep diagnosis observation-only. Never read `truth`, schedules, wear or
   manual fault labels inside the evaluator.
7. Add synthetic sensor-fault variants for every new condition channel so the
   system can distinguish sensor failure from mechanical failure.
8. Expand tests for exact combinations, missing channels, isolated deviations,
   persistence thresholds and fast-versus-slow condition decline.

## Conclusion

The **Engine Evidence Lab** follows the matrix's most important CPAI principle:
do not claim root cause from a single reading or from hidden simulator truth.
It is aligned with the matrix at the level of state conditioning, residuals,
anomalies, persistence, alternatives and technician confirmation.

The project is **not yet aligned at the exact Cat 793F subsystem-signature
level**, primarily because piston/turbo/gearbox/hydraulic signatures and the
engine/gearbox condition indexes are not implemented. The legacy dashboard
should not be described as implementing the fault signature matrix; the
evidence lab should be the basis for completing that work.
