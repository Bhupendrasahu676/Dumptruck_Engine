# Enhanced dataset: essential-point verification

## Verdict

**Pass for the synthetic data contract; partial for full fault-signature/RCA
alignment.** This dataset is suitable for controlled fault-classification
development and feature selection. It is not evidence of reliable physical
root-cause diagnosis, real Cat 793F calibration or prognostic accuracy.

The full-file auditor returned **29 passed checks, one partial check and one
explicitly not-evaluated validation category, with no failed checks**. The
partial check is matrix-signature coverage. See `data_audit.json` for complete
evidence rather than relying on this summary.

## Generated artifacts

- `datasets/dump_truck_enhanced_all_conditions.csv` — **164,160 rows, 446 columns,
  approximately 794.92 MiB**.
- `datasets/dump_truck_enhanced_all_conditions.schema.json` — approved inputs,
  targets, exclusions, units, feature definitions and missingness policy.
- `datasets/dump_truck_enhanced_all_conditions.manifest.json` — per-episode
  seeds/configuration/schedules, coverage, partitions and integrity hashes.
- `datasets/README_ENHANCED_CSV.md` — loading/reproduction instructions.

CSV SHA-256:

```text
2a4790199945f331987897a86f9329bacb03f9b5ac9d0cb893ae2b7d52adf448
```

The original legacy dataset remains unchanged. The fresh file has the same
row/journey sizes as before, but not the same fault-combination semantics.

## Essential checks

| Requirement | Result | Actual evidence |
|---|---|---|
| Fresh sensor-based inputs | Pass | All 113 original numeric channels are present, within 369 approved raw/engineered inputs. |
| Fault labels separate from X | Pass | Sixteen multi-label targets; direct labels, truth, IDs, statuses, health and cause/action outputs excluded by schema. |
| One CSV with reproducible schema | Pass | 446 unique headers; correct widths, declared roles, units and definitions. |
| Same journey sampling as before | Pass | 960 journeys, each 171 samples over 1,800 seconds; approximately 10.588-second interval. |
| C-WTVC phase boundaries | Pass | Urban 81,600 rows, highway 43,200, expressway 39,360; zero phase-boundary violations. |
| Fixed categorical payload | Pass | Category and mass remain fixed inside each journey. |
| Diverse context per primary class | Pass | Every class plus normal has all 30 scenario/payload/start Cartesian combinations. |
| Independent virtual machines | Pass | Twenty stable virtual truck variants, independent episode seeds, noise/ambient/wear variation. Not twenty physical trucks. |
| Independent partitions | Pass | Fourteen train trucks, three validation, three test; no truck crosses partitions. |
| All classes in every partition | Pass | Each split has positive examples of all 16 classes and normal readings. |
| Multiple concurrent faults | Pass | All 120 distinct pairs and 60 selected triples actually occur. Not every Boolean combination is generated. |
| Temporal fault behaviour | Pass | Positive episodes vary in labels; onset/progression/recovery/recurrence and delayed/persistent effects are represented. |
| Sensor-fault diversity | Pass | Biased, drifting, frozen, missing and noisy observations are present, with truth quality kept out of X. |
| Condition-index bounds | Pass | Engine/gearbox/hydraulic/coolant condition readings stay within 0–100. |
| Numeric validity and missingness | Pass | Finite numeric inputs or explicitly documented blanks; 113 binary validity masks checked. |
| Label semantics | Pass | All 16 binary targets equal `effect_level > 0.001`; schedule and effect levels remain separate. |
| Uniqueness and order | Pass | Unique episode/step keys, increasing elapsed/timestamp coordinates, correct cadence. |
| Traceability | Pass | CSV digest and five generator/model source hashes match the manifest. |
| Complete matrix signatures | Partial | Weak/developing, residual, mixed and aged-baseline states do not always satisfy all developed-signature predicates. |
| Full long-term aging | Partial | Thirty-minute journeys contain early/accelerated wear, not weeks of observed degradation. |
| Actual high-frequency evidence in this CSV | Not provided | Scalar CSV is not a vibration waveform. Do not infer bearing frequencies from its sampling. |
| All truck operating modes | Limited | Batch has hauling, idle and downslope; the separate compact-route simulator supports other modes. |
| Real root causes/repair outcomes | Not validated | Cause/mitigation descriptions are synthetic catalogue lookups, not confirmed maintenance records. |
| Trained DL performance | Not evaluated | No DL model trained; matrix match rates are not classifier accuracy. |
| Physical energy balance and Cat calibration | Not validated | Existing energy channels are retained; empirical fault overlays do not recompute a calibrated energy-conserving engine. |
| Physical RUL | Not validated | No independent failure/censoring histories; numerical RUL is not fabricated. |

Partition rows: **114,912 train / 24,624 validation / 24,624 test**.
There are **50,353 normal-labelled readings**. The largest/smallest positive
class frequency ratio is approximately **2.55**, so use class-sensitive metrics
and thresholds rather than assuming exact row balance.

There are **512,493 intentionally blank input cells**, mostly onset associations,
first-window history and sensor/history unavailability. These are not NaN/Inf
corruption. Fit imputation/scaling on training trucks only.

## What the matrix check actually found

All positive readings include early, recovering, residual-damage and mixed-fault
states. It is incorrect to require every such row to look like a fully developed
single fault. A separate scan therefore examined primary single-fault readings
with current schedule strength at least 0.5:

| Signature | Matching screen rows / evaluated rows | Screening match rate |
|---|---:|---:|
| Oil-pump wear | 2,464 / 2,497 | 98.68% |
| Piston/ring damage | 2,421 / 2,564 | 94.42% |
| Turbo/exhaust fault | 2,547 / 2,571 | 99.07% |
| Coolant-system failure | 1,790 / 2,658 | 67.34% |
| Coolant degradation | 1,918 / 2,341 | 81.93% |
| Gearbox seal wear | 1,948 / 2,694 | 72.31% |
| Gearbox bearing wear | 2,233 / 2,596 | 86.02% |

These are **transparent threshold-screen match rates, not DL accuracy**, and
strength >=0.5 does not guarantee that every lagged channel has fully developed.
The screen uses explicit context-adjusted thermal thresholds and declared
pressure/vibration/condition screening cuts, informed by the supplied bands.
These are not an exact OEM-validated matrix evaluator (for example, its gearbox
pressure/vibration screening cuts are <320 kPa and >6.3 mm/s). See
`phase_specific_evidence.json` for the interpretation and `data_audit.json` for
the threshold ranges.

The automatic signature-rate audit covers seven matrix mechanisms plus two
hydraulic trend classes. The other seven classes have structural/label coverage
and behavioural regression checks, but no matrix match rate was evaluated by
that auditor. Do not interpret their `not_evaluated` entries as either passes
or measured detection failures.

In separate fixed-severity-one, approximately 1,000-second controlled probes,
all seven mechanisms matched their screens across all 30 contexts. This verifies
the simulator can produce the developed signatures; it does not establish that
every weak/mixed sample is distinguishable or that real machines behave this way.

### Hydraulic rate caveat

All **40 primary contamination episodes** showed some fast condition decline
and some actuator vibration above 6.3 mm/s. In contrast, only **14/40 primary
normal-aging episodes** showed any vibration above that boundary. Short-window
noise can mimic fast decline even in aging examples (3/40 episodes).

The full auditor also fits a trend across positive portions of each episode;
plateaus/recovery can make a fast onset appear slow in that aggregate metric.
Use decline-phase/window evidence, not the whole-episode slope alone. Do not
interpret low aggregate hydraulic match rates as measured classifier errors.
For robust aging-versus-contamination training, add explicit longer histories,
time-scale-aware rate definitions and independent early/late degradation tests.

## Software and independent checks run

- `python3 -m pytest -q`: **318 passed, 4 skipped** (164.00 seconds).
  Three skips need Streamlit in that interpreter; one is an existing optional
  project check. Skips are not passes.
- `/tmp/opencode/enhanced-ui-test/bin/python -m pytest -q test_enhanced_dashboard.py`:
  **8 passed**.
- `python3 -m pytest -q test_training_csv_audit.py`: **31 passed**, including a
  regression correcting an overly strict pump threshold in the auditor.
- Python compilation of the new generator and auditor passed.
- **26 matched-trajectory contract probes** passed: pre-onset equality,
  localized condition/thermal effects, pressure-first cascade, permanent damage,
  missing-sensor handling and no fabricated root confirmation/RUL.
- Oil-pump first deviations: pressure **105 s**, oil temperature **130 s**,
  engine-block vibration **160 s** in the controlled test.

Evidence files: `data_audit.md`, `data_audit.json`, `mechanism_probes.json`,
`controlled_matrix_probes.json`, `phase_specific_evidence.json`.

## How to use this batch safely

1. Use only `.schema.json` `input_columns` for X and `target_columns` for y.
2. Preserve the supplied truck-level train/validation/test partitions.
3. Fit correlations, selection, imputation and scaling only on training data.
4. Form sequence windows within a journey, never across episode boundaries.
5. Evaluate each class by severity/phase and single/mixed/sensor-quality condition.
6. Treat fuel-level/accumulated-energy signals as potential journey-time proxies;
   test stability on independent onset schedules and seeds.
7. Do not train on `probable_faults_json` or catalogue-derived causes/mitigations.
8. Deploy classifications as probable patterns with evidence/alternatives, not
   confirmed root causes or field-validated RUL.

**Bottom line:** the new CSV meets the core synthetic training-data requirements
and is substantially broader than the old five-toggle dataset. Full RCA claims
remain conditional, particularly for long-term aging and ambiguous mechanisms.
