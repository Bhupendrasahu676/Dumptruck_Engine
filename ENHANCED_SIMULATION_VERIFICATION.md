# Enhanced simulator verification — 2026-10-05

## Executed checks

- `python3 -m pytest -q`: **239 passed, 4 skipped** (137.71 seconds).
  Three skips require Streamlit in that interpreter; one is an existing project
  optional check. These are not reported as passes.
- `/tmp/opencode/enhanced-ui-test/bin/python -m pytest -q test_enhanced_dashboard.py`:
  **8 passed**, including real simulator/analytics integration.
- `python3 -m py_compile enhanced_simulation.py enhanced_analytics.py vibration_analysis.py generate_enhanced_dataset.py enhanced_dashboard.py pages/2_Enhanced_Dumper_Simulator.py`:
  passed.
- Actual Streamlit AppTest flow: configured a 600-second episode, scheduled
  oil-pump wear at 100 seconds, replayed sample 60 (300 seconds), observed the
  matching observation-only hypothesis, generated a waveform and FFT table,
  and verified four separate download controls. No app exception occurred.
- `curl --fail --silent http://localhost:8504/_stcore/health`: `ok`.
- Original dataset SHA-256 still matches its existing no-perception manifest:
  **legacy CSV unchanged**.

The Streamlit test environment emitted non-fatal `use_container_width`
deprecation warnings; no global environment or requirements change was made.

## Generated dataset audit

Generated and regenerated with final code:

```bash
python3 generate_enhanced_dataset.py --output-dir datasets/enhanced_v1 \
  --engines 12 --episodes-per-fault 2 --duration-s 1800 --timestep-s 10 \
  --waveforms --waveform-every 180
```

The final refresh used `--overwrite` for these newly generated artifacts only.
The closing streaming audit verified:

- **12 virtual trucks, 492 episodes, 89,052 rows** in each full-rate table.
- **369 engineered inputs**; all target classes have positive labels.
- **981 high-rate snippets and 981 FFT feature rows**; three snippets omitted
  because the source RPM/vibration measurement was missing.
- Matched join keys across readings/features/labels/diagnostics; finite or
  explicitly missing numeric features; exact binary-label/effect-level rules.
- Disjoint engines: 10 training, one validation, one test (74,210 / 7,421 /
  7,421 rows). This is a small synthetic pilot, not sufficient field validation.
- **8,930 fault/sample pairs** retain delayed or damage labels after the
  corresponding active schedule level becomes zero.
- Source hashes match current simulation, analytics and exporter files.
- Every root cause remains unconfirmed; no numerical RUL is emitted.

The verbose audit tables are large: readings 105.2 MiB, features 294.2 MiB,
labels 407.3 MiB and diagnostics 1,461.2 MiB. Load only manifest-approved
feature/target columns for training; the diagnostics are not required by DL.

## Behaviours protected

Tests cover all 16 simulation classes, deterministic matched trajectories,
condition bounds, onset/progression/recovery/recurrence, overlapping events,
causal lags, permanent wear and explicit lifecycle continuation. Distinct pump
and seal mechanisms preserve oil condition; gearbox damage reduces pressure;
coolant degradation preserves block/oil temperatures relative to healthy context.

Analytics tests mutate hidden truth, schedules, expected readings and injected
quality labels and verify that inference/features stay unchanged. Temporal
windows reset per episode. Missing observations mask inputs and prevent a
fabricated healthy score. Twelve developed mechanisms are checked for their
matching hypotheses. Sparse 20-hour hydraulic aging is checked separately to
avoid assigning a fast-contamination cause solely because the short window
contains only one reading.

Exporter tests cover engine-disjoint splits, joined table keys, repeatability,
overwrite safety, source/input separation and optional waveform/FFT output.

## Interpretation limits

These are software/synthetic-behaviour checks, not diagnostic accuracy estimates
on real trucks. A 24-episode healthy-context stress audit produced zero named
mechanical-fault hypotheses, but 56 isolated/sensor-ambiguity and 368 unmatched
multi-deviation hypotheses over 1,464 samples. Approximate priors therefore
remain unsuitable for claims of a low overall false-alarm rate.

Do not train on `diagnostics.csv`. Use injected mechanism/effect targets from
`labels.csv` and separately assess whether weak/incipient effects are observable.
No DL model was trained, no confirmed maintenance history was fabricated, and
RUL remains unavailable. See `ENHANCED_SIMULATION.md` for the exact schemas and
synthetic physics/spectral assumptions.
