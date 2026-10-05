# Dump-truck digital twin — complete decision flow

The diagrams below describe the complete working of the Streamlit simulator:
input controls, journey-level decisions, per-sample telemetry generation,
fault injection, operational-band decisions, engine-health decisions, 3D
mapping, and explicit logging/export.

> The research cycle is a deterministic C-WTVC proxy calibrated from the
> supplied Fuel 328 (2022) paper and figures. It is not the authors' raw data.

## 1. Whole application flow

```mermaid
flowchart TD
    A([Start Streamlit app]) --> B[Load model, operational ranges,
    CSS and Three.js/Canvas assets]
    B --> C[Read sidebar controls]

    C --> C1[Scenario:
    ideal / stress / overhaul]
    C --> C2[Payload class:
    empty / light / medium / heavy / overload]
    C --> C3[Drive cycle:
    compact route / research C-WTVC]
    C --> C4[Start condition:
    cold / warm]
    C --> C5[Fault toggles:
    overheat, bearing, oil, brakes, perception]
    C --> C6[Seed, sample interval,
    replay setting]

    C2 --> D{Lock payload class
    for the journey}
    D --> E[Resolve rated capacity,
    payload mass and fixed load category]
    C3 --> F{Select route profile}
    F -->|Research C-WTVC| F1[1800 s proxy:
    urban 0–900 s
    highway 900–1368 s
    expressway 1368–1800 s]
    F -->|Compact route| F2[Short deterministic
    operating-mode profile]
    C4 --> G[Initialise thermal state:
    cold lag or warm operating temperature]

    E --> H[Create deterministic journey configuration]
    F1 --> H
    F2 --> H
    G --> H
    C1 --> H
    C6 --> H
    H --> I[Generate one telemetry frame
    per sample]
    I --> J[Apply selected manual faults
    to every generated frame]
    J --> K[Recalculate engine health,
    RUL and recommended action]
    K --> L[Classify all readings against
    Green / Amber / Red bands]
    L --> M[Build dashboard payload]

    M --> N[Update Streamlit metrics,
    warnings and sensor cards]
    M --> O[Replay frame in 3D twin]
    O --> O1[Payload → suspension sag]
    O --> O2[Coolant temperature → thermal colour]
    O --> O3[Vibration → high-frequency shake]
    O --> O4[Hoist angle → dump-body rotation]
    M --> P[Draw health / coolant /
    stress trend chart]

    M --> Q{User presses
    Record journey locally?}
    Q -->|No| R[Keep data in memory;
    no filesystem side effect]
    Q -->|Yes| S[Create stable hash journey ID]
    S --> T{Same journey already
    recorded?}
    T -->|Yes| U[Report already recorded;
    do not duplicate]
    T -->|No| V[Write atomic JSONL + CSV
    to telemetry_logs/]
    M --> W[Optional download:
    JSONL or CSV]
    N --> X([User changes a control])
    X --> C
```

## 2. Journey-level decisions

```mermaid
flowchart LR
    A[Payload selector] --> B{Payload category}
    B -->|Empty| B1[0% capacity]
    B -->|Light| B2[48% capacity]
    B -->|Medium| B3[72% capacity]
    B -->|Heavy| B4[94% capacity]
    B -->|Overload| B5[108% capacity]
    B1 --> C[Fixed for all frames]
    B2 --> C
    B3 --> C
    B4 --> C
    B5 --> C
    C --> D{Utilisation > 85%?}
    D -->|No| E[Normal payload stress]
    D -->|Yes| F[Raise thermal, vibration,
    hydraulic and structural stress]

    G[Scenario selector] --> H{Scenario}
    H -->|Ideal| H1[Healthy baseline]
    H -->|Stress| H2[Heavy payload and rising stress]
    H -->|Overhaul| H3[Service transient fades over journey]

    I[Drive cycle selector] --> J{Cycle}
    J -->|Compact| J1[Existing short route modes]
    J -->|C-WTVC proxy| J2[Time-based phase boundaries]

    K[Start condition] --> L{Temperature state}
    L -->|Cold| L1[Low initial coolant;
    slow coolant-flow response;
    warm-up transient]
    L -->|Warm| L2[Normal initial coolant;
    faster flow response]
```

## 3. Per-sample telemetry decision flow

```mermaid
flowchart TD
    A[Next sample: step, timestamp,
    elapsed time] --> B{Research C-WTVC?}
    B -->|Yes| C{Elapsed time}
    C -->|0–899 s| C1[Urban phase:
    stop/go speed pulses]
    C -->|900–1367 s| C2[Highway phase:
    moderate transients]
    C -->|1368–1800 s| C3[Expressway phase:
    ramp then stable speed]
    B -->|No| D[Compact route mode profile]

    C1 --> E[Vehicle speed, acceleration,
    grade and transient index]
    C2 --> E
    C3 --> E
    D --> E

    E --> F[Calculate engine load,
    RPM, torque and effective power]
    F --> G[Calculate fuel mass flow,
    fuel volume and fuel energy]
    G --> H[Allocate energy:
    coolant, exhaust, mechanical,
    remainder]
    H --> I[Update coolant thermal state]
    I --> J{Coolant temperature
    crosses thermostat region?}
    J -->|No| K[Thermostat closed/part-open;
    low or lagged coolant flow]
    J -->|Yes| L[Thermostat opens with hysteresis;
    coolant flow increases]
    K --> M[Coolant inlet/outlet,
    ΔT and coolant energy flow]
    L --> M

    F --> N[Generate engine, transmission,
    brake and fuel readings]
    E --> O[Generate IMU:
    acceleration, gyro, roll, pitch, yaw]
    E --> P[Generate GNSS/GPS:
    position, speed, heading]
    E --> Q[Generate LiDAR, radar
    and DSS perception readings]
    E --> R[Generate suspension,
    hoist and frame stress readings]
    M --> S[Assemble complete frame]
    N --> S
    O --> S
    P --> S
    Q --> S
    R --> S
```

## 4. Fault injection flow

```mermaid
flowchart TD
    A[Base telemetry frame] --> B{Manual fault selected?}
    B -->|No| C[Keep base readings]
    B -->|Yes| D{Fault type}

    D -->|Overheat| D1[Increase coolant/oil/exhaust
    temperatures; set engine-overheat flag]
    D -->|Bearing failure| D2[Increase engine block and gearbox
    vibration; set IMU fault flag]
    D -->|Low oil pressure| D3[Reduce oil pressure and oil level;
    set low-oil flag]
    D -->|Brake wear| D4[Increase pad wear, stroke,
    brake temperature; set brake flag]
    D -->|Perception degradation| D5[Degrade GNSS, obstruct LiDAR,
    fault radar and set perception flags]

    D1 --> E[Recalculate health]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    C --> E
    E --> F[Store manual-fault metadata
    separately from automatic flags]
```

## 5. Operational-range and health decisions

```mermaid
flowchart TD
    A[Frame readings] --> B[Operational range classifier]
    B --> B1[Engine:
    block temp, oil temp, exhaust temp,
    oil pressure, block vibration]
    B --> B2[Transmission:
    gearbox oil temp, oil pressure,
    gearbox vibration]
    B --> B3[Hydraulic:
    oil condition, actuator vibration]
    B --> B4[Coolant:
    pump temperature, coolant condition]

    B1 --> C{Within reference band?}
    B2 --> C
    B3 --> C
    B4 --> C
    C -->|Green band| C1[Workable]
    C -->|Grey / Amber band| C2[Caution]
    C -->|Red band| C3[Alert]

    A --> D[Engine health calculation]
    D --> D1[Scenario penalty]
    D --> D2[Load and payload penalty]
    D --> D3[Temperature, oil and vibration penalty]
    D --> D4[Automatic and manual fault penalty]
    D1 --> E[Health score 0–100]
    D2 --> E
    D3 --> E
    D4 --> E
    E --> F{Score band}
    F -->|≥90| F1[Excellent / Green]
    F -->|75–89| F2[Good / Green]
    F -->|60–74| F3[Fair / Amber]
    F -->|40–59| F4[Poor / Amber]
    F -->|<40| F5[Critical / Red]
    F1 --> G[Estimate RUL range + action]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G
```

## 6. Output and persistence flow

```mermaid
flowchart LR
    A[Complete frame list] --> B[JSON-safe nested records]
    A --> C[Compact flattened CSV rows]
    B --> D[JSONL download]
    C --> E[CSV download]
    A --> F{Record button pressed?}
    F -->|No| G[No file write]
    F -->|Yes| H[Hash configuration:
    seed, scenario, payload,
    cycle, start, faults]
    H --> I[journey-<hash>.jsonl
    journey-<hash>.csv]
    I --> J{Files exist?}
    J -->|Yes| K[Idempotent: skip duplicate]
    J -->|No| L[Atomic write into telemetry_logs/]
```

## Operational-range reference used by the model

| System | Green / workable | Grey / amber / caution | Red / alert |
|---|---|---|---|
| Engine block temperature | 82–101 °C | 101–108 °C; values 78–82 °C are uncovered and treated cautiously | >108 °C or <78 °C |
| Engine oil temperature | 86–100 °C | 100–106 °C | >106 °C |
| Exhaust temperature | 340–555 °C | 555–615 °C | >615 °C |
| Engine oil pressure | 459–512 kPa | 440–459 or 512–525 kPa | <440 or >525 kPa |
| Engine block vibration | 2.0–6.3 mm/s | 6.3–7.5 mm/s | >7.5 mm/s |
| Gearbox oil temperature | 67–89 °C | 89–96 °C | >96 °C |
| Transmission oil pressure | 329–372 kPa | 310–329 or 372–390 kPa | <310 or >390 kPa |
| Gearbox vibration | 1.4–5.0 mm/s | 5.0–6.0 mm/s | >6.0 mm/s |
| Hydraulic oil condition | ≥90 | 80–90 | <80 |
| Hydraulic actuator vibration | 1.6–5.3 mm/s | 5.3–6.3 mm/s | >6.3 mm/s |
| Coolant pump temperature | 78–92 °C | 92–98 °C | >98 °C |
| Coolant condition | ≥90 | 80–90 | <80 |

The classifier is conservative: a value in an uncovered gap is reported as
amber rather than incorrectly reported as healthy.
