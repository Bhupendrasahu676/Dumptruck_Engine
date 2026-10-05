"""Deterministic, explicitly synthetic extension of the legacy truck journey.

This is a dependency-light demonstration, not a validated physics, diagnostic,
energy-conservation or remaining-life model. ``raw_legacy`` preserves numeric
legacy measurements verbatim. ``expected`` is the same-context, same-engine
fault-free reference; injected labels live only in ``truth``. Baseline legacy
noise is preserved; ``noise_scale`` controls additional bounded sensor noise.

Fault progression is a fraction of the active duration. Recurrence is an
onset-to-onset interval and must include the recovery period. Pressure/flow
effects precede thermal and vibration effects by 0/25/55 s respectively.
Accumulated wear uses operating seconds / 3600 * ``wear_time_scale``; a value
above one explicitly accelerates aging. Condition percentages are synthetic
condition indices, not laboratory oil or coolant measurements.
Explicit ``initial_fault_wear`` and ``initial_operating_hours`` carry mechanical
history between episodes; no process-global state is retained. References
exclude this injected damage, while generic ``initial_wear`` stays a known
baseline characteristic. Schedule levels and physical-effect levels are
separate labels, so a recovered schedule does not erase remaining damage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import math
import random

from telemetry_model import SimulationConfig, simulate_journey


FAULT_CATALOG = {
    "engine_oil_pump_wear": {"subsystem": "engine", "possible_cause": "Oil pump wear or restricted pickup reduces lubrication pressure.", "mitigation": "Check oil pressure independently, pickup and pump; service lubrication system."},
    "piston_ring_damage": {"subsystem": "engine", "possible_cause": "Ring wear or damage causes uneven combustion and oil consumption.", "mitigation": "Inspect compression, blow-by, cylinder balance and oil consumption."},
    "turbo_exhaust_fault": {"subsystem": "engine", "possible_cause": "Turbocharger or exhaust restriction changes boost and exhaust temperature.", "mitigation": "Inspect turbocharger, boost plumbing and exhaust restriction."},
    "coolant_system_failure": {"subsystem": "cooling", "possible_cause": "Coolant flow loss from pump, thermostat or circulation failure.", "mitigation": "Stop safely if overheating; inspect coolant flow, pump and thermostat."},
    "coolant_degradation": {"subsystem": "cooling", "possible_cause": "Coolant contamination or aging reduces the synthetic condition index.", "mitigation": "Sample coolant, inspect leaks and service the cooling circuit."},
    "gearbox_seal_wear": {"subsystem": "transmission", "possible_cause": "Seal leakage lowers transmission oil pressure.", "mitigation": "Check leaks, oil level, seals and transmission pressure."},
    "gearbox_bearing_wear": {"subsystem": "transmission", "possible_cause": "Bearing wear increases gearbox vibration and heat.", "mitigation": "Inspect gearbox bearings and lubricant; acquire real vibration data."},
    "hydraulic_normal_aging": {"subsystem": "hydraulics", "possible_cause": "Gradual hydraulic lubricant aging during use.", "mitigation": "Trend condition and perform scheduled lubricant sampling/service."},
    "hydraulic_contamination": {"subsystem": "hydraulics", "possible_cause": "Hydraulic contamination degrades condition faster than normal aging.", "mitigation": "Sample hydraulic fluid, inspect filtration and contamination entry paths."},
    "bearing_failure": {"subsystem": "engine", "possible_cause": "Engine bearing damage increases vibration and frictional heat.", "mitigation": "Stop safely for severe vibration; inspect bearings and lubrication."},
    "low_oil": {"subsystem": "engine", "possible_cause": "Low oil quantity reduces lubrication pressure.", "mitigation": "Check oil level, leakage and pressure before continued operation."},
    "overheat": {"subsystem": "engine", "possible_cause": "Abnormal heat accumulation beyond same-context reference.", "mitigation": "Reduce demand or stop safely; inspect heat rejection and lubrication."},
    "brake_wear": {"subsystem": "brakes", "possible_cause": "Pad wear increases actuator stroke and braking heat.", "mitigation": "Inspect pad thickness, stroke and brake cooling."},
    "perception_degraded": {"subsystem": "perception", "possible_cause": "Obstruction or environmental interference degrades perception.", "mitigation": "Clean and inspect sensors and confirm operation independently."},
    "sensor_fault": {"subsystem": "sensors", "possible_cause": "Missing, biased, frozen, drifting or noisy sensor measurement.", "mitigation": "Verify with an independent measurement and inspect sensor/wiring."},
    "unrelated_multi_blip": {"subsystem": "multiple", "possible_cause": "Unrelated short disturbances without a shared mechanical cause.", "mitigation": "Trend each signal independently and check operating context."},
}
FAULT_KINDS = tuple(FAULT_CATALOG)
SENSOR_MODES = ("missing", "biased", "frozen", "drifting", "noisy")
_MAX_SAMPLES = 100_000
_MAX_WAVEFORM_SAMPLES = 262_144


def _number(name, value, low=None, high=None, *, positive=False):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be a finite {'positive ' if positive else ''}number")
    if (low is not None and result < low) or (high is not None and result > high):
        raise ValueError(f"{name} must be within {low}..{high}")
    return result


def _integer(name, value):
    # Do not silently truncate floats, accept bools, or overflow a huge integer.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if not -(2**63) <= value < 2**63:
        raise ValueError(f"{name} must fit a signed 64-bit integer")
    return value


def _identifier(name, value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value


@dataclass(frozen=True)
class FaultEvent:
    kind: str
    onset_s: float = 300
    duration_s: float = 600
    severity: float = 0.8
    progression: float = 0.2
    recovery_s: float = 120
    recurrence_s: float | None = None
    target_signal: str | None = None
    sensor_mode: str = "biased"
    bias: float = 20.0
    event_id: str = ""

    def __post_init__(self):
        if not isinstance(self.kind, str) or self.kind not in FAULT_CATALOG:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        for name, low, high, positive in (
            ("onset_s", 0, 1e9, False), ("duration_s", 0, 1e9, True),
            ("severity", 0, 1, False), ("progression", 0, 1, False),
            ("recovery_s", 0, 1e9, False), ("bias", -1e6, 1e6, False),
        ):
            object.__setattr__(self, name, _number(name, getattr(self, name), low, high, positive=positive))
        if self.recurrence_s is not None:
            recurrence = _number("recurrence_s", self.recurrence_s, 0, 1e9, positive=True)
            if recurrence < self.duration_s + self.recovery_s:
                raise ValueError("recurrence_s must include duration_s + recovery_s")
            object.__setattr__(self, "recurrence_s", recurrence)
        if not isinstance(self.sensor_mode, str) or self.sensor_mode not in SENSOR_MODES:
            raise ValueError(f"sensor_mode must be one of {SENSOR_MODES}")
        if self.target_signal is not None:
            _identifier("target_signal", self.target_signal)
        if self.kind == "sensor_fault" and self.target_signal is None:
            object.__setattr__(self, "target_signal", "engine.oil_pressure_kpa")
        if self.kind != "sensor_fault" and self.target_signal is not None:
            raise ValueError("target_signal is only applicable to sensor_fault")
        if self.event_id != "":
            _identifier("event_id", self.event_id)


_DEFAULTS = {
    "seed": 793, "engine_seed": None, "engine_id": "virtual-001",
    "episode_id": "episode-001", "base_trajectory_id": None,
    "scenario": "ideal", "payload_class": "medium", "start_condition": "warm",
    "drive_cycle": "c-wtvc", "duration_s": 1800, "timestep_s": 5,
    "ambient_c": 25, "initial_wear": 0.0, "noise_scale": 1.0,
    "wear_time_scale": 1.0, "start_time": "2026-01-01T00:00:00+00:00",
    "initial_fault_wear": None, "initial_operating_hours": 0.0,
}


def _config(config):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    supplied = dict(config or {})
    # timestamp is a convenient alias for the UTC episode start.
    if "timestamp" in supplied:
        if "start_time" in supplied:
            raise ValueError("supply timestamp or start_time, not both")
        supplied["start_time"] = supplied.pop("timestamp")
    unknown = set(supplied) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(map(repr, unknown))}")
    cfg = {**_DEFAULTS, **supplied}
    cfg["seed"] = _integer("seed", cfg["seed"])
    cfg["engine_seed"] = _integer("engine_seed", cfg["seed"] if cfg["engine_seed"] is None else cfg["engine_seed"])
    for name in ("engine_id", "episode_id"):
        _identifier(name, cfg[name])
    for name, choices in (
        ("scenario", ("ideal", "stress", "overhaul")),
        ("payload_class", ("empty", "light", "medium", "heavy", "overload")),
        ("start_condition", ("cold", "warm")),
        ("drive_cycle", ("c-wtvc", "synthetic_route")),
    ):
        if cfg[name] not in choices:
            raise ValueError(f"{name} must be one of {choices}")
        cfg[name] = str(cfg[name])
    for name, low, high, positive in (
        ("duration_s", 0, 86400 * 365, True), ("timestep_s", 0, 86400, True),
        ("ambient_c", -50, 65, False), ("initial_wear", 0, 1, False),
        ("noise_scale", 0, 10, False), ("wear_time_scale", 0, 10000, False),
        ("initial_operating_hours", 0, None, False),
    ):
        cfg[name] = _number(name, cfg[name], low, high, positive=positive)
    history = cfg["initial_fault_wear"]
    if history is None:
        history = {}
    if not isinstance(history, Mapping):
        raise ValueError("initial_fault_wear must be a mapping of mechanical wear kinds to 0..1")
    unknown_wear = set(history) - set(_WEAR_RATES)
    if unknown_wear:
        raise ValueError(f"unsupported initial_fault_wear kinds: {sorted(map(repr, unknown_wear))}")
    cfg["initial_fault_wear"] = {
        kind: _number(f"initial_fault_wear.{kind}", history.get(kind, 0), 0, 1)
        for kind in _WEAR_RATES
    }
    if cfg["timestep_s"] > cfg["duration_s"]:
        raise ValueError("timestep_s must not exceed duration_s")
    intervals = cfg["duration_s"] / cfg["timestep_s"]
    if not math.isfinite(intervals) or intervals >= _MAX_SAMPLES:
        raise ValueError(f"episode exceeds {_MAX_SAMPLES} samples")
    count = math.floor(intervals) + 1
    start = cfg["start_time"]
    if isinstance(start, str):
        try:
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("start_time must be an ISO-8601 timestamp") from exc
    if not isinstance(start, datetime) or start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("start_time must include a timezone")
    start = start.astimezone(timezone.utc)
    try:
        start + timedelta(seconds=cfg["duration_s"])
    except OverflowError as exc:
        raise ValueError("episode timestamps exceed datetime range") from exc
    cfg["start_time"] = start.isoformat()
    if cfg["base_trajectory_id"] is None:
        identity = tuple((key, cfg[key]) for key in cfg if key not in ("engine_id", "episode_id", "base_trajectory_id"))
        cfg["base_trajectory_id"] = "trajectory-" + hashlib.sha256(repr(identity).encode()).hexdigest()[:16]
    else:
        _identifier("base_trajectory_id", cfg["base_trajectory_id"])
    return cfg, start, count


def _events(events):
    if isinstance(events, (FaultEvent, Mapping)):
        events = [events]
    if isinstance(events, (str, bytes)) or events is None:
        raise ValueError("events must contain FaultEvent objects or event mappings")
    result, ids = [], set()
    try:
        iterator = iter(events)
    except TypeError as exc:
        raise ValueError("events must be iterable") from exc
    for index, event in enumerate(iterator):
        if index >= 256:
            raise ValueError("at most 256 events are supported")
        if isinstance(event, Mapping):
            try:
                event = FaultEvent(**dict(event))
            except TypeError as exc:
                raise ValueError(f"invalid event fields: {exc}") from exc
        if not isinstance(event, FaultEvent):
            raise ValueError("each event must be a FaultEvent or mapping")
        event = replace(event, event_id=event.event_id or f"{event.kind}-{index + 1}")
        if event.event_id in ids:
            raise ValueError(f"duplicate event_id {event.event_id!r}")
        ids.add(event.event_id)
        result.append(event)
    return result


def _flatten_numeric(value, prefix=""):
    result = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_flatten_numeric(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            result.update(_flatten_numeric(child, f"{prefix}.{index}"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def _raw(frame):
    # Deliberately exclude legacy injected flags, operational labels and legacy
    # health/RUL proxies from observations. Preserve all numeric sensor fields.
    data = {}
    for name in ("engine", "transmission", "brake", "suspension", "imu", "frame", "gnss", "lidar", "radar", "dss", "payload"):
        data.update(_flatten_numeric(asdict(getattr(frame, name)), name))
    for name in ("body_position_pct", "hoist_angle_deg", "chassis_pitch_deg", "rack_deg", "cumulative_fuel_energy_kwh", "cumulative_fuel_volume_l", "coolant_condition_pct"):
        data[name] = getattr(frame, name)
    return data


def _bound(signal, value):
    if signal.endswith("_pct"):
        # Capacity utilization is a percentage of rated capacity and may exceed
        # 100 for overload; condition and other sensor percentages never do.
        return max(0.0, min(125.0 if signal == "payload.utilization_pct" else 100.0, value))
    if signal.endswith(("_mm_s", "_kpa", "_bar", "_kg_s", "_rpm", "_kw", "_lph")) or signal == "engine.rpm":
        return max(0.0, value)
    if signal in ("gnss.satellites", "lidar.points_per_scan", "radar.object_count", "dss.camera_fps", "gnss.accuracy_m"):
        return max(0.0, value)
    return value


def _envelope(event, elapsed):
    age = elapsed - event.onset_s
    if age < 0:
        return 0.0, "scheduled", 0
    occurrence = 0
    if event.recurrence_s is not None:
        occurrence = math.floor(age / event.recurrence_s)
        age -= occurrence * event.recurrence_s
    ramp = event.duration_s * event.progression
    if age < event.duration_s:
        fraction = min(1.0, age / ramp) if ramp else 1.0
        return event.severity * fraction, "progressing" if fraction < 1 else "active", occurrence
    if event.recovery_s and age < event.duration_s + event.recovery_s:
        return event.severity * (1 - (age - event.duration_s) / event.recovery_s), "recovering", occurrence
    return 0.0, "between_occurrences" if event.recurrence_s else "complete", occurrence


# channel: (developed delta, causal delay seconds). Conditions and permanent
# mechanical damage also use the cumulative exposure table below.
_EFFECTS = {
    "engine_oil_pump_wear": {"engine.oil_pressure_kpa": (-145, 0), "engine.oil_temp_c": (38, 25), "engine.block_temp_c": (25, 25), "engine.exhaust_temp_c": (290, 25), "engine.block_vibration_mm_s": (6, 55)},
    "piston_ring_damage": {"engine.oil_pressure_kpa": (-110, 0), "engine.oil_level_pct": (-35, 0), "engine.oil_temp_c": (34, 25), "engine.block_temp_c": (27, 25), "engine.exhaust_temp_c": (300, 25), "engine.block_vibration_mm_s": (6, 55), "engine.oil_condition_pct": (-35, 25)},
    "turbo_exhaust_fault": {"engine.boost_kpa": (-65, 0), "engine.exhaust_temp_c": (360, 25)},
    "coolant_system_failure": {"engine.coolant_flow_kg_s": (-1.9, 0), "engine.thermostat_open_pct": (-65, 0), "engine.block_temp_c": (43, 25), "engine.coolant_temp_c": (43, 25), "engine.coolant_inlet_temp_c": (35, 25), "engine.oil_temp_c": (38, 25), "engine.exhaust_temp_c": (360, 25), "engine.coolant_pump_temp_c": (44, 25), "engine.block_vibration_mm_s": (4, 55)},
    "coolant_degradation": {"coolant_condition_pct": (-35, 0), "engine.coolant_flow_kg_s": (-0.45, 0), "engine.coolant_pump_temp_c": (24, 25)},
    "gearbox_seal_wear": {"transmission.oil_pressure_kpa": (-145, 0), "transmission.oil_temp_c": (44, 25), "transmission.vibration_mm_s": (6.5, 55), "transmission.slip_pct": (12, 0)},
    "gearbox_bearing_wear": {"transmission.oil_pressure_kpa": (-125, 0), "transmission.oil_temp_c": (42, 25), "transmission.vibration_mm_s": (8, 55), "transmission.oil_condition_pct": (-25, 25)},
    # Normal aging has no instantaneous step: both condition loss and vibration
    # arise from persistent exposure-integrated wear below.
    "hydraulic_normal_aging": {},
    "hydraulic_contamination": {"suspension.hydraulic_oil_condition_pct": (-37, 0), "suspension.actuator_vibration_mm_s": (7.5, 55), "suspension.shock_temp_c": (17, 25)},
    "bearing_failure": {"engine.oil_pressure_kpa": (-90, 0), "engine.oil_temp_c": (35, 25), "engine.block_temp_c": (20, 25), "engine.block_vibration_mm_s": (10, 55), "engine.oil_condition_pct": (-22, 25)},
    "low_oil": {"engine.oil_pressure_kpa": (-190, 0), "engine.oil_level_pct": (-70, 0), "engine.oil_temp_c": (36, 25), "engine.block_temp_c": (25, 25), "engine.block_vibration_mm_s": (5, 55)},
    "overheat": {"engine.block_temp_c": (43, 25), "engine.coolant_temp_c": (43, 25), "engine.oil_temp_c": (40, 25), "engine.exhaust_temp_c": (360, 25), "engine.coolant_pump_temp_c": (40, 25)},
    "brake_wear": {"brake.pad_wear_pct": (88, 0), "brake.stroke_pct": (80, 0), "brake.disc_temp_c": (150, 25)},
    "perception_degraded": {"gnss.accuracy_m": (18, 0), "gnss.satellites": (-9, 0), "lidar.obstacle_confidence_pct": (-60, 0), "lidar.points_per_scan": (-550, 0), "radar.target_confidence_pct": (-55, 0), "dss.camera_fps": (-22, 0)},
    "sensor_fault": {},
    "unrelated_multi_blip": {"engine.oil_temp_c": (15, 0), "transmission.vibration_mm_s": (2.4, 0), "gnss.accuracy_m": (3, 0)},
}
_WEAR_RATES = {  # latent damage fraction per hour of unit-severity exposure
    "engine_oil_pump_wear": 0.18, "piston_ring_damage": 0.22,
    "gearbox_seal_wear": 0.16, "gearbox_bearing_wear": 0.20,
    "bearing_failure": 0.30, "brake_wear": 0.25,
    "hydraulic_normal_aging": 0.025, "hydraulic_contamination": 0.6,
    "coolant_degradation": 0.3,
}
_WEAR_EFFECTS = {
    # The matrix distinguishes pressure-loss mechanisms by preserved condition
    # (pump/seal) versus degraded oil condition (ring/gear/bearing damage).
    "engine_oil_pump_wear": {"engine.oil_pressure_kpa": -80, "engine.oil_temp_c": 12, "engine.block_vibration_mm_s": 6},
    "piston_ring_damage": {"engine.oil_level_pct": -45, "engine.block_vibration_mm_s": 4, "engine.oil_condition_pct": -50},
    "gearbox_seal_wear": {"transmission.oil_pressure_kpa": -90, "transmission.vibration_mm_s": 5},
    "gearbox_bearing_wear": {"transmission.oil_pressure_kpa": -100, "transmission.vibration_mm_s": 5, "transmission.oil_temp_c": 15, "transmission.oil_condition_pct": -50},
    "bearing_failure": {"engine.block_vibration_mm_s": 6, "engine.oil_condition_pct": -40},
    "brake_wear": {"brake.pad_wear_pct": 90, "brake.stroke_pct": 40},
    "hydraulic_normal_aging": {"suspension.hydraulic_oil_condition_pct": -100, "suspension.actuator_vibration_mm_s": 8},
    "hydraulic_contamination": {"suspension.hydraulic_oil_condition_pct": -100, "suspension.actuator_vibration_mm_s": 2},
    "coolant_degradation": {"coolant_condition_pct": -100, "engine.coolant_pump_temp_c": 24},
}


def _exposure(event, elapsed):
    """Exact envelope integral in seconds, independent of sample interval."""
    age = elapsed - event.onset_s
    if age <= 0:
        return 0.0
    ramp = event.duration_s * event.progression
    full = event.duration_s - ramp / 2 + event.recovery_s / 2
    cycles = 0
    if event.recurrence_s is not None:
        cycles = math.floor(age / event.recurrence_s)
        age -= cycles * event.recurrence_s
    active = min(age, event.duration_s)
    area = active if not ramp else active * active / (2 * ramp) if active < ramp else active - ramp / 2
    tail = min(max(0.0, age - event.duration_s), event.recovery_s)
    if event.recovery_s:
        area += tail - tail * tail / (2 * event.recovery_s)
    return event.severity * (cycles * full + area)


def _fault_wear(cfg, scheduled, elapsed, lag=0):
    """Inherited damage acts immediately; new damage follows channel lags."""
    wear = dict(cfg["initial_fault_wear"])
    for event in scheduled:
        if event.kind in _WEAR_RATES:
            increment = _exposure(event, elapsed - lag) / 3600 * cfg["wear_time_scale"] * _WEAR_RATES[event.kind]
            wear[event.kind] = min(1.0, wear[event.kind] + increment)
    return wear


def _wear_delay(key):
    if key.endswith("vibration_mm_s"):
        return 55
    if key.endswith("_temp_c"):
        return 25
    return 0


def _reference(raw, cfg, variant, elapsed):
    reference = dict(raw)
    temperature_offset, pressure_offset, vibration_offset, condition_offset = variant
    # Known identity and operating context are represented in the reference,
    # so a heavy/hot/cold operating condition is not an injected failure.
    cold = math.exp(-elapsed / 260) if cfg["start_condition"] == "cold" else 0.0
    ambient_effect = (cfg["ambient_c"] - 25) * (0.12 + 0.88 * cold)
    reference["engine.block_temp_c"] = raw["engine.coolant_temp_c"] + 1.5
    reference["engine.oil_condition_pct"] = 97 + condition_offset - cfg["initial_wear"] * 35
    reference["transmission.oil_condition_pct"] = 96 + condition_offset - cfg["initial_wear"] * 30
    for key in ("engine.block_temp_c", "engine.oil_temp_c", "engine.coolant_temp_c", "engine.coolant_inlet_temp_c", "engine.coolant_pump_temp_c", "transmission.oil_temp_c"):
        reference[key] += temperature_offset + ambient_effect
    reference["engine.oil_pressure_kpa"] += pressure_offset - cfg["initial_wear"] * 20
    reference["transmission.oil_pressure_kpa"] += pressure_offset * 0.5
    reference["engine.block_vibration_mm_s"] += vibration_offset + cfg["initial_wear"] * 0.4
    reference["transmission.vibration_mm_s"] += vibration_offset * 0.5
    reference["suspension.hydraulic_oil_condition_pct"] -= cfg["initial_wear"] * 30
    reference["coolant_condition_pct"] -= cfg["initial_wear"] * 25
    # New block channel is distinct; native temperatures, energies and IMU are
    # retained in raw_legacy, without treating mm/s as an acceleration in g.
    return {key: round(_bound(key, value), 6) for key, value in reference.items()}


def _noise_amplitude(key):
    if key.endswith("condition_pct"):
        return 0.25
    if key.endswith("_temp_c") or key.startswith("engine.exhaust_temperatures_c."):
        return 0.35
    if key.endswith("_pressure_kpa"):
        return 1.5
    if key.endswith("_vibration_mm_s") or key.endswith(".vibration_mm_s"):
        return 0.04
    return 0.0


def simulate_enhanced(config=None, events=()):
    """Return config, schedules, metadata and samples; never write files.

    Samples include t=0 and each timestep up to duration_s (inclusive when
    divisible). Events beyond the horizon are permitted and remain scheduled.
    Missing sensor faults are the only source of None observations. A frozen
    fault captures the last valid reading at the start of each occurrence.
    Multiple physical events add bounded effects; same-channel sensor events
    apply in schedule order, with missing taking precedence.
    Inherited fault wear remains outside the healthy reference. Returned
    metadata.final_engine_state can be merged into the next episode's config;
    it carries wear, operating hours and engine identity, not thermal, sensor,
    random-generator or still-running schedule state.
    """
    cfg, start, count = _config(config)
    scheduled = _events(events)
    journey = simulate_journey(SimulationConfig(
        seed=cfg["seed"], scenario=cfg["scenario"], payload_class=cfg["payload_class"],
        num_steps=count, timestep_s=cfg["timestep_s"], start_time=start,
        drive_cycle=cfg["drive_cycle"], start_condition=cfg["start_condition"],
    ))
    vrng = random.Random(cfg["engine_seed"])
    variant = (vrng.uniform(-1.5, 1.5), vrng.uniform(-5, 5), vrng.uniform(-0.12, 0.12), vrng.uniform(-1, 1))
    noise_rng = random.Random(f"enhanced:{cfg['seed']}")
    sensor_rng = random.Random(f"sensor:{cfg['seed']}")
    samples, frozen, previous_good = [], {}, {}
    known = set(_reference(_raw(journey[0]), cfg, variant, 0))
    for event in scheduled:
        if event.kind == "sensor_fault" and event.target_signal not in known:
            raise ValueError(f"unknown sensor target {event.target_signal!r}")
    for frame in journey:
        elapsed = frame.elapsed_s
        raw = _raw(frame)
        expected = _reference(raw, cfg, variant, elapsed)
        observation = {
            key: _bound(key, value + noise_rng.uniform(-1, 1) * _noise_amplitude(key) * cfg["noise_scale"])
            for key, value in expected.items()
        }
        levels = dict.fromkeys(FAULT_KINDS, 0.0)
        effect_envelopes = {kind: {} for kind in FAULT_KINDS}
        wear_by_lag = {lag: _fault_wear(cfg, scheduled, elapsed, lag) for lag in (0, 25, 55)}
        wear = wear_by_lag[0]
        phases, occurrences = {}, {}
        for event in scheduled:
            level, phase, occurrence = _envelope(event, elapsed)
            levels[event.kind] = min(1.0, levels[event.kind] + level)
            phases[event.event_id] = phase
            occurrences[event.event_id] = occurrence
            # Sum same-kind overlapping envelopes at each realized channel lag,
            # then take their maximum; do not count each channel as another fault.
            lags = {lag for _, lag in _EFFECTS[event.kind].values()}
            if event.kind == "sensor_fault":
                lags.add(0)
            for lag in lags:
                delayed_level = _envelope(event, elapsed - lag)[0]
                previous = effect_envelopes[event.kind].get(lag, 0)
                effect_envelopes[event.kind][lag] = min(1.0, previous + delayed_level)
            for key, (magnitude, lag) in _EFFECTS[event.kind].items():
                delayed_level, _, _ = _envelope(event, elapsed - lag)
                observation[key] += magnitude * delayed_level
            if event.kind in ("piston_ring_damage", "turbo_exhaust_fault", "coolant_system_failure", "overheat", "engine_oil_pump_wear"):
                egt_level = _envelope(event, elapsed - 25)[0]
                amplitude = _EFFECTS[event.kind].get("engine.exhaust_temp_c", (0, 0))[0]
                for index in range(6):
                    key = f"engine.exhaust_temperatures_c.{index}"
                    # Ring damage deliberately produces unequal cylinder EGT.
                    imbalance = (-80, -35, 150, 60, -55, 20)[index] if event.kind == "piston_ring_damage" else 0
                    erratic = 35 * math.sin(elapsed / 11 + index * 1.7) if event.kind == "piston_ring_damage" else 0
                    observation[key] += (amplitude + imbalance + erratic) * egt_level
        for kind in wear:
            for key, coefficient in _WEAR_EFFECTS[kind].items():
                observation[key] += coefficient * wear_by_lag[_wear_delay(key)][kind]
        effect_levels = {
            kind: max(max(effect_envelopes[kind].values(), default=0.0), wear.get(kind, 0.0))
            for kind in FAULT_KINDS
        }
        # Related indicators respond to the same mechanical disturbance, but
        # velocity (mm/s) and acceleration (g) are separate assumed responses,
        # never treated as a direct unit conversion.
        mechanical_level = max(effect_levels[k] for k in (
            "engine_oil_pump_wear", "piston_ring_damage", "bearing_failure", "low_oil"))
        observation["engine.vibration_rms_g"] += .32 * mechanical_level
        observation["imu.vibration_rms_g"] += .20 * mechanical_level
        slip_delta = observation["transmission.slip_pct"] - expected["transmission.slip_pct"]
        if slip_delta > 0:
            observation["transmission.efficiency_pct"] -= .45 * slip_delta
            observation["transmission.output_torque_nm"] *= max(0, 1 - slip_delta / 100)
        observation = {key: round(_bound(key, value), 6) for key, value in observation.items()}
        quality = dict.fromkeys(observation, "good")
        # Freeze a physical reading, not an inferred value or another sensor's
        # altered value. Store state per occurrence to reset recurring freezes.
        physical = dict(observation)
        for event in scheduled:
            if event.kind != "sensor_fault":
                continue
            level, _, occurrence = _envelope(event, elapsed)
            if level <= 0:
                continue
            key = event.target_signal
            token = (event.event_id, occurrence)
            if token not in frozen:
                frozen[token] = previous_good.get(key, physical[key])
            if quality[key] == "missing":
                continue
            mode = event.sensor_mode
            quality[key] = mode
            if mode == "missing":
                observation[key] = None
            elif mode == "frozen":
                observation[key] = frozen[token]
            elif mode == "biased":
                observation[key] += event.bias * level
            elif mode == "drifting":
                age = elapsed - event.onset_s - occurrence * (event.recurrence_s or 0)
                observation[key] += event.bias * level * min(1.0, age / event.duration_s)
            else:
                observation[key] += sensor_rng.uniform(-abs(event.bias), abs(event.bias)) * level
            if observation[key] is not None:
                observation[key] = round(_bound(key, observation[key]), 6)
        for key in physical:
            if quality[key] == "good":
                previous_good[key] = physical[key]
        samples.append({
            "elapsed_s": elapsed, "timestamp": frame.timestamp.isoformat(),
            "engine_id": cfg["engine_id"], "episode_id": cfg["episode_id"],
            "base_trajectory_id": cfg["base_trajectory_id"],
            "context": {
                "rpm": expected["engine.rpm"], "load_pct": expected["engine.load_pct"],
                "demand_pct": expected["engine.throttle_position_pct"], "ambient_c": cfg["ambient_c"],
                "payload_utilization_pct": frame.payload_utilization_pct,
                "payload_class": frame.payload_class.value, "start_condition": cfg["start_condition"],
                "mode": frame.mode.value, "transient": expected["engine.transient_index_pct"] / 100,
                "operating_hours": cfg["initial_operating_hours"] + elapsed / 3600, "cycle_phase": frame.cycle_phase,
                "scenario": cfg["scenario"], "drive_cycle": cfg["drive_cycle"],
            },
            "observations": observation, "expected": expected, "quality": quality,
            "raw_legacy": raw,
            "truth": {
                "active_faults": [kind for kind, value in levels.items() if value > 0],
                "fault_levels": levels, "effect_levels": effect_levels, "event_phases": phases,
                "event_occurrences": occurrences,
                "latent_wear": {"initial_wear": cfg["initial_wear"], **wear},
            },
        })
    return {
        "config": cfg, "events": [asdict(event) for event in scheduled],
        "metadata": {
            "schema_version": "enhanced-dumper/1", "source": "simulation",
            "physics_validated": False, "diagnostics_validated": False,
            "accelerated_aging": cfg["wear_time_scale"] > 1,
            "wear_time_scale": cfg["wear_time_scale"],
            "aging_semantics": "Envelope exposure seconds / 3600 times wear_time_scale; normal hydraulic aging 2.5 condition points/hour, contamination 60 points/hour plus reversible effects, at unit severity.",
            "label_semantics": "Injected synthetic schedules, effects and latent wear are simulation labels, never observed diagnoses or confirmed root causes. Schedule recovery does not imply a damage-free engine.",
            "schedule_level_semantics": "truth.fault_levels is the sum of current undelayed event envelopes for each kind, capped at 1. truth.active_faults contains only kinds with nonzero schedule levels, including scheduled recovery; inherited wear does not activate this list.",
            "effect_level_semantics": "truth.effect_levels is the maximum of accumulated mechanical wear (inherited plus this episode, capped at 1) and the maximum per-channel-delay sum of realized same-kind event envelopes (each sum capped at 1). Sensor effects use delay zero. Normal hydraulic aging has no direct envelope effect; its effect level is accumulated wear. Effects can remain after fault_levels becomes zero. These synthetic mechanism strengths are not probabilities or measured fault severities.",
            "history_semantics": "Explicit initial_fault_wear carries catalogue-specific damage outside the fault-free reference; initial_wear remains a generic known baseline characteristic. initial_operating_hours counts actual elapsed hours without wear_time_scale acceleration. No history is retained unless the caller supplies it. Final state is evaluated at the final sampled elapsed time and does not carry pending lagged, thermal, sensor or schedule state.",
            "final_fault_wear": dict(wear),
            "final_operating_hours": samples[-1]["context"]["operating_hours"],
            "final_engine_state": {
                "engine_id": cfg["engine_id"], "engine_seed": cfg["engine_seed"],
                "initial_wear": cfg["initial_wear"], "initial_fault_wear": dict(wear),
                "initial_operating_hours": samples[-1]["context"]["operating_hours"],
            },
            "energy_disclaimer": "Legacy numeric energy channels are preserved verbatim. Fault overlays are empirical synthetic perturbations and are not a recomputed or validated energy balance.",
            "noise_semantics": "noise_scale controls extra bounded uniform noise; legacy bounded journey noise is retained in the same-context reference.",
            "units": {key: _unit(key) for key in sorted(known)},
            "synthetic": True, "sample_count": count,
            "vibration_disclaimer": "mm/s velocity indicators and IMU acceleration in g are separate synthetic channels, not unit conversions.",
        }, "samples": samples,
    }


def _unit(key):
    if key.startswith("engine.exhaust_temperatures_c."):
        return "degC"
    suffixes = (("_mm_s", "mm/s"), ("_mps2", "m/s^2"), ("_mps", "m/s"), ("_dps", "deg/s"), ("_kwh", "kWh"), ("_microstrain", "microstrain"), ("_kg_s", "kg/s"), ("_temp_c", "degC"), ("_kpa", "kPa"), ("_bar", "bar"), ("_pct", "%"), ("_rpm", "rpm"), ("_kw", "kW"), ("_nm", "N m"), ("_lph", "L/h"), ("_mpa", "MPa"), ("_rms_g", "g"), ("_deg", "deg"), ("_hz", "Hz"), ("_mm", "mm"), ("_t", "t"), ("_l", "L"), ("_m", "m"))
    if key == "engine.rpm":
        return "rpm"
    if key == "dss.camera_fps":
        return "Hz"
    for suffix, unit in suffixes:
        if key.endswith(suffix):
            return unit
    return "count" if key.endswith(("satellites", "points_per_scan", "object_count", "gear")) else "1"


def synthesize_vibration(sample, sample_rate_hz=4096, duration_s=1.0, seed=0):
    """Synthesize a bounded velocity waveform from observations only.

    Assumed 1x/2x/4.7x shaft-order components are illustrative, not calibrated
    bearing geometry or a fault classifier. Harmonic weights respond to observed
    cylinder EGT spread and oil condition; optional missing readings have neutral
    weights. Neither schedules, quality labels nor truth are consulted.
    At standstill there are no shaft
    components. Every component must be strictly below Nyquist. Maximum rate
    65536 Hz, duration 10 s, and 262144 samples bound memory use.
    """
    rate = _number("sample_rate_hz", sample_rate_hz, 16, 65536)
    duration = _number("duration_s", duration_s, 0, 10, positive=True)
    _integer("seed", seed)
    size = int(rate * duration)
    if size < 2 or size > _MAX_WAVEFORM_SAMPLES:
        raise ValueError(f"waveform must contain 2..{_MAX_WAVEFORM_SAMPLES} samples")
    if not isinstance(sample, Mapping) or not isinstance(sample.get("observations"), Mapping):
        raise ValueError("sample must have an observations mapping")
    readings = sample["observations"]
    rpm = _number("engine.rpm", readings.get("engine.rpm"), 0, 10000)
    rms = _number("engine.block_vibration_mm_s", readings.get("engine.block_vibration_mm_s"), 0, 100)
    frequencies = [rpm / 60 * order for order in (1, 2, 4.7)] if rpm > 0 else []
    if frequencies and max(frequencies) >= rate / 2:
        raise ValueError("sample_rate_hz violates Nyquist for assumed shaft orders")
    rng = random.Random(seed)
    phases = [rng.uniform(0, 2 * math.pi) for _ in frequencies]
    egts = [readings.get(f"engine.exhaust_temperatures_c.{i}") for i in range(6)]
    egts = [value for value in egts if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
    egt_span = max(egts) - min(egts) if len(egts) >= 2 else None
    if egt_span is not None and not math.isfinite(egt_span):
        egt_span = None
    imbalance = max(0.0, min(1.0, (egt_span - 40) / 250)) if egt_span is not None else 0.0
    condition = readings.get("engine.oil_condition_pct")
    if not isinstance(condition, (int, float)) or isinstance(condition, bool) or not math.isfinite(condition) or not 0 <= condition <= 100:
        condition = None
    condition_deficit = max(0.0, min(1.0, (90 - condition) / 90)) if condition is not None else 0.0
    weights = (0.85, 0.35 + 0.30 * imbalance, 0.25 + 0.30 * condition_deficit)
    waveform = [
        sum(weight * math.sin(2 * math.pi * frequency * index / rate + phase)
            for frequency, phase, weight in zip(frequencies, phases, weights)) + rng.uniform(-0.1, 0.1)
        for index in range(size)
    ]
    mean = sum(waveform) / size
    centered = [value - mean for value in waveform]
    current_rms = math.sqrt(sum(value * value for value in centered) / size)
    waveform = [round(value * rms / current_rms, 8) if current_rms else 0.0 for value in centered]
    return {
        "sample_rate_hz": rate, "units": "mm/s", "rpm": rpm, "waveform": waveform,
        "metadata": {
            "source": "simulation", "synthetic": True, "physics_validated": False,
            "assumed_frequency_components": frequencies, "shaft_orders": [1, 2, 4.7] if frequencies else [],
            "component_semantics": "Assumed synthetic velocity components; no bearing geometry or fault-frequency validation.",
            "component_weights": list(weights) if frequencies else [],
            "observation_drivers": {
                "cylinder_egt_span_c": egt_span, "egt_imbalance_index": imbalance,
                "engine_oil_condition_pct": condition, "oil_condition_deficit_index": condition_deficit,
            },
            "harmonic_semantics": "Bounded illustrative harmonic weights increase with observed EGT spread and low oil condition. This mapping is synthetic, not a combustion or bearing diagnosis; shaft orders are assumed, not identified from data.",
            "target_rms_mm_s": rms, "duration_s": size / rate,
            "requested_duration_s": duration, "seed": seed,
        },
    }


__all__ = ["FAULT_CATALOG", "FAULT_KINDS", "FaultEvent", "simulate_enhanced", "synthesize_vibration"]
