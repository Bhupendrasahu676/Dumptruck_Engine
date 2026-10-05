"""Reproducible, standard-library engine condition simulation.

Healthy priors depend only on observable context. Physical faults and wear
are isolated in truth; sensor faults only change observations. Schedules have
finite active and linear recovery periods. Severity starts at its configured
value and grows by severity * progression, capped at one. Recurrence is the
interval between starts; overlapping repetitions use the strongest envelope.
This is an illustrative lab model, not a calibrated physical engine solver.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

FAULT_KINDS = (
    "lubrication_degradation", "oil_starvation", "bearing_degradation",
    "cooling_restriction", "thermostat_malfunction", "combustion_imbalance",
    "sensor_fault", "incomplete_overhaul",
)
WEAR_KEYS = ("lubrication", "bearing", "cooling", "combustion")
SENSORS = (
    "oil_pressure_kpa", "oil_temp_c", "coolant_temp_c", "coolant_flow_kg_s",
    "thermostat_open_pct", "vibration_mm_s", "egt_c",
)
PAYLOAD_CLASSES = ("empty", "light", "medium", "heavy", "overload")
MAX_SAMPLES = 100_001

__all__ = ["FAULT_KINDS", "WEAR_KEYS", "SENSORS", "PAYLOAD_CLASSES",
           "MAX_SAMPLES", "default_schedule", "expected_readings", "simulate_test"]


def _number(value: Any, name: str, minimum: float = 0.0,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} must be within {minimum}..{maximum or 'infinity'}")
    return result


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def default_schedule(kind: str) -> dict[str, Any]:
    """Return a fresh editable schedule for the default 600-second test."""
    if kind not in FAULT_KINDS:
        raise ValueError(f"Unsupported fault kind: {kind!r}")
    result = {"kind": kind, "onset_s": 60.0, "duration_s": 240.0,
              "severity": 0.65, "recurrence_s": 0.0, "progression": 0.25,
              "recovery_s": 30.0}
    if kind == "thermostat_malfunction":
        result["subtype"] = "delayed"
    elif kind == "sensor_fault":
        result.update(subtype="biased", sensor="oil_pressure_kpa")
    return result


def expected_readings(context: Mapping[str, Any]) -> dict[str, Any]:
    """Healthy means given RPM, demand, start condition and elapsed time.

    Cold temperatures near ambient, lower idle pressure, and transient
    vibration are normal. Extra metadata is ignored; faults and wear are
    never consulted. ``transient`` is intensity 0..1 (or a boolean).
    """
    if not isinstance(context, Mapping):
        raise ValueError("context must be a mapping")
    elapsed = _number(context.get("elapsed_s", 0.0), "elapsed_s")
    rpm = _number(context.get("rpm", 700.0), "rpm", maximum=5000.0)
    demand = _number(context.get("demand_pct", 10.0), "demand_pct", maximum=120.0)
    ambient = _number(context.get("ambient_c", 22.0), "ambient_c", -60.0, 70.0)
    transient = context.get("transient", 0.0)
    transient = float(transient) if isinstance(transient, bool) else transient
    transient = _number(transient, "transient", maximum=1.0)
    start = context.get("start_condition", "cold")
    if start not in ("cold", "warm"):
        raise ValueError("start_condition must be cold or warm")
    oil_start = ambient if start == "cold" else 78.0
    coolant_start = ambient if start == "cold" else 82.0
    oil_target = 78.0 + 0.19 * demand + 0.08 * (ambient - 22.0)
    coolant_target = 82.0 + 0.105 * demand + 0.05 * (ambient - 22.0)
    oil_temp = oil_target + (oil_start - oil_target) * math.exp(-elapsed / 155.0)
    coolant_temp = coolant_target + (coolant_start - coolant_target) * math.exp(-elapsed / 105.0)
    thermostat = 100.0 * _clip((coolant_temp - 73.0) / 15.0)
    oil_pressure = (95.0 + 0.14 * rpm) * (1.0 + 0.0035 * (85.0 - oil_temp))
    oil_pressure *= 1.0 + 0.0005 * demand
    flow = (0.75 + rpm / 850.0) * (0.42 + 0.58 * thermostat / 100.0)
    vibration = 0.75 + rpm / 1700.0 + demand / 160.0 + 1.65 * transient
    egt_mean = 135.0 + demand * 4.05 + rpm * 0.028
    egt_mean -= 32.0 * math.exp(-elapsed / 45.0) if start == "cold" else 0.0
    return {
        "oil_pressure_kpa": oil_pressure, "oil_temp_c": oil_temp,
        "coolant_temp_c": coolant_temp, "coolant_flow_kg_s": flow,
        "thermostat_open_pct": thermostat, "vibration_mm_s": vibration,
        "egt_c": [egt_mean + offset for offset in (-3.0, 1.0, 3.0, -1.0, 2.0, -2.0)],
    }


def _config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    result = {"engine_id": config.get("engine_id", "engine-demo"),
              "overhaul_id": config.get("overhaul_id"),
              "test_id": config.get("test_id", "test-demo")}
    for key, value in result.items():
        if key == "overhaul_id" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty string")
    seed = config.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not -(2 ** 63) <= seed < 2 ** 63:
        raise ValueError("seed must be a signed 64-bit integer")
    result["seed"] = seed
    for key, default in (("duration_s", 600.0), ("timestep_s", 1.0), ("initial_hours", 0.0)):
        result[key] = _number(config.get(key, default), key)
    if result["duration_s"] <= 0 or result["timestep_s"] <= 0:
        raise ValueError("duration_s and timestep_s must be positive")
    ticks = result["duration_s"] / result["timestep_s"]
    if not math.isfinite(ticks) or ticks > MAX_SAMPLES - 1:
        raise ValueError(f"Test exceeds sample limit ({MAX_SAMPLES})")
    if not math.isfinite(result["initial_hours"] + result["duration_s"] / 3600.0):
        raise ValueError("Final operating hours must be finite")
    result["start_condition"] = config.get("start_condition", "cold")
    if result["start_condition"] not in ("cold", "warm"):
        raise ValueError("start_condition must be cold or warm")
    result["payload_class"] = config.get("payload_class", "medium")
    if result["payload_class"] not in PAYLOAD_CLASSES:
        raise ValueError(f"payload_class must be one of {PAYLOAD_CLASSES}")
    result["ambient_c"] = _number(config.get("ambient_c", 22.0), "ambient_c", -60.0, 70.0)
    return result


def _schedules(schedules: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(schedules, (str, bytes)) or not isinstance(schedules, Sequence):
        raise ValueError("schedules must be a sequence of mappings")
    result = []
    for index, supplied in enumerate(schedules):
        if not isinstance(supplied, Mapping):
            raise ValueError(f"Schedule {index} must be a mapping")
        kind = supplied.get("kind")
        schedule = default_schedule(kind)
        for key in ("onset_s", "duration_s", "recurrence_s", "recovery_s", "severity", "progression"):
            schedule[key] = _number(supplied.get(key, schedule[key]), f"schedule[{index}].{key}",
                                    maximum=1.0 if key in ("severity", "progression") else None)
        if schedule["duration_s"] <= 0:
            raise ValueError(f"schedule[{index}].duration_s must be positive")
        if not math.isfinite(schedule["onset_s"] + schedule["duration_s"] + schedule["recovery_s"]):
            raise ValueError(f"schedule[{index}] endpoint must be finite")
        subtype = supplied.get("subtype", schedule.get("subtype"))
        sensor = supplied.get("sensor", schedule.get("sensor"))
        if kind == "thermostat_malfunction":
            if subtype not in ("delayed", "stuck_open", "stuck_closed"):
                raise ValueError("Thermostat subtype must be delayed, stuck_open or stuck_closed")
            schedule["subtype"] = subtype
        elif kind == "sensor_fault":
            if subtype not in ("frozen", "biased", "missing"):
                raise ValueError("Sensor subtype must be frozen, biased or missing")
            if sensor not in SENSORS:
                raise ValueError(f"Unsupported sensor: {sensor!r}")
            schedule.update(subtype=subtype, sensor=sensor)
        elif subtype is not None:
            raise ValueError(f"{kind} does not support a subtype")
        if kind != "sensor_fault" and sensor is not None:
            raise ValueError("sensor is only supported for sensor_fault")
        schedule["schedule_index"] = index
        result.append(schedule)
    return result


def _event(schedule: dict[str, Any], elapsed: float) -> dict[str, Any] | None:
    """Evaluate finite event/recovery envelopes, including overlapping repeats."""
    relative = elapsed - schedule["onset_s"]
    if relative < 0 or schedule["severity"] == 0:
        return None
    period = schedule["recurrence_s"]
    duration, recovery = schedule["duration_s"], schedule["recovery_s"]
    if period:
        quotient = relative / period
        if not math.isfinite(quotient) or quotient > 2 ** 52:
            raise ValueError("recurrence_s is too small to resolve on this timeline")
        latest = math.floor(quotient)
        boundary_ratio = (relative - duration) / period
        boundary = math.floor(boundary_ratio) if boundary_ratio > 0 else 0
        candidates = {latest, boundary, boundary + 1}
    else:
        candidates = {0}
    strongest = None
    for cycle in sorted(candidates):
        onset = schedule["onset_s"] + cycle * period
        age = elapsed - onset
        if age < 0:
            continue
        if age < duration:
            progress = age / duration
            level = _clip(schedule["severity"] * (1.0 + schedule["progression"] * progress))
            phase = "active"
        elif recovery > 0 and age < duration + recovery:
            progress = 1.0
            level = _clip(schedule["severity"] * (1.0 + schedule["progression"]))
            level *= 1.0 - (age - duration) / recovery
            phase = "recovery"
        else:
            continue
        if strongest is None or level > strongest["effective_severity"]:
            strongest = {**schedule, "occurrence": cycle, "event_onset_s": onset,
                         "phase": phase, "event_progress": progress,
                         "effective_severity": level}
    return strongest


def _events(schedules: list[dict[str, Any]], elapsed: float) -> list[dict[str, Any]]:
    return [event for schedule in schedules if (event := _event(schedule, elapsed)) is not None]


def _context(config: dict[str, Any], elapsed: float) -> dict[str, Any]:
    """Smooth idle/ramp/haul/coast cycle with no hidden fault inputs."""
    payload = {"empty": 0.34, "light": 0.49, "medium": 0.67, "heavy": 0.85, "overload": 1.0}[config["payload_class"]]
    phase = elapsed % 180.0
    if phase < 20.0:
        load, transient = 0.0, 0.0
    elif phase < 40.0:
        x = (phase - 20.0) / 20.0
        load, transient = (1.0 - math.cos(math.pi * x)) / 2.0, math.sin(math.pi * x)
    elif phase < 120.0:
        load, transient = 1.0, 0.0
    elif phase < 150.0:
        x = (phase - 120.0) / 30.0
        load, transient = (1.0 + math.cos(math.pi * x)) / 2.0, math.sin(math.pi * x)
    else:
        load, transient = 0.0, 0.0
    return {"elapsed_s": elapsed, "rpm": 680.0 + (970.0 + 360.0 * payload) * load,
            "demand_pct": 9.0 + (102.0 * payload - 9.0) * load,
            "start_condition": config["start_condition"], "ambient_c": config["ambient_c"],
            "transient": transient}


def _levels(events: list[dict[str, Any]]) -> dict[str, float]:
    levels = dict.fromkeys(FAULT_KINDS, 0.0)
    for event in events:
        levels[event["kind"]] = _clip(levels[event["kind"]] + event["effective_severity"])
    return levels


def _advance_wear(wear: dict[str, float], events: list[dict[str, Any]],
                  context: dict[str, Any], dt: float) -> None:
    levels = _levels(events)
    load = 0.3 + context["demand_pct"] / 100.0
    starvation = levels["oil_starvation"] * (0.2 + 0.8 * context["transient"])
    # Fault aging is accelerated for the lab. Sensor events add no wear.
    rates = {
        "lubrication": 0.00032 * levels["lubrication_degradation"] + 0.00065 * starvation,
        "bearing": 0.00042 * levels["bearing_degradation"] + 0.0008 * starvation
                   + 0.00022 * levels["lubrication_degradation"] + 0.0006 * levels["incomplete_overhaul"],
        "cooling": 0.00028 * levels["cooling_restriction"] + 0.00018 * levels["thermostat_malfunction"],
        "combustion": 0.0003 * levels["combustion_imbalance"],
    }
    for key in WEAR_KEYS:
        wear[key] = _clip(wear[key] + dt * load * (0.00000015 + rates[key]))


def _physical(prior: dict[str, Any], context: dict[str, Any], wear: dict[str, float],
              events: list[dict[str, Any]]) -> dict[str, Any]:
    values = {**prior, "egt_c": list(prior["egt_c"])}
    levels = _levels(events)
    lube, bearing = levels["lubrication_degradation"], levels["bearing_degradation"]
    cooling, combustion = levels["cooling_restriction"], levels["combustion_imbalance"]
    starvation = levels["oil_starvation"] * (0.12 + 0.88 * context["transient"])
    incomplete_lube = 0.0
    for event in events:
        if event["kind"] == "incomplete_overhaul":
            settling = 1.0 - 0.82 * _clip((event["event_progress"] - 0.25) / 0.3)
            incomplete_lube += event["effective_severity"] * settling
    lube = _clip(lube + incomplete_lube)
    bearing = _clip(bearing + 0.8 * levels["incomplete_overhaul"])
    pressure_loss = 0.35 * lube + 0.67 * starvation + 0.12 * bearing
    pressure_loss += 0.34 * wear["lubrication"] + 0.22 * wear["bearing"]
    values["oil_pressure_kpa"] *= max(0.06, 1.0 - pressure_loss)
    values["oil_temp_c"] += 23.0 * lube + 15.0 * starvation + 10.0 * bearing
    values["oil_temp_c"] += 16.0 * wear["lubrication"] + 12.0 * wear["bearing"]
    values["vibration_mm_s"] += 7.5 * bearing + 4.8 * starvation + 0.8 * lube
    values["vibration_mm_s"] += 9.0 * wear["bearing"] + 1.4 * wear["lubrication"]
    values["coolant_flow_kg_s"] *= max(0.05, 1.0 - 0.72 * cooling - 0.42 * wear["cooling"])
    heat_phase = 1.0 - math.exp(-context["elapsed_s"] / 70.0)
    if context["start_condition"] == "warm":
        heat_phase = 1.0
    values["coolant_temp_c"] += heat_phase * (31.0 * cooling + 22.0 * wear["cooling"])
    values["thermostat_open_pct"] += 22.0 * cooling + 16.0 * wear["cooling"]
    for event in events:
        if event["kind"] != "thermostat_malfunction":
            continue
        level, subtype = event["effective_severity"], event["subtype"]
        if subtype == "stuck_open":
            values["thermostat_open_pct"] += level * (100.0 - values["thermostat_open_pct"])
            values["coolant_temp_c"] -= 20.0 * level * heat_phase
            values["coolant_flow_kg_s"] *= 1.0 + 0.45 * level
        else:
            restriction = level if subtype == "stuck_closed" else level * 0.72
            if subtype == "delayed":
                restriction *= 1.0 - 0.85 * _clip((event["event_progress"] - 0.35) / 0.35)
            values["thermostat_open_pct"] *= 1.0 - restriction
            values["coolant_temp_c"] += 37.0 * restriction * heat_phase
            values["coolant_flow_kg_s"] *= 1.0 - 0.7 * restriction
    imbalance = combustion + 0.65 * wear["combustion"]
    values["vibration_mm_s"] += 2.5 * imbalance
    egt_rise = 24.0 * lube + 14.0 * bearing + 21.0 * cooling
    egt_rise += 16.0 * wear["bearing"] + 20.0 * wear["cooling"]
    for cylinder, offset in enumerate((115.0, -70.0, 20.0, -15.0, 8.0, -4.0)):
        values["egt_c"][cylinder] += egt_rise + offset * imbalance
    values["thermostat_open_pct"] = _clip(values["thermostat_open_pct"], 0.0, 100.0)
    return values


def _measure(latent: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    noise = {"oil_pressure_kpa": 2.0, "oil_temp_c": 0.22, "coolant_temp_c": 0.2,
             "coolant_flow_kg_s": 0.018, "thermostat_open_pct": 0.5, "vibration_mm_s": 0.055}
    observations = {key: value + rng.gauss(0.0, noise[key])
                    for key, value in latent.items() if key != "egt_c"}
    for key in ("oil_pressure_kpa", "coolant_flow_kg_s", "vibration_mm_s"):
        observations[key] = max(0.0, observations[key])
    observations["thermostat_open_pct"] = _clip(observations["thermostat_open_pct"], 0.0, 100.0)
    observations["egt_c"] = [value + rng.gauss(0.0, 1.4) for value in latent["egt_c"]]
    return observations


def _sensor_faults(observations: dict[str, Any], events: list[dict[str, Any]],
                   frozen: dict[tuple[int, int], Any]) -> None:
    biases = {"oil_pressure_kpa": -155.0, "oil_temp_c": 30.0, "coolant_temp_c": 28.0,
              "coolant_flow_kg_s": -1.4, "thermostat_open_pct": -48.0,
              "vibration_mm_s": 5.0, "egt_c": 90.0}
    original = {key: list(value) if isinstance(value, list) else value
                for key, value in observations.items()}
    missing = set()
    for event in events:
        if event["kind"] != "sensor_fault":
            continue
        sensor, subtype = event["sensor"], event["subtype"]
        level = event["effective_severity"]
        if subtype == "missing":
            missing.add(sensor)
            continue
        value = observations[sensor]
        if subtype == "frozen":
            identity = (event["schedule_index"], event["occurrence"])
            held = frozen.setdefault(identity, original[sensor])
            blend = 1.0 if event["phase"] == "active" else level / _clip(
                event["severity"] * (1.0 + event["progression"]))
            if event["phase"] == "active":
                value = list(held) if isinstance(held, list) else held
            elif isinstance(value, list):
                value = [current + blend * (fixed - current) for current, fixed in zip(value, held)]
            else:
                value += blend * (held - value)
        else:
            bias = biases[sensor] * level
            value = [item + bias for item in value] if isinstance(value, list) else value + bias
        observations[sensor] = value
    for sensor in missing:
        observations[sensor] = [None] * 6 if sensor == "egt_c" else None


def simulate_test(config: Mapping[str, Any], schedules: Sequence[Mapping[str, Any]],
                  initial_wear: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Return config, samples and final_wear without mutating caller data.

    Samples include both endpoints. A partial final timestep ensures final
    hours equal initial_hours + duration_s / 3600. Truth contains
    ``scheduled_events``, ``latent`` readings, and a ``wear`` snapshot.
    All values are JSON-safe, including None for missing sensor readings.
    """
    normalized = _config(config)
    planned = _schedules(schedules)
    if initial_wear is None:
        initial_wear = {}
    if not isinstance(initial_wear, Mapping):
        raise ValueError("initial_wear must be a mapping")
    if any(key not in WEAR_KEYS for key in initial_wear):
        raise ValueError(f"initial_wear supports only {WEAR_KEYS}")
    wear = {key: _number(initial_wear.get(key, 0.0), f"initial_wear.{key}", maximum=1.0)
            for key in WEAR_KEYS}
    rng = random.Random(normalized["seed"])
    duration, timestep = normalized["duration_s"], normalized["timestep_s"]
    count = max(1, math.ceil(duration / timestep))
    samples = []
    frozen: dict[tuple[int, int], Any] = {}
    previous = 0.0
    for index in range(count + 1):
        elapsed = duration if index == count else index * timestep
        if index < count and elapsed >= duration:
            continue
        if index:
            midpoint = previous + (elapsed - previous) / 2.0
            _advance_wear(wear, _events(planned, midpoint), _context(normalized, midpoint), elapsed - previous)
        context = _context(normalized, elapsed)
        events = _events(planned, elapsed)
        latent = _physical(expected_readings(context), context, wear, events)
        observations = _measure(latent, rng)
        _sensor_faults(observations, events, frozen)
        samples.append({"elapsed_s": elapsed,
                        "operating_hours": normalized["initial_hours"] + elapsed / 3600.0,
                        "context": context, "observations": observations,
                        "truth": {"scheduled_events": events, "latent": latent, "wear": dict(wear)}})
        previous = elapsed
    return {"config": normalized, "samples": samples, "final_wear": dict(wear)}
