"""Observation-only, unvalidated feature engineering and fault hypotheses.

Never consumes simulator truth, expected/oracle readings, schedules or named
quality labels. Healthy references are transparent context-conditioned priors,
not paired simulator outputs. Results are hypotheses, not confirmed repairs or
a trained DL classifier. RUL remains unavailable without independent outcomes.
"""
from collections import deque
from dataclasses import fields
import math
import statistics

from enhanced_simulation import FAULT_CATALOG
from operational_ranges import classify
from telemetry_model import (
    EngineTelemetry, TransmissionTelemetry, BrakeTelemetry, SuspensionTelemetry,
    IMUTelemetry, FrameTelemetry, GNSSTelemetry, LiDARTelemetry, RadarTelemetry,
    DSSTelemetry, PayloadTelemetry,
)


def _sensor_columns():
    keys = []
    for prefix, cls in (("engine", EngineTelemetry), ("transmission", TransmissionTelemetry),
                        ("brake", BrakeTelemetry), ("suspension", SuspensionTelemetry),
                        ("imu", IMUTelemetry), ("frame", FrameTelemetry),
                        ("gnss", GNSSTelemetry), ("lidar", LiDARTelemetry),
                        ("radar", RadarTelemetry), ("dss", DSSTelemetry)):
        value = cls()
        keys.extend(f"{prefix}.{f.name}" for f in fields(cls)
                    if isinstance(getattr(value, f.name), (int, float))
                    and not isinstance(getattr(value, f.name), bool))
    keys.extend(f"engine.exhaust_temperatures_c.{i}" for i in range(6))
    keys.extend(("payload.mass_t", "payload.rated_capacity_t", "payload.utilization_pct",
                 "body_position_pct", "hoist_angle_deg", "chassis_pitch_deg", "rack_deg",
                 "cumulative_fuel_energy_kwh", "cumulative_fuel_volume_l", "coolant_condition_pct",
                 "engine.block_temp_c", "engine.oil_condition_pct", "transmission.oil_condition_pct"))
    return tuple(sorted(set(keys)))


SENSOR_COLUMNS = _sensor_columns()
CONTEXT_NUMERIC = ("rpm", "load_pct", "demand_pct", "ambient_c", "payload_utilization_pct", "transient")
MODE_LOAD = {"parked": .02, "idle": .12, "hauling": .62, "upslope": .9,
             "downslope": .42, "braking": .5, "dumping": .38}
BANDS = {
    "engine.block_temp_c": "engine_block_temperature",
    "engine.oil_temp_c": "engine_oil_temperature",
    "engine.exhaust_temp_c": "exhaust_temperature",
    "engine.oil_pressure_kpa": "engine_oil_pressure",
    "engine.block_vibration_mm_s": "engine_block_vibration",
    "transmission.oil_temp_c": "gearbox_oil_temperature",
    "transmission.oil_pressure_kpa": "transmission_oil_pressure",
    "transmission.vibration_mm_s": "gearbox_vibration",
    "suspension.hydraulic_oil_condition_pct": "hydraulic_oil_condition",
    "suspension.actuator_vibration_mm_s": "hydraulic_actuator_vibration",
    "engine.coolant_pump_temp_c": "coolant_pump_temperature",
    "coolant_condition_pct": "coolant_condition",
    "engine.oil_condition_pct": "hydraulic_oil_condition",
    "transmission.oil_condition_pct": "hydraulic_oil_condition",
}
# Thresholds apply to residuals, not raw readings. Widen for normal transients.
THRESHOLDS = {
    "engine.block_temp_c": 12., "engine.coolant_temp_c": 12.,
    "engine.oil_temp_c": 9., "engine.exhaust_temp_c": 55.,
    "engine.coolant_pump_temp_c": 12., "engine.oil_pressure_kpa": 30.,
    "engine.block_vibration_mm_s": 1.6, "engine.oil_condition_pct": 8.,
    "engine.oil_level_pct": 15., "engine.coolant_flow_kg_s": .45,
    "engine.thermostat_open_pct": 25., "engine.boost_kpa": 30.,
    "transmission.oil_temp_c": 10., "transmission.oil_pressure_kpa": 25.,
    "transmission.vibration_mm_s": 1.2, "transmission.oil_condition_pct": 8.,
    "suspension.hydraulic_oil_condition_pct": 8., "suspension.actuator_vibration_mm_s": 1.2,
    "coolant_condition_pct": 8., "brake.pad_wear_pct": 20.,
    "brake.stroke_pct": 15., "brake.disc_temp_c": 35.,
    "gnss.accuracy_m": 4., "lidar.obstacle_confidence_pct": 20.,
    "radar.target_confidence_pct": 20.,
}
SUBSYSTEMS = {
    "engine": ("engine.oil_pressure_kpa", "engine.oil_temp_c", "engine.block_vibration_mm_s", "engine.oil_condition_pct"),
    "cooling": ("engine.coolant_temp_c", "engine.coolant_flow_kg_s", "coolant_condition_pct"),
    "transmission": ("transmission.oil_pressure_kpa", "transmission.vibration_mm_s", "transmission.oil_condition_pct"),
    "hydraulics": ("suspension.hydraulic_oil_condition_pct", "suspension.actuator_vibration_mm_s"),
    "brakes": ("brake.pad_wear_pct", "brake.stroke_pct", "brake.disc_temp_c"),
    "structure": ("frame.strain_microstrain", "frame.fatigue_index_pct"),
    "perception": ("gnss.accuracy_m", "lidar.obstacle_confidence_pct", "radar.target_confidence_pct"),
}


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clip(value, low=0., high=1.):
    return min(high, max(low, value))


def _observed(sample):
    """Explicit allowlist prevents unknown numeric labels becoming inputs."""
    raw = sample.get("observations", {})
    return {key: float(raw[key]) if _finite(raw.get(key)) else None for key in SENSOR_COLUMNS}


def _valid(key, value):
    if value is None:
        return False
    if key.endswith("_pct"):
        return 0 <= value <= (125 if key == "payload.utilization_pct" else 100)
    if key.endswith("_temp_c"):
        return -60 <= value <= (1500 if "exhaust" in key else 250)
    if key.endswith(("_pressure_kpa", "_vibration_mm_s", ".vibration_mm_s", "_flow_kg_s")):
        return 0 <= value <= 10000
    return True


def _episodes(samples):
    """Reset history at engine/episode boundaries; reject non-increasing time."""
    last_id, last_t = None, None
    for sample in samples:
        t = sample.get("elapsed_s")
        if not _finite(t) or t < 0:
            raise ValueError("elapsed_s must be finite and nonnegative")
        identity = (sample.get("engine_id"), sample.get("episode_id"))
        reset = last_t is None or identity != last_id
        if not reset and t <= last_t:
            raise ValueError("elapsed_s must increase within an episode; supply episode IDs when combining episodes")
        yield sample, reset, (0. if reset else float(t - last_t))
        last_id, last_t = identity, t


def _reference(sample, readings, state, dt):
    """Synthetic healthy prior using only operating context and raw sensors.

    Frame strain supplies a load/stress proxy; it is not a hidden scenario label.
    Flow/thermal states evolve independently of faulty flow/temperature sensors.
    Unknown machine offsets require commissioning calibration in real equipment.
    """
    c = sample.get("context", {})
    t = float(sample["elapsed_s"])
    load = c.get("load_pct") if _finite(c.get("load_pct")) else readings.get("engine.load_pct")
    load = _clip(load if load is not None else 50, 0, 100)
    ratio = _clip(c.get("payload_utilization_pct", 72) / 100 if _finite(c.get("payload_utilization_pct")) else .72, 0, 1.25)
    mode = MODE_LOAD.get(c.get("mode"), .62)
    strain = readings.get("frame.strain_microstrain")
    stress = _clip((strain - 45 - ratio * 440 - mode * 220) / 610) if strain is not None else _clip((ratio - .85) / .15)
    original_load = _clip(100 * (.10 + .72 * mode + .16 * max(0, ratio - .5)) + 14 * stress, 0, 100)
    if c.get("mode") == "idle":
        original_load = _clip(original_load, 5, 28)
    elif c.get("mode") == "parked":
        original_load = min(original_load, 12)
    trans = _clip(c.get("transient", 0) if _finite(c.get("transient")) else 0)
    cold = c.get("start_condition") == "cold"
    decay = math.exp(-t / 240) if cold else .05 * math.exp(-t / 60)
    ambient = c.get("ambient_c", 25) if _finite(c.get("ambient_c")) else 25
    ambient_effect = (ambient - 25) * (.12 + .88 * (math.exp(-t / 260) if cold else 0))
    slip = 5 * stress + (3 if c.get("mode") == "upslope" else 0)
    vibration = .035 + .0014 * original_load + .27 * stress
    compression = _clip(63 * ratio + 20 * mode + 14 * stress, 0, 100)
    r = {
        "engine.oil_pressure_kpa": 485 - .20 * original_load - 32 * stress,
        "engine.oil_condition_pct": 97., "engine.oil_level_pct": 92.,
        "transmission.oil_pressure_kpa": 368 - 18 * stress - slip,
        "transmission.vibration_mm_s": 1.4 + .18 * slip + .8 * stress,
        "transmission.oil_condition_pct": 96.,
        "transmission.oil_temp_c": 70 + .1 * load + 18 * stress + ambient_effect,
        "suspension.hydraulic_oil_condition_pct": 98 - .1 * compression - 8 * stress,
        "suspension.actuator_vibration_mm_s": 1.6 + 10 * vibration + 1.2 * stress,
        "coolant_condition_pct": 98. - 9 * stress,
        "brake.pad_wear_pct": 10.,
        "gnss.accuracy_m": .7 + 4.8 * stress,
        "lidar.obstacle_confidence_pct": 92.,
        "radar.target_confidence_pct": 90.,
    }
    # Initial condition offsets may be calibrated from the first observation;
    # absolute low-condition checks below still catch pre-existing degradation.
    for key in ("engine.oil_condition_pct", "transmission.oil_condition_pct", "coolant_condition_pct"):
        if key not in state["offsets"]:
            first = readings.get(key)
            state["offsets"][key] = _clip(first - r[key], -8, 3) if first is not None else 0
        r[key] += state["offsets"][key]
    pedal = readings.get("brake.pedal_pct") or 0
    r["brake.disc_temp_c"] = 35 + .75 * pedal + 80 * stress
    # Reference pad wear is bounded at the initial healthy measurement plus a
    # small allowed service drift, not the current possibly faulty wear signal.
    if "pad" not in state:
        state["pad"] = min(readings.get("brake.pad_wear_pct") or 10, 25)
    r["brake.pad_wear_pct"] = state["pad"] + min(10, t / 180)
    r["brake.stroke_pct"] = .76 * pedal + .18 * r["brake.pad_wear_pct"]
    r["engine.boost_kpa"] = _clip(98 + 1.25 * original_load + 22 * stress, 80, 240)
    if c.get("drive_cycle", "c-wtvc") == "c-wtvc":
        state.setdefault("mean", 28. if cold else 80.)
        state.setdefault("flow", .65 if cold else 2.15)
        state.setdefault("thermostat", 0. if cold else .55)
        target_open = _clip((state["mean"] - 82) / 13) if state["mean"] > 80 else 0
        if state["thermostat"] > 0 and state["mean"] > 80:
            target_open = max(target_open, state["thermostat"])
        step = dt if dt else 5.
        state["flow"] += (.65 + 2.35 * target_open - state["flow"]) * _clip(step / (180 if cold else 45))
        state["thermostat"] += (target_open - state["thermostat"]) * _clip(step / 20)
        target_mean = 25 + 60 * (1 - math.exp(-t / 260)) if cold else 80 + 12 * (1 - math.exp(-t / 500))
        state["mean"] += (target_mean - state["mean"]) * _clip(step / 90)
        energy = readings.get("engine.coolant_energy_flow_kw") or 20.
        delta = _clip(energy / max(.08, state["flow"] * 4.18), .5, 25)
        coolant = state["mean"] + delta * .5 + ambient_effect
        r.update({"engine.coolant_temp_c": coolant, "engine.block_temp_c": coolant + 1.5,
                  "engine.coolant_pump_temp_c": coolant,
                  "engine.oil_temp_c": 86 + .05 * load + 5 * decay + 8 * stress + ambient_effect,
                  "engine.exhaust_temp_c": 340 + 1.8 * load + 35 * trans + 35 * decay,
                  "engine.block_vibration_mm_s": _clip(2 + 30 * vibration + 1.2 * trans, 2, 12),
                  "engine.coolant_flow_kg_s": state["flow"],
                  "engine.thermostat_open_pct": state["thermostat"] * 100})
    else:
        r.update({"engine.coolant_temp_c": 72 + .21 * load + 26 * stress + ambient_effect,
                  "engine.block_temp_c": 73.5 + .21 * load + 26 * stress + ambient_effect,
                  "engine.coolant_pump_temp_c": 72 + .21 * load + 26 * stress + ambient_effect,
                  "engine.oil_temp_c": 68 + .20 * load + 20 * stress + ambient_effect,
                  "engine.exhaust_temp_c": 190 + 4.1 * load + 170 * stress,
                  "engine.block_vibration_mm_s": 2 + 30 * vibration})
    return r


def _unit(name):
    if "exhaust_temperatures_c." in name or name.endswith("_temp_c"):
        return "degC"
    for suffix, unit in (("_kpa", "kPa"), ("_mm_s", "mm/s"), ("_pct", "%"),
                         ("_kg_s", "kg/s"), (".rpm", "rpm"), ("_m", "m")):
        if name.endswith(suffix):
            return unit
    return "signal units"


FEATURE_DEFINITIONS = {}


def _slope(points):
    valid = [(t, v) for t, v in points if v is not None]
    if len(valid) < 2:
        return None
    mt = statistics.fmean(t for t, _ in valid)
    mv = statistics.fmean(v for _, v in valid)
    denominator = sum((t - mt) ** 2 for t, _ in valid)
    return sum((t - mt) * (v - mv) for t, v in valid) / denominator if denominator else None


def _process(samples):
    state, histories, previous, since, onset, frozen_since = None, {}, {}, {}, {}, {}
    result = []
    for sample, reset, dt in _episodes(samples):
        if reset:
            state = {"offsets": {}}
            histories, previous, since, onset, frozen_since = {}, {}, {}, {}, {}
        t, readings = float(sample["elapsed_s"]), _observed(sample)
        context = sample.get("context", {})
        context_valid = all(_finite(context.get(k)) for k in ("rpm", "load_pct", "ambient_c", "payload_utilization_pct"))
        context_valid = context_valid and 0 <= context.get("rpm", -1) <= 3000 and 0 <= context.get("load_pct", -1) <= 100
        context_valid = context_valid and context.get("mode") in MODE_LOAD and context.get("start_condition") in ("cold", "warm")
        reference = _reference(sample, readings, state, dt)
        residuals = {key: (readings.get(key) - value if _valid(key, readings.get(key)) else None)
                     for key, value in reference.items()}
        transient = context.get("transient", 0) if _finite(context.get("transient")) else 0
        cold_warmup = context.get("start_condition") == "cold" and t < 180
        factor = 1.4 if transient > .5 or cold_warmup else 1.
        operating_state = "unknown" if not context_valid else "cold_warmup" if cold_warmup else "transition" if transient > .5 else "steady_idle" if context["rpm"] <= 850 and context["load_pct"] <= 15 else "steady_loaded"
        quality, rates, residual_rates, slopes, persistent = {}, {}, {}, {}, {}
        for key, value in readings.items():
            quality[key] = "good" if _valid(key, value) else "missing" if value is None else "out_of_range"
            old = previous.get(key)
            rates[key] = (value - old[0]) / dt if dt and value is not None and old and old[0] is not None else None
            residual = residuals.get(key)
            residual_rates[key] = (residual - old[1]) / dt if dt and residual is not None and old and old[1] is not None else None
            history = histories.setdefault(key, deque())
            history.append((t, value, residual))
            # Sparse long-history streams still need multiple observations to
            # estimate a rate. Do not silently collapse a 120-s window to one
            # sample when dt is several minutes.
            window_s = max(120., 3 * dt)
            while history and history[0][0] < t - window_s:
                history.popleft()
            slopes[key] = _slope([(a, c) for a, _, c in history])
            # Constant healthy indices are not sensor failures. Only dynamic
            # pressure/temperature channels qualify, with changing context prior.
            dynamic = key.endswith("_pressure_kpa") or key.endswith("_temp_c")
            if dynamic and old and value == old[0] and value is not None:
                frozen_since.setdefault(key, t - dt)
                if t - frozen_since[key] >= 45 and residual is not None and abs(residual) > 1:
                    quality[key] = "frozen_suspected"
            else:
                frozen_since.pop(key, None)
            threshold = THRESHOLDS.get(key)
            above = context_valid and quality[key] == "good" and threshold and residual is not None and abs(residual) >= threshold * factor
            # Low oil/gearbox condition itself is evidence even when initial
            # commissioning offsets could partly absorb a pre-existing issue.
            if key in ("engine.oil_condition_pct", "transmission.oil_condition_pct") and value is not None and value < 80:
                above = context_valid and quality[key] == "good"
            if above:
                direction = 1 if residual > 0 else -1
                if key not in since or since[key][1] != direction:
                    since[key] = (t, direction)
                persistence_s = 15 if key == "engine.oil_pressure_kpa" else 30
                if transient > .5:
                    persistence_s = max(persistence_s, 45)
                persistent[key] = t - since[key][0] >= persistence_s
                if persistent[key]:
                    onset.setdefault(key, since[key][0])
            else:
                since.pop(key, None)
                persistent[key] = False
                onset.pop(key, None)
            previous[key] = (value, residual)
        egt = [readings[f"engine.exhaust_temperatures_c.{i}"] for i in range(6)]
        egt_spread = max(egt) - min(egt) if all(v is not None for v in egt) else None
        features = {key: readings[key] for key in SENSOR_COLUMNS}
        for key in SENSOR_COLUMNS:
            features[key + "__available_mask"] = int(quality[key] == "good")
        for key in CONTEXT_NUMERIC:
            features["context." + key] = float(context[key]) if _finite(context.get(key)) else None
        features["context.cold_start"] = int(context.get("start_condition") == "cold")
        for mode in MODE_LOAD:
            features["context.mode_" + mode] = int(context.get("mode") == mode)
        for key in THRESHOLDS:
            features[key + "__residual"] = residuals.get(key)
            features[key + "__rate"] = rates.get(key)
            features[key + "__residual_slope_120s"] = slopes.get(key)
            values = [v for tt, v, _ in histories[key] if tt >= t - 30 and v is not None]
            features[key + "__mean_30s"] = statistics.fmean(values) if values else None
            features[key + "__std_30s"] = statistics.pstdev(values) if values else None
        features["engine.egt_spread_c"] = egt_spread
        lead_pairs = (("engine.coolant_flow_kg_s", "engine.oil_temp_c"),
                      ("engine.oil_pressure_kpa", "engine.oil_temp_c"),
                      ("engine.oil_temp_c", "engine.block_vibration_mm_s"))
        for first, second in lead_pairs:
            features[f"leadlag.{first}__to__{second}_s"] = onset[second] - onset[first] if first in onset and second in onset else None
        for key in features:
            if key not in FEATURE_DEFINITIONS:
                base = key.split("__", 1)[0]
                if key.endswith("_mask"):
                    definition, unit = "Observed availability/validity mask, never injected sensor-mode labels", "0 or 1"
                elif key.startswith("leadlag."):
                    definition, unit = "Difference between persistent residual onset times; association, not causal proof", "s"
                elif "__residual_slope" in key:
                    definition, unit = "Least-squares residual slope over trailing max(120 seconds, 3 sampling intervals) within episode", _unit(base) + "/s"
                elif key.endswith("__rate"):
                    definition, unit = "First difference of observed readings divided by elapsed seconds, reset per episode", _unit(base) + "/s"
                elif key.endswith("__residual"):
                    definition, unit = "Observed minus transparent healthy context prior; no paired/oracle simulator values", _unit(base)
                elif "__mean" in key or "__std" in key:
                    definition, unit = "Trailing 30-second observed mean or population standard deviation", _unit(base)
                else:
                    definition, unit = "Raw observed sensor or available operating context", "degC" if key == "engine.egt_spread_c" else _unit(base)
                FEATURE_DEFINITIONS[key] = {"definition": definition, "units": unit}
        result.append({"elapsed_s": t, "readings": readings, "reference": reference,
                       "residuals": residuals, "rates": rates, "residual_rates": residual_rates,
                       "slopes": slopes, "persistent": persistent, "onsets": dict(onset),
                       "data_quality": quality, "operating_state": operating_state,
                       "context_valid": context_valid, "features": features, "egt_spread": egt_spread,
                       "threshold_factor": factor})
    return result


def engineer_features(samples):
    """Numeric inputs only; missing values stay None for train-only imputation."""
    return [row["features"] for row in _process(samples)]


def _hypotheses(row):
    x, r, p = row["readings"], row["residuals"], row["persistent"]
    def high(key):
        return p.get(key, False) and r.get(key) is not None and r[key] > 0
    def low(key):
        return p.get(key, False) and r.get(key) is not None and r[key] < 0
    def normal(key):
        return x.get(key) is not None and row["data_quality"].get(key) == "good" and not p.get(key, False)
    def condition_normal(key):
        return normal(key) and x[key] >= 90
    def condition_bad(key):
        return low(key) or (x.get(key) is not None and x[key] < 80 and p.get(key, False))
    output = []
    def add(kind, evidence_keys, alternatives):
        entry = FAULT_CATALOG[kind]
        evidence = [f"{key}: reading={x.get(key)}, residual={r.get(key)}, persistent={p.get(key, False)}"
                    for key in evidence_keys]
        output.append({"fault": kind, "evidence": evidence, "alternatives": alternatives,
                       "possible_cause": entry["possible_cause"], "mitigation": entry["mitigation"],
                       "confidence": "heuristic_not_probability", "root_cause_confirmed": False})
    pressure, vibration, oil = "engine.oil_pressure_kpa", "engine.block_vibration_mm_s", "engine.oil_condition_pct"
    pt, vg, cg = "transmission.oil_pressure_kpa", "transmission.vibration_mm_s", "transmission.oil_condition_pct"
    ch, vh, cc = "suspension.hydraulic_oil_condition_pct", "suspension.actuator_vibration_mm_s", "coolant_condition_pct"
    if low(pressure) and high(vibration) and condition_normal(oil):
        add("engine_oil_pump_wear", [pressure, vibration, oil], ["Pickup obstruction", "Lubrication leak", "Correlated sensor errors"])
    if high(vibration) and high("engine.oil_temp_c") and condition_bad(oil):
        if row["egt_spread"] is not None and row["egt_spread"] > 90:
            add("piston_ring_damage", [vibration, "engine.oil_temp_c", oil, "engine.exhaust_temp_c"], ["Injector imbalance", "Bearing wear", "Oil contamination"])
        else:
            add("bearing_failure", [vibration, "engine.oil_temp_c", oil], ["Misalignment", "Oil supply failure", "Sensor mounting issue"])
    if low(pressure) and low("engine.oil_level_pct"):
        add("low_oil", [pressure, "engine.oil_level_pct"], ["External leakage", "Oil consumption", "Quantity/pressure sensor error"])
    if high("engine.exhaust_temp_c") and normal("engine.block_temp_c") and normal("engine.oil_temp_c"):
        add("turbo_exhaust_fault", ["engine.exhaust_temp_c", "engine.boost_kpa"], ["Injector over-fuelling", "Exhaust sensor bias"])
    thermal = ("engine.block_temp_c", "engine.oil_temp_c", "engine.exhaust_temp_c", "engine.coolant_pump_temp_c")
    if all(high(key) for key in thermal):
        kind = "coolant_system_failure" if low("engine.coolant_flow_kg_s") else "overheat"
        add(kind, list(thermal) + ["engine.coolant_flow_kg_s"], ["Radiator restriction", "Thermostat malfunction", "Excessive heat input"])
    if condition_bad(cc) and high("engine.coolant_pump_temp_c") and normal("engine.block_temp_c"):
        add("coolant_degradation", [cc, "engine.coolant_pump_temp_c", "engine.block_temp_c"], ["Coolant contamination", "Pump-local issue", "Condition sensor bias"])
    if low(pt) and high(vg):
        if condition_bad(cg):
            add("gearbox_bearing_wear", [pt, vg, cg], ["Gear tooth damage", "Contamination", "Combined pressure/sensor issue"])
        elif condition_normal(cg):
            add("gearbox_seal_wear", [pt, vg, cg], ["Transmission pump loss", "External leak", "Pressure sensor bias"])
    if condition_bad(ch) and high(vh):
        slope = row["slopes"].get(ch)
        # 36 points/hour is an illustrative accelerated-demo boundary, not a
        # field hydraulic-fluid specification. Report ambiguous plateaus too.
        if slope is not None and slope < -.01:
            add("hydraulic_contamination", [ch, vh], ["Filter bypass", "Seal breach", "Abrupt condition sensor bias"])
        elif slope is not None and -.01 <= slope < -.0002:
            add("hydraulic_normal_aging", [ch, vh], ["Gradual component wear", "Progressive sensor drift"])
        else:
            add("hydraulic_contamination", [ch, vh], ["Normal aging: rate currently unresolved", "Residual damage", "Condition sensor fault"])
            output[-1]["fault"] = "hydraulic_degradation_unresolved"
            output[-1]["possible_cause"] = "Hydraulic degradation pattern; contamination versus wear rate is unresolved at this sample."
    if high("brake.pad_wear_pct") and high("brake.stroke_pct"):
        add("brake_wear", ["brake.pad_wear_pct", "brake.stroke_pct", "brake.disc_temp_c"], ["Actuator adjustment", "Stroke sensor error"])
    if high("gnss.accuracy_m") and low("lidar.obstacle_confidence_pct") and low("radar.target_confidence_pct"):
        add("perception_degraded", ["gnss.accuracy_m", "lidar.obstacle_confidence_pct", "radar.target_confidence_pct"], ["Environmental interference", "Sensor power/communications fault"])
    bad_quality = [key for key, value in row["data_quality"].items() if value != "good"]
    anomalies = [key for key, value in p.items() if value]
    if bad_quality:
        add("sensor_fault", bad_quality[:8], ["Communications loss", "Mechanical single-channel change", "Measurement saturation"])
    elif len(anomalies) == 1 and not output:
        add("sensor_fault", anomalies, ["Genuine physical single-channel anomaly", "Independent verification needed"])
        output[-1]["possible_cause"] = "Isolated deviation: sensor error and a physical single-channel change cannot be separated yet."
    elif len(anomalies) > 1 and not output:
        add("unrelated_multi_blip", anomalies, ["Unmodeled coupled fault", "Context-reference mismatch", "Several sensor faults"])
        output[-1]["possible_cause"] = "Multiple deviations do not match a complete signature; inspect before assigning a common cause."
    return output


def analyse_episode(samples):
    results = []
    for row in _process(samples):
        health = {}
        for subsystem, keys in SUBSYSTEMS.items():
            if not row["context_valid"] or any(row["data_quality"].get(key) != "good" for key in keys):
                health[subsystem] = None
                continue
            severities = [abs(row["residuals"][key]) / THRESHOLDS[key] / 4
                          for key in keys if key in THRESHOLDS and row["persistent"].get(key) and row["residuals"].get(key) is not None]
            if subsystem == "structure":
                severities.append(_clip((row["readings"]["frame.strain_microstrain"] - 900) / 1200))
            health[subsystem] = round(100 * (1 - _clip(max(severities, default=0))), 2)
        hypotheses = _hypotheses(row) if row["context_valid"] else []
        available = list(health.values())
        overall = min(available) if available and all(v is not None for v in available) else None
        pressure = row["readings"].get("engine.oil_pressure_kpa")
        if pressure is not None and pressure < 300:
            action = "Reduce load and follow approved shutdown/inspection procedures; independently verify oil pressure before further operation."
        elif any(value != "good" for value in row["data_quality"].values()) or not row["context_valid"]:
            action = "Restore and independently verify measurement quality/context before diagnosing the affected subsystem."
        elif hypotheses:
            action = "Arrange targeted inspection: " + "; ".join(h["mitigation"] for h in hypotheses)
        else:
            action = "No persistent modeled fault signature; continue trending. This is not a safety clearance."
        statuses = {key: classify(band, row["readings"][key]) if row["readings"][key] is not None else "unknown" for key, band in BANDS.items()}
        results.append({"elapsed_s": row["elapsed_s"], "statuses": statuses,
                        "expected": row["reference"], "residuals": row["residuals"], "rates": row["rates"],
                        "residual_slopes_per_s": row["slopes"], "operating_state": row["operating_state"],
                        "persistent_anomalies": [key for key, value in row["persistent"].items() if value],
                        "onset_times_s": row["onsets"], "hypotheses": hypotheses,
                        "data_quality": row["data_quality"], "context_valid": row["context_valid"],
                        "subsystem_health": health, "overall_health": overall,
                        "root_cause_confirmed": False, "rul_hours": None, "rul_status": "not_validated",
                        "reference_semantics": "Observation/context prior, not simulator expected/truth; heuristic and not field-calibrated",
                        "recommended_action": action})
    return results
