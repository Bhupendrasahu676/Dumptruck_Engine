"""Observation-only state-conditioned screening and conservatively gated RUL.

Healthy simulation references and health scores are unvalidated heuristics.
Nothing in this module establishes a root cause. No simulator truth, injected
faults, wear, or schedules are read. See the public function docstrings for the
JSON input and output schemas.
"""

from collections import deque
from collections.abc import Mapping
import math

from .simulation import expected_readings


_ROOT_CAUSE = "Not confirmed — technician evidence required"
_SCALARS = ("oil_pressure_kpa", "oil_temp_c", "coolant_temp_c",
            "coolant_flow_kg_s", "thermostat_open_pct", "vibration_mm_s")
_SIGNALS = _SCALARS + tuple("egt_c[%d]" % i for i in range(6))
_BOUNDS = {
    "oil_pressure_kpa": (0.0, 1500.0), "oil_temp_c": (-60.0, 220.0),
    "coolant_temp_c": (-60.0, 180.0), "coolant_flow_kg_s": (0.0, 1000.0),
    "thermostat_open_pct": (0.0, 100.0), "vibration_mm_s": (0.0, 150.0),
    **{"egt_c[%d]" % i: (-60.0, 1500.0) for i in range(6)},
}
_SUBSYSTEMS = {
    "lubrication": ("oil_pressure_kpa", "oil_temp_c"),
    "cooling": ("coolant_temp_c", "coolant_flow_kg_s", "thermostat_open_pct"),
    "rotating_assembly": ("vibration_mm_s",),
    "combustion": tuple("egt_c[%d]" % i for i in range(6)),
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except (OverflowError, ValueError):
        return None


def _flat(readings):
    if not isinstance(readings, Mapping):
        return {key: None for key in _SIGNALS}
    values = {key: readings.get(key) for key in _SCALARS}
    egts = readings.get("egt_c")
    valid_array = isinstance(egts, (list, tuple)) and len(egts) == 6
    values.update({"egt_c[%d]" % i: egts[i] if valid_array else None for i in range(6)})
    return values


def _nested(values):
    result = {key: values.get(key) for key in _SCALARS}
    result["egt_c"] = [values.get("egt_c[%d]" % i) for i in range(6)]
    return result


def _threshold(signal, prior, state):
    limits = {"oil_pressure_kpa": max(25.0, abs(prior) * 0.09),
              "oil_temp_c": 8.0, "coolant_temp_c": 8.0,
              "coolant_flow_kg_s": max(0.5, abs(prior) * 0.20),
              "thermostat_open_pct": 18.0,
              "vibration_mm_s": max(0.85, abs(prior) * 0.30)}
    threshold = limits.get(signal, 45.0)
    if state == "transition":
        threshold *= 1.7 if signal == "vibration_mm_s" else 1.35
    elif state == "cold_warmup" and signal in ("oil_temp_c", "coolant_temp_c"):
        threshold *= 1.5
    return threshold


def _state(context):
    if context["transient"]:
        return "transition"
    if context["start_condition"] == "cold" and context["elapsed_s"] < 180:
        return "cold_warmup"
    if context["rpm"] <= 850 and context["demand_pct"] <= 15:
        return "steady_idle"
    return "steady_loaded"


def _context(sample, elapsed, issues):
    source = sample.get("context")
    if not isinstance(source, Mapping) or elapsed is None or elapsed < 0:
        issues.append({"signal": "context", "type": "invalid", "detail": "A valid elapsed time and operating context are required."})
        return None
    result = {}
    bounds = {"elapsed_s": (0, 1e9), "rpm": (0, 5000), "demand_pct": (0, 120), "ambient_c": (-60, 70)}
    for key, (low, high) in bounds.items():
        value = _number(source.get(key))
        if value is None or not low <= value <= high:
            issues.append({"signal": "context." + key, "type": "invalid", "detail": "Missing or implausible state input; healthy reference unavailable."})
            return None
        result[key] = value
    if abs(result["elapsed_s"] - elapsed) > 1e-6:
        issues.append({"signal": "context.elapsed_s", "type": "invalid", "detail": "Context and sample clocks disagree."})
        return None
    start = source.get("start_condition")
    if start not in ("cold", "warm"):
        issues.append({"signal": "context.start_condition", "type": "invalid", "detail": "Expected cold or warm start context."})
        return None
    transient = source.get("transient")
    if not isinstance(transient, bool) and (_number(transient) is None or not 0 <= transient <= 1):
        issues.append({"signal": "context.transient", "type": "invalid", "detail": "Missing transition state."})
        return None
    result["start_condition"] = start
    result["transient"] = transient
    return result


def _trend(points, threshold):
    if len(points) < 5 or points[-1][0] - points[0][0] < 30:
        return "insufficient_history"
    count = max(1, len(points) // 3)
    early = sum(abs(p[3]) for p in list(points)[:count]) / count
    late = sum(abs(p[3]) for p in list(points)[-count:]) / count
    if late - early > threshold * 0.5:
        return "degrading_residual"
    if early - late > threshold * 0.5:
        return "recovering_residual"
    return "stable_residual"


def _diagnoses(persistent):
    findings = {item["signal"]: item for item in persistent}
    diagnoses = []

    def matches(signal, direction):
        return signal in findings and findings[signal]["direction"] == direction

    def add(hypothesis, signals, alternatives, confirmation):
        diagnoses.append({"hypothesis": hypothesis,
            "evidence": ["%s: %s residual for %.1f s (%s)" % (key, findings[key]["direction"], findings[key]["duration_s"], findings[key]["trend"]) for key in signals if key in findings],
            "alternatives": alternatives, "required_confirmation": confirmation})

    if matches("oil_pressure_kpa", "low"):
        corroborating = [s for s in ("oil_temp_c", "vibration_mm_s") if matches(s, "high")]
        add("Possible lubrication impairment" if corroborating else "Pressure-only deviation — sensor bias and lubrication fault are unresolved",
            ["oil_pressure_kpa"] + corroborating,
            ["Pressure transducer bias or wiring fault", "Oil grade, level or state-reference mismatch", "Lubrication supply loss or increased bearing clearance"],
            ["Compare with a calibrated mechanical pressure gauge at matched speed and oil temperature", "Inspect oil level/grade, leaks and filter; assess bearings if corroborated"])
    cooling = [s for s, direction in (("coolant_temp_c", "high"), ("coolant_flow_kg_s", "low"), ("thermostat_open_pct", "low")) if matches(s, direction)]
    if cooling:
        add("Possible cooling-system performance loss", cooling,
            ["Temperature/flow sensor bias", "Thermostat response mismatch", "Pump, coolant inventory or heat-rejection limitation"],
            ["Independently verify temperature and coolant flow", "Inspect coolant inventory, pump, thermostat and heat exchanger"])
    if matches("vibration_mm_s", "high"):
        add("Persistent excess vibration", ["vibration_mm_s"],
            ["Mounting or sensor coupling", "Misalignment or imbalance", "Bearing or combustion-related excitation"],
            ["Acquire tachometer-referenced vibration spectra under controlled load", "Inspect mounts, alignment and lubrication"])
    exhaust = [key for key in _SIGNALS if key.startswith("egt_c[") and key in findings]
    if exhaust:
        add("Cylinder exhaust-temperature imbalance or thermal deviation", exhaust,
            ["Thermocouple bias or connection fault", "Injector, compression or air-path imbalance", "Fuel/load or healthy-reference mismatch"],
            ["Verify thermocouples against an independent reference", "Perform cylinder balance, injector and compression checks"])
    covered = {*cooling, *exhaust}
    if matches("oil_pressure_kpa", "low"):
        covered.add("oil_pressure_kpa")
        covered.update(s for s in ("oil_temp_c", "vibration_mm_s") if matches(s, "high"))
    if matches("vibration_mm_s", "high"):
        covered.add("vibration_mm_s")
    other = [key for key in findings if key not in covered]
    if other:
        add("Persistent state-conditioned deviation requiring inspection", other,
            ["Sensor bias", "Operating-context/reference mismatch", "Mechanical or thermal deviation"],
            ["Repeat at matched stable conditions and independently verify affected measurements"])
    return diagnoses


def analyse_samples(observed_samples):
    """Return an aligned JSON-safe result list from observation-only samples.

    Input samples: {elapsed_s, operating_hours, context: {elapsed_s, rpm,
    demand_pct, start_condition: cold|warm, ambient_c, transient: 0..1},
    observations: {oil_pressure_kpa, oil_temp_c, coolant_temp_c,
    coolant_flow_kg_s, thermostat_open_pct, vibration_mm_s, egt_c: [6 numbers]}}.
    Unknown input keys are ignored. Output per sample:
      state: str; quality: {status: good|suspect|invalid, issues: [{signal,type,detail}]}
      residuals: same signal schema as observations, numbers or null
      anomalies: [{signal,direction,residual,threshold,onset_s,duration_s,
                   persistence: transient|persistent,trend,evidence}]
      diagnosis: [{hypothesis,evidence:[str],alternatives:[str],required_confirmation:[str]}]
      health: {score: number|null,status,subsystems:{name:number|null},method,validated:false}
      evidence_strength: qualitative str; action: str; root_cause: constant str.

    Call with a complete chronological test. Persistence requires 15 s (pressure),
    30 s (others), or 45 s during a transition. Gaps exceeding max(10 s, three
    recent median steps), invalid readings, and reversed clocks break continuity.
    Scores are unvalidated residual-severity heuristics, not remaining life.
    """
    results = []
    windows = {key: deque(maxlen=256) for key in _SIGNALS}
    freezes, events = {}, {}
    intervals = deque(maxlen=20)
    previous_time = previous_hours = None
    for raw_sample in observed_samples:
        sample = raw_sample if isinstance(raw_sample, Mapping) else {}
        issues = []
        elapsed = _number(sample.get("elapsed_s"))
        hours = _number(sample.get("operating_hours"))
        if hours is None or not 0 <= hours <= 1e9:
            issues.append({"signal": "operating_hours", "type": "invalid", "detail": "Missing or implausible operating-hour reading."})
        elif previous_hours is not None and hours < previous_hours:
            issues.append({"signal": "operating_hours", "type": "invalid", "detail": "Operating hours decreased within this test."})
        clock_valid = elapsed is not None and 0 <= elapsed <= 1e9
        continuous = clock_valid
        if previous_time is not None and clock_valid:
            step = elapsed - previous_time
            sorted_steps = sorted(intervals)
            median_step = sorted_steps[len(sorted_steps) // 2] if sorted_steps else 1.0
            if step <= 0 or step > max(10.0, median_step * 3):
                continuous = False
                issues.append({"signal": "elapsed_s", "type": "time_gap" if step > 0 else "invalid", "detail": "Elapsed-time discontinuity; persistence is restarted."})
                if step <= 0:
                    clock_valid = False
            if step > 0:
                intervals.append(step)
        if not continuous:
            events.clear()
            freezes.clear()
            for window in windows.values():
                window.clear()
        context = _context(sample, elapsed if clock_valid else None, issues)
        state = _state(context) if context else "unknown"
        prior = {}
        if context is not None:
            try:
                prior = _flat(expected_readings(context))
                if any(_number(prior.get(signal)) is None for signal in _SIGNALS):
                    issues.append({"signal": "reference", "type": "invalid", "detail": "Healthy reference contains an unusable channel; affected residuals are masked."})
            except (ValueError, TypeError, KeyError, OverflowError):
                issues.append({"signal": "reference", "type": "invalid", "detail": "Healthy reference could not be evaluated."})
        observed = _flat(sample.get("observations"))
        residuals, thresholds, anomalies = {}, {}, []
        for signal in _SIGNALS:
            value = _number(observed[signal])
            baseline = _number(prior.get(signal))
            low, high = _BOUNDS[signal]
            if value is None or not low <= value <= high:
                kind = "missing" if observed[signal] is None else "nonfinite_or_nonnumeric" if value is None else "out_of_range"
                issues.append({"signal": signal, "type": kind, "detail": "Measurement masked; not mechanical-fault evidence."})
                value = None
            if value is None or baseline is None:
                residuals[signal] = None
                events.pop(signal, None)
                freezes.pop(signal, None)
                windows[signal].clear()
                continue
            threshold = _threshold(signal, baseline, state)
            thresholds[signal] = threshold
            residual = value - baseline
            window = windows[signal]
            window.append((elapsed, value, baseline, residual))
            while window and elapsed - window[0][0] > 180:
                window.popleft()
            frozen = freezes.get(signal)
            if frozen is None or abs(value - frozen["value"]) > 1e-7:
                frozen = {"value": value, "since": elapsed, "min_prior": baseline, "max_prior": baseline}
                freezes[signal] = frozen
            frozen["min_prior"] = min(frozen["min_prior"], baseline)
            frozen["max_prior"] = max(frozen["max_prior"], baseline)
            if elapsed - frozen["since"] >= 45 and frozen["max_prior"] - frozen["min_prior"] > threshold:
                issues.append({"signal": signal, "type": "frozen", "detail": "Unchanging channel while its state-conditioned reference changes; masked pending sensor verification."})
                residuals[signal] = None
                events.pop(signal, None)
                continue
            residuals[signal] = round(residual, 5)
            trend = _trend(window, threshold)
            if abs(residual) <= threshold:
                events.pop(signal, None)
                continue
            direction = "high" if residual > 0 else "low"
            event = events.get(signal)
            if event is None or event["direction"] != direction:
                event = {"onset_s": elapsed, "direction": direction}
                events[signal] = event
            duration = elapsed - event["onset_s"]
            duration_gate = 45.0 if state == "transition" else 15.0 if signal == "oil_pressure_kpa" else 30.0
            persistence = "persistent" if duration >= duration_gate else "transient"
            anomalies.append({"signal": signal, "direction": direction,
                "residual": round(residual, 5), "threshold": round(threshold, 5),
                "onset_s": event["onset_s"], "duration_s": round(duration, 3),
                "persistence": persistence, "trend": trend,
                "evidence": "Observation-minus-state-conditioned healthy reference; cause unresolved"})
        persistent = [a for a in anomalies if a["persistence"] == "persistent"]
        if len(persistent) == 1:
            item = persistent[0]
            if item["trend"] == "degrading_residual" and item["duration_s"] >= 45:
                issues.append({"signal": item["signal"], "type": "drift_suspected", "detail": "Isolated progressive bias: sensor drift and a physical single-channel change remain indistinguishable; verify independently."})
        diagnoses = _diagnoses(persistent)
        subsystems = {}
        anomaly_by_signal = {a["signal"]: a for a in persistent}
        for name, signals in _SUBSYSTEMS.items():
            available = [s for s in signals if residuals.get(s) is not None]
            if len(available) != len(signals):
                subsystems[name] = None
                continue
            severity = max((min(1.0, abs(anomaly_by_signal[s]["residual"]) / thresholds[s] / 4.0) for s in signals if s in anomaly_by_signal), default=0.0)
            subsystems[name] = round(100.0 * (1.0 - severity), 1)
        score = min(subsystems.values()) if all(v is not None for v in subsystems.values()) else None
        hard_types = {"invalid", "missing", "nonfinite_or_nonnumeric", "out_of_range", "frozen"}
        quality_status = "invalid" if any(i["type"] in hard_types for i in issues) else "suspect" if issues else "good"
        if quality_status == "invalid":
            score = None
        severe_pressure = any(a["signal"] == "oil_pressure_kpa" and a["direction"] == "low" and abs(a["residual"]) >= 2 * a["threshold"] for a in persistent)
        if severe_pressure:
            action = "Reduce load and follow the equipment shutdown/inspection procedure; independently verify oil pressure before further operation."
        elif persistent:
            action = "Arrange targeted technician checks; repeat only under an approved controlled procedure and verify affected sensors."
        elif quality_status != "good":
            action = "Restore and independently verify measurement quality before making a condition decision."
        elif anomalies:
            action = "Monitor the deviation through settling; persistence or corroborating signals require inspection."
        else:
            action = "Continue routine monitoring; a healthy-reference match does not establish remaining life."
        strength = "corroborated_observations_not_causal" if len(persistent) >= 2 else "limited_single_channel" if persistent else "transient_only" if anomalies else "no_persistent_deviation_detected"
        if quality_status == "invalid":
            strength = "limited_by_measurement_quality"
        results.append({"state": state, "quality": {"status": quality_status, "issues": issues},
            "residuals": _nested(residuals), "anomalies": anomalies, "diagnosis": diagnoses,
            "health": {"score": score, "status": "unknown" if score is None else "degraded" if persistent else "monitor" if anomalies else "no_persistent_deviation", "subsystems": subsystems,
                       "method": "Unvalidated state-conditioned residual-severity heuristic", "validated": False},
            "evidence_strength": strength, "action": action, "root_cause": _ROOT_CAUSE})
        previous_time = elapsed if elapsed is not None and 0 <= elapsed <= 1e9 else None
        previous_hours = hours if hours is not None and 0 <= hours <= 1e9 else None
    return results


def assess_rul(history, evidence=None):
    """Gate externally supplied RUL predictions with independent outcome evidence.

    History: >=5 test summaries with operating_hours, health_score (0..100),
    test_id and overhaul_id. Require >=100 operating hours, distinct tests and
    a >=5-point sustained decline in one overhaul. Optional history.engine_id
    must match evidence.current_engine_id. Health is NOT a trained RUL model.

    Evidence schema (all entries required):
      current_engine_id, model_id, domain_id, failure_definition: nonempty str
      source: "field" | "simulation"
      training_engine_ids: nonempty list of unique engine IDs
      model_frozen_before_calibration: true
      prediction: {engine_id, operating_hours, overhaul_id, remaining_hours,
                   model_id, domain_id, failure_definition, source}
      calibration_outcomes: >=19 independent outcome records
      test_outcomes: >=10 independent outcome records
    Each outcome contains engine_id, operating_hours, remaining_hours (the
    PRE-outcome model prediction), failure_hours, failure_observed: true,
    predicted_before_failure: true, and matching metadata fields model_id,
    domain_id, failure_definition, source. Actual RUL = failure_hours minus
    operating_hours. Censored records are not failure outcomes. Every engine
    contributes exactly one record. Training/calibration/test/current IDs must
    all be disjoint. Supplied provenance is not independently verified.

    A fixed 90% split-conformal absolute-residual interval uses the finite-sample
    ceil((n+1)*0.90) order statistic from CALIBRATION only. Untouched test outcomes
    gate gross miscalibration (>=80% empirical coverage). Cohort independence
    and exchangeability cannot be proved here. At most 10,000 records per group
    are accepted. Current prediction must match the latest history summary.

    Output: {status, reasons:[str], estimate_hours:number|null,
             interval_hours:[lower,upper]|null, uncertainty_kind:str}.
    Unmet gates mean NO numerical RUL. Synthetic evidence can ONLY yield
    simulation_only, never field validation. No health-score extrapolation or
    invented individual-failure probability is used.
    """
    reasons = []

    def unavailable():
        return {"status": "insufficient_evidence", "reasons": list(dict.fromkeys(reasons)),
                "estimate_hours": None, "interval_hours": None,
                "uncertainty_kind": "not_quantifiable_from_available_evidence"}

    if not isinstance(history, (list, tuple)) or not 5 <= len(history) <= 10000:
        reasons.append("Need 5–10,000 longitudinal test summaries; a short simulator run cannot establish remaining life.")
        return unavailable()
    if not isinstance(evidence, Mapping):
        reasons.append("No explicit failure definition, independent engine outcomes, locked-model provenance or held-out calibration evidence supplied.")
        return unavailable()
    metadata_keys = ("model_id", "domain_id", "failure_definition", "source")
    for key in ("current_engine_id", "model_id", "domain_id", "failure_definition"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            reasons.append("Missing nonempty evidence." + key)
    if evidence.get("source") not in ("field", "simulation"):
        reasons.append("Evidence source must explicitly be field or simulation.")
    if evidence.get("model_frozen_before_calibration") is not True:
        reasons.append("The model must be frozen before independent calibration and test outcomes are inspected.")
    current_id = evidence.get("current_engine_id")
    training = evidence.get("training_engine_ids")
    if not isinstance(training, (list, tuple)) or not 1 <= len(training) <= 10000 or any(not isinstance(i, str) or not i.strip() for i in training):
        reasons.append("Explicit bounded training engine IDs are required to check leakage.")
        training_ids = set()
    else:
        training_ids = set(training)
        if len(training_ids) != len(training):
            reasons.append("Training engine IDs must be unique.")
    valid_history, test_ids, overhaul_ids = [], set(), set()
    for row in history:
        if not isinstance(row, Mapping):
            reasons.append("History entries must be test-summary objects.")
            continue
        hours, score = _number(row.get("operating_hours")), _number(row.get("health_score"))
        test_id, overhaul_id = row.get("test_id"), row.get("overhaul_id")
        if hours is None or not 0 <= hours <= 1e7 or score is None or not 0 <= score <= 100:
            reasons.append("History requires finite plausible operating hours and health scores.")
            continue
        if not isinstance(test_id, str) or not test_id.strip() or not isinstance(overhaul_id, str) or not overhaul_id.strip():
            reasons.append("History requires explicit test and overhaul IDs.")
            continue
        if test_id in test_ids:
            reasons.append("Repeated test IDs cannot count as independent degradation history.")
        test_ids.add(test_id)
        overhaul_ids.add(overhaul_id)
        if "engine_id" in row and row["engine_id"] != current_id:
            reasons.append("History contains an engine other than the current engine.")
        valid_history.append((hours, score, overhaul_id))
    if len(overhaul_ids) != 1:
        reasons.append("Use history from one explicit overhaul; do not bridge maintenance resets.")
    if len(valid_history) != len(history):
        reasons.append("Every history row must be usable.")
    if valid_history:
        hours, scores = [row[0] for row in valid_history], [row[1] for row in valid_history]
        if any(b <= a for a, b in zip(hours, hours[1:])):
            reasons.append("History operating hours must increase strictly in supplied order.")
        if hours[-1] - hours[0] < 100:
            reasons.append("At least 100 operating hours of longitudinal degradation history are required.")
        xs = [h - hours[0] for h in hours]
        mx, my = sum(xs) / len(xs), sum(scores) / len(scores)
        denominator = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, scores)) / denominator if denominator else 0.0
        declining_steps = sum(b < a for a, b in zip(scores, scores[1:]))
        if scores[0] - scores[-1] < 5 or slope >= 0 or declining_steps < math.ceil(0.6 * (len(scores) - 1)):
            reasons.append("History lacks a sustained measurable degrading trend; a numerical RUL is not supportable.")
        if any(b - a > 10 for a, b in zip(scores, scores[1:])):
            reasons.append("Large health recovery suggests an unrecorded reset or incompatible measurements.")

    prediction = evidence.get("prediction")
    remaining = None
    if not isinstance(prediction, Mapping):
        reasons.append("An externally trained, metadata-matched current-engine prediction is required.")
    else:
        remaining = _number(prediction.get("remaining_hours"))
        operating = _number(prediction.get("operating_hours"))
        if remaining is None or not 0 <= remaining <= 1e7:
            reasons.append("Current remaining-hours prediction must be finite, nonnegative and plausible.")
        if prediction.get("engine_id") != current_id:
            reasons.append("Prediction engine ID does not match the current engine.")
        if any(prediction.get(key) != evidence.get(key) for key in metadata_keys):
            reasons.append("Current prediction must use the same model, domain, failure definition and evidence source.")
        if not valid_history or operating is None or abs(operating - valid_history[-1][0]) > 1e-6 or prediction.get("overhaul_id") != valid_history[-1][2]:
            reasons.append("Current prediction must align with the latest history hours and overhaul.")

    groups = {}
    for group, minimum in (("calibration_outcomes", 19), ("test_outcomes", 10)):
        records, parsed, ids = evidence.get(group), [], set()
        if not isinstance(records, (list, tuple)) or not minimum <= len(records) <= 10000:
            reasons.append("%s requires %d–10,000 independent held-out engine failure outcomes." % (group, minimum))
        else:
            for record in records:
                if not isinstance(record, Mapping):
                    reasons.append(group + " contains a non-object record.")
                    continue
                engine_id = record.get("engine_id")
                if not isinstance(engine_id, str) or not engine_id.strip():
                    reasons.append(group + " requires explicit engine IDs.")
                    continue
                if engine_id in ids:
                    reasons.append(group + " repeats an engine; observations from one engine are not independent outcomes.")
                ids.add(engine_id)
                if any(record.get(key) != evidence.get(key) for key in metadata_keys):
                    reasons.append(group + " has incompatible model/domain/failure/source metadata.")
                if record.get("failure_observed") is not True or record.get("predicted_before_failure") is not True:
                    reasons.append(group + " requires observed failures and predictions made before outcome inspection; censoring is not a failure.")
                observed = _number(record.get("operating_hours"))
                failed = _number(record.get("failure_hours"))
                predicted = _number(record.get("remaining_hours"))
                if observed is None or failed is None or predicted is None or not 0 <= observed < failed <= 1e7 or not 0 <= predicted <= 1e7:
                    reasons.append(group + " contains invalid observation, failure or predicted hours.")
                    continue
                parsed.append((predicted, failed - observed))
        groups[group] = (ids, parsed)
    cal_ids, calibration = groups["calibration_outcomes"]
    held_ids, held_out = groups["test_outcomes"]
    if training_ids & cal_ids or training_ids & held_ids or cal_ids & held_ids:
        reasons.append("Training, calibration and test engine IDs must be disjoint.")
    if isinstance(current_id, str) and current_id in training_ids | cal_ids | held_ids:
        reasons.append("The current engine must be excluded from all training, calibration and held-out outcome cohorts.")
    if reasons:
        return unavailable()
    residuals = sorted(abs(actual - predicted) for predicted, actual in calibration)
    rank = math.ceil((len(residuals) + 1) * 0.90)
    if rank > len(residuals):
        reasons.append("Too few calibration outcomes for a finite 90% conformal interval.")
        return unavailable()
    radius = residuals[rank - 1]
    covered = sum(abs(actual - predicted) <= radius for predicted, actual in held_out)
    if covered / len(held_out) < 0.80:
        reasons.append("Untouched test outcomes fail the gross coverage gate (80%); recalibrate independently before reporting numerical RUL.")
        return unavailable()
    synthetic = evidence["source"] == "simulation"
    return {
        "status": "simulation_only" if synthetic else "empirically_calibrated_estimate",
        "reasons": [
            "Simulation-only evidence; not field validated." if synthetic else "Estimate calibrated to supplied field outcomes; provenance has not been independently verified.",
            "90%% nominal split-conformal interval from %d independent calibration engines; %d/%d untouched test outcomes covered." % (len(calibration), covered, len(held_out)),
            "Coverage depends on engine independence, exchangeability, fixed model and matching operating domain; it is not an individual failure probability or a safety guarantee.",
        ],
        "estimate_hours": remaining,
        "interval_hours": [max(0.0, remaining - radius), remaining + radius],
        "uncertainty_kind": "split_conformal_absolute_residual_90pct" + ("_simulation_only" if synthetic else "_conditional_on_exchangeability"),
    }
