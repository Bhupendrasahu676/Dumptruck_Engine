"""Streaming, evidence-based audit of the unified synthetic training CSV.

The auditor checks exported data contracts, not the validity of simulated physics,
diagnoses or lifetime predictions. Sidecars default to ``<csv stem>.schema.json``
and ``<csv stem>.manifest.json``. Only bounded counters/examples and per-episode
summaries are retained; CSV bytes are hashed during the single read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


KINDS = (
    "engine_oil_pump_wear", "piston_ring_damage", "turbo_exhaust_fault",
    "coolant_system_failure", "coolant_degradation", "gearbox_seal_wear",
    "gearbox_bearing_wear", "hydraulic_normal_aging", "hydraulic_contamination",
    "bearing_failure", "low_oil", "overheat", "brake_wear", "perception_degraded",
    "sensor_fault", "unrelated_multi_blip",
)
SENSOR_MODES = {"biased", "missing", "frozen", "drifting", "noisy"}
MAX_EXAMPLES = 5
MAX_EPISODE_KEYS = 100_000


def _columns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict) and ("name" in item or "column" in item):
                result.append(item.get("name", item.get("column")))
            else:
                result.extend(_columns(item))
        return result
    if isinstance(value, dict):
        return [column for item in value.values() for column in _columns(item)]
    return []


def _canon(name: str) -> str:
    for prefix in ("metadata.", "meta.", "observation.", "observations.", "readings.", "raw.", "sensor."):
        if name.startswith(prefix):
            return _canon(name[len(prefix):])
    return name


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _first(mapping: dict, *names: str, default=None):
    return next((mapping[name] for name in names if name in mapping), default)


def _leaks(name: str) -> bool:
    """Do not mistake measurement names or missing/valid masks for truth."""
    name = _canon(name).lower()
    if name.startswith(("fault", "effect_level.", "truth.", "expected.", "diagnostic", "label")):
        return True
    parts = re.split(r"[.\[\]]", name)
    forbidden = {
        "id", "engine_id", "truck_id", "episode_id", "journey_id", "base_trajectory_id",
        "timestamp", "timestamp_s", "datetime", "time", "time_s", "elapsed", "elapsed_s", "elapsed_seconds", "elapsed_time_s", "step", "step_index", "sample_index",
        "scenario", "scenario_name", "payload_class", "start_condition", "split", "status",
        "quality", "sensor_mode", "active_faults", "event_phase", "event_phases",
        "sensor_quality_json", "active_faults_json", "event_phases_json", "event_occurrences_json", "latent_wear", "latent_wear_json",
        "mode", "cycle_phase", "drive_cycle", "coverage_group", "primary_class", "probable_faults_json", "possible_causes_json", "mitigations_json",
        "health", "overall_health", "rul", "rul_hours", "confidence", "hypotheses",
        "possible_cause", "root_cause", "root_cause_confirmed", "mitigation", "recommended_action",
    }
    return any(part in forbidden or part.endswith(("_id", "_status", "_health", "_rul", "_confidence"))
               or bool(re.search(r"(^|_)(health|rul|status)($|_)", part))
               or any(token in part for token in ("fault_truth", "fault_label", "fault_level", "effect_level", "root_cause", "diagnostic_cause", "diagnostic_action"))
               for part in parts)


def _plans(manifest: dict) -> dict[str, dict]:
    entries = _first(manifest, "episodes", "episode_configs", "episode_plans", "plans", "episode_manifest", default=[])
    if isinstance(entries, dict):
        entries = [{"episode_id": key, **value} for key, value in entries.items() if isinstance(value, dict)]
    result = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        config = _first(entry, "config", "configuration", "episode_config", "simulation_config", default={})
        if isinstance(config, dict):
            nested = _first(config, "config", "simulation", default={})
            config = {**config, **(nested if isinstance(nested, dict) else {})}
        else:
            config = {}
        parameters = manifest.get("parameters", {})
        merged = {**(parameters if isinstance(parameters, dict) else {}), **entry, **config}
        episode = _first(merged, "episode_id", "journey_id", "id")
        if episode is not None:
            result[str(episode)] = merged
    return result


def _expected_count(manifest: dict, schema: dict, filename: str) -> int | None:
    def count(value):
        if isinstance(value, dict):
            value = _first(value, "row_count", "rows", "total_rows", "count", "num_rows")
        numeric = _number(value)
        return int(numeric) if numeric is not None and numeric >= 0 and numeric.is_integer() else None
    for source in (manifest, schema):
        for key in ("row_count", "total_rows", "rows", "expected_rows"):
            value = count(source.get(key))
            if value is not None:
                return value
        for key in ("counts", "table_counts", "tables", "artifacts", "files"):
            table = source.get(key, {})
            if isinstance(table, dict):
                for name in (filename, "unified_csv", "training_csv", "dataset", "csv", "rows", "total_rows"):
                    value = count(table.get(name))
                    if value is not None:
                        return value
            elif isinstance(table, list):
                for item in table:
                    if isinstance(item, dict) and _first(item, "name", "path", "file", "filename") == filename:
                        value = count(item)
                        if value is not None:
                            return value
    return None


def _memberships(manifest: dict) -> dict[str, str]:
    memberships = _first(manifest, "split_membership", "engine_splits", "truck_splits", "splits", default={})
    result = {}
    if isinstance(memberships, dict):
        for key, value in memberships.items():
            if isinstance(value, str):
                result[str(key)] = value
            elif isinstance(value, list):
                for engine in value:
                    engine = _first(engine, "engine_id", "truck_id", "id") if isinstance(engine, dict) else engine
                    if engine is not None:
                        # Retain overlapping declaration evidence, rather than overwrite it.
                        previous = result.get(str(engine))
                        result[str(engine)] = value_split = str(key)
                        if previous and previous != value_split:
                            result[str(engine)] = f"{previous}|{value_split}"
            elif isinstance(value, dict):
                nested = _first(value, "engine_ids", "truck_ids", "engines", "members", default=[])
                for engine in nested:
                    previous = result.get(str(engine))
                    result[str(engine)] = str(key) if previous in (None, str(key)) else f"{previous}|{key}"
    return result


def _mask_source(column: str) -> tuple[str | None, str | None]:
    name = _canon(column)
    for prefix, kind in (("missing_mask.", "missing"), ("availability_mask.", "present"),
                         ("available.", "present"), ("availability.", "present"),
                         ("available_mask.", "present"), ("presence_mask.", "present"),
                         ("present_mask.", "present"), ("valid_mask.", "valid"),
                         ("validity_mask.", "valid"), ("mask.", "valid")):
        if name.startswith(prefix):
            return _canon(name[len(prefix):]), kind
    for suffix, kind in (("__missing", "missing"), ("__available", "present"),
                         ("__available_mask", "valid"),
                         ("__present", "present"), ("__valid", "valid"),
                         ("__missing_mask", "missing"), ("__availability_mask", "present"),
                         ("__valid_mask", "valid")):
        if name.endswith(suffix):
            return _canon(name[:-len(suffix)]), kind
    return None, None


def _time(value: str) -> float | None:
    number = _number(value)
    if number is not None:
        return number
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Naive export timestamps are interpreted consistently as UTC.
        if parsed.tzinfo is None:
            from datetime import timezone
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def _signature(kind: str, values: dict[str, float], previous: dict) -> tuple[bool | None, dict]:
    """Transparent observation-only screening thresholds; no truth-derived prior."""
    load = min(100., max(0., values.get("engine.load_pct", values.get("context.load_pct", 50.))))
    ambient = values.get("context.ambient_c", values.get("ambient_c", 25.))
    stress = min(1., max(0., (values.get("frame.strain_microstrain", 500.) - 500.) / 610.))
    thresholds = {"oil_temp_c": 101 + .05 * load + 8 * stress + max(0, ambient - 25) * .3,
                  "block_temp_c": 111 + 12 * stress + max(0, ambient - 25) * .3,
                  "exhaust_temp_c": 440 + 1.8 * load + 35 * stress,
                  # C-WTVC coolant/pump baseline is a thermal state, not the
                  # engine's load/stress oil-temperature formula. Adding 12 C
                  # per stress unit here hid clear >98 C pump abnormalities
                  # while the block stayed normal. Use the published red pump
                  # boundary with only the simulator's warm ambient allowance.
                  "pump_temp_c": 98 + max(0, ambient - 25) * .12}
    required = {
        "engine_oil_pump_wear": ("engine.oil_pressure_kpa", "engine.block_vibration_mm_s", "engine.oil_condition_pct"),
        "piston_ring_damage": ("engine.block_vibration_mm_s", "engine.oil_temp_c", "engine.oil_condition_pct"),
        "turbo_exhaust_fault": ("engine.exhaust_temp_c", "engine.block_temp_c", "engine.oil_temp_c"),
        "coolant_system_failure": ("engine.block_temp_c", "engine.oil_temp_c", "engine.exhaust_temp_c", "engine.coolant_pump_temp_c"),
        "coolant_degradation": ("engine.coolant_pump_temp_c", "coolant_condition_pct", "engine.block_temp_c"),
        "gearbox_seal_wear": ("transmission.oil_pressure_kpa", "transmission.vibration_mm_s", "transmission.oil_condition_pct"),
        "gearbox_bearing_wear": ("transmission.oil_pressure_kpa", "transmission.vibration_mm_s", "transmission.oil_condition_pct"),
    }
    needed = required.get(kind)
    if needed is None or any(name not in values for name in needed):
        return None, thresholds
    v = values
    if kind == "engine_oil_pump_wear":
        return v[needed[0]] < 440 and v[needed[1]] > 6.3 and v[needed[2]] >= 90, thresholds
    if kind == "piston_ring_damage":
        egts = [value for name, value in v.items() if name.startswith("engine.exhaust_temperatures_c.")]
        spread = v.get("engine.egt_spread_c", max(egts) - min(egts) if len(egts) >= 2 else None)
        if spread is None:
            return None, thresholds
        return v[needed[0]] > 6.3 and v[needed[1]] > thresholds["oil_temp_c"] and v[needed[2]] < 90 and spread > 35, thresholds
    if kind == "turbo_exhaust_fault":
        return v[needed[0]] > thresholds["exhaust_temp_c"] and v[needed[1]] <= thresholds["block_temp_c"] and v[needed[2]] <= thresholds["oil_temp_c"], thresholds
    if kind == "coolant_system_failure":
        return all(v[name] > thresholds[threshold] for name, threshold in zip(needed, ("block_temp_c", "oil_temp_c", "exhaust_temp_c", "pump_temp_c"))), thresholds
    if kind == "coolant_degradation":
        return v[needed[0]] > thresholds["pump_temp_c"] and v[needed[1]] < 90 and v[needed[2]] <= thresholds["block_temp_c"], thresholds
    normal_condition = v[needed[2]] >= 90
    return v[needed[0]] < 320 and v[needed[1]] > 6.3 and (normal_condition if kind == "gearbox_seal_wear" else not normal_condition), thresholds


def _report_markdown(audit: dict) -> str:
    lines = ["# Training CSV audit", "", f"**Result: {audit['status']}** — {audit['summary']['rows']} rows, "
             f"{audit['summary']['episodes']} episodes.", "", "| Check | Status | Evidence |", "|---|---|---|"]
    for name, check in audit["checks"].items():
        evidence = json.dumps(check["evidence"], ensure_ascii=False, sort_keys=True)
        if len(evidence) > 900:
            evidence = evidence[:897] + "... (see JSON)"
        lines.append(f"| {name} | {check['status']} | {evidence.replace('|', '/')} |")
    lines += ["", "## Validation limits", "", *[f"- {item}: **NOT validated**." for item in audit["validation_limits"]],
              "", "Matrix match rates describe observed synthetic readings, not diagnostic accuracy. "
              "Incipient, cold-start, mixed-fault, missing-sensor and residual-damage states may only partially match a developed signature.", ""]
    return "\n".join(lines)


def audit_training_csv(csv_path, schema_path=None, manifest_path=None, output_prefix=None) -> dict:
    """Audit once, return a JSON-serializable report and optionally publish reports.

    Invalid rows are evidence-bearing failures, not exceptions. Unreadable CSVs
    raise the underlying I/O error; malformed/missing sidecars are reported.
    """
    path = Path(csv_path)
    schema_file = Path(schema_path) if schema_path else path.with_suffix(".schema.json")
    manifest_file = Path(manifest_path) if manifest_path else path.with_suffix(".manifest.json")
    checks: dict[str, dict] = {}
    errors: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)

    def bad(check, evidence):
        errors[check] += 1
        if len(examples[check]) < MAX_EXAMPLES:
            examples[check].append(evidence)

    def result(name, evaluated=True, evidence=None, partial=False):
        evidence = {} if evidence is None else dict(evidence)
        if errors[name]:
            evidence.update(violations=errors[name], examples=examples[name])
        checks[name] = {"status": "fail" if errors[name] else "not_evaluated" if not evaluated else "partial" if partial else "pass",
                        "evidence": evidence}

    def sidecar(file, name):
        if not file.exists():
            result(name, False, {"reason": "sidecar absent"})
            return {}
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("sidecar must be a JSON object")
            result(name, evidence={"present": True})
            return value
        except (ValueError, OSError) as error:
            bad(name, {"reason": str(error)[:200]})
            result(name)
            return {}

    schema, manifest = sidecar(schema_file, "schema_sidecar"), sidecar(manifest_file, "manifest_sidecar")
    inputs = _columns(schema.get("input_columns"))
    declared_targets = _columns(schema.get("target_columns"))
    plans, declared_splits = _plans(manifest), _memberships(manifest)
    feature_defs = _first(schema, "feature_definitions", "featuredefs", "features", default={})
    feature_defs = feature_defs if isinstance(feature_defs, dict) else {}
    expected_count = _expected_count(manifest, schema, path.name)
    parameters = manifest.get("parameters", {})
    settings = {**(parameters if isinstance(parameters, dict) else {}), **manifest}
    digest = hashlib.sha256()

    def lines(handle):
        first = True
        for line in handle:
            digest.update(line)
            yield line.decode("utf-8-sig" if first else "utf-8")
            first = False

    row_count, good_rows, missing_inputs = 0, 0, 0
    documented_missing: Counter = Counter()
    inferred_source_checks = 0
    episodes: dict[str, dict] = {}
    engines: dict[str, set] = defaultdict(set)
    trucks: dict[str, set] = defaultdict(set)
    split_classes: dict[str, Counter] = defaultdict(Counter)
    positives: Counter = Counter()
    contexts = {kind: {"scenario": set(), "payload_class": set(), "start_condition": set()} for kind in ("normal", *KINDS)}
    pairs, triples = Counter(), Counter()
    sensor_modes, quality_counts, variants = Counter(), Counter(), defaultdict(set)
    signatures = {kind: Counter() for kind in KINDS}
    signature_ranges: dict[str, list[float]] = {}
    recovery: Counter = Counter()
    persistent_wear: Counter = Counter()
    delayed_only: Counter = Counter()
    planned_pairs, planned_triples = Counter(), Counter()
    consistency_seen = set()
    target_seen = Counter()
    normal_rows, mask_values = 0, 0
    with path.open("rb") as handle:
        reader = csv.reader(lines(handle), strict=True)
        try:
            header = next(reader, [])
        except (csv.Error, UnicodeError) as error:
            header = []
            bad("file_structure", {"reason": str(error)[:200]})
        duplicate_headers = [name for name, count in Counter(header).items() if count > 1]
        if not header or any(not name.strip() for name in header) or duplicate_headers:
            bad("file_structure", {"empty_header": not header, "duplicate_headers": duplicate_headers[:MAX_EXAMPLES], "blank_header": any(not name.strip() for name in header)})
        index = {name: i for i, name in enumerate(header)}
        canonical = {_canon(name): name for name in header}

        def find(*names):
            return next((canonical[name] for name in names if name in canonical), None)

        episode_col = find("episode_id", "journey_id")
        engine_col = find("engine_id", "truck_id", "virtual_engine_id")
        truck_col = find("truck_id")
        split_col = find("split", "dataset_split")
        step_col = find("step", "step_index", "sample_index")
        elapsed_col = find("elapsed_s", "elapsed_seconds", "elapsed_time_s")
        timestamp_col = find("timestamp", "datetime", "timestamp_s", "time_s")
        context_cols = {key: find(key) for key in ("scenario", "payload_class", "start_condition")}
        fixed_cols = [column for column in (engine_col, split_col, find("truck_id"), find("base_trajectory_id"),
                      find("payload_kg", "payload.mass_kg", "payload.mass_t", "payload_tonnes"), find("primary_class"), find("coverage_group"), *context_cols.values()) if column]
        targets = declared_targets or [name for name in header if _canon(name).startswith("fault_label.")]
        classes = [_canon(name).split("fault_label.", 1)[-1] for name in targets]
        for kind in classes:
            contexts.setdefault(kind, {key: set() for key in context_cols})
            signatures.setdefault(kind, Counter())
        groups = ("input_columns", "target_columns", "metadata_columns", "label_columns", "diagnostic_columns")
        required = set(column for group in groups for column in _columns(schema.get(group)))
        required.update(_columns(schema.get("columns")))
        absent = sorted(required - set(header))
        if absent:
            bad("file_structure", {"missing_declared_columns": absent[:MAX_EXAMPLES], "missing_count": len(absent)})
        declared_all = _columns(schema.get("columns"))
        extra = sorted(set(header) - set(declared_all)) if declared_all else []
        if extra:
            bad("file_structure", {"unexpected_columns": extra[:MAX_EXAMPLES], "unexpected_count": len(extra)})
        if schema:
            if not inputs:
                bad("schema_contract", {"reason": "input_columns missing or empty"})
            if len(declared_targets) != 16 or set(classes) != set(KINDS):
                bad("schema_contract", {"reason": "expected 16 catalogue fault_label targets", "target_count": len(declared_targets)})
            if len(inputs) != len(set(inputs)):
                bad("schema_contract", {"reason": "duplicate input declaration"})
            overlap = set(inputs) & set(column for group in groups[1:] for column in _columns(schema.get(group)))
            if overlap:
                bad("input_leakage", {"overlap_with_noninputs": sorted(overlap)[:MAX_EXAMPLES]})
        for column in inputs:
            if _leaks(column):
                bad("input_leakage", {"column": column})
        if not episode_col or not (step_col or elapsed_col):
            bad("unique_keys", {"reason": "episode_id and step or elapsed_s required"})
        masks = {column: _mask_source(column) for column in header if "mask" in column.lower() or _mask_source(column)[0]}
        masks = {column: item for column, item in masks.items() if item[0] or "mask" in column.lower()}
        missing_markers = defaultdict(list)
        for column, (source, mode) in masks.items():
            if source:
                missing_markers[source].append((column, mode))
        quality_json_col = find("sensor_quality_json", "quality_json", "sensor_quality")
        active_json_col = find("active_faults_json")
        wear_json_col = find("latent_wear_json")
        normal_col = find("is_normal")
        quality_cols = [name for name in header if name.startswith(("quality.", "metadata.quality.", "status."))]
        quality_sources = {_canon(name.split("quality.", 1)[-1].removeprefix("status.")): name for name in quality_cols}
        raw_indices = [(name, _canon(name), position) for name, position in index.items()
                       if _canon(name).startswith(("engine.", "transmission.", "suspension.", "frame.", "context.")) or _canon(name) in ("coolant_condition_pct", "ambient_c")]
        input_indices = [(name, index[name]) for name in inputs if name in index]
        condition_indices = [(name, position) for name, position in index.items()
                             if _canon(name).endswith("condition_pct") and not name.startswith(("quality.", "status."))
                             and not any(token in name for token in ("mask", "rate", "residual", "slope", "std", "mean"))]
        target_indices = [(kind, index[column]) for kind, column in zip(classes, targets) if column in index]
        level_cols = {kind: find(f"fault_level.{kind}", f"truth.fault_levels.{kind}", f"fault_levels.{kind}") for kind in classes}
        effect_cols = {kind: find(f"effect_level.{kind}", f"truth.effect_levels.{kind}", f"effect_levels.{kind}") for kind in classes}
        meta_variant_cols = [name for name in header if any(token in _canon(name) for token in ("noise_scale", "initial_wear", "ambient_c", "engine_seed", "wear_time_scale"))]
        mode_cols = [name for name in header if _canon(name).endswith("sensor_mode")]
        try:
            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    bad("file_structure", {"row": row_count + 1, "expected_width": len(header), "actual_width": len(row)})
                    continue
                good_rows += 1

                def get(column, default=""):
                    return row[index[column]] if column in index else default

                episode = get(episode_col)
                quality_truth = {}
                wear_truth = {}
                if wear_json_col:
                    try:
                        wear_truth = json.loads(get(wear_json_col))
                        if not isinstance(wear_truth, dict):
                            raise ValueError("wear truth must be an object")
                    except (ValueError, TypeError):
                        wear_truth = {}
                        bad("recovery_with_damage", {"row": row_count + 1, "reason": "invalid latent wear JSON"})
                if quality_json_col:
                    try:
                        quality_truth = json.loads(get(quality_json_col))
                        if not isinstance(quality_truth, dict):
                            raise ValueError("quality truth must be an object")
                    except (ValueError, TypeError):
                        quality_truth = {}
                        bad("sensor_quality_truth", {"row": row_count + 1, "reason": "invalid sensor quality JSON"})
                engine, split = get(engine_col), get(split_col)
                if episode_col and not episode:
                    bad("unique_keys", {"row": row_count + 1, "reason": "empty episode key"})
                plan = plans.get(episode, {})
                episode_context = {key: get(column) or str(plan.get(key, "")) for key, column in context_cols.items()}
                if episode not in episodes:
                    episodes[episode] = {"rows": 0, "keys": set(), "overflow": False, "fixed": {column: get(column) for column in fixed_cols},
                                         "first_t": None, "last_t": None, "last_timestamp": None, "first_timestamp": None,
                                         "dt_min": None, "dt_max": None, "last_step": None, "first_step": None,
                                         "previous_unavailable": set(),
                                         "previous_missing_readings": set(),
                                         "classes": {kind: {"min": 1, "max": 0, "positives": 0} for kind in classes},
                                         "hydraulic": {kind: [0, 0., 0., 0., 0., 0.] for kind in ("hydraulic_normal_aging", "hydraulic_contamination")}}
                state = episodes[episode]
                state["rows"] += 1
                for column in fixed_cols:
                    if state["fixed"][column] != get(column):
                        bad("episode_invariants", {"episode": episode, "column": column, "row": row_count + 1})
                for key, column in (("engine_id", engine_col), ("truck_id", find("truck_id")), ("split", split_col), *context_cols.items()):
                    expected = plan.get(key)
                    if column and expected is not None and get(column) != str(expected):
                        bad("episode_invariants", {"episode": episode, "column": column, "reason": "manifest mismatch"})
                if truck_col and split_col:
                    trucks[get(truck_col)].add(split)
                if engine_col and split_col:
                    if not engine or not split:
                        bad("engine_disjoint_split", {"row": row_count + 1, "reason": "empty engine/split"})
                    engines[engine].add(split)
                    if engine in declared_splits and declared_splits[engine] != split:
                        bad("engine_disjoint_split", {"engine": engine, "reason": "manifest membership mismatch"})
                elapsed = _number(get(elapsed_col))
                step = _number(get(step_col))
                key = get(step_col) if step_col else get(elapsed_col)
                key_number = _number(key)
                if (step_col or elapsed_col) and (key_number is None or (step_col and not key_number.is_integer())):
                    bad("unique_keys", {"row": row_count + 1, "reason": "invalid step/elapsed key"})
                normalized_key = key_number if key_number is not None else key
                if normalized_key in state["keys"]:
                    bad("unique_keys", {"episode": episode, "key": key})
                if len(state["keys"]) < MAX_EPISODE_KEYS:
                    state["keys"].add(normalized_key)
                else:
                    state["overflow"] = True
                duration = _number(_first(plan, "duration_s", "journey_duration_s", default=_first(settings, "duration_s", "journey_duration_s", default=1800)))
                sample_count = _number(_first(plan, "sample_count", "samples", "steps", "num_steps", "samples_per_episode", "samples_per_journey", default=_first(settings, "samples_per_episode", "samples_per_journey", "steps_per_episode", default=171)))
                timestep = _number(_first(plan, "timestep_s", "time_step_s", "cadence_s", default=_first(settings, "timestep_s", "cadence_s")))
                if timestep is None and duration is not None and sample_count is not None and sample_count > 1:
                    timestep = duration / (sample_count - 1)
                tolerance = max(.002, (timestep or 1) * .001)
                if elapsed_col:
                    if elapsed is None or elapsed < -tolerance:
                        bad("temporal_order", {"episode": episode, "row": row_count + 1, "reason": "invalid elapsed time"})
                    elif state["last_t"] is not None:
                        dt = elapsed - state["last_t"]
                        if dt <= 0:
                            bad("temporal_order", {"episode": episode, "reason": "elapsed time not increasing"})
                        if timestep and abs(dt - timestep) > tolerance:
                            bad("cadence", {"episode": episode, "delta_s": dt, "expected_s": timestep})
                        state["dt_min"] = dt if state["dt_min"] is None else min(dt, state["dt_min"])
                        state["dt_max"] = dt if state["dt_max"] is None else max(dt, state["dt_max"])
                    if elapsed is not None:
                        if state["first_t"] is None:
                            state["first_t"] = elapsed
                        state["last_t"] = elapsed
                if step_col and step is not None:
                    if state["last_step"] is not None and step != state["last_step"] + 1:
                        bad("temporal_order", {"episode": episode, "reason": "step not consecutive"})
                    state["first_step"] = step if state["first_step"] is None else state["first_step"]
                    state["last_step"] = step
                stamp = _time(get(timestamp_col)) if timestamp_col else None
                if timestamp_col:
                    if stamp is None or (state["last_timestamp"] is not None and stamp <= state["last_timestamp"]):
                        bad("temporal_order", {"episode": episode, "reason": "invalid or nonincreasing timestamp"})
                    if stamp is not None:
                        if state["first_timestamp"] is None:
                            state["first_timestamp"] = stamp
                        if timestep and state["last_timestamp"] is not None and abs(stamp - state["last_timestamp"] - timestep) > max(1.01, tolerance):
                            bad("cadence", {"episode": episode, "reason": "timestamp cadence mismatch"})
                        if elapsed is not None and state["first_t"] is not None and abs((stamp - state["first_timestamp"]) - (elapsed - state["first_t"])) > 1.01:
                            bad("cadence", {"episode": episode, "reason": "timestamp/elapsed divergence"})
                        state["last_timestamp"] = stamp
                for column, position in input_indices:
                    value = row[position]
                    if not value.strip():
                        missing_inputs += 1
                        markers = missing_markers[_canon(column)]
                        explicit = any((_number(get(mask)) == (1 if mode == "missing" else 0)) for mask, mode in markers)
                        quality = get(quality_sources.get(_canon(column))) or quality_truth.get(_canon(column))
                        definition = feature_defs.get(column, {})
                        missing_policy = schema.get("missingness_handling", {})
                        documented = isinstance(definition, dict) and bool(definition.get("definition")) and isinstance(missing_policy, dict) and "blank" in str(missing_policy.get("csv_representation", "")).lower()
                        canonical_name = _canon(column)
                        base = canonical_name.split("__", 1)[0]
                        source_marked = any(_number(get(mask)) == (1 if mode == "missing" else 0) for mask, mode in missing_markers[base])
                        unavailable_reason = None
                        if documented and canonical_name.startswith("leadlag."):
                            unavailable_reason = "documented_onset_association_unavailable"
                        elif documented and "__" in canonical_name and any(token in canonical_name for token in ("__rate", "__residual", "__mean", "__std")):
                            if source_marked or base in state["previous_unavailable"]:
                                unavailable_reason = "source_or_recent_history_unavailable"
                            elif state["rows"] == 1 and (canonical_name.endswith("__rate") or "__residual_slope" in canonical_name):
                                unavailable_reason = "episode_initial_history_unavailable"
                        elif documented and canonical_name == "engine.egt_spread_c":
                            egt_columns = [name for source, name in canonical.items() if source.startswith("engine.exhaust_temperatures_c.")]
                            if egt_columns and any(not get(name).strip() for name in egt_columns):
                                unavailable_reason = "required_egt_channel_missing"
                        if unavailable_reason:
                            explicit = True
                            documented_missing[unavailable_reason] += 1
                        if not explicit and quality != "missing":
                            bad("numeric_inputs", {"row": row_count + 1, "column": column, "reason": "missing without explicit marker"})
                    elif _number(value) is None:
                        bad("numeric_inputs", {"row": row_count + 1, "column": column, "reason": "nonnumeric or nonfinite"})
                for column, (source, mode) in masks.items():
                    mask = _number(get(column))
                    mask_values += 1
                    if mask not in (0, 1):
                        bad("availability_masks", {"row": row_count + 1, "column": column, "reason": "mask not binary"})
                    source_col = canonical.get(source)
                    # Validity can be inferred from behavior; zero need not mean absent.
                    if source_col and mode in ("missing", "present") and mask in (0, 1):
                        present = bool(get(source_col).strip()) and _number(get(source_col)) is not None
                        expected = int(not present) if mode == "missing" else int(present)
                        if mask != expected:
                            bad("availability_masks", {"row": row_count + 1, "column": column, "reason": "presence/source mismatch"})
                    if source_col and mode == "valid" and mask == 1 and _number(get(source_col)) is None:
                        bad("availability_masks", {"row": row_count + 1, "column": column, "reason": "valid mask requires a finite source"})
                for source, quality in quality_truth.items():
                    source_col = canonical.get(_canon(source))
                    quality_counts[str(quality)] += 1
                    if quality not in ("good", *SENSOR_MODES):
                        bad("sensor_quality_truth", {"row": row_count + 1, "source": source, "reason": "unknown quality mode"})
                    if source_col and (quality == "missing") != (not get(source_col).strip()):
                        bad("sensor_quality_truth", {"row": row_count + 1, "source": source, "reason": "quality missingness disagrees with source presence"})
                for column, definition in feature_defs.items():
                    if not isinstance(definition, dict) or column not in index:
                        continue
                    sources = _columns(_first(definition, "required_sources", "required_source_columns", default=[]))
                    valid_column = _first(definition, "availability_mask", "validity_mask", "valid_mask")
                    if sources and valid_column in index and _number(get(valid_column)) == 1:
                        if any(canonical.get(_canon(source)) is None or _number(get(canonical.get(_canon(source)))) is None for source in sources):
                            bad("required_source_availability", {"row": row_count + 1, "feature": column, "reason": "declared valid with absent required source"})
                for column, position in input_indices:
                    name = _canon(column)
                    if _number(row[position]) is None:
                        continue
                    # Current-reading residuals and first differences require a
                    # current source. Rolling summaries/slopes and onset lags may
                    # legitimately survive a missing current reading.
                    if name.endswith(("__rate", "__residual")):
                        source = name.split("__", 1)[0]
                        source_col = canonical.get(source)
                        inferred_source_checks += 1
                        if not source_col or _number(get(source_col)) is None:
                            bad("required_source_availability", {"row": row_count + 1, "feature": column, "reason": "finite derived feature without a finite current source"})
                        elif name.endswith("__rate") and (state["rows"] == 1 or source in state["previous_missing_readings"]):
                            bad("required_source_availability", {"row": row_count + 1, "feature": column, "reason": "rate lacks finite previous source/history"})
                    elif name == "engine.egt_spread_c":
                        sources = [column_name for source_name, column_name in canonical.items() if source_name.startswith("engine.exhaust_temperatures_c.")]
                        if sources:
                            inferred_source_checks += 1
                            if any(_number(get(source)) is None for source in sources):
                                bad("required_source_availability", {"row": row_count + 1, "feature": column, "reason": "EGT spread lacks required cylinder reading"})
                for column, position in condition_indices:
                    number = _number(row[position])
                    if number is not None and not 0 <= number <= 100:
                        bad("condition_bounds", {"row": row_count + 1, "column": column, "value": number})
                    elif row[position].strip() and number is None:
                        bad("condition_bounds", {"row": row_count + 1, "column": column, "reason": "nonfinite condition"})
                active = []
                for kind, position in target_indices:
                    label = _number(row[position])
                    target_seen[kind] += 1
                    if label not in (0, 1):
                        bad("label_consistency", {"row": row_count + 1, "class": kind, "reason": "target not binary"})
                        continue
                    summary = state["classes"][kind]
                    summary["min"], summary["max"] = min(summary["min"], label), max(summary["max"], label)
                    summary["positives"] += int(label)
                    if label:
                        active.append(kind)
                        positives[kind] += 1
                        for context, value in episode_context.items():
                            if value:
                                contexts[kind][context].add(value)
                    effect_col, level_col = effect_cols[kind], level_cols[kind]
                    effect = _number(get(effect_col)) if effect_col else None
                    level = _number(get(level_col)) if level_col else None
                    for name, column, value in (("effect", effect_col, effect), ("schedule", level_col, level)):
                        if column and (value is None or not 0 <= value <= 1):
                            bad("label_consistency", {"row": row_count + 1, "class": kind, "reason": f"{name} outside finite 0..1"})
                    if effect is not None:
                        consistency_seen.add(kind)
                        if label != int(effect > .001):
                            bad("label_consistency", {"row": row_count + 1, "class": kind, "reason": "label differs from effect > 0.001"})
                    if level is not None and effect is not None and level <= .001 and effect > .001:
                        recovery[kind] += 1
                        if wear_json_col:
                            if (_number(wear_truth.get(kind)) or 0) > 0:
                                persistent_wear[kind] += 1
                            else:
                                delayed_only[kind] += 1
                if normal_col and _number(get(normal_col)) != int(not active):
                    bad("label_consistency", {"row": row_count + 1, "reason": "is_normal differs from fault labels"})
                if active_json_col:
                    try:
                        scheduled_active = json.loads(get(active_json_col))
                        if not isinstance(scheduled_active, list):
                            raise ValueError("active_faults must be a list")
                        if all(level_cols.values()):
                            expected_active = {kind for kind in classes if (_number(get(level_cols[kind])) or 0) > 0}
                            if set(scheduled_active) != expected_active:
                                bad("label_consistency", {"row": row_count + 1, "reason": "active_faults differs from nonzero schedule truth (not effect labels)"})
                    except (ValueError, TypeError):
                        bad("label_consistency", {"row": row_count + 1, "reason": "invalid active_faults JSON"})
                if not active and len(target_indices) == len(classes) and classes:
                    normal_rows += 1
                    split_classes[split]["normal"] += 1
                    for context, value in episode_context.items():
                        if value:
                            contexts["normal"][context].add(value)
                for kind in active:
                    split_classes[split][kind] += 1
                pairs.update(" + ".join(items) for items in itertools.combinations(sorted(active), 2))
                triples.update(" + ".join(items) for items in itertools.combinations(sorted(active), 3))
                values = {name: value for _, name, position in raw_indices if (value := _number(row[position])) is not None}
                for kind in active:
                    match, thresholds = _signature(kind, values, state)
                    signatures[kind]["positive_rows"] += 1
                    if match is not None:
                        signatures[kind]["evaluated"] += 1
                        signatures[kind]["matched"] += int(match)
                        effect = _number(get(effect_cols[kind]))
                        if effect is not None and effect >= .5:
                            signatures[kind]["developed_rows"] += 1
                            signatures[kind]["developed_matches"] += int(match)
                        for key, value in thresholds.items():
                            bounds = signature_ranges.setdefault(key, [value, value])
                            bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
                    if kind in state["hydraulic"] and elapsed is not None:
                        condition = values.get("suspension.hydraulic_oil_condition_pct")
                        vibration = values.get("suspension.actuator_vibration_mm_s")
                        if condition is not None and vibration is not None:
                            stats = state["hydraulic"][kind]
                            for j, increment in enumerate((1, elapsed, condition, elapsed * elapsed, elapsed * condition, vibration)):
                                stats[j] += increment
                for column in quality_cols:
                    value = get(column)
                    if value:
                        quality_counts[value] += 1
                for column in mode_cols:
                    value = get(column)
                    if value:
                        sensor_modes[value] += 1
                for column in meta_variant_cols:
                    value = get(column)
                    if value and len(variants[_canon(column)]) < 100:
                        variants[_canon(column)].add(value)
                state["previous_unavailable"] = {source for source, markers in missing_markers.items()
                                                   if any(_number(get(mask)) == (1 if mode == "missing" else 0) for mask, mode in markers)}
                state["previous_missing_readings"] = {source for source, column in canonical.items()
                                                       if source in missing_markers and _number(get(column)) is None}
        except (csv.Error, UnicodeError) as error:
            bad("file_structure", {"row": row_count + 1, "reason": str(error)[:200]})
            # Complete the byte digest even if decoding/parsing terminated early.
            for remaining in handle:
                digest.update(remaining)

    for episode, state in episodes.items():
        plan = plans.get(episode, {})
        duration = _number(_first(plan, "duration_s", "journey_duration_s", default=_first(settings, "duration_s", "journey_duration_s", default=1800)))
        timestep = _number(_first(plan, "timestep_s", "time_step_s", "cadence_s", default=_first(settings, "timestep_s", "cadence_s")))
        samples = _number(_first(plan, "sample_count", "samples", "steps", "num_steps", "samples_per_episode", "samples_per_journey", default=_first(settings, "samples_per_episode", "samples_per_journey", "steps_per_episode")))
        if samples is None:
            samples = math.floor(duration / timestep + 1e-6) + 1 if duration is not None and timestep else 171
        if state["rows"] != samples:
            bad("cadence", {"episode": episode, "samples": state["rows"], "expected": samples})
        if elapsed_col:
            tolerance = max(.02, (duration or 1800) * .00001)
            if state["first_t"] is None or abs(state["first_t"]) > tolerance or state["last_t"] is None or (duration is not None and abs(state["last_t"] - duration) > tolerance):
                bad("cadence", {"episode": episode, "first_elapsed_s": state["first_t"], "last_elapsed_s": state["last_t"], "expected_end_s": duration})
        elif timestamp_col and duration is not None:
            span = state["last_timestamp"] - state["first_timestamp"] if state["last_timestamp"] is not None and state["first_timestamp"] is not None else None
            if span is None or abs(span - duration) > 1.01:
                bad("cadence", {"episode": episode, "timestamp_span_s": span, "expected_end_s": duration})
        if step_col and state["first_step"] != 0:
            bad("temporal_order", {"episode": episode, "reason": "steps must start at zero"})
        for kind, stats in state["hydraulic"].items():
            n, sx, sy, sxx, sxy, sv = stats
            denominator = n * sxx - sx * sx
            if n >= 3 and denominator > 0:
                drop_rate = -(n * sxy - sx * sy) / denominator
                high_vibration = sv / n > 6.3
                match = high_vibration and (.0001 <= drop_rate <= .01 if kind == "hydraulic_normal_aging" else drop_rate > .01)
                signatures[kind]["evaluated_episodes"] += 1
                signatures[kind]["matched_episodes"] += int(match)
                signatures[kind]["rate_sum"] += drop_rate
    if plans:
        missing_episodes = set(plans) - set(episodes)
        unexpected_episodes = set(episodes) - set(plans)
        if missing_episodes or unexpected_episodes:
            bad("episode_invariants", {"missing_plans": len(missing_episodes), "unexpected_episodes": len(unexpected_episodes)})
    for engine, splits in engines.items():
        if len(splits) > 1:
            bad("engine_disjoint_split", {"engine": engine, "splits": sorted(splits)})
    for truck, splits in trucks.items():
        if not truck or len(splits) > 1:
            bad("engine_disjoint_split", {"truck": truck, "splits": sorted(splits)})
    for engine, split in declared_splits.items():
        if "|" in split:
            bad("engine_disjoint_split", {"engine": engine, "reason": "overlapping sidecar split membership"})
    expected_splits = set(declared_splits.values()) - {value for value in declared_splits.values() if "|" in value}
    expected_splits.update({"train", "validation", "test"})
    for split in expected_splits if split_col else []:
        missing = [kind for kind in ("normal", *classes) if split_classes[split][kind] == 0]
        if missing:
            bad("split_class_coverage", {"split": split, "missing_classes": missing})
    temporal = {}
    for kind in classes:
        positive_episodes = [state["classes"][kind] for state in episodes.values() if state["classes"][kind]["positives"]]
        changing = sum(state["min"] != state["max"] for state in positive_episodes)
        temporal[kind] = {"positive_episodes": len(positive_episodes), "varying_label_episodes": changing,
                          "static_positive_episodes": len(positive_episodes) - changing}
        if positive_episodes and not changing:
            bad("temporal_class_variation", {"class": kind, "reason": "no positive episode has changing label"})
    context_evidence = {}
    for kind in ("normal", *classes):
        context_evidence[kind] = {key: sorted(values) for key, values in contexts[kind].items()}
        for key, expected in (("scenario", 3), ("payload_class", 5), ("start_condition", 2)):
            if (normal_rows if kind == "normal" else positives[kind]) and contexts[kind][key] and len(contexts[kind][key]) < expected:
                bad("per_class_contexts", {"class": kind, "context": key, "observed": len(contexts[kind][key]), "expected": expected})
    for plan in plans.values():
        for key in ("noise_scale", "initial_wear", "ambient_c", "engine_seed", "wear_time_scale"):
            if key in plan and len(variants[key]) < 100:
                variants[key].add(str(plan[key]))
        events = _first(plan, "events", "fault_events", default=[])
        selected = sorted({event.get("kind", event.get("fault_kind")) for event in events if isinstance(event, dict) and event.get("kind", event.get("fault_kind"))}) if isinstance(events, list) else []
        planned_pairs.update(" + ".join(items) for items in itertools.combinations(selected, 2))
        planned_triples.update(" + ".join(items) for items in itertools.combinations(selected, 3))
        for event in events if isinstance(events, list) else []:
            if isinstance(event, dict) and event.get("sensor_mode"):
                sensor_modes[event["sensor_mode"]] += 1
    if expected_count is not None and row_count != expected_count:
        bad("expected_row_count", {"actual": row_count, "expected": expected_count})
    result("file_structure", evidence={"columns": len(header), "rows": row_count, "well_formed_rows": good_rows})
    result("schema_contract", bool(schema), {"input_count": len(inputs), "target_count": len(targets)})
    units = _first(schema, "units", "input_units", default={})
    units = units if isinstance(units, dict) else {}
    taxonomy = _first(schema, "fault_taxonomy", "taxonomy", default={})
    missing_units = [name for name in inputs if name not in units]
    missing_definitions = [name for name in inputs if name not in feature_defs]
    result("schema_documentation", bool(units or feature_defs or taxonomy), {"inputs_missing_units": len(missing_units), "inputs_missing_feature_definitions": len(missing_definitions), "taxonomy_classes": len(taxonomy) if isinstance(taxonomy, (dict, list)) else 0}, partial=bool(missing_units or missing_definitions or not taxonomy))
    result("input_leakage", bool(inputs), {"checked_inputs": len(inputs)})
    result("numeric_inputs", bool(input_indices), {"missing_input_cells": missing_inputs, "checked_columns": len(input_indices), "documented_unavailable_feature_cells": dict(documented_missing)})
    result("availability_masks", bool(masks), {"mask_columns": len(masks), "checked_values": mask_values, "note": "presence/missing masks agree with source presence; inferred validity may be zero for present readings, but one requires a finite source"})
    result("sensor_quality_truth", bool(quality_json_col or quality_cols), {"row_quality_counts": dict(quality_counts)})
    required_sources_seen = any(isinstance(item, dict) and _first(item, "required_sources", "required_source_columns") and _first(item, "availability_mask", "validity_mask", "valid_mask") in index for item in feature_defs.values())
    result("required_source_availability", required_sources_seen or bool(inferred_source_checks), {"inferred_direct_feature_cells_checked": inferred_source_checks, "explicit_relationships_available": bool(required_sources_seen), "reason": "Direct residual/rate/EGT dependencies are inferred by naming; other relationships require explicit source/mask definitions."})
    result("condition_bounds", bool(condition_indices), {"checked_columns": [name for name, _ in condition_indices]})
    result("label_consistency", bool(target_indices), {"target_classes": len(target_indices), "classes_with_effect_truth": len(consistency_seen), "threshold": .001}, partial=len(consistency_seen) < len(classes))
    result("unique_keys", bool(episode_col and (step_col or elapsed_col)), {"key": [episode_col, step_col or elapsed_col], "bounded_key_limit_per_episode": MAX_EPISODE_KEYS}, partial=any(state["overflow"] for state in episodes.values()))
    result("temporal_order", bool(episodes and (elapsed_col or timestamp_col or step_col)), {"timestamp_available": bool(timestamp_col), "elapsed_available": bool(elapsed_col)})
    result("cadence", bool(episodes and (elapsed_col or timestamp_col)), {"default_contract": "171 samples, elapsed 0..1800 seconds; explicit episode config overrides defaults", "elapsed_rounding_tolerance": "max(0.002 s, 0.1% cadence)", "timestamp_rounding_tolerance_s": 1.01}, partial=not bool(timestamp_col))
    result("expected_row_count", expected_count is not None, {"actual": row_count, "expected": expected_count, "reason": "sidecar row count absent"} if expected_count is None else {"actual": row_count, "expected": expected_count})
    result("episode_invariants", bool(episodes and fixed_cols), {"fixed_columns": fixed_cols, "manifest_plans": len(plans)}, partial=not bool(engine_col and split_col))
    config_fields = ("seed", "engine_seed", "duration_s", "timestep_s", "scenario", "payload_class", "start_condition")
    config_coverage = {field: sum(field in plan for plan in plans.values()) for field in config_fields}
    result("manifest_episode_configs", bool(plans), {"plans": len(plans), "field_coverage": config_coverage, "plans_with_event_lists": sum(isinstance(_first(plan, "events", "fault_events"), list) for plan in plans.values()), "reason": "Recorded config/events are checked for availability; simulation was not replayed."}, partial=any(count < len(plans) for count in config_coverage.values()) or any(not isinstance(_first(plan, "events", "fault_events"), list) for plan in plans.values()))
    result("engine_disjoint_split", bool(engine_col and split_col), {"engines": len(engines), "declared_engines": len(declared_splits)})
    result("split_class_coverage", bool(split_col and target_indices), {"positive_rows": {key: dict(value) for key, value in sorted(split_classes.items())}})
    context_available = all(context_cols.values()) or (bool(plans) and all(any(key in plan for plan in plans.values()) for key in context_cols))
    result("per_class_contexts", context_available and bool(target_indices), {"observed": context_evidence, "required_counts": {"scenario": 3, "payload_class": 5, "start_condition": 2}})
    result("temporal_class_variation", bool(target_indices and episodes), {"per_class": temporal}, partial=any(not item["positive_episodes"] for item in temporal.values()))
    result("recovery_with_damage", bool(consistency_seen and any(level_cols.values())), {"rows_by_class": dict(recovery), "persistent_wear_rows_by_class": dict(persistent_wear), "delayed_effect_without_wear_rows_by_class": dict(delayed_only), "wear_truth_available": bool(wear_json_col), "definition": "schedule <= 0.001 and effect > 0.001; wear attribution requires supplied per-class latent wear; counted, not automatically an inconsistency"}, partial=not bool(wear_json_col))
    frequencies = {kind: {"positive_rows": positives[kind], "rate": positives[kind] / good_rows if good_rows else None} for kind in classes}
    nonzero = [positives[kind] for kind in classes if positives[kind]]
    imbalance = max(nonzero) / min(nonzero) if nonzero else None
    result("class_imbalance", bool(target_indices and good_rows), {"per_class": frequencies, "normal_rows": normal_rows, "largest_to_smallest_nonzero_class": imbalance}, partial=bool(imbalance and imbalance > 3) or len(nonzero) < len(classes))
    result("fault_combinations", bool(target_indices), {"distinct_pairs": len(pairs), "distinct_triples": len(triples), "pair_rows": sum(pairs.values()), "triple_rows": sum(triples.values()), "top_pairs": pairs.most_common(10), "top_triples": triples.most_common(10), "planned_distinct_pairs": len(planned_pairs) if plans else None, "planned_distinct_triples": len(planned_triples) if plans else None, "planned_pair_cooccurrence_episodes": sum(planned_pairs.values()) if plans else None, "planned_triple_episodes": sum(planned_triples.values()) if plans else None}, partial=not bool(pairs and triples))
    result("sensor_modes", bool(sensor_modes or quality_counts), {"metadata_or_event_counts": dict(sensor_modes), "row_quality_counts": dict(quality_counts), "required_modes": sorted(SENSOR_MODES)}, partial=not SENSOR_MODES.issubset(set(sensor_modes) | set(quality_counts)))
    result("quality_variants", bool(variants or quality_counts), {"distinct_variant_values": {key: len(value) for key, value in variants.items()}, "quality_counts": dict(quality_counts)}, partial=not any(len(value) > 1 for value in variants.values()))
    matrix = {}
    for kind, counts in signatures.items():
        if kind in ("hydraulic_normal_aging", "hydraulic_contamination"):
            evaluated = counts["evaluated_episodes"]
            matched = counts["matched_episodes"]
            positive_episode_count = temporal.get(kind, {}).get("positive_episodes", 0)
            matrix[kind] = {"status": "not_evaluated" if not evaluated else "pass" if matched == evaluated == positive_episode_count else "partial", "positive_rows": positives[kind], "positive_episodes": positive_episode_count,
                            "evaluated_episodes": evaluated, "matched_episodes": matched, "match_rate": matched / evaluated if evaluated else None,
                            "mean_condition_drop_pct_per_s": counts["rate_sum"] / evaluated if evaluated else None,
                            "rate_threshold": "0.0001..0.01 pct/s" if kind == "hydraulic_normal_aging" else ">0.01 pct/s", "vibration_threshold_mm_s": 6.3}
        else:
            evaluated, matched = counts["evaluated"], counts["matched"]
            matrix[kind] = {"status": "not_evaluated" if not evaluated else "pass" if matched == evaluated and evaluated == positives[kind] else "partial",
                            "positive_rows": positives[kind], "evaluated_rows": evaluated, "matched_rows": matched,
                            "match_rate": matched / evaluated if evaluated else None, "developed_rows": counts["developed_rows"],
                            "developed_matches": counts["developed_matches"]}
    result("matrix_signatures", any(item["status"] != "not_evaluated" for item in matrix.values()),
           {"per_class": matrix, "observed_context_threshold_ranges": signature_ranges,
            "definition": "observation-only screening, effect >= 0.5 developed subset; never a root-cause validation",
            "fixed_thresholds": {"pump_Po_kpa": "<440", "Vb_mm_s": ">6.3", "normal_condition_pct": ">=90", "low_condition_pct": "<90", "gear_Pt_kpa": "<320", "gear_Vg_mm_s": ">6.3", "EGT_spread_c": ">35"}},
           partial=any(item["status"] != "pass" for item in matrix.values()))

    dataset_sha = digest.hexdigest()
    expected_sha = _first(manifest, "dataset_sha256", "csv_sha256", "sha256")
    if isinstance(expected_sha, dict):
        expected_sha = expected_sha.get(path.name)
    for table_name in ("hashes", "files", "artifacts", "tables"):
        table = manifest.get(table_name, {})
        entry = table.get(path.name, table.get("unified_csv", {})) if isinstance(table, dict) else {}
        if isinstance(entry, str) and len(entry) == 64:
            expected_sha = expected_sha or entry
        elif isinstance(entry, dict):
            expected_sha = expected_sha or _first(entry, "sha256", "hash")
    if expected_sha and str(expected_sha).lower() != dataset_sha:
        bad("dataset_hash", {"reason": "CSV SHA-256 mismatch"})
    result("dataset_hash", bool(expected_sha), {"sha256": dataset_sha, "supplied": bool(expected_sha)})
    provenance = manifest.get("provenance", {})
    source_hashes = _first(manifest, "source_hashes", "source_sha256", "sources", default=_first(provenance, "source_hashes", "source_sha256", default={}) if isinstance(provenance, dict) else {})
    checked_sources, unavailable_sources = 0, []
    if isinstance(source_hashes, list):
        source_hashes = {item.get("path", item.get("name", "")): item for item in source_hashes if isinstance(item, dict)}
    for name, supplied in source_hashes.items() if isinstance(source_hashes, dict) else []:
        expected = _first(supplied, "sha256", "hash") if isinstance(supplied, dict) else supplied
        source_name = supplied.get("path", name) if isinstance(supplied, dict) else name
        if not isinstance(source_name, str) or not source_name:
            unavailable_sources.append(str(name))
            continue
        aliases = {"generator": "generate_training_csv.py", "simulation": "enhanced_simulation.py", "analytics": "enhanced_analytics.py"}
        if source_name in aliases:
            source_name = aliases[source_name]
        elif "." not in str(source_name) and "/" not in str(source_name):
            source_name = str(source_name) + ".py"
        if not expected:
            unavailable_sources.append(str(name))
            continue
        candidates = (manifest_file.parent / source_name, path.parent.parent / source_name, Path(__file__).parent / source_name)
        source = next((item for item in candidates if item.is_file()), None)
        if source is None:
            unavailable_sources.append(str(name))
            continue
        source_digest = hashlib.sha256()
        if source.resolve() != path.resolve():
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    source_digest.update(block)
        actual_source_hash = dataset_sha if source.resolve() == path.resolve() else source_digest.hexdigest()
        checked_sources += 1
        if actual_source_hash != str(expected).lower():
            bad("source_hashes", {"source": Path(str(name)).name, "reason": "SHA-256 mismatch"})
    result("source_hashes", bool(checked_sources), {"checked": checked_sources, "unavailable": [Path(item).name for item in unavailable_sources[:MAX_EXAMPLES]], "unavailable_count": len(unavailable_sources)}, partial=bool(unavailable_sources))
    limits = {"synthetic_physics": "NOT validated", "calibrated_deep_learning": "NOT validated", "RUL": "NOT validated", "real_root_cause": "NOT validated"}
    result("validation_limits", False, {"claims": limits, "reason": "CSV consistency cannot establish these claims"})
    states = [check["status"] for key, check in checks.items() if key != "validation_limits"]
    overall = "fail" if "fail" in states else "partial" if any(state != "pass" for state in states) else "pass"
    audit = {"format_version": 1, "dataset": path.name, "status": overall,
             "summary": {"rows": row_count, "well_formed_rows": good_rows, "columns": len(header), "episodes": len(episodes), "engines": len(engines), "dataset_sha256": dataset_sha},
             "checks": checks, "validation_limits": limits,
             "limitations": ["Synthetic data contract audit, not physical or clinical/field validation.",
                             "Explicit sidecar configuration overrides the 171-sample/1800-second journey default.",
                             "Matrix thresholds are transparent screening heuristics; rates are within-positive-episode least-squares trends.",
                             "Per-episode unique-key storage is capped at 100000; exceeding it yields partial key validation.",
                             "Source hash checks require supplied source files; missing files are not evaluated."]}
    if output_prefix is not None:
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ".json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        Path(str(prefix) + ".md").write_text(_report_markdown(audit), encoding="utf-8")
    return audit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args(argv)
    prefix = args.output_prefix or args.csv.with_suffix(".audit")
    try:
        audit = audit_training_csv(args.csv, args.schema, args.manifest, prefix)
    except OSError as error:
        parser.error(str(error))
    print(f"{audit['status']}: {audit['summary']['rows']} rows, {audit['summary']['episodes']} episodes; reports: {prefix}.json, {prefix}.md")
    return 1 if audit["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
