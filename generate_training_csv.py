"""Stream fresh, truck-disjoint enhanced journeys into one training CSV.

The default is 20 trucks x 48 journeys x 171 readings = 164,160 rows.
Each truck has 34 primary, five sensor-mode, six pair and three triple journeys.
The 369 approved numeric features already include the 113 raw channels (shared
names, not duplicate columns) and their observation-derived validity masks.
Truth, quality names, identifiers and heuristic diagnostics are excluded from X.
Adjacent <stem>.schema.json and <stem>.manifest.json describe the entire contract.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any


DEFAULT_OUTPUT = Path("datasets/dump_truck_enhanced_all_conditions.csv")
SCENARIOS = ("ideal", "stress", "overhaul")
PAYLOAD_CLASSES = ("empty", "light", "medium", "heavy", "overload")
START_CONDITIONS = ("cold", "warm")
SENSOR_MODES = ("biased", "missing", "frozen", "drifting", "noisy")
SENSOR_TARGETS = (
    "engine.oil_pressure_kpa", "engine.coolant_temp_c",
    "engine.block_vibration_mm_s", "transmission.oil_temp_c",
    "engine.oil_condition_pct", "transmission.oil_pressure_kpa",
    "engine.boost_kpa", "suspension.hydraulic_oil_condition_pct",
    "coolant_condition_pct", "engine.coolant_flow_kg_s",
)
MAX_ROWS = 2_000_000
RAW_CHANNEL_COUNT = 113
FEATURE_COUNT = 369
FAULT_COUNT = 16
LABEL_THRESHOLD = .001
METADATA_COLUMNS = (
    "engine_id", "episode_id", "base_trajectory_id", "split", "timestamp",
    "elapsed_s", "step", "scenario", "payload_class", "start_condition",
    "mode", "cycle_phase", "drive_cycle", "coverage_group", "primary_class",
)
TRUTH_COLUMNS = (
    "active_faults_json", "event_phases_json", "event_occurrences_json",
    "sensor_quality_json", "latent_wear_json",
)
DIAGNOSTIC_COLUMNS = (
    "probable_faults_json", "possible_causes_json", "mitigations_json",
    "overall_health", "subsystem_health_json", "diagnostic_source",
    "root_cause_confirmed", "rul_status",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seed(seed: int, *parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(_json([seed, *parts]).encode()).digest()[:8], "big") % 2**32


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(_seed(seed, *parts))


def _integer(name: str, value: Any, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _number(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")
    return value


def _splits(engines: int, seed: int) -> dict[str, str]:
    ids = [f"virtual-{index + 1:04d}" for index in range(engines)]
    _rng(seed, "splits").shuffle(ids)
    validation = max(1, round(engines * .15))
    test = max(1, round(engines * .15))
    train = engines - validation - test
    return {engine: "train" if index < train else "validation" if index < train + validation else "test"
            for index, engine in enumerate(ids)}


def _plans(engines: int, episodes_per_fault: int, seed: int, duration_s: float,
           timestep_s: float, kinds: tuple[str, ...]):
    """Yield reproducible plans; primary grids are independent per class.

    Pair-only supplements take six consecutive entries per truck from one
    shuffled 120-pair grid. Twenty trucks cover that grid exactly once; smaller
    requests sample it, and larger requests repeat after the first 120 pairs.
    """
    grids = {}
    for kind in ("normal", *kinds):
        grid = list(itertools.product(SCENARIOS, PAYLOAD_CLASSES, START_CONDITIONS))
        _rng(seed, "conditions", kind).shuffle(grid)
        grids[kind] = grid
    pairs = list(itertools.combinations(kinds, 2))
    triples = list(itertools.combinations(kinds, 3))
    _rng(seed, "pairs").shuffle(pairs)
    _rng(seed, "triples").shuffle(triples)
    for engine_index in range(engines):
        engine_id = f"virtual-{engine_index + 1:04d}"
        specs = [("primary", kind, repeat, () if kind == "normal" else (kind,), None)
                 for kind in ("normal", *kinds) for repeat in range(episodes_per_fault)]
        specs += [("sensor_variant", "sensor_fault", repeat, ("sensor_fault",), mode)
                  for repeat, mode in enumerate(SENSOR_MODES)]
        specs += [("mixed_pair", "mixed", repeat, pairs[(engine_index * 6 + repeat) % len(pairs)], None)
                  for repeat in range(6)]
        specs += [("mixed_triple", "mixed", repeat, triples[(engine_index * 3 + repeat) % len(triples)], None)
                  for repeat in range(3)]
        for index, (group, primary, repeat, selected, sensor_mode) in enumerate(specs):
            episode_id = f"{engine_id}-episode-{index + 1:04d}"
            if group == "primary":
                condition = grids[primary][(engine_index * episodes_per_fault + repeat) % 30]
            else:
                condition = _rng(seed, episode_id, "context").choice(grids["normal"])
            config = {
                "engine_id": engine_id, "episode_id": episode_id,
                "base_trajectory_id": f"{engine_id}-trajectory-{index + 1:04d}",
                "engine_seed": _seed(seed, engine_id, "variant"),
                "seed": _seed(seed, episode_id, "observation_noise"),
                "scenario": condition[0], "payload_class": condition[1], "start_condition": condition[2],
                "drive_cycle": "c-wtvc", "duration_s": duration_s, "timestep_s": timestep_s,
                "start_time": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
            }
            for field, bounds in (("ambient_c", (-5, 42)), ("initial_wear", (0, .18)),
                                  ("noise_scale", (.25, 1.8)), ("wear_time_scale", (1, 4))):
                config[field] = round(_rng(seed, episode_id, field).uniform(*bounds), 6)
            events = []
            for event_index, kind in enumerate(selected):
                def draw(field, low, high):
                    return _rng(seed, episode_id, kind, event_index, field).uniform(low, high)
                recurring = repeat % 2 == 1 and kind not in ("hydraulic_normal_aging", "hydraulic_contamination")
                event = {
                    "event_id": f"{episode_id}-event-{event_index + 1}", "kind": kind,
                    "onset_s": duration_s * draw("onset", .10, .14 if recurring else .24),
                    "duration_s": duration_s * draw("duration", .16 if recurring else .42, .22 if recurring else .56),
                    "recovery_s": duration_s * draw("recovery", .06, .10 if recurring else .12),
                    "severity": round(draw("severity", .45, 1), 6),
                    "progression": round(draw("progression", .05, .4), 6),
                }
                if recurring:
                    event["recurrence_s"] = duration_s * draw("recurrence", .46, .50)
                if kind == "sensor_fault":
                    target = SENSOR_TARGETS[(engine_index + repeat + event_index) % len(SENSOR_TARGETS)]
                    # Biases follow channel units, rather than applying a 20-unit
                    # temperature bias to a sub-unit vibration or flow signal.
                    low, high = ((.12, .8) if target.endswith("_kg_s") else
                                 (1.5, 5) if target.endswith("_mm_s") else (5, 25))
                    event.update(target_signal=target,
                                 sensor_mode=sensor_mode or _rng(seed, episode_id, "sensor_mode").choice(SENSOR_MODES),
                                 bias=round(draw("bias", low, high), 6))
                _validate_event(event, duration_s, timestep_s)
                events.append(event)
            yield {"config": config, "events": events, "coverage_group": group,
                   "primary_class": primary, "repeat": repeat, "condition": list(condition)}


def _validate_event(event: dict[str, Any], duration: float, dt: float) -> None:
    if event["duration_s"] < dt or event["recovery_s"] < dt:
        raise ValueError("Event duration and recovery must each include at least one sampling interval")
    occupied = event["duration_s"] + event["recovery_s"]
    onset = event["onset_s"]
    recurrence = event.get("recurrence_s")
    if recurrence is not None:
        if recurrence < occupied:
            raise ValueError("Recurrence must include event duration and recovery")
        onset += math.floor((duration - onset) / recurrence) * recurrence
    if onset + occupied > duration:
        raise ValueError("Every occurrence and recovery must fit the episode")


def _observed_samples(samples: list[dict[str, Any]], analytics: Any) -> list[dict[str, Any]]:
    """Physical observations and approved context only: no truth or quality names."""
    allowed = (*analytics.CONTEXT_NUMERIC, "mode", "start_condition", "drive_cycle")
    return [{"engine_id": sample["engine_id"], "episode_id": sample["episode_id"],
             "elapsed_s": sample["elapsed_s"], "observations": dict(sample["observations"]),
             "context": {name: sample["context"][name] for name in allowed}}
            for sample in samples]


def _normalise_times(samples: list[dict[str, Any]], count: int, dt: float, duration: float) -> None:
    """Restore exact grid coordinates from the legacy 3-decimal time rendering.

    Simulator effects/truth are evaluated on its millisecond-rounded times.
    CSV/analytics use the exact requested grid; no observation or truth value
    is altered. Maximum difference is half a millisecond.
    """
    if len(samples) != count:
        raise ValueError("Simulator sample count does not match the requested journey length")
    for index, sample in enumerate(samples):
        elapsed = index * dt if index < count - 1 else duration
        if not math.isclose(sample["elapsed_s"], elapsed, rel_tol=0, abs_tol=.000500001):
            raise ValueError("Simulator changed the sample interval")
        sample["elapsed_s"] = elapsed


def _validate_features(feature: dict[str, Any], allowed: set[str]) -> None:
    if not isinstance(feature, dict) or set(feature) != allowed:
        raise ValueError("Feature schema changed or contains unapproved/leaking columns")
    forbidden = ("truth", "expected", "quality", "fault_label", "fault_level", "effect_level",
                 "event", "latent_wear", "rul", "health", "confirmed")
    for name, value in feature.items():
        if name in METADATA_COLUMNS or any(token in name.lower() for token in forbidden):
            raise ValueError(f"Leaking/non-input feature: {name}")
        if value is not None:
            _number(name, value, -1e15, 1e15)
        if name.endswith("__available_mask") and value not in (0, 1):
            raise ValueError(f"Nonbinary availability mask: {name}")


def _approved_feature_names(analytics: Any) -> set[str]:
    """Contract of the existing public pipeline, independent of returned rows."""
    raw = analytics.SENSOR_COLUMNS
    return { *raw, *(name + "__available_mask" for name in raw),
             *("context." + name for name in analytics.CONTEXT_NUMERIC),
             "context.cold_start", *("context.mode_" + name for name in analytics.MODE_LOAD),
             *(name + suffix for name in analytics.THRESHOLDS
               for suffix in ("__residual", "__rate", "__residual_slope_120s", "__mean_30s", "__std_30s")),
             "engine.egt_spread_c",
             "leadlag.engine.coolant_flow_kg_s__to__engine.oil_temp_c_s",
             "leadlag.engine.oil_pressure_kpa__to__engine.oil_temp_c_s",
             "leadlag.engine.oil_temp_c__to__engine.block_vibration_mm_s_s" }


def _compact_diagnostics(diagnostic: dict[str, Any], active: list[str], catalogue: dict) -> dict[str, Any]:
    # Lookup outputs describe injected *effect labels*, not diagnostic inference.
    # Probable faults are supplied independently by observation-only analytics.
    probable = [{"fault": hypothesis["fault"], "source": "enhanced_analytics.analyse_episode",
                 "confidence": hypothesis.get("confidence", "heuristic_not_probability"),
                 "root_cause_confirmed": False}
                for hypothesis in diagnostic.get("hypotheses", [])]
    def lookup(field):
        return [{"fault": kind, field: catalogue[kind][field],
                 "source": "synthetic_catalogue_lookup:enhanced_simulation.FAULT_CATALOG",
                 "basis": "injected_effect_label", "root_cause_confirmed": False}
                for kind in active]
    return {"probable_faults_json": _json(probable),
            "possible_causes_json": _json(lookup("possible_cause")),
            "mitigations_json": _json(lookup("mitigation")),
            "overall_health": diagnostic.get("overall_health"),
            "subsystem_health_json": _json(diagnostic.get("subsystem_health", {})),
            "diagnostic_source": "observation_only_heuristic_not_field_validated",
            "root_cause_confirmed": 0, "rul_status": "unavailable_not_validated"}


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _source_digest(module: Any) -> str | None:
    path = getattr(module, "__file__", None)
    return _digest(Path(path)) if path and Path(path).is_file() else None


def _publish(stage: Path, paths: tuple[Path, ...], overwrite: bool) -> None:
    """Publish fully staged files with exclusive creates or rollback on failure.

    Individual renames are atomic; three adjacent files cannot form a single
    filesystem transaction. The manifest is published last as the completion
    receipt. Backups are restored on a caught publishing failure.
    """
    installed, backups = [], {}
    try:
        for path in paths:
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"Output must be a regular file: {path}")
            if path.exists():
                if not overwrite:
                    raise FileExistsError(f"Output exists: {path}; use overwrite=True")
                backup = stage / (path.name + ".backup")
                path.replace(backup)
                backups[path] = backup
            staged = stage / path.name
            if overwrite:
                staged.replace(path)
            else:
                # Hard-link creation refuses racing existing files, unlike rename.
                path.hardlink_to(staged)
            installed.append(path)
    except BaseException:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        for path, backup in backups.items():
            backup.replace(path)
        raise


def generate_training_csv(output_path=DEFAULT_OUTPUT, engines=20, episodes_per_fault=2,
                          samples_per_journey=171, duration_s=1800, seed=793,
                          overwrite=False) -> dict[str, Any]:
    """Generate one CSV and two sidecars; return their paths and actual counts.

    Episodes are independent experiments with stable per-truck variation, not
    continuous lifecycle histories. Use only schema.input_columns for X and
    schema.target_columns for y. Missing numeric entries are blank CSV cells;
    fit imputation/scaling on training trucks only. Nondefault truck counts have
    explicitly recorded sampled pair coverage (six pair-only journeys/truck).
    """
    engines = _integer("engines", engines, 3, 1000)
    episodes_per_fault = _integer("episodes_per_fault", episodes_per_fault, 1, 100)
    samples_per_journey = _integer("samples_per_journey", samples_per_journey, 21, 10000)
    seed = _integer("seed", seed, 0, 2**32 - 1)
    duration_s = _number("duration_s", duration_s, 1, 86400)
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be boolean")
    if not isinstance(output_path, (str, Path)) or not str(output_path).strip():
        raise ValueError("output_path must be a nonempty CSV path")
    output = Path(output_path).absolute()
    if output.suffix.lower() != ".csv":
        raise ValueError("output_path must have a .csv extension")
    paths = (output, output.with_suffix(".schema.json"), output.with_suffix(".manifest.json"))
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"Output must be a regular file: {path}")
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}; use overwrite=True")
    episodes = engines * (17 * episodes_per_fault + 14)
    expected_rows = episodes * samples_per_journey
    if expected_rows > MAX_ROWS:
        raise ValueError(f"Request exceeds {MAX_ROWS:,} rows")
    dt = duration_s / (samples_per_journey - 1)
    # Existing simulator floors duration/dt. Repair only floating-point horizon
    # rounding, never the interval or the simulator, to retain exactly N points.
    simulation_duration = duration_s
    if math.floor(simulation_duration / dt) + 1 < samples_per_journey:
        simulation_duration = math.nextafter(duration_s, math.inf)
    if math.floor(simulation_duration / dt) + 1 != samples_per_journey:
        raise ValueError("Sampling horizon is not representable at the requested interval")

    # Lazy imports: argument validation/help do not load simulation or analytics.
    import enhanced_simulation as simulation
    import enhanced_analytics as analytics
    import telemetry_model
    import operational_ranges

    kinds = tuple(simulation.FAULT_KINDS)
    if len(kinds) != FAULT_COUNT or len(set(kinds)) != FAULT_COUNT or "sensor_fault" not in kinds:
        raise ValueError("Expected the enhanced 16-class fault taxonomy")
    if len(analytics.SENSOR_COLUMNS) != RAW_CHANNEL_COUNT:
        raise ValueError("Expected 113 approved raw channels")
    approved_features = _approved_feature_names(analytics)
    if len(approved_features) != FEATURE_COUNT:
        raise ValueError("Expected 369 approved feature definitions")
    # Validate every schedule before creating even a staging directory.
    plans = list(_plans(engines, episodes_per_fault, seed, duration_s, dt, kinds))
    splits = _splits(engines, seed)
    targets = [f"fault_label.{kind}" for kind in kinds]
    label_columns = targets + [f"fault_level.{kind}" for kind in kinds] + [f"effect_level.{kind}" for kind in kinds] + ["is_normal", *TRUTH_COLUMNS]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.stem}-staging-", dir=output.parent))
    counts = Counter()
    class_counts, context_counts, pair_counts, triple_counts, positive_rows, sensor_modes, sensor_targets = (Counter() for _ in range(7))
    episode_records = []
    input_columns = columns = None
    supplied_units = {}
    try:
        with (stage / output.name).open("w", newline="", encoding="utf-8") as handle:
            writer = None
            for plan in plans:
                config = {**plan["config"], "duration_s": simulation_duration}
                result = simulation.simulate_enhanced(config, plan["events"])
                samples = result["samples"]
                _normalise_times(samples, samples_per_journey, dt, duration_s)
                activated = set()
                for sample in samples:
                    for key in ("engine_id", "episode_id", "base_trajectory_id"):
                        if sample[key] != config[key]:
                            raise ValueError(f"Simulator changed {key}")
                    if set(sample["observations"]) != set(analytics.SENSOR_COLUMNS):
                        raise ValueError("Simulator raw channel schema changed")
                    for key, value in sample["observations"].items():
                        if value is not None:
                            _number(key, value, -1e12, 1e12)
                        if (value is None) != (sample["quality"].get(key) == "missing"):
                            raise ValueError("Missing reading does not match synthetic quality truth")
                    for level_name in ("fault_levels", "effect_levels"):
                        if set(sample["truth"][level_name]) != set(kinds):
                            raise ValueError("Truth fault taxonomy changed")
                        for kind, level in sample["truth"][level_name].items():
                            _number(f"{level_name}.{kind}", level, 0, 1)
                    activated.update(kind for kind, level in sample["truth"]["fault_levels"].items() if level > 0)
                if not {event["kind"] for event in plan["events"]} <= activated:
                    raise ValueError("Scheduled fault never activated in sampled schedule truth")
                observed = _observed_samples(samples, analytics)
                # Public APIs intentionally used; each resets within this episode.
                features = analytics.engineer_features(observed)
                diagnostics = analytics.analyse_episode(observed)
                if len(features) != len(samples) or len(diagnostics) != len(samples):
                    raise ValueError("Analytics sample count mismatch")
                if input_columns is None:
                    input_columns = list(features[0])
                    if set(input_columns) != approved_features:
                        raise ValueError("Expected 369 engineered inputs containing all 113 raw channels")
                    # Definitions are populated by the public feature pipeline.
                    if set(input_columns) != set(analytics.FEATURE_DEFINITIONS):
                        raise ValueError("Approved feature definitions do not match the feature schema")
                    columns = [*METADATA_COLUMNS, *input_columns, *label_columns, *DIAGNOSTIC_COLUMNS]
                    if len(columns) != len(set(columns)):
                        raise ValueError("Output groups contain duplicate columns")
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    supplied_units = result["metadata"]["units"]
                allowed = set(input_columns)
                for index, (sample, feature, diagnostic) in enumerate(zip(samples, features, diagnostics)):
                    _validate_features(feature, allowed)
                    if diagnostic["elapsed_s"] != sample["elapsed_s"]:
                        raise ValueError("Diagnostic elapsed-time join mismatch")
                    for name, value in sample["observations"].items():
                        if feature[name] != value:
                            raise ValueError(f"Raw channel was changed by feature pipeline: {name}")
                    truth, context = sample["truth"], sample["context"]
                    active = [kind for kind in kinds if truth["effect_levels"][kind] > LABEL_THRESHOLD]
                    if plan["primary_class"] == "normal" and (active or any(truth["fault_levels"].values())):
                        raise ValueError("Normal experiment has active fault truth")
                    metadata = {name: sample[name] for name in ("engine_id", "episode_id", "base_trajectory_id", "timestamp", "elapsed_s")}
                    metadata.update({name: context[name] for name in ("scenario", "payload_class", "start_condition", "mode", "cycle_phase", "drive_cycle")})
                    metadata.update(split=splits[config["engine_id"]], step=index,
                                    coverage_group=plan["coverage_group"], primary_class=plan["primary_class"])
                    labels = {f"fault_label.{kind}": int(kind in active) for kind in kinds}
                    labels.update({f"fault_level.{kind}": truth["fault_levels"][kind] for kind in kinds})
                    labels.update({f"effect_level.{kind}": truth["effect_levels"][kind] for kind in kinds})
                    labels.update(is_normal=int(not active), active_faults_json=_json(truth["active_faults"]),
                                  event_phases_json=_json(truth["event_phases"]),
                                  event_occurrences_json=_json(truth["event_occurrences"]),
                                  sensor_quality_json=_json({name: quality for name, quality in sample["quality"].items() if quality != "good"}),
                                  latent_wear_json=_json(truth["latent_wear"]))
                    writer.writerow({**metadata, **feature, **labels,
                                     **_compact_diagnostics(diagnostic, active, simulation.FAULT_CATALOG)})
                    counts["rows"] += 1
                    positive_rows.update(active)
                counts["episodes"] += 1
                counts[plan["coverage_group"]] += 1
                for event in plan["events"]:
                    if event["kind"] == "sensor_fault":
                        sensor_modes[event["sensor_mode"]] += 1
                        sensor_targets[event["target_signal"]] += 1
                if plan["coverage_group"] == "primary":
                    class_counts[plan["primary_class"]] += 1
                    context_counts[(plan["primary_class"], *plan["condition"])] += 1
                elif plan["coverage_group"] == "mixed_pair":
                    pair_counts[tuple(sorted(event["kind"] for event in plan["events"]))] += 1
                elif plan["coverage_group"] == "mixed_triple":
                    triple_counts[tuple(sorted(event["kind"] for event in plan["events"]))] += 1
                episode_records.append({**plan, "config": config, "split": splits[config["engine_id"]],
                                        "sample_count": len(samples), "first_elapsed_s": 0, "last_elapsed_s": duration_s})
        if counts["rows"] != expected_rows or counts["episodes"] != episodes:
            raise ValueError("Final episode/row count mismatch")
        if set(class_counts.values()) != {engines * episodes_per_fault}:
            raise ValueError("Primary class coverage mismatch")
        if set(sensor_modes) != set(SENSOR_MODES):
            raise ValueError("Sensor mode coverage is incomplete")
        if engines >= 20 and len(pair_counts) != 120:
            raise ValueError("Mixed pair coverage is incomplete")
        if engines * episodes_per_fault >= 30 and any(
                sum(key[0] == kind for key in context_counts) != 30 for kind in ("normal", *kinds)):
            raise ValueError("Primary condition coverage is incomplete")
        definitions = {name: dict(analytics.FEATURE_DEFINITIONS[name]) for name in input_columns}
        context_units = {"context.rpm": "rpm", "context.load_pct": "%", "context.demand_pct": "%",
                         "context.ambient_c": "degC", "context.payload_utilization_pct": "%",
                         "context.transient": "1", "context.cold_start": "0 or 1"}
        units = {}
        for name in input_columns:
            base = name.split("__", 1)[0]
            unit = supplied_units.get(base, context_units.get(name, definitions[name]["units"]))
            if name.endswith("__available_mask") or name.startswith("context.mode_"):
                unit = "0 or 1"
            elif name.endswith("__rate") or "__residual_slope" in name:
                unit += "/s"
            elif name.startswith("leadlag."):
                unit = "s"
            units[name] = unit
        for name in input_columns:
            definitions[name]["units"] = units[name]
        units.update({name: "identifier/category" for name in METADATA_COLUMNS})
        units.update(elapsed_s="s", step="count", timestamp="ISO-8601 UTC")
        units.update({name: "0 or 1" for name in targets + ["is_normal", "root_cause_confirmed"]})
        units.update({name: "synthetic mechanism strength [0,1]" for name in label_columns
                      if name.startswith(("fault_level.", "effect_level."))})
        units.update({name: "JSON" for name in columns if name.endswith("_json")})
        units.update(overall_health="heuristic index [0,100]", diagnostic_source="category", rul_status="category")
        exclusions = [*METADATA_COLUMNS, *label_columns, *DIAGNOSTIC_COLUMNS]
        schema = {
            "schema_version": "enhanced-training-csv/1", "columns": columns,
            "input_columns": input_columns, "target_columns": targets,
            "metadata_columns": list(METADATA_COLUMNS), "diagnostic_columns": list(DIAGNOSTIC_COLUMNS),
            "label_columns": label_columns, "raw_columns": list(analytics.SENSOR_COLUMNS),
            "engineered_columns": input_columns,
            "availability_mask_columns": [name for name in input_columns if name.endswith("__available_mask")],
            "units": units, "feature_definitions": definitions, "fault_taxonomy": simulation.FAULT_CATALOG,
            "label_semantics": {"fault_label": "int(truth.effect_levels[kind] > 0.001)",
                                "fault_level": "truth.fault_levels: undelayed schedule envelope",
                                "effect_level": "truth.effect_levels: lagged realized envelope or persistent synthetic wear",
                                "is_normal": "int(sum(fault_label.*) == 0); derived, not a 17th target",
                                "sensor_quality_json": "Sparse simulator quality truth: only non-good statuses; any omitted raw channel has good quality. Excluded from model inputs."},
            "missingness_handling": {"csv_representation": "blank numeric cells", "feature_representation": "None",
                                    "mask": "Observation-only validity/availability mask; 1 when analytics classifies the observed reading good, 0 for missing/out-of-range/frozen-suspected. Not injected quality truth.",
                                    "imputation": "Fit imputation and scaling on train trucks only; reset temporal features each episode."},
            "exclusions": {"columns": exclusions,
                           "absent_simulator_fields": ["expected", "raw_legacy", "events", "truth"],
                           "reason": "All identifiers, sampling/time metadata, scenario/payload/start/mode/phase categories, truth, named quality, labels, diagnostic health and catalogue lookups are excluded from X; only approved engineered numeric operating context is included."},
            "diagnostic_semantics": {"probable_faults_json": "Compact observation-only analyse_episode hypotheses; heuristic, not confirmed or probabilities.",
                                     "possible_causes_json": "Synthetic FAULT_CATALOG lookup for active effect labels; provenance output, not inferred.",
                                     "mitigations_json": "Synthetic FAULT_CATALOG lookup for active effect labels; provenance output, not inferred.",
                                     "rul": "No numerical RUL column or invented RUL; unavailable_not_validated."},
        }
        (stage / paths[1].name).write_text(_json(schema) + "\n", encoding="utf-8")
        membership = {split: sorted(engine for engine, assigned in splits.items() if assigned == split)
                      for split in ("train", "validation", "test")}
        manifest = {
            "schema_version": "enhanced-training-csv/1", "source": "fresh_enhanced_simulation", "synthetic": True,
            "physics_validated": False, "diagnostics_validated": False,
            "parameters": {"engines": engines, "episodes_per_fault": episodes_per_fault,
                           "samples_per_journey": samples_per_journey, "duration_s": duration_s, "seed": seed,
                           "timestep_s": dt, "simulation_duration_s": simulation_duration},
            "sampling_semantics": "CSV elapsed_s and analytics use step*duration/(samples-1), with exact final duration. Legacy simulator evaluates effects/truth on its 3-decimal elapsed_s grid (maximum 0.0005s difference); only exported/analytics time coordinates are normalised. Simulation horizon is nudged upward by at most one float ULP if flooring would drop the endpoint.",
            "counts": {**dict(counts), "engines": engines, "columns": len(columns),
                       "raw_channels": RAW_CHANNEL_COUNT, "engineered_inputs": FEATURE_COUNT,
                       "targets": FAULT_COUNT, "episodes_per_engine": 17 * episodes_per_fault + 14},
            "source_hashes": {"generator": _digest(Path(__file__)), "simulation": _source_digest(simulation),
                              "analytics": _source_digest(analytics), "telemetry_model": _source_digest(telemetry_model),
                              "operational_ranges": _source_digest(operational_ranges)},
            "seed_derivation": "SHA256 compact sorted JSON [master_seed, stable identifiers, independent parameter name], first 64 bits modulo 2**32; stable variant per truck",
            "split_policy": "Shuffle truck IDs deterministically; val/test each max(1, round(engines*0.15)), remainder train; no truck crosses splits",
            "split_membership": membership, "primary_class_episode_coverage": dict(sorted(class_counts.items())),
            "positive_label_row_counts": {kind: positive_rows[kind] for kind in kinds},
            "sensor_mode_event_coverage": dict(sorted(sensor_modes.items())),
            "sensor_target_event_coverage": dict(sorted(sensor_targets.items())),
            "primary_context_coverage": {
                kind: {"distinct_conditions": sum(key[0] == kind for key in context_counts),
                       "conditions": [{"scenario": key[1], "payload_class": key[2], "start_condition": key[3], "episodes": count}
                                      for key, count in sorted(context_counts.items()) if key[0] == kind]}
                for kind in ("normal", *kinds)},
            "context_policy": "Each primary class independently shuffled 30-condition Cartesian grid indexed by engine_index*episodes_per_fault+repeat modulo 30; all 30 covered when at least 30 class episodes.",
            "pair_coverage": {"possible_pairs": 120, "distinct_pairs": len(pair_counts),
                              "pair_episodes": sum(pair_counts.values()), "complete": len(pair_counts) == 120,
                              "policy": "Six pair-only journeys/truck from shuffled120 grid; 20 trucks cover all120 exactly once, fewer trucks sample6*engines distinct pairs, more repeat after120. Triple pair combinations excluded from this count.",
                              "pairs": [{"faults": list(pair), "episodes": count} for pair, count in sorted(pair_counts.items())]},
            "triple_coverage": {"distinct_triples": len(triple_counts),
                                "triples": [{"faults": list(triple), "episodes": count} for triple, count in sorted(triple_counts.items())]},
            "lifecycle_semantics": "Independent fault experiments per truck; no carried fault-specific damage. Generic initial wear varies independently per episode.",
            "episodes": episode_records,
            "files": {output.name: {"sha256": _digest(stage / output.name), "rows": counts["rows"]},
                      paths[1].name: {"sha256": _digest(stage / paths[1].name)}},
            "manifest_digest_semantics": "Manifest cannot contain its own byte digest; its SHA256 is returned in the API receipt.",
        }
        (stage / paths[2].name).write_text(_json(manifest) + "\n", encoding="utf-8")
        receipt = {"output_path": str(output), "schema_path": str(paths[1]), "manifest_path": str(paths[2]),
                   "engines": engines, "episodes": counts["episodes"], "rows": counts["rows"],
                   "input_columns": len(input_columns), "split_membership": membership,
                   "sha256": {path.name: _digest(stage / path.name) for path in paths}}
        _publish(stage, paths, overwrite)
        return receipt
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", dest="output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--engines", type=int, default=20)
    parser.add_argument("--episodes-per-fault", type=int, default=2)
    parser.add_argument("--samples", dest="samples_per_journey", type=int, default=171)
    parser.add_argument("--duration-s", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=793)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        receipt = generate_training_csv(**vars(args))
    except (ValueError, FileExistsError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
