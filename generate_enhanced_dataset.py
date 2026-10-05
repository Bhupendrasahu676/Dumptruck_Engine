"""Repeatable, engine-disjoint synthetic enhanced datasets (stdlib exporter).

Example: python generate_enhanced_dataset.py --output-dir datasets/enhanced
Truth and heuristic diagnostic outputs are separate from model input tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from datetime import datetime, timedelta
from collections import Counter
from pathlib import Path
from typing import Any


SCENARIOS = ("ideal", "stress", "overhaul")
PAYLOAD_CLASSES = ("empty", "light", "medium", "heavy", "overload")
SENSOR_MODES = ("biased", "missing", "frozen", "drifting", "noisy")
SENSOR_TARGETS = (
    "engine.oil_pressure_kpa", "engine.coolant_temp_c",
    "engine.block_vibration_mm_s", "transmission.oil_temp_c",
    "engine.oil_condition_pct",
)
JOIN_KEYS = ("engine_id", "episode_id", "elapsed_s")
IDENTIFIERS = (*JOIN_KEYS, "base_trajectory_id", "timestamp", "split")
ARTIFACTS = ("readings.csv", "labels.csv", "diagnostics.csv", "features.csv",
             "manifest.json", "waveforms.jsonl", "spectral_features.csv")
MAX_ROWS = 2_000_000
MAX_WAVEFORMS = 5_000
WAVEFORM_RATE_HZ = 4096
WAVEFORM_DURATION_S = 0.25


def _seed(seed: int, *parts: Any) -> int:
    encoded = json.dumps([seed, *parts], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**32)


def _integer(name: str, value: Any, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _number(name: str, value: Any, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]") from error
    if not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be a finite number in [{lower}, {upper}]")
    return value


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_digest(module: Any) -> str | None:
    source = getattr(module, "__file__", None)
    return hashlib.sha256(Path(source).read_bytes()).hexdigest() if source and Path(source).is_file() else None


def _unit(name: str) -> str:
    name = name.removesuffix("__residual").removesuffix("__rate")
    for suffix, unit in (("_mm_s", "mm/s"), ("_kg_s", "kg/s"),
                         ("_kpa", "kPa"), ("_dps", "degrees/s"),
                         ("_rpm", "rpm"), (".rpm", "rpm"), ("_pct", "%"),
                         ("_hz", "Hz"), ("_c", "degC"), ("_g", "g"),
                         ("_m", "m"), ("_hours", "h"), ("_s", "s")):
        if name.endswith(suffix):
            return unit
    return "dimensionless" if "mask" in name else "unspecified (synthetic signal)"


def _feature_definition(name: str) -> dict[str, str]:
    """Fallback definitions; prefer the analytics module's definitions if supplied."""
    unit = _unit(name)
    if "mask" in name:
        description, unit = "Numeric measurement-availability mask; not a fault label", "0 or 1"
    elif "residual" in name:
        description = "Observation residual against an observation/context-derived reference"
    elif "rate" in name or "slope" in name:
        description = "Temporal rate of the named observation, computed within one episode"
        unit += "/s"
    elif any(token in name for token in ("mean", "std", "window", "rolling")):
        description = "Within-episode temporal summary of the named observation"
    else:
        description = "Numeric observation or operating-context feature"
    return {"definition": description, "units": unit}


def _splits(engines: int, seed: int) -> dict[str, str]:
    ids = [f"virtual-{index + 1:04d}" for index in range(engines)]
    random.Random(_seed(seed, "splits")).shuffle(ids)
    validation = max(1, int(engines * 0.15))
    test = max(1, int(engines * 0.15))
    train = engines - validation - test
    return {engine: "train" if index < train else "validation" if index < train + validation else "test"
            for index, engine in enumerate(ids)}


def _plans(engines: int, episodes_per_fault: int, seed: int, duration_s: float,
           timestep_s: float, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    plans = []
    for engine_index in range(engines):
        engine_id = f"virtual-{engine_index + 1:04d}"
        engine_seed = _seed(seed, engine_id, "variant")
        specifications = [("balanced", kind, repeat, None)
                          for kind in ("normal", *kinds)
                          for repeat in range(episodes_per_fault)]
        # All modes are represented even with only one primary sensor episode.
        specifications.extend(("sensor_variant", "sensor_fault", index, mode)
                              for index, mode in enumerate(SENSOR_MODES))
        specifications.extend(("mixed", "mixed", index, None) for index in range(2))
        for index, (group, kind, repeat, sensor_mode) in enumerate(specifications):
            episode_id = f"{engine_id}-episode-{index + 1:04d}"
            episode_seed = _seed(seed, engine_id, episode_id, "episode")
            rng = random.Random(episode_seed)
            config = {
                "seed": episode_seed, "engine_seed": engine_seed,
                "engine_id": engine_id, "episode_id": episode_id,
                "base_trajectory_id": f"{engine_id}-trajectory-{index + 1:04d}",
                "scenario": SCENARIOS[(engine_index + index) % len(SCENARIOS)],
                "payload_class": PAYLOAD_CLASSES[(engine_index + index) % len(PAYLOAD_CLASSES)],
                "start_condition": ("cold", "warm")[(engine_index + index) % 2],
                "drive_cycle": "c-wtvc", "duration_s": duration_s, "timestep_s": timestep_s,
                "ambient_c": round(rng.uniform(-5, 42), 3),
                "initial_wear": round(rng.uniform(0.0, 0.18), 4),
                "noise_scale": round(rng.uniform(0.25, 1.8), 4),
                "wear_time_scale": round(rng.uniform(1.0, 4.0), 4),
            }
            events = []
            if kind != "normal":
                selected = [kind] if kind != "mixed" else [
                    "engine_oil_pump_wear", "coolant_degradation"
                ] if repeat == 0 else ["gearbox_bearing_wear", "sensor_fault"]
                for event_index, fault in enumerate(selected):
                    onset = duration_s * rng.uniform(0.12, 0.22)
                    event = {
                        "kind": fault, "onset_s": onset,
                        "duration_s": duration_s * rng.uniform(0.48, 0.6),
                        "severity": round(rng.uniform(0.45, 1.0), 4),
                        "progression": round(rng.uniform(0.05, 0.4), 4),
                        "recovery_s": duration_s * rng.uniform(0.08, 0.15),
                        "event_id": f"{episode_id}-event-{event_index + 1}",
                    }
                    if fault == "sensor_fault":
                        event.update(target_signal=SENSOR_TARGETS[(engine_index + repeat) % len(SENSOR_TARGETS)],
                                      sensor_mode=sensor_mode or SENSOR_MODES[(engine_index + repeat) % len(SENSOR_MODES)],
                                      bias=round(rng.uniform(5.0, 25.0), 4))
                    if repeat % 2 and fault not in ("hydraulic_normal_aging", "hydraulic_contamination"):
                        # Repeated shorter disturbances in additional repetitions.
                        event["duration_s"] = duration_s * .18
                        event["recovery_s"] = duration_s * .06
                        event["recurrence_s"] = duration_s * .34
                    events.append(event)
            plans.append({"config": config, "events": events, "coverage_group": group,
                          "primary_class": kind})
    return plans


class _Table:
    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = None
        self.columns: list[str] = []
        self.count = 0

    def write(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            self.columns = list(row)
            self.writer = csv.DictWriter(self.handle, fieldnames=self.columns)
            self.writer.writeheader()
        elif set(row) != set(self.columns):
            raise ValueError("CSV schema changed between samples or episodes")
        self.writer.writerow(row)
        self.count += 1

    def close(self) -> None:
        self.handle.close()


def _validate_sample(sample: dict[str, Any], kinds: tuple[str, ...], config: dict[str, Any]) -> None:
    for key in ("engine_id", "episode_id", "base_trajectory_id"):
        if sample[key] != config[key]:
            raise ValueError(f"Simulator changed {key}")
    _number("elapsed_s", sample["elapsed_s"], 0, config["duration_s"])
    observations, quality, expected = sample["observations"], sample["quality"], sample["expected"]
    if not observations or set(observations) != set(quality) or set(observations) != set(expected):
        raise ValueError("Observation, expected and quality signal sets must match")
    for signal, value in observations.items():
        status = quality[signal]
        if status not in ("good", *SENSOR_MODES):
            raise ValueError(f"Invalid quality status for {signal}: {status}")
        if value is None:
            if status != "missing":
                raise ValueError(f"Missing {signal} has no missing quality mask")
        else:
            _number(signal, value, -1e12, 1e12)
            if status == "missing":
                raise ValueError(f"Missing quality mask has a numeric value for {signal}")
            # Rated payload utilization can exceed 100%; condition indices cannot.
            limit = 125 if signal == "payload.utilization_pct" else 100
            if signal.endswith("_pct") and not 0 <= value <= limit:
                raise ValueError(f"Percentage signal outside 0..{limit}: {signal}")
        _number(f"expected.{signal}", expected[signal], -1e12, 1e12)
    truth = sample["truth"]
    if set(truth["fault_levels"]) != set(kinds):
        raise ValueError("Truth must include fault levels for every catalogue class")
    for kind, value in truth["fault_levels"].items():
        _number(f"fault_levels.{kind}", value, 0, 1)
    if any(kind not in kinds for kind in truth["active_faults"]):
        raise ValueError("Unknown active fault label")


def _validate_features(row: dict[str, Any]) -> None:
    if not isinstance(row, dict) or not row:
        raise ValueError("engineer_features must produce nonempty numeric dictionaries")
    for name, value in row.items():
        tokens = name.lower().replace("__", ".").split(".")
        if name in IDENTIFIERS or any(token in tokens for token in (
                "truth", "label", "labels", "diagnostics", "fault_levels", "active_faults",
                "event_phases", "latent_wear", "expected", "rul_hours", "root_cause_confirmed")):
            raise ValueError(f"Leakage/non-feature column from analytics: {name}")
        if value is not None:
            _number(f"feature.{name}", value, -1e15, 1e15)
        if "mask" in name and value not in (0, 1, 0.0, 1.0):
            raise ValueError(f"Feature mask must be binary: {name}")


def _commit(stage: Path, output: Path, overwrite: bool) -> None:
    """Publish only a fully validated directory, preserving unrelated existing files."""
    if output.exists():
        if not overwrite and any((output / name).exists() for name in ARTIFACTS):
            raise FileExistsError(f"Generated output already exists at {output}; use overwrite=True")
        for item in output.iterdir():
            if item.name not in ARTIFACTS:
                destination = stage / item.name
                if item.is_dir() and not item.is_symlink():
                    shutil.copytree(item, destination, symlinks=True)
                else:
                    shutil.copy2(item, destination, follow_symlinks=False)
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}-backup-", dir=output.parent))
        backup.rmdir()
        output.rename(backup)
        try:
            stage.rename(output)
        except BaseException:
            backup.rename(output)
            raise
        shutil.rmtree(backup)
    else:
        stage.rename(output)


def generate_dataset(output_dir: str | Path, engines: int = 12, episodes_per_fault: int = 2,
                     seed: int = 793, duration_s: float = 1800, timestep_s: float = 5,
                     waveforms: bool = False, waveform_every: int = 60,
                     overwrite: bool = False, longitudinal: bool = False) -> dict[str, Any]:
    """Generate complete datasets; never split samples from one engine across sets.

    Balanced coverage counts exclude explicitly marked supplemental sensor/mixed
    episodes. Every engine has all five sensor modes and two mixed episodes.
    Validation failures leave previously generated artifacts intact.
    """
    engines = _integer("engines", engines, 3, 1000)
    episodes_per_fault = _integer("episodes_per_fault", episodes_per_fault, 1, 100)
    seed = _integer("seed", seed, 0, 2**32 - 1)
    duration_s = _number("duration_s", duration_s, 1, 86400)
    timestep_s = _number("timestep_s", timestep_s, 0.05, duration_s / 20)
    waveform_every = _integer("waveform_every", waveform_every, 1, 1_000_000)
    if not all(isinstance(value, bool) for value in (waveforms, overwrite, longitudinal)):
        raise ValueError("waveforms, overwrite and longitudinal must be booleans")
    if not isinstance(output_dir, (str, Path)) or not str(output_dir).strip():
        raise ValueError("output_dir must be a nonempty path")
    output = Path(output_dir).absolute()
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("output_dir must be a directory, not a symlink or file")
    if output.exists() and not overwrite and any((output / name).exists() for name in ARTIFACTS):
        raise FileExistsError(f"Generated output already exists at {output}; use overwrite=True")

    # Lazy imports allow API/help/argument validation during parallel implementation.
    import enhanced_simulation as simulation
    import enhanced_analytics as analytics

    kinds = tuple(simulation.FAULT_KINDS)
    if not kinds or len(set(kinds)) != len(kinds) or "sensor_fault" not in kinds:
        raise ValueError("Invalid enhanced fault catalogue")
    plan_count = engines * ((len(kinds) + 1) * episodes_per_fault + len(SENSOR_MODES) + 2)
    samples_bound = math.ceil(duration_s / timestep_s) + 1
    if plan_count * samples_bound > MAX_ROWS:
        raise ValueError(f"Requested dataset exceeds {MAX_ROWS:,} sample rows")
    if waveforms and plan_count * math.ceil(samples_bound / waveform_every) > MAX_WAVEFORMS:
        raise ValueError(f"Requested waveforms exceed {MAX_WAVEFORMS:,}; increase waveform_every")
    plans = _plans(engines, episodes_per_fault, seed, duration_s, timestep_s, kinds)
    splits = _splits(engines, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    tables: dict[str, _Table] = {}
    waveform_handle = None
    try:
        for name in ("readings", "labels", "diagnostics", "features"):
            tables[name] = _Table(stage / f"{name}.csv")
        if waveforms:
            waveform_handle = (stage / "waveforms.jsonl").open("w", encoding="utf-8")
            tables["spectral_features"] = _Table(stage / "spectral_features.csv")
        episode_records, simulation_metadata = [], []
        coverage = Counter()
        waveform_count = 0
        waveform_skipped_missing = 0
        lifecycle_states = {}
        for plan in plans:
            config = plan["config"]
            if longitudinal and config["engine_id"] in lifecycle_states:
                config.update(lifecycle_states[config["engine_id"]])
            result = simulation.simulate_enhanced(config, plan["events"])
            if longitudinal:
                continuation = result["metadata"]["final_engine_state"]
                lifecycle_states[config["engine_id"]] = {
                    key: continuation[key] for key in ("initial_fault_wear", "initial_operating_hours", "initial_wear")}
                lifecycle_states[config["engine_id"]]["start_time"] = (
                    datetime.fromisoformat(result["samples"][-1]["timestamp"]) + timedelta(seconds=timestep_s)).isoformat()
            samples = result["samples"]
            if not samples or len(samples) > samples_bound:
                raise ValueError("Simulator produced an empty or unexpectedly large episode")
            previous_elapsed = -1.0
            activated = set()
            for sample in samples:
                _validate_sample(sample, kinds, config)
                if sample["elapsed_s"] <= previous_elapsed:
                    raise ValueError("Sample elapsed_s must increase strictly within each episode")
                previous_elapsed = sample["elapsed_s"]
                activated.update(kind for kind, level in sample["truth"]["fault_levels"].items() if level > 0)
            if any(event["kind"] not in activated for event in plan["events"]):
                raise ValueError("Scheduled fault never became active in sampled truth")
            if plan["primary_class"] == "normal" and activated:
                raise ValueError("Normal episode has active injected fault truth")
            # Observation-only inference boundary: schedules, expected and truth are absent.
            observed = [{key: sample[key] for key in (
                "engine_id", "episode_id", "base_trajectory_id", "elapsed_s", "timestamp",
                "context", "observations", "quality")} for sample in samples]
            diagnostics = analytics.analyse_episode(observed)
            features = analytics.engineer_features(observed)
            if len(diagnostics) != len(samples) or len(features) != len(samples):
                raise ValueError("Analytics rows must be parallel to the input samples")
            for index, (sample, diagnostic, feature) in enumerate(zip(samples, diagnostics, features)):
                _validate_features(feature)
                if diagnostic["elapsed_s"] != sample["elapsed_s"]:
                    raise ValueError("Diagnostic join time does not match sample")
                ids = {key: sample[key] for key in IDENTIFIERS if key != "split"}
                ids["split"] = splits[config["engine_id"]]
                context = {f"context.{key}": value for key, value in sample["context"].items()}
                for key, value in context.items():
                    if value is not None and not isinstance(value, (str, bool)):
                        _number(key, value, -1e12, 1e12)
                readings = {**ids, **context,
                            **{f"observation.{key}": value for key, value in sample["observations"].items()},
                            **{f"available.{key}": int(value is not None) for key, value in sample["observations"].items()}}
                effect_levels = sample["truth"].get("effect_levels", sample["truth"]["fault_levels"])
                labels = {**ids, "active_faults_json": _json(sample["truth"]["active_faults"]),
                           **{f"fault_level.{kind}": sample["truth"]["fault_levels"][kind] for kind in kinds},
                           **{f"effect_level.{kind}": effect_levels[kind] for kind in kinds},
                           **{f"fault_label.{kind}": int(effect_levels[kind] > .001) for kind in kinds},
                          "event_phases_json": _json(sample["truth"]["event_phases"]),
                          "latent_wear_json": _json(sample["truth"]["latent_wear"]),
                          "sensor_quality_json": _json(sample["quality"])}
                diagnostic_row = dict(ids)
                for key, value in diagnostic.items():
                    if key != "elapsed_s":
                        diagnostic_row[key] = _json(value) if isinstance(value, (dict, list, tuple)) else value
                for name, row in (("readings", readings), ("labels", labels),
                                  ("diagnostics", diagnostic_row), ("features", {**ids, **feature})):
                    tables[name].write(row)
                if waveform_handle is not None and index % waveform_every == 0:
                    if any(sample["observations"].get(signal) is None for signal in (
                            "engine.rpm", "engine.block_vibration_mm_s")):
                        waveform_skipped_missing += 1
                        continue
                    waveform_seed = _seed(seed, config["episode_id"], index, "waveform")
                    waveform = simulation.synthesize_vibration(observed[index], sample_rate_hz=WAVEFORM_RATE_HZ,
                                                               duration_s=WAVEFORM_DURATION_S, seed=waveform_seed)
                    values = waveform["waveform"]
                    if len(values) != int(WAVEFORM_RATE_HZ * WAVEFORM_DURATION_S):
                        raise ValueError("Waveform length does not match requested bounded size")
                    for value in values:
                        _number("waveform value", value, -1e12, 1e12)
                    waveform_handle.write(_json({**ids, **waveform, "seed": waveform_seed}) + "\n")
                    from vibration_analysis import spectral_features
                    tables["spectral_features"].write({**ids, **spectral_features(waveform)})
                    waveform_count += 1
            if plan["coverage_group"] == "balanced":
                coverage[plan["primary_class"]] += 1
            episode_records.append({**plan, "split": splits[config["engine_id"]],
                                    "sample_count": len(samples)})
            metadata = result["metadata"]
            if metadata not in simulation_metadata:
                simulation_metadata.append(metadata)

        if set(coverage) != {"normal", *kinds} or set(coverage.values()) != {engines * episodes_per_fault}:
            raise ValueError("Balanced primary episode coverage check failed")
        feature_columns = [column for column in tables["features"].columns if column not in IDENTIFIERS]
        shortcut_columns = {"context.scenario", "context.drive_cycle", "context.cycle_phase",
                            "context.payload_class", "context.operating_hours"}
        input_columns = [column for column in tables["readings"].columns
                         if column not in IDENTIFIERS and column not in shortcut_columns]
        supplied_definitions = getattr(analytics, "FEATURE_DEFINITIONS", {})
        definitions = {name: supplied_definitions.get(name, _feature_definition(name)) for name in feature_columns}
        sensor_units = simulation_metadata[0].get("units", {})
        context_units = {"mode": "category", "payload_class": "category", "start_condition": "category",
                         "cycle_phase": "category", "scenario": "category", "drive_cycle": "category",
                         "transient": "dimensionless"}
        input_units = {}
        for name in input_columns:
            group, signal = name.split(".", 1)
            if group == "available":
                input_units[name] = "0 or 1"
            elif group == "observation":
                input_units[name] = sensor_units.get(signal, _unit(signal))
            else:
                input_units[name] = context_units.get(signal, _unit(signal))
        manifest = {
            "schema_version": "enhanced-dataset/1", "source": "simulation",
            "physics_validated": False, "diagnostics_validated": False,
            "provenance": {"generator": "generate_enhanced_dataset.py", "simulation_module": "enhanced_simulation",
                           "analytics_module": "enhanced_analytics", "synthetic": True,
                           "source_sha256": {"simulation": _source_digest(simulation),
                                             "analytics": _source_digest(analytics),
                                             "generator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                           "seed_derivation": "SHA-256 of JSON master seed and stable engine/episode/waveform identifiers, first 64 bits reduced to uint32",
                           "seed": seed, "simulation_metadata": simulation_metadata,
                            "label_semantics": "fault_level is schedule envelope; effect_level includes delayed effects and persistent damage; fault_label is effect_level > 0.001. Neither is a confirmed field root cause or a guarantee of detectability.",
                           "time_scale": "wear_time_scale is sampled per episode and recorded in config; accelerated synthetic aging"},
            "parameters": {"engines": engines, "episodes_per_fault": episodes_per_fault,
                           "duration_s": duration_s, "timestep_s": timestep_s,
                            "waveforms": waveforms, "waveform_every": waveform_every,
                            "longitudinal": longitudinal},
            "join_keys": list(JOIN_KEYS), "identifier_columns": list(IDENTIFIERS),
            "input_columns": {"readings.csv": input_columns, "features.csv": feature_columns},
            "feature_definitions": definitions,
            "input_units": input_units,
            "exclusions": {"tables": ["labels.csv", "diagnostics.csv", "manifest.json"],
                            "columns": list(IDENTIFIERS) + sorted(shortcut_columns) + ["truth", "expected", "events", "fault_levels",
                                       "active_faults", "sensor_quality_json", "latent_wear", "event_phases",
                                       "hypotheses", "root_cause_confirmed", "rul_hours", "overall_health"],
                           "reason": "Identifiers are join-only; truth, simulator baselines, schedules, diagnostic outputs and categorical quality labels are not model inputs"},
            "fault_taxonomy": simulation.FAULT_CATALOG,
            "label_columns": {"multi_label_targets": [f"fault_label.{kind}" for kind in kinds],
                              "schedule_severity": [f"fault_level.{kind}" for kind in kinds],
                              "effect_severity": [f"effect_level.{kind}" for kind in kinds]},
            "lifecycle_semantics": "Independent fault experiments per truck identity" if not longitudinal else "Sequential wear/hours continuation; balanced primary classes describe new injections, not absence of earlier damage",
            "split_policy": "Deterministic 70/15/15 approximate engine-level split, minimum one engine per split; no row splitting",
            "split_membership": {name: sorted(engine for engine, split in splits.items() if split == name)
                                 for name in ("train", "validation", "test")},
            "balanced_class_episode_coverage": dict(sorted(coverage.items())),
            "coverage_semantics": "Equal single-class episode/event counts for every primary class per engine; sensor_variant and mixed supplements are separate and can change aggregate class frequencies",
            "episodes": episode_records,
            "tables": {f"{name}.csv": {"rows": table.count, "columns": table.columns} for name, table in tables.items()},
            "waveforms": {"file": "waveforms.jsonl" if waveforms else None,
                          "count": waveform_count, "sample_rate_hz": WAVEFORM_RATE_HZ,
                          "skipped_missing_source_measurements": waveform_skipped_missing,
                          "duration_s": WAVEFORM_DURATION_S, "units": "mm/s",
                          "synthetic_assumed_frequency_components": True,
                           "model_input": False, "join_keys": list(JOIN_KEYS),
                           "spectral_features_file": "spectral_features.csv" if waveforms else None,
                           "spectral_input_columns": [column for column in tables["spectral_features"].columns if column not in IDENTIFIERS] if waveforms else [],
                           "spectral_semantics": "FFT of optional high-rate synthetic snippets; assumed harmonic components, not validated fault frequencies. Missing snippets must be handled explicitly when joining."},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for table in tables.values():
            table.close()
        if waveform_handle is not None:
            waveform_handle.close()
        _commit(stage, output, overwrite)
        return {"output_dir": str(output), "engines": engines, "episodes": len(plans),
                "rows": tables["readings"].count, "waveforms": waveform_count,
                "split_membership": manifest["split_membership"],
                "balanced_class_episode_coverage": manifest["balanced_class_episode_coverage"]}
    finally:
        for table in tables.values():
            table.close()
        if waveform_handle is not None:
            waveform_handle.close()
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/enhanced"))
    parser.add_argument("--engines", type=int, default=12)
    parser.add_argument("--episodes-per-fault", type=int, default=2)
    parser.add_argument("--seed", type=int, default=793)
    parser.add_argument("--duration-s", type=float, default=1800)
    parser.add_argument("--timestep-s", type=float, default=5)
    parser.add_argument("--waveforms", action="store_true")
    parser.add_argument("--waveform-every", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--longitudinal", action="store_true", help="Carry each engine's wear and operating hours between episodes; can create additional persistent fault labels")
    args = parser.parse_args()
    try:
        summary = generate_dataset(**vars(args))
    except (ValueError, FileExistsError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
