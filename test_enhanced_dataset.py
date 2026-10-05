"""Exporter contract, leakage, split, reproducibility and publication checks."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from generate_enhanced_dataset import (
    ARTIFACTS, SENSOR_MODES, _plans, _splits, generate_dataset,
)


KINDS = (
    "engine_oil_pump_wear", "piston_ring_damage", "turbo_exhaust_fault",
    "coolant_system_failure", "coolant_degradation", "gearbox_seal_wear",
    "gearbox_bearing_wear", "hydraulic_normal_aging", "hydraulic_contamination",
    "bearing_failure", "low_oil", "overheat", "brake_wear", "perception_degraded",
    "sensor_fault", "unrelated_multi_blip",
)


@pytest.fixture
def fake_modules(monkeypatch):
    """Tiny dependency doubles exercise exporter failures, not simulation physics."""
    def simulate(config, events):
        samples = []
        for index in range(21):
            elapsed = index * config["timestep_s"]
            levels = {kind: 0.0 for kind in KINDS}
            for event in events:
                if event["onset_s"] <= elapsed < event["onset_s"] + event["duration_s"]:
                    levels[event["kind"]] = event["severity"]
            value = 75 + config["seed"] % 10
            samples.append({
                **{key: config[key] for key in ("engine_id", "episode_id", "base_trajectory_id")},
                "elapsed_s": elapsed, "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "context": {"rpm": 1000, "ambient_c": config["ambient_c"], "mode": "haul"},
                "observations": {"engine.oil_condition_pct": value},
                "expected": {"engine.oil_condition_pct": value},
                "quality": {"engine.oil_condition_pct": "good"},
                "truth": {"fault_levels": levels,
                          "active_faults": [kind for kind, level in levels.items() if level],
                          "event_phases": {}, "latent_wear": {}},
            })
        return {"samples": samples, "metadata": {"source": "test_double", "physics_validated": False}}

    def assert_observation_boundary(samples):
        assert all("truth" not in sample and "expected" not in sample and "events" not in sample for sample in samples)

    def features(samples):
        assert_observation_boundary(samples)
        return [{"sensor.engine.oil_condition_pct": sample["observations"]["engine.oil_condition_pct"],
                 "missing_mask.engine.oil_condition_pct": 0} for sample in samples]

    def diagnostics(samples):
        assert_observation_boundary(samples)
        return [{"elapsed_s": sample["elapsed_s"], "hypotheses": [], "overall_health": 95.0,
                 "root_cause_confirmed": False, "rul_hours": None} for sample in samples]

    simulation = SimpleNamespace(FAULT_KINDS=KINDS,
                                 FAULT_CATALOG={kind: {"subsystem": "test", "possible_cause": "test", "mitigation": "test"} for kind in KINDS},
                                 simulate_enhanced=simulate)
    analytics = SimpleNamespace(engineer_features=features, analyse_episode=diagnostics)
    monkeypatch.setitem(sys.modules, "enhanced_simulation", simulation)
    monkeypatch.setitem(sys.modules, "enhanced_analytics", analytics)
    return simulation, analytics


def _generate(path, **kwargs):
    return generate_dataset(path, engines=3, episodes_per_fault=1, duration_s=100, timestep_s=5, **kwargs)


def _hashes(path):
    return {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in ARTIFACTS if (path / name).exists()}


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("kwargs", [
    {"engines": 2}, {"engines": True}, {"engines": 3.0},
    {"episodes_per_fault": 0}, {"seed": -1}, {"seed": True},
    {"duration_s": float("nan")}, {"duration_s": float("inf")},
    {"timestep_s": 0}, {"timestep_s": 10}, {"waveforms": "yes"},
    {"overwrite": 1}, {"waveform_every": False},
])
def test_invalid_parameters_do_not_create_output(tmp_path, kwargs):
    parameters = {"engines": 3, "episodes_per_fault": 1, "duration_s": 100, "timestep_s": 5}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "invalid", **parameters)
    assert not (tmp_path / "invalid").exists()


def test_plan_has_balanced_primary_coverage_and_diverse_context():
    plans = _plans(3, 2, 793, 100, 5, KINDS)
    for engine in {plan["config"]["engine_id"] for plan in plans}:
        selected = [plan for plan in plans if plan["config"]["engine_id"] == engine]
        primary = [plan for plan in selected if plan["coverage_group"] == "balanced"]
        assert all(sum(plan["primary_class"] == kind for plan in primary) == 2 for kind in ("normal", *KINDS))
        assert {plan["events"][0]["sensor_mode"] for plan in selected if plan["coverage_group"] == "sensor_variant"} == set(SENSOR_MODES)
        assert sum(plan["coverage_group"] == "mixed" for plan in selected) == 2
        assert len({plan["config"]["engine_seed"] for plan in selected}) == 1
        assert len({plan["config"]["seed"] for plan in selected}) == len(selected)
        assert len({plan["config"]["base_trajectory_id"] for plan in selected}) == len(selected)
        assert len({plan["config"]["payload_class"] for plan in selected}) == 5
        assert len({plan["config"]["ambient_c"] for plan in selected}) > 5
        assert len({plan["config"]["noise_scale"] for plan in selected}) > 5
    assert len({plan["config"]["engine_seed"] for plan in plans}) == 3
    assert set(_splits(3, 793).values()) == {"train", "validation", "test"}


def test_export_join_keys_splits_inputs_and_determinism(tmp_path, fake_modules):
    first, second = tmp_path / "first", tmp_path / "second"
    summary = _generate(first)
    _generate(second)
    assert _hashes(first) == _hashes(second)
    manifest = json.loads((first / "manifest.json").read_text())
    assert summary["rows"] == 3 * 24 * 21
    assert set(manifest["balanced_class_episode_coverage"].values()) == {3}
    assert set(manifest["balanced_class_episode_coverage"]) == {"normal", *KINDS}
    memberships = list(manifest["split_membership"].values())
    assert set(memberships[0]).isdisjoint(memberships[1])
    assert set(memberships[0]).isdisjoint(memberships[2])
    assert set(memberships[1]).isdisjoint(memberships[2])
    key_sets = []
    for table in ("readings", "labels", "diagnostics", "features"):
        rows = _rows(first / f"{table}.csv")
        keys = [tuple(row[key] for key in manifest["join_keys"]) for row in rows]
        assert len(keys) == len(set(keys))
        key_sets.append(set(keys))
        for row in rows:
            assert row["engine_id"] in manifest["split_membership"][row["split"]]
    assert all(keys == key_sets[0] for keys in key_sets)
    for table, columns in manifest["input_columns"].items():
        assert all(not any(token in column for token in ("truth", "fault_level", "active_fault", "hypotheses", "rul_hours", "engine_id", "episode_id", "split")) for column in columns)
    assert set(manifest["feature_definitions"]) == set(manifest["input_columns"]["features.csv"])
    assert all("units" in value and "definition" in value for value in manifest["feature_definitions"].values())
    _generate(second, overwrite=True, seed=794)
    assert _hashes(first) != _hashes(second)


def test_no_overwrite_and_preserve_unrelated_file(tmp_path, fake_modules):
    output = tmp_path / "dataset"
    _generate(output)
    (output / "notes.txt").write_text("user content")
    original = _hashes(output)
    with pytest.raises(FileExistsError):
        _generate(output)
    assert _hashes(output) == original
    _generate(output, overwrite=True, seed=795)
    assert (output / "notes.txt").read_text() == "user content"
    assert _hashes(output) != original


@pytest.mark.parametrize("failure", ["nonfinite", "mask", "leakage", "length", "quality", "never_active"])
def test_failed_validation_leaves_previous_generation_intact(tmp_path, fake_modules, failure):
    simulation, analytics = fake_modules
    output = tmp_path / "dataset"
    _generate(output)
    original = _hashes(output)
    if failure in ("nonfinite", "mask", "leakage", "length"):
        original_features = analytics.engineer_features
        def broken(samples):
            rows = original_features(samples)
            if failure == "nonfinite":
                rows[0]["sensor.engine.oil_condition_pct"] = float("nan")
            elif failure == "mask":
                rows[0]["missing_mask.engine.oil_condition_pct"] = 2
            elif failure == "leakage":
                rows[0]["truth.fault_levels"] = 1
            else:
                rows.pop()
            return rows
        analytics.engineer_features = broken
    else:
        original_simulate = simulation.simulate_enhanced
        def broken(config, events):
            result = original_simulate(config, events)
            if failure == "quality":
                result["samples"][0]["observations"]["engine.oil_condition_pct"] = None
            else:
                for sample in result["samples"]:
                    sample["truth"]["fault_levels"] = dict.fromkeys(KINDS, 0.0)
                    sample["truth"]["active_faults"] = []
            return result
        simulation.simulate_enhanced = broken
    with pytest.raises(ValueError):
        _generate(output, overwrite=True)
    assert _hashes(output) == original
    assert not list(tmp_path.glob(".dataset-staging-*"))
    assert not list(tmp_path.glob(".dataset-backup-*"))


def test_bounds_and_output_path_validation(tmp_path, fake_modules):
    with pytest.raises(ValueError, match="sample rows"):
        generate_dataset(tmp_path / "large", engines=1000, episodes_per_fault=100)
    with pytest.raises(ValueError, match="waveforms exceed"):
        generate_dataset(tmp_path / "waves", engines=3, episodes_per_fault=1,
                         waveforms=True, waveform_every=1, duration_s=1000, timestep_s=5)
    path = tmp_path / "file"
    path.write_text("keep")
    with pytest.raises(ValueError):
        _generate(path)
    assert path.read_text() == "keep"


def test_real_enhanced_export_and_optional_waveforms(tmp_path):
    if any(importlib.util.find_spec(module) is None for module in ("enhanced_simulation", "enhanced_analytics")):
        pytest.skip("Shared enhanced simulation/analytics modules not yet available")
    output = tmp_path / "real"
    summary = _generate(output, waveforms=True, waveform_every=100)
    manifest = json.loads((output / "manifest.json").read_text())
    assert summary["waveforms"] == summary["episodes"]
    readings = _rows(output / "readings.csv")
    missing_count = 0
    for row in readings:
        for column, value in row.items():
            if column.startswith("available."):
                observation = "observation." + column.removeprefix("available.")
                assert value in ("0", "1")
                assert (value == "0") == (row[observation] == "")
                missing_count += value == "0"
    assert missing_count > 0
    qualities = {quality for row in _rows(output / "labels.csv")
                 for quality in json.loads(row["sensor_quality_json"]).values()}
    assert set(SENSOR_MODES) <= qualities
    with (output / "waveforms.jsonl").open() as handle:
        for line in handle:
            waveform = json.loads(line)
            assert len(waveform["waveform"]) == 1024
            assert waveform["units"] == "mm/s"
            assert waveform["engine_id"] in manifest["split_membership"][waveform["split"]]
    assert manifest["physics_validated"] is False
    assert manifest["diagnostics_validated"] is False
