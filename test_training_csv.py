"""Real small journeys plus publication/failure doubles for the training CSV."""

from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

import enhanced_analytics as analytics
import enhanced_simulation as simulation
import generate_training_csv as exporter


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def real_dataset(tmp_path_factory):
    directory = tmp_path_factory.mktemp("training-real")
    path = directory / "fixture.csv"
    seen = Counter()
    original_features, original_diagnostics = analytics.engineer_features, analytics.analyse_episode

    def boundary(samples, function, name):
        seen[name] += 1
        for sample in samples:
            assert set(sample) == {"engine_id", "episode_id", "elapsed_s", "observations", "context"}
            assert not {"truth", "expected", "quality", "events", "raw_legacy"} & sample.keys()
            assert not {"scenario", "payload_class", "cycle_phase", "operating_hours"} & sample["context"].keys()
        return function(samples)

    with patch.object(analytics, "engineer_features", side_effect=lambda samples: boundary(samples, original_features, "features")), \
         patch.object(analytics, "analyse_episode", side_effect=lambda samples: boundary(samples, original_diagnostics, "diagnostics")):
        receipt = exporter.generate_training_csv(path, engines=3, episodes_per_fault=1, samples_per_journey=21)
    assert seen == {"features": 93, "diagnostics": 93}
    return path, receipt, json.loads(path.with_suffix(".schema.json").read_text()), json.loads(path.with_suffix(".manifest.json").read_text())


def test_default_contract_complete_cartesian_context_and_pairs():
    plans = list(exporter._plans(20, 2, 793, 1800, 1800 / 170, simulation.FAULT_KINDS))
    assert len(plans) == 960
    assert len(plans) * 171 == 164160
    assert Counter(plan["coverage_group"] for plan in plans) == {"primary": 680, "sensor_variant": 100, "mixed_pair": 120, "mixed_triple": 60}
    expected_contexts = set(itertools.product(exporter.SCENARIOS, exporter.PAYLOAD_CLASSES, exporter.START_CONDITIONS))
    for kind in ("normal", *simulation.FAULT_KINDS):
        selected = [plan for plan in plans if plan["coverage_group"] == "primary" and plan["primary_class"] == kind]
        assert len(selected) == 40
        assert {tuple(plan["condition"]) for plan in selected} == expected_contexts
        assert set(Counter(tuple(plan["condition"]) for plan in selected).values()) == {1, 2}
    pairs = [tuple(sorted(event["kind"] for event in plan["events"])) for plan in plans if plan["coverage_group"] == "mixed_pair"]
    assert len(pairs) == len(set(pairs)) == 120
    assert set(pairs) == {tuple(sorted(pair)) for pair in itertools.combinations(simulation.FAULT_KINDS, 2)}
    triples = [tuple(sorted(event["kind"] for event in plan["events"])) for plan in plans if plan["coverage_group"] == "mixed_triple"]
    assert len(set(triples)) == 60
    assert len(set(triples)) > 3
    for engine in {plan["config"]["engine_id"] for plan in plans}:
        selected = [plan for plan in plans if plan["config"]["engine_id"] == engine]
        assert len(selected) == 48
        assert len({plan["config"]["engine_seed"] for plan in selected}) == 1
        assert len({plan["config"]["seed"] for plan in selected}) == 48
        assert {plan["events"][0]["sensor_mode"] for plan in selected if plan["coverage_group"] == "sensor_variant"} == set(exporter.SENSOR_MODES)
        for parameter in ("ambient_c", "initial_wear", "noise_scale", "wear_time_scale"):
            assert len({plan["config"][parameter] for plan in selected}) > 40
    assert len({plan["config"]["engine_seed"] for plan in plans}) == 20
    assert len({event["target_signal"] for plan in plans for event in plan["events"] if event["kind"] == "sensor_fault"}) == len(exporter.SENSOR_TARGETS)


def test_recurring_events_are_real_intervals_and_fit_every_occurrence():
    plans = list(exporter._plans(20, 2, 793, 1800, 1800 / 170, simulation.FAULT_KINDS))
    recurring = []
    for plan in plans:
        for event in plan["events"]:
            assert event["duration_s"] >= 1800 / 170
            assert event["recovery_s"] >= 1800 / 170
            onset = event["onset_s"]
            period = event.get("recurrence_s")
            if period:
                recurring.append(event)
                assert period >= event["duration_s"] + event["recovery_s"]
                starts = [onset + period * index for index in range(10) if onset + period * index < 1800]
                assert len(starts) == 2
            else:
                starts = [onset]
            assert all(start + event["duration_s"] + event["recovery_s"] <= 1800 for start in starts)
    assert recurring


def test_real_counts_schema_splits_missingness_and_digest_receipt(real_dataset):
    path, receipt, schema, manifest = real_dataset
    assert receipt["episodes"] == manifest["counts"]["episodes"] == 93
    assert receipt["rows"] == manifest["counts"]["rows"] == 1953
    assert manifest["counts"]["episodes_per_engine"] == 31
    assert set(manifest["primary_class_episode_coverage"].values()) == {3}
    assert manifest["pair_coverage"]["distinct_pairs"] == 18
    assert manifest["pair_coverage"]["complete"] is False
    assert manifest["triple_coverage"]["distinct_triples"] == 9
    assert len(schema["input_columns"]) == 369
    assert len(schema["raw_columns"]) == len(schema["availability_mask_columns"]) == 113
    assert set(schema["raw_columns"]) <= set(schema["input_columns"])
    assert len(schema["target_columns"]) == 16
    assert schema["target_columns"] == ["fault_label." + kind for kind in simulation.FAULT_KINDS]
    assert "is_normal" not in schema["target_columns"]
    assert set(schema["units"]) == set(schema["columns"])
    assert set(schema["feature_definitions"]) == set(schema["input_columns"])
    groups = [set(schema[name]) for name in ("input_columns", "metadata_columns", "label_columns", "diagnostic_columns")]
    assert set().union(*groups) == set(schema["columns"])
    assert sum(map(len, groups)) == len(schema["columns"])
    assert set(schema["exclusions"]["columns"]) == set().union(*groups[1:])
    assert not any(token in name for name in schema["input_columns"]
                   for token in ("quality", "truth", "expected", "fault_label", "event", "is_normal", "health", "rul"))
    memberships = manifest["split_membership"]
    assert all(memberships[name] for name in ("train", "validation", "test"))
    assert sum(map(len, memberships.values())) == len(set().union(*map(set, memberships.values()))) == 3
    seen, missing, positive_rows = defaultdict(list), 0, Counter()
    for row in _rows(path):
        assert row["engine_id"] in memberships[row["split"]]
        seen[row["episode_id"]].append((int(row["step"]), float(row["elapsed_s"])))
        assert row["root_cause_confirmed"] == "0"
        assert row["rul_status"] == "unavailable_not_validated"
        assert int(row["is_normal"]) == int(not any(int(row[name]) for name in schema["target_columns"]))
        for kind in simulation.FAULT_KINDS:
            assert int(row["fault_label." + kind]) == int(float(row["effect_level." + kind]) > .001)
            positive_rows[kind] += int(row["fault_label." + kind])
        for name in schema["input_columns"]:
            if row[name]:
                assert math.isfinite(float(row[name]))
        for name in schema["raw_columns"]:
            if row[name] == "":
                missing += 1
                assert row[name + "__available_mask"] == "0"
                assert json.loads(row["sensor_quality_json"])[name] == "missing"
    assert len(seen) == 93
    assert all(points == [(index, index * 90) for index in range(21)] for points in seen.values())
    assert missing > 0
    assert dict(positive_rows) == manifest["positive_label_row_counts"]
    for name, digest in receipt["sha256"].items():
        assert _sha(path.parent / name) == digest
    for name, info in manifest["files"].items():
        assert _sha(path.parent / name) == info["sha256"]
    assert all(len(value) == 64 for value in manifest["source_hashes"].values())


def test_real_episode_maps_public_features_truth_and_diagnostics_exactly(real_dataset):
    path, _, schema, manifest = real_dataset
    # Select a sensor-missing journey and a mechanical journey, not just healthy
    # placeholders. Recompute directly from the manifest's reproducible schedule.
    selected = [next(record for record in manifest["episodes"] if record["primary_class"] == "engine_oil_pump_wear"),
                next(record for record in manifest["episodes"] if record["coverage_group"] == "sensor_variant" and record["events"][0]["sensor_mode"] == "missing")]
    for record in selected:
        result = simulation.simulate_enhanced(record["config"], record["events"])
        observed = exporter._observed_samples(result["samples"], analytics)
        features, diagnostics = analytics.engineer_features(observed), analytics.analyse_episode(observed)
        rows = [row for row in _rows(path) if row["episode_id"] == record["config"]["episode_id"]]
        assert len(rows) == len(result["samples"]) == 21
        for row, sample, feature, diagnostic in zip(rows, result["samples"], features, diagnostics):
            for name in schema["input_columns"]:
                assert row[name] == ("" if feature[name] is None else str(feature[name]))
            for kind in simulation.FAULT_KINDS:
                assert float(row["fault_level." + kind]) == sample["truth"]["fault_levels"][kind]
                assert float(row["effect_level." + kind]) == sample["truth"]["effect_levels"][kind]
            assert json.loads(row["event_phases_json"]) == sample["truth"]["event_phases"]
            assert json.loads(row["event_occurrences_json"]) == sample["truth"]["event_occurrences"]
            sparse_quality = json.loads(row["sensor_quality_json"])
            assert {name: sparse_quality.get(name, "good") for name in schema["raw_columns"]} == sample["quality"]
            probable = json.loads(row["probable_faults_json"])
            assert [entry["fault"] for entry in probable] == [hypothesis["fault"] for hypothesis in diagnostic["hypotheses"]]
            assert all(entry["source"] == "enhanced_analytics.analyse_episode" and entry["root_cause_confirmed"] is False for entry in probable)
            active = [kind for kind in simulation.FAULT_KINDS if sample["truth"]["effect_levels"][kind] > .001]
            for field, column in (("possible_cause", "possible_causes_json"), ("mitigation", "mitigations_json")):
                entries = json.loads(row[column])
                assert [entry["fault"] for entry in entries] == active
                assert all(entry[field] == simulation.FAULT_CATALOG[entry["fault"]][field]
                           and entry["source"].startswith("synthetic_catalogue_lookup:")
                           and entry["basis"] == "injected_effect_label"
                           and entry["root_cause_confirmed"] is False for entry in entries)


def test_reproducible_complete_bytes_at_different_paths(tmp_path, real_dataset):
    path, receipt, _, _ = real_dataset
    repeat = exporter.generate_training_csv(tmp_path / path.name, engines=3, episodes_per_fault=1, samples_per_journey=21)
    assert repeat["sha256"] == receipt["sha256"]
    assert repeat["split_membership"] == receipt["split_membership"]


@pytest.mark.parametrize("engines,expected", [(3, (1, 1, 1)), (12, (8, 2, 2)), (20, (14, 3, 3)), (30, (22, 4, 4))])
def test_splits_use_round_and_are_disjoint(engines, expected):
    splits = exporter._splits(engines, 793)
    counts = Counter(splits.values())
    assert tuple(counts[name] for name in ("train", "validation", "test")) == expected
    assert len(splits) == engines
    assert splits == exporter._splits(engines, 793)


@pytest.mark.parametrize("kwargs", [
    {"engines": 2}, {"engines": True}, {"engines": 3.0}, {"engines": 1001},
    {"episodes_per_fault": 0}, {"episodes_per_fault": True}, {"episodes_per_fault": 101},
    {"samples_per_journey": 20}, {"samples_per_journey": 10001}, {"samples_per_journey": True},
    {"duration_s": 0}, {"duration_s": float("nan")}, {"duration_s": float("inf")}, {"duration_s": True}, {"duration_s": 10**1000},
    {"seed": -1}, {"seed": True}, {"seed": 2**32}, {"overwrite": 1},
    {"engines": 1000, "episodes_per_fault": 100},
])
def test_validation_and_caps_precede_imports_and_writes(tmp_path, kwargs):
    path = tmp_path / "absent" / "fixture.csv"
    with patch.object(simulation, "simulate_enhanced", side_effect=AssertionError("simulation must not run")):
        with pytest.raises(ValueError):
            exporter.generate_training_csv(path, **kwargs)
    assert not path.parent.exists()


@pytest.mark.parametrize("path", ["", " ", "fixture.txt", None, 42])
def test_invalid_paths(path):
    with pytest.raises(ValueError):
        exporter.generate_training_csv(path)


@pytest.mark.parametrize("suffix", [".csv", ".schema.json", ".manifest.json"])
def test_existing_any_artifact_is_refused_before_simulation(tmp_path, suffix):
    path = tmp_path / "fixture.csv"
    existing = path.with_suffix(suffix)
    existing.write_text("original user artifact")
    with patch.object(simulation, "simulate_enhanced", side_effect=AssertionError("must not simulate")):
        with pytest.raises(FileExistsError):
            exporter.generate_training_csv(path, engines=3, episodes_per_fault=1, samples_per_journey=21)
    assert existing.read_text() == "original user artifact"
    assert list(tmp_path.iterdir()) == [existing]


def test_symlink_and_directory_outputs_are_refused(tmp_path):
    original = tmp_path / "original.csv"
    original.write_text("original")
    link = tmp_path / "link.csv"
    link.symlink_to(original)
    for path in (link, tmp_path / "directory.csv"):
        if path.name == "directory.csv":
            path.mkdir()
        with pytest.raises(ValueError):
            exporter.generate_training_csv(path, overwrite=True)
    assert original.read_text() == "original"


@pytest.mark.parametrize("failure", ["leakage", "nonfinite", "mask", "length", "raw_changed", "join"])
def test_failed_replacement_preserves_old_files_and_cleans_stage(tmp_path, failure):
    path = tmp_path / "fixture.csv"
    paths = [path, path.with_suffix(".schema.json"), path.with_suffix(".manifest.json")]
    for item in paths:
        item.write_text("old " + item.name)
    original = {item: item.read_bytes() for item in paths}
    original_features, original_diagnostics = analytics.engineer_features, analytics.analyse_episode

    def features(samples):
        rows = original_features(samples)
        if failure == "leakage":
            rows[0]["truth.numeric_label"] = rows[0].pop("context.ambient_c")
        elif failure == "nonfinite":
            rows[0]["context.ambient_c"] = float("nan")
        elif failure == "mask":
            rows[0][analytics.SENSOR_COLUMNS[0] + "__available_mask"] = .3
        elif failure == "length":
            rows.pop()
        elif failure == "raw_changed":
            rows[0]["engine.rpm"] += 1
        return rows

    def diagnostics(samples):
        rows = original_diagnostics(samples)
        if failure == "join":
            rows[0]["elapsed_s"] += 1
        return rows

    with patch.object(analytics, "engineer_features", side_effect=features), \
         patch.object(analytics, "analyse_episode", side_effect=diagnostics):
        with pytest.raises(ValueError):
            exporter.generate_training_csv(path, engines=3, episodes_per_fault=1, samples_per_journey=21, overwrite=True)
    assert {item: item.read_bytes() for item in paths} == original
    assert set(tmp_path.iterdir()) == set(paths)


def test_overwrite_success_and_publish_failure_rollback(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    paths = tuple(tmp_path / ("fixture" + suffix) for suffix in (".csv", ".schema.json", ".manifest.json"))
    for path in paths:
        path.write_text("old")
        (stage / path.name).write_text("new")
    original_replace = Path.replace

    def fail_last(source, target):
        if source == stage / paths[-1].name:
            raise OSError("publishing fault")
        return original_replace(source, target)

    with patch.object(Path, "replace", fail_last):
        with pytest.raises(OSError, match="publishing fault"):
            exporter._publish(stage, paths, True)
    assert all(path.read_text() == "old" for path in paths)
    # Recreate staged files moved during the attempted transaction.
    for path in paths:
        (stage / path.name).write_text("new")
    exporter._publish(stage, paths, True)
    assert all(path.read_text() == "new" for path in paths)


def test_api_overwrite_replaces_all_receipts_and_preserves_user_files(tmp_path, real_dataset):
    source, _, _, _ = real_dataset
    path = tmp_path / source.name
    for suffix in (".csv", ".schema.json", ".manifest.json"):
        destination = path.with_suffix(suffix)
        destination.write_bytes(source.with_suffix(suffix).read_bytes())
    notes = tmp_path / "notes.txt"
    notes.write_text("user notes")
    old = _sha(path)
    receipt = exporter.generate_training_csv(path, engines=3, episodes_per_fault=1,
                                             samples_per_journey=21, seed=794, overwrite=True)
    assert _sha(path) != old
    assert receipt["rows"] == 1953
    assert json.loads(path.with_suffix(".manifest.json").read_text())["parameters"]["seed"] == 794
    assert notes.read_text() == "user notes"
    assert set(tmp_path.iterdir()) == {path, path.with_suffix(".schema.json"), path.with_suffix(".manifest.json"), notes}


def test_float_endpoint_contract_171_samples_from_real_episode():
    plan = next(exporter._plans(3, 1, 793, 1800, 1800 / 170, simulation.FAULT_KINDS))
    result = simulation.simulate_enhanced(plan["config"], plan["events"])
    assert len(result["samples"]) == 171
    assert result["samples"][0]["elapsed_s"] == 0
    assert result["samples"][-1]["elapsed_s"] == 1800
    exporter._normalise_times(result["samples"], 171, 1800 / 170, 1800)
    assert all(math.isclose(sample["elapsed_s"], index * (1800 / 170), abs_tol=1e-12)
               for index, sample in enumerate(result["samples"]))


def test_cli_equivalent_arguments_and_defaults(capsys):
    receipt = {"rows": 1953}
    argv = ["generate_training_csv.py", "--output", "custom.csv", "--engines", "3",
            "--episodes-per-fault", "1", "--samples", "21", "--duration-s", "90", "--seed", "5", "--overwrite"]
    with patch("sys.argv", argv), patch.object(exporter, "generate_training_csv", return_value=receipt) as generate:
        exporter.main()
    generate.assert_called_once_with(output_path=Path("custom.csv"), engines=3, episodes_per_fault=1,
                                     samples_per_journey=21, duration_s=90.0, seed=5, overwrite=True)
    assert json.loads(capsys.readouterr().out) == receipt
    with patch("sys.argv", ["generate_training_csv.py"]), patch.object(exporter, "generate_training_csv", return_value=receipt) as generate:
        exporter.main()
    generate.assert_called_once_with(output_path=exporter.DEFAULT_OUTPUT, engines=20, episodes_per_fault=2,
                                     samples_per_journey=171, duration_s=1800, seed=793, overwrite=False)
