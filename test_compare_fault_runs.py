import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.compare_fault_runs import compare


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def runs(tmp_path):
    targets = ["fault_label.brake_wear", "fault_label.perception_degraded"] + [f"fault_label.fault_{i}" for i in range(14)]
    inputs = ["lidar.points_per_scan", "radar.target_distance_m"] + [f"engine.input_{i}" for i in range(30)]
    directories = (tmp_path / "old", tmp_path / "new")
    for directory in directories:
        directory.mkdir()
        write_json(directory / "manifest.json", {
            "source_csv": "/original/data/unchanged.csv", "source_sha256": "same-csv-hash",
            "selected_model": "chosen", "selection_policy": "highest validation macro AP",
            "split_membership": {"train": ["truck1"], "validation": ["truck2", "truck3"], "test": ["truck4"]},
            "calibration_truck": "truck2"})
        write_json(directory / "feature_schema.json", {
            "source_input_columns": inputs, "model_input_columns": inputs,
            "selected_original_attributes": inputs[:16], "subsets": {"32": inputs, "16": inputs[:16]},
            "target_columns": targets})
        write_json(directory / "feature_selection/selected_preliminary_ranks.json", {
            "eligible_names": inputs, "pruned_names": inputs[:24]})
        write_csv(directory / "model_comparison.csv", [
            {"candidate": "chosen_raw", "split": "test", "macro_AP": .7, "macro_F1": .6},
            {"candidate": "chosen_sigmoid_calibrated", "split": "test", "macro_AP": .7, "macro_F1": "null"},
            {"candidate": "test_winner", "split": "test", "macro_AP": 1, "macro_F1": 1},
            {"candidate": "chosen_raw", "split": "validation_tuning", "macro_AP": .99, "macro_F1": .99}])
        write_csv(directory / "per_fault_metrics.csv", [
            {"candidate": "chosen_raw", "split": "test", "fault": targets[0], "AP": .1, "F1": 0, "recall": ""},
            {"candidate": "chosen_sigmoid_calibrated", "split": "test", "fault": targets[1], "AP": .8, "F1": .8, "recall": .8}])
    return directories


def snapshot(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_report_preserves_inputs_and_manifest_selection(runs):
    old, new = runs
    old_before, new_before = snapshot(old), snapshot(new)
    # The old run may contain excluded input names; only the new run must exclude them.
    schema = json.loads((old / "feature_schema.json").read_text())
    schema["source_input_columns"].append("gnss.speed_mps")
    write_json(old / "feature_schema.json", schema)
    old_before = snapshot(old)
    result = compare(old, new)
    assert snapshot(old) == old_before
    assert all((new / name).read_bytes() == value for name, value in new_before.items())
    assert result["source_csv_paths"] == {"old": "/original/data/unchanged.csv", "new": "/original/data/unchanged.csv"}
    assert result["selected_models"] == {"old": "chosen", "new": "chosen"}
    assert result["selected_test_scores"]["raw"]["old"]["macro_AP"] == .7
    assert result["features"]["new"]["counts"]["source_input_columns"] == 32
    assert result["features"]["new"]["counts"]["eligible_names"] == 32
    assert result["features"]["new"]["counts"]["pruned_names"] == 24
    assert result["features"]["new"]["selected_original_attributes"] == schema["selected_original_attributes"]
    assert result["comparability"]["comparable"]
    assert len(result["per_fault"]) == 32  # 16 targets, two calibration phases
    assert "fault_label.brake_wear" in result["weak_classes"]
    saved = json.loads((new / "INPUT_EXCLUSION_COMPARISON.json").read_text())
    assert saved == result
    text = (new / "INPUT_EXCLUSION_COMPARISON.md").read_text()
    for phrase in ("same held-trucks", "not a newly untouched test", "unsupported", "LiDAR/radar", "RUL is not validated", "prior run already tested"):
        assert phrase in text
    assert not list(new.glob(".*.tmp"))
    assert compare(old, new) == result


@pytest.mark.parametrize("branch", ["source_input_columns", "model_input_columns", "selected_original_attributes", "subsets"])
@pytest.mark.parametrize("bad", ["GNSS.latitude_deg__available_mask", "engine.gps_speed", "leadlag.brake.pressure__to__engine.oil", "context.braking"])
def test_excluded_branch_detected_before_any_output(runs, branch, bad):
    old, new = runs
    schema = json.loads((new / "feature_schema.json").read_text())
    if branch == "subsets":
        schema[branch]["nested"] = {"8": [bad]}
    else:
        schema[branch].append(bad)
    write_json(new / "feature_schema.json", schema)
    before = snapshot(new)
    with pytest.raises(ValueError, match="Excluded input names"):
        compare(old, new)
    assert snapshot(new) == before


def test_missing_scores_and_optional_artifacts_are_not_zero(runs):
    old, new = runs
    (new / "per_fault_metrics.csv").unlink()
    (new / "feature_selection/selected_preliminary_ranks.json").unlink()
    result = compare(old, new)
    score = result["selected_test_scores"]["sigmoid_calibrated"]
    assert score["new"]["macro_F1"] is None
    assert score["delta_new_minus_old"]["macro_F1"] is None
    assert "macro_F1" in score["missing_metrics"]["new"]
    assert result["per_fault"][0]["new"]["F1"] is None
    assert result["per_fault"][0]["old"]["F1"] == 0
    assert result["features"]["new"]["counts"]["eligible_names"] is None
    assert result["independent_seed_comparison"]["scores"]["new"]["macro_AP"] is None
    assert "missing" in (new / "INPUT_EXCLUSION_COMPARISON.md").read_text()


def test_numeric_deltas_nonfinite_scores_and_unsuffixed_raw(runs):
    old, new = runs
    write_csv(new / "model_comparison.csv", [
        {"candidate": "chosen", "split": "test", "macro_AP": .8, "macro_F1": .65, "micro_AP": "NaN"},
        {"candidate": "chosen_sigmoid_calibrated", "split": "test", "macro_AP": "Infinity", "macro_F1": "", "micro_AP": ""}])
    result = compare(old, new)
    raw = result["selected_test_scores"]["raw"]
    assert raw["new_candidate"] == "chosen"
    assert raw["delta_new_minus_old"]["macro_AP"] == pytest.approx(.1)
    assert raw["delta_new_minus_old"]["macro_F1"] == pytest.approx(.05)
    assert raw["new"]["micro_AP"] is None
    assert result["selected_test_scores"]["sigmoid_calibrated"]["new"]["macro_AP"] is None
    assert json.loads((new / "INPUT_EXCLUSION_COMPARISON.json").read_text()) == result


@pytest.mark.parametrize("difference", ["hash", "splits", "targets", "calibration"])
def test_noncomparable_evidence_withholds_deltas(runs, difference):
    old, new = runs
    manifest = json.loads((new / "manifest.json").read_text())
    if difference == "hash":
        manifest["source_sha256"] = "different"
    elif difference == "splits":
        manifest["split_membership"]["test"] = ["another-truck"]
    elif difference == "calibration":
        manifest["calibration_truck"] = "truck3"
    else:
        schema = json.loads((new / "feature_schema.json").read_text())
        schema["target_columns"].reverse()
        write_json(new / "feature_schema.json", schema)
    write_json(new / "manifest.json", manifest)
    result = compare(old, new)
    assert not result["comparability"]["comparable"]
    assert result["selected_test_scores"]["raw"]["new"]["macro_AP"] == .7
    assert result["selected_test_scores"]["raw"]["delta_new_minus_old"]["macro_AP"] is None


def test_reused_independent_seed_and_guarded_results_are_separate(runs):
    old, new = runs
    for directory in runs:
        write_json(directory / "independent_seed_validation/RESULTS.json", {
            "seed": 20261006, "rows": 8, "independent_engine_variant_seeds": [71, 82],
            "receipt": {"sha256": {"independent_fault_batch.csv": "same-batch"}},
            "weights_and_thresholds_frozen_before_generation": True, "training_or_tuning_performed": False,
            "score": {"candidate": "frozen_alarm_guarded_model", "split": "fresh_seed_external_all_rows", "macro_AP": .8}})
        write_csv(directory / "threshold_refinement_comparison.csv", [{
            "candidate": "normal_guard_1pct_per_fault", "split": "test_second_use_exploratory", "macro_AP": .9, "macro_F1": .95}])
    result = compare(old, new)
    independent = result["independent_seed_comparison"]
    assert independent["comparable"] and independent["seed_reused"]
    assert "not fresh unseen" in independent["evidence_label"]
    assert independent["scores"]["old"]["macro_AP"] == .8
    assert result["selected_test_scores"]["raw"]["new"]["macro_AP"] == .7
    assert result["guarded_threshold_comparison"]["rows"][0]["new"]["macro_AP"] == .9
    assert "second-use" in result["guarded_threshold_comparison"]["evidence_label"]
    record = json.loads((new / "independent_seed_validation/RESULTS.json").read_text())
    record["seed"] += 1
    write_json(new / "independent_seed_validation/RESULTS.json", record)
    result = compare(old, new)
    assert not result["independent_seed_comparison"]["comparable"]
    assert result["independent_seed_comparison"]["scores"]["delta_new_minus_old"]["macro_AP"] is None


@pytest.mark.parametrize("prefix", ["../old/REPORT", "../escape", "manifest", "feature_schema"])
def test_output_cannot_overwrite_old_or_inputs(runs, prefix):
    old, new = runs
    before = snapshot(old), snapshot(new)
    with pytest.raises(ValueError):
        compare(old, new, prefix)
    assert (snapshot(old), snapshot(new)) == before


def test_custom_prefix_and_missing_input_branch(runs):
    old, new = runs
    result = compare(old, new, new / "custom_report")
    assert Path(result["output_files"]["json"]) == new / "custom_report.json"
    assert compare(old, new, "custom_report.md") == result
    schema = json.loads((new / "feature_schema.json").read_text())
    del schema["subsets"]
    write_json(new / "feature_schema.json", schema)
    before = snapshot(new)
    with pytest.raises(ValueError, match="missing required input branch"):
        compare(old, new)
    assert snapshot(new) == before


def test_cli_writes_the_same_report(runs):
    old, new = runs
    script = Path(__file__).parent / "analysis/compare_fault_runs.py"
    process = subprocess.run([sys.executable, str(script), str(old), str(new)], capture_output=True, text=True, check=True)
    paths = json.loads(process.stdout)
    assert Path(paths["markdown"]).is_file()
    saved = json.loads(Path(paths["json"]).read_text())
    assert saved["selected_models"] == {"old": "chosen", "new": "chosen"}
