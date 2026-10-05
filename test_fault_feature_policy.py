"""Name-only input ablation: source coverage, dependencies and safe boundaries."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("sklearn", reason="Dependency-aware exclusions use the isolated ML environment")

from analysis.fault_feature_policy import (
    assert_allowed_names,
    excluded_feature_reason,
    filter_input_names,
)


def _source_inputs():
    path = Path(__file__).parent / "datasets/dump_truck_enhanced_all_conditions.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))["input_columns"]


def test_default_source_schema_and_complete_enabled_ablation():
    names = _source_inputs()
    original = names.copy()
    assert len(names) == 369
    default, removed = filter_input_names(names)
    assert default == names and default is not names and removed == []
    kept, removed = filter_input_names(names, exclude_gnss_brakes=True)
    assert len(kept) == 318
    assert len(removed) == 51
    assert all(set(record) == {"attribute", "reason"} and record["reason"] for record in removed)
    removed_names = [record["attribute"] for record in removed]
    assert removed_names == [name for name in names if name not in kept]
    assert kept == [name for name in names if name not in removed_names]
    assert {"gnss.longitude_deg", "gnss.latitude_deg", "context.mode_braking"} <= set(removed_names)
    assert names == original
    assert_allowed_names(kept)


def test_explicit_inputs_only_no_labels_or_data_read(monkeypatch):
    # A policy accepts only names. It neither discovers columns nor reads the
    # CSV, targets, fault labels, diagnostic truth or model artifacts.
    from analysis import fault_features

    def unexpected(*args, **kwargs):
        pytest.fail("Name policy attempted file access or label-based feature selection")

    monkeypatch.setattr(Path, "open", unexpected)
    monkeypatch.setattr(fault_features, "select_features", unexpected)
    names = ["engine.coolant_flow_kg_s", "gps.latitude_deg", "imu.pitch_deg"]
    kept, removed = filter_input_names(names, True)
    assert kept == [names[0], names[2]]
    assert [row["attribute"] for row in removed] == [names[1]]
    assert filter_input_names([], True) == ([], [])


@pytest.mark.parametrize("base", ["gnss.longitude_deg", "GPS.latitude_deg", "brake.stroke_pct",
                                  "brakes.line_pressure_bar", "context.mode_braking"])
@pytest.mark.parametrize("suffix", ["", "__available_mask", "__residual", "__rate",
                                    "__residual_slope_120s", "__mean_30s", "__std_30s"])
def test_raw_masks_residuals_rates_windows_and_onehot(base, suffix):
    name = base + suffix
    assert excluded_feature_reason(name)
    assert filter_input_names([name], True)[0] == []
    with pytest.raises(ValueError, match="Excluded GNSS/brake inputs"):
        assert_allowed_names([name])


@pytest.mark.parametrize("name", [
    "leadlag.brake.disc_temp_c__to__engine.oil_temp_c_s",
    "leadlag.engine.oil_temp_c__to__gps.speed_mps_s",
    "engine.braking_torque_nm__mean_30s", "window.GNSS.accuracy_m__std_30s",
])
def test_unknown_direct_names_and_both_leadlag_endpoints(name):
    assert excluded_feature_reason(name)


@pytest.mark.parametrize("name", [
    "engine.coolant_flow_kg_s", "engine.coolant_temp_c__residual",
    "imu.pitch_deg", "lidar.obstacle_confidence_pct__residual", "radar.target_confidence_pct",
    "dss.fatigue_score_pct", "suspension.actuator_vibration_mm_s__std_30s",
    "context.cold_start", "context.mode_loaded", "engine.gpsignal_strength",
    "engine.brakefluidity", "engine.unbraked_flow", "leadlag.unrecognized_feature",
])
def test_unrelated_inputs_and_unknown_names_are_preserved(name):
    assert excluded_feature_reason(name) is None
    assert filter_input_names([name], True) == ([name], [])
    assert assert_allowed_names([name]) is None


@pytest.mark.parametrize("kind, dependency", [
    ("sensors", "GPS.speed_mps"), ("sensors", "brakes.pedal_pct"),
    ("sensors", "vehicle.braking_torque_nm"), ("context", "context.mode_braking"),
])
def test_indirect_dependencies_excluded_even_with_innocuous_name(monkeypatch, kind, dependency):
    from analysis import fault_features

    name = "engine.coolant_flow_kg_s"
    monkeypatch.setattr(fault_features, "sensor_dependencies", lambda _: {
        "sensors": [], "context": [], kind: [dependency],
    })
    reason = excluded_feature_reason(name)
    assert "dependency" in reason and dependency in reason
    assert filter_input_names([name], True) == ([], [{"attribute": name, "reason": reason}])
    with pytest.raises(ValueError, match="engine.coolant_flow_kg_s"):
        assert_allowed_names([name])


def test_unknown_nonmatching_names_do_not_enter_dependency_parser(monkeypatch):
    from analysis import fault_features

    def unexpected(_):
        pytest.fail("Unknown feature passed to dependency parser")

    monkeypatch.setattr(fault_features, "sensor_dependencies", unexpected)
    assert excluded_feature_reason("leadlag.unrecognized_feature") is None


def test_default_and_direct_detection_do_not_import_feature_training_stack():
    script = """
import sys
from analysis.fault_feature_policy import filter_input_names, excluded_feature_reason
assert 'analysis.fault_features' not in sys.modules
assert filter_input_names(['engine.coolant_flow_kg_s', 'gnss.latitude_deg'])[1] == []
assert excluded_feature_reason('gps.latitude_deg')
assert 'analysis.fault_features' not in sys.modules
assert 'numpy' not in sys.modules
assert 'sklearn' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parent, check=True)


def test_stable_order_no_mutation_and_iterable_support():
    names = ["imu.pitch_deg", "gnss.longitude_deg", "engine.coolant_flow_kg_s",
             "context.mode_braking", "radar.target_confidence_pct"]
    original = names.copy()
    expected = filter_input_names(names, True)
    assert expected[0] == [names[0], names[2], names[4]]
    assert [record["attribute"] for record in expected[1]] == [names[1], names[3]]
    assert filter_input_names(iter(names), True) == expected
    assert names == original
    expected[0].append("extra")
    assert names == original


@pytest.mark.parametrize("names", [
    ["gnss.latitude_deg", "gnss.latitude_deg"], ["engine.coolant_flow_kg_s"] * 2,
    [""], [" "], ["engine. bad"], [None], [1], None, "engine.oil_temp_c",
    {"engine.oil_temp_c"}, {"engine.oil_temp_c": 1},
])
def test_invalid_and_duplicate_names_rejected_in_all_modes(names):
    for enabled in (False, True):
        with pytest.raises(ValueError):
            filter_input_names(names, enabled)
    with pytest.raises(ValueError):
        assert_allowed_names(names)


@pytest.mark.parametrize("name", ["", " ", None, 1])
def test_invalid_single_name(name):
    with pytest.raises(ValueError):
        excluded_feature_reason(name)


def test_exclusion_flag_must_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        filter_input_names([], "false")
