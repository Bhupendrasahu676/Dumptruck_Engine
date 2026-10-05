import copy
import json
import math

import pytest

from enhanced_analytics import analyse_episode, engineer_features
from enhanced_simulation import simulate_enhanced
from generate_enhanced_dataset import generate_dataset
from vibration_analysis import spectral_features


CONFIG = {"duration_s": 600, "timestep_s": 5, "noise_scale": 0}


def episode(kind=None, **config):
    events = [] if kind is None else [{"kind": kind, "onset_s": 100, "duration_s": 300,
                                     "severity": 1., "progression": .1, "recovery_s": 60}]
    return simulate_enhanced({**CONFIG, **config}, events)


def test_hidden_truth_expected_and_quality_labels_do_not_affect_inference_or_features():
    samples = episode("piston_ring_damage")["samples"]
    original = copy.deepcopy(samples)
    changed = copy.deepcopy(samples)
    for row in changed:
        row["truth"] = {"active_faults": ["fake_fault"], "fault_levels": {"fake_fault": 1}}
        row["expected"] = {k: 1e9 for k in row["observations"]}
        row["quality"] = {k: "missing" for k in row["observations"]}
        row["schedule"] = [{"kind": "fake_fault"}]
        row["observations"]["fault_type"] = 999
        row["observations"]["engine.root_cause"] = 1
    assert analyse_episode(changed) == analyse_episode(samples)
    assert engineer_features(changed) == engineer_features(samples)
    assert samples == original
    features = engineer_features(samples)
    assert not any("fault" in key or "root_cause" in key or "expected" in key or "scenario" in key
                   for key in features[0])
    json.dumps(features, allow_nan=False)


@pytest.mark.parametrize("kind", ["engine_oil_pump_wear", "piston_ring_damage", "turbo_exhaust_fault",
                                 "coolant_system_failure", "coolant_degradation", "gearbox_seal_wear",
                                 "gearbox_bearing_wear", "bearing_failure", "low_oil", "overheat",
                                 "brake_wear", "perception_degraded"])
def test_developed_fault_generates_matching_observation_only_hypothesis(kind):
    result = analyse_episode(episode(kind)["samples"])
    mature = next(row for row in result if row["elapsed_s"] == 300)
    assert kind in {hypothesis["fault"] for hypothesis in mature["hypotheses"]}
    assert all(h["evidence"] and h["alternatives"] and h["mitigation"] and not h["root_cause_confirmed"]
               for h in mature["hypotheses"])
    assert mature["rul_hours"] is None and mature["rul_status"] == "not_validated"


def test_baseline_context_changes_do_not_become_mechanical_faults():
    # This is a targeted regression, not a calibrated fleet false-alarm estimate.
    result = analyse_episode(episode()["samples"])
    assert not any(h["fault"] not in ("sensor_fault", "unrelated_multi_blip")
                   for row in result for h in row["hypotheses"])
    assert result[60]["overall_health"] == 100


def test_missing_sensor_invalid_health_and_mask_without_reading_hidden_fault():
    samples = episode()["samples"]
    samples[60]["observations"]["engine.oil_pressure_kpa"] = None
    analysis = analyse_episode(samples)
    features = engineer_features(samples)
    assert analysis[60]["overall_health"] is None
    assert analysis[60]["data_quality"]["engine.oil_pressure_kpa"] == "missing"
    assert features[60]["engine.oil_pressure_kpa__available_mask"] == 0
    assert features[60]["engine.oil_pressure_kpa"] is None
    assert features[61]["engine.oil_pressure_kpa__rate"] is None


def test_temporal_features_reset_at_engine_episode_boundaries():
    first = episode()["samples"]
    second = episode(episode_id="second")["samples"]
    joint = engineer_features(first + second)
    separate = engineer_features(second)
    assert joint[len(first):] == separate
    assert joint[len(first)]["engine.oil_pressure_kpa__rate"] is None
    with pytest.raises(ValueError, match="increase"):
        engineer_features(first + first)


def test_fft_known_frequency_and_invalid_waveforms():
    rate, frequency = 1024, 64
    values = [math.sin(2 * math.pi * frequency * i / rate) for i in range(rate)]
    result = spectral_features({"waveform": values, "sample_rate_hz": rate, "rpm": frequency * 60})
    assert result["vibration.dominant_frequency_hz"] == 64
    assert result["vibration.rms_mm_s"] == pytest.approx(1 / math.sqrt(2))
    assert result["vibration.order_1_energy_fraction"] > .99
    with pytest.raises(ValueError):
        spectral_features({"waveform": [math.nan] * 16, "sample_rate_hz": 1024, "rpm": 0})


def test_sparse_long_history_can_identify_slow_aging_without_fast_contamination_label():
    ep = simulate_enhanced({"duration_s": 72000, "timestep_s": 300, "noise_scale": 0},
                          [{"kind": "hydraulic_normal_aging", "onset_s": 0,
                            "duration_s": 72000, "severity": 1, "progression": 0, "recovery_s": 0}])
    last = analyse_episode(ep["samples"])[-1]
    kinds = {h["fault"] for h in last["hypotheses"]}
    assert "hydraulic_normal_aging" in kinds
    assert "hydraulic_contamination" not in kinds
    assert last["residual_slopes_per_s"]["suspension.hydraulic_oil_condition_pct"] is not None


def test_longitudinal_export_preserves_hours_and_damage_and_excludes_shortcuts(tmp_path):
    generate_dataset(tmp_path / "history", engines=3, episodes_per_fault=1,
                     duration_s=100, timestep_s=5, longitudinal=True)
    manifest = json.loads((tmp_path / "history/manifest.json").read_text())
    episodes = [e for e in manifest["episodes"] if e["config"]["engine_id"] == "virtual-0001"]
    assert episodes[1]["config"]["initial_operating_hours"] > 0
    assert any(e["config"].get("initial_fault_wear", {}).get("engine_oil_pump_wear", 0) > 0
               for e in episodes[2:])
    assert episodes[1]["config"]["start_time"] > episodes[0]["config"].get("start_time", "2026-01-01")
    assert "context.scenario" not in manifest["input_columns"]["readings.csv"]
    assert "context.operating_hours" not in manifest["input_columns"]["readings.csv"]
    assert manifest["label_columns"]["multi_label_targets"]
