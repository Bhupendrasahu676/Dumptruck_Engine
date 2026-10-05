"""Behavioral checks for temporal synthetic telemetry and separate labels."""

import copy
from datetime import datetime, timedelta, timezone
import json
import math
import random

import pytest

from enhanced_simulation import (
    FAULT_CATALOG, FAULT_KINDS, FaultEvent, simulate_enhanced,
    synthesize_vibration,
)
from telemetry_model import SimulationConfig, simulate_journey


CONFIG = {"duration_s": 600, "timestep_s": 5, "noise_scale": 0}


def at(episode, time):
    return next(sample for sample in episode["samples"] if sample["elapsed_s"] == time)


def developed(kind, **kwargs):
    severity = kwargs.pop("severity", 1)
    return simulate_enhanced(CONFIG, FaultEvent(kind, onset_s=60, duration_s=300, severity=severity, progression=0.1, recovery_s=60, **kwargs))


def test_reproducible_matched_reference_context_and_global_rng():
    config = {**CONFIG, "engine_seed": 42, "start_condition": "cold", "ambient_c": 8}
    before = random.getstate()
    healthy = simulate_enhanced(config)
    faulty = simulate_enhanced(config, FaultEvent("coolant_system_failure", onset_s=100))
    assert random.getstate() == before
    assert faulty == simulate_enhanced(config, FaultEvent("coolant_system_failure", onset_s=100))
    assert healthy["config"]["base_trajectory_id"] == faulty["config"]["base_trajectory_id"]
    for normal, fault in zip(healthy["samples"], faulty["samples"]):
        assert normal["expected"] == fault["expected"]
        assert normal["context"] == fault["context"]
        assert normal["observations"] == normal["expected"]
        if fault["elapsed_s"] <= 100:
            assert normal["observations"] == fault["observations"]
    assert len(healthy["samples"]) == 121
    assert {s["context"]["payload_class"] for s in healthy["samples"]} == {"medium"}


def test_raw_legacy_measurements_and_energy_preserved_verbatim():
    config = {**CONFIG, "seed": 793}
    enhanced = developed("coolant_system_failure")
    legacy = simulate_journey(SimulationConfig(seed=793, num_steps=121, timestep_s=5, drive_cycle="c-wtvc", payload_class="medium"))
    for frame, sample in zip(legacy, enhanced["samples"]):
        raw = sample["raw_legacy"]
        assert raw["engine.oil_temp_c"] == frame.engine.oil_temp_c
        assert raw["imu.acceleration_z_mps2"] == frame.imu.acceleration_z_mps2
        assert raw["imu.vibration_rms_g"] == frame.imu.vibration_rms_g
        assert raw["frame.strain_microstrain"] == frame.frame.strain_microstrain
        assert raw["lidar.obstacle_distance_m"] == frame.lidar.obstacle_distance_m
        for index, temp in enumerate(frame.engine.exhaust_temperatures_c):
            assert raw[f"engine.exhaust_temperatures_c.{index}"] == temp
        for signal in ("engine.fuel_energy_input_kw", "engine.exhaust_energy_flow_kw", "engine.coolant_energy_flow_kw", "engine.effective_power_kw", "cumulative_fuel_energy_kwh"):
            assert sample["observations"][signal] == raw[signal]
        assert not any("health" in key or "fault" in key or "rul" in key for key in sample["observations"])
    assert "not a recomputed" in enhanced["metadata"]["energy_disclaimer"]
    assert not enhanced["metadata"]["physics_validated"]


def test_identity_offsets_stable_across_episode_seeds_and_ids():
    first = simulate_enhanced({**CONFIG, "seed": 1, "engine_seed": 81})
    second = simulate_enhanced({**CONFIG, "seed": 2, "engine_seed": 81, "episode_id": "another"})
    different = simulate_enhanced({**CONFIG, "seed": 1, "engine_seed": 82})
    for key in ("engine.oil_pressure_kpa", "engine.oil_temp_c", "engine.block_vibration_mm_s"):
        a, b = first["samples"][0], second["samples"][0]
        assert a["expected"][key] - a["raw_legacy"][key] == pytest.approx(b["expected"][key] - b["raw_legacy"][key])
        assert a["expected"][key] != different["samples"][0]["expected"][key]


def test_reference_accounts_for_hot_heavy_and_cold_context_without_fault_labels():
    for settings in ({"ambient_c": 50, "payload_class": "overload", "scenario": "stress"}, {"ambient_c": -25, "start_condition": "cold"}):
        episode = simulate_enhanced({**CONFIG, **settings})
        assert all(sample["observations"] == sample["expected"] for sample in episode["samples"])
        assert all(not sample["truth"]["active_faults"] for sample in episode["samples"])
        assert len({s["context"]["payload_utilization_pct"] for s in episode["samples"]}) == 1
    hot = simulate_enhanced({**CONFIG, "ambient_c": 50})["samples"][0]
    mild = simulate_enhanced(CONFIG)["samples"][0]
    assert hot["expected"]["engine.block_temp_c"] > mild["expected"]["engine.block_temp_c"]


def test_progression_causal_lags_recovery_and_persistent_wear():
    episode = developed("engine_oil_pump_wear")
    onset = at(episode, 60)
    immediate = at(episode, 65)
    thermal = at(episode, 100)
    mature = at(episode, 180)
    done = at(episode, 600)
    assert onset["truth"]["fault_levels"]["engine_oil_pump_wear"] == 0
    assert immediate["observations"]["engine.oil_pressure_kpa"] < immediate["expected"]["engine.oil_pressure_kpa"]
    # Tiny wear-induced heat may start before the explicitly lagged thermal
    # effect; the thermal jump appears only after the causal delay.
    assert immediate["observations"]["engine.oil_temp_c"] - immediate["expected"]["engine.oil_temp_c"] < 0.01
    assert thermal["observations"]["engine.oil_temp_c"] - thermal["expected"]["engine.oil_temp_c"] > 10
    assert at(episode, 90)["observations"]["engine.block_vibration_mm_s"] == at(episode, 90)["expected"]["engine.block_vibration_mm_s"]
    assert mature["observations"]["engine.block_vibration_mm_s"] > mature["expected"]["engine.block_vibration_mm_s"]
    assert at(episode, 390)["truth"]["event_phases"]["engine_oil_pump_wear-1"] == "recovering"
    assert not done["truth"]["active_faults"]
    assert done["truth"]["latent_wear"]["engine_oil_pump_wear"] > 0
    assert done["truth"]["fault_levels"]["engine_oil_pump_wear"] == 0
    assert done["truth"]["effect_levels"]["engine_oil_pump_wear"] == done["truth"]["latent_wear"]["engine_oil_pump_wear"]
    assert done["observations"]["engine.oil_pressure_kpa"] < done["expected"]["engine.oil_pressure_kpa"]
    assert done["truth"]["latent_wear"] == at(episode, 480)["truth"]["latent_wear"]


def test_recurrence_overlap_and_event_identity():
    a = FaultEvent("low_oil", onset_s=20, duration_s=80, progression=0, recovery_s=20, recurrence_s=150, event_id="oil")
    b = FaultEvent("turbo_exhaust_fault", onset_s=60, duration_s=200, progression=0, recovery_s=20, event_id="turbo")
    result = simulate_enhanced(CONFIG, [a, b])
    assert set(at(result, 80)["truth"]["active_faults"]) == {"low_oil", "turbo_exhaust_fault"}
    assert at(result, 140)["truth"]["event_phases"]["oil"] == "between_occurrences"
    assert at(result, 175)["truth"]["event_occurrences"]["oil"] == 1
    assert at(result, 175)["truth"]["fault_levels"]["low_oil"] == a.severity
    assert at(result, 320)["truth"]["event_occurrences"]["oil"] == 2
    # Two simultaneous events of the same kind have bounded summed truth but
    # retain their individual phase identities.
    doubled = simulate_enhanced(CONFIG, [a, FaultEvent("low_oil", onset_s=20, duration_s=80, progression=0, event_id="other")])
    assert at(doubled, 40)["truth"]["fault_levels"]["low_oil"] == 1
    assert set(at(doubled, 40)["truth"]["event_phases"]) == {"oil", "other"}


@pytest.mark.parametrize("kind", FAULT_KINDS)
def test_catalogue_bounded_numeric_schema_and_truth_separation(kind):
    episode = developed(kind)
    for sample in episode["samples"]:
        assert set(sample["expected"]) == set(sample["observations"]) == set(sample["quality"])
        assert set(sample["truth"]["fault_levels"]) == set(FAULT_KINDS)
        assert set(sample["truth"]["effect_levels"]) == set(FAULT_KINDS)
        assert all(0 <= value <= 1 for value in sample["truth"]["fault_levels"].values())
        assert all(0 <= value <= 1 for value in sample["truth"]["effect_levels"].values())
        assert all(value is None or (isinstance(value, (int, float)) and math.isfinite(value)) for value in sample["observations"].values())
        for key, value in sample["observations"].items():
            if key.endswith("condition_pct"):
                assert 0 <= value <= 100
        assert sample["expected"] is not sample["observations"]
    assert all(FAULT_CATALOG[kind][field] for field in ("subsystem", "possible_cause", "mitigation"))


@pytest.mark.parametrize("severity", [0.8, 1.0])
def test_developed_multichannel_fault_signatures(severity):
    pump = at(developed("engine_oil_pump_wear", severity=severity), 240)["observations"]
    assert pump["engine.oil_pressure_kpa"] < 440
    assert pump["engine.oil_temp_c"] > 106
    assert pump["engine.block_vibration_mm_s"] > 7.5
    assert pump["engine.oil_condition_pct"] >= 90
    coolant = at(developed("coolant_system_failure", severity=severity), 240)["observations"]
    assert coolant["engine.block_temp_c"] > 108
    assert coolant["engine.oil_temp_c"] > 106
    assert coolant["engine.exhaust_temp_c"] > 615
    assert coolant["engine.coolant_pump_temp_c"] > 98
    gearbox = at(developed("gearbox_seal_wear", severity=severity), 240)["observations"]
    assert gearbox["transmission.oil_pressure_kpa"] < 310
    assert gearbox["transmission.oil_temp_c"] > 96
    assert gearbox["transmission.vibration_mm_s"] > 6
    assert gearbox["transmission.oil_condition_pct"] >= 90
    bearing = at(developed("gearbox_bearing_wear", severity=severity), 240)["observations"]
    assert bearing["transmission.oil_temp_c"] > 96
    assert bearing["transmission.vibration_mm_s"] > 6
    assert bearing["transmission.oil_pressure_kpa"] < 310
    assert bearing["transmission.oil_condition_pct"] < 90
    hydraulic = at(developed("hydraulic_contamination", severity=severity), 240)["observations"]
    assert hydraulic["suspension.hydraulic_oil_condition_pct"] < 80
    assert hydraulic["suspension.actuator_vibration_mm_s"] > 6.3
    degradation = at(developed("coolant_degradation", severity=severity), 240)
    assert degradation["observations"]["coolant_condition_pct"] < 80
    assert degradation["observations"]["engine.coolant_pump_temp_c"] > 98
    for key in ("engine.block_temp_c", "engine.oil_temp_c", "engine.exhaust_temp_c"):
        assert degradation["observations"][key] == degradation["expected"][key]


def test_turbo_localization_and_piston_individual_egt_imbalance():
    turbo = at(developed("turbo_exhaust_fault"), 240)
    assert turbo["observations"]["engine.exhaust_temp_c"] > 615
    assert turbo["observations"]["engine.boost_kpa"] < turbo["expected"]["engine.boost_kpa"]
    for key in ("engine.block_temp_c", "engine.oil_temp_c", "engine.oil_pressure_kpa", "engine.coolant_pump_temp_c", "transmission.oil_temp_c"):
        assert turbo["observations"][key] == turbo["expected"][key]
    ring = at(developed("piston_ring_damage"), 240)
    temperatures = [ring["observations"][f"engine.exhaust_temperatures_c.{i}"] for i in range(6)]
    assert max(temperatures) - min(temperatures) > 150


def test_contamination_ages_faster_and_wear_is_timestep_independent():
    aging = developed("hydraulic_normal_aging")
    contamination = developed("hydraulic_contamination")
    key = "suspension.hydraulic_oil_condition_pct"
    for time in (180, 240, 600):
        a, c = at(aging, time), at(contamination, time)
        assert c["expected"][key] - c["observations"][key] > a["expected"][key] - a["observations"][key]
    event = FaultEvent("bearing_failure", onset_s=60, duration_s=300)
    coarse = simulate_enhanced({**CONFIG, "timestep_s": 10}, event)
    fine = simulate_enhanced({**CONFIG, "timestep_s": 2}, event)
    assert coarse["samples"][-1]["truth"]["latent_wear"] == fine["samples"][-1]["truth"]["latent_wear"]
    accelerated = simulate_enhanced({**CONFIG, "wear_time_scale": 50}, event)
    assert accelerated["metadata"]["accelerated_aging"]
    assert not fine["metadata"]["accelerated_aging"]
    assert accelerated["samples"][-1]["truth"]["latent_wear"]["bearing_failure"] > fine["samples"][-1]["truth"]["latent_wear"]["bearing_failure"]


@pytest.mark.parametrize("mode", ["missing", "biased", "frozen", "drifting", "noisy"])
def test_sensor_modes_quality_and_recovery(mode):
    key = "engine.oil_pressure_kpa"
    event = FaultEvent("sensor_fault", onset_s=60, duration_s=120, severity=1, progression=0, recovery_s=0, target_signal=key, sensor_mode=mode, bias=30)
    result = simulate_enhanced(CONFIG, event)
    normal = simulate_enhanced(CONFIG)
    for sample, baseline in zip(result["samples"], normal["samples"]):
        assert sample["expected"] == baseline["expected"]
        for other in sample["observations"]:
            if other != key:
                assert sample["observations"][other] == baseline["observations"][other]
        if 60 <= sample["elapsed_s"] < 180:
            assert sample["quality"][key] == mode
            value = sample["observations"][key]
            if mode == "missing":
                assert value is None
            elif mode == "biased":
                assert value - sample["expected"][key] == pytest.approx(30)
            elif mode == "noisy":
                assert abs(value - sample["expected"][key]) <= 30
        else:
            assert sample["quality"][key] == "good"
            assert sample["observations"] == baseline["observations"]
    if mode == "frozen":
        assert len({s["observations"][key] for s in result["samples"] if 60 <= s["elapsed_s"] < 180}) == 1
    if mode == "drifting":
        a, b = at(result, 80), at(result, 160)
        assert b["observations"][key] - b["expected"][key] > a["observations"][key] - a["expected"][key]


def test_recurring_freeze_resets_and_missing_precedence():
    key = "engine.oil_temp_c"
    frozen = FaultEvent("sensor_fault", onset_s=60, duration_s=60, progression=0, recovery_s=0, recurrence_s=150, target_signal=key, sensor_mode="frozen")
    result = simulate_enhanced(CONFIG, frozen)
    assert at(result, 65)["observations"][key] == at(result, 100)["observations"][key]
    assert at(result, 215)["observations"][key] == at(result, 250)["observations"][key]
    assert at(result, 65)["observations"][key] != at(result, 215)["observations"][key]
    missing = FaultEvent("sensor_fault", onset_s=60, progression=0, target_signal=key, sensor_mode="missing")
    mixed = simulate_enhanced(CONFIG, [missing, frozen])
    assert at(mixed, 65)["observations"][key] is None
    assert at(mixed, 65)["quality"][key] == "missing"


def test_noise_is_bounded_and_independent_of_fault_schedule():
    clean = simulate_enhanced(CONFIG)
    noisy = simulate_enhanced({**CONFIG, "noise_scale": 2})
    fault = simulate_enhanced({**CONFIG, "noise_scale": 2}, FaultEvent("brake_wear", onset_s=60))
    for sample, baseline, injected in zip(noisy["samples"], clean["samples"], fault["samples"]):
        assert sample["expected"] == baseline["expected"]
        assert abs(sample["observations"]["engine.oil_pressure_kpa"] - sample["expected"]["engine.oil_pressure_kpa"]) <= 3
        assert abs(sample["observations"]["engine.oil_temp_c"] - sample["expected"]["engine.oil_temp_c"]) <= 0.7
        assert sample["observations"]["engine.oil_pressure_kpa"] == injected["observations"]["engine.oil_pressure_kpa"]


def test_percentage_bounds_include_extreme_wear_and_sensor_bias():
    events = [FaultEvent("hydraulic_contamination", onset_s=0, duration_s=600, progression=0), FaultEvent("brake_wear", onset_s=0, duration_s=600, progression=0), FaultEvent("sensor_fault", onset_s=0, duration_s=600, progression=0, target_signal="engine.oil_condition_pct", bias=1e6)]
    episode = simulate_enhanced({**CONFIG, "wear_time_scale": 10000, "initial_wear": 1, "payload_class": "overload"}, events)
    for sample in episode["samples"]:
        for key, value in sample["observations"].items():
            if key.endswith("_pct"):
                assert 0 <= value <= (125 if key == "payload.utilization_pct" else 100)
    assert episode["samples"][-1]["observations"]["suspension.hydraulic_oil_condition_pct"] == 0


@pytest.mark.parametrize("bad", [
    {"duration_s": 0}, {"duration_s": math.nan}, {"timestep_s": -1},
    {"timestep_s": 1e-320}, {"timestep_s": 900}, {"ambient_c": math.inf},
    {"initial_wear": 1.1}, {"noise_scale": -1}, {"wear_time_scale": -1},
    {"seed": True}, {"seed": 1.2}, {"engine_seed": "7"}, {"engine_id": ""},
    {"scenario": "unknown"}, {"payload_class": "liquid"}, {"start_condition": "hot"},
    {"drive_cycle": "official-trace"}, {"unknown": 1}, {"start_time": "not-date"},
    {3: 1, "unknown": 1},
    {"start_time": "2026-01-01"}, {"start_time": datetime(2026, 1, 1)},
    {"timestamp": "2026-01-01T00:00:00Z", "start_time": "2026-01-01T00:00:00Z"},
    {"initial_operating_hours": -1}, {"initial_operating_hours": math.inf},
    {"initial_operating_hours": True}, {"initial_fault_wear": []},
    {"initial_fault_wear": {"unknown": 0}}, {"initial_fault_wear": {"sensor_fault": 0.2}},
    {"initial_fault_wear": {"low_oil": 0.2}},
    {"initial_fault_wear": {"bearing_failure": -0.1}},
    {"initial_fault_wear": {"bearing_failure": 1.1}},
    {"initial_fault_wear": {"bearing_failure": math.nan}},
    {"initial_fault_wear": {"bearing_failure": "0.2"}},
])
def test_invalid_configuration_rejected(bad):
    with pytest.raises(ValueError):
        simulate_enhanced({**CONFIG, **bad})


@pytest.mark.parametrize("bad", [
    {"kind": "unknown"}, {"kind": "low_oil", "onset_s": -1},
    {"kind": "low_oil", "severity": 2}, {"kind": "low_oil", "severity": math.nan},
    {"kind": "low_oil", "duration_s": 0}, {"kind": "low_oil", "duration_s": math.inf},
    {"kind": "low_oil", "progression": -1}, {"kind": "low_oil", "recovery_s": -1},
    {"kind": "low_oil", "recurrence_s": 60}, {"kind": "sensor_fault", "sensor_mode": "invalid"},
    {"kind": "sensor_fault", "target_signal": "engine.fake"},
    {"kind": "low_oil", "target_signal": "engine.rpm"},
    {"kind": "sensor_fault", "bias": math.inf}, {"kind": "low_oil", "oops": 1},
])
def test_invalid_events_rejected(bad):
    with pytest.raises(ValueError):
        simulate_enhanced(CONFIG, bad)


def test_duplicate_ids_noniterable_inputs_and_future_zero_events():
    with pytest.raises(ValueError, match="duplicate"):
        simulate_enhanced(CONFIG, [FaultEvent("low_oil", event_id="one"), FaultEvent("brake_wear", event_id="one")])
    for bad in (None, "low_oil", 3, [3]):
        with pytest.raises(ValueError):
            simulate_enhanced(CONFIG, bad)
    healthy = simulate_enhanced(CONFIG)
    inactive = simulate_enhanced(CONFIG, [FaultEvent("low_oil", onset_s=1000), FaultEvent("bearing_failure", severity=0)])
    assert [s["observations"] for s in inactive["samples"]] == [s["observations"] for s in healthy["samples"]]


def test_utc_timestamps_inputs_unmodified_and_json_serializable():
    config = {**CONFIG, "timestamp": "2026-01-01T07:30:00+05:30"}
    events = [{"kind": "low_oil"}]
    saved_config, saved_events = copy.deepcopy(config), copy.deepcopy(events)
    episode = simulate_enhanced(config, events)
    assert config == saved_config and events == saved_events
    start = datetime(2026, 1, 1, 2, tzinfo=timezone.utc)
    for sample in episode["samples"]:
        assert datetime.fromisoformat(sample["timestamp"]) == start + timedelta(seconds=sample["elapsed_s"])
    json.dumps(episode, allow_nan=False)
    units = episode["metadata"]["units"]
    assert units["engine.block_vibration_mm_s"] == "mm/s"
    assert units["imu.vibration_rms_g"] == "g"
    assert units["engine.oil_pressure_kpa"] == "kPa"
    assert units["engine.exhaust_temperatures_c.0"] == "degC"
    assert units["dss.camera_fps"] == "Hz"


def test_waveform_deterministic_velocity_rms_and_no_truth_dependency():
    sample = at(developed("bearing_failure"), 240)
    waveform = synthesize_vibration(sample, seed=6)
    assert waveform == synthesize_vibration(sample, seed=6)
    unlabeled = {"observations": sample["observations"]}
    assert waveform == synthesize_vibration(unlabeled, seed=6)
    values = waveform["waveform"]
    assert len(values) == 4096
    assert waveform["units"] == "mm/s"
    assert math.sqrt(sum(x * x for x in values) / len(values)) == pytest.approx(sample["observations"]["engine.block_vibration_mm_s"], rel=1e-7)
    assert abs(sum(values) / len(values)) < 1e-7
    assert max(map(abs, values)) < 3 * sample["observations"]["engine.block_vibration_mm_s"]
    assert waveform["metadata"]["synthetic"]
    assert all(f < waveform["sample_rate_hz"] / 2 for f in waveform["metadata"]["assumed_frequency_components"])


@pytest.mark.parametrize("kwargs", [
    {"sample_rate_hz": 16}, {"sample_rate_hz": math.nan},
    {"sample_rate_hz": 100000}, {"duration_s": 0}, {"duration_s": math.inf},
    {"sample_rate_hz": 65536, "duration_s": 10}, {"duration_s": 1e-9},
    {"seed": True},
])
def test_invalid_waveform_config_rejected(kwargs):
    sample = simulate_enhanced({"duration_s": 10})["samples"][0]
    with pytest.raises(ValueError):
        synthesize_vibration(sample, **kwargs)


def test_missing_waveform_measurements_and_standstill():
    for key in ("engine.rpm", "engine.block_vibration_mm_s"):
        readings = {"engine.rpm": 1000, "engine.block_vibration_mm_s": 2}
        readings[key] = None
        with pytest.raises(ValueError):
            synthesize_vibration({"observations": readings})
    stopped = synthesize_vibration({"observations": {"engine.rpm": 0, "engine.block_vibration_mm_s": 0}}, sample_rate_hz=16)
    assert set(stopped["waveform"]) == {0}
    assert stopped["metadata"]["assumed_frequency_components"] == []


@pytest.mark.parametrize("kind,condition_key", [
    ("engine_oil_pump_wear", "engine.oil_condition_pct"),
    ("gearbox_seal_wear", "transmission.oil_condition_pct"),
])
def test_pressure_loss_preserves_condition_through_progression_and_damage(kind, condition_key):
    event = FaultEvent(kind, onset_s=60, duration_s=300, severity=1, progression=0.1, recovery_s=60)
    episode = simulate_enhanced({**CONFIG, "wear_time_scale": 1000}, event)
    for sample in episode["samples"]:
        assert sample["observations"][condition_key] == sample["expected"][condition_key]
    assert episode["samples"][-1]["truth"]["latent_wear"][kind] == 1
    assert episode["samples"][-1]["truth"]["effect_levels"][kind] == 1
    assert not episode["samples"][-1]["truth"]["active_faults"]


def test_coolant_degradation_remains_local_even_with_persistent_damage():
    episode = simulate_enhanced({**CONFIG, "wear_time_scale": 1000}, FaultEvent("coolant_degradation", onset_s=60, duration_s=300))
    for sample in episode["samples"]:
        for key in ("engine.block_temp_c", "engine.oil_temp_c", "engine.exhaust_temp_c", "engine.coolant_temp_c"):
            assert sample["observations"][key] == sample["expected"][key]
    final = episode["samples"][-1]
    assert final["observations"]["coolant_condition_pct"] < 80
    assert final["observations"]["engine.coolant_pump_temp_c"] > 98


@pytest.mark.parametrize("duration,timestep,scale", [(7200, 60, 1), (600, 5, 25)])
def test_hydraulic_aging_has_gradual_condition_and_persistent_vibration(duration, timestep, scale):
    event = FaultEvent("hydraulic_normal_aging", onset_s=0, duration_s=duration / 2, severity=1, progression=0, recovery_s=0)
    episode = simulate_enhanced({"duration_s": duration, "timestep_s": timestep, "wear_time_scale": scale, "noise_scale": 0}, event)
    condition_key = "suspension.hydraulic_oil_condition_pct"
    vibration_key = "suspension.actuator_vibration_mm_s"
    condition_losses = [s["expected"][condition_key] - s["observations"][condition_key] for s in episode["samples"]]
    vibration_gains = [s["observations"][vibration_key] - s["expected"][vibration_key] for s in episode["samples"]]
    assert condition_losses[0] == vibration_gains[0] == 0
    assert all(a <= b + 1e-6 for a, b in zip(condition_losses, condition_losses[1:]))
    assert all(a <= b + 1e-6 for a, b in zip(vibration_gains, vibration_gains[1:]))
    assert condition_losses[-1] > 2
    assert vibration_gains[-1] > 0.15
    active_losses = [loss for sample, loss in zip(episode["samples"], condition_losses) if sample["elapsed_s"] < duration / 2]
    assert len({round(loss, 4) for loss in active_losses}) > 10
    final = episode["samples"][-1]
    assert final["truth"]["fault_levels"]["hydraulic_normal_aging"] == 0
    assert final["truth"]["effect_levels"]["hydraulic_normal_aging"] > 0
    assert vibration_gains[-1] == pytest.approx(8 * final["truth"]["latent_wear"]["hydraulic_normal_aging"], abs=1e-6)


def test_effect_labels_cover_delayed_recovery_and_clear_reversible_faults():
    event = FaultEvent("turbo_exhaust_fault", onset_s=60, duration_s=120, severity=0.8, progression=0, recovery_s=0)
    episode = simulate_enhanced(CONFIG, event)
    tail = at(episode, 190)
    assert tail["truth"]["fault_levels"][event.kind] == 0
    assert event.kind not in tail["truth"]["active_faults"]
    assert tail["truth"]["event_phases"][f"{event.kind}-1"] == "complete"
    assert tail["truth"]["effect_levels"][event.kind] == 0.8
    assert tail["observations"]["engine.exhaust_temp_c"] > tail["expected"]["engine.exhaust_temp_c"]
    cleared = at(episode, 210)
    assert cleared["truth"]["effect_levels"][event.kind] == 0
    assert cleared["observations"] == cleared["expected"]
    assert "maximum per-channel-delay sum" in episode["metadata"]["effect_level_semantics"]
    assert "active_faults" in episode["metadata"]["schedule_level_semantics"]
    overlapping = simulate_enhanced(CONFIG, [event, FaultEvent("turbo_exhaust_fault", onset_s=60, duration_s=120, progression=0, recovery_s=0)])
    assert at(overlapping, 190)["truth"]["effect_levels"][event.kind] == 1


def test_engine_history_continuation_retains_damage_and_real_operating_hours():
    original_config = {**CONFIG, "engine_seed": 44, "engine_id": "lifecycle-44", "initial_operating_hours": 500, "wear_time_scale": 30}
    event = FaultEvent("gearbox_bearing_wear", onset_s=0, duration_s=300, progression=0, recovery_s=0)
    first = simulate_enhanced(original_config, event)
    metadata = first["metadata"]
    state = metadata["final_engine_state"]
    assert metadata["final_fault_wear"] == state["initial_fault_wear"]
    assert state["initial_operating_hours"] == metadata["final_operating_hours"] == pytest.approx(500 + 600 / 3600)
    assert state["engine_seed"] == 44 and state["engine_id"] == "lifecycle-44"
    saved_state = copy.deepcopy(state)
    continuation = simulate_enhanced({**CONFIG, **state, "seed": 22, "episode_id": "next"})
    assert state == saved_state
    first_next = continuation["samples"][0]
    assert first_next["context"]["operating_hours"] == metadata["final_operating_hours"]
    assert first_next["truth"]["latent_wear"][event.kind] == metadata["final_fault_wear"][event.kind]
    assert first_next["truth"]["effect_levels"][event.kind] > 0
    assert first_next["truth"]["fault_levels"][event.kind] == 0
    assert not first_next["truth"]["active_faults"]
    for key in ("transmission.oil_pressure_kpa", "transmission.oil_condition_pct"):
        assert first_next["observations"][key] < first_next["expected"][key]
    assert first_next["observations"]["transmission.vibration_mm_s"] > first_next["expected"]["transmission.vibration_mm_s"]
    independent = simulate_enhanced({**CONFIG, "seed": 22, "engine_seed": 44})
    assert first_next["expected"] == independent["samples"][0]["expected"]
    assert independent["samples"][0]["observations"] == independent["samples"][0]["expected"]
    assert not any(independent["samples"][0]["truth"]["effect_levels"].values())
    # Additional exposure accumulates on inherited damage, rather than replacing it.
    repeated = simulate_enhanced({**CONFIG, **state, "wear_time_scale": 30}, event)
    assert repeated["metadata"]["final_fault_wear"][event.kind] > metadata["final_fault_wear"][event.kind]
    json.dumps(repeated, allow_nan=False)


def test_history_normalization_independence_saturation_and_final_sample_horizon():
    history = {"engine_oil_pump_wear": 0.9, "hydraulic_normal_aging": 0.4}
    config = {"duration_s": 602, "timestep_s": 5, "noise_scale": 0, "initial_operating_hours": 7, "initial_fault_wear": history, "wear_time_scale": 1000}
    before = copy.deepcopy(config)
    first = simulate_enhanced(config, FaultEvent("engine_oil_pump_wear", onset_s=0, progression=0))
    assert config == before
    assert first["metadata"]["final_fault_wear"]["engine_oil_pump_wear"] == 1
    assert first["metadata"]["final_operating_hours"] == pytest.approx(7 + 600 / 3600)
    assert first["samples"][0]["truth"]["latent_wear"]["hydraulic_normal_aging"] == 0.4
    initial = first["samples"][0]
    assert initial["observations"]["suspension.actuator_vibration_mm_s"] - initial["expected"]["suspension.actuator_vibration_mm_s"] == pytest.approx(3.2)
    reversed_history = dict(reversed(list(history.items())))
    identical = simulate_enhanced({**config, "initial_fault_wear": reversed_history}, FaultEvent("engine_oil_pump_wear", onset_s=0, progression=0))
    assert first == identical
    independent = simulate_enhanced(CONFIG)
    assert not any(independent["metadata"]["final_fault_wear"].values())


def test_waveform_harmonics_respond_to_readings_without_label_dependencies():
    readings = {"engine.rpm": 1200, "engine.block_vibration_mm_s": 4, "engine.oil_condition_pct": 97}
    readings.update({f"engine.exhaust_temperatures_c.{i}": 400 for i in range(6)})
    normal = synthesize_vibration({"observations": readings}, seed=4)
    abnormal_readings = {**readings, "engine.oil_condition_pct": 45, "engine.exhaust_temperatures_c.2": 620}
    abnormal = synthesize_vibration({"observations": abnormal_readings}, seed=4)
    assert abnormal["waveform"] != normal["waveform"]
    assert abnormal["metadata"]["component_weights"][1] > normal["metadata"]["component_weights"][1]
    assert abnormal["metadata"]["component_weights"][2] > normal["metadata"]["component_weights"][2]
    fabricated_labels = {"observations": abnormal_readings, "truth": {"fault_levels": {kind: 1 for kind in FAULT_KINDS}}, "quality": {key: "missing" for key in abnormal_readings}, "expected": {key: 0 for key in abnormal_readings}}
    assert synthesize_vibration(fabricated_labels, seed=4) == abnormal
    assert math.sqrt(sum(x * x for x in abnormal["waveform"]) / len(abnormal["waveform"])) == pytest.approx(4, abs=1e-7)
    missing_optional = {**readings, "engine.oil_condition_pct": None}
    for i in range(6):
        missing_optional[f"engine.exhaust_temperatures_c.{i}"] = None
    neutral = synthesize_vibration({"observations": missing_optional}, seed=4)
    assert neutral["metadata"]["component_weights"] == normal["metadata"]["component_weights"]
    invalid_optional = {**missing_optional, "engine.oil_condition_pct": math.nan, "engine.exhaust_temperatures_c.0": -1e308, "engine.exhaust_temperatures_c.1": 1e308}
    invalid = synthesize_vibration({"observations": invalid_optional}, seed=4)
    assert invalid["metadata"]["component_weights"] == neutral["metadata"]["component_weights"]
    json.dumps(invalid, allow_nan=False)
