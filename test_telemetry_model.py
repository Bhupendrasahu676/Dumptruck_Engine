"""Focused behavioral checks for the dump-truck telemetry model."""

from telemetry_model import (
    DriveCycle,
    EngineHealthClass,
    PayloadClass,
    Scenario,
    classify_engine_health,
    simulate_journey,
)


def test_payload_class_is_discrete_and_fixed_for_a_journey() -> None:
    journey = simulate_journey(seed=17, num_steps=48)

    categories = [frame.payload_class for frame in journey]

    assert categories
    assert len(set(categories)) == 1
    assert categories[0] in set(PayloadClass)
    assert all(isinstance(category, PayloadClass) for category in categories)


def test_stress_scenario_escalates_above_85_percent_capacity() -> None:
    journey = simulate_journey(seed=17, scenario=Scenario.STRESS, num_steps=48)

    utilization = [frame.payload_utilization_pct for frame in journey]
    stress = [frame.stress_level_pct for frame in journey]

    assert journey.payload_class is PayloadClass.HEAVY
    assert max(utilization) > 85.0
    assert stress[-1] > stress[0]
    assert journey[-1].engine.coolant_temp_c > journey[0].engine.coolant_temp_c


def test_health_classification_thresholds_are_stable() -> None:
    assert classify_engine_health(95) is EngineHealthClass.EXCELLENT
    assert classify_engine_health(80) is EngineHealthClass.GOOD
    assert classify_engine_health(65) is EngineHealthClass.FAIR
    assert classify_engine_health(50) is EngineHealthClass.POOR
    assert classify_engine_health(20) is EngineHealthClass.CRITICAL


def test_release_gate_exposes_green_amber_red_labels() -> None:
    assert simulate_journey(seed=2, scenario=Scenario.IDEAL, num_steps=1)[0].health.traffic_light == "green"
    assert simulate_journey(seed=2, scenario=Scenario.STRESS, payload_class="overload", num_steps=24)[-1].health.traffic_light in {"amber", "red"}


def test_engine_load_is_fixed_and_heavy_payload_raises_stress_signals() -> None:
    heavy = simulate_journey(seed=31, scenario=Scenario.IDEAL, payload_class="heavy", num_steps=24)
    medium = simulate_journey(seed=31, scenario=Scenario.IDEAL, payload_class="medium", num_steps=24)

    assert {frame.engine_load_pct for frame in heavy} == {94.0}
    assert {frame.payload_class for frame in heavy} == {PayloadClass.HEAVY}
    assert all(frame.stress_level_pct >= 60.0 for frame in heavy)
    assert heavy[-1].engine.coolant_temp_c > medium[-1].engine.coolant_temp_c
    assert heavy[-1].suspension.hydraulic_pressure_bar > medium[-1].suspension.hydraulic_pressure_bar


def test_extended_sensor_groups_are_present_in_each_frame() -> None:
    frame = simulate_journey(seed=4, num_steps=8)[0]

    assert frame.engine.throttle_position_pct >= 0.0
    assert len(frame.engine.exhaust_temperatures_c) == 6
    assert frame.transmission.tos_rpm == frame.transmission.output_rpm
    assert frame.brake.accumulator_pressure_bar > 0.0
    assert len(frame.suspension.cylinder_pressures_bar) == 4
    assert frame.imu.accel_z >= 0.0
    assert frame.gnss.fix_type
    assert frame.dss.camera_status in {"online", "offline"}


def test_research_c_wtvc_cycle_has_paper_aligned_boundaries_and_energy_flow() -> None:
    journey = simulate_journey(
        seed=9,
        drive_cycle=DriveCycle.C_WTVC,
        start_condition="cold",
        duration_seconds=1800,
        timestep_s=10,
    )

    assert len(journey) == 181
    assert journey[0].elapsed_s == 0.0
    assert journey[90].elapsed_s == 900.0
    assert journey[90].cycle_phase == "highway"
    assert journey[137].cycle_phase == "expressway"
    assert journey[-1].elapsed_s == 1800.0
    assert journey[0].start_condition.value == "cold"
    assert journey[0].engine.coolant_temp_c < journey[-1].engine.coolant_temp_c
    assert 589.0 <= min(frame.engine.rpm for frame in journey)
    assert max(frame.engine.rpm for frame in journey) <= 1883.0
    late_expressway = [frame for frame in journey if frame.elapsed_s >= 1530.0]
    assert 15.0 <= sum(frame.engine.coolant_energy_pct for frame in late_expressway) / len(late_expressway) <= 30.0
    assert 28.0 <= sum(frame.engine.exhaust_energy_pct for frame in late_expressway) / len(late_expressway) <= 36.0
    assert all(frame.engine.coolant_outlet_temp_c >= frame.engine.coolant_inlet_temp_c for frame in journey)
