"""Generate all 171-sample C-WTVC simulation combinations as CSV.

This produces 3 scenarios x 5 payload classes x 2 start conditions x 32
manual-fault combinations = 960 journeys, with 171 samples per journey.
The values are synthetic and deterministic, not certified field data.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from telemetry_model import (
    BrakeTelemetry,
    EngineTelemetry,
    FaultFlags,
    IMUTelemetry,
    calculate_engine_health,
    simulate_journey,
)


SCENARIOS = ("ideal", "stress", "overhaul")
PAYLOAD_CLASSES = ("empty", "light", "medium", "heavy", "overload")
START_CONDITIONS = ("cold", "warm")
FAULT_NAMES = ("overheat", "bearing_failure", "low_oil", "brake_wear", "perception_degraded")


def apply_faults(frames: Iterable[Any], flags: Mapping[str, bool]) -> list[Any]:
    """Apply the same fault semantics exposed by the Streamlit controls."""

    adjusted = []
    for frame in frames:
        engine: EngineTelemetry = frame.engine
        imu: IMUTelemetry = frame.imu
        brake: BrakeTelemetry = frame.brake
        transmission = frame.transmission
        suspension = frame.suspension
        faults: FaultFlags = frame.faults
        gnss, lidar, radar = frame.gnss, frame.lidar, frame.radar

        if flags["overheat"]:
            engine = replace(
                engine,
                coolant_temp_c=min(145.0, engine.coolant_temp_c + 29.0),
                coolant_pump_temp_c=min(145.0, engine.coolant_pump_temp_c + 29.0),
                oil_temp_c=min(150.0, engine.oil_temp_c + 21.0),
                exhaust_temp_c=min(950.0, engine.exhaust_temp_c + 85.0),
                exhaust_temperatures_c=tuple(min(1_050.0, value + 85.0) for value in engine.exhaust_temperatures_c),
            )
            suspension = replace(suspension, hydraulic_oil_condition_pct=max(50.0, suspension.hydraulic_oil_condition_pct - 8.0))
            faults = replace(faults, engine_overheat=True)

        if flags["bearing_failure"]:
            engine = replace(engine, vibration_rms_g=min(1.6, engine.vibration_rms_g + 0.46), block_vibration_mm_s=min(14.0, engine.block_vibration_mm_s + 5.8))
            imu = replace(imu, angular_rate_x_dps=imu.angular_rate_x_dps + 4.0, angular_rate_y_dps=imu.angular_rate_y_dps + 4.8, vibration_rms_g=min(1.8, imu.vibration_rms_g + 0.55))
            transmission = replace(transmission, vibration_mm_s=min(9.0, transmission.vibration_mm_s + 3.4))
            faults = replace(faults, imu_fault=True)

        if flags["low_oil"]:
            engine = replace(engine, oil_pressure_kpa=max(85.0, engine.oil_pressure_kpa - 155.0), oil_level_pct=max(34.0, engine.oil_level_pct - 13.0))
            faults = replace(faults, low_oil_pressure=True)

        if flags["brake_wear"]:
            brake = replace(brake, pad_wear_pct=min(100.0, brake.pad_wear_pct + 32.0), stroke_pct=min(100.0, brake.stroke_pct + 25.0), disc_temp_c=min(180.0, brake.disc_temp_c + 24.0), oil_temp_c=min(150.0, brake.oil_temp_c + 20.0), cooling_temp_c=min(160.0, brake.cooling_temp_c + 22.0))
            faults = replace(faults, brake_wear=True)

        if flags["perception_degraded"]:
            gnss = replace(gnss, accuracy_m=max(10.0, gnss.accuracy_m + 12.0), satellites=min(5, gnss.satellites), fix_type="DEGRADED")
            lidar = replace(lidar, obstacle_distance_m=min(12.0, lidar.obstacle_distance_m), obstacle_confidence_pct=min(25.0, lidar.obstacle_confidence_pct), points_per_scan=min(12_000, lidar.points_per_scan), scan_rate_hz=min(4.0, lidar.scan_rate_hz), blocked=True, obstacle_class="sensor_obstruction")
            radar = replace(radar, target_confidence_pct=min(20.0, radar.target_confidence_pct), object_count=0, sensor_ok=False, target_class="sensor_fault")
            faults = replace(faults, gnss_degraded=True, lidar_obstructed=True, radar_fault=True)

        health = calculate_engine_health(engine, faults, payload_utilization_pct=frame.payload_utilization_pct, stress_level_pct=frame.stress_level_pct, scenario=frame.scenario, mode=frame.mode)
        adjusted.append(replace(frame, engine=engine, imu=imu, brake=brake, transmission=transmission, suspension=suspension, gnss=gnss, lidar=lidar, radar=radar, faults=faults, engine_health=health))
    return adjusted


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested telemetry dictionaries into CSV columns."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(item, child))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}.{index}"))
        return result
    return {prefix: value}


def _confidence(frame: Any) -> tuple[str, int]:
    """Use the dashboard's deterministic confidence-band convention."""

    classification = frame.engine_health.classification.value
    if classification in {"excellent", "good"}:
        return "High", 90
    if classification == "fair":
        return "Medium", 75
    return "Low", 55


def iter_rows(seed: int = 793):
    combination_index = 0
    for scenario, payload, start_condition, fault_values in itertools.product(
        SCENARIOS,
        PAYLOAD_CLASSES,
        START_CONDITIONS,
        itertools.product((False, True), repeat=len(FAULT_NAMES)),
    ):
        flags = dict(zip(FAULT_NAMES, fault_values))
        # Exactly 171 samples, including both endpoints of the 1,800-second
        # research proxy. The interval is 1800 / 170 seconds.
        base = simulate_journey(seed=seed, scenario=scenario, payload_class=payload, drive_cycle="c-wtvc", start_condition=start_condition, num_steps=171, timestep_s=1800.0 / 170.0)
        frames = apply_faults(base.frames, flags)
        combination_id = f"combo-{combination_index:04d}"
        fault_id = "+".join(name for name, enabled in flags.items() if enabled) or "none"

        for frame in frames:
            confidence, confidence_pct = _confidence(frame)
            row = _flatten(frame.as_dict())
            row.update({
                "combination_id": combination_id,
                "combination_index": combination_index,
                "simulation_seed": seed,
                "scenario_input": scenario,
                "payload_class_input": payload,
                "start_condition_input": start_condition,
                "fault_combination": fault_id,
                "fault_overheat": flags["overheat"],
                "fault_bearing_failure": flags["bearing_failure"],
                "fault_low_oil": flags["low_oil"],
                "fault_brake_wear": flags["brake_wear"],
                "fault_perception_degraded": flags["perception_degraded"],
                "engine_health_score": frame.engine_health.score,
                "engine_health_classification": frame.engine_health.classification.value,
                "engine_health_traffic_light": frame.engine_health.traffic_light,
                "rul_hours": frame.engine_health.rul_hours,
                "rul_low_hours": frame.engine_health.rul_range_hours[0],
                "rul_high_hours": frame.engine_health.rul_range_hours[1],
                "health_confidence": confidence,
                "health_confidence_pct": confidence_pct,
                "recommended_action": frame.engine_health.recommended_action,
                "active_fault_count": frame.faults.count,
                "faults_json": json.dumps(frame.faults.as_dict(), sort_keys=True, separators=(",", ":")),
                "operational_statuses_json": json.dumps(frame.operational_statuses, sort_keys=True, separators=(",", ":")),
            })
            yield row
        combination_index += 1


def generate_csv(output_path: Path, seed: int = 793) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = iter_rows(seed=seed)
    first = next(rows)
    fieldnames = list(first.keys())
    count = 0
    combinations: set[str] = set()
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in itertools.chain((first,), rows):
            writer.writerow(row)
            count += 1
            combinations.add(str(row["combination_id"]))
    return count, len(combinations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/dump_truck_cwtvc_all_combinations.csv"))
    parser.add_argument("--seed", type=int, default=793)
    args = parser.parse_args()
    rows, combinations = generate_csv(args.output, seed=args.seed)
    print(f"wrote {rows:,} rows across {combinations:,} combinations to {args.output}")


if __name__ == "__main__":
    main()
