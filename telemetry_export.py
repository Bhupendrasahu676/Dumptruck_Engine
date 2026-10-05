"""Deterministic, local telemetry export helpers.

The Streamlit page reruns whenever a widget changes, so this module keeps file
side effects behind an explicit ``export_journey`` call. JSONL is the
full-fidelity record; CSV is a compact analysis view.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from telemetry_model import TelemetryFrame


SCHEMA_VERSION = 2
MODEL_REVISION = "2026-10-02.c-wtvc"
SOURCE = "synthetic_dump_truck_digital_twin"


@dataclass(frozen=True)
class ExportReceipt:
    """Receipt returned after an explicit local export."""

    journey_id: str
    record_count: int
    jsonl_path: Path
    csv_path: Path
    already_existed: bool


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _manual_fault_map(manual_faults: Mapping[str, bool] | None) -> dict[str, bool]:
    return {
        str(name): bool(enabled)
        for name, enabled in sorted((manual_faults or {}).items(), key=lambda item: str(item[0]))
    }


def build_export_metadata(
    frames: Sequence[TelemetryFrame],
    *,
    seed: int,
    scenario: str,
    payload_class: str,
    manual_faults: Mapping[str, bool] | None = None,
    timestep_s: float = 1.0,
    rated_capacity_t: float = 40.0,
    drive_cycle: str = "synthetic_route",
    start_condition: str = "warm",
    cycle_source: str = "internal synthetic route",
    cycle_version: str = "route-1",
) -> dict[str, Any]:
    """Build stable provenance and a hash-derived journey ID.

    The identity intentionally excludes wall-clock time. Rerunning the same
    deterministic configuration therefore does not silently create duplicate
    log files.
    """

    faults = _manual_fault_map(manual_faults)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "model_revision": MODEL_REVISION,
        "seed": int(seed),
        "scenario": str(_value(scenario)),
        "payload_class": str(_value(payload_class)),
        "steps": len(frames),
        "timestep_s": float(timestep_s),
        "rated_capacity_t": float(rated_capacity_t),
        "drive_cycle": str(_value(drive_cycle)),
        "start_condition": str(_value(start_condition)),
        "cycle_source": cycle_source,
        "cycle_version": cycle_version,
        "manual_faults": faults,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20]
    return {
        **identity,
        "journey_id": f"journey-{digest}",
        "record_type": "telemetry_frame",
        "source": SOURCE,
        "synthetic": True,
    }


def render_jsonl(frames: Iterable[TelemetryFrame], metadata: Mapping[str, Any]) -> str:
    """Render one complete, self-identifying JSON object per telemetry frame."""

    lines: list[str] = []
    journey_id = str(metadata["journey_id"])
    for frame in frames:
        record = {
            **dict(metadata),
            "record_id": f"{journey_id}:{frame.step}",
            "frame_index": frame.step,
            "frame": frame.as_dict(),
        }
        lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    return "\n".join(lines) + ("\n" if lines else "")


CSV_FIELDS = [
    "journey_id", "record_id", "timestamp", "step", "elapsed_s", "scenario", "mode",
    "drive_cycle", "start_condition", "cycle_phase",
    "payload_class", "engine_load_pct", "engine_work_load_pct", "payload_mass_t", "rated_capacity_t",
    "payload_utilization_pct", "stress_level_pct", "engine_health_score",
    "health_classification", "traffic_light", "rul_hours", "rul_low_hours",
    "rul_high_hours", "recommended_action", "rpm", "throttle_position_pct",
    "coolant_temp_c", "oil_pressure_kpa", "oil_level_pct", "fuel_level_pct",
    "fuel_rate_lph", "exhaust_temp_c", "vibration_rms_g", "gear", "tos_rpm",
    "coolant_inlet_temp_c", "coolant_outlet_temp_c", "coolant_delta_t_c", "coolant_flow_kg_s",
    "thermostat_open_pct", "fuel_energy_input_kw", "coolant_energy_flow_kw", "exhaust_energy_flow_kw",
    "coolant_energy_pct", "exhaust_energy_pct", "transient_index_pct", "cumulative_fuel_energy_kwh",
    "cumulative_fuel_volume_l",
    "transmission_oil_temp_c", "powertrain_oil_pressure_bar", "axle_temp_c",
    "brake_oil_temp_c", "brake_cooling_temp_c", "brake_accumulator_pressure_bar",
    "brake_stroke_pct", "brake_wear_pct", "suspension_pressure_bar", "ride_height_mm",
    "body_position_pct", "hoist_angle_deg", "frame_stress_mpa", "frame_strain_microstrain",
    "chassis_pitch_deg", "rack_deg", "imu_accel_x_mps2", "imu_accel_y_mps2",
    "imu_accel_z_mps2", "imu_gyro_x_dps", "imu_gyro_y_dps", "imu_gyro_z_dps",
    "imu_roll_deg", "imu_pitch_deg", "imu_yaw_deg", "latitude_deg", "longitude_deg",
    "altitude_m", "speed_mps", "heading_deg", "gnss_accuracy_m", "gnss_satellites",
    "gnss_fix_type", "lidar_obstacle_distance_m", "lidar_blocked", "radar_target_distance_m",
    "radar_sensor_ok", "dss_alert_level", "dss_camera_online", "dss_fatigue_score_pct",
    "dss_distraction_score_pct", "faults_json", "manual_faults_json",
    "operational_statuses_json", "engine_block_vibration_mm_s", "coolant_pump_temp_c",
    "transmission_oil_pressure_kpa", "gearbox_vibration_mm_s", "hydraulic_oil_condition_pct",
    "hydraulic_actuator_vibration_mm_s", "coolant_condition_pct",
]


def _csv_row(frame: TelemetryFrame, metadata: Mapping[str, Any]) -> dict[str, Any]:
    health = frame.engine_health
    rul_low, rul_high = health.rul_range_hours
    engine = frame.engine
    transmission = frame.transmission
    brake = frame.brake
    suspension = frame.suspension
    imu = frame.imu
    gnss = frame.gnss
    lidar = frame.lidar
    radar = frame.radar
    dss = frame.dss
    faults = json.dumps(frame.faults.as_dict(), sort_keys=True, separators=(",", ":"))
    manual_faults = json.dumps(metadata.get("manual_faults", {}), sort_keys=True, separators=(",", ":"))
    return {
        "journey_id": metadata["journey_id"],
        "record_id": f"{metadata['journey_id']}:{frame.step}",
        "timestamp": frame.timestamp.isoformat(),
        "step": frame.step,
        "elapsed_s": frame.elapsed_s,
        "scenario": _value(frame.scenario),
        "mode": _value(frame.mode),
        "drive_cycle": _value(frame.drive_cycle),
        "start_condition": _value(frame.start_condition),
        "cycle_phase": frame.cycle_phase,
        "payload_class": _value(frame.payload_class),
        "engine_load_pct": frame.engine_load_pct,
        "engine_work_load_pct": engine.load_pct,
        "payload_mass_t": frame.payload_mass_t,
        "rated_capacity_t": frame.rated_capacity_t,
        "payload_utilization_pct": frame.payload_utilization_pct,
        "stress_level_pct": frame.stress_level_pct,
        "engine_health_score": health.score,
        "health_classification": _value(health.classification),
        "traffic_light": health.traffic_light,
        "rul_hours": health.rul_hours,
        "rul_low_hours": rul_low,
        "rul_high_hours": rul_high,
        "recommended_action": health.recommended_action,
        "rpm": engine.rpm,
        "throttle_position_pct": engine.throttle_position_pct,
        "coolant_temp_c": engine.coolant_temp_c,
        "oil_pressure_kpa": engine.oil_pressure_kpa,
        "oil_level_pct": engine.oil_level_pct,
        "fuel_level_pct": engine.fuel_level_pct,
        "fuel_rate_lph": engine.fuel_rate_lph,
        "exhaust_temp_c": engine.exhaust_temp_c,
        "vibration_rms_g": engine.vibration_rms_g,
        "coolant_inlet_temp_c": engine.coolant_inlet_temp_c,
        "coolant_outlet_temp_c": engine.coolant_outlet_temp_c,
        "coolant_delta_t_c": engine.coolant_delta_t_c,
        "coolant_flow_kg_s": engine.coolant_flow_kg_s,
        "thermostat_open_pct": engine.thermostat_open_pct,
        "fuel_energy_input_kw": engine.fuel_energy_input_kw,
        "coolant_energy_flow_kw": engine.coolant_energy_flow_kw,
        "exhaust_energy_flow_kw": engine.exhaust_energy_flow_kw,
        "coolant_energy_pct": engine.coolant_energy_pct,
        "exhaust_energy_pct": engine.exhaust_energy_pct,
        "transient_index_pct": engine.transient_index_pct,
        "cumulative_fuel_energy_kwh": frame.cumulative_fuel_energy_kwh,
        "cumulative_fuel_volume_l": frame.cumulative_fuel_volume_l,
        "gear": transmission.gear,
        "tos_rpm": transmission.output_rpm,
        "transmission_oil_temp_c": transmission.oil_temp_c,
        "powertrain_oil_pressure_bar": transmission.powertrain_oil_pressure_bar,
        "axle_temp_c": transmission.axle_temp_c,
        "brake_oil_temp_c": brake.oil_temp_c,
        "brake_cooling_temp_c": brake.cooling_temp_c,
        "brake_accumulator_pressure_bar": brake.accumulator_pressure_bar,
        "brake_stroke_pct": brake.stroke_pct,
        "brake_wear_pct": brake.pad_wear_pct,
        "suspension_pressure_bar": suspension.hydraulic_pressure_bar,
        "ride_height_mm": suspension.ride_height_mm,
        "body_position_pct": frame.body_position_pct,
        "hoist_angle_deg": frame.hoist_angle_deg,
        "frame_stress_mpa": frame.frame.stress_mpa,
        "frame_strain_microstrain": frame.frame.strain_microstrain,
        "chassis_pitch_deg": frame.chassis_pitch_deg,
        "rack_deg": frame.rack_deg,
        "imu_accel_x_mps2": imu.acceleration_x_mps2,
        "imu_accel_y_mps2": imu.acceleration_y_mps2,
        "imu_accel_z_mps2": imu.acceleration_z_mps2,
        "imu_gyro_x_dps": imu.angular_rate_x_dps,
        "imu_gyro_y_dps": imu.angular_rate_y_dps,
        "imu_gyro_z_dps": imu.angular_rate_z_dps,
        "imu_roll_deg": imu.roll_deg,
        "imu_pitch_deg": imu.pitch_deg,
        "imu_yaw_deg": imu.yaw_deg,
        "latitude_deg": gnss.latitude_deg,
        "longitude_deg": gnss.longitude_deg,
        "altitude_m": gnss.altitude_m,
        "speed_mps": gnss.speed_mps,
        "heading_deg": gnss.heading_deg,
        "gnss_accuracy_m": gnss.accuracy_m,
        "gnss_satellites": gnss.satellites,
        "gnss_fix_type": gnss.fix_type,
        "lidar_obstacle_distance_m": lidar.obstacle_distance_m,
        "lidar_blocked": lidar.blocked,
        "radar_target_distance_m": radar.target_distance_m,
        "radar_sensor_ok": radar.sensor_ok,
        "dss_alert_level": _value(dss.alert_level),
        "dss_camera_online": dss.camera_online,
        "dss_fatigue_score_pct": dss.fatigue_score_pct,
        "dss_distraction_score_pct": dss.distraction_score_pct,
        "faults_json": faults,
        "manual_faults_json": manual_faults,
        "operational_statuses_json": json.dumps(frame.operational_statuses, sort_keys=True, separators=(",", ":")),
        "engine_block_vibration_mm_s": engine.block_vibration_mm_s,
        "coolant_pump_temp_c": engine.coolant_pump_temp_c,
        "transmission_oil_pressure_kpa": transmission.oil_pressure_kpa,
        "gearbox_vibration_mm_s": transmission.vibration_mm_s,
        "hydraulic_oil_condition_pct": suspension.hydraulic_oil_condition_pct,
        "hydraulic_actuator_vibration_mm_s": suspension.actuator_vibration_mm_s,
        "coolant_condition_pct": frame.coolant_condition_pct,
    }


def render_csv(frames: Iterable[TelemetryFrame], metadata: Mapping[str, Any]) -> str:
    """Render a stable, compact CSV view of the same full-fidelity frames."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for frame in frames:
        writer.writerow(_csv_row(frame, metadata))
    return output.getvalue()


def _write_atomic(path: Path, content: str) -> None:
    """Write a complete file then atomically replace its final path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def export_journey(
    frames: Sequence[TelemetryFrame],
    *,
    export_dir: str | Path,
    seed: int,
    scenario: str,
    payload_class: str,
    manual_faults: Mapping[str, bool] | None = None,
    timestep_s: float = 1.0,
    rated_capacity_t: float = 40.0,
    drive_cycle: str = "synthetic_route",
    start_condition: str = "warm",
    cycle_source: str = "internal synthetic route",
    cycle_version: str = "route-1",
) -> ExportReceipt:
    """Persist one journey as idempotent JSONL and CSV files.

    This function is intentionally called only from an explicit Streamlit
    button. If the deterministic journey already exists, no file is rewritten.
    """

    frame_list = list(frames)
    metadata = build_export_metadata(
        frame_list,
        seed=seed,
        scenario=scenario,
        payload_class=payload_class,
        manual_faults=manual_faults,
        timestep_s=timestep_s,
        rated_capacity_t=rated_capacity_t,
        drive_cycle=drive_cycle,
        start_condition=start_condition,
        cycle_source=cycle_source,
        cycle_version=cycle_version,
    )
    root = Path(export_dir)
    journey_id = str(metadata["journey_id"])
    jsonl_path = root / f"{journey_id}.jsonl"
    csv_path = root / f"{journey_id}.csv"
    already_existed = jsonl_path.exists() and csv_path.exists()
    if not already_existed:
        _write_atomic(jsonl_path, render_jsonl(frame_list, metadata))
        _write_atomic(csv_path, render_csv(frame_list, metadata))
    return ExportReceipt(
        journey_id=journey_id,
        record_count=len(frame_list),
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        already_existed=already_existed,
    )
