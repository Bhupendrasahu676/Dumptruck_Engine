"""3D dump-truck digital twin dashboard.

Run with::

    streamlit run app_3d.py

The Python side creates a deterministic, synthetic journey. The browser side
replays that journey in a driver-focused dashboard and maps payload,
temperature, vibration, and hoist angle onto a procedural Three.js truck.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import streamlit as st
import streamlit.components.v1 as components

from telemetry_model import (
    BrakeTelemetry,
    EngineHealth,
    EngineTelemetry,
    FaultFlags,
    IMUTelemetry,
    DriveCycle,
    StartCondition,
    TelemetryFrame,
    calculate_engine_health,
    simulate_journey,
)
from telemetry_export import build_export_metadata, export_journey, render_csv, render_jsonl


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
TELEMETRY_LOG_DIR = ROOT / "telemetry_logs"
DASHBOARD_CSS = (STATIC / "dashboard.css").read_text(encoding="utf-8")
DIGITAL_TWIN_JS = (STATIC / "digital_twin.js").read_text(encoding="utf-8")

SCENARIO_OPTIONS = {
    "Ideal conditions": "ideal",
    "Stress haul": "stress",
    "Overhaul / post-service": "overhaul",
}

PAYLOAD_OPTIONS = {
    "Auto — locked by scenario": None,
    "Empty · 0% capacity": "empty",
    "Light · 48% capacity": "light",
    "Medium · 72% capacity": "medium",
    "Heavy · 94% capacity": "heavy",
    "Overload · 108% capacity": "overload",
}

CYCLE_OPTIONS = {
    "Research C-WTVC · 1800 s": "c-wtvc",
    "Compact synthetic route": "synthetic_route",
}

START_OPTIONS = {
    "Warm start · normal operating temperature": "warm",
    "Cold start · thermal warm-up": "cold",
}

MODE_LABELS = {
    "parked": "Parked",
    "idle": "Idle",
    "hauling": "Hauling",
    "upslope": "Upslope",
    "downslope": "Downslope",
    "braking": "Braking",
    "dumping": "Dumping",
}


def _mode_label(value: str) -> str:
    return MODE_LABELS.get(value, value.replace("_", " ").title())


def _safe_json(value: Any) -> str:
    """Serialize application-owned data without allowing a script close tag."""

    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).replace(
        "</", "<\\/"
    )


def _fault_label(name: str) -> str:
    return name.replace("_", " ").title()


def _apply_fault_injections(
    frames: Iterable[TelemetryFrame],
    *,
    overheat: bool = False,
    bearing_failure: bool = False,
    low_oil: bool = False,
    brake_wear: bool = False,
    perception_degraded: bool = False,
) -> list[TelemetryFrame]:
    """Apply explicit operator faults without changing the fixed payload class."""

    if not any((overheat, bearing_failure, low_oil, brake_wear, perception_degraded)):
        return list(frames)

    adjusted: list[TelemetryFrame] = []
    for frame in frames:
        engine: EngineTelemetry = frame.engine
        imu: IMUTelemetry = frame.imu
        brake: BrakeTelemetry = frame.brake
        transmission = frame.transmission
        suspension = frame.suspension
        faults: FaultFlags = frame.faults

        if overheat:
            engine = replace(
                engine,
                coolant_temp_c=min(145.0, engine.coolant_temp_c + 29.0),
                coolant_pump_temp_c=min(145.0, engine.coolant_pump_temp_c + 29.0),
                oil_temp_c=min(150.0, engine.oil_temp_c + 21.0),
                exhaust_temp_c=min(950.0, engine.exhaust_temp_c + 85.0),
                exhaust_temperatures_c=tuple(
                    min(1_050.0, temperature + 85.0)
                    for temperature in engine.exhaust_temperatures_c
                ),
            )
            faults = replace(faults, engine_overheat=True)

        if bearing_failure:
            engine = replace(
                engine,
                vibration_rms_g=min(1.6, engine.vibration_rms_g + 0.46),
                block_vibration_mm_s=min(14.0, engine.block_vibration_mm_s + 5.8),
            )
            imu = replace(
                imu,
                angular_rate_x_dps=imu.angular_rate_x_dps + 4.0,
                angular_rate_y_dps=imu.angular_rate_y_dps + 4.8,
                vibration_rms_g=min(1.8, imu.vibration_rms_g + 0.55),
            )
            transmission = replace(
                transmission,
                vibration_mm_s=min(9.0, transmission.vibration_mm_s + 3.4),
            )
            faults = replace(faults, imu_fault=True)

        if low_oil:
            engine = replace(
                engine,
                oil_pressure_kpa=max(85.0, engine.oil_pressure_kpa - 155.0),
                oil_level_pct=max(34.0, engine.oil_level_pct - 13.0),
            )
            faults = replace(faults, low_oil_pressure=True)

        if brake_wear:
            brake = replace(
                brake,
                pad_wear_pct=min(100.0, brake.pad_wear_pct + 32.0),
                stroke_pct=min(100.0, brake.stroke_pct + 25.0),
                disc_temp_c=min(180.0, brake.disc_temp_c + 24.0),
                oil_temp_c=min(150.0, brake.oil_temp_c + 20.0),
                cooling_temp_c=min(160.0, brake.cooling_temp_c + 22.0),
            )
            faults = replace(faults, brake_wear=True)

        if overheat:
            suspension = replace(
                suspension,
                hydraulic_oil_condition_pct=max(50.0, suspension.hydraulic_oil_condition_pct - 8.0),
            )

        if perception_degraded:
            gnss = replace(
                frame.gnss,
                accuracy_m=max(10.0, frame.gnss.accuracy_m + 12.0),
                satellites=min(5, frame.gnss.satellites),
                fix_type="DEGRADED",
            )
            lidar = replace(
                frame.lidar,
                obstacle_distance_m=min(12.0, frame.lidar.obstacle_distance_m),
                obstacle_confidence_pct=min(25.0, frame.lidar.obstacle_confidence_pct),
                points_per_scan=min(12_000, frame.lidar.points_per_scan),
                scan_rate_hz=min(4.0, frame.lidar.scan_rate_hz),
                blocked=True,
                obstacle_class="sensor_obstruction",
            )
            radar = replace(
                frame.radar,
                target_confidence_pct=min(20.0, frame.radar.target_confidence_pct),
                object_count=0,
                sensor_ok=False,
                target_class="sensor_fault",
            )
            faults = replace(
                faults,
                gnss_degraded=True,
                lidar_obstructed=True,
                radar_fault=True,
            )

        health: EngineHealth = calculate_engine_health(
            engine,
            faults,
            payload_utilization_pct=frame.payload_utilization_pct,
            stress_level_pct=frame.stress_level_pct,
            scenario=frame.scenario,
            mode=frame.mode,
        )
        adjusted.append(
            replace(
                frame,
                engine=engine,
                imu=imu,
                brake=brake,
                transmission=transmission,
                suspension=suspension,
                gnss=gnss if perception_degraded else frame.gnss,
                lidar=lidar if perception_degraded else frame.lidar,
                radar=radar if perception_degraded else frame.radar,
                faults=faults,
                engine_health=health,
            )
        )
    return adjusted


def _frame_to_dict(frame: TelemetryFrame) -> dict[str, Any]:
    """Return the complete nested frame for browser cards and sensor tables."""

    data = frame.as_dict()
    health = data.get("engine_health", {})
    if isinstance(health, dict):
        health["traffic_light"] = frame.engine_health.traffic_light
        health["rul_range_hours"] = list(frame.engine_health.rul_range_hours)
    return data


@st.cache_data(show_spinner=False)
def _journey(
    seed: int,
    scenario: str,
    payload_class: str | None,
    steps: int,
    drive_cycle: str = "synthetic_route",
    start_condition: str = "warm",
    timestep_s: float = 1.0,
):
    return simulate_journey(
        seed=seed,
        scenario=scenario,
        payload_class=payload_class,
        num_steps=steps,
        drive_cycle=drive_cycle,
        start_condition=start_condition,
        timestep_s=timestep_s,
    )


APP_CSS = """
.dt-alert { display:flex; align-items:center; gap:12px; margin:0 0 14px; padding:12px 14px;
  border:1px solid var(--dt-line); border-radius:12px; background:rgba(8,18,24,.72); }
.dt-alert__dot { flex:0 0 auto; width:10px; height:10px; border-radius:50%; background:var(--dt-good);
  box-shadow:0 0 0 5px rgba(70,211,154,.12); }
.dt-alert.is-warning .dt-alert__dot { background:var(--dt-warning); box-shadow:0 0 0 5px rgba(243,181,72,.12); }
.dt-alert.is-danger .dt-alert__dot { background:var(--dt-danger); box-shadow:0 0 0 5px rgba(255,107,97,.12); }
.dt-alert__content { display:flex; flex-direction:column; gap:3px; min-width:0; }
.dt-alert__title { color:var(--dt-text); font-size:.79rem; font-weight:750; }
.dt-alert__message { color:var(--dt-muted); font-size:.73rem; line-height:1.35; }
.dt-alert__action { margin-left:auto; color:var(--dt-muted); font-size:.68rem; text-align:right; white-space:nowrap; }
.dt-badge--fault { border-color:rgba(255,107,97,.42); color:var(--dt-danger); background:rgba(255,107,97,.08); }
.dt-badge--clear { border-color:rgba(70,211,154,.32); color:var(--dt-good); background:rgba(70,211,154,.06); }
.dt-scene-card { min-height:100%; }
.dt-scene-footer { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-top:10px; color:var(--dt-muted); font-size:.7rem; }
.dt-scene-footer__progress { flex:1; }
.dt-scene-footer__progress .dt-progress { margin-top:5px; }
.dt-pill { display:inline-flex; align-items:center; gap:6px; padding:6px 9px; border:1px solid var(--dt-line-strong);
  border-radius:999px; color:var(--dt-text); background:rgba(7,16,23,.64); font-size:.66rem; font-weight:720; letter-spacing:.06em; text-transform:uppercase; }
.dt-pill.is-warning { border-color:rgba(243,181,72,.42); color:var(--dt-warning); }
.dt-pill.is-danger { border-color:rgba(255,107,97,.42); color:var(--dt-danger); }
.dt-health-score { display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:2px 0 10px; }
.dt-health-score strong { color:var(--dt-text); font-size:2.75rem; letter-spacing:-.08em; line-height:1; }
.dt-health-score span { color:var(--dt-muted); font-size:.73rem; }
.dt-health-class { color:var(--dt-good); font-size:.74rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.dt-health-class.is-warning { color:var(--dt-warning); }
.dt-health-class.is-danger { color:var(--dt-danger); }
.dt-health-meta { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin:14px 0; }
.dt-health-meta__item { padding:10px; border:1px solid var(--dt-line); border-radius:9px; background:rgba(7,16,23,.48); }
.dt-health-meta__label { display:block; color:var(--dt-muted); font-size:.62rem; letter-spacing:.1em; text-transform:uppercase; }
.dt-health-meta__value { display:block; margin-top:5px; color:var(--dt-text); font-size:.88rem; font-weight:680; }
.dt-action { margin:0; padding:10px 11px; border-left:3px solid var(--dt-accent); background:rgba(243,181,72,.08); color:var(--dt-text); font-size:.74rem; line-height:1.45; }
.dt-factors { display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }
.dt-factor { padding:4px 7px; border:1px solid var(--dt-line); border-radius:999px; color:var(--dt-muted); font-size:.62rem; }
.dt-section { margin-top:14px; }
.dt-sensor-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }
.dt-sensor-card { min-width:0; padding:12px; border:1px solid var(--dt-line); border-radius:11px; background:rgba(7,16,23,.44); }
.dt-sensor-card--wide { grid-column:span 2; }
.dt-sensor-card__title { display:flex; align-items:center; justify-content:space-between; gap:8px; margin:0 0 8px; color:var(--dt-text); font-size:.76rem; font-weight:720; }
.dt-sensor-card__tag { color:var(--dt-muted); font-size:.59rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; }
.dt-sensor-row { display:flex; align-items:baseline; justify-content:space-between; gap:9px; padding:6px 0; border-top:1px solid rgba(145,190,204,.10); }
.dt-sensor-row:first-child { border-top:0; }
.dt-sensor-label { min-width:0; color:var(--dt-muted); font-size:.66rem; line-height:1.3; }
.dt-sensor-value { color:var(--dt-text); font-size:.7rem; font-variant-numeric:tabular-nums; font-weight:650; text-align:right; }
.dt-sensor-value.is-warning { color:var(--dt-warning); }
.dt-sensor-value.is-danger { color:var(--dt-danger); }
.dt-sensor-value.is-good { color:var(--dt-good); }
.dt-timeline { position:relative; width:100%; height:180px; overflow:hidden; border:1px solid var(--dt-line); border-radius:10px; background:rgba(7,16,23,.65); }
.dt-timeline canvas { display:block; width:100%; height:100%; }
.dt-chart-legend { display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; color:var(--dt-muted); font-size:.64rem; }
.dt-legend-item { display:inline-flex; align-items:center; gap:5px; }
.dt-legend-swatch { width:20px; height:3px; border-radius:99px; background:var(--dt-accent); }
.dt-legend-swatch.is-health { background:#46d39a; }
.dt-legend-swatch.is-temp { background:#ff6b61; }
.dt-legend-swatch.is-stress { background:#62b8d1; }
.dt-controls { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-top:10px; }
.dt-range { flex:1; min-width:180px; accent-color:var(--dt-accent); }
.dt-control-note { color:var(--dt-muted); font-size:.65rem; }
.dt-status-list { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:10px; }
.dt-operational-strip { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin:0 0 14px; padding:10px 12px; border:1px solid var(--dt-line); border-radius:10px; background:rgba(7,16,23,.56); color:var(--dt-muted); font-size:.68rem; font-variant-numeric:tabular-nums; }
.dt-operational-strip strong { color:var(--dt-text); font-size:.62rem; letter-spacing:.12em; text-transform:uppercase; }
.dt-status-cell { padding:9px; border:1px solid var(--dt-line); border-radius:9px; background:rgba(7,16,23,.42); }
.dt-status-cell__label { display:block; color:var(--dt-muted); font-size:.59rem; letter-spacing:.08em; text-transform:uppercase; }
.dt-status-cell__value { display:block; margin-top:4px; color:var(--dt-text); font-size:.72rem; font-weight:680; }
.dt-footer-note { margin-top:13px; color:var(--dt-muted); font-size:.65rem; line-height:1.45; }
@media (max-width:820px) { .dt-sensor-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .dt-sensor-card--wide { grid-column:span 2; } .dt-status-list { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media (max-width:500px) { .dt-sensor-grid { grid-template-columns:1fr; } .dt-sensor-card--wide { grid-column:span 1; } .dt-alert { align-items:flex-start; } .dt-alert__action { display:none; } }
"""


def _render_dashboard_html(
    frames: list[TelemetryFrame],
    *,
    selected_index: int,
    replay: bool,
    scenario_label: str,
    payload_locked: bool,
    active_faults: list[str],
) -> str:
    """Build the self-contained Streamlit component page."""

    selected_index = max(0, min(selected_index, max(0, len(frames) - 1)))
    payload = {
        "frames": [_frame_to_dict(frame) for frame in frames],
        "selectedIndex": selected_index,
        "replay": replay,
        "scenarioLabel": scenario_label,
        "payloadLocked": payload_locked,
        "activeFaults": active_faults,
    }
    fault_badge = (
        f"Manual faults ({len(active_faults)}): " + " · ".join(active_faults)
        if active_faults
        else "No manual faults selected"
    )
    fault_badge_class = "dt-badge dt-badge--fault" if active_faults else "dt-badge dt-badge--clear"
    app_script = r"""
    (function () {
      'use strict';
      const model = window.__DUMP_TRUCK_MODEL__ || { frames: [] };
      const frames = model.frames || [];
      if (!frames.length) return;
      let index = Math.max(0, Math.min(Number(model.selectedIndex || 0), frames.length - 1));
      let playing = Boolean(model.replay);
      let interval = null;
      const root = document.getElementById('dt-dashboard');
      const twinHost = document.getElementById('dt-truck-twin');
      const chart = document.getElementById('dt-trend-chart');
      const $ = (id) => document.getElementById(id);
      const text = (id, value) => { document.querySelectorAll(`#${id}`).forEach((el) => { el.textContent = value; }); };
      const pct = (value, digits = 0) => `${Number(value || 0).toFixed(digits)}%`;
      const num = (value, digits = 1) => Number(value || 0).toFixed(digits);
      const temp = (value) => `${num(value, 1)} °C`;
      const pressure = (value, unit = 'bar') => `${num(value, 1)} ${unit}`;
      const boolLabel = (value, yes = 'Online', no = 'Offline') => value ? yes : no;
      const modeLabel = (value) => ({ parked:'Parked', idle:'Idle', hauling:'Hauling', upslope:'Upslope', downslope:'Downslope', braking:'Braking', dumping:'Dumping' }[value] || String(value || '—'));
      const statusClass = (classification) => {
        if (classification === 'excellent' || classification === 'good') return 'is-good';
        if (classification === 'fair' || classification === 'poor') return 'is-warning';
        return 'is-danger';
      };
      const classificationLabel = (classification) => String(classification || 'unknown').replace(/_/g, ' ').toUpperCase();
      const trafficLabel = (classification) => (classification === 'critical' ? 'RED' : classification === 'fair' || classification === 'poor' ? 'AMBER' : 'GREEN');
      const setStateClass = (element, classification) => {
        if (!element) return;
        element.classList.remove('is-good', 'is-warning', 'is-danger');
        element.classList.add(statusClass(classification));
      };
      const operationalStatus = (key) => (frame().operational_statuses || {})[key] || 'amber';
      const markOperational = (id, key) => {
        document.querySelectorAll(`#${id}`).forEach((element) => {
          element.classList.remove('is-good', 'is-warning', 'is-danger');
          element.classList.add(operationalStatus(key) === 'green' ? 'is-good' : operationalStatus(key) === 'red' ? 'is-danger' : 'is-warning');
        });
      };
      const frame = () => frames[index];
      const engine = () => frame().engine || {};
      const imu = () => frame().imu || {};
      const trans = () => frame().transmission || {};
      const brakes = () => frame().brake || {};
      const suspension = () => frame().suspension || {};
      const chassis = () => frame().frame || {};
      const gnss = () => frame().gnss || {};
      const lidar = () => frame().lidar || {};
      const radar = () => frame().radar || {};
      const dss = () => frame().dss || {};
      const health = () => frame().engine_health || {};
      const payload = () => frame().payload || {};

      function updateHealth() {
        const h = health();
        const classification = h.classification || 'fair';
        text('dt-health-score', num(h.score, 0));
        text('dt-health-class', `${trafficLabel(classification)} · ${classificationLabel(classification)}`);
        const rulRange = Array.isArray(h.rul_range_hours) ? h.rul_range_hours : [h.rul_hours || 0, h.rul_hours || 0];
        text('dt-health-rul', `${num(rulRange[0], 0)}–${num(rulRange[1], 0)} h`);
        text('dt-health-confidence', classification === 'excellent' || classification === 'good' ? 'High' : classification === 'fair' ? 'Medium' : 'Low');
        text('dt-health-action', h.recommended_action || 'Review telemetry before release.');
        const healthBar = $('dt-health-bar');
        if (healthBar) healthBar.style.setProperty('--dt-progress', `${Math.max(0, Math.min(100, Number(h.score || 0)))}%`);
        setStateClass($('dt-health-class'), classification);
        const status = statusClass(classification);
        const banner = $('dt-alert');
        if (banner) {
          banner.classList.remove('is-warning', 'is-danger');
          if (status === 'is-warning') banner.classList.add('is-warning');
          if (status === 'is-danger') banner.classList.add('is-danger');
        }
        const title = classification === 'excellent' || classification === 'good' ? 'Release-ready system state' :
          classification === 'critical' || classification === 'poor' ? 'Do not deploy without inspection' : 'Caution — inspect before release';
        text('dt-alert-title', title);
        text('dt-alert-message', h.recommended_action || 'Continue monitoring live parameters.');
        text('dt-alert-action', `${trafficLabel(classification)} · ${num(h.score, 0)}/100`);
        const factorHost = $('dt-health-factors');
        if (factorHost) {
          factorHost.textContent = '';
          (h.contributing_factors || []).slice(0, 5).forEach((factor) => {
            const chip = document.createElement('span');
            chip.className = 'dt-factor';
            chip.textContent = factor;
            factorHost.appendChild(chip);
          });
          if (!factorHost.children.length) {
            const chip = document.createElement('span');
            chip.className = 'dt-factor';
            chip.textContent = 'No active penalty factors';
            factorHost.appendChild(chip);
          }
        }
      }

      function updateCards() {
        const f = frame();
        const e = engine();
        const i = imu();
        const t = trans();
        const b = brakes();
        const s = suspension();
        const c = chassis();
        const g = gnss();
        const l = lidar();
        const r = radar();
        const d = dss();
        const p = payload();
        const h = health();
        text('dt-mode', modeLabel(f.mode));
        text('dt-step', `Sample ${Number(f.step || 0) + 1} / ${frames.length}`);
        text('dt-clock', f.timestamp ? new Date(f.timestamp).toLocaleTimeString([], { hour:'2-digit', minute:'2-digit', second:'2-digit' }) : '—');
        text('dt-engine-load', pct(f.engine_load_pct));
        text('dt-engine-load-delta', `fixed category · actual work ${pct(e.load_pct)}`);
        text('dt-temp', temp(e.coolant_temp_c));
        text('dt-temp-delta', `${temp(e.oil_temp_c)} oil · ${num(e.exhaust_temp_c, 0)} °C exhaust`);
        text('dt-vibration', `${num(i.vibration_rms_g, 3)} g`);
        text('dt-vibration-delta', `${num(i.angular_rate_x_dps, 1)} / ${num(i.angular_rate_y_dps, 1)} dps gyro`);
        text('dt-payload', `${num(f.payload_utilization_pct, 0)}%`);
        text('dt-payload-delta', `${num(f.payload_mass_t, 1)} / ${num(f.rated_capacity_t, 0)} t · ${String(f.payload_class || '').toUpperCase()}`);
        text('dt-payload-badge', `${String(f.payload_class || '—').toUpperCase()} PAYLOAD · FIXED FOR JOURNEY`);
        text('dt-scene-speed', `${num(g.speed_mps * 3.6, 1)} km/h`);
        text('dt-scene-heading', `${num(g.heading_deg, 0)}° heading`);
        text('dt-health-rul-mini', `${num(h.rul_hours, 0)} h nominal`);
        text('dt-timepoint', `t + ${num(f.elapsed_s, 0)} s`);
        text('dt-cycle-phase', `${String(f.cycle_phase || 'route').toUpperCase()} · ${String(f.drive_cycle || 'synthetic_route').toUpperCase()}`);
        text('dt-start-condition', `${String(f.start_condition || 'warm').toUpperCase()} START`);
        const op = f.operational_statuses || {};
        const opGroups = [
          ['ENGINE', ['engine_block_temperature', 'engine_oil_temperature', 'exhaust_temperature', 'engine_oil_pressure', 'engine_block_vibration']],
          ['TRANSMISSION', ['gearbox_oil_temperature', 'transmission_oil_pressure', 'gearbox_vibration']],
          ['HYDRAULIC', ['hydraulic_oil_condition', 'hydraulic_actuator_vibration']],
          ['COOLANT', ['coolant_pump_temperature', 'coolant_condition']],
        ];
        text('dt-operational-summary', opGroups.map(([label, keys]) => `${label} ${keys.some((key) => op[key] === 'red') ? 'RED' : keys.some((key) => op[key] === 'amber') ? 'AMBER' : 'GREEN'}`).join(' · '));
        text('dt-payload-lock', model.payloadLocked ? 'Payload category is locked for this journey' : 'Payload category selected for this journey');
        text('dt-overhaul-progress', f.overhaul_progress_pct > 0 ? `${num(f.overhaul_progress_pct, 0)}% service settled` : '—');

        text('dt-rpm', `${num(e.rpm, 0)} rpm`);
        text('dt-throttle', pct(e.throttle_position_pct));
        text('dt-coolant', temp(e.coolant_temp_c));
        text('dt-coolant-inlet', temp(e.coolant_inlet_temp_c));
        text('dt-coolant-delta', `${num(e.coolant_delta_t_c, 1)} °C ΔT · ${num(e.coolant_flow_kg_s, 2)} kg/s`);
        text('dt-thermostat', pct(e.thermostat_open_pct));
        text('dt-oil-pressure', pressure(e.oil_pressure_kpa, 'kPa'));
        text('dt-oil-level', pct(e.oil_level_pct));
        text('dt-fuel-level', pct(e.fuel_level_pct));
        text('dt-fuel-rate', `${num(e.fuel_rate_lph, 1)} L/h`);
        text('dt-fuel-energy', `${num(e.fuel_energy_input_kw, 1)} kW input`);
        text('dt-energy-split', `Coolant ${pct(e.coolant_energy_pct, 0)} · Exhaust ${pct(e.exhaust_energy_pct, 0)}`);
        text('dt-cumulative-energy', `${num(f.cumulative_fuel_energy_kwh, 1)} kWh cumulative`);
        text('dt-air-filter', pressure(e.air_filter_restriction_kpa, 'kPa')); 
        text('dt-exhaust-bank', (e.exhaust_temperatures_c || []).map((value, n) => `C${n + 1} ${num(value, 0)}°`).join(' · ') || '—');
        text('dt-gear-status', String(t.status || '—').toUpperCase());
        text('dt-selected-gear', t.gear === 0 ? 'N' : `G${t.gear}`);
        text('dt-tos', `${num(t.output_rpm, 0)} rpm`);
        text('dt-powertrain-oil', pressure(t.powertrain_oil_pressure_bar));
        text('dt-trans-oil', temp(t.oil_temp_c));
        text('dt-axle-temp', temp(t.axle_temp_c));
        text('dt-slip', pct(t.slip_pct, 1));
        text('dt-brake-oil', temp(b.oil_temp_c));
        text('dt-brake-cooling', temp(b.cooling_temp_c));
        text('dt-accumulator', pressure(b.accumulator_pressure_bar));
        text('dt-brake-stroke', pct(b.stroke_pct));
        text('dt-brake-wear', pct(b.pad_wear_pct));
        text('dt-line-pressure', pressure(b.line_pressure_bar));
        text('dt-suspension-pressure', pressure(s.hydraulic_pressure_bar));
        text('dt-cylinder-pressures', `${num(s.front_left_pressure_bar, 0)} / ${num(s.front_right_pressure_bar, 0)} / ${num(s.rear_left_pressure_bar, 0)} / ${num(s.rear_right_pressure_bar, 0)} bar`);
        text('dt-ride-height', `${num(s.ride_height_mm, 0)} mm`);
        text('dt-body-position', pct(f.body_position_pct));
        text('dt-hoist-angle', `${num(f.hoist_angle_deg, 1)}°`);
        text('dt-frame-stress', `${num(c.stress_mpa, 1)} MPa`);
        text('dt-strain', `${num(c.strain_microstrain, 0)} µε`);
        text('dt-pitch', `${num(f.chassis_pitch_deg, 1)}° / ${num(f.rack_deg, 2)}°`);
        text('dt-accel', `${num(i.acceleration_x_mps2, 2)} / ${num(i.acceleration_y_mps2, 2)} / ${num(i.acceleration_z_mps2, 2)} m/s²`);
        text('dt-gyro', `${num(i.angular_rate_x_dps, 1)} / ${num(i.angular_rate_y_dps, 1)} / ${num(i.angular_rate_z_dps, 1)} °/s`);
        text('dt-attitude', `${num(i.roll_deg, 1)}° roll · ${num(i.pitch_deg, 1)}° pitch · ${num(i.yaw_deg, 1)}° yaw`);
        text('dt-gps', `${num(g.latitude_deg, 5)}, ${num(g.longitude_deg, 5)}`);
        text('dt-altitude', `${num(g.altitude_m, 1)} m`);
        text('dt-heading', `${num(g.heading_deg, 0)}°`);
        text('dt-gnss-quality', f.faults && f.faults.gnss_degraded ? `DEGRADED · ${g.satellites || 0} sats · ±${num(g.accuracy_m, 1)} m` : `${g.fix_type || '—'} · ${g.satellites || 0} sats · ±${num(g.accuracy_m, 1)} m`);
        text('dt-lidar', f.faults && f.faults.lidar_obstructed ? `OBSTRUCTED · ${num(l.obstacle_distance_m, 1)} m` : `${num(l.obstacle_distance_m, 1)} m · ${pct(l.obstacle_confidence_pct, 0)}`);
        text('dt-lidar-scan', `${num(l.points_per_scan, 0)} pts · ${num(l.scan_rate_hz, 1)} Hz`);
        text('dt-radar', `${num(r.target_distance_m, 1)} m · ${r.target_class || 'none'}`);
        text('dt-radar-quality', f.faults && f.faults.radar_fault ? `FAULT · ${pct(r.target_confidence_pct, 0)} confidence` : `${r.sensor_ok ? 'OK' : 'FAULT'} · ${pct(r.target_confidence_pct, 0)} confidence`);
        text('dt-dss-camera', `${boolLabel(d.camera_online)} · ${num(d.camera_fps, 0)} fps`);
        text('dt-driver-presence', pct(d.driver_presence_pct));
        text('dt-seatbelt', boolLabel(d.seatbelt_fastened, 'Fastened', 'Unfastened'));
        text('dt-dss-risk', `${pct(d.distraction_score_pct, 0)} distraction · ${pct(d.fatigue_score_pct, 0)} fatigue`);
        text('dt-dss-alert', d.driver_message || 'No active alerts');
        text('dt-fault-count', `${(Object.values(f.faults || {}).filter(Boolean)).length} active fault flags`);
        markOperational('dt-coolant', 'engine_block_temperature');
        markOperational('dt-coolant-inlet', 'coolant_pump_temperature');
        markOperational('dt-oil-pressure', 'engine_oil_pressure');
        markOperational('dt-exhaust-bank', 'exhaust_temperature');
        markOperational('dt-trans-oil', 'gearbox_oil_temperature');
        markOperational('dt-powertrain-oil', 'transmission_oil_pressure');
        markOperational('dt-slip', 'gearbox_vibration');
        markOperational('dt-suspension-pressure', 'hydraulic_oil_condition');
        markOperational('dt-cylinder-pressures', 'hydraulic_actuator_vibration');
        markOperational('dt-thermostat', 'coolant_condition');
        text('dt-fault-summary', Object.entries(f.faults || {}).filter(([, value]) => value).map(([key]) => key.replace(/_/g, ' ')).join(' · ') || 'No active faults');
        text('dt-timeline-position', `${index + 1} / ${frames.length}`);
        const timeline = $('dt-time-slider');
        if (timeline) timeline.value = String(index);
        const progress = $('dt-journey-progress');
        if (progress) progress.style.setProperty('--dt-progress', `${frames.length > 1 ? index / (frames.length - 1) * 100 : 100}%`);
        const play = $('dt-play');
        if (play) play.textContent = playing ? 'Pause replay' : 'Play replay';
        const modePill = $('dt-mode-pill');
        if (modePill) modePill.className = `dt-pill ${f.mode === 'braking' || f.mode === 'dumping' ? 'is-warning' : ''}`;
        const faultSummary = $('dt-fault-summary');
        if (faultSummary) faultSummary.className = Object.values(f.faults || {}).some(Boolean) ? 'dt-sensor-value is-warning' : 'dt-sensor-value is-good';
        if (window.__DUMP_TRUCK_TWIN__) {
          window.__DUMP_TRUCK_TWIN__.update({
            payloadPercent: f.payload_utilization_pct,
            maxPayload: f.rated_capacity_t,
            temperature: e.coolant_temp_c,
            vibration: i.vibration_rms_g,
            hoistAngle: f.hoist_angle_deg,
            speed: g.speed_mps,
            rpm: e.rpm
          });
        }
        updateHealth();
        drawTrend();
      }

      function drawTrend() {
        if (!chart) return;
        const ctx = chart.getContext('2d');
        const rect = chart.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        const width = Math.max(1, Math.round(rect.width * ratio));
        const height = Math.max(1, Math.round(rect.height * ratio));
        if (chart.width !== width || chart.height !== height) { chart.width = width; chart.height = height; }
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        const w = rect.width || 600;
        const h = rect.height || 180;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#071017'; ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = 'rgba(145,190,204,.13)'; ctx.lineWidth = 1;
        for (let y = 22; y < h - 18; y += (h - 40) / 3) { ctx.beginPath(); ctx.moveTo(12, y); ctx.lineTo(w - 12, y); ctx.stroke(); }
        const series = [
          { key: 'health_score', color: '#46d39a', scale: [0, 100], nested: 'health' },
          { key: 'coolant_temp_c', color: '#ff6b61', scale: [60, 140], nested: 'engine' },
          { key: 'stress_level_pct', color: '#62b8d1', scale: [0, 100] }
        ];
        const x = (n) => 14 + (w - 28) * (frames.length <= 1 ? 0 : n / (frames.length - 1));
        const valueOf = (f, s) => s.nested === 'health' ? Number((f.engine_health || {}).score || 0) : s.nested === 'engine' ? Number((f.engine || {}).coolant_temp_c || 0) : Number(f.stress_level_pct || 0);
        series.forEach((s) => {
          ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 1.8;
          frames.forEach((f, n) => {
            const raw = valueOf(f, s); const norm = Math.max(0, Math.min(1, (raw - s.scale[0]) / (s.scale[1] - s.scale[0])));
            const y = h - 20 - norm * (h - 42);
            if (n === 0) ctx.moveTo(x(n), y); else ctx.lineTo(x(n), y);
          }); ctx.stroke();
        });
        ctx.strokeStyle = '#f3b548'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
        ctx.beginPath(); ctx.moveTo(x(index), 10); ctx.lineTo(x(index), h - 12); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = '#7e9aa4'; ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText('0', 12, h - 5); ctx.fillText(`${frames.length - 1}s`, Math.max(12, w - 45), h - 5);
      }

      function setPlaying(next) {
        playing = next;
        if (interval) { window.clearInterval(interval); interval = null; }
        if (playing) {
          interval = window.setInterval(() => {
            index = index >= frames.length - 1 ? 0 : index + 1;
            updateCards();
          }, 900);
        }
        updateCards();
      }

      const initial = frame();
      const twinTelemetry = {
        payloadPercent: initial.payload_utilization_pct,
        maxPayload: initial.rated_capacity_t,
        temperature: initial.engine.coolant_temp_c,
        vibration: initial.imu.vibration_rms_g,
        hoistAngle: initial.hoist_angle_deg,
        speed: initial.gnss.speed_mps,
        rpm: initial.engine.rpm
      };
      if (window.DigitalTwinTruck && twinHost) {
        window.__DUMP_TRUCK_TWIN__ = window.DigitalTwinTruck.init(twinHost, {
          telemetry: twinTelemetry,
          config: { showHud: true, height: 460, maxPayload: initial.rated_capacity_t, maxBedAngle: 38 }
        });
      }
      if ($('dt-play')) $('dt-play').addEventListener('click', () => setPlaying(!playing));
      if ($('dt-prev')) $('dt-prev').addEventListener('click', () => { index = index <= 0 ? frames.length - 1 : index - 1; updateCards(); });
      if ($('dt-next')) $('dt-next').addEventListener('click', () => { index = index >= frames.length - 1 ? 0 : index + 1; updateCards(); });
      if ($('dt-time-slider')) $('dt-time-slider').addEventListener('input', (event) => { index = Number(event.target.value || 0); updateCards(); });
      window.addEventListener('resize', drawTrend);
      updateCards();
      setPlaying(playing);
    }());
    """

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>"
        + DASHBOARD_CSS
        + APP_CSS
        + "</style></head><body>"
        + "<main class='dt-dashboard' id='dt-dashboard'>"
        + "<header class='dt-dashboard-header'><div>"
        + "<p class='dt-eyebrow'>Caterpillar 793F · synthetic telemetry</p>"
        + "<h1 class='dt-title'>Digital twin control surface</h1>"
        + "<p class='dt-subtitle'>Real-time simulation of machine health, load response, IMU attitude and perimeter awareness. Payload class is categorical and stays fixed for the journey.</p>"
        + "<div class='dt-control-row' style='margin-top:10px'><span class='dt-badge'>"
        + scenario_label
        + "</span><span class='"
        + fault_badge_class
        + "'>"
        + fault_badge
        + "</span></div></div><div class='dt-status'>SIMULATION ONLINE</div></header>"
        + "<section class='dt-alert' id='dt-alert'><span class='dt-alert__dot'></span><div class='dt-alert__content'><strong class='dt-alert__title' id='dt-alert-title'>Caution</strong><span class='dt-alert__message' id='dt-alert-message'>Loading engine health…</span></div><span class='dt-alert__action' id='dt-alert-action'>—</span></section>"
        + "<div class='dt-operational-strip'><strong>Reference operational bands</strong><span id='dt-operational-summary'>ENGINE — · TRANSMISSION — · HYDRAULIC — · COOLANT —</span><span class='dt-kicker'>Green = workable · Amber = caution · Red = alert</span></div>"
        + "<section class='dt-grid dt-grid--metrics'>"
        + "<article class='dt-metric'><span class='dt-metric__label'>Engine health</span><strong class='dt-metric__value'><span id='dt-health-score'>—</span><span class='dt-metric__unit'>/ 100</span></strong><span class='dt-metric__delta' id='dt-health-class'>—</span></article>"
        + "<article class='dt-metric'><span class='dt-metric__label'>Journey load lock</span><strong class='dt-metric__value' id='dt-engine-load'>—</strong><span class='dt-metric__delta' id='dt-engine-load-delta'>—</span></article>"
        + "<article class='dt-metric'><span class='dt-metric__label'>Coolant temperature</span><strong class='dt-metric__value' id='dt-temp'>—</strong><span class='dt-metric__delta' id='dt-temp-delta'>—</span></article>"
        + "<article class='dt-metric'><span class='dt-metric__label'>Payload / load lock</span><strong class='dt-metric__value' id='dt-payload'>—</strong><span class='dt-metric__delta' id='dt-payload-delta'>—</span></article>"
        + "<article class='dt-metric'><span class='dt-metric__label'>Research cycle phase</span><strong class='dt-metric__value' id='dt-cycle-phase'>—</strong><span class='dt-metric__delta' id='dt-start-condition'>—</span></article>"
        + "</section>"
        + "<section class='dt-grid dt-grid--main'>"
        + "<article class='dt-card dt-scene-card'><div class='dt-card__head'><div><span class='dt-card__label'>3D physical response</span><h2 class='dt-card__title'>Suspension · thermal state · dump body</h2></div><span class='dt-pill' id='dt-mode-pill'><span id='dt-mode'>—</span></span></div><div id='dt-truck-twin' class='dt-container'></div><div class='dt-scene-footer'><div class='dt-scene-footer__progress'><span id='dt-step'>—</span><div class='dt-progress'><div class='dt-progress__bar' id='dt-journey-progress'></div></div></div><div><strong id='dt-scene-speed'>—</strong><br><span id='dt-scene-heading'>—</span></div></div><div class='dt-controls'><button type='button' class='dt-button' id='dt-prev'>Previous</button><button type='button' class='dt-button' id='dt-play'>Play replay</button><button type='button' class='dt-button' id='dt-next'>Next</button><input class='dt-range' id='dt-time-slider' type='range' min='0' max='0' value='0' step='1'><span class='dt-control-note' id='dt-timeline-position'>—</span></div></article>"
        + "<aside class='dt-card'><div class='dt-card__head'><div><span class='dt-card__label'>Release gate</span><h2 class='dt-card__title'>Engine health interpretation</h2></div><span class='dt-status' id='dt-health-status'>LIVE</span></div><div class='dt-health-score'><strong id='dt-health-score'>—</strong><span>normalised score</span></div><div class='dt-health-class' id='dt-health-class'>—</div><div class='dt-progress' style='margin-top:9px'><div class='dt-progress__bar' id='dt-health-bar'></div></div><div class='dt-health-meta'><div class='dt-health-meta__item'><span class='dt-health-meta__label'>Remaining useful life</span><strong class='dt-health-meta__value' id='dt-health-rul'>—</strong></div><div class='dt-health-meta__item'><span class='dt-health-meta__label'>Confidence band</span><strong class='dt-health-meta__value' id='dt-health-confidence'>—</strong></div></div><p class='dt-action' id='dt-health-action'>—</p><div class='dt-factors' id='dt-health-factors'></div><div class='dt-status-list'><div class='dt-status-cell'><span class='dt-status-cell__label'>Mode</span><strong class='dt-status-cell__value' id='dt-mode'>—</strong></div><div class='dt-status-cell'><span class='dt-status-cell__label'>Timepoint</span><strong class='dt-status-cell__value' id='dt-timepoint'>—</strong></div><div class='dt-status-cell'><span class='dt-status-cell__label'>RUL</span><strong class='dt-status-cell__value' id='dt-health-rul-mini'>—</strong></div><div class='dt-status-cell'><span class='dt-status-cell__label'>Service</span><strong class='dt-status-cell__value' id='dt-overhaul-progress'>—</strong></div></div></aside>"
        + "</section>"
        + "<section class='dt-section dt-card'><div class='dt-card__head'><div><span class='dt-card__label'>Journey trend</span><h2 class='dt-card__title'>Health, coolant and stress over the route</h2></div><span class='dt-kicker' id='dt-clock'>—</span></div><div class='dt-timeline'><canvas id='dt-trend-chart'></canvas></div><div class='dt-chart-legend'><span class='dt-legend-item'><i class='dt-legend-swatch is-health'></i>Health score</span><span class='dt-legend-item'><i class='dt-legend-swatch is-temp'></i>Coolant temperature</span><span class='dt-legend-item'><i class='dt-legend-swatch is-stress'></i>Stress level</span><span class='dt-legend-item'><i class='dt-legend-swatch'></i>Selected sample</span></div></section>"
        + "<section class='dt-section dt-card'><div class='dt-card__head'><div><span class='dt-card__label'>Machine state</span><h2 class='dt-card__title'>Driver essentials</h2></div><span class='dt-badge' id='dt-payload-badge'>—</span></div><div class='dt-sensor-grid'>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>Engine <span class='dt-sensor-card__tag'>Power + thermal</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Speed</span><strong class='dt-sensor-value' id='dt-rpm'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Throttle position</span><strong class='dt-sensor-value' id='dt-throttle'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Coolant outlet</span><strong class='dt-sensor-value' id='dt-coolant'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Coolant inlet</span><strong class='dt-sensor-value' id='dt-coolant-inlet'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Coolant ΔT / flow</span><strong class='dt-sensor-value' id='dt-coolant-delta'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Thermostat open</span><strong class='dt-sensor-value' id='dt-thermostat'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Oil pressure</span><strong class='dt-sensor-value' id='dt-oil-pressure'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Oil level</span><strong class='dt-sensor-value' id='dt-oil-level'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Air filter restriction</span><strong class='dt-sensor-value' id='dt-air-filter'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>Fuel & exhaust <span class='dt-sensor-card__tag'>Energy flow</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Fuel level</span><strong class='dt-sensor-value' id='dt-fuel-level'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Consumption rate</span><strong class='dt-sensor-value' id='dt-fuel-rate'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Fuel energy input</span><strong class='dt-sensor-value' id='dt-fuel-energy'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Energy split</span><strong class='dt-sensor-value' id='dt-energy-split'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Journey fuel energy</span><strong class='dt-sensor-value' id='dt-cumulative-energy'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Exhaust manifold</span><strong class='dt-sensor-value' id='dt-exhaust-bank'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Engine vibration</span><strong class='dt-sensor-value' id='dt-vibration'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Fault flags</span><strong class='dt-sensor-value' id='dt-fault-count'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>Transmission <span class='dt-sensor-card__tag'>Driveline</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Gear status</span><strong class='dt-sensor-value' id='dt-gear-status'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Selected gear</span><strong class='dt-sensor-value' id='dt-selected-gear'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Output speed (TOS)</span><strong class='dt-sensor-value' id='dt-tos'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Powertrain oil pressure</span><strong class='dt-sensor-value' id='dt-powertrain-oil'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Transmission oil</span><strong class='dt-sensor-value' id='dt-trans-oil'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Axle temperature</span><strong class='dt-sensor-value' id='dt-axle-temp'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>Brakes <span class='dt-sensor-card__tag'>Stopping</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Brake oil temperature</span><strong class='dt-sensor-value' id='dt-brake-oil'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Cooling temperature</span><strong class='dt-sensor-value' id='dt-brake-cooling'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Accumulator pressure</span><strong class='dt-sensor-value' id='dt-accumulator'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Brake stroke</span><strong class='dt-sensor-value' id='dt-brake-stroke'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Wear indicator</span><strong class='dt-sensor-value' id='dt-brake-wear'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Line pressure</span><strong class='dt-sensor-value' id='dt-line-pressure'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>Suspension & frame <span class='dt-sensor-card__tag'>Structure</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Payload / rated capacity</span><strong class='dt-sensor-value' id='dt-payload-delta'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Cylinder pressure</span><strong class='dt-sensor-value' id='dt-suspension-pressure'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Cylinder set F-L / F-R / R-L / R-R</span><strong class='dt-sensor-value' id='dt-cylinder-pressures'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Body position / hoist</span><strong class='dt-sensor-value' id='dt-body-position'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Hoist angle</span><strong class='dt-sensor-value' id='dt-hoist-angle'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Frame stress / strain</span><strong class='dt-sensor-value' id='dt-frame-stress'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Chassis pitch / rack</span><strong class='dt-sensor-value' id='dt-pitch'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>IMU <span class='dt-sensor-card__tag'>6-axis + attitude</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Acceleration X / Y / Z</span><strong class='dt-sensor-value' id='dt-accel'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Gyro X / Y / Z</span><strong class='dt-sensor-value' id='dt-gyro'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Roll / pitch / yaw</span><strong class='dt-sensor-value' id='dt-attitude'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Vibration RMS</span><strong class='dt-sensor-value' id='dt-vibration'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>GNSS speed</span><strong class='dt-sensor-value' id='dt-scene-speed'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>GNSS / GPS <span class='dt-sensor-card__tag'>Position</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>Latitude / longitude</span><strong class='dt-sensor-value' id='dt-gps'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Altitude</span><strong class='dt-sensor-value' id='dt-altitude'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Heading</span><strong class='dt-sensor-value' id='dt-heading'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Fix / quality</span><strong class='dt-sensor-value' id='dt-gnss-quality'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Sample timestamp</span><strong class='dt-sensor-value' id='dt-clock'>—</strong></div></div>"
        + "<div class='dt-sensor-card'><h3 class='dt-sensor-card__title'>Perimeter & DSS <span class='dt-sensor-card__tag'>Awareness</span></h3><div class='dt-sensor-row'><span class='dt-sensor-label'>LiDAR perimeter</span><strong class='dt-sensor-value' id='dt-lidar'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>LiDAR scan</span><strong class='dt-sensor-value' id='dt-lidar-scan'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Radar target</span><strong class='dt-sensor-value' id='dt-radar'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Radar quality</span><strong class='dt-sensor-value' id='dt-radar-quality'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>In-cab camera</span><strong class='dt-sensor-value' id='dt-dss-camera'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Driver / seatbelt</span><strong class='dt-sensor-value' id='dt-driver-presence'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>DSS risk</span><strong class='dt-sensor-value' id='dt-dss-risk'>—</strong></div><div class='dt-sensor-row'><span class='dt-sensor-label'>Driver alert</span><strong class='dt-sensor-value' id='dt-dss-alert'>—</strong></div></div>"
        + "</div></section>"
        + "<p class='dt-footer-note'><span id='dt-payload-lock'>Payload category is locked for this journey</span>. Values are synthetic, bounded telemetry for simulation and interface development; they are not validated maintenance evidence. Use the release gate as a demonstration of the requested health logic, not as a certified safety decision.</p>"
        + "</main>"
        + "<script>window.__DUMP_TRUCK_MODEL__="
        + _safe_json(payload)
        + ";</script><script>"
        + DIGITAL_TWIN_JS
        + "</script><script>"
        + app_script
        + "</script></body></html>"
    )
    return html


st.set_page_config(
    page_title="3D Dump Truck Digital Twin",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #071017; }
      [data-testid="stHeader"] { background: transparent; }
      [data-testid="stSidebar"] { background: #0a151c; border-right: 1px solid rgba(145,190,204,.14); }
      [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
      .stMarkdown, .stCaption { color: #b6cbd1; }
      div[data-testid="stMetric"] { background: rgba(16,33,42,.72); border: 1px solid rgba(145,190,204,.14); border-radius: 10px; padding: .7rem; }
      .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 🚜 Digital twin")
st.sidebar.caption("Synthetic Caterpillar 793F telemetry lab")
st.sidebar.info("New: Engine Evidence Lab in page navigation adds persistent engine history, scheduled faults, independent diagnosis and gated RUL. This page retains the legacy synthetic health/RUL rules.")

scenario_label = st.sidebar.selectbox(
    "Operating scenario",
    list(SCENARIO_OPTIONS),
    help="Ideal, stress, and post-service journeys use different deterministic sensor profiles.",
)
scenario = SCENARIO_OPTIONS[scenario_label]

payload_label = st.sidebar.selectbox(
    "Payload class · fixed for journey",
    list(PAYLOAD_OPTIONS),
    index=0,
    help="Payload is categorical. It is selected once for this generated journey and does not drift sample-to-sample.",
)
payload_class = PAYLOAD_OPTIONS[payload_label]
st.sidebar.info("Payload class is locked for the generated journey. Capacity stress begins above 85%.")

cycle_label = st.sidebar.selectbox(
    "Drive cycle",
    list(CYCLE_OPTIONS),
    help="The research proxy follows the paper's 1,800 s C-WTVC split: urban 0–900 s, highway 900–1368 s, expressway 1368–1800 s.",
)
drive_cycle = CYCLE_OPTIONS[cycle_label]
start_label = st.sidebar.selectbox("Start condition", list(START_OPTIONS), index=0)
start_condition = START_OPTIONS[start_label]

st.sidebar.subheader("Fault injection")
st.sidebar.caption("These synthetic faults are applied to every generated sample and are clearly marked in the log.")
overheat = st.sidebar.checkbox("🔥 Coolant failure / overheat", key="fault_overheat")
bearing_failure = st.sidebar.checkbox("⚙️ Bearing failure / vibration", key="fault_bearing")
low_oil = st.sidebar.checkbox("🛢️ Low engine oil pressure", key="fault_oil")
brake_wear = st.sidebar.checkbox("🛑 Brake wear / long stroke", key="fault_brakes")
perception_degraded = st.sidebar.checkbox("📡 Perception sensor degradation", key="fault_perception")

with st.sidebar.expander("Replay controls", expanded=False):
    seed = st.number_input("Deterministic seed", min_value=0, max_value=999_999, value=793, step=1)
    if drive_cycle == "c-wtvc":
        sample_interval = st.select_slider("Research cycle sample interval", options=[5, 10, 20, 30], value=10, format_func=lambda value: f"{value} s")
        steps = int(1800 / sample_interval) + 1
        st.caption(f"{steps} samples including t=0 and t=1800 s")
    else:
        sample_interval = 1
        steps = st.slider("Journey samples", min_value=24, max_value=120, value=72, step=12)
    replay = st.checkbox("Replay journey in browser", value=True)

journey = _journey(int(seed), scenario, payload_class, int(steps), drive_cycle, start_condition, float(sample_interval))
frames = _apply_fault_injections(
    journey.frames,
    overheat=overheat,
    bearing_failure=bearing_failure,
    low_oil=low_oil,
    brake_wear=brake_wear,
    perception_degraded=perception_degraded,
)

active_faults = [
    label
    for enabled, label in (
        (overheat, "Coolant failure"),
        (bearing_failure, "Bearing failure"),
        (low_oil, "Low oil pressure"),
        (brake_wear, "Brake wear"),
        (perception_degraded, "Perception degraded"),
    )
    if enabled
]
manual_faults = {
    "engine_overheat": overheat,
    "bearing_failure": bearing_failure,
    "low_oil_pressure": low_oil,
    "brake_wear": brake_wear,
    "perception_degraded": perception_degraded,
}

if not frames:
    st.error("No telemetry frames were generated.")
    st.stop()

if active_faults:
    st.warning(
        f"Manual fault injection ACTIVE ({len(active_faults)}): "
        f"{', '.join(active_faults)}. Applied to all {len(frames)} generated samples.",
        icon="⚠️",
    )
else:
    st.info(
        "No manual faults active. Use the visible Fault injection controls in the sidebar to test a failure mode.",
        icon="ℹ️",
    )

export_metadata = build_export_metadata(
    frames,
    seed=int(seed),
    scenario=scenario,
    payload_class=frames[0].payload_class.value,
    manual_faults=manual_faults,
    timestep_s=journey.config.timestep_s,
    rated_capacity_t=frames[0].rated_capacity_t,
    drive_cycle=journey.config.drive_cycle.value,
    start_condition=journey.config.start_condition.value,
    cycle_source=journey.config.cycle_source,
    cycle_version=journey.config.cycle_version,
)
jsonl_data = render_jsonl(frames, export_metadata)
csv_data = render_csv(frames, export_metadata)
journey_id = str(export_metadata["journey_id"])

st.subheader("Telemetry logging")
st.caption(
    "Telemetry is kept in memory while you adjust controls. It is not silently written on every Streamlit rerun. "
    "Use the button below to persist this exact journey locally, or download it for another system."
)
log_col1, log_col2, log_col3 = st.columns([1.05, 1.05, 2.4])
with log_col1:
    if st.button("Record journey locally", type="primary", use_container_width=True):
        receipt = export_journey(
            frames,
            export_dir=TELEMETRY_LOG_DIR,
            seed=int(seed),
            scenario=scenario,
            payload_class=frames[0].payload_class.value,
            manual_faults=manual_faults,
            timestep_s=journey.config.timestep_s,
            rated_capacity_t=frames[0].rated_capacity_t,
            drive_cycle=journey.config.drive_cycle.value,
            start_condition=journey.config.start_condition.value,
            cycle_source=journey.config.cycle_source,
            cycle_version=journey.config.cycle_version,
        )
        if receipt.already_existed:
            st.info(f"Already recorded: {receipt.journey_id} ({receipt.record_count} samples).")
        else:
            st.success(
                f"Recorded {receipt.record_count} samples to "
                f"{receipt.jsonl_path.name} and {receipt.csv_path.name}."
            )
with log_col2:
    st.download_button(
        "Download JSONL log",
        data=jsonl_data,
        file_name=f"{journey_id}.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )
with log_col3:
    st.download_button(
        "Download CSV analysis",
        data=csv_data,
        file_name=f"{journey_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
st.caption(f"Journey ID: `{journey_id}` · Local files are written under `{TELEMETRY_LOG_DIR}` only when you press Record journey locally.")

st.sidebar.divider()
st.sidebar.metric("Current scenario", scenario_label)
st.sidebar.metric("Journey payload", f"{frames[0].payload_utilization_pct:.0f}% · {frames[0].payload_class.value.title()}")
st.sidebar.caption("Use the replay controls inside the dashboard to inspect each mode: parked, idle, hauling, upslope, downslope, braking and dumping.")

dashboard_html = _render_dashboard_html(
    frames,
    selected_index=0,
    replay=replay,
    scenario_label=scenario_label,
    payload_locked=True,
    active_faults=active_faults,
)
components.html(dashboard_html, height=2_520, scrolling=True)
