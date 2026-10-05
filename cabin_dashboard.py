"""Responsive Android/tablet-style cabin dashboard for the dump-truck twin.

Run with ``streamlit run cabin_dashboard.py --server.port 8502``.  The Python
side owns the deterministic journey and fault injections; the embedded browser
surface renders one live frame at a time and can replay the complete journey.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import streamlit as st
import streamlit.components.v1 as components

from operational_ranges import classify
from telemetry_model import (
    BrakeTelemetry,
    EngineTelemetry,
    FaultFlags,
    IMUTelemetry,
    TelemetryFrame,
    calculate_engine_health,
    simulate_journey,
)


ROOT = Path(__file__).resolve().parent
DIGITAL_TWIN_JS = (ROOT / "static" / "digital_twin.js").read_text(encoding="utf-8")

SCENARIOS = {
    "Ideal conditions": "ideal",
    "Stress haul": "stress",
    "Overhaul / post-service": "overhaul",
}
PAYLOADS = {
    "Auto — scenario default": None,
    "Empty · 0%": "empty",
    "Light · 48%": "light",
    "Medium · 72%": "medium",
    "Heavy · 94%": "heavy",
    "Overload · 108%": "overload",
}
STARTS = {
    "Warm start": "warm",
    "Cold start": "cold",
}


def _safe_json(value: Any) -> str:
    """Keep application-owned frame data safe when embedded in a script tag."""

    return json.dumps(value, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")


def _apply_fault_injections(
    frames: Iterable[TelemetryFrame],
    *,
    overheat: bool = False,
    bearing_failure: bool = False,
    low_oil: bool = False,
    brake_wear: bool = False,
    perception_degraded: bool = False,
) -> list[TelemetryFrame]:
    """Apply the five operator toggles to every generated sample."""

    adjusted: list[TelemetryFrame] = []
    for frame in frames:
        engine = frame.engine
        imu = frame.imu
        brake = frame.brake
        transmission = frame.transmission
        suspension = frame.suspension
        faults = frame.faults
        gnss, lidar, radar = frame.gnss, frame.lidar, frame.radar

        if overheat:
            engine = replace(
                engine,
                coolant_temp_c=min(145.0, engine.coolant_temp_c + 29.0),
                coolant_pump_temp_c=min(145.0, engine.coolant_pump_temp_c + 29.0),
                oil_temp_c=min(150.0, engine.oil_temp_c + 21.0),
                exhaust_temp_c=min(950.0, engine.exhaust_temp_c + 85.0),
                exhaust_temperatures_c=tuple(min(1_050.0, value + 85.0) for value in engine.exhaust_temperatures_c),
            )
            faults = replace(faults, engine_overheat=True)
            suspension = replace(
                suspension,
                hydraulic_oil_condition_pct=max(50.0, suspension.hydraulic_oil_condition_pct - 8.0),
            )
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
            transmission = replace(transmission, vibration_mm_s=min(9.0, transmission.vibration_mm_s + 3.4))
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
        if perception_degraded:
            gnss = replace(gnss, accuracy_m=max(10.0, gnss.accuracy_m + 12.0), satellites=min(5, gnss.satellites), fix_type="DEGRADED")
            lidar = replace(
                lidar,
                obstacle_distance_m=min(12.0, lidar.obstacle_distance_m),
                obstacle_confidence_pct=min(25.0, lidar.obstacle_confidence_pct),
                points_per_scan=min(12_000, lidar.points_per_scan),
                scan_rate_hz=min(4.0, lidar.scan_rate_hz),
                blocked=True,
                obstacle_class="sensor_obstruction",
            )
            radar = replace(radar, target_confidence_pct=min(20.0, radar.target_confidence_pct), object_count=0, sensor_ok=False, target_class="sensor_fault")
            faults = replace(faults, gnss_degraded=True, lidar_obstructed=True, radar_fault=True)

        health = calculate_engine_health(
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
                gnss=gnss,
                lidar=lidar,
                radar=radar,
                faults=faults,
                engine_health=health,
            )
        )
    return adjusted


@st.cache_data(show_spinner=False)
def _journey(seed: int, scenario: str, payload_class: str | None, steps: int, start_condition: str):
    return simulate_journey(
        seed=seed,
        scenario=scenario,
        payload_class=payload_class,
        num_steps=steps,
        start_condition=start_condition,
        timestep_s=1.0,
    )


def _frame_to_dict(frame: TelemetryFrame) -> dict[str, Any]:
    """Use the model's complete JSON-safe representation and classifier output."""

    data = frame.as_dict()
    # Keep this explicit call here so the cabin UI shares the canonical
    # operational_ranges classifier even if the model's schema changes later.
    data["operational_statuses"]["engine_block_temperature"] = classify(
        "engine_block_temperature", frame.engine.coolant_outlet_temp_c
    )
    return data


def _dashboard_html(frames: list[TelemetryFrame], scenario_label: str, active_faults: list[str]) -> str:
    payload = {
        "frames": [_frame_to_dict(frame) for frame in frames],
        "scenario": scenario_label,
        "activeFaults": active_faults,
    }
    model = _safe_json(payload)
    twin_js = DIGITAL_TWIN_JS
    app_js = r"""
      (() => {
        const model = window.CABIN_MODEL || {frames: []};
        const frames = model.frames || [];
        if (!frames.length) return;
        let index = 0;
        let playing = true;
        let timer = null;
        const $ = (id) => document.getElementById(id);
        const set = (id, value) => { const el = $(id); if (el) el.textContent = value; };
        const n = (value, digits = 1) => Number(value || 0).toFixed(digits);
        const pct = (value, digits = 0) => `${n(value, digits)}%`;
        const title = (value) => String(value || '—').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
        const statusClass = (status) => status === 'green' ? 'good' : status === 'red' ? 'bad' : 'warn';
        const mode = (value) => ({parked:'Parked', idle:'Idle', hauling:'Hauling', upslope:'Upslope', downslope:'Downslope', braking:'Braking', dumping:'Dumping'}[value] || title(value));
        const current = () => frames[index];

        function renderFaults(frame) {
          const host = $('fault-list');
          const active = Object.entries(frame.faults || {}).filter(([, value]) => value);
          host.innerHTML = active.length ? active.map(([key]) => `<div class="fault-row"><span class="signal bad"></span><span>${title(key)}</span><b>ACTIVE</b></div>`).join('') : '<div class="clear-state"><span class="signal good"></span><span>All monitored faults clear</span></div>';
          set('fault-count', `${active.length} active`);
        }

        function renderTyres(frame) {
          const s = frame.suspension || {};
          const values = [s.front_left_pressure_bar, s.front_right_pressure_bar, s.rear_left_pressure_bar, s.rear_right_pressure_bar];
          ['FL','FR','RL','RR'].forEach((label, i) => {
            const value = Number(values[i] || 0);
            set(`tyre-${label}`, `${n(value, 0)} bar`);
            const el = $(`tyre-${label}-state`);
            if (el) { el.className = `tyre-state ${value < 80 ? 'bad' : value < 110 ? 'warn' : 'good'}`; el.textContent = value < 80 ? 'CHECK' : value < 110 ? 'WATCH' : 'OK'; }
          });
        }

        function renderTrend() {
          const canvas = $('trend');
          const rect = canvas.getBoundingClientRect();
          const ratio = Math.min(window.devicePixelRatio || 1, 2);
          canvas.width = Math.max(1, Math.round(rect.width * ratio)); canvas.height = Math.max(1, Math.round(rect.height * ratio));
          const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
          const w = rect.width || 500, h = rect.height || 140;
          ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#080e18'; ctx.fillRect(0, 0, w, h);
          ctx.strokeStyle = 'rgba(159,177,198,.13)'; ctx.lineWidth = 1;
          for (let y = 20; y < h - 15; y += (h - 35) / 3) { ctx.beginPath(); ctx.moveTo(12, y); ctx.lineTo(w - 12, y); ctx.stroke(); }
          const series = [{color:'#43d6a2', fn:f => Number((f.engine_health || {}).score || 0)}, {color:'#f0a247', fn:f => Number((f.engine || {}).coolant_temp_c || 0), min:60, max:140}, {color:'#78a9ff', fn:f => Number(f.stress_level_pct || 0)}];
          const x = (i) => 14 + (w - 28) * (frames.length < 2 ? 0 : i / (frames.length - 1));
          series.forEach((line) => { ctx.beginPath(); ctx.strokeStyle = line.color; ctx.lineWidth = 2; frames.forEach((f, i) => { const raw = line.fn(f); const norm = Math.max(0, Math.min(1, (raw - (line.min || 0)) / ((line.max || 100) - (line.min || 0)))); const y = h - 18 - norm * (h - 35); i ? ctx.lineTo(x(i), y) : ctx.moveTo(x(i), y); }); ctx.stroke(); });
          ctx.strokeStyle = '#f2bd55'; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(x(index), 8); ctx.lineTo(x(index), h - 8); ctx.stroke(); ctx.setLineDash([]);
        }

        function render() {
          const f = current(), e = f.engine || {}, h = f.engine_health || {}, p = f.payload || {}, g = f.gnss || {}, op = f.operational_statuses || {};
          const traffic = h.traffic_light || 'amber';
          set('health-score', n(h.score, 0)); set('health-class', title(h.classification)); set('health-bar-label', `${n(h.score, 0)} / 100`);
          set('engine-health', title(h.classification)); set('engine-health-sub', `${traffic.toUpperCase()} · ${title(f.mode)}`);
          set('payload', `${n(f.payload_utilization_pct, 0)}%`); set('payload-sub', `${n(f.payload_mass_t, 1)} t · ${title(f.payload_class)}`);
          set('rul', `${n(h.rul_hours, 0)} h`); set('rul-sub', Array.isArray(h.rul_range_hours) ? `${n(h.rul_range_hours[0], 0)}–${n(h.rul_range_hours[1], 0)} h range` : 'planning estimate');
          set('mode', mode(f.mode)); set('mode-small', mode(f.mode)); set('step', `FRAME ${index + 1} / ${frames.length}`); set('elapsed', `T + ${n(f.elapsed_s, 0)} s`);
          set('speed', `${n((g.speed_mps || 0) * 3.6, 1)} km/h`); set('rpm', `${n(e.rpm, 0)} rpm`); set('temp', `${n(e.coolant_temp_c, 1)}°C`); set('oil', `${n(e.oil_pressure_kpa, 0)} kPa`);
          set('action', h.recommended_action || 'Continue monitoring current telemetry.'); set('factors', (h.contributing_factors || []).join(' · ') || 'No active penalty factors'); set('timestamp', f.timestamp ? new Date(f.timestamp).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'}) : '—');
          set('operational-summary', `ENGINE ${String(op.engine_block_temperature || 'amber').toUpperCase()} · TRANS ${String(op.gearbox_oil_temperature || 'amber').toUpperCase()} · HYDRAULIC ${String(op.hydraulic_oil_condition || 'amber').toUpperCase()}`);
          const progress = Math.max(0, Math.min(100, Number(h.score || 0))); $('health-bar').style.width = `${progress}%`; $('timeline').value = String(index); set('timeline-label', `${index + 1} / ${frames.length}`);
          ['health-score','engine-health','health-class'].forEach((id) => { const el = $(id); if (el) el.className = `state-${traffic}`; });
          $('status-chip').className = `chip state-${traffic}`; $('status-chip').textContent = traffic === 'green' ? 'SYSTEM NOMINAL' : traffic === 'red' ? 'STOP / INSPECT' : 'ATTENTION REQUIRED';
          renderFaults(f); renderTyres(f); renderTrend();
          if (window.CABIN_TWIN) window.CABIN_TWIN.update({payloadPercent:f.payload_utilization_pct, maxPayload:f.rated_capacity_t, temperature:e.coolant_temp_c, vibration:(f.imu || {}).vibration_rms_g, hoistAngle:f.hoist_angle_deg, speed:g.speed_mps, rpm:e.rpm});
        }
        function setPlaying(next) { playing = next; if (timer) clearInterval(timer); timer = playing ? setInterval(() => { index = index >= frames.length - 1 ? 0 : index + 1; render(); }, 1100) : null; set('play-label', playing ? 'Pause replay' : 'Play replay'); }
        if (window.DigitalTwinTruck && $('twin')) window.CABIN_TWIN = window.DigitalTwinTruck.init($('twin'), {telemetry:{payloadPercent:frames[0].payload_utilization_pct, maxPayload:frames[0].rated_capacity_t, temperature:frames[0].engine.coolant_temp_c, vibration:frames[0].imu.vibration_rms_g, hoistAngle:frames[0].hoist_angle_deg}, config:{showHud:false, height:360, maxPayload:frames[0].rated_capacity_t, maxBedAngle:38}});
        $('prev').onclick = () => { index = index <= 0 ? frames.length - 1 : index - 1; render(); };
        $('next').onclick = () => { index = index >= frames.length - 1 ? 0 : index + 1; render(); };
        $('play').onclick = () => setPlaying(!playing);
        $('timeline').max = String(frames.length - 1); $('timeline').oninput = (event) => { index = Number(event.target.value); render(); };
        window.addEventListener('resize', renderTrend); render(); setPlaying(true);
      })();
    """
    css = r"""
      :root { color-scheme: dark; --navy:#070b13; --panel:#0d1421; --panel2:#111b2a; --line:#263449; --text:#eff5fc; --muted:#8998ad; --orange:#f0a247; --green:#43d6a2; --blue:#78a9ff; --red:#ff6b6b; --amber:#f4bd5d; }
      * { box-sizing:border-box; } body { margin:0; background:linear-gradient(145deg,#060a12,#0b1220 60%,#080e18); color:var(--text); font:14px Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; } .cab { max-width:1440px; margin:auto; padding:18px 22px 32px; }
      .topbar { display:flex; align-items:center; gap:26px; border-bottom:1px solid var(--line); padding:4px 2px 16px; } .brand { display:flex; align-items:center; gap:10px; min-width:210px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; } .brand-mark { display:grid; place-items:center; width:32px;height:32px;border:1px solid var(--orange);border-radius:9px;color:var(--orange);font-size:17px; } nav { display:flex; gap:24px; flex:1; } nav a { color:var(--muted); text-decoration:none; font-weight:650; } nav a:first-child, nav a:hover { color:var(--text); } .live { color:var(--green); font-size:11px; letter-spacing:.12em; font-weight:800; }
      .hero { display:flex; justify-content:space-between; gap:20px; padding:26px 0 20px; align-items:flex-end; } .eyebrow,.label { color:var(--orange); font-size:10px; letter-spacing:.16em; text-transform:uppercase; font-weight:800; } h1 { margin:7px 0 4px; font-size:clamp(23px,3vw,36px); letter-spacing:-.04em; } .sub { color:var(--muted); margin:0; max-width:720px; line-height:1.5; } .hero-meta { text-align:right; color:var(--muted); font:11px ui-monospace,monospace; }
      .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-bottom:14px; } .metric,.panel { background:linear-gradient(160deg,rgba(17,27,42,.94),rgba(10,16,27,.96)); border:1px solid var(--line); border-radius:13px; box-shadow:0 12px 28px rgba(0,0,0,.14); } .metric { padding:17px; min-height:120px; } .metric .value { display:block; margin-top:13px; font-size:29px; letter-spacing:-.06em; font-weight:800; } .metric .hint { color:var(--muted); font-size:11px; margin-top:6px; } .state-green,.good { color:var(--green)!important; } .state-amber,.warn { color:var(--amber)!important; } .state-red,.bad { color:var(--red)!important; }
      .main-grid { display:grid; grid-template-columns:minmax(0,1.48fr) minmax(300px,.72fr); gap:14px; align-items:stretch; } .panel { padding:17px; } .panel-head { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:12px; } h2 { margin:4px 0 0; font-size:17px; letter-spacing:-.02em; } .chip { border:1px solid var(--line); border-radius:99px; padding:7px 10px; font-size:10px; font-weight:800; letter-spacing:.1em; white-space:nowrap; } .chip.state-green { border-color:rgba(67,214,162,.4); background:rgba(67,214,162,.08); } .chip.state-amber { border-color:rgba(244,189,93,.4); background:rgba(244,189,93,.08); } .chip.state-red { border-color:rgba(255,107,107,.4); background:rgba(255,107,107,.08); }
      #twin { height:360px; border:1px solid rgba(120,169,255,.16); border-radius:10px; overflow:hidden; background:#080e18; } .scene-footer { display:flex; justify-content:space-between; gap:15px; padding-top:12px; color:var(--muted); font:11px ui-monospace,monospace; } .scene-footer strong { color:var(--text); font-size:16px; } .controls { display:flex; align-items:center; gap:8px; margin-top:14px; } button { border:1px solid var(--line); background:#152238; color:var(--text); padding:9px 12px; border-radius:8px; cursor:pointer; font-weight:700; } button:hover { border-color:var(--orange); } input[type=range] { flex:1; accent-color:var(--orange); min-width:80px; }
      .health-number { font-size:53px; letter-spacing:-.09em; font-weight:850; } .health-number small { color:var(--muted); font-size:14px; letter-spacing:0; } .bar { height:7px; background:#1a2536; border-radius:99px; overflow:hidden; margin:11px 0 15px; } .bar > span { display:block; height:100%; width:0; background:var(--green); border-radius:inherit; transition:width .3s,background .3s; } .health-grid { display:grid; grid-template-columns:1fr 1fr; gap:9px; } .health-mini { border:1px solid var(--line); border-radius:9px; padding:10px; } .health-mini span { display:block; color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.1em; } .health-mini b { display:block; margin-top:6px; font-size:14px; } .action { border-left:3px solid var(--orange); background:rgba(240,162,71,.08); padding:11px 12px; margin-top:14px; line-height:1.5; font-size:12px; } .factors { margin-top:10px; color:var(--muted); font-size:11px; line-height:1.5; }
      .lower-grid { display:grid; grid-template-columns:1.1fr .9fr; gap:14px; margin-top:14px; } #trend { display:block; width:100%; height:140px; border:1px solid var(--line); border-radius:8px; } .legend { display:flex; flex-wrap:wrap; gap:13px; margin-top:9px; color:var(--muted); font-size:10px; } .legend i { display:inline-block; width:18px; height:3px; border-radius:99px; background:var(--green); margin-right:4px; vertical-align:middle; } .legend i.orange { background:var(--orange); } .legend i.blue { background:var(--blue); }
      .fault-row,.clear-state { display:flex; align-items:center; gap:9px; padding:10px 0; border-bottom:1px solid rgba(38,52,73,.7); font-size:12px; } .fault-row b { margin-left:auto; color:var(--red); font-size:10px; letter-spacing:.1em; } .signal { width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 0 4px color-mix(in srgb,currentColor 13%,transparent); } .tyres { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:9px; } .tyre { border:1px solid var(--line); border-radius:9px; padding:10px; } .tyre-top { display:flex; justify-content:space-between; color:var(--muted); font-size:11px; } .tyre b { display:block; margin-top:8px; font-size:16px; } .tyre-state { font-size:9px; letter-spacing:.1em; font-weight:800; }
      .strip { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:14px 0 0; padding:11px 13px; border:1px solid var(--line); border-radius:10px; color:var(--muted); font-size:11px; } .strip strong { color:var(--text); text-transform:uppercase; letter-spacing:.1em; font-size:10px; } .foot { color:var(--muted); font-size:10px; line-height:1.5; margin-top:15px; }
      @media (max-width:900px) { .metrics { grid-template-columns:repeat(2,1fr); } .main-grid,.lower-grid { grid-template-columns:1fr; } .hero { align-items:flex-start; flex-direction:column; } .hero-meta { text-align:left; } }
      @media (max-width:560px) { .cab { padding:12px 11px 24px; } .topbar { gap:12px; align-items:flex-start; flex-wrap:wrap; } .brand { min-width:auto; } nav { order:3; width:100%; justify-content:space-between; gap:8px; } nav a { font-size:12px; } .metrics { gap:8px; } .metric { padding:12px; min-height:105px; } .metric .value { font-size:23px; margin-top:10px; } .panel { padding:12px; } #twin { height:290px; } .controls { flex-wrap:wrap; } .controls input { order:5; flex-basis:100%; } }
    """
    active_text = ", ".join(active_faults) if active_faults else "No manual faults selected"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<style>{css}</style></head><body><main class='cab'>"
        "<header class='topbar'><div class='brand'><span class='brand-mark'>◈</span><span>HAUL / CABIN</span></div><nav><a href='#home'>Home</a><a href='#vehicle'>Vehicle</a><a href='#trip'>Trip</a><a href='#settings'>Settings</a></nav><span class='live'>● LIVE SIM</span></header>"
        f"<section class='hero' id='home'><div><div class='eyebrow'>Caterpillar-class haul unit · cabin console</div><h1>Operator situational dashboard</h1><p class='sub'>A focused digital twin view for the current telemetry frame, machine health, payload balance and route replay.</p></div><div class='hero-meta'><div>{scenario_label}</div><div id='timestamp'>—</div></div></section>"
        "<section class='metrics' id='vehicle'><article class='metric'><span class='label'>Health score</span><strong class='value'><span id='health-score'>—</span><small> / 100</small></strong><div class='hint' id='health-class'>—</div></article><article class='metric'><span class='label'>Engine health</span><strong class='value' id='engine-health'>—</strong><div class='hint' id='engine-health-sub'>—</div></article><article class='metric'><span class='label'>Payload</span><strong class='value' id='payload'>—</strong><div class='hint' id='payload-sub'>—</div></article><article class='metric'><span class='label'>RUL estimate</span><strong class='value' id='rul'>—</strong><div class='hint' id='rul-sub'>—</div></article></section>"
        "<div class='main-grid'><section class='panel'><div class='panel-head'><div><div class='label'>Vehicle digital twin</div><h2>Live body, suspension & thermal response</h2></div><span class='chip' id='status-chip'>LOADING</span></div><div id='twin'></div><div class='scene-footer'><div><strong id='mode'>—</strong><br><span id='step'>—</span> · <span id='elapsed'>—</span></div><div style='text-align:right'><strong id='speed'>—</strong><br><span id='rpm'>—</span></div></div><div class='controls'><button id='prev' type='button'>‹</button><button id='play' type='button'><span id='play-label'>Pause replay</span></button><button id='next' type='button'>›</button><input id='timeline' type='range' min='0' value='0' step='1'><span id='timeline-label'>—</span></div></section>"
        "<aside class='panel'><div class='panel-head'><div><div class='label'>Release gate</div><h2>Machine health</h2></div><span class='label' id='mode-small'>—</span></div><div class='health-number'><span id='health-bar-label'>—</span></div><div class='bar'><span id='health-bar'></span></div><div class='health-grid'><div class='health-mini'><span>Coolant</span><b id='temp'>—</b></div><div class='health-mini'><span>Oil pressure</span><b id='oil'>—</b></div></div><div class='label' style='margin-top:15px'>Recommended action</div><div class='action' id='action'>—</div><div class='factors' id='factors'>—</div></aside></div>"
        "<section class='strip' id='trip'><strong>Operational bands</strong><span id='operational-summary'>ENGINE — · TRANS — · HYDRAULIC —</span><span>Green = nominal · Amber = caution · Red = alert</span></section>"
        "<div class='lower-grid'><section class='panel'><div class='panel-head'><div><div class='label'>Trip replay</div><h2>Health / thermal / stress trend</h2></div><span class='label' id='fault-count'>—</span></div><canvas id='trend'></canvas><div class='legend'><span><i></i>Health</span><span><i class='orange'></i>Coolant</span><span><i class='blue'></i>Stress</span></div></section><section class='panel' id='settings'><div class='panel-head'><div><div class='label'>Faults & alerts</div><h2>Operator attention queue</h2></div></div><div id='fault-list'></div><div class='label' style='margin-top:18px'>Tyre status · suspension pressure</div><div class='tyres'><div class='tyre'><div class='tyre-top'><span>FRONT LEFT</span><span id='tyre-FL-state' class='tyre-state'>—</span></div><b id='tyre-FL'>—</b></div><div class='tyre'><div class='tyre-top'><span>FRONT RIGHT</span><span id='tyre-FR-state' class='tyre-state'>—</span></div><b id='tyre-FR'>—</b></div><div class='tyre'><div class='tyre-top'><span>REAR LEFT</span><span id='tyre-RL-state' class='tyre-state'>—</span></div><b id='tyre-RL'>—</b></div><div class='tyre'><div class='tyre-top'><span>REAR RIGHT</span><span id='tyre-RR-state' class='tyre-state'>—</span></div><b id='tyre-RR'>—</b></div></div></section></div>"
        f"<p class='foot'>Current frame data is synthetic telemetry from <code>simulate_journey</code>. Manual toggles: {active_text}. Use replay controls to step through each generated sample.</p></main><script>window.CABIN_MODEL={model};</script><script>{twin_js}</script><script>{app_js}</script></body></html>"
    )


st.set_page_config(page_title="Haul / Cabin Dashboard", page_icon="🚜", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      .stApp { background:#070b13; } [data-testid="stHeader"] { background:transparent; }
      [data-testid="stSidebar"] { background:#0a101b; border-right:1px solid #263449; }
      .block-container { max-width:1500px; padding-top:1rem; padding-bottom:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## 🚜 Cabin controls")
st.sidebar.caption("Responsive dump-truck operational console")
st.sidebar.info("Use Engine Evidence Lab in page navigation for persistent history, scheduled faults and evidence-based diagnosis. Cabin health/RUL and fault labels below remain legacy synthetic outputs, not confirmed causes.")
scenario_label = st.sidebar.selectbox("Scenario", list(SCENARIOS))
scenario = SCENARIOS[scenario_label]
payload_label = st.sidebar.selectbox("Payload", list(PAYLOADS), help="Payload remains categorical and fixed for the journey.")
payload_class = PAYLOADS[payload_label]
start_label = st.sidebar.selectbox("Start condition", list(STARTS))
start_condition = STARTS[start_label]
st.sidebar.divider()
st.sidebar.markdown("### Fault injection")
st.sidebar.caption("Each selected fault is applied to every generated frame.")
overheat = st.sidebar.checkbox("🔥 Coolant failure / overheat", key="cabin_fault_overheat")
bearing_failure = st.sidebar.checkbox("⚙️ Bearing failure / vibration", key="cabin_fault_bearing")
low_oil = st.sidebar.checkbox("🛢️ Low engine oil pressure", key="cabin_fault_oil")
brake_wear = st.sidebar.checkbox("🛑 Brake wear / long stroke", key="cabin_fault_brakes")
perception_degraded = st.sidebar.checkbox("📡 Perception sensor degradation", key="cabin_fault_perception")
with st.sidebar.expander("Replay setup", expanded=False):
    seed = st.number_input("Deterministic seed", min_value=0, max_value=999_999, value=8502, step=1)
    steps = st.slider("Journey frames", min_value=24, max_value=120, value=72, step=12)
    st.caption("The browser replay starts automatically; use ‹ / › or the timeline to step.")

journey = _journey(int(seed), scenario, payload_class, int(steps), start_condition)
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
if active_faults:
    st.sidebar.warning(f"Active: {', '.join(active_faults)}")

if not frames:
    st.error("No telemetry frames were generated.")
else:
    components.html(_dashboard_html(frames, scenario_label, active_faults), height=1_560, scrolling=True)
