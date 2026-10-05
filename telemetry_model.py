"""Deterministic telemetry model for a dump-truck digital twin.

The module deliberately uses only the Python standard library.  It is intended
to be a small, convenient data source for a Streamlit dashboard, but the
individual dataclasses are also useful in tests or in a notebook.

The simulator is not intended to be a vehicle dynamics solver.  It produces
coherent, bounded telemetry with deterministic trends for three operating
scenarios:

``ideal``
    A healthy truck carrying a nominal payload.
``stress``
    A heavy haul whose thermal, structural, and sensor stress increases during
    the journey.
``overhaul``
    A truck immediately after service; the small initial service transient
    settles as the journey progresses.

Payload *class* is categorical and is selected once per journey.  It never
changes between samples.  A separate ``payload_utilization_pct`` value makes
the capacity state easy to plot and filter in a UI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
import math
import random
from typing import Any, Iterator, Mapping, Sequence

from operational_ranges import classify as classify_operational_range


__all__ = [
    "BrakeTelemetry",
    "DSSAlertLevel",
    "DSSTelemetry",
    "EngineHealth",
    "EngineHealthClass",
    "HealthClassification",
    "EngineTelemetry",
    "Engine",
    "FaultFlags",
    "FrameTelemetry",
    "Frame",
    "GNSS",
    "GNSSTelemetry",
    "IMU",
    "IMUTelemetry",
    "Journey",
    "LiDARTelemetry",
    "LiDAR",
    "OperatingMode",
    "PayloadClass",
    "PayloadProfile",
    "PayloadTelemetry",
    "RadarTelemetry",
    "Radar",
    "Scenario",
    "DriveCycle",
    "StartCondition",
    "SimulationConfig",
    "SuspensionTelemetry",
    "Suspension",
    "TelemetryFrame",
    "TransmissionTelemetry",
    "Transmission",
    "Brake",
    "DSS",
    "calculate_engine_health",
    "classify_engine_health",
    "classify_health",
    "engine_health_score",
    "estimate_rul_hours",
    "generate_telemetry",
    "health_classification",
    "payload_profile_for",
    "recommended_action_for_health",
    "run_simulation",
    "simulate_dump_truck",
    "simulate_journey",
    "simulate_telemetry",
]


class _ValueEnum(str, Enum):
    """A lower-case string enum that serializes naturally in a UI."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            text = value.strip().lower()
            for name, member in cls.__members__.items():
                if name.lower() == text:
                    return member
        return None


class PayloadClass(_ValueEnum):
    """Discrete payload categories.

    The aliases are intentionally provided for dashboards that use slightly
    different terminology (for example, ``NOMINAL`` and ``STANDARD`` both
    mean the medium payload class).  Enum iteration still exposes only the
    five actual categories.
    """

    EMPTY = "empty"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    OVERLOAD = "overload"

    # Common vocabulary aliases; these do not create extra categories.
    LOW = "light"
    STANDARD = "medium"
    NOMINAL = "medium"
    HIGH = "heavy"


class OperatingMode(_ValueEnum):
    """Truck operating mode for a telemetry sample."""

    PARKED = "parked"
    IDLE = "idle"
    HAULING = "hauling"
    UPSLOPE = "upslope"
    DOWNSLOPE = "downslope"
    BRAKING = "braking"
    DUMPING = "dumping"

    # Readable aliases for callers that use road terminology.
    PARK = "parked"
    UPHILL = "upslope"
    DOWNHILL = "downslope"


class Scenario(_ValueEnum):
    """Simulation scenario."""

    IDEAL = "ideal"
    STRESS = "stress"
    OVERHAUL = "overhaul"

    NORMAL = "ideal"
    MAINTENANCE = "overhaul"


class DriveCycle(_ValueEnum):
    """Supported route profiles."""

    SYNTHETIC_ROUTE = "synthetic_route"
    C_WTVC = "c-wtvc"

    RESEARCH = "c-wtvc"


class StartCondition(_ValueEnum):
    """Thermal initial condition for a journey."""

    WARM = "warm"
    COLD = "cold"


class EngineHealthClass(_ValueEnum):
    """Human-readable engine health bands."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

    # Useful aliases for compact dashboards.
    HEALTHY = "excellent"
    WARNING = "fair"
    DEGRADED = "poor"


class DSSAlertLevel(_ValueEnum):
    """Driver-support alert severity."""

    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# A compatibility alias that reads naturally in application code.
HealthClassification = EngineHealthClass


def _coerce_enum(value: Any, enum_type: type[_ValueEnum], default: _ValueEnum) -> _ValueEnum:
    """Convert strings and enum values while keeping public APIs forgiving."""

    if value is None:
        return default
    if isinstance(value, enum_type):
        return value
    text = str(value).strip().lower()
    try:
        return enum_type(text)
    except ValueError:
        # Accept member names as well as serialized values.  This makes both
        # ``Scenario.STRESS`` and ``"STRESS"`` convenient inputs, including
        # aliases such as ``"healthy"`` for the excellent health band.
        for name, member in enum_type.__members__.items():
            if name.lower() == text:
                return member
        choices = ", ".join(member.value for member in enum_type)
        raise ValueError(f"unknown {enum_type.__name__} {value!r}; choose one of {choices}")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bounded_round(value: float, digits: int = 2) -> float:
    """Round a generated sensor value without allowing ``-0.0`` in a UI."""

    result = round(float(value), digits)
    return 0.0 if abs(result) < 10 ** (-(digits + 1)) else result


@dataclass(frozen=True)
class PayloadProfile:
    """Static payload metadata selected for one simulated journey."""

    payload_class: PayloadClass
    rated_capacity_t: float
    utilization_ratio: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "payload_class",
            _coerce_enum(self.payload_class, PayloadClass, PayloadClass.MEDIUM),
        )
        capacity = float(self.rated_capacity_t)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("rated_capacity_t must be a positive finite number")
        object.__setattr__(self, "rated_capacity_t", capacity)
        object.__setattr__(
            self,
            "utilization_ratio",
            _clamp(float(self.utilization_ratio), 0.0, 1.25),
        )

    @property
    def utilization_pct(self) -> float:
        return _bounded_round(self.utilization_ratio * 100.0, 2)

    @property
    def capacity_utilization_pct(self) -> float:
        return self.utilization_pct

    @property
    def capacity_pct(self) -> float:
        return self.utilization_pct

    @property
    def payload_percent(self) -> float:
        return self.utilization_pct

    @property
    def class_name(self) -> str:
        return self.payload_class.value

    @property
    def category(self) -> PayloadClass:
        return self.payload_class

    @property
    def payload_mass_t(self) -> float:
        return _bounded_round(self.rated_capacity_t * self.utilization_ratio, 3)

    @property
    def mass_t(self) -> float:
        """Short alias used by compact dashboard cards."""

        return self.payload_mass_t


_PAYLOAD_UTILIZATION: Mapping[PayloadClass, float] = {
    PayloadClass.EMPTY: 0.0,
    PayloadClass.LIGHT: 0.48,
    PayloadClass.MEDIUM: 0.72,
    PayloadClass.HEAVY: 0.94,
    PayloadClass.OVERLOAD: 1.08,
}


def payload_profile_for(
    payload_class: PayloadClass | str,
    rated_capacity_t: float = 40.0,
) -> PayloadProfile:
    """Return the fixed profile associated with a categorical payload class."""

    category = _coerce_enum(payload_class, PayloadClass, PayloadClass.MEDIUM)
    return PayloadProfile(
        payload_class=category,
        rated_capacity_t=float(rated_capacity_t),
        utilization_ratio=_PAYLOAD_UTILIZATION[category],
    )


@dataclass(frozen=True)
class PayloadTelemetry:
    """Payload values copied into each frame for convenient charting."""

    payload_class: PayloadClass
    mass_t: float
    rated_capacity_t: float
    utilization_pct: float
    over_capacity: bool

    @property
    def utilization_ratio(self) -> float:
        return self.utilization_pct / 100.0

    @property
    def capacity_utilization_pct(self) -> float:
        return self.utilization_pct

    @property
    def capacity_pct(self) -> float:
        return self.utilization_pct

    @property
    def payload_percent(self) -> float:
        return self.utilization_pct

    @property
    def class_name(self) -> str:
        return self.payload_class.value

    @property
    def category(self) -> PayloadClass:
        return self.payload_class

    @classmethod
    def from_profile(cls, profile: PayloadProfile) -> "PayloadTelemetry":
        return cls(
            payload_class=profile.payload_class,
            mass_t=profile.payload_mass_t,
            rated_capacity_t=profile.rated_capacity_t,
            utilization_pct=profile.utilization_pct,
            over_capacity=profile.utilization_ratio > 1.0,
        )


@dataclass(frozen=True)
class FaultFlags:
    """Boolean fault and degradation flags emitted with every frame."""

    engine_overheat: bool = False
    low_oil_pressure: bool = False
    transmission_slip: bool = False
    brake_wear: bool = False
    suspension_overload: bool = False
    frame_strain: bool = False
    gnss_degraded: bool = False
    lidar_obstructed: bool = False
    radar_fault: bool = False
    imu_fault: bool = False
    payload_overload: bool = False
    dss_alert: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, bool(getattr(self, name)))

    def as_dict(self) -> dict[str, bool]:
        return {name: bool(getattr(self, name)) for name in self.__dataclass_fields__}

    to_dict = as_dict

    @property
    def any_active(self) -> bool:
        return any(self.as_dict().values())

    @property
    def count(self) -> int:
        return sum(self.as_dict().values())

    @property
    def fault_names(self) -> tuple[str, ...]:
        return tuple(name for name, active in self.as_dict().items() if active)

    def active_faults(self) -> tuple[str, ...]:
        """Method form retained for callers that prefer an explicit action."""

        return self.fault_names

    def __getitem__(self, name: str) -> bool:
        """Allow ``faults["engine_overheat"]`` in table-oriented UIs."""

        if name not in self.__dataclass_fields__:
            raise KeyError(name)
        return bool(getattr(self, name))

    def get(self, name: str, default: bool | None = None) -> bool | None:
        return bool(getattr(self, name)) if name in self.__dataclass_fields__ else default

    def items(self):
        return self.as_dict().items()

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    # Short aliases useful in filters and backwards-compatible with common
    # telemetry naming conventions.
    @property
    def overheating(self) -> bool:
        return self.engine_overheat

    @property
    def oil_pressure_low(self) -> bool:
        return self.low_oil_pressure

    @property
    def overloaded(self) -> bool:
        return self.payload_overload or self.suspension_overload

    @property
    def engine_fault(self) -> bool:
        return self.engine_overheat or self.low_oil_pressure

    @property
    def transmission_fault(self) -> bool:
        return self.transmission_slip

    @property
    def brake_fault(self) -> bool:
        return self.brake_wear


@dataclass(frozen=True)
class IMUTelemetry:
    """Six-axis inertial measurement unit plus derived attitude."""

    acceleration_x_mps2: float = 0.0
    acceleration_y_mps2: float = 0.0
    acceleration_z_mps2: float = 9.81
    angular_rate_x_dps: float = 0.0
    angular_rate_y_dps: float = 0.0
    angular_rate_z_dps: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    vibration_rms_g: float = 0.0

    # Concise aliases for plotting libraries.
    @property
    def accel_x(self) -> float:
        return self.acceleration_x_mps2

    @property
    def accel_y(self) -> float:
        return self.acceleration_y_mps2

    @property
    def accel_z(self) -> float:
        return self.acceleration_z_mps2

    @property
    def gyro_x(self) -> float:
        return self.angular_rate_x_dps

    @property
    def gyro_y(self) -> float:
        return self.angular_rate_y_dps

    @property
    def gyro_z(self) -> float:
        return self.angular_rate_z_dps


@dataclass(frozen=True)
class EngineTelemetry:
    """Engine operating measurements."""

    rpm: float = 0.0
    load_pct: float = 0.0
    torque_nm: float = 0.0
    power_kw: float = 0.0
    coolant_temp_c: float = 70.0
    oil_temp_c: float = 75.0
    oil_pressure_kpa: float = 400.0
    fuel_rate_lph: float = 0.0
    boost_kpa: float = 100.0
    exhaust_temp_c: float = 200.0
    vibration_rms_g: float = 0.0
    throttle_position_pct: float = 0.0
    oil_level_pct: float = 100.0
    fuel_level_pct: float = 100.0
    exhaust_temperatures_c: tuple[float, ...] = ()
    air_filter_restriction_kpa: float = 0.0
    coolant_inlet_temp_c: float = 65.0
    coolant_flow_kg_s: float = 0.0
    thermostat_open_pct: float = 100.0
    coolant_energy_flow_kw: float = 0.0
    fuel_energy_input_kw: float = 0.0
    exhaust_energy_flow_kw: float = 0.0
    effective_power_kw: float = 0.0
    mechanical_loss_kw: float = 0.0
    remainder_loss_kw: float = 0.0
    coolant_energy_pct: float = 0.0
    exhaust_energy_pct: float = 0.0
    transient_index_pct: float = 0.0
    block_vibration_mm_s: float = 2.0
    coolant_pump_temp_c: float = 80.0

    @property
    def load_ratio(self) -> float:
        return self.load_pct / 100.0

    @property
    def temperature_c(self) -> float:
        return self.coolant_temp_c

    @property
    def temperature(self) -> float:
        return self.coolant_temp_c

    @property
    def throttle_pct(self) -> float:
        return self.throttle_position_pct

    @property
    def oil_level(self) -> float:
        return self.oil_level_pct

    @property
    def fuel_level(self) -> float:
        return self.fuel_level_pct

    @property
    def cylinder_exhaust_temperatures_c(self) -> tuple[float, ...]:
        return self.exhaust_temperatures_c

    @property
    def air_filter_pressure_kpa(self) -> float:
        return self.air_filter_restriction_kpa

    @property
    def coolant_outlet_temp_c(self) -> float:
        """Published outlet temperature; legacy coolant_temp remains an alias."""

        return self.coolant_temp_c

    @property
    def coolant_delta_t_c(self) -> float:
        return self.coolant_outlet_temp_c - self.coolant_inlet_temp_c


@dataclass(frozen=True)
class TransmissionTelemetry:
    """Transmission and driveline measurements."""

    gear: int = 0
    input_rpm: float = 0.0
    output_rpm: float = 0.0
    oil_temp_c: float = 70.0
    slip_pct: float = 0.0
    efficiency_pct: float = 98.0
    output_torque_nm: float = 0.0
    status: str = "neutral"
    powertrain_oil_pressure_bar: float = 0.0
    axle_temp_c: float = 70.0
    oil_pressure_kpa: float = 350.0
    vibration_mm_s: float = 1.4

    @property
    def slip(self) -> float:
        return self.slip_pct

    @property
    def tos_rpm(self) -> float:
        return self.output_rpm

    @property
    def selected_gear(self) -> int:
        return self.gear

    @property
    def oil_pressure_bar(self) -> float:
        return self.powertrain_oil_pressure_bar

    @property
    def oil_pressure(self) -> float:
        return self.oil_pressure_kpa


@dataclass(frozen=True)
class BrakeTelemetry:
    """Brake system measurements."""

    pedal_pct: float = 0.0
    line_pressure_bar: float = 0.0
    disc_temp_c: float = 35.0
    pad_wear_pct: float = 0.0
    abs_active: bool = False
    parking_brake: bool = False
    oil_temp_c: float = 35.0
    cooling_temp_c: float = 35.0
    accumulator_pressure_bar: float = 0.0
    stroke_pct: float = 0.0

    @property
    def pressure_bar(self) -> float:
        return self.line_pressure_bar

    @property
    def temperature_c(self) -> float:
        return self.disc_temp_c

    @property
    def wear_pct(self) -> float:
        return self.pad_wear_pct

    @property
    def wear_indicator_pct(self) -> float:
        return self.pad_wear_pct

    @property
    def accumulator_pressure(self) -> float:
        return self.accumulator_pressure_bar


@dataclass(frozen=True)
class SuspensionTelemetry:
    """Suspension and load-balance measurements."""

    ride_height_mm: float = 900.0
    hydraulic_pressure_bar: float = 100.0
    compression_pct: float = 0.0
    left_load_pct: float = 50.0
    right_load_pct: float = 50.0
    shock_temp_c: float = 35.0
    front_left_pressure_bar: float = 100.0
    front_right_pressure_bar: float = 100.0
    rear_left_pressure_bar: float = 100.0
    rear_right_pressure_bar: float = 100.0
    hydraulic_oil_condition_pct: float = 96.0
    actuator_vibration_mm_s: float = 1.6

    @property
    def pressure_bar(self) -> float:
        return self.hydraulic_pressure_bar

    @property
    def load_pct(self) -> float:
        return (self.left_load_pct + self.right_load_pct) / 2.0

    @property
    def cylinder_pressure_bar(self) -> float:
        return self.hydraulic_pressure_bar

    @property
    def cylinder_pressures_bar(self) -> tuple[float, float, float, float]:
        return (
            self.front_left_pressure_bar,
            self.front_right_pressure_bar,
            self.rear_left_pressure_bar,
            self.rear_right_pressure_bar,
        )


@dataclass(frozen=True)
class FrameTelemetry:
    """Structural frame measurements."""

    strain_microstrain: float = 0.0
    stress_mpa: float = 0.0
    deflection_mm: float = 0.0
    twist_deg: float = 0.0
    fatigue_index_pct: float = 0.0

    @property
    def strain(self) -> float:
        return self.strain_microstrain

    @property
    def stress(self) -> float:
        return self.stress_mpa


@dataclass(frozen=True)
class GNSSTelemetry:
    """Global navigation satellite system measurements."""

    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0
    speed_mps: float = 0.0
    heading_deg: float = 0.0
    accuracy_m: float = 1.0
    satellites: int = 12
    fix_type: str = "RTK"

    @property
    def lat(self) -> float:
        return self.latitude_deg

    @property
    def lon(self) -> float:
        return self.longitude_deg

    @property
    def speed(self) -> float:
        return self.speed_mps


# Some applications spell the acronym with a capital S; expose both forms.
GNSS = GNSSTelemetry


@dataclass(frozen=True)
class LiDARTelemetry:
    """LiDAR perception measurements."""

    obstacle_distance_m: float = 100.0
    obstacle_confidence_pct: float = 0.0
    points_per_scan: int = 0
    scan_rate_hz: float = 10.0
    blocked: bool = False
    obstacle_class: str = "none"

    @property
    def distance_m(self) -> float:
        return self.obstacle_distance_m

    @property
    def confidence_pct(self) -> float:
        return self.obstacle_confidence_pct


@dataclass(frozen=True)
class RadarTelemetry:
    """Radar perception measurements."""

    target_distance_m: float = 100.0
    relative_speed_mps: float = 0.0
    target_confidence_pct: float = 0.0
    object_count: int = 0
    sensor_ok: bool = True
    target_class: str = "none"

    @property
    def distance_m(self) -> float:
        return self.target_distance_m

    @property
    def confidence_pct(self) -> float:
        return self.target_confidence_pct


@dataclass(frozen=True)
class DSSTelemetry:
    """Driver-support-system alerts derived from vehicle and perception data."""

    collision_warning: bool = False
    rollover_warning: bool = False
    overspeed_warning: bool = False
    emergency_braking: bool = False
    fatigue_alert: bool = False
    alert_level: DSSAlertLevel = DSSAlertLevel.NONE
    driver_message: str = "No active alerts"
    camera_online: bool = True
    driver_presence_pct: float = 100.0
    seatbelt_fastened: bool = True
    distraction_score_pct: float = 0.0
    fatigue_score_pct: float = 0.0
    camera_fps: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "alert_level",
            _coerce_enum(self.alert_level, DSSAlertLevel, DSSAlertLevel.NONE),
        )

    @property
    def any_alert(self) -> bool:
        return any(
            (
                self.collision_warning,
                self.rollover_warning,
                self.overspeed_warning,
                self.emergency_braking,
                self.fatigue_alert,
            )
        )

    @property
    def alert(self) -> bool:
        return self.any_alert

    @property
    def message(self) -> str:
        return self.driver_message

    @property
    def camera_status(self) -> str:
        return "online" if self.camera_online else "offline"


@dataclass(frozen=True)
class EngineHealth:
    """Engine health score and maintenance interpretation."""

    score: float
    classification: EngineHealthClass
    rul_hours: float
    recommended_action: str
    contributing_factors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _bounded_round(_clamp(float(self.score), 0.0, 100.0), 2))
        object.__setattr__(
            self,
            "classification",
            _coerce_enum(self.classification, EngineHealthClass, EngineHealthClass.FAIR),
        )
        object.__setattr__(self, "rul_hours", _bounded_round(max(0.0, float(self.rul_hours)), 1))
        object.__setattr__(self, "contributing_factors", tuple(str(item) for item in self.contributing_factors))

    @property
    def health_score(self) -> float:
        return self.score

    @property
    def health_class(self) -> EngineHealthClass:
        return self.classification

    @property
    def classification_name(self) -> str:
        return self.classification.value

    @property
    def status(self) -> str:
        """Coarser label for dashboards that use healthy/warning/critical."""

        if self.classification in (EngineHealthClass.EXCELLENT, EngineHealthClass.GOOD):
            return "healthy"
        if self.classification is EngineHealthClass.CRITICAL:
            return "critical"
        return "warning"

    @property
    def traffic_light(self) -> str:
        """Green/amber/red release-gate label for operator dashboards."""

        if self.classification in (EngineHealthClass.EXCELLENT, EngineHealthClass.GOOD):
            return "green"
        if self.classification is EngineHealthClass.CRITICAL:
            return "red"
        return "amber"

    @property
    def remaining_useful_life_hours(self) -> float:
        return self.rul_hours

    @property
    def rul(self) -> float:
        return self.rul_hours

    @property
    def remaining_useful_life(self) -> float:
        return self.rul_hours

    @property
    def rul_range_hours(self) -> tuple[float, float]:
        """Planning range around the point estimate for dashboard display."""

        spread = 0.08 if self.score >= 75.0 else 0.18 if self.score >= 40.0 else 0.28
        low = _bounded_round(max(0.0, self.rul_hours * (1.0 - spread)), 1)
        high = _bounded_round(self.rul_hours * (1.0 + spread), 1)
        return low, high

    @property
    def action(self) -> str:
        return self.recommended_action

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "classification": self.classification.value,
            "status": self.status,
            "traffic_light": self.traffic_light,
            "rul_hours": self.rul_hours,
            "rul_range_hours": list(self.rul_range_hours),
            "recommended_action": self.recommended_action,
            "contributing_factors": list(self.contributing_factors),
        }


@dataclass(frozen=True)
class SimulationConfig:
    """Inputs controlling a deterministic journey simulation."""

    seed: int = 0
    scenario: Scenario | str = Scenario.IDEAL
    payload_class: PayloadClass | str | None = None
    num_steps: int = 60
    timestep_s: float = 1.0
    rated_capacity_t: float = 40.0
    start_time: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    initial_latitude_deg: float = 35.0000
    initial_longitude_deg: float = -117.0000
    initial_altitude_m: float = 850.0
    drive_cycle: DriveCycle | str = DriveCycle.SYNTHETIC_ROUTE
    start_condition: StartCondition | str = StartCondition.WARM
    cycle_source: str = "internal synthetic route"
    cycle_version: str = "route-1"

    def __post_init__(self) -> None:
        try:
            seed = int(self.seed)
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        object.__setattr__(self, "seed", seed)
        object.__setattr__(
            self,
            "scenario",
            _coerce_enum(self.scenario, Scenario, Scenario.IDEAL),
        )
        object.__setattr__(self, "drive_cycle", _coerce_enum(self.drive_cycle, DriveCycle, DriveCycle.SYNTHETIC_ROUTE))
        object.__setattr__(self, "start_condition", _coerce_enum(self.start_condition, StartCondition, StartCondition.WARM))
        if self.payload_class is not None:
            object.__setattr__(
                self,
                "payload_class",
                _coerce_enum(self.payload_class, PayloadClass, PayloadClass.MEDIUM),
            )
        try:
            steps = int(self.num_steps)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_steps must be a positive integer") from exc
        if steps < 1:
            raise ValueError("num_steps must be at least 1")
        object.__setattr__(self, "num_steps", steps)
        timestep = float(self.timestep_s)
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError("timestep_s must be a positive finite number")
        object.__setattr__(self, "timestep_s", timestep)
        capacity = float(self.rated_capacity_t)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("rated_capacity_t must be a positive finite number")
        object.__setattr__(self, "rated_capacity_t", capacity)
        if not isinstance(self.start_time, datetime):
            raise ValueError("start_time must be a datetime")
        if self.start_time.tzinfo is None:
            object.__setattr__(self, "start_time", self.start_time.replace(tzinfo=timezone.utc))

    @property
    def steps(self) -> int:
        return self.num_steps

    @property
    def capacity_t(self) -> float:
        return self.rated_capacity_t

    @property
    def capacity(self) -> float:
        return self.rated_capacity_t


@dataclass(frozen=True)
class TelemetryFrame:
    """One synchronized sample across all simulated subsystems."""

    timestamp: datetime
    step: int
    elapsed_s: float
    scenario: Scenario
    mode: OperatingMode
    payload_class: PayloadClass
    payload_mass_t: float
    rated_capacity_t: float
    payload_utilization_pct: float
    stress_level_pct: float
    payload: PayloadTelemetry
    faults: FaultFlags
    imu: IMUTelemetry
    engine: EngineTelemetry
    transmission: TransmissionTelemetry
    brake: BrakeTelemetry
    suspension: SuspensionTelemetry
    frame: FrameTelemetry
    gnss: GNSSTelemetry
    lidar: LiDARTelemetry
    radar: RadarTelemetry
    dss: DSSTelemetry
    engine_health: EngineHealth
    overhaul_progress_pct: float = 0.0
    body_position_pct: float = 0.0
    hoist_angle_deg: float = 0.0
    chassis_pitch_deg: float = 0.0
    rack_deg: float = 0.0
    engine_load_pct: float = 0.0
    drive_cycle: DriveCycle = DriveCycle.SYNTHETIC_ROUTE
    start_condition: StartCondition = StartCondition.WARM
    cycle_phase: str = "route"
    cycle_source: str = "internal synthetic route"
    cycle_version: str = "route-1"
    cumulative_fuel_energy_kwh: float = 0.0
    cumulative_fuel_volume_l: float = 0.0
    coolant_condition_pct: float = 96.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", _coerce_enum(self.scenario, Scenario, Scenario.IDEAL))
        object.__setattr__(self, "mode", _coerce_enum(self.mode, OperatingMode, OperatingMode.IDLE))
        object.__setattr__(
            self,
            "payload_class",
            _coerce_enum(self.payload_class, PayloadClass, PayloadClass.MEDIUM),
        )
        object.__setattr__(self, "drive_cycle", _coerce_enum(self.drive_cycle, DriveCycle, DriveCycle.SYNTHETIC_ROUTE))
        object.__setattr__(self, "start_condition", _coerce_enum(self.start_condition, StartCondition, StartCondition.WARM))

    @property
    def capacity_utilization_pct(self) -> float:
        return self.payload_utilization_pct

    @property
    def payload_capacity_pct(self) -> float:
        return self.payload_utilization_pct

    @property
    def payload_pct(self) -> float:
        return self.payload_utilization_pct

    @property
    def payload_ratio(self) -> float:
        return self.payload_utilization_ratio

    @property
    def journey_engine_load_pct(self) -> float:
        """Categorical engine/payload load selected once for the journey."""

        return self.engine_load_pct

    @property
    def capacity_ratio(self) -> float:
        return self.payload_utilization_ratio

    @property
    def payload_utilization(self) -> float:
        """Percent form used by compact dashboard cards."""

        return self.payload_utilization_pct

    @property
    def capacity_utilization(self) -> float:
        return self.payload_utilization_pct

    @property
    def class_name(self) -> str:
        return self.payload_class.value

    @property
    def payload_utilization_ratio(self) -> float:
        return self.payload_utilization_pct / 100.0

    @property
    def capacity_t(self) -> float:
        return self.rated_capacity_t

    @property
    def fault_flags(self) -> FaultFlags:
        return self.faults

    @property
    def health(self) -> EngineHealth:
        return self.engine_health

    @property
    def health_score(self) -> float:
        return self.engine_health.score

    @property
    def health_classification(self) -> EngineHealthClass:
        return self.engine_health.classification

    @property
    def rul_hours(self) -> float:
        return self.engine_health.rul_hours

    @property
    def rul(self) -> float:
        return self.engine_health.rul_hours

    @property
    def remaining_useful_life(self) -> float:
        return self.engine_health.rul_hours

    @property
    def recommended_action(self) -> str:
        return self.engine_health.recommended_action

    @property
    def action(self) -> str:
        return self.engine_health.recommended_action

    @property
    def health_status(self) -> str:
        return self.engine_health.status

    @property
    def engine_health_score(self) -> float:
        return self.engine_health.score

    @property
    def operational_statuses(self) -> dict[str, str]:
        """Classify current readings against the supplied reference bands."""

        return {
            "engine_block_temperature": classify_operational_range("engine_block_temperature", self.engine.coolant_outlet_temp_c),
            "engine_oil_temperature": classify_operational_range("engine_oil_temperature", self.engine.oil_temp_c),
            "exhaust_temperature": classify_operational_range("exhaust_temperature", self.engine.exhaust_temp_c),
            "engine_oil_pressure": classify_operational_range("engine_oil_pressure", self.engine.oil_pressure_kpa),
            "engine_block_vibration": classify_operational_range("engine_block_vibration", self.engine.block_vibration_mm_s),
            "gearbox_oil_temperature": classify_operational_range("gearbox_oil_temperature", self.transmission.oil_temp_c),
            "transmission_oil_pressure": classify_operational_range("transmission_oil_pressure", self.transmission.oil_pressure_kpa),
            "gearbox_vibration": classify_operational_range("gearbox_vibration", self.transmission.vibration_mm_s),
            "hydraulic_oil_condition": classify_operational_range("hydraulic_oil_condition", self.suspension.hydraulic_oil_condition_pct),
            "hydraulic_actuator_vibration": classify_operational_range("hydraulic_actuator_vibration", self.suspension.actuator_vibration_mm_s),
            "coolant_pump_temperature": classify_operational_range("coolant_pump_temperature", self.engine.coolant_pump_temp_c),
            "coolant_condition": classify_operational_range("coolant_condition", self.coolant_condition_pct),
        }

    # Frequently plotted scalar aliases keep a Streamlit chart from needing
    # to know the complete nested sensor schema.
    @property
    def temperature(self) -> float:
        return self.engine.coolant_temp_c

    @property
    def vibration(self) -> float:
        return self.imu.vibration_rms_g

    @property
    def speed_mps(self) -> float:
        return self.gnss.speed_mps

    @property
    def speed(self) -> float:
        return self.gnss.speed_mps

    @property
    def rpm(self) -> float:
        return self.engine.rpm

    @property
    def load_pct(self) -> float:
        return self.engine.load_pct

    @property
    def coolant_temp_c(self) -> float:
        return self.engine.coolant_temp_c

    @property
    def oil_pressure_kpa(self) -> float:
        return self.engine.oil_pressure_kpa

    @property
    def payload_t(self) -> float:
        return self.payload_mass_t

    @property
    def payload_load_t(self) -> float:
        return self.payload_mass_t

    @property
    def payload_capacity_t(self) -> float:
        return self.rated_capacity_t

    @property
    def payload_mass(self) -> float:
        return self.payload_mass_t

    @property
    def rated_capacity(self) -> float:
        return self.rated_capacity_t

    @property
    def max_payload(self) -> float:
        return self.rated_capacity_t

    def as_dict(self) -> dict[str, Any]:
        data = _jsonable(asdict(self))
        health = data.get("engine_health", {})
        if isinstance(health, dict):
            health["traffic_light"] = self.engine_health.traffic_light
            health["rul_range_hours"] = list(self.engine_health.rul_range_hours)
        data["operational_statuses"] = self.operational_statuses
        return data

    to_dict = as_dict

    def to_dashboard_dict(self) -> dict[str, Any]:
        """Return a compact flat record suitable for ``st.dataframe`` or JS."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "elapsed_s": self.elapsed_s,
            "scenario": self.scenario.value,
            "drive_cycle": self.drive_cycle.value,
            "start_condition": self.start_condition.value,
            "cycle_phase": self.cycle_phase,
            "mode": self.mode.value,
            "payload_class": self.payload_class.value,
            "payload": self.payload_mass_t,
            "maxPayload": self.rated_capacity_t,
            "payloadPercent": self.payload_utilization_pct,
            "engine_load_pct": self.engine_load_pct,
            "temperature": self.engine.coolant_temp_c,
            "vibration": self.imu.vibration_rms_g,
            "speed": self.gnss.speed_mps,
            "health_score": self.health_score,
            "health_classification": self.health_classification.value,
            "health_status": self.engine_health.status,
            "rul_hours": self.rul_hours,
            "recommended_action": self.recommended_action,
            "stress_level_pct": self.stress_level_pct,
            "faults": self.faults.as_dict(),
        }


# Short acronym-oriented aliases make the model pleasant to use from a small
# Streamlit script while the verbose ``*Telemetry`` names remain canonical.
IMU = IMUTelemetry
Engine = EngineTelemetry
Transmission = TransmissionTelemetry
Brake = BrakeTelemetry
Suspension = SuspensionTelemetry
Frame = FrameTelemetry
LiDAR = LiDARTelemetry
Radar = RadarTelemetry
DSS = DSSTelemetry


@dataclass
class Journey(Sequence[TelemetryFrame]):
    """A sequence-like result returned by :func:`simulate_journey`."""

    frames: list[TelemetryFrame]
    config: SimulationConfig
    payload_profile: PayloadProfile

    def __post_init__(self) -> None:
        self.frames = list(self.frames)
        categories = {frame.payload_class for frame in self.frames}
        if len(categories) > 1:
            raise ValueError("payload_class must remain fixed for a journey")
        if self.frames and next(iter(categories)) != self.payload_profile.payload_class:
            raise ValueError("journey frames and payload_profile use different payload classes")

    @property
    def records(self) -> list[TelemetryFrame]:
        return self.frames

    @property
    def telemetry(self) -> list[TelemetryFrame]:
        return self.frames

    @property
    def samples(self) -> list[TelemetryFrame]:
        return self.frames

    @property
    def scenario(self) -> Scenario:
        return self.config.scenario

    @property
    def payload_class(self) -> PayloadClass:
        return self.payload_profile.payload_class

    @property
    def payload_classes(self) -> tuple[PayloadClass, ...]:
        return tuple(sorted({frame.payload_class for frame in self.frames}, key=lambda item: item.value))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int | slice) -> TelemetryFrame | list[TelemetryFrame]:
        return self.frames[index]

    def __iter__(self) -> Iterator[TelemetryFrame]:
        return iter(self.frames)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [frame.as_dict() for frame in self.frames]

    to_records = to_dicts

    def dashboard_rows(self) -> list[dict[str, Any]]:
        """Return flat rows for a Streamlit table or chart data source."""

        return [frame.to_dashboard_dict() for frame in self.frames]

    def summary(self) -> dict[str, Any]:
        if not self.frames:
            return {
                "samples": 0,
                "scenario": self.scenario.value,
                "payload_class": self.payload_class.value,
            }
        health_scores = [frame.health_score for frame in self.frames]
        modes: dict[str, int] = {}
        fault_counts: dict[str, int] = {}
        for frame in self.frames:
            modes[frame.mode.value] = modes.get(frame.mode.value, 0) + 1
            for name in frame.faults.fault_names:
                fault_counts[name] = fault_counts.get(name, 0) + 1
        return {
            "samples": len(self.frames),
            "scenario": self.scenario.value,
            "payload_class": self.payload_class.value,
            "payload_utilization_pct": self.payload_profile.utilization_pct,
            "mode_counts": modes,
            "fault_counts": fault_counts,
            "health_score_start": health_scores[0],
            "health_score_end": health_scores[-1],
            "health_score_min": min(health_scores),
            "health_score_max": max(health_scores),
            "rul_hours_end": self.frames[-1].rul_hours,
        }


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses, enums, and datetimes for Streamlit."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def classify_engine_health(score: float) -> EngineHealthClass:
    """Map a 0--100 health score to a stable classification.

    Thresholds are intentionally simple and transparent so a dashboard can
    explain why a classification changed:

    * 90--100: excellent
    * 75--89.99: good
    * 60--74.99: fair
    * 40--59.99: poor
    * below 40: critical
    """

    value = _clamp(float(score), 0.0, 100.0)
    if value >= 90.0:
        return EngineHealthClass.EXCELLENT
    if value >= 75.0:
        return EngineHealthClass.GOOD
    if value >= 60.0:
        return EngineHealthClass.FAIR
    if value >= 40.0:
        return EngineHealthClass.POOR
    return EngineHealthClass.CRITICAL


classify_health = classify_engine_health
health_classification = classify_engine_health


def recommended_action_for_health(classification: EngineHealthClass | str | float) -> str:
    """Return a concise operator action for a health classification."""

    if isinstance(classification, (int, float)):
        health = classify_engine_health(float(classification))
    else:
        health = _coerce_enum(classification, EngineHealthClass, EngineHealthClass.FAIR)
    actions = {
        EngineHealthClass.EXCELLENT: "Continue operation and follow the scheduled service plan.",
        EngineHealthClass.GOOD: "Continue operation; trend engine parameters at the next inspection.",
        EngineHealthClass.FAIR: "Schedule an engine inspection and monitor temperature and oil pressure.",
        EngineHealthClass.POOR: "Plan corrective maintenance before the next shift.",
        EngineHealthClass.CRITICAL: "Stop operation safely and perform immediate engine service.",
    }
    return actions[health]


def estimate_rul_hours(
    score: float,
    *,
    scenario: Scenario | str = Scenario.IDEAL,
    design_life_hours: float = 12_000.0,
) -> float:
    """Estimate remaining useful life from health score.

    This is a transparent planning estimate, not a certified maintenance
    interval.  Scenario adjustments make the stress and overhaul simulations
    visibly different while remaining deterministic.
    """

    health = _clamp(float(score), 0.0, 100.0)
    life = max(0.0, float(design_life_hours)) * (health / 100.0) ** 1.35
    selected = _coerce_enum(scenario, Scenario, Scenario.IDEAL)
    if selected is Scenario.STRESS:
        life *= 0.88
    elif selected is Scenario.OVERHAUL:
        life = min(max(0.0, float(design_life_hours)), life * 1.04)
    return _bounded_round(max(0.0, life), 1)


def calculate_engine_health(
    engine: EngineTelemetry | None = None,
    faults: FaultFlags | None = None,
    *,
    payload_utilization_pct: float = 72.0,
    stress_level_pct: float = 0.0,
    scenario: Scenario | str = Scenario.IDEAL,
    mode: OperatingMode | str = OperatingMode.IDLE,
) -> EngineHealth:
    """Calculate score, classification, RUL, and action from telemetry.

    All inputs are optional enough for a Streamlit control panel to call this
    function while a form is only partially filled in.
    """

    engine = engine or EngineTelemetry()
    faults = faults or FaultFlags()
    selected_scenario = _coerce_enum(scenario, Scenario, Scenario.IDEAL)
    selected_mode = _coerce_enum(mode, OperatingMode, OperatingMode.IDLE)
    utilization = max(0.0, float(payload_utilization_pct))
    stress = _clamp(float(stress_level_pct), 0.0, 100.0)

    score = 100.0
    factors: list[str] = []

    # Scenario and operating-load penalties.
    if selected_scenario is Scenario.STRESS:
        penalty = 5.0 + 0.27 * stress
        score -= penalty
        factors.append("stress scenario")
    elif selected_scenario is Scenario.OVERHAUL:
        # A small residual service transient fades over the journey; the
        # caller can pass a lower stress_level_pct as service settles.
        score -= max(0.0, 2.0 + 0.03 * stress)

    if engine.load_pct > 75.0:
        score -= (engine.load_pct - 75.0) * 0.10
        factors.append("high engine load")
    if engine.coolant_temp_c > 95.0:
        score -= (engine.coolant_temp_c - 95.0) * 0.32
        factors.append("high coolant temperature")
    if engine.oil_pressure_kpa < 300.0:
        score -= (300.0 - engine.oil_pressure_kpa) * 0.08
        factors.append("low oil pressure")
    if engine.vibration_rms_g > 0.25:
        score -= (engine.vibration_rms_g - 0.25) * 18.0
        factors.append("elevated vibration")
    if utilization > 85.0:
        score -= (utilization - 85.0) * 0.12
        factors.append("high payload utilization")
    if selected_mode is OperatingMode.UPSLOPE:
        score -= 1.5
    elif selected_mode is OperatingMode.BRAKING:
        score -= 0.5

    fault_weights = {
        "engine_overheat": 12.0,
        "low_oil_pressure": 10.0,
        "transmission_slip": 5.0,
        "brake_wear": 2.0,
        "suspension_overload": 4.0,
        "frame_strain": 7.0,
        "gnss_degraded": 0.5,
        "lidar_obstructed": 0.5,
        "radar_fault": 0.5,
        "imu_fault": 3.0,
        "payload_overload": 8.0,
        "dss_alert": 1.0,
    }
    for name, weight in fault_weights.items():
        if getattr(faults, name):
            score -= weight
            factors.append(name.replace("_", " "))

    score = _clamp(score, 0.0, 100.0)
    classification = classify_engine_health(score)
    return EngineHealth(
        score=score,
        classification=classification,
        rul_hours=estimate_rul_hours(score, scenario=selected_scenario),
        recommended_action=recommended_action_for_health(classification),
        contributing_factors=tuple(dict.fromkeys(factors)),
    )


def engine_health_score(
    engine: EngineTelemetry | None = None,
    faults: FaultFlags | None = None,
    **kwargs: Any,
) -> float:
    """Convenience wrapper returning only the numeric health score."""

    return calculate_engine_health(engine, faults, **kwargs).score


_MODE_LOAD = {
    OperatingMode.PARKED: 0.02,
    OperatingMode.IDLE: 0.12,
    OperatingMode.HAULING: 0.62,
    OperatingMode.UPSLOPE: 0.90,
    OperatingMode.DOWNSLOPE: 0.42,
    OperatingMode.BRAKING: 0.50,
    OperatingMode.DUMPING: 0.38,
}

_MODE_SPEED_MPS = {
    OperatingMode.PARKED: 0.0,
    OperatingMode.IDLE: 1.0,
    OperatingMode.HAULING: 8.5,
    OperatingMode.UPSLOPE: 4.5,
    OperatingMode.DOWNSLOPE: 10.0,
    OperatingMode.BRAKING: 2.0,
    OperatingMode.DUMPING: 1.0,
}


def _mode_for_step(step: int, total: int) -> OperatingMode:
    """Use a repeatable route profile that covers all operating modes."""

    if total <= 1:
        return OperatingMode.PARKED
    # Keep short demonstrations useful: when there are at least six samples,
    # expose one sample of each categorical mode before following the longer
    # route profile below.  This is especially handy for a small Streamlit
    # preview or a unit test with only a handful of rows.
    if total >= 7 and step < 7:
        return (
            OperatingMode.PARKED,
            OperatingMode.IDLE,
            OperatingMode.HAULING,
            OperatingMode.UPSLOPE,
            OperatingMode.DOWNSLOPE,
            OperatingMode.BRAKING,
            OperatingMode.DUMPING,
        )[step]
    fraction = step / float(total - 1)
    if fraction < 0.04:
        return OperatingMode.PARKED
    if fraction < 0.08:
        return OperatingMode.IDLE
    if fraction < 0.28:
        return OperatingMode.HAULING
    if fraction < 0.48:
        return OperatingMode.UPSLOPE
    if fraction < 0.65:
        return OperatingMode.DOWNSLOPE
    if fraction < 0.80:
        return OperatingMode.BRAKING
    if fraction < 0.90:
        return OperatingMode.DUMPING
    return OperatingMode.HAULING


def _research_cycle_profile(elapsed_s: float) -> tuple[str, OperatingMode, float, float]:
    """Return phase, mode, vehicle speed (m/s), and grade for the C-WTVC proxy.

    The paper reports a 1,800 s C-WTVC split into urban (900 s), highway
    (468 s), and expressway (432 s). The attached plots are used as qualitative
    anchors; this is a reproducible proxy trace, not the authors' unavailable
    raw measurements.
    """

    t = max(0.0, float(elapsed_s))
    if t < 900.0:
        phase = "urban"
        # Frequent stop/go transients, with short near-idle windows.
        speed_kph = 30.0 + 20.0 * math.sin(t / 23.0) + 12.0 * math.sin(t / 7.0)
        if int(t // 120.0) % 3 == 1:
            speed_kph -= 22.0 * math.exp(-((t % 120.0 - 74.0) / 18.0) ** 2)
        mode = OperatingMode.IDLE if speed_kph < 5.0 else OperatingMode.HAULING
        grade = 0.012 * math.sin(t / 83.0)
    elif t < 1368.0:
        phase = "highway"
        local = t - 900.0
        speed_kph = 52.0 + 11.0 * math.sin(local / 42.0) + 5.0 * math.sin(local / 9.0)
        mode = OperatingMode.DOWNSLOPE if math.sin(local / 42.0) < -0.55 else OperatingMode.HAULING
        grade = 0.006 * math.sin(local / 51.0)
    else:
        phase = "expressway"
        local = t - 1368.0
        ramp = min(1.0, local / 130.0)
        speed_kph = 18.0 + 53.0 * ramp + 2.2 * math.sin(local / 18.0) * (1.0 - ramp)
        mode = OperatingMode.HAULING if speed_kph > 8.0 else OperatingMode.IDLE
        grade = 0.002 * math.sin(local / 70.0)
    return phase, mode, _clamp(speed_kph, 0.0, 82.0) / 3.6, grade


def _noise(rng: random.Random, amplitude: float) -> float:
    return rng.uniform(-amplitude, amplitude) if amplitude else 0.0


def _stress_level(scenario: Scenario, progress: float) -> tuple[float, float]:
    """Return (stress level percentage, overhaul progress percentage)."""

    if scenario is Scenario.STRESS:
        # Starts elevated and rises to full stress by the end of the haul.
        return 35.0 + 65.0 * progress, 0.0
    if scenario is Scenario.OVERHAUL:
        # Residual service transient fades as the freshly serviced truck runs.
        return 18.0 * (1.0 - progress), progress * 100.0
    return 0.0, 0.0


def _default_payload_class(scenario: Scenario) -> PayloadClass:
    if scenario is Scenario.STRESS:
        # Heavy is 94% of rated capacity, deliberately above the 85% stress
        # threshold while remaining distinct from the explicit overload class.
        return PayloadClass.HEAVY
    return PayloadClass.MEDIUM


def _resolve_config(
    config: SimulationConfig | int | None,
    *,
    seed: int | None,
    scenario: Scenario | str | None,
    payload_class: PayloadClass | str | None,
    num_steps: int | None,
    steps: int | None,
    n_steps: int | None,
    timestep_s: float | None,
    duration_seconds: float | None,
    rated_capacity_t: float | None,
    capacity_t: float | None,
    start_time: datetime | None,
    drive_cycle: DriveCycle | str | None,
    start_condition: StartCondition | str | None,
) -> SimulationConfig:
    """Normalize the convenience arguments accepted by ``simulate_journey``."""

    if isinstance(config, int):
        if num_steps is not None:
            raise ValueError("specify the number of steps only once")
        num_steps = config
        config = None

    aliases = [value for value in (num_steps, steps, n_steps) if value is not None]
    if len({int(value) for value in aliases}) > 1:
        raise ValueError("num_steps, steps, and n_steps disagree")
    requested_steps = int(aliases[0]) if aliases else None

    if rated_capacity_t is not None and capacity_t is not None and float(rated_capacity_t) != float(capacity_t):
        raise ValueError("rated_capacity_t and capacity_t disagree")
    requested_capacity = rated_capacity_t if rated_capacity_t is not None else capacity_t

    if config is None:
        selected_scenario = _coerce_enum(scenario, Scenario, Scenario.IDEAL)
        selected_cycle = _coerce_enum(drive_cycle, DriveCycle, DriveCycle.SYNTHETIC_ROUTE)
        if requested_steps is None:
            requested_steps = 181 if selected_cycle is DriveCycle.C_WTVC else 60
        if duration_seconds is not None:
            tick = 1.0 if timestep_s is None else float(timestep_s)
            if tick <= 0:
                raise ValueError("timestep_s must be positive")
            requested_steps = (
                max(1, int(math.floor(float(duration_seconds) / tick)) + 1)
                if selected_cycle is DriveCycle.C_WTVC
                else max(1, int(math.ceil(float(duration_seconds) / tick)))
            )
        return SimulationConfig(
            seed=0 if seed is None else seed,
            scenario=selected_scenario,
            payload_class=payload_class,
            num_steps=requested_steps,
            timestep_s=(10.0 if selected_cycle is DriveCycle.C_WTVC else 1.0) if timestep_s is None else timestep_s,
            rated_capacity_t=40.0 if requested_capacity is None else requested_capacity,
            start_time=SimulationConfig.start_time if start_time is None else start_time,
            drive_cycle=DriveCycle.SYNTHETIC_ROUTE if drive_cycle is None else drive_cycle,
            start_condition=StartCondition.WARM if start_condition is None else start_condition,
            cycle_source="internal synthetic route" if drive_cycle is None or str(drive_cycle) != DriveCycle.C_WTVC.value else "Ma et al., Fuel 328 (2022) 125307; qualitative proxy",
            cycle_version="route-1" if drive_cycle is None or str(drive_cycle) != DriveCycle.C_WTVC.value else "c-wtvc-proxy-2022-paper",
        )

    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig, an integer step count, or None")
    updates: dict[str, Any] = {}
    if seed is not None:
        updates["seed"] = seed
    if scenario is not None:
        updates["scenario"] = scenario
    if payload_class is not None:
        updates["payload_class"] = payload_class
    if requested_steps is not None:
        updates["num_steps"] = requested_steps
    if timestep_s is not None:
        updates["timestep_s"] = timestep_s
    if requested_capacity is not None:
        updates["rated_capacity_t"] = requested_capacity
    if start_time is not None:
        updates["start_time"] = start_time
    if drive_cycle is not None:
        updates["drive_cycle"] = drive_cycle
        updates["cycle_source"] = "internal synthetic route" if str(drive_cycle) != DriveCycle.C_WTVC.value else "Ma et al., Fuel 328 (2022) 125307; qualitative proxy"
        updates["cycle_version"] = "route-1" if str(drive_cycle) != DriveCycle.C_WTVC.value else "c-wtvc-proxy-2022-paper"
    if start_condition is not None:
        updates["start_condition"] = start_condition
    if duration_seconds is not None:
        tick = float(updates.get("timestep_s", config.timestep_s))
        if tick <= 0:
            raise ValueError("timestep_s must be positive")
        selected_cycle = _coerce_enum(updates.get("drive_cycle", config.drive_cycle), DriveCycle, DriveCycle.SYNTHETIC_ROUTE)
        updates["num_steps"] = (
            max(1, int(math.floor(float(duration_seconds) / tick)) + 1)
            if selected_cycle is DriveCycle.C_WTVC
            else max(1, int(math.ceil(float(duration_seconds) / tick)))
        )
    return replace(config, **updates) if updates else config


def simulate_journey(
    config: SimulationConfig | int | None = None,
    *,
    seed: int | None = None,
    scenario: Scenario | str | None = None,
    payload_class: PayloadClass | str | None = None,
    num_steps: int | None = None,
    steps: int | None = None,
    n_steps: int | None = None,
    timestep_s: float | None = None,
    duration_seconds: float | None = None,
    rated_capacity_t: float | None = None,
    capacity_t: float | None = None,
    start_time: datetime | None = None,
    drive_cycle: DriveCycle | str | None = None,
    start_condition: StartCondition | str | None = None,
) -> Journey:
    """Generate a deterministic multi-sensor dump-truck journey.

    Examples
    --------
    ``simulate_journey(seed=7, scenario="stress", num_steps=120)``
    ``simulate_journey(SimulationConfig(seed=7, num_steps=30))``
    ``simulate_journey(30, seed=7)``

    The same seed and configuration produce byte-for-byte equivalent values.
    A local ``random.Random`` instance is used, so the caller's global random
    state is never modified.
    """

    cfg = _resolve_config(
        config,
        seed=seed,
        scenario=scenario,
        payload_class=payload_class,
        num_steps=num_steps,
        steps=steps,
        n_steps=n_steps,
        timestep_s=timestep_s,
        duration_seconds=duration_seconds,
        rated_capacity_t=rated_capacity_t,
        capacity_t=capacity_t,
        start_time=start_time,
        drive_cycle=drive_cycle,
        start_condition=start_condition,
    )
    rng = random.Random(cfg.seed)
    selected_payload = (
        _coerce_enum(cfg.payload_class, PayloadClass, PayloadClass.MEDIUM)
        if cfg.payload_class is not None
        else _default_payload_class(cfg.scenario)
    )
    profile = payload_profile_for(selected_payload, cfg.rated_capacity_t)
    payload = PayloadTelemetry.from_profile(profile)

    frames: list[TelemetryFrame] = []
    latitude = cfg.initial_latitude_deg
    longitude = cfg.initial_longitude_deg
    altitude = cfg.initial_altitude_m
    heading = 90.0
    previous_cycle_speed = 0.0
    rpm_state = 800.0 if cfg.start_condition is StartCondition.COLD else 1_150.0
    coolant_mean_state = 28.0 if cfg.start_condition is StartCondition.COLD else 80.0
    coolant_flow_state = 0.65 if cfg.start_condition is StartCondition.COLD else 2.15
    thermostat_open_state = 0.0 if cfg.start_condition is StartCondition.COLD else 0.55
    cumulative_fuel_energy_kwh = 0.0
    cumulative_fuel_volume_l = 0.0

    for step in range(cfg.num_steps):
        progress = 0.0 if cfg.num_steps <= 1 else step / float(cfg.num_steps - 1)
        elapsed_s = step * cfg.timestep_s
        if cfg.drive_cycle is DriveCycle.C_WTVC:
            cycle_phase, mode, research_speed, research_grade = _research_cycle_profile(elapsed_s)
        else:
            cycle_phase, mode, research_speed, research_grade = "route", _mode_for_step(step, cfg.num_steps), 0.0, 0.0
        stress_pct, overhaul_pct = _stress_level(cfg.scenario, progress)
        payload_stress_pct = _clamp(
            (profile.utilization_ratio - 0.85) / 0.15 * 100.0,
            0.0,
            100.0,
        )
        # A heavy or overloaded payload is a journey-level condition: it
        # raises thermal, vibration, structural and hydraulic readings even in
        # an otherwise ideal route.  The selected category itself remains
        # unchanged from the first sample to the last.
        stress_pct = max(stress_pct, payload_stress_pct)
        stress_ratio = stress_pct / 100.0
        mode_load = _MODE_LOAD[mode]
        base_speed = _MODE_SPEED_MPS[mode]
        if cfg.drive_cycle is DriveCycle.C_WTVC:
            speed_delta = research_speed - previous_cycle_speed
            transient_index = _clamp(abs(speed_delta) / max(0.01, 0.8 * cfg.timestep_s) + abs(research_grade) * 12.0, 0.0, 1.0)
            previous_cycle_speed = research_speed
        else:
            speed_delta = 0.0
            transient_index = 0.0

        # Engine and driveline.
        load_pct = _clamp(
            100.0
            * (0.10 + 0.72 * mode_load + 0.16 * max(0.0, profile.utilization_ratio - 0.5))
            + 14.0 * stress_ratio
            + _noise(rng, 2.0),
            0.0,
            100.0,
        )
        if mode is OperatingMode.PARKED:
            load_pct = _clamp(load_pct, 0.0, 12.0)
        elif mode is OperatingMode.IDLE:
            load_pct = _clamp(load_pct, 5.0, 28.0)
        rpm = _clamp(680.0 + load_pct * 16.0 + _noise(rng, 18.0), 600.0, 2_200.0)
        torque = _clamp(90.0 + load_pct * 23.0 + _noise(rng, 16.0), 0.0, 2_700.0)
        if cfg.scenario is Scenario.OVERHAUL:
            thermal_residual = 4.0 * (1.0 - progress)
            oil_residual = 8.0 * (1.0 - progress)
        else:
            thermal_residual = 0.0
            oil_residual = 0.0
        coolant = (
            72.0
            + 0.21 * load_pct
            + 26.0 * stress_ratio
            + thermal_residual
            + _noise(rng, 1.2)
        )
        oil_temp = 68.0 + 0.20 * load_pct + 20.0 * stress_ratio + _noise(rng, 1.0)
        oil_pressure = _clamp(
            485.0
            - 0.20 * load_pct
            - 32.0 * stress_ratio
            + oil_residual
            + _noise(rng, 5.0),
            280.0,
            540.0,
        )
        vibration = _clamp(
            0.035
            + 0.0014 * load_pct
            + 0.27 * stress_ratio
            + _noise(rng, 0.012),
            0.01,
            1.2,
        )
        boost = _clamp(98.0 + 1.25 * load_pct + 22.0 * stress_ratio + _noise(rng, 2.0), 80.0, 240.0)
        exhaust = _clamp(190.0 + 4.1 * load_pct + 170.0 * stress_ratio + _noise(rng, 8.0), 120.0, 850.0)
        fuel_rate = _clamp(4.0 + 0.26 * load_pct + 4.0 * stress_ratio + _noise(rng, 0.4), 0.0, 50.0)
        throttle = _clamp(load_pct * 0.92 + _noise(rng, 2.0), 0.0, 100.0)
        oil_level = _clamp(99.0 - 0.04 * step - 2.0 * stress_ratio + _noise(rng, 0.15), 70.0, 100.0)
        fuel_level = _clamp(98.0 - 0.10 * step - 3.5 * stress_ratio + _noise(rng, 0.25), 0.0, 100.0)
        cylinder_exhausts = tuple(
            _bounded_round(exhaust + offset + _noise(rng, 3.0), 1)
            for offset in (-16.0, -6.0, 5.0, 14.0, 2.0, -9.0)
        )
        air_filter_restriction = _clamp(
            2.4 + 0.026 * load_pct + 2.6 * stress_ratio + _noise(rng, 0.2),
            1.0,
            12.0,
        )
        power_kw = torque * rpm / 9549.0
        engine = EngineTelemetry(
            rpm=_bounded_round(rpm, 1),
            load_pct=_bounded_round(load_pct, 2),
            torque_nm=_bounded_round(torque, 1),
            power_kw=_bounded_round(power_kw, 2),
            coolant_temp_c=_bounded_round(coolant, 2),
            oil_temp_c=_bounded_round(oil_temp, 2),
            oil_pressure_kpa=_bounded_round(oil_pressure, 2),
            fuel_rate_lph=_bounded_round(fuel_rate, 2),
            boost_kpa=_bounded_round(boost, 2),
            exhaust_temp_c=_bounded_round(exhaust, 2),
            vibration_rms_g=_bounded_round(vibration, 3),
            throttle_position_pct=_bounded_round(throttle, 2),
            oil_level_pct=_bounded_round(oil_level, 2),
            fuel_level_pct=_bounded_round(fuel_level, 2),
            exhaust_temperatures_c=cylinder_exhausts,
            air_filter_restriction_kpa=_bounded_round(air_filter_restriction, 2),
            block_vibration_mm_s=_bounded_round(_clamp(2.0 + vibration * 30.0 + stress_ratio * 0.8, 2.0, 12.0), 2),
            coolant_pump_temp_c=_bounded_round(coolant, 2),
        )

        speed = _clamp(base_speed + _noise(rng, 0.35 if base_speed else 0.0), 0.0, 15.0)
        if mode is OperatingMode.BRAKING:
            speed = _clamp(base_speed - 1.2 * progress + _noise(rng, 0.2), 0.0, 8.0)
        if cfg.drive_cycle is DriveCycle.C_WTVC:
            speed = _clamp(research_speed + _noise(rng, 0.18), 0.0, 23.0)
            acceleration = speed_delta / max(0.001, cfg.timestep_s)
            cold_decay = math.exp(-elapsed_s / 240.0) if cfg.start_condition is StartCondition.COLD else 0.05 * math.exp(-elapsed_s / 60.0)
            speed_ratio = _clamp(speed / (22.5 if cycle_phase == "expressway" else 20.0), 0.0, 1.0)
            rpm_command = 700.0 + 650.0 * (0.35 + 0.45 * speed_ratio) + 250.0 * transient_index + 150.0 * cold_decay
            if cycle_phase == "expressway":
                rpm_command = 1_180.0 + 130.0 * speed_ratio + 100.0 * transient_index
            rpm_tau = 8.0 if cycle_phase == "urban" else 18.0
            rpm_state += (rpm_command - rpm_state) * _clamp(cfg.timestep_s / rpm_tau, 0.0, 1.0)
            rpm_state += _noise(rng, 12.0 if cycle_phase == "urban" else 5.0)
            rpm = _clamp(rpm_state, 589.0, 1_883.0)
            traction_kw = (
                (49_000.0 + profile.payload_mass_t * 1_000.0) * acceleration * speed
                + (49_000.0 + profile.payload_mass_t * 1_000.0) * 9.81 * research_grade * speed
                + 0.5 * 1.2 * 7.6 * speed**3
                + 0.008 * (49_000.0 + profile.payload_mass_t * 1_000.0) * 9.81 * speed
            ) / 1_000.0
            effective_power = _clamp(traction_kw / 0.88 + 18.0 + 32.0 * cold_decay, 12.0, 420.0)
            load_pct = _clamp(100.0 * effective_power / 420.0 + 8.0 * stress_ratio + _noise(rng, 1.2), 0.0, 100.0)
            if mode is OperatingMode.IDLE:
                load_pct = _clamp(8.0 + 8.0 * cold_decay + _noise(rng, 0.5), 4.0, 20.0)
                effective_power = 24.0 + 18.0 * cold_decay
            torque = _clamp(9549.0 * effective_power / max(rpm, 589.0), 0.0, 2_700.0)
            bsfc = 210.0 + 35.0 * (1.0 - load_pct / 100.0) ** 2 + 20.0 * cold_decay + 12.0 * transient_index
            fuel_mass_flow = bsfc * effective_power / 3_600_000.0
            fuel_rate = _clamp(fuel_mass_flow * 3_600.0 / 0.832, 0.0, 225.0)
            fuel_energy = fuel_mass_flow * 42.5e6 / 1_000.0
            if cycle_phase == "expressway":
                # Late expressway operation is the paper-aligned steady
                # regime: coolant share trends toward roughly 20% while the
                # early transient sections remain much more variable.
                coolant_fraction = _clamp(0.12 + 0.05 * load_pct / 100.0 + 0.06 * transient_index + 0.03 * cold_decay + 0.04 * (1.0 - thermostat_open_state), 0.06, 0.30)
                exhaust_fraction = _clamp(0.23 + 0.12 * load_pct / 100.0 + 0.04 * transient_index + 0.04 * cold_decay + 0.03, 0.20, 0.45)
            else:
                coolant_fraction = _clamp(0.06 + 0.10 * load_pct / 100.0 + 0.35 * transient_index + 0.25 * cold_decay + 0.15 * (1.0 - thermostat_open_state), 0.06, 0.76)
                exhaust_fraction = _clamp(0.23 + 0.12 * load_pct / 100.0 + 0.04 * transient_index + 0.04 * cold_decay, 0.20, 0.45)
            coolant_energy = fuel_energy * coolant_fraction
            exhaust_energy = fuel_energy * exhaust_fraction
            # The thermostat model has hysteresis and deliberately lags more
            # during cold start, matching the qualitative paper observation.
            target_open = 0.0 if coolant_mean_state <= 80.0 else _clamp((coolant_mean_state - 82.0) / 13.0, 0.0, 1.0)
            if thermostat_open_state > 0.0 and coolant_mean_state > 80.0:
                target_open = max(target_open, thermostat_open_state)
            if coolant_mean_state < 80.0:
                target_open = 0.0
            flow_command = 0.65 + 2.35 * target_open
            flow_tau = 180.0 if cfg.start_condition is StartCondition.COLD else 45.0
            coolant_flow_state += (flow_command - coolant_flow_state) * _clamp(cfg.timestep_s / flow_tau, 0.0, 1.0)
            thermostat_open_state += (target_open - thermostat_open_state) * _clamp(cfg.timestep_s / 20.0, 0.0, 1.0)
            coolant_delta_t = _clamp(coolant_energy / max(0.08, coolant_flow_state * 4.18), 0.5, 25.0)
            # First-order warm-up prior calibrated to the paper's qualitative
            # cold-start lag and approximately 85 C thermostat region.
            if cfg.start_condition is StartCondition.COLD:
                target_mean = 25.0 + 60.0 * (1.0 - math.exp(-elapsed_s / 260.0))
            else:
                target_mean = 80.0 + 12.0 * (1.0 - math.exp(-elapsed_s / 500.0))
            coolant_mean_state += (target_mean - coolant_mean_state) * _clamp(cfg.timestep_s / 90.0, 0.0, 1.0)
            coolant_mean_state = _clamp(coolant_mean_state, 20.0, 105.0)
            coolant_inlet = coolant_mean_state - coolant_delta_t * 0.5
            coolant_outlet = coolant_mean_state + coolant_delta_t * 0.5
            cumulative_fuel_energy_kwh += fuel_energy * cfg.timestep_s / 3_600.0
            cumulative_fuel_volume_l += fuel_rate * cfg.timestep_s / 3_600.0
            engine = replace(
                engine,
                rpm=_bounded_round(rpm, 1),
                load_pct=_bounded_round(load_pct, 2),
                torque_nm=_bounded_round(torque, 1),
                power_kw=_bounded_round(effective_power, 2),
                effective_power_kw=_bounded_round(effective_power, 2),
                coolant_temp_c=_bounded_round(coolant_outlet, 2),
                coolant_inlet_temp_c=_bounded_round(coolant_inlet, 2),
                oil_temp_c=_bounded_round(86.0 + 0.05 * load_pct + 5.0 * cold_decay + 8.0 * stress_ratio, 2),
                exhaust_temp_c=_bounded_round(_clamp(340.0 + 1.8 * load_pct + 35.0 * transient_index + 35.0 * cold_decay + _noise(rng, 5.0), 340.0, 700.0), 2),
                exhaust_temperatures_c=tuple(
                    _bounded_round(_clamp(340.0 + 1.8 * load_pct + 35.0 * transient_index + 35.0 * cold_decay + offset + _noise(rng, 2.0), 340.0, 720.0), 1)
                    for offset in (-16.0, -6.0, 5.0, 14.0, 2.0, -9.0)
                ),
                coolant_flow_kg_s=_bounded_round(coolant_flow_state, 4),
                thermostat_open_pct=_bounded_round(thermostat_open_state * 100.0, 2),
                coolant_energy_flow_kw=_bounded_round(coolant_energy, 3),
                fuel_energy_input_kw=_bounded_round(fuel_energy, 3),
                exhaust_energy_flow_kw=_bounded_round(exhaust_energy, 3),
                mechanical_loss_kw=_bounded_round(max(0.0, fuel_energy * 0.09), 3),
                remainder_loss_kw=_bounded_round(max(0.0, fuel_energy - effective_power - coolant_energy - exhaust_energy - fuel_energy * 0.09), 3),
                coolant_energy_pct=_bounded_round(coolant_fraction * 100.0, 2),
                exhaust_energy_pct=_bounded_round(exhaust_fraction * 100.0, 2),
                transient_index_pct=_bounded_round(transient_index * 100.0, 2),
                fuel_rate_lph=_bounded_round(fuel_rate, 2),
                throttle_position_pct=_bounded_round(_clamp(load_pct + transient_index * 18.0, 0.0, 100.0), 2),
                block_vibration_mm_s=_bounded_round(_clamp(2.0 + vibration * 30.0 + transient_index * 1.2, 2.0, 12.0), 2),
                coolant_pump_temp_c=_bounded_round(coolant_outlet, 2),
            )
        gear = 0 if mode is OperatingMode.PARKED else (1 if mode is OperatingMode.IDLE else int(_clamp(round(7.0 - speed / 2.0), 1, 7)))
        slip = _clamp(
            0.5 * stress_pct / 10.0 + (3.0 if mode is OperatingMode.UPSLOPE else 0.0) + _noise(rng, 0.6),
            0.0,
            30.0,
        )
        transmission = TransmissionTelemetry(
            gear=gear,
            input_rpm=_bounded_round(rpm, 1),
            output_rpm=_bounded_round(speed * 36.0, 1),
            oil_temp_c=_bounded_round(70.0 + 0.10 * load_pct + 18.0 * stress_ratio + _noise(rng, 1.0), 2),
            slip_pct=_bounded_round(slip, 2),
            efficiency_pct=_bounded_round(_clamp(99.0 - slip * 0.45, 60.0, 99.5), 2),
            output_torque_nm=_bounded_round(torque * (1.0 - slip / 100.0), 1),
            status=("parked" if mode is OperatingMode.PARKED else "braking" if mode is OperatingMode.BRAKING else "drive"),
            powertrain_oil_pressure_bar=_bounded_round(
                _clamp(18.0 + 0.12 * load_pct - 8.0 * stress_ratio + _noise(rng, 0.7), 7.0, 35.0),
                2,
            ),
            axle_temp_c=_bounded_round(62.0 + 0.18 * load_pct + 28.0 * stress_ratio + _noise(rng, 1.2), 2),
            oil_pressure_kpa=_bounded_round(
                _clamp(350.0 + 18.0 * (1.0 - stress_ratio) - 10.0 * (slip / 10.0) + _noise(rng, 2.0), 250.0, 430.0),
                2,
            ),
            vibration_mm_s=_bounded_round(_clamp(1.4 + slip * 0.18 + stress_ratio * 0.8 + _noise(rng, 0.08), 1.4, 8.0), 2),
        )

        # Brakes and suspension.
        pedal = 72.0 if mode is OperatingMode.BRAKING else (8.0 if mode is OperatingMode.DOWNSLOPE else 0.0)
        pedal = _clamp(pedal + _noise(rng, 3.0), 0.0, 100.0)
        pad_wear = _clamp(
            10.0
            + (55.0 * progress if cfg.scenario is Scenario.STRESS else 8.0 * progress)
            + (8.0 if mode is OperatingMode.BRAKING else 0.0)
            + _noise(rng, 1.0),
            0.0,
            100.0,
        )
        brake = BrakeTelemetry(
            pedal_pct=_bounded_round(pedal, 2),
            line_pressure_bar=_bounded_round(pedal * 1.65 + _noise(rng, 1.5), 2),
            disc_temp_c=_bounded_round(35.0 + pedal * 0.75 + 80.0 * stress_ratio + _noise(rng, 2.0), 2),
            pad_wear_pct=_bounded_round(pad_wear, 2),
            abs_active=bool(mode is OperatingMode.BRAKING and (speed < 2.0 or stress_ratio > 0.85)),
            parking_brake=mode is OperatingMode.PARKED,
            oil_temp_c=_bounded_round(38.0 + pedal * 0.42 + 32.0 * stress_ratio + _noise(rng, 1.0), 2),
            cooling_temp_c=_bounded_round(34.0 + pedal * 0.30 + 44.0 * stress_ratio + _noise(rng, 1.2), 2),
            accumulator_pressure_bar=_bounded_round(
                _clamp(148.0 - pedal * 0.18 - 42.0 * stress_ratio + _noise(rng, 1.5), 70.0, 165.0),
                2,
            ),
            stroke_pct=_bounded_round(_clamp(pedal * 0.76 + pad_wear * 0.18 + _noise(rng, 1.0), 0.0, 100.0), 2),
        )
        compression = _clamp(
            profile.utilization_ratio * 63.0 + mode_load * 20.0 + 14.0 * stress_ratio + _noise(rng, 1.5),
            0.0,
            100.0,
        )
        left_load = _clamp(50.0 + 4.0 * math.sin(step / 5.0) + _noise(rng, 1.0), 35.0, 65.0)
        suspension = SuspensionTelemetry(
            ride_height_mm=_bounded_round(925.0 - compression * 0.75 + _noise(rng, 1.0), 2),
            hydraulic_pressure_bar=_bounded_round(105.0 + compression * 1.25 + 25.0 * stress_ratio + _noise(rng, 2.0), 2),
            compression_pct=_bounded_round(compression, 2),
            left_load_pct=_bounded_round(left_load, 2),
            right_load_pct=_bounded_round(100.0 - left_load, 2),
            shock_temp_c=_bounded_round(34.0 + compression * 0.30 + 30.0 * stress_ratio + _noise(rng, 1.0), 2),
            front_left_pressure_bar=_bounded_round(102.0 + compression * 1.00 + 18.0 * stress_ratio + _noise(rng, 1.5), 2),
            front_right_pressure_bar=_bounded_round(102.0 + compression * 1.04 + 18.0 * stress_ratio + _noise(rng, 1.5), 2),
            rear_left_pressure_bar=_bounded_round(110.0 + compression * 1.12 + 22.0 * stress_ratio + _noise(rng, 1.5), 2),
            rear_right_pressure_bar=_bounded_round(110.0 + compression * 1.16 + 22.0 * stress_ratio + _noise(rng, 1.5), 2),
            hydraulic_oil_condition_pct=_bounded_round(_clamp(98.0 - compression * 0.10 - stress_ratio * 8.0 + _noise(rng, 0.4), 0.0, 100.0), 2),
            actuator_vibration_mm_s=_bounded_round(_clamp(1.6 + vibration * 10.0 + stress_ratio * 1.2 + _noise(rng, 0.08), 1.6, 8.0), 2),
        )
        strain = _clamp(
            45.0
            + profile.utilization_ratio * 440.0
            + mode_load * 220.0
            + 610.0 * stress_ratio
            + _noise(rng, 16.0),
            0.0,
            2_500.0,
        )
        frame = FrameTelemetry(
            strain_microstrain=_bounded_round(strain, 1),
            stress_mpa=_bounded_round(strain * 0.12, 2),
            deflection_mm=_bounded_round(compression * 0.045 + strain / 1_500.0, 3),
            twist_deg=_bounded_round(mode_load * 1.7 + stress_ratio * 2.4 + _noise(rng, 0.08), 3),
            fatigue_index_pct=_bounded_round(_clamp(5.0 + strain / 17.0, 0.0, 100.0), 2),
        )

        # IMU attitude follows route grade and gets noisier under stress.
        pitch = {
            OperatingMode.UPSLOPE: 8.0,
            OperatingMode.DOWNSLOPE: -5.0,
            OperatingMode.BRAKING: -2.0,
        }.get(mode, 0.0)
        pitch += _noise(rng, 0.35 + stress_ratio * 0.5)
        roll = 1.5 * math.sin(step / 4.0) + _noise(rng, 0.25 + stress_ratio * 0.35)
        yaw_rate = (speed / 4.0) * math.sin(step / 8.0) + _noise(rng, 0.15 + stress_ratio * 0.2)
        imu = IMUTelemetry(
            acceleration_x_mps2=_bounded_round((speed - base_speed) * 0.7 + _noise(rng, 0.08), 3),
            acceleration_y_mps2=_bounded_round(roll * 0.04 + _noise(rng, 0.05), 3),
            acceleration_z_mps2=_bounded_round(9.81 + compression * 0.006 + _noise(rng, 0.04), 3),
            angular_rate_x_dps=_bounded_round(roll * 0.6 + _noise(rng, 0.08), 3),
            angular_rate_y_dps=_bounded_round(pitch * 0.35 + _noise(rng, 0.08), 3),
            angular_rate_z_dps=_bounded_round(yaw_rate, 3),
            roll_deg=_bounded_round(roll, 3),
            pitch_deg=_bounded_round(pitch, 3),
            yaw_deg=_bounded_round((heading + step * 0.7) % 360.0, 3),
            vibration_rms_g=_bounded_round(vibration + stress_ratio * 0.04, 3),
        )

        # Position and perception sensors.
        heading = (90.0 + 9.0 * math.sin(step / 13.0) + _noise(rng, 0.4)) % 360.0
        latitude += (speed * math.cos(math.radians(heading)) * cfg.timestep_s) / 111_320.0
        longitude += (speed * math.sin(math.radians(heading)) * cfg.timestep_s) / (111_320.0 * max(0.2, math.cos(math.radians(latitude))))
        altitude += (0.05 if mode is OperatingMode.UPSLOPE else -0.03 if mode is OperatingMode.DOWNSLOPE else 0.0) * cfg.timestep_s
        gnss_accuracy = _clamp(0.7 + 4.8 * stress_ratio + _noise(rng, 0.12), 0.4, 20.0)
        satellites = max(5, int(round(15.0 - 6.0 * stress_ratio + _noise(rng, 1.0))))
        gnss = GNSSTelemetry(
            latitude_deg=_bounded_round(latitude, 7),
            longitude_deg=_bounded_round(longitude, 7),
            altitude_m=_bounded_round(altitude + _noise(rng, 0.4), 2),
            speed_mps=_bounded_round(speed, 2),
            heading_deg=_bounded_round(heading, 2),
            accuracy_m=_bounded_round(gnss_accuracy, 2),
            satellites=satellites,
            fix_type="RTK" if gnss_accuracy < 2.5 else "3D",
        )
        lidar_blocked = cfg.scenario is Scenario.STRESS and stress_ratio >= 0.78
        lidar_distance = _clamp(
            48.0 - 28.0 * stress_ratio + _noise(rng, 3.0),
            4.0 if lidar_blocked else 10.0,
            100.0,
        )
        lidar = LiDARTelemetry(
            obstacle_distance_m=_bounded_round(lidar_distance, 2),
            obstacle_confidence_pct=_bounded_round(92.0 - (30.0 if lidar_blocked else 0.0) + _noise(rng, 2.0), 2),
            points_per_scan=max(0, int(round(48_000 - 22_000 * stress_ratio + _noise(rng, 500)))),
            scan_rate_hz=_bounded_round(10.0 - 2.0 * stress_ratio, 2),
            blocked=lidar_blocked,
            obstacle_class="haul_traffic" if lidar_distance < 25.0 else "none",
        )
        radar_fault = cfg.scenario is Scenario.STRESS and stress_ratio >= 0.90
        radar_distance = _clamp(55.0 - 35.0 * stress_ratio + _noise(rng, 4.0), 6.0, 100.0)
        radar = RadarTelemetry(
            target_distance_m=_bounded_round(radar_distance, 2),
            relative_speed_mps=_bounded_round(-speed * 0.25 + _noise(rng, 0.3), 2),
            target_confidence_pct=_bounded_round(90.0 - (45.0 if radar_fault else 0.0) + _noise(rng, 2.0), 2),
            object_count=1 if radar_distance < 35.0 else 0,
            sensor_ok=not radar_fault,
            target_class="vehicle" if radar_distance < 35.0 else "none",
        )

        # Fault flags are derived from the generated measurements, keeping the
        # telemetry internally explainable instead of using arbitrary labels.
        provisional_collision = lidar_distance < 12.0 or radar_distance < 12.0
        provisional_rollover = strain >= 900.0 or (mode is OperatingMode.UPSLOPE and profile.utilization_ratio > 0.85)
        provisional_overspeed = speed > 12.0
        dss_level = (
            DSSAlertLevel.CRITICAL
            if provisional_collision or provisional_rollover and stress_ratio > 0.8
            else DSSAlertLevel.WARNING
            if provisional_rollover or provisional_overspeed
            else DSSAlertLevel.NONE
        )
        dss = DSSTelemetry(
            collision_warning=provisional_collision,
            rollover_warning=provisional_rollover,
            overspeed_warning=provisional_overspeed,
            emergency_braking=provisional_collision and mode is OperatingMode.BRAKING,
            fatigue_alert=cfg.scenario is Scenario.STRESS and progress > 0.85,
            alert_level=dss_level,
            driver_message=(
                "Immediate hazard: slow down and stop safely"
                if dss_level is DSSAlertLevel.CRITICAL
                else "Review vehicle stability and speed"
                if dss_level is DSSAlertLevel.WARNING
                else "No active alerts"
            ),
            camera_online=not (cfg.scenario is Scenario.STRESS and stress_ratio > 0.94),
            driver_presence_pct=_bounded_round(
                _clamp(100.0 - (12.0 if cfg.scenario is Scenario.STRESS and progress > 0.85 else 0.0) + _noise(rng, 0.5), 0.0, 100.0),
                2,
            ),
            seatbelt_fastened=True,
            distraction_score_pct=_bounded_round(_clamp(3.0 + 14.0 * stress_ratio + _noise(rng, 1.0), 0.0, 100.0), 2),
            fatigue_score_pct=_bounded_round(
                _clamp(4.0 + 56.0 * stress_ratio + 12.0 * progress + _noise(rng, 1.5), 0.0, 100.0),
                2,
            ),
            camera_fps=_bounded_round(30.0 - 8.0 * stress_ratio, 2),
        )
        faults = FaultFlags(
            engine_overheat=engine.coolant_temp_c >= 108.0,
            low_oil_pressure=engine.oil_pressure_kpa <= 250.0,
            transmission_slip=transmission.slip_pct >= 8.0,
            brake_wear=brake.pad_wear_pct >= 70.0,
            suspension_overload=suspension.compression_pct >= 72.0,
            frame_strain=frame.strain_microstrain >= 900.0,
            gnss_degraded=gnss.accuracy_m >= 3.5,
            lidar_obstructed=lidar.blocked,
            radar_fault=not radar.sensor_ok,
            imu_fault=imu.vibration_rms_g >= 0.65,
            payload_overload=payload.utilization_pct > 100.0,
            dss_alert=dss.any_alert,
        )
        health = calculate_engine_health(
            engine,
            faults,
            payload_utilization_pct=payload.utilization_pct,
            stress_level_pct=stress_pct,
            scenario=cfg.scenario,
            mode=mode,
        )
        body_position = _clamp(
            (32.0 if mode is OperatingMode.DUMPING else 4.0 if mode is OperatingMode.HAULING else 0.0)
            + 5.0 * stress_ratio,
            0.0,
            100.0,
        )
        hoist_angle = _clamp(body_position * 0.34, 0.0, 38.0)
        chassis_pitch = pitch
        rack = _clamp(frame.twist_deg + roll * 0.08, -6.0, 6.0)
        coolant_condition = _clamp(
            98.0
            - max(0.0, engine.coolant_outlet_temp_c - 92.0) * 0.45
            - stress_ratio * 5.0
            + _noise(rng, 0.4),
            0.0,
            100.0,
        )
        frames.append(
            TelemetryFrame(
                timestamp=cfg.start_time + timedelta(seconds=step * cfg.timestep_s),
                step=step,
                elapsed_s=_bounded_round(step * cfg.timestep_s, 3),
                scenario=cfg.scenario,
                mode=mode,
                payload_class=payload.payload_class,
                payload_mass_t=payload.mass_t,
                rated_capacity_t=payload.rated_capacity_t,
                payload_utilization_pct=payload.utilization_pct,
                stress_level_pct=_bounded_round(stress_pct, 2),
                payload=payload,
                faults=faults,
                imu=imu,
                engine=engine,
                transmission=transmission,
                brake=brake,
                suspension=suspension,
                frame=frame,
                gnss=gnss,
                lidar=lidar,
                radar=radar,
                dss=dss,
                engine_health=health,
                overhaul_progress_pct=_bounded_round(overhaul_pct, 2),
                body_position_pct=_bounded_round(body_position, 2),
                hoist_angle_deg=_bounded_round(hoist_angle, 2),
                chassis_pitch_deg=_bounded_round(chassis_pitch, 2),
                rack_deg=_bounded_round(rack, 3),
                engine_load_pct=payload.utilization_pct,
                drive_cycle=cfg.drive_cycle,
                start_condition=cfg.start_condition,
                cycle_phase=cycle_phase,
                cycle_source=cfg.cycle_source,
                cycle_version=cfg.cycle_version,
                cumulative_fuel_energy_kwh=_bounded_round(cumulative_fuel_energy_kwh, 4),
                cumulative_fuel_volume_l=_bounded_round(cumulative_fuel_volume_l, 4),
                coolant_condition_pct=_bounded_round(coolant_condition, 2),
            )
        )

    return Journey(frames=frames, config=cfg, payload_profile=profile)


def generate_telemetry(*args: Any, **kwargs: Any) -> list[TelemetryFrame]:
    """List-returning convenience wrapper for dataframe/dashboard code."""

    return simulate_journey(*args, **kwargs).frames


def simulate_telemetry(*args: Any, **kwargs: Any) -> list[TelemetryFrame]:
    """Alias for :func:`generate_telemetry`."""

    return generate_telemetry(*args, **kwargs)


def run_simulation(*args: Any, **kwargs: Any) -> Journey:
    """Alias retaining the terminology used by some Streamlit prototypes."""

    return simulate_journey(*args, **kwargs)


def simulate_dump_truck(*args: Any, **kwargs: Any) -> Journey:
    """Explicitly named alias for applications with multiple simulators."""

    return simulate_journey(*args, **kwargs)
