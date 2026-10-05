"""Operational threshold bands for the dashboard's machine indicators.

The reference ranges intentionally leave some values unspecified (for example,
engine block temperature from 78 to 82).  :func:`classify` treats those gaps as
``"amber"`` so an uncovered reading is never reported as healthy.

The public :data:`RANGE_TABLE` maps canonical snake-case indicator names to
tuples of :class:`RangeBand` objects.  The table stores only the bands shown in
the reference material; the classifier supplies the conservative amber default
for values outside those bands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal


Status = Literal["green", "amber", "red"]


@dataclass(frozen=True)
class RangeBand:
    """One explicitly defined operational band.

    ``None`` denotes an open-ended bound.  Bounds are configurable so shared
    threshold endpoints can be assigned to the caution band without relying on
    floating-point epsilon values.
    """

    status: Status
    minimum: float | None = None
    maximum: float | None = None
    include_minimum: bool = True
    include_maximum: bool = True

    def contains(self, value: float) -> bool:
        """Return whether *value* is inside this band."""

        if self.minimum is not None:
            if value < self.minimum or (value == self.minimum and not self.include_minimum):
                return False
        if self.maximum is not None:
            if value > self.maximum or (value == self.maximum and not self.include_maximum):
                return False
        return True


# For ordinary three-band gauges, a shared green/caution boundary belongs to
# caution.  The red bounds follow the reference's strict ``<`` and ``>``
# notation.  Values in an unlisted gap are handled as amber by classify().
RANGE_TABLE: Final[dict[str, tuple[RangeBand, ...]]] = {
    "engine_block_temperature": (
        RangeBand("green", 82.0, 101.0, include_maximum=False),
        RangeBand("amber", 101.0, 108.0),
        RangeBand("red", maximum=78.0, include_maximum=False),
        RangeBand("red", minimum=108.0, include_minimum=False),
    ),
    "engine_oil_temperature": (
        RangeBand("green", 86.0, 100.0, include_maximum=False),
        RangeBand("amber", 100.0, 106.0),
        RangeBand("red", minimum=106.0, include_minimum=False),
    ),
    "exhaust_temperature": (
        RangeBand("green", 340.0, 555.0, include_maximum=False),
        RangeBand("amber", 555.0, 615.0),
        RangeBand("red", minimum=615.0, include_minimum=False),
    ),
    "engine_oil_pressure": (
        RangeBand("green", 459.0, 512.0, include_minimum=False, include_maximum=False),
        RangeBand("amber", 440.0, 459.0),
        RangeBand("amber", 512.0, 525.0),
        RangeBand("red", maximum=440.0, include_maximum=False),
        RangeBand("red", minimum=525.0, include_minimum=False),
    ),
    "engine_block_vibration": (
        RangeBand("green", 2.0, 6.3, include_maximum=False),
        RangeBand("amber", 6.3, 7.5),
        RangeBand("red", minimum=7.5, include_minimum=False),
    ),
    "gearbox_oil_temperature": (
        RangeBand("green", 67.0, 89.0, include_maximum=False),
        RangeBand("amber", 89.0, 96.0),
        RangeBand("red", minimum=96.0, include_minimum=False),
    ),
    "transmission_oil_pressure": (
        RangeBand("green", 329.0, 372.0, include_minimum=False, include_maximum=False),
        RangeBand("amber", 310.0, 329.0),
        RangeBand("amber", 372.0, 390.0),
        RangeBand("red", maximum=310.0, include_maximum=False),
        RangeBand("red", minimum=390.0, include_minimum=False),
    ),
    "gearbox_vibration": (
        RangeBand("green", 1.4, 5.0, include_maximum=False),
        RangeBand("amber", 5.0, 6.0),
        RangeBand("red", minimum=6.0, include_minimum=False),
    ),
    "hydraulic_oil_condition": (
        RangeBand("green", minimum=90.0),
        RangeBand("amber", 80.0, 90.0, include_maximum=False),
        RangeBand("red", maximum=80.0, include_maximum=False),
    ),
    "hydraulic_actuator_vibration": (
        RangeBand("green", 1.6, 5.3, include_maximum=False),
        RangeBand("amber", 5.3, 6.3),
        RangeBand("red", minimum=6.3, include_minimum=False),
    ),
    "coolant_pump_temperature": (
        RangeBand("green", 78.0, 92.0, include_maximum=False),
        RangeBand("amber", 92.0, 98.0),
        RangeBand("red", minimum=98.0, include_minimum=False),
    ),
    "coolant_condition": (
        RangeBand("green", minimum=90.0),
        RangeBand("amber", 80.0, 90.0, include_maximum=False),
        RangeBand("red", maximum=80.0, include_maximum=False),
    ),
}


# A descriptive alias is convenient for callers that do not want to depend on
# the shorter table name.
OPERATIONAL_RANGES = RANGE_TABLE
OPERATIONAL_RANGE_TABLE = RANGE_TABLE
RANGES = RANGE_TABLE


_ALIASES: Final[dict[str, str]] = {
    "engine_block_temp": "engine_block_temperature",
    "engine_block_temperature": "engine_block_temperature",
    "engine_oil_temp": "engine_oil_temperature",
    "engine_oil_temperature": "engine_oil_temperature",
    "exhaust_temp": "exhaust_temperature",
    "exhaust_temperature": "exhaust_temperature",
    "engine_oil_pressure": "engine_oil_pressure",
    "engine_block_vibration": "engine_block_vibration",
    "gearbox_oil_temp": "gearbox_oil_temperature",
    "gearbox_oil_temperature": "gearbox_oil_temperature",
    "transmission_oil_pressure": "transmission_oil_pressure",
    "gearbox_vibration": "gearbox_vibration",
    "hydraulic_oil_condition": "hydraulic_oil_condition",
    "hydraulic_actuator_vibration": "hydraulic_actuator_vibration",
    "coolant_pump_temp": "coolant_pump_temperature",
    "coolant_pump_temperature": "coolant_pump_temperature",
    "coolant_condition": "coolant_condition",
}


def _canonical_name(name: str) -> str:
    normalized = "_".join(str(name).strip().lower().replace("-", "_").split())
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(RANGE_TABLE)
        raise ValueError(f"unknown operational range {name!r}; choose one of {choices}") from exc


def classify(name: str, value: float) -> Status:
    """Classify a reading as ``green``, ``amber`` or ``red``.

    Values not covered by an explicitly listed band, including finite values in
    an unlisted gap, are conservatively classified as ``amber``.  At shared
    endpoints the table assigns the boundary to caution where the reference
    lists both green and caution; condition indicators retain their explicit
    ``>= 90`` green rule.
    """

    canonical = _canonical_name(name)
    try:
        reading = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value for {name!r} must be numeric") from exc

    # NaN is not inside any mathematical interval and therefore follows the
    # same conservative fallback as any other uncovered value.
    if math.isnan(reading):
        return "amber"
    for band in RANGE_TABLE[canonical]:
        if band.contains(reading):
            return band.status
    return "amber"


__all__ = [
    "OPERATIONAL_RANGES",
    "OPERATIONAL_RANGE_TABLE",
    "RANGE_TABLE",
    "RANGES",
    "RangeBand",
    "classify",
]
