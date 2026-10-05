"""Focused checks for the reference operational threshold bands."""

import pytest

from operational_ranges import RANGE_TABLE, classify


@pytest.mark.parametrize(
    ("name", "green", "amber", "red"),
    [
        ("engine_block_temperature", 82, 101, 109),
        ("engine_oil_temperature", 86, 100, 107),
        ("exhaust_temperature", 340, 555, 616),
        ("engine_oil_pressure", 480, 459, 439),
        ("engine_block_vibration", 2.0, 6.3, 7.6),
        ("gearbox_oil_temperature", 67, 89, 97),
        ("transmission_oil_pressure", 350, 329, 309),
        ("gearbox_vibration", 1.4, 5.0, 6.1),
        ("hydraulic_oil_condition", 90, 80, 79),
        ("hydraulic_actuator_vibration", 1.6, 5.3, 6.4),
        ("coolant_pump_temperature", 78, 92, 99),
        ("coolant_condition", 90, 80, 79),
    ],
)
def test_reference_bands(name: str, green: float, amber: float, red: float) -> None:
    assert classify(name, green) == "green"
    assert classify(name, amber) == "amber"
    assert classify(name, red) == "red"


def test_shared_pressure_boundaries_are_caution() -> None:
    assert classify("engine oil pressure", 459) == "amber"
    assert classify("engine_oil_pressure", 512) == "amber"
    assert classify("transmission-oil-pressure", 329) == "amber"
    assert classify("transmission_oil_pressure", 372) == "amber"


def test_uncovered_gap_defaults_to_caution() -> None:
    assert classify("engine_block_temp", 80) == "amber"
    assert classify("engine_oil_temp", 84) == "amber"
    assert classify("exhaust_temp", 300) == "amber"
    assert classify("engine_block_vibration", 1.0) == "amber"


def test_range_table_exposes_all_reference_indicators() -> None:
    assert len(RANGE_TABLE) == 12
    assert "coolant_condition" in RANGE_TABLE
    assert all(RANGE_TABLE[name] for name in RANGE_TABLE)


def test_unknown_names_and_non_numeric_values_raise() -> None:
    with pytest.raises(ValueError, match="unknown operational range"):
        classify("not-a-sensor", 1)
    with pytest.raises(ValueError, match="must be numeric"):
        classify("coolant_condition", "not-a-number")
