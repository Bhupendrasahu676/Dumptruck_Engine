"""Behavioral checks for explicit telemetry export and idempotence."""

import json

from telemetry_export import (
    build_export_metadata,
    export_journey,
    render_csv,
    render_jsonl,
)
from telemetry_model import simulate_journey


def test_jsonl_and_csv_include_every_frame_and_manual_fault_metadata(tmp_path) -> None:
    journey = simulate_journey(seed=10, scenario="stress", num_steps=5)
    faults = {"overheat": True, "bearing_failure": False}
    metadata = build_export_metadata(
        journey.frames,
        seed=10,
        scenario="stress",
        payload_class=journey.payload_class.value,
        manual_faults=faults,
        drive_cycle="c-wtvc",
        start_condition="cold",
        cycle_source="research proxy",
        cycle_version="test",
    )

    json_lines = [json.loads(line) for line in render_jsonl(journey.frames, metadata).splitlines()]
    csv_lines = render_csv(journey.frames, metadata).splitlines()

    assert len(json_lines) == len(journey.frames)
    assert {record["record_id"] for record in json_lines} == {
        f"{metadata['journey_id']}:{step}" for step in range(len(journey.frames))
    }
    assert json_lines[0]["manual_faults"]["overheat"] is True
    assert json_lines[0]["drive_cycle"] == "c-wtvc"
    assert json_lines[0]["start_condition"] == "cold"
    assert len(csv_lines) == len(journey.frames) + 1
    assert "faults_json" in csv_lines[0]


def test_export_is_idempotent_for_same_deterministic_journey(tmp_path) -> None:
    journey = simulate_journey(seed=11, num_steps=4)
    kwargs = dict(
        export_dir=tmp_path,
        seed=11,
        scenario="ideal",
        payload_class=journey.payload_class.value,
        manual_faults={"overheat": False},
    )

    first = export_journey(journey.frames, **kwargs)
    first_json = first.jsonl_path.read_bytes()
    second = export_journey(journey.frames, **kwargs)

    assert first.already_existed is False
    assert second.already_existed is True
    assert second.jsonl_path.read_bytes() == first_json
    assert len(list(tmp_path.iterdir())) == 2
