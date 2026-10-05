"""Integration boundary: hidden simulator labels never enter inference."""
from copy import deepcopy
import csv
import io
import json
from uuid import uuid4

from .analytics import analyse_samples, assess_rul
from .simulation import simulate_test

SCHEMA_VERSION = "engine-lab/1"
CONTEXT_KEYS = ("elapsed_s", "rpm", "demand_pct", "start_condition", "ambient_c", "transient")
SIGNAL_KEYS = ("oil_pressure_kpa", "oil_temp_c", "coolant_temp_c", "coolant_flow_kg_s",
               "thermostat_open_pct", "vibration_mm_s", "egt_c")


def observation_input(sample):
    """Explicit allowlist, not deletion of known target names."""
    return {"elapsed_s": sample["elapsed_s"], "operating_hours": sample["operating_hours"],
            "context": {key: deepcopy(sample["context"].get(key)) for key in CONTEXT_KEYS},
            "observations": {key: deepcopy(sample["observations"].get(key)) for key in SIGNAL_KEYS}}


def preview_test(engine, settings, schedules):
    """Pure preview: no database write, no accumulated hours until explicit save."""
    config = dict(settings, engine_id=engine["engine_id"],
                  overhaul_id=engine["current_overhaul_id"], test_id=str(uuid4()),
                  initial_hours=engine["operating_hours"])
    result = simulate_test(config, schedules, engine["wear"])
    result["analyses"] = analyse_samples([observation_input(s) for s in result["samples"]])
    result["schema_version"] = SCHEMA_VERSION
    result["schedules"] = deepcopy(schedules)
    result["initial_wear"] = deepcopy(engine["wear"])
    result["provenance"] = {"source": "simulation", "physics_validated": False,
                            "diagnostics_validated": False, "aging": "accelerated lab heuristic"}
    return result


def rul_for_engine(store, engine, evidence=None):
    """Use only committed history in the current overhaul, bound to this engine."""
    history = [dict(row, engine_id=engine["engine_id"]) for row in store.health_history(engine["engine_id"])
               if row["overhaul_id"] == engine["current_overhaul_id"]]
    if evidence is not None and (not isinstance(evidence, dict)
                                 or evidence.get("current_engine_id") != engine["engine_id"]):
        return {"status": "insufficient_evidence", "reasons": ["Evidence must belong to the selected engine."],
                "estimate_hours": None, "interval_hours": None,
                "uncertainty_kind": "not_quantifiable_from_available_evidence"}
    # This workflow produces synthetic history. Never let field-labelled outcomes
    # make its simulated current prediction appear field-validated.
    if isinstance(evidence, dict) and evidence.get("source") != "simulation":
        return {"status": "insufficient_evidence", "reasons": ["This lab has synthetic history; only simulation-labelled evidence is applicable."],
                "estimate_hours": None, "interval_hours": None,
                "uncertainty_kind": "not_quantifiable_from_available_evidence"}
    if history and abs(history[-1]["operating_hours"] - engine["operating_hours"]) > 1e-8:
        return {"status": "insufficient_evidence", "reasons": ["Latest usable health history is stale relative to current engine hours; restore measurement quality and record a valid endpoint."],
                "estimate_hours": None, "interval_hours": None,
                "uncertainty_kind": "not_quantifiable_from_available_evidence"}
    return assess_rul(history, evidence)


def export_jsonl(result, layer):
    """Separate observations, predictions and hidden truth; never mixed inputs."""
    if layer not in ("observations", "predictions", "truth"):
        raise ValueError("Unknown export layer")
    ids = {key: result["config"][key] for key in ("engine_id", "overhaul_id", "test_id")}
    rows = []
    for index, sample in enumerate(result["samples"]):
        row = dict(ids, schema_version=SCHEMA_VERSION, source="simulation", sample_index=index,
                   elapsed_s=sample["elapsed_s"], operating_hours=sample["operating_hours"])
        if layer == "observations":
            row.update(observation_input(sample))
        elif layer == "predictions":
            row["prediction"] = result["analyses"][index]
        else:
            row["ground_truth_not_model_input"] = sample["truth"]
        rows.append(json.dumps(row, allow_nan=False, ensure_ascii=False))
    return "\n".join(rows) + "\n"


def export_observations_csv(result):
    stream = io.StringIO(newline="")
    columns = ["Engine_ID", "Overhaul_ID", "Test_ID", "sample_index", "elapsed_s", "operating_hours"]
    columns += ["context_" + key for key in CONTEXT_KEYS if key != "elapsed_s"]
    columns += list(SIGNAL_KEYS[:-1]) + [f"egt_c_{n + 1}" for n in range(6)]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for index, sample in enumerate(result["samples"]):
        clean = observation_input(sample)
        row = {label: result["config"][key] for label, key in
               (("Engine_ID", "engine_id"), ("Overhaul_ID", "overhaul_id"), ("Test_ID", "test_id"))}
        row.update(sample_index=index, elapsed_s=clean["elapsed_s"], operating_hours=clean["operating_hours"])
        row.update({"context_" + key: clean["context"][key] for key in CONTEXT_KEYS if key != "elapsed_s"})
        row.update({key: clean["observations"][key] for key in SIGNAL_KEYS[:-1]})
        row.update({f"egt_c_{n + 1}": value for n, value in enumerate(clean["observations"]["egt_c"])})
        writer.writerow(row)
    return stream.getvalue()
