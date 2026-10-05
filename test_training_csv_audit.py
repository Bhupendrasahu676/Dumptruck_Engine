"""Pure auditor regressions plus a lazy small unified-generator smoke test."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from audit_training_csv import KINDS, audit_training_csv, main, _signature


class TrainingCsvAuditTests(unittest.TestCase):
    def test_coolant_degradation_screen_does_not_hide_red_pump_in_heavy_context(self):
        values = {"engine.block_temp_c": 90.74, "engine.oil_temp_c": 89.94,
                  "engine.coolant_pump_temp_c": 114.47, "coolant_condition_pct": 54.90,
                  "frame.strain_microstrain": 972.4, "engine.load_pct": 10,
                  "context.ambient_c": 25}
        matched, thresholds = _signature("coolant_degradation", values, {})
        self.assertTrue(matched)
        self.assertEqual(thresholds["pump_temp_c"], 98)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "training.csv"
        self.raw = ["engine.oil_pressure_kpa", "engine.block_vibration_mm_s", "engine.oil_condition_pct"]
        self.mask = "engine.oil_pressure_kpa__available_mask"
        self.targets = [f"fault_label.{kind}" for kind in KINDS]
        self.metadata = ["engine_id", "episode_id", "split", "step", "elapsed_s", "timestamp", "scenario", "payload_class", "start_condition", "payload.mass_t"]
        self.labels = [*self.targets, *[f"fault_level.{kind}" for kind in KINDS], *[f"effect_level.{kind}" for kind in KINDS], "is_normal", "active_faults_json", "sensor_quality_json"]
        self.schema = {"columns": [*self.metadata, *self.raw, self.mask, *self.labels],
                       "metadata_columns": self.metadata, "input_columns": [*self.raw, self.mask],
                       "target_columns": self.targets, "label_columns": self.labels,
                       "diagnostic_columns": [], "units": {column: "synthetic" for column in self.raw},
                       "feature_definitions": {}, "fault_taxonomy": {kind: {} for kind in KINDS}}
        self.rows, self.plans = [], []
        for engine, split in (("a", "train"), ("b", "validation"), ("c", "test")):
            episode = engine + "-episode"
            config = {"episode_id": episode, "engine_id": engine, "duration_s": 20, "timestep_s": 10,
                      "scenario": "ideal", "payload_class": "heavy", "start_condition": "warm"}
            self.plans.append({"config": config, "events": [], "sample_count": 3})
            for step in range(3):
                label = int(step == 1)
                row = {"engine_id": engine, "episode_id": episode, "split": split, "step": step,
                       "elapsed_s": step * 10, "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step * 10)).isoformat(),
                       "scenario": "ideal", "payload_class": "heavy", "start_condition": "warm", "payload.mass_t": 72,
                       "engine.oil_pressure_kpa": 400 if label else 480, "engine.block_vibration_mm_s": 8 if label else 4,
                       "engine.oil_condition_pct": 96, self.mask: 1, "is_normal": 1 - label,
                       "active_faults_json": json.dumps([KINDS[0]] if label else []),
                       "sensor_quality_json": json.dumps({name: "good" for name in self.raw})}
                row.update({f"fault_label.{kind}": label if kind == KINDS[0] else 0 for kind in KINDS})
                row.update({f"fault_level.{kind}": .8 * label if kind == KINDS[0] else 0 for kind in KINDS})
                row.update({f"effect_level.{kind}": .8 * label if kind == KINDS[0] else 0 for kind in KINDS})
                self.rows.append(row)
        self.manifest = {"episodes": self.plans, "counts": {"rows": len(self.rows)},
                         "split_membership": {"train": ["a"], "validation": ["b"], "test": ["c"]}}

    def write(self):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.schema["columns"])
            writer.writeheader()
            writer.writerows(self.rows)
        self.path.with_suffix(".schema.json").write_text(json.dumps(self.schema), encoding="utf-8")
        self.path.with_suffix(".manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def audit(self):
        self.write()
        return audit_training_csv(self.path)

    def status(self, audit, name):
        return audit["checks"][name]["status"]

    def test_focused_contract_and_honest_limits(self):
        report = self.audit()
        for check in ("file_structure", "input_leakage", "numeric_inputs", "availability_masks", "sensor_quality_truth", "condition_bounds", "label_consistency", "unique_keys", "temporal_order", "cadence", "expected_row_count", "episode_invariants", "engine_disjoint_split"):
            with self.subTest(check=check):
                self.assertEqual(self.status(report, check), "pass")
        self.assertEqual(report["summary"]["rows"], 9)
        self.assertEqual(self.status(report, "split_class_coverage"), "fail")
        self.assertEqual(self.status(report, "required_source_availability"), "not_evaluated")
        self.assertTrue(all(value == "NOT validated" for value in report["validation_limits"].values()))
        json.dumps(report, allow_nan=False)

    def test_short_and_long_rows_are_rejected(self):
        self.write()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("short,row\n")
            handle.write(",".join(["extra"] * (len(self.schema["columns"]) + 1)) + "\n")
        report = audit_training_csv(self.path)
        self.assertEqual(self.status(report, "file_structure"), "fail")
        self.assertEqual(report["checks"]["file_structure"]["evidence"]["violations"], 2)
        self.assertEqual(report["summary"]["well_formed_rows"], 9)

    def test_duplicate_and_missing_headers(self):
        self.write()
        text = self.path.read_text()
        self.path.write_text(text.replace("engine_id,", "episode_id,", 1))
        report = audit_training_csv(self.path)
        self.assertEqual(self.status(report, "file_structure"), "fail")
        examples = report["checks"]["file_structure"]["evidence"]["examples"]
        self.assertTrue(any(item.get("duplicate_headers") for item in examples))
        self.assertTrue(any(item.get("missing_declared_columns") for item in examples))

    def test_nonfinite_and_nonnumeric_inputs(self):
        for value in ("nan", "inf", "-inf", "not-a-number"):
            with self.subTest(value=value):
                self.rows[0][self.raw[0]] = value
                report = self.audit()
                self.assertEqual(self.status(report, "numeric_inputs"), "fail")

    def test_missing_requires_explicit_marker(self):
        self.rows[0][self.raw[0]] = ""
        self.rows[0][self.mask] = 1
        self.assertEqual(self.status(self.audit(), "numeric_inputs"), "fail")
        self.rows[0][self.mask] = 0
        quality = {name: "good" for name in self.raw}
        quality[self.raw[0]] = "missing"
        self.rows[0]["sensor_quality_json"] = json.dumps(quality)
        report = self.audit()
        self.assertEqual(self.status(report, "numeric_inputs"), "pass")
        self.assertEqual(self.status(report, "sensor_quality_truth"), "pass")

    def test_inferred_validity_not_equivalent_to_presence(self):
        self.rows[0][self.mask] = 0  # finite but analytics classified invalid
        self.assertEqual(self.status(self.audit(), "availability_masks"), "pass")

    def test_presence_mask_must_match_source(self):
        column = "available.engine.oil_pressure_kpa"
        self.schema["columns"].append(column)
        self.schema["input_columns"].append(column)
        for row in self.rows:
            row[column] = 1
        self.rows[0][column] = 0
        self.assertEqual(self.status(self.audit(), "availability_masks"), "fail")

    def test_mask_binary_and_required_source_availability(self):
        self.rows[0][self.mask] = 2
        self.assertEqual(self.status(self.audit(), "availability_masks"), "fail")
        self.rows[0][self.mask] = 1
        self.rows[0][self.raw[0]] = ""
        self.schema["feature_definitions"][self.raw[1]] = {"required_sources": [self.raw[0]], "validity_mask": self.mask}
        self.assertEqual(self.status(self.audit(), "required_source_availability"), "fail")

    def test_leaking_inputs_are_rejected_without_rejecting_readings(self):
        for name in ("episode_id", "elapsed_s", "scenario", "fault_level.engine_oil_pump_wear", "sensor_quality_json", "overall_health", "rul_hours", "diagnostic.root_cause", "diagnostic.recommended_action", "diagnostic.confidence"):
            with self.subTest(column=name):
                self.schema["input_columns"] = [*self.raw, self.mask, name]
                self.assertEqual(self.status(self.audit(), "input_leakage"), "fail")
        self.schema["input_columns"] = [*self.raw, self.mask]
        self.assertEqual(self.status(self.audit(), "input_leakage"), "pass")

    def test_invalid_labels_and_threshold_consistency(self):
        self.rows[1][self.targets[0]] = 2
        self.assertEqual(self.status(self.audit(), "label_consistency"), "fail")
        self.rows[1][self.targets[0]] = 1
        self.rows[1]["effect_level." + KINDS[0]] = .001
        self.assertEqual(self.status(self.audit(), "label_consistency"), "fail")
        self.rows[1][self.targets[0]] = 0
        self.rows[1]["is_normal"] = 1
        self.assertEqual(self.status(self.audit(), "label_consistency"), "pass")

    def test_invalid_severity_and_condition_bounds(self):
        self.rows[0]["effect_level." + KINDS[0]] = "nan"
        self.rows[0]["engine.oil_condition_pct"] = 101
        report = self.audit()
        self.assertEqual(self.status(report, "label_consistency"), "fail")
        self.assertEqual(self.status(report, "condition_bounds"), "fail")

    def test_duplicate_episode_step_and_numeric_elapsed_keys(self):
        self.rows[1]["step"] = "0.0"
        self.assertEqual(self.status(self.audit(), "unique_keys"), "fail")
        self.rows[1]["step"] = 1
        self.schema["columns"].remove("step")
        self.schema["metadata_columns"].remove("step")
        for row in self.rows:
            row.pop("step")
        self.rows[1]["elapsed_s"] = "0.000"
        self.assertEqual(self.status(self.audit(), "unique_keys"), "fail")

    def test_timestamp_order_and_cadence(self):
        self.rows[1]["timestamp"] = self.rows[0]["timestamp"]
        report = self.audit()
        self.assertEqual(self.status(report, "temporal_order"), "fail")
        self.assertEqual(self.status(report, "cadence"), "fail")

    def test_fixed_payload_and_engine_split(self):
        self.rows[1]["payload.mass_t"] = 80
        self.rows[3]["engine_id"] = "a"
        report = self.audit()
        self.assertEqual(self.status(report, "episode_invariants"), "fail")
        self.assertEqual(self.status(report, "engine_disjoint_split"), "fail")

    def test_alias_config_and_table_counts(self):
        self.manifest = {"episode_configs": {plan["config"]["episode_id"]: {"configuration": {"config": plan["config"]}, "sample_count": 3} for plan in self.plans},
                         "table_counts": {self.path.name: {"row_count": 9}}, "engine_splits": {"a": "train", "b": "validation", "c": "test"}}
        report = self.audit()
        self.assertEqual(self.status(report, "cadence"), "pass")
        self.assertEqual(self.status(report, "expected_row_count"), "pass")
        self.assertEqual(self.status(report, "engine_disjoint_split"), "pass")

    def test_manifest_count_mismatch_and_absent_evidence(self):
        self.manifest["counts"]["rows"] = 99
        self.assertEqual(self.status(self.audit(), "expected_row_count"), "fail")
        self.manifest.pop("counts")
        self.assertEqual(self.status(self.audit(), "expected_row_count"), "not_evaluated")
        self.write()
        self.path.with_suffix(".schema.json").unlink()
        self.path.with_suffix(".manifest.json").unlink()
        report = audit_training_csv(self.path)
        self.assertEqual(self.status(report, "input_leakage"), "not_evaluated")
        self.assertEqual(self.status(report, "source_hashes"), "not_evaluated")

    def test_damaged_recovery_uses_effect_not_schedule_label(self):
        row = self.rows[2]
        row[self.targets[0]] = 1
        row["is_normal"] = 0
        row["effect_level." + KINDS[0]] = .2
        # Active fault JSON is undelayed schedule truth, hence remains [].
        report = self.audit()
        self.assertEqual(self.status(report, "label_consistency"), "pass")
        self.assertEqual(report["checks"]["recovery_with_damage"]["evidence"]["rows_by_class"][KINDS[0]], 1)

    def test_matrix_rates_are_partial_for_developing_states(self):
        self.rows[1][self.raw[0]] = 470
        self.rows[1]["effect_level." + KINDS[0]] = .1
        report = self.audit()
        item = report["checks"]["matrix_signatures"]["evidence"]["per_class"][KINDS[0]]
        self.assertEqual(item["status"], "partial")
        self.assertEqual(item["positive_rows"], 3)
        self.assertEqual(item["matched_rows"], 2)
        self.assertAlmostEqual(item["match_rate"], 2 / 3)
        self.assertEqual(item["developed_rows"], 2)

    def test_every_class_gets_temporal_check(self):
        kind = KINDS[-1]
        for row in self.rows[:3]:
            row["fault_label." + kind] = 1
            row["effect_level." + kind] = .5
            row["is_normal"] = 0
        report = self.audit()
        temporal = report["checks"]["temporal_class_variation"]
        self.assertEqual(temporal["status"], "fail")
        self.assertEqual(set(temporal["evidence"]["per_class"]), set(KINDS))
        self.assertTrue(any(item["class"] == kind for item in temporal["evidence"]["examples"]))

    def test_row_quality_and_metadata_sensor_modes(self):
        self.plans[0]["events"] = [{"kind": "sensor_fault", "sensor_mode": mode} for mode in ("biased", "missing", "frozen", "drifting", "noisy")]
        self.rows[0]["sensor_quality_json"] = json.dumps({self.raw[0]: "frozen"})
        report = self.audit()
        self.assertEqual(self.status(report, "sensor_modes"), "pass")
        self.assertEqual(report["checks"]["sensor_modes"]["evidence"]["row_quality_counts"]["frozen"], 1)
        self.rows[0]["sensor_quality_json"] = json.dumps({self.raw[0]: "missing"})
        self.assertEqual(self.status(self.audit(), "sensor_quality_truth"), "fail")

    def test_source_and_dataset_hashes(self):
        source = self.root / "source.py"
        source.write_text("value = 3\n")
        self.manifest["source_hashes"] = {"source.py": hashlib.sha256(source.read_bytes()).hexdigest()}
        self.write()
        self.manifest["files"] = {self.path.name: {"sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}}
        self.path.with_suffix(".manifest.json").write_text(json.dumps(self.manifest))
        report = audit_training_csv(self.path)
        self.assertEqual(self.status(report, "source_hashes"), "pass")
        self.assertEqual(self.status(report, "dataset_hash"), "pass")
        source.write_text("value = 4\n")
        self.manifest["files"][self.path.name]["sha256"] = "0" * 64
        self.path.with_suffix(".manifest.json").write_text(json.dumps(self.manifest))
        report = audit_training_csv(self.path)
        self.assertEqual(self.status(report, "source_hashes"), "fail")
        self.assertEqual(self.status(report, "dataset_hash"), "fail")

    def test_reports_and_cli(self):
        self.write()
        prefix = self.root / "report"
        audit = audit_training_csv(self.path, output_prefix=prefix)
        saved = json.loads(prefix.with_suffix(".json").read_text())
        self.assertEqual(saved, json.loads(json.dumps(audit)))
        self.assertIn("NOT validated", prefix.with_suffix(".md").read_text())
        self.assertEqual(main(["--csv", str(self.path), "--output-prefix", str(prefix)]), 1)

    def test_old_observation_and_metadata_prefixes(self):
        renames = {name: "observation." + name for name in self.raw}
        renames.update({name: "metadata." + name for name in self.metadata})
        for key in ("columns", "metadata_columns", "input_columns"):
            self.schema[key] = [renames.get(name, name) for name in self.schema[key]]
        self.rows = [{renames.get(name, name): value for name, value in row.items()} for row in self.rows]
        report = self.audit()
        for check in ("input_leakage", "unique_keys", "cadence", "episode_invariants", "numeric_inputs"):
            self.assertEqual(self.status(report, check), "pass")

    def test_empty_csv_and_invalid_sidecar(self):
        self.write()
        self.path.write_text("")
        self.path.with_suffix(".schema.json").write_text("not JSON")
        report = audit_training_csv(self.path)
        self.assertEqual(self.status(report, "file_structure"), "fail")
        self.assertEqual(self.status(report, "schema_sidecar"), "fail")
        json.dumps(report, allow_nan=False)

    def test_documented_temporal_unavailability_is_not_unexplained_missing(self):
        rate = self.raw[0] + "__rate"
        leadlag = "leadlag.engine.oil_pressure_kpa__to__engine.oil_temp_c_s"
        for name in (rate, leadlag):
            self.schema["columns"].append(name)
            self.schema["input_columns"].append(name)
            self.schema["feature_definitions"][name] = {"definition": "Observation-derived temporal feature", "units": "s"}
        self.schema["missingness_handling"] = {"csv_representation": "blank numeric cells"}
        for row in self.rows:
            row[rate] = "" if row["step"] == 0 else -8
            row[leadlag] = ""
        report = self.audit()
        self.assertEqual(self.status(report, "numeric_inputs"), "pass")
        reasons = report["checks"]["numeric_inputs"]["evidence"]["documented_unavailable_feature_cells"]
        self.assertEqual(reasons["episode_initial_history_unavailable"], 3)
        self.assertEqual(reasons["documented_onset_association_unavailable"], 9)
        self.rows[1][rate] = ""
        self.assertEqual(self.status(self.audit(), "numeric_inputs"), "fail")

    def test_derived_reading_cannot_claim_available_with_missing_source(self):
        residual = self.raw[0] + "__residual"
        self.schema["columns"].append(residual)
        self.schema["input_columns"].append(residual)
        for row in self.rows:
            row[residual] = 2
        self.rows[0][self.raw[0]] = ""
        self.rows[0][self.mask] = 0
        quality = {name: "good" for name in self.raw}
        quality[self.raw[0]] = "missing"
        self.rows[0]["sensor_quality_json"] = json.dumps(quality)
        report = self.audit()
        self.assertEqual(self.status(report, "numeric_inputs"), "pass")
        self.assertEqual(self.status(report, "required_source_availability"), "fail")

    def test_hydraulic_slow_and_fast_rates_are_assessed_separately(self):
        condition = "suspension.hydraulic_oil_condition_pct"
        vibration = "suspension.actuator_vibration_mm_s"
        for name in (condition, vibration):
            self.schema["columns"].append(name)
            self.schema["input_columns"].append(name)
        for row in self.rows:
            row[condition], row[vibration] = 99, 8
        for selected, kind, rate in ((self.rows[:3], "hydraulic_normal_aging", .002), (self.rows[3:6], "hydraulic_contamination", .05)):
            for row in selected:
                row[condition] = 99 - rate * row["elapsed_s"]
                row["fault_label." + kind] = 1
                row["effect_level." + kind] = .5
                row["is_normal"] = 0
        report = self.audit()
        matrix = report["checks"]["matrix_signatures"]["evidence"]["per_class"]
        for kind, rate in (("hydraulic_normal_aging", .002), ("hydraulic_contamination", .05)):
            with self.subTest(kind=kind):
                self.assertEqual(matrix[kind]["matched_episodes"], 1)
                self.assertAlmostEqual(matrix[kind]["mean_condition_drop_pct_per_s"], rate)
                self.assertEqual(matrix[kind]["status"], "pass")

    def test_default_171_sample_1800_second_rounding(self):
        first = self.rows[0]
        self.rows = []
        self.manifest = {"counts": {"rows": 171}}
        for step in range(171):
            elapsed = step * 1800 / 170
            row = dict(first)
            row.update(step=step, elapsed_s=round(elapsed, 3), timestamp=(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=round(elapsed))).isoformat())
            self.rows.append(row)
        report = self.audit()
        self.assertEqual(self.status(report, "cadence"), "pass")
        self.assertEqual(self.status(report, "temporal_order"), "pass")
        self.rows[-1]["elapsed_s"] = 1790
        self.assertEqual(self.status(self.audit(), "cadence"), "fail")

    def test_pair_and_triple_counts_use_observed_labels_and_planned_events(self):
        kinds = KINDS[:3]
        row = self.rows[1]
        for kind in kinds:
            row["fault_label." + kind] = 1
            row["effect_level." + kind] = .5
        self.plans[0]["events"] = [{"kind": kind} for kind in kinds]
        report = self.audit()
        evidence = report["checks"]["fault_combinations"]["evidence"]
        self.assertEqual(evidence["distinct_pairs"], 3)
        self.assertEqual(evidence["distinct_triples"], 1)
        self.assertEqual(evidence["planned_distinct_pairs"], 3)
        self.assertEqual(evidence["planned_distinct_triples"], 1)


class UnifiedGeneratorAuditSmokeTest(unittest.TestCase):
    def test_small_real_generator_contract(self):
        # Import only here: pure auditor tests require no simulation dependencies.
        from generate_training_csv import generate_training_csv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            receipt = generate_training_csv(path, engines=3, episodes_per_fault=1,
                                            samples_per_journey=21, duration_s=300, seed=793)
            report = audit_training_csv(path, output_prefix=Path(directory) / "audit")
            self.assertEqual(report["summary"]["rows"], receipt["rows"])
            for check in ("file_structure", "schema_contract", "input_leakage", "numeric_inputs", "availability_masks", "sensor_quality_truth", "condition_bounds", "label_consistency", "unique_keys", "temporal_order", "cadence", "expected_row_count", "episode_invariants", "engine_disjoint_split", "dataset_hash", "source_hashes"):
                with self.subTest(check=check):
                    self.assertEqual(report["checks"][check]["status"], "pass", report["checks"][check])
            self.assertEqual(report["summary"]["episodes"], 93)
            self.assertEqual(report["checks"]["sensor_modes"]["status"], "pass")
            self.assertEqual(len(report["checks"]["temporal_class_variation"]["evidence"]["per_class"]), 16)
            # Three trucks deliberately do not cover the entire 30-context grid.
            self.assertEqual(report["checks"]["per_class_contexts"]["status"], "fail")
            self.assertEqual(report["checks"]["matrix_signatures"]["status"], "partial")


if __name__ == "__main__":
    unittest.main()
