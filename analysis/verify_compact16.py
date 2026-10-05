"""Independent read-back checks of fixed-16 results and optional external data.

No model/threshold fitting or model selection is performed here. Joblib files
must be locally generated trusted artifacts; prediction uses embedded weights.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from analysis.train_compact16 import FIXED_FEATURES, predict_compact_bundle
from analysis.train_fault_models import digest, metrics


def verify(output, external_csv=None):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    frozen = json.loads((output / "frozen_policy.json").read_text())
    schema = json.loads((output / "feature_schema.json").read_text())
    inputs, targets = schema["input_columns"], schema["target_columns"]
    assert inputs == list(FIXED_FEATURES) and len(inputs) == 16
    assert not schema["additional_context_inputs"]
    assert manifest["verification"]["frozen_before_test_features"]
    assert not frozen["test_features_loaded"] and not frozen["test_predictions_computed"]
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    comparison = pd.read_csv(output / "model_comparison.csv")
    receipts = []
    for policy, filename in (("guarded", "selected_model.joblib"), ("maxf1", "selected_maxf1_model.joblib")):
        bundle = joblib.load(output / filename)
        assert bundle["input_columns"] == inputs and bundle["kind"] == frozen["selected_kind"]
        assert bundle["imputer"].n_features_in_ == bundle["scaler"].n_features_in_ == 16
        assert "pca" not in bundle and "autoencoder_path" not in bundle
        predicted = pd.read_csv(output / ("test_predictions_" + policy + ".csv"))
        faults = [t.split(".", 1)[1] for t in targets]
        y = predicted[["truth." + f for f in faults]].to_numpy(np.int8)
        p = predicted[["probability." + f for f in faults]].to_numpy()
        decisions = predicted[["prediction." + f for f in faults]].to_numpy(np.int8)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(decisions, p >= bundle["thresholds"])
        assert len(predicted) == manifest["test_rows"]
        assert set(predicted.engine_id) == set(manifest["split_membership"]["test"])
        assert not predicted.duplicated(["episode_id", "step"]).any()
        score, _ = metrics(y, p, bundle["thresholds"], targets, bundle["kind"], "test")
        row = comparison[(comparison.candidate.eq(bundle["kind"])) & comparison.split.eq("test") & comparison.policy.eq(policy)].iloc[0]
        for key in ("macro_AP", "macro_F1", "macro_recall", "micro_F1", "normal_false_alarm_rate"):
            np.testing.assert_allclose(score[key], row[key], rtol=1e-7, atol=1e-7)
        receipts.append({"policy": policy, "bundle": filename, "verified": True, "score": score})
    result = {"verified": True, "exact_model_input_count": 16, "input_columns": inputs,
              "artifact_hash_checks": len(manifest["artifacts"]), "prediction_checks": receipts,
              "no_weight_or_threshold_changes": True, "deployment_ready": False,
              "original_test_status": "Previously used synthetic benchmark, not newly untouched field data"}
    if external_csv is not None:
        external_csv = Path(external_csv)
        new_manifest = json.loads(external_csv.with_suffix(".manifest.json").read_text())
        original = json.loads(Path(manifest["source_csv"]).with_suffix(".manifest.json").read_text())
        old_seeds = {e["config"]["engine_seed"] for e in original["episodes"]}
        new_seeds = {e["config"]["engine_seed"] for e in new_manifest["episodes"]}
        assert not old_seeds & new_seeds
        assert digest(external_csv) == new_manifest["files"][external_csv.name]["sha256"]
        assert new_manifest["parameters"]["seed"] != original["parameters"]["seed"]
        data = pd.read_csv(external_csv, usecols=inputs + targets + ["engine_id", "episode_id", "step"],
                           dtype={c: "float32" for c in inputs + targets})
        data = data.sort_values(["engine_id", "episode_id", "step"], kind="stable").reset_index(drop=True)
        assert not data.duplicated(["episode_id", "step"]).any()
        bundle = joblib.load(output / "selected_model.joblib")
        p = predict_compact_bundle(bundle, data, data.episode_id.to_numpy(), base_dir=output)
        y = data[targets].to_numpy(np.int8)
        score, per_fault = metrics(y, p, bundle["thresholds"], targets, bundle["kind"], "external_existing_seed_no_tuning")
        pd.DataFrame(per_fault).to_csv(output / "external_per_fault_metrics.csv", index=False)
        rows = data[["engine_id", "episode_id", "step"]].copy()
        for k, target in enumerate(targets):
            fault = target.split(".", 1)[1]
            rows["truth." + fault] = y[:, k]
            rows["probability." + fault] = p[:, k]
            rows["prediction." + fault] = (p[:, k] >= bundle["thresholds"][k]).astype(np.int8)
        rows.to_csv(output / "external_predictions.csv", index=False)
        external = {"scores": score, "data_path": str(external_csv), "source_sha256": digest(external_csv),
                    "independent_variant_seeds": sorted(new_seeds), "model_input_count": 16,
                    "weights_or_thresholds_fit_on_batch": False,
                    "evidence_status": "Existing different-seed batch previously evaluated in another study; confirmatory cross-seed check, not a newly generated untouched experiment",
                    "engine_identity_note": "Truck names are batch-local; engine_seed sets establish independence",
                    "field_validation": False}
        (output / "EXTERNAL_EVALUATION.json").write_text(json.dumps(external, indent=2) + "\n")
        result["external_evaluation"] = external
    (output / "RESULTS_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/results_compact16_only")
    parser.add_argument("--external-csv", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.external_csv), indent=2))
