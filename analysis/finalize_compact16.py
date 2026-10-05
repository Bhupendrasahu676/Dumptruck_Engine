"""Finalize an interrupted compact16 export from existing frozen weights only."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
from analysis.train_compact16 import FIXED_FEATURES, DEFAULT_CSV, DEFAULT_FEATURES, predict_compact_bundle, DEPENDENCY_WARNING
from analysis.train_fault_models import digest, markdown_table
from analysis import fault_dl as dl


def finalize(output):
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed run already has a manifest; verify it instead of finalizing again")
    frozen = json.loads((output / "frozen_policy.json").read_text())
    schema = json.loads((output / "feature_schema.json").read_text())
    assert frozen["input_columns"] == schema["input_columns"] == list(FIXED_FEATURES)
    assert not frozen["test_features_loaded"] and not frozen["test_predictions_computed"]
    source = DEFAULT_CSV
    assert digest(source) == frozen["source_sha256"]
    source_manifest = json.loads(source.with_suffix(".manifest.json").read_text())
    features, targets = list(FIXED_FEATURES), frozen["target_columns"]
    frame = pd.read_csv(source, usecols=features + ["split", "engine_id", "episode_id", "step"],
                        dtype={n: "float32" for n in features})
    te = frame[frame.split.eq("test")].sort_values(["engine_id", "episode_id", "step"]).reset_index(drop=True)
    first = te[te.episode_id.eq(te.episode_id.iloc[0])]
    counts = frame.split.value_counts().to_dict()
    selected_predictions = pd.read_csv(output / "test_predictions.csv")
    expected = selected_predictions[["probability." + t.split(".", 1)[1] for t in targets]].to_numpy()
    checks = []
    for filename in ("dense_model.joblib", "tcn_model.joblib", "selected_model.joblib", "selected_maxf1_model.joblib"):
        bundle = joblib.load(output / filename)
        kind = bundle["kind"]
        assert bundle["input_columns"] == features
        assert bundle["imputer"].n_features_in_ == bundle["scaler"].n_features_in_ == 16
        assert bundle["train_rows"] == counts["train"] == 114912
        assert digest(output / bundle["model_path"]) == frozen["candidates"][kind]["checkpoint_sha256"]
        if filename in ("dense_model.joblib", "tcn_model.joblib"):
            assert digest(output / filename) == frozen["candidates"][kind]["bundle_sha256"]
        checkpoint = dl.load_model(output / bundle["model_path"])
        full = predict_compact_bundle(output / filename, te, te.episode_id.to_numpy())
        if kind == frozen["selected_kind"]:
            np.testing.assert_allclose(full, expected, rtol=1e-5, atol=1e-6)
        prefix = predict_compact_bundle(output / filename, first, first.episode_id.to_numpy())
        error = float(np.max(np.abs(prefix - full[:len(first)])))
        flips = int(((prefix >= bundle["thresholds"]) != (full[:len(first)] >= bundle["thresholds"])).sum())
        checks.append({"bundle": filename, "kind": kind, "full_test_rows": len(te), "passed": True,
                       "first_episode_rows": len(first), "prefix_batch_size_probability_difference": error,
                       "prefix_batch_size_decision_flips": flips,
                       "full_test_probability_matches_saved_selected": kind == frozen["selected_kind"]})
    receipt = {"serialization_reload": checks, "frozen_policy_sha256": digest(output / "frozen_policy.json"),
               "frozen_before_test_features": True, "model_inputs": 16, "trained_on_all_train_rows": counts["train"],
               "no_test_selection_or_calibration": True, "source_sha256_verified": True, "deployment_ready": False,
               "export_recovery": "Existing checkpoints/predictions used; no retraining, weight changes or threshold refitting",
               "numerical_note": "CUDA convolution can vary slightly with batch shape; same-batch checkpoint reload reproduces selected full-test probabilities"}
    (output / "verification_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    comparison = pd.read_csv(output / "model_comparison.csv")
    selected = comparison[(comparison.candidate.eq(frozen["selected_kind"])) & comparison.split.eq("test")]
    report = ["# Fixed compact-16 training results", "", f"Validation-selected model: **{frozen['selected_kind']}**, exactly **16 model input columns**.",
              "No PCA, feature reselection, added context inputs or autoencoder; both models trained on all 114,912 training rows.", "",
              markdown_table(comparison[["candidate", "split", "policy", "macro_AP", "macro_F1", "macro_recall", "normal_false_alarm_rate"]]),
              "", "## Interpretation", "Average precision is not accuracy. The primary guarded thresholds were frozen on validation before test prediction.",
              "The original held-truck benchmark was used in prior studies; these results are comparative synthetic evidence, not a newly untouched field test.",
              DEPENDENCY_WARNING, "Slow aging and ambiguous sensor faults may remain weak. All bundles are deployment_ready=False.",
              "No health/RUL training or real root-cause confirmation. Inspect per_fault_metrics.csv and normal_false_alarms.csv, not aggregate scores alone.",
              "", "## Export verification", "Training finished, but a strict single-episode serialization assertion found CUDA batch-size rounding differences.",
              "The finalizer reused existing frozen weights and predictions; full-test reload matched and the prefix differences/decision flips are recorded.",
              "No training or threshold selection was repeated. verification_receipt.json records recovery and numerical evidence.",
              "", "## Artifacts", "selected_model.joblib is the guarded primary; selected_maxf1_model.joblib is the alternate. Candidate bundles and native checkpoints are also saved.",
              "feature_schema.json and selected_attributes.csv give the exact input order. For portable loading use analysis.train_compact16.predict_compact_bundle.",
              "", "## Reproduce", "```bash", "OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/opencode/fault-training/bin/python analysis/train_compact16.py --output analysis/results_compact16_new --epochs 18 --seed 793", "```"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    model_metadata = {kind: {"best_epoch": joblib.load(output / (kind + "_model.joblib"))["best_epoch"],
                             "epochs_completed": joblib.load(output / (kind + "_model.joblib"))["epochs_completed"],
                             "device": joblib.load(output / (kind + "_model.joblib"))["device"],
                             "selection_macro_AP": frozen["candidates"][kind]["selection_macro_AP"]}
                      for kind in ("dense", "tcn")}
    manifest = {"source_csv": str(source.resolve()), "source_sha256": frozen["source_sha256"],
                "source_schema_sha256": digest(source.with_suffix(".schema.json")), "source_manifest_sha256": digest(source.with_suffix(".manifest.json")),
                "source_selected_attributes": str(DEFAULT_FEATURES.resolve()), "source_features_sha256": digest(DEFAULT_FEATURES),
                "seed": 793, "epochs_budget": 18, "input_columns": features, "target_columns": targets,
                "selected_model": frozen["selected_kind"], "primary_policy": "guarded", "alternate_policy": "maxf1",
                "split_membership": source_manifest["split_membership"], "train_rows": counts["train"],
                "validation_tuning_rows": 16416, "calibration_rows": 8208, "test_rows": counts["test"],
                "calibration_truck": frozen["calibration_truck"], "selection_strategy": frozen["selection_metric"],
                "models": model_metadata, "verification": receipt,
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": dl.torch.__version__},
                "code_sha256": {str(f.relative_to(ROOT)): digest(f) for f in (Path(__file__), ROOT / "analysis/train_compact16.py", ROOT / "analysis/fault_dl.py")},
                "code_revision_note": "ISO8601 timestamp parsing corrected before fitting; reload check corrected after training; checkpoint weights unchanged",
                "health": "not_trained", "RUL": "unavailable_not_validated", "deployment_ready": False,
                "artifacts": {str(f.relative_to(output)): digest(f) for f in output.rglob("*") if f.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest_saved": True, "training_repeated": False, "verification": receipt,
            "selected_test_metrics": selected.to_dict("records")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/results_compact16_only")
    print(json.dumps(finalize(parser.parse_args().output), indent=2))
