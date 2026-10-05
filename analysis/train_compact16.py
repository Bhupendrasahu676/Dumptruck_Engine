"""Fresh dense/TCN training on the frozen sixteen original observation columns.

No feature selection, PCA, autoencoder, health target or RUL target is used.
Test feature rows are read only after the model, calibration and both threshold
policies have been serialized in frozen_policy.json. Existing outputs are refused.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

from analysis import fault_dl as dl
from analysis.fault_features import allowed_input_names, sensor_dependencies
from analysis.train_fault_models import (digest, markdown_table, metrics,
                                        sigmoid_apply, sigmoid_fit, thresholds_for)

DEFAULT_CSV = ROOT / "datasets/dump_truck_enhanced_all_conditions.csv"
DEFAULT_FEATURES = ROOT / "analysis/results_enhanced_multilabel/selected_attributes.csv"
DEFAULT_OUTPUT = ROOT / "analysis/results_compact16_only"
FIXED_FEATURES = (
    "engine.oil_pressure_kpa__residual",
    "engine.oil_condition_pct__residual",
    "engine.boost_kpa__residual",
    "engine.coolant_flow_kg_s__residual",
    "coolant_condition_pct__residual",
    "transmission.oil_pressure_kpa__residual",
    "transmission.oil_condition_pct__residual",
    "gnss.longitude_deg",
    "suspension.hydraulic_oil_condition_pct__residual",
    "engine.oil_level_pct",
    "engine.coolant_temp_c__residual",
    "brake.stroke_pct__residual",
    "gnss.accuracy_m__residual",
    "gnss.latitude_deg",
    "transmission.vibration_mm_s__residual",
    "engine.oil_temp_c__residual",
)
METADATA = ["engine_id", "episode_id", "step", "elapsed_s", "timestamp", "split"]
POLICIES = {"maxf1": "exact_validation_max_F1",
            "guarded": "exact_validation_max_F1_with_1pct_per_label_normal_guard"}
DEPENDENCY_WARNING = (
    "Exactly sixteen model columns do not imply sixteen physical sensors. Residuals "
    "require healthy context priors, commissioning/calibration, source sensors, "
    "trailing observation state and episode resets; context is not added to the "
    "model input. GNSS longitude/latitude remain route proxies and GNSS accuracy "
    "and brake.stroke residual remain explicitly requested inputs. Brake stroke "
    "also depends on pedal and pad-wear measurements. No GNSS/brake ablation or "
    "reselection is performed."
)


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_fixed_features(path):
    """Validate an attribute CSV against the exact frozen order and allowlist.

    The built-in exporter observation allowlist is checked here; run additionally
    checks the particular source CSV schema. An alternate --features file must
    contain the very same sixteen names in the very same order.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["attribute"]:
            raise ValueError("Feature CSV must have exactly one header: attribute")
        rows = list(reader)
    names = [row["attribute"] for row in rows]
    if any(None in row for row in rows) or len(names) != 16 or len(set(names)) != 16:
        raise ValueError("Feature CSV must contain exactly sixteen unique attributes")
    if any(not isinstance(n, str) or not n or n not in allowed_input_names() for n in names):
        raise ValueError("Feature CSV contains names outside the observation schema allowlist")
    if names != list(FIXED_FEATURES):
        raise ValueError("Feature names/order differ from the frozen selected_attributes.csv contract")
    return names


def _feature_frame(frame, names):
    if names != list(FIXED_FEATURES) or frame.columns.duplicated().any():
        raise ValueError("Invalid fixed feature order or duplicated input columns")
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError("Missing model inputs: " + ", ".join(sorted(missing)))
    X = frame.loc[:, names].astype(np.float32)
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Model inputs may contain NaNs, but not infinities")
    return X


def predict_compact_bundle(bundle, frame, episode_ids=None, base_dir=None):
    """Return calibrated (n,16) probabilities in target_columns order.

    Accept a loaded dictionary or a joblib path. A path establishes its own bundle
    root; for a dictionary pass base_dir=the output directory to resolve the
    relative checkpoint. Embedded checkpoint bytes also make each joblib complete
    and usable after relocation without a filesystem checkpoint. TCN rows must be
    chronological contiguous full episodes (or prefixes); call boundaries reset
    history. Extra frame columns are ignored, never learned as inputs.
    """
    if isinstance(bundle, (str, Path)):
        path = Path(bundle)
        base_dir = path.parent if base_dir is None else Path(base_dir)
        bundle = joblib.load(path)
    names = bundle["input_columns"]
    X = _feature_frame(frame, names)
    Z = bundle["scaler"].transform(bundle["imputer"].transform(X)).astype(np.float32)
    if Z.shape[1] != 16:
        raise ValueError("Preprocessing changed the sixteen-input contract")
    checkpoint = Path(bundle["model_path"])
    if checkpoint.is_absolute():
        raise ValueError("Compact checkpoint paths must be relative to the bundle root")
    checkpoint = Path(base_dir) / checkpoint if base_dir is not None else None
    if checkpoint is not None and checkpoint.is_file():
        if digest(checkpoint) != bundle["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash differs from the frozen bundle")
        model = dl.load_model(checkpoint)
    else:
        model = dl.load_model(io.BytesIO(bundle["checkpoint_bytes"]))
    if bundle["kind"] == "tcn":
        if episode_ids is None:
            raise ValueError("TCN prediction requires explicit episode_ids")
        p = dl.predict_tcn(model, Z, episode_ids)
    elif bundle["kind"] == "dense":
        p = dl.predict_dense(model, Z)
    else:
        raise ValueError("Compact bundles only support dense and tcn")
    return sigmoid_apply(bundle["calibrators"], p)


def _validate_source(csv_path, schema, manifest, names):
    expected = manifest.get("files", {}).get(csv_path.name, {}).get("sha256")
    actual = digest(csv_path)
    if not expected or expected != actual:
        raise ValueError("Source CSV SHA256 does not match its manifest")
    schema_path = csv_path.with_suffix(".schema.json")
    schema_hash = manifest.get("files", {}).get(schema_path.name, {}).get("sha256")
    if schema_hash and schema_hash != digest(schema_path):
        raise ValueError("Source schema SHA256 does not match its manifest")
    if not set(names).issubset(schema["input_columns"]):
        raise ValueError("Fixed features are not all approved by the source input schema")
    targets = schema["target_columns"]
    if len(targets) != 16 or len(set(targets)) != 16 or any(
            not t.startswith("fault_label.") for t in targets):
        raise ValueError("Source must declare exactly sixteen unique fault_label targets")
    if set(names) & set(targets):
        raise ValueError("Input/target overlap")
    with csv_path.open(newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError("Source CSV has duplicated column names")
    if not set(names + targets + METADATA).issubset(header):
        raise ValueError("Source CSV lacks required schema columns")
    return actual, targets, header


def _validate_rows(frame, targets, manifest):
    """Validate truth/coordinates only; no test feature values are accepted."""
    if frame[METADATA].isna().any().any():
        raise ValueError("Missing journey coordinates or split identifiers")
    y = frame[targets].to_numpy()
    if not np.isfinite(y).all() or not np.isin(y, (0, 1)).all():
        raise ValueError("All sixteen targets must be finite binary indicators")
    if frame.duplicated(["episode_id", "step"]).any():
        raise ValueError("Duplicate episode/step")
    step = frame.step.to_numpy()
    if not np.isfinite(step).all() or (step < 0).any() or (step != np.floor(step)).any():
        raise ValueError("Step must be a nonnegative integer")
    if not np.isfinite(frame.elapsed_s).all() or (frame.elapsed_s < 0).any():
        raise ValueError("Invalid elapsed timestamps")
    membership = manifest["split_membership"]
    if set(membership) != {"train", "validation", "test"}:
        raise ValueError("Manifest must supply train/validation/test engine splits")
    flat = [engine for engines in membership.values() for engine in engines]
    if len(flat) != len(set(flat)) or any(not engines for engines in membership.values()):
        raise ValueError("Manifest splits must be nonempty and engine-disjoint")
    if len(membership["validation"]) != 3:
        raise ValueError("Exactly three validation trucks are required for 1/2 calibration/tuning")
    if not set(frame.split).issubset(membership):
        raise ValueError("Unknown split")
    for split, engines in membership.items():
        if set(frame.loc[frame.split.eq(split), "engine_id"]) != set(engines):
            raise ValueError("CSV split membership differs from manifest")
    if frame.groupby("engine_id").split.nunique().ne(1).any():
        raise ValueError("An engine crosses partitions")
    groups = frame.groupby("episode_id", sort=False)
    if groups.engine_id.nunique().ne(1).any() or groups.split.nunique().ne(1).any():
        raise ValueError("An episode crosses trucks or splits")
    ordered = frame.sort_values(["engine_id", "episode_id", "step"], kind="stable")
    # Real CSV contains valid ISO8601 timestamps both with and without
    # fractional seconds (171 samples span a non-integer interval).
    times = pd.to_datetime(ordered.timestamp, format="ISO8601", utc=True, errors="raise")
    same = ordered.episode_id.eq(ordered.episode_id.shift()).to_numpy()
    if ((ordered.elapsed_s.diff().to_numpy()[same] <= 0).any() or
            (times.diff().to_numpy()[same] <= np.timedelta64(0, "ns")).any()):
        raise ValueError("Sorted episode timestamps and elapsed_s must strictly increase")
    if "counts" in manifest and manifest["counts"].get("rows", len(frame)) != len(frame):
        raise ValueError("Source row count differs from manifest")
    return ordered.reset_index(drop=True)


def _read_partition(csv_path, coordinates, columns, split, names):
    # skiprows excludes test records in the parser itself, rather than loading a
    # complete feature matrix and subsequently slicing it. Header occupies row 0.
    wanted = coordinates.split.eq(split) if split == "test" else ~coordinates.split.eq("test")
    skip = (np.flatnonzero(~wanted.to_numpy()) + 1).tolist()
    frame = pd.read_csv(csv_path, usecols=columns, skiprows=skip,
                        dtype={n: "float32" for n in names})
    frame = frame.sort_values(["engine_id", "episode_id", "step"], kind="stable").reset_index(drop=True)
    _feature_frame(frame, names)
    return frame


def _macro_ap(y, p):
    # Identical absent-label convention to the checkpoint-selection helper.
    return float(np.mean([average_precision_score(y[:, k], p[:, k]) if y[:, k].any() else 0.
                          for k in range(16)]))


def _normal_rows(y, p, thresholds, targets, kind, split, policy):
    normal = y.sum(axis=1) == 0
    pred = p >= thresholds
    count = int(normal.sum())
    return [{"candidate": kind, "split": split, "policy": policy, "fault": target,
             "normal_rows": count, "false_alarms": int(pred[normal, k].sum()),
             "per_label_normal_FPR": float(pred[normal, k].mean()) if count else None,
             "any_label_normal_FPR": float(pred[normal].any(axis=1).mean()) if count else None}
            for k, target in enumerate(targets)]


def _calibration_rows(y, raw, calibrated, targets, kind, split):
    rows, bins = [], []
    for state, p in (("raw", raw), ("sigmoid_calibrated", calibrated)):
        for k, target in enumerate(targets):
            rows.append({"candidate": kind, "split": split, "state": state, "fault": target,
                         "rows": len(y), "Brier": float(brier_score_loss(y[:, k], p[:, k])),
                         "log_loss": float(log_loss(y[:, k], p[:, k], labels=[0, 1]))})
            for index in range(10):
                lo, hi = index / 10, (index + 1) / 10
                mask = (p[:, k] >= lo) & ((p[:, k] < hi) if index < 9 else (p[:, k] <= hi))
                bins.append({"candidate": kind, "split": split, "state": state,
                             "fault": target, "bin_low": lo, "count": int(mask.sum()),
                             "mean_probability": float(p[mask, k].mean()) if mask.any() else None,
                             "observed_frequency": float(y[mask, k].mean()) if mask.any() else None})
    return rows, bins


def _detection_delays(frame, y, pred, targets, policy):
    rows = []
    for episode, positions in frame.groupby("episode_id", sort=False).indices.items():
        positions = np.asarray(positions)
        times = frame.iloc[positions].elapsed_s.to_numpy()
        for k, target in enumerate(targets):
            truth = y[positions, k].astype(bool)
            starts = np.flatnonzero(truth & ~np.r_[False, truth[:-1]])
            for start in starts:
                ends = np.flatnonzero(~truth[start:])
                end = start + ends[0] if len(ends) else len(truth)
                hits = np.flatnonzero(pred[positions[start:end], k])
                rows.append({"policy": policy, "engine_id": frame.iloc[positions[0]].engine_id,
                             "episode_id": episode, "fault": target, "onset_s": float(times[start]),
                             "segment_end_s": float(times[end - 1]), "detected": bool(len(hits)),
                             "delay_s": float(times[start + hits[0]] - times[start]) if len(hits) else None,
                             "censored": not bool(len(hits)),
                             "evaluation_semantics": "first positive within contiguous active truth segment"})
    return rows


def run(csv_path, output, features_path=None, epochs=18, seed=793):
    csv_path, output = Path(csv_path), Path(output)
    features_path = Path(features_path) if features_path is not None else DEFAULT_FEATURES
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new results folder; existing artifacts are never overwritten")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    started = time.perf_counter()
    names = load_fixed_features(features_path)
    schema_path, manifest_path = csv_path.with_suffix(".schema.json"), csv_path.with_suffix(".manifest.json")
    schema = json.loads(schema_path.read_text())
    source_manifest = json.loads(manifest_path.read_text())
    source_hash, targets, header = _validate_source(csv_path, schema, source_manifest, names)
    print("Validating source coordinates and binary targets (no test features)", flush=True)
    coordinates = pd.read_csv(csv_path, usecols=METADATA + targets)
    ordered = _validate_rows(coordinates, targets, source_manifest)
    file_rows = source_manifest["files"][csv_path.name].get("rows")
    if file_rows is not None and file_rows != len(coordinates):
        raise ValueError("Source file row count differs from manifest")
    if csv_path.resolve() == DEFAULT_CSV.resolve():
        sizes = {k: len(v) for k, v in source_manifest["split_membership"].items()}
        if sizes != {"train": 14, "validation": 3, "test": 3} or int(coordinates.split.eq("train").sum()) != 114912:
            raise ValueError("Default benchmark must retain 14/3/3 trucks and all 114912 training rows")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(features_path, output / "selected_attributes.csv")
    (output / "source").mkdir()
    shutil.copyfile(features_path, output / "source/selected_attributes.csv")
    dependencies = {n: sensor_dependencies(n) for n in names}
    feature_schema = {"input_columns": names, "selected_original_attributes": names,
                      "model_input_columns": names, "target_columns": targets,
                      "input_dimension": 16, "preprocessing": "training-only median imputation (keep_empty_features=True), StandardScaler",
                      "feature_selection": "none; immutable supplied list",
                      "additional_context_inputs": [], "dependencies": dependencies,
                      "dependency_warning": DEPENDENCY_WARNING,
                      "synthetic_only": True, "deployment_ready": False,
                      "health_training": False, "RUL_training": False,
                      "excluded_source_fields": "all non-fixed columns including labels, truth, diagnostics, health, RUL, identifiers, context and timestamps",
                      "tcn": {"window": 12, "causal": True, "episode_reset": True,
                              "left_padding": "repeat episode first observation"}}
    _json(output / "feature_schema.json", feature_schema)
    journey = ordered.groupby("episode_id", sort=False).agg(
        engine_id=("engine_id", "first"), split=("split", "first"), rows=("step", "size"),
        first_step=("step", "min"), last_step=("step", "max"),
        first_elapsed_s=("elapsed_s", "min"), last_elapsed_s=("elapsed_s", "max")).reset_index()
    calibration_truck = sorted(source_manifest["split_membership"]["validation"])[0]
    journey["role"] = journey["split"]
    journey.loc[journey.split.eq("validation"), "role"] = "validation_tuning"
    journey.loc[journey.engine_id.eq(calibration_truck), "role"] = "calibration"
    journey.to_csv(output / "journey_splits.csv", index=False)
    trainval = _read_partition(csv_path, coordinates, names + targets + METADATA, "trainval", names)
    tr = trainval[trainval.split.eq("train")].reset_index(drop=True)
    va = trainval[trainval.split.eq("validation") & ~trainval.engine_id.eq(calibration_truck)].reset_index(drop=True)
    ca = trainval[trainval.engine_id.eq(calibration_truck)].reset_index(drop=True)
    ytr, yva, yca = [f[targets].to_numpy(np.int8) for f in (tr, va, ca)]
    if not (yva.sum(axis=1) == 0).any():
        raise ValueError("Tuning validation has no normal rows; guarded policy cannot be fitted")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    atr = scaler.fit_transform(imputer.fit_transform(_feature_frame(tr, names))).astype(np.float32)
    ava, aca = [scaler.transform(imputer.transform(_feature_frame(f, names))).astype(np.float32) for f in (va, ca)]
    if atr.shape != (len(tr), 16):
        raise ValueError("Preprocessing must preserve all sixteen inputs")
    timings = {"source_validation_and_preprocessing_seconds": time.perf_counter() - started}
    candidates = {}
    summaries, per_fault, thresholds_rows, normal_rows, calibration_rows, calibration_bins = [], [], [], [], [], []
    for kind in ("dense", "tcn"):
        begin = time.perf_counter()
        print(f"Training fresh {kind}: {len(tr)} rows, exactly 16 inputs", flush=True)
        if kind == "dense":
            result = dl.train_dense(atr, ytr, ava, yva, output / "models", name="dense", seed=seed, epochs=epochs)
            raw_va, raw_ca = dl.predict_dense(result, ava), dl.predict_dense(result, aca)
        else:
            result = dl.train_tcn(atr, ytr, tr.episode_id.to_numpy(), ava, yva,
                                  va.episode_id.to_numpy(), output / "models", seed=seed, epochs=epochs, window=12)
            raw_va = dl.predict_tcn(result, ava, va.episode_id.to_numpy())
            raw_ca = dl.predict_tcn(result, aca, ca.episode_id.to_numpy())
        timings[kind + "_training_and_validation_seconds"] = time.perf_counter() - begin
        calibrators = sigmoid_fit(yca, raw_ca)
        pva = sigmoid_apply(calibrators, raw_va)
        pca = sigmoid_apply(calibrators, raw_ca)
        thresholds = {"maxf1": thresholds_for(yva, pva),
                      "guarded": thresholds_for(yva, pva, max_normal_fpr=.01)}
        checkpoint = Path(result["model_path"])
        bundle = {"format_version": 1, "kind": kind, "input_columns": names,
                  "target_columns": targets, "imputer": imputer, "scaler": scaler,
                  "model_path": str(checkpoint.relative_to(output)),
                  "checkpoint_bytes": checkpoint.read_bytes(), "checkpoint_sha256": digest(checkpoint),
                  "window": 12 if kind == "tcn" else None, "calibrators": calibrators,
                  "thresholds": thresholds["guarded"], "thresholds_by_policy": thresholds,
                  "threshold_policy": POLICIES["guarded"], "deployment_ready": False,
                  "synthetic_only": True, "root_cause_confirmed": False,
                  "health_status": "not_trained", "rul_status": "unavailable_not_validated",
                  "dependency_warning": DEPENDENCY_WARNING,
                  "calibration_truck": calibration_truck,
                  "uncalibrated_targets": [targets[k] for k, c in enumerate(calibrators) if c is None],
                  "train_rows": len(tr), "epochs_budget": epochs, "epochs_completed": len(result["history"]),
                  "best_epoch": result["best_epoch"], "device": result["device"], "seed": seed}
        candidates[kind] = {"result": result, "bundle": bundle, "raw_va": raw_va,
                            "pva": pva, "thresholds": thresholds,
                            "selection_macro_AP": _macro_ap(yva, raw_va)}
        joblib.dump(bundle, output / f"{kind}_model.joblib", compress=3)
        for policy, threshold in thresholds.items():
            rows = [{"candidate": kind, "policy": policy, "fault": target, "threshold": float(threshold[k]),
                     "objective": POLICIES[policy], "tuning_partition": "two_validation_trucks",
                     "normal_FPR_guard": .01 if policy == "guarded" else None}
                    for k, target in enumerate(targets)]
            thresholds_rows.extend(rows)
            pd.DataFrame(rows).to_csv(output / f"{kind}_{policy}_thresholds.csv", index=False)
            summary, label_rows = metrics(yva, pva, threshold, targets, kind, "validation_tuning")
            summary.update(policy=policy, inputs=16, checkpoint_selection_macro_AP=_macro_ap(yva, raw_va))
            summaries.append(summary)
            per_fault.extend({**r, "policy": policy} for r in label_rows)
            normal_rows.extend(_normal_rows(yva, pva, threshold, targets, kind, "validation_tuning", policy))
            if policy == "guarded" and any(r["per_label_normal_FPR"] > .01 + 1e-12 for r in normal_rows[-16:]):
                raise AssertionError("Validation normal FPR guard violated")
        for partition, truth, raw, calibrated in (("validation_tuning", yva, raw_va, pva),
                                                   ("calibration_fit_in_sample", yca, raw_ca, pca)):
            rows, bins = _calibration_rows(truth, raw, calibrated, targets, kind, partition)
            calibration_rows.extend(rows); calibration_bins.extend(bins)
        print(f"{kind} best epoch {result['best_epoch']}, validation selection macro AP={_macro_ap(yva, raw_va):.6f}", flush=True)

    # Selection uses the SAME raw macro AP objective as checkpoint selection.
    # Stable kind order gives dense precedence for an exact validation tie.
    best_kind = max(candidates, key=lambda k: candidates[k]["selection_macro_AP"])
    selected_bundle = candidates[best_kind]["bundle"]
    joblib.dump(selected_bundle, output / "selected_model.joblib", compress=3)
    alternate_bundle = {**selected_bundle, "thresholds": selected_bundle["thresholds_by_policy"]["maxf1"],
                        "threshold_policy": POLICIES["maxf1"]}
    joblib.dump(alternate_bundle, output / "selected_maxf1_model.joblib", compress=3)
    pd.DataFrame(thresholds_rows).to_csv(output / "thresholds.csv", index=False)
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": source_hash, "input_columns": names, "target_columns": targets,
              "selected_kind": best_kind, "primary_policy": "guarded", "alternate_policy": "maxf1",
              "policy_names": POLICIES, "selection_metric": "raw validation macro AP; absent labels score zero",
              "tie_break": "dense", "calibration_truck": calibration_truck,
              "threshold_trucks": sorted(va.engine_id.unique().tolist()),
              "calibration": "per-label sigmoid fit ONLY on reserved validation truck; no-variation labels left raw",
              "guard": "1% per-label FPR on all-zero-label validation rows, not any-label or test guarantee",
              "candidates": {k: {"selection_macro_AP": v["selection_macro_AP"],
                                  "best_epoch": v["result"]["best_epoch"],
                                  "thresholds": {p: t.tolist() for p, t in v["thresholds"].items()},
                                  "checkpoint_sha256": v["bundle"]["checkpoint_sha256"],
                                  "bundle_sha256": digest(output / f"{k}_model.joblib")}
                             for k, v in candidates.items()},
              "test_features_loaded": False, "test_predictions_computed": False,
              "deployment_ready": False}
    _json(output / "frozen_policy.json", frozen)
    frozen_digest = digest(output / "frozen_policy.json")
    print(f"Frozen validation choice: {best_kind}; primary guarded policy. Loading test features now.", flush=True)

    begin = time.perf_counter()
    evaluation_columns = [n for n in ["sensor_quality_json", "coverage_group", "primary_class"] if n in header]
    severity = ["effect_level." + target.split(".", 1)[1] for target in targets]
    evaluation_columns += [n for n in severity if n in header]
    te = _read_partition(csv_path, coordinates, names + targets + METADATA + evaluation_columns, "test", names)
    yte = te[targets].to_numpy(np.int8)
    ate = scaler.transform(imputer.transform(_feature_frame(te, names))).astype(np.float32)
    for kind, candidate in candidates.items():
        result = candidate["result"]
        raw = dl.predict_tcn(result, ate, te.episode_id.to_numpy()) if kind == "tcn" else dl.predict_dense(result, ate)
        p = sigmoid_apply(candidate["bundle"]["calibrators"], raw)
        candidate["pte"] = p
        for policy, threshold in candidate["thresholds"].items():
            summary, label_rows = metrics(yte, p, threshold, targets, kind, "test")
            summary.update(policy=policy, inputs=16)
            summaries.append(summary)
            per_fault.extend({**r, "policy": policy} for r in label_rows)
            normal_rows.extend(_normal_rows(yte, p, threshold, targets, kind, "test", policy))
        rows, bins = _calibration_rows(yte, raw, p, targets, kind, "test")
        calibration_rows.extend(rows); calibration_bins.extend(bins)
    timings["test_inference_seconds"] = time.perf_counter() - begin
    p = candidates[best_kind]["pte"]
    strata = {"normal": yte.sum(axis=1) == 0, "single_fault": yte.sum(axis=1) == 1,
              "mixed_fault": yte.sum(axis=1) > 1}
    strata.update({"truck_" + str(engine): te.engine_id.eq(engine).to_numpy() for engine in te.engine_id.unique()})
    gaps = []
    if "sensor_quality_json" in te:
        quality = [json.loads(value) for value in te.sensor_quality_json]
        strata["sensor_quality_good"] = np.asarray([not value for value in quality])
        for status in ("missing", "noisy", "biased", "drifting", "frozen"):
            strata["sensor_quality_" + status] = np.asarray([status in value.values() for value in quality])
    else:
        gaps.append("sensor-quality truth is unavailable in supplied CSV")
    if set(severity).issubset(te.columns):
        strength = te[severity].to_numpy()
        strata["any_weak_effect"] = ((strength > .001) & (strength < .5)).any(axis=1)
        strata["any_developed_effect"] = (strength >= .5).any(axis=1)
    else:
        gaps.append("effect-severity truth is unavailable in supplied CSV")
    stratified, stratified_per_fault, delays = [], [], []
    for policy, threshold in candidates[best_kind]["thresholds"].items():
        pred = p >= threshold
        prediction = te[METADATA].copy()
        for k, target in enumerate(targets):
            fault = target.split(".", 1)[1]
            prediction["truth." + fault] = yte[:, k]
            prediction["probability." + fault] = p[:, k]
            prediction["prediction." + fault] = pred[:, k].astype(np.int8)
        prediction.to_csv(output / f"test_predictions_{policy}.csv", index=False)
        if policy == "guarded":
            prediction.to_csv(output / "test_predictions.csv", index=False)
        for stratum, mask in strata.items():
            if mask.any():
                summary, label_rows = metrics(yte[mask], p[mask], threshold, targets, best_kind, stratum)
                summary.update(policy=policy, truth_fields_evaluation_only=True)
                stratified.append(summary)
                stratified_per_fault.extend({**r, "policy": policy} for r in label_rows)
        delays.extend(_detection_delays(te, yte, pred, targets, policy))
    pd.DataFrame(delays, columns=["policy", "engine_id", "episode_id", "fault", "onset_s", "segment_end_s",
                                 "detected", "delay_s", "censored", "evaluation_semantics"]).to_csv(output / "detection_delay.csv", index=False)
    _json(output / "evaluation_gaps.json", {"gaps": gaps, "detection_delay": "computed separately for both policies; misses censored, not zero delay",
                                           "strata": "overlapping descriptive truth-evaluation groups; not independent populations"})
    for filename, rows in (("model_comparison.csv", summaries), ("per_fault_metrics.csv", per_fault),
                           ("normal_false_alarms.csv", normal_rows), ("calibration_metrics.csv", calibration_rows),
                           ("calibration_bins.csv", calibration_bins), ("stratified_metrics.csv", stratified),
                           ("stratified_per_fault_metrics.csv", stratified_per_fault)):
        pd.DataFrame(rows).to_csv(output / filename, index=False)

    # Compare serialization at the SAME full-test batch sizes. CUDA convolution
    # kernels can change rounding with a smaller single-episode batch; test that
    # separately rather than misreporting it as altered checkpoint weights.
    first_episode = te.episode_id.iloc[0]
    first = te[te.episode_id.eq(first_episode)]
    positions = first.index.to_numpy()
    checks = []
    for filename, kind in (("dense_model.joblib", "dense"), ("tcn_model.joblib", "tcn"),
                           ("selected_model.joblib", best_kind), ("selected_maxf1_model.joblib", best_kind)):
        restored = predict_compact_bundle(output / filename, te, te.episode_id.to_numpy())
        expected = candidates[kind]["pte"]
        np.testing.assert_allclose(restored, expected, rtol=1e-5, atol=1e-6)
        prefix = predict_compact_bundle(output / filename, first, first.episode_id.to_numpy())
        threshold = joblib.load(output / filename)["thresholds"]
        checks.append({"bundle": filename, "kind": kind, "episode_id": first_episode, "rows": len(first),
                       "full_test_rows": len(te), "max_abs_probability_error": float(np.max(np.abs(restored - expected))),
                       "prefix_batch_size_probability_difference": float(np.max(np.abs(prefix - expected[positions]))),
                       "prefix_batch_size_decision_flips": int(((prefix >= threshold) != (expected[positions] >= threshold)).sum()),
                       "passed": True})
    if digest(output / "frozen_policy.json") != frozen_digest:
        raise AssertionError("Frozen policy changed during test evaluation")
    verification = {"serialization_reload": checks, "frozen_policy_sha256": frozen_digest,
                    "frozen_before_test_features": True, "model_inputs": 16,
                    "trained_on_all_train_rows": len(tr), "no_test_selection_or_calibration": True,
                    "source_sha256_verified": True, "deployment_ready": False}
    _json(output / "verification_receipt.json", verification)
    table = pd.DataFrame(summaries)[["candidate", "split", "policy", "macro_AP", "macro_F1", "macro_recall", "normal_false_alarm_rate", "macro_Brier"]]
    report = ["# Fixed compact-16 synthetic fault training", "",
              f"Validation-selected model: **{best_kind}**. Primary policy: **guarded**; alternate: **maxf1**.", "",
              "Both fresh candidates train on every training row. Checkpoints and kind use raw macro AP on two validation trucks; "
              "one separately reserved validation truck fits sigmoid calibration. Both exact PR-grid threshold policies are "
              "frozen before any test feature matrix or prediction is loaded. No test metric selects a model or policy.", "",
              markdown_table(table), "", "## Contract and limitations", DEPENDENCY_WARNING,
              "Synthetic-only evidence; deployment_ready=False. This held-truck benchmark has been used previously and is not a newly untouched field test. "
              "Only three test trucks in the default dataset; overlapping/autocorrelated journey rows are not independent trials. "
              "Calibration is synthetic and uses a single truck; labels without both classes on that truck are left uncalibrated.",
              "The 1% guard is validation per-label FPR on all-zero-label rows, not a test guarantee or a 1% any-label alarm guarantee. "
              "Weak effects or labels without direct selected sensor evidence may remain undetectable. No root cause is confirmed.",
              "No health or RUL model is trained; no diagnostic/health/RUL source field enters the model. "
              "Truth sensor-quality/effect severity and target labels are used only for evaluation groups and delay segments. "
              "Detection delay reports first positive within each active truth segment; undetected segments are censored. "
              "It does not establish real event onset or operational alarm persistence.",
              "No SHAP or replayed sensor-failure robustness experiment is included. Sensor-quality strata describe existing injected variants.",
              "", "## Artifacts", "Candidate and selected joblib bundles embed preprocessing, calibration, ordered schemas, "
              "both policies and checkpoint bytes, with relative checkpoint paths. Use predict_compact_bundle(joblib_path, frame, episode_ids) "
              "for relocated outputs. Supply complete chronological episodes; TCN call boundaries reset history.",
              "", "## Reproduce", "```bash",
              f"OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /tmp/opencode/fault-training/bin/python analysis/train_compact16.py --csv {csv_path} --output {output} --features {features_path} --epochs {epochs} --seed {seed}",
              "```"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    timings["total_seconds"] = time.perf_counter() - started
    result_manifest = {"source_csv": str(csv_path.resolve()), "source_sha256": source_hash,
                       "source_schema_sha256": digest(schema_path), "source_manifest_sha256": digest(manifest_path),
                       "source_selected_attributes": str(features_path.resolve()), "source_features_sha256": digest(features_path),
                       "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                       (Path(__file__).resolve(), ROOT / "analysis/fault_dl.py", ROOT / "analysis/train_fault_models.py",
                                        ROOT / "analysis/fault_features.py", ROOT / "enhanced_analytics.py")},
                       "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                                    "sklearn": sklearn.__version__, "joblib": joblib.__version__, "torch": dl.torch.__version__},
                       "seed": seed, "epochs_budget": epochs, "input_columns": names, "target_columns": targets,
                       "selected_model": best_kind, "primary_policy": "guarded", "alternate_policy": "maxf1",
                       "selection_strategy": frozen["selection_metric"], "calibration_truck": calibration_truck,
                       "split_membership": source_manifest["split_membership"], "train_rows": len(tr),
                       "validation_tuning_rows": len(va), "calibration_rows": len(ca), "test_rows": len(te),
                       "models": {k: {"best_epoch": c["result"]["best_epoch"], "epochs_completed": len(c["result"]["history"]),
                                      "device": c["result"]["device"], "selection_macro_AP": c["selection_macro_AP"]}
                                  for k, c in candidates.items()},
                       "timings": timings, "elapsed_seconds": timings["total_seconds"],
                       "verification": verification, "dependency_warning": DEPENDENCY_WARNING,
                       "health": "not_trained", "RUL": "unavailable_not_validated", "deployment_ready": False,
                       "test_evidence_status": "previously used synthetic held-truck benchmark; not newly untouched field evidence",
                       "artifacts": {str(path.relative_to(output)): digest(path) for path in output.rglob("*") if path.is_file()}}
    _json(output / "manifest.json", result_manifest)
    return result_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=793)
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.output, args.features, args.epochs, args.seed), indent=2))
