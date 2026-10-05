"""Training-only feature selection for the exported 16-fault multilabel task.

No split, model orchestration or feature engineering lives here. The exact
exporter allowlist is reconstructed from observation feature definitions, never
from CSV numeric dtypes. All constant/duplicate checks use the entire training
array; scoring and redundancy checks use at most 30,000 reproducibly sampled
training rows. Neither validation nor test arrays are accepted by this API.

The preliminary ranking is association evidence, not minimum instrumentation.
Derived features require their source sensors, operating context, calibration,
and within-episode history. A smaller feature count need not mean fewer sensors.
Permutation importance must be measured by the caller on truck-independent
validation data. Row-wise cross-validation and SHAP are intentionally absent.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from threadpoolctl import threadpool_limits

from enhanced_analytics import CONTEXT_NUMERIC, MODE_LOAD, SENSOR_COLUMNS, THRESHOLDS


MAX_FIT_ROWS = 30_000
MAX_MI_ROWS = 15_000
CORRELATION_THRESHOLD = .95
THREADS = 2
TREE_COUNT = 64
TREE_DEPTH = 12
_DERIVED_SUFFIXES = (
    "__residual", "__rate", "__residual_slope_120s", "__mean_30s", "__std_30s",
)
_LEAD_PAIRS = (
    ("engine.coolant_flow_kg_s", "engine.oil_temp_c"),
    ("engine.oil_pressure_kpa", "engine.oil_temp_c"),
    ("engine.oil_temp_c", "engine.block_vibration_mm_s"),
)
_STRESS_REFERENCE_CHANNELS = frozenset({
    "engine.oil_pressure_kpa", "transmission.oil_pressure_kpa", "transmission.vibration_mm_s",
    "transmission.oil_temp_c", "suspension.hydraulic_oil_condition_pct",
    "suspension.actuator_vibration_mm_s", "coolant_condition_pct", "gnss.accuracy_m",
    "brake.disc_temp_c", "engine.boost_kpa", "engine.coolant_temp_c", "engine.block_temp_c",
    "engine.coolant_pump_temp_c", "engine.oil_temp_c", "engine.block_vibration_mm_s",
})


def _reference_sensors(base: str) -> list[str]:
    """Dependencies of the existing prior, covering both drive-cycle branches."""
    sensors = [base]
    if base in _STRESS_REFERENCE_CHANNELS:
        sensors.append("frame.strain_microstrain")
    if base in {"engine.coolant_temp_c", "engine.block_temp_c", "engine.coolant_pump_temp_c"}:
        sensors.append("engine.coolant_energy_flow_kw")
    if base in {"brake.disc_temp_c", "brake.stroke_pct"}:
        sensors.append("brake.pedal_pct")
    if base == "brake.stroke_pct":
        sensors.append("brake.pad_wear_pct")
    return sensors


def allowed_input_names() -> frozenset[str]:
    """Return the exporter contract, including approved perception confidence."""
    return frozenset([
        *SENSOR_COLUMNS,
        *(name + "__available_mask" for name in SENSOR_COLUMNS),
        *("context." + name for name in CONTEXT_NUMERIC),
        "context.cold_start",
        *("context.mode_" + mode for mode in MODE_LOAD),
        *(name + suffix for name in THRESHOLDS for suffix in _DERIVED_SUFFIXES),
        "engine.egt_spread_c",
        *(f"leadlag.{first}__to__{second}_s" for first, second in _LEAD_PAIRS),
    ])


def proxy_ablation_reason(name: str) -> str | None:
    """Route/consumption/time shortcuts, not all perception or energy channels."""
    base = name.split("__", 1)[0].lower()
    if base in {"gnss.latitude_deg", "gnss.longitude_deg", "gnss.altitude_m"}:
        return "GNSS absolute position/route proxy"
    if base == "gnss.heading_deg":
        return "GNSS absolute heading/route proxy"
    if base == "engine.fuel_level_pct":
        return "Fuel level/elapsed-consumption proxy"
    if base.startswith("cumulative_"):
        return "Cumulative energy/consumption/time proxy"
    if base in {"time", "time_s", "elapsed_s", "timestamp", "operating_hours"}:
        return "Absolute or cumulative time proxy"
    return None


def sensor_dependencies(name: str) -> dict[str, Any]:
    """Describe required inputs, not a claim about production instrumentation."""
    context = ["context." + key for key in CONTEXT_NUMERIC]
    prior_context = context + ["context.mode", "context.cold_start", "context.drive_cycle"]
    if name.startswith("context."):
        return {"sensors": [], "context": [name], "history": False,
                "requirements": "Available operating-context signal or state"}
    if name == "engine.egt_spread_c":
        return {"sensors": [f"engine.exhaust_temperatures_c.{i}" for i in range(6)],
                "context": [], "history": False, "requirements": "Six exhaust temperature channels"}
    if name.startswith("leadlag."):
        first, second = name[len("leadlag."):].split("__to__")
        sensors = list(dict.fromkeys(_reference_sensors(first) + _reference_sensors(second[:-2])))
        return {"sensors": sensors, "context": prior_context,
                "history": True,
                "requirements": "Persistent residual-onset history, episode resets and elapsed intervals"}
    base = name.split("__", 1)[0]
    residual = "__residual" in name
    history = residual or any(suffix in name for suffix in ("__rate", "__residual_slope", "__mean", "__std"))
    sensors = _reference_sensors(base) if residual else [base]
    requirement = "Observed numeric channel; vehicle availability must be confirmed"
    if residual:
        requirement = "Healthy context prior and commissioning/calibration; no oracle readings"
    if history:
        requirement += "; trailing observations, elapsed intervals and episode resets"
    if name.endswith("__available_mask"):
        requirement = "Observed channel validity/availability checks, never injected quality labels"
        if base in THRESHOLDS and base.endswith(("_pressure_kpa", "_temp_c")):
            residual, history = True, True
            sensors = _reference_sensors(base)
            requirement += "; frozen-suspicion checks require residual prior and repeated observations"
    return {"sensors": sensors, "context": prior_context if residual else [],
            "history": history, "requirements": requirement}


def _names(names: list[str], count: int, label: str) -> None:
    if not isinstance(names, list) or len(names) != count:
        raise ValueError(f"{label} must be a list with {count} names")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != count:
        raise ValueError(f"{label} must contain unique nonempty strings")


def _validate(X: np.ndarray, y: np.ndarray, input_names: list[str],
              target_names: list[str], seed: int, max_fit_rows: int) -> None:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype != np.float32:
        raise ValueError("X_train must be a two-dimensional float32 numpy array")
    if X.shape[0] < 3 or X.shape[1] == 0:
        raise ValueError("X_train must have at least three rows and one column")
    if np.isinf(X).any():
        raise ValueError("X_train may contain NaNs, but not infinities")
    if not isinstance(y, np.ndarray) or y.shape != (X.shape[0], 16):
        raise ValueError("y_train must have shape (training rows, 16)")
    if y.dtype.kind not in "biuf" or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("y_train must contain finite binary multilabel indicators (0 or 1)")
    _names(input_names, X.shape[1], "input_names")
    _names(target_names, 16, "target_names")
    forbidden = sorted(set(input_names) - allowed_input_names())
    if forbidden:
        raise ValueError("Input names outside the observation-only allowlist (possible leakage): "
                         + ", ".join(forbidden))
    if set(input_names) & set(target_names):
        raise ValueError("Input and target names must be disjoint")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if isinstance(max_fit_rows, bool) or not isinstance(max_fit_rows, (int, np.integer)) or max_fit_rows < 3:
        raise ValueError("max_fit_rows must be an integer of at least three")


def _csv(path: Path, fields: list[str], records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    xc, yc = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(np.clip(np.dot(xc, yc) / denominator, -1, 1)) if denominator > 0 else np.nan


def _pairwise_pearson(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete Pearson, preserving NaNs rather than imputing them."""
    mask = (~np.isnan(X)).astype(np.float64)
    values = np.nan_to_num(X.astype(np.float64), nan=0.)
    # Global centering does not change pairwise correlations, and avoids
    # cancellation when absolute readings are large compared with variation.
    observed = mask.sum(axis=0)
    centers = np.divide(values.sum(axis=0), observed, out=np.zeros(X.shape[1]), where=observed > 0)
    values = (values - centers) * mask
    counts = mask.T @ mask
    sums = values.T @ mask
    squares = (values * values).T @ mask
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = values.T @ values - sums * sums.T / counts
        variance = np.maximum(0., squares - sums * sums / counts)
        denominator = np.sqrt(variance * variance.T)
        correlation = covariance / denominator
    correlation[(counts < 3) | (denominator <= 0)] = np.nan
    return np.clip(correlation, -1, 1), counts.astype(np.int64)


def _rank_scores(values: np.ndarray) -> np.ndarray:
    """Tied normalized ranks; a zero/undefined association contributes zero."""
    values = np.nan_to_num(np.abs(values), nan=0.)
    result = np.zeros(len(values), dtype=np.float64)
    positive = values > 0
    if positive.any():
        result[positive] = rankdata(values[positive], method="average") / positive.sum()
    return result


def _diverse_ranking(scores: np.ndarray, active: list[int]) -> tuple[list[int], np.ndarray]:
    """Equal-label rank aggregation with one best-first round for label coverage."""
    aggregate = scores[:, active].mean(axis=1) if active else np.zeros(len(scores))
    global_order = sorted(range(len(scores)), key=lambda index: (-aggregate[index], index))
    selected = []
    seen = set()
    for label in active:
        order = sorted(range(len(scores)), key=lambda index: (-scores[index, label], -aggregate[index], index))
        choice = next((index for index in order if index not in seen and scores[index, label] > 0), None)
        if choice is not None:
            selected.append(choice)
            seen.add(choice)
    return selected + [index for index in global_order if index not in seen], aggregate


def select_features(X_train: np.ndarray, y_train: np.ndarray, input_names: list[str],
                    target_names: list[str], output_dir: Path, seed: int = 793,
                    max_fit_rows: int = 30_000) -> dict[str, Any]:
    """Write reproducible training-only evidence and return original-column mappings.

    ``eligible_*`` exclude constants, all-missing columns and exact duplicates.
    ``pruned_*`` additionally exclude |pairwise Pearson| >= .95 redundancy.
    Names and indices in these two groups follow original input order.
    ``ranking_names`` covers all eligible columns in preliminary ranked order.
    Numeric subsets use that order after redundancy pruning; ``all`` preserves
    all eligible columns and ``proxy_ablated`` is all eligible minus the stated
    shortcut proxies. Every subset contains original names, not generated ones.
    """
    _validate(X_train, y_train, input_names, target_names, seed, max_fit_rows)
    seed, max_fit_rows = int(seed), min(int(max_fit_rows), MAX_FIT_ROWS)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row_count = len(X_train)
    rng = np.random.default_rng(seed)
    fit_rows = np.sort(rng.choice(row_count, min(row_count, max_fit_rows), replace=False))
    mi_rows = np.sort(rng.choice(fit_rows, min(len(fit_rows), MAX_MI_ROWS), replace=False))

    audit, removals, eligible_indices = [], [], []
    fingerprints: dict[bytes, list[int]] = {}
    for index, name in enumerate(input_names):
        column = X_train[:, index]
        observed = column[~np.isnan(column)]
        reason, representative = "eligible", None
        if not len(observed):
            reason = "all_missing"
        elif np.all(observed == observed[0]):
            reason = "constant_nonmissing"
        else:
            canonical = column.copy()
            canonical[np.isnan(canonical)] = np.nan
            canonical[canonical == 0] = 0.  # Equal +0/-0 must hash identically.
            digest = hashlib.sha256(canonical.tobytes()).digest()
            for previous in fingerprints.get(digest, []):
                if np.array_equal(column, X_train[:, previous], equal_nan=True):
                    reason, representative = "exact_duplicate_nan_aware", input_names[previous]
                    break
            if representative is None:
                fingerprints.setdefault(digest, []).append(index)
        dependencies = sensor_dependencies(name)
        audit.append({"column": name, "original_index": index, "dtype": str(X_train.dtype),
                      "training_rows": row_count, "nonmissing": len(observed),
                      "missing": row_count - len(observed), "status": reason,
                      "representative": representative, "proxy_ablation_reason": proxy_ablation_reason(name),
                      "required_sensors": json.dumps(dependencies["sensors"]),
                      "required_context": json.dumps(dependencies["context"]),
                      "requires_history": dependencies["history"], "requirements": dependencies["requirements"]})
        if reason == "eligible":
            eligible_indices.append(index)
        else:
            removals.append({"column": name, "original_index": index, "reason": reason,
                             "representative": representative})

    eligible_names = [input_names[index] for index in eligible_indices]
    feature_count = len(eligible_indices)
    pearson = np.full((feature_count, 16), np.nan)
    spearman = np.full_like(pearson, np.nan)
    mutual_information = np.zeros_like(pearson)
    tree_importance = np.zeros_like(pearson)
    normalized = np.zeros_like(pearson)
    correlations, mi_records = [], []
    label_records = []
    active = []
    X_fit = X_train[np.ix_(fit_rows, eligible_indices)].astype(np.float64)
    X_mi = X_train[np.ix_(mi_rows, eligible_indices)].astype(np.float64)
    y_fit, y_mi = y_train[fit_rows].astype(np.int8), y_train[mi_rows].astype(np.int8)
    medians = np.zeros(feature_count)
    for index in range(feature_count):
        observed = X_fit[:, index][~np.isnan(X_fit[:, index])]
        medians[index] = np.median(observed) if len(observed) else 0.
    imputed_fit = np.where(np.isnan(X_fit), medians, X_fit).astype(np.float32)
    imputed_mi = np.where(np.isnan(X_mi), medians, X_mi).astype(np.float32)
    discrete = np.array([name.endswith("__available_mask") or name.startswith("context.mode_")
                         or name == "context.cold_start" for name in eligible_names], dtype=bool)

    with threadpool_limits(limits=THREADS):
        redundancy, overlap = _pairwise_pearson(X_fit)
        for label, target in enumerate(target_names):
            varying = len(np.unique(y_fit[:, label])) == 2
            label_records.append({"target": target, "training_positive": int(y_train[:, label].sum()),
                                  "fit_positive": int(y_fit[:, label].sum()),
                                  "fit_negative": int(len(fit_rows) - y_fit[:, label].sum()),
                                  "scored": bool(varying and feature_count),
                                  "reason": "no_eligible_features" if not feature_count else
                                            ("" if varying else "single_class_in_fit_rows")})
            if varying and feature_count:
                active.append(label)
            for feature, name in enumerate(eligible_names):
                mask = ~np.isnan(X_fit[:, feature])
                x, y = X_fit[mask, feature], y_fit[mask, label].astype(np.float64)
                pearson[feature, label] = _pearson(x, y)
                spearman[feature, label] = _pearson(rankdata(x), y)
                correlations.append({"target": target, "column": name,
                                     "original_index": eligible_indices[feature], "observed_rows": int(mask.sum()),
                                     "pearson_pointbiserial": _number(pearson[feature, label]),
                                     "spearman": _number(spearman[feature, label])})
            if varying and feature_count:
                # Independent per-label class weights avoid multiplying weights
                # for all 16 targets, as sklearn native multioutput would do.
                model = ExtraTreesClassifier(n_estimators=TREE_COUNT, max_depth=TREE_DEPTH,
                                             min_samples_leaf=2, max_features="sqrt",
                                             class_weight="balanced", n_jobs=THREADS,
                                             random_state=(seed + label) % 2**32)
                model.fit(imputed_fit, y_fit[:, label])
                tree_importance[:, label] = model.feature_importances_
                if len(np.unique(y_mi[:, label])) == 2:
                    mutual_information[:, label] = mutual_info_classif(
                        imputed_mi, y_mi[:, label], discrete_features=discrete,
                        random_state=(seed + label) % 2**32, n_neighbors=min(3, len(mi_rows) - 1),
                        n_jobs=THREADS,
                    )
                normalized[:, label] = np.mean([
                    _rank_scores(pearson[:, label]), _rank_scores(spearman[:, label]),
                    _rank_scores(mutual_information[:, label]), _rank_scores(tree_importance[:, label]),
                ], axis=0)
            for feature, name in enumerate(eligible_names):
                mi_records.append({"target": target, "column": name,
                                   "original_index": eligible_indices[feature], "fit_rows": len(mi_rows),
                                   "single_class": bool(len(np.unique(y_mi[:, label])) < 2),
                                   "mutual_information": float(mutual_information[feature, label])})

    order, aggregate = _diverse_ranking(normalized, active)
    kept, redundant = [], []
    for feature in order:
        representative = next((previous for previous in kept
                               if np.isfinite(redundancy[feature, previous])
                               and abs(redundancy[feature, previous]) >= CORRELATION_THRESHOLD), None)
        if representative is None:
            kept.append(feature)
        else:
            redundant.append({"column": eligible_names[feature], "original_index": eligible_indices[feature],
                              "representative": eligible_names[representative],
                              "representative_index": eligible_indices[representative],
                              "pearson": float(redundancy[feature, representative]),
                              "overlap_rows": int(overlap[feature, representative]),
                              "reason": "Absolute pairwise training Pearson >= 0.95; higher preliminary rank retained"})
    ranking_names = [eligible_names[index] for index in order]
    pruned_ranking = [eligible_names[index] for index in kept]
    pruned_indices = sorted(eligible_indices[index] for index in kept)
    subsets = {str(size): pruned_ranking[:size] for size in (8, 16, 32, 64)}
    subsets.update(all=ranking_names, correlation_pruned=pruned_ranking,
                   proxy_ablated=[name for name in ranking_names if proxy_ablation_reason(name) is None])
    ranks = {feature: rank + 1 for rank, feature in enumerate(order)}
    importance = []
    for feature in order:
        record = {"column": eligible_names[feature], "original_index": eligible_indices[feature],
                  "preliminary_rank": ranks[feature], "aggregate_normalized_rank": float(aggregate[feature]),
                  "correlation_pruned_retained": feature in kept,
                  "max_label_rank_score": float(normalized[feature].max())}
        for label, target in enumerate(target_names):
            record[f"{target}__extra_trees"] = float(tree_importance[feature, label])
            record[f"{target}__normalized_rank"] = float(normalized[feature, label])
        importance.append(record)

    metadata = {
        "seed": seed, "training_rows": row_count, "input_count": len(input_names),
        "eligible_count": feature_count, "pruned_count": len(pruned_indices),
        "target_names": target_names, "fit_rows": len(fit_rows), "mi_rows": len(mi_rows),
        "max_fit_rows": max_fit_rows, "max_mi_rows": MAX_MI_ROWS,
        "correlation_threshold": CORRELATION_THRESHOLD, "thread_limit": THREADS,
        "extra_trees": {"n_estimators_per_label": TREE_COUNT, "max_depth": TREE_DEPTH,
                        "class_weight": "balanced independently per label; no joint weight multiplication"},
        "label_support": label_records, "fit_row_sampling": "Seeded uniform training-only sampling without replacement",
        "redundancy_method": "Pairwise-complete Pearson on fit rows; NaNs not imputed",
        "imputation": "Training fit-row medians for MI and trees only; sampled-all-missing columns use 0",
        "ranking_method": "Equal-weight Pearson/Spearman/MI/ExtraTrees normalized tied ranks per label; "
                          "equal-label mean, preceded by one round of per-label best unused features",
        "subset_semantics": "8/16/32/64 are prefixes of correlation-pruned ranking; all includes every eligible "
                            "feature; proxy_ablated is all minus position/heading/fuel-level/cumulative/time proxies",
        "instrumentation_note": "Preliminary feature ranking is not an inferred minimum instrumentation set. "
                                "Audit required sensors/context/calibration/history before deployment.",
        "permutation_importance": "Not computed: caller must use truck-independent validation; API has no truck IDs",
        "cross_validation": "No row-wise cross-validation performed",
        "shap": "Skipped: optional SHAP dependency not required; association and tree impurity evidence only",
        "feature_engineering": "Existing exported observation-derived columns only; source data not modified",
        "required_sensor_dependencies": {name: sensor_dependencies(name) for name in eligible_names},
    }
    result = {"eligible_names": eligible_names, "eligible_indices": eligible_indices,
              "pruned_names": [input_names[index] for index in pruned_indices],
              "pruned_indices": pruned_indices, "ranking_names": ranking_names,
              "subsets": subsets, "fit_row_indices": fit_rows.tolist(),
              "mi_row_indices": mi_rows.tolist(), "metadata": metadata}

    audit_fields = ["column", "original_index", "dtype", "training_rows", "nonmissing", "missing", "status",
                    "representative", "proxy_ablation_reason", "required_sensors", "required_context",
                    "requires_history", "requirements"]
    _csv(output_dir / "column_audit.csv", audit_fields, audit)
    _csv(output_dir / "numeric_features.csv", audit_fields, audit)
    _csv(output_dir / "constant_duplicates.csv", ["column", "original_index", "reason", "representative"], removals)
    _csv(output_dir / "correlation_pruned.csv", ["column", "original_index", "representative", "representative_index",
                                              "pearson", "overlap_rows", "reason"], redundant)
    importance_fields = ["column", "original_index", "preliminary_rank", "aggregate_normalized_rank",
                         "correlation_pruned_retained", "max_label_rank_score"]
    importance_fields += [f"{target}__{method}" for target in target_names for method in ("extra_trees", "normalized_rank")]
    _csv(output_dir / "feature_importance.csv", importance_fields, importance)
    _csv(output_dir / "preliminary_ranking.csv", importance_fields, importance)
    _csv(output_dir / "per_fault_correlations.csv", ["target", "column", "original_index", "observed_rows",
                                                   "pearson_pointbiserial", "spearman"], correlations)
    _csv(output_dir / "mutual_information.csv", ["target", "column", "original_index", "fit_rows",
                                               "single_class", "mutual_information"], mi_records)
    (output_dir / "subsets.json").write_text(json.dumps(subsets, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "selected_preliminary_ranks.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.save(output_dir / "training_feature_correlation.npy", redundancy)
    return result
