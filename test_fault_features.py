"""Meaningful training-only feature selection and original-column mapping checks."""
import csv
import json

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="Use the isolated fault-training environment")

from analysis.fault_features import allowed_input_names, select_features, sensor_dependencies


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def selection_fixture():
    rng = np.random.default_rng(73)
    rows = 240
    y = np.zeros((rows, 16), dtype=np.int8)
    # Exclusive fault states plus normal make multiclass-looking data, still
    # represented as the required 16 binary labels.
    y[:80, 0] = 1
    y[80:160, 1] = 1
    first = (y[:, 0] * 20 + rng.normal(0, .05, rows)).astype(np.float32)
    first[::11] = np.nan
    second = (y[:, 1] * 20 + rng.normal(0, .05, rows)).astype(np.float32)
    near = first * 2 + 5
    near[::13] = np.nan  # Different NaN pattern, correlated only on overlap.
    constant = np.full(rows, 7., dtype=np.float32)
    constant[::7] = np.nan
    names = ["engine.oil_pressure_kpa", "engine.boost_kpa", "engine.oil_temp_c",
             "engine.coolant_temp_c", "engine.load_pct", "engine.fuel_rate_lph",
             "gnss.latitude_deg", "gnss.heading_deg", "engine.fuel_level_pct",
             "cumulative_fuel_energy_kwh", "lidar.obstacle_confidence_pct",
             "radar.target_confidence_pct", "gnss.accuracy_m", "context.rpm"]
    X = np.column_stack([first, second, first.copy(), near, constant,
                         np.full(rows, np.nan), rng.normal(size=rows), rng.normal(size=rows),
                         rng.normal(size=rows), rng.normal(size=rows), rng.normal(size=rows),
                         rng.normal(size=rows), rng.normal(size=rows), rng.normal(size=rows)]).astype(np.float32)
    targets = [f"fault_label.fixture_{label}" for label in range(16)]
    return X, y, names, targets


def test_constants_nan_duplicates_redundancy_and_original_mapping(selection_fixture, tmp_path):
    X, y, names, targets = selection_fixture
    before = X.copy()
    result = select_features(X, y, names, targets, tmp_path)
    np.testing.assert_array_equal(X, before)
    removed = {row["column"]: row for row in _csv(tmp_path / "constant_duplicates.csv")}
    assert removed["engine.load_pct"]["reason"] == "constant_nonmissing"
    assert removed["engine.fuel_rate_lph"]["reason"] == "all_missing"
    assert removed["engine.oil_temp_c"]["representative"] == "engine.oil_pressure_kpa"
    assert "engine.oil_temp_c" not in result["eligible_names"]
    assert {"engine.oil_pressure_kpa", "engine.coolant_temp_c"} <= set(result["eligible_names"])
    assert len({"engine.oil_pressure_kpa", "engine.coolant_temp_c"} & set(result["pruned_names"])) == 1
    correlations = _csv(tmp_path / "correlation_pruned.csv")
    pair = next(row for row in correlations if row["column"] in {"engine.oil_pressure_kpa", "engine.coolant_temp_c"})
    assert abs(float(pair["pearson"])) >= .95
    assert int(pair["overlap_rows"]) < len(X)
    for group in ("eligible", "pruned"):
        assert [names[index] for index in result[group + "_indices"]] == result[group + "_names"]
    assert set(result["ranking_names"]) == set(result["eligible_names"])
    for size in (8, 16, 32, 64):
        assert result["subsets"][str(size)] == result["subsets"]["correlation_pruned"][:size]
    assert result["subsets"]["all"] == result["ranking_names"]
    assert json.loads((tmp_path / "subsets.json").read_text()) == result["subsets"]
    assert json.loads((tmp_path / "selected_preliminary_ranks.json").read_text()) == result
    assert np.load(tmp_path / "training_feature_correlation.npy").shape == (len(result["eligible_names"]),) * 2


def test_label_diversity_evidence_and_proxy_ablation(selection_fixture, tmp_path):
    X, y, names, targets = selection_fixture
    result = select_features(X, y, names, targets, tmp_path)
    assert result["ranking_names"][0] in {"engine.oil_pressure_kpa", "engine.coolant_temp_c"}
    assert result["ranking_names"][1] == "engine.boost_kpa"
    per_label = _csv(tmp_path / "per_fault_correlations.csv")
    signal = next(row for row in per_label if row["target"] == targets[1] and row["column"] == "engine.boost_kpa")
    assert float(signal["pearson_pointbiserial"]) > .99
    assert len(per_label) == len(result["eligible_names"]) * 16
    importance = {row["column"]: row for row in _csv(tmp_path / "feature_importance.csv")}
    assert float(importance["engine.boost_kpa"][targets[1] + "__extra_trees"]) > .1
    assert float(importance["engine.boost_kpa"][targets[2] + "__extra_trees"]) == 0
    mi = _csv(tmp_path / "mutual_information.csv")
    assert max(float(row["mutual_information"]) for row in mi if row["target"] == targets[1]) > .4
    ablated = set(result["subsets"]["proxy_ablated"])
    assert not {"gnss.latitude_deg", "gnss.heading_deg", "engine.fuel_level_pct", "cumulative_fuel_energy_kwh"} & ablated
    assert {"lidar.obstacle_confidence_pct", "radar.target_confidence_pct", "gnss.accuracy_m"} <= ablated
    assert "truck-independent" in result["metadata"]["permutation_importance"]
    assert "Skipped" in result["metadata"]["shap"]


def test_reproducibility_and_bounded_training_sampling(selection_fixture, tmp_path):
    X, y, names, targets = selection_fixture
    first = select_features(X, y, names, targets, tmp_path / "first", max_fit_rows=120)
    second = select_features(X, y, names, targets, tmp_path / "second", max_fit_rows=120)
    assert first == second
    assert len(first["fit_row_indices"]) == 120
    assert set(first["mi_row_indices"]) <= set(first["fit_row_indices"])
    assert first["fit_row_indices"] == sorted(set(first["fit_row_indices"]))
    assert all(0 <= row < len(X) for row in first["fit_row_indices"])
    assert first["metadata"]["thread_limit"] == 2
    assert first["metadata"]["extra_trees"]["n_estimators_per_label"] == 64
    for filename in ("feature_importance.csv", "per_fault_correlations.csv", "mutual_information.csv"):
        assert (tmp_path / "first" / filename).read_bytes() == (tmp_path / "second" / filename).read_bytes()


@pytest.mark.parametrize("leak", ["fault_label.low_oil", "effect_level.low_oil", "truth.low_oil",
                                 "truck_id", "elapsed_s", "engine.overall_health", "engine.unknown_numeric",
                                 "sensor_quality_json", "lidar.diagnostic_confidence"])
def test_unknown_or_leaking_names_fail_closed(selection_fixture, tmp_path, leak):
    X, y, names, targets = selection_fixture
    names = names.copy()
    names[0] = leak
    with pytest.raises(ValueError, match="allowlist"):
        select_features(X, y, names, targets, tmp_path / "no_artifacts")
    assert not (tmp_path / "no_artifacts").exists()


@pytest.mark.parametrize("bad", ["float64", "infinite_x", "nan_y", "continuous_y", "wrong_labels", "duplicate_names"])
def test_input_validation(selection_fixture, tmp_path, bad):
    X, y, names, targets = selection_fixture
    X, y, names = X.copy(), y.astype(np.float32), names.copy()
    if bad == "float64":
        X = X.astype(np.float64)
    elif bad == "infinite_x":
        X[0, 0] = np.inf
    elif bad == "nan_y":
        y[0, 0] = np.nan
    elif bad == "continuous_y":
        y[0, 0] = .5
    elif bad == "wrong_labels":
        y = y[:, :3]
    else:
        names[0] = names[1]
    with pytest.raises(ValueError):
        select_features(X, y, names, targets, tmp_path)


def test_allowlist_matches_exporter_and_required_dependencies():
    from pathlib import Path
    schema = json.loads((Path(__file__).parent / "datasets/dump_truck_enhanced_all_conditions.schema.json").read_text())
    assert set(schema["input_columns"]) == set(allowed_input_names())
    assert "lidar.obstacle_confidence_pct__residual" in allowed_input_names()
    deps = sensor_dependencies("engine.oil_pressure_kpa__residual_slope_120s")
    assert deps["sensors"] == ["engine.oil_pressure_kpa", "frame.strain_microstrain"]
    assert deps["context"] and deps["history"]
    lead = sensor_dependencies("leadlag.engine.oil_pressure_kpa__to__engine.oil_temp_c_s")
    assert lead["sensors"] == ["engine.oil_pressure_kpa", "frame.strain_microstrain", "engine.oil_temp_c"]
    assert len(sensor_dependencies("engine.egt_spread_c")["sensors"]) == 6
    assert "engine.coolant_energy_flow_kw" in sensor_dependencies("engine.coolant_temp_c__residual")["sensors"]
    assert "brake.pedal_pct" in sensor_dependencies("brake.stroke_pct__residual")["sensors"]


def test_all_16_labels_have_independent_signal_and_equal_ranking_voice(tmp_path):
    rng = np.random.default_rng(9)
    y = rng.binomial(1, .3, size=(320, 16)).astype(np.int8)
    from enhanced_analytics import SENSOR_COLUMNS
    names = list(SENSOR_COLUMNS[:16])
    X = (y * 10 + rng.normal(0, .03, size=y.shape)).astype(np.float32)
    targets = [f"fault_label.{index}" for index in range(16)]
    result = select_features(X, y, names, targets, tmp_path)
    assert result["ranking_names"] == names
    assert all(record["scored"] for record in result["metadata"]["label_support"])
    importance = {row["column"]: row for row in _csv(tmp_path / "feature_importance.csv")}
    for label, target in enumerate(targets):
        strongest = max(names, key=lambda name: float(importance[name][target + "__extra_trees"]))
        assert strongest == names[label]


def test_pairwise_nan_correlation_is_stable_with_large_absolute_offsets():
    from analysis.fault_features import _pairwise_pearson
    first = (100_000_000 + np.arange(80) * 8).astype(np.float32)
    second = (-first * 2).astype(np.float32)
    first[::9], second[::7] = np.nan, np.nan
    X = np.column_stack([first, second])
    correlation, overlap = _pairwise_pearson(X)
    assert correlation[0, 1] == pytest.approx(-1., abs=1e-12)
    assert overlap[0, 1] == int((np.isfinite(first) & np.isfinite(second)).sum())


def test_all_ineligible_or_single_class_is_serializable(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    X[:, 1] = np.nan
    y = np.zeros((8, 16), dtype=np.int8)
    result = select_features(X, y, ["engine.oil_pressure_kpa", "engine.oil_temp_c"],
                             [f"target_{index}" for index in range(16)], tmp_path)
    assert result["eligible_names"] == []
    assert all(subset == [] for subset in result["subsets"].values())
    assert json.dumps(result, allow_nan=False)
