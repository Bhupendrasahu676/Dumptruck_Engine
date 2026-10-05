"""Small checks for the all-combination dataset generator."""

from generate_dataset import FAULT_NAMES, iter_rows


def test_dataset_combinations_have_171_samples_each() -> None:
    rows = iter_rows(seed=793)
    counts = {}
    for row in rows:
        counts[row["combination_id"]] = counts.get(row["combination_id"], 0) + 1
    assert len(counts) == 3 * 5 * 2 * (2 ** len(FAULT_NAMES))
    assert set(counts.values()) == {171}


def test_dataset_rows_include_health_rul_confidence_outputs() -> None:
    row = next(iter_rows(seed=793))
    for field in (
        "engine_health_score",
        "engine_health_classification",
        "rul_hours",
        "rul_low_hours",
        "rul_high_hours",
        "health_confidence",
        "health_confidence_pct",
        "recommended_action",
    ):
        assert field in row
