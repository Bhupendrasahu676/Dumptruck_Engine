"""Fixed input contract and portable 16-column inference checks."""
import csv
import io
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="Use isolated fault-training environment")
torch = pytest.importorskip("torch")
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from analysis import fault_dl as dl
from analysis.train_compact16 import FIXED_FEATURES, load_fixed_features, predict_compact_bundle, run, _validate_rows


def write_features(path, names):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attribute"])
        writer.writerows([[n] for n in names])


def test_frozen_features_exact_order_and_no_reselection(tmp_path):
    path = tmp_path / "features.csv"
    write_features(path, FIXED_FEATURES)
    assert load_fixed_features(path) == list(FIXED_FEATURES)
    for bad in (FIXED_FEATURES[:-1], list(reversed(FIXED_FEATURES)), [FIXED_FEATURES[0]] * 16,
                [*FIXED_FEATURES[:-1], "fault_label.low_oil"]):
        write_features(path, bad)
        with pytest.raises(ValueError):
            load_fixed_features(path)


def test_portable_dense_uses_only_requested16_and_keeps_empty_column(tmp_path, monkeypatch):
    monkeypatch.setenv("FAULT_ML_DEVICE", "cpu")
    frame = pd.DataFrame(np.random.default_rng(793).normal(size=(24, 16)), columns=FIXED_FEATURES)
    frame[FIXED_FEATURES[0]] = np.nan
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(frame)).astype(np.float32)
    assert z.shape == (24, 16)
    model = dl.DenseClassifier(16)
    memory = io.BytesIO()
    torch.save({"format_version": 1,
                "architecture": {"kind": "dense", "input_dim": 16, "seed": 793},
                "state_dict": model.state_dict(), "history": [], "best_epoch": 0}, memory)
    bundle = {"kind": "dense", "input_columns": list(FIXED_FEATURES), "imputer": imputer,
              "scaler": scaler, "model_path": "nonexistent/model.pt",
              "checkpoint_bytes": memory.getvalue(), "calibrators": [None] * 16}
    expected = predict_compact_bundle(bundle, frame)
    noisy = frame.assign(**{"truth.secret": 1, "engine.rpm": 9999, "health_score": 0})
    np.testing.assert_array_equal(predict_compact_bundle(bundle, noisy), expected)
    assert expected.shape == (24, 16)
    assert np.isfinite(expected).all()
    with pytest.raises(ValueError, match="Missing model inputs"):
        predict_compact_bundle(bundle, frame.drop(columns=[FIXED_FEATURES[1]]))


def test_existing_results_refused_without_writes(tmp_path):
    saved = tmp_path / "REPORT.md"
    saved.write_text("preserved")
    with pytest.raises(FileExistsError):
        run("not-opened.csv", tmp_path)
    assert saved.read_text() == "preserved"


def test_real_noninteger_cadence_accepts_mixed_ISO8601_fractional_precision():
    engines = {"train": ["a"], "validation": ["b", "c", "d"], "test": ["e"]}
    targets = [f"fault_label.fixture_{i}" for i in range(16)]
    rows = []
    for partition, ids in engines.items():
        for engine in ids:
            for step, timestamp in enumerate(("2026-01-01T00:00:00+00:00", "2026-01-01T00:00:10.588235+00:00")):
                rows.append({"engine_id": engine, "episode_id": engine+"-journey", "split": partition,
                             "step": step, "elapsed_s": step*1800/170, "timestamp": timestamp,
                             **{key: 0 for key in targets}})
    ordered = _validate_rows(pd.DataFrame(rows), targets, {"split_membership": engines, "counts": {"rows": len(rows)}})
    assert len(ordered) == 10
