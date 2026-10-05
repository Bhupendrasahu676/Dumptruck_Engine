"""Small CPU regressions for native model checkpoints and causal windows."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest
pytest.importorskip("sklearn", reason="Use the isolated fault-training environment")
pytest.importorskip("torch", reason="PyTorch is required for DL helper checks")
import torch

from analysis.fault_dl import (
    CausalConv1d, CausalWindowDataset, DenseClassifier, TCNClassifier,
    encode, load_model, predict_dense, predict_tcn, train_autoencoder,
    train_dense, train_tcn,
)


class FaultDLTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.device = patch.dict(os.environ, {"FAULT_ML_DEVICE": "cpu"})
        self.device.start()
        self.addCleanup(self.device.stop)
        rng = np.random.default_rng(40)
        self.X = rng.normal(size=(48, 5)).astype(np.float32)
        self.val = rng.normal(size=(20, 5)).astype(np.float32)
        self.y = (rng.random((48, 16)) < .25).astype(np.float32)
        self.yval = (rng.random((20, 16)) < .25).astype(np.float32)
        self.y[:, 0], self.yval[:, 0] = 0, 0
        self.y[:, 1], self.yval[:, 1] = 1, 1
        self.ids = np.repeat(["a", "b", "c"], 16)
        self.valids = np.repeat(["d", "e"], 10)

    def test_lazy_window_boundaries_row_zero_and_targets(self):
        X = np.arange(12, dtype=np.float32).reshape(6, 2)
        y = np.zeros((6, 16), dtype=np.float32)
        y[3, 4] = 1
        data = CausalWindowDataset(X, ["a", "a", "a", "b", "b", "b"], y, window=4)
        self.assertEqual(len(data), 6)
        np.testing.assert_array_equal(data[0][0], np.repeat(X[0, :, None], 4, axis=1))
        np.testing.assert_array_equal(data[2][0], X[[0, 0, 1, 2]].T)
        np.testing.assert_array_equal(data[3][0], np.repeat(X[3, :, None], 4, axis=1))
        np.testing.assert_array_equal(data[4][0], X[[3, 3, 3, 4]].T)
        self.assertEqual(tuple(data[3][1].shape), (16,))
        np.testing.assert_array_equal(data[3][1], y[3])
        self.assertEqual(data.episode_starts.shape, (6,))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            CausalWindowDataset(X, ["a", "a", "b", "b", "a", "a"])
        with self.assertRaisesRegex(ValueError, "one ID"):
            CausalWindowDataset(X, ["a"])

    def test_future_rows_and_other_episode_cannot_change_predictions(self):
        model = TCNClassifier(5, window=4).eval()
        before = predict_tcn(model, self.X, self.ids, batch_size=8)
        changed = self.X.copy()
        changed[9:] += 1000
        after = predict_tcn(model, changed, self.ids, batch_size=8)
        np.testing.assert_array_equal(before[:9], after[:9])
        changed = self.X.copy()
        changed[:16] += 1000
        after = predict_tcn(model, changed, self.ids, batch_size=8)
        np.testing.assert_array_equal(before[16:], after[16:])
        # Direct convolution causality, independent of the window builder.
        conv = CausalConv1d(2, 3, dilation=2)
        x = torch.randn(1, 2, 12)
        changed_x = x.clone()
        changed_x[..., 7:] += 100
        torch.testing.assert_close(conv(x)[..., :7], conv(changed_x)[..., :7], rtol=0, atol=0)

    def _check_result(self, result):
        self.assertEqual(result["device"], "cpu")
        self.assertEqual(len(result["history"]), 2)
        self.assertIn(result["best_epoch"], (1, 2))
        path = Path(result["model_path"])
        self.assertTrue(path.is_file())
        self.assertTrue((path.parent / "history.csv").is_file())
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        self.assertTrue(all(value.device.type == "cpu" for value in checkpoint["state_dict"].values()))
        json.dumps(checkpoint["architecture"], allow_nan=False)
        json.dumps(result["history"], allow_nan=False)
        if checkpoint["architecture"]["kind"] == "autoencoder":
            best_index = min(range(2), key=lambda i: result["history"][i]["val_loss"])
        else:
            best_index = max(range(2), key=lambda i: result["history"][i]["val_macro_ap"])
        self.assertEqual(result["best_epoch"], best_index + 1)
        return load_model(path)

    def test_dense_two_epoch_cpu_train_reload_and_local_rng(self):
        rng_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        threads_before = torch.get_num_threads()
        result = train_dense(self.X, self.y, self.val, self.yval, self.temp.name,
                             epochs=2, batch_size=13)
        torch.testing.assert_close(rng_before, torch.random.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(threads_before, torch.get_num_threads())
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        loaded = self._check_result(result)
        scores = predict_dense(result, self.val, batch_size=7)
        self.assertEqual(scores.shape, (20, 16))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        np.testing.assert_array_equal(scores, predict_dense(loaded, self.val, batch_size=7))
        self.assertEqual(predict_dense(result["model"], self.val[:0]).shape, (0, 16))
        checkpoint = torch.load(result["model_path"], weights_only=True)
        weights = checkpoint["architecture"]["pos_weight"]
        self.assertEqual(weights[0], 15)
        self.assertEqual(weights[1], 1)
        self.assertTrue(all(1 <= weight <= 15 for weight in weights))
        second = train_dense(self.X, self.y, self.val, self.yval, self.temp.name,
                             name="repeat", epochs=2, batch_size=13)
        np.testing.assert_array_equal(scores, predict_dense(second, self.val, batch_size=7))

    def test_absent_labels_ap_early_stop_restores_first_checkpoint(self):
        y = np.zeros_like(self.y)
        yval = np.zeros_like(self.yval)
        result = train_dense(self.X, y, self.val, yval, self.temp.name,
                             name="stopped", epochs=8, batch_size=13)
        self.assertEqual(len(result["history"]), 5)
        self.assertEqual(result["best_epoch"], 1)
        self.assertTrue(all(row["val_macro_ap"] == 0 for row in result["history"]))
        first_epoch = train_dense(self.X, y, self.val, yval, self.temp.name,
                                  name="first", epochs=1, batch_size=13)
        np.testing.assert_array_equal(predict_dense(result, self.val),
                                      predict_dense(first_epoch, self.val))

    def test_autoencoder_two_epoch_cpu_train_reload(self):
        result = train_autoencoder(self.X, self.val, self.temp.name,
                                   epochs=2, batch_size=13, latent_dim=3)
        loaded = self._check_result(result)
        latent = encode(result, self.val)
        self.assertEqual(latent.shape, (20, 3))
        self.assertTrue(np.isfinite(latent).all())
        np.testing.assert_array_equal(latent, encode(loaded, self.val))
        self.assertEqual(encode(result["model"], self.val[:0]).shape, (0, 3))

    def test_tcn_two_epoch_cpu_train_reload(self):
        result = train_tcn(self.X, self.y, self.ids, self.val, self.yval, self.valids,
                           self.temp.name, epochs=2, batch_size=13, window=4)
        loaded = self._check_result(result)
        self.assertEqual(result["window"], 4)
        scores = predict_tcn(result, self.val, self.valids, batch_size=7)
        self.assertEqual(scores.shape, (20, 16))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())
        np.testing.assert_array_equal(scores, predict_tcn(loaded, self.val, self.valids, batch_size=7))
        self.assertEqual(predict_tcn(result["model"], self.val[:0], []).shape, (0, 16))

    def test_target_shape_feature_and_batch_validation(self):
        with self.assertRaisesRegex(ValueError, "Targets must have shape"):
            train_dense(self.X, self.y[:, :15], self.val, self.yval, self.temp.name, epochs=2)
        with self.assertRaisesRegex(ValueError, "Feature count"):
            train_autoencoder(self.X, self.val[:, :4], self.temp.name, epochs=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_dense(DenseClassifier(5), np.full((1, 5), np.nan))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            predict_dense(DenseClassifier(5), self.val, batch_size=0)
        with self.assertRaisesRegex(ValueError, "binary"):
            train_dense(self.X, self.y + .1, self.val, self.yval, self.temp.name, epochs=2)
        with self.assertRaisesRegex(ValueError, "contain rows"):
            train_dense(self.X[:0], self.y[:0], self.val, self.yval, self.temp.name, epochs=2)


if __name__ == "__main__":
    unittest.main()
