"""PyTorch helpers for preprocessed, 16-label fault data.

Only training and validation rows are accepted here. Feature fitting, feature
order metadata, thresholds, calibration, and test evaluation belong to callers.
The autoencoder learns reconstruction on *all* supplied training rows, including
faults; its reconstruction error is not a calibrated anomaly probability.
"""
from __future__ import annotations

import csv
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


N_LABELS = 16
PATIENCE = 4


class DenseClassifier(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, N_LABELS),
        )

    def forward(self, x):
        return self.network(x)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32),
            nn.ReLU(), nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 64),
            nn.ReLU(), nn.Linear(64, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CausalConv1d(nn.Module):
    """Pad on both sides, then remove the right/future portion."""
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, 3,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[..., :-self.padding]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, dilation: int):
        super().__init__()
        self.network = nn.Sequential(
            CausalConv1d(in_channels, 32, dilation), nn.ReLU(),
            CausalConv1d(32, 32, dilation), nn.ReLU(),
        )
        self.residual = nn.Conv1d(in_channels, 32, 1) if in_channels != 32 else nn.Identity()

    def forward(self, x):
        return torch.relu(self.network(x) + self.residual(x))


class TCNClassifier(nn.Module):
    def __init__(self, input_dim: int, window: int = 12):
        super().__init__()
        self.input_dim, self.window = input_dim, window
        self.network = nn.Sequential(
            TemporalBlock(input_dim, 1), TemporalBlock(32, 2), TemporalBlock(32, 4),
        )
        self.head = nn.Linear(32, N_LABELS)

    def forward(self, x):
        return self.head(self.network(x)[..., -1])


def _features(X, *, input_dim=None, allow_empty=False):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("X must be a two-dimensional feature matrix with at least one feature")
    if not allow_empty and len(X) == 0:
        raise ValueError("Training and validation matrices must contain rows")
    if input_dim is not None and X.shape[1] != input_dim:
        raise ValueError("Feature count does not match the trained model")
    for start in range(0, len(X), 65536):
        if not np.isfinite(X[start:start + 65536]).all():
            raise ValueError("X must already be imputed and contain only finite values")
    return X


def _targets(y, n):
    y = np.asarray(y, dtype=np.float32)
    if y.shape != (n, N_LABELS):
        raise ValueError(f"Targets must have shape ({n}, {N_LABELS})")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Targets must be binary multilabel indicators")
    return y


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class CausalWindowDataset(Dataset):
    """One current-row target per lazy trailing window (channels, time).

    Short windows repeat their episode's first row on the left. Episode IDs
    must occupy contiguous runs; reappearing IDs are rejected rather than
    silently mixing separate runs. Storage is O(rows), not O(rows * window).
    """
    def __init__(self, X, episode_ids, y=None, window=12):
        _positive_int(window, "window")
        self.X = _features(X, allow_empty=True)
        ids = np.asarray(episode_ids)
        if ids.ndim != 1 or len(ids) != len(self.X):
            raise ValueError("episode_ids must have one ID per feature row")
        self.y = None if y is None else _targets(y, len(self.X))
        self.window = int(window)
        self.episode_starts = np.empty(len(ids), dtype=np.int64)
        seen = set()
        start = 0
        for index, episode in enumerate(ids):
            if index == 0 or episode != ids[index - 1]:
                if episode in seen:
                    raise ValueError("Episode IDs must be contiguous; an episode ID reappeared")
                seen.add(episode)
                start = index
            self.episode_starts[index] = start

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        indices = np.maximum(
            np.arange(index - self.window + 1, index + 1), self.episode_starts[index],
        )
        x = torch.from_numpy(self.X[indices].T.copy())
        return x if self.y is None else (x, torch.from_numpy(self.y[index]))


def _device():
    requested = os.environ.get("FAULT_ML_DEVICE")
    device = torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("FAULT_ML_DEVICE must select cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


@contextmanager
def _training_context(seed, device):
    """Restore caller RNG, thread count, and CUDA backend settings on exit."""
    old_threads = torch.get_num_threads()
    old_benchmark = torch.backends.cudnn.benchmark
    old_deterministic = torch.backends.cudnn.deterministic
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
    try:
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=devices):
            # Seed CPU explicitly: torch.manual_seed also mutates unused CUDA RNGs.
            torch.random.default_generator.manual_seed(int(seed))
            for index in devices:
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(int(seed))
            yield
    finally:
        torch.set_num_threads(old_threads)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_deterministic


def _loader(dataset, batch_size, *, shuffle=False, seed=793):
    _positive_int(batch_size, "batch_size")
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle,
                      num_workers=0, generator=generator)


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _macro_ap(y, prediction):
    # Explicitly define AP for absent positives, avoiding sklearn warning spam.
    scores = [float(average_precision_score(y[:, i], prediction[:, i]))
              if np.any(y[:, i]) else 0.0 for i in range(N_LABELS)]
    return float(np.mean(scores))


def _save(model, metadata, history, output_dir, name, best_epoch, device):
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Model name must be a simple directory name")
    destination = Path(output_dir) / name
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.pt"
    with (destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save({"format_version": 1, "architecture": metadata,
                "state_dict": _cpu_state(model), "best_epoch": best_epoch,
                "history": history}, model_path)
    model.eval()
    result = {"model": model, "history": history, "best_epoch": best_epoch,
              "device": str(device), "model_path": str(model_path)}
    if isinstance(model, TCNClassifier):
        result["window"] = model.window
    return result


def _classifier_validation(model, loader, criterion, device, n):
    scores = np.empty((n, N_LABELS), dtype=np.float32)
    offset, total = 0, 0.0
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += float(criterion(logits, y).item()) * len(x)
            scores[offset:offset + len(x)] = logits.sigmoid().cpu().numpy()
            offset += len(x)
    return total / n, scores


def _train_classifier(model, train_dataset, val_dataset, y_train, y_val,
                      output_dir, name, seed, epochs, batch_size, device, metadata):
    _positive_int(epochs, "epochs")
    train_loader = _loader(train_dataset, batch_size, shuffle=True, seed=seed)
    val_loader = _loader(val_dataset, batch_size)
    positives = y_train.sum(axis=0, dtype=np.float64)
    pos_weight = np.clip((len(y_train) - positives) / np.maximum(positives, 1), 1, 15)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score, best_epoch, best_state, stale = -float("inf"), 0, None, 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(x)
        val_loss, scores = _classifier_validation(model, val_loader, criterion, device, len(y_val))
        val_ap = _macro_ap(y_val, scores)
        history.append({"epoch": epoch, "train_loss": total / len(y_train),
                        "val_loss": val_loss, "val_macro_ap": val_ap})
        if val_ap > best_score:
            best_score, best_epoch, best_state, stale = val_ap, epoch, _cpu_state(model), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(best_state)
    metadata.update({"seed": int(seed), "pos_weight": pos_weight.tolist(),
                     "selection_metric": "val_macro_ap"})
    return _save(model, metadata, history, output_dir, name, best_epoch, device)


def train_dense(X_train, y_train, X_val, y_val, output_dir, name="dense",
                seed=793, epochs=18, batch_size=1024):
    X_train = _features(X_train)
    X_val = _features(X_val, input_dim=X_train.shape[1])
    y_train, y_val = _targets(y_train, len(X_train)), _targets(y_val, len(X_val))
    device = _device()
    with _training_context(seed, device):
        return _train_classifier(
            DenseClassifier(X_train.shape[1]),
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            y_train, y_val, output_dir, name, seed, epochs, batch_size, device,
            {"kind": "dense", "input_dim": X_train.shape[1], "n_outputs": N_LABELS},
        )


def train_tcn(X_train, y_train, train_episode_ids, X_val, y_val, val_episode_ids,
              output_dir, seed=793, epochs=18, batch_size=256, window=12):
    X_train = _features(X_train)
    X_val = _features(X_val, input_dim=X_train.shape[1])
    y_train, y_val = _targets(y_train, len(X_train)), _targets(y_val, len(X_val))
    train_dataset = CausalWindowDataset(X_train, train_episode_ids, y_train, window)
    val_dataset = CausalWindowDataset(X_val, val_episode_ids, y_val, window)
    device = _device()
    with _training_context(seed, device):
        return _train_classifier(
            TCNClassifier(X_train.shape[1], int(window)), train_dataset, val_dataset,
            y_train, y_val, output_dir, "tcn", seed, epochs, batch_size, device,
            {"kind": "tcn", "input_dim": X_train.shape[1], "n_outputs": N_LABELS,
             "window": int(window), "channels": 32, "dilations": [1, 2, 4]},
        )


def train_autoencoder(X_train, X_val, output_dir, seed=793, latent_dim=16,
                      epochs=15, batch_size=1024):
    X_train = _features(X_train)
    X_val = _features(X_val, input_dim=X_train.shape[1])
    _positive_int(latent_dim, "latent_dim")
    _positive_int(epochs, "epochs")
    device = _device()
    with _training_context(seed, device):
        model = Autoencoder(X_train.shape[1], int(latent_dim)).to(device)
        train_loader = _loader(TensorDataset(torch.from_numpy(X_train)), batch_size, shuffle=True, seed=seed)
        val_loader = _loader(TensorDataset(torch.from_numpy(X_val)), batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()
        best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
        history = []
        for epoch in range(1, int(epochs) + 1):
            model.train()
            total = 0.0
            for (x,) in train_loader:
                x = x.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), x)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(x)
            model.eval()
            val_total = 0.0
            with torch.inference_mode():
                for (x,) in val_loader:
                    x = x.to(device)
                    val_total += float(criterion(model(x), x).item()) * len(x)
            val_loss = val_total / len(X_val)
            history.append({"epoch": epoch, "train_loss": total / len(X_train), "val_loss": val_loss})
            if val_loss < best_loss:
                best_loss, best_epoch, best_state, stale = val_loss, epoch, _cpu_state(model), 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        model.load_state_dict(best_state)
        return _save(model, {"kind": "autoencoder", "input_dim": X_train.shape[1],
                            "latent_dim": int(latent_dim), "seed": int(seed),
                            "selection_metric": "val_reconstruction_mse"},
                     history, output_dir, "autoencoder", best_epoch, device)


def _model(result_or_model, expected):
    model = result_or_model["model"] if isinstance(result_or_model, dict) else result_or_model
    if not isinstance(model, expected):
        raise TypeError(f"Expected a {expected.__name__} or its training result")
    return model


def _infer(model, dataset, output_dim, batch_size, *, encoder=False):
    device = next(model.parameters()).device
    loader = _loader(dataset, batch_size)
    predictions = np.empty((len(dataset), output_dim), dtype=np.float32)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            offset = 0
            for x in loader:
                if isinstance(x, (tuple, list)):
                    x = x[0]
                x = x.to(device)
                output = model.encoder(x) if encoder else model(x).sigmoid()
                predictions[offset:offset + len(x)] = output.cpu().numpy()
                offset += len(x)
    finally:
        model.train(was_training)
    return predictions


def predict_dense(result_or_model, X, batch_size=2048):
    model = _model(result_or_model, DenseClassifier)
    X = _features(X, input_dim=model.input_dim, allow_empty=True)
    return _infer(model, TensorDataset(torch.from_numpy(X)), N_LABELS, batch_size)


def encode(result_or_model, X):
    model = _model(result_or_model, Autoencoder)
    X = _features(X, input_dim=model.input_dim, allow_empty=True)
    return _infer(model, TensorDataset(torch.from_numpy(X)), model.latent_dim, 2048, encoder=True)


def predict_tcn(result_or_model, X, episode_ids, batch_size=512):
    model = _model(result_or_model, TCNClassifier)
    X = _features(X, input_dim=model.input_dim, allow_empty=True)
    return _infer(model, CausalWindowDataset(X, episode_ids, window=model.window), N_LABELS, batch_size)


def load_model(path):
    """Load native architecture metadata and tensor weights without pickle code.

    Checkpoints are mapped to CPU first, then moved to FAULT_ML_DEVICE (or the
    available default). No optimizer state or external feature order is loaded.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported fault model checkpoint version")
    metadata = checkpoint["architecture"]
    kind, input_dim = metadata["kind"], metadata["input_dim"]
    device = _device()
    # Model initialization must not alter the caller's RNG either.
    with _training_context(metadata.get("seed", 793), torch.device("cpu")):
        if kind == "dense":
            model = DenseClassifier(input_dim)
        elif kind == "tcn":
            model = TCNClassifier(input_dim, metadata["window"])
        elif kind == "autoencoder":
            model = Autoencoder(input_dim, metadata["latent_dim"])
        else:
            raise ValueError(f"Unsupported architecture: {kind}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    result = {"model": model, "history": checkpoint["history"],
              "best_epoch": checkpoint["best_epoch"], "device": str(device),
              "model_path": str(path)}
    if kind == "tcn":
        result["window"] = model.window
    return result
