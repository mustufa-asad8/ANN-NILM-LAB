"""Train and evaluate two ANN models for household NILM."""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGION_FILES = {
    "newyork": PROJECT_ROOT / "data" / "raw" / "new_york" / "readings_1min.csv",
    "california": PROJECT_ROOT / "data" / "raw" / "california" / "readings_1min.csv",
    "austin": PROJECT_ROOT / "data" / "raw" / "austin" / "readings_1min.csv",
}
CHECKPOINT_FILES = {"MLP": "mlp_best.pt", "1D-CNN": "cnn_1d_best.pt"}
APPLIANCES = ("air1", "furnace1", "drye1")
APPLIANCE_NAMES = {
    "air1": "Air compressor circuit",
    "furnace1": "Furnace and air handler",
    "drye1": "Electric clothes dryer",
}
DEFAULT_HOMES = (
    7951,
    8565,
    2361,
    6139,
    7536,
    8156,
    9019,
    4767,
    3456,
    7901,
    4373,
    2335,
    9922,
    9278,
    661,
)
DEFAULT_TEST_HOME = 7951
POWER_COLUMNS = ("grid", *APPLIANCES)


@dataclass
class WindowSet:
    features: np.ndarray
    status: np.ndarray
    power_kw: np.ndarray
    home_ids: np.ndarray
    timestamps: np.ndarray


class MultiTaskMLP(nn.Module):
    def __init__(self, window_size: int, appliance_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.status_head = nn.Linear(64, appliance_count)
        self.power_head = nn.Linear(64, appliance_count)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return self.status_head(encoded), torch.nn.functional.softplus(
            self.power_head(encoded)
        )


class MultiTaskCNN(nn.Module):
    def __init__(self, appliance_count: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.status_head = nn.Linear(64, appliance_count)
        self.power_head = nn.Linear(64, appliance_count)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x.unsqueeze(1))
        return self.status_head(encoded), torch.nn.functional.softplus(
            self.power_head(encoded)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP and 1D-CNN NILM models on Pecan Street 1-minute data."
    )
    parser.add_argument("--region", choices=REGION_FILES, default="austin")
    parser.add_argument("--csv", type=Path, help="Override the regional CSV path.")
    parser.add_argument(
        "--home-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_HOMES),
        help="Five or more home IDs with all required appliance channels.",
    )
    parser.add_argument(
        "--test-home",
        type=int,
        default=DEFAULT_TEST_HOME,
        help="One complete home reserved for unseen-home testing.",
    )
    parser.add_argument("--window", type=int, default=120, help="Input history in minutes.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train-stride", type=int, default=20)
    parser.add_argument("--seed", type=int, default=466)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate existing checkpoints without training them again.",
    )
    return parser.parse_args()


def validate_evaluation_setup(args: argparse.Namespace) -> None:
    metadata_path = args.output_dir / "metadata" / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Cannot evaluate checkpoints without the original run metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Run metadata must contain a JSON object: {metadata_path}")

    requested = {
        "region": args.region,
        "home_ids": args.home_ids,
        "test_home": args.test_home,
        "window_minutes": args.window,
        "train_stride": args.train_stride,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    mismatches = [
        f"{key}: saved={metadata.get(key)!r}, requested={value!r}"
        for key, value in requested.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "--evaluate-only requires the original training configuration. "
            "Use the saved run settings or train new checkpoints. Mismatches: "
            + "; ".join(mismatches)
        )


def validate_checkpoint_preprocessing(
    checkpoint: dict[str, Any],
    input_mean: float,
    input_std: float,
    thresholds: dict[str, float],
    model_name: str,
) -> None:
    stored_mean = checkpoint.get("input_mean_kw")
    stored_std = checkpoint.get("input_std_kw")
    if (
        stored_mean is None
        or stored_std is None
        or not np.isclose(float(stored_mean), input_mean, rtol=1e-6, atol=1e-8)
        or not np.isclose(float(stored_std), input_std, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError(
            f"{model_name} checkpoint preprocessing does not match the selected "
            "training data; refusing to report incompatible evaluation metrics."
        )

    stored_thresholds = checkpoint.get("on_thresholds_kw")
    if not isinstance(stored_thresholds, dict) or any(
        appliance not in stored_thresholds
        or not np.isclose(
            float(stored_thresholds[appliance]),
            thresholds[appliance],
            rtol=1e-6,
            atol=1e-8,
        )
        for appliance in APPLIANCES
    ):
        raise ValueError(
            f"{model_name} checkpoint ON thresholds do not match the selected "
            "training data; refusing to report incompatible evaluation metrics."
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_selected_homes(csv_path: Path, home_ids: list[int]) -> dict[int, pd.DataFrame]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path.resolve()}")

    columns = ["dataid", "localminute", *POWER_COLUMNS]
    dtypes = {"dataid": "int32", **{column: "float32" for column in POWER_COLUMNS}}
    chunks: dict[int, list[pd.DataFrame]] = {home_id: [] for home_id in home_ids}
    for frame in pd.read_csv(
        csv_path,
        usecols=columns,
        dtype=dtypes,
        chunksize=500_000,
        low_memory=False,
    ):
        selected = frame[frame["dataid"].isin(home_ids)]
        if selected.empty:
            continue
        for home_id, home_frame in selected.groupby("dataid", sort=False):
            chunks[int(home_id)].append(home_frame.drop(columns="dataid"))

    homes: dict[int, pd.DataFrame] = {}
    missing_homes = [home_id for home_id, parts in chunks.items() if not parts]
    if missing_homes:
        raise ValueError(f"No data rows found for home IDs: {missing_homes}")

    for home_id, parts in chunks.items():
        frame = pd.concat(parts, ignore_index=True)
        frame["localminute"] = pd.to_datetime(
            frame["localminute"], errors="coerce", utc=True
        )
        frame = frame.dropna(subset=["localminute"])
        frame = frame.groupby("localminute", as_index=False)[list(POWER_COLUMNS)].mean()
        frame = frame.sort_values("localminute").reset_index(drop=True)
        for column in POWER_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[list(APPLIANCES)] = frame[list(APPLIANCES)].clip(lower=0)
        homes[home_id] = frame
    return homes


def contiguous_segments(
    frame: pd.DataFrame, required_columns: tuple[str, ...]
) -> list[pd.DataFrame]:
    clean = frame.dropna(subset=["localminute", *required_columns]).copy()
    if clean.empty:
        return []
    gaps = clean["localminute"].diff().ne(pd.Timedelta(minutes=1))
    clean["_segment"] = gaps.cumsum()
    return [
        segment.drop(columns="_segment").reset_index(drop=True)
        for _, segment in clean.groupby("_segment", sort=False)
    ]


def select_train_validation_segments(
    segments: list[pd.DataFrame], window_size: int
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    total_rows = sum(len(segment) for segment in segments)
    if total_rows < window_size * 3:
        raise ValueError("A training home has too few complete rows for the 60-minute window.")

    train: list[pd.DataFrame] = []
    validation: list[pd.DataFrame] = []
    test: list[pd.DataFrame] = []
    train_target = int(total_rows * 0.70)
    validation_target = int(total_rows * 0.15)
    train_rows = 0
    validation_rows = 0
    for segment in segments:
        remaining = segment
        if train_rows < train_target:
            take = min(len(remaining), train_target - train_rows)
            train.append(remaining.iloc[:take].copy())
            remaining = remaining.iloc[take:]
            train_rows += take
        if len(remaining) and validation_rows < validation_target:
            take = min(len(remaining), validation_target - validation_rows)
            validation.append(remaining.iloc[:take].copy())
            remaining = remaining.iloc[take:]
            validation_rows += take
        if len(remaining):
            test.append(remaining.copy())
    if not train or not validation:
        raise ValueError("Could not create non-empty chronological train and validation splits.")
    if not test:
        raise ValueError("Could not create a non-empty chronological test split.")
    return train, validation, test


def training_thresholds(train_frames: list[pd.DataFrame]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for appliance in APPLIANCES:
        values = np.concatenate(
            [frame[appliance].to_numpy(dtype=np.float32) for frame in train_frames]
        )
        positive_values = values[values > 0.02]
        if not len(positive_values):
            raise ValueError(f"No positive training readings for {appliance}.")
        thresholds[appliance] = max(0.05, float(np.quantile(positive_values, 0.10)))
    return thresholds


def make_windows(
    frames: list[tuple[int, pd.DataFrame]],
    thresholds: dict[str, float],
    input_mean: float,
    input_std: float,
    window_size: int,
    stride: int,
) -> WindowSet:
    x_parts: list[np.ndarray] = []
    status_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    home_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    threshold_values = np.array([thresholds[name] for name in APPLIANCES], dtype=np.float32)

    for home_id, frame in frames:
        if len(frame) < window_size:
            continue
        input_power = frame["grid"].to_numpy(dtype=np.float32)
        power_targets = frame[list(APPLIANCES)].to_numpy(dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(input_power, window_size)
        target_indices = np.arange(window_size - 1, len(frame), dtype=np.int64)[::stride]
        x_parts.append(((windows[::stride] - input_mean) / input_std).astype(np.float32))
        target_power = power_targets[target_indices]
        power_parts.append(target_power)
        status_parts.append((target_power >= threshold_values).astype(np.float32))
        home_parts.append(np.full(len(target_indices), home_id, dtype=np.int32))
        timestamp_parts.append(
            frame["localminute"].to_numpy(dtype="datetime64[ns]")[target_indices]
        )

    if not x_parts:
        raise ValueError("No complete windows could be generated for a data split.")
    return WindowSet(
        features=np.concatenate(x_parts),
        status=np.concatenate(status_parts),
        power_kw=np.concatenate(power_parts),
        home_ids=np.concatenate(home_parts),
        timestamps=np.concatenate(timestamp_parts),
    )


def build_model(name: str, window_size: int) -> nn.Module:
    if name == "MLP":
        return MultiTaskMLP(window_size, len(APPLIANCES))
    if name == "1D-CNN":
        return MultiTaskCNN(len(APPLIANCES))
    raise ValueError(f"Unknown model: {name}")


def predict(
    model: nn.Module, data: WindowSet, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    power_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data.features), batch_size):
            features = torch.from_numpy(data.features[start : start + batch_size]).to(device)
            logits, power = model(features)
            logits_parts.append(logits.cpu().numpy())
            power_parts.append(power.cpu().numpy())
    return np.concatenate(logits_parts), np.concatenate(power_parts)


def tune_status_thresholds(
    logits: np.ndarray, validation_data: WindowSet
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    selected: dict[str, float] = {}
    validation_metrics: dict[str, dict[str, float]] = {}

    for index, appliance in enumerate(APPLIANCES):
        labels = validation_data.status[:, index].astype(np.int32)
        best_key: tuple[float, float, float] | None = None
        best_threshold = 0.5
        best_scores = (0.0, 0.0)
        for threshold in candidate_thresholds:
            predictions = (probabilities[:, index] >= threshold).astype(np.int32)
            macro_f1 = f1_score(
                labels, predictions, average="macro", zero_division=0
            )
            balanced_accuracy = balanced_accuracy_score(labels, predictions)
            key = (float(macro_f1), float(balanced_accuracy), -abs(threshold - 0.5))
            if best_key is None or key > best_key:
                best_key = key
                best_threshold = float(threshold)
                best_scores = (float(macro_f1), float(balanced_accuracy))
        selected[appliance] = round(best_threshold, 3)
        validation_metrics[APPLIANCE_NAMES[appliance]] = {
            "probability_threshold": best_threshold,
            "macro_f1": best_scores[0],
            "balanced_accuracy": best_scores[1],
        }

    return selected, validation_metrics


def train_model(
    name: str,
    train_data: WindowSet,
    validation_data: WindowSet,
    window_size: int,
    input_mean: float,
    input_std: float,
    thresholds: dict[str, float],
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = build_model(name, window_size).to(device)
    positive = train_data.status.sum(axis=0)
    negative = len(train_data.status) - positive
    if np.any(positive == 0):
        raise ValueError(f"Training split contains no ON labels for {APPLIANCES}.")
    pos_weight = torch.tensor(
        np.clip(negative / positive, 1.0, 50.0), dtype=torch.float32, device=device
    )
    status_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    power_loss = nn.SmoothL1Loss(beta=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_tensors = TensorDataset(
        torch.from_numpy(train_data.features),
        torch.from_numpy(train_data.status),
        torch.from_numpy(train_data.power_kw),
    )
    validation_tensors = TensorDataset(
        torch.from_numpy(validation_data.features),
        torch.from_numpy(validation_data.status),
        torch.from_numpy(validation_data.power_kw),
    )
    train_loader = DataLoader(
        train_tensors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(validation_tensors, batch_size=batch_size, shuffle=False)

    best_validation_key: tuple[float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    log: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for features, status, power in train_loader:
            features = features.to(device)
            status = status.to(device)
            power = power.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, estimated_power = model(features)
            loss = status_loss(logits, status) + 0.25 * power_loss(estimated_power, power)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(features)
            train_count += len(features)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        validation_logits_parts: list[np.ndarray] = []
        with torch.no_grad():
            for features, status, power in validation_loader:
                logits, estimated_power = model(features.to(device))
                loss = status_loss(logits, status.to(device)) + 0.25 * power_loss(
                    estimated_power, power.to(device)
                )
                validation_total += loss.item() * len(features)
                validation_count += len(features)
                validation_logits_parts.append(logits.cpu().numpy())

        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
        validation_logits = np.concatenate(validation_logits_parts)
        _, calibration_metrics = tune_status_thresholds(
            validation_logits, validation_data
        )
        validation_macro_f1 = float(
            np.mean([values["macro_f1"] for values in calibration_metrics.values()])
        )
        validation_balanced_accuracy = float(
            np.mean(
                [values["balanced_accuracy"] for values in calibration_metrics.values()]
            )
        )
        validation_key = (validation_macro_f1, validation_balanced_accuracy)
        log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_macro_f1,
                "validation_balanced_accuracy": validation_balanced_accuracy,
            }
        )
        print(
            f"{name} epoch {epoch:02d}/{epochs}: "
            f"train_loss={train_loss:.5f}, val_loss={validation_loss:.5f}, "
            f"val_macro_f1={validation_macro_f1:.4f}",
            flush=True,
        )
        if best_validation_key is None or validation_key > best_validation_key:
            best_validation_key = validation_key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"{name} did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    torch.save(
        {
            "model": name,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "window_size": window_size,
            "input_mean_kw": input_mean,
            "input_std_kw": input_std,
            "appliances": list(APPLIANCES),
            "on_thresholds_kw": thresholds,
        },
        output_dir / "checkpoints" / CHECKPOINT_FILES[name],
    )
    return model, log


def evaluate(
    name: str,
    logits: np.ndarray,
    predicted_power_kw: np.ndarray,
    test_data: WindowSet,
    thresholds: dict[str, float],
    status_probability_thresholds: dict[str, float],
    test_group: str,
) -> list[dict[str, Any]]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    probability_thresholds = np.array(
        [status_probability_thresholds[appliance] for appliance in APPLIANCES]
    )
    predicted_status = (probabilities >= probability_thresholds).astype(np.int32)
    actual_status = test_data.status.astype(np.int32)
    test_home_ids = ",".join(
        str(home_id) for home_id in np.unique(test_data.home_ids)
    )
    results: list[dict[str, Any]] = []
    for index, appliance in enumerate(APPLIANCES):
        results.append(
            {
                "model": name,
                "appliance": APPLIANCE_NAMES[appliance],
                "channel": appliance,
                "test_group": test_group,
                "threshold_w": round(thresholds[appliance] * 1000, 2),
                "test_home": test_home_ids,
                "status_probability_threshold": status_probability_thresholds[appliance],
                "on_minutes": int(actual_status[:, index].sum()),
                "off_minutes": int(len(actual_status) - actual_status[:, index].sum()),
                "accuracy": accuracy_score(
                    actual_status[:, index], predicted_status[:, index]
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    actual_status[:, index], predicted_status[:, index]
                ),
                "macro_f1": f1_score(
                    actual_status[:, index],
                    predicted_status[:, index],
                    average="macro",
                    zero_division=0,
                ),
                "mae_w": mean_absolute_error(
                    test_data.power_kw[:, index] * 1000,
                    predicted_power_kw[:, index] * 1000,
                ),
                "rmse_w": np.sqrt(
                    mean_squared_error(
                        test_data.power_kw[:, index] * 1000,
                        predicted_power_kw[:, index] * 1000,
                    )
                ),
            }
        )
    return results


def save_plots(
    output_dir: Path,
    test_data: WindowSet,
    thresholds: dict[str, float],
    status_probability_thresholds: dict[str, dict[str, float]],
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    test_group: str,
) -> None:
    figure, axes = plt.subplots(2, len(APPLIANCES), figsize=(15, 8))
    for appliance_index, appliance in enumerate(APPLIANCES):
        truth = test_data.status[:, appliance_index]
        for row, model_name in enumerate(("MLP", "1D-CNN")):
            logits, _ = predictions[model_name]
            probability = 1.0 / (
                1.0 + np.exp(-np.clip(logits[:, appliance_index], -80, 80))
            )
            cutoff = status_probability_thresholds[model_name][appliance]
            matrix = confusion_matrix(
                truth,
                (probability >= cutoff).astype(int),
                labels=[0, 1],
            )
            axis = axes[row, appliance_index]
            image = axis.imshow(matrix, cmap="Blues")
            for (true_status, predicted_status), count in np.ndenumerate(matrix):
                axis.text(predicted_status, true_status, str(count), ha="center", va="center")
            axis.set(
                title=f"{model_name} - {APPLIANCE_NAMES[appliance]}",
                xlabel="Predicted status",
                ylabel="Actual status",
                xticks=[0, 1],
                yticks=[0, 1],
            )
            figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(
        output_dir / "figures" / f"{test_group}_confusion_matrices.png", dpi=160
    )
    plt.close(figure)

    figure, axes = plt.subplots(len(APPLIANCES), 1, figsize=(15, 9), sharex=True)
    shown = min(1440, len(test_data.features))
    for index, (axis, appliance) in enumerate(zip(axes, APPLIANCES)):
        axis.plot(
            np.arange(shown),
            test_data.power_kw[:shown, index] * 1000,
            label="Actual",
            linewidth=1,
        )
        for model_name, (_, estimated_power) in predictions.items():
            axis.plot(
                np.arange(shown),
                estimated_power[:shown, index] * 1000,
                label=f"{model_name} estimate",
                alpha=0.8,
                linewidth=1,
            )
        axis.set_ylabel("Power (W)")
        axis.set_title(
            f"{APPLIANCE_NAMES[appliance]} (ON threshold {thresholds[appliance] * 1000:.0f} W)"
        )
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel(f"Minutes in {test_group.replace('_', ' ')} sequence")
    figure.tight_layout()
    figure.savefig(
        output_dir / "figures" / f"{test_group}_power_predictions.png", dpi=160
    )
    plt.close(figure)


def write_report(
    output_dir: Path,
    region: str,
    home_ids: list[int],
    test_home: int,
    thresholds: dict[str, float],
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    def metric_table(test_group: str) -> str:
        rows = [
            "| Model | Appliance | Accuracy | Balanced accuracy | Macro-F1 | Status cutoff | MAE (W) | RMSE (W) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        rows.extend(
            f"| {row['model']} | {row['appliance']} | {row['accuracy']:.3f} | "
            f"{row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['status_probability_threshold']:.3f} | "
            f"{row['mae_w']:.1f} | {row['rmse_w']:.1f} |"
            for row in results
            if row["test_group"] == test_group
        )
        return "\n".join(rows)

    threshold_lines = "\n".join(
        f"- {APPLIANCE_NAMES[channel]}: {threshold * 1000:.1f} W "
        f"({metadata['training_threshold_rule']})"
        for channel, threshold in thresholds.items()
    )
    count_rows = [
        "| Split | Appliance | ON examples | OFF examples |",
        "|---|---|---:|---:|",
    ]
    for split_name, counts in metadata["status_counts"].items():
        count_rows.extend(
            f"| {split_name} | {appliance} | {values['on']} | {values['off']} |"
            for appliance, values in counts.items()
        )
    calibration_rows = [
        "| Model | Appliance | Status cutoff | Validation macro-F1 | Validation balanced accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name, appliance_metrics in metadata[
        "validation_threshold_metrics"
    ].items():
        calibration_rows.extend(
            f"| {model_name} | {appliance} | "
            f"{values['probability_threshold']:.3f} | {values['macro_f1']:.3f} | "
            f"{values['balanced_accuracy']:.3f} |"
            for appliance, values in appliance_metrics.items()
        )
    def benchmark_check(test_group: str) -> tuple[str, list[str]]:
        test_rows = [row for row in results if row["test_group"] == test_group]
        detail_rows = [
            "| Model | Appliance | Accuracy >=85% | Balanced accuracy >=75% | Macro-F1 >=0.70 |",
            "|---|---|---:|---:|---:|",
        ]
        model_summaries: list[str] = []
        for model_name in sorted({str(row["model"]) for row in test_rows}):
            model_rows = [row for row in test_rows if row["model"] == model_name]
            for row in model_rows:
                accuracy_pass = row["accuracy"] >= 0.85
                balanced_pass = row["balanced_accuracy"] >= 0.75
                f1_pass = row["macro_f1"] >= 0.70
                detail_rows.append(
                    f"| {model_name} | {row['appliance']} | "
                    f"{'PASS' if accuracy_pass else 'FAIL'} | "
                    f"{'PASS' if balanced_pass else 'FAIL'} | "
                    f"{'PASS' if f1_pass else 'FAIL'} |"
                )
            misses = [
                row["appliance"]
                for row in model_rows
                if row["accuracy"] < 0.85
                or row["balanced_accuracy"] < 0.75
                or row["macro_f1"] < 0.70
            ]
            if misses:
                model_summaries.append(
                    f"- {model_name}: below at least one minimum for {', '.join(misses)}."
                )
            else:
                model_summaries.append(
                    f"- {model_name}: meets the minimum for all three appliances."
                )
        return "\n".join(detail_rows), model_summaries

    chronological_benchmarks, chronological_summary = benchmark_check(
        "chronological_test"
    )
    unseen_benchmarks, unseen_summary = benchmark_check("unseen_home")

    def average_macro_f1(test_group: str, model_name: str) -> float:
        values = [
            float(row["macro_f1"])
            for row in results
            if row["test_group"] == test_group and row["model"] == model_name
        ]
        return float(np.mean(values))

    model_names = sorted({str(row["model"]) for row in results})
    best_chronological = max(
        model_names,
        key=lambda model_name: average_macro_f1("chronological_test", model_name),
    )
    best_unseen = max(
        model_names,
        key=lambda model_name: average_macro_f1("unseen_home", model_name),
    )
    def passing_models(test_group: str) -> list[str]:
        return [
            model_name
            for model_name in model_names
            if all(
                row["accuracy"] >= 0.85
                and row["balanced_accuracy"] >= 0.75
                and row["macro_f1"] >= 0.70
                for row in results
                if row["test_group"] == test_group and row["model"] == model_name
            )
            and any(
                row["test_group"] == test_group and row["model"] == model_name
                for row in results
            )
        ]

    chronological_passes = passing_models("chronological_test")
    unseen_passes = passing_models("unseen_home")
    benchmark_conclusion = (
        "Models meeting all appliance minimums: "
        f"chronological test: {', '.join(chronological_passes) or 'none'}; "
        f"unseen-home test: {', '.join(unseen_passes) or 'none'}."
    )
    comparison = (
        f"By average macro-F1 across appliances, {best_chronological} leads the "
        f"chronological test ({average_macro_f1('chronological_test', best_chronological):.3f}), "
        f"while {best_unseen} leads the unseen-home test "
        f"({average_macro_f1('unseen_home', best_unseen):.3f}). Individual "
        "appliance results vary; compare the tables and error plots for status "
        "recognition and power estimation separately."
    )
    text = f"""# NILM ANN CCP Results

## Project

Design and implementation of an ANN-based Non-Intrusive Load Monitoring system.
The models use only the whole-home `grid` meter history to estimate three
appliance ON/OFF states and appliance power. Appliance channels are targets,
never model inputs. Source readings are kW and reported power errors are W.

## Dataset and split

- Region file: {region}
- Selected homes: {", ".join(map(str, home_ids))}
- Training/validation homes: {", ".join(map(str, [hid for hid in home_ids if hid != test_home]))}
- Unseen-home test: {test_home} (complete held-out home; no training or threshold fitting)
- Input column: `grid`; target columns: {", ".join(f"`{appliance}`" for appliance in APPLIANCES)}
- Input window: {metadata['window_minutes']} consecutive one-minute whole-home readings
- Per development home: chronological 70% train, 15% validation, and 15%
  chronological test. The complete final home is a separate unseen-home test.
- Windows do not cross split or missing-data boundaries.
- Seed: {metadata['seed']}

## Appliance ON thresholds

Thresholds are computed using positive readings from training portions only,
then applied unchanged to validation and test.

{threshold_lines}

## Thresholded ON/OFF example counts

Training and validation counts use the configured training stride; test counts
include every scored minute.

{chr(10).join(count_rows)}

## Models and training

- Duplicate timestamps are averaged; rows missing a required meter/target value
  are excluded, and sequences are split at missing-minute gaps.
- Appliance targets are clipped at zero to remove negative sensor artifacts.
  Input normalization statistics are fit using training readings only.
- MLP: flattened {metadata['window_minutes']}-minute input, two hidden layers, shared encoder, status and
  power output heads.
- 1D-CNN: convolutional encoder over the {metadata['window_minutes']}-minute input, with the same two
  output tasks.
- Both models use class-weighted binary cross entropy plus smooth L1 power loss.
  The best checkpoint is selected by mean validation macro-F1, with mean
  balanced accuracy as a tie-breaker.
- Each model's ON/OFF probability cutoff is selected separately per appliance
  from 0.05 to 0.95 in 0.005 steps by maximizing validation macro-F1, with
  balanced accuracy as a tie-breaker.
  The cutoffs are then frozen before either test set is evaluated.
- Training samples use a stride of {metadata['train_stride']} minutes to reduce
  redundancy; both test sets are evaluated every minute.
- Maximum epochs: {metadata['max_epochs']}; early-stopping patience: {metadata['early_stopping_patience']}; batch size: {metadata['batch_size']}.

### Validation-only status cutoff calibration

{chr(10).join(calibration_rows)}

## Chronological test results

The final 15% of each of the {len([hid for hid in home_ids if hid != test_home])}
development homes was kept out of training and validation. Power MAE/RMSE
include appliance-OFF periods.

{metric_table('chronological_test')}

## Unseen-home results

Home {test_home} was excluded completely from training and validation.

{metric_table('unseen_home')}

### Assignment benchmarks

Classification targets are accuracy >= 0.85, balanced accuracy >= 0.75, and
macro-F1 >= 0.70 (excellent: 0.90, 0.85, and 0.80 respectively). Power error
has no fixed assignment target.

### Minimum benchmark check

#### Chronological test

{chronological_benchmarks}

{chr(10).join(chronological_summary)}

#### Unseen-home test

{unseen_benchmarks}

{chr(10).join(unseen_summary)}

## Comparison and conclusion

{comparison}

{benchmark_conclusion}

## Figures and training log

- `artifacts/figures/chronological_test_confusion_matrices.png`
- `artifacts/figures/chronological_test_power_predictions.png`
- `artifacts/figures/unseen_home_confusion_matrices.png`
- `artifacts/figures/unseen_home_power_predictions.png`
- `artifacts/logs/training_log.csv`

## Reproducibility and limitations

Run `python src/nilm_ccp/train.py --region austin`. The selected cohort and settings can
be overridden from the command line. The chronological test measures later-time
performance on the development homes; the unseen-home test measures transfer to
one unseen home, not all Pecan Street homes. `grid` is the available whole-home
net-meter reading; it can be negative when a home exports solar power, so the
input is net grid power rather than gross appliance-sum consumption. This
distinction should be included when discussing limitations.
"""
    (output_dir / "reports" / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.home_ids) < 5:
        raise ValueError("The CCP requires at least five homes.")
    if len(set(args.home_ids)) != len(args.home_ids):
        raise ValueError("Home IDs must be unique.")
    if args.test_home not in args.home_ids:
        raise ValueError("--test-home must be included in --home-ids.")
    if args.window < 2 or args.train_stride < 1:
        raise ValueError("Window size must be >= 2 and train stride must be >= 1.")
    if args.evaluate_only:
        validate_evaluation_setup(args)

    seed_everything(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = args.csv or REGION_FILES[args.region]
    print(f"Reading {csv_path.resolve()} (this scans the regional CSV once)...", flush=True)
    homes = load_selected_homes(csv_path, args.home_ids)
    print(
        "Loaded selected homes: "
        + ", ".join(f"{home_id}={len(frame):,} rows" for home_id, frame in homes.items()),
        flush=True,
    )

    train_frames: list[tuple[int, pd.DataFrame]] = []
    validation_frames: list[tuple[int, pd.DataFrame]] = []
    chronological_test_frames: list[tuple[int, pd.DataFrame]] = []
    unseen_test_frames: list[tuple[int, pd.DataFrame]] = []
    for home_id, frame in homes.items():
        segments = contiguous_segments(frame, POWER_COLUMNS)
        if home_id == args.test_home:
            unseen_test_frames.extend((home_id, segment) for segment in segments)
            continue
        train_segments, validation_segments, test_segments = select_train_validation_segments(
            segments, args.window
        )
        train_frames.extend((home_id, segment) for segment in train_segments)
        validation_frames.extend((home_id, segment) for segment in validation_segments)
        chronological_test_frames.extend((home_id, segment) for segment in test_segments)

    raw_train_frames = [frame for _, frame in train_frames]
    thresholds = training_thresholds(raw_train_frames)
    train_grid = np.concatenate(
        [frame["grid"].to_numpy(dtype=np.float32) for frame in raw_train_frames]
    )
    input_mean = float(train_grid.mean())
    input_std = float(train_grid.std())
    if not np.isfinite(input_std) or input_std < 1e-6:
        input_std = 1.0

    train_data = make_windows(
        train_frames,
        thresholds,
        input_mean,
        input_std,
        args.window,
        args.train_stride,
    )
    validation_data = make_windows(
        validation_frames, thresholds, input_mean, input_std, args.window, args.train_stride
    )
    chronological_test_data = make_windows(
        chronological_test_frames, thresholds, input_mean, input_std, args.window, 1
    )
    unseen_test_data = make_windows(
        unseen_test_frames, thresholds, input_mean, input_std, args.window, 1
    )
    if (
        len(np.unique(unseen_test_data.home_ids)) != 1
        or int(unseen_test_data.home_ids[0]) != args.test_home
    ):
        raise RuntimeError("The held-out test data contains an unexpected home.")

    for directory in ("checkpoints", "figures", "logs", "metadata", "metrics", "reports"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        "chronological_test": {},
        "unseen_home": {},
    }
    status_probability_thresholds: dict[str, dict[str, float]] = {}
    validation_threshold_metrics: dict[str, dict[str, dict[str, float]]] = {}
    log_rows: list[dict[str, Any]] = []
    for name in ("MLP", "1D-CNN"):
        checkpoint_path = args.output_dir / "checkpoints" / CHECKPOINT_FILES[name]
        if args.evaluate_only:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Cannot evaluate without the trained checkpoint: {checkpoint_path}"
                )
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint.get("model") != name or checkpoint.get("window_size") != args.window:
                raise ValueError(f"Checkpoint does not match the requested {name} setup.")
            validate_checkpoint_preprocessing(
                checkpoint,
                input_mean,
                input_std,
                thresholds,
                name,
            )
            model = build_model(name, args.window).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            log_path = args.output_dir / "logs" / "training_log.csv"
            if not log_path.is_file():
                raise FileNotFoundError(f"Training log not found: {log_path}")
            log_frame = pd.read_csv(log_path)
            model_log = log_frame[log_frame["model"] == name]
            if model_log.empty:
                raise ValueError(f"No training history found for {name} in {log_path}.")
            training_log = [
                {"epoch": int(row.epoch), "train_loss": row.train_loss,
                 "validation_loss": row.validation_loss,
                 **(
                     {"validation_macro_f1": row.validation_macro_f1,
                      "validation_balanced_accuracy": row.validation_balanced_accuracy}
                     if "validation_macro_f1" in log_frame.columns
                     else {}
                 )}
                for row in model_log.itertuples(index=False)
            ]
        else:
            model, training_log = train_model(
                name,
                train_data,
                validation_data,
                args.window,
                input_mean,
                input_std,
                thresholds,
                args.seed,
                args.epochs,
                args.batch_size,
                args.patience,
                device,
                args.output_dir,
            )
        validation_logits, _ = predict(
            model, validation_data, device, args.batch_size
        )
        model_thresholds, model_validation_metrics = tune_status_thresholds(
            validation_logits, validation_data
        )
        status_probability_thresholds[name] = model_thresholds
        validation_threshold_metrics[name] = model_validation_metrics
        print(
            f"{name} validation-selected status cutoffs: "
            + ", ".join(
                f"{APPLIANCE_NAMES[appliance]}={threshold:.2f}"
                for appliance, threshold in model_thresholds.items()
            ),
            flush=True,
        )
        for test_group, test_data in (
            ("chronological_test", chronological_test_data),
            ("unseen_home", unseen_test_data),
        ):
            logits, predicted_power = predict(model, test_data, device, args.batch_size)
            predictions[test_group][name] = (logits, predicted_power)
            all_results.extend(
                evaluate(
                    name,
                    logits,
                    predicted_power,
                    test_data,
                    thresholds,
                    model_thresholds,
                    test_group,
                )
            )
        log_rows.extend({"model": name, **row} for row in training_log)
        torch.save(
            {
                "model": name,
                "state_dict": model.state_dict(),
                "best_epoch": int(
                    max(
                        training_log,
                        key=lambda row: (
                            row.get("validation_macro_f1", -1.0),
                            row.get("validation_balanced_accuracy", -1.0),
                        ),
                    )["epoch"]
                ),
                "window_size": args.window,
                "input_mean_kw": input_mean,
                "input_std_kw": input_std,
                "appliances": list(APPLIANCES),
                "on_thresholds_kw": thresholds,
                "status_probability_thresholds": model_thresholds,
            },
            checkpoint_path,
        )

    results_frame = pd.DataFrame(all_results)
    results_frame.to_csv(args.output_dir / "metrics" / "metrics.csv", index=False)
    pd.DataFrame(log_rows).to_csv(args.output_dir / "logs" / "training_log.csv", index=False)
    for test_group, test_data in (
        ("chronological_test", chronological_test_data),
        ("unseen_home", unseen_test_data),
    ):
        save_plots(
            args.output_dir,
            test_data,
            thresholds,
            status_probability_thresholds,
            predictions[test_group],
            test_group,
        )

    metadata = {
        "region": args.region,
        "home_ids": args.home_ids,
        "test_home": args.test_home,
        "window_minutes": args.window,
        "train_stride": args.train_stride,
        "max_epochs": args.epochs,
        "early_stopping_patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(device),
        "python_torch_version": torch.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "input_mean_kw": input_mean,
        "input_std_kw": input_std,
        "training_threshold_rule": "max(50 W, 10th percentile of training readings >20 W)",
        "thresholds_kw": thresholds,
        "status_probability_thresholds": status_probability_thresholds,
        "validation_threshold_metrics": validation_threshold_metrics,
        "train_windows": len(train_data.features),
        "validation_windows": len(validation_data.features),
        "chronological_test_windows": len(chronological_test_data.features),
        "unseen_test_windows": len(unseen_test_data.features),
        "status_counts": {
            split_name: {
                APPLIANCE_NAMES[appliance]: {
                    "on": int(data.status[:, index].sum()),
                    "off": int(len(data.status) - data.status[:, index].sum()),
                }
                for index, appliance in enumerate(APPLIANCES)
            }
            for split_name, data in (
                ("train", train_data),
                ("validation", validation_data),
                ("chronological_test", chronological_test_data),
                ("unseen_test", unseen_test_data),
            )
        },
    }
    (args.output_dir / "metadata" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    write_report(
        args.output_dir,
        args.region,
        args.home_ids,
        args.test_home,
        thresholds,
        all_results,
        metadata,
    )
    result_columns = [
        "model",
        "appliance",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mae_w",
        "rmse_w",
    ]
    for test_group, title in (
        ("chronological_test", "Chronological test"),
        ("unseen_home", f"Unseen home {args.test_home} test"),
    ):
        print(f"\n{title} results:")
        print(
            results_frame.loc[results_frame["test_group"] == test_group, result_columns]
            .to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
    print(f"\nArtifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
