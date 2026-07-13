from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rcd.data import load_preprocessed_raw, set_path
from rcd.download import iter_expected_set_files
from rcd.manifest import write_manifest
from rcd.models.eegnet import EEGNet
from rcd.prediction import CLASSES, _assemble_features, _metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _extract_epoch_cache(
    *,
    project_root: Path,
    config: dict[str, Any],
    table: pd.DataFrame,
) -> tuple[np.ndarray, Path]:
    settings = config["datasets"]["ds004504"]
    preprocessing = config["preprocessing"]
    channels = config["study"]["channels"]["canonical"]
    epoch_seconds = float(preprocessing["model_epoch_seconds"])
    epoch_count = int(config["classification"]["epochs_per_subject"])
    samples = int(round(epoch_seconds * float(preprocessing["resample_hz"])))
    cache_dir = project_root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"eegnet_epochs_{epoch_seconds:g}s_{epoch_count}.npy"
    metadata_path = cache_path.with_suffix(".json")
    participant_ids = table["participant_id"].tolist()
    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_shape = [len(participant_ids), epoch_count, len(channels), samples]
        if (
            metadata.get("participant_ids") == participant_ids
            and metadata.get("shape") == expected_shape
        ):
            return np.load(cache_path, mmap_mode="r"), cache_path

    values = np.empty((len(participant_ids), epoch_count, len(channels), samples), dtype=np.float32)
    dataset_root = project_root / "data" / "ds004504"
    for participant_index, participant_id in enumerate(
        tqdm(participant_ids, desc="EEGNet epoch cache")
    ):
        raw = load_preprocessed_raw(
            set_path(
                dataset_root,
                participant_id,
                settings["task"],
                settings.get("derivative_pipeline"),
            ),
            channels,
            l_freq=float(preprocessing["l_freq"]),
            h_freq=float(preprocessing["h_freq"]),
            resample_hz=float(preprocessing["resample_hz"]),
            rereference=str(preprocessing["rereference"]),
        )
        maximum_start = raw.n_times - samples
        if maximum_start < 0:
            raise ValueError(f"{participant_id} is shorter than one model epoch")
        starts = np.linspace(0, maximum_start, epoch_count, dtype=int)
        for epoch_index, start in enumerate(starts):
            values[participant_index, epoch_index] = raw.get_data(
                start=int(start),
                stop=int(start + samples),
            ).astype(np.float32)
    np.save(cache_path, values)
    metadata_path.write_text(
        json.dumps(
            {
                "participant_ids": participant_ids,
                "shape": list(values.shape),
                "units": "volts",
                "selection": "evenly spaced fixed epochs across each released derivative",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return np.load(cache_path, mmap_mode="r"), cache_path


def _subject_probabilities(logits: torch.Tensor, n_subjects: int, n_epochs: int) -> np.ndarray:
    log_probability = torch.log_softmax(logits, dim=1).reshape(n_subjects, n_epochs, -1)
    mean_log_probability = log_probability.mean(dim=1)
    return torch.softmax(mean_log_probability, dim=1).cpu().numpy()


def _infer_eegnet(
    model: EEGNet,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_subjects, n_epochs = values.shape[:2]
    tensor = torch.from_numpy(np.asarray(values).reshape(-1, *values.shape[2:])).float()
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    logits: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            output = model(batch.to(device))
            logits.append(output.logits.cpu())
            embeddings.append(output.embedding.cpu())
    all_logits = torch.cat(logits)
    all_embeddings = torch.cat(embeddings).reshape(n_subjects, n_epochs, -1).mean(dim=1)
    return _subject_probabilities(all_logits, n_subjects, n_epochs), all_embeddings.numpy()


def _macro_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(
        roc_auc_score(y_true, probability, labels=CLASSES, multi_class="ovr", average="macro")
    )


def _train_eegnet(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[EEGNet, dict[str, Any]]:
    _seed_everything(seed)
    settings = config["classification"]
    n_epochs_per_subject = train_values.shape[1]
    model = EEGNet(n_channels=train_values.shape[2], n_classes=CLASSES.size).to(device)
    epoch_values = torch.from_numpy(
        np.asarray(train_values).reshape(-1, *train_values.shape[2:])
    ).float()
    label_indices = np.searchsorted(CLASSES, train_labels)
    epoch_labels = torch.from_numpy(np.repeat(label_indices, n_epochs_per_subject)).long()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(epoch_values, epoch_labels),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    subject_counts = np.bincount(label_indices, minlength=CLASSES.size)
    class_weights = subject_counts.sum() / (CLASSES.size * subject_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses: list[float] = []
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.to(device)).logits
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probability, _ = _infer_eegnet(
            model,
            validation_values,
            device=device,
            batch_size=int(settings["batch_size"]),
        )
        score = _macro_auc(validation_labels, validation_probability)
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_macro_auc": score}
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= int(settings["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_macro_auc": best_score,
        "epochs_run": len(history),
        "history": history,
    }


class FusionHead(nn.Module):
    def __init__(self, embedding_size: int, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_size + n_features, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, embedding: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([embedding, features], dim=1))


def _train_fusion_head(
    train_embedding: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_embedding: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[FusionHead, dict[str, Any]]:
    _seed_everything(seed)
    settings = config["classification"]
    model = FusionHead(train_embedding.shape[1], train_features.shape[1], CLASSES.size).to(device)
    label_indices = np.searchsorted(CLASSES, train_labels)
    subject_counts = np.bincount(label_indices, minlength=CLASSES.size)
    weights = torch.as_tensor(
        subject_counts.sum() / (CLASSES.size * subject_counts),
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensors = TensorDataset(
        torch.from_numpy(train_embedding).float(),
        torch.from_numpy(train_features).float(),
        torch.from_numpy(label_indices).long(),
    )
    loader = DataLoader(
        train_tensors,
        batch_size=min(32, len(train_tensors)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_embedding_tensor = torch.from_numpy(validation_embedding).float().to(device)
    validation_features_tensor = torch.from_numpy(validation_features).float().to(device)
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    without_improvement = 0
    for epoch in range(int(settings["fusion_head_max_epochs"])):
        model.train()
        for embedding, features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(embedding.to(device), features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(
                model(validation_embedding_tensor, validation_features_tensor), dim=1
            ).cpu().numpy()
        score = _macro_auc(validation_labels, probability)
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= int(settings["fusion_head_patience"]):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_macro_auc": best_score,
        "epochs_run": epoch + 1,
    }


def _stratified_bootstrap_difference(
    y_true: np.ndarray,
    raw_probability: np.ndarray,
    fusion_probability: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    class_indices = {label: np.flatnonzero(y_true == label) for label in CLASSES}
    raw_auc = np.empty(iterations)
    fusion_auc = np.empty(iterations)
    for iteration in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(class_indices[label], size=class_indices[label].size, replace=True)
                for label in CLASSES
            ]
        )
        raw_auc[iteration] = _macro_auc(y_true[indices], raw_probability[indices])
        fusion_auc[iteration] = _macro_auc(y_true[indices], fusion_probability[indices])
    difference = fusion_auc - raw_auc
    raw_interval = np.quantile(raw_auc, [0.025, 0.975])
    fusion_interval = np.quantile(fusion_auc, [0.025, 0.975])
    difference_interval = np.quantile(difference, [0.025, 0.975])
    observed_difference = _macro_auc(y_true, fusion_probability) - _macro_auc(
        y_true, raw_probability
    )
    permutation_difference = np.empty(iterations)
    for iteration in range(iterations):
        swap = rng.random(y_true.size) < 0.5
        permuted_raw = raw_probability.copy()
        permuted_fusion = fusion_probability.copy()
        permuted_raw[swap] = fusion_probability[swap]
        permuted_fusion[swap] = raw_probability[swap]
        permutation_difference[iteration] = _macro_auc(
            y_true, permuted_fusion
        ) - _macro_auc(y_true, permuted_raw)
    permutation_p = (
        1 + np.count_nonzero(np.abs(permutation_difference) >= abs(observed_difference))
    ) / (iterations + 1)
    return {
        "raw_macro_auc_ci95_low": float(raw_interval[0]),
        "raw_macro_auc_ci95_high": float(raw_interval[1]),
        "fusion_macro_auc_ci95_low": float(fusion_interval[0]),
        "fusion_macro_auc_ci95_high": float(fusion_interval[1]),
        "fusion_minus_raw_ci95_low": float(difference_interval[0]),
        "fusion_minus_raw_ci95_high": float(difference_interval[1]),
        "fusion_minus_raw_observed": float(observed_difference),
        "fusion_minus_raw_paired_permutation_p": float(permutation_p),
    }


def run_deep_prediction(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    repeats: int,
) -> tuple[Path, Path, Path]:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, feature_sets = _assemble_features(project_root)
    complexity_columns = feature_sets["complexity"]
    biomarkers = table[complexity_columns].to_numpy(dtype=np.float32)
    labels = table["diagnosis"].to_numpy()
    participant_ids = table["participant_id"].to_numpy()
    epochs, cache_path = _extract_epoch_cache(
        project_root=project_root,
        config=config,
        table=table,
    )
    frozen = pd.read_csv(project_root / "outputs" / "prediction" / "frozen_subject_splits.csv")
    id_to_index = {participant_id: index for index, participant_id in enumerate(participant_ids)}
    seed = int(config["study"]["seed"])
    prediction_records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    checkpoint_dir = project_root / "outputs" / "deep" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    split_groups = list(frozen.loc[frozen["repeat"] < repeats].groupby(["repeat", "fold"]))
    for (repeat, fold), split in tqdm(split_groups, desc="EEGNet outer folds"):
        outer_train_ids = split.loc[split["role"] == "train", "participant_id"].to_numpy()
        test_ids = split.loc[split["role"] == "test", "participant_id"].to_numpy()
        outer_train_indices = np.asarray([id_to_index[item] for item in outer_train_ids])
        test_indices = np.asarray([id_to_index[item] for item in test_ids])
        inner = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=seed + int(repeat) * 100 + int(fold),
        )
        train_local, validation_local = next(
            inner.split(outer_train_indices, labels[outer_train_indices])
        )
        train_indices = outer_train_indices[train_local]
        validation_indices = outer_train_indices[validation_local]

        train_raw = np.asarray(epochs[train_indices], dtype=np.float32)
        validation_raw = np.asarray(epochs[validation_indices], dtype=np.float32)
        test_raw = np.asarray(epochs[test_indices], dtype=np.float32)
        channel_mean = train_raw.mean(axis=(0, 1, 3), keepdims=True)
        channel_sd = train_raw.std(axis=(0, 1, 3), keepdims=True)
        channel_sd = np.maximum(channel_sd, np.finfo(np.float32).eps)
        train_raw = np.clip((train_raw - channel_mean) / channel_sd, -10, 10)
        validation_raw = np.clip((validation_raw - channel_mean) / channel_sd, -10, 10)
        test_raw = np.clip((test_raw - channel_mean) / channel_sd, -10, 10)

        fold_seed = seed + int(repeat) * 1000 + int(fold)
        eegnet, raw_training = _train_eegnet(
            train_raw,
            labels[train_indices],
            validation_raw,
            labels[validation_indices],
            config=config,
            seed=fold_seed,
            device=device,
        )
        raw_probability, _ = _infer_eegnet(
            eegnet,
            test_raw,
            device=device,
            batch_size=int(config["classification"]["batch_size"]),
        )
        _, train_embedding = _infer_eegnet(
            eegnet,
            train_raw,
            device=device,
            batch_size=int(config["classification"]["batch_size"]),
        )
        _, validation_embedding = _infer_eegnet(
            eegnet,
            validation_raw,
            device=device,
            batch_size=int(config["classification"]["batch_size"]),
        )
        _, test_embedding = _infer_eegnet(
            eegnet,
            test_raw,
            device=device,
            batch_size=int(config["classification"]["batch_size"]),
        )
        feature_mean = biomarkers[train_indices].mean(axis=0, keepdims=True)
        feature_sd = biomarkers[train_indices].std(axis=0, keepdims=True)
        feature_sd = np.maximum(feature_sd, np.finfo(np.float32).eps)
        train_features = (biomarkers[train_indices] - feature_mean) / feature_sd
        validation_features = (biomarkers[validation_indices] - feature_mean) / feature_sd
        test_features = (biomarkers[test_indices] - feature_mean) / feature_sd
        fusion, fusion_training = _train_fusion_head(
            train_embedding,
            train_features,
            labels[train_indices],
            validation_embedding,
            validation_features,
            labels[validation_indices],
            config=config,
            seed=fold_seed + 500_000,
            device=device,
        )
        fusion.eval()
        with torch.no_grad():
            fusion_probability = torch.softmax(
                fusion(
                    torch.from_numpy(test_embedding).float().to(device),
                    torch.from_numpy(test_features).float().to(device),
                ),
                dim=1,
            ).cpu().numpy()

        for row_index, participant_index in enumerate(test_indices):
            for model_name, probability in (
                ("eegnet", raw_probability),
                ("eegnet_complexity_fusion", fusion_probability),
            ):
                prediction_records.append(
                    {
                        "model": model_name,
                        "repeat": int(repeat),
                        "fold": int(fold),
                        "participant_id": participant_ids[participant_index],
                        "diagnosis": labels[participant_index],
                        **{
                            f"probability_{label}": float(probability[row_index, class_index])
                            for class_index, label in enumerate(CLASSES)
                        },
                    }
                )
        checkpoint_path = checkpoint_dir / f"repeat-{int(repeat):02d}_fold-{int(fold)}.pt"
        torch.save(
            {
                "eegnet": eegnet.state_dict(),
                "fusion_head": fusion.state_dict(),
                "channel_mean": channel_mean,
                "channel_sd": channel_sd,
                "feature_mean": feature_mean,
                "feature_sd": feature_sd,
                "complexity_columns": complexity_columns,
                "train_ids": participant_ids[train_indices].tolist(),
                "validation_ids": participant_ids[validation_indices].tolist(),
                "test_ids": participant_ids[test_indices].tolist(),
            },
            checkpoint_path,
        )
        training_records.append(
            {
                "repeat": int(repeat),
                "fold": int(fold),
                "device": str(device),
                "raw_best_epoch": raw_training["best_epoch"],
                "raw_epochs_run": raw_training["epochs_run"],
                "raw_validation_macro_auc": raw_training["best_validation_macro_auc"],
                "fusion_best_epoch": fusion_training["best_epoch"],
                "fusion_epochs_run": fusion_training["epochs_run"],
                "fusion_validation_macro_auc": fusion_training["best_validation_macro_auc"],
            }
        )

    output_dir = project_root / "outputs" / "deep"
    prediction_path = output_dir / "outer_predictions.csv"
    training_path = output_dir / "training_summary.csv"
    predictions = pd.DataFrame(prediction_records)
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(training_records).to_csv(training_path, index=False)
    probability_columns = [f"probability_{label}" for label in CLASSES]
    aggregated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary: dict[str, Any] = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "outer_repeats_completed": repeats,
        "models": {},
    }
    for model_name, frame in predictions.groupby("model", sort=False):
        participant = (
            frame.groupby(["participant_id", "diagnosis"], as_index=False)[probability_columns]
            .mean()
            .sort_values("participant_id")
        )
        y_true = participant["diagnosis"].to_numpy()
        probability = participant[probability_columns].to_numpy(dtype=float)
        aggregated[model_name] = (y_true, probability)
        summary["models"][model_name] = _metrics(y_true, probability)
    y_true, raw_probability = aggregated["eegnet"]
    fusion_y, fusion_probability = aggregated["eegnet_complexity_fusion"]
    if not np.array_equal(y_true, fusion_y):
        raise RuntimeError("Deep model prediction aggregation is misaligned")
    summary["bootstrap"] = _stratified_bootstrap_difference(
        y_true,
        raw_probability,
        fusion_probability,
        seed=seed + 7000,
        iterations=int(config["statistics"]["bootstrap_iterations"]),
    )
    summary["h5_success"] = summary["bootstrap"]["fusion_minus_raw_ci95_low"] > 0
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        project_root / "outputs" / "manifests" / "deep_prediction.json",
        stage="deep_prediction",
        project_root=project_root,
        config_path=config_path,
        inputs=[
            project_root / "outputs" / "prediction" / "frozen_subject_splits.csv",
            project_root / "outputs" / "mechanistic" / "regional_features.csv",
            cache_path,
        ],
        outputs=[prediction_path, training_path, summary_path],
        extra={
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "outer_repeats": repeats,
            "checkpoints": len(split_groups),
            "source_set_files": len(
                list(iter_expected_set_files(project_root / "data" / "ds004504"))
            ),
        },
    )
    return prediction_path, training_path, summary_path
