import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted
from torch import Tensor
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SurvivalArray = np.ndarray
ArrayLike = np.ndarray | pd.DataFrame | pd.Series


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_survival_array(times: ArrayLike, events: ArrayLike) -> np.recarray:
    times_array = np.asarray(times, dtype=np.float64)
    event_array = np.asarray(events, dtype=bool)
    if times_array.shape != event_array.shape:
        raise ValueError("times and events must have the same shape")
    result = np.empty(times_array.shape, dtype=[("cens", "?"), ("time", "f8")])
    result["time"] = times_array
    result["cens"] = event_array
    return result


def convert_sksurv_y(y: np.ndarray) -> np.recarray:
    event_key, time_key = y.dtype.names
    return make_survival_array(y[time_key], y[event_key])


def calc_expected_time(survival_table: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    deltas = np.diff(time_grid)
    return time_grid[0] + np.sum(survival_table[:, :-1] * deltas[None, :], axis=1)


def adequate_integrated_brier_score(
    y_train: SurvivalArray,
    y_test: SurvivalArray,
    survival_table: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    left_train = float(y_train["time"].min())
    right_train = float(y_train["time"].max())
    valid_mask = (y_test["time"] >= left_train) & (y_test["time"] <= right_train)
    if not np.any(valid_mask):
        return float("nan")
    filtered_y_test = y_test[valid_mask]
    left = float(filtered_y_test["time"].min())
    right = float(filtered_y_test["time"].max())
    left_idx, right_idx = np.searchsorted(time_grid, [left, right])
    right_idx = max(left_idx + 2, right_idx)
    right_idx = min(right_idx, len(time_grid))
    return float(
        integrated_brier_score(
            y_train,
            filtered_y_test,
            survival_table[valid_mask][:, left_idx:right_idx],
            time_grid[left_idx:right_idx],
        )
    )


def adequate_cumulative_dynamic_auc(
    y_train: SurvivalArray,
    y_test: SurvivalArray,
    survival_table: np.ndarray,
    time_grid: np.ndarray,
) -> float:
    if len(time_grid) < 2:
        return float("nan")
    lower, upper = np.percentile(y_train["time"], [0, 95])
    valid_mask = (lower <= y_test["time"]) & (y_test["time"] <= upper)
    if not np.any(valid_mask):
        return float("nan")
    filtered_y_test = y_test[valid_mask]
    picked_times = np.unique(filtered_y_test["time"])[:-1]
    event_times = filtered_y_test["time"][filtered_y_test["cens"]]
    if len(event_times) == 0:
        return float("nan")
    picked_times = picked_times[picked_times > event_times.min()]
    if len(picked_times) == 0:
        return float("nan")
    picked_indices = np.searchsorted(time_grid, picked_times, side="left")
    picked_indices = np.clip(picked_indices, 0, len(time_grid) - 1)
    risk_scores = -np.log(survival_table[valid_mask][:, picked_indices].clip(min=1e-15))
    try:
        _, auc = cumulative_dynamic_auc(y_train, filtered_y_test, risk_scores, picked_times)
    except Exception:
        auc = float("nan")
    return float(auc)


def calc_survival_metrics(
    y_train: SurvivalArray,
    y_test: SurvivalArray,
    survival_table: np.ndarray,
    time_grid: np.ndarray,
) -> dict[str, float]:
    expected_time = calc_expected_time(survival_table, time_grid)
    return {
        "C-index": float(
            concordance_index_censored(y_test["cens"], y_test["time"], -expected_time)[0]
        ),
        "IBS": adequate_integrated_brier_score(y_train, y_test, survival_table, time_grid),
        "CDAUC": adequate_cumulative_dynamic_auc(y_train, y_test, survival_table, time_grid),
    }


def parse_columns(raw_value: str | None) -> list[str]:
    if raw_value is None or raw_value.strip() == "":
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def load_array(path: str | Path) -> np.ndarray:
    loaded = np.load(Path(path), allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return loaded
    keys = list(loaded.keys())
    if len(keys) != 1:
        raise ValueError(f"Expected exactly one array in {path}, got keys={keys}")
    return loaded[keys[0]]


def load_relations_csv(
    path: str | Path,
    expected_rows: int,
    expected_background_size: int,
) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(
            f"Relation CSV {path} has {len(frame)} rows, expected {expected_rows}"
        )
    column_names = list(frame.columns)
    has_grouped_columns = all("__" in column_name for column_name in column_names)
    if not has_grouped_columns:
        values = frame.to_numpy(dtype=np.float32, copy=True)
        if values.shape[1] != expected_background_size:
            raise ValueError(
                f"Relation CSV {path} must have {expected_background_size} columns for a single relation matrix "
                f"or use grouped headers like relation_name__0, relation_name__1, ..."
            )
        return values[:, :, None]

    grouped_columns: dict[str, list[tuple[int, str]]] = {}
    for column_name in column_names:
        relation_name, background_idx_raw = column_name.rsplit("__", 1)
        try:
            background_idx = int(background_idx_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid relation column {column_name!r} in {path}; expected suffix '__<train_index>'"
            ) from exc
        grouped_columns.setdefault(relation_name, []).append((background_idx, column_name))

    relation_blocks: list[np.ndarray] = []
    for relation_name in sorted(grouped_columns):
        indexed_columns = sorted(grouped_columns[relation_name], key=lambda item: item[0])
        indices = [idx for idx, _ in indexed_columns]
        expected_indices = list(range(expected_background_size))
        if indices != expected_indices:
            raise ValueError(
                f"Columns for relation {relation_name!r} in {path} must cover train indices 0..{expected_background_size - 1}"
            )
        ordered_column_names = [column_name for _, column_name in indexed_columns]
        relation_blocks.append(frame.loc[:, ordered_column_names].to_numpy(dtype=np.float32, copy=True))
    return np.stack(relation_blocks, axis=2)


def load_relations_tensor(
    path: str | Path,
    expected_rows: int,
    expected_background_size: int,
) -> np.ndarray:
    relation_path = Path(path)
    if relation_path.suffix.lower() == ".csv":
        return load_relations_csv(
            relation_path,
            expected_rows=expected_rows,
            expected_background_size=expected_background_size,
        )
    return np.asarray(load_array(relation_path), dtype=np.float32)


def save_pickle(obj: Any, path: str | Path) -> None:
    with Path(path).open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def infer_feature_columns(
    frame: pd.DataFrame,
    time_col: str,
    event_col: str,
    relation_cat_cols: Sequence[str],
    feature_cols: Sequence[str] | None = None,
) -> list[str]:
    if feature_cols:
        return list(feature_cols)
    excluded = {time_col, event_col, *relation_cat_cols}
    return [column for column in frame.columns if column not in excluded]


def frame_to_features(frame: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    if not feature_cols:
        raise ValueError("No feature columns were selected")
    return frame.loc[:, feature_cols].astype(np.float32).to_numpy(copy=True)


def build_categorical_relations(
    train_frame: pd.DataFrame,
    relation_columns: Sequence[str],
    test_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not relation_columns:
        train_rel = np.zeros((len(train_frame), len(train_frame), 0), dtype=np.float32)
        test_rel = None if test_frame is None else np.zeros((len(test_frame), len(train_frame), 0), dtype=np.float32)
        return train_rel, test_rel
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    for column in relation_columns:
        train_values = train_frame[column].astype("string").fillna("__MISSING__").to_numpy()
        train_blocks.append((train_values[:, None] == train_values[None, :]).astype(np.float32))
        if test_frame is not None:
            test_values = test_frame[column].astype("string").fillna("__MISSING__").to_numpy()
            test_blocks.append((test_values[:, None] == train_values[None, :]).astype(np.float32))
    train_relations = np.stack(train_blocks, axis=2)
    test_relations = None if test_frame is None else np.stack(test_blocks, axis=2)
    return train_relations, test_relations


def make_balanced_sample_weights(events: np.ndarray) -> torch.DoubleTensor:
    counts = np.bincount(events.astype(int), minlength=2).astype(np.float64)
    counts[counts == 0.0] = 1.0
    class_weights = counts.sum() / counts
    sample_weights = class_weights[events.astype(int)]
    return torch.as_tensor(sample_weights, dtype=torch.double)


class SurvivalTensorDataset(Dataset):
    def __init__(self, features: np.ndarray, y: SurvivalArray, relations: np.ndarray):
        self.features = np.asarray(features, dtype=np.float32)
        self.times = np.asarray(y["time"], dtype=np.float32)
        self.events = np.asarray(y["cens"], dtype=np.bool_)
        self.relations = np.asarray(relations, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "x": self.features[index],
            "t": self.times[index],
            "event": self.events[index],
            "relations": self.relations[index],
            "index": index,
        }


@dataclass
class BackgroundTensors:
    x: Tensor
    t: Tensor
    event: Tensor
    relations: Tensor
    index: Tensor


def make_background_tensors(features: np.ndarray, y: SurvivalArray, relations: np.ndarray) -> BackgroundTensors:
    return BackgroundTensors(
        x=torch.as_tensor(features, dtype=torch.float32),
        t=torch.as_tensor(y["time"].copy(), dtype=torch.float32),
        event=torch.as_tensor(y["cens"].copy(), dtype=torch.bool),
        relations=torch.as_tensor(relations, dtype=torch.float32),
        index=torch.arange(len(features)),
    )


def collate_query_key_batch(
    x_query: Tensor,
    t_query: Tensor,
    event_query: Tensor | None,
    x_key: Tensor,
    t_key: Tensor,
    event_key: Tensor,
    relations: Tensor,
    attention_mask: Tensor | None = None,
    original_key_index: Tensor | None = None,
) -> dict[str, Tensor]:
    sorted_key_idx = torch.argsort(t_key)
    sorted_x_key = x_key[sorted_key_idx]
    sorted_t_key = t_key[sorted_key_idx]
    sorted_event_key = event_key[sorted_key_idx]
    sorted_relations = relations[:, sorted_key_idx]
    query_time_idx_raw = torch.searchsorted(sorted_t_key, t_query, right=False)
    query_time_idx = torch.clamp_max(query_time_idx_raw, len(sorted_t_key) - 1)
    if event_query is not None:
        event_query = event_query.clone()
        event_query[query_time_idx_raw == len(sorted_t_key)] = False
    batch = {
        "x_query": x_query,
        "x_key": sorted_x_key,
        "relations": sorted_relations,
        "event_key": sorted_event_key,
        "event_query": event_query,
        "query_time_idx": query_time_idx,
        "time_grid": sorted_t_key,
    }
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask[:, sorted_key_idx]
    if original_key_index is not None:
        batch["original_key_index"] = original_key_index[sorted_key_idx]
    return batch


class SimpleLeaveOneOutLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for batch in self.loader:
            sample_index = batch["index"]
            relations = batch["relations"][:, sample_index]
            attention_mask = torch.eye(len(sample_index), dtype=torch.bool)
            yield collate_query_key_batch(
                x_query=batch["x"],
                t_query=batch["t"],
                event_query=batch["event"],
                x_key=batch["x"],
                t_key=batch["t"],
                event_key=batch["event"],
                relations=relations,
                attention_mask=attention_mask,
                original_key_index=sample_index,
            )


class CombinedSurvivalLoader:
    def __init__(self, loader: DataLoader, background: BackgroundTensors):
        self.loader = loader
        self.background = background

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for batch in self.loader:
            yield collate_query_key_batch(
                x_query=batch["x"],
                t_query=batch["t"],
                event_query=batch["event"],
                x_key=self.background.x,
                t_key=self.background.t,
                event_key=self.background.event,
                relations=batch["relations"],
                original_key_index=self.background.index,
            )


class MLPBlock(Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: type[Module] = torch.nn.ReLU,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[Module] = []
        current_dim = input_dim
        for _ in range(max(0, n_layers - 1)):
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(torch.nn.LayerNorm(hidden_dim))
            layers.append(activation())
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class BeranEstimatorCore(Module):
    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, weights: Tensor, event_key: Tensor) -> tuple[Tensor, Tensor]:
        cumulative_weights = torch.cumsum(weights, dim=1)
        previous_cumulative = torch.clamp_max(cumulative_weights - weights, 1.0 - self.eps)
        cumulative_weights = torch.clamp_max(cumulative_weights, 1.0 - self.eps)
        log_hazard_increment = torch.log1p(-previous_cumulative) - torch.log1p(-cumulative_weights)
        filtered_increment = event_key.to(weights.dtype) * log_hazard_increment
        cumulative_hazard = torch.cumsum(filtered_increment, dim=1)
        survival = torch.exp(-cumulative_hazard)
        steps = torch.cat([1.0 - survival[:, :1], survival[:, :-1] - survival[:, 1:]], dim=1)
        return survival, steps


class GaussianRelationKernel(Module):
    def __init__(self, n_features: int, n_relations: int) -> None:
        super().__init__()
        self.distance_scale = Parameter(torch.ones(n_features))
        self.relation_weight = Parameter(torch.zeros(n_relations))

    def forward(self, x_query: Tensor, x_key: Tensor, relations: Tensor) -> Tensor:
        squared_distances = (
            self.distance_scale.view(1, 1, -1) * (x_query.unsqueeze(1) - x_key.unsqueeze(0))
        ).pow(2)
        logits = -squared_distances.sum(dim=2)
        if relations.shape[2] > 0:
            logits = logits + torch.einsum("qkr,r->qk", relations.to(x_query.dtype), self.relation_weight)
        return logits


class RelationMLPKernel(Module):
    def __init__(
        self,
        n_features: int,
        n_relations: int,
        embed_dim: int,
        embed_layers: int,
        merge_hidden_dim: int,
        merge_layers: int,
    ) -> None:
        super().__init__()
        self.query_key_network = MLPBlock(
            n_layers=embed_layers,
            input_dim=n_features,
            hidden_dim=max(embed_dim, n_features),
            output_dim=embed_dim,
        )
        merge_input_dim = 1 + n_relations
        self.merge_network = MLPBlock(
            n_layers=merge_layers,
            input_dim=merge_input_dim,
            hidden_dim=merge_hidden_dim,
            output_dim=1,
        )

    def forward(self, x_query: Tensor, x_key: Tensor, relations: Tensor) -> Tensor:
        embedded_query = self.query_key_network(x_query)
        embedded_key = self.query_key_network(x_key)
        symmetric_distance = (embedded_query.unsqueeze(1) - embedded_key.unsqueeze(0)).pow(2).sum(dim=2, keepdim=True)
        merge_input = torch.cat([symmetric_distance, relations.to(x_query.dtype)], dim=2)
        return self.merge_network(merge_input.view(-1, merge_input.shape[2])).view(x_query.shape[0], x_key.shape[0])


class KernelBeranModel(Module):
    def __init__(self, kernel: Module, inverse_temperature: float = 1.0) -> None:
        super().__init__()
        self.kernel = kernel
        self.beran = BeranEstimatorCore()
        self.inverse_temperature = inverse_temperature

    def forward(
        self,
        x_query: Tensor,
        x_key: Tensor,
        relations: Tensor,
        event_key: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        logits = self.kernel(x_query, x_key, relations) * self.inverse_temperature
        if attention_mask is not None:
            logits = logits + torch.where(attention_mask, -torch.inf, 0.0)
        weights = torch.softmax(logits, dim=1)
        return self.beran(weights, event_key)


def aggregate_tied_times(
    time_grid: Tensor,
    survival: Tensor,
    survival_steps: Tensor,
    query_time_idx: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    sorted_time = torch.sort(time_grid).values
    unique_time, inverse_index, counts = torch.unique_consecutive(
        sorted_time,
        return_inverse=True,
        return_counts=True,
    )
    if len(unique_time) == len(sorted_time):
        return survival, survival_steps, query_time_idx, unique_time
    ends = counts.cumsum(dim=0) - 1
    previous = torch.cat(
        [
            torch.zeros((survival.shape[0], 1), device=survival.device, dtype=survival.dtype),
            (1.0 - survival)[:, ends[:-1]],
        ],
        dim=1,
    )
    aggregated_fail = (1.0 - survival)[:, ends] - previous
    aggregated_survival = 1.0 - torch.cumsum(aggregated_fail, dim=1)
    cumulative_steps = torch.cumsum(survival_steps, dim=1)
    previous_steps = torch.cat(
        [torch.zeros((survival.shape[0], 1), device=survival.device, dtype=survival.dtype), cumulative_steps[:, ends[:-1]]],
        dim=1,
    )
    aggregated_steps = cumulative_steps[:, ends] - previous_steps
    remapped_idx = None if query_time_idx is None else inverse_index[query_time_idx]
    return aggregated_survival, aggregated_steps, remapped_idx, unique_time


def k_survival_loss(
    survival: Tensor,
    _survival_steps: Tensor,
    event_query: Tensor,
    query_time_idx: Tensor,
    k: int = 3,
    eps: float = 1e-9,
) -> Tensor:
    extra = (k - 1) // 2
    extended_survival = torch.cat(
        [
            torch.ones((survival.shape[0], extra + 1), dtype=survival.dtype, device=survival.device),
            survival,
            torch.zeros((survival.shape[0], extra), dtype=survival.dtype, device=survival.device),
        ],
        dim=1,
    )
    widened_steps = extended_survival[:, :-k] - extended_survival[:, k:]
    picked = torch.where(
        event_query,
        torch.take_along_dim(widened_steps, query_time_idx[:, None], dim=1),
        torch.take_along_dim(survival, query_time_idx[:, None], dim=1),
    )
    return -torch.log(picked + eps).mean()


class RelationAwareBeranEstimator(BaseEstimator):
    def __init__(
        self,
        model_type: str = "mlp",
        n_epochs: int = 300,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        inverse_temperature: float = 1.0,
        loss_k: int = 3,
        embed_dim: int = 8,
        embed_layers: int = 2,
        merge_hidden_dim: int = 16,
        merge_layers: int = 2,
        validation_fraction: float = 0.2,
        balance_events: bool = True,
        tied_times_correction: bool = True,
        device: str = "cpu",
        random_state: int = 123,
        verbose: bool = False,
    ) -> None:
        self.model_type = model_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.inverse_temperature = inverse_temperature
        self.loss_k = loss_k
        self.embed_dim = embed_dim
        self.embed_layers = embed_layers
        self.merge_hidden_dim = merge_hidden_dim
        self.merge_layers = merge_layers
        self.validation_fraction = validation_fraction
        self.balance_events = balance_events
        self.tied_times_correction = tied_times_correction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _build_model(self, n_features: int, n_relations: int) -> KernelBeranModel:
        if self.model_type == "gaussian":
            kernel = GaussianRelationKernel(n_features=n_features, n_relations=n_relations)
        elif self.model_type == "mlp":
            kernel = RelationMLPKernel(
                n_features=n_features,
                n_relations=n_relations,
                embed_dim=self.embed_dim,
                embed_layers=self.embed_layers,
                merge_hidden_dim=self.merge_hidden_dim,
                merge_layers=self.merge_layers,
            )
        else:
            raise ValueError(f"Unknown model_type={self.model_type!r}")
        return KernelBeranModel(kernel=kernel, inverse_temperature=self.inverse_temperature)

    def fit(self, X: np.ndarray, y: SurvivalArray, relations: np.ndarray) -> "RelationAwareBeranEstimator":
        set_seed(self.random_state)
        features = np.asarray(X, dtype=np.float32)
        relation_tensor = np.asarray(relations, dtype=np.float32)
        if relation_tensor.ndim != 3:
            raise ValueError("relations must have shape (n_samples, n_background, n_relations)")
        if relation_tensor.shape[:2] != (len(features), len(features)):
            raise ValueError("fit() expects train-train relations of shape (n_samples, n_samples, n_relations)")
        self.n_features_in_ = features.shape[1]
        self.n_relations_ = relation_tensor.shape[2]
        self.model_ = self._build_model(self.n_features_in_, self.n_relations_)
        self.model_.to(self.device)
        self.training_features_ = features.copy()
        self.training_y_ = y.copy()
        self.training_relations_ = relation_tensor.copy()
        self.train_time_grid_ = np.sort(self.training_y_["time"])
        self.history_ = {"train_loss": [], "val_loss": []}

        dataset = SurvivalTensorDataset(features, y, relation_tensor)
        if self.balance_events:
            sampler = WeightedRandomSampler(
                make_balanced_sample_weights(y["cens"]),
                num_samples=len(dataset),
                replacement=True,
            )
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        train_loader = SimpleLeaveOneOutLoader(loader)

        val_loader: CombinedSurvivalLoader | None = None
        if 0.0 < self.validation_fraction < 1.0 and len(dataset) > 4:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=self.validation_fraction,
                random_state=self.random_state,
            )
            (_, val_idx), = splitter.split(features, y["cens"])
            val_relations = relation_tensor[val_idx]
            val_dataset = SurvivalTensorDataset(features[val_idx], y[val_idx], val_relations)
            val_background = make_background_tensors(features, y, relation_tensor)
            val_loader = CombinedSurvivalLoader(
                DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False),
                val_background,
            )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.n_epochs))
        loss_fn = partial(k_survival_loss, k=self.loss_k)

        for epoch in range(self.n_epochs):
            train_loss = self._run_epoch(
                loader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                training=True,
            )
            scheduler.step()
            self.history_["train_loss"].append(train_loss)
            if val_loader is not None:
                val_loss = self._run_epoch(
                    loader=val_loader,
                    optimizer=None,
                    loss_fn=loss_fn,
                    training=False,
                )
                self.history_["val_loss"].append(val_loss)
            if self.verbose and (
                epoch == 0 or (epoch + 1) % max(1, self.n_epochs // 10) == 0 or epoch + 1 == self.n_epochs
            ):
                message = f"[{epoch + 1:4d}/{self.n_epochs}] train_loss={train_loss:.4f}"
                if val_loader is not None:
                    message += f" val_loss={self.history_['val_loss'][-1]:.4f}"
                print(message)

        self.model_.to("cpu")
        return self

    def _run_epoch(
        self,
        loader: Iterator[dict[str, Tensor]],
        optimizer: torch.optim.Optimizer | None,
        loss_fn: Any,
        training: bool,
    ) -> float:
        total_loss = 0.0
        total_count = 0
        self.model_.train(training)
        for batch in loader:
            with torch.set_grad_enabled(training):
                survival, survival_steps = self.model_(
                    x_query=batch["x_query"].to(self.device),
                    x_key=batch["x_key"].to(self.device),
                    relations=batch["relations"].to(self.device),
                    event_key=batch["event_key"].to(self.device),
                    attention_mask=batch.get("attention_mask", None).to(self.device)
                    if "attention_mask" in batch
                    else None,
                )
                time_grid = batch["time_grid"].to(self.device)
                query_time_idx = batch["query_time_idx"].to(self.device)
                if self.tied_times_correction:
                    survival, survival_steps, query_time_idx, _ = aggregate_tied_times(
                        time_grid=time_grid,
                        survival=survival,
                        survival_steps=survival_steps,
                        query_time_idx=query_time_idx,
                    )
                loss = loss_fn(
                    survival,
                    survival_steps,
                    batch["event_query"].to(self.device),
                    query_time_idx,
                )
                if training:
                    assert optimizer is not None
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            batch_size = len(batch["x_query"])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
        return total_loss / max(total_count, 1)

    def predict_survival_table(
        self,
        X: np.ndarray,
        relations: np.ndarray,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, "model_")
        features = np.asarray(X, dtype=np.float32)
        relation_tensor = np.asarray(relations, dtype=np.float32)
        if relation_tensor.shape[:2] != (len(features), len(self.training_features_)):
            raise ValueError(
                "relations must have shape (n_samples, n_train_samples, n_relations) for prediction"
            )
        self.model_.to(self.device)
        prediction_batch_size = batch_size or self.batch_size
        all_survival: list[np.ndarray] = []
        output_grid: np.ndarray | None = None
        with torch.inference_mode():
            for start in range(0, len(features), prediction_batch_size):
                stop = min(len(features), start + prediction_batch_size)
                batch = collate_query_key_batch(
                    x_query=torch.as_tensor(features[start:stop], dtype=torch.float32),
                    t_query=torch.zeros(stop - start, dtype=torch.float32),
                    event_query=None,
                    x_key=torch.as_tensor(self.training_features_, dtype=torch.float32),
                    t_key=torch.as_tensor(self.training_y_["time"].copy(), dtype=torch.float32),
                    event_key=torch.as_tensor(self.training_y_["cens"].copy(), dtype=torch.bool),
                    relations=torch.as_tensor(relation_tensor[start:stop], dtype=torch.float32),
                )
                survival, survival_steps = self.model_(
                    x_query=batch["x_query"].to(self.device),
                    x_key=batch["x_key"].to(self.device),
                    relations=batch["relations"].to(self.device),
                    event_key=batch["event_key"].to(self.device),
                )
                if self.tied_times_correction:
                    survival, survival_steps, _, unique_time = aggregate_tied_times(
                        time_grid=batch["time_grid"].to(self.device),
                        survival=survival,
                        survival_steps=survival_steps,
                    )
                    time_grid = unique_time.cpu().numpy()
                else:
                    time_grid = batch["time_grid"].cpu().numpy()
                all_survival.append(survival.cpu().numpy())
                output_grid = time_grid
        self.model_.to("cpu")
        assert output_grid is not None
        return np.vstack(all_survival), output_grid

    def predict_survival_function(
        self,
        X: np.ndarray,
        relations: np.ndarray,
        return_array: bool = True,
    ) -> np.ndarray | list[Any]:
        survival_table, time_grid = self.predict_survival_table(X, relations)
        if return_array:
            return survival_table
        return [pd.Series(row, index=time_grid) for row in survival_table]

    def predict_expected_time(self, X: np.ndarray, relations: np.ndarray) -> np.ndarray:
        survival_table, time_grid = self.predict_survival_table(X, relations)
        return calc_expected_time(survival_table, time_grid)

    def predict(self, X: np.ndarray, relations: np.ndarray) -> np.ndarray:
        return self.predict_expected_time(X, relations)


@dataclass
class PreparedTableData:
    X: np.ndarray
    y: np.recarray
    train_relations: np.ndarray
    test_X: np.ndarray | None = None
    test_relations: np.ndarray | None = None
    feature_columns: list[str] | None = None


@dataclass
class PreparedPredictionData:
    X_test: np.ndarray
    test_relations: np.ndarray
    feature_columns: list[str]


def prepare_table_data(
    train_csv: str | Path,
    time_col: str,
    event_col: str,
    feature_cols: Sequence[str] | None = None,
    relation_cat_cols: Sequence[str] | None = None,
    train_relations_path: str | None = None,
    test_csv: str | Path | None = None,
    test_relations_path: str | None = None,
) -> PreparedTableData:
    relation_cat_cols = list(relation_cat_cols or [])
    train_frame = pd.read_csv(train_csv)
    selected_feature_cols = infer_feature_columns(
        train_frame,
        time_col=time_col,
        event_col=event_col,
        relation_cat_cols=relation_cat_cols,
        feature_cols=feature_cols,
    )
    X_train = frame_to_features(train_frame, selected_feature_cols)
    y_train = make_survival_array(train_frame[time_col], train_frame[event_col].astype(bool))

    test_frame = pd.read_csv(test_csv) if test_csv is not None else None
    X_test = None if test_frame is None else frame_to_features(test_frame, selected_feature_cols)

    if relation_cat_cols:
        train_relations, test_relations = build_categorical_relations(
            train_frame=train_frame,
            relation_columns=relation_cat_cols,
            test_frame=test_frame,
        )
    elif train_relations_path is not None:
        train_relations = load_relations_tensor(
            train_relations_path,
            expected_rows=len(train_frame),
            expected_background_size=len(train_frame),
        )
        test_relations = (
            None
            if test_relations_path is None
            else load_relations_tensor(
                test_relations_path,
                expected_rows=len(test_frame),
                expected_background_size=len(train_frame),
            )
        )
    else:
        train_relations = np.zeros((len(train_frame), len(train_frame), 0), dtype=np.float32)
        test_relations = None if test_frame is None else np.zeros((len(test_frame), len(train_frame), 0), dtype=np.float32)

    if train_relations.shape[:2] != (len(train_frame), len(train_frame)):
        raise ValueError("Train relations must have shape (n_train, n_train, n_relations)")
    if test_frame is not None and test_relations is None:
        raise ValueError("Prediction requires either categorical relation columns or --test-relations")
    if test_relations is not None and test_relations.shape[:2] != (len(test_frame), len(train_frame)):
        raise ValueError("Test relations must have shape (n_test, n_train, n_relations)")

    return PreparedTableData(
        X=X_train,
        y=y_train,
        train_relations=train_relations,
        test_X=X_test,
        test_relations=test_relations,
        feature_columns=selected_feature_cols,
    )


def infer_prediction_feature_columns(
    test_frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    saved_feature_columns: Sequence[str] | None = None,
) -> list[str]:
    if feature_cols:
        return list(feature_cols)
    if saved_feature_columns:
        return list(saved_feature_columns)
    numeric_columns = test_frame.select_dtypes(include=[np.number, bool]).columns.tolist()
    if not numeric_columns:
        raise ValueError(
            "Could not infer numeric feature columns automatically; please provide --feature-cols"
        )
    return numeric_columns


def prepare_prediction_data(
    model: Any,
    test_csv: str | Path,
    feature_cols: Sequence[str] | None = None,
    relation_cat_cols: Sequence[str] | None = None,
    train_csv: str | Path | None = None,
    test_relations_path: str | None = None,
) -> PreparedPredictionData:
    test_frame = pd.read_csv(test_csv)
    selected_feature_cols = infer_prediction_feature_columns(
        test_frame,
        feature_cols=feature_cols,
        saved_feature_columns=getattr(model, "feature_columns_", None),
    )
    X_test = frame_to_features(test_frame, selected_feature_cols)

    relation_cat_cols = list(relation_cat_cols or [])
    if relation_cat_cols:
        if train_csv is None:
            raise ValueError("--train-csv is required to build relations from categorical columns during prediction")
        train_frame = pd.read_csv(train_csv)
        _, test_relations = build_categorical_relations(
            train_frame=train_frame,
            relation_columns=relation_cat_cols,
            test_frame=test_frame,
        )
        assert test_relations is not None
    elif test_relations_path is not None:
        test_relations = load_relations_tensor(
            test_relations_path,
            expected_rows=len(test_frame),
            expected_background_size=len(model.training_features_),
        )
    else:
        expected_relations = getattr(model, "n_relations_", 0)
        test_relations = np.zeros(
            (len(test_frame), len(model.training_features_), expected_relations),
            dtype=np.float32,
        )

    if test_relations.shape[:2] != (len(test_frame), len(model.training_features_)):
        raise ValueError("Test relations must have shape (n_test, n_train, n_relations)")

    return PreparedPredictionData(
        X_test=X_test,
        test_relations=test_relations,
        feature_columns=selected_feature_cols,
    )


def save_survival_predictions(path: str | Path, survival_table: np.ndarray, time_grid: np.ndarray) -> None:
    columns = [str(float(time)) for time in time_grid]
    pd.DataFrame(survival_table, columns=columns).to_csv(path, index=False)


def save_expected_times(path: str | Path, expected_time: np.ndarray) -> None:
    pd.DataFrame({"expected_time": expected_time}).to_csv(path, index=False)


def save_relations_csv(path: str | Path, relations: np.ndarray, prefix: str = "rel") -> None:
    relation_tensor = np.asarray(relations, dtype=np.float32)
    if relation_tensor.ndim != 3:
        raise ValueError("relations must have shape (n_rows, n_background, n_relations)")
    columns: dict[str, np.ndarray] = {}
    for relation_idx in range(relation_tensor.shape[2]):
        relation_name = f"{prefix}{relation_idx}"
        for background_idx in range(relation_tensor.shape[1]):
            columns[f"{relation_name}__{background_idx}"] = relation_tensor[:, background_idx, relation_idx]
    pd.DataFrame(columns).to_csv(path, index=False)


def load_survival_predictions_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if frame.shape[1] == 0:
        raise ValueError(f"Survival CSV {path} has no columns")
    try:
        time_grid = np.asarray([float(column_name) for column_name in frame.columns], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(
            f"Survival CSV {path} must have column names equal to prediction times"
        ) from exc
    survival_table = frame.to_numpy(dtype=np.float64, copy=True)
    return survival_table, time_grid


def load_expected_time_csv(path: str | Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if "expected_time" in frame.columns:
        return frame["expected_time"].to_numpy(dtype=np.float64, copy=True)
    if frame.shape[1] != 1:
        raise ValueError(
            f"Expected-time CSV {path} must contain exactly one column or a column named 'expected_time'"
        )
    return frame.iloc[:, 0].to_numpy(dtype=np.float64, copy=True)


def calc_c_index_from_expected_time(y_test: SurvivalArray, expected_time: np.ndarray) -> float:
    if len(expected_time) != len(y_test):
        raise ValueError("expected_time length must match test set size")
    return float(concordance_index_censored(y_test["cens"], y_test["time"], -expected_time)[0])


def build_estimator_from_args(args: argparse.Namespace) -> RelationAwareBeranEstimator:
    return RelationAwareBeranEstimator(
        model_type=args.model_type,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        inverse_temperature=args.inverse_temperature,
        loss_k=args.loss_k,
        embed_dim=args.embed_dim,
        embed_layers=args.embed_layers,
        merge_hidden_dim=args.merge_hidden_dim,
        merge_layers=args.merge_layers,
        validation_fraction=args.validation_fraction,
        balance_events=not args.no_balance_events,
        tied_times_correction=not args.no_tied_times_correction,
        device=args.device,
        random_state=args.random_state,
        verbose=args.verbose,
    )


def evaluate_model_cv(
    estimator: RelationAwareBeranEstimator,
    X: np.ndarray,
    y: SurvivalArray,
    relations: np.ndarray,
    n_splits: int,
    random_state: int,
) -> dict[str, float]:
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: dict[str, list[float]] = {"C-index": [], "IBS": [], "CDAUC": []}
    for train_idx, test_idx in kfold.split(X, y["cens"]):
        fold_estimator = RelationAwareBeranEstimator(**estimator.get_params(deep=False))
        train_relations = relations[train_idx][:, train_idx]
        test_relations = relations[test_idx][:, train_idx]
        fold_estimator.fit(X[train_idx], y[train_idx], train_relations)
        survival_table, time_grid = fold_estimator.predict_survival_table(X[test_idx], test_relations)
        metrics = calc_survival_metrics(y[train_idx], y[test_idx], survival_table, time_grid)
        for metric_name, value in metrics.items():
            fold_scores[metric_name].append(value)
    return {metric_name: float(np.nanmean(values)) for metric_name, values in fold_scores.items()}


def evaluate_rsf_cv(
    X: np.ndarray,
    y: SurvivalArray,
    n_splits: int,
    random_state: int,
    params: dict[str, Any],
) -> dict[str, float]:
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: dict[str, list[float]] = {"C-index": [], "IBS": [], "CDAUC": []}
    for train_idx, test_idx in kfold.split(X, y["cens"]):
        rsf = RandomSurvivalForest(random_state=random_state, **params)
        rsf.fit(X[train_idx], y[train_idx])
        survival_table = rsf.predict_survival_function(X[test_idx], return_array=True)
        metrics = calc_survival_metrics(y[train_idx], y[test_idx], survival_table, rsf.unique_times_)
        for metric_name, value in metrics.items():
            fold_scores[metric_name].append(value)
    return {metric_name: float(np.nanmean(values)) for metric_name, values in fold_scores.items()}


def metric_to_objective_sign(metric_name: str) -> int:
    if metric_name in {"C-index", "CDAUC"}:
        return 1
    if metric_name == "IBS":
        return -1
    raise ValueError(f"Unsupported metric={metric_name}")


def sample_relation_model_params(
    trial: 'optuna.Trial',
    model_name: str,
    base_params: dict[str, Any],
    n_features: int,
    n_relations: int,
) -> dict[str, Any]:
    params = dict(base_params)
    params["model_type"] = model_name
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    params["inverse_temperature"] = trial.suggest_float("inverse_temperature", 0.5, 3.0)
    params["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    params["loss_k"] = trial.suggest_categorical("loss_k", [1, 3, 5])
    if model_name == "mlp":
        default_embed = max(4, min(4 * max(1, n_features), 64))
        params["embed_dim"] = trial.suggest_categorical("embed_dim", sorted({4, 8, 16, default_embed}))
        params["embed_layers"] = trial.suggest_int("embed_layers", 1, 3)
        hidden_candidates = sorted({8, 16, 32, max(8, 2 * max(1, n_features + n_relations))})
        params["merge_hidden_dim"] = trial.suggest_categorical("merge_hidden_dim", hidden_candidates)
        params["merge_layers"] = trial.suggest_int("merge_layers", 1, 3)
    return params


def sample_rsf_params(trial: 'optuna.Trial') -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300]),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0]),
    }


def run_optuna_benchmark(
    X: np.ndarray,
    y: SurvivalArray,
    relations: np.ndarray,
    model_names: Sequence[str],
    metric_name: str,
    n_splits: int,
    n_trials: int,
    random_state: int,
    device: str,
    n_epochs: int,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Benchmark mode requires Optuna, but it is not available in the current environment."
        ) from exc

    direction = "maximize" if metric_to_objective_sign(metric_name) > 0 else "minimize"
    records: list[dict[str, Any]] = []

    for model_name in model_names:
        base_params = {
            "n_epochs": n_epochs,
            "device": device,
            "random_state": random_state,
            "verbose": False,
        }

        def objective(trial: 'optuna.Trial') -> float:
            params = sample_relation_model_params(
                trial=trial,
                model_name=model_name,
                base_params=base_params,
                n_features=X.shape[1],
                n_relations=relations.shape[2],
            )
            estimator = RelationAwareBeranEstimator(**params)
            scores = evaluate_model_cv(estimator, X, y, relations, n_splits=n_splits, random_state=random_state)
            return float(scores[metric_name])

        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_trial.params
        full_params = {
            **base_params,
            **best_params,
            "model_type": model_name,
        }
        estimator = RelationAwareBeranEstimator(**full_params)
        scores = evaluate_model_cv(estimator, X, y, relations, n_splits=n_splits, random_state=random_state)
        records.append({"model": model_name, **scores, "best_params": json.dumps(full_params, ensure_ascii=False)})

    def rsf_objective(trial: 'optuna.Trial') -> float:
        params = sample_rsf_params(trial)
        scores = evaluate_rsf_cv(X, y, n_splits=n_splits, random_state=random_state, params=params)
        return float(scores[metric_name])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rsf_study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        rsf_study.optimize(rsf_objective, n_trials=n_trials, show_progress_bar=False)
        rsf_scores = evaluate_rsf_cv(
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            params=rsf_study.best_trial.params,
        )
    records.append(
        {
            "model": "rsf",
            **rsf_scores,
            "best_params": json.dumps(rsf_study.best_trial.params, ensure_ascii=False),
        }
    )

    result = pd.DataFrame.from_records(records).sort_values(by=metric_name, ascending=(direction == "minimize"))
    if output_csv is not None:
        result.to_csv(output_csv, index=False)
    return result


def add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--event-col", required=True)
    parser.add_argument("--feature-cols", default=None, help="Comma-separated list of numeric feature columns.")
    parser.add_argument(
        "--relation-categorical-cols",
        default=None,
        help="Comma-separated categorical columns used to build relation matrices.",
    )
    parser.add_argument(
        "--train-relations",
        default=None,
        help=(
            "Path to train-train relations (.npy/.npz or .csv). "
            "For CSV, use either one matrix with n_train columns or grouped columns like rel0__0,...,rel0__N,rel1__0,..."
        ),
    )


def add_relations_builder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--relation-categorical-cols", required=True)
    parser.add_argument("--train-out", required=True, help="Output CSV for train-train relations.")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--test-out", default=None, help="Output CSV for test-train relations.")
    parser.add_argument("--relation-prefix", default="rel")


def add_evaluate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--time-col", required=True)
    parser.add_argument("--event-col", required=True)
    parser.add_argument("--survival-csv", default=None)
    parser.add_argument("--expected-time-csv", default=None)
    parser.add_argument("--output-csv", default=None)


def add_predict_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-in", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--feature-cols", default=None, help="Comma-separated list of feature columns for prediction.")
    parser.add_argument(
        "--relation-categorical-cols",
        default=None,
        help="Comma-separated categorical columns used to build test-train relation matrices.",
    )
    parser.add_argument("--train-csv", default=None, help="Training CSV needed when relations are built from categories.")
    parser.add_argument(
        "--test-relations",
        default=None,
        help="Path to test-train relations (.npy/.npz or .csv).",
    )
    parser.add_argument("--survival-out", default=None)
    parser.add_argument("--expected-time-out", default=None)


def add_shared_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-type", choices=["gaussian", "mlp"], default="mlp")
    parser.add_argument("--n-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--inverse-temperature", type=float, default=1.0)
    parser.add_argument("--loss-k", type=int, default=3)
    parser.add_argument("--embed-dim", type=int, default=8)
    parser.add_argument("--embed-layers", type=int, default=2)
    parser.add_argument("--merge-hidden-dim", type=int, default=16)
    parser.add_argument("--merge-layers", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-balance-events", action="store_true")
    parser.add_argument("--no-tied-times-correction", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python surv_rel.py",
        description="Relation-aware Beran survival models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one model, optionally save it and/or write predictions.")
    add_shared_data_args(train_parser)
    add_shared_model_args(train_parser)
    train_parser.add_argument("--test-csv", default=None)
    train_parser.add_argument(
        "--test-relations",
        default=None,
        help=(
            "Path to test-train relations (.npy/.npz or .csv). "
            "For CSV, rows are test samples and grouped columns must use train indices in the suffix."
        ),
    )
    train_parser.add_argument("--model-out", default=None)
    train_parser.add_argument("--survival-out", default=None)
    train_parser.add_argument("--expected-time-out", default=None)

    relations_parser = subparsers.add_parser(
        "make-relations-csv",
        help="Build train-train and optional test-train relation CSV files from categorical columns.",
    )
    add_relations_builder_args(relations_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate already saved predictions on a labeled test set.",
    )
    add_evaluate_args(evaluate_parser)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Load a trained model and generate predictions for new data.",
    )
    add_predict_args(predict_parser)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Tune several models with Optuna and compare them to RandomSurvivalForest on cross-validation.",
    )
    add_shared_data_args(benchmark_parser)
    benchmark_parser.add_argument("--models", default="gaussian,mlp")
    benchmark_parser.add_argument("--metric", choices=["C-index", "IBS", "CDAUC"], default="C-index")
    benchmark_parser.add_argument("--cv-splits", type=int, default=5)
    benchmark_parser.add_argument("--optuna-trials", type=int, default=15)
    benchmark_parser.add_argument("--n-epochs", type=int, default=250)
    benchmark_parser.add_argument("--device", default="cpu")
    benchmark_parser.add_argument("--random-state", type=int, default=123)
    benchmark_parser.add_argument("--output-csv", default=None)
    return parser


def handle_train_command(args: argparse.Namespace) -> int:
    prepared = prepare_table_data(
        train_csv=args.train_csv,
        time_col=args.time_col,
        event_col=args.event_col,
        feature_cols=parse_columns(args.feature_cols),
        relation_cat_cols=parse_columns(args.relation_categorical_cols),
        train_relations_path=args.train_relations,
        test_csv=args.test_csv,
        test_relations_path=args.test_relations,
    )
    estimator = build_estimator_from_args(args)
    estimator.fit(prepared.X, prepared.y, prepared.train_relations)
    estimator.feature_columns_ = list(prepared.feature_columns or [])

    if args.model_out:
        save_pickle(estimator, args.model_out)

    if prepared.test_X is not None:
        survival_table, time_grid = estimator.predict_survival_table(prepared.test_X, prepared.test_relations)
        if args.survival_out:
            save_survival_predictions(args.survival_out, survival_table, time_grid)
        if args.expected_time_out:
            save_expected_times(args.expected_time_out, calc_expected_time(survival_table, time_grid))

    return 0


def handle_make_relations_csv_command(args: argparse.Namespace) -> int:
    train_frame = pd.read_csv(args.train_csv)
    test_frame = None if args.test_csv is None else pd.read_csv(args.test_csv)
    relation_columns = parse_columns(args.relation_categorical_cols)
    if not relation_columns:
        raise ValueError("At least one categorical relation column must be provided")
    train_relations, test_relations = build_categorical_relations(
        train_frame=train_frame,
        relation_columns=relation_columns,
        test_frame=test_frame,
    )
    save_relations_csv(args.train_out, train_relations, prefix=args.relation_prefix)
    if test_frame is not None:
        if args.test_out is None:
            raise ValueError("--test-out is required when --test-csv is provided")
        assert test_relations is not None
        save_relations_csv(args.test_out, test_relations, prefix=args.relation_prefix)
    return 0


def handle_evaluate_command(args: argparse.Namespace) -> int:
    if args.survival_csv is None and args.expected_time_csv is None:
        raise ValueError("At least one of --survival-csv or --expected-time-csv must be provided")

    train_frame = pd.read_csv(args.train_csv)
    test_frame = pd.read_csv(args.test_csv)
    y_train = make_survival_array(train_frame[args.time_col], train_frame[args.event_col].astype(bool))
    y_test = make_survival_array(test_frame[args.time_col], test_frame[args.event_col].astype(bool))

    metrics: dict[str, float] = {}
    if args.survival_csv is not None:
        survival_table, time_grid = load_survival_predictions_csv(args.survival_csv)
        if len(survival_table) != len(test_frame):
            raise ValueError("Number of rows in survival CSV must match test set size")
        metrics.update(calc_survival_metrics(y_train, y_test, survival_table, time_grid))

    if args.expected_time_csv is not None:
        expected_time = load_expected_time_csv(args.expected_time_csv)
        metrics["C-index-from-expected-time"] = calc_c_index_from_expected_time(y_test, expected_time)
    elif args.survival_csv is not None:
        metrics["C-index-from-expected-time"] = metrics["C-index"]

    result = pd.DataFrame([metrics])
    if args.output_csv is not None:
        result.to_csv(args.output_csv, index=False)
    print(result.to_string(index=False))
    return 0


def handle_predict_command(args: argparse.Namespace) -> int:
    model = load_pickle(args.model_in)
    prediction_data = prepare_prediction_data(
        model=model,
        test_csv=args.test_csv,
        feature_cols=parse_columns(args.feature_cols),
        relation_cat_cols=parse_columns(args.relation_categorical_cols),
        train_csv=args.train_csv,
        test_relations_path=args.test_relations,
    )
    survival_table, time_grid = model.predict_survival_table(
        prediction_data.X_test,
        prediction_data.test_relations,
    )
    if args.survival_out:
        save_survival_predictions(args.survival_out, survival_table, time_grid)
    if args.expected_time_out:
        save_expected_times(args.expected_time_out, calc_expected_time(survival_table, time_grid))
    if args.survival_out is None and args.expected_time_out is None:
        print(pd.DataFrame(survival_table, columns=[str(float(time)) for time in time_grid]).to_string(index=False))
    return 0


def handle_benchmark_command(args: argparse.Namespace) -> int:
    prepared = prepare_table_data(
        train_csv=args.train_csv,
        time_col=args.time_col,
        event_col=args.event_col,
        feature_cols=parse_columns(args.feature_cols),
        relation_cat_cols=parse_columns(args.relation_categorical_cols),
        train_relations_path=args.train_relations,
    )
    model_names = parse_columns(args.models)
    result = run_optuna_benchmark(
        X=prepared.X,
        y=prepared.y,
        relations=prepared.train_relations,
        model_names=model_names,
        metric_name=args.metric,
        n_splits=args.cv_splits,
        n_trials=args.optuna_trials,
        random_state=args.random_state,
        device=args.device,
        n_epochs=args.n_epochs,
        output_csv=args.output_csv,
    )
    print(result.to_string(index=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return handle_train_command(args)
        if args.command == "make-relations-csv":
            return handle_make_relations_csv_command(args)
        if args.command == "evaluate":
            return handle_evaluate_command(args)
        if args.command == "predict":
            return handle_predict_command(args)
        if args.command == "benchmark":
            return handle_benchmark_command(args)
        raise ValueError(f"Unknown command {args.command!r}")
    except (ImportError, ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


__all__ = [
    "RelationAwareBeranEstimator",
    "PreparedTableData",
    "build_categorical_relations",
    "calc_expected_time",
    "calc_survival_metrics",
    "convert_sksurv_y",
    "load_pickle",
    "make_survival_array",
    "main",
    "prepare_table_data",
    "save_pickle",
]


if __name__ == "__main__":
    main()
