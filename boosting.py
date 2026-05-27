from __future__ import annotations

import numpy as np
import json
import inspect

from collections import defaultdict
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    def tqdm(iterable):
        return iterable

try:
    import polars as pl
except ModuleNotFoundError:
    pl = None

try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.base import ClassifierMixin, BaseEstimator, TransformerMixin
except ModuleNotFoundError:
    DecisionTreeRegressor = None

    class ClassifierMixin:
        pass

    class BaseEstimator:
        pass

    class TransformerMixin:
        pass

    def train_test_split(*args, **kwargs):
        raise ModuleNotFoundError("sklearn is required for train_test_split")

    def roc_auc_score(y_true, y_score):
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1)
        positives = y_true.astype(bool)
        n_pos = np.sum(positives)
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return np.nan
        return (np.sum(ranks[positives]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

# note: да я знаю что такое literal и named tuple
from typing import Literal, NamedTuple, Iterable

# TODO: можно попробовать сделать функцию для вычисления метрик которая будет считать сразу все

type Metrix = Literal["val_loss", "train_loss", "val_roc_auc", "train_roc_auc"]


class BestIteration(NamedTuple):
    step: int
    score: float


class BoostingClassifier(ClassifierMixin):
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = 42,
        verbose: bool = True,
        early_stopping_rounds: int | None = None,
        eval_metric: Metrix | None = None,
        use_best_model: bool = False,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        history_commit_path: Path | None = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = "Bernoulli",
        rsm: float = 1.0,
        goss: bool = False,
        goss_k: float = 0.2,
    ):
        super().__init__()
        if base_model_class is None:
            raise ModuleNotFoundError(
                "sklearn is required for the default DecisionTreeRegressor; "
                "pass base_model_class explicitly to use another model"
            )
        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.history_commit_path = history_commit_path

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0.0] * (n_estimators)
        self.feature_indices = [None] * n_estimators

        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
        self.best = BestIteration(step=0, score=-np.inf)
        self.eval_set = eval_set

        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.eval_metric = eval_metric
        self.use_best_model = use_best_model

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.goss = goss
        self.goss_k = goss_k

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.unti_grad_fn = lambda y, z: y / (1 + np.exp(y * z))

    def _validate_sampling_params(self) -> None:
        if not 0 < self.subsample <= 1:
            raise ValueError("subsample should be in (0, 1]")
        if self.bagging_temperature < 0:
            raise ValueError("bagging_temperature should be non-negative")
        if not 0 < self.rsm <= 1:
            raise ValueError("rsm should be in (0, 1]")
        if not 0 < self.goss_k < 1:
            raise ValueError("goss_k should be in (0, 1)")
        if self.bootstrap_type not in (None, "Bernoulli", "Bayesian"):
            raise ValueError("bootstrap_type should be one of None, 'Bernoulli', 'Bayesian'")

    def _sample_features(self, n_features: int) -> np.ndarray:
        n_selected = max(1, int(np.ceil(self.rsm * n_features)))
        if n_selected == n_features:
            return np.arange(n_features)
        return np.sort(self._rng.choice(n_features, size=n_selected, replace=False))

    def _bernoulli_mask(self, n_objects: int, subsample: float) -> np.ndarray:
        mask = self._rng.random(n_objects) < subsample
        if not np.any(mask):
            mask[self._rng.integers(0, n_objects)] = True
        return mask

    def _sample_objects(self, anti_gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        n_objects = anti_gradient.shape[0]

        if self.goss:
            n_large = max(1, int(np.ceil(self.goss_k * n_objects)))
            order = np.argsort(np.abs(anti_gradient))[::-1]
            large_idx = order[:n_large]
            small_idx = order[n_large:]

            selected = np.zeros(n_objects, dtype=bool)
            selected[large_idx] = True

            weights = np.ones(n_objects)
            if len(small_idx) > 0:
                small_mask = self._rng.random(len(small_idx)) < self.subsample
                if not np.any(small_mask):
                    small_mask[self._rng.integers(0, len(small_idx))] = True
                sampled_small_idx = small_idx[small_mask]
                selected[sampled_small_idx] = True
                weights[sampled_small_idx] = (1.0 - self.goss_k) / self.subsample

            return selected, weights[selected]

        if self.bootstrap_type is None:
            return np.ones(n_objects, dtype=bool), None

        if self.bootstrap_type == "Bernoulli":
            return self._bernoulli_mask(n_objects, self.subsample), None

        weights = (-np.log(self._rng.uniform(1e-12, 1.0, size=n_objects))) ** self.bagging_temperature
        return np.ones(n_objects, dtype=bool), weights

    def _fit_base_model(
        self,
        base_model,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        if sample_weight is None:
            base_model.fit(X, y)
            return

        fit_signature = inspect.signature(base_model.fit)
        if "sample_weight" in fit_signature.parameters:
            base_model.fit(X, y, sample_weight=sample_weight)
            return

        base_model.fit(X, y * sample_weight)

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, step: int, train_predictions: np.ndarray
    ) -> None:
        base_model = self.base_model_class(**self.base_model_params)
        anti_gradient = self.unti_grad_fn(y, train_predictions)

        object_mask, sample_weight = self._sample_objects(anti_gradient)
        feature_idx = self._sample_features(X.shape[1])

        X_train = X[object_mask][:, feature_idx]
        y_train = anti_gradient[object_mask]

        self._fit_base_model(base_model, X_train, y_train, sample_weight)

        new_predictions = base_model.predict(X[:, feature_idx])
        self.models[step] = base_model
        self.feature_indices[step] = feature_idx
        self.gammas[step] = self._find_optimal_gamma(
            y, train_predictions, new_predictions
        )
        train_predictions += self.learning_rate * self.gammas[step] * new_predictions

    def _split_val(self, X_train, y_train):
        if self.eval_set:
            X_val, y_val = self.eval_set
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.val_size,
                random_state=self.random_state,
            )
        return X_train, X_val, y_train, y_val

    @property
    def __early_stopping_on(self) -> bool:
        if self.early_stopping_rounds is None:
            return False

        if self.eval_set is None:
            raise ValueError("ERROR: валидационная выборка не задана")

        if self.eval_metric is None:
            print("WARNING:укажите метрику останова! Поставил её дефолтной: `val_loss`")
            self.eval_metric = "val_loss"

        return True

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        self._validate_sampling_params()
        self._rng = np.random.default_rng(self.random_state)
        self.models = [0] * self.n_estimators
        self.gammas = [0.0] * self.n_estimators
        self.feature_indices = [None] * self.n_estimators
        self.history = defaultdict(list)
        self.best = BestIteration(step=0, score=-np.inf)

        if self.__early_stopping_on:
            val_predictions = np.zeros(self.eval_set[0].shape[0])

        train_predictions = np.zeros(X_train.shape[0])

        self.classes_ = np.unique(y_train)
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for i in estimator_range:
            self.partial_fit(X_train, y_train, i, train_predictions)
            self._record_metrics("train", y_train, train_predictions)
            if self.__early_stopping_on and self._early_stopping(i, val_predictions):
                break

        self.commit_history()

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])  # type: ignore

    def commit_history(self, path: Path | None = None):
        if self.history_commit_path is None and path is None:
            return

        if path is None:
            path = self.history_commit_path

        if not path.exists():
            path.touch()

        if not path.is_file():
            raise FileNotFoundError("Путь не является файлом")

        # [{histpry}]
        with open(path, "r", encoding="utf-8") as file:
            res = json.load(file)

        if res is None:
            res = []

        if not isinstance(res, list):
            raise TypeError("В JSON должен быть список словарей")

        with open(path, "w", encoding="utf-8") as file:
            json.dump(res, file, ensure_ascii=False, indent=4)

    def _record_metrics(
        self, prefix: Literal["train", "val"], y: np.ndarray, predictions: np.ndarray
    ) -> None:
        proba = self.sigmoid(predictions)
        self.history[f"{prefix}_loss"].append(self.loss_fn(y, predictions))
        self.history[f"{prefix}_roc_auc"].append(
            roc_auc_score(y == 1, proba) if np.unique(y).size > 1 else np.nan
        )

    # NOTE: chat-gpt когда оптимизировал код
    def _is_better(self, metric: str, score: float) -> bool:
        if not np.isfinite(score):
            return False
        if len(self.history[metric]) == 1:
            return True
        if metric.endswith("roc_auc"):
            return score > self.best.score
        return score < self.best.score

    def _early_stopping(
        self,
        step: int,
        val_predictions: np.ndarray,
    ) -> bool:
        X_val, y_val = self.eval_set
        feature_idx = self.feature_indices[step]
        val_predictions += (
            self.learning_rate
            * self.gammas[step]
            * self.models[step].predict(X_val[:, feature_idx])
        )

        self._record_metrics("val", y_val, val_predictions)
        metric = self.eval_metric or "val_loss"
        score = self.history[metric][-1]

        if self._is_better(metric, score):
            self.best = BestIteration(step=step, score=score)

        if (
            self.early_stopping_rounds
            and self.early_stopping_rounds <= step - self.best.step
            and self.use_best_model
        ):
            self.models = self.models[: self.best.step + 1]
            self.gammas = self.gammas[: self.best.step + 1]
            self.feature_indices = self.feature_indices[: self.best.step + 1]
            return True

        return False

    # NOTE: chat-pgt
    def plot_history(self, keys: Metrix | list[Metrix]) -> None:
        if plt is None:
            raise ModuleNotFoundError("matplotlib is required for plot_history")

        if isinstance(keys, str):
            keys = [keys]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

        for key in keys:
            values = np.asarray(self.history[key])
            if values.size == 0:
                continue

            steps = np.arange(1, values.size + 1)
            ax.plot(
                steps,
                values,
                linewidth=2.4,
                marker="o",
                markersize=3.5,
                label=key,
            )

            if np.all(np.isnan(values)):
                continue

            best_idx = (
                np.nanargmax(values)
                if key.endswith("roc_auc")
                else np.nanargmin(values)
            )
            ax.scatter(
                steps[best_idx],
                values[best_idx],
                s=70,
                zorder=3,
                edgecolor="white",
                linewidth=1.2,
            )

        ax.set_title("Boosting metrics history", fontsize=14, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Metric value")
        ax.legend(frameon=True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros(X.shape[0])
        for gamma, model, feature_idx in zip(
            self.gammas, self.models, self.feature_indices
        ):
            if model == 0 or feature_idx is None:
                continue
            predictions += gamma * model.predict(X[:, feature_idx])

        res = self.sigmoid(
            self.learning_rate * predictions
        )
        return np.vstack([1 - res, res]).T

    def _find_optimal_gamma(
        self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray
    ) -> float:
        # NOTE: ускорил chat-gpt
        gammas = np.linspace(0, 1, 100)
        z = old_predictions[None, :] + gammas[:, None] * new_predictions[None, :]
        losses = -np.log(self.sigmoid(y[None, :] * z)).mean(axis=1)
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # type: ignore
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])  # type: ignore


class CatEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: list[str]):
        self.columns = columns
        self.values = dict()

    def fit_transform(self, X, y):
        import polars as pl

        X = pl.from_arrow(X)
        y = pl.Series("__target__", y)

        df = X.with_columns(y)
        self.global_mean = y.mean()
        self.values = {}

        for key in self.columns:
            prev_sum = pl.col("__target__").cum_sum().over(key) - pl.col("__target__")
            prev_cnt = pl.col("__target__").cum_count().over(key) - 1

            df = df.with_columns(
                (
                    pl.when(prev_cnt > 0)
                    .then(prev_sum / prev_cnt)
                    .otherwise(self.global_mean)
                ).alias(key)
            )

        return df.drop("__target__")

    def fit(self, X, y):
        X = pl.from_arrow(X)
        y = pl.Series("__target__", y)

        df = X.with_columns(y)
        self.global_mean = y.mean()

        for key in self.columns:
            self.values[key] = df.group_by(key).agg(
                pl.col("__target__").mean().alias("mean_target")
            )

        return self

    def transform(self, X):
        X_transformed = pl.from_arrow(X)

        for key in self.columns:
            X_transformed = (
                X_transformed.join(self.values[key], on=key, how="left")
                .with_columns(
                    pl.col("mean_target").fill_null(self.global_mean).alias(key)
                )
                .drop("mean_target")
            )

        return X_transformed
