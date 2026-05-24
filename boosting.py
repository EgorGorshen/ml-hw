from __future__ import annotations

import logging
from collections import defaultdict
import matplotlib.pyplot as pl
import numpy as np
import json
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.base import ClassifierMixin

# note: да я знаю что такое literal и named tuple
from typing import Literal, NamedTuple

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
        early_stopping_rounds: int | None = 0,
        eval_metric: Metrix | None = None,
        val_size: float | None = 0.1,
        use_best_model: bool = False,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
        history_commit_path: Path | None = None,
    ):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.history_commit_path = history_commit_path

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0.0] * (n_estimators)

        self.random_state = random_state
        self.best = BestIteration(step=0, score=-np.inf)
        self.eval_set = eval_set

        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.val_size = val_size
        self.eval_metric = eval_metric
        self.use_best_model = use_best_model

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.unti_grad_fn = lambda y, z: y / (1 + np.exp(y * z))

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, step: int, train_predictions: np.ndarray
    ) -> None:
        base_model = self.base_model_class(**self.base_model_params)
        anti_gradient = self.unti_grad_fn(y, train_predictions)
        base_model.fit(X, anti_gradient)
        new_predictions = base_model.predict(X)
        self.models[step] = base_model
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
            logging.warning(
                "укажите метрику останова! Поставил её дефолтной: `val_loss`"
            )
            self.eval_metric = "val_loss"

        return True

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
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
        val_predictions += (
            self.learning_rate * self.gammas[step] * self.models[step].predict(X_val)
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
            return True

        return False

    # NOTE: chat-pgt
    def plot_history(self, keys: Metrix | list[Metrix]) -> None:
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
        res = self.sigmoid(
            self.learning_rate
            * sum(g * model.predict(X) for (g, model) in zip(self.gammas, self.models))  # type: ignore
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
