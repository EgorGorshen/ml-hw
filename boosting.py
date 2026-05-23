from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class BoostingClassifier(ClassifierMixin):
    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: dict | None = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = 42,
        verbose: bool = True,
        time_series: bool = False,
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = [0] * (n_estimators)
        self.gammas = [0] * (n_estimators)

        self.random_state = (
            random_state  # не забудьте вставить его везде, где у вас возникает рандом
        )
        self.verbose = verbose
        self.time_series = time_series

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.unti_grad_fn = lambda y, z: y / (1 + np.exp(y * z))

    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, step: int, train_predictions: np.ndarray
    ) -> None:
        base_model = self.base_model_class(**self.base_model_params)
        base_model.fit(X, y)
        new_predictions = base_model.predict(X)
        self.models[step] = base_model
        self.gammas[step] = self._find_optimal_gamma(
            y, train_predictions, new_predictions
        )
        train_predictions += self.learning_rate * self.gammas[step] * new_predictions

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        train_predictions = np.zeros(X_train.shape[0])
        self.classes_ = np.unique(
            y_train
        )  # не рекомендуется убирать, нужно для калибровки
        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for i in estimator_range:
            self.partial_fit(X_train, y_train, i, train_predictions)
            sigmoid = self.sigmoid(train_predictions)
            self.history["train_loss"].append(self.loss_fn(y_train, sigmoid))
            self.history["train_roc-auc"].append(
                roc_auc_score(y_true=(y_train == 1), y_score=sigmoid)
            )

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        res = self.sigmoid(
            self.learning_rate
            * sum(g * model.predict(X) for (g, model) in zip(self.gammas, self.models))
        )
        return np.vstack([1 - res, res]).T

    def _find_optimal_gamma(
        self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray
    ) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])
