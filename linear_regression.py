import numpy as np
from interfaces import (
    LossFunction,
    LossFunctionClosedFormMixin,
    LinearRegressionInterface,
    AbstractOptimizer,
)
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
import numpy as np
from scipy.sparse.linalg import svds


class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(
        self,
        analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        v = X @ w - y
        return float(v @ v) / X.shape[0]

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        return (2.0 / X.shape[0]) * (X.T @ (X @ w - y))

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса;
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.inv(X.T @ X) @ X.T @ y

    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        n, m = X.shape
        k = min(n, m) - 1
        if k <= 0:
            return np.zeros(m)

        U, s, Vt = svds(X, k=k, solver="arpack", tol=0.0)

        idx = np.argsort(s)[::-1]
        U, s, Vt = U[:, idx], s[idx], Vt[idx, :]

        tol = 1e-12 * s[0]
        s_inv = np.where(s > tol, 1.0 / s, 0.0)

        return Vt.T @ (s_inv * (U.T @ y))


class L2Regularization(LossFunction):

    def __init__(
        self,
        core_loss: LossFunction,
        mu_rate: float = 1.0,
        analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ):
        self.core_loss = core_loss
        self.mu_rate = mu_rate

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        penalty_part = self.mu_rate * w
        return core_part + penalty_part

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        return self.core_loss.loss(X, y, w) + self.mu_rate / 2 * (w @ w)


class LogCosh(LossFunction):
    def __init__(
        self,
        analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ) -> None:
        pass

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        e = X @ w - y
        return (X.T @ np.tanh(e)) / X.shape[0]

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        e = X @ w - y
        return float(np.mean(np.log(np.cosh(e))))


class HuberLoss(LossFunction):
    def __init__(
        self,
        delta: float = 1.0,
        analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ) -> None:
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        e = np.abs(X @ w - y)

        return float(
            np.mean(
                np.where(
                    e < self.delta,
                    0.5 * e**2,
                    self.delta * (e - 0.5 * self.delta),
                )
            )
        )

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        e = X @ w - y
        psi = np.where(np.abs(e) <= self.delta, e, self.delta * np.sign(e))
        return (X.T @ psi) / X.shape[0]


class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss(),
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function: LossFunction = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        return X @ self.w

    def compute_gradients(
        self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None
    ) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if (X_batch is None) or (y_batch is None):
            return self.loss_function.gradient(self.X_train, self.y_train, self.w)
        return self.loss_function.gradient(X_batch, y_batch, self.w)

    def compute_loss(
        self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None
    ) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """

        if (X_batch is None) or (y_batch is None):
            return self.loss_function.loss(self.X_train, self.y_train, self.w)
        return self.loss_function.loss(X_batch, y_batch, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.

        X: np.ndarray,
        y: np.ndarray
        """
        self.X_train, self.y_train = X, y
        self.w = np.zeros(self.X_train.shape[1], dtype=float)

        self.optimizer.optimize()
