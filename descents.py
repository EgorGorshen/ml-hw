import numpy as np
from abc import ABC, abstractmethod
from interfaces import (
    LearningRateSchedule,
    AbstractOptimizer,
    LinearRegressionInterface,
)


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR(),
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """

        raise NotImplementedError()

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """

        self.model.loss_history.append(self.model.compute_loss())
        while self.iteration < self.max_iter:
            delta = self._step()
            self.model.loss_history.append(self.model.compute_loss())
            if (
                np.isnan(delta).any()
                or self.tolerance > 0
                and np.dot(delta, delta) < self.tolerance
            ):
                break


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        delta = self.lr_schedule.get_lr(self.iteration) * self.model.compute_gradients()
        self.model.w -= delta
        return -delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X = self.model.X_train
        y = self.model.y_train

        # 1) выбрать случайный батч
        idx = np.random.randint(0, X.shape[0], size=self.batch_size)

        # 2) вычислить градиенты на батче
        grad = self.model.compute_gradients(X[idx], y[idx])

        # 3) обновить веса модели
        lr = self.lr_schedule.get_lr(self.iteration)
        delta = lr * grad
        self.model.w -= delta

        return -delta


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        n, d = self.model.X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros(self.model.X_train.shape)
            self.avg_grad = np.zeros(d)

        idx = np.random.randint(n, size=self.batch_size)
        xb, yb = self.model.X_train[idx], self.model.y_train[idx]

        new_grads = self.model.compute_gradients(xb, yb)
        corr = (new_grads - self.grad_memory[idx]).mean(axis=0) / n * self.batch_size
        self.avg_grad += corr
        self.grad_memory[idx] = new_grads

        diff = self.lr_schedule.get_lr(self.iteration) * self.avg_grad
        self.model.w -= diff
        return -diff


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:

        if self.velocity is None:
            self.velocity = np.zeros(self.model.X_train.shape[1])

        self.velocity = (
            self.beta * self.velocity
            + self.lr_schedule.get_lr(self.iteration)
            * self.model.compute_gradients()
            * self.model.compute_loss()
        )

        self.model.w -= self.velocity

        return -self.velocity


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros(self.model.X_train.shape[1])
        if self.v is None:
            self.v = np.zeros(self.model.X_train.shape[1])

        grad = self.model.compute_gradients()

        self.m = self.m * self.beta1 + (1 - self.beta1) * grad
        self.v = self.v * self.beta2 + (1 - self.beta2) * grad**2

        m_hat = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.iteration + 1))

        dif = 0.01 * m_hat / (np.sqrt(v_hat) + self.eps)

        self.model.w -= dif

        return -dif


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """

        self.model.w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )
