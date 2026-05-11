import numpy as np


class LinearRegression:
    """
    Линейная регрессия с аналитическим решением и градиентными методами.

    Поддерживаемые методы:
    - 'analytical' : нормальное уравнение (точное, быстро для p < 10^4)
    - 'gd'         : полный градиентный спуск (batch)
    - 'sgd'        : стохастический градиентный спуск (один объект за раз)
    - 'mini-batch' : мини-батчевый градиентный спуск (батчи фиксированного размера)
    """

    def __init__(
        self,
        method: str = "sgd",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int | None = None,
        batch_size: int = 256,
        patience: int = 10,
        optimizer: str = "sgd",
        eps: float = 1e-8,
    ):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer = optimizer
        self.eps = eps
        self.epoch_callback = None

        self.weights = None
        self.bias = None
        self.loss_history = []
        self._random_state = (
            np.random.RandomState(random_state) if random_state else np.random
        )

    def _add_bias(self, x: np.ndarray) -> np.ndarray:
        """
        Добавляет столбец единиц для учёта bias в аналитическом решении.
        """
        return np.c_[np.ones(x.shape[0]), x]

    def _analytical_solution(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Вычисляет веса через нормальное уравнение: θ = (X^T X)^{-1} X^T y.
        Bias хранится отдельно, weights — без единичного столбца.
        """
        x_b = self._add_bias(x)
        theta = np.linalg.pinv(x_b.T @ x_b) @ x_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def _compute_gradient(
        self, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Вычисляет градиенты MSE и регуляризации, суммирует их.

        Параметры
        ----------
        X_batch : np.ndarray
            Батч признаков.
        y_batch : np.ndarray
            Батч целевых значений.

        Возвращает
        -------
        grad_weights : np.ndarray
            Градиент по весам с учётом регуляризации.
        grad_bias : float
            Градиент по смещению (без регуляризации, штраф не применяется к bias).
        """
        y_pred = x_batch @ self.weights + self.bias
        errors = y_pred - y_batch
        batch_size = x_batch.shape[0]
        grad_weights = (2 / batch_size) * (x_batch.T @ errors)
        grad_bias = (2 / batch_size) * np.sum(errors)
        return grad_weights, grad_bias

    def _initialize_weights(self, n_features: int) -> None:
        """
        Инициализирует веса и смещение малыми случайными числами.

        Параметры
        ----------
        n_features : int
            Количество признаков (размерность весов).
        """
        self.weights = self._random_state.randn(n_features) * 0.01
        self.bias = 0.0

    def _gradient_descent(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Выполняет градиентный спуск (полный, стохастический или мини-батчевый)
        в зависимости от self.method.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Обучающие признаки.
        y : np.ndarray, shape (n_samples,)
            Целевые значения.

        Примечания
        ----------
        - Для метода 'gd' используется полный батч (все данные).
        - Для 'sgd' – батч размером 1 с перемешиванием на каждую эпоху.
        - Для 'mini-batch' – батч размера self.batch_size.
        - Ранняя остановка срабатывает, если loss не улучшается на self.patience эпох.
        """
        n_samples, n_features = x.shape
        self._initialize_weights(n_features)
        self.loss_history = []

        if self.optimizer == "adagrad":
            cache_w = np.zeros_like(self.weights)
            cache_b = 0.0

        if self.method == "gd":
            batch_size = n_samples
            shuffle = False
        elif self.method == "sgd":
            batch_size = 1
            shuffle = True
        elif self.method == "mini-batch":
            batch_size = self.batch_size
            shuffle = True
        else:
            raise ValueError(f"Unknown method: {self.method}")

        best_loss = float("inf")
        no_improvement = 0

        for epoch in range(self.n_iterations):
            if shuffle:
                indices = self._random_state.permutation(n_samples)
                x_epoch = x[indices]
                y_epoch = y[indices]
            else:
                x_epoch = x
                y_epoch = y

            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                x_batch = x_epoch[start : start + batch_size]
                y_batch = y_epoch[start : start + batch_size]

                grad_w, grad_b = self._compute_gradient(x_batch, y_batch)

                max_grad_norm = 1e3
                grad_norm = np.sqrt(np.sum(grad_w**2) + grad_b**2)
                if grad_norm > max_grad_norm:
                    scale = max_grad_norm / grad_norm
                    grad_w *= scale
                    grad_b *= scale
                    
                if self.optimizer == "sgd":
                    self.weights -= self.learning_rate * grad_w
                    self.bias -= self.learning_rate * grad_b
                elif self.optimizer == "adagrad":
                    cache_w += grad_w**2
                    cache_b += grad_b**2
                    self.weights -= (
                        self.learning_rate / (np.sqrt(cache_w) + self.eps)
                    ) * grad_w
                    self.bias -= (
                        self.learning_rate / (np.sqrt(cache_b) + self.eps)
                    ) * grad_b
                else:
                    raise ValueError(f"Unknown optimizer: {self.optimizer}")

                # Накопление MSE loss
                y_pred = x_batch @ self.weights + self.bias
                epoch_loss += np.sum((y_pred - y_batch) ** 2)

            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)

            if self.epoch_callback is not None:
                self.epoch_callback(epoch, avg_loss)

            # Ранняя остановка
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= self.patience:
                    break

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Обучение модели.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Обучающая выборка
        y : np.ndarray, shape (n_samples,)
            Целевые значения

        Возвращает
        -------
        self : LinearRegression
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "Несовпадение размерностей: x и y должны иметь одинаковое количество строк"
            )

        self.loss_history = []
        if self.method == "analytical":
            self._analytical_solution(x, y)
        else:
            self._gradient_descent(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Предсказание для новых данных.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Возвращает
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        x = np.asarray(x, dtype=float)
        return x @ self.weights + self.bias

    def rscore(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Коэффициент детерминации R².

        Параметры
        ----------
        X : np.ndarray
            Данные для предсказания
        y : np.ndarray
            Истинные значения

        Возвращает
        -------
        r2 : float
            R² (чем ближе к 1, тем лучше)
        """

        y_pred = self.predict(x)
        y_true = np.asarray(y).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


class RegularizedLinearRegression(LinearRegression):
    """
    Базовый класс для линейной регрессии с регуляризацией (Ridge, Lasso, ElasticNet).

    Добавляет регуляризационный член к функции потерь и градиенту.
    Сам по себе не предназначен для прямого использования, только как родительский.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        method: str = "sgd",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int | None = None,
        batch_size: int = 256,
        patience: int = 10,
        optimizer: str = "sgd",
        eps: float = 1e-8,
    ):
        super().__init__(
            method,
            learning_rate,
            n_iterations,
            random_state,
            batch_size,
            patience,
            optimizer=optimizer,
            eps=eps,
        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент регуляризационного члена.

        Параметры
        ----------
        weights : np.ndarray
            Вектор весов модели.

        Возвращает
        -------
        np.ndarray
            Градиент регуляризации (той же формы, что и weights).
            По умолчанию (без регуляризации) возвращает нулевой вектор.
        """
        return np.zeros_like(weights)

    def _compute_gradient(
        self, x_batch: np.ndarray, y_batch: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Переопределяет вычисление градиента, добавляя регуляризацию.
        """
        grad_weights_mse, grad_bias = super()._compute_gradient(x_batch, y_batch)
        grad_weights_reg = self._regularization_gradient(self.weights)

        grad_weights = grad_weights_mse + grad_weights_reg

        return grad_weights, grad_bias


class RidgeRegression(RegularizedLinearRegression):
    """
    Гребневая регрессия (Ridge) с L2-регуляризацией.

    Штраф: alpha * sum(w^2). Градиент регуляризации: 2 * alpha * w.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        method: str = "sgd",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int | None = None,
        batch_size: int = 256,
        patience: int = 10,
        optimizer: str = "sgd",
        eps: float = 1e-8,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=0.0,
            method=method,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state,
            batch_size=batch_size,
            patience=patience,
            optimizer=optimizer,
            eps=eps,
        )

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент L2-регуляризации: 2 * alpha * weights.

        Параметры
        ----------
        weights : np.ndarray
            Веса модели.

        Возвращает
        -------
        np.ndarray
            Градиент регуляризации.
        """
        return 2 * self.alpha * weights


class LassoRegression(RegularizedLinearRegression):
    """
    Лассо-регрессия (Lasso) с L1-регуляризацией.

    Штраф: alpha * sum(|w|). Градиент регуляризации: alpha * sign(w).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        method: str = "sgd",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int | None = None,
        batch_size: int = 256,
        patience: int = 10,
        optimizer: str = "sgd",
        eps: float = 1e-8,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,
            method=method,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state,
            batch_size=batch_size,
            patience=patience,
            optimizer=optimizer,
            eps=eps,
        )

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент L1-регуляризации: alpha * sign(weights),
        с клиппингом для численной стабильности.

        Параметры
        ----------
        weights : np.ndarray
            Веса модели.

        Возвращает
        -------
        np.ndarray
            Градиент регуляризации.
        """
        return self.alpha * np.sign(weights)


class ElasticNetRegression(RegularizedLinearRegression):
    """
    Эластичная сеть (ElasticNet) – комбинация L1 и L2 регуляризации.

    Штраф: alpha * (l1_ratio * sum(|w|) + (1-l1_ratio) * sum(w^2)).
    Градиент: alpha * (l1_ratio * sign(w) + 2*(1-l1_ratio)*w).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        method: str = "sgd",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        random_state: int | None = None,
        batch_size: int = 256,
        patience: int = 10,
        optimizer: str = "sgd",
        eps: float = 1e-8,
    ):
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            method=method,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state,
            batch_size=batch_size,
            patience=patience,
            optimizer=optimizer,
            eps=eps,
        )

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент регуляризации ElasticNet.

        Параметры
        ----------
        weights : np.ndarray
            Веса модели.

        Возвращает
        -------
        np.ndarray
            Градиент регуляризации.
        """
        l1_grad = self.l1_ratio * np.sign(weights)
        l2_grad = 2 * (1 - self.l1_ratio) * weights
        return self.alpha * (l1_grad + l2_grad)
