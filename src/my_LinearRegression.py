from typing import Optional, Tuple
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

    def __init__(self,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None,
                 batch_size: int = 256,
                 patience: int = 10):
        """
        Параметры
        ----------
        method : str
            Метод оптимизации ('analytical', 'gd', 'sgd', 'mini-batch')
        learning_rate : float
            Шаг градиентного спуска (для всех градиентных методов)
        n_iterations : int
            Максимальное количество эпох (для градиентных методов)
        random_state : int, optional
            Seed для генератора случайных чисел (перемешивание данных)
        batch_size : int
            Размер батча для метода 'mini-batch' (по умолч. 256)
        patience : int
            Количество эпох без улучшения loss для ранней остановки
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.batch_size = batch_size
        self.patience = patience

        self.weights = None      
        self.bias = None         
        self.loss_history = []   

        self._random_state = np.random.RandomState(random_state) if random_state else np.random

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Добавляет столбец единиц для учёта bias в аналитическом решении.
        """
        return np.c_[np.ones(X.shape[0]), X]

    def _analytical_solution(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Вычисляет веса через нормальное уравнение: θ = (X^T X)^{-1} X^T y.
        Bias хранится отдельно, weights — без единичного столбца.
        """
        X_b = self._add_bias(X)
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def _compute_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Вычисляет градиенты MSE для одного батча.

        Возвращает
        ----------
        grad_weights : np.ndarray
            Градиент по весам (размер n_features)
        grad_bias : float
            Градиент по смещению
        """
        batch_size = X_batch.shape[0]
        y_pred = X_batch @ self.weights + self.bias
        errors = y_pred - y_batch
        grad_weights = (2 / batch_size) * (X_batch.T @ errors)
        grad_bias = (2 / batch_size) * np.sum(errors)
        return grad_weights, grad_bias

    def _initialize_weights(self, n_features: int) -> None:
        """
        Инициализация весов и bias в зависимости от метода.
        Для GD — маленькие случайные значения, для SGD/mini-batch — нули.
        """
        if self.method in ('gd', 'mini-batch'):
            self.weights = self._random_state.randn(n_features) * 0.01
        else:
            self.weights = np.zeros(n_features)
        self.bias = 0.0

    def _batch_generator(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, batch_size: int, shuffle_data: bool):
        """
        Генератор батчей.

        Параметры
        ----------
        X, y : np.ndarray
            Входные данные и целевые значения
        indices : np.ndarray
            Базовый массив индексов (обычно np.arange(n_samples))
        batch_size : int
            Размер батча
        shuffle_data : bool
            Перемешивать ли данные перед генерацией
        """
        n_samples = X.shape[0]
        if shuffle_data:
            shuffled_indices = indices.copy()
            self._random_state.shuffle(shuffled_indices)
            iter_indices = shuffled_indices
        else:
            iter_indices = indices

        for start in range(0, n_samples, batch_size):
            idx = iter_indices[start:start + batch_size]
            yield X[idx], y[idx]

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        self.loss_history = []
        base_indices = np.arange(n_samples)
        
        if self.method == 'gd':
            batch_size = n_samples
            shuffle_data = False
        elif self.method == 'sgd':
            batch_size = 1
            shuffle_data = True
        else:  # mini-batch
            batch_size = self.batch_size
            shuffle_data = True

        best_loss = float('inf')
        no_improvement_count = 0

        for epoch in range(self.n_iterations):
            epoch_loss = 0.0
            for X_batch, y_batch in self._batch_generator(X, y, base_indices, batch_size, shuffle_data):
                grad_weights, grad_bias = self._compute_gradient(X_batch, y_batch)
                
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias
                
                self.weights = np.clip(self.weights, -1e6, 1e6)
                self.bias = np.clip(self.bias, -1e6, 1e6)
                
                if np.isnan(self.weights).any() or np.isinf(self.weights).any():
                    self.loss_history.append(np.inf)
                    return
                
                pred = X_batch @ self.weights + self.bias
                epoch_loss += np.sum((pred - y_batch) ** 2)

            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
            if avg_loss < 0 or np.isnan(avg_loss) or np.isinf(avg_loss) or avg_loss > 1e12:
                # Обучение разошлось
                self.loss_history.append(np.inf)
                return

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    break

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
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
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        if X.shape[0] != y.shape[0]:
            raise ValueError("Несовпадение размерностей: X и y должны иметь одинаковое количество строк")

        self.loss_history = []
        if self.method == 'analytical':
            self._analytical_solution(X, y)
        else:
            self._gradient_descent(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для новых данных.

        Параметры
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Возвращает
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        X = np.asarray(X, dtype=float)
        return X @ self.weights + self.bias

    def rscore(self, X: np.ndarray, y: np.ndarray) -> float:
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
        
        y_pred = self.predict(X)
        y_true = np.asarray(y).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot


class RegularizedLinearRegression(LinearRegression):
    """
    Базовый класс для линейной регрессии с регуляризацией (Ridge, Lasso, ElasticNet).
    Добавляет штраф к функции потерь и градиенту.
    """

    def __init__(self,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None,
                 batch_size: int = 256,
                 patience: int = 10):
        super().__init__(method, learning_rate, n_iterations, random_state, batch_size, patience)
        self.alpha = alpha          # сила регуляризации
        self.l1_ratio = l1_ratio    # доля L1 (0 – только L2, 1 – только L1)

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент регуляризационного члена (по умолчанию без регуляризации).
        """
        return np.zeros_like(weights)

    def _compute_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Переопределяет вычисление градиента, добавляя регуляризацию.
        """
        grad_weights_mse, grad_bias = super()._compute_gradient(X_batch, y_batch)
        grad_weights_reg = self._regularization_gradient(self.weights)
        
        grad_weights = grad_weights_mse + grad_weights_reg
        grad_weights = np.clip(grad_weights, -1e3, 1e3)
        
        return grad_weights, grad_bias


class RidgeRegression(RegularizedLinearRegression):
    """
    Ridge регрессия (L2-регуляризация).
    """

    def __init__(self,
                 alpha: float = 1.0,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None,
                 batch_size: int = 256,
                 patience: int = 10):
        super().__init__(alpha=alpha, l1_ratio=0.0, method=method,
                         learning_rate=learning_rate, n_iterations=n_iterations,
                         random_state=random_state, batch_size=batch_size, patience=patience)

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """Градиент L2: 2 * alpha * w"""
        return 2 * self.alpha * weights


class LassoRegression(RegularizedLinearRegression):
    """
    Lasso регрессия (L1-регуляризация).
    """

    def __init__(self,
                 alpha: float = 1.0,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None,
                 batch_size: int = 256,
                 patience: int = 10):
        super().__init__(alpha=alpha, l1_ratio=1.0, method=method,
                         learning_rate=learning_rate, n_iterations=n_iterations,
                         random_state=random_state, batch_size=batch_size, patience=patience)

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        grad = self.alpha * np.sign(weights)
        return np.clip(grad, -1e3, 1e3)


class ElasticNetRegression(RegularizedLinearRegression):
    """
    ElasticNet регрессия (комбинация L1 и L2).
    """

    def __init__(self,
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None,
                 batch_size: int = 256,
                 patience: int = 10):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, method=method,
                         learning_rate=learning_rate, n_iterations=n_iterations,
                         random_state=random_state, batch_size=batch_size, patience=patience)

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент ElasticNet: alpha * (l1_ratio * sign(w) + 2*(1-l1_ratio)*w)
        """
        l1_grad = self.l1_ratio * np.sign(weights)
        l2_grad = 2 * (1 - self.l1_ratio) * weights
        return self.alpha * (l1_grad + l2_grad)