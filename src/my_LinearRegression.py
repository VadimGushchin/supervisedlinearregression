import numpy as np
from typing import Optional, Tuple

class LinearRegression:
    """
    Линейная регрессия с тремя методами оптимизации.
    """
    
    def __init__(self, 
                 method: str = 'sgd', 
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None):
        
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.weights = None  
        self.bias = None     
        self.loss_history = []  
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Добавляет столбец единиц для bias term."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _analytical_solution(self, X: np.ndarray, y: np.ndarray) -> None:
        """Аналитическое решение через нормальное уравнение."""
        X_b = self._add_bias(X)
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _compute_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Вычисляет градиент функции потерь MSE для батча.

        """
        batch_size = X_batch.shape[0]
        
        y_pred = X_batch @ self.weights + self.bias
        
        errors = y_pred - y_batch
        
        dw = (2 / batch_size) * (X_batch.T @ errors)
        db = (2 / batch_size) * np.sum(errors)
        
        return dw, db
    
    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Полный градиентный спуск (Batch Gradient Descent).
        Использует ВСЕ данные для вычисления градиента на каждой итерации.

        """
        n_features = X.shape[1] 
        
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for _ in range(self.n_iterations):
            dw, db = self._compute_gradient(X, y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            y_pred = X @ self.weights + self.bias
            current_mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(current_mse)
            
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                    break
    
    def _stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Стохастический градиентный спуск (SGD) с ранней остановкой
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        best_loss = float('inf')
        patience = 10  
        wait = 0
        
        for iteration in range(self.n_iterations):
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state + iteration)
                indices = rng.permutation(n_samples)
            else:
                indices = np.random.permutation(n_samples)
            
            epoch_loss = 0.0
            
            for idx in indices:
                xi = X[idx]
                yi = y[idx]
                
                y_pred = np.dot(xi, self.weights) + self.bias
                error = y_pred - yi
                
                dw = 2 * error * xi
                db = 2 * error
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                epoch_loss += error * error
            
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < 1e-6:
                    break
    
    def _mini_batch_gradient_descent(self, X, y, batch_size=256):
        """
        Оптимизированный Mini-Batch градиентный спуск
        Без лишних созданий массивов, с векторизацией
        """
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        
        best_loss = float('inf')
        patience = 10
        wait = 0
        
        for epoch in range(self.n_iterations):
            # Перемешиваем данные
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # Mini-batch цикл
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                batch_len = len(X_batch)
                
                # Векторизованные вычисления (без циклов по объектам!)
                y_pred = X_batch @ self.weights + self.bias
                errors = y_pred - y_batch
                
                # Градиенты за один раз
                dw = (2 / batch_len) * (X_batch.T @ errors)
                db = (2 / batch_len) * np.sum(errors)
                
                # Обновление
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Считаем loss для мониторинга
                epoch_loss += np.sum(errors ** 2)
                n_batches += 1
            
            avg_loss = epoch_loss / n_samples
            self.loss_history.append(avg_loss)
            
            # Ранняя остановка
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Обучение модели линейной регрессии.

        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).flatten()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Несовпадение размерностей")
        
        self.loss_history = []
        
        if self.method == 'analytical':
            self._analytical_solution(X, y)

        elif self.method == 'gd':
            self._gradient_descent(X, y)

        elif self.method == 'sgd':
            self._stochastic_gradient_descent(X, y)
        
        elif self.method == 'mini-batch':
            self._mini_batch_gradient_descent(X, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание для новых данных.
        
        """        
        X = np.array(X, dtype=float)
        return X @ self.weights + self.bias
    
    def rscore(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисляет коэффициент детерминации R².
        
        """
        y_pred = self.predict(X)
        y_true = np.array(y).flatten()
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
class RegularizedLinearRegression(LinearRegression):
    """
    Базовый класс для регуляризованной линейной регрессии.
    Наследуется от LinearRegression и добавляет регуляризацию.

    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None):
        
        super().__init__(method, 
                         learning_rate, 
                         n_iterations, 
                         random_state
                        )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Вычисляет градиент регуляризационного члена.
        
        """
        return np.zeros_like(weights)
    
    def _compute_gradient(self, X_batch: np.ndarray, y_batch: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Переопределяет вычисление градиента с добавлением регуляризации.
        
        """
        dw_mse, db = super()._compute_gradient(X_batch, y_batch)
        
        dw_reg = self._regularization_gradient(self.weights)
        
        dw = dw_mse + dw_reg
        
        return dw, db


class RidgeRegression(RegularizedLinearRegression):
    """
    Ridge регрессия с L2 регуляризацией.

    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None):
        
        super().__init__(
            alpha=alpha,
            l1_ratio=0.0,
            method=method,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state
        )
    
    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент L2 регуляризации.
        
        """
        return 2 * self.alpha * weights


class LassoRegression(RegularizedLinearRegression):
    """
    Lasso регрессия с L1 регуляризацией.
    
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None):
        
        super().__init__(
            alpha=alpha,
            l1_ratio=1.0,
            method=method,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state
        )
    
    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Субградиент L1 регуляризации.

        """
        return self.alpha * np.sign(weights)


class ElasticNetRegression(RegularizedLinearRegression):
    """
    ElasticNet регрессия с комбинацией L1 и L2 регуляризации.
    
    """
    
    def __init__(self, 
                 alpha: float = 1.0,
                 l1_ratio: float = 0.5,
                 method: str = 'sgd',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 random_state: Optional[int] = None):
        
        super().__init__(
            alpha=alpha,
            l1_ratio=l1_ratio,
            method=method,
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            random_state=random_state
        )
    
    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Градиент ElasticNet регуляризации.

        """
        l1_component = self.l1_ratio * np.sign(weights)
        l2_component = 2 * (1 - self.l1_ratio) * weights
        return self.alpha * (l1_component + l2_component)
