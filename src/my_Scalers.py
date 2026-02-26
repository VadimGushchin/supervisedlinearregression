import numpy as np


class CustomMinMaxScaler:
    """
    Кастомная реализация MinMaxScaler для нормализации признаков в диапазон [0, 1].

    Формула: X_scaled = (X - min) / (max - min)
    """

    def __init__(self):
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        """
        Вычисляет min, max и размах для каждого признака.

        """
        X = np.array(X)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1.0

        return self

    def transform(self, X, inverse=False):
        """
        Прямое (inverse=False) или обратное (inverse=True) преобразование.

        """
        X = np.array(X)

        if inverse:
            return X * self.data_range_ + self.data_min_
        else:
            return (X - self.data_min_) / self.data_range_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CustomStandardScaler:
    """
    Кастомная реализация StandardScaler для стандартизации признаков (μ=0, σ=1).

    Формула: X_scaled = (X - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Вычисляет среднее и стандартное отклонение для каждого признака.

        """
        X = np.array(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X, inverse=False):
        """
        Прямое (inverse=False) или обратное (inverse=True) преобразование.

        """
        X = np.array(X)

        if inverse:
            return X * self.std_ + self.mean_
        else:
            return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
