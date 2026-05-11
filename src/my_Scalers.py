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

    def fit(self, x):
        """
        Вычисляет min, max и размах для каждого признака.

        """
        x = np.array(x)
        self.data_min_ = x.min(axis=0)
        self.data_max_ = x.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1.0

        return self

    def transform(self, x, inverse=False):
        """
        Прямое (inverse=False) или обратное (inverse=True) преобразование.

        """
        x = np.array(x)

        if inverse:
            return x * self.data_range_ + self.data_min_
        else:
            return (x - self.data_min_) / self.data_range_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class CustomStandardScaler:
    """
    Кастомная реализация StandardScaler для стандартизации признаков (μ=0, σ=1).

    Формула: X_scaled = (X - mean) / std
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        """
        Вычисляет среднее и стандартное отклонение для каждого признака.

        """
        x = np.array(x)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x, inverse=False):
        """
        Прямое (inverse=False) или обратное (inverse=True) преобразование.

        """
        x = np.array(x)

        if inverse:
            return x * self.std_ + self.mean_
        else:
            return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        return self.fit(x).transform(x)
