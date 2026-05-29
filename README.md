# Supervised Linear Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Реализация линейной регрессии **с нуля** с поддержкой различных методов оптимизации: аналитическое решение, полный, стохастический и мини-батч градиентный спуск. Проект включает полный пайплайн обработки данных, продвинутую инженерию признаков и сравнение с эталонной реализацией из scikit-learn.

---

## 📌 О проекте

**Цель** – построить модель для предсказания цены аренды жилья в Нью-Йорке (датасет с Kaggle) и сравнить качество кастомной реализации с `sklearn.linear_model.LinearRegression`.

**Главная особенность** – все ключевые компоненты написаны вручную:
- Линейная регрессия (с градиентным спуском и аналитическим решением)
- Масштабирование признаков (MinMax, Standard)
- Пайплайн обработки колонок
- Функции визуализации

---

## 📂 Структура репозитория

```
supervisedlinearregression/
├── my_LinearRegression.py          # Кастомная линейная регрессия (GD, SGD, MBGD, Closed-form)
├── my_DataProcessing.py            # Утилиты: обработка выбросов, метрики (R2, MAE, RMSE)
├── my_Scalers.py                   # Кастомные MinMaxScaler и StandardScaler
├── feature_engineering.py          # Преобразование категориальных признаков (amenities, features)
├── columns_proicessing_pipeline.py # Главный пайплайн обработки всех колонок
├── columns_processing/             # Модули обработки специфичных колонок:
│   ├── features_column_pipeline.py
│   ├── street_adress_column_processing.py
│   ├── created_column_processing.py
│   └── location_processing.py
├── my_graphs.py                    # Быстрые функции для EDA (гистограммы, boxplot'ы)
├── ML2.ipynb                       # Основной Jupyter Notebook с демонстрацией полного пайплайна
├── requirements.txt
├── ruff.toml                       # Конфигурация линтера
└── README.md
```

---

## Быстрый старт

### 1. Клонировать репозиторий

```bash
git clone https://github.com/VadimGushchin/supervisedlinearregression.git
cd supervisedlinearregression
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

Основные библиотеки: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `category_encoders`, `kagglehub`, `jupyter`.

### 3. Запустить Jupyter Notebook

```bash
jupyter notebook ML2.ipynb
```

В ноутбуке вы увидите полный цикл: загрузку данных → EDA → обработку колонок → обучение кастомной модели → сравнение с sklearn.

---

## Ключевые модули и их использование

### `my_LinearRegression.py`

Поддерживает четыре режима обучения:

| Метод         | Параметр `method` | Описание                                                                         |
|---------------|-------------------|----------------------------------------------------------------------------------|
| Analytical    | `"analytical"`    | Аналитическое решение через псевдообратную матрицу (только для небольших данных) |
| Full GD       | `"gd"`            | Полный градиентный спуск по всей выборке                                         |
| Stochastic GD | `"sgd"`           | Обновление весов на каждом примере                                               |
| Mini-batch GD | `"mini_batch"`    | Компромисс между GD и SGD                                                        |

Пример:

```python
from my_LinearRegression import MyLinearRegression

model = MyLinearRegression(method="sgd", learning_rate=0.01, n_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### `my_Scalers.py`

```python
from my_Scalers import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### `columns_proicessing_pipeline.py`

Главный пайплайн, который:
- обрабатывает адреса (`street_adress_column_processing`)
- извлекает циклические признаки из даты (`created_column_processing`)
- считает расстояние до центра Манхэттена (`location_processing`)
- превращает категориальные списки (`amenities`, `features`) в бинарные признаки
- кодирует оставшиеся категории с помощью TargetEncoder / CountEncoder
- заполняет пропуски

Использование:

```python
from columns_proicessing_pipeline import process_columns

X_processed = process_columns(df, target_column="price")
```

---

## ✍️ Автор

[Vadim Gushchin](https://github.com/VadimGushchin) – пет проект в рамках обучения в Школе21.
```
