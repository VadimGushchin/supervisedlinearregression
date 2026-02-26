import matplotlib.pyplot as plt
import pandas as pd
import my_LinearRegression as linreg
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.base import clone


def remove_price_outliers(df, column="price", lower_q=0.01, upper_q=0.99):
    """
    Удаляет выбросы в цене по квантилям
    """
    q_low = df[column].quantile(lower_q)
    q_hi = df[column].quantile(upper_q)

    return df[(df[column] > q_low) & (df[column] < q_hi)].copy()


def clean_features(feature_str):
    """
    Преобразует строку вида "['Elevator', 'CatsAllowed']" в список ['Elevator', 'CatsAllowed'].
    """
    if isinstance(feature_str, float) and pd.isna(feature_str):
        return []

    if feature_str is None:
        return []

    if isinstance(feature_str, str) and feature_str.strip() == "":
        return []

    text = str(feature_str)

    text = text.replace("[", "").replace("]", "")

    text = text.replace("'", "").replace('"', "")

    items = []
    for item in text.split(","):
        cleaned_item = item.strip()
        if cleaned_item:
            items.append(cleaned_item)

    return items


def normalize_feature_name(name):
    """
    Нормализует названия признаков для учёта опечаток.
    """
    name = name.lower()
    replacements = {
        "laundryinbuilding": "LaundryInBuilding",
        "laundryinunit": "LaundryInUnit",
        "pre-war": "Pre-War",
        "highspeedinternet": "HighSpeedInternet",
        "outdoorspace": "OutdoorSpace",
        "roofdeck": "RoofDeck",
        "fitnesscenter": "FitnessCenter",
        "newconstruction": "NewConstruction",
        "nofee": "NoFee",
        "catsallowed": "CatsAllowed",
        "dogsallowed": "DogsAllowed",
        "hardwoodfloors": "HardwoodFloors",
        "dishwasher": "Dishwasher",
        "doorman": "Doorman",
        "elevator": "Elevator",
        "balcony": "Balcony",
        "swimmingpool": "SwimmingPool",
        "diningroom": "DiningRoom",
        "terrace": "Terrace",
    }

    for key, value in replacements.items():
        if key in name:
            return value

    return name.capitalize()


def norm_table():
    normalization_table = pd.DataFrame(
        {
            "Метод ML": [
                "Линейная регрессия (GD/SGD)",
                "Линейная регрессия (аналитика)",
                "Ridge/Lasso/ElasticNet (GD)",
                "Ridge (аналитика)",
                "Логистическая регрессия",
                "SVM (RBF, полиномиальное)",
                "SVM (линейное ядро)",
                "KNN, K-Means",
                "PCA, SVD",
                "Нейронные сети",
                "Деревья решений",
                "Random Forest",
                "Gradient Boosting (деревья)",
                "Наивный Байес",
                "LDA",
            ],
            "Нужна нормализация?": [
                "ДА",
                "НЕТ",
                "ДА",
                "НЕТ",
                "ДА",
                "ДА",
                "Желательно",
                "ДА",
                "ДА",
                "ДА",
                "НЕТ",
                "НЕТ",
                "НЕТ",
                "НЕТ",
                "Желательно",
            ],
            "Обоснование": [
                "Разный масштаб → зигзаги → медленная сходимость",
                "Веса автоматически масштабируются обратно " "пропорционально",
                "Регуляризация штрафует веса одинаково, " "масштаб критичен",
                "Аналитическое решение подстраивает " "веса под масштаб",
                "Градиентный спуск + регуляризация",
                "Чувствителен к евклидовым расстояниям",
                "Без регуляризации можно без, " "с регуляризацией — обязательно",
                "Евклидово расстояние — признаки " "с большим масштабом доминируют",
                "Чувствителен к дисперсии " "признаков",
                "Насыщение активаций, ускорение сходимости",
                "Работают с порогами, масштаб не важен",
                "Наследует от деревьев",
                "На деревьях — не нужна",
                "Работает с распределениями, не с расстояниями",
                "Чувствителен к масштабу, но внутри нормирует",
            ],
        }
    )

    return normalization_table


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Обучает модель и возвращает словарь с метриками.

    """
    start_time = time.time()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    fit_time = time.time() - start_time

    # MAE и RMSE из sklearn
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_val = root_mean_squared_error(y_val, y_pred_val)

    # R2 (для кастомных используем rscore, для sklearn - r2_score)
    if "Custom" in model_name:
        r2_train = model.rscore(X_train, y_train)
        r2_val = model.rscore(X_val, y_val)
    else:
        r2_train = r2_score(y_train, y_pred_train)
        r2_val = r2_score(y_val, y_pred_val)

    return {
        "model": model_name,
        "mae_train": mae_train,
        "mae_val": mae_val,
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "r2_train": r2_train,
        "r2_val": r2_val,
        "fit_time": fit_time,
    }


def train_and_predict(model, X_train, y_train, X_val):
    """Обучает модель и возвращает предсказания."""
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    return {
        "model": model_clone,
        "pred_train": model_clone.predict(X_train),
        "pred_val": model_clone.predict(X_val),
    }


def metrics_in_dollars(
    y_true_train, y_true_val, y_pred_train, y_pred_val, target_scaler
):
    """Пересчитывает предсказания из scaled в доллары и считает метрики."""
    y_pred_train_dollars = target_scaler.inverse_transform(
        y_pred_train.reshape(-1, 1)
    ).ravel()
    y_pred_val_dollars = target_scaler.inverse_transform(
        y_pred_val.reshape(-1, 1)
    ).ravel()

    return {
        "mae_train": mean_absolute_error(y_true_train, y_pred_train_dollars),
        "mae_val": mean_absolute_error(y_true_val, y_pred_val_dollars),
        "rmse_train": root_mean_squared_error(y_true_train, y_pred_train_dollars),
        "rmse_val": root_mean_squared_error(y_true_val, y_pred_val_dollars),
        "r2_train": r2_score(y_true_train, y_pred_train_dollars),
        "r2_val": r2_score(y_true_val, y_pred_val_dollars),
    }

def format_model_name(model):
    model_str = str(model)
    
    # Определяем тип регуляризации
    if 'Ridge' in model_str:
        reg_type = 'Ridge'
    elif 'Lasso' in model_str:
        reg_type = 'Lasso'
    elif 'ElasticNet' in model_str:
        reg_type = 'ElasticNet'
    elif 'LinearRegression' in model_str:
        reg_type = 'Linear'
    else:
        reg_type = 'Unknown'
    
    # Определяем наличие Scaler
    if 'StandardScaler' in model_str:
        scaler = 'Standard'
    elif 'MinMaxScaler' in model_str:
        scaler = 'MinMax'
    else:
        scaler = 'NoScaler'
    
    # Извлекаем alpha (если есть)
    import re
    alpha_match = re.search(r'alpha=([\d.]+)', model_str)
    alpha = f"alpha={alpha_match.group(1)}" if alpha_match else ""
    
    # Для ElasticNet извлекаем l1_ratio
    l1_match = re.search(r'l1_ratio=([\d.]+)', model_str)
    l1 = f", l1={l1_match.group(1)}" if l1_match else ""
    
    # Формируем итоговое название
    if reg_type == 'Linear':
        return f"{scaler}+Linear"
    elif reg_type in ['Ridge', 'Lasso']:
        return f"{scaler}+{reg_type}({alpha})"
    elif reg_type == 'ElasticNet':
        return f"{scaler}+{reg_type}({alpha}{l1})"
    else:
        return model_str[:50]


class LogTransformedModel:
    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, y):
        y_log = np.log1p(y)  # log(1+y) для защиты от 0
        self.base_model.fit(X, y_log)
        return self

    def predict(self, X):
        y_pred_log = self.base_model.predict(X)
        return np.expm1(y_pred_log)
