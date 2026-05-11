import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np


def remove_price_outliers(df, column="price", lower_q=0.01, upper_q=0.99):
    q_low = df[column].quantile(lower_q)
    q_hi = df[column].quantile(upper_q)
    return df[(df[column] > q_low) & (df[column] < q_hi)].copy()


def clean_features(feature_str):
    if feature_str is None or (isinstance(feature_str, float) and pd.isna(feature_str)):
        return []
    if isinstance(feature_str, list):
        return feature_str
    text = str(feature_str).strip()
    if text in ("", "[]", "nan", "None"):
        return []
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return [item.strip() for item in text.split(",") if item.strip()]


def evaluate_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    model_name,
    y_train_orig=None,
    y_val_orig=None,
    log_base=np.exp(1)
):
    """
    Оценивает уже обученную модель на train и val.
    
    Параметры
    ----------
    model : обученная модель с методом predict
    x_train, y_train : обучающие данные (в том масштабе, в котором модель обучалась)
    x_val, y_val : валидационные данные
    model_name : str
    y_train_orig, y_val_orig : исходные целевые переменные (до логарифмирования).
                               Если не указаны, метрики считаются в той же шкале, что и y_train/y_val.
    log_base : float, необязательный
               Основание логарифма, если модель обучалась на log_base(y_target).
               При указании вместе с y_train_orig/y_val_orig предсказания переводятся
               в исходную шкалу возведением основания в степень предсказания.
    """

    y_pred_train = model.predict(x_train)
    y_pred_val = model.predict(x_val)

    if log_base is not None and y_train_orig is not None and y_val_orig is not None:
        y_pred_train_orig = np.exp(y_pred_train)
        y_pred_val_orig = np.exp(y_pred_val)

        mae_train = mean_absolute_error(y_train_orig, y_pred_train_orig)
        mae_val   = mean_absolute_error(y_val_orig, y_pred_val_orig)
        rmse_train = root_mean_squared_error(y_train_orig, y_pred_train_orig)
        rmse_val   = root_mean_squared_error(y_val_orig, y_pred_val_orig)

        r2_train = r2_score(y_train_orig, y_pred_train_orig)
        r2_val   = r2_score(y_val_orig, y_pred_val_orig)
    else:

        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_val   = mean_absolute_error(y_val, y_pred_val)
        rmse_train = root_mean_squared_error(y_train, y_pred_train)
        rmse_val   = root_mean_squared_error(y_val, y_pred_val)

        if "Custom" in model_name:
            r2_train = model.rscore(x_train, y_train)
            r2_val   = model.rscore(x_val, y_val)
        else:
            r2_train = r2_score(y_train, y_pred_train)
            r2_val   = r2_score(y_val, y_pred_val)

    return {
        "model": model_name,
        "mae_train": mae_train,
        "mae_val": mae_val,
        "rmse_train": rmse_train,
        "rmse_val": rmse_val,
        "r2_train": r2_train,
        "r2_val": r2_val,
    }
