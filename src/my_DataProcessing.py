import matplotlib.pyplot as plt
import pandas as pd
import my_LinearRegression as linreg
import time
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.base import clone


def remove_price_outliers(df, column="price", lower_q=0.01, upper_q=0.99):
    q_low = df[column].quantile(lower_q)
    q_hi = df[column].quantile(upper_q)
    return df[(df[column] > q_low) & (df[column] < q_hi)].copy()


def clean_features(feature_str):
    try:
        if pd.isna(feature_str) or feature_str is None or feature_str == "":
            return []
    except:
        pass
    if isinstance(feature_str, list):
        return feature_str
    text = str(feature_str)
    if text in ["", "[]", "nan", "None"]:
        return []

    text = text.replace("[", "").replace("]", "")
    text = text.replace("'", "").replace('"', "")

    items = [item.strip() for item in text.split(",") if item.strip()]
    return items


def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """
    Оценивает уже обученную модель на train и val.
    """
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_val = root_mean_squared_error(y_val, y_pred_val)

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

