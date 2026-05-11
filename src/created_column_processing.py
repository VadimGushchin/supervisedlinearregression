import numpy as np
import pandas as pd


def process_created(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    created_col: str = "created",
    drop_original: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Преобразует колонку created в циклические признаки (sin/cos) месяца, дня недели, дня месяца.
    Удаляет константные компоненты (например, год).

    Параметры
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Датафреймы с колонкой created_col.
    created_col : str
        Название колонки с датой/временем.
    drop_original : bool
        Удалить ли исходную колонку created_col после обработки.

    Возвращает
    -------
    train_df, val_df, test_df : pd.DataFrame
        Датафреймы с добавленными колонками:
        - created_month_sin, created_month_cos
        - created_dayofweek_sin, created_dayofweek_cos
        - created_day_sin, created_day_cos
    """

    def _transform(df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        dates = pd.to_datetime(df[created_col], errors="coerce")

        year = dates.dt.year
        month = dates.dt.month
        dayofweek = dates.dt.dayofweek
        day = dates.dt.day

        components = {"year": year, "month": month, "dayofweek": dayofweek, "day": day}

        if is_train:
            constant_cols = [
                name for name, series in components.items() if series.nunique() == 1
            ]
        else:
            constant_cols = []

        cyclic_cols = [("month", 12), ("dayofweek", 7), ("day", 31)]

        for name, max_val in cyclic_cols:
            col_series = components[name]
            if name in constant_cols:
                continue
            angle = 2 * np.pi * (col_series - 1) / max_val
            if name == "dayofweek":
                angle = 2 * np.pi * col_series / max_val
            else:
                angle = 2 * np.pi * (col_series - 1) / max_val
            df[f"{created_col}_{name}_sin"] = np.sin(angle)
            df[f"{created_col}_{name}_cos"] = np.cos(angle)

        if drop_original and created_col in df.columns:
            df.drop(columns=[created_col], inplace=True)

        return df

    train_df = _transform(train_df, is_train=True)
    val_df = _transform(val_df, is_train=False)
    test_df = _transform(test_df, is_train=False)

    return train_df, val_df, test_df
