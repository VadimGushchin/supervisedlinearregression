import pandas as pd
import numpy as np
from my_DataProcessing import clean_features
from feature_engineering import canonicalize, map_to_groups, build_feature_groups
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter


def prepare_features(
    df_train_raw: pd.DataFrame,
    df_val_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    min_freq: int = 3,
    corr_threshold: float = 0.9,
    drop_original_cols: list = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн feature engineering.

    Параметры
    ----------
    df_train_raw, df_val_raw, df_test_raw : pd.DataFrame
        Исходные датафреймы. Должны содержать колонки:
        - 'features' (строки-списки) – будет обработана и удалена.
        - 'bathrooms', 'bedrooms' (числовые) – остаются как есть.
        Остальные колонки (latitude, building_id, created и т.д.) сохраняются.
    min_freq : int
        Минимальная частота группы для включения в one‑hot.
    corr_threshold : float
        Порог корреляции для удаления одной из двух бинарных групп.
    drop_original_cols : list, optional
        Дополнительные колонки, которые нужно удалить (например, ['photos', 'listing_id']).

    Возвращает
    -------
    df_train, df_val, df_test : pd.DataFrame
        Датафреймы с сохранёнными исходными колонками (кроме 'features' и указанных в drop_original_cols)
        и добавленными бинарными группами (отфильтрованными).
        Колонка 'price' остаётся нетронутой.
    """
    # Копируем, чтобы не менять оригиналы
    df_train = df_train_raw.copy()
    df_val = df_val_raw.copy()
    df_test = df_test_raw.copy()

    # 1. Очистка и канонизация features
    df_train["Features_cleaned"] = (
        df_train["features"]
        .apply(clean_features)
        .apply(lambda x: [canonicalize(i) for i in x])
    )
    df_val["Features_cleaned"] = (
        df_val["features"]
        .apply(clean_features)
        .apply(lambda x: [canonicalize(i) for i in x])
    )
    df_test["Features_cleaned"] = (
        df_test["features"]
        .apply(clean_features)
        .apply(lambda x: [canonicalize(i) for i in x])
    )

    # Удаляем исходную колонку features
    df_train.drop(columns=["features"], inplace=True)
    df_val.drop(columns=["features"], inplace=True)
    df_test.drop(columns=["features"], inplace=True)

    # 2. Построение групп на основе всех канонических фич из TRAIN
    all_canonical = []
    for lst in df_train["Features_cleaned"]:
        all_canonical.extend(lst)
    groups = build_feature_groups(all_canonical)

    # 3. Маппинг в группы
    df_train["Feature_groups"] = df_train["Features_cleaned"].apply(
        lambda x: map_to_groups(x, groups)
    )
    df_val["Feature_groups"] = df_val["Features_cleaned"].apply(
        lambda x: map_to_groups(x, groups)
    )
    df_test["Feature_groups"] = df_test["Features_cleaned"].apply(
        lambda x: map_to_groups(x, groups)
    )

    # 4. Сбор всех групп из TRAIN и фильтрация по частоте
    all_features = []
    for lst in df_train["Feature_groups"]:
        all_features.extend(lst)
    counter = Counter(all_features)
    common_features = {f for f, cnt in counter.items() if cnt >= min_freq}
    print(f"Уникальных групп без фильтра: {len(set(all_features))}")
    print(f"Оставлено (частота >= {min_freq}): {len(common_features)}")
    unique_features = sorted(common_features)

    # 5. One‑hot кодирование групп
    mlb = MultiLabelBinarizer(classes=unique_features)
    train_bin = mlb.fit_transform(df_train["Feature_groups"])
    val_bin = mlb.transform(df_val["Feature_groups"])
    test_bin = mlb.transform(df_test["Feature_groups"])

    train_bin_df = pd.DataFrame(train_bin, columns=mlb.classes_, index=df_train.index)
    val_bin_df = pd.DataFrame(val_bin, columns=mlb.classes_, index=df_val.index)
    test_bin_df = pd.DataFrame(test_bin, columns=mlb.classes_, index=df_test.index)

    # Объединяем с исходными данными
    df_train = pd.concat([df_train, train_bin_df], axis=1)
    df_val = pd.concat([df_val, val_bin_df], axis=1)
    df_test = pd.concat([df_test, test_bin_df], axis=1)

    # Удаляем вспомогательные колонки
    df_train.drop(columns=["Features_cleaned", "Feature_groups"], inplace=True)
    df_val.drop(columns=["Features_cleaned", "Feature_groups"], inplace=True)
    df_test.drop(columns=["Features_cleaned", "Feature_groups"], inplace=True)

    binary_cols = mlb.classes_
    if len(binary_cols) > 0:
        # Константные
        constant_binary = [col for col in binary_cols if df_train[col].nunique() == 1]
        if constant_binary:
            print(f"Константные бинарные колонки (удаляем): {len(constant_binary)}")
            df_train.drop(columns=constant_binary, inplace=True)
            df_val.drop(columns=constant_binary, inplace=True)
            df_test.drop(columns=constant_binary, inplace=True)
            binary_cols = [col for col in binary_cols if col not in constant_binary]

        # Коррелированные
        if len(binary_cols) > 1:
            df_corr = df_train[binary_cols].corr().abs()
            upper_triangle = df_corr.where(
                np.triu(np.ones(df_corr.shape), k=1).astype(bool)
            )
            to_drop_corr = [
                col
                for col in upper_triangle.columns
                if any(upper_triangle[col] > corr_threshold)
            ]
            if to_drop_corr:
                print(
                    f"Удаляемые из‑за корреляции >{corr_threshold}: {len(to_drop_corr)}"
                )
                df_train.drop(columns=to_drop_corr, inplace=True)
                df_val.drop(columns=to_drop_corr, inplace=True)
                df_test.drop(columns=to_drop_corr, inplace=True)

    if drop_original_cols is not None:
        for col in drop_original_cols:
            for df in (df_train, df_val, df_test):
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
                else:
                    print(
                        f"Предупреждение: колонка {col} отсутствует в одном из датафреймов, пропускаем"
                    )

    print(f"Итоговое количество колонок в train: {df_train.shape[1]}")
    return df_train, df_val, df_test
