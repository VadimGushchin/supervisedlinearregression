import re
import pandas as pd


def clean_street_address(address: str) -> str:
    """
    Удаляет номера домов из адреса, оставляя название улицы и суффикс (Street, Avenue и т.п.).
    """
    if not isinstance(address, str):
        return "unknown"

    # Удаляем номер дома в начале строки (например, "257 Gold Street" -> "Gold Street")
    s = re.sub(r"^\s*[\d\-]+(?:st|nd|rd|th)?\s+", "", address, flags=re.IGNORECASE)

    # Удаляем номера домов в любом месте (например, "West 125th Street" -> "West Street")
    s = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", "", s, flags=re.IGNORECASE)

    # Убираем лишние пробелы
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else "unknown"


def prepare_street_addresses(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    address_col: str = "street_address",
    new_col: str = "street_clean",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Очищает адреса от номеров домов и добавляет новую колонку в каждый датафрейм.

    Параметры
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Исходные датафреймы с колонкой address_col.
    address_col : str, default='street_address'
        Название колонки с адресами.
    new_col : str, default='street_clean'
        Название новой колонки с очищенным адресом.

    Возвращает
    -------
    train_df, val_df, test_df : pd.DataFrame
        Те же датафреймы с добавленной колонкой new_col.
    """
    for df in (train_df, val_df, test_df):
        df[new_col] = df[address_col].apply(clean_street_address)
        df.drop(columns=["street_address"], inplace=True)
    return train_df, val_df, test_df
