import numpy as np


def great_circle_distance(latitude_1, longitude_1, latitude_2, longitude_2):
    """
    Вычисляет расстояние между двумя точками на сфере по формуле гаверсинуса.

    Параметры
    ----------
    latitude_1, longitude_1 : float или array-like
        Широта и долгота первой точки (в градусах).
    latitude_2, longitude_2 : float или array-like
        Широта и долгота второй точки (в градусах).

    Возвращает
    -------
    distance : float или array-like
        Расстояние в километрах.
    """
    earth_radius_km = 6371
    latitude_1_rad, longitude_1_rad, latitude_2_rad, longitude_2_rad = map(
        np.radians, [latitude_1, longitude_1, latitude_2, longitude_2]
    )
    delta_latitude = latitude_2_rad - latitude_1_rad
    delta_longitude = longitude_2_rad - longitude_1_rad
    a = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(latitude_1_rad)
        * np.cos(latitude_2_rad)
        * np.sin(delta_longitude / 2) ** 2
    )
    central_angle = 2 * np.arcsin(np.sqrt(a))
    return earth_radius_km * central_angle


def add_distances_to_landmarks(
    training_dataframe,
    validation_dataframe,
    testing_dataframe,
    landmarks=None,
    latitude_column="latitude",
    longitude_column="longitude",
):
    """
    Добавляет признаки-расстояния от каждой точки в датафрейме до заданных географических ориентиров.

    Особенности:
    - Расстояния вычисляются по формуле гаверсинуса (в километрах).
    - Исходные колонки широты и долготы не удаляются.
    - Для каждого переданного ориентира создаётся новая колонка `distance_to_<имя_ориентира>`.
    - Если список ориентиров не указан, используются центры районов Нью-Йорка и крупные аэропорты.

    Параметры
    ----------
    training_dataframe, validation_dataframe, testing_dataframe : pandas.DataFrame
        Датафреймы, содержащие колонки с координатами.
    landmarks : list of tuples, optional
        Список кортежей вида (широта, долгота, имя_ориентира), где:
        - широта : float – широта центра в градусах,
        - долгота : float – долгота центра,
        - имя_ориентира : str – суффикс для имени колонки (без пробелов, на английском).
        Если None, используются ориентиры:
        - Манхэттен (40.7128, -74.0060, 'manhattan_center')
        - Бруклин (40.6782, -73.9442, 'brooklyn_center')
        - Куинс (40.7282, -73.7949, 'queens_center')
        - Аэропорт имени Кеннеди (40.6413, -73.7781, 'jfk_airport')
        - Аэропорт Ла-Гуардия (40.7769, -73.8740, 'lga_airport')
    latitude_column, longitude_column : str
        Названия колонок с широтой и долготой в датафреймах (по умолчанию 'latitude' и 'longitude').

    Возвращает
    -------
    training_dataframe, validation_dataframe, testing_dataframe : pandas.DataFrame
        Те же датафреймы, но с добавленными колонками расстояний.
        Пример: 'distance_to_manhattan_center', 'distance_to_jfk_airport' и т.д.
    """
    if landmarks is None:
        landmarks = [
            (40.7128, -74.0060, "manhattan_center"),
            (40.6782, -73.9442, "brooklyn_center"),
            (40.7282, -73.7949, "queens_center"),
            (40.6413, -73.7781, "jfk_airport"),
            (40.7769, -73.8740, "lga_airport"),
        ]

    for dataframe in (training_dataframe, validation_dataframe, testing_dataframe):
        latitudes = dataframe[latitude_column].values
        longitudes = dataframe[longitude_column].values
        for landmark_latitude, landmark_longitude, landmark_name in landmarks:
            dataframe[f"distance_to_{landmark_name}"] = great_circle_distance(
                latitudes, longitudes, landmark_latitude, landmark_longitude
            )

    return training_dataframe, validation_dataframe, testing_dataframe
