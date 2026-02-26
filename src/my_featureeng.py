# ============================================
# КЛАСС ДЛЯ FEATURE ENGINEERING И ОТБОРА ПРИЗНАКОВ
# ============================================

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Полная обработка признаков(доп задание):
    - Добавление базовых признаков (lat_lon, bathrooms/bedrooms, luxury)
    - Полиномиальные признаки 2 степени
    - Отбор признаков по важности (топ-50%)
    - Текстовые признаки из description
    - Взаимодействия текста с топ-признаками
    - Финальный отбор
    """
    
    def __init__(self, top_features_ratio=0.5, n_top_interactions=20, alpha_selector=250):
        self.top_features_ratio = top_features_ratio
        self.n_top_interactions = n_top_interactions
        self.alpha_selector = alpha_selector
        self.poly = None
        self.scaler_poly = None
        self.scaler_text = None
        self.important_features = None
        self.top20_indices = None
        self.final_important = None
        self.text_cols = None
        
    def _add_base_features(self, df):
        """Добавление базовых признаков"""
        df = df.copy()
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df['lat_lon_interaction'] = df['latitude'] * df['longitude']
        if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
            df['rooms'] = df['bathrooms'] + df['bedrooms']
            df['bath_squared'] = df['bathrooms'] ** 2
            df['bed_squared'] = df['bedrooms'] ** 2
        luxury = [c for c in ['Doorman', 'Elevator', 'Dishwasher'] if c in df.columns]
        if luxury:
            df['luxury_score'] = df[luxury].sum(axis=1)
        return df
    
    def _get_text_features(self, df):
        """Извлечение текстовых признаков"""
        df = df.copy()
        df['desc_length'] = df['description'].fillna('').str.len()
        df['desc_word_count'] = df['description'].fillna('').str.split().str.len()
        for word in ['luxury', 'penthouse', 'view', 'renovated', 'modern', 'spacious', 
                     'balcony', 'terrace', 'elevator', 'doorman']:
            df[f'has_{word}'] = df['description'].fillna('').str.contains(word, case=False).astype(int)
        return df
    
    def fit(self, X, y=None, train_mask=None):
        """
        Обучение пайплайна на тренировочных данных
        X - исходный DataFrame с данными
        y - целевая переменная (логарифмированная и очищенная)
        train_mask - маска для отфильтрованных данных
        """
        # Сохраняем исходные данные
        self.df_train = X.copy()
        self.train_mask = train_mask
        
        # Добавляем базовые признаки
        df_with_features = self._add_base_features(self.df_train)
        
        # Убираем лишние колонки
        cols_to_drop = ['features', 
                        'description', 
                        'display_address', 
                        'street_address', 
                        'image_urls', 
                        'Features_cleaned', 
                        'created'
                        ]
        
        cols_to_drop = [c for c in cols_to_drop if c in df_with_features.columns]

        df_numeric = df_with_features.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
        
        # Отделяем признаки от таргета
        self.feature_names = [c for c in df_numeric.columns if c != 'price']
        X_numeric = df_numeric[self.feature_names].values
        X_numeric = X_numeric[train_mask] if train_mask is not None else X_numeric
        
        # Масштабирование для полиномов
        self.scaler_poly = StandardScaler()
        X_scaled = self.scaler_poly.fit_transform(X_numeric)
        
        # Полиномиальные признаки
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = self.poly.fit_transform(X_scaled)
        
        # Отбор признаков по важности
        selector = Ridge(alpha=self.alpha_selector).fit(X_poly, y)
        coefs = np.abs(selector.coef_)
        threshold = np.percentile(coefs, self.top_features_ratio * 100)
        self.important_features = coefs > threshold
        
        # Отобранные полиномиальные признаки
        X_selected = X_poly[:, self.important_features]
        
        # Определяем топ-N признаков для взаимодействий
        selector_top = Ridge(alpha=self.alpha_selector).fit(X_selected, y)
        self.top20_indices = np.argsort(np.abs(selector_top.coef_))[-self.n_top_interactions:]
        
        # Текстовые признаки
        df_text = self._get_text_features(self.df_train)
        self.text_cols = [c for c in df_text.columns if c.startswith(('desc_', 'has_'))]
        X_text = df_text[self.text_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_text = X_text[train_mask] if train_mask is not None else X_text
        
        self.scaler_text = StandardScaler()
        self.scaler_text.fit(X_text)
        
        return self
    
    def transform(self, X, mask=None):
        """
        Преобразование данных с использованием обученных параметров
        """
        # Добавляем базовые признаки
        df = self._add_base_features(X)
        
        # Убираем лишние колонки
        cols_to_drop = ['features', 
                        'description', 
                        'display_address', 
                        'street_address', 
                        'image_urls', 
                        'Features_cleaned', 
                        'created'
                        ]
        
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]
        df_numeric = df.drop(columns=cols_to_drop).select_dtypes(include=[np.number])
        
        # Числовые признаки
        X_numeric = df_numeric[self.feature_names].values
        X_numeric = X_numeric[mask] if mask is not None else X_numeric
        
        # Полиномиальные признаки
        X_scaled = self.scaler_poly.transform(X_numeric)
        X_poly = self.poly.transform(X_scaled)
        X_selected = X_poly[:, self.important_features]
        
        # Топ-признаки для взаимодействий
        X_top20 = X_selected[:, self.top20_indices]
        
        # Текстовые признаки
        df_text = self._get_text_features(X)
        X_text = df_text[self.text_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
        X_text = X_text[mask] if mask is not None else X_text
        X_text_scaled = self.scaler_text.transform(X_text)
        
        # Взаимодействия
        n_text, n_top = X_text_scaled.shape[1], X_top20.shape[1]
        X_interact = np.zeros((X_selected.shape[0], n_text * n_top))
        for i in range(n_text):
            for j in range(n_top):
                X_interact[:, i*n_top + j] = X_text_scaled[:, i] * X_top20[:, j]
        
        # Объединение всех признаков
        X_all = np.hstack([X_selected, X_text_scaled, X_interact])
        
        return X_all
    
    def fit_transform(self, X, y=None, mask=None):
        """fit + transform"""
        self.fit(X, y, mask)
        return self.transform(X, mask)