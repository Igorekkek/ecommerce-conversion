"""Новые поведенческие признаки и препроцессинг для моделей.

Все новые признаки считаются по одной строке (сессии), поэтому трансформеру
нечему "учиться" на данных и утечки через него быть не может. Он всё равно
живёт внутри Pipeline, чтобы train и test проходили одинаковый путь.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import PAGE_COUNTS, PAGE_DURATIONS

ENGINEERED = [
    "total_pages",          # всего страниц за сессию
    "total_duration",       # общая длительность сессии, сек
    "product_share",        # доля товарных страниц среди всех
    "avg_product_time",     # среднее время на одну товарную страницу, сек
    "avg_page_time",        # среднее время на одну страницу, сек
    "visited_admin",        # заходил ли на служебные страницы (аккаунт, вход и т.п.)
]

# Технические коды без расшифровки (ОС, браузер, регион)
TECH_CODES = ["OperatingSystems", "Browser", "Region"]


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Деление, где 0/0 даёт 0, а не NaN."""
    return (num / den.where(den > 0)).fillna(0.0)


def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_pages"] = df[PAGE_COUNTS].sum(axis=1)
    df["total_duration"] = df[PAGE_DURATIONS].sum(axis=1)
    df["product_share"] = _safe_ratio(df["ProductRelated"], df["total_pages"])
    df["avg_product_time"] = _safe_ratio(df["ProductRelated_Duration"], df["ProductRelated"])
    df["avg_page_time"] = _safe_ratio(df["total_duration"], df["total_pages"])
    df["visited_admin"] = (df["Administrative"] > 0).astype(int)
    return df


class BehaviorFeatures(BaseEstimator, TransformerMixin):
    """Обёртка над add_behavior_features для sklearn Pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_behavior_features(X)


def make_preprocessor(numeric: list[str], categorical: list[str],
                      log: bool = True, scale: bool = True) -> ColumnTransformer:
    """Числовые: log1p (сжимает длинные хвосты) + стандартизация.
    Категориальные: one-hot, редкие значения (<30 сессий) сливаются в одну колонку."""
    num_steps = []
    if log:
        num_steps.append(FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
    if scale:
        num_steps.append(StandardScaler())
    num_pipe = make_pipeline(*num_steps) if num_steps else "passthrough"
    cat_pipe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                             sparse_output=False)
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        verbose_feature_names_out=False,
    )


def make_model_pipeline(model, numeric: list[str], categorical: list[str],
                        log: bool = True, scale: bool = True):
    """Полный пайплайн: новые признаки -> препроцессинг -> модель."""
    return make_pipeline(
        BehaviorFeatures(),
        make_preprocessor(numeric, categorical, log=log, scale=scale),
        model,
    )


# Итоговый набор признаков, выбран в ноутбуке 04:
# без PageValues (утечка), без нерасшифрованных кодов ОС/браузера/региона,
# новые поведенческие признаки не дали прироста и в модель не входят.
MAIN_NUMERIC = [
    "Administrative", "Informational", "ProductRelated",
    "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "SpecialDay",
]
MAIN_CATEGORICAL = ["Month", "VisitorType", "Weekend", "TrafficType"]
