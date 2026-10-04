"""Загрузка, очистка и разбиение данных."""
import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_URL, PROCESSED_DIR, RANDOM_STATE, RAW_PATH, TARGET, TEST_PATH,
    TEST_SIZE, TRAIN_PATH,
)


def download_raw(force: bool = False) -> None:
    """Скачивает CSV с UCI, если его ещё нет в data/raw."""
    if RAW_PATH.exists() and not force:
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as resp:
        archive = zipfile.ZipFile(io.BytesIO(resp.read()))
    with archive.open(RAW_PATH.name) as src, open(RAW_PATH, "wb") as dst:
        dst.write(src.read())


def load_raw() -> pd.DataFrame:
    """Исходные данные без изменений."""
    download_raw()
    return pd.read_csv(RAW_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Минимальная очистка: единый формат месяцев и булевы колонки в 0/1."""
    df = df.copy()
    # В исходнике все месяцы трёхбуквенные, кроме 'June'
    df["Month"] = df["Month"].replace({"June": "Jun"})
    df["Weekend"] = df["Weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return df


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение train/test, сохраняет доли покупок."""
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )


def save_split(train: pd.DataFrame, test: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # index=True сохраняет номер строки исходного файла, чтобы можно было сверить
    train.to_csv(TRAIN_PATH, index_label="row_id")
    test.to_csv(TEST_PATH, index_label="row_id")


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH, index_col="row_id")


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH, index_col="row_id")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=TARGET), df[TARGET]
