"""Общие настройки проекта: пути, random_state, списки колонок."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "online_shoppers_intention.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
FIGURES_DIR = ROOT / "reports" / "figures"
EXPERIMENTS_CSV = ROOT / "reports" / "experiments.csv"
EXPERIMENTS_MD = ROOT / "reports" / "experiments_log.md"
MODELS_DIR = ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
# Предсказания, которые сохраняет ноутбук 05 для 06 и 07
OOF_PATH = PROCESSED_DIR / "oof_train.csv"
TEST_PRED_PATH = PROCESSED_DIR / "test_predictions.csv"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

TARGET = "Revenue"

# Порядок месяцев для графиков (в данных нет января и апреля)
MONTH_ORDER = ["Feb", "Mar", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_RU = {
    "Feb": "фев", "Mar": "мар", "May": "май", "Jun": "июн", "Jul": "июл",
    "Aug": "авг", "Sep": "сен", "Oct": "окт", "Nov": "ноя", "Dec": "дек",
}
VISITOR_RU = {
    "New_Visitor": "новый", "Returning_Visitor": "вернувшийся", "Other": "другое",
}

# Исходные признаки
PAGE_COUNTS = ["Administrative", "Informational", "ProductRelated"]
PAGE_DURATIONS = [
    "Administrative_Duration", "Informational_Duration", "ProductRelated_Duration",
]
GA_RATES = ["BounceRates", "ExitRates", "PageValues"]
NUMERIC_RAW = PAGE_COUNTS + PAGE_DURATIONS + GA_RATES + ["SpecialDay"]
FEATURE_RU = {
    "Administrative": "Служебные страницы, шт",
    "Administrative_Duration": "Время на служебных, сек",
    "Informational": "Информационные страницы, шт",
    "Informational_Duration": "Время на информационных, сек",
    "ProductRelated": "Товарные страницы, шт",
    "ProductRelated_Duration": "Время на товарных, сек",
    "BounceRates": "Показатель отказов",
    "ExitRates": "Показатель выхода",
    "PageValues": "Ценность страниц (PageValues)",
    "SpecialDay": "Близость праздника",
}

CATEGORICAL_RAW = [
    "Month", "VisitorType", "Weekend",
    "OperatingSystems", "Browser", "Region", "TrafficType",
]
