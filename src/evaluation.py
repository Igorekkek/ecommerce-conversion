"""Кросс-валидация, метрики, пороги, permutation importance."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate,
)

from src.config import N_SPLITS, RANDOM_STATE

SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}


def get_cv(n_repeats: int = 1):
    """Одинаковые фолды для всех экспериментов: честное сравнение моделей."""
    if n_repeats == 1:
        return StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats,
                                   random_state=RANDOM_STATE)


def cv_scores(pipe, X, y, cv, n_jobs: int = -1) -> pd.DataFrame:
    """Метрики на каждом фолде (строка = фолд)."""
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    return pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})


def compare_pipelines(pipes: dict, X, y, cv, baseline: str,
                      metric: str = "pr_auc") -> pd.DataFrame:
    """Сравнивает пайплайны на одних и тех же фолдах.

    Кроме средних считает разницу с базовым вариантом на каждом фолде:
    так видно, улучшение стабильное или это шум разбиения.
    """
    folds = {name: cv_scores(p, X, y, cv) for name, p in pipes.items()}
    base = folds[baseline][metric]
    rows = []
    for name, f in folds.items():
        diff = f[metric] - base
        rows.append({
            "вариант": name,
            "ROC-AUC": f["roc_auc"].mean(),
            "PR-AUC": f["pr_auc"].mean(),
            "PR-AUC std": f["pr_auc"].std(),
            f"Δ {metric} к базе": diff.mean(),
            "фолдов лучше базы": f"{(diff > 0).sum()}/{len(diff)}",
        })
    return pd.DataFrame(rows).set_index("вариант")


def oof_proba(pipe, X, y, cv, n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold вероятности: каждую сессию оценивает модель, которая её не видела."""
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                             n_jobs=n_jobs)[:, 1]


def evaluate_oof(name: str, pipe, X, y, cv) -> tuple[dict, np.ndarray]:
    """Одна прогонка кросс-валидации: OOF-вероятности, метрики по фолдам
    и precision/recall/F1 при пороге, максимизирующем F1."""
    p = oof_proba(pipe, X, y, cv)
    folds = np.array([
        (roc_auc_score(y.iloc[va], p[va]), average_precision_score(y.iloc[va], p[va]))
        for _, va in cv.split(X, y)
    ])
    thr = best_f1_threshold(y, p)
    at_thr = metrics_at_threshold(y, p, thr)
    row = {
        "модель": name,
        "ROC-AUC": folds[:, 0].mean(), "ROC-AUC std": folds[:, 0].std(),
        "PR-AUC": folds[:, 1].mean(), "PR-AUC std": folds[:, 1].std(),
        "порог (max F1)": thr,
        "precision": at_thr["precision"], "recall": at_thr["recall"], "F1": at_thr["F1"],
    }
    return row, p


def metrics_at_threshold(y, p, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "порог": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "доля отмеченных": pred.mean(),
    }


def best_f1_threshold(y, p) -> float:
    """Порог, при котором F1 максимальна."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def threshold_table(y, p, thresholds) -> pd.DataFrame:
    rows = [metrics_at_threshold(y, p, t) for t in thresholds]
    return pd.DataFrame(rows).drop(columns=["ROC-AUC", "PR-AUC"]).set_index("порог")


def cv_permutation_importance(pipe, X: pd.DataFrame, y, cv, scoring="average_precision",
                              n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance на отложенных фолдах train.

    Модель обучается на 4/5 train, а важность считается на оставшейся 1/5:
    насколько падает метрика, если перемешать значения одной исходной колонки.
    Тест для этого не нужен.
    """
    results = []
    for i, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        model = clone(pipe).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        imp = permutation_importance(
            model, X.iloc[va_idx], y.iloc[va_idx], scoring=scoring,
            n_repeats=n_repeats, random_state=RANDOM_STATE + i, n_jobs=-1,
        )
        results.append(pd.Series(imp.importances_mean, index=X.columns))
    df = pd.concat(results, axis=1)
    return (pd.DataFrame({"важность": df.mean(axis=1), "std по фолдам": df.std(axis=1)})
            .sort_values("важность", ascending=False))
