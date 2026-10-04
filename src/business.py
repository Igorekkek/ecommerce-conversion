"""Бизнес-симуляция промокода.

Ожидаемая дополнительная прибыль от показа промокода одной сессии
(по сравнению с ситуацией "промокод никому"):

* сессия и так закончилась бы покупкой (y = 1):
  с вероятностью cannibalization покупатель применит код, и магазин теряет скидку:
  -cannibalization * discount * aov;
* сессия закончилась бы без покупки (y = 0):
  с вероятностью q промокод превращает её в покупку, магазин получает
  маржу минус скидку: q * (margin - discount) * aov.

Вероятность q ("сила" промокода) в данных не наблюдается, это допущение:
q = persuasion * ((1 - warmth) + warmth * p / p_ref),
где p это оценка модели, p_ref средняя оценка среди не купивших.
warmth = 0: промокод одинаково действует на всех ушедших;
warmth = 1: действует пропорционально "теплоте" (оценке модели).
При любом warmth средняя q среди не купивших примерно равна persuasion.
"""
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    aov: float = 3000.0           # средний чек, ₽
    margin: float = 0.30          # маржа, доля от чека
    discount: float = 0.10        # скидка по промокоду, доля от чека
    cannibalization: float = 1.0  # доля "и так покупателей", которые применят код
    persuasion: float = 0.05      # средняя доля ушедших, которых промокод превращает в покупателей
    warmth: float = 0.5           # насколько эффект сосредоточен на "тёплых" (0..1)

    def with_(self, **kwargs) -> "Assumptions":
        return replace(self, **kwargs)

    def as_dict(self) -> dict:
        return asdict(self)


def persuasion_probability(p, a: Assumptions, p_ref: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = a.persuasion * ((1 - a.warmth) + a.warmth * p / p_ref)
    return np.clip(q, 0.0, 1.0)


def session_value(y, p, a: Assumptions, p_ref: float) -> np.ndarray:
    """Ожидаемая дополнительная прибыль (₽), если показать промокод этой сессии."""
    y = np.asarray(y)
    q = persuasion_probability(p, a, p_ref)
    loss = -a.cannibalization * a.discount * a.aov
    gain = q * (a.margin - a.discount) * a.aov
    return np.where(y == 1, loss, gain)


def strategy_summary(name: str, mask, y, p, a: Assumptions, p_ref: float) -> dict:
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y)
    value = session_value(y, p, a, p_ref)
    q = persuasion_probability(p, a, p_ref)
    profit = value[mask].sum()
    return {
        "стратегия": name,
        "получили промокод": int(mask.sum()),
        "доля сессий": mask.mean(),
        "из них купили бы и так": int((mask & (y == 1)).sum()),
        "ожидаемо дожато покупок": float((q * (mask & (y == 0))).sum()),
        "прибыль, ₽": profit,
        "прибыль на 1000 сессий, ₽": profit / len(y) * 1000,
    }


def threshold_curve(y, p, a: Assumptions, p_ref: float, thresholds,
                    direction: str = "above") -> np.ndarray:
    """Прибыль стратегии "промокод, если оценка >= t" (above) или "< t" (below)."""
    value = session_value(y, p, a, p_ref)
    p = np.asarray(p)
    if direction == "above":
        return np.array([value[p >= t].sum() for t in thresholds])
    return np.array([value[p < t].sum() for t in thresholds])


def best_strategies(y, p, is_new, a: Assumptions, p_ref: float, thresholds) -> pd.Series:
    """Прибыль на 1000 сессий для всех стратегий (порог берётся лучший на этих же данных).

    thresholds должны включать 0 и 1: тогда кривые идут от "никому" до "всем",
    и если ни один промежуточный порог не выгоден, лучший вариант честно даёт 0.
    """
    n = len(y)
    value = session_value(y, p, a, p_ref)
    above = threshold_curve(y, p, a, p_ref, thresholds, "above")
    below = threshold_curve(y, p, a, p_ref, thresholds, "below")
    return pd.Series({
        "никому": 0.0,
        "всем новым": value[np.asarray(is_new, dtype=bool)].sum() / n * 1000,
        "всем": value.sum() / n * 1000,
        "модель: оценка ≥ t": above.max() / n * 1000,
        "модель: оценка < t": below.max() / n * 1000,
    })
