"""Конверсия с доверительными интервалами и размеры эффекта для тестов."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint


def conversion_table(df: pd.DataFrame, col: str, target: str = "Revenue",
                     alpha: float = 0.05) -> pd.DataFrame:
    """Конверсия по группам с 95% интервалом Уилсона.

    Интервал Уилсона ведёт себя разумно и при маленьких группах,
    и при конверсии около нуля (обычный "±1.96·SE" там врёт).
    """
    g = df.groupby(col, observed=True)[target].agg(sessions="size", purchases="sum")
    g["conversion"] = g["purchases"] / g["sessions"]
    lo, hi = proportion_confint(g["purchases"], g["sessions"], alpha=alpha, method="wilson")
    g["ci_low"], g["ci_high"] = lo, hi
    return g


def cramers_v(table: pd.DataFrame) -> float:
    """V Крамера: сила связи двух категориальных признаков от 0 до 1."""
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k)))


def rank_biserial(x, y) -> tuple[float, float]:
    """Размер эффекта для U-критерия Манна-Уитни.

    Возвращает (ps, r):
    ps: вероятность, что случайное значение из x больше случайного из y
         (ничьи считаются за половину); 0.5 = различий нет;
    r: ранговая бисериальная корреляция, r = 2*ps - 1, от -1 до 1.
    """
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    ps = u / (len(x) * len(y))
    return float(ps), float(2 * ps - 1)


def bootstrap_median_diff(x, y, n_boot: int = 2000, random_state: int = 42):
    """Разница медиан x - y и её 95% бутстрэп-интервал."""
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x), np.asarray(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = (np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))))
    return float(np.median(x) - np.median(y)), np.percentile(diffs, [2.5, 97.5])
