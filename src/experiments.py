"""Журнал экспериментов: reports/experiments.csv -> reports/experiments_log.md.

Ноутбуки вызывают log_experiment(...) с числами, посчитанными в коде.
Повторный запуск ноутбука перезаписывает строку с тем же номером, а не дублирует её.
"""
import pandas as pd

from src.config import EXPERIMENTS_CSV, EXPERIMENTS_MD

COLUMNS = ["№", "ноутбук", "идея", "что изменил", "метрика на CV", "вывод"]

HEADER = """# Журнал экспериментов

Все числа получены кодом в ноутбуках (таблица пересобирается при их запуске).
Метрика на CV: среднее по фолдам кросс-валидации **на train**, тест здесь не участвует.
PR-AUC: площадь под кривой precision-recall, главная метрика при редких покупках
(случайная модель даёт примерно долю покупок, около 0.155).

"""


def log_experiment(exp_id: str, notebook: str, idea: str, change: str,
                   cv_metric: str, decision: str) -> None:
    row = dict(zip(COLUMNS, [exp_id, notebook, idea, change, cv_metric, decision]))
    if EXPERIMENTS_CSV.exists():
        log = pd.read_csv(EXPERIMENTS_CSV, dtype=str)
        log = log[log["№"] != exp_id]
    else:
        log = pd.DataFrame(columns=COLUMNS)
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log = log.sort_values("№", key=lambda s: s.str.split(".").map(
        lambda parts: tuple(int(x) for x in parts)))
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(EXPERIMENTS_CSV, index=False)
    _render_md(log)


def _render_md(log: pd.DataFrame) -> None:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for _, r in log.iterrows():
        cells = [str(r[c]).replace("|", "/").replace("\n", " ") for c in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    EXPERIMENTS_MD.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")
