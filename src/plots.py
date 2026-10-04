"""Единый стиль графиков и сохранение в reports/figures."""
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

# Палитра: категориальные цвета идут в фиксированном порядке
BLUE = "#2a78d6"      # слот 1: "не купили" / основная серия
ORANGE = "#eb6834"    # слот 2: "купили" / вторая серия
AQUA = "#1baf7a"      # слот 3
YELLOW = "#eda100"    # слот 4
GRAY = "#898781"      # вспомогательные линии, ориентиры
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

CLASS_COLORS = {0: BLUE, 1: ORANGE}
CLASS_LABELS = {0: "не купили", 1: "купили"}
SEQ_CMAP = "Blues"
DIV_CMAP = "RdBu_r"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
    })


def save_fig(fig, name: str) -> None:
    """Сохраняет график в reports/figures/<name>.png."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
