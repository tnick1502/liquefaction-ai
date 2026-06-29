"""
Единая тема оформления графиков (matplotlib, журнальный стиль Q1).

Задаёт мягкую пастельную палитру, чистый sans-serif, тонкие оси (без верхней и правой
рамок), светлую сетку и высокое разрешение экспорта — единый «публикационный» стиль для
всех ноутбуков. Также предоставляет согласованные цветовые карты для типов грунта, режимов
нагружения и моделей и пастельные непрерывную/расходящуюся colormap.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "TEMPLATE_NAME",
    "QUALITATIVE",
    "SEQUENTIAL",
    "SEQUENTIAL_NAME",
    "DIVERGING",
    "DIVERGING_NAME",
    "INK",
    "GRID",
    "register_theme",
    "plain_log_axis",
    "soil_color_map",
    "load_color_map",
    "model_color_map",
]

TEMPLATE_NAME = "liquefaction"

# Основной тёмный «чернильный» цвет текста/осей и цвет сетки
INK = "#2b2f36"
GRID = "#dfe3e8"

# Мягкая пастельная качественная палитра (десатурированная, журнальная)
QUALITATIVE: List[str] = [
    "#5B8FB0",  # пыльно-синий
    "#E39C9C",  # пастельно-коралловый
    "#7FB286",  # шалфейно-зелёный
    "#E1B877",  # песочно-золотой
    "#9D8EC1",  # пастельно-фиолетовый
    "#5FB0AB",  # пастельно-бирюзовый
    "#CB8FB3",  # пастельно-розовый
    "#8C9BA8",  # серо-голубой сланец
]
"""Качественная палитра для категориальных рядов (пастельная, Q1)."""

# Пастельная последовательная colormap: светлый → пыльно-сине-зелёный
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "liq_seq", ["#f3f6f5", "#bfd8d2", "#7fb0b8", "#4f8aa8", "#3c5f86"])
SEQUENTIAL_NAME = "liq_seq"

# Пастельная расходящаяся colormap для корреляций: коралл → светлый → синий
DIVERGING = LinearSegmentedColormap.from_list(
    "liq_div", ["#c46b6b", "#e8a598", "#f4ece4", "#9fc0cf", "#4f7f9e"])
DIVERGING_NAME = "liq_div"


def _register_cmaps() -> None:
    """Зарегистрировать пастельные colormap в matplotlib (идемпотентно)."""
    for name, cmap in [(SEQUENTIAL_NAME, SEQUENTIAL), (DIVERGING_NAME, DIVERGING)]:
        try:
            mpl.colormaps.register(cmap, name=name, force=True)
        except (AttributeError, ValueError):
            try:
                mpl.cm.register_cmap(name=name, cmap=cmap)
            except Exception:
                pass


def register_theme() -> str:
    """
    Активировать единый журнальный (Q1) стиль matplotlib.

    Настраивает rcParams: пастельный цикл цветов, чистый sans-serif, тонкие оси без верхней
    и правой рамок, мягкую сетку, белый фон и высокое разрешение сохранения. Регистрирует
    пастельные colormap. Возвращает имя темы.

    :return: имя зарегистрированной темы
    """
    _register_cmaps()
    plt.rcParams.update({
        # Шрифт
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.titlesize": 15,
        "figure.titleweight": "semibold",
        # Цвета текста/осей
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#9aa3ad",
        "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
        # Оси/рамки
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.direction": "out", "ytick.direction": "out",
        # Сетка
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 0.9,
        "axes.axisbelow": True,
        # Фон
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        # Легенда
        "legend.frameon": True, "legend.framealpha": 0.85, "legend.edgecolor": GRID,
        "legend.borderpad": 0.5, "legend.handlelength": 1.6,
        # Линии/маркеры
        "lines.linewidth": 2.0, "lines.markersize": 5.5, "lines.solid_capstyle": "round",
        "patch.linewidth": 0.6,
        # Экспорт высокого разрешения
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        # Цикл пастельных цветов
        "axes.prop_cycle": mpl.cycler(color=QUALITATIVE),
        "image.cmap": SEQUENTIAL_NAME,
    })
    return TEMPLATE_NAME


def plain_log_axis(ax, axis: str = "x") -> None:
    """
    Лог-шкала с обычными (не mathtext) подписями делений: 1, 10, 100, 1000, …

    matplotlib по умолчанию рисует деления лог-оси как mathtext (``$10^{n}$``), для чего нужен
    математический шрифт DejaVu Sans. Если кэш шрифтов matplotlib повреждён, рендер падает с
    ``ValueError: Failed to find font DejaVu Sans``. Обычный формат делений mathtext не использует
    и потому устойчив к состоянию кэша шрифтов (а для числа циклов N подписи 10/100/1000 даже
    читабельнее, чем 10^n).

    :param ax: оси matplotlib
    :param axis: ``"x"`` или ``"y"`` — какую ось форматировать
    """
    from matplotlib.ticker import FuncFormatter, NullFormatter
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def soil_color_map(soil_names: List[str]) -> Dict[str, str]:
    """
    Построить фиксированное соответствие «класс грунта → цвет».

    :param soil_names: упорядоченный список идентификаторов классов грунта
    :return: словарь соответствий цветов из качественной палитры
    """
    return {name: QUALITATIVE[i % len(QUALITATIVE)] for i, name in enumerate(soil_names)}


def load_color_map(load_names: List[str]) -> Dict[str, str]:
    """
    Построить фиксированное соответствие «режим нагружения → цвет».

    :param load_names: упорядоченный список идентификаторов режимов нагружения
    :return: словарь соответствий цветов из качественной палитры
    """
    return {name: QUALITATIVE[i % len(QUALITATIVE)] for i, name in enumerate(load_names)}


def model_color_map(model_names: List[str]) -> Dict[str, str]:
    """
    Построить фиксированное соответствие «модель → цвет».

    :param model_names: список имён моделей
    :return: словарь соответствий цветов из качественной палитры
    """
    return {name: QUALITATIVE[i % len(QUALITATIVE)] for i, name in enumerate(model_names)}
