"""
Единая тема оформления графиков Plotly.

Регистрирует именованный шаблон ``liquefaction`` (светлый фон, аккуратная сетка,
единый шрифт и hover-режим) и предоставляет согласованные палитры: основную
качественную последовательность цветов и фиксированные цвета для типов грунта,
режимов нагружения и моделей. Это обеспечивает единый «конференционный» стиль во всех
ноутбуках проекта.
"""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go
import plotly.io as pio

from liquefaction_ai.config import DEMO_PALETTE

__all__ = [
    "TEMPLATE_NAME",
    "QUALITATIVE",
    "SEQUENTIAL",
    "register_theme",
    "soil_color_map",
    "load_color_map",
    "model_color_map",
]

TEMPLATE_NAME = "liquefaction"

QUALITATIVE: List[str] = [
    "#0b6efd",  # синий
    "#d63384",  # малиновый
    "#198754",  # зелёный
    "#fd7e14",  # оранжевый
    "#6610f2",  # фиолетовый
    "#0dcaf0",  # голубой
    "#c99a3d",  # песочный
    "#dc3545",  # красный
]
"""Качественная палитра для категориальных рядов."""

SEQUENTIAL = "Viridis"
"""Последовательная цветовая карта для непрерывных величин (риск, PPR и т.п.)."""


def register_theme() -> str:
    """
    Зарегистрировать и активировать единый шаблон оформления Plotly.

    Создаёт шаблон ``liquefaction`` со светлым фоном, мягкой сеткой, единым шрифтом,
    унифицированным hover и качественной палитрой, после чего делает его шаблоном по
    умолчанию для всех последующих фигур.

    :return: имя зарегистрированного шаблона
    """
    layout = go.Layout(
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13, color=DEMO_PALETTE["dark"]),
        title=dict(font=dict(size=18, color=DEMO_PALETTE["dark"]), x=0.5, xanchor="center"),
        paper_bgcolor="white",
        plot_bgcolor="#fbfbfd",
        colorway=QUALITATIVE,
        hovermode="closest",
        hoverlabel=dict(font_size=12, font_family="Inter, Arial, sans-serif"),
        margin=dict(l=70, r=30, t=70, b=60),
        xaxis=dict(showgrid=True, gridcolor="#e9ecef", zeroline=False, linecolor="#ced4da", ticks="outside"),
        yaxis=dict(showgrid=True, gridcolor="#e9ecef", zeroline=False, linecolor="#ced4da", ticks="outside"),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="#e9ecef", borderwidth=1),
        colorscale=dict(sequential=SEQUENTIAL),
    )
    pio.templates[TEMPLATE_NAME] = go.layout.Template(layout=layout)
    pio.templates.default = f"plotly_white+{TEMPLATE_NAME}"
    return TEMPLATE_NAME


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
