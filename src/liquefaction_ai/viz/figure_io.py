"""
Сохранение фигур Plotly в каталог результатов.

Все построители графиков проекта принимают флаг ``save``. При ``save=True`` фигура
сохраняется в ``results/figs/`` с именем вида ``{номер_ноутбука}_{подноутбук}_{название}``.
Сохраняются интерактивная версия (``.html``) и, при доступности экспортного движка,
статическая картинка (``.png``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import plotly.graph_objects as go

__all__ = ["find_repo_root", "resolve_results_dir", "save_figure"]


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Найти корень репозитория по наличию ``pyproject.toml`` вверх по дереву каталогов.

    :param start: каталог, от которого начинается поиск (по умолчанию текущий рабочий)
    :return: путь к корню репозитория (или исходный каталог, если корень не найден)
    """
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start


def resolve_results_dir(results_dir: Optional[Path] = None) -> Path:
    """
    Определить и создать каталог для сохранения фигур.

    :param results_dir: явный каталог; при ``None`` используется ``<корень>/results/figs``
    :return: существующий каталог для сохранения фигур
    """
    if results_dir is None:
        results_dir = find_repo_root() / "results" / "figs"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_figure(
    fig: go.Figure,
    fig_id: str,
    save: bool = False,
    results_dir: Optional[Path] = None,
    width: int = 1100,
    height: int = 600,
    scale: int = 2,
) -> go.Figure:
    """
    Сохранить фигуру Plotly в каталог результатов, если включён флаг ``save``.

    Имя файла формируется из ``fig_id`` (рекомендуемый формат
    ``{номер_ноутбука}_{подноутбук}_{название}``, например ``2_1_training_curves``).
    Всегда сохраняется интерактивный ``.html``; статический ``.png`` сохраняется при
    наличии движка экспорта (kaleido) и пропускается с предупреждением при его отсутствии.

    :param fig: фигура Plotly
    :param fig_id: идентификатор/имя файла без расширения
    :param save: выполнять ли сохранение (если False — функция ничего не делает)
    :param results_dir: каталог сохранения (по умолчанию ``<корень>/results/figs``)
    :param width: ширина статической картинки, пикс.
    :param height: высота статической картинки, пикс.
    :param scale: масштаб (множитель разрешения) статической картинки
    :return: та же фигура (для удобной цепочки вызовов)
    """
    if not save:
        return fig

    target_dir = resolve_results_dir(results_dir)
    html_path = target_dir / f"{fig_id}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

    try:
        png_path = target_dir / f"{fig_id}.png"
        fig.write_image(str(png_path), width=width, height=height, scale=scale)
    except Exception as exc:  # noqa: BLE001 — статический экспорт необязателен
        print(f"[save_figure] PNG для '{fig_id}' не сохранён (нет движка экспорта): {exc}")

    return fig
