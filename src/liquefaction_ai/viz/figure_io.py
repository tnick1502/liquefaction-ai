"""
Обёртка фигур matplotlib и сохранение в каталог результатов.

Все построители проекта возвращают :class:`MplFig` — лёгкую обёртку над фигурой matplotlib,
которая корректно отображается в ноутбуке (как при ``fig`` последней строкой, так и при
``fig.show()``) и проксирует атрибуты к самой фигуре. :func:`save_figure` сохраняет фигуру в
``results/figs/`` в **высоком разрешении PNG** (а также векторный PDF). Для совместимости на
время перехода поддерживаются и фигуры Plotly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = ["MplFig", "new_figure", "find_repo_root", "resolve_results_dir", "save_figure"]


class MplFig:
    """
    Обёртка над :class:`matplotlib.figure.Figure` с управляемым отображением.

    Отображается в Jupyter и при ``fig`` последней строкой ячейки (через ``_repr_png_``),
    и при явном ``fig.show()``. Прочие атрибуты (``savefig``, ``axes`` …) проксируются к
    обёрнутой фигуре.
    """

    def __init__(self, fig: Figure):
        self._fig = fig
        if fig.canvas is None or not isinstance(fig.canvas, FigureCanvasAgg):
            FigureCanvasAgg(fig)

    @property
    def figure(self) -> Figure:
        """Обёрнутая фигура matplotlib."""
        return self._fig

    def _png_bytes(self, dpi: int = 130) -> bytes:
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()

    def _repr_png_(self) -> bytes:
        """Отображение в Jupyter, когда фигура — последнее выражение ячейки."""
        return self._png_bytes()

    def show(self):
        """Явно отобразить фигуру в ноутбуке (совместимость с ``fig.show()``)."""
        try:
            from IPython.display import Image, display
            display(Image(data=self._png_bytes()))
        except Exception:
            pass
        return None

    # --- совместимость с plotly-идиомами в ноутбуках ---
    _DASH = {"dash": "--", "dot": ":", "dashdot": "-.", "solid": "-", "longdash": (0, (6, 3))}

    def add_hline(self, y, line_dash="dash", line_color="#c46b6b", line_width=1.3, **_kw):
        """Горизонтальная опорная линия на всех осях фигуры (совместимость с Plotly)."""
        for ax in self._fig.axes:
            ax.axhline(y, linestyle=self._DASH.get(line_dash, "--"), color=line_color, linewidth=line_width)
        return self

    def add_vline(self, x, line_dash="dash", line_color="#c46b6b", line_width=1.3, **_kw):
        """Вертикальная опорная линия на всех осях фигуры (совместимость с Plotly)."""
        for ax in self._fig.axes:
            ax.axvline(x, linestyle=self._DASH.get(line_dash, "--"), color=line_color, linewidth=line_width)
        return self

    def update_layout(self, title=None, xaxis_title=None, yaxis_title=None, **_kw):
        """Минимальная совместимость с ``fig.update_layout`` (title/оси на первую ось)."""
        axes = self._fig.axes
        if axes:
            if title is not None:
                axes[0].set_title(title if isinstance(title, str) else title.get("text", ""))
            if xaxis_title is not None:
                axes[0].set_xlabel(xaxis_title)
            if yaxis_title is not None:
                axes[0].set_ylabel(yaxis_title)
        return self

    def __getattr__(self, name):
        return getattr(self._fig, name)


def new_figure(figsize: Tuple[float, float] = (7.2, 4.4)) -> Tuple[MplFig, Figure]:
    """
    Создать новую фигуру matplotlib вне глобального реестра pyplot.

    Использование объектного API (без ``pyplot``) исключает двойное авто-отображение
    inline-бэкендом — фигура показывается только через обёртку :class:`MplFig`.

    :param figsize: размер фигуры в дюймах
    :return: кортеж ``(обёртка MplFig, фигура matplotlib)``
    """
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return MplFig(fig), fig


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
    fig,
    fig_id: str,
    save: bool = False,
    results_dir: Optional[Path] = None,
    width: int = 1100,
    height: int = 600,
    scale: int = 2,
    dpi: int = 300,
    also_pdf: bool = True,
):
    """
    Сохранить фигуру в ``results/figs`` в высоком разрешении PNG (и векторный PDF).

    Поддерживает фигуры matplotlib (обёртка :class:`MplFig` или сырой ``Figure``) и, для
    совместимости на время перехода, фигуры Plotly. Имя файла — ``fig_id`` (рекомендуемый
    формат ``{номер_ноутбука}_{подноутбук}_{название}``).

    :param fig: фигура (``MplFig``/``matplotlib.Figure`` или ``plotly.Figure``)
    :param fig_id: имя файла без расширения
    :param save: выполнять ли сохранение
    :param results_dir: каталог сохранения (по умолчанию ``<корень>/results/figs``)
    :param width: ширина для экспорта Plotly, пикс. (для matplotlib не используется)
    :param height: высота для экспорта Plotly, пикс. (для matplotlib не используется)
    :param scale: масштаб для экспорта Plotly
    :param dpi: разрешение PNG для matplotlib
    :param also_pdf: дополнительно сохранять векторный PDF (matplotlib)
    :return: та же фигура (для цепочки вызовов)
    """
    if not save:
        return fig

    target_dir = resolve_results_dir(results_dir)
    mpl_fig = fig.figure if isinstance(fig, MplFig) else fig

    if isinstance(mpl_fig, Figure):
        mpl_fig.savefig(str(target_dir / f"{fig_id}.png"), dpi=dpi, bbox_inches="tight",
                        facecolor="white")
        if also_pdf:
            try:
                mpl_fig.savefig(str(target_dir / f"{fig_id}.pdf"), bbox_inches="tight",
                                facecolor="white")
            except Exception as exc: # noqa: BLE001 — вектор необязателен
                print(f"[save_figure] PDF для '{fig_id}' не сохранён: {exc}")
        return fig

    # Совместимость: фигура Plotly
    try:
        fig.write_html(str(target_dir / f"{fig_id}.html"), include_plotlyjs="cdn", full_html=True)
        fig.write_image(str(target_dir / f"{fig_id}.png"), width=width, height=height, scale=scale)
    except Exception as exc: # noqa: BLE001
        print(f"[save_figure] '{fig_id}' не сохранён: {exc}")
    return fig
