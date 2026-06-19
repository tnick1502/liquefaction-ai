"""
Топологический анализ данных (TDA) для моделей разжижения.

Модуль предоставляет переиспользуемые инструменты для двух направлений:

1. **Топология латентного пространства DPI-Flow** — извлечение латента θ из обученной
   модели и его анализ методами UMAP, Mapper и устойчивой гомологии (persistent homology).
   Показывает, что в латентном пространстве автоматически формируются физические режимы
   поведения грунта (по типам грунта, режимам CSR, механизмам разжижения).

2. **Топологический ранний предвестник разжижения** — построение фазового пространства
   циклического отклика ``X = (PPR, q, ε)`` и расчёт устойчивой гомологии по скользящим
   окнам. Полная персистентность H0 (разрастание траектории) и H1 (богатство гистерезисных
   петель) растут при приближении к разжижению, что даёт ранний индикатор.

Зависимости TDA (``ripser``, ``umap-learn``, ``kmapper``) импортируются лениво; при их
отсутствии функции деградируют корректно (например, UMAP → PCA).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "tda_available",
    "collect_latent",
    "compute_persistence",
    "persistence_summary",
    "umap_embed",
    "mapper_graph",
    "csr_regime",
    "mechanism_label",
    "build_phase_space",
    "sliding_window_persistence",
    "topological_early_warning",
]


def _probe(module: str) -> bool:
    """Тихо проверить, импортируется ли опциональная TDA-библиотека."""
    import importlib.util

    return importlib.util.find_spec(module) is not None


# Доступность опциональных TDA-зависимостей определяется один раз при импорте модуля.
HAS_RIPSER = _probe("ripser")
HAS_KMAPPER = _probe("kmapper")
HAS_UMAP = _probe("umap")
_INSTALL_HINT = "pip install ripser kmapper umap-learn kaleido"


def tda_available() -> Dict[str, bool]:
    """
    Сообщить о наличии опциональных TDA-библиотек (graceful degradation).

    Ноутбуки серии 4 должны проверять этот флаг и корректно пропускать недоступные блоки
    (persistent homology → ripser, граф Mapper → kmapper), а не падать с ImportError.

    :return: словарь ``{"ripser": bool, "kmapper": bool, "umap": bool}``
    """
    return {"ripser": HAS_RIPSER, "kmapper": HAS_KMAPPER, "umap": HAS_UMAP}


# ============================ Латент DPI-Flow ============================

def collect_latent(model, split: Dict[str, object], config, device,
                   batch_size: int = 256, which: str = "theta_raw") -> np.ndarray:
    """
    Извлечь латент θ из обученной модели DPI-Flow по выборке.

    Прогоняет ``model.forward_batch`` по мини-батчам и собирает запрошенный латент
    (``theta_raw`` — физические параметры θ после flow; ``mu`` — среднее энкодера).

    :param model: обученная модель с методами ``forward_batch`` (например, DPIFlow)
    :param split: выборка из ``prepare_benchmark_dataset`` (ключи ``static``/``seq_in``/…)
    :param config: конфигурация эксперимента (для размера батча)
    :param device: устройство инференса
    :param batch_size: размер батча инференса
    :param which: какой латент собрать — ``"theta_raw"`` или ``"mu"``
    :return: массив латентов формы (n, theta_dim)
    """
    import torch
    from liquefaction_ai.data.splits import iterate_minibatches

    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in iterate_minibatches(split, batch_size, device, shuffle=False):
            out = model.forward_batch(batch)
            chunks.append(out[which].detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ============================ Persistent homology ============================

def compute_persistence(points: np.ndarray, maxdim: int = 1,
                        n_subsample: Optional[int] = None, seed: int = 0,
                        metric: str = "euclidean") -> List[np.ndarray]:
    """
    Рассчитать диаграммы устойчивой гомологии (Vietoris–Rips) для облака точек.

    :param points: облако точек, форма (n, d)
    :param maxdim: максимальная размерность гомологий (0 — компоненты, 1 — петли)
    :param n_subsample: при большом облаке — случайно проредить до этого размера
    :param seed: зерно прореживания
    :param metric: метрика расстояния
    :return: список диаграмм ``[H0, H1, …]``; каждая — массив (k, 2) пар (рождение, смерть)
    :raises ImportError: если не установлен ``ripser`` (см. :func:`tda_available`)
    """
    if not HAS_RIPSER:
        raise ImportError(
            "Устойчивая гомология требует пакет 'ripser', который не установлен. "
            f"Установите TDA-зависимости: {_INSTALL_HINT}. "
            "Либо проверьте tda_available() и пропустите этот блок.")
    from ripser import ripser

    X = np.asarray(points, dtype=np.float64)
    X = X[np.isfinite(X).all(axis=1)]
    if n_subsample is not None and len(X) > n_subsample:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), n_subsample, replace=False)]
    dgms = ripser(X, maxdim=maxdim, metric=metric)["dgms"]
    return dgms


def persistence_summary(dgms: List[np.ndarray]) -> Dict[str, float]:
    """
    Свести диаграммы устойчивости к скалярным дескрипторам.

    Для каждой размерности возвращает число особенностей (число Бетти), суммарную и
    максимальную персистентность (время жизни ``death − birth``) и персистентную энтропию
    (нормированную меру разнообразия времён жизни).

    :param dgms: список диаграмм ``[H0, H1, …]``
    :return: словарь скаляров с ключами ``betti_k``, ``total_pers_k``, ``max_pers_k``,
             ``entropy_k`` для каждой размерности ``k``
    """
    out: Dict[str, float] = {}
    for k, dgm in enumerate(dgms):
        if dgm is None or len(dgm) == 0:
            out[f"betti_{k}"] = 0.0
            out[f"total_pers_{k}"] = 0.0
            out[f"max_pers_{k}"] = 0.0
            out[f"entropy_{k}"] = 0.0
            continue
        life = dgm[:, 1] - dgm[:, 0]
        life = life[np.isfinite(life)]            # убрать бесконечную особенность H0
        life = life[life > 0]
        total = float(life.sum())
        out[f"betti_{k}"] = float(len(life))
        out[f"total_pers_{k}"] = total
        out[f"max_pers_{k}"] = float(life.max()) if len(life) else 0.0
        if total > 0 and len(life) > 1:
            p = life / total
            out[f"entropy_{k}"] = float(-(p * np.log(p + 1e-12)).sum())
        else:
            out[f"entropy_{k}"] = 0.0
    return out


# ============================ UMAP / Mapper ============================

def umap_embed(X: np.ndarray, n_neighbors: int = 20, min_dist: float = 0.1,
               seed: int = 42, standardize: bool = True) -> Tuple[np.ndarray, str]:
    """
    Спроецировать признаки в 2D через UMAP (с откатом на PCA при отсутствии библиотеки).

    :param X: матрица признаков, форма (n, d)
    :param n_neighbors: число соседей UMAP
    :param min_dist: минимальная дистанция UMAP
    :param seed: зерно
    :param standardize: стандартизовать ли признаки перед проекцией
    :return: кортеж ``(coords (n, 2), method)``, где method — ``"UMAP"`` или ``"PCA"``
    """
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X) if standardize else np.asarray(X, dtype=float)
    try:
        import umap

        coords = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                           random_state=seed).fit_transform(Z)
        return np.asarray(coords), "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(Z), "PCA"


def mapper_graph(X: np.ndarray, n_cubes: int = 12, perc_overlap: float = 0.45,
                 eps: float = 0.8, min_samples: int = 3, seed: int = 42) -> Dict[str, object]:
    """
    Построить граф Mapper по облаку признаков (KeplerMapper).

    Линза — двумерная UMAP/PCA-проекция; покрытие интервалами, кластеризация DBSCAN внутри
    прообразов. Возвращает узлы (множества индексов точек) и рёбра (общие точки), а также
    2D-координаты узлов (среднее проекции их точек) для отрисовки.

    :param X: матрица признаков, форма (n, d)
    :param n_cubes: число интервалов покрытия по каждой оси линзы
    :param perc_overlap: доля перекрытия интервалов
    :param eps: радиус DBSCAN
    :param min_samples: минимум точек в кластере DBSCAN
    :param seed: зерно проекции
    :return: словарь ``{"nodes", "edges", "node_xy", "node_members", "lens", "method"}``
    :raises ImportError: если не установлен ``kmapper`` (см. :func:`tda_available`)
    """
    if not HAS_KMAPPER:
        raise ImportError(
            "Граф Mapper требует пакет 'kmapper', который не установлен. "
            f"Установите TDA-зависимости: {_INSTALL_HINT}. "
            "Либо проверьте tda_available() и пропустите этот блок.")
    import kmapper as km
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    lens, method = umap_embed(X, seed=seed)
    mapper = km.KeplerMapper(verbose=0)
    graph = mapper.map(lens, Z,
                       cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
                       clusterer=DBSCAN(eps=eps, min_samples=min_samples))
    node_members = {nid: list(members) for nid, members in graph["nodes"].items()}
    node_xy = {nid: lens[members].mean(axis=0) for nid, members in node_members.items()}
    edges = [(a, b) for a, bs in graph["links"].items() for b in bs]
    return {"nodes": list(node_members.keys()), "edges": edges, "node_xy": node_xy,
            "node_members": node_members, "lens": lens, "method": method}


# ============================ Метки режимов ============================

def csr_regime(csr_base: np.ndarray, thresholds: Tuple[float, float] = (0.15, 0.25)) -> np.ndarray:
    """
    Разбить циклическое напряжение CSR на качественные режимы.

    :param csr_base: массив базового CSR
    :param thresholds: границы (низкий|умеренный|высокий)
    :return: массив строковых меток режима CSR
    """
    csr = np.asarray(csr_base, dtype=float)
    lo, hi = thresholds
    out = np.where(csr < lo, "low CSR", np.where(csr < hi, "moderate CSR", "high CSR"))
    return out.astype(object)


def mechanism_label(liq_label: np.ndarray, n_liq: np.ndarray,
                    fast_cycles: float = 15.0) -> np.ndarray:
    """
    Грубая классификация механизма разжижения по метке и числу циклов до разрушения.

    :param liq_label: бинарная метка разжижения
    :param n_liq: число циклов до разжижения
    :param fast_cycles: порог «быстрого» разжижения (циклы)
    :return: массив меток механизма (``flow (fast)`` / ``cyclic (gradual)`` / ``stable``)
    """
    liq = np.asarray(liq_label).astype(bool)
    n = np.asarray(n_liq, dtype=float)
    out = np.where(~liq, "stable",
                   np.where(n <= fast_cycles, "flow (fast)", "cyclic (gradual)"))
    return out.astype(object)


# ============================ Ранний предвестник ============================

def build_phase_space(ppr: np.ndarray, q: np.ndarray, eps: np.ndarray,
                      baseline_frac: float = 0.25) -> np.ndarray:
    """
    Построить безразмерное фазовое пространство ``X = (PPR, q, ε)``.

    PPR берётся как есть (0…1). Девиатор ``q`` и деформация ``ε`` масштабируются на свою
    амплитуду в **базовой** (начальной) части опыта — так последующий рост амплитуды
    (предвестник разжижения) сохраняется как растяжение облака, а не «съедается» нормировкой.

    :param ppr: поровое давление по точкам
    :param q: девиатор по точкам
    :param eps: осевая деформация по точкам
    :param baseline_frac: доля начала опыта для оценки базовой амплитуды
    :return: фазовое облако формы (m, 3)
    """
    ppr = np.asarray(ppr, float); q = np.asarray(q, float); eps = np.asarray(eps, float)
    m = np.isfinite(ppr) & np.isfinite(q) & np.isfinite(eps)
    ppr, q, eps = ppr[m], q[m], eps[m]
    n0 = max(int(len(ppr) * baseline_frac), 10)
    q_scale = np.ptp(q[:n0]) + 1e-9
    eps_scale = np.ptp(eps[:n0]) + 1e-12
    return np.column_stack([ppr, q / q_scale, (eps - eps[:n0].mean()) / eps_scale])


def sliding_window_persistence(cycles: np.ndarray, phase: np.ndarray,
                               points_per_cycle: int, window_cycles: int = 4,
                               step_cycles: int = 1, maxdim: int = 1) -> Dict[str, np.ndarray]:
    """
    Рассчитать устойчивую гомологию по скользящим окнам вдоль числа циклов.

    Для каждого окна (несколько циклов) считается PH фазового облака и сводки H0/H1.

    :param cycles: номер цикла по точкам
    :param phase: фазовое облако (m, d) из :func:`build_phase_space`
    :param points_per_cycle: число точек на цикл
    :param window_cycles: ширина окна в циклах
    :param step_cycles: шаг окна в циклах
    :param maxdim: максимальная размерность гомологий
    :return: словарь массивов ``cycle``, ``H0_total``, ``H1_total``, ``H1_max``, ``H0_max``
    """
    win = max(points_per_cycle * window_cycles, 20)
    step = max(points_per_cycle * step_cycles, 1)
    cyc, H0t, H1t, H1m, H0m = [], [], [], [], []
    for s in range(0, len(phase) - win, step):
        dgms = compute_persistence(phase[s:s + win], maxdim=maxdim)
        summ = persistence_summary(dgms)
        idx = min(s + win // 2, len(cycles) - 1)
        cyc.append(float(cycles[idx]))
        H0t.append(summ["total_pers_0"]); H0m.append(summ["max_pers_0"])
        H1t.append(summ.get("total_pers_1", 0.0)); H1m.append(summ.get("max_pers_1", 0.0))
    return {"cycle": np.array(cyc), "H0_total": np.array(H0t), "H0_max": np.array(H0m),
            "H1_total": np.array(H1t), "H1_max": np.array(H1m)}


def topological_early_warning(cycles: np.ndarray, ppr: np.ndarray, q: np.ndarray,
                              eps: np.ndarray, window_cycles: int = 4,
                              step_cycles: int = 1) -> Dict[str, np.ndarray]:
    """
    Топологический ранний индикатор разжижения (TEWI) по одному опыту.

    Строит фазовое пространство ``(PPR, q, ε)`` и считает по скользящим окнам полную
    персистентность H0 (разрастание траектории ≈ рост амплитуды деформации) и H1 (богатство
    гистерезисных петель). Индикатор ``TEWI = H0_total + H1_total`` растёт при приближении к
    разжижению.

    :param cycles: номер цикла по точкам
    :param ppr: поровое давление по точкам
    :param q: девиатор по точкам
    :param eps: осевая деформация по точкам
    :param window_cycles: ширина окна в циклах
    :param step_cycles: шаг окна в циклах
    :return: словарь массивов ``cycle``, ``H0_total``, ``H1_total``, ``tewi`` (вдоль опыта)
    """
    cycles = np.asarray(cycles, float)
    n_total = max(np.nanmax(cycles), 1.0)
    ppc = max(int(round(len(cycles) / n_total)), 4)
    phase = build_phase_space(ppr, q, eps)
    pw = sliding_window_persistence(cycles, phase, ppc, window_cycles, step_cycles)
    pw["tewi"] = pw["H0_total"] + pw["H1_total"]
    return pw
