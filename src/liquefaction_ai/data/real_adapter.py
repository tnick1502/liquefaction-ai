"""
Адаптер реальных лабораторных данных к формату артефакта проекта.

Позволяет подставить массивы реальных циклических испытаний вместо синтетических, не меняя
ноутбуки анализа, обучения и оценки. Принимает только **наблюдаемые** в опыте величины:
свойства грунта, историю нагружения (число циклов и CSR), измеренную траекторию порового
давления PPR(N), длину валидного участка, бинарную метку разжижения и число циклов до
разжижения N_liq. Синтетические «латентные» поля (скрытое повреждение z, триггер g, истинная
CRR, непрерывный риск-скор) не используются и не требуются.

Признаки, строящие CRR (α, β), вычисляются из физических свойств грунта физической моделью
:func:`liquefaction_ai.physics.crr_physical.compute_crr_components` — это входные признаки, а
не супервизия, поэтому доступны и на реальных данных.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.constants import LOAD_NAMES, SOIL_NAMES
from liquefaction_ai.data.observed import derive_observed_targets
from liquefaction_ai.data.splits import make_benchmark_splits
from liquefaction_ai.data.synthetic import build_feature_matrices
from liquefaction_ai.physics.crr_physical import RESPONSE_TYPES, compute_crr_components

__all__ = ["compute_crr_features", "enrich_crr_breakdown", "build_observed_prefix",
           "build_population_from_experiments", "ensure_analysis_columns"]


def _col(df: pd.DataFrame, name: str, default) -> np.ndarray:
    """Вернуть колонку как массив или заполнить значением по умолчанию при отсутствии."""
    if name in df.columns:
        return df[name].to_numpy()
    return np.full(len(df), default, dtype=float)


# Характерные значения по типу грунта ГОСТ (1…9): D50 (мм) — для образцов без грансостава;
# e_min/e_max — если рыхлое/плотное состояние не измерено в опыте
_TYPE_D50 = {1: 3.0, 2: 0.7, 3: 0.4, 4: 0.15, 5: 0.08, 6: 0.03, 7: 0.012, 8: 0.002, 9: 0.05}
_TYPE_EMIN = {1: 0.40, 2: 0.45, 3: 0.50, 4: 0.55, 5: 0.55, 6: 0.50, 7: 0.55, 8: 0.60, 9: 1.50}
_TYPE_EMAX = {1: 0.75, 2: 0.80, 3: 0.85, 4: 0.95, 5: 1.05, 6: 0.95, 7: 1.10, 8: 1.35, 9: 3.00}
_GRAN_COLS = ["gran_10", "gran_5", "gran_2", "gran_1", "gran_05", "gran_025",
              "gran_01", "gran_005", "gran_001", "gran_0002", "gran_0000"]


def ensure_analysis_columns(soil_df: pd.DataFrame, load_df: pd.DataFrame,
                            crr_obs_mask: Optional[np.ndarray] = None) -> None:
    """
    Дополнить таблицы свойств/нагрузки колонками, совместимыми с синтетическим артефактом.

    Делает мету реальных данных column-complete для ноутбуков анализа/оценки. Производные
    гранулометрические параметры ``D10/D50/D60`` и ``plaxis_class`` вычисляются **по
    алгоритму digitrock** (:func:`liquefaction_ai.data.grainsize.plaxis_classification`:
    лог-интерполяция кумулятивной кривой грансостава, бины PLAXIS по D50). Остальные
    производные (``e_min/e_max``, ``n_porosity``, ``soil_name_en`` …) — из имеющихся свойств
    или характерных значений по типу грунта ГОСТ. Латентные «истины» (``risk_score_true`` и
    т.п.) на реальных данных **не добавляются** — ноутбуки анализа сами подставляют
    наблюдаемый прокси (пик PPR). Изменяет ``soil_df`` и ``load_df`` на месте.

    :param soil_df: таблица свойств грунта (изменяется на месте)
    :param load_df: таблица параметров нагружения (изменяется на месте)
    :param crr_obs_mask: маска наличия измеренной кривой CRR (для ``has_measured_crr``)
    :return: None
    """
    from liquefaction_ai.data.grainsize import FRACTION_KEYS, plaxis_classification

    n = len(soil_df)
    cls = soil_df["class_id"].astype(int).to_numpy() if "class_id" in soil_df.columns else np.zeros(n, int)
    tg = np.clip(cls + 1, 1, 9)

    def setdefault(df: pd.DataFrame, col: str, values) -> None:
        if col not in df.columns:
            df[col] = values

    setdefault(soil_df, "soil_name_en", [SOIL_NAMES[int(i)] for i in cls])

    # Грансостав → D10/D50/D60/Cu/plaxis_class по алгоритму digitrock там, где грансостав
    # задан; где не задан (типично для глин/суглинков) — D50 по типу ГОСТ, далее те же бины
    gran_cols = [f"gran_{k}" for k in FRACTION_KEYS]
    for col in gran_cols:
        setdefault(soil_df, col, np.zeros(n))
    fractions = soil_df[gran_cols].to_numpy(dtype=float)
    pc = plaxis_classification(fractions)
    has_gran = fractions.sum(axis=1) > 1.0 # есть измеренный грансостав
    type_d50 = np.array([_TYPE_D50.get(int(t), 0.01) for t in tg])
    d50 = np.where(has_gran, pc["D50"].astype(float), type_d50)
    d10 = np.where(has_gran, pc["D10"].astype(float), type_d50 / 5.0)
    d60 = np.where(has_gran, pc["D60"].astype(float), type_d50 * 1.5)
    plaxis = np.select([d50 > 10.0, d50 > 2.0, d50 > 0.25, d50 > 0.075],
                       ["very coarse", "coarse", "medium", "fine"], default="very fine")
    setdefault(soil_df, "D10", d10)
    setdefault(soil_df, "D50", d50)
    setdefault(soil_df, "D60", d60)
    setdefault(soil_df, "plaxis_class", plaxis.astype(object))
    cu_prev = _col(soil_df, "Cu", 5.0)
    soil_df["Cu"] = np.where(has_gran, pc["Cu"].astype(float), cu_prev)

    setdefault(soil_df, "e_min", np.array([_TYPE_EMIN.get(int(t), 0.5) for t in tg]))
    setdefault(soil_df, "e_max", np.array([_TYPE_EMAX.get(int(t), 1.0) for t in tg]))
    e_arr = _col(soil_df, "e", 0.7)
    setdefault(soil_df, "n_porosity", e_arr / (1.0 + np.maximum(e_arr, 1e-3)) * 100.0)
    setdefault(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0)
    setdefault(soil_df, "saturation", np.ones(n))
    setdefault(soil_df, "B_value", np.full(n, 0.95))
    setdefault(soil_df, "aging_years", np.ones(n))
    setdefault(soil_df, "cementation_index", np.zeros(n))

    # Колонки, которых может не быть на реальных данных (физика/химия) — NaN
    for col in ["rs", "rd", "r", "W", "Wl", "Wp", "ground_water_depth",
                "cohesion", "phi", "E_modulus", "calcite", "dolomite", "insoluble_residue"]:
        setdefault(soil_df, col, np.full(n, np.nan))

    # Флаг наличия измеренной кривой CRR (серия потенциала разжижения)
    has_crr = np.zeros(n, bool) if crr_obs_mask is None else (np.asarray(crr_obs_mask) > 0)
    setdefault(soil_df, "has_measured_crr", has_crr)

    # Параметры формы нагрузки (для реальных опытов — стационарные)
    setdefault(load_df, "phase", np.zeros(n))
    setdefault(load_df, "burst_1", np.zeros(n))
    setdefault(load_df, "burst_2", np.zeros(n))


def compute_crr_features(soil_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Вычислить параметры кривой CRR (α, β) из физических свойств грунта.

    Использует физическую модель CRR на референсном режиме нагружения. Недостающие
    свойства заполняются разумными значениями по умолчанию.

    :param soil_df: таблица свойств грунта (Dr, Ip, fines, clay, σ′, OCR, K0, Vs1 и т.д.)
    :return: словарь с массивами ``crr_alpha``, ``crr_betta``, ``crr_ref``
    """
    n = len(soil_df)
    crr = compute_crr_components(
        Dr=_col(soil_df, "D_r", 0.5),
        Ip=_col(soil_df, "I_p", 0.0),
        Il=_col(soil_df, "Il", 0.0),
        Ir=_col(soil_df, "Ir", 0.0),
        fines_content=_col(soil_df, "fines_content", 10.0),
        clay_fraction=_col(soil_df, "clay_fraction", 2.0),
        sigma_eff=_col(soil_df, "sigma_eff", 100.0),
        OCR=_col(soil_df, "OCR", 1.0),
        K0=_col(soil_df, "K0", 0.5),
        Vs1=_col(soil_df, "Vs1", _col(soil_df, "V_s", 180.0)),
        static_shear_ratio=_col(soil_df, "static_shear_ratio", 0.0),
        cementation_index=_col(soil_df, "cementation_index", 0.0),
        aging_years=_col(soil_df, "aging_years", 1.0),
        saturation=_col(soil_df, "saturation", 1.0),
        B=_col(soil_df, "B_value", 0.95),
        damping_percent=_col(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0),
        frequency=np.ones(n),
        loading_mode_factor=np.ones(n),
    )
    return {"crr_alpha": crr["alpha"].astype(np.float32),
            "crr_betta": crr["betta"].astype(np.float32),
            "crr_ref": crr["crr_ref"].astype(np.float32)}


def enrich_crr_breakdown(soil_df: pd.DataFrame, load_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Добавить к таблице свойств полное разложение CRR из физической модели.

    Дописывает колонки ``crr_alpha``, ``crr_betta``, ``crr_ref``, ``crr_cycle_slope``,
    ``crr_state_proxy``, ``response_type`` и факторы ``crr_f_*``. Это делает ноутбук анализа
    параметров CRR (серия 1.3) работоспособным и на реальных данных.

    Частота нагружения берётся из ``load_df["frequency"]`` (реально измерена и варьируется
    0.1…5 Гц) и подаётся в digitrock-формулу частотного фактора — иначе частота, входящая в
    модель как признак, была бы проигнорирована в самой физике CRR. Множитель режима нагружения
    оставлен референсным (=1): digitrock не задаёт статического фактора CRR по типу режима, а сам
    режим (storm/seismic) уже входит в модель как категориальный признак и через динамическую
    голову CRR. При отсутствии ``load_df`` используется референсная частота 1 Гц (синтетический
    путь, где частота применяется в порождающем ODE).

    :param soil_df: таблица свойств грунта
    :param load_df: опциональная таблица нагружения (для реальной частоты ``frequency``)
    :return: та же таблица с добавленными CRR-колонками
    """
    n = len(soil_df)
    if load_df is not None and "frequency" in load_df.columns:
        frequency = np.asarray(load_df["frequency"], dtype=np.float64)
    else:
        frequency = np.ones(n)
    crr = compute_crr_components(
        Dr=_col(soil_df, "D_r", 0.5), Ip=_col(soil_df, "I_p", 0.0), Il=_col(soil_df, "Il", 0.0),
        Ir=_col(soil_df, "Ir", 0.0), fines_content=_col(soil_df, "fines_content", 10.0),
        clay_fraction=_col(soil_df, "clay_fraction", 2.0), sigma_eff=_col(soil_df, "sigma_eff", 100.0),
        OCR=_col(soil_df, "OCR", 1.0), K0=_col(soil_df, "K0", 0.5),
        Vs1=_col(soil_df, "Vs1", _col(soil_df, "V_s", 180.0)),
        static_shear_ratio=_col(soil_df, "static_shear_ratio", 0.0),
        cementation_index=_col(soil_df, "cementation_index", 0.0), aging_years=_col(soil_df, "aging_years", 1.0),
        saturation=_col(soil_df, "saturation", 1.0), B=_col(soil_df, "B_value", 0.95),
        damping_percent=_col(soil_df, "damping_ratio", _col(soil_df, "xi", 0.03) * 100.0),
        frequency=frequency, loading_mode_factor=np.ones(n),
    )
    soil_df["crr_alpha"] = crr["alpha"].astype(np.float32)
    soil_df["crr_betta"] = crr["betta"].astype(np.float32)
    soil_df["crr_ref"] = crr["crr_ref"].astype(np.float32)
    soil_df["crr_cycle_slope"] = crr["cycle_slope"].astype(np.float32)
    soil_df["crr_state_proxy"] = crr["state_proxy"].astype(np.float32)
    soil_df["response_type"] = np.array(RESPONSE_TYPES, dtype=object)[crr["response_idx"]]
    for key, value in crr.items():
        if key.startswith("f_"):
            soil_df[f"crr_{key}"] = value.astype(np.float32)
    return soil_df


def strict_pre_onset_prefix_mask(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    prefix_len: int,
    *,
    strict: bool = True,
    onset_threshold: float = 0.95,
    margin: int = 1,
    min_len: int = 3,
) -> np.ndarray:
    """
    Маска наблюдаемого префикса, **обрезанного строго до onset разжижения** (P0-c, анти-утечка).

    Без обрезки префикс = первые ``prefix_len`` валидных шагов; на быстрых опытах это окно уже
    содержит момент разжижения (ru пересекает ``onset_threshold``), и модель «видит ответ».
    Здесь для каждого образца последний шаг префикса гарантированно ``< onset_idx`` (а с учётом
    ``margin`` — ``< onset_idx − margin``), поэтому во входе нет ни одной post-onset точки.

    Гарантия отсутствия утечки: все индексы префикса ``< onset_idx`` ⇒ ru на них ``< onset_threshold``.
    Пол ``min_len`` применяется только если он не пересекает onset (иначе берётся более короткий
    префикс — такие сверхбыстрые опыты помечаются вызывающей стороной как трудные для onset-прогноза).

    :param r_obs: измеренная траектория PPR (ru), форма (n, seq_len)
    :param valid_mask: маска валидной длины, форма (n, seq_len)
    :param prefix_len: максимальная длина префикса
    :param strict: если False — старое поведение (первые ``prefix_len`` шагов, БЕЗ обрезки)
    :param onset_threshold: порог ru, определяющий onset
    :param margin: доп. буфер шагов перед onset
    :param min_len: желаемая минимальная длина (только когда не пересекает onset)
    :return: бинарная маска префикса, форма (n, seq_len), float32
    """
    n, seq_len = r_obs.shape
    idx = np.arange(seq_len)[None, :]
    if not strict:
        return ((idx < prefix_len) & (valid_mask > 0)).astype(np.float32)
    onset_hit = (r_obs >= onset_threshold) & (valid_mask > 0)
    has_onset = onset_hit.any(axis=1)
    onset_idx = np.where(has_onset, onset_hit.argmax(axis=1), seq_len).astype(int)
    cut = onset_idx - int(margin) # последний допустимый индекс = cut−1
    floor = np.minimum(int(min_len), np.where(has_onset, onset_idx, prefix_len))
    cut = np.maximum(cut, floor) # пол min_len, но не дальше onset
    cut = np.clip(cut, 0, prefix_len) # не превышать prefix_len
    mask = (idx < cut[:, None]) & (valid_mask > 0)
    return mask.astype(np.float32)


def landmark_prefix_mask(cycles: np.ndarray, valid_mask: np.ndarray, landmark_cycles: float,
                         prefix_len: Optional[int] = None) -> np.ndarray:
    """
    Префикс-маска по ФИЗИЧЕСКОМУ landmark: наблюдаемы шаги с числом циклов ``≤ landmark_cycles``,
    но НЕ больше первых ``prefix_len`` шагов (архитектурное окно префикса).

    В отличие от окна по индексам сетки (``fixed_k``), окно одинаково в ФИЗИЧЕСКИХ циклах и не
    зависит от исхода. Ограничение по ``prefix_len`` гарантирует, что ВСЕ ветви модели (prefix_obs,
    seq-вход, prefix_summary) видят один и тот же префикс ≤ prefix_len точек (иначе prefix_summary
    использовал бы до 20 точек, а seq-вход — только 12), и что ``prefix_coverage = count/prefix_len ≤ 1``.

    :param cycles: сетка числа циклов на образец, форма (n, seq_len)
    :param valid_mask: маска валидной длины, форма (n, seq_len)
    :param landmark_cycles: физический горизонт landmark N₀ (циклы)
    :param prefix_len: верхняя граница числа шагов префикса (архитектурная); None — без ограничения
    :return: бинарная маска префикса, форма (n, seq_len), float32
    """
    m = (np.asarray(cycles) <= float(landmark_cycles)) & (valid_mask > 0)
    if prefix_len is not None:
        step_idx = np.arange(m.shape[1])[None, :]
        m = m & (step_idx < int(prefix_len)) # не больше первых prefix_len шагов
    return m.astype(np.float32)


def build_observed_prefix(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    prefix_len: int,
    *,
    strict_preonset: bool = True,
    onset_threshold: float = 0.95,
    margin: int = 1,
    min_len: int = 3,
    landmark_cycles: Optional[float] = None,
    cycles: Optional[np.ndarray] = None,
):
    """
    Сформировать наблюдаемый префикс из измеренной траектории (без добавления шума).

    По умолчанию префикс обрезается строго до onset (см. :func:`strict_pre_onset_prefix_mask`),
    что устраняет утечку метки через вход.

    :param r_obs: измеренная траектория PPR, форма (n, seq_len)
    :param valid_mask: маска валидной длины, форма (n, seq_len)
    :param prefix_len: длина префикса
    :param strict_preonset: обрезать строго до onset (рекоменд.)
    :param onset_threshold: порог ru для onset
    :param margin: буфер шагов перед onset
    :param min_len: минимальная длина префикса (если не пересекает onset)
    :return: словарь с ``prefix_obs`` и ``prefix_mask``
    """
    if landmark_cycles is not None and cycles is not None:
        # landmark-протокол: окно по физическим циклам ≤ N₀, capped по prefix_len (одинаковый
        # префикс во всех ветвях модели; prefix_coverage≤1).
        prefix_mask = landmark_prefix_mask(cycles, valid_mask, landmark_cycles, prefix_len=prefix_len)
    else:
        prefix_mask = strict_pre_onset_prefix_mask(
            r_obs, valid_mask, prefix_len, strict=strict_preonset,
            onset_threshold=onset_threshold, margin=margin, min_len=min_len,
        )
    prefix_obs = (r_obs * prefix_mask).astype(np.float32)
    return {"prefix_obs": prefix_obs, "prefix_mask": prefix_mask}


def build_population_from_experiments(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    cycles: np.ndarray,
    csr: np.ndarray,
    r_measured: np.ndarray,
    valid_mask: np.ndarray,
    liq_label: np.ndarray,
    n_liq: np.ndarray,
    config: ExperimentConfig,
    prefix_source: Optional[np.ndarray] = None,
    crr_obs: Optional[np.ndarray] = None,
    crr_obs_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Собрать артефакт популяции из наблюдаемых данных реальных испытаний.

    Результат полностью совместим с :func:`liquefaction_ai.data.io.save_population_artifact` и
    :func:`liquefaction_ai.data.splits.prepare_benchmark_dataset`: содержит признаки, метаданные,
    измеренную траекторию и стратифицированное benchmark-разбиение. Синтетические латентные
    поля не создаются.

    Требования к входным таблицам:
    - ``soil_df`` — свойства грунта; обязательны колонки ``class_id`` (или ``type_ground``),
      ``soil_type`` и физ/мех колонки, используемые в признаках (``e, D_r, I_p, V_s, xi,
      sigma_eff, permeability, fines_content, clay_fraction, Cu``). Если ``crr_alpha``/``crr_betta``
      отсутствуют — вычисляются из свойств.
    - ``load_df`` — параметры нагружения: ``CSR_base, frequency, amp_scale, N_max,
      nonstationarity, load_mode, mode_id``.

    :param soil_df: таблица свойств грунта по испытаниям
    :param load_df: таблица параметров нагружения по испытаниям
    :param cycles: сетка числа циклов измерений, форма (n, seq_len)
    :param csr: история CSR на каждом узле, форма (n, seq_len)
    :param r_measured: измеренная траектория PPR, форма (n, seq_len)
    :param valid_mask: маска валидной длины измерений, форма (n, seq_len)
    :param liq_label: бинарная метка разжижения, форма (n,)
    :param n_liq: число циклов до разжижения (или N_max, если не разжижился), форма (n,)
    :param config: конфигурация эксперимента (длины, нормировки, размеры выборок)
    :param prefix_source: опциональная leakage-free PPR-кривая для входного префикса. Для landmark
        она должна быть построена только из наблюдений до N0; target ``r_measured`` может использовать
        всю траекторию для сглаживания.
    :param crr_obs: опциональная измеренная кривая CRR(N), форма (n, seq_len) (например, по серии
        из 6 образцов); используется как наблюдаемая супервизия границы CRR там, где доступна
    :param crr_obs_mask: опциональная по-образцовая маска наличия измеренной CRR, форма (n,)
    :return: словарь популяции в формате артефакта (только наблюдаемые поля)
    """
    soil_df = soil_df.reset_index(drop=True).copy()
    load_df = load_df.reset_index(drop=True).copy()
    n, seq_len = r_measured.shape

    # Производные классификационные колонки
    if "class_id" not in soil_df.columns and "type_ground" in soil_df.columns:
        soil_df["class_id"] = soil_df["type_ground"].astype(int) - 1
    if "soil_type" not in soil_df.columns:
        soil_df["soil_type"] = [SOIL_NAMES[int(i)] for i in soil_df["class_id"]]
    if "mode_id" not in load_df.columns and "load_mode" in load_df.columns:
        load_df["mode_id"] = load_df["load_mode"].map({name: i for i, name in enumerate(LOAD_NAMES)})

    # Полное разложение CRR из физической модели (входные признаки + диагностика)
    if "crr_alpha" not in soil_df.columns or "crr_betta" not in soil_df.columns:
        soil_df = enrich_crr_breakdown(soil_df, load_df)
    if "Cu" not in soil_df.columns:
        soil_df["Cu"] = 5.0
    if "Vs1" not in soil_df.columns:
        soil_df["Vs1"] = _col(soil_df, "V_s", 180.0)

    # Колонки, совместимые с синтетическим артефактом (для ноутбуков анализа/оценки)
    ensure_analysis_columns(soil_df, load_df, crr_obs_mask)

    delta_cycles = np.diff(np.concatenate([np.zeros((n, 1)), cycles], axis=1), axis=1).astype(np.float32)
    # Протокол префикса. "landmark" (рекоменд., leakage-free): окно по ФИЗИЧЕСКИМ циклам ≤ N₀ для
    # всех опытов. "fixed_k": фиксированное окно первых K ШАГОВ сетки. "preonset": обрезка до onset
    # (длина зависит от исхода). Все три outcome-independent по входу, кроме preonset.
    _mode = getattr(config, "prefix_mode", "preonset")
    _fixed_k = _mode == "fixed_k"
    _landmark = _mode == "landmark"
    _pref_window = int(getattr(config, "prefix_fixed_k", 6)) if _fixed_k else config.prefix_len
    prefix_values = r_measured if prefix_source is None else np.asarray(prefix_source)
    if prefix_values.shape != r_measured.shape:
        raise ValueError("prefix_source должен иметь ту же форму, что r_measured")
    observations = build_observed_prefix(
        prefix_values.astype(np.float32), valid_mask.astype(np.float32), _pref_window,
        strict_preonset=(False if (_fixed_k or _landmark) else getattr(config, "prefix_strict_preonset", True)),
        onset_threshold=getattr(config, "prefix_onset_threshold", config.liq_threshold),
        margin=getattr(config, "prefix_onset_margin", 1),
        min_len=getattr(config, "prefix_min_len", 3),
        landmark_cycles=(float(getattr(config, "prefix_landmark_cycles", 20.0)) if _landmark else None),
        cycles=(cycles.astype(np.float32) if _landmark else None),
    )
    features = build_feature_matrices(soil_df, load_df, cycles.astype(np.float32), delta_cycles,
                                      csr.astype(np.float32), observations, config.prefix_len,
                                      max_cycle_reference=float(config.max_cycle_reference))

    meta = pd.concat([soil_df, load_df], axis=1)
    meta["liq_label"] = np.asarray(liq_label).astype(int)
    meta["N_liq_true"] = np.asarray(n_liq).astype(np.float32)
    meta["PPR_max_true"] = r_measured.max(axis=1)
    meta["CSR_max"] = csr.max(axis=1)

    # Наблюдаемые вспомогательные цели из измеренной кривой PPR
    obs_targets = derive_observed_targets(r_measured.astype(np.float32), valid_mask.astype(np.float32))

    benchmark = make_benchmark_splits(meta, min(config.benchmark_subset, n), config.seed, config)

    population = {
        "meta": meta,
        "cycles": cycles.astype(np.float32),
        "delta_cycles": delta_cycles,
        "csr": csr.astype(np.float32),
        "r_obs": r_measured.astype(np.float32),
        "valid_mask": valid_mask.astype(np.float32),
        "prefix_mask": observations["prefix_mask"],
        "prefix_obs": observations["prefix_obs"],
        "liq_label": np.asarray(liq_label).astype(np.float32),
        "n_liq_true": np.asarray(n_liq).astype(np.float32),
        "g_obs": obs_targets["g_obs"],
        "risk_proxy": obs_targets["risk_proxy"],
        "static_features": features["static_features"],
        "static_feature_names": features["static_feature_names"],
        "prefix_summary": features["prefix_summary"],
        "prefix_summary_names": features["prefix_summary_names"],
        "seq_inputs": features["seq_inputs"],
        "seq_feature_names": features["seq_feature_names"],
        "benchmark": benchmark,
    }
    # Опциональная измеренная кривая CRR(N) (если есть серия из 6 образцов)
    if crr_obs is not None:
        population["crr_obs"] = np.asarray(crr_obs, dtype=np.float32)
        population["crr_obs_mask"] = (np.ones(n, dtype=np.float32) if crr_obs_mask is None
                                      else np.asarray(crr_obs_mask, dtype=np.float32))
    return population
