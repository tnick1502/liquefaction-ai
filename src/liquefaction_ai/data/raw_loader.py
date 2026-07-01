"""
Чтение и обогащение СЫРЫХ данных циклических испытаний (пиклы digitrock + ведомости .xls).

Этот модуль собирает всю «грязную» работу с исходными данными, которая раньше была зашита прямо
в ноутбук-загрузчик: распаковку пиклов ``data/handler CyclicModel`` без зависимости от digitrock,
извлечение массивов PPR(N) и свойств грунта одного образца, обход папок объектов, парсинг
ведомости свойств (.xls) и подгонку измеренной кривой CRR(N) по группам ИГЭ. Верхнеуровневая
функция :func:`build_real_objects_population` принимает путь(и) к папке «Облако разжижения» и
возвращает готовый артефакт популяции (через :func:`build_population_from_experiments`) — тот же,
что используется для обучения. Таким образом ноутбукам остаётся только вызвать одну функцию и
сохранить результат.

Сглаживание квазисинусоиды PPR в монотонную огибающую берётся из :mod:`liquefaction_ai.data.ppr_envelope`,
а физическое обогащение (CRR, Vs, PLAXIS-класс и т.д.) — из :mod:`liquefaction_ai.data.real_adapter`.
"""
from __future__ import annotations

import glob
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from liquefaction_ai.config import ExperimentConfig, LIQ_THRESHOLD
from liquefaction_ai.data.ppr_envelope import (causal_monotone_smooth, extract_cycle_amplitude,
                                               extract_upper_envelope,
                                               landmark_aware_cycles, monotone_smooth,
                                               smooth_ppr_trajectory)
from liquefaction_ai.data.real_adapter import build_population_from_experiments


def _cycle_bins(cycles) -> np.ndarray:
    """Целочисленный НОМЕР ЦИКЛА для каждого пика (устойчиво к субцикловому дрожанию фазы).

    В режиме ``points_in_cycle`` номера циклов пиков — ИЗМЕРЕННЫЕ float-значения (аргмакс внутри
    окна одного цикла), которые около onset дрожат по фазе: два ПОДРЯД идущих цикла могут иметь пики
    в k+0.9 и (k+1)+0.1 → сырой diff ≈ 0.2 и «выглядит» как один цикл, хотя это два соседних. Поэтому
    последовательность определяется по ЦЕЛОМУ номеру цикла ``floor(cyc)``, а НЕ по сырым float-разностям.
    Небольшой ε страхует от fp-погрешности (2.0 хранится как 1.9999999)."""
    return np.floor(np.asarray(cycles, dtype=float) + 1e-6).astype(np.int64)


def _consecutive_cycles(cycles) -> bool:
    """Все соседние пики окна принадлежат ПОСЛЕДОВАТЕЛЬНЫМ целым циклам (шаг ровно 1)."""
    if cycles is None:
        return True
    bins = _cycle_bins(cycles)
    return bins.size <= 1 or bool(np.all(np.diff(bins) == 1))


def sustained_first_crossing(curve, thr: float = LIQ_THRESHOLD, sustain: int = 3,
                             cycles=None) -> int:
    """
    Индекс первого ЦИКЛА устойчивого пересечения порога ``thr`` на СЫРОЙ (не монотонной) кривой.

    Причинно/физически корректный критерий onset разжижения:

    * применять к СЫРЫМ поцикловым пикам, НЕ к изотонической огибающей — на монотонной кривой после
      первого пересечения значение по построению остаётся ≥ порога, и любое sustain-окно там no-op;
    * требуется ПОЛНОЕ окно из ``sustain`` подряд идущих ЦЕЛЫХ циклов ≥ порога (последовательность —
      по целому номеру цикла ``floor(cyc)``, устойчиво к субцикловому дрожанию фазы пиков в режиме
      ``points_in_cycle``; сырые float-разности для этого не годятся, см. :func:`_cycle_bins`);
    * усечённый хвост НЕ принимается как событие. Такие случаи аудируются отдельно как
      terminal-ambiguous (:func:`terminal_onset_ambiguous`), поскольку правило «два последних
      превышения вместо трёх» меняло бы определение события в зависимости от момента остановки опыта.

    :param curve: поцикловые значения ru (сырые пики), 1-D
    :param thr: порог разжижения ru (обычно :data:`LIQ_THRESHOLD`)
    :param sustain: требуемое число подряд идущих циклов ≥ порога
    :param cycles: физические номера циклов пиков; если заданы, окно должно покрывать ``sustain``
        последовательных целых циклов
    :return: индекс первого устойчивого пересечения; ``-1`` если события нет
    """
    c = np.asarray(curve)
    cyc = np.arange(c.size, dtype=float) if cycles is None else np.asarray(cycles, dtype=float)
    if cyc.shape != c.shape:
        raise ValueError("cycles и curve должны иметь одинаковую форму")
    above = c >= thr
    n = c.size
    s = int(max(1, sustain))
    for i in range(n):
        if not above[i]:
            continue
        j = i + s
        if j <= n and bool(above[i:j].all()):
            if cycles is None or _consecutive_cycles(cyc[i:j]):
                return i
    return -1


def terminal_onset_ambiguous(curve, thr: float = LIQ_THRESHOLD, sustain: int = 3,
                             cycles=None, min_tail: int = 2) -> bool:
    """Есть ли у конца записи правдоподобное, но недостаточно длинное sustained-превышение."""
    c = np.asarray(curve)
    cyc = np.arange(c.size, dtype=float) if cycles is None else np.asarray(cycles, dtype=float)
    if c.shape != cyc.shape or c.size == 0:
        return False
    above = c >= thr
    start = c.size
    while start > 0 and above[start - 1]:
        start -= 1
    tail = c.size - start
    s = max(int(sustain), 1)
    if not (max(int(min_tail), 1) <= tail < s):
        return False
    if cycles is not None and tail > 1 and not _consecutive_cycles(cyc[start:]):
        return False
    return True


def _peak_on_grid(cyc, signal, grid, points_in_cycle=None) -> np.ndarray:
    """Пик произвольного циклического сигнала по циклам, интерполированный на сетку ``grid``.

    Используется для девиатора q(N) и деформации ε(N): берётся по одному пику на цикл (как у PPR),
    затем значения переносятся на ту же сетку циклов, что и сглаженная кривая PPR. Без PPR-клипа
    [0,1] — сохраняется реальная амплитуда напряжения/деформации.
    """
    cp, pk = extract_upper_envelope(cyc, signal, points_in_cycle)
    if cp.size < 2:
        return np.zeros(len(grid), np.float32)
    return np.interp(np.asarray(grid, float), cp, pk).astype(np.float32)

__all__ = [
    "RealUnpickler", "load_pickle", "gv",
    "fines_clay_cu", "dr_proxy", "extract_test",
    "find_object_pickles", "load_object", "discover_objects",
    "read_statement", "fit_alpha_betta", "build_crr_obs",
    "find_cloud_root", "DEFAULT_TEST_TYPES", "TYPE_TO_MODE", "sustained_first_crossing",
    "terminal_onset_ambiguous",
    "build_real_objects_population", "build_cohort_manifest", "canonical_site_id",
]


def build_cohort_manifest(population: Dict[str, object], raw_count: Optional[int] = None) -> Dict[str, object]:
    """
    Манифест когорты: различает RAW (все извлечённые опыты) и ANALYTIC risk set (после landmark-
    фильтра событий до N₀). Возвращает счётчики для §4 статьи и проверки консистентности артефакта.

    :param population: артефакт популяции (из :func:`build_real_objects_population`)
    :param raw_count: число опытов в RAW; если не задано, берётся из ``cohort_filter_counts``
    :return: словарь манифеста (raw/analytic N, классы, объекты, режимы, CRR образцы/объекты)
    """
    meta = population["meta"]; lab = np.asarray(population["liq_label"])
    filt = dict(population.get("cohort_filter_counts", {}))
    raw_n = int(raw_count if raw_count is not None else filt.get("raw_specimens", len(meta)))
    excluded_event = int(filt.get("excluded_event_before_N0", 0))
    excluded_censored = int(filt.get("excluded_censored_before_N0", 0))
    crrm = population.get("crr_obs_mask")
    crr_n = int((np.asarray(crrm) > 0.5).sum()) if crrm is not None else 0
    crr_obj = int(meta.loc[np.asarray(crrm) > 0.5, "object"].nunique()) if crrm is not None else 0
    crr_site = int(meta.loc[np.asarray(crrm) > 0.5, "site_id"].nunique()) \
        if crrm is not None and "site_id" in meta else crr_obj
    events = np.asarray(population.get("n_liq_true", []), dtype=float)[lab > 0.5]
    return {
        "raw_specimens": raw_n,
        "analytic_risk_set": int(len(meta)),
        "excluded_before_N0_total": int(raw_n - len(meta)),
        "excluded_event_before_N0": excluded_event,
        "excluded_censored_before_N0": excluded_censored,
        "n_liq": int((lab > 0.5).sum()), "n_nonliq": int((lab < 0.5).sum()),
        "n_objects": int(meta["object"].nunique()),
        "n_sites": int(meta["site_id"].nunique()) if "site_id" in meta else int(meta["object"].nunique()),
        "raw_objects": int(filt.get("raw_objects", meta["object"].nunique())),
        "modes": {str(k): int(v) for k, v in meta["load_mode"].value_counts().items()},
        "crr_samples": crr_n, "crr_objects": crr_obj, "crr_sites": crr_site,
        "terminal_onset_ambiguous_raw": int(filt.get("terminal_onset_ambiguous", 0)),
        "terminal_onset_ambiguous_analytic": int(meta.get(
            "onset_terminal_ambiguous", pd.Series(np.zeros(len(meta), dtype=int))).sum()),
        "audit_prefix_crossed_but_target_after_N0": int(
            filt.get("audit_prefix_crossed_but_target_after_N0", 0)),
        "audit_target_by_N0_but_prefix_not_crossed": int(
            filt.get("audit_target_by_N0_but_prefix_not_crossed", 0)),
        "event_nliq_quantiles": ({str(q): float(np.quantile(events, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
                                 if events.size else {}),
        "site_object_map": ({str(site): sorted(map(str, grp["object"].unique()))
                             for site, grp in meta.groupby("site_id")}
                            if "site_id" in meta else {}),
    }

# Фракции грансостава (как в digitrock) и их характерные размеры, мм
GRAN = ["10", "5", "2", "1", "05", "025", "01", "005", "001", "0002", "0000"]
GRAN_SIZE = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01, 0.002, 0.001]
GRAN_COLS = ["gran_10", "gran_5", "gran_2", "gran_1", "gran_05", "gran_025",
             "gran_01", "gran_005", "gran_001", "gran_0002", "gran_0000"]
# Проницаемость по типу грунта ГОСТ (1…9), м/с — порядковые оценки
PERM_BY_TYPE = {1: 1e-4, 2: 5e-5, 3: 1e-5, 4: 5e-6, 5: 1e-6, 6: 1e-7, 7: 1e-8, 8: 1e-9, 9: 1e-7}
FINES_BY_TYPE = {1: 3, 2: 4, 3: 5, 4: 8, 5: 20, 6: 45, 7: 75, 8: 90, 9: 60}
CLAY_BY_TYPE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 8, 7: 20, 8: 40, 9: 15}
# Тип воздействия (имя папки) → режим нагружения модели
TYPE_TO_MODE = {"Потенциал разжижения": "seismic", "Сейсморазжижение": "seismic",
                "Сейсмо": "seismic", "Штормовое разжижение": "storm"}
DEFAULT_TEST_TYPES = ["Потенциал разжижения", "Штормовое разжижение", "Сейсмо"]


# ============================ Распаковка пиклов ============================

class _Stub:
    """Заглушка под любой неизвестный класс digitrock: принимает state в ``__dict__``."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class RealUnpickler(pickle.Unpickler):
    """Unpickler пиклов digitrock без установленного пакета digitrock."""

    def find_class(self, module, name):
        if module.startswith("numpy") or module in ("builtins", "collections", "_codecs", "copy_reg"):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def load_pickle(path: str):
    """Прочитать пикл ``data``/``handler`` без зависимостей digitrock."""
    with open(path, "rb") as f:
        return RealUnpickler(f).load()


def gv(obj, name, default=None):
    """Достать атрибут объекта; если это обёртка результата — вернуть её ``_value``/``value``."""
    d = getattr(obj, "__dict__", {})
    if name not in d:
        return default
    v = d[name]
    if hasattr(v, "__dict__"):
        vd = v.__dict__
        if "_value" in vd:
            return vd["_value"]
        if "value" in vd:
            return vd["value"]
    return v


# ============================ Свойства грунта ============================

def fines_clay_cu(phys, type_ground: int, Ip) -> Tuple[float, float, float]:
    """Вернуть ``(fines_content %, clay_fraction %, Cu)`` из грансостава либо оценкой по типу ГОСТ."""
    gd = {k: gv(phys, f"granulometric_{k}", None) for k in GRAN}
    if any(v not in (None, "") for v in gd.values()):
        v = {k: (float(gd[k]) if gd[k] not in (None, "") else 0.0) for k in GRAN}
        fines = v["005"] + v["001"] + v["0002"] + v["0000"] # ≤ 0.05 мм
        clay = v["0000"] # < 0.002 мм
        frac_pass = np.clip(np.cumsum([v[k] for k in GRAN][::-1])[::-1], 0, 100)
        d = np.array(GRAN_SIZE, float); order = np.argsort(frac_pass)
        try:
            D10 = np.interp(10, frac_pass[order], d[order])
            D60 = np.interp(60, frac_pass[order], d[order])
            Cu = float(np.clip(D60 / max(D10, 1e-4), 1.0, 200.0))
        except Exception:
            Cu = 5.0
        return float(fines), float(clay), Cu
    clay = float(np.clip(Ip * 1.4, 2, 60)) if Ip else CLAY_BY_TYPE.get(int(type_ground), 5)
    return float(FINES_BY_TYPE.get(int(type_ground), 30)), float(clay), 5.0


def dr_proxy(e) -> float:
    """Прокси относительной плотности из коэффициента пористости (e≈0.40 плотный → 1.10 рыхлый)."""
    if e is None:
        return 0.5
    return float(np.clip((1.10 - float(e)) / (1.10 - 0.40), 0.0, 1.0))


# ============================ Извлечение одного образца ============================

def extract_test(data_obj, handler_obj, test_type: str, seq_len: int,
                 landmark_n0: Optional[float] = None, landmark_k: Optional[int] = None,
                 horizon_default: Optional[float] = None, onset_sustain_cycles: int = 3):
    """
    Извлечь из одного образца строки свойств/нагрузки и массивы PPR(N).

    :param data_obj: объект ``data`` (содержит ``cycles``/``PPR``)
    :param handler_obj: объект ``handler`` (содержит ``_test_params``/``_test_result``)
    :param test_type: тип воздействия (имя папки)
    :param seq_len: длина целевой сетки PPR(N)
    :return: кортеж ``(soil, load, arrays, liq_label, n_liq)`` либо ``None``, если данных нет
    """
    cyc = gv(data_obj, "cycles"); ppr = gv(data_obj, "PPR")
    if cyc is None or ppr is None or len(cyc) < 3:
        return None
    tp = gv(handler_obj, "_test_params")
    phys = gv(tp, "physical") if tp is not None else None
    tr = gv(handler_obj, "_test_result")
    if tp is None or phys is None:
        return None

    # гладкая линия PPR(N) по верхней огибающей квазисинусоиды
    pic = gv(tp, "points_in_cycle", None)
    coarse_grid, coarse_r, coarse_mask, _cp, ppr_peaks = smooth_ppr_trajectory(
        cyc, ppr, seq_len, points_in_cycle=pic, return_peaks=True
    )
    # Полноразрешённая поцикловая огибающая. Landmark-сетка должна интерполироваться именно из неё:
    # повторная интерполяция из coarse_grid теряет ранние наблюдения и позволяет узлам после N0
    # влиять на префикс до N0.
    ppr_sm_pc = monotone_smooth(ppr_peaks) if len(ppr_peaks) else np.zeros(0, float)
    ppr_causal_pc = causal_monotone_smooth(ppr_peaks) if len(ppr_peaks) else np.zeros(0, float)
    peak = float(ppr_peaks.max()) if len(ppr_peaks) else float(np.nanmax(np.nan_to_num(ppr)))

    # ---- нагрузка и ГОРИЗОНТЫ (a-priori plan vs наблюдаемая длительность) ----
    sigma_1 = gv(tp, "sigma_1", 100.0) or 100.0
    t = gv(tp, "t", None)
    csr = float(t) / float(sigma_1) if t else 0.2
    n_total = max(int(np.floor(np.nanmax(cyc))), 1)
    # last_obs — ФАКТический последний наблюдённый цикл (длительность опыта). Это censor time для
    # N_liq, но он НЕ должен попадать во ВХОД (grid/seq_in/delta): у разжижившихся опыт обрывают на
    # onset, поэтому last_obs≈N_liq и его утечка в сетку даёт corr(log last_obs, log N_liq)≈0.96.
    last_obs = float(_cp[-1]) if len(_cp) else float(
        coarse_grid[coarse_mask > 0].max() if np.any(coarse_mask > 0) else coarse_grid[-1]
    )
    # N_max — ПЛАНОВЫЙ (a-priori) горизонт: число циклов плана, заданное ДО опыта. Им задаётся
    # endpoint входной/query-сетки (одинаково для liquefied и non-liq, без утечки факт. длительности).
    cycles_count = gv(tp, "cycles_count", None)
    n_max_planned = float(cycles_count) if (cycles_count and float(cycles_count) > 0) else None
    n_max = n_max_planned if n_max_planned is not None else float(n_total)
    # ЕДИНЫЙ горизонт задачи: endpoint входной/query-сетки = ФИКСИРОВАННЫЙ horizon_default
    # (=max_cycle_reference) для ВСЕХ опытов, а НЕ cycles_count. Аудит 1093 опытов: cycles_count≈
    # last_obs (corr с N_liq≈0.96) — это суррогат длительности, не строго a-priori горизонт. Единая
    # сетка делает prediction horizon одинаковым и убирает утечку длительности во входы.
    grid_horizon = float(horizon_default) if (horizon_default and horizon_default > 0) else float(n_total)

    # Событие (РЕТРОСПЕКТИВНЫЙ таргет) = первый ЦИКЛ устойчивого пересечения ru≥LIQ_THRESHOLD.
    # ВАЖНО про строгость критерия:
    #   • Критерий применяется к СЫРЫМ поцикловым пикам ``ppr_peaks``, а НЕ к монотонной огибающей:
    #     на изотонической ``ppr_sm_pc`` после первого пересечения значение по построению остаётся
    #     ≥ порога, поэтому sustain-окно там НИЧЕГО не фильтровало (был no-op). На сырых пиках sustain
    #     реально отсекает одиночные числовые всплески у порога.
    #   • Требуется ПОЛНОЕ окно из ``sustain`` подряд идущих ЦЕЛЫХ циклов ≥ порога. Усечённый хвост
    #     (полное окно не влезает у конца записи) событием НЕ считается — такие случаи отдельно
    #     помечаются как terminal-ambiguous (onset_terminal_ambiguous) и аудируются, а не «дотягиваются»
    #     до события правилом «2 вместо 3», которое привязало бы определение onset к моменту остановки.
    # Это РЕТРОСПЕКТИВНЫЙ таргет (пики берутся по всей записи) — он корректен как метка/цель, но НЕ
    # является причинным; причинным (для входа) является ТОЛЬКО префикс ниже. fail_cycle лаборатории
    # как независимый триггер НЕ используется (расходился с порогом).
    def _sustained_first_crossing(curve, cycle_values):
        return sustained_first_crossing(curve, thr=LIQ_THRESHOLD,
                                        sustain=int(max(1, onset_sustain_cycles)), cycles=cycle_values)
    _cross_i = _sustained_first_crossing(ppr_peaks, _cp) if len(ppr_peaks) else -1
    onset_terminal_ambiguous = int(terminal_onset_ambiguous(
        ppr_peaks, thr=LIQ_THRESHOLD, sustain=int(max(1, onset_sustain_cycles)), cycles=_cp,
    )) if len(ppr_peaks) else 0
    if _cross_i >= 0 and len(_cp):
        liq = 1
        n_liq = float(_cp[int(_cross_i)]) # цикл устойчивого пересечения (поцикловое разрешение)
    else:
        # нет пересечения → событие не наступило в окне; N_liq право-цензурирован на ФАКТическом
        # последнем наблюдённом цикле (наблюдённая длительность = нижняя граница N_liq), а НЕ на
        # плановом горизонте: planned и censoring — разные величины.
        liq = 0
        n_liq = float(last_obs)

    # ЕДИНАЯ early-cycle grid (landmark): первые k узлов = geomspace(1, N₀, k), endpoint =
    # ФИКСИРОВАННЫЙ горизонт (max_cycle_reference) — сетка ИДЕНТИЧНА для всех опытов (uniform horizon,
    # без утечки длительности). mask валиден только до last_obs — это доступность ТАРГЕТА, не длина
    # входной сетки. За last_obs np.interp держит последнее наблюдённое значение (точки замаскированы).
    event_in_prefix = 0 # причинный флаг «разжижение наблюдаемо к N₀»
    if landmark_n0 is not None:
        grid = landmark_aware_cycles(
            grid_horizon, seq_len, float(landmark_n0), int(landmark_k or 12)
        ).astype(np.float32)
        if len(_cp):
            r = np.interp(grid, _cp, ppr_sm_pc).astype(np.float32)
            r_causal = np.interp(grid, _cp, ppr_causal_pc).astype(np.float32)
        else:
            r = np.zeros(seq_len, np.float32)
            r_causal = np.zeros(seq_len, np.float32)
        mask = (grid <= last_obs + 1e-6).astype(np.float32)

        # ПРИЧИННЫЙ префикс: сглаживается ТОЛЬКО по наблюдениям до landmark N₀ (никакого будущего).
        # НИКАКОГО клиппинга по метке — прежний клип `min(prefix, порог) при grid<n_liq` был
        # outcome-conditioned (зависел от будущего n_liq/liq, неизвестного в prospective inference) и
        # к тому же лепил плато ровно 0.949, которое само становилось proxy-сигналом. Вместо
        # переписывания входа: причинный префикс строим как есть, а образцы, у которых он УЖЕ пересёк
        # порог к N₀, ПОМЕЧАЕМ (event_in_prefix) для исключения из landmark risk set в сборке.
        early = np.asarray(_cp) <= float(landmark_n0) + 1e-9
        if int(early.sum()) > 0:
            cp_early = np.asarray(_cp)[early]
            peaks_early = np.asarray(ppr_peaks)[early]
            ppr_early = monotone_smooth(peaks_early)
            prefix_r = np.interp(grid, cp_early, ppr_early).astype(np.float32)
            # event_in_prefix — БЕЗ обращения к метке: устойчивое пересечение порога на СЫРЫХ пиках
            # причинного окна ≤ N₀ (тот же sustained-критерий). Если наблюдаемое разжижение УЖЕ
            # случилось к N₀, образец не может быть forecasting-таргетом → пометка на исключение.
            event_in_prefix = int(_sustained_first_crossing(peaks_early, cp_early) >= 0)
        else:
            prefix_r = np.zeros(seq_len, np.float32)
            event_in_prefix = 0
    else:
        grid = coarse_grid.astype(np.float32)
        r = coarse_r.astype(np.float32)
        mask = coarse_mask.astype(np.float32)
        prefix_r = r.copy()
        r_causal = (np.interp(grid, _cp, ppr_causal_pc).astype(np.float32)
                    if len(_cp) else np.zeros(seq_len, np.float32))

    # q(N) и ε(N) переносятся на ФИНАЛЬНУЮ сетку напрямую из поцикловых пиков, без двойного
    # ресэмплинга. Это сохраняет физическое раннее разрешение так же, как для PPR.
    dev = gv(data_obj, "deviator"); eps_sig = gv(data_obj, "strain")
    q_grid = _peak_on_grid(cyc, dev, grid, pic) if dev is not None else np.zeros(seq_len, np.float32)
    eps_grid = _peak_on_grid(cyc, eps_sig, grid, pic) if eps_sig is not None else np.zeros(seq_len, np.float32)

    # ИЗМЕРЕННАЯ история нагружения CSR(N) из амплитуды девиатора (вместо КОНСТАНТЫ и ручного
    # nonstationarity). CSR(N) привязан к номиналу CSR_base = t/σ₁ через отношение амплитуд (единицы
    # консистентны), для переменно-амплитудных опытов варьируется. Замечание: на этих данных опыты
    # контролируемо-амплитудные (амплитуда девиатора почти постоянна, CV≈0.02), поэтому CSR(N) выходит
    # ~плоской — это ФАКТ данных, а не выдумка. nonstationarity тоже становится DATA-DERIVED (CV
    # измеренной амплитуды), а не зашитой 0.30/0.05. Нет девиатора → откат на константу CSR_base.
    meas_nonstat = 0.05
    if dev is not None:
        _ca, _amp = extract_cycle_amplitude(cyc, dev, pic)
        _amp = np.asarray(_amp, float)
        if _amp.size >= 3 and np.nanmax(_amp) > 0:
            _ref = float(np.nanmedian(_amp[:max(3, _amp.size // 10)]))
            _csr_pc = (np.clip(csr * _amp / _ref, 0.0, csr * 3.0 + 1e-6) if _ref > 1e-9
                       else np.full(_amp.size, csr))
            csr_series = np.interp(grid, _ca, _csr_pc).astype(np.float32)
            meas_nonstat = float(np.clip(np.nanstd(_amp) / max(abs(np.nanmean(_amp)), 1e-6), 0.0, 1.0))
        else:
            csr_series = np.full(seq_len, csr, np.float32)
    else:
        csr_series = np.full(seq_len, csr, np.float32)

    # свойства грунта
    tg = int(gv(phys, "type_ground", 7) or 7)
    Ip = gv(phys, "Ip", None); e = gv(phys, "e", None)
    rho = gv(phys, "r", 2.0) or 2.0
    # Vs по формуле digitrock: Vs = √(G/ρ), G = E/(2(1+ν)). Источники по убыванию приоритета:
    #   1) измеренная скорость поперечной волны (result.transverse_waves_velocity, ед. √(МПа/(г/см³)) → ×31.6228);
    #   2) динамический G (result.G, МПа) — там же, где tw (динамический тест);
    #   3) СТАТИЧЕСКИЙ модуль деформации E0/E из _test_params (есть у ~100% опытов; значения в кПа →
    #      переводим в МПа), G = E/(2(1+ν)). Это и есть «формула», а не зашитая константа.
    # Хардкод G=25 МПа — только если ВСЕ источники отсутствуют (на реальных данных ≈никогда).
    def _modulus_mpa(x):
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        if v <= 0:
            return None
        return v / 1000.0 if v > 200.0 else v # >200 → кПа → МПа; иначе уже МПа
    nu = gv(tp, "unload_poisons_ratio", 0.2) or 0.2
    tw = gv(tr, "transverse_waves_velocity", None)
    _G_dyn = gv(tr, "G", None)
    _E = _modulus_mpa(gv(tr, "E0", None)) or _modulus_mpa(gv(tp, "E0", None)) or _modulus_mpa(gv(tp, "E", None))
    _vs_measured = 1.0
    if tw and float(tw) > 0:
        Vs = float(tw) * 31.6228
    elif _G_dyn and float(_G_dyn) > 0:
        Vs = float((float(_G_dyn) * 1e6 / (float(rho) * 1e3)) ** 0.5)
    elif _E is not None: # статический E0/E → G → Vs (формула)
        G_mpa = float(_E) / (2 * (1 + float(nu)))
        Vs = float((G_mpa * 1e6 / (float(rho) * 1e3)) ** 0.5)
    else:
        Vs = float((25.0 * 1e6 / (float(rho) * 1e3)) ** 0.5)
        _vs_measured = 0.0 # ни одного источника — зашитый G=25 (fabricated)
    Vs = float(np.clip(Vs, 40, 600))
    sigma_eff = float(sigma_1)
    fines, clay, Cu = fines_clay_cu(phys, tg, Ip)

    soil = dict(
        laboratory_number=gv(phys, "laboratory_number", ""), borehole=gv(phys, "borehole", ""),
        depth=gv(phys, "depth", None), soil_name=gv(phys, "soil_name", ""),
        type_ground=tg, class_id=tg - 1,
        e=float(e) if e is not None else 0.7, D_r=dr_proxy(e),
        I_p=float(Ip) if Ip is not None else 0.0,
        Il=float(gv(phys, "Il", 0.0) or 0.0), Ir=float(gv(phys, "Ir", 0.0) or 0.0),
        V_s=Vs, Vs1=float(np.clip(Vs * (100.0 / max(sigma_eff, 1.0)) ** 0.25, 60, 450)),
        xi=float((gv(tp, "damping_ratio", 1.5) or 1.5) / 100.0),
        sigma_eff=sigma_eff, permeability=PERM_BY_TYPE.get(tg, 1e-7),
        fines_content=fines, clay_fraction=clay, Cu=Cu,
        K0=float(gv(tp, "K0", 0.5) or 0.5), static_shear_ratio=0.0,
        ige=str(gv(phys, "ige", "") or ""), # ИГЭ — для группировки кривых CRR
    )

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    soil.update(dict(
        rs=_f(gv(phys, "rs", None)), rd=_f(gv(phys, "rd", None)), r=_f(rho),
        n_porosity=_f(gv(phys, "n", None)), W=_f(gv(phys, "W", None)),
        Wl=_f(gv(phys, "Wl", None)), Wp=_f(gv(phys, "Wp", None)),
        saturation=_f(gv(phys, "Sr", None)), ground_water_depth=_f(gv(phys, "ground_water_depth", None)),
        B_value=_f(gv(phys, "skempton_initial", None)), cohesion=_f(gv(tp, "c", None)),
        phi=_f(gv(tp, "fi", None)), E_modulus=_f(gv(tp, "E", None)),
        damping_ratio=_f(gv(tp, "damping_ratio", None)),
        calcite=_f(gv(phys, "calcite", None)), dolomite=_f(gv(phys, "dolomite", None)),
    ))
    for col, key in zip(GRAN_COLS, GRAN):
        g = gv(phys, f"granulometric_{key}", None)
        soil[col] = 0.0 if g in (None, "") else float(g)

    # ИНДИКАТОРЫ ПРОПУСКОВ: 1 = свойство ИМПУТИРОВано константой (а не измерено). Делает явным
    # то, что статья ошибочно отрицала («rather than fabricated defaults»): модель видит, какие входы
    # суррогатные, и может это учесть; отдельно позволяет оценить чувствительность к импутации.
    _gran_present = any(gv(phys, f"granulometric_{k}", None) not in (None, "") for k in GRAN)
    soil.update(dict(
        miss_e=0.0 if e is not None else 1.0,
        miss_Ip=0.0 if Ip is not None else 1.0,
        miss_K0=0.0 if gv(tp, "K0", None) is not None else 1.0,
        miss_vs=float(1.0 - _vs_measured),
        miss_gran=0.0 if _gran_present else 1.0,
    ))

    # ФЛАГ причинной утечки в префикс: 1 = разжижение НАБЛЮДАЕМО (устойчивое пересечение
    # на сырых пиках) уже в причинном окне ≤ N₀ → запись нельзя использовать как forecasting-таргет.
    # Определён БЕЗ обращения к ретроспективной метке (см. выше), пробрасывается в meta; landmark risk
    # set в сборке = {event_in_prefix == 0}. Ретроспективный контроль (n_liq ≤ N₀) считается там же и
    # расхождения аудируются отдельно, а не «чинятся» переписыванием входа.
    load = dict(
        CSR_base=csr, frequency=float(gv(tp, "frequency", 0.5) or 0.5),
        amp_scale=1.0, N_max=n_max, N_max_is_planned=(n_max_planned is not None),
        nonstationarity=meas_nonstat, # DATA-DERIVED (не зашитая 0.30/0.05)
        load_mode=TYPE_TO_MODE.get(test_type, "seismic"),
        event_in_prefix=event_in_prefix, # 1 = onset попал в окно N₀ (исключить из risk set)
        onset_terminal_ambiguous=onset_terminal_ambiguous,
    )
    arrays = dict(
        cycles=grid.astype(np.float32), csr=csr_series, # измеренная CSR(N), не константа
        r=r.astype(np.float32), mask=mask.astype(np.float32),
        prefix_r=prefix_r.astype(np.float32),
        r_causal=r_causal.astype(np.float32),
        q=q_grid, eps=eps_grid, # девиатор и деформация на сетке циклов (для топологии 4_2)
    )
    return soil, load, arrays, int(liq), float(n_liq)


# ============================ Обход папок объектов ============================

def find_object_pickles(obj_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Найти пути к пиклам ``data``/``handler CyclicModel`` внутри папки объекта."""
    dp = glob.glob(os.path.join(obj_dir, "**", "data CyclicModel*"), recursive=True)
    hp = glob.glob(os.path.join(obj_dir, "**", "handler CyclicModel*"), recursive=True)
    return (dp[0], hp[0]) if dp and hp else (None, None)


def load_object(obj_dir: str, test_type: str, seq_len: int,
                landmark_n0: Optional[float] = None, landmark_k: Optional[int] = None,
                horizon_default: Optional[float] = None,
                onset_sustain_cycles: int = 3) -> List[Tuple[str, tuple]]:
    """Извлечь все образцы одного объекта → список ``(ключ, запись extract_test)``."""
    dpath, hpath = find_object_pickles(obj_dir)
    if not dpath:
        return []
    D = load_pickle(dpath)["data"]
    H = load_pickle(hpath)["data"]
    out = []
    for key in D:
        if key not in H:
            continue
        rec = extract_test(D[key], H[key], test_type, seq_len, landmark_n0=landmark_n0,
                           landmark_k=landmark_k, horizon_default=horizon_default,
                           onset_sustain_cycles=onset_sustain_cycles)
        if rec is not None:
            out.append((key, rec))
    return out


def discover_objects(type_dir: str) -> List[Tuple[str, str]]:
    """Список ``(имя, путь)`` объектов внутри папки типа воздействия (где есть пиклы)."""
    if not os.path.isdir(type_dir):
        return []
    res = []
    for name in sorted(os.listdir(type_dir)):
        p = os.path.join(type_dir, name)
        if os.path.isdir(p) and glob.glob(os.path.join(p, "**", "data CyclicModel*"), recursive=True):
            res.append((name, p))
    return res


def find_cloud_root(candidates: Sequence[os.PathLike | str]) -> Optional[Path]:
    """
    Вернуть первый кандидат, который реально СОДЕРЖИТ сырые пиклы опытов (или None).

    Проверяется не только существование папки, но и наличие хотя бы одного файла
    ``data CyclicModel*`` внутри — пустая папка «Облако разжижения» пропускается, чтобы
    подготовка корректно откатилась на уже сохранённый артефакт.
    """
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists() and glob.glob(os.path.join(str(p), "**", "data CyclicModel*"), recursive=True):
            return p
    return None


# ============================ Ведомость свойств (.xls) ============================

# Карта позиций колонок ведомости (как в digitrock: PhysicalPropertyPosition)
STATEMENT_COLS = {
    "laboratory_number": 0, "borehole": 1, "depth": 2, "soil_name": 3,
    "granulometric_10": 4, "granulometric_5": 5, "granulometric_2": 6, "granulometric_1": 7,
    "granulometric_05": 8, "granulometric_025": 9, "granulometric_01": 10, "granulometric_005": 11,
    "granulometric_001": 12, "granulometric_0002": 13, "granulometric_0000": 14,
    "rs": 15, "r": 16, "rd": 17, "n": 18, "e": 19, "W": 20, "Sr": 21, "Wl": 22, "Wp": 23,
    "Ip": 24, "Il": 25, "Ir": 30, "ground_water_depth": 35,
}


def read_statement(xls_path: str) -> pd.DataFrame:
    """
    Прочитать ведомость .xls в таблицу свойств (индекс — лабораторный номер).

    Требует пакет ``xlrd`` (для старого формата .xls). Строка данных определяется автоматически —
    первая, где в колонке A непустой лабораторный номер вида ``'1-1'``/``'12'``.
    """
    raw = pd.read_excel(xls_path, header=None, engine="xlrd")
    start = None
    for i in range(min(40, len(raw))):
        a = str(raw.iat[i, 0]).strip()
        if re.match(r"^\d+(-\d+)?$", a):
            start = i; break
    if start is None:
        return pd.DataFrame()
    rows = []
    for i in range(start, len(raw)):
        a = str(raw.iat[i, 0]).strip()
        if not re.match(r"^\d+(-\d+)?$", a):
            continue
        rows.append({name: raw.iat[i, col] for name, col in STATEMENT_COLS.items()})
    return pd.DataFrame(rows).set_index("laboratory_number")


# ============================ Измеренная CRR(N) по ИГЭ ============================

def _hyp(N, alpha, betta):
    """Кривая CSR(N)=β/N^(1−α) (define_cyclic_stress_ratio_hyp из digitrock)."""
    return betta / (N ** (1.0 - alpha))


def fit_alpha_betta(cycles, csr) -> Optional[Tuple[float, float]]:
    """Подгонка ``(α, β)`` по точкам ``(N_fail, CSR)`` (как approximate_test_data в digitrock)."""
    from scipy.optimize import curve_fit, differential_evolution

    cycles = np.asarray(cycles, float); csr = np.asarray(csr, float)
    m = (cycles > 0) & np.isfinite(cycles) & np.isfinite(csr) & (csr > 0)
    cycles, csr = cycles[m], csr[m]
    if len(cycles) < 3:
        return None
    try:
        de = differential_evolution(lambda p: float(np.sum((csr - _hyp(cycles, *p)) ** 2)),
                                    [(0, 1), (0, 1.5)], seed=3, maxiter=60, tol=1e-3, polish=False)
        popt, _ = curve_fit(_hyp, cycles, csr, p0=de.x, bounds=([0, 0], [0.999, 3.0]), maxfev=5000)
        a, b = float(popt[0]), float(popt[1])
        return (a, b) if (np.isfinite(a) and np.isfinite(b)) else None
    except Exception:
        return None


def build_crr_obs(soil_df: pd.DataFrame, load_df: pd.DataFrame, cycles: np.ndarray,
                  n_liq: np.ndarray, liq_label: np.ndarray, tags: Sequence[str],
                  seq_len: int, min_group: int = 3):
    """CRR(N) по ИГЭ: подгонка ``β/N^(1−α)`` по ``(N_fail, CSR)`` разрушившихся образцов потенциала."""
    n = len(soil_df)
    crr = np.zeros((n, seq_len), np.float32); mask = np.zeros(n, np.float32)
    is_pot = np.array(["Потенциал" in str(t) for t in tags])
    csr_base = load_df["CSR_base"].to_numpy()
    key = (soil_df["object"].astype(str) + "|" + soil_df["ige"].astype(str)).to_numpy()
    df = pd.DataFrame({"i": np.arange(n), "key": key})
    n_groups = 0
    for _, grp in df.groupby("key"):
        idx = grp["i"].to_numpy(); pidx = idx[is_pot[idx]]
        fl = pidx[liq_label[pidx] == 1]
        if len(fl) < min_group:
            continue
        res = fit_alpha_betta(n_liq[fl], csr_base[fl])
        if res is None:
            continue
        a, b = res; n_groups += 1
        for i in pidx:
            crr[i] = np.clip(b / np.maximum(cycles[i], 1.0) ** (1.0 - a), 0.02, 0.9).astype(np.float32)
            mask[i] = 1.0
    return crr, mask, n_groups


def canonical_site_id(object_name: str) -> str:
    """
    Канонический ИД ПЛОЩАДКИ (геологический сайт) из имени папки объекта.

    Геологически связанные проекты на одном адресе (напр. «852-23 …вл.5 (ВГК-5)» и
    «856-23 …вл.5 (ВГК-6)») — это ОДНА площадка; при object-held-out они обязаны быть в одном фолде,
    иначе утечка между скоррелированными площадками. Нормализуем имя: убираем ведущий код проекта
    (``NNN-NN``), скобочные пометки ``(...)``, технические токены (plaxis/мех/вибро/сейсмо/g0/этап),
    пунктуацию и ё→е, схлопываем пробелы. Совпавшие адреса → один ``site_id``.

    :param object_name: имя папки объекта (``oname``)
    :return: нормализованный адрес-ключ площадки
    """
    s = str(object_name).replace("ё", "е").replace("Ё", "Е").lower()
    s = re.sub(r"\([^)]*\)", " ", s) # (вгк-5), (96), (plaxis…)
    s = re.sub(r"^\s*\d+\s*[-_]\s*\d+\s*", " ", s) # ведущий код проекта NNN-NN
    s = re.sub(r"\b(plaxis|мех|вибро|сейсмо|сейсмо|g0|gо|этап)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s) # пунктуация → пробел (пр-д → пр д)
    s = re.sub(r"\s+", " ", s).strip()
    return s or str(object_name).strip().lower()


# ============================ Верхнеуровневая сборка популяции ============================

def build_real_objects_population(
    source_specs: Sequence[Tuple[os.PathLike | str, Sequence[str]]],
    config: ExperimentConfig,
    seq_len: Optional[int] = None,
    max_objects: Optional[int] = None,
) -> Dict[str, object]:
    """
    Прочитать сырые объекты «Облако разжижения» и собрать готовый артефакт популяции.

    Полностью повторяет прежнюю inline-логику ноутбука-загрузчика, но в одном вызове:
    обход объектов → извлечение образцов (``extract_test``) → сборка таблиц свойств/нагрузки и
    массивов PPR → подгонка измеренной CRR(N) по ИГЭ → :func:`build_population_from_experiments`.

    :param source_specs: список ``(корень, [типы воздействия])``; типы — имена подпапок
        (``"Потенциал разжижения"``, ``"Штормовое разжижение"``)
    :param config: конфигурация эксперимента (длины, нормировки, сплит)
    :param seq_len: длина сетки PPR(N) (по умолчанию ``config.seq_len``)
    :param max_objects: ограничение числа объектов на тип (для быстрых прогонов; ``None`` — все)
    :return: словарь популяции в формате артефакта (как у :func:`build_population_from_experiments`)
    :raises ValueError: если не найдено ни одного образца
    """
    seq_len = int(seq_len or config.seq_len)
    soil_rows: List[dict] = []; load_rows: List[dict] = []
    CY, CS, RM, RC, PR, VM, LB, NL, TAG, QM, EM = [], [], [], [], [], [], [], [], [], [], []
    SITE: List[str] = [] # site_id (геологическая площадка)
    multi = len(source_specs) > 1

    for root, types in source_specs:
        root = Path(root)
        rtag = root.name
        for test_type in types:
            objects = discover_objects(str(root / test_type))
            if max_objects:
                objects = objects[:max_objects]
            _lm = getattr(config, "prefix_mode", "preonset") == "landmark"
            _lm_n0 = float(getattr(config, "prefix_landmark_cycles", 20.0)) if _lm else None
            _lm_k = int(getattr(config, "prefix_len", 12)) if _lm else None
            # endpoint входной сетки = a-priori горизонт; если у опыта нет planned cycles_count —
            # глобальный horizon (max_cycle_reference), но НИКОГДА не фактический last_obs.
            _hz = float(getattr(config, "max_cycle_reference", 3000.0))
            _sustain = int(getattr(config, "onset_sustain_cycles", 3))
            for oname, opath in objects:
                recs = load_object(opath, test_type, seq_len, landmark_n0=_lm_n0, landmark_k=_lm_k,
                                   horizon_default=_hz, onset_sustain_cycles=_sustain)
                otag = f"{rtag} · {test_type}/{oname}" if multi else f"{test_type}/{oname}"
                _site = canonical_site_id(oname) # геологическая площадка (адрес)
                for _key, (soil, load, arr, liq, nl) in recs:
                    soil_rows.append(soil); load_rows.append(load)
                    CY.append(arr["cycles"]); CS.append(arr["csr"]); RM.append(arr["r"])
                    RC.append(arr.get("r_causal", arr["r"]))
                    PR.append(arr.get("prefix_r", arr["r"])); VM.append(arr["mask"])
                    QM.append(arr["q"]); EM.append(arr["eps"])
                    LB.append(liq); NL.append(nl); TAG.append(otag); SITE.append(_site)

    if not soil_rows:
        raise ValueError("Объекты не найдены — проверьте source_specs (путь к «Облако разжижения»).")

    # #6: импутация ПЛАНОВОГО горизонта N_max там, где cycles_count отсутствовал. НЕ использовать
    # achieved-длину (у liquefied это onset → утечка). Берём медиану планового N_max по тому же
    # ОБЪЕКТУ (один протокол нагружения площадки), глобальный fallback — медиана всех плановых.
    from collections import defaultdict as _dd
    _planned_all = [float(r["N_max"]) for r in load_rows if r.get("N_max_is_planned")]
    _glob = float(np.median(_planned_all)) if _planned_all else float(config.max_cycle_reference)
    _obj_planned = _dd(list)
    for r, tag in zip(load_rows, TAG):
        if r.get("N_max_is_planned"):
            _obj_planned[tag].append(float(r["N_max"]))
    for i, (r, tag) in enumerate(zip(load_rows, TAG)):
        if not r.get("N_max_is_planned"):
            # ЧИСТО плановая оценка планового горизонта (медиана объекта / глобальная) — ТОЛЬКО для
            # ПРИЗНАКА N_max. НЕ берём max(.., N_liq) (вернуло бы исход в признак) и НЕ трогаем
            # N_liq: censor time неразжижившегося = ФАКТический last_obs (см. extract_test). Смешивать
            # planned-горизонт и censoring-время нельзя — это разные величины.
            r["N_max"] = float(np.median(_obj_planned[tag])) if _obj_planned.get(tag) else _glob
    _imp_vals = [float(r["N_max"]) for r in load_rows if not r.get("N_max_is_planned")]
    _n_imputed = len(_imp_vals)
    if _n_imputed:
        _iv = np.array(_imp_vals)
        # аудит происхождения N_max: доля и распределение импутированных горизонтов (planned отсутствовал)
        print(f"[N_max] planned cycles_count отсутствует у {_n_imputed}/{len(load_rows)} "
              f"({100*_n_imputed/len(load_rows):.1f}%) → импутация медианой объекта (без max(N_liq)). "
              f"Imputed N_max: min/median/max = {_iv.min():.0f}/{np.median(_iv):.0f}/{_iv.max():.0f}.")
    # Причинный флаг event_in_prefix снимаем ИЗ load_rows ДО построения load_df (иначе он утёк бы как
    # входной признак), сохраняя в отдельный массив для формирования risk set и аудита.
    EIP = [int(r.get("event_in_prefix", 0)) for r in load_rows]
    OTA = [int(r.get("onset_terminal_ambiguous", 0)) for r in load_rows]
    for r in load_rows:
        r.pop("N_max_is_planned", None) # служебный флаг — не в признаки
        r.pop("event_in_prefix", None) # причинный флаг — не в признаки (только в meta ниже)
        r.pop("onset_terminal_ambiguous", None) # аудит метки, не входной признак

    # #7/#9: ЕДИНОЕ определение события «разжижение BY горизонта» уже НА СБОРКЕ — чтобы meta.parquet,
    # EDA, стратификация фолдов и CRR-сборка использовали ТО ЖЕ определение, что обучение/метрики
    # (раньше поздние события были label=1 в meta и становились отрицательными только в splits).
    _H = float(config.max_cycle_reference)
    for i in range(len(LB)):
        if LB[i] > 0.5 and NL[i] > _H:
            LB[i] = 0.0 # разжижение после горизонта → не-событие
            NL[i] = _H # право-цензура на горизонт

    cohort_filter_counts = {
        "raw_specimens": int(len(LB)),
        "raw_objects": int(len(set(TAG))),
        "excluded_event_in_prefix_causal": 0, # разжижение НАБЛЮДАЕМО в причинном окне ≤ N₀ (первично)
        "excluded_event_before_N0": 0, # ретроспективный onset ≤ N₀ (таргет уже наступил)
        "excluded_censored_before_N0": 0, # цензура до N₀ (нет валидного risk-периода)
        "audit_prefix_crossed_but_target_after_N0": 0, # причинно пересёк, а ретро-onset > N₀
        "audit_target_by_N0_but_prefix_not_crossed": 0, # ретро-onset ≤ N₀, а причинно не пересёк
        "terminal_onset_ambiguous": int(sum(OTA)), # всего терминально-неоднозначных
        "excluded_terminal_ambiguous": 0, # из них исключено из когорты (если включён флаг)
    }
    _excl_ambig = bool(getattr(config, "exclude_terminal_ambiguous", True))
    # Landmark risk set = образцы, у которых разжижение ещё НЕ наблюдаемо и НЕ наступило к N₀.
    # ПЕРВИЧНЫЙ критерий исключения — ПРИЧИННЫЙ event_in_prefix (наблюдаемо к N₀, без метки). Плюс
    # ретроспективные исключения (onset ≤ N₀ и цензура ≤ N₀). Противоречия причинного и ретро-критериев
    # НЕ «чинятся» переписыванием входа, а ОТДЕЛЬНО аудируются (см. счётчики) — это честный сигнал о
    # рассогласовании сырых пиков и монотонной цели у самой границы окна.
    if getattr(config, "prefix_mode", "preonset") == "landmark":
        _N0 = float(getattr(config, "prefix_landmark_cycles", 20.0))
        cohort_filter_counts["excluded_event_in_prefix_causal"] = int(sum(EIP))
        cohort_filter_counts["excluded_event_before_N0"] = int(sum(
            LB[i] > 0.5 and NL[i] <= _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["excluded_censored_before_N0"] = int(sum(
            LB[i] < 0.5 and NL[i] <= _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["audit_prefix_crossed_but_target_after_N0"] = int(sum(
            EIP[i] == 1 and NL[i] > _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["audit_target_by_N0_but_prefix_not_crossed"] = int(sum(
            EIP[i] == 0 and NL[i] <= _N0 for i in range(len(LB))
        ))
        cohort_filter_counts["excluded_terminal_ambiguous"] = int(sum(
            _excl_ambig and OTA[i] and EIP[i] == 0 and NL[i] > _N0 for i in range(len(LB))
        ))
        # keep = НЕ наблюдаемо причинно, ретро-onset строго после N₀, и (опц.) не терминально-неоднозначно
        _keep = [i for i in range(len(LB))
                 if (EIP[i] == 0 and NL[i] > _N0 and not (_excl_ambig and OTA[i]))]
        if len(_keep) < len(LB):
            soil_rows = [soil_rows[i] for i in _keep]; load_rows = [load_rows[i] for i in _keep]
            CY = [CY[i] for i in _keep]; CS = [CS[i] for i in _keep]; RM = [RM[i] for i in _keep]
            RC = [RC[i] for i in _keep]
            PR = [PR[i] for i in _keep]; VM = [VM[i] for i in _keep]
            QM = [QM[i] for i in _keep]; EM = [EM[i] for i in _keep]
            LB = [LB[i] for i in _keep]; NL = [NL[i] for i in _keep]; TAG = [TAG[i] for i in _keep]
            SITE = [SITE[i] for i in _keep]
            OTA = [OTA[i] for i in _keep]

    soil_df = pd.DataFrame(soil_rows); load_df = pd.DataFrame(load_rows)
    cycles = np.array(CY); csr = np.array(CS)
    r_measured = np.array(RM); valid_mask = np.array(VM)
    liq_label = np.array(LB); n_liq = np.array(NL)
    soil_df["object"] = TAG
    soil_df["site_id"] = SITE # геологическая площадка (группировка CV)
    soil_df["onset_terminal_ambiguous"] = np.asarray(OTA, dtype=np.int8) # meta-only audit column

    crr_obs, crr_obs_mask, _n_groups = build_crr_obs(
        soil_df, load_df, cycles, n_liq, liq_label, soil_df["object"].tolist(), seq_len)
    if crr_obs_mask.sum() == 0:
        crr_obs = crr_obs_mask = None

    population = build_population_from_experiments(
        soil_df=soil_df, load_df=load_df, cycles=cycles, csr=csr, r_measured=r_measured,
        valid_mask=valid_mask, liq_label=liq_label, n_liq=n_liq, config=config,
        prefix_source=np.array(PR, np.float32),
        crr_obs=crr_obs, crr_obs_mask=crr_obs_mask)
    population["cohort_filter_counts"] = cohort_filter_counts
    population["r_causal"] = np.array(RC, np.float32)
    # Девиатор q(N) и деформация ε(N) — сохраняются в артефакт (io пишет любые ndarray-поля),
    # чтобы топология (4_2) строила фазовое пространство X=(PPR,q,ε) на данных проекта.
    population["q_obs"] = np.array(QM, np.float32)
    population["eps_obs"] = np.array(EM, np.float32)
    return population
