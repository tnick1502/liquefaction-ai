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
from liquefaction_ai.data.ppr_envelope import extract_upper_envelope, smooth_ppr_trajectory
from liquefaction_ai.data.real_adapter import build_population_from_experiments


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
    "find_cloud_root", "DEFAULT_TEST_TYPES", "TYPE_TO_MODE",
    "build_real_objects_population",
]

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
        fines = v["005"] + v["001"] + v["0002"] + v["0000"]   # ≤ 0.05 мм
        clay = v["0000"]                                        # < 0.002 мм
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

def extract_test(data_obj, handler_obj, test_type: str, seq_len: int):
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
    grid, r, mask, _cp, ppr_peaks = smooth_ppr_trajectory(cyc, ppr, seq_len, points_in_cycle=pic,
                                                          return_peaks=True)
    peak = float(ppr_peaks.max()) if len(ppr_peaks) else float(np.nanmax(np.nan_to_num(ppr)))

    # Девиатор q(N) и осевая деформация ε(N) — по одному пику на цикл, на ту же сетку, что и PPR.
    # Сохраняются в артефакт (q_obs/eps_obs), чтобы фазовое пространство X=(PPR,q,ε) для топологии
    # (ноутбук 4_2) строилось из данных проекта без повторного парсинга сырых пиклов.
    dev = gv(data_obj, "deviator"); eps_sig = gv(data_obj, "strain")
    q_grid = _peak_on_grid(cyc, dev, grid, pic) if dev is not None else np.zeros(seq_len, np.float32)
    eps_grid = _peak_on_grid(cyc, eps_sig, grid, pic) if eps_sig is not None else np.zeros(seq_len, np.float32)

    # нагрузка
    sigma_1 = gv(tp, "sigma_1", 100.0) or 100.0
    t = gv(tp, "t", None)
    csr = float(t) / float(sigma_1) if t else 0.2
    n_total = max(int(np.floor(np.nanmax(cyc))), 1)
    cycles_count = gv(tp, "cycles_count", n_total) or n_total
    n_max = float(max(cycles_count, n_total))

    # разжижение / N_liq (порог события — единый LIQ_THRESHOLD)
    fail = gv(tr, "fail_cycle", None) if tr is not None else None
    if fail in (None, 0):
        fail = gv(tp, "n_fail", None)
    liq = 1 if (peak >= LIQ_THRESHOLD or fail not in (None, 0)) else 0
    n_liq = float(fail) if (liq and fail not in (None, 0)) else float(n_total)

    # свойства грунта
    tg = int(gv(phys, "type_ground", 7) or 7)
    Ip = gv(phys, "Ip", None); e = gv(phys, "e", None)
    rho = gv(phys, "r", 2.0) or 2.0
    # Vs по формуле digitrock: Vs = √(G/r), G = E0/(2(1+ν)). Готовое значение хранится в
    # result.transverse_waves_velocity (ед. √(МПа/(г/см³))) → ×31.6228 = м/с.
    tw = gv(tr, "transverse_waves_velocity", None)
    if tw and float(tw) > 0:
        Vs = float(tw) * 31.6228
    else:
        G = gv(tr, "G", None)
        if G and float(G) > 0:
            Vs = float((float(G) * 1e6 / (float(rho) * 1e3)) ** 0.5)
        else:
            E0d = gv(tr, "E0", None); nu = gv(tp, "unload_poisons_ratio", 0.2) or 0.2
            G_mpa = (float(E0d) / (2 * (1 + float(nu)))) if E0d else 25.0
            Vs = float((G_mpa * 1e6 / (float(rho) * 1e3)) ** 0.5)
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
        ige=str(gv(phys, "ige", "") or ""),   # ИГЭ — для группировки кривых CRR
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

    load = dict(
        CSR_base=csr, frequency=float(gv(tp, "frequency", 0.5) or 0.5),
        amp_scale=1.0, N_max=n_max,
        nonstationarity=0.30 if test_type == "Штормовое разжижение" else 0.05,
        load_mode=TYPE_TO_MODE.get(test_type, "seismic"),
    )
    arrays = dict(
        cycles=grid.astype(np.float32), csr=np.full(seq_len, csr, np.float32),
        r=r.astype(np.float32), mask=mask.astype(np.float32),
        q=q_grid, eps=eps_grid,        # девиатор и деформация на сетке циклов (для топологии 4_2)
    )
    return soil, load, arrays, int(liq), float(n_liq)


# ============================ Обход папок объектов ============================

def find_object_pickles(obj_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Найти пути к пиклам ``data``/``handler CyclicModel`` внутри папки объекта."""
    dp = glob.glob(os.path.join(obj_dir, "**", "data CyclicModel*"), recursive=True)
    hp = glob.glob(os.path.join(obj_dir, "**", "handler CyclicModel*"), recursive=True)
    return (dp[0], hp[0]) if dp and hp else (None, None)


def load_object(obj_dir: str, test_type: str, seq_len: int) -> List[Tuple[str, tuple]]:
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
        rec = extract_test(D[key], H[key], test_type, seq_len)
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
    CY, CS, RM, VM, LB, NL, TAG, QM, EM = [], [], [], [], [], [], [], [], []
    multi = len(source_specs) > 1

    for root, types in source_specs:
        root = Path(root)
        rtag = root.name
        for test_type in types:
            objects = discover_objects(str(root / test_type))
            if max_objects:
                objects = objects[:max_objects]
            for oname, opath in objects:
                recs = load_object(opath, test_type, seq_len)
                otag = f"{rtag} · {test_type}/{oname}" if multi else f"{test_type}/{oname}"
                for _key, (soil, load, arr, liq, nl) in recs:
                    soil_rows.append(soil); load_rows.append(load)
                    CY.append(arr["cycles"]); CS.append(arr["csr"]); RM.append(arr["r"]); VM.append(arr["mask"])
                    QM.append(arr["q"]); EM.append(arr["eps"])
                    LB.append(liq); NL.append(nl); TAG.append(otag)

    if not soil_rows:
        raise ValueError("Объекты не найдены — проверьте source_specs (путь к «Облако разжижения»).")

    soil_df = pd.DataFrame(soil_rows); load_df = pd.DataFrame(load_rows)
    cycles = np.array(CY); csr = np.array(CS)
    r_measured = np.array(RM); valid_mask = np.array(VM)
    liq_label = np.array(LB); n_liq = np.array(NL)
    soil_df["object"] = TAG

    crr_obs, crr_obs_mask, _n_groups = build_crr_obs(
        soil_df, load_df, cycles, n_liq, liq_label, soil_df["object"].tolist(), seq_len)
    if crr_obs_mask.sum() == 0:
        crr_obs = crr_obs_mask = None

    population = build_population_from_experiments(
        soil_df=soil_df, load_df=load_df, cycles=cycles, csr=csr, r_measured=r_measured,
        valid_mask=valid_mask, liq_label=liq_label, n_liq=n_liq, config=config,
        crr_obs=crr_obs, crr_obs_mask=crr_obs_mask)
    # Девиатор q(N) и деформация ε(N) — сохраняются в артефакт (io пишет любые ndarray-поля),
    # чтобы топология (4_2) строила фазовое пространство X=(PPR,q,ε) на данных проекта.
    population["q_obs"] = np.array(QM, np.float32)
    population["eps_obs"] = np.array(EM, np.float32)
    return population
