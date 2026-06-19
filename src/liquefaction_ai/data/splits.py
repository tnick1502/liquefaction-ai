"""
Разбиение популяции на выборки и формирование benchmark-тензоров.

Модуль отвечает за три связанные задачи:
1. стратифицированный отбор benchmark-подмножества из полной популяции;
2. разбиение benchmark на обучающую / валидационную / тестовую выборки;
3. сборку нормированных тензоров и итератор мини-батчей для обучения моделей.

Стратификация выполняется по тройке (тип грунта, режим нагружения, метка
разжижения) с безопасным огрублением страт, если какие-то комбинации слишком редки.
"""

from __future__ import annotations

from typing import Dict, Iterator, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from liquefaction_ai.config import ExperimentConfig

__all__ = [
    "safe_strata",
    "stratified_subset_indices",
    "make_benchmark_splits",
    "prepare_benchmark_dataset",
    "iterate_minibatches",
]


def _terminal_observability(
    r_obs: np.ndarray,
    valid_mask: np.ndarray,
    liq_label: np.ndarray,
    cycles: np.ndarray | None = None,
    tail_frac: float = 0.20,
    rise_eps: float = 0.02,
    min_complete_cycles: float = 500.0,
) -> np.ndarray:
    """
    Определить, наблюдаема ли терминальная точка N_liq у каждого опыта (три режима).

    Возвращает бинарную маску ``n_liq_observed`` (1 — терминал наблюдаем/корректно
    цензурирован, 0 — оценить нельзя):

    * **разжижение** (``liq_label==1``) → 1: N_liq наблюдён точно;
    * **нет разжижения + стабилизация** (длина опыта ≥ ``min_complete_cycles`` и хвост PPR
      плоский, прирост < ``rise_eps``)
      → 1: N_liq право-цензурирован на практическом горизонте оценки;
    * **нет разжижения + нет стабилизации** (длина опыта < ``min_complete_cycles`` или PPR на
      хвосте ещё растёт)
      → 0: конечную точку оценить нельзя; такие образцы исключаются из супервизии N_liq
      (обучаемся только динамике кривой).

    :param r_obs: измеренная траектория PPR, форма (n, seq_len)
    :param valid_mask: маска валидной длины измерений, форма (n, seq_len)
    :param liq_label: бинарная метка разжижения, форма (n,)
    :param cycles: сетка циклов, форма (n, seq_len); если задана, неразжижившийся опыт короче
        ``min_complete_cycles`` считается незавершённым независимо от хвоста PPR
    :param tail_frac: доля хвоста валидной части для оценки прироста PPR
    :param rise_eps: порог прироста PPR на хвосте, ниже которого кривая считается стабильной
    :param min_complete_cycles: минимальная длительность завершённого неразжижившегося опыта
    :return: маска ``n_liq_observed`` формы (n,), значения {0., 1.}
    """
    n = r_obs.shape[0]
    observed = np.ones(n, dtype=np.float32)
    for i in range(n):
        if liq_label[i] >= 0.5:
            continue                                  # разжижение — наблюдаем точно
        idx = np.where(valid_mask[i] > 0)[0]
        if idx.size == 0:
            observed[i] = 0.0
            continue
        if cycles is not None:
            last_cycle = float(cycles[i, idx[-1]])
            if last_cycle < min_complete_cycles:       # короткий non-liq опыт — терминал неизвестен
                observed[i] = 0.0
                continue
        if idx.size < 5:
            observed[i] = 0.0                          # хвост нельзя надёжно оценить
            continue
        k = max(3, int(tail_frac * idx.size))
        tail = idx[-k:]
        rise = float(r_obs[i, tail[-1]] - r_obs[i, tail[0]])
        if rise >= rise_eps:                          # нет разжижения и нет стабилизации
            observed[i] = 0.0
    return observed


def safe_strata(meta: pd.DataFrame, fine_columns: List[str]) -> np.ndarray:
    """
    Построить безопасные метки страт для стратифицированного разбиения.

    Сначала формируются «тонкие» страты как конкатенация значений ``fine_columns``.
    Если в них встречаются классы с единственным представителем (что ломает
    ``train_test_split`` со стратификацией), такие наблюдения огрубляются до страты
    (режим нагружения, метка разжижения); при необходимости — до одной метки разжижения.

    :param meta: таблица метаданных популяции
    :param fine_columns: список колонок для построения наиболее детальных страт
    :return: массив строковых меток страт длины ``len(meta)``
    """
    fine = meta[fine_columns].astype(str).agg("|".join, axis=1)
    fine_counts = fine.value_counts()
    if fine_counts.min() >= 2:
        return fine.to_numpy()

    medium = meta[["load_mode", "liq_label"]].astype(str).agg("|".join, axis=1)
    medium = fine.where(fine.map(fine_counts) >= 2, medium)
    medium_counts = medium.value_counts()
    if medium_counts.min() >= 2:
        return medium.to_numpy()

    # Огрубление до одной метки разжижения
    coarse = meta["liq_label"].astype(str)
    coarse_counts = coarse.value_counts()
    if coarse_counts.min() >= 2:
        return coarse.to_numpy()

    # Вырожденный случай (на реальных данных разжижается почти 100 %, минорный класс
    # может содержать <2 наблюдений) — отказываемся от стратификации: единая страта.
    return np.zeros(len(meta), dtype=int)


def stratified_subset_indices(meta: pd.DataFrame, subset_size: int, seed: int) -> np.ndarray:
    """
    Отобрать стратифицированное подмножество индексов из популяции.

    :param meta: таблица метаданных популяции
    :param subset_size: желаемый размер подмножества (обрезается до размера популяции)
    :param seed: случайное зерно для воспроизводимого отбора
    :return: отсортированный массив абсолютных индексов отобранных сценариев
    """
    subset_size = min(subset_size, len(meta))
    idx = np.arange(len(meta))
    if subset_size >= len(meta):  # подмножество — вся популяция
        return idx
    strata = safe_strata(meta, ["soil_type", "load_mode", "liq_label"])
    keep_idx, _ = train_test_split(idx, train_size=subset_size, stratify=strata, random_state=seed)
    return np.sort(keep_idx)


def make_benchmark_splits(meta: pd.DataFrame, subset_size: int, seed: int, config: ExperimentConfig) -> Dict[str, np.ndarray]:
    """
    Сформировать benchmark-подмножество и его разбиение train/val/test.

    Сначала из популяции отбирается стратифицированное benchmark-подмножество, затем
    оно разбивается на обучающую, валидационную и тестовую части с сохранением
    стратификации. Доли задаются полями ``benchmark_train_fraction`` и
    ``benchmark_val_fraction`` конфигурации.

    :param meta: таблица метаданных полной популяции
    :param subset_size: размер benchmark-подмножества
    :param seed: случайное зерно для воспроизводимости
    :param config: конфигурация эксперимента (доли train/val)
    :return: словарь с ключами ``benchmark_idx`` (абсолютные индексы) и
             ``train_rel`` / ``val_rel`` / ``test_rel`` (индексы относительно benchmark)
    """
    benchmark_idx = stratified_subset_indices(meta, subset_size, seed)
    benchmark_meta = meta.iloc[benchmark_idx].reset_index(drop=True)
    benchmark_rel = np.arange(len(benchmark_idx))

    # Leakage-free разбиение по объекту/площадке: ни один объект не попадает одновременно в
    # train/val/test (важно для геотехники — образцы одной площадки коррелированы). При наличии
    # measured-CRR объектов test/val получают хотя бы один CRR-объект и один обычный объект, чтобы
    # основной grouped leaderboard не терял CRR_RMSE и бинарные риск-метрики.
    if getattr(config, "group_split_by_object", False) and "object" in benchmark_meta.columns:
        objects = benchmark_meta["object"].to_numpy()
        obj_stats = (
            benchmark_meta.groupby("object")
            .agg(
                n=("object", "size"),
                liq_rate=("liq_label", "mean"),
                has_crr=("has_measured_crr", "max") if "has_measured_crr" in benchmark_meta.columns else ("liq_label", "min"),
            )
            .reset_index()
        )
        uniq = np.array(sorted(obj_stats["object"].tolist()))
        rng = np.random.default_rng(seed)
        perm = list(uniq[rng.permutation(len(uniq))])
        n_obj = len(perm)
        test_fraction = max(0.0, 1.0 - config.benchmark_train_fraction - config.benchmark_val_fraction)
        n_te = max(1, int(round(test_fraction * n_obj)))
        if n_obj >= 6:
            n_te = max(2, n_te)
        n_va = max(1, int(round(config.benchmark_val_fraction * n_obj)))
        n_va = min(n_va, max(1, n_obj - n_te - 1))
        n_tr = max(1, n_obj - n_va - n_te)

        stats = obj_stats.set_index("object")

        def take_balanced(pool: List[str], k: int) -> List[str]:
            """Выбрать k объектов, сохранив CRR/non-CRR покрытие, если это возможно."""
            chosen: List[str] = []
            crr = [o for o in pool if bool(stats.loc[o, "has_crr"])]
            non = [o for o in pool if not bool(stats.loc[o, "has_crr"])]
            if k >= 2 and crr and non:
                chosen.extend([crr[0], non[0]])
            elif crr:
                chosen.append(crr[0])
            elif non:
                chosen.append(non[0])
            for o in pool:
                if len(chosen) >= k:
                    break
                if o not in chosen:
                    chosen.append(o)
            return chosen[:k]

        te_obj = set(take_balanced(perm, n_te))
        remaining = [o for o in perm if o not in te_obj]
        va_obj = set(take_balanced(remaining, n_va))
        tr_obj = set(o for o in remaining if o not in va_obj)
        if len(tr_obj) < n_tr:
            deficit = n_tr - len(tr_obj)
            move = list(va_obj)[:deficit]
            tr_obj.update(move)
            va_obj.difference_update(move)
        train_rel = benchmark_rel[np.isin(objects, list(tr_obj))]
        val_rel = np.sort(benchmark_rel[np.isin(objects, list(va_obj))])
        test_rel = np.sort(benchmark_rel[np.isin(objects, list(te_obj))])
        return {"benchmark_idx": benchmark_idx, "train_rel": np.sort(train_rel),
                "val_rel": val_rel, "test_rel": test_rel}

    strata = safe_strata(benchmark_meta, ["soil_type", "load_mode", "liq_label"])

    train_rel, temp_rel = train_test_split(
        benchmark_rel,
        train_size=config.benchmark_train_fraction,
        stratify=strata,
        random_state=seed,
    )
    temp_meta = benchmark_meta.iloc[temp_rel].reset_index(drop=True)
    temp_strata = safe_strata(temp_meta, ["soil_type", "load_mode", "liq_label"])
    val_fraction_relative = config.benchmark_val_fraction / (1.0 - config.benchmark_train_fraction)
    val_rel_local, test_rel_local = train_test_split(
        np.arange(len(temp_rel)),
        train_size=val_fraction_relative,
        stratify=temp_strata,
        random_state=seed,
    )
    val_rel = np.sort(temp_rel[val_rel_local])
    test_rel = np.sort(temp_rel[test_rel_local])

    return {
        "benchmark_idx": benchmark_idx,
        "train_rel": np.sort(train_rel),
        "val_rel": val_rel,
        "test_rel": test_rel,
    }


def prepare_benchmark_dataset(
    population_dict: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    """
    Собрать нормированные benchmark-тензоры и разбить их на выборки.

    По индексам benchmark из артефакта извлекаются все нужные массивы, статические и
    префиксные признаки масштабируются ``StandardScaler`` (обученным только на train,
    чтобы исключить утечку), последовательностные признаки нормируются поканально.
    Дополнительно вычисляются логарифмически-нормированный таргет ``n_liq_norm`` и
    бинарная зона триггера ``trigger_zone``. Результат раскладывается в готовые к
    обучению словари ``train`` / ``val`` / ``test``.

    :param population_dict: словарь популяции (из генератора или артефакта)
    :param config: конфигурация эксперимента (нормировки, опорное N)
    :param device: целевое устройство (используется итератором мини-батчей)
    :return: словарь с выборками ``train``/``val``/``test``, метаданными,
             обученными скейлерами и именами признаков
    """
    # По умолчанию используем готовое (запечённое в артефакт) разбиение. Если запрошено
    # leakage-free разбиение по объекту — пересобираем его на лету (это меняет train/val/test,
    # обеспечивая, что ни один объект не попадает одновременно в train и test).
    if getattr(config, "group_split_by_object", False):
        bsplit = make_benchmark_splits(population_dict["meta"],
                                       min(config.benchmark_subset, len(population_dict["meta"])),
                                       config.seed, config)
    else:
        bsplit = population_dict["benchmark"]
    benchmark_idx = bsplit["benchmark_idx"]
    bench_meta = population_dict["meta"].iloc[benchmark_idx].reset_index(drop=True)

    # Наблюдаемые (доступные в реальном опыте) массивы — обязательные
    static_raw = population_dict["static_features"][benchmark_idx]
    prefix_raw = population_dict["prefix_summary"][benchmark_idx]
    seq_raw = population_dict["seq_inputs"][benchmark_idx]
    cycles = population_dict["cycles"][benchmark_idx]
    delta_cycles = population_dict["delta_cycles"][benchmark_idx]
    csr = population_dict["csr"][benchmark_idx]
    r_obs = population_dict["r_obs"][benchmark_idx]
    valid_mask = population_dict["valid_mask"][benchmark_idx]
    prefix_mask = population_dict["prefix_mask"][benchmark_idx]
    prefix_obs = population_dict["prefix_obs"][benchmark_idx]
    liq_label = population_dict["liq_label"][benchmark_idx]
    n_liq_true = population_dict["n_liq_true"][benchmark_idx]

    train_rel = bsplit["train_rel"]
    val_rel = bsplit["val_rel"]
    test_rel = bsplit["test_rel"]

    static_scaler = StandardScaler().fit(static_raw[train_rel])
    prefix_scaler = StandardScaler().fit(prefix_raw[train_rel])
    seq_train = seq_raw[train_rel].reshape(-1, seq_raw.shape[-1])
    seq_mean = seq_train.mean(axis=0)
    seq_std = seq_train.std(axis=0) + 1e-6

    static_scaled = static_scaler.transform(static_raw).astype(np.float32)
    prefix_scaled = prefix_scaler.transform(prefix_raw).astype(np.float32)
    seq_scaled = ((seq_raw - seq_mean[None, None, :]) / seq_std[None, None, :]).astype(np.float32)
    # N_liq обучается и оценивается на практическом горизонте max_cycle_reference:
    # после него разжижение считается инженерно ненаступившим, поэтому правую цензуру
    # и редкие поздние события капируем на этот горизонт.
    n_liq_target = np.minimum(n_liq_true, config.max_cycle_reference).astype(np.float32)
    n_liq_norm = (np.log1p(n_liq_target) / np.log1p(config.max_cycle_reference)).astype(np.float32)

    # --- Маска наблюдаемости терминальной точки N_liq (три режима опыта) ---
    # Режим 1 (разжижение, label=1): N_liq наблюдён точно → маска 1.
    # Режим 2 (нет разжижения, длинный опыт, хвост PPR плоский): право-цензура на N_max/horizon
    #          → маска 1 (корректная цензура).
    # Режим 3 (нет разжижения, опыт короче min_nonliq_complete_cycles или PPR ещё растёт):
    #          конечную точку оценить нельзя → маска 0 (N_liq-loss отключаем, динамику PPR учим).
    n_liq_observed = _terminal_observability(
        r_obs, valid_mask, liq_label, cycles=cycles,
        min_complete_cycles=getattr(config, "min_nonliq_complete_cycles", 500.0),
    )

    benchmark_arrays = {
        "static": static_scaled,
        "static_raw": static_raw.astype(np.float32),
        "prefix_summary": prefix_scaled,
        "prefix_summary_raw": prefix_raw.astype(np.float32),
        "seq_in": seq_scaled,
        "seq_in_raw": seq_raw.astype(np.float32),
        "cycles": cycles.astype(np.float32),
        "delta_cycles": delta_cycles.astype(np.float32),
        "csr": csr.astype(np.float32),
        "r_obs": r_obs.astype(np.float32),
        "mask": valid_mask.astype(np.float32),
        "prefix_mask": prefix_mask.astype(np.float32),
        "prefix_obs": prefix_obs.astype(np.float32),
        "label": liq_label.astype(np.float32),
        "n_liq_true": n_liq_target.astype(np.float32),
        "n_liq_raw": n_liq_true.astype(np.float32),
        "n_liq_norm": n_liq_norm.astype(np.float32),
        "n_liq_observed": n_liq_observed.astype(np.float32),
    }

    # Наблюдаемые вспомогательные цели (выводятся из измеренной PPR — доступны и на реальных
    # данных) и опциональная измеренная CRR(N); плюс синтетические латентные поля (только для
    # диагностики, не для обучения/оценки на реальных данных).
    optional_fields = {
        # наблюдаемые
        "g_obs": "g_obs",
        "risk_proxy": "risk_proxy",
        "crr_obs": "crr_obs",
        "crr_obs_mask": "crr_obs_mask",
        # синтетические латентные (диагностика)
        "r_true": "r_true",
        "z_true": "z_true",
        "g_true": "g_true",
        "uncertainty_proxy": "uncertainty_proxy",
        "crr_mix_true": "crr_mix",
    }
    for split_key, pop_key in optional_fields.items():
        if pop_key in population_dict and population_dict[pop_key] is not None:
            benchmark_arrays[split_key] = population_dict[pop_key][benchmark_idx].astype(np.float32)
    if "g_true" in benchmark_arrays:
        benchmark_arrays["trigger_zone"] = (benchmark_arrays["g_true"] > 0.70).astype(np.float32)

    def make_split(rel_idx: np.ndarray) -> Dict[str, object]:
        """Собрать словарь одной выборки из benchmark-массивов по относительным индексам."""
        split = {key: torch.from_numpy(value[rel_idx]) for key, value in benchmark_arrays.items()}
        split["meta"] = bench_meta.iloc[rel_idx].reset_index(drop=True)
        split["indices"] = rel_idx
        return split

    return {
        "train": make_split(train_rel),
        "val": make_split(val_rel),
        "test": make_split(test_rel),
        "meta": bench_meta,
        "scalers": {
            "static": static_scaler,
            "prefix": prefix_scaler,
            "seq_mean": seq_mean.astype(np.float32),
            "seq_std": seq_std.astype(np.float32),
        },
        "feature_names": {
            "static": population_dict["static_feature_names"],
            "prefix": population_dict["prefix_summary_names"],
            "seq": population_dict["seq_feature_names"],
        },
    }


def iterate_minibatches(
    split: Dict[str, object],
    batch_size: int,
    device: torch.device,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Dict[str, object]]:
    """
    Итерировать мини-батчи по выборке с переносом тензоров на устройство.

    Все тензорные поля выборки нарезаются на батчи; колонка ``meta`` для каждого
    батча формируется как соответствующее подмножество таблицы метаданных.

    :param split: словарь выборки (из :func:`prepare_benchmark_dataset`)
    :param batch_size: размер мини-батча
    :param device: устройство, на которое переносятся тензоры батча
    :param shuffle: перемешивать ли порядок наблюдений
    :param seed: случайное зерно для перемешивания
    :yield: словарь батча с тензорами на ``device`` и под-таблицей ``meta``
    """
    idx = np.arange(split["static"].shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    tensor_keys = [key for key, value in split.items() if isinstance(value, torch.Tensor)]
    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start : start + batch_size]
        batch = {key: split[key][batch_idx].to(device) for key in tensor_keys}
        batch["meta"] = split["meta"].iloc[batch_idx].reset_index(drop=True)
        yield batch
