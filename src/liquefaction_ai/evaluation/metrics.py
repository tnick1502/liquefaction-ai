"""
Метрики качества, агрегаты и утилиты экспериментов оценки.

Модуль собирает выходы моделей по выборке, считает траекторные и вероятностные
метрики (MSE/MAE/RMSE, AUROC/AUPRC/Brier/ECE, покрытие и ширину интервалов),
агрегирует их по группам (тип грунта, режим нагружения), а также предоставляет
инструменты для абляций и out-of-distribution экспериментов и локализацию таблиц.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.data.meta import localize_meta_frame
from liquefaction_ai.data.splits import iterate_minibatches

__all__ = [
    "collect_outputs",
    "resolve_nliq_prediction",
    "expected_calibration_error",
    "safe_binary_metrics",
    "compute_metrics",
    "grouped_metrics",
    "subsample_split",
    "filter_split",
    "run_quick_experiment",
    "is_holdout_region",
    "MODEL_DISPLAY_NAMES",
    "METRIC_COLUMN_TRANSLATIONS",
    "METRIC_COLUMN_EN",
    "localize_model_names_in_df",
    "localize_metric_table",
    "english_metric_table",
    "MetricInfo",
    "METRICS",
    "metrics_catalog",
    "metric_info",
    "list_metrics",
    "rank_by_metric",
]


@dataclass(frozen=True)
class MetricInfo:
    """
    Описание метрики качества для каталога и отбора в grid search.

    :param key: технический ключ метрики (как в выходе ``compute_metrics``)
    :param name: отображаемое имя метрики
    :param description: подробное описание, что измеряет метрика
    :param units: единицы измерения (``–`` для безразмерных)
    :param lower_is_better: True, если меньшее значение лучше
    :param target: целевое значение (если задано, отбор — по близости к нему), иначе None
    :param fmt: формат отображения числа
    """

    key: str
    name: str
    description: str
    units: str
    lower_is_better: bool
    target: Optional[float] = None
    fmt: str = ".4f"


METRICS: Dict[str, "MetricInfo"] = {
    "val_loss": MetricInfo(
        "val_loss", "Validation loss",
        "Mean validation value of the model's training objective; combines all supervised "
        "and physics-informed loss terms. Useful as a generic, always-available selection criterion.",
        "–", lower_is_better=True),
    "Traj_RMSE": MetricInfo(
        "Traj_RMSE", "Trajectory RMSE",
        "Root-mean-square error of the predicted pore-pressure-ratio trajectory PPR(N) over the "
        "observed horizon. Primary measure of how accurately the model reproduces the PPR curve.",
        "–", lower_is_better=True),
    "Traj_MAE": MetricInfo(
        "Traj_MAE", "Trajectory MAE",
        "Mean absolute error of the predicted PPR(N) trajectory; more robust to outliers than RMSE.",
        "–", lower_is_better=True),
    "Traj_MSE": MetricInfo(
        "Traj_MSE", "Trajectory MSE",
        "Mean squared error of the predicted PPR(N) trajectory; penalises large deviations strongly.",
        "–", lower_is_better=True),
    "N_liq_MAE": MetricInfo(
        "N_liq_MAE", "MAE of N_liq",
        "Mean absolute error of the predicted number of cycles to liquefaction N_liq. Directly "
        "reflects timing accuracy of the liquefaction event.",
        "cycles", lower_is_better=True, fmt=".1f"),
    "AUROC": MetricInfo(
        "AUROC", "AUROC",
        "Area under the ROC curve for liquefaction-risk classification; ability to rank liquefying "
        "scenarios above stable ones, independent of the decision threshold.",
        "–", lower_is_better=False),
    "AUPRC": MetricInfo(
        "AUPRC", "AUPRC",
        "Area under the precision–recall curve; classification quality emphasising the positive "
        "(liquefying) class, informative under class imbalance.",
        "–", lower_is_better=False),
    "Brier": MetricInfo(
        "Brier", "Brier score",
        "Mean squared error of the predicted liquefaction-risk probabilities. Lower values mean "
        "sharper and better-calibrated probability estimates.",
        "–", lower_is_better=True),
    "ECE": MetricInfo(
        "ECE", "Expected calibration error",
        "Average absolute gap between predicted confidence and observed liquefaction frequency across "
        "probability bins. Measures how trustworthy the predicted probabilities are.",
        "–", lower_is_better=True),
    "Coverage_90": MetricInfo(
        "Coverage_90", "90% interval coverage",
        "Empirical fraction of true PPR values that fall inside the predicted 90% interval. The "
        "ideal value is 0.90 — both under- and over-coverage are undesirable.",
        "–", lower_is_better=False, target=0.90, fmt=".3f"),
    "Interval_Width_90": MetricInfo(
        "Interval_Width_90", "90% interval width",
        "Mean width of the predicted 90% interval for PPR(N). At fixed coverage, narrower intervals "
        "mean sharper, more informative uncertainty.",
        "–", lower_is_better=True),
}
"""Каталог метрик качества с подробными описаниями (используется ноутбуками и grid search)."""


def metric_info(key: str) -> "MetricInfo":
    """
    Получить описание метрики по её ключу.

    :param key: ключ метрики
    :return: объект :class:`MetricInfo`
    :raises KeyError: если метрика не зарегистрирована
    """
    return METRICS[key]


def list_metrics() -> List[str]:
    """
    Список доступных ключей метрик.

    :return: список ключей метрик из каталога
    """
    return list(METRICS.keys())


def metrics_catalog() -> pd.DataFrame:
    """
    Вернуть таблицу-каталог метрик с описаниями для отображения в ноутбуках.

    :return: DataFrame с колонками Metric/Name/Units/Direction/Target/Description
    """
    rows = []
    for m in METRICS.values():
        direction = "lower is better" if m.lower_is_better else "higher is better"
        if m.target is not None:
            direction = f"target ≈ {m.target}"
        rows.append({"Metric": m.key, "Name": m.name, "Units": m.units,
                     "Direction": direction, "Description": m.description})
    return pd.DataFrame(rows)


def rank_by_metric(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    """
    Отсортировать таблицу результатов по метрике с учётом её направления.

    Учитывает направление из каталога: «меньше — лучше», «больше — лучше» или близость к
    целевому значению (например, для покрытия интервала). Пропуски (NaN) уходят в конец.

    :param df: таблица с колонкой ``metric_key``
    :param metric_key: ключ метрики, по которой ранжируем
    :return: отсортированная копия таблицы (лучшие — сверху), с обновлённым индексом
    """
    info = METRICS.get(metric_key)
    if info is not None and info.target is not None:
        order = (df[metric_key] - info.target).abs().sort_values(na_position="last").index
        return df.loc[order].reset_index(drop=True)
    lower = info.lower_is_better if info is not None else True
    return df.sort_values(metric_key, ascending=lower, na_position="last").reset_index(drop=True)


def collect_outputs(
    model: nn.Module,
    split: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Прогнать модель по всей выборке и собрать тензорные выходы в numpy-массивы.

    :param model: обученная модель с методом ``forward_batch``
    :param split: выборка (из ``prepare_benchmark_dataset``)
    :param config: конфигурация эксперимента (размер батча)
    :param device: устройство инференса
    :return: словарь конкатенированных выходов модели в виде массивов numpy
    """
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in iterate_minibatches(split, config.batch_size, device, shuffle=False):
            outputs = model.forward_batch(batch)
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    collected.setdefault(key, []).append(value.detach().cpu())
    return {key: torch.cat(value, dim=0).numpy() for key, value in collected.items()}


def resolve_nliq_prediction(outputs: Dict[str, np.ndarray], max_cycle_reference: float) -> np.ndarray:
    """
    Извлечь предсказание числа циклов до разжижения N_liq из выходов модели.

    Поддерживает разные представления: прямое ``nliq`` либо нормированные ``nliq_pred``/
    ``nliq_norm`` (последние обратно преобразуются через ``expm1`` и опорное N).

    :param outputs: словарь выходов модели
    :param max_cycle_reference: опорное N для обратной логарифмической нормировки
    :return: массив предсказанных N_liq
    :raises KeyError: если ни одно из поддерживаемых полей не найдено
    """
    if "nliq" in outputs:
        return outputs["nliq"]
    if "nliq_pred" in outputs:
        return np.expm1(outputs["nliq_pred"] * math.log1p(max_cycle_reference))
    if "nliq_norm" in outputs:
        return np.expm1(outputs["nliq_norm"] * math.log1p(max_cycle_reference))
    raise KeyError("В outputs не найдено предсказание для N_liq.")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Ожидаемая ошибка калибровки (Expected Calibration Error, ECE).

    Формула:
        ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|,

    где предсказанные вероятности разбиваются на ``n_bins`` корзин по [0, 1], а для
    каждой корзины сравниваются средняя наблюдаемая частота и средняя уверенность.

    :param y_true: бинарные истинные метки
    :param y_prob: предсказанные вероятности
    :param n_bins: число корзин разбиения по вероятности
    :return: значение ECE (float)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / len(y_true))
    return float(ece)


def safe_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """
    Бинарные метрики качества с защитой от вырожденного случая одного класса.

    Если в ``y_true`` присутствует только один класс, AUROC и AUPRC не определены и
    возвращаются как NaN; Brier-score считается всегда.

    :param y_true: бинарные истинные метки
    :param y_prob: предсказанные вероятности
    :return: кортеж (AUROC, AUPRC, Brier)
    """
    auroc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if np.unique(y_true).size > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)
    return float(auroc), float(auprc), float(brier)


def compute_metrics(
    model_name: str,
    outputs: Dict[str, np.ndarray],
    split: Dict[str, object],
    config: ExperimentConfig,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Вычислить сводные метрики и пообъектную таблицу качества для модели.

    Считаются MAE по N_liq, бинарные метрики риска (AUROC/AUPRC/Brier/ECE) и, при
    наличии траекторной головы, траекторные MSE/MAE/RMSE, а также покрытие и средняя
    ширина 90%-интервала. Дополнительно формируется пообъектная таблица для групповых
    агрегатов и анализа кейсов.

    :param model_name: отображаемое имя модели
    :param outputs: словарь выходов модели (из :func:`collect_outputs`)
    :param split: выборка с истинными значениями и масками
    :param config: конфигурация эксперимента (опорное N)
    :return: кортеж (словарь сводных метрик, пообъектная таблица DataFrame)
    """
    meta_df = split["meta"].copy().reset_index(drop=True)
    y_true = split["label"].cpu().numpy()
    y_prob = outputs["risk_prob"]
    nliq_pred = resolve_nliq_prediction(outputs, config.max_cycle_reference)
    nliq_true = split["n_liq_true"].cpu().numpy()

    sample_df = localize_meta_frame(meta_df.copy()).reset_index(drop=True)
    sample_df["risk_prob_pred"] = y_prob
    sample_df["liq_label"] = y_true
    sample_df["nliq_pred"] = nliq_pred
    sample_df["nliq_abs_err"] = np.abs(nliq_pred - nliq_true)

    metrics: Dict[str, object] = {
        "model": model_name,
        "N_liq_MAE": float(np.mean(np.abs(nliq_pred - nliq_true))),
    }

    auroc, auprc, brier = safe_binary_metrics(y_true, y_prob)
    metrics["AUROC"] = auroc
    metrics["AUPRC"] = auprc
    metrics["Brier"] = brier
    metrics["ECE"] = expected_calibration_error(y_true, y_prob)

    if "traj_mean" in outputs:
        pred = outputs["traj_mean"]
        true = split["r_true"].cpu().numpy()
        mask = split["mask"].cpu().numpy()
        mse = float(np.sum(((pred - true) ** 2) * mask) / np.maximum(mask.sum(), 1.0))
        mae = float(np.sum(np.abs(pred - true) * mask) / np.maximum(mask.sum(), 1.0))
        rmse = float(np.sqrt(mse))
        metrics["Traj_MSE"] = mse
        metrics["Traj_MAE"] = mae
        metrics["Traj_RMSE"] = rmse

        sample_mask_count = np.maximum(mask.sum(axis=1), 1.0)
        sample_df["traj_rmse"] = np.sqrt(np.sum(((pred - true) ** 2) * mask, axis=1) / sample_mask_count)

        if "traj_logvar" in outputs:
            std = np.sqrt(np.exp(outputs["traj_logvar"]))
            lower = pred - 1.64 * std
            upper = pred + 1.64 * std
            coverage = float(np.sum(((true >= lower) & (true <= upper)) * mask) / np.maximum(mask.sum(), 1.0))
            width = float(np.sum((upper - lower) * mask) / np.maximum(mask.sum(), 1.0))
            metrics["Coverage_90"] = coverage
            metrics["Interval_Width_90"] = width
            sample_df["interval_width"] = np.sum((upper - lower) * mask, axis=1) / sample_mask_count
        else:
            metrics["Coverage_90"] = float("nan")
            metrics["Interval_Width_90"] = float("nan")
            sample_df["interval_width"] = np.nan
    else:
        metrics["Traj_MSE"] = float("nan")
        metrics["Traj_MAE"] = float("nan")
        metrics["Traj_RMSE"] = float("nan")
        metrics["Coverage_90"] = float("nan")
        metrics["Interval_Width_90"] = float("nan")
        sample_df["traj_rmse"] = np.nan
        sample_df["interval_width"] = np.nan

    return metrics, sample_df


def grouped_metrics(sample_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Агрегировать пообъектные метрики по заданной группирующей колонке.

    Для каждой группы считаются размер, доля разжижения, средний предсказанный риск,
    средняя абсолютная ошибка N_liq, средний RMSE траектории, средняя ширина интервала
    и групповой AUROC (NaN при единственном классе в группе).

    :param sample_df: пообъектная таблица из :func:`compute_metrics`
    :param group_col: имя колонки группировки (например, ``soil_type_ru``)
    :return: таблица групповых агрегатов, отсортированная по размеру группы
    """
    def safe_group_auc(df: pd.DataFrame) -> float:
        return roc_auc_score(df["liq_label"], df["risk_prob_pred"]) if df["liq_label"].nunique() > 1 else float("nan")

    grouped = (
        sample_df.groupby(group_col)
        .agg(
            samples=("liq_label", "size"),
            liquefaction_rate=("liq_label", "mean"),
            mean_risk_pred=("risk_prob_pred", "mean"),
            mean_nliq_abs_err=("nliq_abs_err", "mean"),
            mean_traj_rmse=("traj_rmse", "mean"),
            mean_interval_width=("interval_width", "mean"),
        )
        .reset_index()
    )
    grouped["AUROC"] = [safe_group_auc(sample_df[sample_df[group_col] == val]) for val in grouped[group_col]]
    return grouped.sort_values("samples", ascending=False)


def subsample_split(split: Dict[str, object], max_size: int, seed: int) -> Dict[str, object]:
    """
    Случайно проредить выборку до заданного максимального размера.

    Используется для облегчения абляционных экспериментов. Если выборка уже не больше
    ``max_size``, возвращается без изменений.

    :param split: исходная выборка
    :param max_size: максимальный размер результата
    :param seed: случайное зерно для воспроизводимого отбора
    :return: прореженная выборка (с согласованными тензорами и таблицей ``meta``)
    """
    current_size = split["static"].shape[0]
    if current_size <= max_size:
        return split
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(current_size, size=max_size, replace=False))
    new_split: Dict[str, object] = {}
    for key, value in split.items():
        if isinstance(value, torch.Tensor):
            new_split[key] = value[idx]
        elif key == "meta":
            new_split[key] = value.iloc[idx].reset_index(drop=True)
        else:
            new_split[key] = value
    return new_split


def filter_split(split: Dict[str, object], mask: np.ndarray) -> Dict[str, object]:
    """
    Отфильтровать выборку по булевой маске наблюдений.

    Используется для построения out-of-distribution подвыборок (удержанный регион
    грунтов, невидимый режим нагружения и т.п.).

    :param split: исходная выборка
    :param mask: булева маска длины числа наблюдений выборки
    :return: отфильтрованная выборка (с согласованными тензорами и таблицей ``meta``)
    """
    idx = np.where(mask)[0]
    filtered: Dict[str, object] = {}
    for key, value in split.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value[idx]
        elif key == "meta":
            filtered[key] = value.iloc[idx].reset_index(drop=True)
        else:
            filtered[key] = value
    return filtered


def run_quick_experiment(
    model_name: str,
    model: nn.Module,
    train_split: Dict[str, object],
    val_split: Dict[str, object],
    test_split: Dict[str, object],
    epochs: int,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    """
    Быстро обучить модель и вернуть её метрики на тестовой выборке.

    Удобная обёртка для абляций и OOD-экспериментов: обучает модель на коротком числе
    эпох, собирает выходы и считает сводные метрики.

    :param model_name: отображаемое имя эксперимента
    :param model: необученная модель
    :param train_split: обучающая выборка
    :param val_split: валидационная выборка
    :param test_split: тестовая выборка
    :param epochs: число эпох обучения
    :param config: конфигурация эксперимента
    :param device: устройство
    :return: словарь сводных метрик на тестовой выборке
    """
    from liquefaction_ai.training.loop import train_model

    model = model.to(device)
    model, _ = train_model(model, train_split, val_split, epochs=epochs, model_name=model_name, config=config, device=device)
    outputs = collect_outputs(model, test_split, config, device)
    metrics, _ = compute_metrics(model_name, outputs, test_split, config)
    return metrics


def is_holdout_region(meta_df: pd.DataFrame, e_threshold: float, vs_threshold: float) -> np.ndarray:
    """
    Определить наблюдения удержанного региона грунтов (рыхлые слабые грунты).

    Регион задаётся как высокий коэффициент пористости и низкая скорость волн:
    ``e ≥ e_threshold`` и ``V_s ≤ vs_threshold``.

    :param meta_df: таблица метаданных
    :param e_threshold: порог по коэффициенту пористости
    :param vs_threshold: порог по скорости поперечных волн
    :return: булева маска принадлежности к удержанному региону
    """
    return (meta_df["e"].to_numpy() >= e_threshold) & (meta_df["V_s"].to_numpy() <= vs_threshold)


MODEL_DISPLAY_NAMES = {
    "MLP_risk": "MLP риск",
    "GRU": "GRU",
    "TCN": "TCN",
    "DPI-Flow": "DPI-Flow",
    "EVT-NeuralSSM": "EVT-NeuralSSM",
    "DPI-Flow без калибровки": "DPI-Flow без калибровки",
    "DPI-Flow без вероятностной головы": "DPI-Flow без вероятностной головы",
    "DPI-Flow без ODE-слоя": "DPI-Flow без ODE-слоя",
    "EVT-NeuralSSM без trigger-head": "EVT-NeuralSSM без trigger-head",
    "EVT-NeuralSSM без постсобытийной динамики": "EVT-NeuralSSM без постсобытийной динамики",
    "EVT-NeuralSSM без CRR-уравнения повреждения": "EVT-NeuralSSM без CRR-уравнения повреждения",
    "TCN: короткий->длинный": "TCN: короткий→длинный",
    "DPI-Flow: короткий->длинный": "DPI-Flow: короткий→длинный",
    "EVT-NeuralSSM: короткий->длинный": "EVT-NeuralSSM: короткий→длинный",
    "TCN: holdout": "TCN: удержанный набор",
    "DPI-Flow: holdout": "DPI-Flow: удержанный набор",
    "EVT-NeuralSSM: holdout": "EVT-NeuralSSM: удержанный набор",
    "TCN: удержанный набор": "TCN: удержанный набор",
    "DPI-Flow: удержанный набор": "DPI-Flow: удержанный набор",
    "EVT-NeuralSSM: удержанный набор": "EVT-NeuralSSM: удержанный набор",
}
"""Соответствия технических имён моделей их отображаемым подписям."""

METRIC_COLUMN_TRANSLATIONS = {
    "model": "модель",
    "N_liq_MAE": "MAE по N_liq",
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Brier": "Brier",
    "ECE": "ECE",
    "Traj_MSE": "MSE траектории",
    "Traj_MAE": "MAE траектории",
    "Traj_RMSE": "RMSE траектории",
    "Coverage_90": "покрытие_90",
    "Interval_Width_90": "ширина_интервала_90",
    "samples": "число_образцов",
    "liquefaction_rate": "доля_разжижения",
    "mean_risk_pred": "средний_предсказанный_риск",
    "mean_nliq_abs_err": "средняя_абс_ошибка_N_liq",
    "mean_traj_rmse": "средний_RMSE_траектории",
    "mean_interval_width": "средняя_ширина_интервала",
}
"""Соответствия технических имён колонок метрик их русским подписям."""


def localize_model_names_in_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Локализовать колонку ``model`` таблицы метрик русскими подписями.

    :param df: таблица метрик с технической колонкой ``model``
    :return: копия таблицы с локализованными именами моделей и суффиксами экспериментов
    """
    localized = df.copy()
    if "model" in localized.columns:
        localized["model"] = localized["model"].map(MODEL_DISPLAY_NAMES).fillna(localized["model"])
        localized["model"] = localized["model"].str.replace(": holdout", ": удержанный набор", regex=False)
        localized["model"] = localized["model"].str.replace(" | region", " | удержанный_регион", regex=False)
        localized["model"] = localized["model"].str.replace(" | unseen_regime", " | невидимый_режим", regex=False)
    return localized


def localize_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полностью локализовать таблицу метрик: имена моделей и заголовки колонок.

    :param df: таблица метрик с техническими именами
    :return: копия таблицы с русскими именами моделей и заголовками колонок
    """
    localized = localize_model_names_in_df(df)
    rename_map = {key: value for key, value in METRIC_COLUMN_TRANSLATIONS.items() if key in localized.columns}
    return localized.rename(columns=rename_map)


METRIC_COLUMN_EN = {
    "model": "Model",
    "N_liq_MAE": "MAE N_liq (cycles)",
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Brier": "Brier",
    "ECE": "ECE",
    "Traj_MSE": "Trajectory MSE",
    "Traj_MAE": "Trajectory MAE",
    "Traj_RMSE": "Trajectory RMSE",
    "Coverage_90": "Coverage@90%",
    "Interval_Width_90": "Interval width@90%",
    "samples": "N samples",
    "liquefaction_rate": "Liquefaction rate",
    "mean_risk_pred": "Mean predicted risk",
    "mean_nliq_abs_err": "Mean |ΔN_liq| (cycles)",
    "mean_traj_rmse": "Mean trajectory RMSE",
    "mean_interval_width": "Mean interval width",
}
"""Соответствия технических имён колонок метрик их англоязычным публикационным подписям."""


def english_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переименовать колонки таблицы метрик в англоязычные публикационные подписи.

    Имена моделей не изменяются (в ноутбуках они задаются на английском). Подходит для
    подготовки таблиц результатов к публикации.

    :param df: таблица метрик с техническими именами колонок
    :return: копия таблицы с англоязычными заголовками колонок
    """
    rename_map = {key: value for key, value in METRIC_COLUMN_EN.items() if key in df.columns}
    return df.rename(columns=rename_map)
