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
from scipy.special import erf
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from liquefaction_ai.config import ExperimentConfig, set_global_seed
from liquefaction_ai.data.meta import localize_meta_frame
from liquefaction_ai.data.splits import iterate_minibatches

__all__ = [
    "collect_outputs",
    "resolve_nliq_prediction",
    "expected_calibration_error",
    "safe_binary_metrics",
    "fit_temperature",
    "apply_temperature",
    "compute_metrics",
    "grouped_metrics",
    "subsample_split",
    "filter_split",
    "stress_split",
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
        "whole observed horizon, including the conditioning prefix when prefix observations are "
        "available. This is a reconstruction/continuation metric; use Traj_RMSE_continuation for "
        "the strictly post-prefix forecast portion.",
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
        "Censored mean absolute error of the predicted number of cycles to liquefaction N_liq. "
        "Liquefied samples use absolute error; non-liquefied stabilized samples penalise only "
        "too-early predictions; unfinished non-liquefied samples are excluded.",
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
    "N_liq_RMSE": MetricInfo(
        "N_liq_RMSE", "RMSE of N_liq",
        "Censored root-mean-square error of predicted cycles to liquefaction N_liq; uses the same "
        "right-censoring protocol as N_liq_MAE.",
        "cycles", lower_is_better=True, fmt=".1f"),
    "N_liq_logMAE": MetricInfo(
        "N_liq_logMAE", "log-MAE of N_liq",
        "Censored mean absolute error of N_liq in log1p scale. Cycles span orders of magnitude "
        "and scale non-linearly, so log-error reflects relative timing accuracy more fairly than raw cycles.",
        "log-cycles", lower_is_better=True, fmt=".3f"),
    "N_liq_logRMSE": MetricInfo(
        "N_liq_logRMSE", "log-RMSE of N_liq",
        "Censored root-mean-square error of N_liq in log1p scale; relative timing error robust to the wide dynamic range of cycles.",
        "log-cycles", lower_is_better=True, fmt=".3f"),
    "Physics_Violation_Rate": MetricInfo(
        "Physics_Violation_Rate", "Monotonicity-assumption violation rate",
        "Fraction of predicted PPR(N) curves that break the undrained monotonic-accumulation modelling "
        "assumption: ru either decreases or leaves [0, 1.05]. This is the assumption adopted here (and "
        "enforced structurally for the proposed models through monotone projection) for undrained cyclic "
        "loading, NOT a universal physical law. A zero value for structurally-constrained models is a "
        "feasibility guarantee, not independent empirical evidence of better physics.",
        "–", lower_is_better=True, target=0.0, fmt=".3f"),
    "Coverage_80": MetricInfo(
        "Coverage_80", "80% interval coverage",
        "Empirical fraction of true PPR values inside the predicted 80% interval (ideal = 0.80).",
        "–", lower_is_better=False, target=0.80, fmt=".3f"),
    "Coverage_95": MetricInfo(
        "Coverage_95", "95% interval coverage",
        "Empirical fraction of true PPR values inside the predicted 95% interval (ideal = 0.95).",
        "–", lower_is_better=False, target=0.95, fmt=".3f"),
    "Calibration_Error": MetricInfo(
        "Calibration_Error", "Calibration error",
        "Mean absolute gap between empirical and nominal interval coverage across the 80/90/95% levels. "
        "Lower means more trustworthy uncertainty — a key strength of the probabilistic physics models.",
        "–", lower_is_better=True, target=0.0, fmt=".3f"),
    "Traj_NLL": MetricInfo(
        "Traj_NLL", "Trajectory NLL",
        "Gaussian negative log-likelihood of the observed PPR under the predicted mean/variance. A proper "
        "scoring rule rewarding both accuracy and well-calibrated uncertainty.",
        "nats", lower_is_better=True, fmt=".3f"),
    "Traj_CRPS": MetricInfo(
        "Traj_CRPS", "Trajectory CRPS",
        "Continuous ranked probability score of the predicted PPR distribution. A proper scoring rule; "
        "lower values reward sharp, calibrated probabilistic trajectories.",
        "–", lower_is_better=True, fmt=".4f"),
    "CRR_RMSE": MetricInfo(
        "CRR_RMSE", "CRR-curve RMSE",
        "RMSE between the predicted cyclic-resistance curve CRR(N) and the measured liquefaction-potential "
        "curve, where available. Only the physics-structured models (DPI-Flow, EVT-NeuralSSM, DPI-EVT) output a CRR "
        "boundary at all — a capability black-box baselines lack.",
        "–", lower_is_better=True, fmt=".4f"),
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
    # Единый сид проекта делает вероятностную оценку (MC-сэмплирование θ) детерминированной.
    # Состояние RNG сохраняется и восстанавливается, чтобы оценка не сбивала обучение
    # (например, при трекинге метрик по эпохам).
    import random as _random
    _torch_state = torch.get_rng_state()
    _np_state = np.random.get_state()
    _py_state = _random.getstate()
    set_global_seed(config.seed)
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {}
    try:
        with torch.no_grad():
            for batch in iterate_minibatches(split, config.batch_size, device, shuffle=False):
                outputs = model.forward_batch(batch)
                for key, value in outputs.items():
                    if torch.is_tensor(value):
                        collected.setdefault(key, []).append(value.detach().cpu())
    finally:
        torch.set_rng_state(_torch_state)
        np.random.set_state(_np_state)
        _random.setstate(_py_state)
    return {key: torch.cat(value, dim=0).numpy() for key, value in collected.items()}


def fit_interval_scale(model, val_split: Dict[str, object], config: ExperimentConfig,
                       device: torch.device, level: float = 0.90) -> float:
    """
    Пост-hoc конформная калибровка интервалов: подобрать скаляр s, выравнивающий покрытие к номиналу.

    На валидации ищется множитель ``s`` для стандартного отклонения, при котором эмпирическое
    покрытие интервала уровня ``level`` совпадает с номиналом. Результат записывается в буфер
    модели ``calib_log_scale`` (= ln s) и автоматически применяется в ``forward_batch``.

    :param model: модель с буфером ``calib_log_scale`` и траекторной головой
    :param val_split: валидационная выборка
    :param config: конфигурация (размер батча)
    :param device: устройство
    :param level: целевой уровень покрытия (0.90 по умолчанию)
    :return: подобранный множитель s (1.0, если модель не поддерживает калибровку)
    """
    if not hasattr(model, "calib_log_scale"):
        return 1.0
    with torch.no_grad():
        model.calib_log_scale.zero_()
    out = collect_outputs(model, val_split, config, device)
    if "traj_logvar" not in out:
        return 1.0
    pred = out["traj_mean"]; std = np.sqrt(np.exp(out["traj_logvar"]))
    true = val_split["r_obs"].cpu().numpy(); mask = val_split["mask"].cpu().numpy()
    z = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600}.get(level, 1.6449)
    denom = max(mask.sum(), 1.0)

    def coverage(s):
        lo = pred - z * s * std; hi = pred + z * s * std
        return float(np.sum(((true >= lo) & (true <= hi)) * mask) / denom)

    grid = np.linspace(0.3, 4.0, 75)
    best = min(grid, key=lambda s: abs(coverage(s) - level))
    with torch.no_grad():
        model.calib_log_scale.fill_(float(np.log(best)))
    return float(best)


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


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """
    Подобрать температуру калибровки T, минимизируя BCE на валидации.

    Post-hoc калибровка Платта/температурой: масштабирование логитов риска на скаляр T
    приближает предсказанные вероятности к наблюдаемым частотам, улучшая Brier/ECE без
    изменения ранжирования (AUROC сохраняется).

    :param logits: логиты риска на валидации (если есть только вероятности — передайте logit(p))
    :param labels: бинарные истинные метки разжижения
    :param max_iter: максимум итераций оптимизатора
    :return: оптимальная температура T > 0
    """
    logit = torch.tensor(np.asarray(logits, dtype=np.float64))
    target = torch.tensor(np.asarray(labels, dtype=np.float64))
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.2, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit / torch.exp(log_t), target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).detach().item())


def apply_temperature(risk_prob: np.ndarray, temperature: float) -> np.ndarray:
    """
    Применить температуру калибровки к вероятностям риска.

    :param risk_prob: исходные вероятности риска в (0, 1)
    :param temperature: температура T > 0 (из :func:`fit_temperature`)
    :return: откалиброванные вероятности sigmoid(logit(p) / T)
    """
    p = np.clip(np.asarray(risk_prob, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-logit / max(temperature, 1e-6)))


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

    # Единый цензур-протокол N_liq (как при обучении): образцы без наблюдаемого терминала N_liq
    # (3-й режим — нет разжижения и нет стабилизации, маска n_liq_observed==0) исключаются
    # из ВСЕХ ошибок N_liq — агрегатных, пообъектных и групповых (OOD), иначе метрика штрафовала бы
    # за «ошибку» по точке, которой нет. Для исключения используется маска obs.
    if "n_liq_observed" in split:
        obs = split["n_liq_observed"].cpu().numpy() > 0.5
    else:
        obs = np.ones_like(nliq_true, dtype=bool)
    obs_any = bool(obs.any())

    sample_df = localize_meta_frame(meta_df.copy()).reset_index(drop=True)
    sample_df["risk_prob_pred"] = y_prob
    sample_df["liq_label"] = y_true
    sample_df["nliq_pred"] = nliq_pred

    nliq_err = nliq_pred - nliq_true
    log_pred = np.log1p(np.maximum(nliq_pred, 0.0))
    log_true = np.log1p(np.maximum(nliq_true, 0.0))
    log_err = log_pred - log_true
    liq = y_true > 0.5
    # Censored timing error: exact liquefaction is absolute error; stabilized non-liquefaction
    # is right-censored, so only too-early predictions are errors; unfinished non-liq is masked out.
    nliq_cens_err = np.where(liq, np.abs(nliq_err), np.maximum(nliq_true - nliq_pred, 0.0))
    log_cens_err = np.where(liq, np.abs(log_err), np.maximum(log_true - log_pred, 0.0))
    # Пообъектные ошибки: NaN для неучтённых образцов → groupby.mean() их автоматически пропустит,
    # поэтому групповые (OOD) средние считаются по той же маске, что и агрегат.
    sample_df["nliq_abs_err"] = np.where(obs, nliq_cens_err, np.nan)
    sample_df["nliq_log_err"] = np.where(obs, log_cens_err, np.nan)
    sample_df["n_liq_observed"] = obs.astype(float)

    if obs_any:
        nliq_mae = float(np.mean(nliq_cens_err[obs]))
        nliq_rmse = float(np.sqrt(np.mean(nliq_cens_err[obs] ** 2)))
        nliq_logmae = float(np.mean(log_cens_err[obs]))
        nliq_logrmse = float(np.sqrt(np.mean(log_cens_err[obs] ** 2)))
    else:
        nliq_mae = nliq_rmse = nliq_logmae = nliq_logrmse = float("nan")

    metrics: Dict[str, object] = {
        "model": model_name,
        "N_liq_MAE": nliq_mae,
        "N_liq_RMSE": nliq_rmse,
        "N_liq_logMAE": nliq_logmae,
        "N_liq_logRMSE": nliq_logrmse,
        "N_liq_n_observed": int(obs.sum()),
    }

    auroc, auprc, brier = safe_binary_metrics(y_true, y_prob)
    metrics["AUROC"] = auroc
    metrics["AUPRC"] = auprc
    metrics["Brier"] = brier
    metrics["ECE"] = expected_calibration_error(y_true, y_prob)

    if "traj_mean" in outputs:
        pred = outputs["traj_mean"]
        # Эталон траектории — измеренное поровое давление (как в реальном опыте), а не
        # синтетически «чистая» кривая. Для реальных данных доступно только измеренное.
        true = split["r_obs"].cpu().numpy()
        mask = split["mask"].cpu().numpy()
        mse = float(np.sum(((pred - true) ** 2) * mask) / np.maximum(mask.sum(), 1.0))
        mae = float(np.sum(np.abs(pred - true) * mask) / np.maximum(mask.sum(), 1.0))
        rmse = float(np.sqrt(mse))
        metrics["Traj_MSE"] = mse
        metrics["Traj_MAE"] = mae
        metrics["Traj_RMSE"] = rmse

        sample_mask_count = np.maximum(mask.sum(axis=1), 1.0)
        sample_df["traj_rmse"] = np.sqrt(np.sum(((pred - true) ** 2) * mask, axis=1) / sample_mask_count)

        # Prefix-conditioned task: models observe the early PPR prefix as context. Report the full
        # reconstruction/continuation error above, but make the strictly post-prefix forecast error
        # explicit so it cannot be mistaken for a pure-from-zero trajectory forecast.
        if "prefix_mask" in split:
            prefix_mask = split["prefix_mask"].cpu().numpy()
            continuation_mask = mask * (1.0 - np.minimum(prefix_mask, 1.0))
        else:
            continuation_mask = mask
        cont_denom = np.maximum(continuation_mask.sum(), 1.0)
        cont_mse = float(np.sum(((pred - true) ** 2) * continuation_mask) / cont_denom)
        cont_mae = float(np.sum(np.abs(pred - true) * continuation_mask) / cont_denom)
        cont_rmse = float(np.sqrt(cont_mse))
        metrics["Traj_MSE_continuation"] = cont_mse
        metrics["Traj_MAE_continuation"] = cont_mae
        metrics["Traj_RMSE_continuation"] = cont_rmse
        sample_cont_count = np.maximum(continuation_mask.sum(axis=1), 1.0)
        sample_df["traj_rmse_continuation"] = np.sqrt(
            np.sum(((pred - true) ** 2) * continuation_mask, axis=1) / sample_cont_count
        )

        # --- Траекторная ошибка по трём СОСТОЯНИЯМ ОПЫТА (а не по типу воздействия) ---
        # Состояния: разжижение (liq_label==1); нет разжижения + стабилизация (obs==1);
        # нет разжижения + нет стабилизации (obs==0). Balanced = макро-среднее по
        # присутствующим состояниям (нечувствительно к дисбалансу классов и не даёт модели
        # «спрятать» провал режима за счёт лёгкого большинства); worst = худшее состояние
        # (используется gate-ом компетентности в P³, чтобы коллапс целого режима не проходил).
        _se = ((pred - true) ** 2) * mask
        _states = {"liq": y_true > 0.5,
                   "stab": (y_true < 0.5) & obs,
                   "nostab": (y_true < 0.5) & (~obs)}
        _present = []
        for _nm, _m in _states.items():
            if int(_m.sum()) > 0:
                _v = float(np.sqrt(_se[_m].sum() / np.maximum(mask[_m].sum(), 1.0)))
            else:
                _v = float("nan")
            metrics[f"Traj_RMSE_{_nm}"] = _v
            if _v == _v:
                _present.append(_v)
        metrics["Traj_RMSE_balanced"] = float(np.mean(_present)) if _present else float("nan")
        metrics["Traj_RMSE_worst"] = float(np.max(_present)) if _present else float("nan")

        _se_cont = ((pred - true) ** 2) * continuation_mask
        _present_cont = []
        for _nm, _m in _states.items():
            if int(_m.sum()) > 0 and float(continuation_mask[_m].sum()) > 0:
                _v = float(np.sqrt(_se_cont[_m].sum() / np.maximum(continuation_mask[_m].sum(), 1.0)))
            else:
                _v = float("nan")
            metrics[f"Traj_RMSE_continuation_{_nm}"] = _v
            if _v == _v:
                _present_cont.append(_v)
        metrics["Traj_RMSE_continuation_balanced"] = (
            float(np.mean(_present_cont)) if _present_cont else float("nan")
        )
        metrics["Traj_RMSE_continuation_worst"] = (
            float(np.max(_present_cont)) if _present_cont else float("nan")
        )

        # Физические нарушения: доля предсказаний с «невозможной» кривой PPR(N) — заметно
        # убывающей (ru должна монотонно расти) или выходящей за физические границы [0, 1.05].
        diffs = pred[:, 1:] - pred[:, :-1]
        decreasing = ((diffs < -0.02) * mask[:, 1:]).sum(axis=1) > 0
        out_of_bounds = (((pred > 1.05) | (pred < -0.02)) * mask).sum(axis=1) > 0
        violation = decreasing | out_of_bounds
        metrics["Physics_Violation_Rate"] = float(violation.mean())
        sample_df["physics_violation"] = violation.astype(float)

        if "traj_logvar" in outputs:
            std = np.maximum(np.sqrt(np.exp(outputs["traj_logvar"])), 1e-6)
            # Калибровка: эмпирическое покрытие интервалов на нескольких уровнях
            cov_gaps = []
            for level, z in [(80, 1.2816), (90, 1.6449), (95, 1.9600)]:
                lower = pred - z * std
                upper = pred + z * std
                cov = float(np.sum(((true >= lower) & (true <= upper)) * mask) / np.maximum(mask.sum(), 1.0))
                width = float(np.sum((upper - lower) * mask) / np.maximum(mask.sum(), 1.0))
                metrics[f"Coverage_{level}"] = cov
                metrics[f"Interval_Width_{level}"] = width
                cov_gaps.append(abs(cov - level / 100.0))
            # Сводная ошибка калибровки интервалов (среднее |покрытие − номинал| по уровням)
            metrics["Calibration_Error"] = float(np.mean(cov_gaps))
            # Гауссовская NLL — собственно правило (proper scoring) для вероятностного прогноза
            nll = 0.5 * (np.log(2 * np.pi * std ** 2) + ((true - pred) ** 2) / (std ** 2))
            metrics["Traj_NLL"] = float(np.sum(nll * mask) / np.maximum(mask.sum(), 1.0))
            # CRPS для гауссовского предиктива (награждает калиброванную остроту)
            z0 = (true - pred) / std
            phi = np.exp(-0.5 * z0 ** 2) / np.sqrt(2 * np.pi)
            Phi = 0.5 * (1.0 + erf(z0 / np.sqrt(2.0)))
            crps = std * (z0 * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
            metrics["Traj_CRPS"] = float(np.sum(crps * mask) / np.maximum(mask.sum(), 1.0))
            std90 = 1.6449 * std
            sample_df["interval_width"] = np.sum((2 * std90) * mask, axis=1) / sample_mask_count
        else:
            for level in (80, 90, 95):
                metrics[f"Coverage_{level}"] = float("nan")
                metrics[f"Interval_Width_{level}"] = float("nan")
            metrics["Calibration_Error"] = float("nan")
            metrics["Traj_NLL"] = float("nan")
            metrics["Traj_CRPS"] = float("nan")
            sample_df["interval_width"] = np.nan
    else:
        metrics["Traj_MSE"] = float("nan")
        metrics["Traj_MAE"] = float("nan")
        metrics["Traj_RMSE"] = float("nan")
        metrics["Traj_MSE_continuation"] = float("nan")
        metrics["Traj_MAE_continuation"] = float("nan")
        metrics["Traj_RMSE_continuation"] = float("nan")
        metrics["Physics_Violation_Rate"] = float("nan")
        for level in (80, 90, 95):
            metrics[f"Coverage_{level}"] = float("nan")
            metrics[f"Interval_Width_{level}"] = float("nan")
        metrics["Calibration_Error"] = float("nan")
        metrics["Traj_NLL"] = float("nan")
        metrics["Traj_CRPS"] = float("nan")
        sample_df["traj_rmse"] = np.nan
        sample_df["traj_rmse_continuation"] = np.nan
        sample_df["interval_width"] = np.nan
        sample_df["physics_violation"] = np.nan

    # Восстановление границы CRR(N): уникальная способность физически-структурированных моделей
    # (DPI-Flow / EVT-NeuralSSM / DPI-EVT). Сравнение с измеренной кривой потенциала разжижения, где она есть.
    crr_rmse = float("nan")
    n_crr_test = 0
    n_crr_objects = 0
    if "crr" in outputs and "crr_obs" in split and "crr_obs_mask" in split:
        crr_pred = outputs["crr"]
        crr_true = split["crr_obs"].cpu().numpy()
        crr_m = split["crr_obs_mask"].cpu().numpy()
        tmask = split["mask"].cpu().numpy()
        per = np.sqrt(np.sum(((crr_pred - crr_true) ** 2) * tmask, axis=1) / np.maximum(tmask.sum(axis=1), 1.0))
        sel = crr_m > 0
        n_crr_test = int(sel.sum())
        # Сколько разных объектов/площадок стоят за измеренной CRR — важно для честной интерпретации:
        # CRR-метрика опирается на малую выборку из немногих объектов (раскрываем это в таблицах).
        if "object" in sample_df.columns and n_crr_test > 0:
            n_crr_objects = int(pd.Series(sample_df["object"].to_numpy()[sel]).nunique())
        if sel.any():
            crr_rmse = float(per[sel].mean())
    metrics["CRR_RMSE"] = crr_rmse
    metrics["N_CRR_test"] = n_crr_test            # число тест-образцов с измеренной CRR (мощность выборки)
    metrics["N_CRR_objects"] = n_crr_objects      # число объектов/площадок за этими образцами
    metrics["Produces_CRR"] = "crr" in outputs

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
            n_nliq_observed=("n_liq_observed", "sum"),
            mean_nliq_abs_err=("nliq_abs_err", "mean"),
            mean_nliq_log_err=("nliq_log_err", "mean"),
            mean_traj_rmse=("traj_rmse", "mean"),
            mean_interval_width=("interval_width", "mean"),
            physics_violation_rate=("physics_violation", "mean"),
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


def stress_split(
    split: Dict[str, object],
    *,
    no_prefix: bool = False,
    drop_derived_aux: bool = False,
) -> Dict[str, object]:
    """
    Сформировать копию выборки для стресс-протоколов без переупаковки артефакта.

    ``no_prefix=True`` зануляет наблюдаемый префикс PPR и его summary-признаки. Это проверяет,
    насколько результат зависит от постановки "forecast from observed prefix" и не должен
    смешиваться с основной prefix-conditioned таблицей.

    ``drop_derived_aux=True`` удаляет ``g_obs`` и ``risk_proxy`` — auxiliary targets, выводимые из
    полной PPR-кривой. Используйте это при переобучении с ``config.use_observed_aux_loss=False``
    для стресс-теста "no-derived-threshold auxiliaries". На test-time модели не читают эти поля,
    но явное удаление защищает от случайного использования в новых экспериментах.

    :param split: выборка из :func:`prepare_benchmark_dataset`
    :param no_prefix: убрать prefix-conditioning признаки
    :param drop_derived_aux: удалить производные auxiliary цели
    :return: копия split с теми же meta/indices и изменёнными тензорными полями
    """
    out: Dict[str, object] = {}
    for key, value in split.items():
        out[key] = value.clone() if torch.is_tensor(value) else value
    if no_prefix:
        for key in ("prefix_summary", "prefix_summary_raw", "prefix_obs", "prefix_mask"):
            if key in out and torch.is_tensor(out[key]):
                out[key] = torch.zeros_like(out[key])
        # В текущем артефакте последние два sequence-канала — prefix_obs/prefix_mask. Зануляем их
        # как дополнительную защиту для моделей, читающих весь seq_in.
        for key in ("seq_in", "seq_in_raw"):
            if key in out and torch.is_tensor(out[key]) and out[key].ndim == 3 and out[key].shape[-1] >= 2:
                value = out[key].clone()
                value[..., -2:] = 0.0
                out[key] = value
    if drop_derived_aux:
        for key in ("g_obs", "risk_proxy"):
            out.pop(key, None)
    return out


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
    "Traj_MSE_continuation": "MSE прогноза после префикса",
    "Traj_MAE_continuation": "MAE прогноза после префикса",
    "Traj_RMSE_continuation": "RMSE прогноза после префикса",
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
    "N_liq_RMSE": "RMSE N_liq (cycles)",
    "N_liq_logMAE": "log-MAE N_liq",
    "N_liq_logRMSE": "log-RMSE N_liq",
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Brier": "Brier",
    "ECE": "ECE",
    "Traj_MSE": "Trajectory MSE",
    "Traj_MAE": "Trajectory MAE",
    "Traj_RMSE": "Trajectory RMSE",
    "Traj_MSE_continuation": "Post-prefix MSE",
    "Traj_MAE_continuation": "Post-prefix MAE",
    "Traj_RMSE_continuation": "Post-prefix RMSE",
    "Physics_Violation_Rate": "Physics violations",
    "Calibration_Error": "Calibration error",
    "Traj_NLL": "Trajectory NLL",
    "Traj_CRPS": "Trajectory CRPS",
    "CRR_RMSE": "CRR-curve RMSE",
    "Produces_CRR": "Produces CRR",
    "Coverage_80": "Coverage@80%",
    "Coverage_90": "Coverage@90%",
    "Coverage_95": "Coverage@95%",
    "Interval_Width_80": "Interval width@80%",
    "Interval_Width_90": "Interval width@90%",
    "Interval_Width_95": "Interval width@95%",
    "samples": "N samples",
    "liquefaction_rate": "Liquefaction rate",
    "mean_risk_pred": "Mean predicted risk",
    "mean_nliq_abs_err": "Mean |ΔN_liq| (cycles)",
    "mean_nliq_log_err": "Mean log-error N_liq",
    "mean_traj_rmse": "Mean trajectory RMSE",
    "mean_interval_width": "Mean interval width",
    "physics_violation_rate": "Physics violations",
    "Traj_RMSE_liq": "Trajectory RMSE (liquefied)",
    "Traj_RMSE_stab": "Trajectory RMSE (no-liq, stabilized)",
    "Traj_RMSE_nostab": "Trajectory RMSE (no-liq, not stabilized)",
    "Traj_RMSE_balanced": "Trajectory RMSE (balanced over states)",
    "Traj_RMSE_worst": "Trajectory RMSE (worst state)",
    "Traj_RMSE_continuation_liq": "Post-prefix RMSE (liquefied)",
    "Traj_RMSE_continuation_stab": "Post-prefix RMSE (no-liq, stabilized)",
    "Traj_RMSE_continuation_nostab": "Post-prefix RMSE (no-liq, not stabilized)",
    "Traj_RMSE_continuation_balanced": "Post-prefix RMSE (balanced over states)",
    "Traj_RMSE_continuation_worst": "Post-prefix RMSE (worst state)",
}
"""Соответствия технических имён колонок метрик их англоязычным публикационным подписям."""


def _register_state_traj_metrics() -> None:
    """Зарегистрировать в каталоге траекторные метрики по трём состояниям опыта."""
    _defs = {
        "Traj_RMSE_liq": ("Trajectory RMSE (liquefied)",
            "PPR(N) RMSE restricted to liquefied experiments (liq_label=1)."),
        "Traj_RMSE_stab": ("Trajectory RMSE (no-liq, stabilized)",
            "PPR(N) RMSE on non-liquefying experiments whose pore pressure stabilized "
            "(observable censored terminal)."),
        "Traj_RMSE_nostab": ("Trajectory RMSE (no-liq, not stabilized)",
            "PPR(N) RMSE on non-liquefying experiments that neither liquefied nor stabilized."),
        "Traj_RMSE_balanced": ("Trajectory RMSE (balanced)",
            "Macro-average of per-experiment-state PPR(N) RMSE over the three observed states. "
            "Insensitive to class imbalance: a model cannot hide a collapsed regime behind an easy "
            "majority. Used as the trajectory component of the core P³ score."),
        "Traj_RMSE_worst": ("Trajectory RMSE (worst state)",
            "Worst (maximum) per-state PPR(N) RMSE across the three experiment states. Used by the "
            "P³ competence gate to exclude models that collapse on an entire regime."),
        "Traj_RMSE_continuation": ("Post-prefix trajectory RMSE",
            "RMSE of PPR(N) only after the observed conditioning prefix. This is the primary "
            "trajectory metric for claims about forecasting/continuation rather than reconstruction."),
        "Traj_RMSE_continuation_liq": ("Post-prefix RMSE (liquefied)",
            "Post-prefix PPR(N) RMSE restricted to liquefied experiments."),
        "Traj_RMSE_continuation_stab": ("Post-prefix RMSE (no-liq, stabilized)",
            "Post-prefix PPR(N) RMSE on stabilized non-liquefying experiments."),
        "Traj_RMSE_continuation_nostab": ("Post-prefix RMSE (no-liq, not stabilized)",
            "Post-prefix PPR(N) RMSE on non-liquefying experiments that did not stabilize."),
        "Traj_RMSE_continuation_balanced": ("Post-prefix RMSE (balanced)",
            "Macro-average of post-prefix per-state RMSE. Used as the preferred P³ trajectory "
            "component when available because the task is prefix-conditioned forecasting."),
        "Traj_RMSE_continuation_worst": ("Post-prefix RMSE (worst state)",
            "Worst post-prefix per-state RMSE; used by the P³ competence gate when available."),
    }
    for key, (name, desc) in _defs.items():
        METRICS[key] = MetricInfo(key, name, desc, "–", lower_is_better=True, fmt=".3f")


_register_state_traj_metrics()


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


# --- P³-ranking вынесен в отдельный модуль (работает поверх metrics_df) ---
from liquefaction_ai.evaluation.p3_ranking import (  # noqa: E402,F401
    metric_direction, compute_physical_admissibility, compute_p3_score,
    build_pareto_objectives, pareto_rank, publication_ranking_table,
)
