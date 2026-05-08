from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.utils.meta import localize_meta_frame
from liquefaction_ai.utils.splits import iterate_minibatches


def collect_outputs(
    model: nn.Module,
    split: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
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
    if "nliq" in outputs:
        return outputs["nliq"]
    if "nliq_pred" in outputs:
        return np.expm1(outputs["nliq_pred"] * math.log1p(max_cycle_reference))
    if "nliq_norm" in outputs:
        return np.expm1(outputs["nliq_norm"] * math.log1p(max_cycle_reference))
    raise KeyError("В outputs не найдено предсказание для N_liq.")


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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
    current_size = split["static"].shape[0]
    if current_size <= max_size:
        return split
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(current_size, size=max_size, replace=False))
    new_split = {}
    for key, value in split.items():
        if isinstance(value, torch.Tensor):
            new_split[key] = value[idx]
        elif key == "meta":
            new_split[key] = value.iloc[idx].reset_index(drop=True)
        else:
            new_split[key] = value
    return new_split


def filter_split(split: Dict[str, object], mask: np.ndarray) -> Dict[str, object]:
    idx = np.where(mask)[0]
    filtered: Dict[str,object] = {}
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
    from liquefaction_ai.utils.train_loop import train_model

    model = model.to(device)
    model, _ = train_model(model, train_split, val_split, epochs=epochs, model_name=model_name, config=config, device=device)
    outputs = collect_outputs(model, test_split, config, device)
    metrics, _ = compute_metrics(model_name, outputs, test_split, config)
    return metrics


def is_holdout_region(meta_df: pd.DataFrame, e_threshold: float, vs_threshold: float) -> np.ndarray:
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


def localize_model_names_in_df(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    if "model" in localized.columns:
        localized["model"] = localized["model"].map(MODEL_DISPLAY_NAMES).fillna(localized["model"])
        localized["model"] = localized["model"].str.replace(": holdout", ": удержанный набор", regex=False)
        localized["model"] = localized["model"].str.replace(" | region", " | удержанный_регион", regex=False)
        localized["model"] = localized["model"].str.replace(" | unseen_regime", " | невидимый_режим", regex=False)
    return localized


def localize_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    localized = localize_model_names_in_df(df)
    rename_map = {key: value for key, value in METRIC_COLUMN_TRANSLATIONS.items() if key in localized.columns}
    return localized.rename(columns=rename_map)


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
    "localize_model_names_in_df",
    "localize_metric_table",
]
