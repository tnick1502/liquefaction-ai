from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from liquefaction_ai.config import ExperimentConfig


def safe_strata(meta: pd.DataFrame, fine_columns: List[str]) -> np.ndarray:
    fine = meta[fine_columns].astype(str).agg("|".join, axis=1)
    fine_counts = fine.value_counts()
    if fine_counts.min() >= 2:
        return fine.to_numpy()

    medium = meta[["load_mode", "liq_label"]].astype(str).agg("|".join, axis=1)
    medium = fine.where(fine.map(fine_counts) >= 2, medium)
    medium_counts = medium.value_counts()
    if medium_counts.min() >= 2:
        return medium.to_numpy()

    return meta["liq_label"].astype(str).to_numpy()


def stratified_subset_indices(meta: pd.DataFrame, subset_size: int, seed: int) -> np.ndarray:
    subset_size = min(subset_size, len(meta))
    strata = safe_strata(meta, ["soil_type", "load_mode", "liq_label"])
    idx = np.arange(len(meta))
    keep_idx, _ = train_test_split(idx, train_size=subset_size, stratify=strata, random_state=seed)
    return np.sort(keep_idx)


def make_benchmark_splits(meta: pd.DataFrame, subset_size: int, seed: int, config: ExperimentConfig) -> Dict[str, np.ndarray]:
    benchmark_idx = stratified_subset_indices(meta, subset_size, seed)
    benchmark_meta = meta.iloc[benchmark_idx].reset_index(drop=True)
    benchmark_rel = np.arange(len(benchmark_idx))
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


def subset_array(arr: object, idx: np.ndarray) -> object:
    if isinstance(arr, np.ndarray):
        return arr[idx]
    return arr


def prepare_benchmark_dataset(
    population_dict: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    benchmark_idx = population_dict["benchmark"]["benchmark_idx"]
    bench_meta = population_dict["meta"].iloc[benchmark_idx].reset_index(drop=True)

    static_raw = population_dict["static_features"][benchmark_idx]
    prefix_raw = population_dict["prefix_summary"][benchmark_idx]
    seq_raw = population_dict["seq_inputs"][benchmark_idx]
    cycles = population_dict["cycles"][benchmark_idx]
    delta_cycles = population_dict["delta_cycles"][benchmark_idx]
    csr = population_dict["csr"][benchmark_idx]
    r_obs = population_dict["r_obs"][benchmark_idx]
    r_true = population_dict["r_true"][benchmark_idx]
    z_true = population_dict["z_true"][benchmark_idx]
    g_true = population_dict["g_true"][benchmark_idx]
    valid_mask = population_dict["valid_mask"][benchmark_idx]
    prefix_mask = population_dict["prefix_mask"][benchmark_idx]
    prefix_obs = population_dict["prefix_obs"][benchmark_idx]
    liq_label = population_dict["liq_label"][benchmark_idx]
    risk_score_true = population_dict["risk_score_true"][benchmark_idx]
    n_liq_true = population_dict["n_liq_true"][benchmark_idx]
    uncertainty_proxy = population_dict["uncertainty_proxy"][benchmark_idx]
    crr_mix = population_dict["crr_mix"][benchmark_idx]
    crr_weights = population_dict["crr_weights"][benchmark_idx]

    train_rel = population_dict["benchmark"]["train_rel"]
    val_rel = population_dict["benchmark"]["val_rel"]
    test_rel = population_dict["benchmark"]["test_rel"]

    static_scaler = StandardScaler().fit(static_raw[train_rel])
    prefix_scaler = StandardScaler().fit(prefix_raw[train_rel])
    seq_train = seq_raw[train_rel].reshape(-1, seq_raw.shape[-1])
    seq_mean = seq_train.mean(axis=0)
    seq_std = seq_train.std(axis=0) + 1e-6

    static_scaled = static_scaler.transform(static_raw).astype(np.float32)
    prefix_scaled = prefix_scaler.transform(prefix_raw).astype(np.float32)
    seq_scaled = ((seq_raw - seq_mean[None, None, :]) / seq_std[None, None, :]).astype(np.float32)
    n_liq_norm = (np.log1p(n_liq_true) / np.log1p(config.max_cycle_reference)).astype(np.float32)
    trigger_zone = (g_true > 0.70).astype(np.float32)

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
        "r_true": r_true.astype(np.float32),
        "z_true": z_true.astype(np.float32),
        "g_true": g_true.astype(np.float32),
        "trigger_zone": trigger_zone,
        "mask": valid_mask.astype(np.float32),
        "prefix_mask": prefix_mask.astype(np.float32),
        "prefix_obs": prefix_obs.astype(np.float32),
        "label": liq_label.astype(np.float32),
        "risk_true": risk_score_true.astype(np.float32),
        "n_liq_true": n_liq_true.astype(np.float32),
        "n_liq_norm": n_liq_norm.astype(np.float32),
        "uncertainty_proxy": uncertainty_proxy.astype(np.float32),
        "crr_mix_true": crr_mix.astype(np.float32),
        "crr_weights_true": crr_weights.astype(np.float32),
    }

    def make_split(rel_idx: np.ndarray) -> Dict[str, object]:
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
