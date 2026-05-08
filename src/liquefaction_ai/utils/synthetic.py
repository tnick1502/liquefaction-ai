from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from liquefaction_ai.config import ExperimentConfig
from liquefaction_ai.utils.constants import LOAD_NAMES, SOIL_NAMES
from liquefaction_ai.utils.splits import make_benchmark_splits


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def normalize_range(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def build_log_dense_cycles(n_max: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    base = np.geomspace(1e-3, 1.0, seq_len)
    base = (base - base.min()) / (base.max() - base.min())
    cycles = 1.0 + (n_max[:, None] - 1.0) * base[None, :]
    cycles = np.maximum.accumulate(cycles, axis=1)
    delta_cycles = np.diff(np.concatenate([np.zeros((n_max.shape[0], 1)), cycles], axis=1), axis=1)
    return cycles.astype(np.float32), delta_cycles.astype(np.float32)


def sample_soils(n: int, rng: np.random.Generator, soil_specs: dict, soil_names: List[str], soil_class_probs: np.ndarray) -> pd.DataFrame:
    class_id = rng.choice(len(soil_names), size=n, p=soil_class_probs)
    density_latent = rng.normal(0.0, 1.0, size=n)
    fabric_latent = rng.normal(0.0, 1.0, size=n)

    data: Dict[str, np.ndarray] = {
        "class_id": class_id.astype(int),
        "soil_type": np.array([soil_names[idx] for idx in class_id], dtype=object),
        "e": np.zeros(n, dtype=np.float32),
        "D_r": np.zeros(n, dtype=np.float32),
        "I_p": np.zeros(n, dtype=np.float32),
        "V_s": np.zeros(n, dtype=np.float32),
        "xi": np.zeros(n, dtype=np.float32),
        "sigma_eff": np.zeros(n, dtype=np.float32),
        "permeability": np.zeros(n, dtype=np.float32),
    }

    for idx, soil_name in enumerate(soil_names):
        mask = class_id == idx
        count = int(mask.sum())
        if count == 0:
            continue
        spec = soil_specs[soil_name]
        dr = np.clip(rng.uniform(*spec["D_r"], size=count) + 0.08 * density_latent[mask], 0.15, 0.98)
        e = np.clip(rng.uniform(*spec["e"], size=count) - 0.07 * density_latent[mask], 0.38, 1.15)
        ip = np.clip(rng.uniform(*spec["I_p"], size=count) + 1.4 * fabric_latent[mask], 0.0, 35.0)
        vs = np.clip(rng.uniform(*spec["V_s"], size=count) + 12.0 * density_latent[mask], 90.0, 420.0)
        xi = np.clip(rng.uniform(*spec["xi"], size=count) + 0.0025 * ip, 0.01, 0.18)
        sigma_eff = np.clip(rng.uniform(*spec["sigma_eff"], size=count) + 8.0 * density_latent[mask], 45.0, 350.0)
        log_perm = rng.uniform(*spec["log_perm"], size=count) - 0.18 * density_latent[mask]
        permeability = np.power(10.0, log_perm)

        data["e"][mask] = e.astype(np.float32)
        data["D_r"][mask] = dr.astype(np.float32)
        data["I_p"][mask] = ip.astype(np.float32)
        data["V_s"][mask] = vs.astype(np.float32)
        data["xi"][mask] = xi.astype(np.float32)
        data["sigma_eff"][mask] = sigma_eff.astype(np.float32)
        data["permeability"][mask] = permeability.astype(np.float32)

    return pd.DataFrame(data)


def sample_loads(n: int, rng: np.random.Generator, load_specs: dict, load_names: List[str], load_mode_probs: np.ndarray) -> pd.DataFrame:
    mode_id = rng.choice(len(load_names), size=n, p=load_mode_probs)
    data: Dict[str, np.ndarray] = {
        "mode_id": mode_id.astype(int),
        "load_mode": np.array([load_names[idx] for idx in mode_id], dtype=object),
        "CSR_base": np.zeros(n, dtype=np.float32),
        "frequency": np.zeros(n, dtype=np.float32),
        "amp_scale": np.zeros(n, dtype=np.float32),
        "N_max": np.zeros(n, dtype=np.float32),
        "nonstationarity": np.zeros(n, dtype=np.float32),
        "phase": rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32),
        "burst_1": rng.uniform(0.15, 0.45, size=n).astype(np.float32),
        "burst_2": rng.uniform(0.45, 0.85, size=n).astype(np.float32),
    }

    for idx, mode_name in enumerate(load_names):
        mask = mode_id == idx
        count = int(mask.sum())
        if count == 0:
            continue
        spec = load_specs[mode_name]
        data["CSR_base"][mask] = rng.uniform(*spec["CSR"], size=count).astype(np.float32)
        data["frequency"][mask] = rng.uniform(*spec["frequency"], size=count).astype(np.float32)
        data["amp_scale"][mask] = rng.uniform(*spec["amp_scale"], size=count).astype(np.float32)
        data["N_max"][mask] = rng.uniform(*spec["N_max"], size=count).astype(np.float32)
        data["nonstationarity"][mask] = rng.uniform(*spec["nonstationarity"], size=count).astype(np.float32)

    return pd.DataFrame(data)


def build_csr_history(load_df: pd.DataFrame, cycles: np.ndarray, max_csr_clip: float) -> np.ndarray:
    n = len(load_df)
    u = cycles / load_df["N_max"].to_numpy()[:, None]
    phase = load_df["phase"].to_numpy()[:, None] / (2.0 * np.pi)
    burst_1 = load_df["burst_1"].to_numpy()[:, None]
    burst_2 = load_df["burst_2"].to_numpy()[:, None]
    amp_scale = load_df["amp_scale"].to_numpy()[:, None]
    nonstationarity = load_df["nonstationarity"].to_numpy()[:, None]
    csr_base = load_df["CSR_base"].to_numpy()[:, None]
    mode_id = load_df["mode_id"].to_numpy()

    csr = np.zeros_like(cycles, dtype=np.float32)

    storm = mode_id == LOAD_NAMES.index("storm")
    if storm.any():
        csr[storm] = (
            csr_base[storm]
            * (
                0.78
                + 0.18 * np.sin(2.0 * np.pi * (1.5 * u[storm] + phase[storm]))
                + 0.20 * np.power(u[storm], 1.10)
                + 0.10 * np.exp(-18.0 * np.square(u[storm] - burst_1[storm]))
            )
            * (1.0 + 0.35 * nonstationarity[storm])
            * amp_scale[storm]
        )

    seismic = mode_id == LOAD_NAMES.index("seismic")
    if seismic.any():
        csr[seismic] = (
            csr_base[seismic]
            * (
                0.45
                + 1.10 * np.exp(-5.5 * u[seismic])
                + 0.70 * np.exp(-140.0 * np.square(u[seismic] - burst_1[seismic]))
                + 0.45 * np.exp(-90.0 * np.square(u[seismic] - burst_2[seismic]))
            )
            * (1.0 + 0.25 * nonstationarity[seismic])
            * amp_scale[seismic]
        )

    technogenic = mode_id == LOAD_NAMES.index("technogenic")
    if technogenic.any():
        csr[technogenic] = (
            csr_base[technogenic]
            * (
                0.95
                + 0.06 * np.sin(2.0 * np.pi * (5.0 * u[technogenic] + phase[technogenic]))
                + 0.08 * u[technogenic]
            )
            * (1.0 + 0.12 * nonstationarity[technogenic])
            * amp_scale[technogenic]
        )

    stationary = mode_id == LOAD_NAMES.index("stationary_cyclic")
    if stationary.any():
        csr[stationary] = (
            csr_base[stationary]
            * (
                1.00
                + 0.03 * np.sin(2.0 * np.pi * (1.2 * u[stationary] + phase[stationary]))
                + 0.04 * nonstationarity[stationary]
            )
            * amp_scale[stationary]
        )

    variable = mode_id == LOAD_NAMES.index("variable_amplitude")
    if variable.any():
        csr[variable] = (
            csr_base[variable]
            * (
                0.72
                + 0.18 * np.sin(2.0 * np.pi * (3.0 * u[variable] + phase[variable]))
                + 0.28 * (u[variable] > burst_1[variable]).astype(np.float32)
                + 0.16 * (u[variable] > burst_2[variable]).astype(np.float32)
            )
            * (1.0 + 0.30 * nonstationarity[variable])
            * amp_scale[variable]
        )

    return np.clip(csr, 0.02, max_csr_clip).astype(np.float32)


def build_hidden_parameters(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    cycles: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    e = soil_df["e"].to_numpy()
    dr = soil_df["D_r"].to_numpy()
    ip = soil_df["I_p"].to_numpy()
    vs = soil_df["V_s"].to_numpy()
    xi = soil_df["xi"].to_numpy()
    sigma_eff = soil_df["sigma_eff"].to_numpy()
    perm = soil_df["permeability"].to_numpy()
    mode_id = load_df["mode_id"].to_numpy()
    csr_base = load_df["CSR_base"].to_numpy()
    amp_scale = load_df["amp_scale"].to_numpy()
    n_max = load_df["N_max"].to_numpy()

    e_n = normalize_range(e, 0.38, 1.15)
    ip_n = normalize_range(ip, 0.0, 35.0)
    vs_n = normalize_range(vs, 90.0, 420.0)
    xi_n = normalize_range(xi, 0.01, 0.18)
    sigma_n = normalize_range(np.log1p(sigma_eff), np.log1p(45.0), np.log1p(350.0))
    perm_n = normalize_range(np.log10(perm), -7.0, -2.5)

    family_logits = np.stack(
        [
            1.20 + 1.05 * dr - 0.20 * e_n + 0.20 * vs_n,
            0.95 + 0.85 * dr + 0.10 * sigma_n,
            0.70 + 0.55 * (mode_id == LOAD_NAMES.index("seismic")) + 0.30 * csr_base,
            0.85 + 0.65 * ip_n + 0.35 * (mode_id == LOAD_NAMES.index("storm")),
        ],
        axis=1,
    )
    family_logits += rng.normal(0.0, 0.35, size=family_logits.shape)
    weights = softmax_np(family_logits, axis=1).astype(np.float32)
    family_id = np.argmax(weights, axis=1).astype(np.int64)

    n_rel = cycles / 100.0
    strength = np.clip(
        0.08
        + 0.18 * dr
        + 0.08 * vs_n
        + 0.04 * sigma_n
        - 0.06 * e_n
        - 0.04 * ip_n
        - 0.02 * xi_n,
        0.04,
        0.42,
    ).astype(np.float32)

    h_tail = np.clip(0.35 * strength + 0.02 * dr, 0.02, 0.24)
    p_tail = np.clip(0.30 * strength + 0.02 * vs_n, 0.02, 0.24)
    e_tail = np.clip(0.18 * strength + 0.015 * sigma_n, 0.01, 0.18)
    l_tail = np.clip(0.28 * strength + 0.01 * ip_n, 0.015, 0.22)

    crr_h = h_tail[:, None] + (0.70 * strength)[:, None] / (
        1.0 + (0.18 + 0.65 * (1.0 - dr) + 0.15 * xi_n)[:, None] * n_rel
    )
    crr_p = p_tail[:, None] + (0.78 * strength)[:, None] * np.power(
        1.0 + n_rel, -(0.22 + 0.80 * (1.0 - dr) + 0.20 * ip_n)[:, None]
    )
    crr_e = e_tail[:, None] + (0.88 * strength)[:, None] * np.exp(
        -(0.12 + 0.35 * (1.0 - dr) + 0.20 * (mode_id == LOAD_NAMES.index("seismic")))[:, None] * n_rel
    )
    crr_l = l_tail[:, None] + (0.90 * strength)[:, None] / (
        1.0
        + (0.55 + 0.50 * (1.0 - dr) + 0.25 * (mode_id == LOAD_NAMES.index("variable_amplitude")))[:, None]
        * np.log1p(4.0 * n_rel)
    )

    crr_mix = (
        weights[:, 0:1] * crr_h
        + weights[:, 1:2] * crr_p
        + weights[:, 2:3] * crr_e
        + weights[:, 3:4] * crr_l
    ).astype(np.float32)

    lambda_damage = np.clip(
        (0.0012 + 0.0024 * (1.0 - dr) + 0.0008 * amp_scale)
        * np.power(300.0 / np.maximum(n_max, 300.0), 0.35),
        0.00015,
        0.0050,
    ).astype(np.float32)
    exponent_m = np.clip(1.15 + 1.40 * (1.0 - dr) + 0.15 * ip_n, 1.05, 3.25).astype(np.float32)
    exponent_nu = np.clip(1.05 + 1.05 * dr + 0.20 * vs_n, 1.0, 2.6).astype(np.float32)

    alpha = np.clip(0.0009 + 0.0022 * (1.0 - dr) + 0.0008 * csr_base, 0.0004, 0.0050).astype(np.float32)
    beta = np.clip(0.045 + 0.070 * perm_n + 0.025 * (mode_id == LOAD_NAMES.index("storm")), 0.02, 0.18).astype(
        np.float32
    )
    gamma = np.clip(
        0.00012 + 0.00055 * xi_n + 0.00018 * (mode_id == LOAD_NAMES.index("technogenic")),
        0.00005,
        0.0012,
    ).astype(np.float32)
    exponent_p = np.clip(1.10 + 1.10 * dr + 0.25 * vs_n, 1.0, 2.8).astype(np.float32)
    tau = np.clip(5.0 + 18.0 * perm_n + 12.0 * (mode_id == LOAD_NAMES.index("storm")), 3.0, 40.0).astype(np.float32)

    alpha_post = np.clip(
        alpha * (0.85 + 0.55 * (mode_id == LOAD_NAMES.index("seismic")) + 0.35 * (mode_id == LOAD_NAMES.index("variable_amplitude"))),
        0.0004,
        0.0075,
    ).astype(np.float32)
    beta_post = np.clip(0.015 + 0.050 * ip_n + 0.020 * (mode_id == LOAD_NAMES.index("storm")), 0.01, 0.16).astype(
        np.float32
    )
    gamma_post = np.clip(gamma * (1.20 + 0.70 * ip_n), 0.00008, 0.0020).astype(np.float32)
    exponent_p_post = np.clip(0.95 + 0.85 * dr, 0.9, 2.1).astype(np.float32)
    tau_post = np.clip(0.70 * tau + 5.0, 5.0, 35.0).astype(np.float32)

    kappa = np.clip(8.0 + 10.0 * amp_scale + 2.0 * csr_base, 6.0, 24.0).astype(np.float32)
    z0 = np.clip(0.55 + 0.16 * dr - 0.08 * (mode_id == LOAD_NAMES.index("seismic")), 0.40, 0.82).astype(np.float32)
    entropy = -(weights * np.log(weights + 1e-8)).sum(axis=1).astype(np.float32)

    return {
        "weights": weights,
        "family_id": family_id,
        "crr_h": crr_h.astype(np.float32),
        "crr_p": crr_p.astype(np.float32),
        "crr_e": crr_e.astype(np.float32),
        "crr_l": crr_l.astype(np.float32),
        "crr_mix": crr_mix,
        "lambda_damage": lambda_damage,
        "m": exponent_m,
        "nu": exponent_nu,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "p": exponent_p,
        "tau": tau,
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "gamma_post": gamma_post,
        "p_post": exponent_p_post,
        "tau_post": tau_post,
        "kappa": kappa,
        "z0": z0,
        "entropy": entropy,
    }


def integrate_physics(
    hidden: Dict[str, np.ndarray],
    csr: np.ndarray,
    cycles: np.ndarray,
    delta_cycles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, seq_len = csr.shape
    z = np.zeros((n, seq_len), dtype=np.float32)
    r = np.zeros((n, seq_len), dtype=np.float32)
    g = np.zeros((n, seq_len), dtype=np.float32)
    eps = 1e-6

    for step in range(seq_len - 1):
        crr_step = hidden["crr_mix"][:, step]
        ratio = csr[:, step] / (crr_step + eps)
        phi = np.log1p(np.exp(6.0 * (ratio - 0.92))) / 6.0
        g[:, step] = expit(hidden["kappa"] * (z[:, step] - hidden["z0"]))

        dz = (
            hidden["lambda_damage"]
            * np.power(np.clip(ratio, 0.0, None) + eps, hidden["m"])
            * np.power(np.clip(1.0 - z[:, step], eps, 1.0), hidden["nu"])
        )
        pre_event = (
            hidden["alpha"] * phi * np.power(np.clip(1.0 - r[:, step], eps, 1.0), hidden["p"])
            + hidden["beta"] / (cycles[:, step] + hidden["tau"])
            - hidden["gamma"] * r[:, step]
        )
        post_event = (
            hidden["alpha_post"] * np.power(np.clip(1.0 - r[:, step], eps, 1.0), hidden["p_post"])
            + hidden["beta_post"] / (cycles[:, step] + hidden["tau_post"])
            - hidden["gamma_post"] * r[:, step]
        )
        dr = (1.0 - g[:, step]) * pre_event + g[:, step] * post_event

        z[:, step + 1] = np.clip(z[:, step] + delta_cycles[:, step + 1] * dz, 0.0, 0.9995)
        r[:, step + 1] = np.clip(r[:, step] + delta_cycles[:, step + 1] * dr, 0.0, 1.02)

    g[:, -1] = expit(hidden["kappa"] * (z[:, -1] - hidden["z0"]))
    return z, r, g


def build_observations(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    hidden: Dict[str, np.ndarray],
    z_true: np.ndarray,
    r_true: np.ndarray,
    g_true: np.ndarray,
    cycles: np.ndarray,
    rng: np.random.Generator,
    prefix_len: int,
) -> Dict[str, np.ndarray]:
    n, seq_len = r_true.shape
    dr = soil_df["D_r"].to_numpy()
    mode_id = load_df["mode_id"].to_numpy()

    obs_fraction = np.clip(rng.beta(4.0, 1.5, size=n), 0.60, 1.0)
    observed_length = np.maximum(prefix_len + 8, np.floor(obs_fraction * seq_len).astype(int))
    observed_length = np.clip(observed_length, prefix_len + 4, seq_len)
    valid_mask = (np.arange(seq_len)[None, :] < observed_length[:, None]).astype(np.float32)

    noise_scale = (
        0.010
        + 0.030 * (1.0 - dr)
        + 0.015 * (mode_id == LOAD_NAMES.index("variable_amplitude"))
        + 0.010 * g_true.max(axis=1)
        + 0.020 * hidden["entropy"]
    ).astype(np.float32)
    noise = noise_scale[:, None] * (0.35 + 0.65 * r_true) * rng.normal(size=r_true.shape)
    outliers = (rng.random(size=r_true.shape) < 0.0025).astype(np.float32) * rng.normal(
        loc=0.06, scale=0.025, size=r_true.shape
    )
    r_obs = np.clip(r_true + noise + outliers, 0.0, 1.05).astype(np.float32)

    prefix_mask = ((np.arange(seq_len)[None, :] < prefix_len) & (valid_mask > 0)).astype(np.float32)
    prefix_obs = (r_obs * prefix_mask).astype(np.float32)

    liq_mask = (r_true >= 0.985) | (g_true >= 0.95)
    hit_any = liq_mask.any(axis=1)
    first_idx = liq_mask.argmax(axis=1)
    n_liq = np.where(hit_any, cycles[np.arange(n), first_idx], load_df["N_max"].to_numpy()).astype(np.float32)
    liq_label = hit_any.astype(np.float32)

    risk_score = expit(
        4.0 * (0.45 * r_true.max(axis=1) + 0.30 * z_true.max(axis=1) + 0.25 * g_true.max(axis=1) - 0.72)
    ).astype(np.float32)
    uncertainty_proxy = (
        noise_scale
        + 0.08 * hidden["entropy"]
        + 0.03 * (mode_id == LOAD_NAMES.index("variable_amplitude"))
        + 0.02 * (mode_id == LOAD_NAMES.index("seismic"))
    ).astype(np.float32)

    return {
        "r_obs": r_obs,
        "valid_mask": valid_mask,
        "prefix_mask": prefix_mask,
        "prefix_obs": prefix_obs,
        "liq_label": liq_label,
        "n_liq_true": n_liq,
        "risk_score": risk_score,
        "uncertainty_proxy": uncertainty_proxy,
    }


def build_feature_matrices(
    soil_df: pd.DataFrame,
    load_df: pd.DataFrame,
    cycles: np.ndarray,
    delta_cycles: np.ndarray,
    csr: np.ndarray,
    observations: Dict[str, np.ndarray],
    prefix_len: int,
) -> Dict[str, object]:
    n = len(soil_df)
    soil_onehot = np.eye(len(SOIL_NAMES), dtype=np.float32)[soil_df["class_id"].to_numpy()]
    mode_onehot = np.eye(len(LOAD_NAMES), dtype=np.float32)[load_df["mode_id"].to_numpy()]

    prefix_mask = observations["prefix_mask"]
    prefix_obs = observations["prefix_obs"]
    prefix_count = np.clip(prefix_mask.sum(axis=1), 1.0, None)
    prefix_mean = (prefix_obs.sum(axis=1) / prefix_count).astype(np.float32)
    last_prefix_idx = np.maximum(prefix_count.astype(int) - 1, 0)
    first_prefix = prefix_obs[:, 0]
    last_prefix = prefix_obs[np.arange(n), last_prefix_idx]
    prefix_peak = prefix_obs.max(axis=1)
    prefix_var = (
        (np.square(prefix_obs - prefix_mean[:, None]) * prefix_mask).sum(axis=1) / prefix_count
    ).astype(np.float32)
    delta_log_n = np.log1p(cycles[np.arange(n), last_prefix_idx]) - np.log1p(cycles[:, 0])
    prefix_slope = ((last_prefix - first_prefix) / np.maximum(delta_log_n, 1e-3)).astype(np.float32)
    prefix_coverage = (prefix_count / prefix_len).astype(np.float32)

    static_feature_names = (
        [
            "e",
            "D_r",
            "I_p",
            "V_s",
            "xi",
            "sigma_eff",
            "log10_permeability",
            "CSR_base",
            "frequency",
            "amp_scale",
            "N_max",
            "nonstationarity",
        ]
        + [f"soil_{name}" for name in SOIL_NAMES]
        + [f"mode_{name}" for name in LOAD_NAMES]
    )

    static_features = np.column_stack(
        [
            soil_df["e"].to_numpy(),
            soil_df["D_r"].to_numpy(),
            soil_df["I_p"].to_numpy(),
            soil_df["V_s"].to_numpy(),
            soil_df["xi"].to_numpy(),
            soil_df["sigma_eff"].to_numpy(),
            np.log10(soil_df["permeability"].to_numpy()),
            load_df["CSR_base"].to_numpy(),
            load_df["frequency"].to_numpy(),
            load_df["amp_scale"].to_numpy(),
            load_df["N_max"].to_numpy(),
            load_df["nonstationarity"].to_numpy(),
            soil_onehot,
            mode_onehot,
        ]
    ).astype(np.float32)

    prefix_summary_names = [
        "prefix_mean",
        "prefix_last",
        "prefix_peak",
        "prefix_slope",
        "prefix_var",
        "prefix_coverage",
    ]
    prefix_summary = np.column_stack(
        [
            prefix_mean,
            last_prefix.astype(np.float32),
            prefix_peak.astype(np.float32),
            prefix_slope,
            prefix_var,
            prefix_coverage,
        ]
    ).astype(np.float32)

    log_cycle_norm = np.log1p(cycles) / np.log1p(load_df["N_max"].to_numpy()[:, None])
    delta_cycle_norm = delta_cycles / np.maximum(load_df["N_max"].to_numpy()[:, None], 1.0)
    seq_inputs = np.stack(
        [
            csr,
            log_cycle_norm.astype(np.float32),
            delta_cycle_norm.astype(np.float32),
            observations["prefix_obs"],
            observations["prefix_mask"],
        ],
        axis=-1,
    ).astype(np.float32)
    seq_feature_names = ["CSR", "log_cycle_norm", "delta_cycle_norm", "prefix_obs", "prefix_mask"]

    return {
        "static_features": static_features,
        "static_feature_names": static_feature_names,
        "prefix_summary": prefix_summary,
        "prefix_summary_names": prefix_summary_names,
        "seq_inputs": seq_inputs,
        "seq_feature_names": seq_feature_names,
    }


def generate_population(config: ExperimentConfig) -> Dict[str, object]:
    from liquefaction_ai.utils.constants import LOAD_MODE_SPECS, SOIL_CLASS_SPECS

    soil_class_probs = np.array([0.22, 0.18, 0.18, 0.18, 0.24])
    load_mode_probs = np.array([0.24, 0.18, 0.16, 0.20, 0.22])
    rng = np.random.default_rng(config.seed)
    soil_df = sample_soils(config.n_scenarios, rng, SOIL_CLASS_SPECS, SOIL_NAMES, soil_class_probs)
    load_df = sample_loads(config.n_scenarios, rng, LOAD_MODE_SPECS, LOAD_NAMES, load_mode_probs)
    cycles, delta_cycles = build_log_dense_cycles(load_df["N_max"].to_numpy(), config.seq_len)
    csr = build_csr_history(load_df, cycles, config.max_csr_clip)
    hidden = build_hidden_parameters(soil_df, load_df, cycles, rng)
    z_true, r_true, g_true = integrate_physics(hidden, csr, cycles, delta_cycles)
    observations = build_observations(
        soil_df, load_df, hidden, z_true, r_true, g_true, cycles, rng, config.prefix_len
    )
    features = build_feature_matrices(
        soil_df, load_df, cycles, delta_cycles, csr, observations, config.prefix_len
    )

    meta = pd.concat([soil_df, load_df], axis=1)
    meta["liq_label"] = observations["liq_label"].astype(int)
    meta["risk_score_true"] = observations["risk_score"]
    meta["N_liq_true"] = observations["n_liq_true"]
    meta["uncertainty_proxy"] = observations["uncertainty_proxy"]
    meta["generator_family_id"] = hidden["family_id"]
    meta["generator_family"] = np.array(["hyperbolic", "power", "exponential", "logarithmic"])[hidden["family_id"]]
    meta["PPR_max_true"] = r_true.max(axis=1)
    meta["damage_max_true"] = z_true.max(axis=1)
    meta["trigger_max_true"] = g_true.max(axis=1)
    meta["CSR_max"] = csr.max(axis=1)

    benchmark = make_benchmark_splits(meta, config.benchmark_subset, config.seed, config)

    return {
        "meta": meta,
        "cycles": cycles.astype(np.float32),
        "delta_cycles": delta_cycles.astype(np.float32),
        "csr": csr.astype(np.float32),
        "crr_mix": hidden["crr_mix"].astype(np.float32),
        "crr_families": {
            "hyperbolic": hidden["crr_h"],
            "power": hidden["crr_p"],
            "exponential": hidden["crr_e"],
            "logarithmic": hidden["crr_l"],
        },
        "crr_weights": hidden["weights"].astype(np.float32),
        "z_true": z_true.astype(np.float32),
        "r_true": r_true.astype(np.float32),
        "g_true": g_true.astype(np.float32),
        "r_obs": observations["r_obs"].astype(np.float32),
        "valid_mask": observations["valid_mask"].astype(np.float32),
        "prefix_mask": observations["prefix_mask"].astype(np.float32),
        "prefix_obs": observations["prefix_obs"].astype(np.float32),
        "liq_label": observations["liq_label"].astype(np.float32),
        "risk_score_true": observations["risk_score"].astype(np.float32),
        "n_liq_true": observations["n_liq_true"].astype(np.float32),
        "uncertainty_proxy": observations["uncertainty_proxy"].astype(np.float32),
        "static_features": features["static_features"].astype(np.float32),
        "static_feature_names": features["static_feature_names"],
        "prefix_summary": features["prefix_summary"].astype(np.float32),
        "prefix_summary_names": features["prefix_summary_names"],
        "seq_inputs": features["seq_inputs"].astype(np.float32),
        "seq_feature_names": features["seq_feature_names"],
        "benchmark": benchmark,
    }
