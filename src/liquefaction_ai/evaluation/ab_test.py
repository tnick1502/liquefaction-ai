"""
A/B-сравнение «flow vs gaussian posterior» для DPI-Flow (#3).

Прямой тест: даёт ли conditional RealNVP-поток выигрыш по ПРАВИЛЬНЫМ вероятностным скорам
против гауссова постериора (use_flow=False) при прочих равных. Ключевые принципы:

* **Скоры считаются на СМЕСИ** (mixture-NLL через logsumexp + sample energy-CRPS), а НЕ на
  схлопнутом в один гаусс предиктиве — иначе теряется skewness/мультимодальность RealNVP
  (то самое, ради чего поток и нужен). Это согласовано с обучающим объективом.
* **CI — кластерный bootstrap по ОБЪЕКТАМ с кратностью** (ресэмпл площадок с возвращением;
  объект, выбранный дважды, входит дважды). Точечная оценка и CI берутся из ОДНОГО оценщика
  (balanced object-mean), чтобы точка не оказалась вне интервала.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def mixture_scores_per_sample(model, split, config, device, n_samples: int = 32, seed: int = 0) -> pd.DataFrame:
    """
    Per-sample mixture-NLL и energy-CRPS по S сэмплам θ из постериора (flow/gaussian).

    NLL — proper плотность смеси ``−log[(1/S)Σ N(y;μ_s,σ_s²)]`` (logsumexp, БЕЗ схлопывания в один
    гаусс). CRPS — sample energy-score по предиктивным сэмплам ``μ_s+σ_s·ε``. Усреднены по валидным
    шагам опыта. Если модель не вероятностная — одиночный проход (S=1).

    :param model: обученная модель с ``forward_batch`` (и опц. ``_force_sample``)
    :param split: выборка (с ``r_obs``, ``mask``, ``meta['object']``)
    :param config: конфиг (batch_size)
    :param device: устройство
    :param n_samples: число сэмплов θ
    :param seed: сид воспроизводимости
    :return: DataFrame (object, nll, crps) — по строке на опыт
    """
    import torch
    from liquefaction_ai.config import set_global_seed
    from liquefaction_ai.data.splits import iterate_minibatches

    S = int(n_samples) if getattr(model, "probabilistic", False) else 1
    set_global_seed(seed)
    prev = getattr(model, "_force_sample", None)
    if hasattr(model, "_force_sample"):
        model._force_sample = S > 1
    model.eval()
    mus, lvs = [], []
    with torch.no_grad():
        for _ in range(S):
            mm, ll = [], []
            for batch in iterate_minibatches(split, config.batch_size, device, shuffle=False):
                o = model.forward_batch(batch)
                mm.append(o["traj_mean"]); ll.append(o["traj_logvar"])
            mus.append(torch.cat(mm, 0)); lvs.append(torch.cat(ll, 0))
    if hasattr(model, "_force_sample") and prev is not None:
        model._force_sample = prev
    M = torch.stack(mus, 0)
    LV = torch.clamp(torch.stack(lvs, 0), -6.0, 3.0)                 # (S, N, T)
    true = split["r_obs"].to(M.device); mask = split["mask"].to(M.device)
    t = true.unsqueeze(0)
    comp = -0.5 * (math.log(2 * math.pi) + LV + (t - M) ** 2 * torch.exp(-LV))
    logp = torch.logsumexp(comp, 0) - math.log(M.shape[0])          # (N, T) — log плотности смеси
    cnt = mask.sum(1).clamp_min(1.0)
    nll = (-(logp) * mask).sum(1) / cnt
    Y = M + torch.exp(0.5 * LV) * torch.randn_like(M)               # предиктивные сэмплы смеси
    term1 = (Y - t).abs().mean(0)
    term2 = (Y.unsqueeze(0) - Y.unsqueeze(1)).abs().mean((0, 1))
    crps = ((term1 - 0.5 * term2) * mask).sum(1) / cnt
    obj = split["meta"]["object"].to_numpy() if "object" in split["meta"].columns else np.arange(M.shape[1])
    return pd.DataFrame({"object": obj, "nll": nll.cpu().numpy(), "crps": crps.cpu().numpy()})


def _object_cluster_ab(sf: pd.DataFrame, sg: pd.DataFrame, metric: str,
                       nboot: int, rng) -> Dict[str, float]:
    """Balanced object-mean точка + кластерный bootstrap по объектам (с кратностью) для разницы."""
    gf = sf.groupby("object")[metric].mean()
    gg = sg.groupby("object")[metric].mean()
    uo = np.array(sorted(set(gf.index) & set(gg.index)))
    f_pt = float(gf.loc[uo].mean()); g_pt = float(gg.loc[uo].mean())
    draws = np.empty(nboot)
    for b in range(nboot):
        bo = rng.choice(uo, size=len(uo), replace=True)            # .loc[bo] СОХРАНЯЕТ кратность
        draws[b] = float(gg.loc[bo].mean() - gf.loc[bo].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"flow": round(f_pt, 4), "gaussian": round(g_pt, 4),
            "diff_gauss_minus_flow": round(g_pt - f_pt, 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            "P(flow_better)": round(float(np.mean(draws > 0)), 3)}


def ab_flow_vs_gaussian(model_flow, model_gauss, split, config, device,
                        n_samples: int = 32, nboot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    A/B flow vs gaussian по mixture-NLL и mixture-CRPS с кластерным bootstrap по объектам.

    Знак разницы ``gaussian − flow`` (>0 ⇒ flow лучше, т.к. меньше NLL/CRPS = лучше);
    ``P(flow_better)`` — доля bootstrap-итераций с разницей >0.

    :param model_flow: обученная модель с flow (use_flow=True)
    :param model_gauss: обученная модель-близнец без flow (use_flow=False)
    :param split: общая тестовая выборка
    :param config: конфиг; :param device: устройство
    :param n_samples: сэмплов θ в mixture-скорах; :param nboot: bootstrap-итераций; :param seed: сид
    :return: DataFrame по метрикам: flow, gaussian, разница, CI, P(flow лучше)
    """
    sf = mixture_scores_per_sample(model_flow, split, config, device, n_samples, seed)
    sg = mixture_scores_per_sample(model_gauss, split, config, device, n_samples, seed)
    rng = np.random.default_rng(seed)
    rows = []
    for name, col in (("Traj_mixNLL", "nll"), ("Traj_mixCRPS", "crps")):
        rows.append({"metric": name, **_object_cluster_ab(sf, sg, col, nboot, rng)})
    return pd.DataFrame(rows)


def train_ab_pair(benchmark: Dict[str, object], config, device, model_kwargs: Dict[str, object],
                  epochs: int = None, seed: int = 42):
    """
    Обучить ПАРУ DPI-Flow для A/B «flow vs gaussian»: идентичные модели, отличающиеся только
    ``use_flow`` (True = conditional RealNVP-поток над θ, False = гауссов постериор). Прочее равно
    (одни kwargs, один сид, одни эпохи/расписание) → разница в скорах относится к самому flow.

    :param benchmark: словарь выборок (нужны ``train``/``val``) из ``prepare_benchmark_dataset``
    :param config: конфигурация эксперимента (эпохи/расписание/сид)
    :param device: устройство обучения
    :param model_kwargs: kwargs конструктора DPIFlow (из ``hyperparams.json``; ``use_flow`` будет задан)
    :param epochs: число эпох (по умолчанию ``config.publication_physics_epochs``)
    :param seed: общий сид обеих моделей (идентичная инициализация/порядок батчей)
    :return: кортеж ``(model_flow, model_gauss)`` — обученные модели с потоком и без
    """
    from liquefaction_ai.config import set_global_seed
    from liquefaction_ai.models import DPIFlow
    from liquefaction_ai.training.loop import train_model

    ep = int(epochs if epochs is not None else getattr(config, "publication_physics_epochs", 200))
    base = {k: v for k, v in model_kwargs.items() if k != "use_flow"}
    trained = []
    for use_flow in (True, False):
        set_global_seed(seed)
        model = DPIFlow(**base, use_flow=use_flow).to(device)
        model, _ = train_model(model, benchmark["train"], benchmark["val"], epochs=ep,
                               model_name=f"AB-{'flow' if use_flow else 'gauss'}",
                               config=config, device=device, verbose=False, scheduler="cosine")
        trained.append(model)
    return trained[0], trained[1]
