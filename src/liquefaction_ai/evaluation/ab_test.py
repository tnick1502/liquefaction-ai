"""
A/B-сравнение «flow vs gaussian posterior» для DPI-Flow.

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
    LV = torch.clamp(torch.stack(lvs, 0), -6.0, 3.0) # (S, N, T)
    true = split["r_obs"].to(M.device); mask = split["mask"].to(M.device)
    # СТРОГО ПОСТ-ПРЕФИКСНАЯ (continuation) область: скоры на прогнозе после наблюдаемого окна, а не
    # на реконструкции префикса. Иначе A/B «flow vs gaussian» замеряется в т.ч. на
    # входных точках, где обе модели тривиально совпадают. Совпадает с Traj_*_continuation в metrics.
    if "prefix_mask" in split:
        pmask = split["prefix_mask"].to(M.device)
        smask = mask * (1.0 - torch.clamp(pmask, max=1.0))
    else:
        smask = mask
    t = true.unsqueeze(0)
    comp = -0.5 * (math.log(2 * math.pi) + LV + (t - M) ** 2 * torch.exp(-LV))
    logp = torch.logsumexp(comp, 0) - math.log(M.shape[0]) # (N, T) — log плотности смеси
    raw_cnt = smask.sum(1)
    valid = raw_cnt > 0
    cnt = raw_cnt.clamp_min(1.0)
    nll = (-(logp) * smask).sum(1) / cnt
    Y = M + torch.exp(0.5 * LV) * torch.randn_like(M) # предиктивные сэмплы смеси
    term1 = (Y - t).abs().mean(0)
    term2 = (Y.unsqueeze(0) - Y.unsqueeze(1)).abs().mean((0, 1))
    crps = ((term1 - 0.5 * term2) * smask).sum(1) / cnt
    # Per-sample эмпирическое покрытие 90%-предиктивной полосы смеси (calibration): доля валидных
    # continuation-точек внутри [q05, q95] по предиктивным сэмплам Y. Честный A/B-калибровочный скор.
    ql = torch.quantile(Y, 0.05, dim=0); qh = torch.quantile(Y, 0.95, dim=0)
    inside = ((true >= ql) & (true <= qh)).to(M.dtype)
    coverage_hits = (inside * smask).sum(1)
    cov90 = coverage_hits / cnt
    nan = torch.full_like(nll, float("nan"))
    nll = torch.where(valid, nll, nan)
    crps = torch.where(valid, crps, nan)
    cov90 = torch.where(valid, cov90, nan)
    meta = split["meta"]
    site = (meta["site_id"].to_numpy() if "site_id" in meta.columns
            else (meta["object"].to_numpy() if "object" in meta.columns else np.arange(M.shape[1])))
    return pd.DataFrame({"site_id": site, "nll": nll.cpu().numpy(),
                         "crps": crps.cpu().numpy(), "cov90": cov90.cpu().numpy(),
                         "coverage90_hits": coverage_hits.cpu().numpy(),
                         "continuation_points": raw_cnt.cpu().numpy()})


def _object_cluster_ab(sf: pd.DataFrame, sg: pd.DataFrame, metric: str,
                       nboot: int, rng, transform=None,
                       numerator: str | None = None, denominator: str | None = None) -> Dict[str, float]:
    """
    Balanced SITE-mean точка + кластерный bootstrap по ПЛОЩАДКАМ (site_id, с кратностью) для разницы.

    Для coverage передайте numerator/denominator: сначала суммируются hits/points внутри site,
    затем применяется ``transform``. Это оценивает |site coverage - target|, а не среднюю
    абсолютную ошибку покрытия отдельных траекторий.
    """
    def aggregate(df: pd.DataFrame) -> pd.Series:
        if "site_id" not in df.columns:
            if "object" not in df.columns:
                raise KeyError("A/B scoring требует site_id или object")
            df = df.assign(site_id=df["object"].to_numpy())
        if numerator and denominator:
            use = df[df[denominator] > 0]
            num = use.groupby("site_id")[numerator].sum()
            den = use.groupby("site_id")[denominator].sum()
            values = num / den.replace(0, np.nan)
        else:
            values = df.groupby("site_id")[metric].mean()
        if transform is not None:
            values = transform(values)
        return values.dropna()

    gf = aggregate(sf); gg = aggregate(sg)
    uo = np.array(sorted(set(gf.index) & set(gg.index)))
    if len(uo) == 0:
        raise ValueError(f"Нет общих площадок с валидной метрикой {metric}")
    f_pt = float(gf.loc[uo].mean()); g_pt = float(gg.loc[uo].mean())
    draws = np.empty(nboot)
    for b in range(nboot):
        bo = rng.choice(uo, size=len(uo), replace=True) # .loc[bo] СОХРАНЯЕТ кратность
        draws[b] = float(gg.loc[bo].mean() - gf.loc[bo].mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"flow": round(f_pt, 4), "gaussian": round(g_pt, 4),
            "diff_gauss_minus_flow": round(g_pt - f_pt, 4),
            "ci95_low": round(float(lo), 4), "ci95_high": round(float(hi), 4),
            "n_sites": int(len(uo)),
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
    # Калибровка: |coverage90 − 0.90| (меньше = лучше), тот же site-clustered bootstrap. Показывает,
    # улучшает ли поток именно КАЛИБРОВКУ прогноза (а не только острые NLL/CRPS).
    rows.append({"metric": "Cov90_abs_miscal",
                 **_object_cluster_ab(sf, sg, "cov90", nboot, rng,
                                      transform=lambda s: (s - 0.90).abs(),
                                      numerator="coverage90_hits", denominator="continuation_points")})
    return pd.DataFrame(rows)


def ab_flow_vs_gaussian_pooled(pairs, config, device, n_samples: int = 32,
                               nboot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    Multi-fold A/B: пул per-sample скоров ПО ВСЕМ фолдам, затем ОДИН site-кластерный
    bootstrap. Так одна маленькая площадка в одном фолде не выдаётся за независимое повторение —
    единица ресэмплинга остаётся site_id по всему объединению out-of-fold тестов.

    :param pairs: список ``(model_flow, model_gauss, test_split)`` — по одному на fold
    :param config: конфиг; :param device: устройство
    :param n_samples: сэмплов θ; :param nboot: bootstrap-итераций; :param seed: сид
    :return: DataFrame по метрикам (как :func:`ab_flow_vs_gaussian`), но на пуле фолдов
    """
    sf_all, sg_all = [], []
    for k, (mf, mg, sp) in enumerate(pairs):
        sf_all.append(mixture_scores_per_sample(mf, sp, config, device, n_samples, seed + k))
        sg_all.append(mixture_scores_per_sample(mg, sp, config, device, n_samples, seed + k))
    sf = pd.concat(sf_all, ignore_index=True); sg = pd.concat(sg_all, ignore_index=True)
    rng = np.random.default_rng(seed)
    rows = []
    for name, col in (("Traj_mixNLL", "nll"), ("Traj_mixCRPS", "crps")):
        rows.append({"metric": name, **_object_cluster_ab(sf, sg, col, nboot, rng)})
    rows.append({"metric": "Cov90_abs_miscal",
                 **_object_cluster_ab(sf, sg, "cov90", nboot, rng,
                                      transform=lambda s: (s - 0.90).abs(),
                                      numerator="coverage90_hits", denominator="continuation_points")})
    return pd.DataFrame(rows)


def train_ab_pair(benchmark: Dict[str, object], config, device, model_kwargs: Dict[str, object],
                  epochs: int = None, seed: int = 42, mc_train_samples: int = 4,
                  mc_crps_weight: float = 0.3):
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
    base.update(mc_train_samples=int(mc_train_samples), mc_crps_weight=float(mc_crps_weight),
                calibration_steps=0)
    trained = []
    for use_flow in (True, False):
        set_global_seed(seed)
        model = DPIFlow(**base, use_flow=use_flow).to(device)
        model, _ = train_model(model, benchmark["train"], benchmark["val"], epochs=ep,
                               model_name=f"AB-{'flow' if use_flow else 'gauss'}",
                               config=config, device=device, verbose=False, scheduler="cosine")
        trained.append(model)
    return trained[0], trained[1]
