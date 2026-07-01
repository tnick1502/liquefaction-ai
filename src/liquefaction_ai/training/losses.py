"""
Функции потерь и вспомогательные операции для обучения моделей.

Содержит маскированные варианты ошибок (учитывают валидную длину наблюдения),
гауссовскую отрицательную лог-правдоподобность для вероятностных голов и утилиту
копирования состояния модели для механизма ранней остановки по лучшей валидации.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["masked_mean", "masked_mse", "masked_mae", "gaussian_nll", "gaussian_mixture_nll",
           "masked_bce_with_logits", "energy_crps", "clone_state_dict",
           "observed_aux_loss", "soft_auc_loss", "monotone_clip", "beta_nll", "censored_nliq_loss",
           "masked_censored_nliq_loss", "risk_observation_mask", "nliq_censor_mask"]


def risk_observation_mask(batch: Dict[str, torch.Tensor]):
    """Маска образцов с известным бинарным исходом by-H; legacy fallback оставлен для старых артефактов."""
    return batch.get("risk_label_observed", batch.get("n_liq_observed"))


def nliq_censor_mask(batch: Dict[str, torch.Tensor]):
    """Маска точных/право-цензурированных event-time наблюдений; не равна risk mask в общем случае."""
    return batch.get("nliq_censor_valid", batch.get("n_liq_observed"))


def soft_auc_loss(logit: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """
    Гладкая аппроксимация (1 − AUROC): парный logistic-ранжирующий лосс.

    Для каждой пары (положительный, отрицательный) штрафует случаи, когда логит риска
    положительного класса не выше логита отрицательного. Напрямую оптимизирует ранжирование
    (AUROC), в отличие от поэлементного BCE. При отсутствии одного из классов в батче — 0.

    :param logit: логиты риска, форма (batch,)
    :param label: бинарные метки разжижения, форма (batch,)
    :return: скалярный ранжирующий лосс
    """
    pos = logit[label > 0.5]
    neg = logit[label < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logit.new_zeros(())
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return F.softplus(-diff).mean()


def beta_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """
    β-NLL (Seitzer et al., 2022): гауссовская NLL, взвешенная на ``var.detach()**β``.

    При β>0 точки с малой дисперсией меньше доминируют в обновлении дисперсии, что заметно
    улучшает калибровку неопределённости по сравнению с обычной NLL (β=0). Маскируется по
    валидной длине наблюдения.

    :param mean: предсказанное среднее, форма (batch, seq_len)
    :param logvar: предсказанный логарифм дисперсии, форма (batch, seq_len)
    :param target: измеренные значения, форма (batch, seq_len)
    :param mask: маска валидной длины, форма (batch, seq_len)
    :param beta: степень взвешивания (0 = обычная NLL, 0.5 — рекомендуемое)
    :return: скалярный β-NLL
    """
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)
    weighted = nll * var.detach() ** beta
    return masked_mean(weighted, mask)


def censored_nliq_loss(nliq_pred: torch.Tensor, nliq_target: torch.Tensor,
                       liq_label: torch.Tensor) -> torch.Tensor:
    """
    Цензурированная потеря для N_liq (Tobit-стиль).

    Разжижившиеся образцы (метка 1) — обычная Smooth-L1 к наблюдаемому N_liq. Неразжижившиеся
    (правое цензурирование при N_max, метка 0) штрафуются только за **занижение** прогноза
    (предсказание разжижения раньше точки цензурирования), но не за «перелёт».

    :param nliq_pred: предсказанный нормированный N_liq, форма (batch,)
    :param nliq_target: целевой нормированный N_liq (для цензурированных — точка N_max), (batch,)
    :param liq_label: бинарная метка разжижения, форма (batch,)
    :return: скалярная цензурированная потеря
    """
    obs = F.smooth_l1_loss(nliq_pred, nliq_target, reduction="none")
    cens = F.relu(nliq_target - nliq_pred)
    return (liq_label * obs + (1.0 - liq_label) * cens).mean()


def masked_censored_nliq_loss(nliq_pred: torch.Tensor, nliq_target: torch.Tensor,
                              liq_label: torch.Tensor,
                              observed: torch.Tensor = None) -> torch.Tensor:
    """
    Цензурированная потеря N_liq с независимой event-time маской.

    Объединяет корректную обработку всех трёх типов опыта:

    * **разжижение** (``liq_label==1``): обычная Smooth-L1 к наблюдаемому N_liq;
    * **нет разжижения** (``liq_label==0``, ``observed==1``):
      право-цензурирование на фактическом последнем наблюдённом цикле —
      штраф только за **занижение** прогноза (предсказание разжижения раньше точки цензуры),
      «перелёт» не штрафуется (Tobit);
    * ``observed==0``: образец не принадлежит landmark risk set или не имеет корректного времени
      наблюдения и исключается. Стабилизация является отдельным физическим режимом и не определяет
      валидность обычной правой цензуры.

    :param nliq_pred: предсказанный нормированный N_liq, форма (batch,)
    :param nliq_target: целевой нормированный N_liq (для цензурированных — точка N_max), (batch,)
    :param liq_label: бинарная метка разжижения, форма (batch,)
    :param observed: маска ``nliq_censor_valid`` ∈ {0,1}, форма (batch,);
        ``None`` — все образцы наблюдаемы (обратная совместимость)
    :return: скалярная потеря (взвешенное среднее по наблюдаемым образцам)
    """
    obs_term = F.smooth_l1_loss(nliq_pred, nliq_target, reduction="none")
    cens = F.relu(nliq_target - nliq_pred)
    per_sample = liq_label * obs_term + (1.0 - liq_label) * cens
    if observed is None:
        return per_sample.mean()
    weight = observed.to(per_sample.dtype)
    return (per_sample * weight).sum() / weight.sum().clamp(min=1.0)


def monotone_clip(traj: torch.Tensor, lo: float = 0.0, hi: float = 1.05) -> torch.Tensor:
    """
    Спроецировать траекторию PPR(N) на монотонно неубывающую в [lo, hi].

    Реализуется как накопительный максимум по времени с клиппингом. Это **модельное допущение**
    недренированного монотонного накопления порового давления при циклическом нагружении
    (ru не убывает), а не универсальный физический закон: при дренированном/сильно переменном
    воздействии с диссипацией ru может и снижаться. В рамках принятого здесь недренированного
    допущения проекция гарантирует неубывание и ограниченность по построению.

    :param traj: траектория, форма (batch, seq_len)
    :return: монотонно неубывающая ограниченная траектория той же формы
    """
    return torch.clamp(torch.cummax(traj, dim=1).values, lo, hi)


def monotone_residual_scale(traj: torch.Tensor, residual: torch.Tensor, span: float = 0.10) -> torch.Tensor:
    """
    МОНОТОННО-СОХРАНЯЮЩАЯ обучаемая коррекция траектории PPR(N).

    Вместо аддитивного residual (``traj + r``, который может сделать кривую убывающей и нарушить
    физику) масштабирует НЕОТРИЦАТЕЛЬНЫЕ приращения базовой кривой множителем
    ``g = 1 + span·tanh(residual) ∈ [1−span, 1+span] > 0``. Поскольку приращения ``Δ = clamp(diff, ≥0)``
    остаются неотрицательными, ``traj₀ + cumsum(Δ·g)`` неубывающая ПО ПОСТРОЕНИЮ. Модель корректирует
    ТЕМП накопления (±span), но не направление → не может стать «physically unreliable» из-за residual,
    независимо от post-hoc проекции.

    :param traj: базовая (неубывающая) траектория, форма (batch, seq_len)
    :param residual: выход residual-головы, форма (batch, seq_len)
    :param span: предел относительной коррекции темпа (0.10 = ±10%)
    :return: скорректированная неубывающая траектория той же формы
    """
    base0 = traj[:, :1]
    dr = torch.clamp(traj[:, 1:] - traj[:, :-1], min=0.0)
    gate = 1.0 + span * torch.tanh(residual[:, 1:])
    return torch.cat([base0, base0 + torch.cumsum(dr * gate, dim=1)], dim=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Среднее по элементам с учётом бинарной маски валидности.

    Формула:
        masked_mean = Σ(values · mask) / max(Σ mask, 1)

    :param values: тензор значений
    :param mask: бинарная маска той же формы (1 — валидно, 0 — игнорировать)
    :return: скалярный тензор взвешенного среднего
    """
    return (values * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная среднеквадратичная ошибка.

    Формула:
        MSE = masked_mean((pred − target)²)

    :param pred: предсказанные значения
    :param target: целевые значения
    :param mask: бинарная маска валидности
    :return: скалярный тензор MSE по валидным элементам
    """
    return masked_mean((pred - target) ** 2, mask)


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная средняя абсолютная ошибка.

    Формула:
        MAE = masked_mean(|pred − target|)

    :param pred: предсказанные значения
    :param target: целевые значения
    :param mask: бинарная маска валидности
    :return: скалярный тензор MAE по валидным элементам
    """
    return masked_mean(torch.abs(pred - target), mask)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная гауссовская отрицательная лог-правдоподобность.

    Для гетероскедастичной модели с предсказанными средним μ и логарифмом дисперсии
    log σ² минимизируется
        NLL = masked_mean( 0.5·(log σ² + (target − μ)² · exp(−log σ²)) ).

    Логарифм дисперсии отсекается в [−6, 3] для численной устойчивости.

    :param mean: предсказанное среднее μ
    :param logvar: предсказанный логарифм дисперсии log σ²
    :param target: целевые значения
    :param mask: бинарная маска валидности
    :return: скалярный тензор отрицательного лог-правдоподобия
    """
    logvar = torch.clamp(logvar, min=-6.0, max=3.0)
    inv_var = torch.exp(-logvar)
    return masked_mean(0.5 * (logvar + (target - mean) ** 2 * inv_var), mask)


def masked_bce_with_logits(logit: torch.Tensor, label: torch.Tensor,
                           observed: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    BCE риск-классификации ТОЛЬКО по образцам с НАБЛЮДАЕМЫМ исходом.

    Незавершённые non-liq (``observed==0``) имеют неизвестный исход — учить их как истинный
    отрицательный (label=0) было бы ложным негативом (≈⅕ датасета). Маскируем единым образом для
    ВСЕХ моделей (proposed и baselines), иначе leaderboard методологически нечестен: метрики
    исключают эти образцы, а обучение baseline'ов — нет. ``observed=None`` → обычный BCE.

    :param logit: логиты риска, форма (N,)
    :param label: бинарные метки разжижения, форма (N,)
    :param observed: маска известности бинарного исхода (``risk_label_observed``); None → без маски
    :return: скалярный BCE (0, если в батче нет наблюдаемых исходов)
    """
    if observed is not None:
        m = observed > 0.5
        if bool(m.any()):
            return F.binary_cross_entropy_with_logits(logit[m], label[m])
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logit, label)


def gaussian_mixture_nll(means: torch.Tensor, logvars: torch.Tensor,
                         target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    NLL равновесной гауссовой смеси — proper density предиктивного распределения.

    Предиктив DPI-Flow как смесь по S сэмплам θ из conditional flow:
        p(y) = (1/S) Σ_s N(y; μ_s, σ_s²),
        NLL = − masked_mean[ log p(y) ].
    Логарифм считается устойчиво через ``logsumexp`` по компонентам. В отличие от одиночного
    :func:`gaussian_nll`, минимизация ЭТОГО лосса привязывает РАЗБРОС компонент (= неопределённость
    flow-постериора над θ) к фактической предиктивной ошибке — иначе разброс flow не калибруется и
    поток не даёт выигрыша по NLL/CRPS/покрытию.

    :param means: средние компонент μ_s, форма (S, B, T)
    :param logvars: логдисперсии компонент log σ_s², форма (S, B, T)
    :param target: наблюдаемые значения, форма (B, T)
    :param mask: бинарная маска валидности, форма (B, T)
    :return: скалярный NLL смеси
    """
    import math
    logvars = torch.clamp(logvars, min=-6.0, max=3.0)
    s = means.shape[0]
    t = target.unsqueeze(0)
    comp_logp = -0.5 * (math.log(2.0 * math.pi) + logvars
                        + (t - means) ** 2 * torch.exp(-logvars)) # (S, B, T)
    logp = torch.logsumexp(comp_logp, dim=0) - math.log(float(s)) # (B, T)
    return masked_mean(-logp, mask)


def energy_crps(samples: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Маскированная sample-оценка CRPS (энергетический скор, β=1) — proper scoring остроты.

        CRPS ≈ mean_s |y_s − y| − ½·mean_{s,s'} |y_s − y_s'|,
    где ``y_s`` — сэмплы предиктива (μ_s + σ_s·ε). Для детерминированного прогноза (все сэмплы
    равны) сводится к |ŷ − y| (MAE). Дифференцируема; награждает калиброванную остроту смеси.

    :param samples: сэмплы предиктива, форма (S, B, T)
    :param target: наблюдаемые значения, форма (B, T)
    :param mask: бинарная маска валидности, форма (B, T)
    :return: скалярный CRPS
    """
    t = target.unsqueeze(0)
    term1 = torch.abs(samples - t).mean(dim=0) # (B, T)
    diff = torch.abs(samples.unsqueeze(0) - samples.unsqueeze(1)).mean(dim=(0, 1)) # (B, T)
    return masked_mean(term1 - 0.5 * diff, mask)


def observed_aux_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    use_states: bool = True,
    w_g: float = 0.10,
    w_risk: float = 0.0,
    w_crr: float = 0.10,
) -> torch.Tensor:
    """
    Наблюдаемая вспомогательная супервизия, выводимая из измеренной кривой PPR.

    Аналог deep-supervision и калибровки риска, но с **наблюдаемыми** целями (доступными и на
    реальных данных): мягкий триггер ``g_obs`` (момент PPR≈1), мягкий риск ``risk_proxy``
    (пиковое PPR) и, опционально, измеренная граница ``crr_obs`` (с по-образцовой маской
    ``crr_obs_mask``). Все слагаемые подключаются только при наличии соответствующих целей.

    :param outputs: выходы модели (ожидаются ``risk_prob`` и, для физических моделей, ``g``, ``crr``)
    :param batch: словарь батча с наблюдаемыми целями (``risk_proxy``/``g_obs``/``crr_obs``/``mask``)
    :param use_states: применять ли супервизию латентных состояний g и границы CRR
    :param w_g: вес супервизии триггера g
    :param w_risk: вес калибровки риска к наблюдаемому риск-прокси
    :param w_crr: вес супервизии измеренной границы CRR
    :return: скалярный тензор суммарной вспомогательной потери
    """
    device = outputs["risk_prob"].device if "risk_prob" in outputs else outputs["traj_mean"].device
    total = torch.zeros((), device=device)
    # ВНИМАНИЕ (калибровка): risk_proxy = PPR_max — НЕПРЕРЫВНЫЙ прокси, а НЕ событие «разжижение by 3000».
    # Регрессия калиброванного risk_prob к нему противоречит масочному BCE (у stabilized non-liq, label=0,
    # медиана proxy≈0.63 → толкает вероятность негатива вверх) и портит Brier/ECE/AUPRC. Поэтому по
    # умолчанию ВЫКЛЮЧЕНО (w_risk=0): risk_prob калибруется ТОЛЬКО событийным масочным BCE. Если включить —
    # хотя бы маскируем по наблюдаемости (как BCE), но семантический конфликт всё равно остаётся.
    if w_risk > 0 and "risk_proxy" in batch and "risk_prob" in outputs:
        _o = risk_observation_mask(batch)
        if _o is not None and bool((_o > 0.5).any()):
            _m = _o > 0.5
            total = total + w_risk * F.mse_loss(outputs["risk_prob"][_m], batch["risk_proxy"][_m])
        elif _o is None:
            total = total + w_risk * F.mse_loss(outputs["risk_prob"], batch["risk_proxy"])
    if use_states and "g_obs" in batch and "g" in outputs:
        total = total + w_g * masked_mse(outputs["g"], batch["g_obs"], batch["mask"])
    if use_states and "crr_obs" in batch and "crr" in outputs:
        mask = batch["mask"]
        per_sample = (((outputs["crr"] - batch["crr_obs"]) ** 2) * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        crr_mask = batch.get("crr_obs_mask")
        if crr_mask is not None:
            total = total + w_crr * (per_sample * crr_mask).sum() / torch.clamp(crr_mask.sum(), min=1.0)
        else:
            total = total + w_crr * per_sample.mean()
    return total


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Создать глубокую копию весов модели на CPU.

    Используется для сохранения лучшего по валидации состояния без удержания
    ссылок на графы вычислений и без привязки к устройству.

    :param model: модель PyTorch
    :return: словарь весов (detached-копии тензоров на CPU)
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
