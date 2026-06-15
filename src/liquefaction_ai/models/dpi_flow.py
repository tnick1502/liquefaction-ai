"""
DPI-Flow — вероятностный физически-структурированный вывод параметров.

Модель не выдаёт траекторию PPR(N) как black-box последовательность, а кодирует
контекст (дескрипторы грунта, режим нагружения, короткий префикс PPR), предсказывает
распределение по физическим ODE-параметрам θ, преобразует латент через лёгкий
аффинный flow, при необходимости уточняет θ дифференцируемым шагом калибровки по
наблюдаемому префиксу и пропускает параметры через аналитический ODE-слой, который
интегрирует те же уравнения CRR/повреждения/PPR, что и синтетический генератор.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.models.blocks import ResidualMLP
from liquefaction_ai.models.heads import RiskHead, SeqLogvarHead, physics_summary
from liquefaction_ai.training.losses import gaussian_nll, masked_mse

__all__ = ["ConditionalAffineFlow", "AnalyticalLiquefactionLayer", "DPIFlow"]


class ConditionalAffineFlow(nn.Module):
    """
    Условный аффинный поток (лёгкий нормализующий flow).

    Два аффинных преобразования латента с масштабом и сдвигом, обусловленными
    контекстом. Масштаб ограничен ``0.35·tanh(...)`` для устойчивости обучения.
    """

    def __init__(self, latent_dim: int, context_dim: int):
        """
        :param latent_dim: размерность латентного вектора θ
        :param context_dim: размерность контекстного представления
        """
        super().__init__()
        self.scale_1 = nn.Linear(context_dim, latent_dim)
        self.shift_1 = nn.Linear(context_dim, latent_dim)
        self.scale_2 = nn.Linear(context_dim, latent_dim)
        self.shift_2 = nn.Linear(context_dim, latent_dim)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Применить два условных аффинных преобразования к латенту.

        :param z: латентный вектор формы (batch, latent_dim)
        :param context: контекст формы (batch, context_dim)
        :return: преобразованный латент формы (batch, latent_dim)
        """
        s1 = 0.35 * torch.tanh(self.scale_1(context))
        h = z * torch.exp(s1) + self.shift_1(context)
        s2 = 0.35 * torch.tanh(self.scale_2(context))
        h = h * torch.exp(s2) + self.shift_2(context)
        return h


class AnalyticalLiquefactionLayer(nn.Module):
    """
    Аналитический дифференцируемый ODE-слой разжижения.

    По вектору параметров θ распаковывает физические величины, строит границу CRR как
    дифференцируемую смесь четырёх семейств и пошагово интегрирует связанные ODE
    повреждения z и порового давления r с мягким триггером g, повторяя структуру
    синтетического генератора, но в виде вычислительного графа PyTorch.
    """

    def __init__(self, seq_len: int, max_cycle_reference: float):
        """
        :param seq_len: длина временной последовательности
        :param max_cycle_reference: опорное N для логарифмической нормировки N_liq
        """
        super().__init__()
        self.seq_len = seq_len
        self.max_cycle_reference = max_cycle_reference

    def unpack_theta(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Распаковать сырой вектор θ в физически ограниченные параметры.

        Каждая компонента θ через sigmoid/softmax отображается в допустимый диапазон
        соответствующего физического параметра (веса смеси CRR, амплитуды, скорости,
        хвосты, параметры ODE повреждения и роста PPR до/после события, триггер, шум).

        :param theta: сырой тензор параметров формы (batch, theta_dim)
        :return: словарь физически ограниченных параметров
        """
        weights = torch.softmax(theta[:, 0:4], dim=-1)
        amp = 0.03 + 0.28 * torch.sigmoid(theta[:, 4:8])
        rate = 0.05 + 1.20 * torch.sigmoid(theta[:, 8:12])
        tail = 0.005 + 0.18 * torch.sigmoid(theta[:, 12:16])
        lambda_damage = 0.0001 + 0.0050 * torch.sigmoid(theta[:, 16])
        exponent_m = 1.0 + 2.5 * torch.sigmoid(theta[:, 17])
        exponent_nu = 1.0 + 1.8 * torch.sigmoid(theta[:, 18])
        alpha = 0.0004 + 0.0060 * torch.sigmoid(theta[:, 19])
        beta = 0.01 + 0.18 * torch.sigmoid(theta[:, 20])
        gamma = 0.00005 + 0.0020 * torch.sigmoid(theta[:, 21])
        exponent_p = 0.95 + 1.9 * torch.sigmoid(theta[:, 22])
        tau = 3.0 + 42.0 * torch.sigmoid(theta[:, 23])
        alpha_post = 0.0004 + 0.0080 * torch.sigmoid(theta[:, 24])
        beta_post = 0.01 + 0.16 * torch.sigmoid(theta[:, 25])
        gamma_post = 0.00005 + 0.0024 * torch.sigmoid(theta[:, 26])
        exponent_p_post = 0.85 + 1.4 * torch.sigmoid(theta[:, 27])
        kappa = 6.0 + 20.0 * torch.sigmoid(theta[:, 28])
        z0 = 0.35 + 0.50 * torch.sigmoid(theta[:, 29])
        noise = 0.01 + 0.08 * torch.sigmoid(theta[:, 30])
        return {
            "weights": weights,
            "amp": amp,
            "rate": rate,
            "tail": tail,
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
            "kappa": kappa,
            "z0": z0,
            "noise": noise,
        }

    def compute_crr(self, params: Dict[str, torch.Tensor], cycles: torch.Tensor) -> torch.Tensor:
        """
        Построить границу CRR(N) как дифференцируемую смесь четырёх семейств.

        Семейства: гиперболическое, степенное, экспоненциальное, логарифмическое;
        смешиваются весами softmax. Результат отсекается снизу для положительности.

        :param params: распакованные параметры (веса, амплитуды, скорости, хвосты)
        :param cycles: сетка числа циклов, форма (batch, seq_len)
        :return: граница CRR(N), форма (batch, seq_len)
        """
        n_rel = cycles / 100.0
        crr_h = params["tail"][:, 0:1] + params["amp"][:, 0:1] / (1.0 + params["rate"][:, 0:1] * n_rel)
        crr_p = params["tail"][:, 1:2] + params["amp"][:, 1:2] * torch.pow(1.0 + n_rel, -(0.15 + params["rate"][:, 1:2]))
        crr_e = params["tail"][:, 2:3] + params["amp"][:, 2:3] * torch.exp(-params["rate"][:, 2:3] * n_rel)
        crr_l = params["tail"][:, 3:4] + params["amp"][:, 3:4] / (1.0 + params["rate"][:, 3:4] * torch.log1p(4.0 * n_rel))
        crr = (
            params["weights"][:, 0:1] * crr_h
            + params["weights"][:, 1:2] * crr_p
            + params["weights"][:, 2:3] * crr_e
            + params["weights"][:, 3:4] * crr_l
        )
        return torch.clamp(crr, min=1e-4)

    def soft_first_hitting(self, r: torch.Tensor, g: torch.Tensor, cycles: torch.Tensor) -> torch.Tensor:
        """
        Дифференцируемая оценка числа циклов до разжижения N_liq.

        Вероятность события на шаге берётся как максимум из сглаженного достижения
        порога PPR и триггера g. По выживаемости (cumprod) вычисляется масса первого
        достижения; ожидание числа циклов с остаточной массой на конце горизонта даёт
        дифференцируемый аналог момента первого достижения.

        :param r: траектория PPR, форма (batch, seq_len)
        :param g: траектория триггера, форма (batch, seq_len)
        :param cycles: сетка числа циклов, форма (batch, seq_len)
        :return: оценка N_liq, форма (batch,)
        """
        event_prob = torch.maximum(torch.sigmoid(12.0 * (r - 0.985)), g)
        shifted_survival = torch.cumprod(
            torch.cat([torch.ones_like(event_prob[:, :1]), torch.clamp(1.0 - event_prob[:, :-1], min=1e-4)], dim=1),
            dim=1,
        )
        event_mass = event_prob * shifted_survival
        residual_mass = torch.clamp(1.0 - event_mass.sum(dim=1), min=0.0, max=1.0)
        nliq = (event_mass * cycles).sum(dim=1) + residual_mass * cycles[:, -1]
        return nliq

    def simulate(self, theta: torch.Tensor, cycles: torch.Tensor, delta_cycles: torch.Tensor, csr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Проинтегрировать ODE повреждения и PPR по параметрам θ.

        Явная схема Эйлера по сетке циклов: на каждом шаге вычисляются отношение
        CSR/CRR, приращение повреждения dz, до- и постсобытийные приращения PPR,
        смешиваемые мягким триггером g. Возвращает траектории и производные величины.

        :param theta: сырой тензор параметров формы (batch, theta_dim)
        :param cycles: сетка числа циклов, форма (batch, seq_len)
        :param delta_cycles: приращения ΔN, форма (batch, seq_len)
        :param csr: история CSR(N), форма (batch, seq_len)
        :return: словарь с ``traj_mean`` (PPR), ``traj_logvar``, ``z``, ``g``, ``crr``,
                 ``nliq``, ``nliq_norm`` и распакованными ``theta_params``
        """
        params = self.unpack_theta(theta)
        crr = self.compute_crr(params, cycles)
        batch_size, seq_len = cycles.shape
        eps = 1e-6

        z_states: List[torch.Tensor] = [torch.zeros(batch_size, device=cycles.device)]
        r_states: List[torch.Tensor] = [torch.zeros(batch_size, device=cycles.device)]
        g_states: List[torch.Tensor] = []

        for step in range(seq_len):
            z_curr = z_states[-1]
            r_curr = r_states[-1]
            g_curr = torch.sigmoid(params["kappa"] * (z_curr - params["z0"]))
            g_states.append(g_curr)
            if step == seq_len - 1:
                break

            ratio = csr[:, step] / (crr[:, step] + eps)
            phi = F.softplus(6.0 * (ratio - 0.90)) / 6.0
            dz = (
                params["lambda_damage"]
                * torch.pow(torch.clamp(ratio, min=eps), params["m"])
                * torch.pow(torch.clamp(1.0 - z_curr, min=eps), params["nu"])
            )
            pre_event = (
                params["alpha"] * phi * torch.pow(torch.clamp(1.0 - r_curr, min=eps), params["p"])
                + params["beta"] / (cycles[:, step] + params["tau"])
                - params["gamma"] * r_curr
            )
            post_event = (
                params["alpha_post"] * torch.pow(torch.clamp(1.0 - r_curr, min=eps), params["p_post"])
                + params["beta_post"] / (cycles[:, step] + 0.70 * params["tau"])
                - params["gamma_post"] * r_curr
            )
            dr = (1.0 - g_curr) * pre_event + g_curr * post_event

            z_next = torch.clamp(z_curr + delta_cycles[:, step + 1] * dz, min=0.0, max=0.999)
            r_next = torch.clamp(r_curr + delta_cycles[:, step + 1] * dr, min=0.0, max=1.02)
            z_states.append(z_next)
            r_states.append(r_next)

        z = torch.stack(z_states, dim=1)
        r = torch.stack(r_states, dim=1)
        g = torch.stack(g_states, dim=1)
        nliq = self.soft_first_hitting(r, g, cycles)
        logvar = 2.0 * torch.log(params["noise"].unsqueeze(1).expand_as(r))
        return {
            "traj_mean": r,
            "traj_logvar": logvar,
            "z": z,
            "g": g,
            "crr": crr,
            "nliq": nliq,
            "nliq_norm": torch.log1p(nliq) / torch.log1p(
                torch.as_tensor(self.max_cycle_reference, device=nliq.device, dtype=nliq.dtype)
            ),
            "theta_params": params,
        }


class DPIFlow(nn.Module):
    """
    DPI-Flow: вероятностный вывод физических параметров через ODE-слой.

    Энкодер контекста выдаёт среднее и логдисперсию распределения θ; при
    ``probabilistic=True`` используется репараметризация. Аффинный flow преобразует
    латент, опциональный внутренний шаг калибровки подстраивает θ под наблюдаемый
    префикс, после чего аналитический ODE-слой моделирует траектории. При
    ``use_analytical_layer=False`` модель работает как прямой black-box декодер
    (для абляции вклада физического слоя).
    """

    def __init__(
        self,
        static_dim: int,
        prefix_dim: int,
        seq_len: int,
        prefix_len: int,
        max_cycle_reference: float,
        theta_dim: int = 31,
        hidden_dim: int = 160,
        probabilistic: bool = True,
        calibration_steps: int = 2,
        calibration_lr: float = 0.10,
        use_analytical_layer: bool = True,
    ):
        """
        :param static_dim: размерность статических признаков
        :param prefix_dim: размерность сводки префикса
        :param seq_len: длина временной последовательности
        :param prefix_len: длина наблюдаемого префикса
        :param max_cycle_reference: опорное N для нормировки N_liq
        :param theta_dim: размерность вектора физических параметров θ
        :param hidden_dim: размерность скрытого представления энкодера
        :param probabilistic: использовать ли вероятностную голову (репараметризацию)
        :param calibration_steps: число шагов внутренней калибровки θ по префиксу
        :param calibration_lr: шаг градиентной калибровки θ
        :param use_analytical_layer: использовать ли аналитический ODE-слой (иначе black-box декодер)
        """
        super().__init__()
        self.theta_dim = theta_dim
        self.seq_len = seq_len
        self.probabilistic = probabilistic
        self.calibration_steps = calibration_steps
        self.calibration_lr = calibration_lr
        self.use_analytical_layer = use_analytical_layer
        self.prefix_len = prefix_len
        self.max_cycle_reference = max_cycle_reference

        context_dim = static_dim + prefix_dim + 2 * self.prefix_len
        self.context_encoder = ResidualMLP(context_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.mu_head = nn.Linear(hidden_dim, theta_dim)
        self.logvar_head = nn.Linear(hidden_dim, theta_dim)
        self.flow = ConditionalAffineFlow(theta_dim, hidden_dim)
        self.ode_layer = AnalyticalLiquefactionLayer(seq_len=seq_len, max_cycle_reference=max_cycle_reference)

        # Обучаемые головы: калиброванный риск и гетероскедастичная неопределённость
        self.risk_head = RiskHead(hidden_dim)
        self.logvar_head_seq = SeqLogvarHead(hidden_dim, seq_len)

        self.direct_decoder = ResidualMLP(context_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.direct_traj_head = nn.Linear(hidden_dim, seq_len)
        self.direct_logvar_head = nn.Linear(hidden_dim, seq_len)
        self.direct_risk_head = nn.Linear(hidden_dim, 1)
        self.direct_nliq_head = nn.Linear(hidden_dim, 1)

    def build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Собрать контекстный вектор из статических, префиксных и масочных признаков.

        :param batch: словарь батча
        :return: контекст формы (batch, static_dim + prefix_dim + 2·prefix_len)
        """
        prefix_values = batch["prefix_obs"][:, : self.prefix_len]
        prefix_mask = batch["prefix_mask"][:, : self.prefix_len]
        return torch.cat([batch["static"], batch["prefix_summary"], prefix_values, prefix_mask], dim=-1)

    def sample_theta(self, encoded_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Сэмплировать вектор параметров θ из предсказанного распределения.

        При вероятностном режиме на обучении применяется репараметризация
        ``μ + exp(0.5·logσ²)·ε``; на инференсе берётся среднее. Затем латент проходит
        через аффинный flow.

        :param encoded_context: закодированный контекст формы (batch, hidden_dim)
        :return: кортеж (θ, μ, logσ²)
        """
        mu = self.mu_head(encoded_context)
        raw_logvar = torch.clamp(self.logvar_head(encoded_context), min=-5.0, max=3.0)
        if self.probabilistic:
            eps = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
            latent = mu + torch.exp(0.5 * raw_logvar) * eps
        else:
            latent = mu
            raw_logvar = torch.zeros_like(mu)
        theta = self.flow(latent, encoded_context)
        return theta, mu, raw_logvar

    def calibrate_theta(self, theta: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Уточнить θ несколькими шагами градиентного спуска по ошибке на префиксе.

        Это дифференцируемая идентификация системы: θ подстраивается так, чтобы
        смоделированный ODE-слоем префикс PPR соответствовал наблюдаемому. Поправка
        переносится на исходный θ без удержания внутреннего графа (straight-through).

        :param theta: исходный вектор параметров формы (batch, theta_dim)
        :param batch: словарь батча (наблюдаемый префикс и маски)
        :return: уточнённый вектор θ
        """
        if self.calibration_steps <= 0 or not self.use_analytical_layer:
            return theta

        theta_anchor = theta
        theta_work = theta.detach().clone()
        for _ in range(self.calibration_steps):
            theta_work = theta_work.detach().requires_grad_(True)
            with torch.enable_grad():
                sim = self.ode_layer.simulate(theta_work, batch["cycles"], batch["delta_cycles"], batch["csr"])
                prefix_loss = masked_mse(sim["traj_mean"], batch["prefix_obs"], batch["prefix_mask"])
            grad = torch.autograd.grad(prefix_loss, theta_work, create_graph=False, retain_graph=False)[0]
            theta_work = theta_work - self.calibration_lr * grad

        return theta_anchor + (theta_work.detach() - theta_anchor.detach())

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Прямой проход по батчу: вывод θ, калибровка и моделирование траекторий.

        :param batch: словарь батча
        :return: словарь выходов (траектория PPR, риск, N_liq, скрытые состояния, KL и θ)
        """
        context = self.build_context(batch)
        encoded = self.context_encoder(context)
        theta, mu, raw_logvar = self.sample_theta(encoded)
        theta = self.calibrate_theta(theta, batch)

        if self.use_analytical_layer:
            outputs = self.ode_layer.simulate(theta, batch["cycles"], batch["delta_cycles"], batch["csr"])
            summary = physics_summary(outputs["traj_mean"], outputs["z"], outputs["g"], outputs["nliq_norm"])
            # Физический prior риска + обучаемая остаточная поправка (калибровка)
            risk_prior = 6.0 * (
                0.50 * outputs["traj_mean"].amax(dim=1)
                + 0.25 * outputs["g"].amax(dim=1)
                + 0.25 * outputs["z"].amax(dim=1)
                - 0.75
            )
            risk_logit = risk_prior + self.risk_head(encoded, summary)
            # Гетероскедастичная неопределённость (зависит от сценария и шага)
            outputs["traj_logvar"] = self.logvar_head_seq(encoded)
            outputs.update(
                {
                    "risk_logit": risk_logit,
                    "risk_prob": torch.sigmoid(risk_logit),
                    "kl": 0.5 * (torch.exp(raw_logvar) + mu.pow(2) - 1.0 - raw_logvar).mean(dim=1),
                    "theta_raw": theta,
                    "mu": mu,
                    "raw_logvar": raw_logvar,
                }
            )
            return outputs

        decoded = self.direct_decoder(context)
        traj_mean = torch.sigmoid(self.direct_traj_head(decoded))
        traj_logvar = torch.clamp(self.direct_logvar_head(decoded), min=-6.0, max=2.0)
        risk_logit = self.direct_risk_head(decoded).squeeze(-1)
        nliq_norm = torch.sigmoid(self.direct_nliq_head(decoded).squeeze(-1))
        mcr = torch.as_tensor(self.max_cycle_reference, device=traj_mean.device, dtype=traj_mean.dtype)
        return {
            "traj_mean": traj_mean,
            "traj_logvar": traj_logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq_norm": nliq_norm,
            "nliq": torch.expm1(nliq_norm * torch.log1p(mcr)),
            "g": torch.sigmoid(traj_mean * 8.0 - 4.0),
            "z": torch.sigmoid(traj_mean * 6.0 - 2.5),
            "crr": torch.ones_like(traj_mean) * 0.20,
            "kl": 0.5 * (torch.exp(raw_logvar) + mu.pow(2) - 1.0 - raw_logvar).mean(dim=1),
            "theta_raw": theta,
            "mu": mu,
            "raw_logvar": raw_logvar,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Вычислить суммарную функцию потерь и выходы по батчу.

        Складывает гауссовскую NLL траектории, BCE-риск, Smooth-L1 по N_liq и
        физически-мотивированные регуляризаторы (монотонность CRR, ограниченность,
        гладкость второго порядка) и KL-дивергенцию вероятностной головы.

        :param batch: словарь батча с таргетами и масками
        :return: словарь выходов с добавленным ключом ``loss``
        """
        outputs = self.forward_batch(batch)
        traj_loss = gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch["mask"])
        nliq_loss = F.smooth_l1_loss(outputs["nliq_norm"], batch["n_liq_norm"])
        risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["label"])
        # Калибровка риска к мягкой метке риск-скора
        risk_cal = F.mse_loss(outputs["risk_prob"], batch["risk_true"])
        monotonicity = torch.relu(outputs["crr"][:, 1:] - outputs["crr"][:, :-1]).mean()
        boundedness = (torch.relu(outputs["traj_mean"] - 1.02) + torch.relu(-outputs["traj_mean"])).mean()
        smoothness = torch.abs(
            outputs["traj_mean"][:, 2:] - 2.0 * outputs["traj_mean"][:, 1:-1] + outputs["traj_mean"][:, :-2]
        ).mean()
        kl_loss = outputs["kl"].mean() if self.probabilistic else torch.zeros(1, device=outputs["traj_mean"].device).squeeze()

        # Глубокая супервизия скрытой физики (только при аналитическом ODE-слое)
        if self.use_analytical_layer:
            z_loss = masked_mse(outputs["z"], batch["z_true"], batch["mask"])
            g_loss = masked_mse(outputs["g"], batch["g_true"], batch["mask"])
            crr_loss = masked_mse(outputs["crr"], batch["crr_mix_true"], batch["mask"])
            physics_sup = 0.10 * z_loss + 0.06 * g_loss + 0.05 * crr_loss
        else:
            physics_sup = torch.zeros((), device=outputs["traj_mean"].device)

        loss = (
            traj_loss
            + 0.40 * risk_loss
            + 0.20 * risk_cal
            + 0.25 * nliq_loss
            + physics_sup
            + 0.03 * monotonicity
            + 0.03 * boundedness
            + 0.01 * smoothness
            + 0.02 * kl_loss
        )
        outputs["loss"] = loss
        return outputs
