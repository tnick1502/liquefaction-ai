"""
EVT-NeuralSSM — событийно-переключаемая нейронная модель пространства состояний.

Модель эволюционирует структурированное скрытое состояние h_t = [z_t, r_t, c_t]
(повреждение, поровое давление, вспомогательная память) под управлением физики
CRR/CSR. До- и постсобытийная динамики смешиваются мягким триггером
g_t = sigmoid(κ·(z_t − z0)), что даёт дифференцируемое переключение режима вместо
хрупкого порога. Небольшая нейронная поправка действует только через вспомогательное
состояние c_t, сохраняя физическую доминанту.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.models.blocks import ResidualMLP
from liquefaction_ai.training.losses import gaussian_nll

__all__ = ["EVTNeuralSSM"]


class EVTNeuralSSM(nn.Module):
    """
    Событийно-активируемая структурированная модель пространства состояний.

    Контекст кодируется остаточным MLP в физические параметры; рекуррентно
    интегрируются связанные уравнения повреждения z и порового давления r с мягким
    переключением до/после события. Флаги конструктора позволяют отключать ключевые
    механизмы для абляционных исследований.
    """

    def __init__(
        self,
        static_dim: int,
        prefix_dim: int,
        seq_dim: int,
        seq_len: int,
        prefix_len: int,
        max_cycle_reference: float,
        hidden_dim: int = 144,
        use_trigger_head: bool = True,
        structured_post_event: bool = True,
        use_crr_damage: bool = True,
    ):
        """
        :param static_dim: размерность статических признаков
        :param prefix_dim: размерность сводки префикса
        :param seq_dim: размерность последовательностных признаков на шаге
        :param seq_len: длина временной последовательности
        :param prefix_len: длина наблюдаемого префикса
        :param max_cycle_reference: опорное N для нормировки N_liq
        :param hidden_dim: размерность скрытого представления энкодера
        :param use_trigger_head: использовать ли мягкий триггер события g
        :param structured_post_event: использовать ли структурированную постсобытийную динамику
        :param use_crr_damage: использовать ли CRR-основанное уравнение повреждения z
        """
        super().__init__()
        self.seq_len = seq_len
        self.use_trigger_head = use_trigger_head
        self.structured_post_event = structured_post_event
        self.use_crr_damage = use_crr_damage
        self.prefix_len = prefix_len
        self.max_cycle_reference = max_cycle_reference

        context_dim = static_dim + prefix_dim + 2 * self.prefix_len
        self.context_encoder = ResidualMLP(context_dim, hidden_dim=hidden_dim, depth=3, dropout=0.10)
        self.param_head = nn.Linear(hidden_dim, 33)
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim + seq_dim + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def build_context(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Собрать контекстный вектор из статических, префиксных и масочных признаков.

        :param batch: словарь батча
        :return: контекст формы (batch, static_dim + prefix_dim + 2·prefix_len)
        """
        return torch.cat(
            [
                batch["static"],
                batch["prefix_summary"],
                batch["prefix_obs"][:, : self.prefix_len],
                batch["prefix_mask"][:, : self.prefix_len],
            ],
            dim=-1,
        )

    def unpack_params(self, theta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Распаковать сырой вектор параметров в физически ограниченные величины.

        Включает веса смеси CRR, амплитуды/скорости/хвосты, параметры ODE повреждения
        и роста PPR до и после события, скорость затухания вспомогательной памяти и шум.

        :param theta: сырой тензор параметров формы (batch, 33)
        :return: словарь физически ограниченных параметров
        """
        weights = torch.softmax(theta[:, 0:4], dim=-1)
        amp = 0.03 + 0.28 * torch.sigmoid(theta[:, 4:8])
        rate = 0.05 + 1.15 * torch.sigmoid(theta[:, 8:12])
        tail = 0.005 + 0.18 * torch.sigmoid(theta[:, 12:16])
        lambda_pre = 0.0001 + 0.0050 * torch.sigmoid(theta[:, 16])
        exponent_m = 1.0 + 2.4 * torch.sigmoid(theta[:, 17])
        exponent_nu = 1.0 + 1.8 * torch.sigmoid(theta[:, 18])
        alpha_pre = 0.0004 + 0.0065 * torch.sigmoid(theta[:, 19])
        beta_pre = 0.01 + 0.18 * torch.sigmoid(theta[:, 20])
        gamma_pre = 0.00005 + 0.0020 * torch.sigmoid(theta[:, 21])
        exponent_p_pre = 0.95 + 1.8 * torch.sigmoid(theta[:, 22])
        tau_pre = 3.0 + 42.0 * torch.sigmoid(theta[:, 23])
        lambda_post = 0.0002 + 0.0075 * torch.sigmoid(theta[:, 24])
        alpha_post = 0.0004 + 0.0080 * torch.sigmoid(theta[:, 25])
        beta_post = 0.01 + 0.16 * torch.sigmoid(theta[:, 26])
        gamma_post = 0.00005 + 0.0024 * torch.sigmoid(theta[:, 27])
        exponent_p_post = 0.80 + 1.6 * torch.sigmoid(theta[:, 28])
        kappa = 6.0 + 20.0 * torch.sigmoid(theta[:, 29])
        z0 = 0.35 + 0.50 * torch.sigmoid(theta[:, 30])
        c_decay = 0.50 + 0.45 * torch.sigmoid(theta[:, 31])
        noise = 0.01 + 0.08 * torch.sigmoid(theta[:, 32])
        return {
            "weights": weights,
            "amp": amp,
            "rate": rate,
            "tail": tail,
            "lambda_pre": lambda_pre,
            "m": exponent_m,
            "nu": exponent_nu,
            "alpha_pre": alpha_pre,
            "beta_pre": beta_pre,
            "gamma_pre": gamma_pre,
            "p_pre": exponent_p_pre,
            "tau_pre": tau_pre,
            "lambda_post": lambda_post,
            "alpha_post": alpha_post,
            "beta_post": beta_post,
            "gamma_post": gamma_post,
            "p_post": exponent_p_post,
            "kappa": kappa,
            "z0": z0,
            "c_decay": c_decay,
            "noise": noise,
        }

    def compute_crr(self, params: Dict[str, torch.Tensor], cycles: torch.Tensor) -> torch.Tensor:
        """
        Построить границу CRR(N) как дифференцируемую смесь четырёх семейств.

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
        return (event_mass * cycles).sum(dim=1) + residual_mass * cycles[:, -1]

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Прямой проход: рекуррентное интегрирование состояний с переключением режима.

        На каждом шаге вычисляются отношение CSR/CRR, нейронная поправка (через
        вспомогательную память c), до- и постсобытийные приращения z и r, смешиваемые
        триггером g. Возвращает траектории, риск и оценку N_liq.

        :param batch: словарь батча
        :return: словарь выходов: ``traj_mean``, ``traj_logvar``, ``risk_logit``,
                 ``risk_prob``, ``nliq``, ``nliq_norm``, ``z``, ``g``, ``c``, ``crr``
        """
        context = self.build_context(batch)
        encoded = self.context_encoder(context)
        params = self.unpack_params(self.param_head(encoded))
        crr = self.compute_crr(params, batch["cycles"])

        batch_size, seq_len = batch["cycles"].shape
        eps = 1e-6
        z_states: List[torch.Tensor] = [torch.zeros(batch_size, device=batch["cycles"].device)]
        r_states: List[torch.Tensor] = [batch["prefix_obs"][:, 0]]
        c_states: List[torch.Tensor] = [torch.zeros(batch_size, device=batch["cycles"].device)]
        g_states: List[torch.Tensor] = []

        for step in range(seq_len):
            z_curr = z_states[-1]
            r_curr = r_states[-1]
            c_curr = c_states[-1]
            g_step = torch.sigmoid(params["kappa"] * (z_curr - params["z0"])) if self.use_trigger_head else torch.zeros_like(z_curr)
            g_states.append(g_step)
            if step == seq_len - 1:
                break

            ratio = batch["csr"][:, step] / (crr[:, step] + eps)
            phi = F.softplus(6.0 * (ratio - 0.90)) / 6.0

            correction_input = torch.cat(
                [
                    encoded,
                    batch["seq_in"][:, step, :],
                    z_curr[:, None],
                    r_curr[:, None],
                    c_curr[:, None],
                    g_step[:, None],
                ],
                dim=-1,
            )
            correction = 0.05 * torch.tanh(self.correction_net(correction_input))
            dz_corr, dr_corr, dc_corr = correction.unbind(dim=-1)

            if self.use_crr_damage:
                dz_pre = (
                    params["lambda_pre"]
                    * torch.pow(torch.clamp(ratio, min=eps), params["m"])
                    * torch.pow(torch.clamp(1.0 - z_curr, min=eps), params["nu"])
                    + dz_corr
                )
            else:
                dz_pre = 0.10 * torch.tanh(dz_corr) + 0.02 * phi

            dr_pre = (
                params["alpha_pre"] * phi * torch.pow(torch.clamp(1.0 - r_curr, min=eps), params["p_pre"])
                + params["beta_pre"] / (batch["cycles"][:, step] + params["tau_pre"])
                - params["gamma_pre"] * r_curr
                + dr_corr
            )

            if self.structured_post_event:
                dz_post = params["lambda_post"] * torch.clamp(1.0 - z_curr, min=eps) + 0.5 * dz_corr
                dr_post = (
                    params["alpha_post"] * torch.pow(torch.clamp(1.0 - r_curr, min=eps), params["p_post"])
                    + params["beta_post"] / (batch["cycles"][:, step] + 0.70 * params["tau_pre"])
                    - params["gamma_post"] * r_curr
                    + 0.5 * dr_corr
                )
            else:
                dz_post = dz_pre
                dr_post = dr_pre

            dz = (1.0 - g_step) * dz_pre + g_step * dz_post
            dr = (1.0 - g_step) * dr_pre + g_step * dr_post
            z_next = torch.clamp(z_curr + batch["delta_cycles"][:, step + 1] * dz, min=0.0, max=0.999)
            r_next = torch.clamp(r_curr + batch["delta_cycles"][:, step + 1] * dr, min=0.0, max=1.02)
            c_next = torch.tanh(params["c_decay"] * c_curr + dc_corr)
            z_states.append(z_next)
            r_states.append(r_next)
            c_states.append(c_next)

        z = torch.stack(z_states, dim=1)
        r = torch.stack(r_states, dim=1)
        c = torch.stack(c_states, dim=1)
        g = torch.stack(g_states, dim=1)
        nliq = self.soft_first_hitting(r, g, batch["cycles"])
        mcr = torch.as_tensor(self.max_cycle_reference, device=nliq.device, dtype=nliq.dtype)
        nliq_norm = torch.log1p(nliq) / torch.log1p(mcr)
        risk_logit = 6.0 * (0.50 * r.amax(dim=1) + 0.25 * g.amax(dim=1) + 0.25 * z.amax(dim=1) - 0.75)
        traj_logvar = 2.0 * torch.log(params["noise"].unsqueeze(1).expand_as(r))
        return {
            "traj_mean": r,
            "traj_logvar": traj_logvar,
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "nliq": nliq,
            "nliq_norm": nliq_norm,
            "z": z,
            "g": g,
            "c": c,
            "crr": crr,
        }

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Вычислить суммарную функцию потерь и выходы по батчу.

        Складывает гауссовскую NLL траектории, BCE по зоне триггера и риску, Smooth-L1
        по N_liq и регуляризаторы переключения/гладкости/ограниченности состояний.

        :param batch: словарь батча с таргетами и масками
        :return: словарь выходов с добавленным ключом ``loss``
        """
        outputs = self.forward_batch(batch)
        traj_loss = gaussian_nll(outputs["traj_mean"], outputs["traj_logvar"], batch["r_obs"], batch["mask"])
        trigger_loss = F.binary_cross_entropy(outputs["g"], batch["trigger_zone"])
        risk_loss = F.binary_cross_entropy_with_logits(outputs["risk_logit"], batch["label"])
        nliq_loss = F.smooth_l1_loss(outputs["nliq_norm"], batch["n_liq_norm"])
        switch_reg = torch.abs(outputs["g"][:, 1:] - outputs["g"][:, :-1]).mean()
        state_smoothness = (
            torch.abs(outputs["traj_mean"][:, 2:] - 2.0 * outputs["traj_mean"][:, 1:-1] + outputs["traj_mean"][:, :-2]).mean()
            + torch.abs(outputs["z"][:, 2:] - 2.0 * outputs["z"][:, 1:-1] + outputs["z"][:, :-2]).mean()
        )
        boundedness = (
            torch.relu(outputs["traj_mean"] - 1.02)
            + torch.relu(-outputs["traj_mean"])
            + torch.relu(outputs["z"] - 1.0)
            + torch.relu(-outputs["z"])
        ).mean()
        loss = (
            traj_loss
            + 0.30 * trigger_loss
            + 0.30 * risk_loss
            + 0.25 * nliq_loss
            + 0.02 * switch_reg
            + 0.01 * state_smoothness
            + 0.03 * boundedness
        )
        outputs["loss"] = loss
        return outputs
