"""
DPI-EVT — объединённая модель: идентификация физических параметров (DPI) + событийный движок (EVT).

Идея (объединение двух архитектур проекта):
- **DPI как модуль идентификации параметров.** Энкодер по статическим свойствам грунта и
  наблюдаемому префиксу PPR выводит апостериор θ (+ нормализующий поток); опционально θ
  уточняется 1–2 шагами дифференцируемой калибровки по наблюдаемому префиксу через сам движок.
- **EVT как основной движок.** Из θ разворачивается единое латентное состояние: повреждение
  z(N) (damage), поровое давление r(N)=ru(N)=PPR(N), триггер g(N), число циклов N_liq.
- **Дифференцируемая физика CRR.** Отчётная CRR(N) — физический степенной закон (следствие
  закона повреждения), а не эмпирическая подгонка. Режим ``decoupled`` разделяет сопротивление,
  ведущее динамику, и отчётную CRR (связаны регуляризатором), снимая компромисс «траектория↔CRR».
- **Когерентность единого состояния.** N_liq берётся из самой кривой PPR (момент пересечения
  порога), а joint-consistency лоссы связывают выходы: CRR(N_liq)≈CSR, Damage(N_liq)≈порог.

Из одного состояния одновременно предсказываются PPR(N)=ru(N), Damage(N), CRR(N), N_liq и риск.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from liquefaction_ai.models.dpi_flow import ConditionalCouplingFlow, flow_kl_per_dim
from liquefaction_ai.models.evt_ssm import EVTNeuralSSM
from liquefaction_ai.models.heads import physics_summary
from liquefaction_ai.training.losses import (energy_crps, gaussian_nll, gaussian_mixture_nll,
                                             interpolated_crossing,
                                             masked_bce_with_logits, masked_censored_nliq_loss, masked_mean,
                                             masked_mse, monotone_clip, monotone_residual_scale,
                                             normalized_free_increments, nliq_censor_mask,
                                             risk_observation_mask, soft_auc_loss)

__all__ = ["DPIEvtNet"]


class DPIEvtNet(EVTNeuralSSM):
    """Идентификация параметров (DPI) + событийный движок (EVT) + дифференцируемая CRR-физика."""

    def __init__(self, static_dim: int, prefix_dim: int, seq_dim: int, seq_len: int, prefix_len: int,
                 max_cycle_reference: float, hidden_dim: int = 144, probabilistic: bool = True,
                 use_flow: bool = True, crr_from_damage: bool = True, crr_mode: str = None,
                 nliq_from_curve: bool = True, calibration_steps: int = 0, calibration_lr: float = 0.05,
                 use_traj_residual: bool = False, traj_residual_span: float = 0.10,
                 use_free_increment: bool = False, liq_threshold: float = 0.95,
                 use_observed_aux_loss: bool = True,
                 mc_train_samples: int = 0, mc_crps_weight: float = 0.0, mc_predict_samples: int = 0,
                 report_nliq_from_curve: bool = True, nliq_head_aux_weight: float = 0.10,
                 **kwargs):
        """
        :param crr_mode: "damage" | "empirical" | "hybrid" | "decoupled" (см. модульную документацию)
        :param nliq_from_curve: брать N_liq из момента пересечения порога кривой PPR (а не из first-hitting)
        :param calibration_steps: число шагов дифференцируемой калибровки θ по префиксу (0 = выкл)
        :param use_traj_residual: малый zero-init резидуал поверх траектории движка
        :param liq_threshold: порог PPR для определения разжижения (для N_liq из кривой)
        :param report_nliq_from_curve: публиковать физическое пересечение PPR как N_liq; direct head
            при этом остаётся auxiliary-головой
        :param nliq_head_aux_weight: вес censored-loss auxiliary N_liq-головы
        """
        super().__init__(static_dim, prefix_dim, seq_dim, seq_len, prefix_len, max_cycle_reference,
                         hidden_dim=hidden_dim, **kwargs)
        self.probabilistic = probabilistic
        self.use_flow = use_flow
        self.crr_mode = crr_mode if crr_mode is not None else ("damage" if crr_from_damage else "empirical")
        self.crr_from_damage = self.crr_mode in ("damage", "hybrid")
        self.nliq_from_curve = nliq_from_curve
        self.calibration_steps = calibration_steps
        self.calibration_lr = calibration_lr
        self.use_traj_residual = use_traj_residual
        self.liq_threshold = liq_threshold
        self.use_observed_aux_loss = use_observed_aux_loss
        # MC-микстура предиктива по θ (#3, симметрично DPI-Flow): opt-in калибровка разброса
        # постериора под предиктивную ошибку (mixture-NLL + energy-CRPS). 0 → выкл (прежнее поведение).
        self.mc_train_samples = int(mc_train_samples)
        self.mc_crps_weight = float(mc_crps_weight)
        self.mc_predict_samples = int(mc_predict_samples)
        self.report_nliq_from_curve = bool(report_nliq_from_curve)
        self.nliq_head_aux_weight = float(nliq_head_aux_weight)
        self._force_sample = False
        self.logvar_head = nn.Linear(hidden_dim, 33)
        # Conditional RealNVP с latent-зависимым coupling и log-det (вместо диагонального flow).
        self.flow = ConditionalCouplingFlow(33, hidden_dim, n_layers=4, hidden=64)
        self.crr_ref_head = nn.Linear(hidden_dim, 3) # [CRR_ref, λ_crr, m_crr]
        self.traj_residual = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                           nn.Linear(hidden_dim, seq_len))
        nn.init.zeros_(self.traj_residual[-1].weight)
        nn.init.zeros_(self.traj_residual[-1].bias)
        self.traj_residual_span = float(traj_residual_span)
        # Выразительный монотонный канал: gate задаёт ОБЩУЮ неотрицательную массу коррекции,
        # softmax-голова распределяет её по сетке. Масштаб не зависит от seq_len; gate=-6 даёт
        # почти no-op (~0.0025 PPR суммарно). Монотонность и ru≤1 сохранены по построению.
        self.use_free_increment = bool(use_free_increment)
        self.free_increment_head = nn.Linear(hidden_dim, seq_len)
        self.free_increment_gate = nn.Parameter(torch.tensor(-6.0))

    # ---------- DPI: идентификация θ ----------
    def infer_theta(self, encoded):
        mu = self.param_head(encoded)
        raw_logvar = torch.clamp(self.logvar_head(encoded), min=-5.0, max=3.0)
        if self.probabilistic:
            stochastic = self.training or self._force_sample # _force_sample → MC-сэмплы на инференсе
            eps = torch.randn_like(mu) if stochastic else torch.zeros_like(mu)
            latent = mu + torch.exp(0.5 * raw_logvar) * eps
        else:
            latent = mu
            raw_logvar = torch.zeros_like(mu)
        return latent, mu, raw_logvar

    def _theta_from_latent(self, latent, encoded):
        """θ из латента через conditional RealNVP; возвращает (θ, log|det ∂θ/∂z|)."""
        if self.use_flow:
            return self.flow(latent, encoded)
        return latent, torch.zeros(latent.shape[0], device=latent.device, dtype=latent.dtype)

    # ---------- CRR ----------
    def crr_from_damage_law(self, params, crr_ref, cycles):
        inv_m = 1.0 / torch.clamp(params["m"], min=0.5).unsqueeze(1)
        arg = 1.0 + params["lambda_pre"].unsqueeze(1) * cycles
        return torch.clamp(crr_ref * torch.pow(arg, -inv_m), min=1e-4)

    def _compute_crr_pair(self, encoded, params, cycles):
        """Вернуть (crr_dyn — ведёт динамику, crr_out — отчётная, crr_consistency (B,))."""
        head = self.crr_ref_head(encoded)
        crr_ref = 0.15 + 0.45 * torch.sigmoid(head[:, 0:1])
        cons = torch.zeros(encoded.shape[0], device=encoded.device)

        def power_crr():
            lam = 0.0005 + 0.02 * torch.sigmoid(head[:, 1:2])
            m = 1.0 + 2.4 * torch.sigmoid(head[:, 2:3])
            return torch.clamp(crr_ref * torch.pow(1.0 + lam * cycles, -1.0 / m), min=1e-4), lam, m

        if self.crr_mode == "empirical":
            crr_dyn = crr_out = self.compute_crr(params, cycles)
        elif self.crr_mode == "hybrid":
            crr_out, lam, m = power_crr(); crr_dyn = crr_out
            lp = params["lambda_pre"].unsqueeze(1); mp = params["m"].unsqueeze(1)
            cons = ((torch.log(lam) - torch.log(lp)) ** 2 + (1.0 / m - 1.0 / mp) ** 2).squeeze(1)
        elif self.crr_mode == "decoupled":
            crr_dyn = self.compute_crr(params, cycles)
            crr_out, _, _ = power_crr()
            cons = ((torch.log(crr_out + 1e-6) - torch.log(crr_dyn + 1e-6)) ** 2).mean(dim=1)
        else: # damage
            crr_dyn = crr_out = self.crr_from_damage_law(params, crr_ref, cycles)
        return crr_dyn, crr_out, cons

    # ---------- движок EVT (рекуррентный rollout) ----------
    def _engine(self, encoded, params, crr_dyn, batch, n_steps=None):
        """Развернуть динамику (z, r, c, g) на n_steps шагов (по умолчанию вся последовательность)."""
        batch_size, seq_len = batch["cycles"].shape
        n_steps = seq_len if n_steps is None else min(n_steps, seq_len)
        eps = 1e-6
        cycles, dcyc, csr, seq_in = batch["cycles"], batch["delta_cycles"], batch["csr"], batch["seq_in"]

        def trigger(z_s):
            return torch.sigmoid(params["kappa"] * (z_s - params["z0"])) if self.use_trigger_head else torch.zeros_like(z_s)

        def derivatives(z_s, r_s, c_s, g_s, step):
            ratio = csr[:, step] / (crr_dyn[:, step] + eps)
            phi = F.softplus(6.0 * (ratio - 0.90)) / 6.0
            corr_in = torch.cat([encoded, seq_in[:, step, :], z_s[:, None], r_s[:, None], c_s[:, None], g_s[:, None]], dim=-1)
            raw = torch.tanh(self.correction_net(corr_in))
            dz_c = 0.004 * raw[:, 0]; dr_c = 0.004 * raw[:, 1]; dc_c = 0.5 * raw[:, 2]
            if self.use_crr_damage:
                dz_pre = (params["lambda_pre"] * torch.pow(torch.clamp(ratio, min=eps), params["m"])
                          * torch.pow(torch.clamp(1.0 - z_s, min=eps), params["nu"]) + dz_c)
            else:
                dz_pre = 0.10 * torch.tanh(dz_c) + 0.02 * phi
            dr_pre = (params["alpha_pre"] * phi * torch.pow(torch.clamp(1.0 - r_s, min=eps), params["p_pre"])
                      + params["beta_pre"] / (cycles[:, step] + params["tau_pre"]) - params["gamma_pre"] * r_s + dr_c)
            if self.structured_post_event:
                dz_post = params["lambda_post"] * torch.clamp(1.0 - z_s, min=eps) + 0.5 * dz_c
                dr_post = (params["alpha_post"] * torch.pow(torch.clamp(1.0 - r_s, min=eps), params["p_post"])
                           + params["beta_post"] / (cycles[:, step] + 0.70 * params["tau_pre"]) - params["gamma_post"] * r_s + 0.5 * dr_c)
            else:
                dz_post, dr_post = dz_pre, dr_pre
            return (1.0 - g_s) * dz_pre + g_s * dz_post, (1.0 - g_s) * dr_pre + g_s * dr_post, dc_c

        z_curr = torch.zeros(batch_size, device=cycles.device)
        r_curr = 0.10 * torch.sigmoid(self.r0_head(encoded)).squeeze(-1)
        c_curr = torch.zeros(batch_size, device=cycles.device)
        zs, rs, cs, gs = [z_curr], [r_curr], [c_curr], []
        for step in range(n_steps):
            g_step = trigger(z_curr); gs.append(g_step)
            if step == n_steps - 1:
                break
            dn = dcyc[:, step + 1]
            dz1, dr1, dc1 = derivatives(z_curr, r_curr, c_curr, g_step, step)
            if self.integrator == "heun":
                z_e = torch.clamp(z_curr + dn * dz1, 0.0, 0.999); r_e = torch.clamp(r_curr + dn * dr1, 0.0, 1.0)
                c_e = torch.tanh(params["c_decay"] * c_curr + dc1)
                dz2, dr2, dc2 = derivatives(z_e, r_e, c_e, trigger(z_e), step)
                dz_s, dr_s, dc_s = 0.5 * (dz1 + dz2), 0.5 * (dr1 + dr2), 0.5 * (dc1 + dc2)
            else:
                dz_s, dr_s, dc_s = dz1, dr1, dc1
            z_curr = torch.clamp(z_curr + dn * dz_s, 0.0, 0.999)
            # Урон необратим: без проекции нейронная поправка dz_c<0 роняет z и триггер g на хвосте.
            z_curr = torch.maximum(z_curr, zs[-1])
            r_curr = torch.clamp(r_curr + dn * dr_s, 0.0, 1.0)
            c_curr = torch.tanh(params["c_decay"] * c_curr + dc_s)
            zs.append(z_curr); rs.append(r_curr); cs.append(c_curr)
        # Поглощающее событие: триггер g неубывающий (z уже спроецирован; cummax — явная гарантия).
        return torch.stack(zs, 1), torch.stack(rs, 1), torch.stack(cs, 1), torch.cummax(torch.stack(gs, 1), dim=1).values

    # ---------- дифференцируемая калибровка θ по префиксу (DPI-identification) ----------
    def calibrate_theta(self, latent, encoded, batch):
        if self.calibration_steps <= 0:
            return latent
        anchor = latent
        work = latent.detach()
        for _ in range(self.calibration_steps):
            work = work.detach().requires_grad_(True)
            with torch.enable_grad():
                theta, _ = self._theta_from_latent(work, encoded) # (θ, log_det) — берём θ
                params = self.unpack_params(theta)
                crr_dyn, _, _ = self._compute_crr_pair(encoded, params, batch["cycles"])
                _, r_pref, _, _ = self._engine(encoded, params, crr_dyn, batch, n_steps=self.prefix_len)
                r_pref = monotone_clip(r_pref)
                loss = masked_mse(r_pref, batch["prefix_obs"][:, : self.prefix_len],
                                  batch["prefix_mask"][:, : self.prefix_len])
            grad = torch.autograd.grad(loss, work, retain_graph=False, create_graph=False)[0]
            work = work - self.calibration_lr * grad
        return anchor + (work.detach() - anchor.detach()) # straight-through

    # ---------- N_liq из кривой PPR ----------
    def _nliq_from_curve(self, r, cycles, beta: float = 25.0):
        """Дифференцируемый момент пересечения порога ru монотонной кривой PPR. Возвращает (N_liq, pdf, mass).

        N_liq считается ЖЁСТКОЙ линейной интерполяцией пересечения (``interpolated_crossing``): оценка
        лежит ТОЧНО на кривой между узлами-скобками и не смещена размазыванием ``(pdf·cycles).sum()``
        по редким поздним узлам лог-сетки (систематический сдвиг тайминга у DPI-EVT). Мягкие pdf/mass
        по-прежнему возвращаются для supervision CRR-в-момент-разжижения (cross_pdf/cross_mass)."""
        p = torch.sigmoid(beta * (r - self.liq_threshold)) # монотонно растёт (r монотонна)
        dp = torch.clamp(p[:, 1:] - p[:, :-1], min=0.0)
        pdf = torch.cat([p[:, :1], dp], dim=1) # мягкая масса пересечения по шагам (для CRR@crossing)
        mass = pdf.sum(dim=1)
        nliq = interpolated_crossing(r, cycles, self.liq_threshold, beta=beta)
        return nliq, pdf, mass

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = self.build_context(batch)
        encoded = self.context_encoder(context)
        latent, mu, raw_logvar = self.infer_theta(encoded)
        latent = self.calibrate_theta(latent, encoded, batch) # DPI: уточнение θ по префиксу (steps=0 → no-op)
        theta, log_det = self._theta_from_latent(latent, encoded)
        params = self.unpack_params(theta)
        crr_dyn, crr_out, crr_cons = self._compute_crr_pair(encoded, params, batch["cycles"])

        z, r, c, g = self._engine(encoded, params, crr_dyn, batch)
        if self.use_traj_residual or self.use_free_increment:
            # МОНОТОННО-СОХРАНЯЮЩАЯ коррекция: ±span масштаб темпа + опц. свободный неотрицательный
            # аддитивный канал. Кривая остаётся неубывающей по построению (не «physically unreliable»).
            resid = self.traj_residual(encoded) if self.use_traj_residual else torch.zeros_like(r)
            free = None
            if self.use_free_increment:
                free = normalized_free_increments(
                    self.free_increment_head(encoded), self.free_increment_gate)
            r = monotone_residual_scale(r, resid, span=self.traj_residual_span, free_increment=free)
        r = monotone_clip(r, hi=1.0) # ru≤1 физически (не 1.05)

        mcr = torch.as_tensor(self.max_cycle_reference, device=r.device, dtype=r.dtype)
        if self.nliq_from_curve:
            nliq_curve, cross_pdf, cross_mass = self._nliq_from_curve(r, batch["cycles"])
        else:
            nliq_curve = self.soft_first_hitting(r, g, batch["cycles"])
            cross_pdf = torch.zeros_like(r); cross_mass = torch.zeros(r.shape[0], device=r.device)
        nliq_norm_curve = torch.log1p(nliq_curve) / torch.log1p(mcr)
        # Direct head остаётся auxiliary-предиктором. Headline DPI-EVT публикует физически проверяемое
        # пересечение финальной PPR-кривой; это исключает скрытое рассогласование «N_liq не лежит на PPR».
        nliq_head, nliq_norm_head, nliq_norm_curve = self._apply_nliq_head(encoded, nliq_norm_curve, mcr)
        if self.report_nliq_from_curve:
            nliq, nliq_norm = nliq_curve, nliq_norm_curve
        else:
            nliq, nliq_norm = nliq_head, nliq_norm_head
        summary = physics_summary(r, z, g, nliq_norm)
        risk_prior = 6.0 * (0.50 * r.amax(dim=1) + 0.25 * g.amax(dim=1) + 0.25 * z.amax(dim=1) - 0.75)
        risk_logit = (self.risk_clf(encoded).squeeze(-1) + self.prior_gate * risk_prior
                      + self.risk_head(encoded, summary))
        traj_logvar = self.logvar_head_seq(encoded) + 2.0 * self.calib_log_scale
        # Корректная плотность conditional flow: KL с log-det (при выключенном flow — обычный гауссов KL).
        if self.probabilistic and self.use_flow:
            kl = flow_kl_per_dim(latent, mu, raw_logvar, theta, log_det)
        else:
            kl = 0.5 * (torch.exp(raw_logvar) + mu.pow(2) - 1.0 - raw_logvar).mean(dim=1)
        return {
            "traj_mean": r, "ru": r, "damage": z, "traj_logvar": traj_logvar,
            "risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit),
            "nliq": nliq, "nliq_norm": nliq_norm, "nliq_norm_curve": nliq_norm_curve,
            "nliq_head": nliq_head, "nliq_norm_head": nliq_norm_head,
            "z": z, "g": g, "c": c, "crr": crr_out, "kl": kl,
            "crr_consistency": crr_cons, "cross_pdf": cross_pdf, "cross_mass": cross_mass,
        }

    def predictive(self, batch: Dict[str, torch.Tensor], mc_samples: int = 8) -> Dict[str, torch.Tensor]:
        """
        MC-предиктив: K стохастических проходов с сэмплированием θ (через flow/гауссов постериор).

        Дисперсия траектории = aleatoric (среднее exp(logvar) по сэмплам) + epistemic (разброс
        средних между сэмплами θ). Если модель не вероятностная или mc_samples<=1 — forward_batch.
        """
        if not self.probabilistic or mc_samples <= 1:
            return self.forward_batch(batch)
        prev = self._force_sample
        self._force_sample = True
        try:
            means, alea, risks, nliqs, crrs, last = [], [], [], [], [], None
            for _ in range(int(mc_samples)):
                o = self.forward_batch(batch)
                means.append(o["traj_mean"]); alea.append(torch.exp(o["traj_logvar"]))
                risks.append(o["risk_prob"])
                if "nliq_norm" in o:
                    nliqs.append(o["nliq_norm"])
                if "crr" in o and torch.is_tensor(o["crr"]):
                    crrs.append(o["crr"])
                last = o
        finally:
            self._force_sample = prev
        M = torch.stack(means, 0)
        pred_var = torch.stack(alea, 0).mean(0) + M.var(0, unbiased=False)
        out = dict(last)
        out["traj_mean"] = M.mean(0)
        out["traj_logvar"] = torch.log(pred_var.clamp_min(1e-12))
        out["traj_epistemic_var"] = M.var(0, unbiased=False)
        rp = torch.stack(risks, 0).mean(0).clamp(1e-6, 1 - 1e-6)
        out["risk_prob"] = rp
        out["risk_logit"] = torch.log(rp) - torch.log1p(-rp)
        if crrs:
            out["crr"] = torch.stack(crrs, 0).mean(0)
        if nliqs:
            nn_samples = torch.stack(nliqs, 0)
            mcr = torch.as_tensor(self.max_cycle_reference, device=out["traj_mean"].device,
                                  dtype=out["traj_mean"].dtype)
            if self.report_nliq_from_curve:
                # Point estimate согласуем с опубликованной MC-средней траекторией. Квантили ниже
                # остаются квантилями индивидуальных crossing и сохраняют эпистемическую вариацию.
                nliq_point, cross_pdf, cross_mass = self._nliq_from_curve(
                    out["traj_mean"], batch["cycles"],
                )
                out["nliq"] = nliq_point
                out["nliq_norm"] = torch.log1p(nliq_point) / torch.log1p(mcr)
                out["nliq_norm_curve"] = out["nliq_norm"]
                out["cross_pdf"] = cross_pdf
                out["cross_mass"] = cross_mass
            else:
                nn_ = nn_samples.mean(0)
                out["nliq_norm"] = nn_
                out["nliq"] = torch.expm1(nn_ * torch.log1p(mcr))
            cyc_samples = torch.expm1(nn_samples * torch.log1p(mcr))
            out["nliq_q05"] = torch.quantile(cyc_samples, 0.05, dim=0)
            out["nliq_q95"] = torch.quantile(cyc_samples, 0.95, dim=0)
        return out

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from liquefaction_ai.training.losses import observed_aux_loss
        out = self.forward_batch(batch)
        # Траекторный лосс: одиночный gaussian_nll по умолчанию; при mc_train_samples>0 — proper
        # NLL гауссовой смеси по S сэмплам θ (+ опц. energy-CRPS), что калибрует разброс постериора.
        if self.mc_train_samples > 0 and self.probabilistic:
            mus, logvars = [], []
            prev = self._force_sample
            self._force_sample = True
            try:
                for _ in range(self.mc_train_samples):
                    o = self.forward_batch(batch)
                    mus.append(o["traj_mean"]); logvars.append(o["traj_logvar"])
            finally:
                self._force_sample = prev
            mc_means = torch.stack(mus, 0); mc_logvars = torch.stack(logvars, 0)
            traj_loss = gaussian_mixture_nll(mc_means, mc_logvars, batch["r_obs"], batch["mask"])
            if self.mc_crps_weight > 0.0:
                samples = mc_means + torch.exp(0.5 * mc_logvars) * torch.randn_like(mc_means)
                traj_loss = traj_loss + self.mc_crps_weight * energy_crps(samples, batch["r_obs"], batch["mask"])
        else:
            traj_loss = gaussian_nll(out["traj_mean"], out["traj_logvar"], batch["r_obs"], batch["mask"])
        # Риск-лосс только по наблюдаемым (хелпер: all-unobserved → 0); soft-AUC тоже по наблюдаемым.
        _robs = risk_observation_mask(batch)
        risk_loss = masked_bce_with_logits(out["risk_logit"], batch["label"], _robs)
        _m = (_robs > 0.5) if _robs is not None else torch.ones_like(batch["label"], dtype=torch.bool)
        rank_loss = (soft_auc_loss(out["risk_logit"][_m], batch["label"][_m]) if bool(_m.any())
                     else out["risk_logit"].sum() * 0.0)
        nliq_loss = masked_censored_nliq_loss(out["nliq_norm"], batch["n_liq_norm"],
                                              batch["label"], nliq_censor_mask(batch))
        # При curve-first output direct head обучается отдельно как auxiliary fallback. Не дублируем
        # этот loss в legacy head-first режиме, где основной nliq_loss уже приложен к той же голове.
        if self.report_nliq_from_curve and getattr(self, "use_nliq_head", False):
            nliq_head_loss = masked_censored_nliq_loss(
                out["nliq_norm_head"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch),
            )
        else:
            nliq_head_loss = torch.zeros((), device=out["traj_mean"].device)
        switch_reg = torch.abs(out["g"][:, 1:] - out["g"][:, :-1]).mean()
        state_smooth = (torch.abs(out["traj_mean"][:, 2:] - 2 * out["traj_mean"][:, 1:-1] + out["traj_mean"][:, :-2]).mean()
                        + torch.abs(out["z"][:, 2:] - 2 * out["z"][:, 1:-1] + out["z"][:, :-2]).mean())
        kl_loss = out["kl"].mean() if self.probabilistic else torch.zeros((), device=out["traj_mean"].device)

        # joint-consistency: связать выходы единого состояния в момент разжижения
        liq = batch["label"]; denom = torch.clamp(liq.sum(), min=1.0)
        joint = torch.zeros((), device=out["traj_mean"].device)
        if out["cross_mass"].sum() > 0:
            m = torch.clamp(out["cross_mass"], min=1e-4)
            crr_at = (out["cross_pdf"] * out["crr"]).sum(1) / m # CRR в момент разжижения
            z_at = (out["cross_pdf"] * out["z"]).sum(1) / m # Damage в момент разжижения
            csr_app = batch["csr"].amax(dim=1)
            crr_cons_liq = (liq * (crr_at - csr_app) ** 2).sum() / denom # CRR(N_liq) ≈ CSR
            dmg_cons_liq = (liq * (z_at - 0.90) ** 2).sum() / denom # Damage(N_liq) ≈ порог
            joint = crr_cons_liq + dmg_cons_liq

        # --- подавление ложного роста PPR и ложного триггера на НЕразжижающихся опытах ---
        # overshoot: одностороннее превышение измеренной кривой (безопасно и для незавершённых,
        # правоцензурированных опытов). Подавление триггера g — только на уверенно неразжижающихся
        # (regime_stable==1): незавершённые опыты могли бы продолжить рост, их не штрафуем.
        noliq = (1.0 - batch["label"]).unsqueeze(1)
        overshoot = masked_mean(torch.relu(out["traj_mean"] - batch["r_obs"]) * noliq, batch["mask"])
        stable = batch.get("regime_stable")
        stab = noliq if stable is None else stable.unsqueeze(1)
        trigger_noliq = masked_mean(out["g"] * stab, batch["mask"]) if "g" in out else torch.zeros((), device=out["traj_mean"].device)

        # consistency: N_liq-голова ≈ физическое пересечение кривой (где событие произошло)
        nliq_consistency = self._nliq_consistency_loss(out, batch)
        loss = (traj_loss + 0.80 * risk_loss + 0.30 * rank_loss + 0.25 * nliq_loss
                + self.nliq_head_aux_weight * nliq_head_loss
                + 0.10 * nliq_consistency
                + 0.02 * switch_reg + 0.01 * state_smooth + 0.02 * kl_loss
                + 0.01 * out["crr_consistency"].mean() + 0.05 * joint
                + 0.20 * overshoot + 0.05 * trigger_noliq)
        if self.use_observed_aux_loss:
            loss = loss + observed_aux_loss(out, batch, use_states=True)
        out["loss"] = loss
        return out
