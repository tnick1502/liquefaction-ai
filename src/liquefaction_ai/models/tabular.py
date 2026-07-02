"""
Табличные baseline по статическим признакам грунта/нагружения (без траектории PPR).

- :class:`FTTransformer` — Feature-Tokenizer Transformer (Gorishniy et al., 2021): каждый
  числовой признак токенизируется в d-мерный эмбеддинг, добавляется обучаемый CLS-токен,
  стек слоёв Transformer-энкодера агрегирует признаки; из CLS предсказываются риск разжижения
  и нормированный N_liq. Реализует контракт ``forward_batch`` / ``compute_loss`` (нейросеть).
- :class:`CatBoostBaseline` — обёртка над градиентным бустингом CatBoost (классификатор риска +
  регрессор N_liq). Не нейросеть, но повторяет интерфейс ``eval`` / ``forward_batch`` (возвращает
  torch-тензоры), поэтому совместима с :func:`liquefaction_ai.evaluation.collect_outputs`.
  Обучение — методом :meth:`CatBoostBaseline.fit`; сохранение/загрузка — нативным форматом CatBoost.

Оба предсказывают только риск и N_liq (без кривой PPR(N)), поэтому Traj_RMSE для них не
определён (NaN в лидерборде, как у статического MLP-Risk).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from liquefaction_ai.training.losses import (masked_bce_with_logits, masked_censored_nliq_loss,
                                             nliq_censor_mask, risk_observation_mask)

__all__ = ["FTTransformer", "CatBoostBaseline"]


class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer для табличных статических признаков."""

    def __init__(self, static_dim: int, prefix_dim: int, d_token: int = 64,
                 n_heads: int = 4, n_layers: int = 3):
        """
        :param static_dim: размерность статических признаков
        :param prefix_dim: размерность сводки префикса
        :param d_token: размерность токена признака (d_model)
        :param n_heads: число голов внимания
        :param n_layers: число слоёв энкодера
        """
        super().__init__()
        self.n_features = static_dim + prefix_dim
        self.feature_weight = nn.Parameter(torch.empty(self.n_features, d_token))
        self.feature_bias = nn.Parameter(torch.zeros(self.n_features, d_token))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.trunc_normal_(self.feature_weight, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads, dim_feedforward=4 * d_token,
                                           dropout=0.10, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.risk_head = nn.Linear(d_token, 1)
        self.nliq_head = nn.Linear(d_token, 1)

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        :param batch: словарь батча (поля ``static``, ``prefix_summary``)
        :return: словарь выходов: ``risk_logit``, ``risk_prob``, ``nliq_pred``
        """
        x = torch.cat([batch["static"], batch["prefix_summary"]], dim=-1) # (B, F)
        tokens = x.unsqueeze(-1) * self.feature_weight + self.feature_bias # (B, F, d)
        cls = self.cls_token.expand(x.shape[0], -1, -1) # (B, 1, d)
        h = self.encoder(torch.cat([cls, tokens], dim=1))
        cls_out = self.norm(h[:, 0])
        risk_logit = self.risk_head(cls_out).squeeze(-1)
        nliq_pred = torch.sigmoid(self.nliq_head(cls_out).squeeze(-1))
        return {"risk_logit": risk_logit, "risk_prob": torch.sigmoid(risk_logit), "nliq_pred": nliq_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """BCE-риск + Smooth-L1 по нормированному N_liq."""
        outputs = self.forward_batch(batch)
        risk_loss = masked_bce_with_logits(outputs["risk_logit"], batch["label"], risk_observation_mask(batch))
        nliq_loss = masked_censored_nliq_loss(outputs["nliq_pred"], batch["n_liq_norm"], batch["label"], nliq_censor_mask(batch))
        outputs["loss"] = risk_loss + 0.45 * nliq_loss
        return outputs


class CatBoostBaseline:
    """
    Обёртка CatBoost (классификатор риска + регрессор N_liq) под интерфейс оценки проекта.

    Не является ``nn.Module``: обучается :meth:`fit` (а не ``train_model``), но предоставляет
    ``eval`` и ``forward_batch`` (с torch-тензорами на выходе), что делает её совместимой с
    :func:`collect_outputs` / :func:`compute_metrics`.
    """

    def __init__(self, static_dim: int, prefix_dim: int, iterations: int = 500,
                 depth: int = 6, learning_rate: float = 0.05):
        self.static_dim = static_dim
        self.prefix_dim = prefix_dim
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.clf = None
        self.reg = None

    # --- совместимость с циклом инференса ---
    def eval(self):
        """No-op (CatBoost не имеет режима train/eval); для совместимости с collect_outputs."""
        return self

    def to(self, device):
        """No-op: CatBoost работает на CPU; метод для единообразия с torch-моделями."""
        return self

    @staticmethod
    def _features(batch: Dict[str, torch.Tensor]) -> np.ndarray:
        return torch.cat([batch["static"], batch["prefix_summary"]], dim=-1).detach().cpu().numpy()

    def fit(self, train_split: Dict[str, torch.Tensor], val_split: Dict[str, torch.Tensor]) -> "CatBoostBaseline":
        """Обучить классификатор риска и регрессор N_liq на статических признаках."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        Xtr = self._features(train_split); Xv = self._features(val_split)
        ytr_risk = train_split["label"].detach().cpu().numpy()
        yv_risk = val_split["label"].detach().cpu().numpy()
        ytr_nliq = train_split["n_liq_norm"].detach().cpu().numpy()
        yv_nliq = val_split["n_liq_norm"].detach().cpu().numpy()
        common = dict(iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                      random_seed=42, verbose=0, allow_writing_files=False)
        # Риск-классификатор: маска наблюдаемости исхода (незавершённые non-liq исключены — единый
        # цензур-протокол с proposed-моделями).
        otr = risk_observation_mask(train_split); ov = risk_observation_mask(val_split)
        mtr = (otr.detach().cpu().numpy() > 0.5) if otr is not None else np.ones(len(ytr_nliq), bool)
        mv = (ov.detach().cpu().numpy() > 0.5) if ov is not None else np.ones(len(yv_nliq), bool)
        self.clf = CatBoostClassifier(loss_function="Logloss", **common)
        self.clf.fit(Xtr[mtr], ytr_risk[mtr], eval_set=(Xv[mv], yv_risk[mv])) # риск ТОЛЬКО по наблюдаемым
        # N_liq-РЕГРЕССОР: обычный RMSE не выражает право-цензуру. Стабилизированные non-liq имеют
        # лишь НИЖНЮЮ границу N_liq (censoring time), подавать их как ТОЧНЫЙ таргет нельзя (proposed-
        # модели используют односторонний censored loss). Поэтому регрессор учим ТОЛЬКО на разжижившихся
        # (label==1). В evaluation он получает только liquefied-only N_liq metric; censored-aware metric
        # помечается N/A, чтобы не сравнивать разные estimands.
        rtr = mtr & (ytr_risk > 0.5); rv = mv & (yv_risk > 0.5)
        self.reg = CatBoostRegressor(loss_function="RMSE", **common)
        # fallback: если в VAL нет разжижившихся — обучаем БЕЗ eval_set (нельзя индексировать Xv
        # train-маской rtr — это был баг с несовпадением длин). Иначе обычный val-eval.
        if int(rv.sum()) == 0:
            self.reg.fit(Xtr[rtr], ytr_nliq[rtr])
        else:
            self.reg.fit(Xtr[rtr], ytr_nliq[rtr], eval_set=(Xv[rv], yv_nliq[rv]))
        return self

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Предсказать риск (вероятность) и нормированный N_liq; вернуть torch-тензоры."""
        X = self._features(batch)
        prob = np.clip(self.clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6).astype(np.float32)
        nliq = np.clip(self.reg.predict(X), 0.0, 1.0).astype(np.float32)
        risk_prob = torch.from_numpy(prob)
        return {
            "risk_prob": risk_prob,
            "risk_logit": torch.logit(risk_prob),
            "nliq_pred": torch.from_numpy(nliq),
            # RMSE-регрессор обучен только на exact events и не является censored-survival моделью.
            "supports_censored_nliq": torch.zeros(len(nliq), dtype=torch.float32),
        }

    def save(self, models_dir, name: str = "catboost") -> None:
        """Сохранить обе модели в нативном формате CatBoost (``risk.cbm`` / ``nliq.cbm``)."""
        d = Path(models_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        self.clf.save_model(str(d / "risk.cbm"))
        self.reg.save_model(str(d / "nliq.cbm"))

    def load(self, models_dir, name: str = "catboost") -> "CatBoostBaseline":
        """Загрузить обе модели из нативного формата CatBoost."""
        from catboost import CatBoostClassifier, CatBoostRegressor

        d = Path(models_dir) / name
        self.clf = CatBoostClassifier(); self.clf.load_model(str(d / "risk.cbm"))
        self.reg = CatBoostRegressor(); self.reg.load_model(str(d / "nliq.cbm"))
        return self
