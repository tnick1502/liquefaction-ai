"""
Подпакет нейросетевых архитектур.

Содержит переиспользуемые блоки (``blocks``), базовые модели (``baselines``) и две
физически-структурированные архитектуры: ``DPIFlow`` (вероятностный вывод параметров
через аналитический ODE-слой) и ``EVTNeuralSSM`` (событийно-переключаемая модель
пространства состояний). Все модели реализуют единый контракт ``forward_batch`` /
``compute_loss``.
"""

from liquefaction_ai.models.ablations import FlowNoODE, NeuralODENoPhysics
from liquefaction_ai.models.baselines import (GRUBaseline, LSTMBaseline, RiskMLP,
                                              TCNBaseline, TransformerBaseline)
from liquefaction_ai.models.blocks import CausalTemporalBlock, ResidualMLP
from liquefaction_ai.models.dpi_flow import AnalyticalLiquefactionLayer, ConditionalAffineFlow, DPIFlow
from liquefaction_ai.models.evt_ssm import EVTNeuralSSM
from liquefaction_ai.models.heads import RiskHead, SeqLogvarHead, physics_summary
from liquefaction_ai.models.physics_baselines import PINNBaseline
from liquefaction_ai.models.probabilistic import DeepStateBaseline, NeuralSplineFlow, RealNVPFlow
from liquefaction_ai.models.tabular import CatBoostBaseline, FTTransformer

__all__ = [
    "ResidualMLP",
    "CausalTemporalBlock",
    "RiskMLP",
    "GRUBaseline",
    "TCNBaseline",
    "LSTMBaseline",
    "TransformerBaseline",
    "FTTransformer",
    "CatBoostBaseline",
    "PINNBaseline",
    "DeepStateBaseline",
    "RealNVPFlow",
    "NeuralSplineFlow",
    "ConditionalAffineFlow",
    "AnalyticalLiquefactionLayer",
    "DPIFlow",
    "NeuralODENoPhysics",
    "FlowNoODE",
    "EVTNeuralSSM",
    "RiskHead",
    "SeqLogvarHead",
    "physics_summary",
]
