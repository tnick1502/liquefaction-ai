"""
Подпакет нейросетевых архитектур.

Содержит переиспользуемые блоки (``blocks``), базовые модели (``baselines``) и две
физически-структурированные архитектуры: ``DPIFlow`` (вероятностный вывод параметров
через аналитический ODE-слой) и ``EVTNeuralSSM`` (событийно-переключаемая модель
пространства состояний). Все модели реализуют единый контракт ``forward_batch`` /
``compute_loss``.
"""

from liquefaction_ai.models.baselines import GRUBaseline, RiskMLP, TCNBaseline
from liquefaction_ai.models.blocks import CausalTemporalBlock, ResidualMLP
from liquefaction_ai.models.dpi_flow import AnalyticalLiquefactionLayer, ConditionalAffineFlow, DPIFlow
from liquefaction_ai.models.evt_ssm import EVTNeuralSSM
from liquefaction_ai.models.heads import RiskHead, SeqLogvarHead, physics_summary

__all__ = [
    "ResidualMLP",
    "CausalTemporalBlock",
    "RiskMLP",
    "GRUBaseline",
    "TCNBaseline",
    "ConditionalAffineFlow",
    "AnalyticalLiquefactionLayer",
    "DPIFlow",
    "EVTNeuralSSM",
    "RiskHead",
    "SeqLogvarHead",
    "physics_summary",
]
