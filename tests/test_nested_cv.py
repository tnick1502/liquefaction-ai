"""
Тест API вложенной object-CV (#2): селекция гиперпараметров внутри фолда — opt-in и совместима.
"""
import inspect

from liquefaction_ai.evaluation import cross_validation as CV


def test_nested_is_optin_and_default_off():
    assert "nested" in inspect.signature(CV.evaluate_fold).parameters
    assert inspect.signature(CV.evaluate_fold).parameters["nested"].default is False
    assert "nested" in inspect.signature(CV._train_one).parameters


def test_nested_grids_cover_structural_models():
    # внутренние сетки заданы для всех трёх структурных моделей
    for name in ("dpi_flow", "evt_ssm", "dpi_evt"):
        assert name in CV.NESTED_GRIDS and len(CV.NESTED_GRIDS[name]) >= 1
    assert CV.NESTED_SELECT_METRIC == "Traj_RMSE_continuation"
