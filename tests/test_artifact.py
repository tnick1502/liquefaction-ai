"""
Тесты целостности артефакта данных и батча.

Проверяют наличие обязательных полей/колонок артефакта популяции, появление маски цензуры
``n_liq_observed`` в батче и информативность частотного фактора CRR (частота реально вошла в
физику CRR, а не занулена).
"""
import numpy as np
import pytest
import torch

from conftest import REAL_OBJECTS
from liquefaction_ai import get_default_config, prepare_benchmark_dataset
from liquefaction_ai.data.io import load_population_artifact

_skip = pytest.mark.skipif(not REAL_OBJECTS.exists(), reason="нет артефакта data/real_objects")


@_skip
def test_population_has_required_fields():
    pop, _ = load_population_artifact(REAL_OBJECTS)
    for key in ["r_obs", "valid_mask", "liq_label", "n_liq_true", "g_obs", "static_features"]:
        assert key in pop, f"в артефакте нет поля {key}"
    for col in ["crr_alpha", "crr_betta", "crr_ref", "liq_label", "N_liq_true", "frequency"]:
        assert col in pop["meta"].columns, f"в meta нет колонки {col}"


@_skip
def test_split_exposes_censoring_mask():
    pop, cfg = load_population_artifact(REAL_OBJECTS)
    bench = prepare_benchmark_dataset(pop, cfg, torch.device("cpu"))
    for name in ["train", "val", "test"]:
        assert "n_liq_observed" in bench[name], f"в выборке {name} нет n_liq_observed"
        vals = set(np.unique(bench[name]["n_liq_observed"].cpu().numpy()).tolist())
        assert vals.issubset({0.0, 1.0})


@_skip
def test_crr_uses_real_frequency():
    # Частота нагружения (0.1…5 Гц) должна входить в физику CRR, а не зануляться.
    pop, _ = load_population_artifact(REAL_OBJECTS)
    meta = pop["meta"]
    assert meta["frequency"].nunique() > 1, "частота в данных не варьируется?"
    assert meta["crr_f_frequency"].nunique() > 1, "частотный фактор CRR занулён (константа)"


@_skip
def test_no_leakage_columns():
    # Убранные ранее синтетические/утечные поля не должны вернуться.
    pop, _ = load_population_artifact(REAL_OBJECTS)
    for bad in ["risk_score_true", "OCR"]:
        assert bad not in pop["meta"].columns
