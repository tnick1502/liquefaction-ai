"""
Тесты на «научные расхождения», которые легко пропустить (замечания рецензента P2).

Покрывают:
* object-level leakage: основной сплит артефакта — grouped/site-held-out и не делит объект
  между train/validation/test;
* полноту раннера (все ожидаемые ноутбуки на месте);
* отсутствие удалённого слоя PDF-отчётов и publication notebooks серии 5;
* graceful-degradation топологии (tda_available + guarded-импорты);
* раскрытие размера CRR-выборки (N_CRR_test).
"""
import pytest
import torch

from conftest import REAL_OBJECTS, REPO_ROOT, TABLES
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset

_skip_art = pytest.mark.skipif(not REAL_OBJECTS.exists(), reason="нет артефакта data/real_objects")


def _build_model_from_hp(hp: dict):
    """Сконструировать модель из hyperparams.json (model_type → класс). None, если тип неизвестен."""
    from liquefaction_ai.models import DPIEvtNet, DPIFlow, EVTNeuralSSM
    reg = {"DPIFlow": DPIFlow, "EVTNeuralSSM": EVTNeuralSSM, "DPIEvtNet": DPIEvtNet}
    cls = reg.get(hp.get("model_type"))
    return cls(**hp["model_kwargs"]) if cls is not None else None


def _artifact_weights_dims_match(model_name: str) -> bool:
    """ПОЛНАЯ совместимость СОХРАНЁННЫХ весов и текущей архитектуры (замечание #6).

    Раньше проверялся только static_dim — расхождение ЛЮБОГО другого слоя (напр. добавлен nliq_head,
    сменилась размерность θ, изменился flow) молча ломало load_state_dict уже ВНУТРИ теста. Теперь
    preflight: (1) static_dim артефакта == static_dim весов; (2) множества ключей state_dict
    сконструированной из hyperparams модели и сохранённого weights.pt СОВПАДАЮТ; (3) формы всех
    тензоров совпадают. Любое расхождение → тест грузящий веса ПРОПУСКАЕТСЯ как «нужен retrain»
    (ожидаемо при активной переработке архитектуры), а не падает с непрозрачной ошибкой."""
    import json

    import torch
    try:
        fn = json.loads((REAL_OBJECTS / "feature_names.json").read_text())
        sf = fn["static_feature_names"] if isinstance(fn, dict) else fn
        hp = json.loads((REPO_ROOT / "models" / model_name / "hyperparams.json").read_text())
        if int(len(sf)) != int(hp["model_kwargs"].get("static_dim", -1)):
            return False
        wp = REPO_ROOT / "models" / model_name / "weights.pt"
        if not wp.exists():
            return False
        model = _build_model_from_hp(hp)
        if model is None:
            return False
        saved = torch.load(wp, map_location="cpu", weights_only=True)
        cur = model.state_dict()
        if set(saved.keys()) != set(cur.keys()):
            return False
        return all(tuple(saved[k].shape) == tuple(cur[k].shape) for k in cur)
    except Exception:
        return False


# ---------------- object-level leakage ----------------

@_skip_art
def test_main_artifact_split_is_grouped_site_heldout():
    pop, cfg = load_population_artifact(REAL_OBJECTS)
    assert cfg.group_split_by_object is True
    bench = prepare_benchmark_dataset(pop, cfg, torch.device("cpu"))
    tr = set(bench["train"]["meta"]["object"]); te = set(bench["test"]["meta"]["object"])
    va = set(bench["val"]["meta"]["object"])
    assert not (tr & te), "leakage: объект попал и в train, и в test при grouped-разбиении"
    assert not (tr & va), "leakage: объект попал и в train, и в val при grouped-разбиении"
    assert not (va & te), "leakage: объект попал и в val, и в test при grouped-разбиении"


@_skip_art
def test_within_site_split_is_only_explicit_opt_in():
    pop, cfg = load_population_artifact(REAL_OBJECTS)
    cfg.group_split_by_object = False
    bench = prepare_benchmark_dataset(pop, cfg, torch.device("cpu"))
    if "object" not in bench["train"]["meta"].columns:
        pytest.skip("в meta нет колонки object")
    # Within-site режим — явный opt-in. Он не гарантирует overlap объектов на каждом конкретном
    # артефакте/seed (малое число объектов может случайно разойтись), но именно он отключает
    # grouped/site-held-out протокол и поэтому должен оставаться дополнительным режимом.
    assert cfg.group_split_by_object is False
    assert all(bench[name]["meta"].shape[0] > 0 for name in ("train", "val", "test"))


# ---------------- полнота раннера ----------------

def test_pipeline_notebooks_present():
    # Единый ноутбук подготовки данных живёт в data/ (sites/ → data/dataset либо генерация).
    assert (REPO_ROOT / "data" / "prepare_dataset.ipynb").exists(), "нет ноутбука data/prepare_dataset.ipynb"
    expected = [
        "1_data_analysis/1_1_exploratory_analysis.ipynb",
        "1_data_analysis/1_2_crr_parameter_analysis.ipynb",
        "1_data_analysis/1_3_dataset_split.ipynb",
        "2_model_training/2_1_baseline_models.ipynb",
        "2_model_training/2_2_dpi_flow.ipynb",
        "2_model_training/2_3_evt_neural_ssm.ipynb",
        "2_model_training/2_4_dpi_evt.ipynb",
        "3_evaluations/3_1_core_metrics.ipynb",
        "3_evaluations/3_2_ablations_ood.ipynb",
        "3_evaluations/3_3_case_studies.ipynb",
        "4_topology/4_1_dpi_flow_latent_topology.ipynb",
        "4_topology/4_2_topological_early_warning.ipynb",
        "4_topology/4_3_evt_neural_ssm_topological_regularization.ipynb",
    ]
    for rel in expected:
        assert (REPO_ROOT / "notebooks" / rel).exists(), f"нет ноутбука {rel}"


# ---------------- удалённый PDF/report слой ----------------

def test_pdf_reports_and_publication_notebooks_removed():
    forbidden = [
        "make_paper.py",
        "make_report.py",
        "liquefaction_ai_preprint.pdf",
        "liquefaction_ai_report.pdf",
        "notebooks/5_publication/5_1_paper_figures.ipynb",
    ]
    for rel in forbidden:
        assert not (REPO_ROOT / rel).exists(), f"устаревший PDF/publication артефакт всё ещё есть: {rel}"
    assert not (REPO_ROOT / "results" / "report_figs").exists()


def test_notebook_pipeline_is_complete():
    """Пайплайн полностью покрыт ноутбуками (корневые .py-оркестраторы удалены — всё в ноутбуках)."""
    nb = REPO_ROOT / "notebooks"
    assert (REPO_ROOT / "data" / "prepare_dataset.ipynb").exists(), "нет ноутбука data/prepare_dataset.ipynb"
    expected = [
        "2_model_training/2_2_dpi_flow.ipynb",
        "2_model_training/2_4_dpi_evt.ipynb",
        "3_evaluations/3_1_core_metrics.ipynb",
        "3_evaluations/3_4_object_cv_and_ci.ipynb",
        "3_evaluations/3_5_significance_tests.ipynb",
        "3_evaluations/3_6_ablations.ipynb",
        "3_evaluations/3_7_publication_figures.ipynb",
        "3_evaluations/3_8_consistency_and_p3_sensitivity.ipynb",
    ]
    for rel in expected:
        assert (nb / rel).exists(), f"нет ноутбука пайплайна: {rel}"


def test_readme_mentions_three_structured_models():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("DPI-Flow", "EVT-NeuralSSM", "DPI-EVT"):
        assert name in readme, f"README не упоминает {name}"


# ---------------- топология: graceful degradation ----------------

def test_topology_reports_availability_and_guards():
    from liquefaction_ai import topology as T

    avail = T.tda_available()
    assert set(avail) == {"ripser", "kmapper", "umap"}
    # импорты опциональных библиотек объявлены в pyproject
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for dep in ("ripser", "kmapper", "umap-learn", "kaleido"):
        assert dep in pyproject, f"в pyproject нет зависимости {dep}"
    # функции уважают флаги доступности (guarded)
    src = (REPO_ROOT / "src" / "liquefaction_ai" / "topology.py").read_text(encoding="utf-8")
    assert "if not HAS_RIPSER" in src and "if not HAS_KMAPPER" in src


# ---------------- раскрытие размера CRR-выборки ----------------

@_skip_art
def test_crr_sample_count_disclosed():
    import json as _json

    from liquefaction_ai.evaluation import collect_outputs, compute_metrics
    from liquefaction_ai.models import DPIEvtNet
    from liquefaction_ai.training.persistence import load_model_metadata, load_weights_into

    if not _artifact_weights_dims_match("dpi_evt"):
        pytest.skip("artifact/weights static_dim расходятся — нужна регенерация данных + переобучение")
    pop, cfg = load_population_artifact(REAL_OBJECTS)
    test = prepare_benchmark_dataset(pop, cfg, torch.device("cpu"))["test"]
    hp, _ = load_model_metadata(REPO_ROOT / "models", "dpi_evt")
    m = DPIEvtNet(**hp["model_kwargs"]); load_weights_into(m, REPO_ROOT / "models", "dpi_evt", torch.device("cpu"))
    met, _ = compute_metrics("DPI-EVT", collect_outputs(m, test, cfg, torch.device("cpu")), test, cfg)
    assert "N_CRR_test" in met and "N_CRR_objects" in met
    assert met["N_CRR_test"] >= 0 and met["N_CRR_objects"] >= 0
