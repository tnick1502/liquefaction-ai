"""
Run manifest: воспроизводимая «шапка прогона».

Единый артефакт, привязывающий отчётные числа к КОНКРЕТНОМУ состоянию проекта: хэш данных,
полный конфиг, git-commit и архитектуры обученных моделей. Пишется в ``results/run_manifest.json``
из ноутбука оценки (3_0). Позволяет рецензенту/себе однозначно ответить «на каких данных, коде и
гиперпараметрах получена таблица», и ловит рассинхрон «старые веса × новый артефакт».
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["build_run_manifest", "write_run_manifest", "validate_run_manifest",
           "publication_preflight"]


def _git_commit(repo_root: Path) -> Optional[str]:
    """Текущий git-commit (короткий hash) или None, если репозиторий/GIT недоступен."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except Exception:
        return None


def _git_dirty(repo_root: Path) -> Optional[bool]:
    """True, если есть незакоммиченные изменения (грязное дерево); None при недоступности git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return bool(out.stdout.strip())
    except Exception:
        return None


def publication_preflight(repo_root: str | Path, *, quick: bool, nested: bool,
                          run_loo: bool, run_ablations: bool,
                          run_ood: bool = True, run_ab: bool = True,
                          n_repeats: int = 3, ablation_seeds=(0, 1, 2),
                          output_root: str | Path = "results",
                          require_clean: bool = False) -> None:
    """Fail-fast, если отчётный (PUBLICATION_RUN) конфиг неполный. **Git-состояние НЕ проверяется**
    (проверка удалена: она мешала запуску из рабочего дерева). ``require_clean`` оставлен как no-op
    ради обратной совместимости сигнатуры. Для быстрых прогонов ставьте PUBLICATION_RUN=False —
    тогда preflight не вызывается и артефакты пишутся в results/smoke.
    """
    repo_root = Path(repo_root).resolve()
    out = Path(output_root)
    if not out.is_absolute():
        out = repo_root / out
    errors = []
    if quick:
        errors.append("QUICK must be False")
    if not nested:
        errors.append("NESTED must be True")
    if not run_loo:
        errors.append("RUN_LOO must be True")
    if not run_ablations:
        errors.append("RUN_ABLATIONS must be True")
    if not run_ood:
        errors.append("RUN_OOD must be True")
    if not run_ab:
        errors.append("RUN_AB must be True")
    if int(n_repeats) < 3:
        errors.append("N_REPEATS must be at least 3")
    if len(tuple(ablation_seeds)) < 3:
        errors.append("at least 3 ABLATION_SEEDS are required")
    if out.resolve() != (repo_root / "results").resolve():
        errors.append("publication output_root must be repository/results")
    if errors:
        raise RuntimeError(
            "Publication preflight failed: " + "; ".join(errors)
            + ".  Для БЫСТРОГО/пробного прогона поставьте PUBLICATION_RUN=False — тогда preflight "
            + "пропускается, а все артефакты пишутся в results/smoke (headline-таблицы не затираются). "
            + "Для ОТЧЁТНОГО прогона исправьте перечисленные тумблеры (напр. QUICK=False)."
        )


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _update_array_hash(h, key: str, value: Any) -> list:
    a = np.ascontiguousarray(np.asarray(value))
    h.update(key.encode("utf-8")); h.update(str(a.dtype).encode("ascii")); h.update(str(a.shape).encode("ascii"))
    h.update(a.tobytes())
    return list(a.shape)


def _data_fingerprint(pop: Dict[str, Any]) -> Dict[str, Any]:
    """SHA256 по фактическим ключам publication-артефакта, meta и split indices."""
    h = hashlib.sha256()
    dims: Dict[str, Any] = {}
    keys = ("liq_label", "n_liq_true", "r_obs", "r_causal", "q_obs", "eps_obs",
            "prefix_obs", "prefix_mask", "valid_mask", "cycles", "delta_cycles", "csr",
            "g_obs", "risk_proxy", "static_features", "prefix_summary", "seq_inputs",
            "crr_obs", "crr_obs_mask")
    for key in keys:
        arr = pop.get(key)
        if arr is None:
            continue
        dims[key] = _update_array_hash(h, key, arr)
    meta = pop.get("meta")
    if isinstance(meta, pd.DataFrame):
        h.update(json.dumps(list(meta.columns), ensure_ascii=False).encode("utf-8"))
        h.update(json.dumps([str(x) for x in meta.dtypes]).encode("utf-8"))
        h.update(pd.util.hash_pandas_object(meta, index=True).to_numpy(dtype=np.uint64).tobytes())
    split = pop.get("benchmark", {})
    split_dims = {}
    if isinstance(split, dict):
        for key in sorted(split):
            split_dims[key] = _update_array_hash(h, f"benchmark/{key}", split[key])
    for key in ("static_feature_names", "prefix_summary_names", "seq_feature_names"):
        if key in pop:
            h.update(key.encode("utf-8"))
            h.update(json.dumps(list(pop[key]), ensure_ascii=False).encode("utf-8"))
    lab = np.asarray(pop.get("liq_label", []))
    return {
        "sha256": h.hexdigest(),
        "n_samples": int(lab.shape[0]) if lab.size else 0,
        "liq_rate": float(np.mean(lab)) if lab.size else float("nan"),
        "array_dims": dims,
        "split_dims": split_dims,
        "meta_rows": int(len(meta)) if isinstance(meta, pd.DataFrame) else 0,
    }


def build_run_manifest(pop: Dict[str, Any], config, repo_root: str | Path,
                       models_dir: str | Path = "models",
                       run_metadata: Optional[Dict[str, Any]] = None,
                       fold_hyperparameters: Optional[list] = None) -> Dict[str, Any]:
    """
    Собрать манифест прогона: git, конфиг, отпечаток данных, архитектуры моделей, состав когорты.

    :param pop: артефакт популяции (из ``load_population_artifact``)
    :param config: ``ExperimentConfig`` данного прогона
    :param repo_root: корень репозитория (для git и путей к моделям)
    :param models_dir: относительный/абсолютный путь к каталогу моделей
    :return: словарь-манифест (JSON-сериализуемый)
    """
    repo_root = Path(repo_root)
    models_root = Path(models_dir)
    if not models_root.is_absolute():
        models_root = repo_root / models_root

    if dataclasses.is_dataclass(config):
        cfg = dataclasses.asdict(config)
    elif isinstance(config, dict):
        cfg = dict(config)
    else:
        cfg = dict(vars(config))

    architectures: Dict[str, Any] = {}
    if models_root.exists():
        for sub in sorted(models_root.iterdir()):
            hp_path = sub / "hyperparams.json"
            if not hp_path.exists():
                continue
            try:
                hp = json.loads(hp_path.read_text())
            except Exception:
                continue
            architectures[sub.name] = {
                "model_type": hp.get("model_type"),
                "display_name": hp.get("display_name"),
                "model_kwargs": hp.get("model_kwargs"),
                "has_weights": (sub / "weights.pt").exists(),
                "hyperparams_sha256": _file_sha256(hp_path),
                "weights_sha256": _file_sha256(sub / "weights.pt"),
                "artifacts_sha256": {p.name: _file_sha256(p) for p in sorted(sub.iterdir())
                                     if p.is_file() and p.name != "hyperparams.json"},
            }

    result_hashes: Dict[str, str] = {}
    results_root = repo_root / "results"
    if results_root.exists():
        for path in sorted(results_root.rglob("*")):
            if path.is_file() and path.name != "run_manifest.json" and path.suffix.lower() in {".csv", ".json"}:
                digest = _file_sha256(path)
                if digest:
                    result_hashes[str(path.relative_to(repo_root))] = digest

    return {
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "config": cfg,
        "data": _data_fingerprint(pop),
        "cohort_filter_counts": pop.get("cohort_filter_counts"),
        # These are global training artifacts. The actual nested-CV architectures are recorded
        # separately in fold_hyperparameters; conflating the two made old manifests misleading.
        "architectures": architectures,
        "global_training_artifacts": architectures,
        "evaluation_protocol": dict(run_metadata or {}),
        "fold_hyperparameters": list(fold_hyperparameters or []),
        "result_files_sha256": result_hashes,
        "environment": {
            "pyproject_sha256": _file_sha256(repo_root / "pyproject.toml"),
            "poetry_lock_sha256": _file_sha256(repo_root / "poetry.lock"),
        },
    }


def write_run_manifest(pop: Dict[str, Any], config, repo_root: str | Path,
                       models_dir: str | Path = "models",
                       out_path: str | Path = "results/run_manifest.json",
                       require_clean: bool = False,
                       run_metadata: Optional[Dict[str, Any]] = None,
                       fold_hyperparameters: Optional[list] = None) -> Path:
    """Собрать и записать манифест на диск; вернуть путь к файлу."""
    manifest = build_run_manifest(pop, config, repo_root, models_dir,
                                  run_metadata=run_metadata,
                                  fold_hyperparameters=fold_hyperparameters)
    # Git-состояние больше НЕ блокирует прогон (записывается в манифест как справочная информация).
    out = Path(out_path)
    if not out.is_absolute():
        out = Path(repo_root) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return out


def validate_run_manifest(manifest: Dict[str, Any], required_models=(), required_results=(),
                          require_fold_hyperparameters: bool = False) -> None:
    """Publication gate: полный data hash, model artifacts и итоговые таблицы. Git НЕ проверяется."""
    errors = []
    data = manifest.get("data", {})
    required_arrays = {"liq_label", "n_liq_true", "r_obs", "prefix_obs", "prefix_mask", "cycles"}
    missing_arrays = sorted(required_arrays - set(data.get("array_dims", {})))
    if missing_arrays:
        errors.append(f"data fingerprint misses {missing_arrays}")
    if len(str(data.get("sha256", ""))) != 64:
        errors.append("invalid data sha256")
    models = manifest.get("architectures", {})
    for name in required_models:
        entry = models.get(name)
        if not entry:
            errors.append(f"missing model {name}")
        elif not any(entry.get("artifacts_sha256", {}).values()):
            errors.append(f"model {name} has no hashed artifacts")
    results = manifest.get("result_files_sha256", {})
    for suffix in required_results:
        if not any(path.endswith(suffix) for path in results):
            errors.append(f"missing result {suffix}")
    if require_fold_hyperparameters and not manifest.get("fold_hyperparameters"):
        errors.append("missing fold-specific hyperparameters")
    if errors:
        raise RuntimeError("Publication manifest invalid: " + "; ".join(errors))
