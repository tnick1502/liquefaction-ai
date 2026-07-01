"""
Run manifest: воспроизводимая «шапка прогона».

Единый артефакт, привязывающий отчётные числа к КОНКРЕТНОМУ состоянию проекта: хэш данных,
полный конфиг, git-commit и архитектуры обученных моделей. Пишется в ``results/run_manifest.json``
из ноутбука оценки (3_1). Позволяет рецензенту/себе однозначно ответить «на каких данных, коде и
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

__all__ = ["build_run_manifest", "write_run_manifest", "validate_run_manifest"]


def _git_commit(repo_root: Path) -> Optional[str]:
    """Текущий git-commit (короткий hash) или None, если репозиторий/GIT недоступен."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
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
        return bool(out.stdout.strip())
    except Exception:
        return None


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
    keys = ("liq_label", "n_liq_true", "r_obs", "prefix_obs", "prefix_mask", "valid_mask",
            "cycles", "csr", "static_features", "prefix_summary", "seq_inputs", "crr_obs",
            "crr_obs_mask")
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
                       models_dir: str | Path = "models") -> Dict[str, Any]:
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
        "architectures": architectures,
        "result_files_sha256": result_hashes,
        "environment": {
            "pyproject_sha256": _file_sha256(repo_root / "pyproject.toml"),
            "poetry_lock_sha256": _file_sha256(repo_root / "poetry.lock"),
        },
    }


def write_run_manifest(pop: Dict[str, Any], config, repo_root: str | Path,
                       models_dir: str | Path = "models",
                       out_path: str | Path = "results/run_manifest.json",
                       require_clean: bool = False) -> Path:
    """Собрать и записать манифест на диск; вернуть путь к файлу."""
    manifest = build_run_manifest(pop, config, repo_root, models_dir)
    if require_clean and manifest["git_dirty"]:
        raise RuntimeError("Publication run запрещён из dirty git tree: сначала зафиксируйте код и конфиг.")
    out = Path(out_path)
    if not out.is_absolute():
        out = Path(repo_root) / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return out


def validate_run_manifest(manifest: Dict[str, Any], required_models=(), required_results=()) -> None:
    """Строгий publication gate: clean git, полный data hash, model artifacts и итоговые таблицы."""
    errors = []
    if manifest.get("git_dirty"):
        errors.append("git tree dirty")
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
    if errors:
        raise RuntimeError("Publication manifest invalid: " + "; ".join(errors))
