import json

import numpy as np
import pandas as pd

import pytest

from liquefaction_ai.evaluation.manifest import build_run_manifest, validate_run_manifest


def _population():
    return {
        "liq_label": np.array([0, 1], np.float32),
        "n_liq_true": np.array([100, 20], np.float32),
        "r_obs": np.array([[0.1, 0.2], [0.4, 0.96]], np.float32),
        "prefix_obs": np.array([[0.1, 0.0], [0.4, 0.0]], np.float32),
        "prefix_mask": np.array([[1, 0], [1, 0]], np.float32),
        "valid_mask": np.ones((2, 2), np.float32),
        "cycles": np.array([[1, 2], [1, 2]], np.float32),
        "csr": np.full((2, 2), 0.2, np.float32),
        "benchmark": {"benchmark_idx": np.array([0, 1]), "train_rel": np.array([0]),
                      "val_rel": np.array([], dtype=int), "test_rel": np.array([1])},
        "meta": pd.DataFrame({"object": ["a", "b"], "site_id": ["s1", "s2"]}),
        "static_feature_names": ["x"], "prefix_summary_names": ["p"],
        "seq_feature_names": ["csr"],
    }


def test_manifest_hashes_real_artifact_keys_and_changes_with_targets(tmp_path):
    pop = _population()
    m1 = build_run_manifest(pop, {"seed": 1}, tmp_path)
    assert set(("n_liq_true", "r_obs", "prefix_obs")) <= set(m1["data"]["array_dims"])
    assert len(m1["data"]["sha256"]) == 64
    pop["n_liq_true"][1] += 1
    m2 = build_run_manifest(pop, {"seed": 1}, tmp_path)
    assert m1["data"]["sha256"] != m2["data"]["sha256"]


def test_manifest_hashes_weight_contents(tmp_path):
    d = tmp_path / "models" / "m"; d.mkdir(parents=True)
    (d / "hyperparams.json").write_text(json.dumps({"model_type": "X", "model_kwargs": {}}))
    (d / "weights.pt").write_bytes(b"first")
    m1 = build_run_manifest(_population(), {"seed": 1}, tmp_path)
    (d / "weights.pt").write_bytes(b"second")
    m2 = build_run_manifest(_population(), {"seed": 1}, tmp_path)
    assert m1["architectures"]["m"]["weights_sha256"] != m2["architectures"]["m"]["weights_sha256"]


def test_publication_manifest_gate_rejects_missing_artifacts(tmp_path):
    manifest = build_run_manifest(_population(), {"seed": 1}, tmp_path)
    manifest["git_dirty"] = False
    with pytest.raises(RuntimeError, match="missing model"):
        validate_run_manifest(manifest, required_models=("dpi_flow",))
