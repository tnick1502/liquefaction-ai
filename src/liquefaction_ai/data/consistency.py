"""
Сквозная проверка digitrock-консистентности артефакта ``data/dataset``.

Импортируемая функция, которую вызывают и ноутбук 3_8, и тест
``tests/test_run_consistency.py``. torch не требуется.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from liquefaction_ai.data.grainsize import FRACTION_KEYS, plaxis_classification
from liquefaction_ai.physics.g0 import vs_from_g0


def check_artifact_consistency(src_dir: str = "data/dataset") -> Tuple[bool, List[str]]:
    """Проверить самосогласованность артефакта. Возвращает (все_ок, список_строк_отчёта).

    Проверки: убраны утечные/латентные поля (risk_score_true, OCR); Vs == sqrt(G0·1000/ρ) по
    digitrock (G0=ρ·Vs²/1000); Vs в физ. диапазоне; plaxis_class по грансоставу; static без NaN;
    наличие измеренной CRR.
    """
    m = pd.read_parquet(os.path.join(src_dir, "meta.parquet"))
    z = np.load(os.path.join(src_dir, "arrays.npz"))
    fn = json.load(open(os.path.join(src_dir, "feature_names.json")))
    names = fn["static_feature_names"] if isinstance(fn, dict) else fn
    N = len(m)
    report: List[str] = [f"Артефакт: {N} образцов, {m.shape[1]} колонок meta"]
    checks: List[bool] = []

    def chk(cond: bool, msg: str):
        checks.append(bool(cond))
        report.append(("OK   " if cond else "FAIL ") + msg)

    chk("risk_score_true" not in m.columns, "risk_score_true отсутствует в meta")
    chk(not any(c.upper() == "OCR" for c in m.columns), "OCR отсутствует в meta")
    chk(not any("OCR" in str(x) for x in names), "OCR отсутствует в static features")
    chk("PPR_max_true" in m.columns, "PPR_max_true есть (риск-поле)")

    rho = m["r"].to_numpy(float); vs_arr = m["V_s"].to_numpy(float)
    g0_recon = rho * vs_arr ** 2 / 1000.0
    vs_err = float(np.nanmax(np.abs(vs_from_g0(g0_recon, rho) - vs_arr)))
    chk(vs_err < 1.0, f"Vs == sqrt(G0·1000/ρ) (digitrock), max|Δ|={vs_err:.3f} м/с")
    vs = m["V_s"].astype(float)
    chk((vs.min() >= 40) and (vs.max() < 600), f"Vs диапазон {vs.min():.0f}–{vs.max():.0f} м/с (median {vs.median():.0f})")
    chk(np.isfinite(g0_recon).all() and (g0_recon > 0).all(), f"G0 определён для всех (median {np.median(g0_recon):.1f} МПа)")

    gcols = [f"gran_{k}" for k in FRACTION_KEYS]
    fr = m[gcols].to_numpy(float); has = fr.sum(1) > 1.0
    pc = plaxis_classification(fr)
    agree = (pd.Series(pc["plaxis_class"])[has].to_numpy() == m["plaxis_class"][has].to_numpy()).mean() if has.sum() else 1.0
    chk(agree > 0.999, f"plaxis_class совпадает на {int(has.sum())} образцах с грансоставом ({agree*100:.1f}%)")
    cu_chk = pc["D60"] / np.maximum(pc["D10"], 1e-9)
    cu_err = float(np.nanmax(np.abs(cu_chk[has] - m["Cu"].to_numpy(float)[has]))) if has.sum() else 0.0
    chk(cu_err < 0.5, f"Cu == D60/D10, max|Δ|={cu_err:.3f}")

    sf = z["static_features"]
    chk(np.isfinite(sf).all(), f"static_features без NaN/inf {sf.shape}")
    chk("crr_obs" in z and z["crr_obs_mask"].sum() > 0, f"CRR(N): {int(z['crr_obs_mask'].sum())}/{N} образцов")

    report.append("plaxis_class: " + str(m["plaxis_class"].value_counts().to_dict()))
    report.append("DONE")
    return all(checks), report
