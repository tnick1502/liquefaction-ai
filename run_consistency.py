import os, sys, warnings, json
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np, pandas as pd
from liquefaction_ai.data.grainsize import plaxis_classification, FRACTION_KEYS
from liquefaction_ai.physics.g0 import g0_mpa, vs_from_g0

m = pd.read_parquet("data/real_objects/meta.parquet")
z = np.load("data/real_objects/arrays.npz")
fn = json.load(open("data/real_objects/feature_names.json"))
names = fn["static_feature_names"] if isinstance(fn, dict) else fn
N = len(m)
ok = lambda b: "OK " if b else "FAIL"
print(f"=== Артефакт: {N} образцов, {m.shape[1]} колонок meta ===")

# 1) латентные/OCR убраны
print(ok("risk_score_true" not in m.columns), "risk_score_true отсутствует в meta")
print(ok(not any(c.upper()=="OCR" for c in m.columns)), "OCR отсутствует в meta")
print(ok(not any("OCR" in str(x) for x in names)), "OCR отсутствует в static features")
print(ok("PPR_max_true" in m.columns), "PPR_max_true есть (риск-поле)")

# 2) Vs из G0 (digitrock) — пересчёт и сверка
sig1 = m["sigma_eff"].to_numpy(float)          # = sigma_1
# p_ref как в экстракторе: (sig1+2*sig3)/3/1000; sig3 не в meta → реконструируем из K0? используем сохранённый G0
g0_saved = m["G0"].to_numpy(float)
vs_chk = vs_from_g0(g0_saved, m["r"].to_numpy(float))
vs_err = np.nanmax(np.abs(vs_chk - m["V_s"].to_numpy(float)))
print(ok(vs_err < 1.0), f"Vs == sqrt(G0*1000/r) (digitrock РК), max|Δ|={vs_err:.3f} м/с")
vs = m["V_s"].astype(float)
print(ok((vs.min()>40) and (vs.max()<600)), f"Vs в физ. диапазоне: {vs.min():.0f}–{vs.max():.0f} м/с (median {vs.median():.0f})")
print(ok(m["G0"].notna().all()), f"G0 определён для всех (нет пропусков), G0 median={m['G0'].median():.1f} МПа")

# 3) plaxis_class согласован с digitrock по грансоставу
gcols=[f"gran_{k}" for k in FRACTION_KEYS]
fr=m[gcols].to_numpy(float); has=fr.sum(1)>1.0
pc=plaxis_classification(fr)
agree=(pd.Series(pc["plaxis_class"])[has].to_numpy()==m["plaxis_class"][has].to_numpy()).mean() if has.sum() else 1.0
print(ok(agree>0.999), f"plaxis_class == digitrock-грансостав для {int(has.sum())} образцов с грансоставом (совпадение {agree*100:.1f}%)")
# Cu = D60/D10 согласованность
cu_chk=pc["D60"]/np.maximum(pc["D10"],1e-9)
cu_err=np.nanmax(np.abs(cu_chk[has]-m["Cu"].to_numpy(float)[has]))
print(ok(cu_err<0.5), f"Cu == D60/D10 для образцов с грансоставом, max|Δ|={cu_err:.3f}")

# 4) static features без NaN/inf
sf=z["static_features"]
print(ok(np.isfinite(sf).all()), f"static_features без NaN/inf (форма {sf.shape})")

# 5) CRR для нейронки
print(ok("crr_obs" in z and z["crr_obs_mask"].sum()>0),
      f"CRR(N) есть: {int(z['crr_obs_mask'].sum())}/{N} образцов с измеренной кривой (ИГЭ-фит β/N^(1−α))")

# 6) типы грунта и распределение plaxis
print("Распределение plaxis_class:", m["plaxis_class"].value_counts().to_dict())
print("DONE")
