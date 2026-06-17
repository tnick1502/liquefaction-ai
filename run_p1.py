import os, sys, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
from dataclasses import replace
import torch
from pathlib import Path
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.data.synthetic import generate_population
from liquefaction_ai.training import read_hyperparams, write_hyperparams, save_trained_model
from liquefaction_ai.evaluation import collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import DPIFlow, EVTNeuralSSM

name = sys.argv[1]
dev = torch.device("cpu"); MD = Path("models")
cfg = None
pop, cfg = load_population_artifact(Path("data/demo_run"))
real = prepare_benchmark_dataset(pop, cfg, dev)
sd = real["train"]["static"].shape[1]; pdim = real["train"]["prefix_summary"].shape[1]; qd = real["train"]["seq_in"].shape[-1]

# --- синтетика для предобучения ---
syn_cfg = replace(cfg, n_scenarios=1600, benchmark_subset=1100)
syn = prepare_benchmark_dataset(generate_population(syn_cfg), syn_cfg, dev)

P0 = {"dpi_flow": dict(RMSE=0.128, CRPS=0.067, AUROC=1.000, Brier=0.009, Phys=0.000, Calib=0.083, Cov90=0.971, logMAE=1.401, CRR=0.183),
      "evt_ssm": dict(RMSE=0.147, CRPS=0.095, AUROC=1.000, Brier=0.012, Phys=0.000, Calib=0.101, Cov90=0.950, logMAE=1.562, CRR=0.182)}

if name == "dpi_flow":
    hp = read_hyperparams(MD, "dpi_flow"); model = DPIFlow(**hp["model_kwargs"]).to(dev); disp = "DPI-Flow"
else:
    hp = read_hyperparams(MD, "evt_ssm"); model = EVTNeuralSSM(**hp["model_kwargs"]).to(dev); disp = "EVT-NeuralSSM"

# 1) предобучение на синтетике
model, _ = train_model(model, syn["train"], syn["val"], epochs=4, model_name=disp + " (pretrain)",
                       config=cfg, device=dev, track_metrics=False, scheduler="cosine")
# 2) дообучение на реальных данных (меньший LR)
ft_cfg = replace(cfg, learning_rate=cfg.learning_rate * 0.3)
model, hist = train_model(model, real["train"], real["val"], epochs=cfg.physics_epochs, model_name=disp + " (finetune)",
                          config=ft_cfg, device=dev, track_metrics=True, scheduler="cosine")
# 3) пост-hoc конформная калибровка интервалов на валидации
s = fit_interval_scale(model, real["val"], cfg, dev, level=0.90)
# 4) сохранение и оценка
save_trained_model(model, MD, name, {**hp, "epochs": cfg.physics_epochs, "calib_scale": s}, hist)
r, _ = compute_metrics(disp, collect_outputs(model, real["test"], cfg, dev), real["test"], cfg)
o = P0[name]
print(f"== {disp} (P1: pretrain+finetune+censored+βNLL+conformal s={s:.2f}) ==")
print(f"  PPR RMSE   {o['RMSE']:.3f} -> {r['Traj_RMSE']:.3f}")
print(f"  CRPS       {o['CRPS']:.3f} -> {r['Traj_CRPS']:.3f}")
print(f"  NLL        (P0 n/a) -> {r['Traj_NLL']:.3f}")
print(f"  Calib err  {o['Calib']:.3f} -> {r['Calibration_Error']:.3f}")
print(f"  Cov@90     {o['Cov90']:.3f} -> {r['Coverage_90']:.3f}")
print(f"  AUROC      {o['AUROC']:.3f} -> {r['AUROC']:.3f}")
print(f"  Brier      {o['Brier']:.3f} -> {r['Brier']:.3f}")
print(f"  PhysViol   {o['Phys']:.3f} -> {r['Physics_Violation_Rate']:.3f}")
print(f"  Nliq logMAE {o['logMAE']:.3f} -> {r['N_liq_logMAE']:.3f}")
print(f"  CRR RMSE   {o['CRR']:.3f} -> {r['CRR_RMSE']:.3f}")
print("DONE")
