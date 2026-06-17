import os, sys, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import torch
from pathlib import Path
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.training import read_hyperparams
from liquefaction_ai.evaluation import collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import DPIFlow, EVTNeuralSSM, EnsembleModel

name = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
dev = torch.device("cpu"); MD = Path("models")
CLS = {"dpi_flow": DPIFlow, "evt_ssm": EVTNeuralSSM}[name]
pop, cfg = load_population_artifact(Path("data/demo_run")); b = prepare_benchmark_dataset(pop, cfg, dev)
hp = read_hyperparams(MD, name)

members = []
for k in range(K):
    torch.manual_seed(1000 + k)
    m = CLS(**hp["model_kwargs"]).to(dev)
    m, _ = train_model(m, b["train"], b["val"], epochs=cfg.physics_epochs, model_name=f"{name}#{k}",
                       config=cfg, device=dev, track_metrics=False, scheduler="cosine")
    fit_interval_scale(m, b["val"], cfg, dev, level=0.90)
    members.append(m)

def metrics(model, tag):
    r, _ = compute_metrics(tag, collect_outputs(model, b["test"], cfg, dev), b["test"], cfg)
    return r

single = metrics(members[0], "single")
ens = metrics(EnsembleModel(members).to(dev), "ensemble")
print(f"== {name}: одиночная модель vs ансамбль (K={K}) ==")
for k, key in [("PPR RMSE", "Traj_RMSE"), ("CRPS", "Traj_CRPS"), ("NLL", "Traj_NLL"),
               ("Calib err", "Calibration_Error"), ("Cov@90", "Coverage_90"), ("AUROC", "AUROC"),
               ("Brier", "Brier"), ("PhysViol", "Physics_Violation_Rate"), ("Nliq logMAE", "N_liq_logMAE"),
               ("CRR RMSE", "CRR_RMSE")]:
    print(f"  {k:11s} {single[key]:.3f} -> {ens[key]:.3f}")
print("DONE")
