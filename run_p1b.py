import os, sys, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import torch
from pathlib import Path
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.training import grid_search, write_hyperparams, read_hyperparams, save_trained_model
from liquefaction_ai.evaluation import subsample_split, collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import DPIFlow, EVTNeuralSSM

name = sys.argv[1]
dev = torch.device("cpu"); MD = Path("models")
pop, cfg = load_population_artifact(Path("data/demo_run")); b = prepare_benchmark_dataset(pop, cfg, dev)
sd = b["train"]["static"].shape[1]; pdim = b["train"]["prefix_summary"].shape[1]; qd = b["train"]["seq_in"].shape[-1]
gst = subsample_split(b["train"], 2000, cfg.seed); gsv = subsample_split(b["val"], 600, cfg.seed + 1)

P0 = {"dpi_flow": dict(RMSE=0.128, CRPS=0.067, AUROC=1.000, Brier=0.009, Phys=0.000, Calib=0.083, Cov90=0.971, logMAE=1.401, CRR=0.183),
      "evt_ssm": dict(RMSE=0.147, CRPS=0.095, AUROC=1.000, Brier=0.012, Phys=0.000, Calib=0.101, Cov90=0.950, logMAE=1.562, CRR=0.182)}
SPEC = {"dpi_flow": ("DPI-Flow", DPIFlow, dict(static_dim=sd, prefix_dim=pdim, seq_len=cfg.seq_len, prefix_len=cfg.prefix_len,
                     max_cycle_reference=cfg.max_cycle_reference, theta_dim=31, probabilistic=True, use_analytical_layer=True),
                     {"hidden_dim": [128, 160], "calibration_steps": [1, 2]}),
        "evt_ssm": ("EVT-NeuralSSM", EVTNeuralSSM, dict(static_dim=sd, prefix_dim=pdim, seq_dim=qd, seq_len=cfg.seq_len,
                    prefix_len=cfg.prefix_len, max_cycle_reference=cfg.max_cycle_reference), {"hidden_dim": [96, 128]})}
disp, cls, fixed, grid = SPEC[name]
res, best = grid_search(lambda p, cls=cls, fixed=fixed: cls(**fixed, **p), grid, gst, gsv, cfg, dev, search_epochs=1, score_metric="Traj_RMSE")
write_hyperparams(MD, name, {"model_type": cls.__name__, "display_name": disp, "model_kwargs": {**fixed, **best}, "search": {"best": best}})
hp = read_hyperparams(MD, name); m = cls(**hp["model_kwargs"]).to(dev)
m, hist = train_model(m, b["train"], b["val"], epochs=cfg.physics_epochs, model_name=disp, config=cfg, device=dev, track_metrics=True, scheduler="cosine")
s = fit_interval_scale(m, b["val"], cfg, dev, level=0.90)
save_trained_model(m, MD, name, {**hp, "epochs": cfg.physics_epochs, "calib_scale": s}, hist)
r, _ = compute_metrics(disp, collect_outputs(m, b["test"], cfg, dev), b["test"], cfg)
o = P0[name]
print(f"== {disp} (P1: censored N_liq + conformal s={s:.2f}, без β-NLL/предобучения) ==")
for k, key in [("PPR RMSE", "Traj_RMSE"), ("CRPS", "Traj_CRPS"), ("Calib err", "Calibration_Error"),
               ("Cov@90", "Coverage_90"), ("AUROC", "AUROC"), ("Brier", "Brier"),
               ("PhysViol", "Physics_Violation_Rate"), ("Nliq logMAE", "N_liq_logMAE"), ("CRR RMSE", "CRR_RMSE")]:
    ov = o.get({"Traj_RMSE": "RMSE", "Traj_CRPS": "CRPS", "Calibration_Error": "Calib", "Coverage_90": "Cov90",
                "Physics_Violation_Rate": "Phys", "N_liq_logMAE": "logMAE", "CRR_RMSE": "CRR"}.get(key, key), float("nan"))
    print(f"  {k:11s} {ov:.3f} -> {r[key]:.3f}")
print("DONE")
