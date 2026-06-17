import os, sys, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import torch
from pathlib import Path
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.training import write_hyperparams, read_hyperparams, save_trained_model
from liquefaction_ai.evaluation import collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import DPIEvtNet

# key=val overrides: crr_mode, nliq_curve(0/1), calib(int), resid(0/1), ema(float), epochs(int), save(0/1)
opt = {"crr_mode": "decoupled", "nliq_curve": "1", "calib": "0", "resid": "0", "ema": "0.0", "epochs": "0", "save": "0"}
for a in sys.argv[1:]:
    if "=" in a:
        k, v = a.split("=", 1); opt[k] = v
dev = torch.device("cpu"); MD = Path("models")
pop, cfg = load_population_artifact(Path("data/demo_run")); b = prepare_benchmark_dataset(pop, cfg, dev)
sd = b["train"]["static"].shape[1]; pdim = b["train"]["prefix_summary"].shape[1]; qd = b["train"]["seq_in"].shape[-1]
epochs = int(opt["epochs"]) or cfg.physics_epochs
kw = dict(static_dim=sd, prefix_dim=pdim, seq_dim=qd, seq_len=cfg.seq_len, prefix_len=cfg.prefix_len,
          max_cycle_reference=cfg.max_cycle_reference, probabilistic=True, use_flow=True,
          crr_mode=opt["crr_mode"], nliq_from_curve=bool(int(opt["nliq_curve"])),
          calibration_steps=int(opt["calib"]), use_traj_residual=bool(int(opt["resid"])))
torch.manual_seed(0)
m = DPIEvtNet(**kw).to(dev)
m, hist = train_model(m, b["train"], b["val"], epochs=epochs, model_name="DPI-EVT",
                      config=cfg, device=dev, track_metrics=True, scheduler="cosine", ema_decay=float(opt["ema"]))
s = fit_interval_scale(m, b["val"], cfg, dev, level=0.90)
r, _ = compute_metrics("DPI-EVT", collect_outputs(m, b["test"], cfg, dev), b["test"], cfg)
print(f"== DPI-EVT [{opt['crr_mode']} nliq_curve={opt['nliq_curve']} calib={opt['calib']} resid={opt['resid']} "
      f"ema={opt['ema']} ep={epochs}] s={s:.2f} | base(D): RMSE0.114 CRPS0.057 CRR0.118 logMAE2.95 ==")
for k, key in [("PPR RMSE", "Traj_RMSE"), ("CRPS", "Traj_CRPS"), ("Calib err", "Calibration_Error"),
               ("AUROC", "AUROC"), ("Brier", "Brier"), ("PhysViol", "Physics_Violation_Rate"),
               ("Nliq logMAE", "N_liq_logMAE"), ("Nliq MAE", "N_liq_MAE"), ("CRR RMSE", "CRR_RMSE")]:
    v = r[key]; print(f"  {k:11s} {v:.4f}" if v == v else f"  {k:11s} nan")
if int(opt["save"]):
    write_hyperparams(MD, "dpi_evt", {"model_type": "DPIEvtNet", "display_name": "DPI-EVT", "model_kwargs": kw})
    save_trained_model(m, MD, "dpi_evt", {**read_hyperparams(MD, "dpi_evt"), "epochs": epochs, "calib_scale": s}, hist)
    print("saved dpi_evt")
print("DONE")
