"""
Целевое переобучение DPI-EVT после правки лосса (анти-overshoot + censoring-aware trigger).

Как и у EVT: больше эпох и сидов, выбор лучшего сида по ВАЛИДАЦИОННОЙ worst-state траектории.
Воспроизводит model_kwargs ноутбука 2_4 (crr_mode='decoupled', nliq_from_curve и т.д.).
Запускать повторно до «DPIEVT DONE». Параметры: DPIEVT_EPOCHS (14), DPIEVT_NSEEDS (5).
"""
import os, sys, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
import torch

from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.training import write_hyperparams, save_trained_model
from liquefaction_ai.evaluation import collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import DPIEvtNet

DATA_DIR = REPO / "data" / "demo_run"; MODELS_DIR = REPO / "models"
STATE = MODELS_DIR / ".dpi_evt_retrain.json"
EPOCHS = int(os.environ.get("DPIEVT_EPOCHS", "14")); NSEEDS = int(os.environ.get("DPIEVT_NSEEDS", "5"))
BUDGET = float(os.environ.get("INCR_BUDGET", "34"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
population, config = load_population_artifact(DATA_DIR)
benchmark = prepare_benchmark_dataset(population, config, device)
train, val, test = benchmark["train"], benchmark["val"], benchmark["test"]
static_dim = train["static"].shape[1]; prefix_dim = train["prefix_summary"].shape[1]; seq_dim = train["seq_in"].shape[-1]

MK = dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_dim=seq_dim,
          seq_len=config.seq_len, prefix_len=config.prefix_len,
          max_cycle_reference=config.max_cycle_reference, probabilistic=True, use_flow=True,
          crr_mode="decoupled", nliq_from_curve=True, liq_threshold=config.liq_threshold,
          calibration_steps=0, use_traj_residual=False,
          use_observed_aux_loss=config.use_observed_aux_loss)

def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "seed_val": {}}
def save_state(s): STATE.write_text(json.dumps(s, indent=1))
state = load_state(); done = set(state["done"])

def do_seed(seed):
    torch.manual_seed(seed)
    m = DPIEvtNet(**MK).to(device)
    m, hist = train_model(m, train, val, epochs=EPOCHS, model_name=f"DPI-EVT (seed {seed})",
                          config=config, device=device, track_metrics=True, scheduler="cosine", ema_decay=0.0)
    vr, _ = compute_metrics("val", collect_outputs(m, val, config, device), val, config)
    (MODELS_DIR / "dpi_evt").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), MODELS_DIR / "dpi_evt" / f"_seed{seed}.pt")
    hist.to_parquet(MODELS_DIR / "dpi_evt" / f"_seed{seed}_hist.parquet")
    state["seed_val"][str(seed)] = {"worst": float(vr.get("Traj_RMSE_worst", vr["Traj_RMSE"])),
                                     "balanced": float(vr.get("Traj_RMSE_balanced", vr["Traj_RMSE"])),
                                     "global": float(vr["Traj_RMSE"])}
    save_state(state)

def do_finalize():
    import pandas as pd
    sv = state["seed_val"]
    best_seed = min(sv, key=lambda k: (sv[k]["worst"], sv[k]["balanced"]))
    m = DPIEvtNet(**MK).to(device)
    m.load_state_dict(torch.load(MODELS_DIR / "dpi_evt" / f"_seed{best_seed}.pt", map_location=device))
    history = pd.read_parquet(MODELS_DIR / "dpi_evt" / f"_seed{best_seed}_hist.parquet")
    calib = fit_interval_scale(m, val, config, device, level=0.90)
    hp = {"model_type": "DPIEvtNet", "display_name": "DPI-EVT", "model_kwargs": MK}
    write_hyperparams(MODELS_DIR, "dpi_evt", hp)
    save_trained_model(m, MODELS_DIR, "dpi_evt", {**hp, "epochs": EPOCHS, "calib_scale": float(calib),
                       "best_seed": int(best_seed), "seed_val": sv, "selection": "val Traj_RMSE_worst"}, history)
    for f in list((MODELS_DIR / "dpi_evt").glob("_seed*.pt")) + list((MODELS_DIR / "dpi_evt").glob("_seed*_hist.parquet")):
        try: f.unlink()
        except OSError: pass
    print("best_seed:", best_seed, "| val:", sv[best_seed], "| calib:", round(float(calib), 3))

UNITS = [(f"seed{s}", lambda s=s: do_seed(s)) for s in range(NSEEDS)] + [("finalize", do_finalize)]
t0 = time.time(); ran = []
for uid, fn in UNITS:
    if uid in done: continue
    if ran and (time.time() - t0) > BUDGET: break
    s0 = time.time(); print(f">>> {uid}", flush=True); fn()
    done.add(uid); state["done"] = sorted(done); save_state(state); ran.append(uid)
    print(f"<<< {uid} ({time.time()-s0:.1f}s)", flush=True)
pending = [u for u, _ in UNITS if u not in done]
print(f"RAN: {ran}"); print(f"PENDING: {pending}")
print("DPIEVT DONE" if not pending else "")
