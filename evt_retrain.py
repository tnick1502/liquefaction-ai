"""
Целевое переобучение EVT-NeuralSSM после правки лосса (анти-overshoot на неразжижающихся опытах).

Возвращает EVT «в форму» на режиме «нет разжижения»: больше эпох и сидов, выбор лучшего сида по
ВАЛИДАЦИОННОЙ worst-state траекторной ошибке (а не глобальной) — чтобы не выбрать сид, хороший
только на лёгком большинстве. Запускать повторно до «EVT DONE» (бюджет вызова ~38 с).
Параметры через env: EVT_EPOCHS (по умолч. 14), EVT_NSEEDS (5).
"""
import os, sys, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
import torch

from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.training import grid_search, write_hyperparams, read_hyperparams, save_trained_model
from liquefaction_ai.evaluation import subsample_split, collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import EVTNeuralSSM

DATA_DIR = REPO / "data" / "demo_run"; MODELS_DIR = REPO / "models"
STATE = MODELS_DIR / ".evt_retrain.json"
EPOCHS = int(os.environ.get("EVT_EPOCHS", "14")); NSEEDS = int(os.environ.get("EVT_NSEEDS", "5"))
BUDGET = float(os.environ.get("INCR_BUDGET", "38"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
population, config = load_population_artifact(DATA_DIR)
benchmark = prepare_benchmark_dataset(population, config, device)
train, val, test = benchmark["train"], benchmark["val"], benchmark["test"]
static_dim = train["static"].shape[1]; prefix_dim = train["prefix_summary"].shape[1]; seq_dim = train["seq_in"].shape[-1]
gs_train = subsample_split(train, 2000, config.seed); gs_val = subsample_split(val, 600, config.seed + 1)

def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"done": [], "seed_val": {}}
def save_state(s): STATE.write_text(json.dumps(s, indent=1))
state = load_state(); done = set(state["done"])

fixed = dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_dim=seq_dim, seq_len=config.seq_len,
             prefix_len=config.prefix_len, max_cycle_reference=config.max_cycle_reference,
             use_trigger_head=True, structured_post_event=True, use_crr_damage=True,
             integrator="euler", liq_threshold=config.liq_threshold)

def do_grid():
    res, best = grid_search(lambda p: EVTNeuralSSM(**fixed, **p), {"hidden_dim": [96, 128]},
                            gs_train, gs_val, config, device, search_epochs=2, score_metric="Traj_RMSE")
    write_hyperparams(MODELS_DIR, "evt_ssm", {"model_type": "EVTNeuralSSM", "display_name": "EVT-NeuralSSM",
                      "model_kwargs": {**fixed, **best},
                      "search": {"grid": {"hidden_dim": [96, 128]}, "score_metric": "Traj_RMSE", "best": best}})

def do_seed(seed):
    hp = read_hyperparams(MODELS_DIR, "evt_ssm")
    torch.manual_seed(seed)
    m = EVTNeuralSSM(**hp["model_kwargs"]).to(device)
    m, hist = train_model(m, train, val, epochs=EPOCHS, model_name=f"EVT-NeuralSSM (seed {seed})",
                          config=config, device=device, track_metrics=True, scheduler="cosine")
    vr, _ = compute_metrics("val", collect_outputs(m, val, config, device), val, config)
    (MODELS_DIR / "evt_ssm").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), MODELS_DIR / "evt_ssm" / f"_seed{seed}.pt")
    hist.to_parquet(MODELS_DIR / "evt_ssm" / f"_seed{seed}_hist.parquet")
    # выбор по worst-state траектории (а не глобальной), с тай-брейком по balanced
    state["seed_val"][str(seed)] = {"worst": float(vr.get("Traj_RMSE_worst", vr["Traj_RMSE"])),
                                     "balanced": float(vr.get("Traj_RMSE_balanced", vr["Traj_RMSE"])),
                                     "global": float(vr["Traj_RMSE"])}
    save_state(state)

def do_finalize():
    import pandas as pd
    hp = read_hyperparams(MODELS_DIR, "evt_ssm")
    sv = state["seed_val"]
    best_seed = min(sv, key=lambda k: (sv[k]["worst"], sv[k]["balanced"]))
    m = EVTNeuralSSM(**hp["model_kwargs"]).to(device)
    m.load_state_dict(torch.load(MODELS_DIR / "evt_ssm" / f"_seed{best_seed}.pt", map_location=device))
    history = pd.read_parquet(MODELS_DIR / "evt_ssm" / f"_seed{best_seed}_hist.parquet")
    calib = fit_interval_scale(m, val, config, device, level=0.90)
    save_trained_model(m, MODELS_DIR, "evt_ssm", {**hp, "epochs": EPOCHS, "learning_rate": config.learning_rate,
                       "weight_decay": config.weight_decay, "batch_size": config.batch_size,
                       "calib_scale": float(calib), "best_seed": int(best_seed),
                       "seed_val": sv, "selection": "val Traj_RMSE_worst"}, history)
    for f in list((MODELS_DIR / "evt_ssm").glob("_seed*.pt")) + list((MODELS_DIR / "evt_ssm").glob("_seed*_hist.parquet")):
        try: f.unlink()
        except OSError: pass
    print("best_seed:", best_seed, "| val worst/balanced/global:", sv[best_seed], "| calib:", round(float(calib),3))

UNITS = [("grid", do_grid)] + [(f"seed{s}", lambda s=s: do_seed(s)) for s in range(NSEEDS)] + [("finalize", do_finalize)]
t0 = time.time(); ran = []
for uid, fn in UNITS:
    if uid in done: continue
    if ran and (time.time() - t0) > BUDGET: break
    s0 = time.time(); print(f">>> {uid}", flush=True); fn()
    done.add(uid); state["done"] = sorted(done); save_state(state); ran.append(uid)
    print(f"<<< {uid} ({time.time()-s0:.1f}s)", flush=True)
pending = [u for u, _ in UNITS if u not in done]
print(f"RAN: {ran}"); print(f"PENDING: {pending}")
print("EVT DONE" if not pending else "")
