"""
Инкрементальное переобучение всех моделей бенчмарка на текущем data/demo_run.

Среда исполнения ограничивает один вызов ~45 с и не сохраняет фоновые процессы,
поэтому модели обучаются по одной единице за вызов. Состояние прогресса хранится в
models/.incr_state.json; запускать скрипт повторно, пока не появится "ALL DONE".

Методика воспроизводит ноутбуки 2_1–2_4 один-в-один:
* baseline (2_1): grid_search(search_epochs=1) → train (baseline/physics epochs) → save;
* CatBoost (2_1): .fit → save;
* DPI-Flow (2_2) и EVT-NeuralSSM (2_3): grid_search(search_epochs=2) → best-of-seeds [0,1,2]
  по val Traj_RMSE (physics_epochs, scheduler cosine) → fit_interval_scale → save;
* DPI-EVT (2_4): единый сид, train (physics_epochs, cosine) → fit_interval_scale → save.
"""
import os, sys, json, time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
import torch

from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model, set_global_seed
from liquefaction_ai.training import grid_search, write_hyperparams, read_hyperparams, save_trained_model
from liquefaction_ai.evaluation import subsample_split, collect_outputs, compute_metrics, fit_interval_scale
from liquefaction_ai.models import (GRUBaseline, LSTMBaseline, RiskMLP, TCNBaseline,
                                    TransformerBaseline, FTTransformer, CatBoostBaseline,
                                    PINNBaseline, DeepStateBaseline, RealNVPFlow, NeuralSplineFlow,
                                    DPIFlow, EVTNeuralSSM, DPIEvtNet)

DATA_DIR = REPO / "data" / "demo_run"
MODELS_DIR = REPO / "models"
STATE = MODELS_DIR / ".incr_state.json"
BUDGET = float(os.environ.get("INCR_BUDGET", "38"))   # сек на вызов (запас до лимита 45 с)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
population, config = load_population_artifact(DATA_DIR)
benchmark = prepare_benchmark_dataset(population, config, device)
train, val, test = benchmark["train"], benchmark["val"], benchmark["test"]
static_dim = train["static"].shape[1]; prefix_dim = train["prefix_summary"].shape[1]
seq_dim = train["seq_in"].shape[-1]
gs_train = subsample_split(train, 2000, config.seed)
gs_val = subsample_split(val, 600, config.seed + 1)

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"done": [], "flow_seed_val": {}, "evt_seed_val": {}}
def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))

state = load_state()
done = set(state["done"])

# ---------- спецификации baseline (как в 2_1 c6) ----------
base_specs = {
    "mlp_risk": dict(display="MLP-Risk", cls=RiskMLP,
                     fixed=dict(static_dim=static_dim, prefix_dim=prefix_dim),
                     grid={"hidden_dim": [64, 128]}, score="Brier"),
    "gru": dict(display="GRU", cls=GRUBaseline,
                fixed=dict(static_dim=static_dim, seq_dim=seq_dim),
                grid={"hidden_dim": [64, 96]}, score="Traj_RMSE"),
    "tcn": dict(display="TCN", cls=TCNBaseline,
                fixed=dict(static_dim=static_dim, seq_dim=seq_dim),
                grid={"hidden_dim": [64, 96]}, score="Traj_RMSE"),
    "lstm": dict(display="LSTM", cls=LSTMBaseline,
                 fixed=dict(static_dim=static_dim, seq_dim=seq_dim),
                 grid={"hidden_dim": [64, 96]}, score="Traj_RMSE"),
    "transformer": dict(display="Transformer", cls=TransformerBaseline,
                 fixed=dict(static_dim=static_dim, seq_dim=seq_dim, seq_len=config.seq_len),
                 grid={"hidden_dim": [64, 96]}, score="Traj_RMSE"),
    "ft_transformer": dict(display="FT-Transformer", cls=FTTransformer,
                 fixed=dict(static_dim=static_dim, prefix_dim=prefix_dim),
                 grid={"n_layers": [2, 3]}, score="Brier"),
    "pinn": dict(display="PINN", cls=PINNBaseline,
                 fixed=dict(static_dim=static_dim, seq_dim=seq_dim),
                 grid={"hidden_dim": [64, 96]}, score="Traj_RMSE"),
    "deepstate": dict(display="DeepState", cls=DeepStateBaseline,
                 fixed=dict(static_dim=static_dim, seq_dim=seq_dim),
                 grid={"hidden_dim": [64, 96]}, score="Traj_RMSE"),
    "realnvp": dict(display="RealNVP", cls=RealNVPFlow,
                 fixed=dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_len=config.seq_len),
                 grid={"n_layers": [4, 6]}, score="Traj_RMSE"),
    "nsf": dict(display="Neural Spline Flow", cls=NeuralSplineFlow,
                 fixed=dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_len=config.seq_len),
                 grid={"n_layers": [4, 5]}, score="Traj_RMSE"),
}
CLS = {c.__name__: c for c in [RiskMLP, GRUBaseline, TCNBaseline, LSTMBaseline, TransformerBaseline,
                               FTTransformer, PINNBaseline, DeepStateBaseline, RealNVPFlow, NeuralSplineFlow]}

def train_baseline(name):
    spec = base_specs[name]
    cls, fixed, grid, score = spec["cls"], spec["fixed"], spec["grid"], spec["score"]
    res, best = grid_search(lambda p, cls=cls, fixed=fixed: cls(**fixed, **p),
                            grid, gs_train, gs_val, config, device, search_epochs=1, score_metric=score)
    write_hyperparams(MODELS_DIR, name, {"model_type": cls.__name__, "display_name": spec["display"],
                      "model_kwargs": {**fixed, **best},
                      "search": {"grid": grid, "score_metric": score, "best": best}})
    epochs = config.physics_epochs if name == "pinn" else config.baseline_epochs
    model = CLS[cls.__name__](**{**fixed, **best}).to(device)
    model, history = train_model(model, train, val, epochs=epochs, model_name=spec["display"],
                                 config=config, device=device, track_metrics=True)
    save_trained_model(model, MODELS_DIR, name, {"model_type": cls.__name__, "display_name": spec["display"],
                       "model_kwargs": {**fixed, **best}, "epochs": epochs,
                       "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                       "batch_size": config.batch_size, "seed": config.seed}, history)

def train_catboost(name):
    cb = CatBoostBaseline(static_dim, prefix_dim).fit(train, val)
    cb.save(MODELS_DIR, "catboost")
    write_hyperparams(MODELS_DIR, "catboost", {"model_type": "CatBoostBaseline", "display_name": "CatBoost",
                      "model_kwargs": dict(static_dim=static_dim, prefix_dim=prefix_dim)})

# ---------- DPI-Flow / EVT: grid, seeds, finalize ----------
def flow_fixed():
    return dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_len=config.seq_len,
                prefix_len=config.prefix_len, max_cycle_reference=config.max_cycle_reference,
                theta_dim=31, probabilistic=True, use_analytical_layer=True,
                liq_threshold=config.liq_threshold,
                use_observed_aux_loss=config.use_observed_aux_loss)
def evt_fixed():
    return dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_dim=seq_dim, seq_len=config.seq_len,
                prefix_len=config.prefix_len, max_cycle_reference=config.max_cycle_reference,
                use_trigger_head=True, structured_post_event=True, use_crr_damage=True,
                integrator="euler", liq_threshold=config.liq_threshold,
                use_observed_aux_loss=config.use_observed_aux_loss)

def grid_unit(name, builder_cls, fixed, grid, mtype, disp):
    res, best = grid_search(lambda p: builder_cls(**fixed, **p), grid, gs_train, gs_val,
                            config, device, search_epochs=2, score_metric="Traj_RMSE")
    write_hyperparams(MODELS_DIR, name, {"model_type": mtype, "display_name": disp,
                      "model_kwargs": {**fixed, **best},
                      "search": {"grid": grid, "score_metric": "Traj_RMSE", "best": best}})

def seed_unit(name, builder_cls, seed, val_store):
    hp = read_hyperparams(MODELS_DIR, name)
    torch.manual_seed(seed)
    cand = builder_cls(**hp["model_kwargs"]).to(device)
    cand, hist = train_model(cand, train, val, epochs=config.physics_epochs,
                             model_name=f"{hp['display_name']} (seed {seed})", config=config,
                             device=device, track_metrics=True, scheduler="cosine")
    vr, _ = compute_metrics("val", collect_outputs(cand, val, config, device), val, config)
    (MODELS_DIR / name).mkdir(parents=True, exist_ok=True)
    torch.save(cand.state_dict(), MODELS_DIR / name / f"_seed{seed}.pt")
    hist.to_parquet(MODELS_DIR / name / f"_seed{seed}_hist.parquet")
    val_store[str(seed)] = float(vr["Traj_RMSE"])
    save_state(state)

def finalize_unit(name, builder_cls, val_store):
    hp = read_hyperparams(MODELS_DIR, name)
    best_seed = min(val_store, key=lambda k: val_store[k])
    import pandas as pd
    model = builder_cls(**hp["model_kwargs"]).to(device)
    model.load_state_dict(torch.load(MODELS_DIR / name / f"_seed{best_seed}.pt", map_location=device))
    history = pd.read_parquet(MODELS_DIR / name / f"_seed{best_seed}_hist.parquet")
    calib = fit_interval_scale(model, val, config, device, level=0.90)
    save_trained_model(model, MODELS_DIR, name, {**hp, "epochs": config.physics_epochs,
                       "learning_rate": config.learning_rate, "weight_decay": config.weight_decay,
                       "batch_size": config.batch_size, "calib_scale": float(calib),
                       "best_seed": int(best_seed), "seed_val_rmse": val_store}, history)
    for f in list((MODELS_DIR / name).glob("_seed*.pt")) + list((MODELS_DIR / name).glob("_seed*_hist.parquet")):
        try:
            f.unlink()
        except OSError:
            pass            # смонтированная ФС может запрещать удаление — временные файлы безвредны

def train_dpi_evt(name):
    mk = dict(static_dim=static_dim, prefix_dim=prefix_dim, seq_dim=seq_dim,
              seq_len=config.seq_len, prefix_len=config.prefix_len,
              max_cycle_reference=config.max_cycle_reference, probabilistic=True, use_flow=True,
              crr_mode="decoupled", nliq_from_curve=True, liq_threshold=config.liq_threshold,
              calibration_steps=0, use_traj_residual=False,
              use_observed_aux_loss=config.use_observed_aux_loss)
    set_global_seed(config.seed)
    model = DPIEvtNet(**mk).to(device)
    model, history = train_model(model, train, val, epochs=config.physics_epochs, model_name="DPI-EVT",
                                 config=config, device=device, track_metrics=True, scheduler="cosine", ema_decay=0.0)
    calib = fit_interval_scale(model, val, config, device, level=0.90)
    hp = {"model_type": "DPIEvtNet", "display_name": "DPI-EVT", "model_kwargs": mk}
    write_hyperparams(MODELS_DIR, "dpi_evt", hp)
    save_trained_model(model, MODELS_DIR, "dpi_evt", {**hp, "epochs": config.physics_epochs,
                       "calib_scale": float(calib)}, history)

# ---------- очередь единиц ----------
UNITS = []
for n in base_specs:
    UNITS.append((f"base:{n}", lambda n=n: train_baseline(n)))
UNITS.append(("catboost", lambda: train_catboost("catboost")))
UNITS.append(("flow:grid", lambda: grid_unit("dpi_flow", DPIFlow, flow_fixed(),
              {"hidden_dim": [128, 160], "calibration_steps": [1, 2]}, "DPIFlow", "DPI-Flow")))
for s in (0, 1, 2):
    UNITS.append((f"flow:seed{s}", lambda s=s: seed_unit("dpi_flow", DPIFlow, s, state["flow_seed_val"])))
UNITS.append(("flow:finalize", lambda: finalize_unit("dpi_flow", DPIFlow, state["flow_seed_val"])))
UNITS.append(("evt:grid", lambda: grid_unit("evt_ssm", EVTNeuralSSM, evt_fixed(),
              {"hidden_dim": [96, 128]}, "EVTNeuralSSM", "EVT-NeuralSSM")))
for s in (0, 1, 2):
    UNITS.append((f"evt:seed{s}", lambda s=s: seed_unit("evt_ssm", EVTNeuralSSM, s, state["evt_seed_val"])))
UNITS.append(("evt:finalize", lambda: finalize_unit("evt_ssm", EVTNeuralSSM, state["evt_seed_val"])))
UNITS.append(("dpi_evt", lambda: train_dpi_evt("dpi_evt")))

t0 = time.time(); ran = []
for uid, fn in UNITS:
    if uid in done:
        continue
    if ran and (time.time() - t0) > BUDGET:
        break                       # бюджет вызова исчерпан — продолжим в следующем
    s0 = time.time()
    print(f">>> {uid}", flush=True)
    fn()
    done.add(uid); state["done"] = sorted(done); save_state(state)
    ran.append(uid)
    print(f"<<< {uid} ({time.time()-s0:.1f}s)", flush=True)

pending = [u for u, _ in UNITS if u not in done]
print(f"RAN: {ran}")
print(f"PENDING ({len(pending)}): {pending[:6]}{'...' if len(pending)>6 else ''}")
if not pending:
    print("ALL DONE")
