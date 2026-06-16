import os, sys, warnings, time
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import torch
from pathlib import Path
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.training import grid_search, write_hyperparams, read_hyperparams, save_trained_model
from liquefaction_ai.evaluation import subsample_split
from liquefaction_ai.models import (RiskMLP, GRUBaseline, TCNBaseline, LSTMBaseline, TransformerBaseline,
                                    FTTransformer, CatBoostBaseline, PINNBaseline, DeepStateBaseline,
                                    RealNVPFlow, NeuralSplineFlow, DPIFlow, EVTNeuralSSM)

dev = torch.device("cpu"); MD = Path("models")
pop, cfg = load_population_artifact(Path("data/demo_run"))
b = prepare_benchmark_dataset(pop, cfg, dev)
sd = b["train"]["static"].shape[1]; pdim = b["train"]["prefix_summary"].shape[1]; qd = b["train"]["seq_in"].shape[-1]
gst = subsample_split(b["train"], 2000, cfg.seed); gsv = subsample_split(b["val"], 600, cfg.seed + 1)
be, pe = cfg.baseline_epochs, cfg.physics_epochs

SPECS = [
 ("mlp_risk","RiskMLP",RiskMLP,dict(static_dim=sd,prefix_dim=pdim),{"hidden_dim":[64,128]},"Brier",be),
 ("gru","GRU",GRUBaseline,dict(static_dim=sd,seq_dim=qd),{"hidden_dim":[64,96]},"Traj_RMSE",be),
 ("tcn","TCN",TCNBaseline,dict(static_dim=sd,seq_dim=qd),{"hidden_dim":[64,96]},"Traj_RMSE",be),
 ("lstm","LSTM",LSTMBaseline,dict(static_dim=sd,seq_dim=qd),{"hidden_dim":[64,96]},"Traj_RMSE",be),
 ("transformer","Transformer",TransformerBaseline,dict(static_dim=sd,seq_dim=qd,seq_len=cfg.seq_len),{"hidden_dim":[64,96]},"Traj_RMSE",be),
 ("ft_transformer","FT-Transformer",FTTransformer,dict(static_dim=sd,prefix_dim=pdim),{"n_layers":[2,3]},"Brier",be),
 ("pinn","PINN",PINNBaseline,dict(static_dim=sd,seq_dim=qd),{"hidden_dim":[64,96]},"Traj_RMSE",pe),
 ("deepstate","DeepState",DeepStateBaseline,dict(static_dim=sd,seq_dim=qd),{"hidden_dim":[64,96]},"Traj_RMSE",be),
 ("realnvp","RealNVP",RealNVPFlow,dict(static_dim=sd,prefix_dim=pdim,seq_len=cfg.seq_len),{"n_layers":[4,6]},"Traj_RMSE",be),
 ("nsf","Neural Spline Flow",NeuralSplineFlow,dict(static_dim=sd,prefix_dim=pdim,seq_len=cfg.seq_len),{"n_layers":[4,5]},"Traj_RMSE",be),
 ("dpi_flow","DPI-Flow",DPIFlow,dict(static_dim=sd,prefix_dim=pdim,seq_len=cfg.seq_len,prefix_len=cfg.prefix_len,max_cycle_reference=cfg.max_cycle_reference,theta_dim=31,probabilistic=True,use_analytical_layer=True),{"hidden_dim":[128,160],"calibration_steps":[1,2]},"Traj_RMSE",pe),
 ("evt_ssm","EVT-NeuralSSM",EVTNeuralSSM,dict(static_dim=sd,prefix_dim=pdim,seq_dim=qd,seq_len=cfg.seq_len,prefix_len=cfg.prefix_len,max_cycle_reference=cfg.max_cycle_reference),{"hidden_dim":[96,128]},"Traj_RMSE",pe),
]
DONE = "/tmp/retrain_done.txt"
done = set(open(DONE).read().split("\n")) if os.path.exists(DONE) else set()

def mark(n):
    done.add(n); open(DONE,"w").write("\n".join(sorted(done)))

START=time.time(); BUDGET=33.0
for name, disp, cls, fixed, grid, score, epochs in SPECS:
    if name in done: continue
    if time.time()-START > BUDGET: break
    t0=time.time()
    res, best = grid_search(lambda p, cls=cls, fixed=fixed: cls(**fixed, **p), grid, gst, gsv, cfg, dev,
                            search_epochs=1, score_metric=score)
    write_hyperparams(MD, name, {"model_type": cls.__name__, "display_name": disp,
                                 "model_kwargs": {**fixed, **best}, "search": {"best": best}})
    hp = read_hyperparams(MD, name); m = cls(**hp["model_kwargs"]).to(dev)
    m, hist = train_model(m, b["train"], b["val"], epochs=epochs, model_name=disp, config=cfg,
                          device=dev, track_metrics=True, scheduler="cosine")
    save_trained_model(m, MD, name, {**hp, "epochs": epochs}, hist)
    mark(name); print(f"OK {name} ({time.time()-t0:.0f}s) best={best}", flush=True)

if "catboost" not in done and all(s[0] in done for s in SPECS):
    cb = CatBoostBaseline(sd, pdim).fit(b["train"], b["val"]); cb.save(MD, "catboost")
    write_hyperparams(MD, "catboost", {"model_type":"CatBoostBaseline","display_name":"CatBoost",
                      "model_kwargs": dict(static_dim=sd, prefix_dim=pdim)})
    mark("catboost"); print("OK catboost", flush=True)

remaining = [s[0] for s in SPECS if s[0] not in done] + ([] if "catboost" in done else ["catboost"])
print("REMAINING:", remaining if remaining else "NONE — ALL DONE")
