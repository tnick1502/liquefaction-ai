import os, sys, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
import numpy as np, torch
from pathlib import Path
from liquefaction_ai import load_population_artifact, prepare_benchmark_dataset, train_model
from liquefaction_ai.models import DPIFlow
from liquefaction_ai.training.losses import observed_aux_loss as observed_auxiliary_loss
from liquefaction_ai.data.splits import iterate_minibatches

dev = torch.device("cpu")
pop, cfg = load_population_artifact(Path("data/demo_run"))
bench = prepare_benchmark_dataset(pop, cfg, dev)
tr = bench["train"]
static_dim = tr["static"].shape[1]; prefix_dim = tr["prefix_summary"].shape[1]
print("crr_obs in train split:", "crr_obs" in tr, "| crr_obs_mask sum:",
      float(tr["crr_obs_mask"].sum()) if "crr_obs_mask" in tr else "NA")

model = DPIFlow(static_dim=static_dim, prefix_dim=prefix_dim, seq_len=cfg.seq_len,
                prefix_len=cfg.prefix_len, max_cycle_reference=cfg.max_cycle_reference,
                theta_dim=31, probabilistic=True, use_analytical_layer=True,
                hidden_dim=128, calibration_steps=1).to(dev)

# one batch -> confirm CRR supervision term is active and >0
batch = next(iterate_minibatches(tr, batch_size=64, device=dev, shuffle=True, seed=1))
model.eval()
with torch.no_grad():
    out = model.forward_batch(batch)
    has_crr_out = "crr" in out
    aux_no = observed_auxiliary_loss(out, {k:v for k,v in batch.items() if k!="crr_obs"}, use_states=True)
    aux_yes = observed_auxiliary_loss(out, batch, use_states=True)
print("model emits 'crr' output:", has_crr_out, "| crr output shape:", tuple(out["crr"].shape) if has_crr_out else None)
print("aux loss WITHOUT crr_obs: %.5f | WITH crr_obs: %.5f | crr term active: %s"
      % (float(aux_no), float(aux_yes), bool(abs(float(aux_yes)-float(aux_no))>1e-9)))

# CRR MSE (masked) before vs after a short train
def crr_mse(m):
    m.eval()
    with torch.no_grad():
        o = m.forward_batch(batch); mask = batch["mask"]; cm = batch["crr_obs_mask"]
        ps = (((o["crr"]-batch["crr_obs"])**2)*mask).sum(1)/torch.clamp(mask.sum(1),min=1.)
        return float((ps*cm).sum()/torch.clamp(cm.sum(),min=1.))
before = crr_mse(model)
model, hist = train_model(model, tr, bench["val"], epochs=3, model_name="DPI-Flow",
                          config=cfg, device=dev, track_metrics=False, scheduler="cosine")
after = crr_mse(model)
print("masked CRR MSE  before=%.5f  after 3 epochs=%.5f  (improved: %s)" % (before, after, after < before))
print('hist rows:', len(hist))
print("CRR-FOR-NN CHECK DONE")
