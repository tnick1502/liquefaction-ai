import os, sys, json, time, warnings, traceback
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("LIQ_DATASET", "real_objects")
os.environ.setdefault("LIQ_QUICK", "1")          # быстрая синтетика (на real не влияет)
os.environ.setdefault("LIQ_REAL_MAXOBJ", "0")

ORDER = [
    "notebooks/1_data_preparation/1_0_select_dataset.ipynb",
    "notebooks/1_data_preparation/1_1_data_generation.ipynb",
    "notebooks/1_data_preparation/1_1_2_real_data_adapter.ipynb",
    "notebooks/1_data_preparation/1_1_3_real_objects_loader.ipynb",
    "notebooks/1_data_preparation/1_2_exploratory_analysis.ipynb",
    "notebooks/1_data_preparation/1_3_crr_parameter_analysis.ipynb",
    "notebooks/1_data_preparation/1_4_dataset_split.ipynb",
    "notebooks/2_model_training/2_1_baseline_models.ipynb",
    "notebooks/2_model_training/2_2_dpi_flow.ipynb",
    "notebooks/2_model_training/2_3_evt_neural_ssm.ipynb",
    "notebooks/2_model_training/2_4_dpi_evt.ipynb",
    "notebooks/3_evaluations/3_1_core_metrics.ipynb",
    "notebooks/3_evaluations/3_2_ablations_ood.ipynb",
    "notebooks/3_evaluations/3_3_case_studies.ipynb",
    "notebooks/4_topology/4_1_dpi_flow_latent_topology.ipynb",
    "notebooks/4_topology/4_2_topological_early_warning.ipynb",
    "notebooks/4_topology/4_3_evt_neural_ssm_topological_regularization.ipynb",
]
DONE = "/tmp/nb_done.txt"
done = set(open(DONE).read().split("\n")) if os.path.exists(DONE) else set()

def run_nb(path):
    nb = json.load(open(path)); ns = {"__name__": "__main__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            exec(compile(src, f"<{os.path.basename(path)}:c{i}>", "exec"), ns)
        except SystemExit:
            pass
        except Exception as e:
            print(f"  !! ERROR cell {i}: {type(e).__name__}: {e}")
            traceback.print_exc(); return False
    return True

for path in ORDER:
    if path in done:
        continue
    t0 = time.time()
    print(f">>> RUN {path}", flush=True)
    ok = run_nb(path)
    dt = time.time() - t0
    if ok:
        done.add(path); open(DONE, "w").write("\n".join(sorted(done)))
        print(f"<<< OK  {path}  ({dt:.1f}s)", flush=True)
    else:
        print(f"<<< FAIL {path}  ({dt:.1f}s)", flush=True); break
else:
    print("ALL NOTEBOOKS DONE")
