import os, sys, time, nbformat
from nbclient import NotebookClient

REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
os.environ["LIQ_REAL_ROOT"] = "/sessions/determined-cool-fermat/mnt/Облако разжижения"
os.environ["LIQ_DATASET"] = "real_objects"
os.environ["LIQ_QUICK"] = "1"
os.environ["LIQ_REAL_MAXOBJ"] = "0"
os.environ["PYTHONPATH"] = os.path.join(REPO, "src")

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
    "notebooks/3_evaluations/3_1_core_metrics.ipynb",
    "notebooks/3_evaluations/3_2_ablations_ood.ipynb",
    "notebooks/3_evaluations/3_3_case_studies.ipynb",
]
DONE = "/tmp/nbexec_done.txt"
done = set(open(DONE).read().split("\n")) if os.path.exists(DONE) else set()

for path in ORDER:
    if path in done:
        continue
    t0 = time.time()
    print(f">>> EXEC {path}", flush=True)
    nb = nbformat.read(path, as_version=4)
    nb_dir = os.path.dirname(path)
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": nb_dir}})
    try:
        client.execute()
        nbformat.write(nb, path)
        done.add(path); open(DONE, "w").write("\n".join(sorted(done)))
        print(f"<<< OK {path} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"<<< FAIL {path} ({time.time()-t0:.1f}s): {type(e).__name__}: {str(e)[:300]}", flush=True)
        break
else:
    print("ALL NOTEBOOKS EXECUTED IN-PLACE")
