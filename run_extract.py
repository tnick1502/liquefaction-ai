import os, sys, json, warnings
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ["LIQ_REAL_ROOT"] = "/sessions/determined-cool-fermat/mnt/Облако разжижения"

nb = json.load(open("notebooks/1_data_preparation/1_1_3_real_objects_loader.ipynb"))
ns = {"__name__": "__main__"}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    if i > 20:          # stop after save_population_artifact (cell 20)
        break
    src = "".join(c["source"])
    try:
        exec(compile(src, f"<cell{i}>", "exec"), ns)
    except SystemExit:
        pass
    except Exception as e:
        import traceback
        print(f"!!! ERROR in cell {i}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
print("EXTRACTION DONE")
